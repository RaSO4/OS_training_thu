#![feature(portable_simd)]
//! TinyVLA board runner: closed-loop grid-world demo + benchmarks.

include!("../../weights/vocab.rs");

mod env;
mod model;
mod sys;

use env::{build_features, build_features_from, Env, Rng, DIR_NAMES, NN};
use model::{Model, Scratch};
use std::time::Instant;

pub const BOW_DIM: usize = 17;

static WEIGHTS: &[u8] = include_bytes!("../../weights/vla_weights.bin");
static TESTVEC: &[u8] = include_bytes!("../../weights/vla_test.bin");

#[derive(Clone)]
struct Cfg {
    mode: String,
    episodes: usize,
    batch: usize,
    threads: usize,
    cpu: i64,
    seed: u64,
    max_steps: u32,
    residual: f32,
    gain_scale: f32,
    iters: usize,
    quant: bool,
    json: bool,
}

impl Default for Cfg {
    fn default() -> Self {
        Self {
            mode: "bench".into(),
            episodes: 100,
            batch: 1,
            threads: 1,
            cpu: -1,
            seed: 1,
            max_steps: 64,
            residual: 0.01,
            gain_scale: 4.0,
            iters: 20000,
            quant: false,
            json: false,
        }
    }
}

fn parse_args() -> Cfg {
    let mut c = Cfg::default();
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--mode" => {
                i += 1;
                c.mode = args[i].clone();
            }
            "--episodes" => {
                i += 1;
                c.episodes = args[i].parse().unwrap();
            }
            "--batch" => {
                i += 1;
                c.batch = args[i].parse().unwrap();
            }
            "--threads" => {
                i += 1;
                c.threads = args[i].parse().unwrap();
            }
            "--cpu" => {
                i += 1;
                c.cpu = args[i].parse().unwrap();
            }
            "--seed" => {
                i += 1;
                c.seed = args[i].parse().unwrap();
            }
            "--max-steps" => {
                i += 1;
                c.max_steps = args[i].parse().unwrap();
            }
            "--residual" => {
                i += 1;
                c.residual = args[i].parse().unwrap();
            }
            "--gain-scale" => {
                i += 1;
                c.gain_scale = args[i].parse().unwrap();
            }
            "--iters" => {
                i += 1;
                c.iters = args[i].parse().unwrap();
            }
            "--quant" => {
                i += 1;
                c.quant = args[i].as_str() == "i8";
            }
            "--json" => c.json = true,
            "--help" | "-h" => {
                println!(
                    "tiny-vla: --mode demo|bench|selftest --episodes N --batch B --threads T \
                     --cpu C --seed S --max-steps N --residual F --gain-scale G --iters N --quant none|i8 --json"
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown argument: {other}");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    if c.mode == "demo" && c.episodes == 100 {
        c.episodes = 3;
    }
    c
}

#[derive(Default, Clone)]
struct Metrics {
    episodes: usize,
    success: usize,
    wrong: usize,
    timeout: usize,
    steps: u64,
    collisions: u64,
    color_correct: u64,
    color_total: u64,
    feature_ticks: u64,
    forward_ticks: u64,
    forward_calls: u64,
    wall_secs: f64,
}

impl Metrics {
    fn merge(&mut self, o: &Metrics) {
        self.episodes += o.episodes;
        self.success += o.success;
        self.wrong += o.wrong;
        self.timeout += o.timeout;
        self.steps += o.steps;
        self.collisions += o.collisions;
        self.color_correct += o.color_correct;
        self.color_total += o.color_total;
        self.feature_ticks += o.feature_ticks;
        self.forward_ticks += o.forward_ticks;
        self.forward_calls += o.forward_calls;
        if o.wall_secs > self.wall_secs {
            self.wall_secs = o.wall_secs;
        }
    }
}

fn run_thread(model: &Model, cfg: &Cfg, tid: usize) -> Metrics {
    if cfg.cpu >= 0 {
        let cpu = cfg.cpu as usize + tid;
        let ret = sys::set_affinity(cpu);
        if ret != 0 {
            eprintln!("thread {tid}: affinity cpu {cpu} failed ret {ret}");
        }
    }
    let actual_cpu = sys::get_cpu();
    let mut rng = Rng::new(cfg.seed ^ ((tid as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15)));
    let mut envs: Vec<Env> = (0..cfg.batch)
        .map(|_| Env::new(&mut rng, cfg.max_steps))
        .collect();
    let mut scratch = Scratch::new(model, cfg.batch);
    let mut x = vec![0.0f32; cfg.batch * model.in_dim];
    let mut out = vec![0.0f32; cfg.batch * N_ACTIONS];
    let mut colors = vec![0usize; cfg.batch];
    let mut m = Metrics::default();
    let start = Instant::now();

    while m.episodes < cfg.episodes {
        let t0 = sys::rdtime();
        for i in 0..cfg.batch {
            colors[i] = build_features(
                model,
                &envs[i],
                &mut x[i * model.in_dim..(i + 1) * model.in_dim],
            );
        }
        let t1 = sys::rdtime();
        model.policy_batch(
            &x,
            cfg.batch,
            cfg.gain_scale,
            cfg.residual,
            cfg.quant,
            &mut scratch,
            &mut out,
        );
        let t2 = sys::rdtime();
        m.feature_ticks += t1.saturating_sub(t0);
        m.forward_ticks += t2.saturating_sub(t1);
        m.forward_calls += 1;

        for i in 0..cfg.batch {
            if envs[i].done {
                continue;
            }
            let a = Model::argmax(&out[i * N_ACTIONS..(i + 1) * N_ACTIONS]);
            m.color_total += 1;
            if colors[i] == envs[i].true_color {
                m.color_correct += 1;
            }
            m.steps += 1;
            if envs[i].step(a) {
                m.episodes += 1;
                m.collisions += envs[i].collisions as u64;
                if envs[i].success {
                    m.success += 1;
                } else if envs[i].wrong {
                    m.wrong += 1;
                } else {
                    m.timeout += 1;
                }
                if m.episodes >= cfg.episodes {
                    break;
                }
                envs[i] = Env::new(&mut rng, cfg.max_steps);
            }
        }
    }
    m.wall_secs = start.elapsed().as_secs_f64();
    eprintln!(
        "thread {tid} cpu {actual_cpu} episodes {} success {} steps {} wall {:.3}s",
        m.episodes, m.success, m.steps, m.wall_secs
    );
    m
}

fn print_map(env: &Env) {
    for y in 0..N {
        let mut line = String::new();
        for x in 0..N {
            let idx = y * N + x;
            let mut ch = if env.map.wall[idx] != 0 { '#' } else { '.' };
            for c in 0..3 {
                if env.map.goals[c] == (x, y) {
                    ch = ['r', 'g', 'b'][c];
                }
            }
            if env.robot == (x, y) {
                ch = 'R';
            }
            line.push(ch);
        }
        println!("{line}");
    }
}

fn run_demo(model: &Model, cfg: &Cfg) {
    let mut rng = Rng::new(cfg.seed);
    let mut scratch = Scratch::new(model, 1);
    let mut x = vec![0.0f32; model.in_dim];
    let mut out = [0.0f32; N_ACTIONS];
    for ep in 0..cfg.episodes.min(5) {
        let mut env = Env::new(&mut rng, cfg.max_steps);
        println!(
            "\n=== EPISODE {ep} ===\ninstruction: \"{}\"  true colour: {}  start: {:?}",
            env.instruction,
            crate::COLORS[env.true_color],
            env.robot
        );
        print_map(&env);
        let mut actions = Vec::new();
        while !env.done {
            let color = build_features(model, &env, &mut x);
            model.policy_batch(&x, 1, cfg.gain_scale, cfg.residual, cfg.quant, &mut scratch, &mut out);
            let a = Model::argmax(&out);
            actions.push(a);
            if actions.len() <= 8 || env.done {
                println!(
                    "step {:02}: colour={} action={} pos={:?}",
                    actions.len(),
                    crate::COLORS[color],
                    DIR_NAMES[a],
                    env.robot
                );
            }
            env.step(a);
        }
        println!(
            "result: success={} wrong_goal={} steps={} collisions={}",
            env.success, env.wrong, env.steps, env.collisions
        );
        println!(
            "actions: {}",
            actions
                .iter()
                .map(|a| DIR_NAMES[*a])
                .collect::<Vec<_>>()
                .join(" ")
        );
        print_map(&env);
    }
}

fn read_u32(d: &[u8], o: &mut usize) -> u32 {
    let v = u32::from_le_bytes([d[*o], d[*o + 1], d[*o + 2], d[*o + 3]]);
    *o += 4;
    v
}

fn read_f32(d: &[u8], o: &mut usize) -> f32 {
    let v = f32::from_le_bytes([d[*o], d[*o + 1], d[*o + 2], d[*o + 3]]);
    *o += 4;
    v
}

fn run_modelbench(model: &Model, cfg: &Cfg) {
    if cfg.cpu >= 0 {
        let ret = sys::set_affinity(cfg.cpu as usize);
        if ret != 0 {
            eprintln!("affinity cpu {} failed ret {ret}", cfg.cpu);
        }
    }
    let actual_cpu = sys::get_cpu();
    let mut rng = Rng::new(cfg.seed);
    let env = Env::new(&mut rng, cfg.max_steps);
    let mut x = vec![0.0f32; cfg.batch * model.in_dim];
    build_features(model, &env, &mut x[..model.in_dim]);
    for b in 1..cfg.batch {
        let (src, dst) = x.split_at_mut(b * model.in_dim);
        dst[..model.in_dim].copy_from_slice(&src[..model.in_dim]);
    }
    let mut scratch = Scratch::new(model, cfg.batch);
    let mut out = vec![0.0f32; cfg.batch * N_ACTIONS];
    for _ in 0..200 {
        model.policy_batch(&x, cfg.batch, cfg.gain_scale, cfg.residual, cfg.quant, &mut scratch, &mut out);
    }
    let w0 = Instant::now();
    let t0 = sys::rdtime();
    let mut checksum = 0.0f32;
    for _ in 0..cfg.iters {
        model.policy_batch(&x, cfg.batch, cfg.gain_scale, cfg.residual, cfg.quant, &mut scratch, &mut out);
        checksum += out[0];
    }
    let t1 = sys::rdtime();
    let elapsed = w0.elapsed().as_secs_f64();
    let calls = cfg.iters as f64;
    let samples = calls * cfg.batch as f64;
    let ticks = t1.saturating_sub(t0) as f64;
    println!(
        "VLA_MODELBENCH cpu={} batch={} iters={} quant={} ticks_per_call={:.1} ticks_per_sample={:.2} calls_per_sec={:.1} samples_per_sec={:.1} wall_secs={:.4} checksum={:.6}",
        actual_cpu,
        cfg.batch,
        cfg.iters,
        cfg.quant,
        ticks / calls,
        ticks / samples,
        calls / elapsed,
        samples / elapsed,
        elapsed,
        checksum
    );
}

fn run_selftest(model: &Model, cfg: &Cfg) {
    let d = TESTVEC;
    assert_eq!(&d[0..4], b"TVT3", "bad test-vector magic");
    let mut o = 4usize;
    let count = read_u32(d, &mut o) as usize;
    let in_dim = read_u32(d, &mut o) as usize;
    let bow_dim = read_u32(d, &mut o) as usize;
    assert_eq!(in_dim, model.in_dim);
    assert_eq!(bow_dim, BOW_DIM);
    let mut max_logit_err = 0.0f32;
    let mut max_color_err = 0.0f32;
    let mut scratch = Scratch::new(model, 1);
    let mut x = vec![0.0f32; model.in_dim];
    let mut logits = [0.0f32; N_ACTIONS];
    for _ in 0..count {
        let mut wall = [0u8; NN];
        wall.copy_from_slice(&d[o..o + NN]);
        o += NN;
        o += 3 * NN; // goal channels are not used by the policy
        let mut dist = [0i16; NN];
        for v in dist.iter_mut() {
            *v = i16::from_le_bytes([d[o], d[o + 1]]);
            o += 2;
        }
        let px = read_f32(d, &mut o);
        let py = read_f32(d, &mut o);
        let mut bow = [0.0f32; BOW_DIM];
        for v in bow.iter_mut() {
            *v = d[o] as f32;
            o += 1;
        }
        let mut expect = [0.0f32; N_ACTIONS];
        for v in expect.iter_mut() {
            *v = read_f32(d, &mut o);
        }
        let mut expect_color = [0.0f32; 3];
        for v in expect_color.iter_mut() {
            *v = read_f32(d, &mut o);
        }
        let rx = (px * (N - 1) as f32).round() as usize;
        let ry = (py * (N - 1) as f32).round() as usize;
        build_features_from(model, &wall, &dist, (rx, ry), &bow, &mut x);
        model.policy_batch(&x, 1, 4.0, 0.01, cfg.quant, &mut scratch, &mut logits);
        let mut cl = [0.0f32; 3];
        model.color_logits(&bow, &mut cl);
        for a in 0..N_ACTIONS {
            max_logit_err = max_logit_err.max((logits[a] - expect[a]).abs());
        }
        for c in 0..3 {
            max_color_err = max_color_err.max((cl[c] - expect_color[c]).abs());
        }
    }
    let tol = if cfg.quant { 3.0 } else { 1e-3 };
    let ok = max_logit_err < tol && max_color_err < 1e-3;
    println!(
        "SELFTEST {} samples={} quant={} max_logit_err={:.3e} max_color_err={:.3e}",
        if ok { "PASS" } else { "FAIL" },
        count,
        cfg.quant,
        max_logit_err,
        max_color_err
    );
}

fn main() {
    let cfg = parse_args();
    let model = Model::from_bytes(WEIGHTS);
    assert_eq!(model.bow_dim, BOW_DIM, "vocabulary size mismatch");
    eprintln!(
        "TinyVLA in_dim={} h1={} h2={} bow={} actions={} weights={}B",
        model.in_dim,
        model.h1,
        model.h2,
        model.bow_dim,
        model.n_actions,
        WEIGHTS.len()
    );
    match cfg.mode.as_str() {
        "demo" => run_demo(&model, &cfg),
        "modelbench" => run_modelbench(&model, &cfg),
        "selftest" => run_selftest(&model, &cfg),
        "bench" => {
            let results: Vec<Metrics> = std::thread::scope(|s| {
                let mut handles = Vec::new();
                for t in 0..cfg.threads {
                    let m = &model;
                    let c = &cfg;
                    handles.push(s.spawn(move || run_thread(m, c, t)));
                }
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
            let mut total = Metrics::default();
            for m in &results {
                total.merge(m);
            }
            let n = total.episodes.max(1) as f64;
            let steps = total.steps.max(1) as f64;
            let success = total.success as f64 / n;
            let wrong = total.wrong as f64 / n;
            let timeout = total.timeout as f64 / n;
            let color_acc = if total.color_total > 0 {
                total.color_correct as f64 / total.color_total as f64
            } else {
                0.0
            };
            let avg_steps = total.steps as f64 / n;
            let avg_collisions = total.collisions as f64 / n;
            let fwd_per_step = total.forward_ticks as f64 / steps;
            let feat_per_step = total.feature_ticks as f64 / steps;
            let fwd_per_call = if total.forward_calls > 0 {
                total.forward_ticks as f64 / total.forward_calls as f64
            } else {
                0.0
            };
            let steps_per_sec = total.steps as f64 / total.wall_secs.max(1e-9);
            println!(
                "VLA_RESULT mode=bench episodes={} threads={} batch={} quant={} success={:.4} wrong_goal={:.4} timeout={:.4} colour_acc={:.4} avg_steps={:.2} avg_collisions={:.3} steps_per_sec={:.1} feature_ticks_per_step={:.1} forward_ticks_per_step={:.1} forward_ticks_per_call={:.1} forward_calls={} wall_secs={:.3}",
                total.episodes,
                cfg.threads,
                cfg.batch,
                cfg.quant,
                success,
                wrong,
                timeout,
                color_acc,
                avg_steps,
                avg_collisions,
                steps_per_sec,
                feat_per_step,
                fwd_per_step,
                fwd_per_call,
                total.forward_calls,
                total.wall_secs
            );
            if cfg.json {
                println!(
                    "VLA_METRICS {{\"episodes\":{},\"threads\":{},\"batch\":{},\"quant\":{},\"success\":{:.6},\"wrong_goal\":{:.6},\"timeout\":{:.6},\"colour_acc\":{:.6},\"avg_steps\":{:.3},\"avg_collisions\":{:.3},\"steps_per_sec\":{:.3},\"feature_ticks_per_step\":{:.1},\"forward_ticks_per_step\":{:.1},\"forward_ticks_per_call\":{:.1},\"forward_calls\":{},\"wall_secs\":{:.6}}}",
                    total.episodes,
                    cfg.threads,
                    cfg.batch,
                    cfg.quant,
                    success,
                    wrong,
                    timeout,
                    color_acc,
                    avg_steps,
                    avg_collisions,
                    steps_per_sec,
                    feat_per_step,
                    fwd_per_step,
                    fwd_per_call,
                    total.forward_calls,
                    total.wall_secs
                );
            }
        }
        other => {
            eprintln!("unknown mode: {other}");
            std::process::exit(2);
        }
    }
}

// rv-bench — X100 vs A100 核矩阵乘基准 (K3 真板)
// 用法: rv-bench [cpu_id] [n] [iters]
//   cpu_id: 绑定的 hart 编号 (0-7=X100, 8-15=A100); 255 = 不绑
// 输出: RV_BENCH_RESULT cpu=<n> n=<n> scalar=<GFLOPS> vec=<GFLOPS> vlenb=<B>

use std::arch::asm;
use std::time::Instant;

fn vlenb() -> usize {
    // vlenb CSR(0xC22): 向量寄存器字节长度 (RVV)
    let v: usize;
    unsafe { asm!("csrr {}, 0xc22", out(reg) v) }
    v
}

fn rdcycle() -> u64 {
    // time CSR(0xC01) (rdtime 在 S 态映射 wallclock; 用户态读 cycle 需 mcounteren,
    // 尝试 cycle(0xC00), 失败风险已知 — 若全 0 则退回 time)
    let v: u64;
    unsafe { asm!("csrr {}, 0xc01", out(reg) v) }
    v
}

fn set_affinity(cpu: i32) -> Result<(), i32> {
    if cpu < 0 {
        return Ok(());
    }
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    unsafe { libc::CPU_SET(cpu as usize, &mut set) };
    let r = unsafe { libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) };
    if r != 0 {
        return Err(unsafe { *libc::__errno_location() });
    }
    Ok(())
}

/// 普通标量核 (i-k-j, autovec 可能命中; 用 black_box 防折叠)
#[inline(never)]
fn matmul_scalar(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    for i in 0..n {
        for k in 0..n {
            let aik = std::hint::black_box(a[i * n + k]);
            for j in 0..n {
                c[i * n + j] += aik * b[k * n + j];
            }
        }
    }
}

/// 分块核 (4x4 分块提升寄存器复用, 利于编译器向量化)
#[inline(never)]
fn matmul_blocked(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    const BC: usize = 64;
    for ii in (0..n).step_by(BC) {
        for kk in (0..n).step_by(BC) {
            for jj in (0..n).step_by(BC) {
                let imax = (ii + BC).min(n);
                let kmax = (kk + BC).min(n);
                let jmax = (jj + BC).min(n);
                for i in ii..imax {
                    for k in kk..kmax {
                        let aik = std::hint::black_box(a[i * n + k]);
                        let crow = &mut c[i * n..];
                        let brow = &b[k * n..];
                        for j in jj..jmax {
                            crow[j] += aik * brow[j];
                        }
                    }
                }
            }
        }
    }
}

fn bench<F: Fn(&[f32], &[f32], &mut [f32], usize)>(name: &str, f: F, cpu: i32, n: usize, iters: usize) {
    let mut a = vec![0.5f32; n * n];
    let mut b = vec![0.25f32; n * n];
    let mut c = vec![0f32; n * n];
    let _ = &mut a; let _ = &mut b;

    // 预热一次 (页错误/缓存)
    f(&a, &b, &mut c, n);

    let t0 = Instant::now();
    for _ in 0..iters {
        f(&a, &b, &mut c, n);
    }
    let dt = t0.elapsed().as_secs_f64();
    let gflops = 2.0 * n as f64 * n as f64 * n as f64 * iters as f64 / dt / 1e9;
    let checksum: f32 = c.iter().sum::<f32>() * 0.0 + c.iter().sum::<f32>(); // 防 DCE, 不打印大数
    println!(
        "RV_BENCH_RESULT cpu={} kernel={} n={} iters={} time={:.3}s gflops={:.3} chk={:.1}",
        if cpu < 0 { -1 } else { cpu },
        name, n, iters, dt, gflops, checksum
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // 探针模式: 连续采样 (getcpu, vlenb) 跳变 + rdtime 估算实际主频
    if args.get(1).map(|s| s.as_str()) == Some("probe") {
        let cpu: i32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(-1);
        let _ = set_affinity(cpu);
        let mut last = (unsafe { libc::sched_getcpu() }, vlenb() as i32);
        println!("PROBE_START pinned={} cpu={} vlenb={}", cpu, last.0, last.1);
        let mut transitions = 0;
        for _ in 0..2_000_000 {
            let cur = (unsafe { libc::sched_getcpu() }, vlenb() as i32);
            if cur != last {
                transitions += 1;
                println!("PROBE_JUMP {} -> {:?}", transitions, cur);
                last = cur;
            }
        }
        println!("PROBE_TRANSITIONS {}", transitions);
        // rdtime 主频估算: 数一段固定循环的 time 增量
        let t0 = rdcycle();
        let mut x: u64 = 0;
        for i in 0..5_000_000u64 {
            x = x.wrapping_add(i ^ (x >> 3));
        }
        std::hint::black_box(x);
        let t1 = rdcycle();
        // 5M 次迭代 ≈ 5M 条加法/移位 ≈ 10M "周期级"操作; 只报原始增量, 留给离线换算
        println!("PROBE_TIME_DELTA {} ticks pin={} cpu={}", t1.wrapping_sub(t0), cpu, unsafe { libc::sched_getcpu() });
        println!("PROBE_DONE");
        return;
    }

    let cpu: i32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(-1);
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(256);
    let iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);

    println!("RV_BENCH_START cpu_arg={} n={} iters={}", cpu, n, iters);
    let vb = vlenb();
    println!("RV_BENCH_VLENB {} (vlen={}b)", vb, vb * 8);

    match set_affinity(cpu) {
        Ok(()) => {
            let cur = unsafe { libc::sched_getcpu() };
            println!("RV_BENCH_AFFINITY requested={} current_cpu={}", cpu, cur);
        }
        Err(e) => println!("RV_BENCH_AFFINITY_FAIL cpu={} errno={}", cpu, e),
    }

    bench("scalar", matmul_scalar, cpu, n, iters);
    bench("blocked", matmul_blocked, cpu, n, iters);
    println!("RV_BENCH_DONE");
}

//! Grid world, BFS distance fields, and the VLA observation builder.

use crate::model::{Model, N};

pub const NN: usize = 16 * 16;
pub const DX: [i32; 4] = [0, 0, -1, 1];
pub const DY: [i32; 4] = [-1, 1, 0, 0];
pub const DIR_NAMES: [&str; 4] = ["up", "down", "left", "right"];

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) | 1)
    }
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    #[inline]
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

fn largest_component(wall: &[u8; NN]) -> Vec<usize> {
    let mut seen = [false; NN];
    let mut best: Vec<usize> = Vec::new();
    let mut queue = [0u16; NN];
    for start in 0..NN {
        if wall[start] != 0 || seen[start] {
            continue;
        }
        let mut head = 0usize;
        let mut tail = 0usize;
        queue[tail] = start as u16;
        tail += 1;
        seen[start] = true;
        let mut comp = Vec::new();
        while head < tail {
            let idx = queue[head] as usize;
            head += 1;
            comp.push(idx);
            let x = (idx % N) as i32;
            let y = (idx / N) as i32;
            for d in 0..4 {
                let nx = x + DX[d];
                let ny = y + DY[d];
                if nx < 0 || ny < 0 || nx >= N as i32 || ny >= N as i32 {
                    continue;
                }
                let ni = ny as usize * N + nx as usize;
                if wall[ni] == 0 && !seen[ni] {
                    seen[ni] = true;
                    queue[tail] = ni as u16;
                    tail += 1;
                }
            }
        }
        if comp.len() > best.len() {
            best = comp;
        }
    }
    best
}

fn bfs(wall: &[u8; NN], goals: &[(usize, usize); 3], color: usize) -> [i16; NN] {
    let mut dist = [-1i16; NN];
    let mut queue = [0u16; NN];
    let (gx, gy) = goals[color];
    let gi = gy * N + gx;
    dist[gi] = 0;
    let mut head = 0usize;
    let mut tail = 1usize;
    queue[0] = gi as u16;
    while head < tail {
        let idx = queue[head] as usize;
        head += 1;
        let x = (idx % N) as i32;
        let y = (idx / N) as i32;
        for d in 0..4 {
            let nx = x + DX[d];
            let ny = y + DY[d];
            if nx < 0 || ny < 0 || nx >= N as i32 || ny >= N as i32 {
                continue;
            }
            let ni = ny as usize * N + nx as usize;
            if wall[ni] != 0 || dist[ni] >= 0 {
                continue;
            }
            let mut blocked = false;
            for c in 0..3 {
                if c != color && goals[c].0 == nx as usize && goals[c].1 == ny as usize {
                    blocked = true;
                    break;
                }
            }
            if blocked {
                continue;
            }
            dist[ni] = dist[idx] + 1;
            queue[tail] = ni as u16;
            tail += 1;
        }
    }
    dist
}

#[derive(Clone, Copy)]
pub struct Map {
    pub wall: [u8; NN],
    pub goals: [(usize, usize); 3],
    pub target: usize,
    pub start: (usize, usize),
    pub dist_by_color: [[i16; NN]; 3],
}

fn choose_goals(comp: &[usize], rng: &mut Rng) -> Option<[(usize, usize); 3]> {
    let mut goals: Vec<(usize, usize)> = Vec::with_capacity(3);
    for _ in 0..800 {
        let idx = comp[rng.below(comp.len())];
        let x = idx % N;
        let y = idx / N;
        if goals.iter().any(|g| *g == (x, y)) {
            continue;
        }
        if goals
            .iter()
            .all(|g| (g.0 as i32 - x as i32).abs() + (g.1 as i32 - y as i32).abs() >= 3)
        {
            goals.push((x, y));
        }
        if goals.len() == 3 {
            return Some([goals[0], goals[1], goals[2]]);
        }
    }
    None
}

impl Map {
    pub fn generate(rng: &mut Rng) -> Self {
        for _ in 0..200 {
            let mut raw = [0u8; NN];
            let nrect = 4 + rng.below(5);
            for _ in 0..nrect {
                let x0 = rng.below(N);
                let y0 = rng.below(N);
                let w = 1 + rng.below(4);
                let h = 1 + rng.below(4);
                for y in y0..(y0 + h).min(N) {
                    for x in x0..(x0 + w).min(N) {
                        raw[y * N + x] = 1;
                    }
                }
            }
            let comp = largest_component(&raw);
            if comp.len() < 90 {
                continue;
            }
            let mut wall = [1u8; NN];
            for idx in &comp {
                wall[*idx] = 0;
            }
            let goals = match choose_goals(&comp, rng) {
                Some(g) => g,
                None => continue,
            };
            let target = rng.below(3);
            let dist_by_color = [
                bfs(&wall, &goals, 0),
                bfs(&wall, &goals, 1),
                bfs(&wall, &goals, 2),
            ];
            let mut candidates: Vec<usize> = Vec::new();
            for idx in comp.iter().copied() {
                let x = idx % N;
                let y = idx / N;
                if goals.iter().any(|g| *g == (x, y)) {
                    continue;
                }
                if dist_by_color[target][idx] >= 6 {
                    candidates.push(idx);
                }
            }
            if candidates.is_empty() {
                continue;
            }
            let start_idx = candidates[rng.below(candidates.len())];
            return Self {
                wall,
                goals,
                target,
                start: (start_idx % N, start_idx / N),
                dist_by_color,
            };
        }
        panic!("map generation failed");
    }
}

pub struct Env {
    pub map: Map,
    pub robot: (usize, usize),
    pub bow: [f32; crate::VOCAB.len()],
    pub tokens: [&'static str; 5],
    pub true_color: usize,
    pub steps: u32,
    pub max_steps: u32,
    pub done: bool,
    pub success: bool,
    pub wrong: bool,
    pub collisions: u32,
    pub instruction: String,
}

impl Env {
    pub fn new(rng: &mut Rng, max_steps: u32) -> Self {
        let map = Map::generate(rng);
        let true_color = rng.below(3);
        let verb = crate::VERBS[rng.below(crate::VERBS.len())];
        let prep = crate::PREPS[rng.below(crate::PREPS.len())];
        let noun = crate::NOUNS[rng.below(crate::NOUNS.len())];
        let tokens = [verb, prep, "the", crate::COLORS[true_color], noun];
        let mut bow = [0.0f32; crate::VOCAB.len()];
        for t in tokens.iter() {
            if let Some(i) = crate::VOCAB.iter().position(|v| v == t) {
                bow[i] = 1.0;
            }
        }
        let instruction = tokens.join(" ");
        Self {
            map,
            robot: map.start,
            bow,
            tokens,
            true_color,
            steps: 0,
            max_steps,
            done: false,
            success: false,
            wrong: false,
            collisions: 0,
            instruction,
        }
    }

    pub fn dist_for(&self, color: usize) -> &[i16; NN] {
        &self.map.dist_by_color[color]
    }

    /// Returns true when the episode just finished.
    pub fn step(&mut self, action: usize) -> bool {
        if self.done {
            return true;
        }
        self.steps += 1;
        let x = self.robot.0 as i32 + DX[action];
        let y = self.robot.1 as i32 + DY[action];
        if x < 0 || y < 0 || x >= N as i32 || y >= N as i32 {
            self.collisions += 1;
            if self.steps >= self.max_steps {
                self.done = true;
            }
            return self.done;
        }
        let nx = x as usize;
        let ny = y as usize;
        let idx = ny * N + nx;
        if self.map.wall[idx] != 0 {
            self.collisions += 1;
            if self.steps >= self.max_steps {
                self.done = true;
            }
            return self.done;
        }
        let (tx, ty) = self.map.goals[self.true_color];
        if (nx, ny) == (tx, ty) {
            self.success = true;
            self.done = true;
            return true;
        }
        for c in 0..3 {
            if c != self.true_color && self.map.goals[c] == (nx, ny) {
                self.wrong = true;
                self.done = true;
                return true;
            }
        }
        self.robot = (nx, ny);
        if self.steps >= self.max_steps {
            self.done = true;
        }
        self.done
    }
}

/// Build the 640-float observation and return the colour selected by the
/// language head.  `dist` is the distance field for the selected colour.
pub fn build_features_from(
    model: &Model,
    wall: &[u8; NN],
    dist: &[i16; NN],
    robot: (usize, usize),
    bow: &[f32],
    x: &mut [f32],
) -> usize {
    debug_assert_eq!(x.len(), model.in_dim);
    let mut cl = [0.0f32; 3];
    model.color_logits(bow, &mut cl);
    let color = Model::argmax(&cl);

    // global wall
    for i in 0..NN {
        x[i] = wall[i] as f32;
    }
    // global normalised distance
    for i in 0..NN {
        x[NN + i] = if dist[i] >= 0 {
            dist[i] as f32 / 4.0
        } else {
            16.0
        };
    }
    // local views
    let (rx, ry) = robot;
    let mut local_wall_off = 2 * NN;
    let mut local_dist_off = 2 * NN + 49;
    for ly in 0..7i32 {
        for lx in 0..7i32 {
            let sx = rx as i32 + lx - 3;
            let sy = ry as i32 + ly - 3;
            let mut wv = 0.0f32;
            let mut dv = 16.0f32;
            if sx >= 0 && sy >= 0 && sx < N as i32 && sy < N as i32 {
                let idx = sy as usize * N + sx as usize;
                wv = wall[idx] as f32;
                if dist[idx] >= 0 {
                    dv = dist[idx] as f32 / 4.0;
                }
            }
            x[local_wall_off] = wv;
            x[local_dist_off] = dv;
            local_wall_off += 1;
            local_dist_off += 1;
        }
    }
    let pos_off = 2 * NN + 98;
    x[pos_off] = rx as f32 / (N - 1) as f32;
    x[pos_off + 1] = ry as f32 / (N - 1) as f32;

    let lang_off = pos_off + 2;
    model.lang(bow, &mut x[lang_off..lang_off + model.lang_dim]);

    let nbd_off = lang_off + model.lang_dim;
    let nbw_off = nbd_off + 4;
    let gain_off = nbw_off + 4;
    for a in 0..4 {
        x[nbd_off + a] = 16.0;
        x[nbw_off + a] = 1.0;
        x[gain_off + a] = -16.0;
    }
    let cur = x[NN + ry * N + rx];
    for a in 0..4 {
        let nx = rx as i32 + DX[a];
        let ny = ry as i32 + DY[a];
        if nx < 0 || ny < 0 || nx >= N as i32 || ny >= N as i32 {
            continue;
        }
        let idx = ny as usize * N + nx as usize;
        if wall[idx] != 0 {
            continue;
        }
        x[nbw_off + a] = 0.0;
        if dist[idx] >= 0 {
            let d = dist[idx] as f32 / 4.0;
            x[nbd_off + a] = d;
            x[gain_off + a] = cur - d;
        }
    }
    color
}

pub fn build_features(model: &Model, env: &Env, x: &mut [f32]) -> usize {
    let mut cl = [0.0f32; 3];
    model.color_logits(&env.bow, &mut cl);
    let color = Model::argmax(&cl);
    build_features_from(model, &env.map.wall, env.dist_for(color), env.robot, &env.bow, x);
    color
}

pub fn argmax4(x: &[f32]) -> usize {
    Model::argmax(x)
}

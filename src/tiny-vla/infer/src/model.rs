//! TinyVLA model: flat weight parser + MLP forward passes.

pub const N: usize = 16;
pub const N_ACTIONS: usize = 4;
pub const GAIN_SCALE: f32 = 4.0;
pub const TIE_BIAS: [f32; 4] = [0.0, -1e-4, -2e-4, -3e-4];

pub struct Mat {
    pub out: usize,
    pub inn: usize,
    /// Transposed to out-major: `w[j * inn + i]`.
    pub w: Vec<f32>,
    pub b: Vec<f32>,
}

#[cfg(target_feature = "v")]
#[inline(always)]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    use core::simd::f32x8;
    use std::simd::StdFloat;
    let n = a.len() / 8 * 8;
    let mut acc = f32x8::splat(0.0);
    let mut i = 0;
    while i < n {
        let va = f32x8::from_slice(&a[i..i + 8]);
        let vb = f32x8::from_slice(&b[i..i + 8]);
        acc = va.mul_add(vb, acc);
        i += 8;
    }
    let mut s: f32 = acc.to_array().iter().sum();
    while i < a.len() {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

#[cfg(not(target_feature = "v"))]
#[inline(always)]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    let mut i = 0;
    while i < a.len() {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

impl Mat {
    fn from_row_major(raw: &[f32], inn: usize, out: usize, bias: &[f32]) -> Self {
        let mut w = vec![0.0f32; inn * out];
        for i in 0..inn {
            for j in 0..out {
                w[j * inn + i] = raw[i * out + j];
            }
        }
        Self {
            out,
            inn,
            w,
            b: bias.to_vec(),
        }
    }

    #[inline]
    pub fn mv(&self, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!(x.len(), self.inn);
        for j in 0..self.out {
            let row = &self.w[j * self.inn..(j + 1) * self.inn];
            out[j] = self.b[j] + dot(x, row);
        }
    }

    #[inline]
    pub fn mv_batch(&self, x: &[f32], batch: usize, out: &mut [f32]) {
        if batch == 1 {
            self.mv(x, out);
            return;
        }
        // Weight-row outer loop: the same row is reused across the batch.
        for j in 0..self.out {
            let row = &self.w[j * self.inn..(j + 1) * self.inn];
            for bi in 0..batch {
                let xb = &x[bi * self.inn..(bi + 1) * self.inn];
                let mut acc = self.b[j];
                for i in 0..self.inn {
                    acc += xb[i] * row[i];
                }
                out[bi * self.out + j] = acc;
            }
        }
    }
}


/// Int8 quantised matrix: per-output-row symmetric scale, i32 accumulation.
pub struct QMat {
    pub out: usize,
    pub inn: usize,
    pub wq: Vec<i8>,
    pub scale: Vec<f32>,
    pub b: Vec<f32>,
}

#[inline(always)]
fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    let mut s = 0i32;
    let mut i = 0usize;
    while i < a.len() {
        s += (a[i] as i32) * (b[i] as i32);
        i += 1;
    }
    s
}

impl QMat {
    fn from_mat(m: &Mat) -> Self {
        let mut wq = vec![0i8; m.inn * m.out];
        let mut scale = vec![0.0f32; m.out];
        for j in 0..m.out {
            let row = &m.w[j * m.inn..(j + 1) * m.inn];
            let max = row.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            let s = if max > 0.0 { max / 127.0 } else { 1.0 };
            scale[j] = s;
            let inv = 1.0 / s;
            for i in 0..m.inn {
                wq[j * m.inn + i] =
                    (row[i] * inv).round().clamp(-127.0, 127.0) as i8;
            }
        }
        Self {
            out: m.out,
            inn: m.inn,
            wq,
            scale,
            b: m.b.clone(),
        }
    }

    #[inline]
    pub fn mv_i8(&self, xq: &[i8], x_scale: f32, out: &mut [f32]) {
        for j in 0..self.out {
            let row = &self.wq[j * self.inn..(j + 1) * self.inn];
            let acc = dot_i8(xq, row);
            out[j] = acc as f32 * x_scale * self.scale[j] + self.b[j];
        }
    }
}

#[inline]
fn quantize(src: &[f32], dst: &mut [i8]) -> f32 {
    let max = src.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let s = if max > 0.0 { max / 127.0 } else { 1.0 };
    let inv = 1.0 / s;
    for i in 0..src.len() {
        dst[i] = (src[i] * inv).round().clamp(-127.0, 127.0) as i8;
    }
    s
}

pub struct Model {
    pub in_dim: usize,
    pub h1: usize,
    pub h2: usize,
    pub n_actions: usize,
    pub lang_dim: usize,
    pub bow_dim: usize,
    pub color_dim: usize,
    pub wl: Mat,
    pub wc: Mat,
    pub w1: Mat,
    pub w2: Mat,
    pub w3: Mat,
    pub q1: QMat,
    pub q2: QMat,
    pub q3: QMat,
}

fn u32_at(d: &[u8], off: &mut usize) -> u32 {
    let v = u32::from_le_bytes([d[*off], d[*off + 1], d[*off + 2], d[*off + 3]]);
    *off += 4;
    v
}

fn f32s(d: &[u8], off: &mut usize, n: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        v.push(f32::from_le_bytes([
            d[*off],
            d[*off + 1],
            d[*off + 2],
            d[*off + 3],
        ]));
        *off += 4;
    }
    v
}

impl Model {
    pub fn from_bytes(d: &[u8]) -> Self {
        assert_eq!(&d[0..4], b"TVLA", "bad weight magic");
        let mut off = 4usize;
        let version = u32_at(d, &mut off);
        assert_eq!(version, 3, "unsupported weight version");
        let in_dim = u32_at(d, &mut off) as usize;
        let h1 = u32_at(d, &mut off) as usize;
        let h2 = u32_at(d, &mut off) as usize;
        let n_actions = u32_at(d, &mut off) as usize;
        let lang_dim = u32_at(d, &mut off) as usize;
        let bow_dim = u32_at(d, &mut off) as usize;
        let color_dim = u32_at(d, &mut off) as usize;

        let wl = f32s(d, &mut off, bow_dim * lang_dim);
        let bl = f32s(d, &mut off, lang_dim);
        let wc = f32s(d, &mut off, bow_dim * color_dim);
        let bc = f32s(d, &mut off, color_dim);
        let w1 = f32s(d, &mut off, in_dim * h1);
        let b1 = f32s(d, &mut off, h1);
        let w2 = f32s(d, &mut off, h1 * h2);
        let b2 = f32s(d, &mut off, h2);
        let w3 = f32s(d, &mut off, h2 * n_actions);
        let b3 = f32s(d, &mut off, n_actions);

        let w1m = Mat::from_row_major(&w1, in_dim, h1, &b1);
        let w2m = Mat::from_row_major(&w2, h1, h2, &b2);
        let w3m = Mat::from_row_major(&w3, h2, n_actions, &b3);
        let q1 = QMat::from_mat(&w1m);
        let q2 = QMat::from_mat(&w2m);
        let q3 = QMat::from_mat(&w3m);
        Self {
            in_dim,
            h1,
            h2,
            n_actions,
            lang_dim,
            bow_dim,
            color_dim,
            wl: Mat::from_row_major(&wl, bow_dim, lang_dim, &bl),
            wc: Mat::from_row_major(&wc, bow_dim, color_dim, &bc),
            w1: w1m,
            w2: w2m,
            w3: w3m,
            q1,
            q2,
            q3,
        }
    }

    /// Language encoder: relu(bow @ Wl + bl).
    #[inline]
    pub fn lang(&self, bow: &[f32], out: &mut [f32]) {
        self.wl.mv(bow, out);
        for v in out.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
    }

    /// Colour logits: bow @ Wc + bc.
    #[inline]
    pub fn color_logits(&self, bow: &[f32], out: &mut [f32]) {
        self.wc.mv(bow, out);
    }

    #[inline]
    pub fn argmax(x: &[f32]) -> usize {
        let mut best = 0usize;
        for i in 1..x.len() {
            if x[i] > x[best] {
                best = i;
            }
        }
        best
    }
}

pub struct Scratch {
    pub h1: Vec<f32>,
    pub h2: Vec<f32>,
    pub logits: Vec<f32>,
    pub xq: Vec<i8>,
    pub h1q: Vec<i8>,
    pub h2q: Vec<i8>,
}

impl Scratch {
    pub fn new(m: &Model, batch: usize) -> Self {
        Self {
            h1: vec![0.0; batch * m.h1],
            h2: vec![0.0; batch * m.h2],
            logits: vec![0.0; batch * m.n_actions],
            xq: vec![0i8; batch * m.in_dim],
            h1q: vec![0i8; batch * m.h1],
            h2q: vec![0i8; batch * m.h2],
        }
    }
}

impl Model {
    /// Batched policy forward.  `x` is `batch * in_dim`; `out` is
    /// `batch * n_actions`.  The gain values are the last four features.
    #[inline]
    pub fn policy_batch(
        &self,
        x: &[f32],
        batch: usize,
        gain_scale: f32,
        residual_scale: f32,
        quant: bool,
        scratch: &mut Scratch,
        out: &mut [f32],
    ) {
        if quant {
            self.policy_batch_quant(x, batch, gain_scale, residual_scale, scratch, out);
            return;
        }
        self.w1.mv_batch(x, batch, &mut scratch.h1);
        for v in scratch.h1.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
        self.w2
            .mv_batch(&scratch.h1, batch, &mut scratch.h2);
        for v in scratch.h2.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
        self.w3
            .mv_batch(&scratch.h2, batch, &mut scratch.logits);
        let gain_off = self.in_dim - 4;
        for bi in 0..batch {
            let g = &x[bi * self.in_dim + gain_off..bi * self.in_dim + gain_off + 4];
            for a in 0..self.n_actions {
                out[bi * self.n_actions + a] = gain_scale * g[a]
                    + residual_scale * scratch.logits[bi * self.n_actions + a]
                    + TIE_BIAS[a];
            }
        }
    }

    #[inline]
    pub fn policy_batch_quant(
        &self,
        x: &[f32],
        batch: usize,
        gain_scale: f32,
        residual_scale: f32,
        scratch: &mut Scratch,
        out: &mut [f32],
    ) {
        for bi in 0..batch {
            let xb = &x[bi * self.in_dim..(bi + 1) * self.in_dim];
            let xs = quantize(xb, &mut scratch.xq[bi * self.in_dim..(bi + 1) * self.in_dim]);
            self.q1.mv_i8(
                &scratch.xq[bi * self.in_dim..(bi + 1) * self.in_dim],
                xs,
                &mut scratch.h1[bi * self.h1..(bi + 1) * self.h1],
            );
            for v in scratch.h1[bi * self.h1..(bi + 1) * self.h1].iter_mut() {
                if *v < 0.0 {
                    *v = 0.0;
                }
            }
            let h1s = quantize(
                &scratch.h1[bi * self.h1..(bi + 1) * self.h1],
                &mut scratch.h1q[bi * self.h1..(bi + 1) * self.h1],
            );
            self.q2.mv_i8(
                &scratch.h1q[bi * self.h1..(bi + 1) * self.h1],
                h1s,
                &mut scratch.h2[bi * self.h2..(bi + 1) * self.h2],
            );
            for v in scratch.h2[bi * self.h2..(bi + 1) * self.h2].iter_mut() {
                if *v < 0.0 {
                    *v = 0.0;
                }
            }
            let h2s = quantize(
                &scratch.h2[bi * self.h2..(bi + 1) * self.h2],
                &mut scratch.h2q[bi * self.h2..(bi + 1) * self.h2],
            );
            self.q3.mv_i8(
                &scratch.h2q[bi * self.h2..(bi + 1) * self.h2],
                h2s,
                &mut scratch.logits[bi * self.n_actions..(bi + 1) * self.n_actions],
            );
            let gain_off = self.in_dim - 4;
            let g = &x[bi * self.in_dim + gain_off..bi * self.in_dim + gain_off + 4];
            for a in 0..self.n_actions {
                out[bi * self.n_actions + a] = gain_scale * g[a]
                    + residual_scale * scratch.logits[bi * self.n_actions + a]
                    + TIE_BIAS[a];
            }
        }
    }
}

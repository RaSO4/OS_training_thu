//! k3_ggml_bridge —— llama.cpp/ggml F32 MatMul 桥接核心路径（171c70e 旧 runtime API 适配版）。
//!
//! 上游来源: k3x-HERA-RT @88ae696 k3_ggml_bridge/src/lib.rs。
//! 适配说明:
//! - 上游按 2026-09 的新 frontend API 写(DimSize/ElemStride/OpFlags/TensorCount/UserToken 包装)。
//! - 本版本配对的 runtime 是 2026-06-25 的旧 API: MatMulAttr 字段为裸整数,
//!   AiKernelDesc::new 直接接收整数 tensor 计数, submit_entry 接收裸 u32 token。
//! - 仅保留 F32 MatMul 桥接路径(k3_ggml_matmul_f32); 上游的整图 k3_ggml_graph_compute
//!   入口依赖新版 AiKernelDesc::new_with_op, 旧 API 下由 k3_test 的 DAG 测试覆盖等价语义。
//! - AiTargetHint 在新旧 API 中均为 pub struct AiTargetHint(pub u8) + is_known(), 无需改动。

use std::{
    slice,
    sync::{
        Mutex,
        atomic::{AtomicU32, Ordering},
    },
};

use k3_ai_runtime::fronted::{
    AiDtype, AiTargetHint, GraphManager, MatMulAttr, TensorManager,
    kd_uring::{UringChannel, build_channel, submit_graph, wait_graph_complete},
};

/// ggml 侧成功返回码。
const K3_GGML_OK: i32 = 1;
/// ggml 侧失败返回码。
const K3_GGML_FAILED: i32 = 0;

/// 表示 RHS 逻辑矩阵按转置存储的 MatMul flag。
const MATMUL_RHS_TRANSPOSED: u32 = 1 << 1;

/// 惰性建立的 runtime 提交 channel。
static CHANNEL: Mutex<Option<UringChannel>> = Mutex::new(None);
/// 桥接单算子图的递增 token 源。
static NEXT_TOKEN: AtomicU32 = AtomicU32::new(0x4B33_0000);

/// 展平的 F32 MatMul 请求(C ABI 入口参数)。
#[repr(C)]
pub struct K3GgmlMatmulF32 {
    /// 左矩阵基址。
    pub lhs: *const f32,
    /// 右矩阵基址。
    pub rhs: *const f32,
    /// 输出矩阵基址。
    pub out: *mut f32,
    /// 输出行数 m。
    pub m: u32,
    /// 输出列数 n。
    pub n: u32,
    /// 归约维 k。
    pub k: u32,
    /// LHS 相邻行跨度(F32 元素数)。
    pub lhs_row_stride: u32,
    /// LHS 相邻列跨度(F32 元素数)。
    pub lhs_col_stride: u32,
    /// RHS 相邻行跨度(F32 元素数)。
    pub rhs_row_stride: u32,
    /// RHS 相邻列跨度(F32 元素数)。
    pub rhs_col_stride: u32,
    /// 输出相邻行跨度(F32 元素数)。
    pub out_row_stride: u32,
    /// 输出相邻列跨度(F32 元素数)。
    pub out_col_stride: u32,
    /// 与 backend attr 共享的 MatMul flag 位。
    pub flags: u32,
    /// K3 target hint 字节。
    pub target_hint: u8,
}

/// # Safety
///
/// `req` 必须指向合法的 K3GgmlMatmulF32; 其引用的缓冲区在调用返回前必须保持有效。
#[unsafe(no_mangle)]
pub unsafe extern "C" fn k3_ggml_matmul_f32(req: *const K3GgmlMatmulF32) -> i32 {
    if req.is_null() {
        return K3_GGML_FAILED;
    }
    // SAFETY: 上面已判空; C 调用方保证请求在调用期间有效。
    let req = unsafe { &*req };
    match run_matmul_f32(req) {
        Ok(()) => K3_GGML_OK,
        Err(_) => K3_GGML_FAILED,
    }
}

/// 校验、暂存、提交并回拷一次 F32 matmul 请求。
fn run_matmul_f32(req: &K3GgmlMatmulF32) -> Result<(), ()> {
    validate_request(req)?;

    let lhs_len = strided_len(req.m, req.k, req.lhs_row_stride, req.lhs_col_stride)?;
    let rhs_rows = if req.flags & MATMUL_RHS_TRANSPOSED != 0 {
        req.n
    } else {
        req.k
    };
    let rhs_cols = if req.flags & MATMUL_RHS_TRANSPOSED != 0 {
        req.k
    } else {
        req.n
    };
    let rhs_len = strided_len(rhs_rows, rhs_cols, req.rhs_row_stride, req.rhs_col_stride)?;
    let out_len = strided_len(req.m, req.n, req.out_row_stride, req.out_col_stride)?;
    let lhs_shape_len = u32::try_from(lhs_len).map_err(|_| ())?;
    let rhs_shape_len = u32::try_from(rhs_len).map_err(|_| ())?;
    let out_shape_len = u32::try_from(out_len).map_err(|_| ())?;

    // SAFETY: validate_request 已校验非空与维度; 调用方契约保证原始缓冲区覆盖跨度元素。
    let lhs_src = unsafe { slice::from_raw_parts(req.lhs, lhs_len) };
    // SAFETY: 同上; rhs_len 覆盖提交的 matmul 引用的全部 RHS 元素。
    let rhs_src = unsafe { slice::from_raw_parts(req.rhs, rhs_len) };
    // SAFETY: 同上, 且在回拷结果期间独占可变访问输出跨度。
    let out_dst = unsafe { slice::from_raw_parts_mut(req.out, out_len) };

    let tensor_mgr = TensorManager::new();
    let mut lhs = tensor_mgr.alloc_tensor(AiDtype::F32, &[lhs_shape_len]).map_err(|_| ())?;
    let mut rhs = tensor_mgr.alloc_tensor(AiDtype::F32, &[rhs_shape_len]).map_err(|_| ())?;
    let out = tensor_mgr.alloc_tensor(AiDtype::F32, &[out_shape_len]).map_err(|_| ())?;

    lhs.as_f32_mut_slice().copy_from_slice(lhs_src);
    rhs.as_f32_mut_slice().copy_from_slice(rhs_src);

    submit_single_matmul(req, &lhs, &rhs, &out)?;

    copy_output(req, out.as_f32_slice(), out_dst);
    Ok(())
}

/// ABI 层请求不变量检查。
fn validate_request(req: &K3GgmlMatmulF32) -> Result<(), ()> {
    if req.lhs.is_null()
        || req.rhs.is_null()
        || req.out.is_null()
        || req.m == 0
        || req.n == 0
        || req.k == 0
    {
        return Err(());
    }

    if (req.m > 1 && req.lhs_row_stride == 0)
        || (req.k > 1 && req.lhs_col_stride == 0)
        || (req.n > 1 && req.rhs_row_stride == 0 && req.flags & MATMUL_RHS_TRANSPOSED != 0)
        || (req.k > 1 && req.rhs_col_stride == 0 && req.flags & MATMUL_RHS_TRANSPOSED != 0)
        || (req.k > 1 && req.rhs_row_stride == 0 && req.flags & MATMUL_RHS_TRANSPOSED == 0)
        || (req.n > 1 && req.rhs_col_stride == 0 && req.flags & MATMUL_RHS_TRANSPOSED == 0)
        || (req.m > 1 && req.out_row_stride == 0)
        || (req.n > 1 && req.out_col_stride == 0)
    {
        return Err(());
    }

    if req.flags & !MATMUL_RHS_TRANSPOSED != 0 {
        return Err(());
    }

    let target = AiTargetHint(req.target_hint);
    if !target.is_known() {
        return Err(());
    }

    Ok(())
}

/// 提交只含一个 matmul 算子的 K3 graph。
fn submit_single_matmul(
    req: &K3GgmlMatmulF32,
    lhs: &k3_ai_runtime::fronted::Tensor,
    rhs: &k3_ai_runtime::fronted::Tensor,
    out: &k3_ai_runtime::fronted::Tensor,
) -> Result<(), ()> {
    let mut graph = GraphManager::new();
    // 旧内核 backend 不解释 flags, 只按 strides 寻址。RHS 按转置存储时,
    // 物理 (n 行, k 列) 布局映射到逻辑 (p, j) 下标需要交换行/列跨度,
    // 并在提交的 attr 中清除转置 flag。
    let transposed = req.flags & MATMUL_RHS_TRANSPOSED != 0;
    let (rhs_row_stride, rhs_col_stride) = if transposed {
        (req.rhs_col_stride, req.rhs_row_stride)
    } else {
        (req.rhs_row_stride, req.rhs_col_stride)
    };
    let attr = MatMulAttr {
        m: req.m,
        n: req.n,
        k: req.k,
        batch: 0,
        lhs_row_stride: req.lhs_row_stride,
        lhs_col_stride: req.lhs_col_stride,
        lhs_batch_stride: 0,
        rhs_row_stride,
        rhs_col_stride,
        rhs_batch_stride: 0,
        out_row_stride: req.out_row_stride,
        out_col_stride: req.out_col_stride,
        out_batch_stride: 0,
        flags: 0,
        accum_dtype: AiDtype::F32,
        reserved: [0; 3],
    };

    graph
        .push_kernel_no_depend(k3_ai_runtime::fronted::AiKernelDesc::new(
            &attr,
            AiTargetHint(req.target_hint),
            2,
            1,
            &[lhs.desc(), rhs.desc(), out.desc()],
        ))
        .map_err(|_| ())?;

    submit_frozen_graph(graph)
}

/// 冻结 graph 并经惰性建立的 channel 提交、等待完成。
fn submit_frozen_graph(graph: GraphManager) -> Result<(), ()> {
    let blob = graph.freeze().map_err(|_| ())?;
    let token = NEXT_TOKEN.fetch_add(1, Ordering::Relaxed);
    let entry = blob.submit_entry(token);
    let mut channel = CHANNEL.lock().map_err(|_| ())?;
    if channel.is_none() {
        *channel = Some(build_channel().map_err(|_| ())?);
    }
    let channel = channel.as_ref().ok_or(())?;
    submit_graph(channel, &entry).map_err(|_| ())?;
    wait_graph_complete(&entry, channel).map_err(|e| {
        eprintln!("k3_ggml_bridge: wait_graph_complete failed: {e:?}");
    })?;
    Ok(())
}

/// 把带跨度的 K3 matmul 输出拷回 ggml 内存。
fn copy_output(req: &K3GgmlMatmulF32, src: &[f32], dst: &mut [f32]) {
    let m = req.m as usize;
    let n = req.n as usize;
    let row_stride = req.out_row_stride as usize;
    let col_stride = req.out_col_stride as usize;

    for row in 0..m {
        for col in 0..n {
            let idx = row * row_stride + col * col_stride;
            dst[idx] = src[idx];
        }
    }
}

/// 覆盖一个带跨度二维矩阵所需的最小连续元素数。
fn strided_len(rows: u32, cols: u32, row_stride: u32, col_stride: u32) -> Result<usize, ()> {
    let rows = rows as usize;
    let cols = cols as usize;
    let row_stride = row_stride as usize;
    let col_stride = col_stride as usize;
    if rows == 0 || cols == 0 {
        return Err(());
    }
    let last = (rows - 1)
        .checked_mul(row_stride)
        .and_then(|v| v.checked_add((cols - 1).checked_mul(col_stride)?))
        .ok_or(())?;
    Ok(last + 1)
}

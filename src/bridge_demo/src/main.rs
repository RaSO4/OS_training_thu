//! bridge_demo —— 验证 k3_ggml_bridge 的 F32 MatMul 桥接核心路径。
//!
//! 模拟 ggml 后端的调用方式: 以裸指针 + 跨度构造展平请求, 走 C ABI 入口
//! k3_ggml_matmul_f32, 由 bridge 内部完成 tensor 暂存 -> k3_aiRuntime graph
//! 构建 -> /dev/k3_airunner 提交 -> 内核调度执行 -> completion -> 结果回拷。

use k3_ggml_bridge::{K3GgmlMatmulF32, k3_ggml_matmul_f32};

fn main() {
    let mut all_pass = true;

    // case 1: 普通行主序 2x3 @ 3x2 -> 2x2
    let lhs = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    let mut out = [0.0f32; 4];
    let expect = [58.0f32, 64.0, 139.0, 154.0];

    let req = K3GgmlMatmulF32 {
        lhs: lhs.as_ptr(),
        rhs: rhs.as_ptr(),
        out: out.as_mut_ptr(),
        m: 2,
        n: 2,
        k: 3,
        lhs_row_stride: 3,
        lhs_col_stride: 1,
        rhs_row_stride: 2,
        rhs_col_stride: 1,
        out_row_stride: 2,
        out_col_stride: 1,
        flags: 0,
        target_hint: 1, // PREFER_CPU
    };
    // SAFETY: 请求内三个指针均指向上方存活的本栈帧数组。
    let rc = unsafe { k3_ggml_matmul_f32(&req) };
    let ok = rc == 1 && out == expect;
    println!("bridge_demo case1 row-major 2x3@3x2: rc={} out={:?} expect={:?} => {}", rc, out, expect, if ok { "PASS" } else { "FAIL" });
    all_pass &= ok;

    // case 2: RHS 转置存储(flag=2), 同一数学结果
    let rhs_t = [7.0f32, 9.0, 11.0, 8.0, 10.0, 12.0]; // n x k = 2x3
    let mut out2 = [0.0f32; 4];

    let req2 = K3GgmlMatmulF32 {
        lhs: lhs.as_ptr(),
        rhs: rhs_t.as_ptr(),
        out: out2.as_mut_ptr(),
        m: 2,
        n: 2,
        k: 3,
        lhs_row_stride: 3,
        lhs_col_stride: 1,
        rhs_row_stride: 3,
        rhs_col_stride: 1,
        out_row_stride: 2,
        out_col_stride: 1,
        flags: 2, // MATMUL_RHS_TRANSPOSED
        target_hint: 1, // PREFER_CPU
    };
    // SAFETY: 同 case1, 指针指向存活的栈数组。
    let rc2 = unsafe { k3_ggml_matmul_f32(&req2) };
    let ok2 = rc2 == 1 && out2 == expect;
    println!("bridge_demo case2 rhs-transposed:    rc={} out={:?} expect={:?} => {}", rc2, out2, expect, if ok2 { "PASS" } else { "FAIL" });
    all_pass &= ok2;

    if all_pass {
        println!("bridge_demo: ALL PASS");
    } else {
        println!("bridge_demo: FAILED");
        std::process::exit(1);
    }
}

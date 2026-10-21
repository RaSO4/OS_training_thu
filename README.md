# 基于 Starry 的机器人应用

操作系统专题训练（2026 秋）大作业。基准项目 [k3x-HERA-RT](https://github.com/Dirinkbottle/k3x-HERA-RT)：在 SpacemiT K3（RISC-V，X100 通用核 + A100 AI 核）上把 AI 算力纳入 StarryOS 内核统一调度。本仓库覆盖两个阶段：QEMU 复现与真板（BananaPi BPI-SM10）实验。

## 结果

| 任务 | 状态 | 证据 |
| --- | --- | --- |
| #1 QEMU 启动 StarryOS 到 shell | ✅ | `docs/runs/boot.log` |
| #2 k3_airunner + k3_test 最小闭环（DAG→内核调度→数值验证） | ✅ | `docs/runs/k3_test.log` |
| #3 k3_ggml_bridge F32 MatMul 桥接（适配旧 API，修复转置 stride） | ✅ | `docs/runs/bridge_demo.log` |
| #4 真板 StarryOS 启动（TFTP+bootm，16 核全部上线） | ✅ | `docs/runs/board/starryos-first-boot.log` |
| #5 用户态 RVV 内核使能（sstatus.VS，修复真板 SIGILL） | ✅ | `patches/tgoskits-bananapi-heralab.patch` |
| #6 X100 vs A100 实测（VLEN/亲和性/GFLOPS） | ✅ | `docs/runs/board/x100-vs-a100-bench.txt` |
| #7 llama.cpp riscv64 musl 交叉编译（静态 llama-cli 9.7MB） | ✅ | `env/bin/build-llama-rv.sh`、`patches/llama-cpp-riscv-zig.patch` |
| #8 首次真板 LLM 推理（模型已上板，SD 降速内核验证中） | 🔧 | 见下"当前进度" |

## 布局

```
├── docs/runs/            # 运行证据（QEMU + 真板）
├── env/bin/              # 工具脚本：QEMU 复现(run-closed-loop)、真板操作(board-console/flash-*)、llama 构建
├── patches/              # 阶段一: tgoskits-dev / k3x-171c70e；阶段二: tgoskits-bananapi-heralab / llama-cpp-riscv-zig
└── src/                  # k3_ggml_bridge / bridge_demo（阶段一）、rv-bench（阶段二基准）
```

## 复现（QEMU）

Ubuntu 22.04 x86_64，无需 sudo 装包。

```bash
git clone https://github.com/Dirinkbottle/tgoskits.git && cd tgoskits && git checkout 4b23208 && cd ..
git clone https://github.com/Dirinkbottle/k3x-HERA-RT.git && cd k3x-HERA-RT && git checkout 171c70e && cd ..
cd tgoskits     && git apply <repo>/patches/tgoskits-dev.patch  && cd ..
cd k3x-HERA-RT  && git apply <repo>/patches/k3x-171c70e.patch   && cd ..
cp -r <repo>/src/k3_ggml_bridge <repo>/src/bridge_demo k3x-HERA-RT/
# 准备自包含工具链（Rust nightly-2026-05-28、qemu-system-riscv64、riscv64-musl 头文件，
# 见 env/bin/ 各脚本头部注释；rootfs 取 Starry-OS/rootfs release 20260214）
<repo>/env/bin/run-closed-loop.sh bg
```

## 复现（真板）

```bash
git clone -b bananapi https://github.com/rcore-os/tgoskits.git && cd tgoskits
git apply <repo>/patches/tgoskits-bananapi-heralab.patch
# 依赖: u-boot-tools device-tree-compiler；构建配置 max_cpu_num=16
cargo starry build --config os/StarryOS/configs/board/bananapi-bpi-sm10.toml
# 板卡操作（ostool 平台客户端，详见脚本注释）:
#   board-console.sh  — 串口会话（script+pty，支持 send/wait 驱动）
#   session-put.py    — 会话文件上传（配 U-Boot tftpboot 使用）
#   flash-rootfs.sh   — 自制 rootfs 一键刷写（上传→tftp→mmc write）
#   flash-model.sh    — 大文件分块刷写（>64MiB 自动切块、断点续传）
# 基准: /bin/rv-bench <cpu> 256 3   （matmul）；/bin/rv-bench probe <cpu>（VLEN/亲和性探针）
```

## 要点与结论

- **版本配对**：tgoskits@dev(4b23208) 必须配 k3x-HERA-RT@171c70e；QEMU 必须 `-smp 16`（worker 钉 CPU10）。
- **用户态 RVV**：内核原只设 sstatus.FS 不设 VS，真板上用户态向量指令一律 SIGILL（QEMU 不查 VS 故不暴露）；补丁在 axcpu 用户上下文初始化加 `VS::Initial`。暂未做每任务向量寄存器保存/恢复（实验性，钉核场景够用）。
- **X100 vs A100 实测**：亲和性真实有效（钉核后 200 万采样零跳变）；X100 VLEN 256 约 4 GFLOPS，A100 VLEN 1024 同代码仅 0.5-0.8 GFLOPS——A100 普通 RVV 反而慢约 5 倍。**结论：A100 提速必须走 IME 自定义指令**（llama.cpp 已有 `GGML_CPU_RISCV64_SPACEMIT` 路径，需 SpacemiT 定制 GCC≥15）。
- **平台坑**：会话约 5 分钟强制回收、单文件上传限 64MiB（分块解决）、U-Boot `mmc write` 参数须 0x 前缀十六进制、网口初始化抖动重上电即恢复。
- **SD 稳定性**：大流量读触发 CRC 错误（内核驱动缺 SpacemiT 延迟调谐），已用 ×4 分频保守修复；正式方案为移植调谐逻辑。

## 当前进度

模型（SmolLM2-135M Q4）与 llama-cli 已刷入真板 rootfs；SD 降速内核已编好待部署验证，通过后即可完成首次真板推理。下一步：真板推理基准 → SpacemiT GCC 的 IME 路线 → k3_airunner 真板移植 → VLA 组合栈。

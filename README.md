# 基于 Starry 的机器人应用 —— 复现实验仓库

操作系统专题训练（2026 秋）大作业。基准项目为 [k3x-HERA-RT](https://github.com/Dirinkbottle/k3x-HERA-RT)（文档：[dirinkbottle.github.io/k3x-HERA-RT](https://dirinkbottle.github.io/k3x-HERA-RT/)）：在 SpacemiT K3（RISC-V，X100 通用核 + A100 AI 核）上把 AI 算力纳入 StarryOS 内核统一调度。

本仓库在 **QEMU(riscv64)** 上复现了基准系统中不依赖真板硬件的部分，并完成 ggml F32 MatMul 桥接的版本适配。

## 复现结果

| 任务 | 内容 | 状态 | 证据 |
| --- | --- | --- | --- |
| #1 | StarryOS 在 QEMU 启动到 shell | ✅ | `docs/runs/boot.log` |
| #2 | /dev/k3_airunner + k3_test 最小闭环（建 DAG → 提交 → 内核调度 → CPU fallback → completion → 数值验证） | ✅ | `docs/runs/k3_test.log` |
| #3 | k3_ggml_bridge F32 MatMul 桥接路径（从上游 88ae696 提取并适配 171c70e 旧 API；修复 RHS 转置 stride 语义） | ✅ | `docs/runs/bridge_demo.log` |
| #4 | llama.cpp 小模型（加分项）/ A100、YOLOv5（真板） | 未做 | 等待真板 |

## 仓库布局

```
├── README.md               # 本文件
├── docs/runs/              # 运行证据日志（QEMU 实际输出）
├── env/
│   ├── env.sh              # 自包含构建环境入口（source 后使用）
│   └── bin/                # 工具 wrapper 与复现脚本
├── patches/
│   ├── tgoskits-dev.patch      # 对 tgoskits@dev 的适配改动
│   └── k3x-171c70e.patch       # 对 k3x-HERA-RT@171c70e 的适配改动
└── src/
    ├── k3_ggml_bridge/     # ggml F32 MatMul 桥接（核心新代码）
    └── bridge_demo/        # 桥接验证程序
```

## 复现步骤

前置：Linux x86_64（Ubuntu 22.04 实测），git、clang、curl、debugfs(e2fsprogs)；全程不需要 sudo 安装任何系统包。

```bash
# 1. 克隆两个上游仓库（必须为同级目录），锁定到配套版本
git clone https://github.com/Dirinkbottle/tgoskits.git
cd tgoskits && git checkout 4b23208 && cd ..
git clone https://github.com/Dirinkbottle/k3x-HERA-RT.git
cd k3x-HERA-RT && git checkout 171c70e && cd ..

# 2. 应用本仓库的适配补丁
cd tgoskits     && git apply <repo>/patches/tgoskits-dev.patch  && cd ..
cd k3x-HERA-RT  && git apply <repo>/patches/k3x-171c70e.patch   && cd ..

# 3. 把本仓库的新代码放进 k3x-HERA-RT（与上游 crate 并列）
cp -r <repo>/src/k3_ggml_bridge <repo>/src/bridge_demo k3x-HERA-RT/

# 4. 按环境脚本注释准备自包含工具链（Rust nightly-2026-05-28、
#    qemu-system-riscv64 9.x、riscv64-musl 头文件；详见 env/bin/ 各脚本头部注释）
#    rootfs 取自 https://github.com/Starry-OS/rootfs release 20260214

# 5. 一键复现（启动 StarryOS 并自动运行 bridge_demo）
<repo>/env/bin/run-closed-loop.sh bg   # 日志见 runs/ 目录
```

## 关键注意事项

- **版本配对**：tgoskits@dev(4b23208, 2026-07-01) 必须配 k3x-HERA-RT@171c70e(2026-06-25)；用 k3x 主分支（09-14）会因 K3SchedulerOps trait 改版编译失败。
- **QEMU 必须 `-smp 16`**：内核把 scheduler worker 亲和性固定在 CPU10（真板 AI 核），单核下线程永远不会被调度。
- **旧内核 backend 不解释 MatMul flags**：RHS 转置存储需在 bridge 里交换行/列 stride 再提交（见 `src/k3_ggml_bridge/src/lib.rs` 注释）。
- k3_test 图后的"共享区保活"阶段会挂起且影响后续提交，与 bridge_demo 分开两次启动运行。

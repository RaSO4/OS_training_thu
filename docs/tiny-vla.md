# TinyVLA：StarryOS 真板上的语言条件导航

一个从零训练的小型 VLA，跑在 BananaPi BPI-SM10（SpacemiT K3）的 StarryOS 上。
它看 16x16 的网格地图、读一句英文指令，输出机器人的移动方向；闭环成功率
99.8%，单个 X100 核每秒 22,196 次决策，8 个核并行 175,123 次。为了跑稳多进程
的向量负载，还给内核补上了每任务的 RVV 寄存器保存/恢复。

## 任务

网格世界里随机放 4 到 8 个矩形墙和红绿蓝三个目标，机器人从离目标至少 6 步
的位置出发。指令由 17 个词的词表拼成，比如 `navigate to the blue target`，
一共 180 种组合。动作是上下左右四个方向。走到指令说的颜色算成功，走错颜色
直接失败，64 步没到算超时。每一小步都要重新观测。

## 模型

```
指令 BoW(17) ─ 颜色头(17→3) ─ argmax 选目标颜色 ─ 选用哪个 BFS 距离场
指令 BoW(17) ─ 语言编码(17→16)

观测 640 维 = 墙体图 256 + 距离场 256 + 局部7x7墙体 49 + 局部7x7距离 49
            + 位置 2 + 语言嵌入 16 + 四邻域距离 4 + 四邻域可通行 4 + 距离增益 4

策略 MLP：640 → 128 → 64 → 4（约 90k 参数，f32 权重 364KB）
动作 logits = 4·gain + 0.01·MLP输出 + tie_bias
```

语言和导航是分开的：颜色头单独用交叉熵训练，选中哪个目标就用哪个目标的
距离场，语言上的判断错误不会连累底层走路的策略。距离场由 BFS 预先算好，
相当于一张代价图，网络不需要自己学寻路。动作头把距离增益当先验、MLP 输出
当残差，增益保证不会选远离目标的邻居，残差负责平局时的取舍。

消融（各 1000 回合）：

| 动作头 | 成功率 |
| --- | --- |
| 只有 MLP（增益=0） | 99.0% |
| 只有增益先验（残差=0） | 100% |
| 两者混合（默认） | 99.9% |

训练数据不足时差距很大：2,500 张地图、8 个 epoch 的早期版本里纯 MLP 只有
31%，混合头不受影响。

## 训练

12,000 张随机地图，BFS 专家（从目标反向搜，其他颜色的目标当墙）在每张图的
最短路径上取状态，共 198,418 条样本。行为克隆 24 个 epoch，再加两轮 DAgger，
每轮让当前策略自己 rollout、专家纠正约 31,500 条。优化器是手写的 numpy
反向传播加 Adam，没有用 torch。1,500 张留出地图上的结果：动作准确率
99.85%，颜色判断 100%，闭环 99.73%，平均 12.5 步，零碰撞。

## 板上运行

推理是 Rust 写的静态 riscv64-musl 二进制，权重、词表和 8 条测试向量全部
`include_bytes!` 进可执行文件，单文件约 820KB，没有运行时依赖。四个子命令：
`demo` 打印地图和逐步动作，`bench` 测闭环吞吐，`modelbench` 测纯前向延迟，
`selftest` 用训练时导出的测试向量校验 Rust 和 numpy 的前向是否一致（f32
误差小于 1.6e-5）。计时用 rdtime，绑核用 sched_setaffinity 的裸 ecall。

往板上放程序走的是 SD 裸扇区：用 `env/bin/flash-file.sh` 把内核刷到
LBA 0x500000、DTB 到 0x600000，之后 `env/bin/board-fastboot.sh` 让 U-Boot
直接 mmc read 加 booti，18 秒到 shell。

## 性能

优化分四步走，每步都有板上实测（X100 单核）：

| 阶段 | 改动 | 决策/秒 |
| --- | --- | --- |
| v1 | 编译器自动向量化（f32） | 3,461 |
| v2 | portable_simd 的 mul_add 生成 vfmacc，替代每段归约（f32） | 8,849 |
| v3 | int8 量化，vwmacc 累加 | 11,840 |
| v5 | 隐藏层 256/128 减到 128/64 | 22,196 |
| v5 x8 进程 | 每核一个进程，绑 cpu0-7 | 175,123 |

8 进程的扩展效率 98.6%，每个进程 500 回合全部成功。纯前向从 v1 的 6,809
ticks 降到 v5 的 836 ticks。相对最初的 v1，单进程 6.4 倍，8 进程 50.6 倍。
量化后 logits 和 f32 最多差 1.45，argmax 和闭环不受影响。详细的分阶段数据
在 `docs/runs/board/tiny-vla/`。

v1 到 v2 提升最大的原因值得记一笔：编译器自动向量化点积时用的是 vfmul 加
vfredosum，每 8 个元素归约一次；换成 portable_simd 的 mul_add 后循环体内
一直是 vfmacc 累加，最后归约一次，快了 2.56 倍。

## 内核：向量上下文

跑多进程向量负载时发现一个内核问题。此前的 RVV 补丁只把 sstatus.VS 设成
Initial，任务切换时不保存向量寄存器，于是一旦有第二个任务用过向量单元，
v0-v31 就会被互相覆盖。8 进程的 int8 负载随机 SIGSEGV，PC 飞到非法地址；
llama.cpp 的多线程版也是同一个原因崩。

修复在 `patches/tgoskits-bananapi-heralab.patch` 里（components/axcpu 的
riscv/vector.rs）。TaskContext 增加 VectorState，保存 v0-v31 外加 vcsr、
vtype、vl、vstart，切换时走和浮点状态一样的 Dirty/Clean/Initial 状态机。
寄存器搬运用 m1 逐个传，VLEN 无关，X100 的 256 位和 A100 的 1024 位都能用；
保存前先把 4KB 镜像清零，窄 VLEN 写不到的高位不会漏出上一个任务的数据。

修的过程有两课。第一版只存了寄存器、漏了 vtype/vl/vstart，任务恢复后向量
形状对不上，8 进程在 vwmacc 处 IllegalInstruction：这几个 CSR 是向量上下文
的一部分，光存寄存器不够。第二版把 CSR 补上了，却在保存汇编之后才去读，
而汇编开头一条 vsetvli 早已把它们改成临时布局，存下来的还是错的。改成先读
CSR 再存寄存器后，8 进程全部跑完，99.67% 成功。

## A100

A100 的普通 RVV 做不了这个负载：标量 int8 稳定但只有 1,800 决策/秒；向量的
纯前向不比 X100 慢（29,088 对 22,196 样本/秒），可是一进闭环或者多进程就
随机 PC 飞掉，和 k3x-HERA-RT 文档里记的 A100 偶发非法 stval 是同一类现象。
结论是 A100 要用 IME 指令而不是通用向量，这留作后续工作。

## 复现

```bash
# 训练（numpy 即可，约 3 分钟）
python3 src/tiny-vla/train/vla_train.py --worlds 12000 --epochs 24 \
    --h1 128 --h2 64 --out src/tiny-vla/weights

# 主机推理
cd src/tiny-vla/infer
cargo build --release --target x86_64-unknown-linux-gnu
target/x86_64-unknown-linux-gnu/release/tiny-vla --mode demo --episodes 2

# 交叉编译后按 docs/k3-remote-usage-guide.md 的流程刷上板
cargo build --release --target riscv64gc-unknown-linux-musl
# 板上：
/bin/vla-vector --mode selftest
/bin/vla-vector --mode bench --episodes 1000 --cpu 0
```

## 局限

视觉是符号化的网格地图，语言是封闭词表的 BoW，都不是真实传感器的输入。
距离场由 BFS 计算，属于系统里的经典模块而非网络的一部分。StarryOS 的进程
内线程启动有独立的问题（标量程序也崩），所以并行用的是多进程而不是多线程。

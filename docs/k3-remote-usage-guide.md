# 远程使用 K3 开发板帮助文档（新手 step-by-step）

> 适用对象：第一次使用课程 K3 开发板共享平台的同学
> 平台地址：https://ostool.muxai.net:14321/
> 板卡型号：BananaPi BPI-SM10（SpacemiT K3 SoC）
> 文档整理：hera-rt-lab 项目组（2026-10，全部坑均实测踩过）
> 仓库内相关文件：`env/bin/`（串口/刷写脚本）、`patches/tgoskits-bananapi-heralab.patch`（内核补丁）

---

## 目录

1. [平台与板卡简介](#1-平台与板卡简介)
2. [准备工作](#2-准备工作)
3. [安装 ostool 客户端](#3-安装-ostool-客户端)
4. [注册账号与申请板卡](#4-注册账号与申请板卡)
5. [登录认证](#5-登录认证)
6. [常用命令](#6-常用命令)
7. [交互使用串口终端](#7-交互使用串口终端)
8. [往板子传文件](#8-往板子传文件)
9. [在板上运行自己的系统（以 StarryOS 为例）](#9-在板上运行自己的系统以-starryos-为例)
10. [脚本化 / 自动化操作](#10-脚本化--自动化操作)
11. [常见问题与坑（必读）](#11-常见问题与坑必读)
12. [板卡硬件速查表](#12-板卡硬件速查表)
13. [参考链接](#13-参考链接)

---

## 1. 平台与板卡简介

课程开发板共享平台运行 **ostool-server**，你通过 **ostool 命令行客户端** 远程使用开发板：

```
你的服务器 ──HTTPS(认证)──> 平台网关 ──局域网──> 板卡服务器 ──串口/网络/TFTP──> K3 板子
```

- 你拿到的不是 ssh，而是**串口终端转发 + 文件服务 + 电源管理**
- 每次连接会话 = 板子**重新上电**
- 板子上电后停在 **U-Boot 提示符 `=>`**（板内没有常驻操作系统，系统镜像由你从网络加载）

板卡规格见[第 12 节](#12-板卡硬件速查表)。

---

## 2. 准备工作

需要一台能联网的 Linux 服务器（Ubuntu 22.04 实测可行）：

```bash
# 基础工具（一般都有）
sudo apt update
sudo apt install -y curl git build-essential pkg-config

# Rust 工具链（ostool 是 Rust 写的，用 cargo 安装）
# 国内走 rsproxy 镜像快
export RUSTUP_DIST_SERVER=https://rsproxy.cn
export RUSTUP_UPDATE_ROOT=https://rsproxy.cn/rustup
curl --proto '=https' --tlsv1.2 -sSf https://rsproxy.cn/rustup-init.sh | sh -s -- -y --default-toolchain stable
source "$HOME/.cargo/env"

# libudev 开发文件（ostool 编译依赖，Ubuntu 22.04 默认没有）
sudo apt install -y libudev-dev
```

> 不能用 apt / 想自包含安装：可以手写一个 libudev.pc 指向系统 libudev.so.1，但不要用 `-l:libudev.so.1` 写法，pkg-config 0.3.33 会把它变成 rustc-link-arg，对依赖 crate 不传递，最终链接报 undefined udev_*；正确做法是建 libudev.so 符号链接后用普通 `-ludev`。

---

## 3. 安装 ostool 客户端

```bash
cargo install ostool
ostool --version    # 应输出类似 ostool 0.31.0
```

然后写入全局配置 `~/.ostool/config.toml`（首次执行 ostool 命令也会自动生成默认模板再改）：

```toml
[board]
server = "https://ostool.muxai.net:14321"
auth_mode = "required"
```

> `server` 必须是**完整 URL**（带 `https://`）。公网认证模式强制 HTTPS，证书是 Let's Encrypt 的，无需额外配置。

---

## 4. 注册账号与申请板卡

1. **注册**：浏览器打开 https://ostool.muxai.net:14321/ ，注册个人账号（记住用户名密码，后面浏览器授权要用）。
2. **申请租约**：联系课程老师/助教，在你的租约池里分配一块板卡（管理后台操作）。租期信息之后可以用 `ostool board ls` 看到。
3. 分配好后 `ostool board ls` 会显示类似：

```
BOARD TYPE  AVAILABLE  TOTAL  TAGS  BOARD ID    DATE BEGIN           DATE END
BananaPi            1      1  -     BananaPi-2  2026-10-12 22:37:00  2026-11-12 22:37:00
```

> ⚠️ 注意：`BOARD ID` 列的显示偶尔与实际分配不符（我们遇到过显示 BananaPi-2、实际是 BananaPi-1 的情况）。**连接时不指定 `--board-id`，让服务器自动分配**最稳妥。

---

## 5. 登录认证

平台使用 OAuth 设备授权流程（浏览器登录一次，令牌长期自动续期）：

```bash
ostool login --server https://ostool.muxai.net:14321
```

输出类似：

```
Open this URL to sign in:
  https://ostool.muxai.net:14321/oauth/activate?user_code=C37W-JWPU
Code: C37W-JWPU
```

**在你自己电脑的浏览器**打开这个链接 → 用平台账号登录 → 点确认授权。终端会自动完成（打印 `Logged in successfully`）。

验证状态：

```bash
ostool auth status --server https://ostool.muxai.net:14321
# credential: OAuth  + expires_at 表示令牌会自动刷新，不用重复登录
```

> 令牌保存在 `~/.config/ostool/hosts.json`（系统 keyring 不可用时）。不要把它提交进任何仓库。

---

## 6. 常用命令

```bash
ostool board ls                          # 看板卡类型/可用数/租期
ostool board connect -b BananaPi         # 连接并打开串口终端（板子会上电复位）
ostool board run -c <构建配置.toml> --board-config <板级配置.toml>   # 自动构建+部署+启动+判活
```

- `-b BananaPi`：板卡类型名。**建议不写 `--board-id`**（见第 4 节注意事项）。
- 串口终端退出：按 `Ctrl+A`，松开，再按 `x`。
- 详细的 `board run` 用法见[第 9 节](#9-在板上运行自己的系统以-starryos-为例)。

---

## 7. 交互使用串口终端

连接后你会看到板子从 BootROM → U-Boot SPL → U-Boot 2022.10 一路滚屏，最后停在：

```
=>
```

这就是 U-Boot 提示符，可以输入 U-Boot 命令。常用：

```
=> version                 # U-Boot 版本
=> printenv bootcmd        # 看默认启动命令
=> mmc list                # 存储设备（SD 卡在 sdh@d4280000: 0）
=> part list mmc 0         # SD 分区表
=> help                    # 全部命令
```

在 U-Boot 里配网络（板卡服务器已规划好 IP，照抄即可）：

```
=> setenv serverip 192.168.1.2      # 板卡服务器
=> setenv ipaddr 192.168.1.157      # 本板 IP（如老师另有分配请按老师的来）
=> setenv netmask 255.255.255.0
=> ping 192.168.1.2                 # 测试连通
```

> ⚠️ `net list` 命令在这块板上会触发异常复位（Load access fault），别用；`ostool` 内部探测时踩到它属于正常现象，它会自己重试。

---

## 8. 往板子传文件

核心机制：**文件先上传到你当前会话的目录，再由 U-Boot 经 TFTP 下载到内存**。

### 8.1 上传到会话（HTTP API）

会话建立后（比如 `board connect` 挂着），拿到会话 ID 和令牌：

```bash
SID=$(grep -a session_id /path/to/你的会话日志 | head -1 | tr -d '\r' | awk '{print $2}')
# 令牌在 hosts.json 的嵌套结构里 (board-server:<hash> -> access_token), 用递归取
TOK=$(python3 -c "
import json
d = json.load(open('$HOME/.config/ostool/hosts.json'))
def find(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == 'access_token' and isinstance(v, str): return v
            r = find(v)
            if r: return r
    return None
print(find(d) or '')")

curl -X PUT \
  -H "Authorization: Bearer $TOK" \
  -H "X-File-Path: boot/myfile.bin" \
  --data-binary @myfile.bin \
  https://ostool.muxai.net:14321/api/v1/sessions/$SID/files
```

返回的 JSON 里有 `tftp_url`（如 `tftp://192.168.1.2/ostool/sessions/<会话ID>/boot/myfile.bin`）。

### 8.2 在 U-Boot 里下载

```
=> tftpboot 0x180000000 ostool/sessions/<会话ID>/boot/myfile.bin
# 成功会打印 Bytes transferred = <字节数>
```

### 8.3 限制（重要）

| 限制 | 值 | 对策 |
| --- | --- | --- |
| 单文件上传上限 | **64 MiB**（超了 HTTP 500） | 大文件切 <64MB 的块多次传（`env/bin/flash-model.sh` 就是这么做的） |
| 会话寿命 | **约 5 分钟强制回收** | 每块单独开会话；板子上 SD 卡数据跨会话持久 |
| TFTP 速度 | ~4-8 MB/s（局域网） | — |
| 外网上传速度 | ~1-1.6 MB/s | 大文件耐心等或分块 |

---

## 9. 在板上运行自己的系统（以 StarryOS 为例）

### 9.1 方式一：ostool board run（全自动，推荐入门）

需要两个配置文件：

**构建配置**（指向已编译好的内核 ELF，不重新编译）：

```toml
# prebuilt.toml —— 直接用现成内核
[system.Custom]
build_cmd = "true"
elf_path = "/绝对路径/starryos"     # 注意是 ELF 不是 .bin
```

**板级配置**（`board.toml`）：

```toml
board_type = "BananaPi"
kernel_load_addr = "0x102200000"
fit_load_addr = "0x140000000"
bootm_addr = "0x140000000"
timeout = 240

[[shell_check_steps]]
success_regex = ["(?m)^MY_BOOT_OK\\s*$"]
```

运行：

```bash
cd 你的/tgoskits/工作区     # board run 需要在工作区里找 manifest
ostool board run -c prebuilt.toml --board-config board.toml
```

流程：ostool 自动 strip 内核 → 打包 FIT（含服务器提供的 DTB）→ TFTP 推送 → `bootm` → 等待 `shell_check_steps` 匹配。

> ⚠️ ostool 0.31+ **拒绝**旧版顶层 `shell_prefix` / `shell_init_cmd` / `success_regex` 键（报 "removed U-Boot config key"），必须用上面的 `shell_check_steps` 新格式。

### 9.2 方式二：手动 booti（最灵活，调试首选）

把内核和 DTB 分别传到会话（见第 8 节），然后：

```
=> tftpboot 0x102200000 ostool/sessions/<会话ID>/boot/starryos.bin
=> tftpboot 0x110000000 ostool/sessions/<会话ID>/boot/board.dtb
=> booti 0x102200000 - 0x110000000
```

- 内核加载地址 `0x102200000`、DTB 放 `0x110000000`（自由内存区）
- 板卡的设备树可以从会话 API 直接下载：
  `curl -H "Authorization: Bearer $TOK" .../api/v1/sessions/$SID/dtb/download -o board.dtb`

### 9.3 关于 rootfs

- 板上 SD 卡的分区 1（LBA 0x2000 起，约 30GB）是课程预置的 ext4 rootfs（busybox 环境），StarryOS 会自动挂载它
- **不要轻易覆盖**；要放自己的程序/模型进 rootfs，用 `env/bin/flash-file.sh`（小文件直刷裸扇区）或 `env/bin/flash-model.sh`（大文件分块）
- 大文件（如 AI 模型）可以像刷 rootfs 一样分块写到 SD 卡的空闲 LBA 区域

---

## 10. 脚本化 / 自动化操作

想用程序驱动串口（而不是人肉敲），关键点：

1. **ostool 的串口终端要求 stdin/stdout 都是 tty 才转发输入**。管道/FIFO 重定向会静默失效。
   解决：用 `script(1)` 套伪终端：
   ```bash
   script -qefc "ostool board connect -b BananaPi" session.log < input.fifo
   ```
2. **`script` 的日志带 CRLF**，从日志提取会话 ID 等字段时要 `tr -d '\r'`，否则 URL 里混入控制字符报 "URL bad format"。
3. 现成参考实现：`env/bin/board-console.sh`（send/wait/log 四件套）。

---

## 11. 常见问题与坑（必读）

全部是实测踩过的，按出现频率排序：

| # | 现象 | 原因/解法 |
| --- | --- | --- |
| 1 | 串口会话莫名断开（websocket reset） | **平台约 5 分钟强制回收会话**，属正常设计。长任务拆成多个会话，SD/UFS 数据跨会话保留 |
| 2 | 文件上传返回 HTTP 500 | 单文件超 **64MiB** 上限，切小块 |
| 3 | `mmc write` 报 `block number exceeds max` | U-Boot 参数**必须 0x 前缀十六进制**：`mmc write 0x180000000 0x2000 0x20000`。裸十进制会被按 16 进制解析 |
| 4 | TFTP 报 `EQOS_DMA_MODE_SWR stuck -110` | 网口初始化抖动，约半数上电出现。**断开会话重连**（=重新上电）即恢复，多试几次 |
| 5 | 指定 `--board-id` 被拒：`server allocated board X while board Y was requested` | 租约实际对应的板与你想的不同。**去掉 `--board-id`**，用 `-b BananaPi` 自动分配 |
| 6 | `net list` 后板子重启 | 该命令触发 U-Boot 异常（已知问题），避免使用 |
| 7 | 提取会话 ID 后 API 报 URL 错误 | 日志行尾有 `\r`（pty 所致），`tr -d '\r'` |
| 8 | 上传 API 时通时断（HTTP 000） | 令牌过期后 ostool 已自动刷新，但你的脚本还在用旧 token——每次重新读取 `hosts.json` |
| 9 | board run 结束报 `shell check sequence ended before ... completed` | 串口 WebSocket 在判定瞬间被平台回收。看日志：如果判活标记已输出，**实际是成功的** |
| 10 | 编译 ostool 报 `undefined reference to udev_*` | 系统缺 libudev 开发文件：`apt install libudev-dev`，或建 libudev.so 符号链接 + 普通 `-ludev`（`-l:libudev.so.1` 写法在新版会静默失效） |
| 11 | 内核对 SD 大流量读报 CRC mismatch 后持续超时 | 原内核 SDHCI 驱动缺延迟调谐；`patches/tgoskits-bananapi-heralab.patch` 里有调谐移植 + ×4 分频；遇到先重上电 |
| 12 | StarryOS 里向量指令程序 SIGILL (exit 132) | 内核未启用用户态向量扩展；同上补丁里包含 sstatus.VS 使能和每任务向量上下文保存 |

---

## 12. 板卡硬件速查表

来源：板卡设备树实测解析（完整 DTS 见 hera-rt-lab 项目 `runs/board/board.dts`）。

| 资源 | 规格 |
| --- | --- |
| SoC | SpacemiT K3（`spacemit,k3-com260-ifx`） |
| CPU | **16 核异构**：8× X100 通用核（rv64imafdcvh，VLEN 256）+ 8× A100 AI 核（`cpu-ai`，rv64imafdcv，**VLEN 1024**，含 RVV/FP16 向量扩展） |
| DRAM | 8 GiB（DDR 6400MT/s；注意内核默认可见约 4GB，高 4GB 在设备树中保留） |
| AI 加速 | IME 矩阵引擎（在 A100 核内，自定义指令 vmadot/vfwmadot 等）+ TCM 8×384KB |
| 实时核 | 2× RT24（remoteproc 资源已预留） |
| 存储 | 32GB SD（课程 rootfs 所在）+ 板载 UFS + 8MB NOR（U-Boot 所在） |
| 机器人接口 | 8× FDCAN、37× PWM、ri2c/rspi（RT24 域）、摄像头管线（imx219/ov5647/imx415+GMSL） |
| 网络 | 4× 千兆网口（StarryOS 现有构建未带网卡驱动，U-Boot 网络可用） |
| 其他 | USB3×4、PCIe×4、GPU、显示（DSI/DP） |
| 调频 | X100 至 2.4GHz / A100 至 2.0GHz（独立 OPP 表） |

---

## 13. 参考链接

| 资源 | 地址 |
| --- | --- |
| 开发板平台 | https://ostool.muxai.net:14321/ |
| ostool 客户端源码 | https://github.com/drivercraft/ostool |
| ostool API 契约 | 仓库 `docs/api.md`（OAuth / 会话 / 文件上传 / TFTP 全接口） |
| StarryOS 构建仓库 | https://github.com/rcore-os/tgoskits（bananapi 分支为本板适配） |
| 本项目内核改动 | `patches/tgoskits-bananapi-heralab.patch` |
| 开发日志 | 同目录 `log.md` |

---

*文档版本：2026-10-21 v1.0。平台行为（会话时长、上传限额、板卡分配）可能调整，以实际为准；发现新坑欢迎补充进第 11 节。*

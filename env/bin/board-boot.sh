#!/bin/bash
# board-boot.sh — 一键交互式启动: 会话→上传内核+DTB→tftp→booti→等 shell
# 网卡抖动/EQOS stuck 自动整轮重试。用法: board-boot.sh [重试次数]
set -u
LAB=/data/codes/OS-train-26fall/hera-rt-lab
BC=$LAB/tools/bin/board-console.sh
TRIES=${1:-4}

for TRY in $(seq 1 $TRIES); do
  echo "== boot try $TRY/$TRIES =="
  $BC stop 2>/dev/null || true
  sleep 8
  $BC start 1500
  SID=""
  for t in $(seq 1 60); do
    SID=$(grep -a "session_id" $LAB/runs/board/session.log | head -1 | tr -d '\r' | awk '{print $2}')
    [ -n "$SID" ] && break; sleep 1
  done
  [ -n "$SID" ] || { echo "  会话建立失败"; continue; }
  echo "$SID" > /tmp/board-sid
  python3 $LAB/tools/bin/session-put.py "$SID" boot/starryos.bin \
    $LAB/src/tgoskits-bananapi/target/riscv64gc-unknown-none-elf/release/starryos.bin 2>/dev/null | grep -q "HTTP 201" || { echo "  内核上传失败"; continue; }
  python3 $LAB/tools/bin/session-put.py "$SID" boot/board.dtb $LAB/runs/board/board.dtb 2>/dev/null | grep -q "HTTP 201" || { echo "  DTB 上传失败"; continue; }
  $BC wait "=>" 100 || { echo "  无 U-Boot 提示符"; continue; }
  $BC send 'setenv serverip 192.168.1.2'; sleep 1
  $BC send 'setenv ipaddr 192.168.1.157'; sleep 1
  $BC send "tftpboot 0x102200000 ostool/sessions/$SID/boot/starryos.bin"
  $BC wait "Bytes transferred = 14077656" 130 || { echo "  内核 tftp 失败(网卡抖动?)"; continue; }
  $BC send "tftpboot 0x110000000 ostool/sessions/$SID/boot/board.dtb"
  $BC wait "Bytes transferred = 142149" 60 || { echo "  DTB tftp 失败"; continue; }
  $BC send 'booti 0x102200000 - 0x110000000'
  $BC wait "root@starry" 200 || { echo "  内核未起来"; continue; }
  sleep 18   # 等 shell 稳定
  echo "BOOT_OK sid=$SID"
  exit 0
done
echo "BOOT_FAILED after $TRIES tries"
exit 1

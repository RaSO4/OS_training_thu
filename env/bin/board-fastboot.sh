#!/bin/bash
# board-fastboot.sh — 从 SD 裸扇区 mmc read 内核/DTB 后 booti (无需 TFTP 内核)
# 依赖: 已用 flash-file.sh 把 starryos.bin 刷到 LBA 0x500000, board.dtb 到 0x600000
set -u
LAB=/data/codes/OS-train-26fall/hera-rt-lab
BC=$LAB/tools/bin/board-console.sh
KERNEL=$LAB/src/tgoskits-bananapi/target/riscv64gc-unknown-none-elf/release/starryos.bin
KSEC=$(( ($(stat -c %s "$KERNEL") + 511) / 512 ))
TRIES=${1:-3}

for TRY in $(seq 1 $TRIES); do
  echo "== fastboot try $TRY/$TRIES =="
  $BC stop 2>/dev/null || true
  sleep 8
  $BC start 1500
  $BC wait "=>" 100 || { echo "  无 U-Boot 提示符"; continue; }
  $BC send 'mmc dev 0'
  sleep 2
  $BC send "mmc read 0x102200000 0x500000 0x$(printf %x $KSEC)"
  sleep 3
  $BC send 'mmc read 0x110000000 0x600000 0x116'
  sleep 3
  $BC send 'booti 0x102200000 - 0x110000000'
  $BC wait "root@starry" 120 || { echo "  内核未起来"; $BC log 40; continue; }
  sleep 10
  echo "FASTBOOT_OK"
  exit 0
done
echo "FASTBOOT_FAILED"
exit 1

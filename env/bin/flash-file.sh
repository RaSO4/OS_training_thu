#!/bin/bash
# flash-file.sh — 把单个小文件刷到 SD 裸扇区 (会话上传 -> TFTP -> mmc write)
# 用法: flash-file.sh <local_file> [remote_name] [base_lba_hex]
# 默认 base LBA 0x380000 (rootfs 从 0x2000 起 64MB, 模型从 0x400000 起)
set -u
LAB=/data/codes/OS-train-26fall/hera-rt-lab
BC=$LAB/tools/bin/board-console.sh
FILE=${1:?usage: flash-file.sh <file> [name] [lba_hex]}
NAME=${2:-$(basename "$FILE")}
LBA_HEX=${3:-0x380000}
LBA=$((LBA_HEX))
SIZE=$(stat -c %s "$FILE")
SECTORS=$(( (SIZE + 511) / 512 ))
MAX_TRY=3

if [ "$SIZE" -gt $((64*1024*1024)) ]; then
  echo "file too large for one session upload (>64MiB)"; exit 1
fi
echo "file=$FILE name=$NAME size=$SIZE sectors=$SECTORS base_lba=$LBA_HEX"

for TRY in $(seq 1 $MAX_TRY); do
  echo "== flash try $TRY/$MAX_TRY =="
  $BC stop 2>/dev/null || true
  sleep 12
  $BC start 600
  SID=""
  for t in $(seq 1 45); do
    SID=$(grep -a "session_id" $LAB/runs/board/session.log | head -1 | tr -d '\r' | awk '{print $2}')
    [ -n "$SID" ] && break
    sleep 1
  done
  [ -n "$SID" ] || { echo "  会话建立失败"; continue; }

  if ! python3 $LAB/tools/bin/session-put.py "$SID" "boot/$NAME" "$FILE" 2>/dev/null | grep -q "HTTP 201"; then
    echo "  上传失败"; continue
  fi
  $BC wait "=>" 90 || { echo "  无 U-Boot 提示符"; continue; }
  $BC send 'setenv serverip 192.168.1.2'; sleep 1
  $BC send 'setenv ipaddr 192.168.1.157'; sleep 1
  $BC send 'setenv netmask 255.255.255.0'; sleep 1
  $BC send "tftpboot 0x180000000 ostool/sessions/$SID/boot/$NAME"
  $BC wait "Bytes transferred = $SIZE" 180 || { echo "  tftp 失败/大小不符"; continue; }
  $BC send 'mmc dev 0'; sleep 2
  $BC send "mmc write 0x180000000 0x$(printf %x $LBA) 0x$(printf %x $SECTORS)"
  $BC wait "blocks written: OK" 120 || { echo "  mmc write 失败"; continue; }
  $BC stop
  echo "FLASH_FILE_OK name=$NAME lba=$LBA_HEX sectors=$SECTORS size=$SIZE"
  echo "板上提取: dd if=/dev/blk0 bs=512 skip=$LBA count=$SECTORS of=/ram/$NAME"
  exit 0
done
echo "FLASH_FILE_FAILED"
$BC stop
exit 1

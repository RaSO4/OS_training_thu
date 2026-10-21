#!/bin/bash
# flash-model.sh — 模型分块刷写(断点续传版)
# 布局: rootfs LBA 0x2000(64MB); 模型固定 LBA 0x400000 起连续分块
# 每块独立会话(避开 5min 回收), 块级重试, /tmp/chunk-done.N 即断点标记
set -u
LAB=/data/codes/OS-train-26fall/hera-rt-lab
BC=$LAB/tools/bin/board-console.sh
MODEL=${1:?usage: flash-model.sh <model.gguf>}
CHUNK_MB=48
BASE_LBA=${FLASH_BASE_LBA:-$((0x400000))}
SPT=512
MAX_TRY=5

SIZE=$(stat -c %s "$MODEL")
SECTORS=$(( (SIZE + SPT - 1) / SPT ))
echo "model=$MODEL size=$SIZE sectors=$SECTORS base_lba=0x$(printf %x $BASE_LBA)"

rm -f /tmp/model-chunk.*
split -b ${CHUNK_MB}M -d -a 3 "$MODEL" /tmp/model-chunk.
CHUNKS=(/tmp/model-chunk.*)
N=${#CHUNKS[@]}
echo "chunks: $N"

i=0
while [ $i -lt $N ]; do
  C=${CHUNKS[$i]}
  MARK=/tmp/chunk-done.$i
  [ -f $MARK ] && { echo "chunk $i 已完成(跳过)"; i=$((i+1)); continue; }
  CSZ=$(stat -c %s "$C")
  CSEC=$(( (CSZ + SPT - 1) / SPT ))
  CLBA=$(( BASE_LBA + i * CHUNK_MB * 1024 * 1024 / SPT ))
  echo "=== chunk $i/$((N-1)): $CSZ bytes -> LBA 0x$(printf %x $CLBA) ==="

  OK=0
  for TRY in $(seq 1 $MAX_TRY); do
    $BC stop 2>/dev/null || true
    sleep 12   # 会话间隔: 避开服务端清理竞态
    $BC start 600
    SID=""
    for t in $(seq 1 45); do
      SID=$(grep -a "session_id" $LAB/runs/board/session.log | head -1 | tr -d '\r' | awk '{print $2}')
      [ -n "$SID" ] && break; sleep 1
    done
    [ -n "$SID" ] || { echo "  try$TRY: 会话建立失败"; continue; }
    python3 $LAB/tools/bin/session-put.py "$SID" boot/chunk$i "$C" 2>/dev/null | grep -q "HTTP 201" || { echo "  try$TRY: 上传失败"; continue; }
    $BC wait "=>" 75 || { echo "  try$TRY: 无提示符"; continue; }
    $BC send 'setenv serverip 192.168.1.2'; sleep 1
    $BC send 'setenv ipaddr 192.168.1.157'; sleep 1
    $BC send "tftpboot 0x180000000 ostool/sessions/$SID/boot/chunk$i"
    $BC wait "Bytes transferred = $CSZ" 150 || { echo "  try$TRY: tftp 失败"; continue; }
    $BC send 'mmc dev 0'; sleep 2
    $BC send "mmc write 0x180000000 0x$(printf %x $CLBA) 0x$(printf %x $CSEC)"
    $BC wait "blocks written: OK" 90 || { echo "  try$TRY: mmc write 失败"; continue; }
    touch $MARK; OK=1; break
  done
  [ $OK -eq 1 ] || { echo "chunk $i 连续 $MAX_TRY 次失败, 退出"; $BC stop; exit 1; }
  i=$((i+1))
done

$BC stop
echo "MODEL_FLASH_DONE size=$SIZE sectors=$SECTORS base_lba=0x$(printf %x $BASE_LBA)"
echo "板上提取: dd if=<partdev> bs=512 skip=$((BASE_LBA-0x2000)) count=$SECTORS of=/tmp/model.gguf"

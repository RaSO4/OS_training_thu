#!/usr/bin/env bash
# board-console.sh — BananaPi-1 串口会话管理
# 用法:
#   board-console.sh start [秒]   起后台 connect 会话(FIFO stdin, 日志 runs/board/session.log)
#   board-console.sh send 'cmd'   往串口写一行(自动补 \n)
#   board-console.sh wait '正则' [秒]  等日志出现正则
#   board-console.sh log [行数]   看会话日志尾部
#   board-console.sh stop         结束会话
set -u
LAB=/data/codes/OS-train-26fall/hera-rt-lab
FIFO=$LAB/tmp/board-in
LOGDIR=$LAB/runs/board
LOG=$LOGDIR/session.log
PIDF=$LOGDIR/session.pid
BIN=$LAB/tools/cargo/bin/ostool

mkdir -p "$LOGDIR" "$LAB/tmp"

case "${1:-}" in
start)
  DUR="${2:-3600}"
  [ -p "$FIFO" ] || mkfifo "$FIFO"
  # 保持 FIFO 写端, 防止读端 EOF (同 qin 技巧)
  tail -f /dev/null > "$FIFO" &
  echo $! > "$PIDF.tail"
  : > "$LOG"
  # script(1) 套 pty: ostool sterm 要求 stdin/stdout 均为终端才转发交互输入
  ( source "$LAB/env.sh"; exec timeout "$DUR" script -qefc \
      "$BIN board connect -b BananaPi" "$LOG" < "$FIFO" > /dev/null 2>&1 ) &
  echo $! > "$PIDF"
  sleep 3
  echo "session pid $(cat "$PIDF"), log: $LOG"
  ;;
send)
  printf '%s\n' "${2:-}" > "$FIFO"
  ;;
wait)
  PAT="${2:-}"; TMO="${3:-60}"
  for i in $(seq 1 "$TMO"); do
    grep -qE "$PAT" "$LOG" 2>/dev/null && { echo "matched after ${i}s"; exit 0; }
    sleep 1
  done
  echo "TIMEOUT waiting for: $PAT"; exit 1
  ;;
log)
  tail -n "${2:-40}" "$LOG"
  ;;
stop)
  [ -f "$PIDF" ] && kill "$(cat "$PIDF")" 2>/dev/null
  [ -f "$PIDF.tail" ] && kill "$(cat "$PIDF.tail")" 2>/dev/null
  rm -f "$PIDF" "$PIDF.tail"
  echo stopped
  ;;
*)
  echo "usage: $0 start [sec] | send 'cmd' | wait 'regex' [sec] | log [n] | stop" >&2
  exit 2
  ;;
esac

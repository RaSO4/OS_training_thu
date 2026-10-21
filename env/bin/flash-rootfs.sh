#!/bin/bash
# flash-rootfs.sh — 一气呵成: 会话→上传镜像→TFTP→mmc write 刷 SD rootfs 分区
# 背景: 平台 ~5min 回收会话, 全流程必须快
set -e
LAB=/data/codes/OS-train-26fall/hera-rt-lab
BC=$LAB/tools/bin/board-console.sh
IMG=$LAB/tmp/rootfs-board.img
SIDF=/tmp/board-sid
TOKF=/tmp/board-tok

$BC stop 2>/dev/null || true

token() {
  python3 - "$TOKF" <<'PYEOF'
import json,sys
d=json.load(open('/root/.config/ostool/hosts.json'))
def f(o):
    if isinstance(o,dict):
        for k,v in o.items():
            if k=='access_token' and isinstance(v,str): return v
            r=f(v)
            if r: return r
    return None
t=f(d) or ''
open(sys.argv[1],'w').write(t)
PYEOF
}

echo "[1/6] 起新会话"
$BC start 600
SID=""
for i in $(seq 1 60); do
  SID=$(grep -a "session_id" $LAB/runs/board/session.log | head -1 | tr -d '\r' | awk '{print $2}')
  [ -n "$SID" ] && break
  sleep 1
done
[ -n "$SID" ] || { echo "会话建立失败"; $BC log 10; exit 1; }
echo "$SID" > $SIDF
echo "    session=$SID"
token; TOK=$(cat $TOKF)

echo "[2/6] 上传 rootfs 镜像 ($(du -m $IMG | cut -f1)MB)"
python3 $LAB/tools/bin/session-put.py "$SID" boot/rootfs-board.img $IMG | tee /tmp/put-resp.txt
grep -q "HTTP 201" /tmp/put-resp.txt || { echo "上传失败"; exit 1; }
TFTPPATH="ostool/sessions/$SID/boot/rootfs-board.img"

echo "[3/6] 等 U-Boot 提示符"
$BC wait "=>" 90

echo "[4/6] 配网 + TFTP 下载到 0x180000000"
$BC send 'setenv serverip 192.168.1.2'
sleep 1
$BC send 'setenv ipaddr 192.168.1.157'
sleep 1
$BC send 'setenv netmask 255.255.255.0'
sleep 1
$BC send "tftpboot 0x180000000 $TFTPPATH"
$BC wait "Bytes transferred = 67108864" 180 || { echo "TFTP 大小不匹配或失败"; $BC log 30; exit 1; }

echo "[5/6] mmc write (LBA 0x2000, 0x80000 扇区)"
$BC send 'mmc dev 0'
sleep 2
$BC send 'mmc write 0x180000000 0x2000 0x20000'
$BC wait "blocks written: OK" 120 || { echo "mmc write 失败"; $BC log 30; exit 1; }

echo "[6/6] 完成; 回读首块验证"
$BC send 'mmc read 0x180100000 0x2000 0x8'
sleep 4
$BC log 15 | grep -a "MMC read" || true
$BC stop
echo "FLASH_DONE — 下一步: ostool board run 启动验证"

#!/bin/sh
# 将 k3_test 二进制注入 rootfs 镜像 /root/k3_test (qemu 必须已停止)
set -e
LAB=/data/codes/OS-train-26fall/hera-rt-lab
IMG=$1
BIN=$LAB/src/k3x-HERA-RT/k3_test/target/riscv64gc-unknown-linux-musl/release/k3_test
[ -f "$IMG" ] || { echo "usage: inject-k3-test.sh <rootfs.img>"; exit 1; }
debugfs -w "$IMG" -f /dev/stdin <<CMDS
cd /root
rm k3_test
write $BIN k3_test
chmod 755 k3_test
CMDS
echo "injected k3_test into $IMG"

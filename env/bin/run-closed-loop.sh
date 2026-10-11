#!/bin/sh
# 一键复现 #1..#3: 启动 StarryOS(QEMU) 并在 guest 内跑 bridge_demo(F32 MatMul 桥接; 交互模式可手动跑 k3_test)
# 用法: run-closed-loop.sh          (前台观察输出, Ctrl-A X 退出 QEMU)
#       run-closed-loop.sh bg       (后台模式, 日志在 runs/qemu3/run3.log)
L=/data/codes/OS-train-26fall/hera-rt-lab
KB=$L/src/tgoskits/target/riscv64gc-unknown-linux-musl/release/starryos.bin
IMG=$L/downloads/rootfs-test.img
RUN=$L/runs/qemu3
. $L/env.sh
if [ "$1" = bg ]; then
  rm -f $RUN/run3.log
  setsid nohup sh -c "tail -f /dev/null > $RUN/qin & sleep 0.3; qemu-system-riscv64 -machine virt -m 4096M -smp 16 -nographic -cpu rv64,v=true,zfh=true,zvfh=true -kernel $KB -device virtio-blk-pci,drive=disk0 -drive id=disk0,if=none,format=raw,file=$IMG -device virtio-net-pci,netdev=net0 -netdev user,id=net0 < $RUN/qin > $RUN/run3.log 2>&1" </dev/null >/dev/null 2>&1 &
  sleep 25 && echo /root/bridge_demo > $RUN/qin && echo "started, log: $RUN/run3.log"
else
  exec qemu-system-riscv64 -machine virt -m 4096M -smp 16 -nographic -cpu rv64,v=true,zfh=true,zvfh=true -kernel $KB -device virtio-blk-pci,drive=disk0 -drive id=disk0,if=none,format=raw,file=$IMG -device virtio-net-pci,netdev=net0 -netdev user,id=net0
fi

#!/bin/sh
# llama.cpp riscv64-musl 交叉构建 (chroot alpine + zig cc, RVV)
# 用法: build-llama-rv.sh [configure|build|clean]
set -e
LAB=/data/codes/OS-train-26fall/hera-rt-lab
R=$LAB/tools/qemu-alpine
DIST=$LAB/tmp/llama-dist
SRCDIR=/tmp/llama.cpp

# chroot 内挂载: /proc (zig 需要 /proc/self/exe), 源码与产物目录 bind
mountpoint -q $R/proc  || mount --bind /proc $R/proc
mkdir -p $R/tmp/llama.cpp $R/tmp/dist "$DIST"
mountpoint -q $R/tmp/llama.cpp || mount --bind $SRCDIR $R/tmp/llama.cpp
mountpoint -q $R/tmp/dist     || mount --bind $DIST $R/tmp/dist

case "${1:-build}" in
configure)
  chroot $R /bin/sh -c '
    cd /tmp/llama.cpp
    exec cmake -B /tmp/dist/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=riscv64 \
      -DGGML_NATIVE=OFF \
      -DGGML_BACKEND_DL=OFF \
      -DGGML_CPU_ALL_VARIANTS=OFF \
      -DCMAKE_C_COMPILER="zig;cc;-target;riscv64-linux-musl" \
      -DCMAKE_CXX_COMPILER="zig;c++;-target;riscv64-linux-musl" \
      -DCMAKE_C_FLAGS="-mcpu=baseline_rv64+v" \
      -DCMAKE_CXX_FLAGS="-mcpu=baseline_rv64+v" \
      -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON \
      \
      -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF
  '
  ;;
build)
  chroot $R /bin/sh -c 'exec cmake --build /tmp/dist/build --target llama-cli llama-bench --parallel 4'
  ;;
clean)
  chroot $R /bin/sh -c 'exec rm -rf /tmp/dist/build'
  ;;
esac

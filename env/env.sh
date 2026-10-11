# hera-rt-lab 自包含环境 — 每个新 shell 先 source 本文件
export LAB=/data/codes/OS-train-26fall/hera-rt-lab
export RUSTUP_HOME=$LAB/tools/rustup
export CARGO_HOME=$LAB/tools/cargo
export PATH=$LAB/tools/cargo/bin:$LAB/tools/qemu-env/bin:$PATH
export GIT_CONFIG_GLOBAL=$LAB/tools/gitconfig-global
export GIT_CONFIG_NOSYSTEM=1
export PATH=$LAB/tools/bin:$PATH
export PKG_CONFIG_PATH=$LAB/tools/pkgconfig:${PKG_CONFIG_PATH:-}
# bindgen(lwprintf-rs 等) 解析 riscv64 musl 头文件所需
export BINDGEN_EXTRA_CLANG_ARGS="--target=riscv64-unknown-linux-musl -isystem /data/codes/OS-train-26fall/hera-rt-lab/tools/riscv64-musl-headers/usr/include"

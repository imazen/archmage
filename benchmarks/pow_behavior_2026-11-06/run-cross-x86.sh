#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/share/mise/shims:$HOME/.local/bin:$PATH" TMPDIR="$HOME/tmp"
echo "$BASHPID" > active.pid
echo native-docs > active-stage.txt
cargo fmt -p zenmath -p magemath
RUSTDOCFLAGS=-Dwarnings cargo test -p zenmath -p magemath --doc
RUSTDOCFLAGS=-Dwarnings cargo doc -p zenmath -p magemath --no-deps
echo v3-build > active-stage.txt
CARGO_TARGET_DIR=target-v3 RUSTFLAGS=-Ctarget-cpu=x86-64-v3 cargo test --release -p zenmath -p magemath --lib --tests --examples
CARGO_TARGET_DIR=target-v3 RUSTFLAGS=-Ctarget-cpu=x86-64-v3 cargo build --release -p magemath --example pow_policy_check
target-v3/release/examples/pow_policy_check artifacts/corpus/pow.bin artifacts/v3
echo no-fma > active-stage.txt
qemu-x86_64 -cpu Nehalem target/release/examples/pow_policy_check artifacts/corpus/pow.bin artifacts/nehalem
echo wasm-build > active-stage.txt
CARGO_TARGET_DIR=target-wasm RUSTFLAGS=-Ctarget-feature=+simd128 cargo build --release -p magemath --example pow_policy_check --target wasm32-wasip1
CARGO_TARGET_DIR=target-wasm RUSTFLAGS=-Ctarget-feature=+simd128 CARGO_TARGET_WASM32_WASIP1_RUNNER="$HOME/tmp/pow-behavior-sol/wasmtime40 -W simd=y" cargo test -p zenmath -p magemath --target wasm32-wasip1 --lib --tests --examples
./wasmtime40 run --dir . target-wasm/wasm32-wasip1/release/examples/pow_policy_check.wasm artifacts/corpus/pow.bin artifacts/wasm
echo cross-complete > active-stage.txt

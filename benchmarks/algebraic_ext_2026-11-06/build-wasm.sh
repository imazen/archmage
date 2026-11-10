#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.cargo/bin:$PATH" TMPDIR="$HOME/tmp" RUSTFLAGS='-Ctarget-feature=+simd128 -Clink-dead-code=yes -Clink-arg=--export-all'
cd "$(dirname "$0")"
for mode in off thin; do
    if test "$mode" = off; then export CARGO_PROFILE_RELEASE_LTO=false; else export CARGO_PROFILE_RELEASE_LTO=thin; fi
    cargo build --release --target wasm32-wasip1 --bin wasm-kernel --bin results --target-dir "target-wasm-$mode"
    cargo rustc --release --target wasm32-wasip1 --lib -p algebraic-ext-consumer --target-dir "target-wasm-$mode" -- --emit=llvm-ir,asm
done

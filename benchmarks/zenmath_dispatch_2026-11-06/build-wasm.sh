#!/usr/bin/env bash
set -euo pipefail
unset CARGO_ENCODED_RUSTFLAGS
export RUSTFLAGS='-Ctarget-feature=+simd128 -Clink-dead-code=yes -Clink-arg=--export-all'
export CARGO_PROFILE_RELEASE_LTO=false
cargo build --manifest-path benchmarks/zenmath_dispatch_2026-11-06/Cargo.toml --locked --release --target wasm32-wasip1 --bin wasm-kernel --target-dir target-dispatch-wasm
cargo test -p zenmath --all-features --release --target wasm32-wasip1 --test acceleration --no-run --target-dir target-dispatch-wasm

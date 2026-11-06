#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.cargo/bin:$PATH" TMPDIR="$HOME/tmp"
cd "$(dirname "$0")/../.."
CARGO_PROFILE_RELEASE_LTO=false RUSTFLAGS='-Ctarget-feature=+simd128 -Clink-dead-code=yes -Clink-arg=--export-all' cargo build --manifest-path benchmarks/math_matrix_2026-11-06/Cargo.toml --locked --release --target wasm32-wasip1 --bin wasm-kernel --bin validate --target-dir target-wasm

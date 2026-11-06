#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.cargo/bin:$PATH" TMPDIR="$HOME/tmp" RUSTUP_TOOLCHAIN=1.99.0
cd "$(dirname "$0")/../.."
echo "$BASHPID" > matrix-active.pid
cargo build --manifest-path benchmarks/math_matrix_2026-11-06/Cargo.toml --release --bins
cargo clippy --manifest-path benchmarks/math_matrix_2026-11-06/Cargo.toml --all-targets -- -D warnings
cargo rustc --manifest-path benchmarks/math_matrix_2026-11-06/Cargo.toml --release --lib -- --emit=llvm-ir,asm
MATH_MATRIX_REQUIRE_TIERS=scalar,neon benchmarks/math_matrix_2026-11-06/target/release/validate matrix-values

#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.cargo/bin:$PATH" TMPDIR="$HOME/tmp"
cd "$(dirname "$0")/../.."
echo "$BASHPID" > matrix-active.pid
cargo build --manifest-path benchmarks/math_matrix_2026-11-06/Cargo.toml --release --features avx512 --bins
cargo clippy --manifest-path benchmarks/math_matrix_2026-11-06/Cargo.toml --all-targets --features avx512 -- -D warnings
cargo rustc --manifest-path benchmarks/math_matrix_2026-11-06/Cargo.toml --release --features avx512 --lib -- --emit=llvm-ir,asm
mkdir -p algebraic-evidence
rustc +1.98.1 --crate-type=lib benchmarks/math_matrix_2026-11-06/algebraic-basic.rs --out-dir algebraic-evidence --emit=llvm-ir -O
if rustc +1.89.0 --crate-type=lib benchmarks/math_matrix_2026-11-06/algebraic-basic.rs --out-dir algebraic-evidence > algebraic-evidence/msrv-1.89.log 2>&1; then echo 'ERROR: algebraic methods unexpectedly compiled on 1.89'; exit 1; fi
cp choosing-math.md docs/site/content/magetypes/math/choosing-math.md
cp reductions.md docs/site/content/magetypes/operations/reductions.md
python3 xtask/check_docs.py docs/site/content/magetypes/math/choosing-math.md docs/site/content/magetypes/operations/reductions.md
python3 xtask/check_docs.py --features avx512 docs/site/content/magetypes/math/choosing-math.md docs/site/content/magetypes/operations/reductions.md
MATH_MATRIX_REQUIRE_TIERS=scalar,v3,v4x benchmarks/math_matrix_2026-11-06/target/release/validate matrix-values

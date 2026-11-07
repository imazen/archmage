#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.cargo/bin:$PATH" TMPDIR="$HOME/tmp"
cd "$(dirname "$0")/../.."
echo "$BASHPID" > matrix-health.pid
cargo run -p xtask -- generate
cargo run -p xtask -- validate-registry
cargo run -p xtask -- validate
cargo run -p xtask -- soundness
cp choosing-math.md docs/site/content/magetypes/math/choosing-math.md
cp reductions.md docs/site/content/magetypes/operations/reductions.md
sha256sum docs/site/content/magetypes/math/choosing-math.md docs/site/content/magetypes/operations/reductions.md
python3 xtask/check_docs.py docs/site/content/magetypes/math/choosing-math.md docs/site/content/magetypes/operations/reductions.md
python3 xtask/check_docs.py --features avx512 docs/site/content/magetypes/math/choosing-math.md docs/site/content/magetypes/operations/reductions.md
export CARGO_TARGET_WASM32_WASIP1_RUNNER="$PWD/wasmtime40 run -W simd=y,relaxed-simd=n"
RUSTFLAGS=-Ctarget-feature=+simd128 python3 xtask/check_docs.py --target wasm32-wasip1 docs/site/content/magetypes/math/choosing-math.md docs/site/content/magetypes/operations/reductions.md

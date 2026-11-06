#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.cargo/bin:$PATH" TMPDIR="$HOME/tmp"
cd "$(dirname "$0")"
runner=$1
for mode in off thin; do
    "$runner" run -W simd=y,relaxed-simd=n --env ALG_EXT_REQUIRE_TIERS=scalar,wasm128 "../../wasm-bin/wasm-kernel-$mode.wasm" > "catalog-wasm-$mode.tsv"
    "$runner" run -W simd=y,relaxed-simd=n --env ALG_EXT_REQUIRE_TIERS=scalar,wasm128 --dir . "../../wasm-bin/results-$mode.wasm" "outputs-wasm-$mode"
done
python3 analyze.py .

#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.cargo/bin:$PATH" TMPDIR="$HOME/tmp" MATH_MATRIX_REQUIRE_TIERS=scalar,wasm128
cd "$(dirname "$0")/../.."
mkdir -p wasm-isolation values-wasm
wasmtime run -W simd=y,relaxed-simd=n target-wasm/wasm32-wasip1/release/wasm-kernel.wasm > wasm-catalog.tsv 2> wasm-catalog.log
wasmtime run -W simd=y,relaxed-simd=n --dir . target-wasm/wasm32-wasip1/release/validate.wasm values-wasm '!metallic'
: > wasm-traps.tsv
: > wasm-isolation/status.tsv
for op in powf pow_fixed exp ln log2 log10 round log2_f64; do
    for band in ordinary edge special; do
        dir="wasm-isolation/$op-$band"
        set +e
        wasmtime run -W simd=y,relaxed-simd=n --dir . target-wasm/wasm32-wasip1/release/validate.wasm "$dir" metallic "$op" "$band" > "$dir.log" 2>&1
        code=$?
        set -e
        printf '%s\t%s\t%s\n' "$op" "$band" "$code" >> wasm-isolation/status.tsv
        if test "$code" != 0; then
            printf '%s/%s\tmetallic/baseline\n' "$op" "$band" >> wasm-traps.tsv
        else
            test -s "$dir/inventory.tsv"
            cat "$dir/inventory.tsv" >> values-wasm/inventory.tsv
            cp "$dir/"*.bin values-wasm/
        fi
    done
done
cat wasm-isolation/status.tsv

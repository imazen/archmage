#!/usr/bin/env bash
set -euo pipefail
runner=$1
module=$2
out=$3
mkdir -p "$out/values" "$out/isolation"
"$runner" --version
"$runner" run -W simd=y,relaxed-simd=n --env MATH_MATRIX_REQUIRE_TIERS=scalar,wasm128 --dir . "$module" "$out/values" '!metallic'
: > "$out/traps.tsv"
: > "$out/status.tsv"
for op in powf pow_fixed exp ln log2 log10 round log2_f64; do
    for band in ordinary edge special; do
        dir="$out/isolation/$op-$band"
        set +e
        "$runner" run -W simd=y,relaxed-simd=n --env MATH_MATRIX_REQUIRE_TIERS=scalar,wasm128 --dir . "$module" "$dir" metallic "$op" "$band" > "$dir.log" 2>&1
        code=$?
        set -e
        printf '%s\t%s\t%s\n' "$op" "$band" "$code" >> "$out/status.tsv"
        if test "$code" != 0; then
            printf '%s/%s\tmetallic/baseline\n' "$op" "$band" >> "$out/traps.tsv"
        else
            test -s "$dir/inventory.tsv"
            cat "$dir/inventory.tsv" >> "$out/values/inventory.tsv"
            cp "$dir/"*.bin "$out/values/"
        fi
    done
done
cat "$out/status.tsv"

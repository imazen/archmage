#!/usr/bin/env bash
# Verify the cached f32/f64 screens agree with both actual engine choices.
set -euo pipefail
out=$1
wasmtime=$2
[[ ! -e "$out" ]]
mkdir -p "$out"
CARGO_TARGET_DIR="$out/target" RUSTFLAGS='-C target-feature=+simd128,+relaxed-simd' cargo test -p magetypes --target wasm32-wasip1 --lib --no-run --message-format=json > "$out/build.jsonl" 2> "$out/build.log"
module=$(python3 -c 'import json,sys; xs=[json.loads(s)for s in open(sys.argv[1])]; bins=[r["executable"]for r in xs if r.get("reason")=="compiler-artifact" and r.get("executable") and r.get("target",{}).get("name")=="magetypes"];assert len(bins)==1; print(bins[0])' "$out/build.jsonl")
"$wasmtime" run -C cranelift-has_fma=true --env ARCHMAGE_EXPECT_RELAXED_FUSION=yes "$module" --exact wasm_fma::tests::relaxed_probe_matches_engine --nocapture > "$out/fused.log" 2>&1
"$wasmtime" run -C cranelift-has_fma=false --env ARCHMAGE_EXPECT_RELAXED_FUSION=no "$module" --exact wasm_fma::tests::relaxed_probe_matches_engine --nocapture > "$out/separate.log" 2>&1

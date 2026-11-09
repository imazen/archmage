#!/usr/bin/env bash
# Execute both engine FMA choices against the canonical graph, without timing.
set -euo pipefail
out=$1
expected=$2
modules=$3
wasmtime=$4
[[ ! -e "$out" ]]
mkdir -p "$out"
for library in std nostd; do
    module="$modules/target-wasm-relaxed-$library/wasm32-wasip1/release/examples/powf_highp_eval.wasm"
    [[ -f "$module" ]]
    for fusion in true false; do
        command=$(python3 -c 'import json,sys;print(json.dumps([sys.argv[1],"run","-C","cranelift-has_fma="+sys.argv[2],sys.argv[3]]))' "$wasmtime" "$fusion" "$module")
        python3 -B benchmarks/powf_providers_2026-11-09/highp_verify.py "$expected" "$out/$library-$fusion" --command "$command" > "$out/$library-$fusion.log" 2>&1
    done
done

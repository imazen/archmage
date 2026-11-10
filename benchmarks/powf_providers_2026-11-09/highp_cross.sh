#!/usr/bin/env bash
# Numerical cross execution only. Keep every binary, log and variance ledger.
set -euo pipefail
out=$1
expected=$2
wasmtime=$3
[[ ! -e "$out" ]]
mkdir -p "$out"
export TMPDIR="$HOME/tmp"
for profile in wasm-strict-std wasm-strict-nostd wasm-relaxed-std wasm-relaxed-nostd; do
    flags='+simd128'
    [[ $profile != *relaxed* ]] || flags='+simd128,+relaxed-simd'
    features=()
    [[ $profile != *nostd ]] || features=(--no-default-features --features w512)
    CARGO_TARGET_DIR="$out/target-$profile" RUSTFLAGS="-C target-feature=$flags" cargo build --manifest-path benchmarks/powf_validation/Cargo.toml --release --locked --target wasm32-wasip1 --bin powf_highp_eval "${features[@]}" > "$out/$profile.build.log" 2>&1
    module="$out/target-$profile/wasm32-wasip1/release/powf_highp_eval.wasm"
    command=$(python3 -c 'import json,sys;print(json.dumps([sys.argv[1],"run",sys.argv[2]]))' "$wasmtime" "$module")
    python3 -B benchmarks/powf_providers_2026-11-09/highp_verify.py "$expected" "$out/$profile" --command "$command" > "$out/$profile.verify.log" 2>&1
    if [[ $profile == *relaxed* ]]; then
        command=$(python3 -c 'import json,sys;print(json.dumps([sys.argv[1],"run","-W","relaxed-simd-deterministic=y",sys.argv[2]]))' "$wasmtime" "$module")
        python3 -B benchmarks/powf_providers_2026-11-09/highp_verify.py "$expected" "$out/$profile-deterministic" --command "$command" > "$out/$profile-deterministic.verify.log" 2>&1
    fi
done
CARGO_TARGET_DIR="$out/target-i686" cargo build --manifest-path benchmarks/powf_validation/Cargo.toml --release --locked --target i686-unknown-linux-gnu --bin powf_highp_eval > "$out/i686.build.log" 2>&1
command=$(python3 -c 'import json,sys;print(json.dumps([sys.argv[1]]))' "$out/target-i686/i686-unknown-linux-gnu/release/powf_highp_eval")
python3 -B benchmarks/powf_providers_2026-11-09/highp_verify.py "$expected" "$out/i686" --command "$command" --contexts scalar automatic > "$out/i686.verify.log" 2>&1

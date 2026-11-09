#!/usr/bin/env bash
set -euo pipefail
out=$1
fixture=$2
sde=$3
wasm_runner=${4:-wasmtime}
mkdir "$out"
study=benchmarks/powf_module_2026-11-07
python3 -B "$study/highp_accuracy.py" "$study/target/release/highp_eval" "$fixture" "$out/nehalem.json" --tier scalar --runner "$sde -nhm --"
cargo build --release --locked --manifest-path "$study/Cargo.toml" --target i686-unknown-linux-gnu --bin highp_eval
python3 -B "$study/highp_accuracy.py" "$study/target/i686-unknown-linux-gnu/release/highp_eval" "$fixture" "$out/i686.json" --tier scalar
for mode in strict relaxed; do
 feature=+simd128
 if [[ $mode == relaxed ]];then feature=+simd128,+relaxed-simd;fi
 export RUSTFLAGS="-Ctarget-feature=$feature"
 export CARGO_TARGET_DIR="$PWD/$study/target/policy-$mode"
 cargo build --release --locked --manifest-path "$study/Cargo.toml" --target wasm32-wasip1 --bin highp_eval
 cp "$CARGO_TARGET_DIR/wasm32-wasip1/release/highp_eval.wasm" "$out/$mode.wasm"
 if [[ $mode == strict ]];then
  for tier in scalar native;do python3 -B "$study/highp_accuracy.py" "$out/$mode.wasm" "$fixture" "$out/$mode-$tier.json" --tier "$tier" --runner "$wasm_runner run -W relaxed-simd=false";done
 else
  for fusion in true false;do for tier in scalar native;do python3 -B "$study/highp_accuracy.py" "$out/$mode.wasm" "$fixture" "$out/$mode-$fusion-$tier.json" --tier "$tier" --runner "$wasm_runner run -C cranelift-has_fma=$fusion";done;done
 fi
done

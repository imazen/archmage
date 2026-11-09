#!/usr/bin/env bash
set -euo pipefail
out=$1
fixture=$2
sde=$3
wasm_runner=${4:-wasmtime}
mkdir "$out"
"$sde" -nhm -- benchmarks/powf_module_2026-11-07/target/release/policy_widths scalar < "$fixture"
cargo build --release --locked --manifest-path benchmarks/powf_module_2026-11-07/Cargo.toml --target i686-unknown-linux-gnu --bin policy_widths
benchmarks/powf_module_2026-11-07/target/i686-unknown-linux-gnu/release/policy_widths scalar < "$fixture"
for mode in strict relaxed; do
 feature=+simd128
 if [[ $mode == relaxed ]];then feature=+simd128,+relaxed-simd;fi
 export RUSTFLAGS="-Ctarget-feature=$feature"
 export CARGO_TARGET_DIR="$PWD/benchmarks/powf_module_2026-11-07/target/policy-$mode"
 cargo build --release --locked --manifest-path benchmarks/powf_module_2026-11-07/Cargo.toml --target wasm32-wasip1 --bin policy_widths
 cp "$CARGO_TARGET_DIR/wasm32-wasip1/release/policy_widths.wasm" "$out/$mode.wasm"
 if [[ $mode == strict ]];then
  for tier in scalar native;do "$wasm_runner" run --env POWF_EXPECT_RELAXED_FUSED=false -W relaxed-simd=false "$out/$mode.wasm" "$tier" < "$fixture";done
 else
  for fusion in true false;do for tier in scalar native;do "$wasm_runner" run --env "POWF_EXPECT_RELAXED_FUSED=$fusion" -C "cranelift-has_fma=$fusion" "$out/$mode.wasm" "$tier" < "$fixture";done;done
 fi
 cargo rustc --release --locked --manifest-path benchmarks/powf_module_2026-11-07/Cargo.toml --target wasm32-wasip1 --bin policy_codegen -- --emit=asm
 python3 - "$CARGO_TARGET_DIR" "$out" "$mode" <<'PYS'
from pathlib import Path
import subprocess,sys
root,out,mode=map(Path,sys.argv[1:])
files=[p for p in (root/'wasm32-wasip1/release/deps').glob('policy_codegen*.s')if 'POLICYTABLE_X4_SCALAR' in p.read_text()]
assert len(files)==1,files
subprocess.run(['just','powf-policy-codegen-audit',str(files[0]),str(out/(str(mode)+'-codegen.json')),'--wasm'],check=True)
(out/(str(mode)+'-codegen.s')).write_bytes(files[0].read_bytes())
PYS
done

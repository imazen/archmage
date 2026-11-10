#!/usr/bin/env bash
# Full policy/tail grids on strict and both relaxed-WASM engine choices.
set -euo pipefail
out=$(realpath -m "$1")
mkdir "$out"
tests=(--test powf_tails_portable_midp --test powf_tails_portable_lowp --test powf_tails_backend_midp --test powf_tails_backend_lowp --test powf_engine_remap)
wasmtime --version > "$out/wasmtime.txt"
for mode in strict relaxed; do
  features=+simd128
  [[ $mode != relaxed ]] || features=+simd128,+relaxed-simd
  export CARGO_TARGET_DIR="$out/targets/$mode" RUSTFLAGS="-Ctarget-feature=$features"
  cargo test -p magetypes --release --all-features --target wasm32-wasip1 "${tests[@]}" --no-run --message-format=json > "$out/$mode-build.jsonl"
  python3 - "$out" "$mode" <<'PY'
from pathlib import Path
import json,shutil,sys
out=Path(sys.argv[1]);mode=sys.argv[2]
modules=[Path(x['executable'])for s in(out/f'{mode}-build.jsonl').read_text().splitlines()if (x:=json.loads(s)).get('reason')=='compiler-artifact'and x.get('executable')and x['target']['kind']==['test']]
assert len(modules)==5,modules
for p in modules:shutil.copy2(p,out/f'{mode}-{p.name}')
PY
  for module in "$out/$mode-"*.wasm; do
    if [[ $mode == strict ]]; then
      wasmtime run -W relaxed-simd=false "$module" --nocapture > "$module.log" 2>&1
      if [[ $module != *powf_engine_remap* ]]; then
        rg -q 'powf-tail-token=WASM SIMD128.*relaxed-f32-madd-bits=00000000' "$module.log"
      fi
    else
      for fusion in true false; do
        wasmtime run -C "cranelift-has_fma=$fusion" "$module" --nocapture > "$module.$fusion.log" 2>&1
        expected=00000000
        [[ $fusion != true ]] || expected=28800000
        if [[ $module != *powf_engine_remap* ]]; then
          rg -q "powf-tail-token=WASM SIMD128.*relaxed-f32-madd-bits=$expected" "$module.$fusion.log"
        fi
      done
    fi
  done
done
find "$out" -maxdepth 1 -name '*.wasm' -exec sha256sum {} + > "$out/MODULES.sha256"
echo 'all 64 policies and legacy remapping, strict and both relaxed choices: scalar/vector/tails pass'

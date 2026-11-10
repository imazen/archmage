#!/usr/bin/env bash
# Run under run-heavy on a worker. Full logs and reference hashes belong outside target/.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
strict=target/powf-wasm-strict/wasm32-wasip1/release/deps/magetypes-f1b616e26215cb72.wasm
relaxed=target/powf-wasm-relaxed/wasm32-wasip1/release/deps/magetypes-ca8f50a61aaeba92.wasm
for mode in strict fusing nonfusing; do
    binary=$relaxed
    extra=()
    expected=wasm-fused
    if [[ $mode == strict ]]; then binary=$strict; expected=wasm-software; fi
    if [[ $mode == nonfusing ]]; then extra=(-C cranelift-has-fma=false); expected=wasm-software; fi
    wasmtime run "${extra[@]}" --env ARCHMAGE_POWF_TIER=wasm128 --env ARCHMAGE_POWF_EXPECT_PLAN="$expected" "$binary" powf:: --nocapture
    wasmtime run "${extra[@]}" --env ARCHMAGE_POWF_TIER=wasm128 --env ARCHMAGE_NAN_STRIDE=1 "$binary" powf::tests::nan_sign_payload --nocapture
    sha256sum "$binary"
done

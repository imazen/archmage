#!/usr/bin/env bash
# Stream the same independent C oracle in strict, relaxed and no-FMA engines.
set -euo pipefail
reference=$(realpath "$1")
density=${2:---dense}
binary=${3:-recip_qualification}
manifest=benchmarks/primitive_variants_2026-11-10/Cargo.toml
for mode in strict relaxed relaxed-no-fma no-std-strict no-std-relaxed; do
    flags=+simd128
    features=(--no-default-features --features 'std w512')
    engine=()
    case "$mode" in *relaxed*) flags=+simd128,+relaxed-simd;; esac
    case "$mode" in no-std-*) features=(--no-default-features --features w512);; esac
    case "$mode" in relaxed-no-fma) engine=(-W relaxed-simd-deterministic=n -C cranelift-has-fma=false);; esac
    echo "QUALIFICATION mode=$mode flags=$flags features=${features[*]} engine=${engine[*]}"
    RUSTFLAGS="-Ctarget-feature=$flags" cargo build --release --target wasm32-wasip1 --manifest-path "$manifest" --bin "$binary" "${features[@]}"
    module="benchmarks/primitive_variants_2026-11-10/target/wasm32-wasip1/release/$binary.wasm"
    if [[ -n ${ARCHMAGE_QUAL_MODULES:-} ]]; then
        mkdir -p "$ARCHMAGE_QUAL_MODULES"
        cp "$module" "$ARCHMAGE_QUAL_MODULES/$mode-$binary.wasm"
        sha256sum "$ARCHMAGE_QUAL_MODULES/$mode-$binary.wasm"
    fi
    wasmtime "${engine[@]}" --dir "$(dirname "$reference")::/refs" "$module" --reference "/refs/$(basename "$reference")" --tier wasm "$density"
done

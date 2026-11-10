#!/usr/bin/env bash
# Fixed correctness matrix; no timing or engine-dependent accuracy thresholds.
set -euo pipefail
for flags in "" +simd128 +simd128,+relaxed-simd; do
    export RUSTFLAGS="-Ctarget-feature=$flags"
    export CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime"
    cargo test --target wasm32-wasip1 --features testable_dispatch --test wasm_token_testing
    token_test=()
    if [[ "$flags" == *simd128* ]]; then token_test=(--test sqrt_token_testing); fi
    for options in 'w512 archmage/testable_dispatch' 'std w512 archmage/testable_dispatch'; do
        cargo test -p magetypes --target wasm32-wasip1 --no-default-features --features "$options" --test sqrt_portability "${token_test[@]}" --test fused_arithmetic
    done
    if [[ "$flags" == *relaxed* ]]; then
        for fusion in yes no; do
            if [[ "$fusion" == yes ]]; then
                export CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime --env ARCHMAGE_EXPECT_RELAXED_FUSION=yes"
            else
                export CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime -W relaxed-simd-deterministic=n -C cranelift-has-fma=false --env ARCHMAGE_EXPECT_RELAXED_FUSION=no"
            fi
            cargo test -p magetypes --target wasm32-wasip1 --features 'std w512 archmage/testable_dispatch' --lib wasm_fma::tests::
            cargo test -p magetypes --target wasm32-wasip1 --no-default-features --features 'w512 archmage/testable_dispatch' --test fused_arithmetic
        done
    fi
done

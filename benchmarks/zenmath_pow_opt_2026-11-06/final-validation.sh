#!/usr/bin/env bash
set -euo pipefail
cargo fmt -p zenmath -p magemath --check
cargo test -p zenmath -p magemath --all-features --release --lib --tests --examples
cargo test -p zenmath -p magemath --all-features --doc
cargo clippy -p zenmath -p magemath --all-features --all-targets -- -D warnings
for features in '' correct-rounding normal configurable correct-rounding,normal,configurable std archmage runtime-dispatch avx512 correct-rounding,archmage normal,runtime-dispatch configurable,runtime-dispatch; do
    if [[ -n "$features" ]]; then cargo +1.89.0 check -p zenmath --no-default-features --features "$features"; else cargo +1.89.0 check -p zenmath --no-default-features; fi
done
for features in configurable configurable,std configurable,archmage; do
    cargo test -p zenmath --release --no-default-features --features "$features" --test pow_policy
done
cargo rustc --manifest-path benchmarks/zenmath_pow_opt_2026-11-06/compile-consumer/Cargo.toml --release --features unified-std -- --emit=llvm-ir
cargo run --manifest-path benchmarks/zenmath_pow_opt_2026-11-06/compile-consumer/Cargo.toml --release --features unified-std
python3 benchmarks/zenmath_pow_opt_2026-11-06/check-unification.py "$CARGO_TARGET_DIR"/release/deps/zenmath_pow_consumer-*.ll
cargo check -p zenmath --no-default-features --target aarch64-unknown-none
cargo check -p zenmath --no-default-features --features correct-rounding,normal,configurable,archmage --target aarch64-unknown-none

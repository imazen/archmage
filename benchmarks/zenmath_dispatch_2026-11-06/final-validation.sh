#!/usr/bin/env bash
set -euo pipefail
cargo fmt -p zenmath -p magemath --check
cargo test -p zenmath -p magemath --all-features --release --lib --tests --examples
cargo test -p zenmath -p magemath --all-features --doc
cargo clippy -p zenmath -p magemath --all-features --all-targets -- -D warnings
for features in '' std archmage runtime-dispatch avx512; do
    if [[ -n "$features" ]]; then
        cargo +1.89.0 check -p zenmath --no-default-features --features "$features"
    else
        cargo +1.89.0 check -p zenmath --no-default-features
    fi
done
cargo rustc --manifest-path benchmarks/zenmath_dispatch_2026-11-06/compile-consumer/Cargo.toml --release --features unified-std -- --emit=llvm-ir
cargo run --manifest-path benchmarks/zenmath_dispatch_2026-11-06/compile-consumer/Cargo.toml --release --features unified-std
python3 benchmarks/zenmath_dispatch_2026-11-06/check-unification.py benchmarks/zenmath_dispatch_2026-11-06/compile-consumer/target/release/deps/zenmath_compile_consumer-*.ll
cargo tree -p zenmath --no-default-features

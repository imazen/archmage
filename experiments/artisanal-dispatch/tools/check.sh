#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
ARTISANAL_EXPECT_TIER=v3 cargo test --workspace --all-features
ARTISANAL_EXPECT_TIER=scalar cargo test --no-default-features
python3 tools/ui.py
cargo +1.89 check --workspace --all-features
cargo check --target i686-unknown-linux-gnu --all-targets --all-features
ARTISANAL_EXPECT_TIER=scalar CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER='qemu-x86_64 -cpu Nehalem' \
    cargo test --target x86_64-unknown-linux-gnu --test behavior --features std
ARTISANAL_EXPECT_TIER=neon CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
    CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUNNER='qemu-aarch64 -L /usr/aarch64-linux-gnu -cpu cortex-a53' \
    cargo test --target aarch64-unknown-linux-gnu --test behavior --features std
: "${WASMTIME:?Set WASMTIME to the Wasmtime executable}"
CARGO_TARGET_WASM32_WASIP1_RUNNER="$WASMTIME run --env ARTISANAL_EXPECT_TIER=scalar" \
    cargo test --target wasm32-wasip1 --test behavior --features std
CARGO_TARGET_WASM32_WASIP1_RUNNER="$WASMTIME run --env ARTISANAL_EXPECT_TIER=wasm128" \
    RUSTFLAGS='-C target-feature=+simd128' \
    cargo test --target wasm32-wasip1 --test behavior --features std

CARGO_TARGET_WASM32_WASIP1_RUNNER="$WASMTIME run --env ARTISANAL_EXPECT_TIER=wasm128" \
    RUSTFLAGS='-C target-feature=+simd128,+relaxed-simd' \
    cargo test --target wasm32-wasip1 --test behavior --features std
cargo check --target aarch64-unknown-none --no-default-features
cargo check --target thumbv7m-none-eabi --no-default-features
python3 tools/codegen.py

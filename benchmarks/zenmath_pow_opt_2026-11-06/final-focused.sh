#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root, under run-heavy. Kernels are frozen at 4da80e0a.
cargo test --locked -p zenmath --all-features --release --lib --test core_math --test pow_policy --test acceleration
cargo clippy --locked -p zenmath -p magemath --all-features --lib -- -D warnings
cargo +1.89.0 check --locked -p zenmath --all-features
cargo check --locked -p zenmath --no-default-features --features correct-rounding,normal,configurable,archmage --target aarch64-unknown-none
if [[ $(uname -m) == x86_64 ]]; then
    CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER='qemu-x86_64 -cpu Nehalem' cargo test --locked -p zenmath --target x86_64-unknown-linux-gnu --no-default-features --features std,correct-rounding,configurable,runtime-dispatch --release --test acceleration
fi

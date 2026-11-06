#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/share/mise/shims:$HOME/.local/bin:$PATH"
export TMPDIR="$HOME/tmp"
echo "$BASHPID" > active.pid
cargo test -p zenmath -p magemath --lib --tests --examples
cargo test -p zenmath -p magemath --no-default-features --lib --tests --examples
cargo check -p zenmath -p magemath --no-default-features
cd benchmarks/pow_behavior_2026-11-06
cargo fmt
cargo clippy --all-targets --all-features -- -D warnings
cargo build --release --bins
cd ../..
sudo systemd-run --scope --quiet -p AllowedCPUs=0-7 -p MemoryMax=10G -p MemorySwapMax=0 runuser -u lilith -- env TMPDIR="$HOME/tmp" taskset -c 0 benchmarks/pow_behavior_2026-11-06/target/release/policy_cost policy-cost-x86.json

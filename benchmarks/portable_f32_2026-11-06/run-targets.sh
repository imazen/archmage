#!/usr/bin/env bash
# Invoked under the sole run-heavy/heavy.lock owner, on eligible x86 host.
set -euo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/share/mise/shims:$HOME/.local/bin:$PATH"
export TMPDIR="$HOME/tmp"
root=$(pwd)
harness="$root/benchmarks/portable_f32_2026-11-06"
echo "$BASHPID" > active.pid
stage() { date -u +%FT%TZ; echo "$*" | tee active-stage.txt; }
run_outputs() {
    local label=$1 binary=$2
    shift 2
    mkdir -p "artifacts/$label"
    for op in exp ln log2 pow; do
        "$@" "$binary" "$op" "artifacts/corpus/$op.bin" "artifacts/$label"
    done
}
stage boundary-corpus
python3 "$harness/boundaries.py" artifacts/corpus
stage same-binary-native
sha256sum "$harness/target/release/results" > artifacts/binary-native.sha256
run_outputs x86-native "$harness/target/release/results"
stage same-binary-no-fma
run_outputs x86-nehalem "$harness/target/release/results" qemu-x86_64 -cpu Nehalem
stage global-v3-build
CARGO_TARGET_DIR="$harness/target-v3" RUSTFLAGS='-Ctarget-cpu=x86-64-v3' cargo build --locked --release --bins --manifest-path "$harness/Cargo.toml"
run_outputs x86-v3 "$harness/target-v3/release/results"
stage force-soft-build
CARGO_TARGET_DIR="$harness/target-soft" cargo build --locked --release --bin results --features soft --manifest-path "$harness/Cargo.toml"
run_outputs x86-soft "$harness/target-soft/release/results"
stage wasm-build
CARGO_TARGET_DIR="$harness/target-wasm" RUSTFLAGS='-Ctarget-feature=+simd128' cargo build --locked --release --bin results --target wasm32-wasip1 --manifest-path "$harness/Cargo.toml"
stage true-real-analysis
python3 "$harness/analyze.py" artifacts --true-real
stage complete-x86

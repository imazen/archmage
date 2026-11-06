#!/usr/bin/env bash
set -euo pipefail
output=$(realpath "$1")
source=$(realpath "$(dirname "$0")/exp2.rs")
rustc --version --verbose > "$output/rust.version"
for cpu in x86-64 x86-64-v3; do
 rustc --edition 2024 -O -C target-cpu="$cpu" "$source" -o "$output/rust-$cpu"
 objdump -d -C -Mintel "$output/rust-$cpu" > "$output/rust-$cpu.asm"
 taskset -c "${PROBE_CPU:-2}" "$output/rust-$cpu" "$output/exp2-golden.bin" "$output/exp2-input.bin" > "$output/rust-$cpu.csv"
done

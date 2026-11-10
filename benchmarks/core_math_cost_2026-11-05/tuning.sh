#!/usr/bin/env bash
set -euo pipefail
output=$(realpath "$1")
source=$(realpath "$(dirname "$0")/exp2.rs")
for mode in clearance nightly-baseline nightly-tuned; do
 flags=()
 if [[ $mode == clearance ]]; then flags+=(-C llvm-args=-partial-reg-update-clearance=1000);
 elif [[ $mode == nightly-tuned ]]; then flags+=(+nightly -Z tune-cpu=arrowlake);
 else flags+=(+nightly); fi
 rustc "${flags[@]}" --edition 2024 -O -C target-cpu=x86-64 "$source" -o "$output/rust-$mode"
 objdump -d -C -Mintel "$output/rust-$mode" > "$output/rust-$mode.asm"
 taskset -c "${PROBE_CPU:-2}" "$output/rust-$mode" "$output/exp2-golden.bin" "$output/exp2-input.bin" > "$output/rust-$mode.csv"
done
rustc +nightly --version --verbose > "$output/rust-nightly.version"

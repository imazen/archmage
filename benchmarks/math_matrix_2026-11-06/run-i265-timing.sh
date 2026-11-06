#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.cargo/bin:$PATH" TMPDIR="$HOME/tmp" MATH_MATRIX_REQUIRE_TIERS=scalar,v3
cd "$(dirname "$0")/../.."
echo "$BASHPID" > matrix-timing.pid
sudo systemd-run --scope --quiet -p AllowedCPUs=0-7 -p MemoryMax=4G -p MemorySwapMax=0 runuser -u lilith -- env TMPDIR="$TMPDIR" MATH_MATRIX_REQUIRE_TIERS=scalar,v3 taskset -c 0 benchmarks/math_matrix_2026-11-06/target/release/timing timing-i265.json
sudo systemd-run --scope --quiet -p AllowedCPUs=0-7 -p MemoryMax=4G -p MemorySwapMax=0 runuser -u lilith -- env TMPDIR="$TMPDIR" MATH_MATRIX_REQUIRE_TIERS=scalar,v3 taskset -c 0 benchmarks/math_matrix_2026-11-06/target/release/algebraic algebraic-i265.json

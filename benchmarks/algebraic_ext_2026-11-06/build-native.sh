#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.cargo/bin:$PATH" TMPDIR="$HOME/tmp"
cd "$(dirname "$0")"
echo "$BASHPID" > build.pid
for mode in off thin; do
    if test "$mode" = off; then export CARGO_PROFILE_RELEASE_LTO=false; else export CARGO_PROFILE_RELEASE_LTO=thin; fi
    cargo build --release --features avx512 --bins --target-dir "target-$mode"
    cargo rustc --release --features avx512 --lib -p algebraic-ext-consumer --target-dir "target-$mode" -- --emit=llvm-ir,asm
    ALG_EXT_REQUIRE_TIERS=scalar,v3,v4x "target-$mode/release/results" "outputs-$mode"
done

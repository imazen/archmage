#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.cargo/bin:$PATH" TMPDIR="$HOME/tmp" RUSTUP_TOOLCHAIN=1.99.0 ALG_EXT_REQUIRE_TIERS=scalar,neon
cd "$(dirname "$0")"
echo "$BASHPID" > build.pid
cargo test --release --lib
cargo clippy --all-targets -- -D warnings
for mode in off thin; do
    if test "$mode" = off; then export CARGO_PROFILE_RELEASE_LTO=false; else export CARGO_PROFILE_RELEASE_LTO=thin; fi
    cargo build --release --bin timing --bin proof --bin results --target-dir "target-$mode"
    cargo rustc --release --lib -p algebraic-ext-consumer --target-dir "target-$mode" -- --emit=llvm-ir,asm
    "target-$mode/release/results" "outputs-$mode"
done

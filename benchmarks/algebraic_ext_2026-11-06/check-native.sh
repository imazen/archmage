#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.cargo/bin:$PATH" TMPDIR="$HOME/tmp" ALG_EXT_REQUIRE_TIERS=scalar,v3,v4x
cd "$(dirname "$0")"
cargo test --release --features avx512 --lib
cargo test --doc --features avx512
cargo clippy --all-targets --features avx512 -- -D warnings
RUSTDOCFLAGS=-Dwarnings cargo doc --no-deps --features avx512
cargo check -p algebraic-magetypes-prototype --no-default-features
cargo +1.98.1 check -p algebraic-magetypes-prototype --no-default-features
if cargo +1.89.0 check -p algebraic-magetypes-prototype --no-default-features > msrv189.log 2>&1; then echo 'ERROR: 1.89 unexpectedly accepted Rust1.98 crate'; exit 1; fi
cargo run --manifest-path published-beta/Cargo.toml --release
cargo tree --manifest-path published-beta/Cargo.toml
cargo run --manifest-path published-030/Cargo.toml --release
cargo tree --manifest-path published-030/Cargo.toml
cargo build --release --features avx512 --bin timing --bin proof

#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/share/mise/installs/just/latest:/usr/bin:/bin:/usr/local/bin"
export GAMMA_TASK_ROOT="$HOME/tmp/zenmath-gamma-opt-sol"
export CARGO_HOME="$GAMMA_TASK_ROOT/cargo"
export RUSTUP_HOME="$GAMMA_TASK_ROOT/rustup-home"
export TMPDIR="$GAMMA_TASK_ROOT/scratch"
export RUSTUP_TOOLCHAIN=1.99.1
export RUSTFLAGS='' CARGO_ENCODED_RUSTFLAGS=''
export RUSTC="$(rustup which --toolchain 1.99.1 rustc)"
export RUSTDOC="$(rustup which --toolchain 1.99.1 rustdoc)"
export CARGO_TARGET_DIR="${GAMMA_TARGET_DIR:?fresh isolated target required}"
cd "$GAMMA_TASK_ROOT/source/benchmarks/zenmath_gamma_opt_2026-11-06"
case "${1:?job required}" in
 health)
  rustup run 1.99.1 rustc -vV
  rustup run 1.99.1 cargo -V
  cd ../..
  just generate
  just validate-registry
  just validate-tokens
  just soundness
  ;;
 analysis-build)
  "$(rustup which --toolchain 1.99.1 rustfmt)" --edition 2024 src/bin/paired.rs
  cargo +1.99.1 clippy --locked --release --bin paired -- -D warnings
  cargo +1.99.1 build --locked --release --bin paired
  ;;
 paired)
  shift;exec "$CARGO_TARGET_DIR/release/paired" "$@"
  ;;
 freeze)
  python3 freeze.py "${2:?output directory required}"
  ;;
 build)
  : "${GAMMA_RUN_DIR:?unique run directory required}"
  mkdir "$GAMMA_RUN_DIR"
  python3 generate.py --rust src/generated.rs --rustfmt "$(rustup which --toolchain 1.99.1 rustfmt)" --proof "$GAMMA_RUN_DIR/bound-proof.json" --summary "$GAMMA_RUN_DIR/bound-summary.csv"
  python3 generate_piecewise.py --rust src/piecewise_generated.rs --rustfmt "$(rustup which --toolchain 1.99.1 rustfmt)" --proof "$GAMMA_RUN_DIR/piece-bound-proof.json" --summary "$GAMMA_RUN_DIR/piece-bound-summary.csv"
  python3 oracle.py --out "$GAMMA_RUN_DIR/oracle.csv"
  cargo +1.99.1 fetch --locked
  cargo +1.99.1 fmt
  cargo +1.99.1 test --locked --release
  cargo +1.99.1 clippy --locked --all-targets --release -- -D warnings
  cargo +1.99.1 build --locked --release
  "$CARGO_TARGET_DIR/release/zenmath-gamma-opt" facts
  "$CARGO_TARGET_DIR/release/zenmath-gamma-opt" oracle "$GAMMA_RUN_DIR/oracle.csv" "$GAMMA_RUN_DIR/oracle-results.csv"
  "$CARGO_TARGET_DIR/release/zenmath-gamma-opt" vector-oracle "$GAMMA_RUN_DIR/oracle.csv" "$GAMMA_RUN_DIR/vector-oracle-results.csv"
  ;;
 *) shift;exec "$CARGO_TARGET_DIR/release/zenmath-gamma-opt" "$@" ;;
esac

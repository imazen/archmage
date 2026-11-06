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
 freeze)
  python3 freeze.py "${2:?output directory required}"
  ;;
 build)
  mkdir -p "$GAMMA_TASK_ROOT/runs/initial-v3"
  python3 generate.py --rust src/generated.rs --rustfmt "$(rustup which --toolchain 1.99.1 rustfmt)" --proof "$GAMMA_TASK_ROOT/runs/initial-v3/bound-proof.json" --summary "$GAMMA_TASK_ROOT/runs/initial-v3/bound-summary.csv"
  python3 oracle.py --out "$GAMMA_TASK_ROOT/runs/initial-v3/oracle.csv"
  cargo +1.99.1 update
  cargo +1.99.1 fmt
  cargo +1.99.1 test --release
  cargo +1.99.1 clippy --all-targets --release -- -D warnings
  cargo +1.99.1 build --release
  "$CARGO_TARGET_DIR/release/zenmath-gamma-opt" facts
  "$CARGO_TARGET_DIR/release/zenmath-gamma-opt" oracle "$GAMMA_TASK_ROOT/runs/initial-v3/oracle.csv" "$GAMMA_TASK_ROOT/runs/initial-v3/oracle-results.csv"
  ;;
 *) shift;exec "$CARGO_TARGET_DIR/release/zenmath-gamma-opt" "$@" ;;
esac

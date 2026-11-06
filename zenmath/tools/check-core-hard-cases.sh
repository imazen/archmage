#!/usr/bin/env bash
# Release checks of every available upstream hard-case file in the port subset.
set -euo pipefail
records=${1:?usage: check-core-hard-cases.sh RECORD_DIRECTORY}
features=()
if [[ -n ${ZENMATH_FEATURES:-} ]]; then features=(--features "$ZENMATH_FEATURES"); fi
runner=()
if [[ -n ${ZENMATH_RUNNER:-} ]]; then read -r -a runner <<< "$ZENMATH_RUNNER"; fi
bindir="${CARGO_TARGET_DIR:-target}${CARGO_BUILD_TARGET:+/$CARGO_BUILD_TARGET}/release/examples"
suffix=""
if [[ ${CARGO_BUILD_TARGET:-} == wasm32-* ]]; then suffix=.wasm; fi
cargo build -p zenmath --release --examples "${features[@]}"
for f in powf atan2f hypotf; do
    "${runner[@]}" "$bindir/core_math_check$suffix" "$f" "$records/$f-wc.bin"
done
for f in cbrt exp2 exp log2 log log10 atan log1p sin cos atan2 tan pow atanh tanh asinh; do
    "${runner[@]}" "$bindir/core_math_check64$suffix" "$f" "$records/$f-wc.bin"
done

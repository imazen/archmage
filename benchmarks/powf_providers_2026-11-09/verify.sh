#!/usr/bin/env bash
set -euo pipefail
directory=$1
fixtures=$2
tier=${3:-baseline}
study=benchmarks/powf_providers_2026-11-09
[[ $tier == baseline || $tier == v3 ]]
test ! -e "$directory/manifest-verified.json"
cargo +nightly-2026-10-06 fmt --manifest-path "$study/Cargo.toml" --check
cargo +nightly-2026-10-06 clippy --release --locked --manifest-path "$study/Cargo.toml" --all-targets -- -D warnings
just powf-external-build
just powf-external-native-build "$directory/native-worker-verified"
cp "$study/target/release/rust_provider_worker" "$directory/rust-worker-verified"
cp "$study/target/release/external_providers_bench" "$directory/bench-verified"
options=()
if [[ $tier == v3 ]];then
 "$directory/native-worker-verified" --list > "$directory/native-list.txt" 2> "$directory/native-list.stderr"
 grep -q '^glibc_2_43_libmvec_AVX2x8 ' "$directory/native-list.txt"
 CARGO_TARGET_DIR="$directory/target-v3" RUSTFLAGS='-C target-feature=+avx2,+fma' \
  cargo +nightly-2026-10-06 build --release --locked --manifest-path "$study/Cargo.toml" --bin rust_provider_worker
 cp "$directory/target-v3/release/rust_provider_worker" "$directory/rust-worker-v3-verified"
 options=(--v3 "$directory/rust-worker-v3-verified")
fi
just powf-external-discover "$directory/rust-worker-verified" "$directory/native-worker-verified" "$directory/manifest-verified.json" "${options[@]}"
just powf-external-accuracy "$directory/manifest-verified.json" "$fixtures" "$directory/accuracy-verified"
just powf-external-traces "$directory/manifest-verified.json" "$directory/trajectories-verified"

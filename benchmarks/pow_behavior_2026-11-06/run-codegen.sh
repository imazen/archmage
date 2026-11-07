#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/share/mise/shims:$HOME/.local/bin:$PATH" TMPDIR="$HOME/tmp"
root=$(pwd)
echo "$BASHPID" > active.pid
mkdir -p artifacts/codegen
python3 benchmarks/pow_behavior_2026-11-06/compare_results.py artifacts
cargo check -p zenmath -p magemath --no-default-features --target aarch64-unknown-none
cd benchmarks/pow_behavior_2026-11-06
cargo fmt
for mode in base extra o0; do
    echo "codegen-$mode" > "$root/active-stage.txt"
    target="$PWD/target-codegen-$mode"
    flags=(-Ctarget-cpu=x86-64-v3)
    features=()
    if [ "$mode" = extra ]; then features=(--features probe-extra); fi
    export CARGO_PROFILE_RELEASE_OPT_LEVEL=3 CARGO_PROFILE_RELEASE_LTO=thin CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS=false CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1
    if [ "$mode" = o0 ]; then export CARGO_PROFILE_RELEASE_OPT_LEVEL=0 CARGO_PROFILE_RELEASE_LTO=false; fi
    /usr/bin/time -f 'wall_seconds=%e user_seconds=%U system_seconds=%S max_rss_kb=%M' -o "$root/artifacts/codegen/build-$mode.txt" env CARGO_TARGET_DIR="$target" RUSTFLAGS="${flags[*]}" cargo rustc --locked --release --bin codegen "${features[@]}" -- --emit=asm,link
    "$target/release/codegen"
    cp "$target/release/codegen" "$root/artifacts/codegen/$mode"
    nm -S -C "$target/release/codegen" > "$root/artifacts/codegen/symbols-$mode.txt"
    size -A "$target/release/codegen" > "$root/artifacts/codegen/size-$mode.txt"
    objdump -d --demangle "$target/release/codegen" > "$root/artifacts/codegen/disassembly-$mode.txt"
    find "$target/release/deps" -maxdepth 1 -name 'codegen-*.s' -exec cp '{}' "$root/artifacts/codegen/" \;
done
echo codegen-complete > "$root/active-stage.txt"

#!/bin/bash
# Compile-time measurements for zenmath and magemath, three runs each, as TSV.
# Run on a quiet host: compile_time.sh <checkout> <scratch dir>
set -euo pipefail
SRC=$(realpath "$1")
# Preserve previous measurements; each invocation needs a new directory.
mkdir "$2"
WORK=$(realpath "$2")
export CARGO_TERM_QUIET=true CARGO_INCREMENTAL=0
t() {
  local s=$(date +%s.%N)
  if ! "$@" >"$WORK/command.log" 2>&1; then
    echo "FAILED: $*" >&2
    cat "$WORK/command.log" >&2
    return 1
  fi
  local e=$(date +%s.%N)
  printf '%.2f' "$(echo "$e - $s" | bc)"
}
cold() { # label, checkout, cargo args...
  local label=$1 dir=$2; shift 2; printf '%s' "$label"
  for _ in 1 2 3; do
    rm -rf "$WORK/t"
    local elapsed
    elapsed=$(cd "$dir" && CARGO_TARGET_DIR=$WORK/t t cargo "$@")
    printf '\t%s' "$elapsed"
  done
  printf '\n'
}
echo -e "what\trun1\trun2\trun3"
cold "zenmath check" "$SRC" check -p zenmath
cold "zenmath build --release" "$SRC" build -p zenmath --release
cold "magetypes check, no w512" "$SRC" check -p magetypes --no-default-features --features std
cold "magetypes check, default (w512)" "$SRC" check -p magetypes
cold "magemath check" "$SRC" check -p magemath
cold "magemath build --release" "$SRC" build -p magemath --release
cold "magemath check --features avx512" "$SRC" check -p magemath --features avx512
cold "magemath build --release --features avx512" "$SRC" build -p magemath --release --features avx512

# Isolate the cost of each additional surface, with the same dependencies and bodies.
for surfaces in portable portable_relaxed; do
  dir="$WORK/$surfaces"
  mkdir "$dir"
  rsync -a --exclude target --exclude .git --exclude .jj --exclude .workongoing "$SRC"/ "$dir"/
  sed -i '/^pub mod unfused;/d' "$dir/magemath/src/lib.rs"
  if [ "$surfaces" = portable ]; then
    sed -i '/^pub mod relaxed;/d' "$dir/magemath/src/lib.rs"
  fi
  cold "magemath $surfaces check" "$dir" check -p magemath
  cold "magemath $surfaces build --release" "$dir" build -p magemath --release
done

# What the #[inline] dispatchers save, on a scratch copy without them.
mkdir -p "$WORK/noinline"
rsync -a --exclude target --exclude .git --exclude .jj "$SRC"/ "$WORK/noinline"/
sed -i 's/dispatch(inline(hint))/dispatch/' \
  "$WORK/noinline/magemath/src/surface.rs"
cold "magemath build --release, non-inline dispatchers" "$WORK/noinline" build -p magemath --release

# A downstream crate: what the ops it calls cost in its own build.
mk() { # name, body
  local d=$WORK/$1; mkdir -p "$d/src"
  cat > "$d/Cargo.toml" <<TOML
[package]
name = "$1"
version = "0.0.0"
edition = "2024"
[dependencies]
magemath = { path = "$SRC/magemath" }
[workspace]
TOML
  printf 'use magemath::portable::slice;\nfn main() {\n    let mut v = vec![0.5f32; std::env::args().count() * 1000];\n%b    println!("{}", v[0]);\n}\n' "$2" > "$d/src/main.rs"
}
ALL=""
for s in sqrt_f32 floor_f32 ceil_f32 roundeven_f32 log2_lowp_f32 exp2_lowp_f32 ln_lowp_f32 \
  exp_lowp_f32 log10_lowp_f32 cbrt_lowp_f32 log2_midp_f32 exp2_midp_f32 ln_midp_f32 exp_midp_f32 \
  log10_midp_f32 cbrt_midp_f32 cbrt_midp_precise_f32 sigmoid_midp_f32 silu_midp_f32; do
  ALL="$ALL    slice::$s(&mut v);\n"
done
ALL="$ALL    slice::pow_midp_f32(&mut v, 2.4);\n    slice::pow_lowp_f32(&mut v, 2.4);\n"
mk use_none "    let _ = slice::sqrt_f32 as fn(&mut [f32]); v[0] = 1.0;\n"
mk use_one "    slice::exp2_midp_f32(&mut v);\n"
mk use_all "$ALL"
for c in use_none use_one use_all; do
  (cd "$WORK/$c" && cargo build --release >/dev/null 2>&1)
  printf '%s rebuild of the crate alone' "$c"
  for _ in 1 2 3; do
    touch "$WORK/$c/src/main.rs"
    elapsed=$(cd "$WORK/$c" && t cargo build --release)
    printf '\t%s' "$elapsed"
  done
  printf '\n'
done
for c in use_none use_one use_all; do
  cp "$WORK/$c/target/release/$c" "$WORK/$c.bin"; strip "$WORK/$c.bin"
  printf '%s stripped binary bytes\t%s\n' "$c" "$(stat -c %s "$WORK/$c.bin")"
done

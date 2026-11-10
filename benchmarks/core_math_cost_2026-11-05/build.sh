#!/usr/bin/env bash
set -euo pipefail
upstream=$(realpath "$1")
output=$(realpath -m "$2")
mkdir -p "$output"
sources=()
for f in exp2 log2 exp log cbrt sin tanh pow; do sources+=("$upstream/src/binary32/$f/${f}f.c"); done
for cc in gcc /opt/intel/oneapi/compiler/latest/bin/icx; do
 name=$(basename "$cc")
 "$cc" --version > "$output/$name.version"
 for contract in off on; do
 for cpu in x86-64 x86-64-v3; do
  flags=(-O3 -march="$cpu" -frounding-math)
  if [[ $name == icx ]]; then flags+=(-fp-model=strict); fi
  flags+=(-ffp-contract="$contract")
  suffix="$name-$cpu"
  if [[ $contract == on ]]; then suffix+=-contract; fi
  objects=()
  mkdir -p "$output/$suffix"
  for source in "${sources[@]}"; do
   object="$output/$suffix/$(basename "${source%.c}").o"
   "$cc" "${flags[@]}" -fPIC -c "$source" -o "$object"
   objects+=("$object")
  done
  # Use the same system libm for both compilers' explicit builtin libcalls.
  gcc -shared "${objects[@]}" -lm -o "$output/$suffix.so"
  objdump -d -Mintel "$output/$suffix.so" > "$output/$suffix.asm"
 done
 done
done
prefix=$HOME/tmp/core-math-study/deps/usr/include
gcc -O3 -ffp-contract=off -frounding-math -I"$prefix" -I"$prefix/x86_64-linux-gnu" "$(dirname "$0")/probe.c" -l:libmpfr.so.6 -l:libgmp.so.10 -lm -ldl -o "$output/probe"

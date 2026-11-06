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
 for cpu in x86-64 x86-64-v3; do
  flags=(-O3 -march="$cpu" -ffp-contract=off -frounding-math)
  if [[ $name == icx ]]; then flags+=(-fp-model=strict); fi
  "$cc" "${flags[@]}" -fPIC -shared "${sources[@]}" -lm -o "$output/$name-$cpu.so"
  objdump -d -Mintel "$output/$name-$cpu.so" > "$output/$name-$cpu.asm"
 done
done
prefix=$HOME/tmp/core-math-study/deps/usr/include
gcc -O3 -ffp-contract=off -frounding-math -I"$prefix" -I"$prefix/x86_64-linux-gnu" "$(dirname "$0")/probe.c" -l:libmpfr.so.6 -l:libgmp.so.10 -lm -ldl -o "$output/probe"

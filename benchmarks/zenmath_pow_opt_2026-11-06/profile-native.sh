#!/usr/bin/env bash
set -euo pipefail
binary=$(realpath "$1")
out=$(realpath -m "$2")
mkdir "$out"
for variant in current/correct ordinary/correct current/midp-pre-fused; do
    label=${variant//\//-}
    taskset -c 0 perf record -F 999 -g --call-graph dwarf -o "$out/$label.data" -- "$binary" gamma-decode "$variant" 2000 > "$out/$label.run.log" 2>&1
    perf report --stdio --no-children -i "$out/$label.data" > "$out/$label.report.txt"
    taskset -c 0 perf stat -e cycles,instructions,branches,branch-misses -o "$out/$label.stat.txt" -- "$binary" gamma-decode "$variant" 1000 > "$out/$label.stat.run.log" 2>&1
done

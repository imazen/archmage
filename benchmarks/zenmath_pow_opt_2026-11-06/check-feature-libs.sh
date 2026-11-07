#!/usr/bin/env bash
set -euo pipefail
for families in '' correct-rounding normal configurable correct-rounding,normal correct-rounding,configurable normal,configurable correct-rounding,normal,configurable; do
    for backend in '' archmage runtime-dispatch; do
        flags="$families"
        if [[ -n "$backend" ]]; then flags="${flags:+$flags,}zenmath/$backend"; fi
        if [[ -n "$flags" ]]; then
            cargo clippy -p zenmath -p magemath --no-default-features --features "$flags" --lib -- -D warnings
        else
            cargo clippy -p zenmath -p magemath --no-default-features --lib -- -D warnings
        fi
    done
done

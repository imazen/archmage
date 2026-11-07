#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p timed-codegen
for mode in off thin; do
    file="target-$mode/release/timing"
    sha256sum "$file" > "timed-codegen/$mode.sha256"
    objdump -d -C "$file" > "timed-codegen/$mode.disasm"
    nm -n -C "$file" > "timed-codegen/$mode.nm"
done

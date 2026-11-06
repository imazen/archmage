#!/usr/bin/env bash
# Caller supplies binary/runner; invoke only while owning heavy.lock + run-heavy.
set -euo pipefail
: "${REPLAY_ROOT:?directory containing corpus/}"
: "${REPLAY_LABEL:?target label}"
: "${RESULTS_BINARY:?native executable or WASM module}"
echo "$BASHPID" > "${REPLAY_ROOT}/active.pid"
for op in exp ln log2 pow; do
    if [[ "$RESULTS_BINARY" = *.wasm ]]; then
        wasmtime run --dir . "$RESULTS_BINARY" "$op" "$REPLAY_ROOT/corpus/$op.bin" "$REPLAY_ROOT/$REPLAY_LABEL"
    else
        "$RESULTS_BINARY" "$op" "$REPLAY_ROOT/corpus/$op.bin" "$REPLAY_ROOT/$REPLAY_LABEL"
    fi
done

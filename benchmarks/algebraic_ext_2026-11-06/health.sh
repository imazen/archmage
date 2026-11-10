#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.cargo/bin:$PATH" TMPDIR="$HOME/tmp"
cd "$(dirname "$0")/../.."
cargo run -p xtask -- generate
cargo run -p xtask -- validate-registry
cargo run -p xtask -- validate
cargo run -p xtask -- soundness

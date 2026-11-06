#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/share/mise/shims:$HOME/.local/bin:$PATH"
date -u +%FT%TZ
uname -a
cat /etc/os-release
rustc -Vv
cargo -V
ldd --version
command -v qemu-x86_64 >/dev/null && qemu-x86_64 --version
command -v wasmtime >/dev/null && wasmtime --version
lscpu
printf 'Compile environment flags: RUSTFLAGS=%s CARGO_ENCODED_RUSTFLAGS=%s\n' "${RUSTFLAGS-}" "${CARGO_ENCODED_RUSTFLAGS-}"

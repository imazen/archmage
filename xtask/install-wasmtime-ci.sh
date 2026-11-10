#!/usr/bin/env bash
# Keep CI's engine bytes fixed while avoiding failing release-page redirects.
set -euo pipefail

[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || {
    echo 'This CI installer requires Linux x86_64.' >&2
    exit 1
}
version=v48.0.1
# Immutable release asset and GitHub's published SHA-256 for this exact archive.
# https://github.com/bytecodealliance/wasmtime/releases/tag/v48.0.1
asset=528119176
digest=4c2e31b68ad99e0a519f225a261fda099eb15f056d4a24fdb3c2a46517bde1df
mkdir -p "$HOME/tmp"
scratch=$(mktemp -d "$HOME/tmp/archmage-wasmtime-ci.XXXXXXXX")
archive="$scratch/wasmtime.tar.xz"
auth=()
if [[ -n ${GH_TOKEN:-} ]]; then
    auth=(--header "Authorization: Bearer $GH_TOKEN")
fi
curl --fail --location --show-error --retry 4 --retry-all-errors \
    --retry-delay 2 --retry-max-time 240 --connect-timeout 15 --max-time 90 \
    --header 'Accept: application/octet-stream' "${auth[@]}" \
    --output "$archive" \
    "https://api.github.com/repos/bytecodealliance/wasmtime/releases/assets/$asset"
printf '%s  %s\n' "$digest" "$archive" | sha256sum --check
tar -xJf "$archive" -C "$scratch"
source_binary="$scratch/wasmtime-$version-x86_64-linux/wasmtime"
install_root=${WASMTIME_CI_HOME:-"$HOME/.wasmtime"}
mkdir -p "$install_root/bin"
destination="$install_root/bin/wasmtime"
if [[ -e "$destination" ]]; then
    mv "$destination" "$destination.bak.$(date -u +%Y%m%dT%H%M%SZ).$$"
fi
install -m 755 "$source_binary" "$destination"
reported=$("$destination" --version)
[[ "$reported" == 'wasmtime 48.0.1'* ]]
printf '%s\n' "$reported"

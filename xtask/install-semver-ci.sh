#!/usr/bin/env bash
# Preserve the checker's pinned bytes without depending on release-page redirects.
set -euo pipefail

[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || {
    echo 'This CI installer requires Linux x86_64.' >&2
    exit 1
}
# https://github.com/obi1kenobi/cargo-semver-checks/releases/tag/v0.51.0
asset=608677291
digest=396c8508d973c83de52660943a057e971b586f7511c3ba695762373452b43d5b
mkdir -p "$HOME/tmp"
scratch=$(mktemp -d "$HOME/tmp/archmage-semver-ci.XXXXXXXX")
archive="$scratch/cargo-semver-checks.tar.gz"
auth=()
if [[ -n ${GH_TOKEN:-} ]]; then
    auth=(--header "Authorization: Bearer $GH_TOKEN")
fi
curl --fail --location --show-error --retry 4 --retry-all-errors \
    --retry-delay 2 --retry-max-time 240 --connect-timeout 15 --max-time 90 \
    --header 'Accept: application/octet-stream' "${auth[@]}" \
    --output "$archive" \
    "https://api.github.com/repos/obi1kenobi/cargo-semver-checks/releases/assets/$asset"
printf '%s  %s\n' "$digest" "$archive" | sha256sum --check
tar -xzf "$archive" -C "$scratch"
install_root=${SEMVER_CI_HOME:-${CARGO_HOME:-"$HOME/.cargo"}}
mkdir -p "$install_root/bin"
destination="$install_root/bin/cargo-semver-checks"
if [[ -e "$destination" ]]; then
    mv "$destination" "$destination.bak.$(date -u +%Y%m%dT%H%M%SZ).$$"
fi
install -m 755 "$scratch/cargo-semver-checks" "$destination"
reported=$("$destination" --version)
[[ "$reported" == 'cargo-semver-checks 0.51.0'* ]]
printf '%s\n' "$reported"

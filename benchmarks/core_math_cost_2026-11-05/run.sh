#!/usr/bin/env bash
set -euo pipefail
cd "$1"
exec taskset -c "${PROBE_CPU:-2}" ./probe ./gcc-x86-64.so ./gcc-x86-64-v3.so ./icx-x86-64.so ./icx-x86-64-v3.so ./gcc-x86-64-contract.so ./gcc-x86-64-v3-contract.so ./icx-x86-64-contract.so ./icx-x86-64-v3-contract.so libm.so.6

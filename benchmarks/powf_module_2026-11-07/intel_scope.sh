#!/usr/bin/env bash
# Constrain this run-heavy scope, preserving its memory cap and common gates.
set -euo pipefail
cpus=$1
shift
path=$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup)
unit=${path##*/}
case "$unit" in run-p*-i*.scope) ;; *) echo 'requires the current transient run-heavy cgroup scope' >&2; exit 2;; esac
cap=$(cat "/sys/fs/cgroup${path}/memory.max")
[[ $cap != max ]] || { echo 'requires a hard memory cap' >&2; exit 2; }
case "$path" in
    /system.slice/*) sudo -n systemctl set-property --runtime "$unit" "AllowedCPUs=$cpus" ;;
    /user.slice/*) systemctl --user set-property --runtime "$unit" "AllowedCPUs=$cpus" ;;
    *) echo 'unexpected current cgroup placement' >&2; exit 2;;
esac
[[ -r "/sys/fs/cgroup${path}/cpuset.cpus.effective" ]] || { echo 'cpuset controller is unavailable in this scope' >&2; exit 2; }
echo "verified scope memory.max=$cap cpuset=$(cat "/sys/fs/cgroup${path}/cpuset.cpus.effective")"
export ZENBENCH_CPU_POLICY=performance ZENBENCH_CPU_SET="$cpus"
exec "$@"

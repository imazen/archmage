#!/usr/bin/env python3
"""Run every nontrapping WASI test plus a separately asserted exact abort."""

from pathlib import Path
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("engine")
parser.add_argument("target")
parser.add_argument("log")
parser.add_argument("--require-tiers", required=True)
parser.add_argument("--forbid-tiers", default="")
parser.add_argument("--disable-simd", action="store_true")
args = parser.parse_args()
engine = args.engine
binaries = list(Path(args.target).glob("acceleration-*.wasm"))
assert len(binaries) == 1, binaries
binary = str(binaries[0])
base = [engine, "run"]
if args.disable_simd:
    base += ["-W", "simd=n", "-W", "relaxed-simd=n"]
base += [
    "--env",
    "ZENMATH_REQUIRE_TIERS=" + args.require_tiers,
    "--env",
    "ZENMATH_FORBID_TIERS=" + args.forbid_tiers,
    binary,
]
subprocess.run(
    base + ["--skip", "power_length_failure_precedes_mutation", "--nocapture"],
    check=True,
)
negative = subprocess.run(
    base + ["--exact", "power_length_failure_precedes_mutation", "--nocapture"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)
Path(args.log).write_bytes(negative.stdout)
assert negative.returncode != 0
for expected in [b"power input lengths must match", b"left: 3", b"right: 2"]:
    assert expected in negative.stdout, expected
print(
    f"PASS exact WASI length trap rc={negative.returncode}; native verifies unchanged output"
)

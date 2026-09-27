#!/usr/bin/env python3
"""Pin the unresolved #117 failure; this is NOT a passing compatibility claim."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--target", help="defaults to the native rustc host")
args = parser.parse_args()
if args.target is None:
    rustc = subprocess.check_output(["rustc", "-vV"], text=True, encoding="utf-8")
    args.target = next(line.removeprefix("host: ") for line in rustc.splitlines()
                       if line.startswith("host: "))
manifest = Path(__file__).with_name("Cargo.toml")
command = ["cargo", "check", "--manifest-path", str(manifest), "--target", args.target,
           "--message-format=json"]
result = subprocess.run(command, stdout=subprocess.PIPE, text=True, encoding="utf-8")
errors = []
for line in result.stdout.splitlines():
    message = json.loads(line)
    if message.get("reason") != "compiler-message":
        continue
    diagnostic = message["message"]
    if diagnostic["level"] == "error":
        errors.append(diagnostic)
        print(diagnostic.get("rendered", diagnostic["message"]), file=sys.stderr)
expected_line = {"aarch64-unknown-linux-gnu": 421, "wasm32-wasip1": 563,
                 "wasm32-unknown-unknown": 563}.get(args.target)
if expected_line is None:
    assert result.returncode == 0, "published x86 consumer must compile"
else:
    assert result.returncode != 0, "#117 now compiles: remove the expected-failure assertion"
    assert len(errors) == 1, f"expected only the known conversion error, got {len(errors)}"
    error = errors[0]
    assert error.get("code", {}).get("code") == "E0061", error
    assert "2 arguments" in error["message"] and "1 argument" in error["message"], error
    assert any(span["is_primary"] and span["file_name"].replace("\\", "/").endswith("/src/dequant.rs")
               and span["line_start"] == expected_line for span in error["spans"]), error
    print("KNOWN INCOMPATIBILITY #117: one-argument from_i32x4 remains; all other code compiled.")

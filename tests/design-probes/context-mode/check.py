#!/usr/bin/env python3
"""Check design probes serially; preserve each compiler diagnostic in full."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--log-dir", type=Path, required=True)
args = parser.parse_args()
args.log_dir.mkdir(parents=True, exist_ok=True)
manifest = Path(__file__).resolve().parent / "Cargo.toml"
cases = [
    ("positive", None),
    ("reject-missing-context", "E0133"),
    ("reject-weaker-context", "E0133"),
    ("reject-implicit-conversion", "E0308"),
    ("reject-legacy-inference", "E0034"),
    ("reject-local-missing-context", "E0133"),
]
results = []
# Attribute arguments are token streams: `use` being a keyword is not a Rust
# grammar barrier. Keep this separate from the production magetypes parser.
probe_dir = manifest.parent
artifacts = args.log_dir.resolve() / "keyword-artifacts"
artifacts.mkdir(exist_ok=True)
suffix = ".dll" if sys.platform == "win32" else ".dylib" if sys.platform == "darwin" else ".so"
macro = artifacts / ("libkeyword_attribute" + suffix)
for name, command in [
    ("keyword-macro", ["rustc", "--edition=2024", "--crate-type=proc-macro", str(probe_dir / "keyword_attribute.rs"), "-o", str(macro)]),
    ("keyword-use", ["rustc", "--edition=2024", "--crate-type=lib", "--emit=metadata", str(probe_dir / "keyword_use.rs"), "--extern", f"keyword_attribute={macro}", "-o", str(artifacts / "keyword_use.rmeta")]),
]:
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log = args.log_dir / (name + ".log")
    log.write_text(result.stdout)
    results.append({"case": name, "command": command, "exit_code": result.returncode,
                    "expected_diagnostic": None, "passed": result.returncode == 0, "log": str(log)})
    (args.log_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(("PASS" if result.returncode == 0 else "FAIL") + " " + name + " " + str(log), flush=True)
    if result.returncode != 0:
        raise SystemExit(1)
for feature, expected in cases:
    command = ["cargo", "check", "--locked", "--manifest-path", str(manifest)]
    if expected:
        command += ["--features", feature]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log = args.log_dir / (feature + ".log")
    log.write_text(result.stdout)
    passed = result.returncode == 0 if expected is None else (
        result.returncode != 0 and "error[" + expected + "]" in result.stdout
    )
    results.append({"case": feature, "command": command, "exit_code": result.returncode,
                    "expected_diagnostic": expected, "passed": passed, "log": str(log)})
    (args.log_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(("PASS" if passed else "FAIL") + " " + feature + " " + str(log), flush=True)
if not all(row["passed"] for row in results):
    raise SystemExit(1)

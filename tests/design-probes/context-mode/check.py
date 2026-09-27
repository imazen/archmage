#!/usr/bin/env python3
"""Check design probes serially; preserve each compiler diagnostic in full."""
from pathlib import Path
import argparse
import json
import subprocess

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

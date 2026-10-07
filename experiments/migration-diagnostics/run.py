#!/usr/bin/env python3
"""Probe diagnostics on installed stable Rust; save every command and result."""
import argparse
import json
import pathlib
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=pathlib.Path, required=True)
args = parser.parse_args()
out = args.out.resolve()
out.mkdir(parents=True, exist_ok=False)
root = pathlib.Path(__file__).resolve().parent
results = []


def run(name, command, expected=0):
    result = subprocess.run(command, text=True, capture_output=True)
    (out / f"{name}.stdout").write_text(result.stdout)
    (out / f"{name}.stderr").write_text(result.stderr)
    results.append({"case": name, "command": command, "exit": result.returncode,
                    "expected": expected})
    (out / "commands.json").write_text(json.dumps(results, indent=2) + "\n")
    assert result.returncode == expected, (name, result.stderr)
    return result


run("version", ["rustc", "--version", "--verbose"])
library = str(out / "libdiagnostic_probe.so")
run("macros", ["rustc", "--edition=2024", "--crate-type=proc-macro",
               "--crate-name=diagnostic_probe", str(root / "macros.rs"), "-o", library])
base = ["rustc", "--edition=2024", str(root / "consumer.rs"), "--extern",
        f"diagnostic_probe={library}", "--error-format=json",
        "-Dunfulfilled_lint_expectations", "-o", str(out / "consumer")]
compiled = run("consumer", base)
diagnostics = [json.loads(line) for line in compiled.stderr.splitlines()]
warnings = [d for d in diagnostics if (d.get("code") or {}).get("code") == "deprecated"]
assert len(warnings) == 3, warnings
messages = [w["message"] for w in warnings]
assert any("\n#[inline(always)]" in m for m in messages), messages
assert any("evaluations += 1" in m for m in messages), messages
assert any("Static macro-level note" in m for m in messages), messages
# Deprecation messages are human-readable text, not rustfix edits.
def spans(diagnostic):
    yield from diagnostic.get("spans", [])
    for child in diagnostic.get("children", []):
        yield from spans(child)
assert all(s.get("suggested_replacement") is None for d in warnings for s in spans(d))
run("runtime", [str(out / "consumer")])
denied = run("deny", base + ["-Ddeprecated"], expected=1)
assert "ARCHMAGE-MIGRATE probe" in denied.stderr
unstable = run("diagnostic_api", ["rustc", "--edition=2024", "--crate-type=proc-macro",
                                  str(root / "nightly_api.rs"), "--error-format=json",
                                  "-o", str(out / "libnightly_api.so")], expected=1)
assert '"code":"E0658"' in unstable.stderr and "proc_macro_diagnostic" in unstable.stderr
(out / "summary.json").write_text(json.dumps({
    "dynamic_attribute_warning": True,
    "dynamic_expression_warning": True,
    "static_macro_deprecation": True,
    "allow_and_expect_respected": True,
    "deny_promotes_to_error": True,
    "argument_evaluated_once": True,
    "machine_applicable_suggestions": False,
    "proc_macro_diagnostic_requires_unstable": True,
}, indent=2) + "\n")
print("PASS: dynamic and static warnings, lint controls, single evaluation, JSON inspection, unstable API rejection")

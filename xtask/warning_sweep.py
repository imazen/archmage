#!/usr/bin/env python3
"""Run clippy over every target/feature/CPU configuration and summarize warnings.

Downstream crates see two kinds of archmage warnings: diagnostics in macro
output (they land in the caller's crate, so `cargo clippy -D warnings` fails
there even for registry dependencies) and diagnostics in our own sources on
targets or feature sets that CI does not lint. Each configuration runs
`cargo clippy --message-format=json`; the report groups diagnostics by lint,
file, and whether the primary span comes from a macro expansion.

    python3 xtask/warning_sweep.py                 # all configurations
    python3 xtask/warning_sweep.py --only wasm     # names containing "wasm"
    python3 xtask/warning_sweep.py --pedantic      # add clippy::pedantic; report
                                                   # expansion spans only
    python3 xtask/warning_sweep.py --list

Exit status is 1 when any diagnostic remains (2 for a configuration that
failed to build). Full records go to target/warning-sweep/<name>.jsonl.
"""
import argparse
import collections
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "target/warning-sweep"

WS = ["--workspace", "--all-targets"]
# Benches use zenbench, which does not build for wasm32; CI never builds them there.
WS_WASM = ["--workspace", "--lib", "--bins", "--tests", "--examples"]
LIBS = ["-p", "archmage", "-p", "magetypes", "--lib"]
AVX512 = ["--features", "archmage/avx512,magetypes/avx512"]
SIMD128 = "-Ctarget-feature=+simd128"

# (name, target or None for host, cargo args, extra RUSTFLAGS)
CONFIGS = [
    ("x86_64", None, WS, ""),
    ("x86_64-avx512", None, WS + AVX512, ""),
    ("x86_64-testable", None, WS + AVX512 + ["--features", "archmage/testable_dispatch"], ""),
    # An explicit --target keeps CPU flags off build scripts and proc macros,
    # which run on the (possibly older) host CPU.
    ("x86_64-v3-cpu", "x86_64-unknown-linux-gnu", WS + AVX512, "-Ctarget-cpu=x86-64-v3"),
    ("x86_64-v4-cpu", "x86_64-unknown-linux-gnu", WS + AVX512, "-Ctarget-cpu=x86-64-v4"),
    ("x86_64-nostd", None, LIBS + ["--no-default-features"], ""),
    ("x86_64-nostd-avx512", None, LIBS + ["--no-default-features", "--features", "archmage/avx512,magetypes/avx512"], ""),
    ("i686", "i686-unknown-linux-gnu", WS + AVX512, ""),
    ("aarch64-linux", "aarch64-unknown-linux-gnu", WS + AVX512, ""),
    ("aarch64-linux-nostd", "aarch64-unknown-linux-gnu", LIBS + ["--no-default-features"], ""),
    ("aarch64-testable", "aarch64-unknown-linux-gnu", WS + ["--features", "archmage/testable_dispatch"], ""),
    ("aarch64-macos", "aarch64-apple-darwin", WS + AVX512, ""),
    ("aarch64-windows", "aarch64-pc-windows-msvc", WS + AVX512, ""),
    ("aarch64-v3-cpu", "aarch64-unknown-linux-gnu", WS, "-Ctarget-cpu=neoverse-v2"),
    ("wasm32-simd128", "wasm32-wasip1", WS_WASM + AVX512, SIMD128),
    ("wasm32-relaxed", "wasm32-wasip1", WS_WASM, SIMD128 + ",+relaxed-simd"),
    ("wasm32-scalar", "wasm32-unknown-unknown", LIBS, ""),
    ("armv7", "armv7-unknown-linux-gnueabihf", LIBS, ""),
    ("thumbv7em-nostd", "thumbv7em-none-eabihf", LIBS + ["--no-default-features"], ""),
]


def expansion_chain(span):
    names = []
    exp = span.get("expansion")
    while exp:
        names.append(exp.get("macro_decl_name", "?"))
        exp = exp.get("span", {}).get("expansion")
    return names


def run(name, target, args, rustflags, pedantic):
    env = os.environ.copy()
    env.setdefault("CARGO_TARGET_DIR", str(ROOT / "target/sweep"))
    flags = " ".join(f for f in [env.get("RUSTFLAGS", ""), rustflags] if f)
    env["RUSTFLAGS"] = flags
    cmd = ["cargo", "clippy", "--keep-going", "--message-format=json", *args]
    if target:
        cmd += ["--target", target]
    cmd += ["--"]
    if pedantic:
        cmd += ["-W", "clippy::pedantic"]
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    records = []
    seen = set()
    for line in proc.stdout.splitlines():
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("reason") != "compiler-message":
            continue
        diag = msg["message"]
        if diag["level"] not in ("warning", "error"):
            continue
        if diag["message"].endswith("generated") or "warnings emitted" in diag["message"]:
            continue
        spans = [s for s in diag["spans"] if s.get("is_primary")] or diag["spans"]
        if not spans:
            continue
        span = spans[0]
        chain = expansion_chain(span)
        code = (diag.get("code") or {}).get("code") or diag["message"]
        rec = {
            "config": name,
            "level": diag["level"],
            "code": code,
            "message": diag["message"],
            "file": span["file_name"],
            "line": span["line_start"],
            "macro": chain,
            "target": msg.get("target", {}).get("name"),
            "kind": msg.get("target", {}).get("kind", []),
        }
        if pedantic and not chain:
            continue
        key = (rec["level"], code, rec["file"], rec["line"], rec["message"])
        if key in seen:
            continue
        seen.add(key)
        records.append(rec)
    built = proc.returncode == 0 or bool(records)
    return records, proc.returncode, proc.stderr if not built else ""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--pedantic", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true", help="print every diagnostic")
    args = parser.parse_args()
    configs = [c for c in CONFIGS if not args.only or any(o in c[0] for o in args.only)]
    if args.list:
        for c in configs:
            print(c[0])
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    status = 0
    total = collections.Counter()
    for name, target, cargs, rustflags in configs:
        print(f"== {name}", flush=True)
        records, rc, err = run(name, target, cargs, rustflags, args.pedantic)
        suffix = "-pedantic" if args.pedantic else ""
        with open(OUT / f"{name}{suffix}.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        if rc != 0 and not records:
            print(f"   BUILD FAILED (rc={rc})\n{err[-3000:]}")
            status = 2
            continue
        groups = collections.Counter()
        for r in records:
            origin = "macro:" + r["macro"][-1] if r["macro"] else "source"
            groups[(r["code"], r["file"], origin)] += 1
            total[(r["code"], origin)] += 1
            if args.verbose:
                print(f"   {r['level']} {r['code']} {r['file']}:{r['line']} [{origin}] {r['message']}")
        for (code, file, origin), n in sorted(groups.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"   {n:4} {code:45} {origin:22} {file}")
        if records:
            status = max(status, 1)
        else:
            print("   clean")
    print("== totals by lint/origin")
    for (code, origin), n in sorted(total.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"   {n:4} {code:45} {origin}")
    return status


if __name__ == "__main__":
    sys.exit(main())

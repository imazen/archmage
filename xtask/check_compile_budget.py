#!/usr/bin/env python3
"""Measure rustc peak RSS and defined code symbols for `magemath`.

Catches accidental monomorphization / compile-time / memory blowups early,
even on high-RAM hosts where builds would otherwise succeed silently.

Stages checked (isolated `target/compile-budget`, `CARGO_INCREMENTAL=0`,
`CARGO_BUILD_JOBS=1`):
  1. `cargo build -p magemath --lib --features "std avx512"` (debug)
  2. `cargo test -p magemath --lib --no-run --features "std avx512"` (unit tests)
  3. `cargo build -p magemath --lib --release --features "std avx512"` (release)
"""
import argparse
import collections
import glob
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TARGET_DIR = ROOT / "target" / "compile-budget"
TMP_DIR = Path.home() / "tmp"

# Budgets (MiB peak RSS for a single cargo/rustc invocation, defined code symbols
# in libmagemath-*.rlib). Keep ~2-4x headroom above clean baseline (observed ~421 MiB
# debug-lib / 1834 symbols, ~509 MiB debug-unit-tests, ~434 MiB release-lib / 43
# symbols) while failing well below low-RAM host limits on generic explosions.
DEBUG_LIB_MAX_RSS_MIB = 1536.0
DEBUG_TEST_MAX_RSS_MIB = 2048.0
RELEASE_LIB_MAX_RSS_MIB = 1536.0
DEBUG_LIB_MAX_SYMBOLS = 4000
RELEASE_LIB_MAX_SYMBOLS = 200

FORBIDDEN_STEM_FRAGMENTS = ("Options<", "const_api::Of<")


def find_nm() -> str | None:
    for candidate in ("nm", "llvm-nm"):
        path = shutil.which(candidate)
        if path:
            return path
    sysroot = subprocess.run(
        ["rustc", "--print", "sysroot"], capture_output=True, text=True
    )
    if sysroot.returncode == 0:
        matches = sorted(
            glob.glob(f"{sysroot.stdout.strip()}/lib/rustlib/*/bin/llvm-nm")
        )
        if matches:
            return matches[0]
    return None


def make_env() -> dict[str, str]:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(TARGET_DIR)
    env["CARGO_INCREMENTAL"] = "0"
    env["CARGO_BUILD_JOBS"] = "1"
    env["TMPDIR"] = str(TMP_DIR)
    return env


def run_measured(cmd: list[str], env: dict[str, str]) -> tuple[int, float, float]:
    """Run `cmd` in a fresh child process and return `(rc, elapsed_s, peak_rss_mib)`."""
    if sys.platform == "win32":
        t0 = time.monotonic()
        proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
        elapsed = time.monotonic() - t0
        if proc.returncode != 0:
            if proc.stdout:
                sys.stdout.write(proc.stdout)
            if proc.stderr:
                sys.stderr.write(proc.stderr)
        return proc.returncode, elapsed, 0.0
    script = (
        "import resource, subprocess, sys; "
        "rc = subprocess.call(sys.argv[1:]); "
        "rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss; "
        "mib = rss / (1024.0 * 1024.0) if sys.platform == 'darwin' else rss / 1024.0; "
        "print(f'__COMPILE_BUDGET_RSS_MIB__={mib:.2f}', file=sys.stderr); "
        "sys.exit(rc)"
    )
    t0 = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-c", script, *cmd],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    elapsed = time.monotonic() - t0
    peak_mib = 0.0
    stderr_lines = []
    for line in proc.stderr.splitlines():
        if line.startswith("__COMPILE_BUDGET_RSS_MIB__="):
            peak_mib = float(line.split("=", 1)[1])
        else:
            stderr_lines.append(line)
    if proc.returncode != 0:
        if proc.stdout:
            sys.stdout.write(proc.stdout)
        if stderr_lines:
            sys.stderr.write("\n".join(stderr_lines) + "\n")
    return proc.returncode, elapsed, peak_mib


def stem_of(sym: str) -> str:
    s = re.sub(r"::h[0-9a-f]{16}$", "", sym)
    s = re.sub(r"\.\d+$", "", s)
    s = re.sub(r"<.*>", "<_>", s)
    return s


def inspect_rlib_symbols(
    profile_dir: Path, nm_bin: str | None
) -> tuple[int, list[tuple[str, int]], list[str]]:
    rlibs = sorted(
        list((profile_dir / "deps").glob("libmagemath-*.rlib"))
        + list((profile_dir / "deps").glob("magemath-*.rlib")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not rlibs:
        raise RuntimeError(f"missing libmagemath-*.rlib under {profile_dir / 'deps'}")
    if not nm_bin:
        return 0, [], []
    proc = subprocess.run(
        [nm_bin, "--defined-only", "-C", str(rlibs[0])],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        proc = subprocess.run(
            [nm_bin, "--defined-only", str(rlibs[0])],
            capture_output=True,
            text=True,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"{nm_bin} failed: {proc.stderr}")
    count = 0
    stems: collections.Counter[str] = collections.Counter()
    forbidden: list[str] = []
    for line in proc.stdout.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        kind, sym = parts[1], parts[2]
        if kind not in ("T", "t", "W", "w"):
            continue
        count += 1
        stems[stem_of(sym)] += 1
        for bad in FORBIDDEN_STEM_FRAGMENTS:
            if bad in sym:
                forbidden.append(sym)
    return count, stems.most_common(10), sorted(set(forbidden))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep",
        action="store_true",
        help="keep target/compile-budget after exit for inspection",
    )
    args = parser.parse_args()

    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)

    nm_bin = find_nm()
    env = make_env()
    dep_features = "archmage/std,archmage/avx512,magetypes/std,magetypes/avx512"

    try:
        # Warm dependencies outside the `magemath` measurement so peak RSS reflects `magemath` only.
        for release in (False, True):
            dep_cmd = [
                "cargo",
                "build",
                "-p",
                "archmage",
                "-p",
                "magetypes",
                "--lib",
                "--features",
                dep_features,
            ]
            if release:
                dep_cmd.append("--release")
            rc, elapsed, _ = run_measured(dep_cmd, env)
            if rc != 0:
                print("failed pre-building archmage/magetypes dependencies", file=sys.stderr)
                return rc

        stages = [
            (
                "debug-lib",
                [
                    "cargo",
                    "build",
                    "-p",
                    "magemath",
                    "--lib",
                    "--features",
                    "std avx512",
                ],
                DEBUG_LIB_MAX_RSS_MIB,
                TARGET_DIR / "debug",
                DEBUG_LIB_MAX_SYMBOLS,
            ),
            (
                "debug-unit-tests",
                [
                    "cargo",
                    "test",
                    "-p",
                    "magemath",
                    "--lib",
                    "--no-run",
                    "--features",
                    "std avx512",
                ],
                DEBUG_TEST_MAX_RSS_MIB,
                None,
                None,
            ),
            (
                "release-lib",
                [
                    "cargo",
                    "build",
                    "-p",
                    "magemath",
                    "--lib",
                    "--release",
                    "--features",
                    "std avx512",
                ],
                RELEASE_LIB_MAX_RSS_MIB,
                TARGET_DIR / "release",
                RELEASE_LIB_MAX_SYMBOLS,
            ),
        ]

        failures = []
        print(f"compile budget check (nm={nm_bin or 'unavailable'}):")
        for name, cmd, rss_limit, profile_dir, sym_limit in stages:
            rc, elapsed, peak_mib = run_measured(cmd, env)
            if rc != 0:
                print(f"  FAIL {name}: exited with rc={rc}", file=sys.stderr)
                return rc
            sym_str = "n/a"
            top10: list[tuple[str, int]] = []
            forbidden: list[str] = []
            if profile_dir is not None and sym_limit is not None:
                sym_count, top10, forbidden = inspect_rlib_symbols(profile_dir, nm_bin)
                sym_str = f"{sym_count} / {sym_limit}"
                if sym_count > sym_limit:
                    failures.append(
                        f"{name}: {sym_count} defined text symbols > budget {sym_limit}"
                    )
                if forbidden:
                    failures.append(
                        f"{name}: forbidden generic instantiations present: {forbidden[:5]}"
                    )
            if peak_mib > rss_limit:
                failures.append(
                    f"{name}: peak RSS {peak_mib:.1f} MiB > budget {rss_limit:.1f} MiB"
                )
            print(
                f"  {name:18s} {elapsed:5.2f}s | peak-RSS {peak_mib:6.1f} / {rss_limit:.0f} MiB | symbols {sym_str}"
            )
            if top10 and (failures or os.environ.get("CI_VERBOSE")):
                print("    top stems:")
                for stem, cnt in top10:
                    print(f"      {cnt:4d}  {stem}")

        if failures:
            print("compile budget exceeded:", file=sys.stderr)
            for msg in failures:
                print(f"  - {msg}", file=sys.stderr)
            return 1
        return 0
    finally:
        if not args.keep and TARGET_DIR.exists():
            shutil.rmtree(TARGET_DIR, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

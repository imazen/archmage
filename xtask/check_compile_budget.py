#!/usr/bin/env python3
"""Measure rustc peak RSS and defined code symbols for `magemath` and an
optimized downstream consumer crate.

Catches accidental monomorphization / compile-time / memory blowups early,
even on high-RAM hosts where builds would otherwise succeed silently.

Stages checked (isolated `target/compile-budget`, `CARGO_INCREMENTAL=0`,
`CARGO_BUILD_JOBS=1`):
  1. `cargo build -p magemath --lib --features "std avx512"` (debug)
  2. `cargo test -p magemath --lib --no-run --features "std avx512"` (unit tests)
  3. `cargo build -p magemath --lib --release --features "std avx512"` (release)
  4. `cargo build --lib --release` on a synthetic downstream consumer crate
     instantiating representative `const CFG: u128` profiles, `SimdMath` inline
     vector chains (`f32x4`, `f32x8`, `f32x16`), slices, and `.runtime()` calls.
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

DEBUG_LIB_MAX_RSS_MIB = 1536.0
DEBUG_TEST_MAX_RSS_MIB = 2048.0
RELEASE_LIB_MAX_RSS_MIB = 1536.0
RELEASE_CONSUMER_MAX_RSS_MIB = 1536.0
DEBUG_LIB_MAX_SYMBOLS = 4000
RELEASE_LIB_MAX_SYMBOLS = 200
RELEASE_CONSUMER_MAX_SYMBOLS = 500

FORBIDDEN_RLIB_FRAGMENTS = ("Options<", "const_api::Of<", "const_api::Cfg<")

CONSUMER_CARGO_TOML = """\
[package]
name = "magemath_budget_consumer"
version = "0.1.0"
edition = "2024"
publish = false

[workspace]

[dependencies]
archmage = {{ path = "{root}", features = ["std", "avx512"] }}
magetypes = {{ path = "{root}/magetypes", features = ["std", "avx512"] }}
magemath = {{ path = "{root}/magemath", features = ["std", "avx512"] }}
"""

CONSUMER_LIB_RS = """\
use archmage::prelude::*;
use magemath::{
    SimdMath,
    cbrtf::{Cbrtf, CbrtfOptions},
    expf::{Expf, ExpfOptions},
    logf::{Logf, LogfOptions},
    powf::{Powf, PowfError, PowfOptions},
};

const P_GAMMA: u128 = PowfOptions::new().gamma_domain().midp().bits();
const P_SRGB: u128 = PowfOptions::new().srgb_encode_domain().clamp_out_of_bounds().bits();
const P_LOWP: u128 = PowfOptions::new().unit_gamma_domain().lowp().assume_in_bounds().caller().bits();
const P_HIGHP: u128 = PowfOptions::new().admit_all_floats().highp().bits();
const E_MIDP: u128 = ExpfOptions::new().unclamped_exp_range().bits();
const E_SMALL: u128 = ExpfOptions::new().small_exp2_range().assume_in_bounds().caller().bits();
const E_CLAMP: u128 = ExpfOptions::new().input_range(-4.0, 4.0).clamp_out_of_bounds().bits();
const L_MIDP: u128 = LogfOptions::new().admit_zero().bits();
const L_SUB: u128 = LogfOptions::new().admit_subnormals().bits();
const L_NEAR: u128 = LogfOptions::new().near_one_range().assume_in_bounds().caller().bits();
const L_1P: u128 = LogfOptions::new().small_log1p_range().clamp_out_of_bounds().bits();
const C_UNIT: u128 = CbrtfOptions::new().unit_range().clamp_out_of_bounds().caller().bits();
const C_FULL: u128 = CbrtfOptions::new().admit_all_floats().lowp().bits();

#[attune(_*_t, +v4x, define(f32x4, f32x8, f32x16))]
fn color_chain_inner(token: Token, src: &[f32], dst: &mut [f32], gamma: f32) -> Result<(), PowfError> {
    assert_eq!(src.len(), dst.len());
    let (sc, st) = f32x8::partition_slice_t(token, src);
    let (dc, dt) = f32x8::partition_slice_mut_t(token, dst);
    for (s, d) in sc.iter().zip(dc.iter_mut()) {
        let v = f32x8::load_t(token, s);
        let y = v
            .powf::<P_LOWP>(gamma)
            .exp2::<E_SMALL>()
            .ln::<L_NEAR>()
            .cbrt::<C_UNIT>();
        y.store(d);
    }
    if st.len() >= 4 {
        let mut head = [0.0; 4];
        head.copy_from_slice(&st[..4]);
        let v4 = f32x4::from_array_t(token, head)
            .powf::<P_SRGB>(gamma)
            .exp::<E_CLAMP>()
            .log1p::<L_1P>();
        dt[..4].copy_from_slice(&v4.to_array());
        for (&x, y) in st[4..].iter().zip(dt[4..].iter_mut()) {
            *y = Powf::<P_GAMMA>::scalar_with_token(token, x, gamma)?;
        }
    } else {
        for (&x, y) in st.iter().zip(dt.iter_mut()) {
            *y = Powf::<P_GAMMA>::scalar_with_token(token, x, gamma)?;
        }
    }
    Ok(())
}

pub fn run_consumer_pipeline(src: &[f32], dst: &mut [f32], gamma: f32) -> f32 {
    let _ = attuned!(color_chain_inner(src, dst, gamma), [_v4x, _v3, _neon, _wasm128, _scalar]);
    let _ = Powf::<P_GAMMA>::slice(src, dst, gamma);
    let _ = Powf::<P_HIGHP>::slice(src, dst, gamma);
    let _ = Expf::<E_MIDP>::exp_slice(src, dst);
    let _ = Logf::<L_MIDP>::ln_slice(src, dst);
    let _ = Logf::<L_SUB>::log2_slice(src, dst);
    let _ = Cbrtf::<C_FULL>::slice(src, dst);
    if src.len() >= 16 {
        let mut a16 = [0.0; 16];
        a16.copy_from_slice(&src[..16]);
        let v16 = magetypes::simd::generic::f32x16::from_array_t(archmage::ScalarToken, a16);
        let out16 = v16.powf::<P_SRGB>(gamma).cbrt::<C_UNIT>().to_array();
        dst[..16].copy_from_slice(&out16);
    }
    PowfOptions::new()
        .gamma_domain()
        .clamp_out_of_bounds()
        .runtime()
        .scalar(dst.first().copied().unwrap_or(1.0), gamma)
        .unwrap_or(0.0)
}
"""


def find_nm() -> str | None:
    for candidate in ("llvm-nm", "nm"):
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


def run_measured(
    cmd: list[str], env: dict[str, str], cwd: Path = ROOT
) -> tuple[int, float, float]:
    """Run `cmd` in a fresh child process and return `(rc, elapsed_s, peak_rss_mib)`."""
    if sys.platform == "win32":
        t0 = time.monotonic()
        proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
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
        cwd=cwd,
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
    profile_dir: Path,
    crate_stem: str,
    nm_bin: str,
    forbidden_fragments: tuple[str, ...],
    required_fragments: tuple[str, ...] = (),
) -> tuple[int, list[tuple[str, int]], list[str]]:
    candidates = [
        p
        for p in (
            profile_dir / f"lib{crate_stem}.rlib",
            profile_dir / f"{crate_stem}.rlib",
            *(profile_dir / "deps").glob(f"lib{crate_stem}-*.rlib"),
            *(profile_dir / "deps").glob(f"{crate_stem}-*.rlib"),
        )
        if p.exists()
    ]
    rlibs = sorted(
        candidates,
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not rlibs:
        raise RuntimeError(f"missing lib{crate_stem}.rlib under {profile_dir}")
    proc = subprocess.run(
        [nm_bin, "--defined-only", "-C", str(rlibs[0])],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{nm_bin} --defined-only -C failed (demangling is required): {proc.stderr}"
        )
    count = 0
    demangled_seen = 0
    raw_v0_mangled = 0
    stems: collections.Counter[str] = collections.Counter()
    forbidden: list[str] = []
    seen_syms: list[str] = []
    for line in proc.stdout.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        kind, sym = parts[1], parts[2]
        if kind not in ("T", "t", "W", "w"):
            continue
        count += 1
        seen_syms.append(sym)
        if "::" in sym:
            demangled_seen += 1
        if sym.startswith("_RNv") or sym.startswith("__RNv"):
            raw_v0_mangled += 1
        stems[stem_of(sym)] += 1
        for bad in forbidden_fragments:
            if bad in sym:
                forbidden.append(sym)
    if count == 0:
        raise RuntimeError(
            f"{nm_bin} -C recognized 0 text/weak symbols in {rlibs[0]}"
        )
    if demangled_seen == 0 or raw_v0_mangled > 0:
        raise RuntimeError(
            f"{nm_bin} -C did not demangle Rust symbols in {rlibs[0]} "
            f"(count={count}, demangled={demangled_seen}, raw_v0={raw_v0_mangled})"
        )
    for req in required_fragments:
        if not any(req in s for s in seen_syms):
            raise RuntimeError(
                f"expected demangled symbol containing {req!r} in {rlibs[0]}"
            )
    return count, stems.most_common(10), sorted(set(forbidden))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep",
        action="store_true",
        help="keep target/compile-budget after exit for inspection",
    )
    args = parser.parse_args()

    nm_bin = find_nm()
    if not nm_bin:
        print(
            "FAIL: neither `llvm-nm` nor `nm` found; symbol budget check requires demangling nm",
            file=sys.stderr,
        )
        return 1

    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)

    env = make_env()
    consumer_dir = TMP_DIR / f"magemath-budget-consumer-{os.getpid()}"
    dep_features = "archmage/std,archmage/avx512,magetypes/std,magetypes/avx512"

    try:
        consumer_src = consumer_dir / "src"
        consumer_src.mkdir(parents=True, exist_ok=True)
        (consumer_dir / "Cargo.toml").write_text(
            CONSUMER_CARGO_TOML.format(root=ROOT.as_posix())
        )
        shutil.copy2(ROOT / "Cargo.lock", consumer_dir / "Cargo.lock")
        (consumer_src / "lib.rs").write_text(CONSUMER_LIB_RS)

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
            rc, _, _ = run_measured(dep_cmd, env)
            if rc != 0:
                print(
                    "failed pre-building archmage/magetypes dependencies",
                    file=sys.stderr,
                )
                return rc

        stages = [
            (
                "debug-lib",
                ROOT,
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
                "magemath",
                DEBUG_LIB_MAX_SYMBOLS,
                FORBIDDEN_RLIB_FRAGMENTS,
            ),
            (
                "debug-unit-tests",
                ROOT,
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
                "",
                None,
                (),
            ),
            (
                "release-lib",
                ROOT,
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
                "magemath",
                RELEASE_LIB_MAX_SYMBOLS,
                FORBIDDEN_RLIB_FRAGMENTS,
            ),
            (
                "release-consumer",
                consumer_dir,
                [
                    "cargo",
                    "build",
                    "--lib",
                    "--release",
                ],
                RELEASE_CONSUMER_MAX_RSS_MIB,
                TARGET_DIR / "release",
                "magemath_budget_consumer",
                RELEASE_CONSUMER_MAX_SYMBOLS,
                ("Options<",),
            ),
        ]

        failures = []
        print(f"compile budget check (nm={nm_bin}):")
        for (
            name,
            cwd,
            cmd,
            rss_limit,
            profile_dir,
            crate_stem,
            sym_limit,
            forbidden_frags,
        ) in stages:
            rc, elapsed, peak_mib = run_measured(cmd, env, cwd=cwd)
            if rc != 0:
                print(f"  FAIL {name}: exited with rc={rc}", file=sys.stderr)
                return rc
            sym_str = "n/a"
            top10: list[tuple[str, int]] = []
            forbidden: list[str] = []
            if profile_dir is not None and sym_limit is not None:
                req_frags = (
                    ("run_consumer_pipeline",) if name == "release-consumer" else ()
                )
                sym_count, top10, forbidden = inspect_rlib_symbols(
                    profile_dir, crate_stem, nm_bin, forbidden_frags, req_frags
                )
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
        shutil.rmtree(consumer_dir, ignore_errors=True)
        if not args.keep and TARGET_DIR.exists():
            shutil.rmtree(TARGET_DIR, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

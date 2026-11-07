#!/usr/bin/env python3
"""Retain precise implementation/build source, compiler and actual binary hashes."""

from pathlib import Path
import shutil
import hashlib, json, os, subprocess, sys, tarfile

root = Path.cwd()
out = root / sys.argv[1]
out.mkdir(parents=True, exist_ok=False)
files = []
for folder in [
    "src",
    "xtask/src",
    "tests/no-features-crate/src",
    "tests/clippy-consumer/src",
    "archmage-macros/src",
    "magetypes/src",
    "zenmath/src",
    "zenmath/tools",
    "zenmath/docs",
    "magemath/tools",
    "magemath/docs",
    "benchmarks/zenmath_pow_opt_2026-11-06/gamma-validation/src",
    "benchmarks/zenmath_pow_opt_2026-11-06/feature-consumer/src",
    "magemath/src",
    "zenmath/tests",
    "magemath/tests",
    "benchmarks/zenmath_pow_opt_2026-11-06/compile-consumer/src",
    "benchmarks/zenmath_pow_opt_2026-11-06/analysis/src",
    "benchmarks/zenmath_pow_opt_2026-11-06/src",
    "benchmarks/zenmath_pow_opt_2026-11-06/baseline/zenmath/src",
    "benchmarks/zenmath_pow_opt_2026-11-06/baseline/zenmath-before-gamma/src",
    "benchmarks/zenmath_pow_opt_2026-11-06/baseline/magemath/src",
]:
    files.extend(p for p in (root / folder).rglob("*") if p.is_file())
for folder in [
    ".",
    "archmage-macros",
    "xtask",
    "tests/no-features-crate",
    "tests/clippy-consumer",
    "magetypes",
    "zenmath",
    "magemath",
    "benchmarks/zenmath_pow_opt_2026-11-06",
    "benchmarks/zenmath_pow_opt_2026-11-06/compile-consumer",
    "benchmarks/zenmath_pow_opt_2026-11-06/gamma-validation",
    "benchmarks/zenmath_pow_opt_2026-11-06/feature-consumer",
    "benchmarks/zenmath_pow_opt_2026-11-06/analysis",
    "benchmarks/zenmath_pow_opt_2026-11-06/baseline/zenmath",
    "benchmarks/zenmath_pow_opt_2026-11-06/baseline/zenmath-before-gamma",
    "benchmarks/zenmath_pow_opt_2026-11-06/baseline/magemath",
]:
    for n in ["Cargo.toml", "Cargo.lock"]:
        p = root / folder / n
        if p.exists():
            files.append(p)
study = root / "benchmarks/zenmath_pow_opt_2026-11-06"
for pattern in ["*.py", "*.sh", "justfile"]:
    files.extend(study.glob(pattern))
for relative in ["token-registry.toml", "magemath/README.md", "magemath/docs/POW_BEHAVIOR.md", "benchmarks/zenmath_pow_opt_2026-11-06/baseline/magemath/README.md", "benchmarks/zenmath_pow_opt_2026-11-06/baseline/magemath/docs/POW_BEHAVIOR.md"]:
    files.append(root / relative)
# Validation metadata supplements the exact build closure.
for relative in ["baseline/PROVENANCE.json","baseline/zenmath-before-gamma/PROVENANCE.json","baseline/zenmath-before-gamma/PROVENANCE.md","linear-srgb-provenance.json"]:
    p=study/relative
    if p.exists():files.append(p)
files = sorted(set(files))
manifest = "".join(
    hashlib.sha256(p.read_bytes()).hexdigest() + "  " + str(p.relative_to(root)) + "\n"
    for p in files
)
(out / "source.sha256").write_text(manifest)
with tarfile.open(out / "source.tar.gz", "w:gz") as tar:
    for p in files:
        tar.add(p, arcname=str(p.relative_to(root)), recursive=False)
meta = dict(
    source_revision=os.environ["POW_SOURCE_REV"],
    source_manifest_sha256=hashlib.sha256(manifest.encode()).hexdigest(),
    rustc=subprocess.check_output(["rustc", "-vV"], text=True),
    environment={
        k: os.environ.get(k, "")
        for k in [
            "RUSTFLAGS",
            "CARGO_ENCODED_RUSTFLAGS",
            "RUSTUP_TOOLCHAIN",
            "CARGO_PROFILE_RELEASE_LTO",
        ]
    },
    binaries=[
        dict(
            path=p,
            sha256=hashlib.sha256((root / p).read_bytes()).hexdigest(),
            bytes=(root / p).stat().st_size,
        )
        for p in sys.argv[2:]
    ],
)
(out / "binaries").mkdir(exist_ok=True)
for binary in meta["binaries"]:
    dest = out / "binaries" / (binary["sha256"][:16] + "-" + Path(binary["path"]).name)
    temp = dest.with_suffix(".copying")
    shutil.copy2(root / binary["path"], temp)
    assert hashlib.sha256(temp.read_bytes()).hexdigest() == binary["sha256"]
    temp.replace(dest)
    binary["archived_as"] = str(dest.relative_to(out))
(out / "identity.json").write_text(json.dumps(meta, indent=2) + "\n")
print(json.dumps(meta))

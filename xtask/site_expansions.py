#!/usr/bin/env python3
"""Regenerate/check the website's actual cargo-expand examples.

Pinned tools keep the displayed output reproducible. Each complete fixture is
type-checked before expansion: cargo-expand alone is not a compile-test oracle.
Run under the same resource limiter used for other builds.
"""
import argparse
import os
from pathlib import Path
import re
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/site-expansion"
OUTPUT = ROOT / "docs/site/static/expansions"
TOOLCHAIN = "1.99.0"
EXPAND = "cargo-expand 1.0.126"
PRESETS = {
    "x86_64": ("x86_64-unknown-linux-gnu", ""),
    "x86_64-avx512": ("x86_64-unknown-linux-gnu", "avx512"),
    "aarch64": ("aarch64-unknown-linux-gnu", ""),
    "wasm32": ("wasm32-unknown-unknown", ""),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    env = os.environ.copy()
    env["RUSTUP_TOOLCHAIN"] = TOOLCHAIN
    env["CARGO_TARGET_DIR"] = str(ROOT / "target/site-expansion")
    env.setdefault("TMPDIR", str(Path.home() / "tmp"))
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    env.pop("RUSTFLAGS", None)
    env.pop("CARGO_ENCODED_RUSTFLAGS", None)
    version = subprocess.check_output(["cargo", "expand", "--version"], env=env, text=True).strip()
    if version != EXPAND:
        raise SystemExit(f"Expected {EXPAND}, found {version}")
    examples = tomllib.loads((ROOT / "docs/site/expansions.toml").read_text())["examples"]
    source = (FIXTURE / "src/lib.rs").read_text()
    parts = re.split(r"^// example: ([a-z_]+)\n", source, flags=re.M)
    sources = dict(zip(parts[1::2], parts[2::2]))
    ids = [example["id"] for example in examples]
    assert len(ids) == len(set(ids)) and set(ids) == set(sources), "fixture/index mismatch"
    expected = {}
    expected["source.rs"] = source
    for preset, (target, features) in PRESETS.items():
        common = ["--manifest-path", str(FIXTURE / "Cargo.toml"), "--lib", "--target", target]
        if features:
            common += ["--features", features]
        subprocess.run(["cargo", "check", *common], cwd=ROOT, env=env, check=True)
        modules = []
        for name in ids:
            print(f"Expand {preset}: {name}", flush=True)
            expanded = subprocess.check_output(
                ["cargo", "expand", "--color", "never", *common, name],
                cwd=ROOT, env=env, text=True,
            )
            assert f"pub mod {name}" in expanded, f"missing module: {preset}/{name}"
            modules.append(f"// example: {name}\n" + expanded.strip() + "\n")
        expected[f"{preset}.rs"] = (
            f"// Rust {TOOLCHAIN}; {EXPAND}; {target}; features: {features or 'default'}\n"
            + "\n".join(modules)
        )
    changed = []
    for relative, contents in expected.items():
        destination = OUTPUT / relative
        if not destination.exists() or destination.read_text() != contents:
            changed.append(relative)
            if not args.check:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(contents)
    stale = sorted(str(p.relative_to(OUTPUT)) for p in OUTPUT.rglob("*.rs")
                   if str(p.relative_to(OUTPUT)) not in expected)
    if stale:
        raise SystemExit(f"Unexpected expansion files; review them explicitly: {stale}")
    if args.check and changed:
        raise SystemExit(f"Website expansions are stale: {changed}")
    print(f"Verified {len(ids)} examples across {len(PRESETS)} target/feature configurations.")


if __name__ == "__main__":
    main()

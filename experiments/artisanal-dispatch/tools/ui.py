#!/usr/bin/env python3
"""Compile each UI fixture against the exact fresh Cargo artifact; no snapshots."""
import json
import os
import platform
from pathlib import Path
import subprocess

assert platform.machine() == "x86_64", "UI safety fixtures require an x86_64 host"
root = Path(__file__).resolve().parents[1]
build = subprocess.run(["cargo", "build", "--features", "std", "--message-format=json"],
                       cwd=root, text=True, capture_output=True, check=True)
artifacts = [m for line in build.stdout.splitlines()
             for m in [json.loads(line)] if m.get("reason") == "compiler-artifact"]
rlibs = [Path(f) for m in artifacts if m["target"]["name"] == "artisanal_dispatch"
         for f in m["filenames"] if f.endswith(".rlib")]
assert len(rlibs) == 1, rlibs
dep_dirs = sorted({str(Path(f).parent) for m in artifacts for f in m["filenames"]}
                  | ({str(rlibs[0].parent / "deps")} if (rlibs[0].parent / "deps").is_dir() else set()))
out = root / "target" / "ui"
out.mkdir(parents=True, exist_ok=True)
for fixture in sorted((root / "tests" / "ui").glob("*.rs")):
    first = fixture.read_text().splitlines()[0]
    assert first.startswith("// error: "), fixture
    expected = first.removeprefix("// error: ")
    result = subprocess.run([os.environ.get("RUSTC", "rustc"), "--edition=2024", "--crate-type=lib",
                             "--emit=metadata", "--extern", f"artisanal_dispatch={rlibs[0]}",
                             *[f"-Ldependency={d}" for d in dep_dirs], "--out-dir", str(out), str(fixture)],
                            text=True, capture_output=True)
    (out / (fixture.stem + ".stderr")).write_text(result.stderr)
    assert result.returncode != 0 and expected in result.stderr, (fixture, result.stderr)
    print(f"PASS {fixture.name}: {expected}")

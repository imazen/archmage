#!/usr/bin/env python3
"""Reconstruct frozen first timing bytes, requiring original SHA-256 equality."""

from pathlib import Path
import hashlib, json, shutil, subprocess, tarfile

root = Path.cwd()
snapshot = root / "frozen-native"
source = root / "benchmarks/zenmath_dispatch_2026-11-06/src/bin/timing.rs"
identity = json.loads((snapshot / "identity.json").read_text())
expected = identity["binaries"][0]["sha256"]
backup = snapshot / "driver-before-reconstruction.rs"
current = source.read_bytes()
backup.write_bytes(current)
try:
    with tarfile.open(snapshot / "source.tar.gz") as tar:
        original = tar.extractfile(
            "benchmarks/zenmath_dispatch_2026-11-06/src/bin/timing.rs"
        ).read()
    source.write_bytes(original)
    subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "--features",
            "avx512,native-bench",
            "--manifest-path",
            "benchmarks/zenmath_dispatch_2026-11-06/Cargo.toml",
            "--bin",
            "timing",
        ],
        check=True,
    )
    binary = root / "benchmarks/zenmath_dispatch_2026-11-06/target/release/timing"
    actual = hashlib.sha256(binary.read_bytes()).hexdigest()
    assert actual == expected, (actual, expected)
    dest = snapshot / "binaries" / ("timing-" + actual)
    dest.parent.mkdir(exist_ok=True)
    shutil.copy2(binary, dest)
    (snapshot / "binary-reconstruction.json").write_text(
        json.dumps(
            dict(
                original_sha256=expected,
                reconstructed_sha256=actual,
                exact_match=True,
                original_source_sha256=hashlib.sha256(original).hexdigest(),
                note="Original Cargo output was replaced by driver rebuild before archive copy. Reconstructed from frozen source with same compiler/config/path; exact original SHA-256 required.",
            ),
            indent=2,
        )
        + "\n"
    )
    print("PASS exact original timed executable SHA-256", actual)
finally:
    source.write_bytes(current)
    assert source.read_bytes() == current

#!/usr/bin/env python3
"""Fail closed on missing platforms, coverage, accepted gates or frozen bytes."""

from pathlib import Path
import hashlib
import json
import sys
import tarfile

base = Path(sys.argv[1]).resolve()
expected = [
    ("native/timing-native.json", 94, 1094, False, False),
    ("native/timing-native-rerun-v2.json", 10, 119, True, False),
    ("arm/timing-arm.json", 94, 678, True, False),
    ("wasm/timing-wasm49-final.json", 94, 678, True, False),
    ("native/timing-facade-native.json", 12, 147, False, True),
    ("native/timing-facade-repeat.json", 2, 24, True, True),
    ("wasm/timing-wasm-policy-v2.json", 3, 24, True, True),
]
for path, groups, cells, clean, ordinary in expected:
    suite = json.loads((base / path).read_text())
    assert not suite["unreliable"] and not suite["incomplete"], path
    assert len(suite["comparisons"]) == groups, path
    assert sum(len(g["benchmarks"]) for g in suite["comparisons"]) == cells, path
    for group in suite["comparisons"]:
        assert group["completed_rounds"] >= 30
        if clean:
            assert all(r["gate_clean"] is True for r in group["samples"]), (
                path,
                group["group_name"],
            )
        if ordinary:
            assert any(b["name"] == "ordinary/baseline" for b in group["benchmarks"]), (
                path
            )
    print(f"PASS {path}: {groups} groups/{cells} cells")
for name in [
    "native/frozen-source",
    "native/frozen-rerun-v2",
    "native/frozen-facade",
    "arm/frozen-source",
    "wasm/frozen-source",
    "wasm/frozen-policy-v2",
    "wasm/frozen-baseline-verified",
]:
    folder = base / name
    identity = json.loads((folder / "identity.json").read_text())
    manifest = (folder / "source.sha256").read_bytes()
    assert hashlib.sha256(manifest).hexdigest() == identity["source_manifest_sha256"], (
        name
    )
    with tarfile.open(folder / "source.tar.gz", "r:gz") as source:
        for line in manifest.decode().splitlines():
            sha, path = line.split("  ", 1)
            assert hashlib.sha256(source.extractfile(path).read()).hexdigest() == sha, (
                name,
                path,
            )
    for binary in identity["binaries"]:
        matches = [
            p
            for p in (folder / "binaries").iterdir()
            if p.stat().st_size == binary["bytes"]
            and hashlib.sha256(p.read_bytes()).hexdigest() == binary["sha256"]
        ]
        if (
            not matches
            and name == "native/frozen-source"
            and binary["path"].endswith("/proof")
        ):
            status = json.loads((base / "retention-status.json").read_text())
            assert status["missing_untimed_helpers"][0]["snapshot"] == name
            assert status["missing_untimed_helpers"][0]["sha256"] == binary["sha256"]
            print(
                "MISSING original untimed proof helper; original measured timing executable retained exactly"
            )
            continue
        assert matches, (name, binary["path"])
    print(
        f"PASS {name}: exact frozen source and retained executable bytes (untimed retention exception reported above)"
    )
if len(sys.argv) > 2:
    mirror = Path(sys.argv[2]).resolve()
    count = 0
    for line in (base / "SHA256SUMS").read_text().splitlines():
        sha, path = line.split("  ", 1)
        for root in [base, mirror]:
            assert hashlib.sha256((root / path).read_bytes()).hexdigest() == sha, (
                str(root),
                path,
            )
        count += 1
    print(f"PASS local/NAS manifest: {count} files byte-verified")

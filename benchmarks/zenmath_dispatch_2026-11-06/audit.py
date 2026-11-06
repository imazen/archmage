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
suites = {}
for path, groups, cells, clean, ordinary in expected:
    suite = json.loads((base / path).read_text())
    suites[path] = suite
    group_names = [g["group_name"] for g in suite["comparisons"]]
    assert len(group_names) == len(set(group_names)), ("duplicate groups", path)
    assert not suite["unreliable"] and not suite["incomplete"], path
    assert len(suite["comparisons"]) == groups, path
    assert sum(len(g["benchmarks"]) for g in suite["comparisons"]) == cells, path
    for group in suite["comparisons"]:
        case_names = [b["name"] for b in group["benchmarks"]]
        assert len(case_names) == len(set(case_names)), (
            "duplicate cases",
            path,
            group["group_name"],
        )
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
ops32 = [
    "sqrt",
    "floor",
    "ceil",
    "trunc",
    "roundeven",
    "round",
    "fma",
    "exp",
    "ln",
    "log2",
    "pow",
    "pow_rescue",
]
ops64 = ops32[:7]
modes = ["stream1", "stream16", "stream1024", "stream65536", "feedback"]
full_names = {
    f"{op}/{ty}/{mode}"
    for ty, ops in [("f32", ops32), ("f64", ops64)]
    for op in ops
    for mode in modes
    if not (op == "pow_rescue" and mode == "feedback")
}
assert len(full_names) == 94
for path in [
    "native/timing-native.json",
    "arm/timing-arm.json",
    "wasm/timing-wasm49-final.json",
]:
    assert {g["group_name"] for g in suites[path]["comparisons"]} == full_names, path


def verify_replacements(original_path, repeat_path, expected_names):
    original = {g["group_name"]: g for g in suites[original_path]["comparisons"]}
    repeat = {g["group_name"]: g for g in suites[repeat_path]["comparisons"]}
    dirty = {
        name
        for name, g in original.items()
        if any(r["gate_clean"] is not True for r in g["samples"])
    }
    assert set(repeat) == dirty, (
        "replacement coverage differs from rejected groups",
        original_path,
        dirty,
        set(repeat),
    )
    effective = original | repeat
    assert set(effective) == expected_names, (
        "effective coverage differs",
        original_path,
    )
    assert all(
        r["gate_clean"] is True for g in effective.values() for r in g["samples"]
    ), ("effective dirty group", original_path)
    print(
        f"PASS effective {original_path}: {len(effective)} unique expected groups, all clean; exact rejected-group replacement"
    )


verify_replacements(
    "native/timing-native.json", "native/timing-native-rerun-v2.json", full_names
)
facade_names = {
    "sqrt/f32/stream1",
    "sqrt/f32/stream65536",
    "sqrt/f32/feedback",
    "floor/f32/stream65536",
    "round/f32/stream65536",
    "pow/f32/stream65536",
    "pow/f32/feedback",
    "sqrt/f64/stream65536",
    "fma/f64/stream1",
    "fma/f64/stream65536",
    "fma/f64/feedback",
    "round/f64/stream65536",
}
verify_replacements(
    "native/timing-facade-native.json", "native/timing-facade-repeat.json", facade_names
)
assert {
    g["group_name"] for g in suites["wasm/timing-wasm-policy-v2.json"]["comparisons"]
} == {"sqrt/f32/stream65536", "floor/f32/stream65536", "round/f64/feedback"}
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

#!/usr/bin/env python3
"""Render measured large-buffer controls, preserving replacement provenance."""

from pathlib import Path
import csv
import sys

base = Path(sys.argv[1])
out = Path(sys.argv[2])
sections = [
    "# Measured dispatch controls\n",
    """All values are ns/value for 65,536-value streams, measured without native CPU flags.
Reference/branch/cached means below use identical raw rounds. Differences and 95%
bootstrap intervals also use all identical rounds (10,000 resamples), with no new
exclusions. These differ from Zenbench's retained filtered-pair intervals, preserved
in the artifact CSVs. Intervals overlapping zero are unresolved, not wins.
Native replaces only the ten gate-dirty original groups with the clean ten-group
repeat. ARM has clean gates throughout; 83 original filtered comparisons retain
fewer than 30 pairs; WASM has 58 such comparisons. Both are reported separately
from these 30–60-round direct intervals.
WASM uses Wasmtime 49.0.5 on Neoverse N1, and includes per-host-call allocation
amortized over explicit repeats. Native allocates before timed iterations.
""",
]
for label, folder, extra in [
    ("Ryzen 7900X", "native", "analysis-initial"),
    ("Neoverse N1", "arm", "analysis"),
    ("WASM SIMD128 on N1", "wasm", "analysis"),
]:
    root = base / folder
    assert (root / extra / "direct-pairs.csv").is_file(), (
        f"missing required platform: {label}"
    )
    rows = list(csv.DictReader((root / extra / "direct-pairs.csv").open()))
    if folder == "native":
        replacement = list(
            csv.DictReader((root / "analysis-rerun-v2/direct-pairs.csv").open())
        )
        groups = {r["group"] for r in replacement}
        rows = [r for r in rows if r["group"] not in groups] + replacement
    sections += [
        f"\n## {label}\n",
        "| Operation/type | Reference | Branch | Cached | Branch − reference [95% CI] | Cached − branch [95% CI] |\n|---|---:|---:|---:|---:|---:|\n",
    ]
    bygroup = {}
    for r in rows:
        if r["group"].endswith("stream65536"):
            bygroup.setdefault(r["group"], {})[(r["baseline"], r["candidate"])] = r
    for group, pairs in bygroup.items():
        r = pairs[("reference/baseline", "branch/baseline")]
        c = pairs[("branch/baseline", "cached/baseline")]
        assert r["policy_eligible"] == "true" and c["policy_eligible"] == "true", group
        ci = (
            lambda x: f"{float(x['diff_mean']):.3f} [{float(x['ci_lo']):.3f}, {float(x['ci_hi']):.3f}]"
        )
        sections.append(
            f"| {group.removesuffix('/stream65536')} | {float(r['baseline_mean']):.3f} | {float(r['candidate_mean']):.3f} | {float(c['candidate_mean']):.3f} | {ci(r)} | {ci(c)} |\n"
        )
out.write_text("\n".join(sections))

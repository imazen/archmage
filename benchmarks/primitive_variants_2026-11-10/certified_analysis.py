#!/usr/bin/env python3
"""Retained-round diagnostics for the bounded/canonical reciprocal studies.

Requires NumPy. Never clears admission failures, filters rounds, or substitutes
within-process bootstrap intervals for independent-process confirmation.
"""
import gzip
import json
import math
from pathlib import Path
import sys

import numpy as np


def t4_critical(probability):
    # Exact df=4 CDF, using s=t/sqrt(t*t+4).
    low, high = 0.0, 1.0
    for _ in range(80):
        s = (low + high) / 2
        if 0.5 + 0.75 * s - 0.25 * s**3 < probability:
            low = s
        else:
            high = s
    s = (low + high) / 2
    return 2 * s / math.sqrt(1 - s * s)


assert abs(t4_critical(0.975) - 2.7764451051977987) < 1e-10
root = Path(sys.argv[1]).resolve()
assert sys.argv[2] in ("first", "cold")
cold = sys.argv[2] == "cold"
contexts = ["zen5/scalar", "zen5/v3", "zen5/v4", "i265/v3", "mac/neon",
            "wasm/strict", "wasm/fusing", "wasm/nonfusing"]
rows = []
for context in contexts:
    packet = json.loads(gzip.decompress((root / (context + ".json.gz")).read_bytes()))
    replicates = packet.get("replicates") or [packet]
    assert len(replicates) == (1 if context == "mac/neon" else 5)
    assert all(len(r["comparisons"]) == 13 for r in replicates)
    # Include the explicitly reported non-baseline comparisons too: widened /
    # serial, or the two cold / inline pairs. Baseline-only multiplicity alone
    # would undercount the family of intervals printed by this analysis.
    family = sum(len(g["benchmarks"]) - 1 + (2 if cold else 1)
                 for g in replicates[0]["comparisons"])
    assert family == 91
    for gi, group in enumerate(replicates[0]["comparisons"]):
        groups = [r["comparisons"][gi] for r in replicates]
        names = [b["name"] for b in group["benchmarks"]]
        assert len(names) == (6 if cold else 7)
        assert all(g["group_name"] == group["group_name"] for g in groups)
        assert all([b["name"] for b in g["benchmarks"]] == names for g in groups)
        assert all(len(g["samples"]) == 50 for g in groups)
        assert all(not g["config"]["auto_rounds"] and g["config"]["max_rounds"] == 50
                   and g["completed_rounds"] == 50 for g in groups)
        assert all(s["iterations"] > 0 and len(s["compensated_ns"]) == len(names)
                   and len(s["elapsed_ns"]) == len(names)
                   and sorted(s["execution_order"]) == list(range(len(names)))
                   for g in groups for s in g["samples"])
        batch = (4096 if group["group_name"].endswith("64")
                 and "/chain" not in group["group_name"] else 256) if context.startswith("wasm/") else 1
        values = np.array([[[s["compensated_ns"][i] / s["iterations"] / batch
                             for i in range(len(names))] for s in g["samples"]] for g in groups])
        assert np.all(np.isfinite(values)) and np.all(values > 0)
        means = values.mean(axis=(0, 1))
        # Equal process weight, paired round resampling within each process.
        rng = np.random.default_rng(0x531324)
        resamples = np.zeros((10000, len(names)))
        for process in values:
            index = rng.integers(0, 50, size=(10000, 50))
            resamples += process[index].mean(axis=1) / len(replicates)
        pairs = [(i, 0) for i in range(1, len(names))]
        pairs += [(4, 2), (5, 3)] if cold else [(2, 1)]
        intervals = []
        for numerator, denominator in pairs:
            boot = resamples[:, numerator] / resamples[:, denominator]
            process_ci = None
            if len(replicates) == 5:
                process_means = values.mean(axis=1)
                logs = np.log(process_means[:, numerator] / process_means[:, denominator])
                floor = max(2 * r["timer_resolution_ns"] / batch /
                            min(s["iterations"] for s in g["samples"]) /
                            process_means[j, numerator]
                            for j, (r, g) in enumerate(zip(replicates, groups)))
                half = max(t4_critical(1 - 0.05 / (2 * family)) * logs.std(ddof=1) / math.sqrt(5), floor)
                process_ci = [math.exp(logs.mean() - half), math.exp(logs.mean() + half)]
            intervals.append(dict(numerator=names[numerator], denominator=names[denominator],
                                  ratio=means[numerator] / means[denominator],
                                  paired_round95=np.quantile(boot, [0.025, 0.975]).tolist(),
                                  process_family95=process_ci))
        clean = sum(s.get("gate_clean") is True for g in groups for s in g["samples"])
        retained = 50 * len(replicates)
        suite_valid = not packet["unreliable"] and not packet.get("incomplete", False)
        suite_valid &= all(not r["unreliable"] and not r.get("incomplete", False) for r in replicates)
        # A true clean flag must also have a documented fresh observation.
        observations = all(s.get("gate_clean") is True and
                           s.get("gate_observation_age_ns") is not None and
                           s["gate_observation_age_ns"] <= 1_000_000_000 and
                           bool(s.get("gate_observation")) for g in groups for s in g["samples"])
        row = dict(context=context, group=group["group_name"], processes=len(replicates),
                   clean=clean, retained=retained, suite_valid=bool(suite_valid),
                   all_rounds_clean=observations, exploratory=context == "mac/neon",
                   admissible=bool(suite_valid and observations and context != "mac/neon"),
                   ns_per_workload=dict(zip(names, means.tolist())), batch=batch,
                   intervals=intervals, process_family=family,
                   cpu_placement=packet.get("cpu_placement"))
        rows.append(row)
        print(context, row["group"], f"{clean}/{retained} clean", "admissible=" + str(row["admissible"]),
              ", ".join(f"{n}={m:.6f}ns" for n, m in zip(names, means)), flush=True)
output = root / "certified-analysis.json"
assert not output.exists(), output
output.write_text(json.dumps(dict(
    study=sys.argv[2], numpy=np.__version__,
    method="All retained rounds; equal process weighting; 10,000 paired within-process bootstrap resamples, NumPy PCG64 seed 0x531324; diagnostic df4 log process mean Student-t Bonferroni family95 plus timer floor; no IQR removal; admission failures retained. Mac exploratory only.",
    rows=rows), indent=2) + "\n")

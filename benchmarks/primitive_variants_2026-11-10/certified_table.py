#!/usr/bin/env python3
"""Render the complete measured matrix from retained-round analysis (no timing)."""
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
analysis = json.loads((root / "certified-analysis.json").read_text())
cold = analysis["study"] == "cold"
rows = analysis["rows"]
names = list(rows[0]["ns_per_workload"])
letters = "ABCDEF" if cold else "ABCDEFG"
lines = ["# " + ("Cold-repair" if cold else "Bounded/canonical") + " reciprocal timing matrix",
         "", "All values are **ns per complete workload**, all retained rounds, equal process weight.",
         "Short/long streams process 64/4096 elements; chain64 is 64 reciprocal-plus-0.5 steps.",
         "WASM host-call times are divided by their fixed 4096/256 batch. They include amortized host/guest-lock overhead.",
         "M6 is unconfined exploratory timing. A clean group inside a failed suite does not promote a default.",
         "No flagged rounds or outliers are removed. Full diagnostic intervals are in the hashed artifact packet.", ""]
for letter, name in zip(letters, names):
    lines.append(f"- **{letter}: {name}**.")
lines += ["", "Source/commands, numerical bounds, admission rules, bars and interpretation: [study](CERTIFIED-RECIP.md).", ""]
for context in dict.fromkeys(r["context"] for r in rows):
    subset = [r for r in rows if r["context"] == context]
    assert len(subset) == 13
    lines += ["## " + context, "", "| Width/workload | " + " | ".join(letters) + " | Clean/retained |",
              "|---|" + "---:|" * len(letters) + "---:|"]
    for r in subset:
        assert list(r["ns_per_workload"]) == names
        label = r["group"].split("/", 1)[1]
        lines.append("| " + label + " | " + " | ".join(f"{x:.3f}" for x in r["ns_per_workload"].values()) +
                     f" | {r['clean']}/{r['retained']} |")
    lines += ["", "Aggregate incomplete/unreliable flags clear: **" + str(subset[0]["suite_valid"]) + "**; " +
              str(sum(r["all_rounds_clean"] for r in subset)) + "/13 groups have all observations clean.", ""]
destination = root / "timings.md"
assert not destination.exists(), destination
text = "\n".join(lines) + "\n"
assert len(text.encode()) <= 30000
destination.write_text(text)
print(destination)

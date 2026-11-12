#!/usr/bin/env python3
"""Render all builder permutations from the actual Rust encoders/validators.

Canonical flag aliases are redirects. Algorithm/codegen aliases and timing
evidence have separate identities and must never be inferred from a label.
"""
import argparse
import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PREFIX = dict(minmax="M", recip="R", powf="P", expf="E", logf="L")
AXES = ["domain", "precision_or_nan", "portability_or_zero", "validation", "execution"]
COLS = [("Checked", "Automatic"), ("Checked", "Caller"),
        ("AssumeValid", "Automatic"), ("AssumeValid", "Caller")]


def read(path):
    with path.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    assert len(rows) == 304
    for r in rows:
        r["bits"] = int(r["bits"])
        r["supported"] = r["supported"] == "true"
    return rows


def identity(row):
    return tuple(row[k] for k in ["family", *AXES])


def cell(row):
    return PREFIX[row["family"]] + f'{row["bits"]:02x}'


def render(family, rows, nostd):
    rs = [r for r in rows if r["family"] == family]
    canonical = {}
    for r in rs:
        if r["supported"]:
            canonical.setdefault(r["bits"], r)
    expected = dict(minmax=(72,24), recip=(48,22), powf=(120,104), expf=(32,32), logf=(32,32))
    assert (len(rs), len(canonical)) == expected[family]
    axes = ["Domain", "NaN", "Zero"] if family == "minmax" else ["Domain", "Precision", "Portability"]
    lines = ["", "# Complete option permutation matrix", "",
             "Generated from the actual Rust builders and validity predicates, in std and no_std builds.",
             "Each cell fixes the row's three axes plus the column's validation/execution choices.",
             f"This covers **{len(rs)} raw combinations**, **{sum(r['supported'] for r in rs)} supported with std**,",
             f"and **{len(canonical)} distinct supported flag sets**. A flag ID identifies a configuration, not a timing result.", "",
             "- **Supported ID**: legal configuration; see its canonical row below.",
             "- **→ ID**: `.bits()` returns exactly the same flags as that canonical row. This is a configuration alias, not just similar accuracy.",
             "- **Unsupported**: rejected option combination; const API compilation fails. Transcendental runtime adapters return UnsupportedOptions.",
             "- **std only**: that combination is unsupported without std.",
             "- Execution Caller/Automatic remains a distinct flag when not canonicalized. Typed entries retain their supplied token; array dispatch is a separate timing dimension.", ""]
    for domain in dict.fromkeys(r["domain"] for r in rs):
        subset = [r for r in rs if r["domain"] == domain]
        lines += [f"<details><summary>{domain}: all precision/result-policy × validation × execution choices</summary>", "",
                  "| " + " | ".join(axes[1:]) + " | Checked / Auto | Checked / Caller | AssumeValid / Auto | AssumeValid / Caller |",
                  "|---|---|---|---|---|---|"]
        for pair in dict.fromkeys((r["precision_or_nan"], r["portability_or_zero"]) for r in subset):
            values = []
            for validation, execution in COLS:
                r = next(r for r in subset if (r["precision_or_nan"], r["portability_or_zero"], r["validation"], r["execution"]) == (*pair, validation, execution))
                if not r["supported"]:
                    values.append("Unsupported")
                    continue
                label = cell(r)
                alias = identity(r) != identity(canonical[r["bits"]])
                value = ("→ " if alias else "Supported ") + f"[{label}](#{label.lower()})"
                if not nostd[identity(r)]["supported"]:
                    value += " (std only)"
                values.append(value)
            lines.append("| " + " | ".join([*pair, *values]) + " |")
        lines += ["", "</details>", ""]
    lines += ["## Canonical cells", "", "| ID | " + " | ".join(axes) + " | Validation | Execution | Features |",
              "|---|---|---|---|---|---|---|"]
    for r in sorted(canonical.values(), key=lambda r:r["bits"]):
        label = cell(r)
        feature = "std required" if not nostd[identity(r)]["supported"] else "std or no_std"
        lines.append("| " + " | ".join([f'<a id="{label.lower()}"></a>{label}', r["domain"], r["precision_or_nan"], r["portability_or_zero"], r["validation"], r["execution"], feature]) + " |")
    lines += ["", "IDs expose the current encoded flags for traceability; construct options with the public builder, not handwritten integers.",
              "Numerical contracts and measured error are distinct from timing. Timing identities also include operation, scalar type, logical width, ISA, entry form, workload, compiler and source.",
              "A redirected configuration may reuse its canonical cell's exact evidence. Other reuse needs a separately verified graph or compiled-code identity.", ""]
    if family in ("minmax", "recip"):
        lines += ["## Precision and timing evidence", "",
                  "The [primitive permutation reference](https://github.com/imazen/archmage/blob/feat/primitive-contracts/docs/PRIMITIVE-PERMUTATIONS.md) joins these IDs to numerical contracts, measured error and per-ISA/width/workload timings.",
                  "Missing measurements are explicitly unmeasured; dirty retained runs are descriptive; M6 scheduler-selected runs are exploratory. Those states do not mean a demonstrated slowdown.", ""]
    else:
        lines += ["## Measurement coverage", "",
                  "This expansion inventories every option for this family. New timing coverage is currently scoped to min/max, reciprocal and rsqrt; the transcendental measurement gaps are not presented as completed work.", ""]
    text = "\n".join(lines).strip() + "\n"
    assert len(text.encode()) <= 30000, family
    return text


def main():
    p = argparse.ArgumentParser();p.add_argument("inventory", type=Path);p.add_argument("--check", action="store_true")
    args = p.parse_args()
    rows = read(args.inventory/"std.tsv")
    no = {identity(r):r for r in read(args.inventory/"no-std.tsv")}
    assert all(r["bits"] == no[identity(r)]["bits"] for r in rows)
    for family in PREFIX:
        text = render(family, rows, no)
        target = REPO/"magemath/src"/family/"permutations.md"
        if args.check:
            assert target.read_text() == text, f"stale {target}"
        else:
            target.write_text(text)
        print(family, len(text.encode()), "bytes")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Inventory actual body-policy decisions in an isolated copy of a measured case.

No instrumentation enters the production macros or the measured source/binaries.
Run through run-heavy; this performs one fresh Cargo check, not a timing run.
"""

import argparse
import collections
import csv
import json
import shutil
import subprocess
from pathlib import Path

from run import HERE, clean_env, replace, save, sha

FIELDS = [
    "package",
    "version",
    "macro",
    "file",
    "line",
    "column",
    "body_file",
    "body_line",
    "function",
    "source_visibility",
    "body_visibility",
    "kind",
    "tier",
    "target_arch",
    "feature_gate",
    "selected",
    "policy",
]
PREFIX = "ARCHMAGE_INLINE_INVENTORY\t"


def instrument(stack):
    root = stack / "archmage-macros/src"
    shutil.copyfile(HERE / "inventory_emit.rs", root / "inline_inventory.rs")
    with (root / "lib.rs").open("a") as f:
        f.write("\nmod inline_inventory;\n")
    replace(
        root / "arcane.rs",
        "    // Scalar has no instruction-set boundary.",
        """    let inventory_direct = features_csv.is_empty() || target_arch == Some("wasm32");
    let inventory_body_vis = if inventory_direct { &input_fn.vis } else { &syn::Visibility::Inherited };
    let inventory_kind = if inventory_direct { "direct" } else if args.nested { "hidden_nested" } else { "hidden_sibling" };
    crate::inline_inventory::record(crate::inline_inventory::Event {
        macro_name, ident: &input_fn.sig.ident, body: &input_fn.body, source_vis: &input_fn.vis,
        body_vis: inventory_body_vis, kind: inventory_kind,
        tier: token_type_name.as_deref().unwrap_or("trait-bound"),
        arch: target_arch, gate: args.shared.cfg_feature.as_deref(),
        selected: if args.inline_always { "always" } else if inline_attr.is_some() { "inline" } else { "none" },
        policy: if args.inline_always { "explicit_always" } else { "default" },
    });
    if !inventory_direct {
        crate::inline_inventory::record(crate::inline_inventory::Event {
            macro_name, ident: &input_fn.sig.ident, body: &input_fn.body, source_vis: &input_fn.vis,
            body_vis: &input_fn.vis, kind: "proof_wrapper",
            tier: token_type_name.as_deref().unwrap_or("trait-bound"),
            arch: target_arch, gate: args.shared.cfg_feature.as_deref(),
            selected: "always", policy: "unchanged",
        });
    }
    // Scalar has no instruction-set boundary.""",
    )
    replace(
        root / "rite.rs",
        "    variant_fn.attrs = new_attrs;",
        """    crate::inline_inventory::record(crate::inline_inventory::Event {
        macro_name: "rite", ident: &variant_fn.sig.ident, body: &variant_fn.body,
        source_vis: &variant_fn.vis, body_vis: &variant_fn.vis, kind: "direct",
        tier: tier.suffix.unwrap_or("trait-bound"), arch: tier.target_arch,
        gate: args.shared.cfg_feature.as_deref(),
        selected: if new_attrs.iter().any(|a| a.path().is_ident("inline")) { "inline" } else { "none" },
        policy: "default",
    });
    variant_fn.attrs = new_attrs;""",
    )


def write_csv(path, fields, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


def collect(measurement, out):
    out.mkdir(parents=True, exist_ok=False)
    source = measurement / "body_default"
    case = out / "case"
    for name in ("archmage", "zenav1-svt", "rav1d-safe", "driver"):
        shutil.copytree(
            source / name, case / name, ignore=shutil.ignore_patterns("target")
        )
    instrument(case / "archmage")
    driver = case / "driver"
    # Calibration functions need a direct import, without enabling new features.
    replace(
        driver / "Cargo.toml",
        "[dependencies]\n",
        '[dependencies]\narchmage={path="../archmage",default-features=false}\n',
    )
    shutil.copyfile(
        HERE / "inventory_controls.rs", driver / "src/inline_inventory_controls.rs"
    )
    with (driver / "src/main.rs").open("a") as f:
        f.write("\nmod inline_inventory_controls;\n")
    env = clean_env()
    metadata = json.loads(
        subprocess.check_output(
            ["cargo", "metadata", "--format-version", "1"],
            cwd=driver,
            env=env,
        )
    )
    save(out / "metadata.json", metadata)
    original_metadata = json.loads(
        subprocess.check_output(
            ["cargo", "metadata", "--locked", "--format-version", "1"],
            cwd=source / "driver",
            env=env,
        )
    )

    def feature_map(meta):
        resolved = {n["id"]: n["features"] for n in meta["resolve"]["nodes"]}
        return {
            (p["name"], p["version"]): sorted(resolved[p["id"]])
            for p in meta["packages"]
        }

    assert feature_map(metadata) == feature_map(original_metadata), (
        "calibration changed resolved package versions/features"
    )
    target_cfg = subprocess.check_output(
        ["rustc", "--print", "cfg"], env=env, text=True
    )
    arch = next(
        line.split('"')[1]
        for line in target_cfg.splitlines()
        if line.startswith("target_arch=")
    )
    command = ["cargo", "check", "--profile", "release", "--locked"]
    with (out / "compile.log").open("w") as log:
        subprocess.run(
            command,
            cwd=driver,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    nodes = {n["id"]: n["features"] for n in metadata["resolve"]["nodes"]}
    features = {(p["name"], p["version"]): nodes[p["id"]] for p in metadata["packages"]}
    rows = []
    for line in (out / "compile.log").read_text().splitlines():
        if not line.startswith(PREFIX):
            continue
        parts = line[len(PREFIX) :].split("\t")
        assert len(parts) == len(FIELDS), line
        row = dict(zip(FIELDS, parts, strict=True))
        for field in ("file", "body_file"):
            path = Path(row[field])
            if path.is_absolute() and path.is_relative_to(case):
                row[field] = str(path.relative_to(case))
        row["enabled_features"] = ",".join(
            sorted(features[(row["package"], row["version"])])
        )
        row["arch_matches"] = row["target_arch"] in ("any", arch)
        row["feature_matches"] = (
            not row["feature_gate"]
            or row["feature_gate"] in features[(row["package"], row["version"])]
        )
        row["generated_guards_match"] = row["arch_matches"] and row["feature_matches"]
        # Compare reported decisions with the documented rule, not source text heuristics.
        if row["policy"] == "default":
            assert row["selected"] == (
                "inline" if row["body_visibility"] == "pub" else "none"
            ), row
        rows.append(row)
    assert rows, "instrumentation produced no events"
    controls = [r for r in rows if r["package"] == "inline-consumer-probe"]
    expected = {
        ("public_direct", "direct", "inline"),
        ("crate_direct", "direct", "none"),
        ("private_direct", "direct", "none"),
        ("public_wrapped", "hidden_sibling", "none"),
        ("public_wrapped", "proof_wrapper", "always"),
        ("public_scalar", "direct", "inline"),
        ("explicit_always", "direct", "always"),
    }
    assert len(controls) == len(expected)
    assert {(r["function"], r["kind"], r["selected"]) for r in controls} == expected
    write_csv(
        out / "decisions.csv",
        [
            *FIELDS,
            "enabled_features",
            "arch_matches",
            "feature_matches",
            "generated_guards_match",
        ],
        rows,
    )
    summary = []
    dimensions = [
        "package",
        "macro",
        "source_visibility",
        "body_visibility",
        "kind",
        "selected",
        "policy",
        "generated_guards_match",
    ]
    counts = collections.Counter(tuple(r[k] for k in dimensions) for r in rows)
    for key, count in sorted(counts.items()):
        summary.append(dict(zip(dimensions, key, strict=True)) | {"count": count})
    write_csv(out / "summary.csv", [*dimensions, "count"], summary)
    file_counts = collections.Counter(
        (r["package"], r["file"], r["selected"])
        for r in rows
        if r["policy"] == "default" and r["generated_guards_match"]
    )
    write_csv(
        out / "by-file.csv",
        ["package", "file", "selected", "count"],
        [
            dict(zip(["package", "file", "selected"], key, strict=True))
            | {"count": count}
            for key, count in sorted(file_counts.items())
        ],
    )
    save(
        out / "provenance.json",
        {
            "measurement_root": str(measurement),
            "measurement_plan_sha256": sha(measurement / "plan.json"),
            "measurement_plan": json.loads((measurement / "plan.json").read_text()),
            "command": command,
            "target_cfg": target_cfg,
            "rustc": subprocess.check_output(["rustc", "-Vv"], text=True),
            "inventory_script_sha256": sha(Path(__file__)),
            "instrumentation_sha256": sha(HERE / "inventory_emit.rs"),
            "controls_sha256": sha(HERE / "inventory_controls.rs"),
            "artifacts": {
                name: {"bytes": (out / name).stat().st_size, "sha256": sha(out / name)}
                for name in (
                    "compile.log",
                    "decisions.csv",
                    "summary.csv",
                    "by-file.csv",
                )
            },
        },
    )
    print(
        f"Recorded {len(rows)} emission events; full inventory: {out / 'decisions.csv'}"
    )
    for package in sorted({r["package"] for r in rows}):
        selected = [
            r
            for r in rows
            if r["package"] == package
            and r["policy"] == "default"
            and r["generated_guards_match"]
        ]
        print(package, dict(collections.Counter(r["selected"] for r in selected)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    collect(args.measurement, args.out)

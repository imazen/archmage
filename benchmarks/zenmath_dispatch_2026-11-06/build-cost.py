#!/usr/bin/env python3
"""Cold consumer release builds: independent Cargo output per feature/replicate."""

from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
study = Path(__file__).resolve().parent
out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True, exist_ok=False)
manifest = study / "compile-consumer/Cargo.toml"
records = []
for feature in ["default", "accelerated", "runtime-dispatch"]:
    for rep in range(1, 4):
        cell = out / f"{feature}-{rep}"
        cell.mkdir()
        target = cell / "target"
        env = dict(os.environ, CARGO_TARGET_DIR=str(target))
        command = [
            "cargo",
            "build",
            "--locked",
            "--release",
            "--manifest-path",
            str(manifest),
            "--no-default-features",
        ]
        if feature != "default":
            command += ["--features", feature]
        with (cell / "build.log").open("w") as log:
            subprocess.run(
                ["/usr/bin/time", "-v", "-o", str(cell / "resources.txt"), *command],
                cwd=root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        binary = target / "release/zenmath-compile-consumer"
        records.append(
            dict(
                feature=feature,
                replicate=rep,
                command=command,
                binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                binary_bytes=binary.stat().st_size,
                resources=(cell / "resources.txt").read_text(),
            )
        )
        print(f"PASS cold build {feature}/{rep}", flush=True)
    tree = [
        "cargo",
        "tree",
        "--locked",
        "--manifest-path",
        str(manifest),
        "--no-default-features",
        "-e",
        "features",
    ]
    if feature != "default":
        tree += ["--features", feature]
    (out / f"{feature}-dependencies.txt").write_text(
        subprocess.check_output(tree, cwd=root, text=True)
    )
(out / "results.json").write_text(
    json.dumps(
        dict(
            rustc=subprocess.check_output(["rustc", "-vV"], text=True), records=records
        ),
        indent=2,
    )
    + "\n"
)

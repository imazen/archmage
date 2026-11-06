#!/usr/bin/env python3
"""Lift exact lockfile/checksum and path-source identities from a frozen snapshot."""

from pathlib import Path
import hashlib, json, sys, tarfile, tomllib

snapshot = Path(sys.argv[1])
identity = json.loads((snapshot / "identity.json").read_text())
out = []
with tarfile.open(snapshot / "source.tar.gz") as tar:
    for name in ["Cargo.lock", "benchmarks/zenmath_dispatch_2026-11-06/Cargo.lock"]:
        data = tar.extractfile(name).read()
        lock = tomllib.loads(data.decode())
        out.append(
            dict(
                path=name,
                sha256=hashlib.sha256(data).hexdigest(),
                packages=[
                    dict(
                        name=p["name"],
                        version=p["version"],
                        source=p.get("source", "path-source frozen in source.tar.gz"),
                        checksum=p.get("checksum"),
                        path_source_revision=None
                        if "source" in p
                        else identity["source_revision"],
                    )
                    for p in lock["package"]
                ],
            )
        )
(snapshot / "dependency-locks.json").write_text(
    json.dumps(
        dict(source_manifest_sha256=identity["source_manifest_sha256"], lockfiles=out),
        indent=2,
    )
    + "\n"
)
print(
    "Retained",
    len(out),
    "exact Cargo locks and",
    sum(len(x["packages"]) for x in out),
    "locked package/source identifiers",
)

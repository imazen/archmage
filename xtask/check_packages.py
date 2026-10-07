#!/usr/bin/env python3
"""Build the three packaged libraries together, without publishing anything."""
import argparse
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]
CRATES = ["archmage", "archmage-macros", "magetypes"]

def run(command, **kwargs):
    print("+ " + " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=ROOT, check=True, **kwargs)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--target", action="append", default=[])
args = parser.parse_args()
metadata = json.loads(subprocess.check_output(
    ["cargo", "metadata", "--no-deps", "--format-version", "1"], cwd=ROOT))
versions = {p["name"]: p["version"] for p in metadata["packages"]}
command = ["cargo", "package", "--no-verify", "--allow-dirty"]
for crate in CRATES:
    command += ["-p", crate]
run(command)
parent = ROOT / "target/package-verification"
parent.mkdir(parents=True, exist_ok=True)
staging = Path(tempfile.mkdtemp(prefix="sources-", dir=parent))
members = [f"{crate}-{versions[crate]}" for crate in CRATES]
for member in members:
    archive = Path(metadata["target_directory"]) / "package" / f"{member}.crate"
    with tarfile.open(archive) as tar:
        tar.extractall(staging, filter="data")
# Check the published dependency protocol, not just local path resolution.
archmage_manifest = tomllib.loads((staging / members[0] / "Cargo.toml").read_text())
magetypes_manifest = tomllib.loads((staging / members[2] / "Cargo.toml").read_text())
assert archmage_manifest["dependencies"]["archmage-macros"]["version"] == "=" + versions["archmage-macros"]
assert magetypes_manifest["dependencies"]["archmage"]["version"] == versions["archmage"]
manifest = '[workspace]\nresolver = "3"\nmembers = ' + json.dumps(members)
manifest += '\n\n[patch.crates-io]\n'
for crate, member in zip(CRATES, members):
    manifest += f'{crate} = {{ path = {json.dumps(member)} }}\n'
(staging / "Cargo.toml").write_text(manifest)
for target in args.target or [None]:
    command = ["cargo", "check", "--workspace", "--all-features", "--lib",
               "--manifest-path", str(staging / "Cargo.toml")]
    if target:
        command += ["--target", target]
    run(command)
print(f"Verified packaged sources: {staging}", flush=True)

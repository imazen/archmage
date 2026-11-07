#!/usr/bin/env python3
"""Independently check untouched published manifests with configuration-only patches."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import urllib.request

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--inventory', type=Path, required=True)
a = p.parse_args()
src, out = a.source.resolve(), a.output.resolve()
out.mkdir(parents=True, exist_ok=False)
rows = []
for spec in json.loads(a.inventory.read_text()):
    name, version = spec.split(':')
    archive = out / f'{name}-{version}.crate'
    req = urllib.request.Request(f'https://static.crates.io/crates/{name}/{name}-{version}.crate',
        headers={'User-Agent': 'archmage-release-compat/0.9.31'})
    with urllib.request.urlopen(req, timeout=60) as response:
        archive.write_bytes(response.read())
    for mode in ['baseline', 'candidate']:
        work = out / name / mode
        work.mkdir(parents=True)
        with tarfile.open(archive) as tar:
            tar.extractall(work, filter='data')
        package = work / f'{name}-{version}'
        lock = package / 'Cargo.lock'
        if lock.exists():
            lock.rename(package / 'Cargo.lock.published')
        manifest = package / 'Cargo.toml'
        before = hashlib.sha256(manifest.read_bytes()).hexdigest()
        patches = []
        if mode == 'candidate':
            for crate, suffix in [('archmage', ''), ('archmage-macros', 'archmage-macros'), ('magetypes', 'magetypes')]:
                patches += ['--config', f'patch.crates-io.{crate}.path={json.dumps(str(src / suffix))}']
        command = ['cargo', 'check', *patches]
        (work / 'command.json').write_text(json.dumps(command))
        env = dict(os.environ, CARGO_TARGET_DIR=str(out.parent.parent / 'target-consumers'))
        with (work / 'check.log').open('w') as log:
            rc = subprocess.run(command, cwd=package, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        metadata = subprocess.run(['cargo', 'metadata', '--format-version', '1', *patches],
            cwd=package, env=env, capture_output=True, text=True)
        (work / 'metadata.json').write_text(metadata.stdout)
        (work / 'metadata.stderr').write_text(metadata.stderr)
        selected = []
        if metadata.returncode == 0:
            data = json.loads(metadata.stdout)
            resolved = {n['id'] for n in data['resolve']['nodes']}
            selected = [{k: v[k] for k in ['name', 'version', 'source', 'manifest_path']}
                for v in data['packages'] if v['id'] in resolved and v['name'] in ['archmage','archmage-macros','magetypes']]
        after = hashlib.sha256(manifest.read_bytes()).hexdigest()
        if before != after:
            raise RuntimeError(f'Manifest mutated: {manifest}')
        rows.append(dict(consumer=spec, mode=mode, rc=rc, metadata_rc=metadata.returncode,
            selected=selected, manifest_sha256=before,
            archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest()))
        (out / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
        print(spec, mode, 'rc', rc, [(x['name'], x['version'], x['source']) for x in selected], flush=True)

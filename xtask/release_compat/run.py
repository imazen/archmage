#!/usr/bin/env python3
"""Validate an isolated stable candidate; never point --source at a live checkout.
Run under run-heavy. Keeps registry responses, commands, logs, and exit codes.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import urllib.request

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--phase', choices=['prepare', 'semver', 'copter'], required=True)
a = p.parse_args()
src, out = a.source.resolve(), a.output.resolve()
out.mkdir(parents=True, exist_ok=True)

def run(label, command, cwd):
    log = out / (label + '.log')
    if log.exists():
        raise RuntimeError(f'Refusing to overwrite {log}')
    (out / (label + '.command.json')).write_text(json.dumps(command, indent=2) + '\n')
    with log.open('w') as stream:
        rc = subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT).returncode
    (out / (label + '.exit')).write_text(str(rc) + '\n')
    print(f'{label}: exit {rc}; {log}', flush=True)
    return rc

def api(path):
    req = urllib.request.Request('https://crates.io/api/v1/' + path,
        headers={'User-Agent': 'archmage-release-compat/0.9.31'})
    with urllib.request.urlopen(req, timeout=60) as response:
        data = json.load(response)
    (out / ('registry-' + path.replace('/', '_').replace('?', '_') + '.json')).write_text(json.dumps(data))
    return data

if a.phase == 'prepare':
    if (out / 'candidate-manifests.json').exists():
        raise RuntimeError('Candidate already prepared')
    changed = []
    for manifest in src.rglob('Cargo.toml'):
        old = manifest.read_text()
        new = old.replace('0.9.31-beta.2', '0.9.31')
        if new != old:
            manifest.write_text(new)
            changed.append(str(manifest.relative_to(src)))
    (out / 'candidate-manifests.json').write_text(json.dumps(changed, indent=2))
    for crate in ['archmage', 'magetypes']:
        names = set()
        page = 1
        while True:
            d = api(f'crates/{crate}/reverse_dependencies?page={page}&per_page=100')
            names.update(v['crate'] for v in d['versions'])
            if page * 100 >= d['meta']['total']:
                break
            page += 1
        selected = []
        for name in sorted(names):
            d = api(f'crates/{name}')
            versions = [v for v in d['versions'] if not v['yanked']]
            stable = d['crate'].get('max_stable_version')
            version = next((v for v in versions if v['num'] == stable), versions[0])
            selected.append(f"{name}:{version['num']}")
        (out / f'{crate}-dependents.json').write_text(json.dumps(selected, indent=2))
        print(crate, len(selected), 'published consumers', flush=True)
elif a.phase == 'semver':
    failures = []
    for crate in ['archmage', 'magetypes']:
        for baseline in ['0.9.30', '0.9.31-beta.2']:
            label = f'semver-{crate}-{baseline}'
            if run(label, ['python3', 'xtask/check_semver.py', crate, '--baseline-version', baseline], src):
                failures.append(label)
    raise SystemExit(bool(failures))
else:
    config = out / '.cargo/config.toml'
    config.parent.mkdir(exist_ok=True)
    content = '[build]\ntarget-dir = ' + json.dumps(str(out.parent / 'target-consumers')) + '\n'
    if config.exists() and config.read_text() != content:
        raise RuntimeError(f'Refusing to replace {config}')
    config.write_text(content)
    failures = []
    for crate, suffix in [('archmage', ''), ('magetypes', 'magetypes')]:
        work = out / ('copter-' + crate)
        work.mkdir(exist_ok=True)
        deps = json.loads((out / f'{crate}-dependents.json').read_text())
        cmd = ['cargo-copter', '--path', str(src / suffix), '--dependents', *deps,
               '--test-versions', '0.9.30', '0.9.31-beta.2', '--only-check', '--simple',
               '--staging-dir', str(work / 'staging')]
        if run('copter-' + crate, cmd, work):
            failures.append(crate)
    raise SystemExit(bool(failures))

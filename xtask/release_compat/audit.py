#!/usr/bin/env python3
"""Verify candidate crates actually compiled on the host, not just metadata resolution."""
import argparse
import json
import os
from pathlib import Path
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('directory', type=Path)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--consumer')
p.add_argument('--features', default='')
a = p.parse_args()
root = a.directory.resolve()
source = a.source.resolve()
results = []
for row in json.loads((root / 'results.json').read_text()):
    if row['mode'] != 'candidate':
        continue
    if a.consumer and row['consumer'] != a.consumer:
        continue
    name, version = row['consumer'].split(':')
    work = root / name / 'candidate'
    suffix = '-' + a.features if a.features else ''
    log = work / ('artifact-check' + suffix + '.jsonl')
    if log.exists():
        raise RuntimeError(f'Refusing to overwrite {log}')
    command = json.loads((work / 'command.json').read_text()) + ['--message-format=json']
    if a.features:
        command += ['--features', a.features]
    env = dict(os.environ, CARGO_TARGET_DIR=str(root.parent.parent / 'target-consumers'))
    with log.open('w') as stdout, (work / ('artifact-check' + suffix + '.stderr')).open('w') as stderr:
        rc = subprocess.run(command, cwd=work / f'{name}-{version}', env=env,
            stdout=stdout, stderr=stderr).returncode
    compiled = set()
    for line in log.read_text().splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get('reason') == 'compiler-artifact':
            path = Path(event.get('manifest_path', ''))
            if path.is_relative_to(source):
                compiled.add(event['target']['name'])
    results.append(dict(consumer=row['consumer'], rc=rc, compiled=sorted(compiled)))
    (root / ('artifact-audit' + suffix + '.json')).write_text(json.dumps(results, indent=2) + '\n')
    print(row['consumer'], rc, sorted(compiled), flush=True)

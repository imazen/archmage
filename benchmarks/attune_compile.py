#!/usr/bin/env python3
"""Measure the unchanged downstream compile-cost fixture in fresh target dirs.

Run through run-heavy. Each output retains Cargo's full log and time's max RSS.
Never deletes results or reuses an existing output directory.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, required=True)
p.add_argument('--runs', type=int, default=3)
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
a.out.mkdir(parents=True, exist_ok=False)
metadata = {
    'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
    'rustc': subprocess.check_output(['rustc', '-Vv'], text=True),
    'cargo': subprocess.check_output(['cargo', '-V'], text=True).strip(),
    'runs': [],
}
for features in ('', 'avx512,use_magetypes'):
    for iteration in range(a.runs):
        label = f"{'all' if features else 'macros'}-{iteration}"
        env = dict(os.environ, CARGO_TARGET_DIR=str(a.out / label / 'target'))
        command = ['cargo', 'check', '--manifest-path', str(root / 'tests/downstream-compat/compile-cost/Cargo.toml')]
        if features:
            command += ['--features', features]
        start = time.perf_counter()
        with (a.out / f'{label}.log').open('w') as log:
            run = subprocess.run(['/usr/bin/time', '-v', *command], env=env, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        record = {'label': label, 'command': command, 'seconds': time.perf_counter()-start, 'exit_code': run.returncode}
        metadata['runs'].append(record)
        (a.out / 'results.json').write_text(json.dumps(metadata, indent=2)+'\n')
        print(json.dumps(record), flush=True)
        if run.returncode:
            raise SystemExit(run.returncode)

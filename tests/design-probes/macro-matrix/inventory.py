#!/usr/bin/env python3
"""Record compiler acceptance/diagnostics for the capability matrix, not a test gate."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
project = Path(__file__).resolve().parent
root = project.parents[2]
env = os.environ.copy()
for key in ('RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS', 'CARGO_BUILD_TARGET'):
    env.pop(key, None)
env.setdefault('CARGO_TARGET_DIR', str(root / 'target/macro-matrix'))
toolchain = subprocess.check_output(['rustc', '-vV'], text=True)
if 'host: x86_64-' not in toolchain:
    parser.error('These capability probes require an x86_64 host (no emulation needed).')
(args.output / 'metadata.json').write_text(json.dumps({
    'rustc': toolchain,
    'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
    'purpose': 'Compile-only inventory; rejection is an observed result, not a skipped test',
}, indent=2) + '\n')
rows = []
for features in ('', 'avx512', 'ungated-autoversion', 'ungated-autoversion,avx512',
                 'signature-alias', 'tokenless-arcane', 'magetypes-without-token',
                 'rite-placeholder', 'rite-tier-gate'):
    label = features or 'default'
    print(f'START {label}', flush=True)
    result = subprocess.run(
        ['cargo', 'check', '--manifest-path', str(project / 'Cargo.toml'),
         '--features', features, '--color', 'never'],
        env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    (args.output / (label.replace(',', '-') + '.log')).write_text(result.stdout)
    row = {'features': label, 'exit_code': result.returncode,
           'diagnostics': '; '.join(re.findall(r'^error[^\n]*', result.stdout, re.MULTILINE))}
    rows.append(row)
    with (args.output / 'results.csv').open('w') as output:
        writer = csv.DictWriter(output, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(rows)
    print(f'RECORDED {label}: {row}', flush=True)

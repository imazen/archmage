#!/usr/bin/env python3
"""Paired cold-target consumer builds. Run under run-heavy; never deletes caches."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--before', type=Path, required=True)
p.add_argument('--after', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--runs', type=int, default=3)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
env = os.environ.copy()
for key in ('RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS', 'RUSTC_WRAPPER', 'RUSTC_WORKSPACE_WRAPPER'):
    env.pop(key, None)
env.update(CARGO_INCREMENTAL='0', CARGO_BUILD_JOBS='8')
metadata = {'rustc': subprocess.check_output(['rustc', '-vV'], text=True),
            'host': platform.node(), 'platform': platform.platform(), 'runs': a.runs,
            'jobs': 8, 'incremental': False, 'profile': 'release',
            'cold': 'fresh Cargo target directory; filesystem cache not flushed',
            'sources': {k: str(v.resolve()) for k, v in [('before', a.before), ('after', a.after)]}}
(a.output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
rows = []
for features in ['default', 'avx512']:
    consumers = {}
    for label, source, local in [('before', a.before, False), ('after_explicit', a.after, False), ('after_local', a.after, True)]:
        project = a.output / f'{features}-{label}'
        (project / 'src').mkdir(parents=True)
        deps = source.resolve()
        feature_list = '["avx512"]' if features == 'avx512' else '[]'
        (project / 'Cargo.toml').write_text(f'''[package]
name = "local-mode-consumer"
version = "0.0.0"
edition = "2024"
[lib]
path = "src/lib.rs"
[dependencies]
archmage = {{ path = "{deps}", features = {feature_list} }}
magetypes = {{ path = "{deps}/magetypes", features = {feature_list} }}
[workspace]
''')
        mode, token = ('local', '') if local else ('define', 'token, ')
        source_text = f'''use archmage::{{magetypes, incant}};
#[magetypes({mode}(f32x8, i32x8), v3, scalar)]
fn kernel(token: Token, values: &[f32]) -> f32 {{
    let mut sum = f32x8::zero({"" if local else "token"});
    let factor = f32x8::splat({token}0.5);
    let (chunks, tail) = f32x8::partition_slice({token}values);
    for chunk in chunks {{
        let v = f32x8::load({token}chunk).mul_add(factor, factor);
        let ints: i32x8 = v.to_i32_saturating();
        sum += f32x8::from_i32({token}ints);
    }}
    sum.reduce_add() + tail.iter().sum::<f32>()
}}
pub fn run(values: &[f32]) -> f32 {{ incant!(kernel(values), [v3, scalar]) }}
'''
        if local:
            source_text = source_text.replace('fn kernel(token:', 'fn kernel(_token:')
        (project / 'src/lib.rs').write_text(source_text)
        shutil.copy2(source / 'Cargo.lock', project / 'Cargo.lock')
        with (project / 'resolve.log').open('w') as log:
            subprocess.run(['cargo', 'generate-lockfile', '--offline'], cwd=project, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        consumers[label] = project
    for run in range(a.runs):
        # Rotate first position to reduce ordering effects while keeping jobs serial.
        labels = list(consumers)
        labels = labels[run % 3:] + labels[:run % 3]
        for label in labels:
            project = consumers[label]
            target = project / f'target-{run}'
            assert not target.exists()
            command = ['cargo', 'build', '--release', '--locked', '--offline', '--timings', '--target-dir', str(target)]
            log_path = project / f'build-{run}.log'
            print(f'START {features} {label} run={run + 1}', flush=True)
            start = time.perf_counter()
            with log_path.open('w') as log:
                subprocess.run(command, cwd=project, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            elapsed = time.perf_counter() - start
            html = (target / 'cargo-timings/cargo-timing.html').read_text()
            units = json.loads(re.search(r'const UNIT_DATA = (\[.*?\]);', html, re.S).group(1))
            (project / f'units-{run}.json').write_text(json.dumps(units, indent=2))
            times = {u['name']: u['duration'] for u in units}
            row = dict(features=features, label=label, run=run+1, total_seconds=elapsed,
                       magetypes_seconds=times['magetypes'], consumer_seconds=times['local-mode-consumer'])
            rows.append(row)
            with (a.output / 'results.csv').open('w') as f:
                writer = csv.DictWriter(f, fieldnames=list(row)); writer.writeheader(); writer.writerows(rows)
            print(f'END {features} {label} total={elapsed:.3f}s magetypes={times["magetypes"]:.3f}s', flush=True)
summary = []
for features in ['default', 'avx512']:
    for label in ['before', 'after_explicit', 'after_local']:
        matches = [r for r in rows if r['features'] == features and r['label'] == label]
        summary.append(dict(features=features, label=label, **{
            key: {'median': statistics.median(r[key] for r in matches),
                  'min': min(r[key] for r in matches), 'max': max(r[key] for r in matches)}
            for key in ['total_seconds', 'magetypes_seconds', 'consumer_seconds']}))
(a.output / 'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)

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
p.add_argument('--middle', type=Path, help='Optional attribute-only candidate')
p.add_argument('--before-local', action='store_true', help='Baseline already supports local constructors')
p.add_argument('--tier-width-probe', action='store_true',
               help='Compare fixed f32x8 with per-tier existing types; does not implement use(f32x)')
p.add_argument('--adaptive-use-probe', action='store_true',
               help='Compare manual natural-width aliases before/after with implemented use(f32xN)')
p.add_argument('--sde', type=Path, help='SDE executable for mandatory AVX-512 width-probe tests')
p.add_argument('--runs', type=int, default=3)
a = p.parse_args()
width_probe = a.tier_width_probe or a.adaptive_use_probe
if width_probe and (not a.sde or not a.sde.is_file()):
    p.error('width probes require --sde for AVX-512 execution')
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
cases = [('before', a.before, False)]
if a.before_local:
    cases.append(('before_local', a.before, True))
if a.middle:
    cases.extend([('middle_explicit', a.middle, False), ('middle_local', a.middle, True)])
cases.extend([('after_explicit', a.after, False), ('after_local', a.after, True)])
if a.tier_width_probe:
    cases = [('fixed8', a.after, False), ('tier_width', a.after, True)]
    metadata['experiment'] = 'Width selection expansion shape; parser cost and scalar x1 are not measured'
if a.adaptive_use_probe:
    cases = [('before_manual', a.before, True), ('after_manual', a.after, True), ('after_adaptive', a.after, True)]
    metadata['experiment'] = 'Actual adaptive use parser versus identical manual per-tier aliases'
metadata['cases'] = {label: {'source': str(source.resolve()), 'local': local} for label, source, local in cases}
(a.output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
rows = []
for features in ['default', 'avx512']:
    consumers = {}
    for label, source, local in cases:
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
        if width_probe:
            manifest = project / 'Cargo.toml'
            defaults = '["avx512"]' if features == 'avx512' else '[]'
            manifest.write_text(manifest.read_text().replace('[workspace]',
                '[features]\ndefault = ' + defaults + '\navx512 = ["archmage/avx512", "magetypes/avx512"]\n[workspace]'))
        context_option = 'use' if 'Token![use]' in (source / 'archmage-macros/src/lib.rs').read_text() else 'local'
        mode, token = (context_option, '') if local else ('define', 'token, ')
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
        if width_probe:
            source_text = (Path(__file__).resolve().parent.parent /
                           'tests/design-probes/context-mode/width_kernel.in.rs').read_text()
            widths = {'V3': 8, 'NEON': 4 if local else 8,
                      'WASM': 4 if local else 8, 'SCALAR': 4 if local else 8}
            for tier, width in widths.items():
                source_text = source_text.replace(f'@{tier}@', f'f32x{width}')
            v4_width = 16 if local else 8
            source_text = source_text.replace('@V4_KERNEL@',
                f'kernel!(kernel_v4, v4, archmage::X64V4Token, f32x{v4_width});' if features == 'avx512' else '')
            source_text = source_text.replace('@V4_ENTRY@',
                '#[arcane] fn apply_v4(_: archmage::X64V4Token, v: &mut [f32]) { kernel_v4(v); }' if features == 'avx512' else '')
            source_text = source_text.replace('@V4_TEST@', """
    #[cfg(target_arch="x86_64")]
    #[test] fn v4_tails_and_rows() {
        let token = archmage::X64V4Token::summon().expect("caller must provide a V4 CPU");
        exercise(|v| apply_v4(token, v));
    }
""" if features == 'avx512' else '')
            source_text = source_text.replace('@TIERS@', 'v4, v3, ' if features == 'avx512' else 'v3, ')
        if a.adaptive_use_probe and label == 'after_adaptive':
            source_text = source_text.replace('#[archmage::rite($tier)]', '#[archmage::rite($tier, use(f32xN))]')
            source_text = source_text.replace('type V = magetypes::simd::generic::local::$vector<$token>;', 'type V = f32xN;')
        (project / 'src/lib.rs').write_text(source_text)
        shutil.copy2(source / 'Cargo.lock', project / 'Cargo.lock')
        with (project / 'resolve.log').open('w') as log:
            subprocess.run(['cargo', 'generate-lockfile', '--offline'], cwd=project, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        consumers[label] = project
    for run in range(a.runs):
        # Rotate first position to reduce ordering effects while keeping jobs serial.
        labels = list(consumers)
        labels = labels[run % len(labels):] + labels[:run % len(labels)]
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
                       macros_seconds=times['archmage-macros'], magetypes_seconds=times['magetypes'], consumer_seconds=times['local-mode-consumer'])
            rows.append(row)
            with (a.output / 'results.csv').open('w') as f:
                writer = csv.DictWriter(f, fieldnames=list(row)); writer.writeheader(); writer.writerows(rows)
            print(f'END {features} {label} total={elapsed:.3f}s magetypes={times["magetypes"]:.3f}s', flush=True)
    if width_probe:
        for label, project in consumers.items():
            print(f'TEST {features} {label}', flush=True)
            cmd = ['cargo', 'test', '--release', '--locked', '--offline', '--no-run',
                   '--message-format=json', '--target-dir', str(project / 'target-0')]
            built = subprocess.run(cmd, cwd=project, env=env, text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
            (project / 'test-build.log').write_text(built.stdout)
            built.check_returncode()
            binaries = [d['executable'] for line in built.stdout.splitlines() if line.startswith('{')
                        for d in [json.loads(line)] if d.get('executable') and d.get('profile', {}).get('test')]
            assert binaries
            with (project / 'test-run.log').open('w') as log:
                for binary in binaries:
                    command = [str(a.sde), '-skx', '--', binary] if features == 'avx512' else [binary]
                    subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
summary = []
for features in ['default', 'avx512']:
    for label, _, _ in cases:
        matches = [r for r in rows if r['features'] == features and r['label'] == label]
        summary.append(dict(features=features, label=label, **{
            key: {'median': statistics.median(r[key] for r in matches),
                  'min': min(r[key] for r in matches), 'max': max(r[key] for r in matches)}
            for key in ['total_seconds', 'macros_seconds', 'magetypes_seconds', 'consumer_seconds']}))
(a.output / 'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)

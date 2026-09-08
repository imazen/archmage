#!/usr/bin/env python3
"""Compare consumer cold builds and ordinary/macro-body edits on an idle machine.
Usage: python3 xtask/macro_perf.py ROOT
ROOT contains before/ and after/ archmage source trees, plus linear.tar.gz and
zenpixels.tar.gz source archives and their linear.Cargo.lock/zenpixels.Cargo.lock.
The archives must have Cargo.toml at their root. This harness patches only the
archmage family (and the same linear-srgb into zenpixels), verifies equal resolved
dependency trees, fetches before timing, alternates order, and uses six independent
empty target directories per configuration. Cold means empty Cargo artifacts,
not a cleared OS disk cache. Source edits must rebuild the consumer only.
Results include raw timings, load, Cargo artifact freshness, locks, and source hashes.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

root = Path(sys.argv[1]).resolve()
variants = ['before', 'after']
def arch_path(variant):
    return root/variant
results = root / 'results'
results.mkdir(exist_ok=True)
env = dict(os.environ, CARGO_TERM_COLOR='never', TMPDIR=str(Path.home()/'tmp'))
for key in ['CARGO_INCREMENTAL', 'RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS',
            'RUSTC_WRAPPER', 'RUSTC_WORKSPACE_WRAPPER', 'CARGO_TARGET_DIR', 'CARGO_BUILD_TARGET']:
    env.pop(key, None)

def run(cmd, cwd, sample_env=env):
    p = subprocess.run(cmd, cwd=cwd, env=sample_env, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stdout + p.stderr)
    return p

for variant in variants:
    base = root / 'consumers' / variant
    for name in ['linear', 'zenpixels']:
        dest = base/name
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(root/(name+'.tar.gz')) as archive:
            archive.extractall(dest, filter='data')
        shutil.copy(root/(name+'.Cargo.lock'), dest/'Cargo.lock')
        arch = arch_path(variant)
        patch = '\n[patch.crates-io]\n'
        for crate, path in [('archmage',arch), ('archmage-macros',arch/'archmage-macros'), ('magetypes',arch/'magetypes')]:
            patch += f'{crate} = {{ path = "{path}" }}\n'
        if name == 'zenpixels':
            patch += f'linear-srgb = {{ path = "{base / "linear"}" }}\n'
        with (dest/'Cargo.toml').open('a') as f:
            f.write(patch)
        if 'BENCH_LOCKS_DIR' in os.environ:
            shutil.copy(Path(os.environ['BENCH_LOCKS_DIR']) / f'{variant}-{name}.Cargo.lock', dest/'Cargo.lock')
        else:
            run(['cargo', '+stable', 'generate-lockfile'], dest)
        p = run(['cargo', '+stable', 'fetch', '--locked'], dest)
        (results/f'{variant}-{name}-fetch.log').write_text(p.stdout+p.stderr)
        shutil.copy(dest/'Cargo.lock', results/f'{variant}-{name}.Cargo.lock')

workloads = {
    'linear-srgb': dict(directory='linear', features=['--features', 'transfer'], edits=[
        ('ordinary', 'src/scalar.rs', 'pub fn srgb_to_linear(gamma: f32) -> f32 {', '\n    let gamma = core::hint::black_box(gamma);'),
        ('arcane', 'src/simd.rs', 'fn $tier_v3(token: X64V3Token, values: &mut [f32]) {', '\n            let values = core::hint::black_box(values);'),
    ]),
    'zenpixels-convert': dict(directory='zenpixels', features=[], edits=[
        ('ordinary', 'zenpixels-convert/src/fast_gamut.rs', 'pub fn convert_linear_rgb(m: &[[f32; 3]; 3], data: &mut [f32]) {', '\n    let data = core::hint::black_box(data);'),
        ('arcane', 'zenpixels-convert/src/fast_gamut.rs', 'fn [<convert_rgb_ $name _v3>](token: X64V3Token, m: &[[f32; 3]; 3], data: &mut [f32]) {', '\n                let data = core::hint::black_box(data);'),
    ]),
}
for package, config in workloads.items():
    trees = []
    for variant in variants:
        dest = root/'consumers'/variant/config['directory']
        cmd = ['cargo', '+stable', 'tree', '--locked', '--offline', '-p', package, '-e', 'normal,build'] + config['features']
        tree = run(cmd, dest).stdout
        (results/f'{package}-{variant}-tree.txt').write_text(tree)
        tree = tree.replace('archmage v0.9.28', 'archmage v0.9.29').replace('archmage-macros v0.9.28', 'archmage-macros v0.9.29').replace('magetypes v0.9.28', 'magetypes v0.9.29')
        trees.append(tree.replace(str(root/variant), 'CONSUMER').replace(str(arch_path(variant)), 'ARCHMAGE'))
    assert len(set(trees)) == 1, ('dependency trees differ', package)

rows = []
metadata = dict(rustc=run(['rustc', '+stable', '-Vv'], root).stdout,
                cpu=run(['lscpu'], root).stdout,
                profiles='Cargo defaults: dev incremental on, release incremental off',
                edits=workloads, sources_cached=True,
                archive_sha256={name: __import__('hashlib').sha256((root/name).read_bytes()).hexdigest()
                                for name in ['linear.tar.gz', 'zenpixels.tar.gz']})
(results/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')

def build(package, config, variant, mode, trial, stage, target):
    cwd = root/'consumers'/variant/config['directory']
    cmd = ['cargo', '+stable', 'build', '--offline', '--locked', '--lib', '-p', package,
           '--message-format=json'] + config['features']
    if mode == 'release':
        cmd.append('--release')
    sample_env = dict(env, CARGO_TARGET_DIR=str(target))
    before = Path('/proc/loadavg').read_text().strip()
    start = time.perf_counter()
    p = run(cmd, cwd, sample_env)
    elapsed = time.perf_counter()-start
    name = f'{package}-{mode}-{trial}-{variant}-{stage}'
    (results/(name+'.log')).write_text(p.stdout+p.stderr)
    rebuilt = []
    fresh = []
    for line in p.stdout.splitlines():
        item = json.loads(line)
        if item.get('reason') == 'compiler-artifact':
            (fresh if item['fresh'] else rebuilt).append(item['target']['name'])
    if stage != 'cold':
        assert not set(rebuilt) & {'archmage', 'archmage_macros', 'magetypes'}, rebuilt
        assert package.replace('-', '_') in rebuilt, rebuilt
    row = dict(package=package, mode=mode, trial=trial, variant=variant, stage=stage,
               seconds=elapsed, rebuilt=rebuilt, fresh=fresh, load_before=before,
               load_after=Path('/proc/loadavg').read_text().strip())
    rows.append(row)
    (results/'runs.json').write_text(json.dumps(rows, indent=2)+'\n')
    print(json.dumps(row), flush=True)

for package, config in workloads.items():
    for mode in ['debug', 'release']:
        for trial, order in enumerate([variants, variants[::-1]] * 3, 1):
            for variant in order:
                cwd = root/'consumers'/variant/config['directory']
                target = root/'targets'/f'{package}-{mode}-{trial}-{variant}'
                target.mkdir(parents=True, exist_ok=False)
                build(package, config, variant, mode, trial, 'cold', target)
                # A single edit to an ordinary function, then a single edit to
                # an arcane macro_rules template (expands to multiple functions).
                # Keep the first edit present during the second, so each timed
                # transition changes only one function/template.
                originals = {}
                for stage, rel, needle, addition in config['edits']:
                    path = cwd/rel
                    source = path.read_text()
                    originals.setdefault(path, source)
                    assert source.count(needle) == 1, (path, needle)
                    path.write_text(source.replace(needle, needle+addition))
                    build(package, config, variant, mode, trial, stage, target)
                for path, source in originals.items():
                    path.write_text(source)
                shutil.rmtree(target)  # Only the sample directory we just created.

#!/usr/bin/env python3
"""Time cold builds of the magetypes crate itself, before vs after a change.

Usage: python3 benchmarks/magetypes_compile_perf.py ROOT [PAIRS]
ROOT contains before/ and after/ archmage source trees (for example from
`git archive <rev> | tar -x -C ROOT/before`). Each tree keeps its own Cargo.lock
and target directory. Dependencies are built once per tree and configuration;
each sample then removes only magetypes' artifacts (`cargo clean -p magetypes`)
and rebuilds it, so the timed work is the magetypes unit that every downstream
build waits on. CARGO_INCREMENTAL=0 matches how a dependency is compiled.
Order alternates every pair. Results go to ROOT/results.json.
"""
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

root = Path(sys.argv[1]).resolve()
pairs = int(sys.argv[2]) if len(sys.argv) > 2 else 6
variants = ['before', 'after']
configs = [
    ('dev', []),
    ('release', ['--release']),
    ('release+avx512', ['--release', '--features', 'avx512']),
]
env = dict(os.environ, CARGO_TERM_COLOR='never', CARGO_INCREMENTAL='0',
           TMPDIR=str(Path.home() / 'tmp'))
for key in ['RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS', 'RUSTC_WRAPPER',
            'RUSTC_WORKSPACE_WRAPPER', 'CARGO_TARGET_DIR', 'CARGO_BUILD_TARGET']:
    env.pop(key, None)

# t(0.975, df) for small samples; the interval is approximate.
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201}


def run(cmd, cwd):
    p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(f"{' '.join(cmd)} failed in {cwd}:\n{p.stdout}{p.stderr}")
    return p


def clean_args(args):
    return ['--release'] if '--release' in args else []


def sample(variant, args):
    cwd = root / variant
    run(['cargo', 'clean', '-p', 'magetypes', *clean_args(args)], cwd)
    t0 = time.perf_counter()
    p = run(['cargo', 'build', '-p', 'magetypes', *args], cwd)
    elapsed = time.perf_counter() - t0
    compiled = re.findall(r'^\s*Compiling (\S+)', p.stderr, re.M)
    if compiled != ['magetypes']:
        raise RuntimeError(f'{variant} {args}: expected only magetypes to rebuild, got {compiled}')
    return elapsed


rustc = run(['rustc', '--version'], root).stdout.strip()
results = {'rustc': rustc, 'pairs': pairs, 'configs': {}}
for name, args in configs:
    for variant in variants:
        run(['cargo', 'build', '-p', 'magetypes', *args], root / variant)
    times = {v: [] for v in variants}
    for i in range(pairs):
        order = variants if i % 2 == 0 else variants[::-1]
        for variant in order:
            times[variant].append(sample(variant, args))
    logs = [math.log(a / b) for a, b in zip(times['after'], times['before'])]
    mean = statistics.fmean(logs)
    half = T975[pairs - 1] * statistics.stdev(logs) / math.sqrt(pairs) if pairs > 1 else float('nan')
    summary = {
        'before_median_s': statistics.median(times['before']),
        'after_median_s': statistics.median(times['after']),
        'paired_change_pct': 100 * (math.exp(mean) - 1),
        'ci95_pct': [100 * (math.exp(mean - half) - 1), 100 * (math.exp(mean + half) - 1)],
        'before_s': times['before'],
        'after_s': times['after'],
    }
    results['configs'][name] = summary
    lo, hi = summary['ci95_pct']
    print(f"{name:16s} before {summary['before_median_s']:6.2f} s  after {summary['after_median_s']:6.2f} s  "
          f"paired {summary['paired_change_pct']:+5.1f}% [{lo:+.1f}, {hi:+.1f}]", flush=True)

(root / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
print(f"{rustc}; {pairs} alternating pairs per configuration; results in {root / 'results.json'}")

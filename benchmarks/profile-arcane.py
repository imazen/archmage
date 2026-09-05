#!/usr/bin/env python3
"""Profile macro execution in two consumer worktrees (not macro-crate compilation)."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('baseline', type=Path)
parser.add_argument('candidate', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--toolchain', default='nightly-2026-09-02')
parser.add_argument('--runs', type=int, default=6)
args = parser.parse_args()
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, CARGO_INCREMENTAL='0', CARGO_TERM_COLOR='never')
env.pop('RUSTC_WRAPPER', None)
variants = [('baseline', args.baseline.resolve()), ('candidate', args.candidate.resolve())]
cmd = ['cargo', '+' + args.toolchain, 'rustc', '--locked', '-p', 'magetypes', '--features', 'avx512', '--lib']

def run(command, root):
    return subprocess.run(command, cwd=root, env=env, check=True, capture_output=True, text=True)

def seconds(value):
    return value['secs'] + value['nanos'] / 1e9

# Warm dependencies before timing. Each profile run recompiles only magetypes.
for _, root in variants:
    run(cmd, root)
rows = []
for trial in range(args.runs):
    for label, root in variants[::1 if trial % 2 == 0 else -1]:
        directory = output / f'{label}-{trial + 1}'
        directory.mkdir()  # Refuse stale traces rather than silently mixing runs.
        run(['cargo', '+' + args.toolchain, 'clean', '-p', 'magetypes'], root)
        start = time.perf_counter()
        result = run(cmd + ['--', '-Zmacro-stats', '-Zself-profile=' + str(directory)], root)
        elapsed = time.perf_counter() - start
        (directory / 'build.log').write_text(result.stdout + result.stderr)
        traces = list(directory.glob('*.mm_profdata'))
        if len(traces) != 1:
            raise RuntimeError(f'Expected one trace in {directory}, found {traces}')
        trace = traces[0]
        summary = run(['summarize', 'summarize', str(trace)], root)
        (directory / 'summary.txt').write_text(summary.stdout)
        run(['summarize', 'summarize', '--json', str(trace)], root)
        data = json.loads(trace.with_suffix('.json').read_text())
        events = {event['label']: event for event in data['query_data']}
        selected = {}
        for name in ['expand_proc_macro', 'expand_crate', 'typeck_root', 'mir_borrowck']:
            event = events[name]
            selected[name] = {'self_seconds': seconds(event['self_time']),
                              'inclusive_seconds': seconds(event['time']),
                              'count': event['invocation_count']}
        row = dict(variant=label, trial=trial + 1, wall_seconds=elapsed, events=selected)
        rows.append(row)
        print(json.dumps(row), flush=True)
metadata = {
    'rustc': run(['rustc', '+' + args.toolchain, '-Vv'], args.baseline).stdout,
    'command': cmd,
    'baseline_commit': run(['git', 'rev-parse', 'HEAD'], args.baseline).stdout.strip(),
    'candidate_base_commit': run(['git', 'rev-parse', 'HEAD'], args.candidate).stdout.strip(),
    'candidate_diff': run(['git', 'diff', '--', 'archmage-macros/src'], args.candidate).stdout,
    'runs': rows,
}
(output / 'results.json').write_text(json.dumps(metadata, indent=2) + '\n')

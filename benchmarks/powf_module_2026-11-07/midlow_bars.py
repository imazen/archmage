#!/usr/bin/env python3
"""Preserve zenbench bars and complete method names with ns per buffer."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('directory', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
assert not a.output.exists()
files = sorted(a.directory.glob('[0-9][0-9][0-9].json'))
assert files
parts = ['# Retained zenbench bars and ns means\n\n'
         'Times are ns per complete buffer, including scalar tails; the group names '
         'specify buffer size. Bars are copied verbatim from zenbench. Their '
         'displayed units and abbreviated labels are unchanged; the adjacent '
         'tables preserve complete method names. All rounds are retained.\n']
for path in files:
    x = json.loads(path.read_text())
    assert len(x['comparisons']) == 1
    g = x['comparisons'][0]
    assert len(g['samples']) == 50
    clean = sum(s['gate_clean'] is True for s in g['samples'])
    parts += [f"\n## {g['group_name']}\n\nPlacement: {x['cpu_placement']['status']}; "
              f"clean rounds: {clean}/50; unreliable: {x['unreliable']}.\n\n"
              '| Actual implementation | Mean ns |\n|---|---:|\n']
    for b in g['benchmarks']:
        parts += [f"| {b['name']} | {b['summary']['mean']:,.2f} |\n"]
    bars = [s for s in path.with_suffix('.log').read_text().splitlines() if '█' in s]
    assert bars
    parts += ['\n```text\n' + '\n'.join(bars) + '\n```\n']
a.output.write_text(''.join(parts))
print(json.dumps({'cells': len(files), 'output': str(a.output)}))

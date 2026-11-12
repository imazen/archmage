#!/usr/bin/env python3
"""Inventory outlined helper calls in timed selectors of a frozen guest WAT.

This describes module call sites, not dynamic counts or JIT machine code.
Read the source's SOFTWARE const before interpreting the mangled bool.
"""
import argparse
from collections import Counter
from pathlib import Path
import re

p = argparse.ArgumentParser()
p.add_argument('wat', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
assert not a.output.exists(), a.output
counts = Counter()
examples = {}
function = ''
for number, line in enumerate(a.wat.open(), 1):
    if line.startswith('  (func '):
        function = line.strip()
    if '5guest' not in function or 'select_' not in function:
        continue
    call = re.search(r'\bcall (\$\S+)', line)
    if call is None:
        continue
    name = call[1]
    if '14primitive_wasm6kernel' in name:
        label = 'minmax helper'
    elif '4powf4wasm5lanesKb0_' in name:
        label = 'fused lanes'
    elif '4powf4wasm5lanesKb1_' in name:
        label = 'software lanes'
    else:
        continue
    counts[label] += 1
    examples.setdefault(label, []).append((number, function, line.strip()))
assert counts, 'No matching calls: inspect module names before interpreting this output'
lines = ['Timed-selector WAT call sites (not dynamic or machine-instruction counts):',
         repr(dict(counts)), '']
for label, entries in examples.items():
    lines.append(label + ': first four call-site examples')
    for number, function, call in entries[:4]:
        lines.extend([f'{number}: {function}', call, ''])
a.output.write_text('\n'.join(lines) + '\n')
print(dict(counts))

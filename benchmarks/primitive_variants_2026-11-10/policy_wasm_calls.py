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
selectors = 0
for number, line in enumerate(a.wat.open(), 1):
    if line.startswith('  (func '):
        function = line.strip()
        if '5guest' in function and 'select_' in function:
            selectors += 1
    if '5guest' not in function or 'select_' not in function:
        continue
    call = re.search(r'\bcall (\$\S+)', line)
    if call is None:
        continue
    name = call[1]
    if '14primitive_wasm' in name:
        label = 'minmax helper'
    elif '4powf4wasm5lanesKb0_' in name:
        label = 'fused lanes'
    elif '4powf4wasm5lanesKb1_' in name:
        label = 'software lanes'
    elif '8wasm_fma' in name and 'fuses' in name:
        label = 'fuses probe'
    elif '11try_from_fn' in name:
        label = 'array::try_from_fn'
    else:
        continue
    counts[label] += 1
    examples.setdefault(label, []).append((number, function, line.strip()))
assert selectors > 0, 'No timed select_* functions found in WAT module'
lines = [f'Timed-selector WAT call sites across {selectors} selectors (not dynamic or machine-instruction counts):',
         repr(dict(counts)), '']
for label, entries in examples.items():
    lines.append(label + ': first four call-site examples')
    for number, function, call in entries[:4]:
        lines.extend([f'{number}: {function}', call, ''])
a.output.write_text('\n'.join(lines) + '\n')
print({'selectors': selectors, 'outlined_calls': dict(counts)})

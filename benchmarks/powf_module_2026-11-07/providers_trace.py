#!/usr/bin/env python3
"""Preserve every within-cell provider/literal/opaque trace difference."""
import argparse
import itertools
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('trace', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
assert not a.output.exists()
budget = 512 * 1024 * 1024
used = 0
report = {'trace': str(a.trace), 'cells': 0, 'elements': 0, 'edges': {},
          'chain_semantics': 'normalized trajectories; after divergence each provider has its own previous state',
          'ledger_budget_bytes': budget}
with a.output.with_suffix('.variances.jsonl').open('x') as out:
    for line in a.trace.open():
        cell = json.loads(line)
        report['cells'] += 1
        report['elements'] += len(cell['inputs'])
        for left, right in itertools.combinations(cell['outputs'], 2):
            edge = left['name'] + ' / ' + right['name']
            stats = report['edges'].setdefault(edge, {'differences': 0, 'by_cell': {}})
            count = 0
            for i, (x, y) in enumerate(zip(left['bits'], right['bits'], strict=True)):
                if x == y:
                    continue
                count += 1
                width = cell['width']
                prior = i-width
                chained = cell['chain_normalized'] and prior >= 0
                record = {'group': cell['group'], 'index': i, 'edge': edge,
                          'exponent_bits': cell['exponent_bits'],
                          'left_input': f"{(left['bits'][prior] if chained else cell['inputs'][i]):08x}",
                          'right_input': f"{(right['bits'][prior] if chained else cell['inputs'][i]):08x}",
                          'left_output': f'{x:08x}', 'right_output': f'{y:08x}',
                          'normalized_chain': cell['chain_normalized']}
                encoded = json.dumps(record)+'\n'
                used += len(encoded.encode())
                if used > budget:
                    raise RuntimeError('ledger budget exceeded; partial output retained')
                out.write(encoded)
            stats['differences'] += count
            stats['by_cell'][cell['group']] = count
assert report['cells'] == 56
report['ledger_bytes'] = used
a.output.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='edges'}, indent=2))
print(json.dumps({k:v['differences'] for k,v in report['edges'].items()}, indent=2))

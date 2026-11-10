#!/usr/bin/env python3
"""Compare retained policy bodies, ignoring LLVM private-symbol unique suffixes."""
import argparse
import json
from pathlib import Path
import re

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('before', type=Path)
p.add_argument('after', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
left, right = (json.loads(path.read_text()) for path in [a.before, a.after])
report = {'before': str(a.before), 'after': str(a.after),
          'normalization': 'LLVM .llvm.<digits> unique private-symbol suffix only',
          'collections': {}}
for collection, count in [('rows', 384), ('extra_shapes', 22)]:
    def key(row):
        prefix = row['flags'] if collection == 'rows' else row['kind']
        return prefix, row['width'], row['tier']
    def bodies(document):
        rows = document[collection]
        assert len(rows) == count
        result = {key(row): [re.sub(r'\.llvm\.[0-9]+', '.llvm.UNIQUE', line)
                            for line in row['body']] for row in rows}
        assert len(result) == count, 'duplicate probe keys'
        return result
    before, after = bodies(left), bodies(right)
    assert before.keys() == after.keys(), 'different probe coverage'
    changed = [probe for probe in before if before[probe] != after[probe]]
    assert not changed, (collection, 'changed instruction bodies', changed)
    report['collections'][collection] = {'identical': count}
with a.output.open('x') as output:
    json.dump(report, output, indent=2)
    output.write('\n')
print('384 policy and 22 shape bodies identical; operands and instruction order retained')

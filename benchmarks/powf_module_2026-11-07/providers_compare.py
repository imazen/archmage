#!/usr/bin/env python3
"""Record cross-execution non-NaN bit/class differences without assuming equality."""
import argparse
import json
from pathlib import Path
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('output', type=Path)
p.add_argument('actuals', nargs='+', type=Path)
a = p.parse_args()
assert len(a.actuals) >= 2 and not a.output.exists()
all_rows = [[line.split() for line in path.read_text().splitlines()] for path in a.actuals]
left = all_rows[0]
report = {'baseline': str(a.actuals[0]), 'cases': len(left), 'comparisons': []}
def nan(bits):
    return bits & 0x7fffffff > 0x7f800000
for path, right in zip(a.actuals[1:], all_rows[1:], strict=True):
    assert len(right) == len(left)
    counts = [{'provider': name, 'non_nan_bit_differences': 0, 'NaN_class_differences': 0,
               'NaN_payload_differences': 0, 'witnesses': []}
              for name in ['std_f32_powf', 'Rust_libm_0_2_16_powf']]
    for row, other in zip(left, right, strict=True):
        assert row[:3] == other[:3] and len(row) == len(other) == 5
        for i, stats in enumerate(counts):
            x, y = int(row[3+i], 16), int(other[3+i], 16)
            key = ('NaN_class_differences' if nan(x) != nan(y) else
                   'NaN_payload_differences' if nan(x) and x != y else
                   'non_nan_bit_differences' if not nan(x) and x != y else None)
            if key:
                stats[key] += 1
                if len(stats['witnesses']) < 16:
                    stats['witnesses'].append({'input': row[:3], 'left': row[3+i], 'right': other[3+i], 'kind': key})
    report['comparisons'].append({'actual': str(path), 'providers': counts})
a.output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))

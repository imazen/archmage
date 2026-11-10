#!/usr/bin/env python3
"""Independent MPFR accuracy and special-class comparison for actual providers."""
import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import shlex
import subprocess
import rounding_accuracy as reference
import rounding_oracle as model
from replacement_accuracy import rounded_reference

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('binary', type=Path)
p.add_argument('fixtures', type=Path)
p.add_argument('output', type=Path)
p.add_argument('--tier', choices=['scalar', 'native'], required=True)
p.add_argument('--runner', default='')
a = p.parse_args()
assert not a.output.exists()
request = ''.join(' '.join(row.split()[:3]) + '\n' for row in a.fixtures.read_text().splitlines())
request += ''.join(f'{b:08x} {n:08x} 3\n'
    for b in [0, 0x80000000, 1, 0x007fffff, 0x7f7fffff, 0x7f800000, 0xff800000,
              0x7fc00001, 0x7f800001, 0xbf800000, 0xc0000000]
    for n in [0, 0x80000000, 1, 0x3f000000, 0x40000000, 0x40400000, 0xbf800000,
              0x7f800000, 0x7fc00001])
run = subprocess.run(shlex.split(a.runner) + [str(a.binary.resolve()), a.tier],
                     input=request, text=True, capture_output=True)
a.output.with_suffix('.actual').write_text(run.stdout)
a.output.with_suffix('.stderr').write_text(run.stderr)
run.check_returncode()
a.output.with_suffix('.inputs').write_text(request)

def classify(bits):
    x = model.fp.value(bits)
    return ('nan' if math.isnan(x) else 'infinity' if math.isinf(x) else 'zero' if x == 0
            else 'subnormal' if abs(x) < 2**-126 else 'normal')

variances = a.output.with_suffix('.variances.jsonl').open('x')
report = {'cases': len(request.splitlines()), 'tier': a.tier, 'runner': a.runner,
          'precision': 256, 'libm_version': '0.2.16', 'libm_features': ['arch'],
          'claim': 'sampled actual outputs; no whole-domain ULP/portability guarantee',
          'widths_checked': [1, 4, 8, 16], 'mixed_lane_checks': 'bits or NaN class',
          'providers': {}, 'pairwise_non_nan_bit_differences': 0,
          'pairwise_class_differences': 0, 'class_witnesses': []}
for row, output in zip(request.splitlines(), run.stdout.splitlines(), strict=True):
    fields, got = row.split(), output.split()
    assert got[:3] == fields and len(got) == 5
    xb, nb = int(fields[0], 16), int(fields[1], 16)
    x, n = model.fp.value(xb), model.fp.value(nb)
    bits = [int(b, 16) for b in got[3:]]
    classes = [classify(b) for b in bits]
    if classes[0] != classes[1]:
        report['pairwise_class_differences'] += 1
        if len(report['class_witnesses']) < 16:
            report['class_witnesses'].append({'input': row, 'std': got[3], 'Rust_libm': got[4]})
    if 'nan' not in classes and bits[0] != bits[1]:
        report['pairwise_non_nan_bit_differences'] += 1
    valid = x > 0 and math.isfinite(x) and math.isfinite(n)
    rounded = rounded_reference(xb, nb) if valid else None
    if bits[0] != bits[1]:
        kind = ('classification' if classes[0] != classes[1] else
                'NaN_payload_or_sign' if classes[0] == 'nan' else 'non_NaN_bits')
        variances.write(json.dumps({'input': fields, 'std_bits': got[3],
            'Rust_libm_bits': got[4], 'classes': classes, 'kind': kind,
            'rounded_MPFR': f'{rounded:08x}' if rounded is not None else None}) + '\n')
    bounds, quantum = reference.reference(xb, nb) if valid else (None, None)
    for name, b, cls in zip(['std_f32_powf', 'Rust_libm_0_2_16_powf'], bits, classes, strict=True):
        stats = report['providers'].setdefault(name, {'classes': {}, 'rounded_differences': 0,
                 'max_finite_output_ulp_upper': 0., 'max_relative_upper': 0., 'worst': None,
                 'nonfinite_mismatches_against_MPFR': 0, 'range_witnesses': []})
        stats['classes'][cls] = stats['classes'].get(cls, 0) + 1
        if not valid:
            continue
        stats['rounded_differences'] += b != rounded
        y = model.fp.value(b)
        if not math.isfinite(y) and b != rounded:
            stats['nonfinite_mismatches_against_MPFR'] += 1
            if len(stats['range_witnesses']) < 16:
                stats['range_witnesses'].append({'input': row, 'actual': f'{b:08x}',
                                               'rounded_MPFR': f'{rounded:08x}'})
        if not math.isfinite(y) or not all(math.isfinite(z) for z in bounds):
            continue
        value = Fraction.from_float(y)
        error = max(abs(value - Fraction.from_float(z)) for z in bounds) / (Fraction(2)**quantum)
        error = math.nextafter(float(error), math.inf) if error else 0.
        if error > stats['max_finite_output_ulp_upper']:
            stats.update(max_finite_output_ulp_upper=error, worst={'input': row, 'actual': f'{b:08x}', 'reference': bounds})
        if all(z > 0 for z in bounds):
            relative = max(abs(value - Fraction.from_float(z)) / Fraction.from_float(z) for z in bounds)
            relative = math.nextafter(float(relative), math.inf) if relative else 0.
            stats['max_relative_upper'] = max(stats['max_relative_upper'], relative)
variances.close()
report['all_variances'] = str(a.output.with_suffix('.variances.jsonl'))
a.output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))

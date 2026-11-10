#!/usr/bin/env python3
"""Fixed-graph exactness and directed MPFR accuracy of actual library outputs.

MPFR bindings adapted from the independent study at archmage da295fadb24fa.
No production dependency; std powf is compared only as a candidate, never truth.
"""
import argparse
import ctypes as C
import ctypes.util
from fractions import Fraction
import json
import math
from pathlib import Path
import random
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'xtask'))
import powf_vectors as oracle

class MP(C.Structure):
    _fields_ = [('prec', C.c_long), ('sign', C.c_int), ('exp', C.c_long), ('d', C.POINTER(C.c_ulong))]
mp = C.CDLL(ctypes.util.find_library('mpfr'))
P = C.POINTER(MP)
for name, args, result in [('init2', [P, C.c_long], None), ('clear', [P], None), ('set_d', [P, C.c_double, C.c_int], C.c_int), ('get_d', [P, C.c_int], C.c_double), ('get_version', [], C.c_char_p), ('pow', [P, P, P, C.c_int], C.c_int)]:
    fn = getattr(mp, 'mpfr_' + name)
    fn.argtypes, fn.restype = args, result

def enclosure(base, exponent):
    a, b, c = MP(), MP(), MP()
    for z in [a, b, c]: mp.mpfr_init2(C.byref(z), 256)
    mp.mpfr_set_d(C.byref(a), oracle.value(base), 0)
    mp.mpfr_set_d(C.byref(b), oracle.value(exponent), 0)
    bounds = []
    for rounding in [3, 2]:
        mp.mpfr_pow(C.byref(c), C.byref(a), C.byref(b), rounding)
        bounds.append(mp.mpfr_get_d(C.byref(c), rounding))
    for z in [a, b, c]: mp.mpfr_clear(C.byref(z))
    return bounds

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary')
    parser.add_argument('output')
    args = parser.parse_args()
    assert not Path(args.output).exists(), 'preserve earlier evidence'
    rng = random.Random(0x504f5746)
    cases = [(b, n, d) for d in range(3) for b, n in oracle.cases()]
    exponents = list(map(oracle.bits, [0., .5, 2.4, 1/2.4, .1593017578125, 1/.1593017578125, 78.84375, 1/78.84375, 4.]))
    for domain in range(3):
        cases += [(rng.randrange(oracle.bits(2.**-16), oracle.bits(2.**16)+1) if domain == 2 else rng.randrange(0x00800000, 0x7f800000), rng.choice(exponents), domain) for _ in range(8192)]
    request = ''.join(f'{b:08x} {n:08x} {d}\n' for b,n,d in cases)
    run = subprocess.run([args.binary], input=request, text=True, capture_output=True, check=True)
    Path(args.output + '.inputs').write_text(request)
    Path(args.output + '.actual').write_text(run.stdout)
    lines = run.stdout.splitlines()
    assert len(lines) == len(cases)
    summary = {'mpfr': mp.mpfr_get_version().decode(), 'precision':256, 'cases': len(cases), 'mismatches': [], 'profiles': {}}
    for (base, exponent, domain), line in zip(cases, lines):
        fields = line.split()
        assert fields[:3] == [f'{base:08x}', f'{exponent:08x}', str(domain)]
        expected, error = oracle.mirror(base, exponent, domain != 0, domain == 2)
        q = summary['profiles'].setdefault(str(domain), {'accepted':0, 'rejected':0, 'std_bit_differences':0, 'max_ulp_upper':0., 'worst':None})
        if error:
            assert fields[3] == {1:'InvalidExponent',2:'InvalidBase',3:'OutsideApproximationRange'}[error]
            q['rejected'] += 1
            continue
        actual, std = int(fields[3],16), int(fields[4],16)
        if actual != expected: summary['mismatches'].append([base,exponent,domain,actual,expected])
        q['accepted'] += 1
        q['std_bit_differences'] += actual != std
        lower, upper = enclosure(base, exponent)
        assert math.isfinite(lower) and math.isfinite(upper)
        # Define ULP using the exact reference binade. Refine ambiguous binades
        # rather than selecting an endpoint; exact powers may straddle only if
        # the two directed references are equal at the boundary.
        e1 = math.frexp(lower)[1] if lower else -125
        e2 = math.frexp(upper)[1] if upper else -125
        assert e1 == e2, ('reference binade unresolved', base, exponent, lower, upper)
        quantum = Fraction(2) ** max(e1 - 24, -149)
        y = Fraction.from_float(oracle.value(actual))
        err = max(abs(y-Fraction.from_float(lower)),abs(y-Fraction.from_float(upper)))/quantum
        bound = math.nextafter(float(err), math.inf) if err else 0.
        if bound > q['max_ulp_upper']:
            q['max_ulp_upper'], q['worst'] = bound, {'base':f'{base:08x}', 'exponent':f'{exponent:08x}', 'actual':f'{actual:08x}', 'reference_interval':[lower,upper]}
    Path(args.output).write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    assert not summary['mismatches'], 'fixed graph contract failed'
if __name__ == '__main__': main()

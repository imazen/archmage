#!/usr/bin/env python3
"""Compare every actual method on identical pairs using directed MPFR truth."""
import argparse
import ctypes as C
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
import accuracy

accuracy.mp.mpfr_get_flt.argtypes=[accuracy.P,C.c_int]
accuracy.mp.mpfr_get_flt.restype=C.c_float

def rounded_reference(base,n):
    values=[accuracy.MP()for _ in range(3)]
    for v in values:accuracy.mp.mpfr_init2(C.byref(v),256)
    a,b,c=values
    accuracy.mp.mpfr_set_d(C.byref(a),accuracy.oracle.value(base),0)
    accuracy.mp.mpfr_set_d(C.byref(b),accuracy.oracle.value(n),0)
    bits=[]
    for rnd in [3,2]:
        accuracy.mp.mpfr_pow(C.byref(c),C.byref(a),C.byref(b),rnd)
        bits.append(accuracy.oracle.bits(accuracy.mp.mpfr_get_flt(C.byref(c),0)))
    for v in values:accuracy.mp.mpfr_clear(C.byref(v))
    assert bits[0]==bits[1],('unresolved MPFR rounding',base,n,bits)
    return bits[0]

NAMES = ['canonical', 'pow_lowp', 'pow_lowp_unchecked', 'pow_midp',
         'pow_midp_unchecked', 'pow_midp_precise', 'std_f32_powf',
         'std_promoted_f64_powf']

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('binary', type=Path)
    p.add_argument('inputs', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--witnesses', type=Path)
    a = p.parse_args()
    assert not a.output.exists(), 'preserve prior evidence'
    request = a.inputs.read_text()
    if a.witnesses:
        known = set(request.splitlines())
        request += ''.join(row+'\n' for row in sorted(set(a.witnesses.read_text().splitlines())-known))
    a.output.with_suffix('.inputs').write_text(request)
    actual = {}
    for tier in ['scalar', 'native']:
        run = subprocess.run([str(a.binary.resolve()), tier], input=request,
                             text=True, capture_output=True, check=True)
        a.output.with_suffix('.'+tier+'.actual').write_text(run.stdout)
        actual[tier] = [line.split() for line in run.stdout.splitlines()]
        assert len(actual[tier]) == len(request.splitlines())
    report = {'mpfr':accuracy.mp.mpfr_get_version().decode(), 'precision':256,
              'cases':len(request.splitlines()), 'tiers':{},
              'claim':'largest measured error; no full-domain accuracy bound',
              'fixed_graph_mismatches':[], 'canonical_tier_mismatches':[]}
    for index, row in enumerate(request.splitlines()):
        fields = row.split()
        base, n, domain = int(fields[0],16), int(fields[1],16), int(fields[2])
        expected, error = accuracy.oracle.mirror(base,n,domain!=0,domain==2)
        enclosure = None
        if not error:
            enclosure = accuracy.enclosure(base,n)
            lower, upper = enclosure
            assert math.isfinite(lower) and math.isfinite(upper)
            binades = [math.frexp(v)[1] if v else -125 for v in enclosure]
            assert binades[0] == binades[1], ('ambiguous reference binade',row,enclosure)
            quantum = Fraction(2)**max(binades[0]-24,-149)
            reference_bits = [rounded_reference(base,n)]
        canonical = []
        for tier in actual:
            line = actual[tier][index]
            assert line[:3] == fields
            values = [int(x,16) for x in line[3:]]
            assert len(values) == len(NAMES)
            canonical.append(values[0])
            profile = report['tiers'].setdefault(tier,{}).setdefault(str(domain),
                {'accepted':0,'rejected':0,'methods':{},'rejected_outputs':{}})
            if error:
                assert values[0]>>32 == error, (row,tier,values[0],error)
                profile['rejected'] += 1
                for name,bits in zip(NAMES[1:],values[1:]):
                    value = accuracy.oracle.value(bits)
                    cls = 'nan' if math.isnan(value) else 'infinity' if math.isinf(value) else 'zero' if value==0 else 'finite'
                    counts = profile['rejected_outputs'].setdefault(name,{})
                    key = str(error)+':'+cls
                    counts[key] = counts.get(key,0)+1
                continue
            assert values[0]>>32 == 0, (row,tier,values[0])
            profile['accepted'] += 1
            if values[0] != expected:
                report['fixed_graph_mismatches'].append([row,tier,values[0],expected])
            assert values[1] == values[2] and values[3] == values[5], ('legacy aliases',row,tier)
            for name,bits in zip(NAMES,values):
                scopes=[profile['methods']]
                if base & 0x7fffffff and n & 0x7fffffff:
                    scopes.append(profile.setdefault('methods_without_zero_rails',{}))
                for scope in scopes:
                    stats = scope.setdefault(name, {'nonfinite':0,'correctly_rounded':0,
                        'canonical_bit_differences':0,'max_ulp_upper':0.,'worst':None})
                    stats['canonical_bit_differences'] += bits != values[0]
                    stats['correctly_rounded'] += bits == reference_bits[0]
                    value = accuracy.oracle.value(bits)
                    if not math.isfinite(value):
                        stats['nonfinite'] += 1
                        continue
                    y = Fraction.from_float(value)
                    bound = max(abs(y-Fraction.from_float(v)) for v in enclosure)/quantum
                    bound = math.nextafter(float(bound),math.inf) if bound else 0.
                    if bound > stats['max_ulp_upper']:
                        stats['max_ulp_upper'] = bound
                        stats['worst'] = {'base':fields[0],'exponent':fields[1],
                            'actual':f'{bits:08x}','reference_interval':enclosure}
        if len(set(canonical))!=1:
            report['canonical_tier_mismatches'].append([row,canonical])
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    assert not report['fixed_graph_mismatches'] and not report['canonical_tier_mismatches']

if __name__ == '__main__':
    main()

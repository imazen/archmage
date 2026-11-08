#!/usr/bin/env python3
"""Actual forced-rounding graphs, exact rational bits, directed MPFR accuracy."""
import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
import accuracy
import rounding_oracle as model
from replacement_accuracy import rounded_reference

NAMES = ['midp_separate', 'midp_fused', 'lowp_separate', 'lowp_fused']


def corpus(path):
    rows = path.read_text().splitlines()
    added = []
    # Fixed exponent and exp2 split boundaries, with neighboring f32 exponents.
    for base in [0x37800000, 0x3eaaaaab, 0x3f000000, 0x3f7fffff,
                 0x3f800000, 0x3f800001, 0x3fc00000, 0x47800000]:
        x = model.fp.value(base)
        ns = [0., .5, 2.4, 4., -4., 65536., 2.**24]
        if x != 1:
            for k in [-126., -64.5, -1., -.5, 0., .5, 1., 63.5, 127.5]:
                ns.append(k / math.log2(x))
        for n in ns:
            bits = model.fp.bits(n)
            for nb in [bits, bits-1, bits+1]:
                if 0 <= nb <= 0xffffffff:
                    for domain in range(3):
                        added.append(f'{base:08x} {nb:08x} {domain}')
    return '\n'.join(rows + sorted(set(added) - set(rows))) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('binary', type=Path)
    p.add_argument('inputs', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--witnesses', type=Path)
    a = p.parse_args()
    assert not a.output.exists(), 'preserve prior evidence'
    request = corpus(a.inputs)
    if a.witnesses:
        request += ''.join(r+'\n' for r in sorted(set(a.witnesses.read_text().splitlines())-set(request.splitlines())))
    a.output.with_suffix('.inputs').write_text(request)
    actual = {}
    for tier in ['scalar', 'native']:
        run = subprocess.run([str(a.binary.resolve()), tier], input=request,
                             text=True, capture_output=True, check=True)
        a.output.with_suffix('.'+tier+'.actual').write_text(run.stdout)
        actual[tier] = [line.split() for line in run.stdout.splitlines()]
        assert len(actual[tier]) == len(request.splitlines())
    report = dict(cases=len(request.splitlines()), precision=256,
                  mpfr=accuracy.mp.mpfr_get_version().decode(),
                  gamma_certificates=model.gamma_certificates(),
                  claim='sampled maxima; no full-domain ULP guarantee',
                  mismatches=[], tiers={}, rounding_differences={})
    for index, row in enumerate(request.splitlines()):
        fields = row.split()
        base, n, domain = int(fields[0],16), int(fields[1],16), int(fields[2])
        expected = [model.mirror(base,n,domain,m>=2,m%2==1) for m in range(4)]
        values = {}
        for tier, lines in actual.items():
            assert lines[index][:3] == fields
            values[tier] = [int(v,16) for v in lines[index][3:]]
            assert len(values[tier]) == 6
        if values['scalar'][:4] != values['native'][:4]:
            report['mismatches'].append(['tier',row,values])
        valid = not model.preflight(base,n,domain)
        enclosure = accuracy.enclosure(base,n) if valid else None
        ref_bits = rounded_reference(base,n) if valid else None
        for m, (bits,error) in enumerate(expected):
            for tier in actual:
                out = values[tier][m]
                if out != ((error << 32) if error else bits):
                    report['mismatches'].append(['oracle',row,tier,NAMES[m],hex(out),hex(bits),error])
                stats = report['tiers'].setdefault(tier,{}).setdefault(str(domain),{}).setdefault(NAMES[m],
                    dict(accepted=0,rejected=0,correctly_rounded=0,max_ulp_upper=0.,
                         max_relative_upper=0.,worst=None,reference_classes={},legacy_compared=0,legacy_mismatches=0))
                stats['rejected' if error else 'accepted'] += 1
                if error:
                    continue
                stats['correctly_rounded'] += bits == ref_bits
                cls = ('overflow' if ref_bits == model.fp.INF else
                       'zero' if ref_bits == 0 else 'subnormal' if ref_bits < 0x00800000 else 'normal')
                stats['reference_classes'][cls] = stats['reference_classes'].get(cls,0)+1
                # Legacy has different zero and range handling; compare only
                # ordinary positive arguments supported by both families.
                if base and ((tier == 'scalar' and m%2==0) or (tier=='native' and m%2==1)):
                    stats['legacy_compared'] += 1
                    old = values[tier][4 if m>=2 else 5]
                    stats['legacy_mismatches'] += bits != old
                lower, upper = enclosure
                if not (math.isfinite(lower) and math.isfinite(upper)):
                    continue
                e = [math.frexp(v)[1] if v else -125 for v in enclosure]
                assert e[0] == e[1], ('unresolved reference binade',row,enclosure)
                quantum = Fraction(2)**max(e[0]-24,-149)
                y = Fraction.from_float(model.fp.value(bits))
                delta = max(abs(y-Fraction.from_float(v)) for v in enclosure)
                ulp = math.nextafter(float(delta/quantum),math.inf) if delta else 0.
                relative = float(delta / Fraction.from_float(lower)) if lower else (math.inf if delta else 0.)
                if relative > stats['max_relative_upper']:
                    stats['max_relative_upper'] = math.nextafter(relative,math.inf)
                if ulp > stats['max_ulp_upper']:
                    stats.update(max_ulp_upper=ulp,worst=dict(input=row,actual=f'{bits:08x}',reference=enclosure))
        for family, pair in [('midp',(0,1)),('lowp',(2,3))]:
            s = report['rounding_differences'].setdefault(str(domain),{}).setdefault(family,dict(common_accepted=0,bit_differences=0,rejection_differences=0))
            x,y = pair
            s['rejection_differences'] += expected[x][1] != expected[y][1]
            if expected[x][1] == expected[y][1] == 0:
                s['common_accepted'] += 1
                s['bit_differences'] += expected[x][0] != expected[y][0]
        if index % 1024 == 0:
            print(json.dumps(dict(processed=index,required=report['cases'],mismatches=len(report['mismatches']))),flush=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    assert not report['mismatches'], 'exact graph mismatch'


if __name__ == '__main__':
    main()

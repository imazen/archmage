#!/usr/bin/env python3
"""Independent restricted-range graph identity and directed MPFR sample errors."""
import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import analyze_quality as q
p=argparse.ArgumentParser(description=__doc__);p.add_argument('actual',type=Path);p.add_argument('output',type=Path);p.add_argument('--scalar',action='store_true');a=p.parse_args()
fp=q.fp;model=q.model
r=dict(claim='sampled MPFR256 maxima, not exhaustive global guarantees',graph_mismatches=[],stats={})
labels=['MIDP_finite_fused','MIDP_normal_fused','MIDP_normal_checked','MIDP_normal_relaxed','LOWP_finite_cubic','LOWP_normal_cubic','LOWP_normal_quadratic','LOWP_normal_quadratic_checked','LOWP_normal_quadratic_relaxed']
for line in a.actual.read_text().splitlines():
    raw,*outputs=[int(v,16)for v in line.split()];assert len(outputs)==9
    x=fp.value(raw);assert -126<=x<127.5
    bounds,quantum=q.reference('E',raw)
    for i,(label,got)in enumerate(zip(labels,outputs)):
        fused=not(a.scalar and i in [3,8])
        if i>=6:
            whole=round(x);frac=fp.sub(raw,fp.bits(float(whole)))
            want=fp.mul(model.horner(frac,[1.,.701271514,.239625156],fused),(whole+127)<<23)
        else:want=q.core_mirror('E',raw,8 if i>=4 else 0,fused)
        if got!=want:r['graph_mismatches'].append([f'{raw:08x}',label,f'{got:08x}',f'{want:08x}'])
        value=fp.value(got);assert math.isfinite(value)
        error=max(abs(Fraction(value)-Fraction(v))for v in bounds)
        ulp=float(error/Fraction(2)**quantum);relative=float(error/min(abs(Fraction(v))for v in bounds))
        s=r['stats'].setdefault(label,dict(count=0,max_ulp_upper=0.,max_relative_upper=0.))
        s['count']+=1
        if ulp>s['max_ulp_upper']:s.update(max_ulp_upper=ulp,worst_ulp=f'{raw:08x}')
        if relative>s['max_relative_upper']:s.update(max_relative_upper=relative,worst_relative=f'{raw:08x}')
a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
assert not r['graph_mismatches']

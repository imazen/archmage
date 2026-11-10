#!/usr/bin/env python3
"""Sampled accuracy against directed MPFR, separate from exact graph identity."""
import argparse
import ctypes as C
from fractions import Fraction
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks/powf_module_2026-11-07"))
import accuracy
import rounding_oracle as model
fp = model.fp
mp, P = accuracy.mp, accuracy.P
for name in ["exp2", "log2"]:
    fn = getattr(mp, "mpfr_" + name)
    fn.argtypes, fn.restype = [P, P, C.c_int], C.c_int
mp.mpfr_get_exp.argtypes, mp.mpfr_get_exp.restype = [P], C.c_long
mp.mpfr_zero_p.argtypes, mp.mpfr_zero_p.restype = [P], C.c_int


def mirror(op, raw, flags, fused):
    low, legacy = bool(flags & 8), bool(flags & 64)
    if op == "E":
        x = fp.value(raw)
        reduced = min(max(x, -126.), 127.99 if low else 128.)
        whole = math.floor(reduced) if low else min(round(reduced), 127)
        frac = fp.sub(fp.bits(reduced), fp.bits(float(whole)))
        if low:
            cs = [1., math.log(2), .2402265, .05550411] if legacy else [1., math.log(2), .23433638464483292, .07191699702349297]
        else:
            cs = [1.,math.log(2),.24022646,.055504545,.009618055,.00133337,.00015447] if legacy else [1.,math.log(2),.2402264639347140,.05550369554299311,.00961871577914975,.00133860472782621,.00015252789649879]
        value = fp.mul(model.horner(frac, cs, fused), (whole + 127) << 23)
        if not low:
            if x < -126: return 0
            if x >= 128: return fp.INF
        return value
    if low and legacy:
        delta = raw - 0x3f2aaaab
        whole = delta >> 23
        m = fp.sub(raw - (whole << 23), fp.ONE)
        p = model.horner(m, [-1.8503833e-6,1.4287161,.7424587], fused)
        q = model.horner(m, [.99032814,1.0096718,.17409343], fused)
        return fp.add(fp.divide(p,q), fp.bits(float(whole)))
    adjusted = raw + fp.ONE - 0x3f3504f3
    whole = (adjusted >> 23) - 127
    a = (adjusted & 0x7fffff) + 0x3f3504f3
    y = fp.divide(fp.sub(a,fp.ONE),fp.add(a,fp.ONE))
    cs = ([2.88539,.96180076,.57697445,.43441197] if legacy else
          [2.8853900817779268,.9616492894288035,.5936509524041969] if low else
          [2.8853900817779268,.9617982234575578,.5767676021306191,.4304685970349457])
    p = model.horner(fp.mul(y,y),cs,fused)
    return model.madd(p,y,fp.bits(float(whole)),fused)


def reference(op, raw):
    a,b = accuracy.MP(),accuracy.MP()
    for v in [a,b]: mp.mpfr_init2(C.byref(v),256)
    mp.mpfr_set_d(C.byref(a),fp.value(raw),0)
    bounds, quanta = [], []
    for rounding in [3,2]:
        getattr(mp,"mpfr_" + ("exp2" if op == "E" else "log2"))(C.byref(b),C.byref(a),rounding)
        bounds.append(mp.mpfr_get_d(C.byref(b),rounding))
        quanta.append(-149 if mp.mpfr_zero_p(C.byref(b)) else max(mp.mpfr_get_exp(C.byref(b))-24,-149))
    for v in [a,b]: mp.mpfr_clear(C.byref(v))
    if bounds == [0.,0.]: return bounds, -149
    assert quanta[0] == quanta[1], (op,raw,bounds,quanta)
    return bounds,quanta[0]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("actual",type=Path);p.add_argument("output",type=Path)
    p.add_argument("--native-separate",action="store_true",help="strict/nonfusing WASM or a scalar-only automatic backend")
    args=p.parse_args();assert not args.output.exists()
    report=dict(mpfr=mp.mpfr_get_version().decode(),precision=256,claim="sampled maxima, not exhaustive bounds",graph_mismatches=[],portable_mismatches=[],token_width_mismatches=[],stats={})
    refs={}
    for line in args.actual.read_text().splitlines():
        op,raw,flags,*out=line.split();raw=int(raw,16);flags=int(flags);out=[int(x,16)for x in out]
        expected=[mirror(op,raw,flags,not bool(flags&16))]+[mirror(op,raw,flags,not bool(flags&16) or not args.native_separate)]*5
        for context,(got,want)in enumerate(zip(out,expected)):
            if got!=want:report['graph_mismatches'].append([op,f'{raw:08x}',flags,context,f'{got:08x}',f'{want:08x}'])
        if flags&16==0 and len(set(out))!=1:report['portable_mismatches'].append([op,f'{raw:08x}',flags,out])
        if len(set(out[1:]))!=1:report['token_width_mismatches'].append([op,f'{raw:08x}',flags,out])
        key=(op,raw)
        if key not in refs:refs[key]=reference(op,raw)
        bounds,q=refs[key]
        for context,got in enumerate(out[:2]):
            stats=report['stats'].setdefault(f'{op}/{flags}/'+('scalar'if context==0 else'native'),dict(count=0,max_ulp_upper=0.,max_relative_upper=0.,max_absolute_upper=0.,max_core_ulp_upper=0.,max_core_relative_upper=0.,worst=None))
            stats['count']+=1
            value=fp.value(got)
            if not all(map(math.isfinite,[value,*bounds])):continue
            error=max(abs(Fraction(value)-Fraction(v))for v in bounds)
            ulp=float(error/Fraction(2)**q)
            relative=float(error/min(abs(Fraction(v))for v in bounds))if all(bounds)else 0.
            if ulp>stats['max_ulp_upper']:stats.update(max_ulp_upper=ulp,worst=f'{raw:08x}')
            stats['max_relative_upper']=max(relative,stats['max_relative_upper'])
            stats['max_absolute_upper']=max(float(error),stats['max_absolute_upper'])
            core = op == 'L' or (-126 <= fp.value(raw) <= 127.99 if flags&8 else -126 <= fp.value(raw) < 127.5)
            if core:
                stats['max_core_ulp_upper']=max(ulp,stats['max_core_ulp_upper'])
                stats['max_core_relative_upper']=max(relative,stats['max_core_relative_upper'])
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:len(v)if isinstance(v,list)else v for k,v in report.items()if k!='stats'}))
    for k,s in report['stats'].items():print(k,s)

if __name__=="__main__":main()

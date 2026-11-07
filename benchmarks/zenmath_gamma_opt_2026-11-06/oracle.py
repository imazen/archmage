#!/usr/bin/env python3
"""Independent directed MPFR samples, including inverse images of f32 midpoints."""
import argparse
import ctypes as C
from fractions import Fraction as F
import math
import random
from pathlib import Path
from mpfr_support import (Value, lib, RN, UP, DOWN, bits32, bits64, f32,
                          fraction_interval, certify_power)
from generate import facts


def midpoint_inputs(ybits, p):
    # Midpoint of adjacent normal positive f32 outputs is exactly binary64.
    mid=(f32(ybits)+f32(ybits+1))*0.5
    reciprocal=fraction_interval(1/F(p),256)
    values=[Value(256) for _ in range(3)]
    base,lo,hi=values
    base.set_float(mid)
    # 0<mid<1: power decreases with exponent. Reverse reciprocal endpoints.
    lib.mpfr_pow(lo.ptr,base.ptr,reciprocal[-1].ptr,DOWN)
    lib.mpfr_pow(hi.ptr,base.ptr,reciprocal[-2].ptr,UP)
    start=bits32(lib.mpfr_get_flt(lo.ptr,DOWN))
    end=bits32(lib.mpfr_get_flt(hi.ptr,UP))
    for v in values+reciprocal:v.close()
    if end-start>2:raise ValueError('inverse midpoint enclosure unexpectedly wide')
    return range(max(0,start-4),min(0x7f800000,end+4)+1)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=4096)
    args=parser.parse_args()
    info=facts()
    with args.out.open('x') as out:
        out.write('kind,x_bits,p_bits,expected_bits,lo_f64_bits,hi_f64_bits,label\n')
        for k,kind in enumerate(['decode','encode']):
            start=info[kind+'_min_bits'];end=bits32(1.0);pb=info[kind+'_p_bits'];p=f32(pb)
            inputs={}
            def add(x,label):
                if 0<=x<=0xffffffff:inputs.setdefault(x,label)
            for b in [start,end]:
                for d in range(-16,17):add(b+d,'domain-edge')
            for e in range(-9,1):
                for m in [1.0,1.5]:
                    b=bits32(math.ldexp(m,e))
                    for d in range(-8,9):add(b+d,'power2-or-mantissa-split')
            for e in range(-9,1):
                for j in range(33):
                    b=bits32(math.ldexp(1+j/32,e))
                    for d in range(-8,9):add(b+d,'piecewise-center-or-edge')
            rng=random.Random(0x47414d4d41+k)
            for _ in range(args.samples):add(rng.randrange(start,end+1),'random-bits')
            for i in range(args.samples):add(start+(end-start)*i//(args.samples-1),'stratified-bits')
            low=bits32(certify_power(f32(start),p,width=32)['value'])
            for i in range(512):
                yb=low+(bits32(1.0)-1-low)*i//511
                for xb in midpoint_inputs(yb,p):add(xb,'directed-midpoint-neighbor')
            for b in [0,0x80000000,1,0x007fffff,0x00800000,0x7f7fffff,
                      0x7f800000,0xff800000,0x7fc00000,0xffc00000,0xbf800000]:add(b,'fallback-edge')
            for xb,label in sorted(inputs.items()):
                r=certify_power(f32(xb),p,precision=256,width=32)
                out.write(f'{k},{xb:08x},{pb:08x},{bits32(r["value"]):08x},{bits64(r["lower"]):016x},{bits64(r["upper"]):016x},{label}\n')
            print(f'{kind}: {len(inputs)} independent inputs')
    print('MPFR version',lib.mpfr_get_version().decode())

if __name__=='__main__':main()

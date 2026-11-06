#!/usr/bin/env python3
"""Two distinct oracles: stage-rounded f32 composition and real C0 curve with f64 constants."""
import argparse
from fractions import Fraction as F
import math
from pathlib import Path
import random
from generate import facts
from mpfr_support import (Value,lib,DOWN,UP,bits32,bits64,f32,f64,rounded_fraction,
                          certify_power,fraction_interval)

INFO=facts();L=f64(INFO['linear_cutoff_f64_bits']);S=f64(INFO['srgb_cutoff_f64_bits']);A=f64(INFO['a_f64_bits']);AP=f64(INFO['ap1_f64_bits'])
L32=f32(INFO['linear_cutoff_bits']);S32=f32(INFO['srgb_cutoff_bits']);A32=f32(INFO['a_bits']);AP32=f32(INFO['ap1_bits']);SCALE=f32(INFO['linear_scale_bits'])
def rn32(q):return rounded_fraction(q,32)
def composition(k,xb):
    x=f32(xb)
    if x<0:return 0
    if x==0:return xb # Preserve negative zero in the scalar linear branch.
    if k==0:
        if x<S32:return bits32(rn32(F(x)*F(SCALE)))
        if x<1:
            base=rn32(F(rn32(F(x)+F(A32)))/F(AP32));return bits32(certify_power(base,f32(INFO['decode_p_bits']),width=32)['value'])
    else:
        if x<L32:return bits32(rn32(F(x)*F(f32(bits32(12.92)))))
        if x<1:
            power=certify_power(x,f32(INFO['encode_p_bits']),width=32)['value'];return bits32(rn32(F(rn32(F(AP32)*F(power)))-F(A32)))
    return bits32(1.0)

def curve(k,x):
    if x<0:return 0.0,0.0
    if x==0:return x,x
    cutoff=S if k==0 else L
    if x<cutoff:
        q=F(x)/F(12.92) if k==0 else F(x)*F(12.92)
        values=fraction_interval(q);lo=lib.mpfr_get_d(values[-2].ptr,DOWN);hi=lib.mpfr_get_d(values[-1].ptr,UP)
        for v in values:v.close()
        return lo,hi
    if not x<1:return 1.0,1.0
    base=(F(x)+F(A))/F(AP) if k==0 else F(x)
    exponent=2.4 if k==0 else rounded_fraction(1/F(2.4))
    interval=fraction_interval(base);p=Value();p.set_float(exponent);lo=Value();hi=Value()
    lib.mpfr_pow(lo.ptr,interval[-2].ptr,p.ptr,DOWN);lib.mpfr_pow(hi.ptr,interval[-1].ptr,p.ptr,UP)
    extra=[]
    if k==1:
        scale=Value();offset=Value();scale.set_float(AP);offset.set_float(A);extra=[scale,offset]
        lib.mpfr_mul(lo.ptr,lo.ptr,scale.ptr,DOWN);lib.mpfr_mul(hi.ptr,hi.ptr,scale.ptr,UP)
        lib.mpfr_sub(lo.ptr,lo.ptr,offset.ptr,DOWN);lib.mpfr_sub(hi.ptr,hi.ptr,offset.ptr,UP)
    result=(lib.mpfr_get_d(lo.ptr,DOWN),lib.mpfr_get_d(hi.ptr,UP))
    for v in interval+[p,lo,hi]+extra:v.close()
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);args=p.parse_args()
    with args.out.open('x') as out:
        out.write('kind,x_bits,expected_f32_bits,curve_lo_f64_bits,curve_hi_f64_bits,label\n')
        for k in [0,1]:
            inputs={0:'zero',0x80000000:'negative-zero',1:'subnormal',0x007fffff:'max-subnormal',0xbf800000:'negative',0x7f800000:'inf',0xff800000:'negative-inf',0x7fc00000:'nan'}
            for edge in [bits32(S32 if k==0 else L32),bits32(1.0)]:
                for d in range(-32,33):inputs[edge+d]='C0-edge'
            rng=random.Random(0x433047414d4d41+k)
            for i in range(4096):
                inputs[bits32(i/4095)]='value-uniform';inputs[rng.randrange(0,bits32(1.0)+1)]='bit-random'
            for xb,label in sorted(inputs.items()):
                lo,hi=curve(k,f32(xb));expected=composition(k,xb)
                out.write(f'{k},{xb:08x},{expected:08x},{bits64(lo):016x},{bits64(hi):016x},{label}\n')
            print(k,len(inputs),'C0 samples, f32 composition and separate f64-constant MPFR curve')

if __name__=='__main__':main()

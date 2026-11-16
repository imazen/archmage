#!/usr/bin/env python3
"""Independent canonical powf oracle: integer bits and exact-rational IEEE rounding.

No Rust/C powf or host floating arithmetic supplies graph answers. Constants
are the independently frozen C-extracted TOINT=0 table; every binary64 node
rounds through fma_vectors.expected, and the final binary32 cast rounds exactly.
Mathematical accuracy versus MPFR is a separate experiment.
"""
import argparse
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
import random
import fma_vectors as ieee

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'magetypes/tests/common/highp_vectors.rs'
DATA={k:[]for k in 'LATCSOB'}
for line in Path(__file__).with_name('highp_coefficients.txt').read_text().splitlines():
    if line.startswith('#'):continue
    key,*values=line.split();DATA[key].append(tuple(int(v,16)for v in values))
assert [len(DATA[k])for k in 'LATCSOB']==[16,5,32,3,1,1,1]
ONE64=0x3ff0000000000000
NAN32=0x7fc00000


def decode(bits,p,e):
    bias=(1<<(e-1))-1;field=(bits>>p)&((1<<e)-1)
    assert field != (1<<e)-1
    q=Fraction((bits&((1<<p)-1))+((1<<p)if field else 0))*ieee.power((field or 1)-bias-p)
    return -q if bits>>(p+e)else q


def encode(q,p,e,negative_zero=False):
    sign=(1<<(p+e))if q<0 or(q==0 and negative_zero)else 0
    if not q:return sign
    q=abs(q);bias=(1<<(e-1))-1
    exponent=q.numerator.bit_length()-q.denominator.bit_length()
    if q<ieee.power(exponent):exponent-=1
    quantum=max(exponent,1-bias)-p
    scaled=q/ieee.power(quantum);n,r=divmod(scaled.numerator,scaled.denominator)
    n+=2*r>scaled.denominator or(2*r==scaled.denominator and n&1)
    if n>=1<<(p+1):n>>=1;quantum+=1
    exponent=quantum+p
    if exponent>bias:return sign|(((1<<e)-1)<<p)
    if n<1<<p:return sign|n
    return sign|((exponent+bias)<<p)|(n-(1<<p))


@lru_cache(maxsize=131072)
def fma(a,b,c):return ieee.expected((a,b,c),52,11)
def add(a,b):return fma(a,ONE64,b)
def sub(a,b):return add(a,b^(1<<63))
def mul(a,b):return fma(a,b,(a^b)&(1<<63))
def as64(b):return encode(decode(b,23,8),52,11, bool(b>>31))
def constant(n):return encode(Fraction(n),52,11)
def special(b):return (((2*b)&0xffffffff)-1)&0xffffffff>=0xfeffffff

def integer_kind(b):
    e=(b>>23)&255
    if e<127:return 0
    if e>150:return 2
    if b&((1<<(150-e))-1):return 0
    return 1 if b&(1<<(150-e))else 2

def signaling(b):return (b&0x7fffffff)>0x7f800000 and b&0x400000==0


def canonical(x,n):
    ix=x;iy=n;sign=0
    if ix-0x00800000&0xffffffff>=0x7f000000 or special(iy):
        if special(iy):
            if iy&0x7fffffff==0:return NAN32 if signaling(ix)else 0x3f800000
            if ix==0x3f800000:return NAN32 if signaling(iy)else 0x3f800000
            if ix&0x7fffffff>0x7f800000 or iy&0x7fffffff>0x7f800000:return NAN32
            if ix&0x7fffffff==0x3f800000:return 0x3f800000
            return 0 if ((ix&0x7fffffff)<0x3f800000)==(iy>>31==0)else 0x7f800000
        if special(ix):
            if ix&0x7fffffff>0x7f800000:return NAN32
            negative=bool(ix>>31 and integer_kind(iy)==1)
            inf=(ix&0x7fffffff)==0x7f800000
            result=0x7f800000 if inf!=(iy>>31!=0)else 0
            return result|(0x80000000 if negative else 0)
        if ix>>31:
            kind=integer_kind(iy)
            if not kind:return NAN32
            if kind==1:sign=1<<16
            ix&=0x7fffffff
        if ix<0x800000:
            norm=ieee.expected((x,0x4b000000,x&0x80000000),23,8)&0x7fffffff
            ix=(norm-(23<<23))&0xffffffff
    tmp=(ix-0x3f330000)&0xffffffff;index=(tmp>>19)&15;top=tmp&0xff800000
    z=as64((ix-top)&0xffffffff);k=(top-(1<<32)if top>>31 else top)>>23
    invc,logc=DATA['L'][index]
    r=fma(z,invc,ONE64|(1<<63));y0=add(logc,constant(k));r2=mul(r,r)
    a=[v[0]for v in DATA['A']]
    h=fma(a[0],r,a[1]);p=fma(a[2],r,a[3]);r4=mul(r2,r2)
    q=fma(a[4],r,y0);q=fma(p,r2,q);logx=fma(h,r4,q)
    t=mul(as64(iy),logx);tv=decode(t,52,11)
    if abs(tv)>=126:
        sb=0x80000000 if sign else 0
        if tv<=-150:return sb
        if tv<-149:return sb|1
        if tv>decode(DATA['O'][0][0],52,11):return sb|0x7f800000
        if tv>decode(DATA['B'][0][0],52,11):return sb|0x7f7fffff
    shift=DATA['S'][0][0];kd=add(t,shift);ki=kd;kd=sub(kd,shift);r=sub(t,kd)
    scale=(DATA['T'][ki&31][0]+(((ki+sign)&((1<<64)-1))<<47))&((1<<64)-1)
    c=[v[0]for v in DATA['C']]
    z=fma(c[0],r,c[1]);r2=mul(r,r);tail=fma(c[2],r,ONE64)
    value=mul(fma(z,r2,tail),scale)
    return encode(decode(value,52,11),23,8,bool(value>>63))


def cases():
    edge=[0,0x80000000,1,0x80000001,0x007fffff,0x00800000,0x3f7fffff,0x3f800000,0x3f800001,0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,0x7fc00001,0x7f800001,0xbf800000,0xc0000000]
    exp=[0,0x80000000,1,0x3f000000,0x40000000,0x40400000,0xbf800000,0x7f800000,0x7fc00001,0x7f800001]
    rows=[(x,n)for x in edge for n in exp]
    rows += [(0x3f7ffff6,0xcd000001),(0x3fd2ae44,0x4019999a)]
    rng=random.Random(0x4849474850)
    rows +=[(rng.randrange(0x00800000,0x7f800000),rng.getrandbits(32))for _ in range(84)]
    rows += [(rng.randrange(0x37800000,0x47800001),rng.choice([0x3f000000,0x4019999a,0x40800000]))for _ in range(128)]
    rows += [(rng.randrange(0x00800000,0x7f800000),rng.choice([0x3f000000,0x4019999a,0xc019999a]))for _ in range(128)]
    return rows

def render():
    rows=cases();s='// Generated by xtask/highp_powf_vectors.py; exact-rational binary64 graph.\n// NaN payload/sign are unspecified. This is not a mathematical pow oracle.\n#[rustfmt::skip]\npub const CASES: &[[u32;3]] = &[\n'
    for x,n in rows:s+=f'    [0x{x:08x},0x{n:08x},0x{canonical(x,n):08x}],\n'
    return s+'];\n'

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');p.add_argument('--write',action='store_true');p.add_argument('--inputs',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    assert sum([a.check,a.write,a.inputs is not None])==1
    if a.inputs:
        assert a.output and not a.output.exists()
        with a.output.open('w')as out:
            for line in a.inputs.read_text().splitlines():
                x,n=map(lambda s:int(s,16),line.split()[:2]);out.write(f'{x:08x} {n:08x} {canonical(x,n):08x}\n')
    elif a.check:assert OUTPUT.read_text()==render(),'canonical HIGHP fixtures differ; review before regeneration'
    else:OUTPUT.write_text(render())

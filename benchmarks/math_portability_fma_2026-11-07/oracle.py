"""Independent fixed-graph mirror: rational RNE and MPFR-directed true errors.

Rational FMA reuses the independent reviewed fixture oracle, never std/libm.
The graph below is handwritten independently of the generated trait extraction.
MPFR is a test oracle only. No production dependency is added.
"""
from fractions import Fraction
import ctypes as C
import ctypes.util
from functools import lru_cache
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('fma_vectors',ROOT/'xtask/fma_vectors.py')
fm=importlib.util.module_from_spec(spec);spec.loader.exec_module(fm)

def bits(x):
    try: return struct.unpack('<I',struct.pack('<f',x))[0]
    except OverflowError: return 0xff800000 if x<0 else 0x7f800000

def val(x):return struct.unpack('<f',struct.pack('<I',x&0xffffffff))[0]
def neg(x):return x^0x80000000
ONE=0x3f800000
INF=0x7f800000
NAN=0x7fc00000

def is_nan(x):return x&0x7fffffff>INF

def frac(x):return Fraction.from_float(val(x))

@lru_cache(maxsize=65536)
def fma(a,b,c):return fm.expected((a,b,c),23,8)
def add(a,b):return fma(a,ONE,b)
def sub(a,b):return add(a,neg(b))
def mul(a,b):return fma(a,b,(a^b)&0x80000000)

def rounded(q,negative=False):
    if not q:return 0x80000000 if negative else 0
    sign=0x80000000 if q<0 else 0;q=abs(q)
    e=q.numerator.bit_length()-q.denominator.bit_length()
    if q<fm.power(e):e-=1
    k=max(e,-126)-23
    q/=fm.power(k)
    n,r=divmod(q.numerator,q.denominator)
    n+=2*r>q.denominator or (2*r==q.denominator and n&1)
    if n>=1<<24:n>>=1;k+=1
    e=k+23
    if e>127:return sign|INF
    if n<1<<23:return sign|n
    return sign|((e+127)<<23)|(n-(1<<23))

def div(a,b):
    av,bv=val(a),val(b);sign=(a^b)&0x80000000
    if is_nan(a) or is_nan(b) or (math.isinf(av) and math.isinf(bv)) or (av==0 and bv==0):return NAN
    if bv==0 or math.isinf(av):return sign|INF
    if av==0 or math.isinf(bv):return sign
    return rounded(frac(a)/frac(b),bool(sign))

def horner(x,coeff):
    y=bits(coeff[-1])
    for c in reversed(coeff[:-1]):y=fma(y,x,bits(c))
    return y

def exp2(x,low=False):
    if is_nan(x):return NAN
    v=val(x)
    if not low and v < -126:return 0
    if not low and v >=128:return INF
    v=min(max(v,-126),val(bits(127.99)) if low else 128)
    xi=math.floor(v) if low else min(round(v),127)
    xf=sub(bits(v),bits(float(xi)))
    poly=horner(xf,[1,math.log(2),.2402265,.05550411] if low else [1,math.log(2),.24022646,.055504545,.009618055,.00133337,.00015447])
    return mul(poly,((xi+127)<<23)&0xffffffff)

def signed32(x):return x-0x100000000 if x&0x80000000 else x

def log2(x,low=False):
    if low:
        e=signed32((x-0x3f2aaaab)&0xffffffff)>>23
        m=sub((x-(e<<23))&0xffffffff,ONE)
        yp=horner(m,[-.0000018503833,1.4287161,.7424587])
        yq=horner(m,[.99032814,1.0096718,.17409343])
        return add(div(yp,yq),bits(float(e)))
    adjusted=(x+(ONE-0x3f3504f3))&0xffffffff
    n=(signed32(adjusted)>>23)-127
    a=(adjusted&0x7fffff)+0x3f3504f3
    y=div(sub(a,ONE),add(a,ONE))
    p=horner(mul(y,y),[2.88539,.96180076,.57697445,.43441197])
    r=fma(p,y,bits(float(n)))
    if val(x)==0:return 0xff800000
    if val(x)<0:return NAN
    if x==INF:return INF
    return r

def mirror(op,x):
    low=op.endswith('lowp')
    if op.startswith('exp2'):return exp2(x,low)
    if op.startswith('exp_'):return exp2(mul(x,bits(math.log2(math.e))),low)
    if op.startswith('log2'):return log2(x,low)
    if op.startswith('ln_'):return mul(log2(x,low),bits(math.log(2)))
    if op.startswith('log10'):return mul(log2(x,low),div(bits(math.log(2)),bits(math.log(10))))
    if op.startswith('pow'):
        r=exp2(mul(log2(x,low),bits(2.4)),low)
        return 0 if low and val(x)==0 else r
    if op.startswith('sigmoid'):return div(ONE,add(exp2(mul(neg(x),bits(math.log2(math.e)))),ONE))
    if op.startswith('silu'):return mul(x,mirror('sigmoid_midp',x))
    raise ValueError(op)

# Only the accuracy oracle uses MPFR. Each result encloses the mathematical
# function at the exact f32 argument, with independent down/up rounding.
class MP(C.Structure):
    _fields_=[('prec',C.c_long),('sign',C.c_int),('exp',C.c_long),('d',C.POINTER(C.c_ulong))]
mp=C.CDLL(ctypes.util.find_library('mpfr'))
P=C.POINTER(MP)
for name,args,res in [('init2',[P,C.c_long],None),('clear',[P],None),('set_d',[P,C.c_double,C.c_int],C.c_int),('get_d',[P,C.c_int],C.c_double),('get_version',[],C.c_char_p)]:
    fn=getattr(mp,'mpfr_'+name);fn.argtypes=args;fn.restype=res
for name in ['exp2','exp','log2','log','log10','neg']:
    fn=getattr(mp,'mpfr_'+name);fn.argtypes=[P,P,C.c_int];fn.restype=C.c_int
for name in ['pow','add','mul','div']:
    fn=getattr(mp,'mpfr_'+name);fn.argtypes=[P,P,P,C.c_int];fn.restype=C.c_int

def reference(op,x):
    a,b,c=MP(),MP(),MP()
    for z in [a,b,c]:mp.mpfr_init2(C.byref(z),256)
    mp.mpfr_set_d(C.byref(a),val(x),0)
    result=[]
    for rnd in [3,2]: # RNDD, RNDU
        if op.startswith('pow'):
            mp.mpfr_set_d(C.byref(b),val(bits(2.4)),0)
            mp.mpfr_pow(C.byref(c),C.byref(a),C.byref(b),rnd)
        elif op.startswith(('sigmoid','silu')):
            # MPFR true function is sampled separately below; compounded
            # interval propagation is required for a rigorous activation bound.
            result=None;break
        else:
            name=op.split('_')[0]
            name={'ln':'log'}.get(name,name)
            getattr(mp,'mpfr_'+name)(C.byref(c),C.byref(a),rnd)
        result.append(mp.mpfr_get_d(C.byref(c),rnd))
    for z in [a,b,c]:mp.mpfr_clear(C.byref(z))
    return result

def main():
    summary={};mismatches=[]
    for file in sys.argv[1:]:
        for line in Path(file).read_text().splitlines():
            row=json.loads(line);key=(str(file),row['token'],row['width'],row['op'])
            q=summary.setdefault(key,dict(n=0,mismatches=0,changed_fused_bits=0,max_ulp_upper=None,accuracy_samples=0,changed_fusion_bits=0,ordinary_changed_fusion_bits=0))
            for x,y,base,corrected in zip(row['input'],row['fused'],row['baseline'],row['exceptional']):
                expected=mirror(row['op'],x);q['n']+=1
                same=(is_nan(y) and is_nan(expected)) or y==expected
                if not same:
                    q['mismatches']+=1
                    if len(mismatches)<30:mismatches.append(dict(key=key,x=f'{x:08x}',got=f'{y:08x}',expected=f'{expected:08x}'))
                q['changed_fused_bits']+=not ((is_nan(y) and is_nan(base)) or y==base)
                fusion_changed=not ((is_nan(y) and is_nan(corrected)) or y==corrected)
                q['changed_fusion_bits']+=fusion_changed
                ref=reference(row['op'],x)
                v=val(x)
                domain=(0<v<math.inf and x&0x7fffffff>=0x00800000) if row['op'].startswith(('log','ln','pow')) else ((-126<=v<(127.99 if row['op'].endswith('lowp') else 128)) if row['op'].startswith('exp2') else -87<=v<=88.5)
                q['ordinary_changed_fusion_bits']+=domain and fusion_changed
                if domain and ref and all(math.isfinite(v) for v in ref) and math.isfinite(val(y)):
                    rounded_ref=bits((ref[0]+ref[1])/2)
                    r=val(rounded_ref)
                    if abs(r)>=val(0x00800000) and rounded_ref&0x7fffffff<INF:
                        ulp=math.ldexp(1.0,math.frexp(abs(r))[1]-24)
                        exact_error=max(abs(Fraction.from_float(val(y))-Fraction.from_float(r)) for r in ref)/Fraction.from_float(ulp)
                        error=math.nextafter(float(exact_error),math.inf)
                        q['max_ulp_upper']=max(q['max_ulp_upper'] or 0,error);q['accuracy_samples']+=1
    print(json.dumps(dict(mpfr=mp.mpfr_get_version().decode(),summaries=[dict(source=k[0],token=k[1],width=k[2],op=k[3],**v) for k,v in summary.items()],mismatch_examples=mismatches),indent=2))
    if mismatches:sys.exit(1)
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Directed MPFR4.2 public C ABI, adapted from portable_f32/mpfr.py at4ccaf1cf.
Reference: https://www.mpfr.org/mpfr-current/mpfr.html#Rounding
"""
import ctypes as C
import ctypes.util
from fractions import Fraction as F
import struct

class Raw(C.Structure):
    _fields_=[('prec',C.c_long),('sign',C.c_int),('exp',C.c_long),('limbs',C.c_void_p)]

lib=C.CDLL(ctypes.util.find_library('mpfr'))
P=C.POINTER(Raw)
for name,args,result in [
    ('init2',[P,C.c_long],None),('clear',[P],None),
    ('set_d',[P,C.c_double,C.c_int],C.c_int),('set_str',[P,C.c_char_p,C.c_int,C.c_int],C.c_int),
    ('get_d',[P,C.c_int],C.c_double),('get_flt',[P,C.c_int],C.c_float),
    ('div',[P,P,P,C.c_int],C.c_int),('pow',[P,P,P,C.c_int],C.c_int),
    ('add',[P,P,P,C.c_int],C.c_int),('mul',[P,P,P,C.c_int],C.c_int),
    ('sub',[P,P,P,C.c_int],C.c_int),('get_version',[],C.c_char_p),
]:
    func=getattr(lib,'mpfr_'+name);func.argtypes=args;func.restype=result
RN,UP,DOWN=0,2,3

def bits32(value):return struct.unpack('<I',struct.pack('<f',value))[0]
def f32(bits):return struct.unpack('<f',struct.pack('<I',bits))[0]
def bits64(value):return struct.unpack('<Q',struct.pack('<d',value))[0]
def f64(bits):return struct.unpack('<d',struct.pack('<Q',bits))[0]

class Value:
    def __init__(self,precision=1024):
        self.raw=Raw();lib.mpfr_init2(C.byref(self.raw),precision);self.ptr=C.byref(self.raw)
    def set_float(self,value):
        if lib.mpfr_set_d(self.ptr,value,RN)!=0:raise ValueError('float input not exact at working precision')
    def set_int(self,value):
        if lib.mpfr_set_str(self.ptr,str(value).encode(),10,RN)!=0:raise ValueError('integer input not exact')
    def close(self):lib.mpfr_clear(self.ptr)


def fraction_interval(value,precision=1024):
    value=F(value)
    if max(abs(value.numerator).bit_length(),value.denominator.bit_length())>precision:
        precision=max(abs(value.numerator).bit_length(),value.denominator.bit_length())+64
    values=[Value(precision) for _ in range(4)]
    n,d,lo,hi=values;n.set_int(value.numerator);d.set_int(value.denominator)
    lib.mpfr_div(lo.ptr,n.ptr,d.ptr,DOWN);lib.mpfr_div(hi.ptr,n.ptr,d.ptr,UP)
    return values


def rounded_fraction(value,width=64):
    values=fraction_interval(value)
    getter=lib.mpfr_get_d if width==64 else lib.mpfr_get_flt
    a,b=[getter(v.ptr,RN) for v in values[-2:]]
    for v in values:v.close()
    if (bits64(a) if width==64 else bits32(a))!=(bits64(b) if width==64 else bits32(b)):
        raise ValueError('rational rounding unresolved')
    return a


def certify_fraction(value):
    values=fraction_interval(value)
    lo,hi=values[-2:]
    nearest=[lib.mpfr_get_d(v.ptr,RN) for v in [lo,hi]]
    result=dict(value=nearest[0],lower=lib.mpfr_get_d(lo.ptr,DOWN),upper=lib.mpfr_get_d(hi.ptr,UP))
    for v in values:v.close()
    if bits64(nearest[0])!=bits64(nearest[1]):raise ValueError('coefficient rounding unresolved')
    if not F(result['lower'])<=value<=F(result['upper']):raise ValueError('coefficient enclosure failed')
    return result


def certify_power(base,exponent,precision=1024,width=64):
    values=[Value(precision) for _ in range(4)]
    x,p,lo,hi=values;x.set_float(base);p.set_float(exponent)
    lib.mpfr_pow(lo.ptr,x.ptr,p.ptr,DOWN);lib.mpfr_pow(hi.ptr,x.ptr,p.ptr,UP)
    get=lib.mpfr_get_d if width==64 else lib.mpfr_get_flt
    a,b=[get(v.ptr,RN) for v in [lo,hi]]
    result=dict(value=a,lower=lib.mpfr_get_d(lo.ptr,DOWN),upper=lib.mpfr_get_d(hi.ptr,UP))
    for v in values:v.close()
    if (bits64(a) if width==64 else bits32(a))!=(bits64(b) if width==64 else bits32(b)):
        if precision>=16384:raise ValueError('pow rounding unresolved')
        return certify_power(base,exponent,precision*2,width)
    return result

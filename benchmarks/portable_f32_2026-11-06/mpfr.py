#!/usr/bin/env python3
"""Independent directed MPFR binary32 oracle, via its public C ABI."""
import ctypes as C
import ctypes.util
import math
import struct

class Value(C.Structure):
    _fields_ = [('prec', C.c_long), ('sign', C.c_int), ('exp', C.c_long), ('limbs', C.c_void_p)]

lib = C.CDLL(ctypes.util.find_library('mpfr'))
P = C.POINTER(Value)
for name, args, result in [
    ('init2', [P, C.c_long], None), ('set_d', [P, C.c_double, C.c_int], C.c_int),
    ('set_prec', [P, C.c_long], None), ('get_d', [P, C.c_int], C.c_double),
    ('get_flt', [P, C.c_int], C.c_float), ('get_exp', [P], C.c_long),
    ('exp', [P, P, C.c_int], C.c_int), ('log', [P, P, C.c_int], C.c_int),
    ('log2', [P, P, C.c_int], C.c_int), ('pow', [P, P, P, C.c_int], C.c_int),
    ('get_version', [], C.c_char_p),
]:
    f = getattr(lib, 'mpfr_' + name)
    f.argtypes, f.restype = args, result

def flt(bits):
    return struct.unpack('<f', struct.pack('<I', bits))[0]

def bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]

def isnan(b):
    return (b & 0x7fffffff) > 0x7f800000

def same(a, b):
    return a == b or (isnan(a) and isnan(b))

def ordered(b):
    return (~b & 0xffffffff) if b & 0x80000000 else b | 0x80000000

class Oracle:
    def __init__(self):
        self.x, self.y, self.lo, self.hi = [Value() for _ in range(4)]
        for v in [self.x, self.y, self.lo, self.hi]:
            lib.mpfr_init2(C.byref(v), 128)

    def reference(self, op, xb, yb):
        lib.mpfr_set_d(C.byref(self.x), flt(xb), 0)
        lib.mpfr_set_d(C.byref(self.y), flt(yb), 0)
        f = getattr(lib, 'mpfr_' + ('log' if op == 'ln' else op))
        precision = 128
        while True:
            for v, rnd in [(self.lo, 3), (self.hi, 2)]:
                lib.mpfr_set_prec(C.byref(v), precision)
                args = [C.byref(v), C.byref(self.x)]
                if op == 'pow':
                    args.append(C.byref(self.y))
                f(*args, rnd)
            lower = bits(lib.mpfr_get_flt(C.byref(self.lo), 0))
            upper = bits(lib.mpfr_get_flt(C.byref(self.hi), 0))
            if same(lower, upper):
                break
            precision *= 2
            if precision > 16384:
                raise RuntimeError((op, xb, yb, 'unresolved rounding'))
        # Preserve the existing CORE-MATH signaling-NaN convention for pow.
        if op == 'pow' and any(isnan(b) and not b & 0x400000 for b in [xb, yb]):
            lower = 0x7fc00000
        lo = lib.mpfr_get_d(C.byref(self.lo), 3)
        hi = lib.mpfr_get_d(C.byref(self.hi), 2)
        exponent = lib.mpfr_get_exp(C.byref(self.lo)) if lo and math.isfinite(lo) else -125
        return lower, lo, hi, max(-149, exponent - 24)

def true_error(out, lo, hi, exponent):
    actual = flt(out)
    if not math.isfinite(actual) or not math.isfinite(lo) or not math.isfinite(hi):
        return None
    ulp = math.ldexp(1.0, exponent)
    # Each f64 endpoint subtraction and power-of-two division is widened
    # explicitly. This remains conservative for inaccurate approximations;
    # it does not rely on Sterbenz cancellation for near-correct candidates.
    distances = [abs(actual-lo), abs(actual-hi)]
    numerator_upper = math.nextafter(max(distances), math.inf)
    upper = math.nextafter(numerator_upper / ulp, math.inf)
    if lo <= actual <= hi:
        lower = 0.0
    else:
        numerator_lower = max(0.0, math.nextafter(min(distances), -math.inf))
        lower = max(0.0, math.nextafter(numerator_lower / ulp, -math.inf))
    return lower, upper

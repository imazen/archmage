#!/usr/bin/env python3
"""Directed MPFR rounding gates plus explicitly observed sampled error estimates."""
import csv
import ctypes as C
import ctypes.util
import math
from pathlib import Path
import struct
import sys

class V(C.Structure):
    _fields_ = [('p', C.c_long), ('sign', C.c_int), ('exp', C.c_long), ('limbs', C.c_void_p)]
lib = C.CDLL(ctypes.util.find_library('mpfr'))
P = C.POINTER(V)
def bind(name, args, result=C.c_int):
    f = getattr(lib, 'mpfr_' + name)
    f.argtypes, f.restype = args, result
    return f
init = bind('init2', [P, C.c_long], None)
setd = bind('set_d', [P, C.c_double, C.c_int])
getd = bind('get_d', [P, C.c_int], C.c_double)
getf = bind('get_flt', [P, C.c_int], C.c_float)
pow_ = bind('pow', [P, P, P, C.c_int])
for op in ['add', 'sub', 'mul', 'div']:
    bind(op, [P, P, P, C.c_int])
a, b, lo, hi, c, t = [V() for _ in range(6)]
for v in [a, b, lo, hi, c, t]:
    init(C.byref(v), 256)
ref = C.byref

def bits(v):
    return int.from_bytes(struct.pack('<f', v), 'little')
def f32(v):
    return struct.unpack('<f', struct.pack('<f', v))[0]
def equivalent(x, y):
    return bits(x) == bits(y) or math.isnan(x) and math.isnan(y)
def ordered(x):
    v = bits(x)
    return (~v & 0xffffffff) if v >> 31 else v | 0x80000000
cache = {}
def oracle(x, y):
    # Raw bits preserve signed zero; all arguments are f32 inputs.
    key = ('f32', bits(x), bits(y))
    if key not in cache:
        setd(ref(a), x, 0)
        setd(ref(b), y, 0)
        pow_(ref(lo), ref(a), ref(b), 3)  # RNDD
        pow_(ref(hi), ref(a), ref(b), 2)  # RNDU
        p, q = getf(ref(lo), 0), getf(ref(hi), 0)
        assert equivalent(p, q), ('MPFR endpoints unresolved', key, p, q)
        cache[key] = (p, getd(ref(lo), 0))
    return cache[key]
transfer_cache = {}
def transfer_oracle(domain, value):
    # Separate high-precision full-transfer reference. Parameters and exponents
    # are the exact f64 constants of linear-srgb 0.6.12, not f32 power inputs.
    key = (domain, bits(value))
    if key not in transfer_cache:
        cutoff = 0.003041282560127521
        threshold = 12.92 * cutoff if domain == 'gamma-decode' else cutoff
        setd(ref(a), value, 0)
        if value < threshold:
            setd(ref(b), 1.0 / 12.92 if domain == 'gamma-decode' else 12.92, 0)
            lib.mpfr_mul(ref(t), ref(a), ref(b), 0)
        elif value < 1.0:
            if domain == 'gamma-decode':
                setd(ref(b), 0.0550107189475866, 0)
                lib.mpfr_add(ref(a), ref(a), ref(b), 0)
                setd(ref(b), 1.0550107189475866, 0)
                lib.mpfr_div(ref(a), ref(a), ref(b), 0)
            setd(ref(b), 2.4 if domain == 'gamma-decode' else 1.0 / 2.4, 0)
            pow_(ref(t), ref(a), ref(b), 0)
            if domain == 'gamma-encode':
                setd(ref(b), 1.0550107189475866, 0)
                lib.mpfr_mul(ref(t), ref(t), ref(b), 0)
                setd(ref(b), 0.0550107189475866, 0)
                lib.mpfr_sub(ref(t), ref(t), ref(b), 0)
        else:
            setd(ref(t), 1.0, 0)
        transfer_cache[key] = getd(ref(t), 0)
    return transfer_cache[key]

def ulp_at(x):
    # Subnormals and zero use 2^-149; normals use binade spacing at magnitude x.
    return 2.0 ** -149 if abs(x) < 2.0 ** -126 else 2.0 ** (math.frexp(abs(x))[1] - 24)
def quantize(x, maximum):
    return math.floor(min(max(x, 0.0), 1.0) * maximum + 0.5)
root = Path(sys.argv[1])
summary = []
for kind, domain, distribution, candidate, policy, composition, file in csv.reader((root / 'inventory.tsv').open(), delimiter='\t'):
    rows = list(struct.iter_unpack('<fff' if kind == 'pow' else '<ffff', (root / file).read_bytes()))
    rounded_equal, finite, max_steps, max_abs, max_ulp = 0, 0, 0, 0.0, 0.0
    q8 = q16 = 0
    for row in rows:
        if kind == 'pow':
            x, y, actual = row
            rounded, value = oracle(x, y)
            rounded_equal += equivalent(actual, rounded)
            if candidate.endswith('/correct') and not candidate.startswith('std'):
                assert equivalent(actual, rounded), (candidate, hex(bits(x)), hex(bits(y)), hex(bits(actual)), hex(bits(rounded)))
        else:
            v, x, y, actual = row
            value = transfer_oracle(domain, v)
            rounded = f32(value)
            rounded_equal += equivalent(actual, rounded)
            q8 += quantize(actual, 255) != quantize(value, 255)
            q16 += quantize(actual, 65535) != quantize(value, 65535)
        if math.isfinite(actual) and math.isfinite(value) and math.isfinite(rounded):
            finite += 1
            max_steps = max(max_steps, abs(ordered(actual) - ordered(rounded)))
            error = abs(actual - value)
            max_abs = max(max_abs, error)
            max_ulp = max(max_ulp, error / ulp_at(rounded))
    summary.append(dict(kind=kind, domain=domain, distribution=distribution, candidate=candidate, policy=policy, composition=composition,
        samples=len(rows), finite=finite, rounded_equal=rounded_equal, max_representable_steps=max_steps,
        max_abs_observed_f64_estimate=max_abs, max_ulp_observed_f64_estimate=max_ulp, quantized8_differences=q8, quantized16_differences=q16))
with (root / 'accuracy.csv').open('w') as out:
    w = csv.DictWriter(out, fieldnames=list(summary[0]))
    w.writeheader()
    w.writerows(summary)
print(f'PASS {len(summary)} cells. Correct pow directed endpoint gate separate from observed estimates; no global proof.')

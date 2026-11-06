#!/usr/bin/env python3
"""Derive exact negative-integer sign cases from independent positive MPFR data.
RNE rounding commutes with sign. Python integer parity is independent of the Rust
bit classifier; preserve all original records and append only valid finite pairs.
"""
from pathlib import Path
import math,struct,sys
source,out=map(Path,sys.argv[1:]);data=source.read_bytes();result=bytearray(data);count=0
flt=lambda b:struct.unpack('<f',struct.pack('<I',b))[0]
for x,y,z in struct.iter_unpack('<III',data):
    a,b=flt(x),flt(y)
    if a>=0 and not x&0x80000000 and math.isfinite(a) and math.isfinite(b) and b.is_integer() and not math.isnan(flt(z)):
        sign=0x80000000 if int(b)&1 else 0
        result.extend(struct.pack('<III',x|0x80000000,y,z|sign));count+=1
out.write_bytes(result);print('original',len(data)//12,'derived negative integer',count,'total',len(result)//12)

#!/usr/bin/env python3
"""Append directed-MPFR boundary cases to the existing common corpus."""
import math
from pathlib import Path
import struct
import sys
from mpfr import Oracle,bits,lib
root=Path(sys.argv[1]); oracle=Oracle()
edges=[0,0x80000000,1,0x80000001,0x7fffff,0x800000,0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,0x7fc00000,0xffc00000,0x7f800001,0xff800001]
for op in ['exp','ln','log2','pow']:
    cases=set()
    if op in ['ln','log2','pow']:
        centers=[k<<23 for k in range(1,255)]+[0x3f800000,0x3f000000,0x40000000,0x800000,0x7fffff,1,0x7f7fffff]
    else:
        centers=[bits(float(k)*math.log(2)/16) for k in range(-2400,2065)]+[bits(-103.97208),bits(-87.33655),bits(88.72284)]
    xs=set(edges)
    for b in centers:
        xs.update(v for d in range(-8,9) if 0 <= (v:=b+d) <= 0xffffffff)
    if op=='pow':
        ys=[bits(v) for v in [-16777216,-150,-149,-128,-2.4,-2,-1,-0.5,0,0.5,1,2,2.4,127,128,149,16777216]]+edges
        for x in xs:
            for y in ys: cases.add((x,y))
    else:
        cases={(x,0) for x in xs}
    path=root/f'{op}.bin'
    original=path.read_bytes()
    saved=path.with_suffix('.original.bin')
    if saved.exists():assert original==saved.read_bytes(),'do not append twice'
    else:saved.write_bytes(original)
    with path.open('ab') as f:
        for x,y in sorted(cases):
            r,_,_,_=oracle.reference(op,x,y)
            f.write(struct.pack('<III',x,y,r))
    print(op,'appended',len(cases),'total',path.stat().st_size//12,flush=True)
print('MPFR',lib.mpfr_get_version().decode())

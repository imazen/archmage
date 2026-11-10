#!/usr/bin/env python3
"""Deterministic structured/random unary inputs, padded to mixed-width groups."""
import argparse
from pathlib import Path
import random
import struct

p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
def bits(x):return struct.unpack('<I',struct.pack('<f',x))[0]
def write(name,kind,xs):
    xs += [xs[-1]]*((-len(xs))%16)
    (a.output/name).write_text(''.join(f'{kind} {x:08x}\n'for x in xs))
rng=random.Random(147);base=[]
for log in [False,True]:
    xs=([0x00800000,0x00800001,0x3f7fffff,0x3f800000,0x3f800001,0x7f7fffff]if log else[bits(x)for x in [-126.,-125.99999,-.5,0.,.5,127.,127.49999,127.5,127.99,128.]])
    for _ in range(2048):xs.append(rng.randrange(0x00800000,0x7f800000)if log else bits(rng.uniform(-126.,127.99999)))
    for _ in range(1024):xs.append(rng.randrange(0x3f3504f3,0x3fb504f3)if log else bits(rng.uniform(-.5,.5)))
    xs += [xs[-1]]*((-len(xs))%16)
    base += [('L'if log else'E')+f' {x:08x}'for x in xs]
(a.output/'quality-inputs.txt').write_text('\n'.join(base)+'\n')
rng=random.Random(148)
write('exp.txt','E',[bits(x)for x in [-87.,-1.,-.5,-0.,0.,.5,1.,88.4,88.5,88.7]]+[bits(rng.uniform(-87.,88.4))for _ in range(2048)])
logs=[0x00800000,0x00800001,0x3f7fffff,0x3f800000,0x3f800001,0x7f7fffff]+[rng.randrange(0x00800000,0x7f800000)for _ in range(2048)]
for name in ['ln','log10']:write(name+'.txt','L',logs.copy())
xs=[0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0xbf7fffff,bits(-.5),bits(-.4142),bits(1e-8),bits(-1e-8),0x7f7fffff]
xs += [bits(rng.uniform(-.999999,2.))for _ in range(1024)]
xs += [rng.randrange(1,0x7f800000)for _ in range(1024)]
write('log1p.txt','L',xs)

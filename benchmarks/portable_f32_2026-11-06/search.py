#!/usr/bin/env python3
"""Bounded adaptation of math_accuracy's max-error retained-interval search.

Algorithm source: check_sample{,2}.c at c2b897fb, GPL, Zimmermann et al.
This independently written driver uses the same split/sample/retain/exhaust
strategy, but evaluates Rust candidates in batches and scores directed-MPFR
true-real ULP lower bounds. It is heuristic, never a global bound.
"""
import argparse
import array
import itertools
import json
import math
from pathlib import Path
import random
import struct
import subprocess
from mpfr import Oracle,flt,ordered,true_error
p=argparse.ArgumentParser()
p.add_argument('binary',type=Path)
p.add_argument('out',type=Path)
p.add_argument('--samples',type=int,default=64)
p.add_argument('--beam',type=int,default=12)
p.add_argument('--threshold',type=int,default=256)
p.add_argument('--ops',default='exp,ln,log2,pow')
a=p.parse_args();a.binary=a.binary.resolve();a.out.mkdir(parents=True,exist_ok=True)
rng=random.Random(0x20261106);oracle=Oracle()
candidates=['micromath','libm','fpmath','std','metallic','core_scalar','core_vector']
# Ordered finite bit indices allow sign-crossing domains without NaNs.
def raw(k):return (k & 0x7fffffff) if k & 0x80000000 else (~k & 0xffffffff)
def volume(cell):return math.prod(hi-lo for lo,hi in cell)
def split(cell):
    axis=max(range(len(cell)),key=lambda j:cell[j][1]-cell[j][0]);lo,hi=cell[axis];mid=(lo+hi)//2
    if mid==lo:return [cell]
    x=list(cell);y=list(cell);x[axis]=(lo,mid);y[axis]=(mid,hi);return [tuple(x),tuple(y)]
report=[]
for op in a.ops.split(','):
    selected=candidates+(['arm_pow_core_fallback'] if op=='pow' else [])
    domain=((ordered(0xff7fffff),ordered(0x7f7fffff)+1),) if op=='exp' else ((ordered(1),ordered(0x7f7fffff)+1),)
    if op=='pow':domain+=((ordered(0xff7fffff),ordered(0x7f7fffff)+1),)
    beams={c:[domain] for c in selected};best={c:None for c in selected};evaluated=0
    for depth in range(70):
        cells=set(itertools.chain.from_iterable(beams.values()))
        children=sorted(set(itertools.chain.from_iterable(split(cell) if volume(cell)>a.threshold else [cell] for cell in cells)))
        inputs=[];which=[]
        for ci,cell in enumerate(children):
            if volume(cell)<=a.threshold:
                points=itertools.product(*(range(lo,hi) for lo,hi in cell))
            else:
                points=[tuple(rng.randrange(lo,hi) for lo,hi in cell) for _ in range(a.samples)]
            for point in points:
                x=raw(point[0]);y=raw(point[1]) if len(point)>1 else 0
                ref,lo,hi,e=oracle.reference(op,x,y)
                inputs.append((x,y,ref,lo,hi,e));which.append(ci)
        stage=a.out/op/f'level-{depth:02d}';stage.mkdir(parents=True,exist_ok=True)
        records=stage/'inputs.bin';records.write_bytes(b''.join(struct.pack('<III',*v[:3]) for v in inputs))
        subprocess.run([str(a.binary),op,str(records),str(stage)],check=True,stdout=subprocess.DEVNULL)
        next_beams={}
        for candidate in selected:
            outputs=array.array('I');outputs.frombytes((stage/f'{candidate}-{op}.bin').read_bytes())
            scores=[-1.0]*len(children)
            for inp,b,ci in zip(inputs,outputs,which):
                x,y,ref,lo,hi,e=inp
                if not math.isfinite(flt(ref)):continue
                err=true_error(b,lo,hi,e)
                if not err:continue
                lower,upper=err
                scores[ci]=max(scores[ci],lower)
                if best[candidate] is None or lower>best[candidate]['lower']:
                    best[candidate]=dict(candidate=candidate,op=op,x=f'{x:08x}',y=f'{y:08x}',actual=f'{b:08x}',cr=f'{ref:08x}',lower=lower,upper=upper,depth=depth)
            ranked=sorted(range(len(children)),key=lambda j:scores[j],reverse=True)
            next_beams[candidate]=[children[j] for j in ranked[:a.beam] if scores[j]>=0]
            if not next_beams[candidate]:next_beams[candidate]=beams[candidate]
        evaluated+=len(inputs);beams=next_beams
        status=dict(op=op,depth=depth,evaluated=evaluated,cells=len(children),best=best)
        (a.out/'live.json').write_text(json.dumps(status,indent=2)+'\n')
        print(json.dumps(status),flush=True)
        if all(volume(c)<=a.threshold for cs in beams.values() for c in cs):break
    report.extend(dict(v,evaluated=evaluated) for v in best.values() if v is not None)
(a.out/'results.json').write_text(json.dumps(report,indent=2)+'\n')

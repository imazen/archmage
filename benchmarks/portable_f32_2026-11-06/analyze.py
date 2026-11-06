#!/usr/bin/env python3
"""Compare result files and directed MPFR intervals. No candidate is an oracle."""
import argparse
import array
import csv
import itertools
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
from mpfr import Oracle, flt, isnan, lib, ordered, same, true_error

p=argparse.ArgumentParser()
p.add_argument('root',type=Path)
p.add_argument('--true-real',action='store_true')
a=p.parse_args()
assert sys.byteorder=='little'
oracle=Oracle()
summary=[]; pairs=[]; examples=[]
for op in ['exp','ln','log2','pow']:
    records=list(struct.iter_unpack('<III',(a.root/'corpus'/f'{op}.bin').read_bytes()))
    runs={}
    for folder in sorted(a.root.iterdir()):
        if folder.is_dir() and folder.name not in ['corpus','search']:
            for f in sorted(folder.glob(f'*-{op}.bin')):
                out=array.array('I');out.frombytes(f.read_bytes())
                assert len(out)==len(records),(f,len(out),len(records))
                runs[(folder.name,f.stem[:-len(op)-1])]=out
    references=[]
    if a.true_real:
        digest=hashlib.sha256((a.root/'corpus'/f'{op}.bin').read_bytes()).hexdigest()
        cache=a.root/'reference'/f'{op}-directed-v2-{digest}.bin'
        if cache.exists():
            references=list(struct.iter_unpack('<Iddi',cache.read_bytes()))
            assert len(references)==len(records)
        else:
            for i,(x,y,_) in enumerate(records):
                references.append(oracle.reference(op,x,y))
                if i%100000==0: print(f'MPFR {op} {i}/{len(records)}',flush=True)
            cache.parent.mkdir(exist_ok=True)
            cache.write_bytes(b''.join(struct.pack('<Iddi',*r) for r in references))
    for (target,candidate),out in runs.items():
        row=dict(target=target,candidate=candidate,op=op,n=len(out),correct=0,finite_pairs=0,finite_wrong=0,max_bit_steps=0,special_mismatch=0,zero_sign=0,nan_pairs=0,true_max_lower=0.0,true_max_upper=0.0,true_over1=0,central_n=0,central_wrong=0,central_true_max_upper=0.0,central_max_abs=0.0,central_max_rel=0.0)
        worst=None
        for i,((x,y,ref),b) in enumerate(zip(records,out)):
            if a.true_real: assert same(ref,references[i][0]),('old oracle differs',op,i,hex(ref),hex(references[i][0]))
            if same(ref,b):row['correct']+=1
            if 100000<=i<200000:
                row['central_n']+=1
                if not same(ref,b):row['central_wrong']+=1
            if isnan(ref) and isnan(b):row['nan_pairs']+=1
            rf,bf=flt(ref),flt(b)
            if math.isfinite(rf) and math.isfinite(bf):
                row['finite_pairs']+=1
                steps=abs(ordered(ref)-ordered(b))
                if b!=ref: row['finite_wrong']+=1
                if steps>row['max_bit_steps']:
                    row['max_bit_steps']=steps
                    worst=dict(target=target,candidate=candidate,op=op,index=i,x=f'{x:08x}',y=f'{y:08x}',actual=f'{b:08x}',cr=f'{ref:08x}',bit_steps=steps)
                if rf==bf==0 and ref!=b: row['zero_sign']+=1
            elif not same(ref,b):row['special_mismatch']+=1
            if a.true_real:
                _,lo,hi,e=references[i]
                # Overflow and exact nonfinite values are classification tests,
                # not finite true-real ULP measurements.
                if math.isfinite(rf):
                    err=true_error(b,lo,hi,e)
                    if err:
                        lower,upper=err
                        if 100000<=i<200000:
                            row['central_true_max_upper']=max(row['central_true_max_upper'],upper)
                            abs_upper=math.nextafter(max(abs(flt(b)-lo),abs(flt(b)-hi)),math.inf)
                            row['central_max_abs']=max(row['central_max_abs'],abs_upper)
                            if not lo<=0<=hi:
                                rel_upper=math.nextafter(abs_upper/min(abs(lo),abs(hi)),math.inf)
                                row['central_max_rel']=max(row['central_max_rel'],rel_upper)
                        if lower>1:row['true_over1']+=1
                        if upper>row['true_max_upper']:
                            row['true_max_lower']=lower;row['true_max_upper']=upper
                            row['true_worst']=f'{x:08x},{y:08x},{b:08x},{ref:08x}'
        if worst:examples.append(worst)
        summary.append(row)
    targets=sorted({t for t,c in runs})
    candidates=sorted({c for t,c in runs})
    for c in candidates:
        for t,u in itertools.combinations(targets,2):
            if (t,c) not in runs or (u,c) not in runs:continue
            row=dict(candidate=c,op=op,a=t,b=u,n=len(records),mismatches=0,max_finite_steps=0,special_mismatch=0,zero_sign=0)
            for i,(v,w) in enumerate(zip(runs[t,c],runs[u,c])):
                if same(v,w):continue
                row['mismatches']+=1
                if math.isfinite(flt(v)) and math.isfinite(flt(w)):
                    row['max_finite_steps']=max(row['max_finite_steps'],abs(ordered(v)-ordered(w)))
                    if flt(v)==flt(w)==0:row['zero_sign']+=1
                else:row['special_mismatch']+=1
                if 'example' not in row:
                    x,y,_=records[i];row['example']=f'{x:08x},{y:08x},{v:08x},{w:08x}'
            pairs.append(row)
for name,rows in [('accuracy',summary),('pairwise',pairs)]:
    columns=list(dict.fromkeys(k for r in rows for k in r))
    with (a.root/f'{name}.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)
(a.root/'worst-examples.json').write_text(json.dumps(examples,indent=2)+'\n')
print('MPFR',lib.mpfr_get_version().decode(), 'summary rows',len(summary),'pair rows',len(pairs))

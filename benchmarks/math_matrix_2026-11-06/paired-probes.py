#!/usr/bin/env python3
"""Direct paired bootstrap for selected algebraic counterexamples, all rounds."""
from pathlib import Path
import csv,json,random,statistics,sys
root=Path(sys.argv[1]);out=Path(sys.argv[2]);rows=[]
for plat,file,group,a,b in [
('i265','algebraic-i265.json','dot/f32','dot_vector_fused/v3','dot_array_algebraic/v3'),
('arm','algebraic-arm-final.json','dot/f32','dot_vector_fused/neon','dot_array_algebraic/neon'),
('arm','algebraic-arm-final.json','dot/f32','dot_vector_fused/neon','dot_algebraic/neon'),
('r7900x','algebraic-r7900x-final.json','poly/f32','poly_ordinary/v4x','poly_algebraic/v4x'),
('i265','algebraic-i265.json','dot/f32','dot_ordinary/v3','dot_algebraic/v3'),
('r7900x','algebraic-r7900x-final.json','dot/f32','dot_ordinary/v3','dot_algebraic/v3'),
('arm','algebraic-arm-final.json','poly/f32','poly_vector_fused/neon','poly_array_algebraic/neon')]:
 d=json.loads((root/plat/file).read_text());g=next(g for g in d['comparisons']if g['group_name']==group);names=[b['name']for b in g['benchmarks']];ia,ib=names.index(a),names.index(b)
 aa=[s['compensated_ns'][ia]/s['iterations']/1024 for s in g['samples']];bb=[s['compensated_ns'][ib]/s['iterations']/1024 for s in g['samples']];delta=[b-a for a,b in zip(aa,bb)];rng=random.Random(20261106);boot=sorted(statistics.mean(rng.choices(delta,k=len(delta)))for _ in range(10000));rows.append(dict(platform=plat,group=group,a=a,b=b,rounds=len(delta),mean_a=statistics.mean(aa),mean_b=statistics.mean(bb),delta_b_minus_a=statistics.mean(delta),ci95_lo=boot[250],ci95_hi=boot[9750],method='all retained rounds; paired mean difference bootstrap seed20261106 10000resamples'))
with out.open('w')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for r in rows:print(r)

#!/usr/bin/env python3
"""Require complete timings and retain per-operation paired means/CIs compactly."""
from pathlib import Path
import csv,json,sys
root,out=map(Path,sys.argv[1:]);out.mkdir(parents=True,exist_ok=True)
rows=[];algebraic=[];coverage=[]
for platform,files in [('i265',['timing-i265.json','algebraic-i265.json','latency-i265.json']),('r7900x',['timing-r7900x-final.json','algebraic-r7900x-final.json','latency-r7900x.json']),('arm',['timing-arm-final.json','algebraic-arm-final.json','latency-arm.json']),('wasm',['timing-wasm.json'])]:
 for file in files:
  p=root/platform/file
  if not p.exists():continue # This reporter enumerates available runs; final coverage separately asserts requested files.
  d=json.loads(p.read_text());assert not d['incomplete']and not d['unreliable'],p
  n=0;mn=100000
  for g in d['comparisons']:
   scale=1024*(256 if platform=='wasm'else 1)
   analyses={b:(a,r)for a,b,r in g['analyses']}
   for b in g['benchmarks']:
    bs=b['summary'];assert bs['n']>=30,(p,g['group_name'],b['name'],bs['n']);n+=1;mn=min(mn,bs['n'])
    a,r=analyses.get(b['name'],('',{}))
    row=dict(platform=platform,group=g['group_name'],candidate=b['name'],rounds=bs['n'],mean_ns=bs['mean']/scale,mad_ns=bs['mad']/scale,baseline=a,paired_baseline_ns=r.get('baseline',{}).get('mean',0)/scale,paired_candidate_ns=r.get('candidate',{}).get('mean',0)/scale,delta_ci_lo_ns=r.get('ci_lower',0)/scale,delta_ci_hi_ns=r.get('ci_upper',0)/scale,outliers=r.get('n_outliers',0),drift=r.get('drift_correlation',0),resolution_limited=r.get('resolution_limited',False))
    rows.append(row)
    if file.startswith('algebraic')or g['group_name'].startswith('algebraic/'):algebraic.append(row)
  coverage.append(dict(platform=platform,file=file,groups=len(d['comparisons']),cells=n,min_rounds=mn,incomplete=d['incomplete'],unreliable=d['unreliable']))
def write(p,rr):
 with p.open('w')as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
write(root/'timing-all.csv',rows);write(root/'algebraic-all.csv',algebraic);write(out/'coverage.csv',coverage)
# Compact committed tables, one per op/platform: no generated file >30KB.
for platform in sorted({r['platform']for r in rows}):
 for op in sorted({r['group'].split('/')[0]for r in rows if r['platform']==platform}):
  rr=[r for r in rows if r['platform']==platform and r['group'].split('/')[0]==op]
  rr=[{k:(round(v,6)if isinstance(v,float)else v)for k,v in r.items()if k not in ['platform','drift','resolution_limited','paired_baseline_ns','paired_candidate_ns']}for r in rr]
  p=out/f'{platform}-{op}.csv';write(p,rr);assert p.stat().st_size<=30000,(p,p.stat().st_size)
print('completed',len(rows),'timed cells',len(coverage),'runs')

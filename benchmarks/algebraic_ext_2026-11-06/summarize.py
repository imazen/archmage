#!/usr/bin/env python3
"""All workload tables with direct retained-round paired bootstrap intervals."""
from pathlib import Path
import csv,json,random,statistics,sys
root,out=map(Path,sys.argv[1:]);out.mkdir(parents=True,exist_ok=True);allrows=[];coverage=[];tables=['# Focused workload results\n\nAll times are ns/input value for 1,024 values. Dot returns one reduction result.\nIntervals are 95% paired mean-difference bootstrap intervals using all retained\nrounds, 10,000 resamples, seed 20261106. No post-hoc round exclusion here; this is\nseparate from zenbench\'s own baseline analysis. Mean/median gaps and drift\nremain visible in CSVs and raw JSON. Close point estimates are not rankings.\n\nWASM setup allocates cases/inputs/output per host call, amortized over 256 repeats;\nnative setup is outside iteration. Do not treat absolute cross-platform values\nas identical wrapper workloads. LTO modes are separate measured binaries.\n']
def ci(g,a,b,scale):
 names=[b['name']for b in g['benchmarks']];ia,ib=names.index(a),names.index(b);ds=[(s['compensated_ns'][ib]-s['compensated_ns'][ia])/s['iterations']/scale for s in g['samples']];rng=random.Random(20261106);means=sorted(sum(rng.choices(ds,k=len(ds)))/len(ds)for _ in range(10000));return statistics.mean(ds),means[250],means[9750]
for target,prefix,tiers in [('native','timing-',['scalar','v3','v4x']),('arm','timing-',['scalar','neon']),('wasm','timing-wasm-',['scalar','wasm128'])]:
 for mode in ['off','thin']:
  p=root/target/(prefix+mode+'.json');d=json.loads(p.read_text());assert not d['incomplete']and not d['unreliable'];gs=d['comparisons'];assert len(gs)==8;bs=[b for g in gs for b in g['benchmarks']];assert min(b['summary']['n']for b in bs)>=30;coverage.append(dict(target=target,lto=mode,groups=8,cells=len(bs),min_rounds=min(b['summary']['n']for b in bs),incomplete=d['incomplete'],unreliable=d['unreliable']))
  controls=[]
  for g in gs:
   scale=1024*(256 if target=='wasm'else 1)
   for b in g['benchmarks']:
    v=b['summary'];controls.append(dict(workload=g['group_name'],candidate=b['name'],rounds=v['n'],mean=v['mean']/scale,median=v['median']/scale,mad=v['mad']/scale))
  with(out/f'controls-{target}-{mode}.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=list(controls[0]));w.writeheader();w.writerows(controls)
  for tier in tiers:
   rows=[]
   for g in gs:
    v={b['name']:b for b in g['benchmarks']};scale=1024*(256 if target=='wasm'else 1);index='index/'+tier;token='token/'+tier;mapping='map/'+tier;ordinary='ordinary/'+tier
    im=ci(g,mapping,index,scale);tm=ci(g,mapping,token,scale);io=ci(g,ordinary,index,scale)
    row=dict(workload=g['group_name'],rounds=v[index]['summary']['n'])
    for label,name in [('ordinary',ordinary),('tokenless',index),('map',mapping),('token',token)]:
     row[label+'_mean']=v[name]['summary']['mean']/scale;row[label+'_median']=v[name]['summary']['median']/scale;row[label+'_mad']=v[name]['summary']['mad']/scale
    for label,values in [('index_minus_map',im),('token_minus_map',tm),('index_minus_ordinary',io)]:
     for suffix,value in zip(['mean','ci_lo','ci_hi'],values):row[label+'_'+suffix]=value
    rows.append(row);allrows.append(dict(target=target,lto=mode,tier=tier,**row))
   dest=out/f'{target}-{mode}-{tier}.csv'
   with dest.open('w')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
   assert dest.stat().st_size<=30000
   if tier=='scalar':continue
   tables.append(f'\n## {target} / {tier} / LTO {mode}\n\n| Workload | Ordinary | Tokenless | Local map | Index−map 95% CI | Index−ordinary 95% CI |\n|---|---:|---:|---:|---|---|\n')
   for r in rows:tables.append(f'| {r["workload"]} | {r["ordinary_mean"]:.5f} | {r["tokenless_mean"]:.5f} | {r["map_mean"]:.5f} | [{r["index_minus_map_ci_lo"]:.5f}, {r["index_minus_map_ci_hi"]:.5f}] | [{r["index_minus_ordinary_ci_lo"]:.5f}, {r["index_minus_ordinary_ci_hi"]:.5f}] |\n')
for p,rows in[(root/'timing-summary-all.csv',allrows),(out/'coverage.csv',coverage)]:
 with p.open('w')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(out.parent/'RESULTS.md').write_text(''.join(tables));print('accepted cells',sum(r['cells']for r in coverage),'rows',len(allrows),'all eight workloads in each tier/mode table')

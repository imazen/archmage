#!/usr/bin/env python3
"""Analyze every frozen cell once, preserving noisy/exploratory qualifications."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('analyzer',type=Path)
p.add_argument('output',type=Path)
p.add_argument('runs',nargs='+',type=Path)
p.add_argument('--require-cells',type=int,required=True)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
inputs=[];seen=set();admission=[];identity=None;cells=None
for run in a.runs:
    manifest=json.loads((run/'MATRIX.json').read_text())
    if identity is None:identity=manifest['sha256'];cells=manifest['cells']
    assert identity==manifest['sha256'] and cells==manifest['cells']
    for path in sorted(run.glob('[0-9][0-9][0-9].json.gz')):
        index=int(path.name.split('.')[0])
        assert index not in seen, ('repeated cell; do not select a rerun',index)
        seen.add(index)
        raw=gzip.decompress(path.read_bytes());result=json.loads(raw)
        assert len(result['comparisons'])==1
        group=result['comparisons'][0]
        assert group['group_name']==cells[index]
        samples=group['samples']
        assert len(samples)==50 and all(len(s['compensated_ns'])==len(group['benchmarks'])for s in samples)
        assert result['execution']['binary_sha256']==identity
        dirty=sum(s['gate_clean']is not True for s in samples)
        clean=dirty==0 and not result['unreliable'] and not result['incomplete']
        admission.append({'index':index,'cell':cells[index],'pairs':len(samples),'dirty_pairs':dirty,
            'unreliable':result['unreliable'],'resource_clean':clean,
            'placement':result['cpu_placement']['status'],'sha256':hashlib.sha256(raw).hexdigest(),
            'run':str(run.resolve())})
        target=a.output/f'{index:03d}.json';target.write_bytes(raw);inputs.append(str(target.resolve()))
assert len(seen)==a.require_cells, ('missing cells',len(seen),a.require_cells)
assert seen==set(range(a.require_cells)), ('missing original indices',seen)
(a.output/'admission.json').write_text(json.dumps({'binary_sha256':identity,'required':a.require_cells,
    'cells':admission,'resource_clean':sum(r['resource_clean']for r in admission),
    'resource_dirty':sum(not r['resource_clean']for r in admission)},indent=2)+'\n')
with(a.output/'paired.jsonl').open('w')as output:
    subprocess.run([str(a.analyzer.resolve()),*inputs],stdout=output,check=True)
summary={}
for line in(a.output/'paired.jsonl').read_text().splitlines():
    row=json.loads(line)
    key=row['candidate']+'/'+row['baseline']
    counts=summary.setdefault(key,{'all':{},'resource_clean':{},'resource_dirty':{}})
    for bucket in ['all','resource_clean'if row['resource_clean']else'resource_dirty']:
        verdict=row['verdict'];counts[bucket][verdict]=counts[bucket].get(verdict,0)+1
(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'cells':len(seen),'resource_clean':sum(r['resource_clean']for r in admission),
    'resource_dirty':sum(not r['resource_clean']for r in admission),'comparisons':summary},indent=2),flush=True)

#!/usr/bin/env python3
"""Freeze 96 cells before timing, preserve every cell and every dirty verdict."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('binary',type=Path);p.add_argument('output',type=Path)
p.add_argument('--exploratory',action='store_true')
p.add_argument('--policies',action='store_true',help='64 policy-ablation cells, 25 explicitly named implementations')
p.add_argument('--normal-range',action='store_true',help='eight experimental range-contract cells at width 8')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
binary=a.binary.resolve()
cells=[f'{op}/{precision}/{width}/{context}/{size}/{work}'
       for op in ['exp2','log2']for precision in ['midp','lowp']
       for width in ([4,8]if a.policies else[4,8,16])for context in ['native','scalar']
       for size in [64,4096]for work in ['stream','chain']]
assert not(a.normal_range and a.policies)
if a.normal_range:cells=[f'exp2/normal-range/8/{context}/{size}/{work}'for context in ['native','scalar']for size in [64,4096]for work in ['stream','chain']]
assert len(cells)==(8 if a.normal_range else 64 if a.policies else 96)
manifest=dict(binary=str(binary),sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
              cells=cells,pairs=50,exploratory=a.exploratory,
              thresholds=dict(no_regression_ci95_upper=1.02,optimization_ci95_upper=1.00),
              input='0.5+(i%256)/512; chain renormalizes result mantissa to [0.5,1)',
              claim='ordinary-input matrix; bit-distributed and exceptional timing coverage absent')
if a.policies:
    manifest.update(policy_bits=[f for f in range(128)if not f&2 and not(f&1 and f&32)],
        excluded_reason='Full/Specials+AssumeValid fail accepted-domain correctness; retained in contract witnesses, excluded from speed promotion',
        token_arm='BitExact Finite/Positive AssumeValid Caller, scalar lane map preserving actual caller token')
(a.output/'MATRIX.json').write_text(json.dumps(manifest,indent=2)+'\n')
for index,cell in enumerate(cells):
    progress=dict(pid=os.getpid(),stage='running',index=index,required=len(cells),cell=cell,
                  last_check=datetime.now(timezone.utc).isoformat())
    (a.output/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
    result=a.output/f'{index:03d}.json'; log=a.output/f'{index:03d}.log'
    env=dict(os.environ,AUDIT_CELL=cell,AUDIT_RESULT=str(result.resolve()),ZENBENCH_LAUNCHER_PIDS=str(os.getpid()))
    if a.exploratory:env['ZENBENCH_CPU_POLICY']='unrestricted'
    else:assert env.get('ZENBENCH_CPU_POLICY','auto')!='unrestricted'
    with log.open('w')as handle:
        run=subprocess.run([str(binary)],env=env,stdout=handle,stderr=subprocess.STDOUT)
    progress.update(returncode=run.returncode,stage='failed')
    if run.returncode==0:
        data=json.loads(result.read_text());assert len(data['comparisons'])==1
        samples=data['comparisons'][0]['samples']
        clean=not data['unreliable']and not data['incomplete']and len(samples)==50 and all(s['gate_clean']is True for s in samples)
        progress.update(stage='complete'if clean else'inadmissible',pairs=len(samples),unreliable=data['unreliable'])
    progress['last_check']=datetime.now(timezone.utc).isoformat()
    (a.output/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
    print(json.dumps(progress),flush=True)
    if run.returncode:raise SystemExit(run.returncode)

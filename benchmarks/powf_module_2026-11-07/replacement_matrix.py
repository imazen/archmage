#!/usr/bin/env python3
"""Freeze/run replacement cells; retain all pairs and stop on dirty admission."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('binary',type=Path)
p.add_argument('directory',type=Path)
p.add_argument('--pilot',action='store_true')
p.add_argument('--policies',action='store_true',help='56 public-policy cells, 69 named implementations each')
p.add_argument('--rounding',action='store_true',help='the separately frozen 96-cell forced-rounding matrix')
p.add_argument('--record-all',action='store_true',help='record every unmeasured cell once; preserve every dirty verdict')
p.add_argument('--exploratory',action='store_true',help='explicit scheduler-controlled study; never promotion evidence')
p.add_argument('--continue-unmeasured',type=Path,help='retain prior failed cell; measure only its previously unmeasured suffix')
a=p.parse_args()
a.directory.mkdir(parents=True,exist_ok=False)
binary=a.binary.resolve()
env=dict(os.environ,POWF_LIST='1')
cells=subprocess.check_output([str(binary)],env=env,text=True).splitlines()
assert len(cells)==(56 if a.policies else 96 if a.rounding else 672)
if a.pilot:
    native='neon' if any('/neon/' in c for c in cells) else 'v3'
    cells=[f'policy/x4/{native}/64/stream/4019999a' if a.policies else f'rounding/x4/{native}/64/stream/Gamma/4019999a/checked' if a.rounding else f'replacement/x4/{native}/64/stream/2/4019999a/literal-n/unorm8']
start=0
if a.continue_unmeasured:
    assert not a.pilot and not a.exploratory
    prior=json.loads((a.continue_unmeasured/'MATRIX.json').read_text())
    progress=json.loads((a.continue_unmeasured/'progress.json').read_text())
    assert prior['sha256']==hashlib.sha256(binary.read_bytes()).hexdigest()
    assert prior['cells']==cells and progress['stage']=='inadmissible'
    start=progress['index']+1
    assert 0<start<len(cells)
policy=os.environ.get('ZENBENCH_CPU_POLICY','auto')
assert (policy=='unrestricted')==a.exploratory, 'unrestricted requires explicit exploratory labeling'
manifest={'binary':str(binary),'sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'cells':cells,'pairs':50,'pilot':a.pilot,'exploratory':a.exploratory,'promotion_eligible':not a.exploratory,'cpu_policy':policy,'record_all':a.record_all,'start_index':start,'prior_run':str(a.continue_unmeasured.resolve())if a.continue_unmeasured else None}
(a.directory/'MATRIX.json').write_text(json.dumps(manifest,indent=2)+'\n')
for index,cell in enumerate(cells):
    if index<start:continue
    output=(a.directory/f'{index:03d}.json').resolve()
    log=a.directory/f'{index:03d}.log'
    progress={'pid':os.getpid(),'binary_sha256':manifest['sha256'],'exploratory':a.exploratory,'stage':'running','index':index,'required':len(cells),'cell':cell,'last_check':datetime.now(timezone.utc).isoformat()}
    (a.directory/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
    child_env={k:v for k,v in os.environ.items() if k not in ['POWF_LIST','POWF_VERIFY_ONLY']}
    child_env.update(POWF_FILTER=cell,POWF_RESULT=str(output))
    with log.open('w') as handle:
        run=subprocess.run([str(binary)],env=child_env,stdout=handle,stderr=subprocess.STDOUT)
    if run.returncode:
        progress.update(stage='failed',returncode=run.returncode)
    else:
        result=json.loads(output.read_text())
        assert len(result['comparisons'])==1 and result['comparisons'][0]['group_name']==cell
        samples=result['comparisons'][0]['samples']
        clean=not result['unreliable'] and len(samples)>=50 and all(s['gate_clean']is True for s in samples)
        progress.update(stage='complete'if clean else'inadmissible',pairs=len(samples),unreliable=result['unreliable'])
    # Lossless compression bounds disk usage on the admitted homogeneous ARM
    # worker. Compare decompressed bytes before removing task-owned originals.
    for source in [output,log]:
        if source.exists():
            raw=source.read_bytes(); packed=source.with_suffix(source.suffix+'.gz')
            assert not packed.exists()
            with gzip.open(packed,'wb')as handle:handle.write(raw)
            assert gzip.decompress(packed.read_bytes())==raw
            source.unlink()
    progress['last_check']=datetime.now(timezone.utc).isoformat()
    (a.directory/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
    print(json.dumps(progress),flush=True)
    if progress['stage']!='complete' and not((a.exploratory or a.record_all) and progress['stage']=='inadmissible'):raise SystemExit('stopped: preserved all evidence, no favorable rerun')

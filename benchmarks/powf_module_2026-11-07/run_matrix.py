#!/usr/bin/env python3
"""Run frozen native cells separately; retain and stop on the first dirty cell."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('binary')
p.add_argument('directory', type=Path)
p.add_argument('--reject', action='store_true')
p.add_argument('--native', choices=['v3','neon'], required=True)
a = p.parse_args()
a.directory.mkdir(parents=True, exist_ok=False)
cells = []
for size in [64, 4096]:
    for chain in ['stream','chain']:
        for domain in ['PositiveNormal','NonNegativeNormal','Gamma']:
            for distribution in ['ordinary','bits']:
                for width, context in [(1,'scalar'), *[(w,c) for c in ['scalar',a.native] for w in [4,8,16]]]:
                    cells.append(f'powf/x{width}/{context}/{size}/{chain}/{domain}/{distribution}')
assert len(cells) == 168
if a.reject:
    cells = [f'reject/x{width}/{context}/{size}/{density}/{kind}' for size in [64,4096] for context in ['scalar',a.native] for density in ['sparse','dense'] for kind in ['nan','subnormal'] for width in [4,8,16]]
    assert len(cells) == 48
(a.directory/'MATRIX.json').write_text(json.dumps(cells,indent=2)+'\n')
for i, cell in enumerate(cells):
    result = (a.directory/f'{i:03d}.json').resolve()
    progress = {'pid':os.getpid(),'stage':'running','index':i,'required':len(cells),'cell':cell,'last_check':datetime.now(timezone.utc).isoformat()}
    (a.directory/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
    env = dict(os.environ, POWF_FILTER=cell, POWF_RESULT=str(result))
    with (a.directory/f'{i:03d}.log').open('w') as log:
        run = subprocess.run([str(Path(a.binary).resolve())],env=env,stdout=log,stderr=subprocess.STDOUT)
    if run.returncode:
        progress.update(stage='failed',returncode=run.returncode)
    else:
        data=json.loads(result.read_text())
        assert len(data['comparisons'])==1 and data['comparisons'][0]['group_name']==cell
        samples=data['comparisons'][0]['samples']
        clean = not data['unreliable'] and len(samples)>=50 and all(s['gate_clean'] is True for s in samples)
        progress.update(stage='complete' if clean else 'inadmissible',pairs=len(samples),unreliable=data['unreliable'])
    progress['last_check']=datetime.now(timezone.utc).isoformat()
    (a.directory/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
    print(json.dumps(progress),flush=True)
    if progress['stage']!='complete': raise SystemExit('stopped: evidence preserved; no favorable rerun')

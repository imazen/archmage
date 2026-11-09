#!/usr/bin/env python3
"""Run each frozen bounded replacement cell once, retaining every round."""
import argparse, hashlib, json, os, subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('binary',type=Path);p.add_argument('directory',type=Path)
p.add_argument('--pilot',action='store_true');p.add_argument('--policies',action='store_true')
a=p.parse_args();a.directory.mkdir(parents=True,exist_ok=False)
binary=a.binary.resolve()
if a.policies:
 cells=[('v3',8,size,chain,'4019999a',list(range(51)))for size in [64,4096]for chain in [False,True]]
else:
 cells=[(ctx,width,size,chain,exp,[29,35,41,47,48,49,50])for ctx,width in [('v3',4),('v3',8),('v3',16),('v4x',16)]for size in [64,4096]for chain in [False,True]for exp in ['4019999a','3f000000']]
 cells += [(ctx,width,size,False,'4019999a',[29,35,41,47,48,49,50])for ctx,width in [('v3',4),('v3',8),('v3',16),('v4x',16)]for size in [65,4097]]
 assert len(cells)==40
if a.pilot:cells=cells[:1]
(a.directory/'MATRIX.json').write_text(json.dumps({'binary':str(binary),'sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'cells':cells,'pairs':50,'source':subprocess.check_output(['jj','log','-r','@','--no-graph','-T','commit_id'],text=True).strip()},indent=2)+'\n')
for index,(context,width,size,chain,exp,arms)in enumerate(cells):
 output=(a.directory/f'{index:03}.json').resolve()
 env=dict(os.environ,MIDLOW_CONTEXT=context,MIDLOW_WIDTH=str(width),MIDLOW_SIZE=str(size),MIDLOW_CHAIN=str(int(chain)),MIDLOW_EXP=exp,MIDLOW_ARMS=','.join(map(str,arms)),POWF_RESULT=str(output),ZENBENCH_LAUNCHER_PIDS=str(os.getpid()))
 progress={'pid':os.getpid(),'index':index,'required':len(cells),'cell':cells[index],'stage':'running'}
 (a.directory/'progress.json').write_text(json.dumps(progress)+'\n')
 with(a.directory/f'{index:03}.log').open('w')as log:
  run=subprocess.run([str(binary)],env=env,stdout=log,stderr=subprocess.STDOUT)
 if run.returncode:progress.update(stage='failed',returncode=run.returncode)
 else:
  data=json.loads(output.read_text());samples=data['comparisons'][0]['samples']
  clean=sum(s['gate_clean']is True for s in samples)
  progress.update(stage='complete'if clean==50 and not data['unreliable']else'inadmissible',clean=clean,pairs=len(samples),unreliable=data['unreliable'])
 (a.directory/'progress.json').write_text(json.dumps(progress)+'\n');print(json.dumps(progress),flush=True)
 if progress['stage']=='failed':raise SystemExit('failed cell; evidence retained')
 # Collect each frozen cell once, retaining dirty evidence without promotion.

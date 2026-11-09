#!/usr/bin/env python3
"""Hardware-counter diagnostic on fixed provider work, without wall-clock claims."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from external import load_manifest

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('manifest',type=Path);p.add_argument('directory',type=Path)
p.add_argument('methods',nargs='+',help='exact provider labels')
a=p.parse_args();a.directory.mkdir(parents=True,exist_ok=False)
manifest=load_manifest(a.manifest);records=[]
for label in a.methods:
    provider=next(p for p in manifest['providers']if p['label']==label)
    worker=manifest['workers'][provider['worker']]
    request=(f'B {provider["method"]} 8 4096 0 4019999a 256\n')*100
    command=['perf','stat','-e','cycles:u,instructions:u,branches:u,branch-misses:u','-x',',',
             '-o',str(a.directory/(label+'.csv')),'--',worker['binary']]
    run=subprocess.run(command,input=request,capture_output=True,text=True,timeout=120)
    (a.directory/(label+'.stdout')).write_text(run.stdout)
    (a.directory/(label+'.stderr')).write_text(run.stderr);run.check_returncode()
    lines=run.stdout.splitlines();assert len(lines)==100 and all(len(line.split())==8 for line in lines)
    assert len(set(lines))==1
    records.append({'label':label,'worker':worker,'command':command,
                    'input_sha256':hashlib.sha256(request.encode()).hexdigest(),
                    'evaluations':104857600,'claim':'whole-process hardware counters including startup and adapter; no single-call latency claim'})
(a.directory/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps({'profiles':len(records),'evaluations_per_profile':104857600}))

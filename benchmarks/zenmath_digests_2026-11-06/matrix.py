#!/usr/bin/env python3
"""Sequential admitted-host cell runner; max4 compute threads, durable live state."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import os
import sys
import time

HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--build',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--isa',choices=['arm','x86','x86-v3','x86-v4x','wasm'],required=True)
p.add_argument('--group',choices=['top','primitives','pow-families'],default='top')
p.add_argument('--threads',type=int,default=4)
p.add_argument('--count',type=int,default=2**32)
a=p.parse_args()
if not 1<=a.threads<=4 or (a.isa=='wasm' and a.threads!=1): p.error('invalid thread budget')
if a.group=='top':
    ops=['exp','ln','log2','pow:4019999a']
    backends={'arm':['neon','reference','packed-scalar','scalar-neon'], 'x86':['v3','v4x8','v4x','scalar-v3','scalar-v4x'], 'x86-v3':['v3','scalar-v3'], 'x86-v4x':['v4x8','v4x','scalar-v4x'], 'wasm':['wasm128','scalar-wasm128']}[a.isa]
    cells=[(op,backend) for backend in backends for op in ops]
elif a.group=='primitives':
    ops=['sqrt','floor','ceil','trunc','round','roundeven']
    if a.isa in ['x86','x86-v3','x86-v4x']: cells=[(op,'v1' if op=='sqrt' else 'v2') for op in ops]
    else: cells=[(op,backend) for backend in (['neon','reference'] if a.isa=='arm' else ['wasm128']) for op in ops]
else:
    cells=[(op,backend) for backend in ({'arm':['neon','reference'],'x86':['v3','v4x8'],'x86-v3':['v3'],'x86-v4x':['v4x8','v4x','scalar-v4x'],'wasm':['wasm128']}[a.isa]) for op in ['pow:3f000000','pow:40000000','pow:bf800000']]
a.out.mkdir(parents=True,exist_ok=True)
plan=dict(isa=a.isa,group=a.group,threads=a.threads,count=a.count,cells=cells,build=str(a.build))
planpath=a.out/'plan.json'
if planpath.exists():
    if json.loads(planpath.read_text())!=json.loads(json.dumps(plan)):raise ValueError('existing matrix plan differs')
else:planpath.write_text(json.dumps(plan,indent=2)+'\n')
completed=[]
def record(stage,**extra):
    state=dict(updated=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),stage=stage,completed=completed,required=len(cells),**extra)
    temp=a.out/'progress.pending';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(a.out/'progress.json')
record('starting')
try:
    for op,backend in cells:
        name=op.replace(':','-')+'-'+backend;out=a.out/name
        command=[sys.executable,str(HERE/'drive.py'),'run','--build',str(a.build),'--out',str(out),'--op',op,'--backend',backend,'--threads',str(a.threads),'--count',str(a.count),'--compare']
        if (out/'metadata.txt').exists(): command.append('--resume')
        logfile=a.out/(name+f'.{time.time_ns()}.log')
        with logfile.open('x') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            record('running',cell=name,child_pid=child.pid,command=command,log=str(logfile))
            rc=child.wait()
        if rc:raise RuntimeError(f'{name} rc={rc}; log={logfile}')
        completed.append(name)
        record('cell-completed',cell=name)
    record('complete')
except BaseException as error:
    record('failed',error=str(error))
    raise

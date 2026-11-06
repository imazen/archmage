#!/usr/bin/env python3
"""Minimize an observed external candidate trap by corpus bisection."""
import argparse
import json
from pathlib import Path
import struct
import subprocess
p=argparse.ArgumentParser();p.add_argument('module',type=Path);p.add_argument('corpus',type=Path);p.add_argument('out',type=Path);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
data=a.corpus.read_bytes();assert len(data)%12==0
start=0;count=len(data)//12;tries=0
while count>1:
    half=count//2;trial=a.out/f'trial-{tries:02d}.bin';trial.write_bytes(data[start*12:(start+half)*12])
    cmd=['wasmtime','run','--dir','.',str(a.module),'pow',str(trial),str(a.out/f'outputs-{tries:02d}'),'metallic']
    r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (a.out/f'trial-{tries:02d}.log').write_bytes(r.stdout)
    if r.returncode:count=half
    else:start+=half;count-=half
    tries+=1
minimal=a.out/'minimal.bin';minimal.write_bytes(data[start*12:(start+1)*12]);r=subprocess.run(['wasmtime','run','--dir','.',str(a.module),'pow',str(minimal),str(a.out/'minimal-output'),'metallic'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
assert r.returncode,'minimal input must reproduce observed trap'
(a.out/'minimal.log').write_bytes(r.stdout)
x,y,ref=struct.unpack('<III',minimal.read_bytes());finding=dict(candidate='metallic0.3.1',op='pow',index=start,x=f'{x:08x}',y=f'{y:08x}',cr=f'{ref:08x}',returncode=r.returncode,tries=tries)
(a.out/'finding.json').write_text(json.dumps(finding,indent=2)+'\n');print(json.dumps(finding),flush=True)

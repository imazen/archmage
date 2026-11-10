#!/usr/bin/env python3
"""Run the independent power corpus under an explicitly selected SDE CPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('sde')
p.add_argument('--cpu',default='-nhm')
p.add_argument('--expected',default='software')
a=p.parse_args()
root=Path(__file__).resolve().parents[2]
build=subprocess.run(['cargo','test','--release','-p','magetypes','--features','avx512','--lib','--no-run','--message-format=json'],cwd=root,text=True,capture_output=True,check=True)
print(build.stderr,end='')
rows=[json.loads(line) for line in build.stdout.splitlines() if line.startswith('{')]
executables=[r['executable'] for r in rows if r.get('reason')=='compiler-artifact' and r.get('executable') and r['target']['name']=='magetypes']
assert len(executables)==1,executables
env=dict(os.environ,ARCHMAGE_POWF_EXPECT_PLAN=a.expected)
subprocess.run([a.sde,a.cpu,'--',executables[0],'powf::','--nocapture'],cwd=root,env=env,check=True)

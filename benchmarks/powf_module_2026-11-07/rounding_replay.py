#!/usr/bin/env python3
"""Replay actual controlled outputs across targets; no timing claims."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('binary',type=Path)
p.add_argument('inputs',type=Path)
p.add_argument('reference',type=Path,help='prefix of preserved scalar/native actual files')
p.add_argument('output',type=Path)
p.add_argument('--runner',default='')
p.add_argument('--tiers',nargs='+',choices=['scalar','native'],default=['scalar','native'])
a=p.parse_args()
assert not a.output.exists()
request=a.inputs.read_text()
report=dict(binary_sha256=hashlib.sha256(a.binary.read_bytes()).hexdigest(),
            inputs_sha256=hashlib.sha256(request.encode()).hexdigest(),tiers={})
for tier in a.tiers:
    command=shlex.split(a.runner)+[str(a.binary.resolve()),tier]
    result=subprocess.run(command,input=request,text=True,capture_output=True,check=True)
    actual=a.output.with_suffix('.'+tier+'.actual')
    assert not actual.exists()
    actual.write_text(result.stdout)
    a.output.with_suffix('.'+tier+'.stderr').write_text(result.stderr)
    expected=Path(str(a.reference)+'.'+tier+'.actual').read_text().splitlines()
    lines=result.stdout.splitlines()
    assert len(lines)==len(expected)==len(request.splitlines())
    mismatches=[i for i,(x,y)in enumerate(zip(lines,expected))if x.split()[:7]!=y.split()[:7]]
    report['tiers'][tier]=dict(command=command,cases=len(lines),mismatches=mismatches,
                              actual_sha256=hashlib.sha256(result.stdout.encode()).hexdigest())
a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
assert all(not x['mismatches']for x in report['tiers'].values())

#!/usr/bin/env python3
"""Exact dirty-group replacement; immutable original source/binaries/results."""
import json
from pathlib import Path
import subprocess
import sys
snapshot, original, out = map(Path, sys.argv[1:])
snapshot=snapshot.resolve();out.mkdir(parents=True,exist_ok=False)
identity=json.loads((snapshot/'identity.json').read_text())
programs={Path(b['path']).name:snapshot/b['archived_as'] for b in identity['binaries']}
for stage,binary in [('pow','timing'),('transfer','transfer')]:
    first=json.loads((original/(stage+'.json')).read_text())
    dirty={g['group_name'] for g in first['comparisons'] if any(r['gate_clean'] is not True for r in g['samples'])}
    assert len(dirty)==3,(stage,dirty)
    args=[str(programs[binary]),str(out/(stage+'.json'))]
    for group in sorted(dirty):
        if stage=='pow':args.append('--study-group')
        args.append(group)
    with(out/(stage+'.log')).open('w')as f:
        child=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT)
        print(json.dumps(dict(stage=stage,pid=child.pid,groups=sorted(dirty))),flush=True)
        assert child.wait()==0
    second=json.loads((out/(stage+'.json')).read_text())
    assert not second['unreliable'] and not second['incomplete']
    names=[g['group_name'] for g in second['comparisons']]
    assert len(names)==len(set(names)) and set(names)==dirty
    for g in second['comparisons']:
        baseline=next(x for x in first['comparisons'] if x['group_name']==g['group_name'])
        assert [b['name'] for b in baseline['benchmarks']]==[b['name'] for b in g['benchmarks']]
    print(json.dumps(dict(stage=stage,completed=len(names),dirty=[g['group_name']for g in second['comparisons']if any(r['gate_clean']is not True for r in g['samples'])])),flush=True)

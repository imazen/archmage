#!/usr/bin/env python3
"""Produce compact committed tables from retained raw artifacts."""
import csv
import json
from pathlib import Path
import struct
import sys
root,out=map(Path,sys.argv[1:]);out.mkdir(parents=True,exist_ok=True)
def write(name,rows):
    columns=list(dict.fromkeys(k for r in rows for k in r))
    with (out/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)
rows=[]
for platform,filename,repeats in [('x86','timing-x86.json',1),('x86-final-pow','timing-x86-hybrid.json',1),('arm','timing-arm.json',1),('wasm49','timing-wasm.json',256)]:
    data=json.loads((root/filename).read_text())
    assert not data['unreliable'],filename
    for group in data['comparisons']:
        if platform=='x86' and group['group_name'].startswith('pow'):continue
        for b in group['benchmarks']:
            r=b['summary'];rows.append(dict(platform=platform,op=group['group_name'],candidate=b['name'],rounds=r['n'],ns_per_value=r['mean']/(1024*repeats),mad_ns_per_value=r['mad']/(1024*repeats)))
write('timings.csv',rows)
accuracy=list(csv.DictReader((root/'accuracy.csv').open()))
fields=['candidate','op','n','correct','finite_pairs','finite_wrong','central_n','central_wrong','true_max_upper','central_true_max_upper','central_max_abs','central_max_rel','special_mismatch','zero_sign']
write('accuracy-x86.csv',[{**{f:r[f] for f in fields},'correct_percent':100*int(r['correct'])/int(r['n']),'finite_correct_percent':100*(int(r['finite_pairs'])-int(r['finite_wrong']))/int(r['finite_pairs'])} for r in accuracy if r['target']=='x86-native'])
pairs=list(csv.DictReader((root/'pairwise.csv').open()))
selected=[r for r in pairs if r['a']=='x86-native' and r['b'] in ['x86-nehalem','x86-v3'] or r['b']=='x86-native' and r['a'] in ['arm-neon','wasm-strict']]
write('consistency.csv',selected)
write('std-consistency.csv',[r for r in selected if r['candidate']=='std'])
write('search-maxima.csv',json.loads((root/'search-results.json').read_text()))
write('replay-consistency.csv',list(csv.DictReader((root/'worst-replay/pairwise.csv').open())))
cases=[]
for op in ['exp','ln','log2','pow']:
    for x,y,cr in struct.iter_unpack('<III',(root/'worst-replay/corpus'/f'{op}.bin').read_bytes()):
        cases.append(dict(op=op,x=f'{x:08x}',y=f'{y:08x}',correct=f'{cr:08x}'))
write('retained-inputs.csv',cases)
for f in out.iterdir():
    if f.is_file():
        assert f.stat().st_size<=30000,(f,'generated table exceeds git limit')

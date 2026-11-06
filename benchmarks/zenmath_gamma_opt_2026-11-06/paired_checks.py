#!/usr/bin/env python3
"""Negative CLI regressions and exact eight-group acceptance at the caller."""
import argparse
import copy
import csv
import json
from pathlib import Path
import subprocess


def validate(source):
    d=json.loads(source.read_text())
    if d.get('incomplete',False) or d['unreliable']:raise ValueError('incomplete/unreliable')
    expected={f'{k}/{dist}/{n}' for k in ['decode','encode'] for dist in ['values','bits'] for n in [64,4096]}
    if len(d['comparisons'])!=8 or {g['group_name'] for g in d['comparisons']}!=expected:raise ValueError('expected eight groups')
    return d

def main():
    p=argparse.ArgumentParser();p.add_argument('binary',type=Path);p.add_argument('input',type=Path);p.add_argument('out',type=Path);args=p.parse_args()
    d=validate(args.input);args.out.mkdir(exist_ok=False)
    cases=[]
    bad=copy.deepcopy(d);bad['comparisons']=[];cases.append(('empty',bad))
    for label,value in [('zero','0'),('nan','NaN'),('inf','inf')]:
        bad=copy.deepcopy(d);bad['comparisons'][0]['group_name']='decode/values/'+value;cases.append((label+'-scale',bad))
    bad=copy.deepcopy(d);bad['comparisons'][0]['samples'][0]['compensated_ns'].pop();cases.append(('sample-width',bad))
    for label,bad in cases:
        fixture=args.out/(label+'.json');fixture.write_text(json.dumps(bad))
        r=subprocess.run([str(args.binary),str(fixture),str(args.out/(label+'.csv')),'piece_e5'],capture_output=True,text=True)
        (args.out/(label+'.log')).write_text(r.stdout+r.stderr)
        if r.returncode==0:raise ValueError('negative fixture accepted '+label)
    fixture=args.out/'incomplete.json';fixture.write_text('# zenbench results (INCOMPLETE — benchmark still running)\n')
    r=subprocess.run([str(args.binary),str(fixture),str(args.out/'incomplete.csv'),'piece_e5'],capture_output=True,text=True)
    (args.out/'incomplete.log').write_text(r.stdout+r.stderr)
    if r.returncode==0:raise ValueError('incomplete text accepted')
    for i,bad in enumerate([dict(d,incomplete=True),dict(d,comparisons=d['comparisons'][:-1])]):
        fixture=args.out/f'caller-negative-{i}.json';fixture.write_text(json.dumps(bad))
        try:validate(fixture)
        except ValueError:pass
        else:raise ValueError('caller incomplete/inventory accepted')
    final=args.out/'fixed-e5-all-round.csv'
    r=subprocess.run([str(args.binary),str(args.input),str(final),'piece_e5'],capture_output=True,text=True,check=True)
    (args.out/'fixed-e5-all-round.log').write_text(r.stdout+r.stderr)
    rows=list(csv.DictReader(final.open()))
    if len(rows)!=8:raise ValueError('paired output expected eight groups')
    print(r.stdout,end='');print('PASS six Rust CLI negative fixtures + two caller completeness checks; fixed E5 uses all retained paired rounds')

if __name__=='__main__':main()

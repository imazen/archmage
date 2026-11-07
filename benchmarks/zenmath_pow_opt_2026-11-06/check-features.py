#!/usr/bin/env python3
"""Availability and exact contract snapshots for every additive family union."""
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

root=Path.cwd()
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
manifest=root/'benchmarks/zenmath_pow_opt_2026-11-06/feature-consumer/Cargo.toml'
expected={}
rows=[]
for mask in range(8):
    families=[name for bit,name in enumerate(['correct-rounding','normal','configurable'])if mask&(1<<bit)]
    for backend in ['', 'archmage','runtime-dispatch']:
        for vector in [False,True]:
            features=families+([backend]if backend else [])+(['vector']if vector else [])
            label=f'{mask}-{backend or "plain"}-'+('vector'if vector else 'scalar')
            args=['cargo','run','--quiet','--release','--manifest-path',str(manifest)]
            if features:args+=['--features',','.join(features)]
            p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
            (out/(label+'.stdout')).write_bytes(p.stdout);(out/(label+'.stderr')).write_bytes(p.stderr)
            assert not p.stderr,(label,p.stderr.decode())
            for line in p.stdout.decode().splitlines():
                key,value=line.split('=',1);value=json.loads(value)
                if key in expected:assert expected[key]==value,(label,key,value,expected[key])
                else:expected[key]=value
            rows.append(dict(families=families,backend=backend or 'plain',vector=vector,log=label))
            print('PASS',label,flush=True)
# Precise negative compilation gates: absence must fail for the named API,
# never merely for a network/build error. No existing test assertion is relaxed.
probes=[
 ('correct-off-normal-on','normal','zenmath::scalar::powf(0.5,2.4)','powf'),
 ('normal-off-correct-on','correct-rounding','zenmath::normal::powf(0.5,2.4)','normal'),
 ('configurable-off-correct-on','correct-rounding','zenmath::PowBehavior::MIDP','PowBehavior'),
]
for label,feature,expression,symbol in probes:
    folder=out/label;folder.mkdir();(folder/'src').mkdir()
    (folder/'Cargo.toml').write_text('[package]\nname="negative-'+label+'"\nversion="0.0.0"\nedition="2024"\n[workspace]\n[dependencies]\nzenmath={path='+json.dumps(str(root/'zenmath'))+',default-features=false,features=['+json.dumps(feature)+']}\n')
    (folder/'src/main.rs').write_text('fn main(){let _='+expression+';}\n')
    p=subprocess.run(['cargo','check','--manifest-path',str(folder/'Cargo.toml')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (folder/'compile.log').write_bytes(p.stdout)
    text=p.stdout.decode()
    assert p.returncode!=0 and symbol in text and ('E0425' in text or 'E0433' in text),text
(out/'inventory.json').write_text(json.dumps(dict(positive=rows,negative=[x[0]for x in probes],snapshots=expected),indent=2)+'\n')
print('PASS 48 family/backend/consumer configurations; 3 explicit missing-API gates; all shared outputs identical (NaNs canonicalized for contract comparison only)')

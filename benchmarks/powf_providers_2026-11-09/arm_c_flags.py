#!/usr/bin/env python3
"""Compare scalar C result bits, errno and FP exceptions to installed powf."""
import argparse
import gzip
import json
from pathlib import Path
import subprocess
from external import load_manifest

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('manifest',type=Path);p.add_argument('fixtures',type=Path);p.add_argument('directory',type=Path)
a=p.parse_args();a.directory.mkdir(parents=True,exist_ok=False)
manifest=load_manifest(a.manifest)
inputs=[line.split()[:2]for line in a.fixtures.read_text().splitlines()]
assert len(inputs)in [27909,515]
all_outputs={};summary={'cases':len(inputs),'reference':'system_C_powf','providers':{}}
for provider in manifest['providers']:
    if provider['worker']!=1:continue
    method=provider['method'];worker=manifest['workers'][1]['binary']
    request=''.join(f'F {method} 1 {x} {n}\n'for x,n in inputs)
    run=subprocess.run([worker],input=request,capture_output=True,text=True,timeout=120)
    (a.directory/(method+'.stderr')).write_text(run.stderr);run.check_returncode()
    with gzip.open(a.directory/(method+'.actual.gz'),'wt')as f:f.write(run.stdout)
    rows=[line.split()for line in run.stdout.splitlines()]
    assert len(rows)==len(inputs)and all(len(row)==3 for row in rows)
    all_outputs[method]=[(int(b,16),int(e),int(f))for b,e,f in rows]
base=all_outputs['system_C_powf']
with gzip.open(a.directory/'all-variances.jsonl.gz','wt')as ledger:
    for method,actual in all_outputs.items():
        summary['providers'][method]={'bit_differences':sum(x[0]!=y[0]for x,y in zip(actual,base,strict=True)),
                                     'errno_differences':sum(x[1]!=y[1]for x,y in zip(actual,base,strict=True)),
                                     'FP_exception_differences':sum(x[2]!=y[2]for x,y in zip(actual,base,strict=True))}
    for index,pair in enumerate(inputs):
        values={method:[f'{actual[index][0]:08x}',*actual[index][1:]]for method,actual in all_outputs.items()}
        if len({tuple(value)for value in values.values()})>1:
            ledger.write(json.dumps({'input':pair,'outputs':values},separators=(',',':'))+'\n')
(a.directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))

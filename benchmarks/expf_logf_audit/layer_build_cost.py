#!/usr/bin/env python3
"""Matched cold, no-change and math-edit builds; never delete measured targets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('sources',type=Path);p.add_argument('output',type=Path)
a=p.parse_args();a.sources=a.sources.resolve();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
rows=[]; expected=None
modes=['direct','facade','owned-compat']
for profile in ['dev','release','thin']:
    for repeat in range(3):
        order=modes[repeat:]+modes[:repeat]
        for mode in order:
            source=a.sources/mode; root=a.output/f'{profile}-{repeat}-{mode}';root.mkdir()
            target=root/'target';env=dict(os.environ,CARGO_TARGET_DIR=str(target),CARGO_BUILD_JOBS='2',RUSTFLAGS=f'--remap-path-prefix={source}=/layer-study')
            command=['cargo','build','--manifest-path',str(source/'consumer/Cargo.toml'),'--profile',profile,'--offline']
            for stage in (['cold','no-change','math-edit']if repeat==0 else['cold']):
                math=source/('magemath'if mode=='owned-compat'else'archmage/magetypes')/'src/expf/graph.rs'
                original=math.read_text()
                if stage=='math-edit':
                    needle='let bits = input.bitcast_to_i32();'
                    assert needle in original
                    math.write_text(original.replace(needle,'let bits = { let input_bits = input.bitcast_to_i32(); input_bits };',1))
                try:
                    with (root/f'{stage}.log').open('w')as log:
                        timed=['/usr/bin/time','-v','-o',str(root/f'{stage}.time'),*command]
                        run=subprocess.run(timed,env=env,stdout=log,stderr=subprocess.STDOUT)
                    binary=target/('debug'if profile=='dev'else profile)/'magemath-layer-consumer'
                    row=dict(profile=profile,repeat=repeat,mode=mode,stage=stage,command=timed,returncode=run.returncode,rustflags=env['RUSTFLAGS'])
                    if run.returncode==0:
                        output=subprocess.check_output([str(binary)])
                        digest=hashlib.sha256(output).hexdigest()
                        if expected is None:expected=digest
                        assert digest==expected,(mode,profile,stage,digest,expected)
                        (root/f'{stage}.outputs').write_bytes(output)
                        row.update(output_sha256=digest,binary_bytes=binary.stat().st_size,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),rlib_bytes={name:sum(f.stat().st_size for f in (binary.parent/'deps').glob('lib'+name+'-*.rlib'))for name in ['archmage','magetypes','magemath']})
                    rows.append(row);(a.output/'builds.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
                    if run.returncode:raise SystemExit(run.returncode)
                finally:
                    if stage=='math-edit':math.write_text(original)

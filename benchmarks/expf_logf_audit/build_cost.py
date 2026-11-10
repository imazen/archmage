#!/usr/bin/env python3
"""Three alternating cold dev builds; preserve times, peak RSS and rlib bytes."""
import argparse
import json
import os
from pathlib import Path
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('baseline',type=Path);p.add_argument('pr',type=Path);p.add_argument('output',type=Path)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
rows=[]
for pair in range(3):
    for label in (['main','PR147']if pair%2==0 else['PR147','main']):
        source=(a.baseline if label=='main'else a.pr).resolve()
        root=a.output/f'{pair}-{label}';root.mkdir()
        target=root/'target';target.mkdir()
        env=dict(os.environ,CARGO_TARGET_DIR=str(target.resolve()),RUSTFLAGS=f'--remap-path-prefix={source}=/audit-source')
        command=['/usr/bin/time','-v','-o',str((root/'time.txt').resolve()),'cargo','build','--locked','-p','magetypes','--lib']
        with (root/'build.log').open('w')as log:
            run=subprocess.run(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
        libs=list((target/'debug/deps').glob('libmagetypes-*.rlib'))
        row=dict(pair=pair,implementation=label,source=str(source),command=command,
                 rustflags=env['RUSTFLAGS'],returncode=run.returncode,
                 rlib_bytes=libs[0].stat().st_size if len(libs)==1 else None)
        rows.append(row);(a.output/'builds.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(json.dumps(row),flush=True)
        if run.returncode:raise SystemExit(run.returncode)

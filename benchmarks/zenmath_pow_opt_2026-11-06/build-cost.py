#!/usr/bin/env python3
"""Three fresh Cargo targets per configuration; one identical gamma consumer."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

root=Path.cwd()
out=Path(sys.argv[1]).resolve()
out.mkdir(parents=True,exist_ok=False)
manifest=root/'benchmarks/zenmath_pow_opt_2026-11-06/compile-consumer/Cargo.toml'
rows=[]
expected=None
for mode in ['default','archmage','runtime-dispatch']:
    for repeat in range(3):
        target=out/f'target-{mode}-{repeat}'
        env=os.environ.copy();env['CARGO_TARGET_DIR']=str(target)
        args=['cargo','build','--manifest-path',str(manifest),'--locked','--release']
        if mode!='default':args+=['--features',mode]
        log=out/f'{mode}-{repeat}.log'
        resource=out/f'{mode}-{repeat}.resource'
        with log.open('w')as f:
            subprocess.run(['/usr/bin/time','-f','%e %M','-o',str(resource),*args],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        binary=target/'release/zenmath-pow-consumer'
        observed=subprocess.check_output([str(binary)],env=env)
        (out/f'{mode}-{repeat}.stdout').write_bytes(observed)
        if expected is None:expected=observed
        assert observed==expected,'configuration changed consumer outputs'
        elapsed,rss=resource.read_text().split()
        rows.append(dict(configuration=mode,replicate=repeat,wall_seconds=elapsed,max_rss_kib=rss,binary_bytes=binary.stat().st_size))
        print(rows[-1],flush=True)
with(out/'build-cost.csv').open('w')as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(out/'commands.json').write_text(json.dumps(dict(source_manifest=str(manifest),cargo_lock=str(manifest.with_name('Cargo.lock')),toolchain=subprocess.check_output(['rustc','-vV'],text=True),rustflags=os.environ.get('RUSTFLAGS',''),encoded_rustflags=os.environ.get('CARGO_ENCODED_RUSTFLAGS',''),note='Cargo registry/toolchain already cached; target directories all fresh'),indent=2)+'\n')

#!/usr/bin/env python3
"""Fresh targets, three replicates, explicit family workloads and Cargo timelines."""
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
ROOT=Path.cwd()
OUT=Path(sys.argv[1]).resolve();OUT.mkdir(parents=True,exist_ok=False)
MANIFEST=ROOT/'benchmarks/zenmath_pow_opt_2026-11-06/feature-consumer/Cargo.toml'
configs=[]
for label,families in [('primitives',[]),('correct',['correct-rounding']),('normal',['normal']),('configurable',['configurable']),('all',['correct-rounding','normal','configurable'])]:
    for backend in ['', 'archmage','runtime-dispatch']:
        configs.append((label+'-'+(backend or 'plain'),families+([backend]if backend else [])))
for backend in ['', 'archmage','runtime-dispatch']:
    configs.append(('combined-'+(backend or 'plain'),['correct-rounding','normal','configurable','vector']+([backend]if backend else [])))
expected={};rows=[];overlaps=[]
(OUT/'binaries').mkdir()
for label,features in configs:
    for replicate in range(3):
        target=OUT/f'target-{label}-{replicate}'
        env=os.environ.copy();env['CARGO_TARGET_DIR']=str(target)
        args=['cargo','build','--locked','--release','--timings','--manifest-path',str(MANIFEST)]
        if features:args+=['--features',','.join(features)]
        resource=OUT/f'{label}-{replicate}.resource'
        with (OUT/f'{label}-{replicate}.log').open('w')as log:
            subprocess.run(['/usr/bin/time','-f','%e %M','-o',str(resource),*args],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        binary=target/'release/math-feature-consumer';data=binary.read_bytes();digest=hashlib.sha256(data).hexdigest()
        shutil.copy2(binary,OUT/'binaries'/f'{label}-{replicate}-{digest[:16]}')
        stdout=subprocess.check_output([str(binary)],env=env);(OUT/f'{label}-{replicate}.stdout').write_bytes(stdout)
        for line in stdout.decode().splitlines():
            key,value=line.split('=',1)
            if key in expected:assert value==expected[key],(label,key)
            else:expected[key]=value
        seconds,rss=resource.read_text().split()
        row=dict(configuration=label,features=','.join(features),replicate=replicate,wall_seconds=seconds,max_rss_kib=rss,binary_bytes=len(data),binary_sha256=digest)
        rows.append(row);print(row,flush=True)
        html=target/'cargo-timings/cargo-timing.html';text=html.read_text()
        (OUT/f'{label}-{replicate}.timing.html').write_text(text)
        if label.startswith('combined'):
            match=re.search(r'(?:const|var|let) UNIT_DATA\s*=\s*(\[.*?\]);',text,re.S)
            assert match,'Cargo timeline unit data missing'
            units=json.loads(match[1]);(OUT/f'{label}-{replicate}.units.json').write_text(json.dumps(units,indent=2)+'\n')
            z=[u for u in units if u['name']=='zenmath'];m=[u for u in units if u['name']=='magetypes'];assert z and m,(label,units)
            for a in z:
                for b in m:
                    overlap=max(0.0,min(a['start']+a['duration'],b['start']+b['duration'])-max(a['start'],b['start']))
                    overlaps.append(dict(configuration=label,replicate=replicate,zenmath_start=a['start'],zenmath_duration=a['duration'],magetypes_start=b['start'],magetypes_duration=b['duration'],observed_overlap_seconds=overlap))
        with (OUT/'build-cost.csv').open('w')as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        (OUT/'overlap.json').write_text(json.dumps(overlaps,indent=2)+'\n')
(OUT/'identity.json').write_text(json.dumps(dict(configurations=len(configs),builds=len(rows),compiler=subprocess.check_output(['rustc','-vV'],text=True),cargo_lock_sha256=hashlib.sha256(MANIFEST.with_name('Cargo.lock').read_bytes()).hexdigest(),note='Toolchain and dependency downloads cached; every Cargo target fresh. Same family workloads across backends; primitives, precise and configurable workloads differ by API availability. Configurable consumes nine policies. Combined consumer uses scalar plus packed power. Cargo timeline interval overlap is observed compilation scheduling, not an isolated causal speedup.',env={k:os.environ.get(k,'')for k in ['RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS','CARGO_BUILD_JOBS']}),indent=2)+'\n')
print('PASS 54 fresh builds, matching shared output bits, retained exact binaries/timelines',flush=True)

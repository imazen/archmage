#!/usr/bin/env python3
"""Build pinned CORE-MATH powf only, with explicit distribution/FMA-only flags."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import shutil
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('crate',type=Path,help='cargo read core-math-sys==1.4.0 --path-only')
p.add_argument('directory',type=Path)
a=p.parse_args();a.directory.mkdir(parents=True,exist_ok=False)
source=a.crate/'vendor/src/binary32/pow/powf.c'
assert source.is_file()
shutil.copy2(source,a.directory/'powf.c')
shutil.copy2(a.crate/'Cargo.toml',a.directory/'source-Cargo.toml')
for name in ['LICENSE','README.md','.cargo_vcs_info.json']:
    candidate=a.crate/name
    if candidate.is_file():shutil.copy2(candidate,a.directory/('source-'+name.lstrip('.')))
compiler='cc'
flags=['-std=c11','-O3','-g','-fPIC','-shared','-frounding-math','-fno-fast-math','-ffp-contract=off','-fexcess-precision=standard']
machine=platform.machine()
if machine in ['x86_64','AMD64']:flags+=['-march=x86-64']
elif machine in ['aarch64','arm64']:flags+=['-march=armv8-a']
else:raise RuntimeError('unregistered architecture')
manifest={'crate':'core-math-sys=1.4.0','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
          'compiler':subprocess.check_output([compiler,'--version'],text=True),'builds':[],
          'errno':'CORE_MATH_SUPPORT_ERRNO is not defined',
          'baseline_note':'explicit fma operations may call dynamically dispatched system fma; baseline does not mean unfused arithmetic'}
variants=[('baseline',[])]
if machine in ['x86_64','AMD64']:variants.append(('fma-no-avx2',['-mavx','-mfma','-mno-avx2']))
for name,features in variants:
    library=(a.directory/f'core-powf-{name}.so').resolve()
    command=[compiler,*flags,*features,str(source.resolve()),'-lm','-o',str(library)]
    with (a.directory/f'{name}.log').open('w')as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    manifest['builds'].append({'name':name,'command':command,'binary':str(library),'sha256':hashlib.sha256(library.read_bytes()).hexdigest()})
(a.directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))

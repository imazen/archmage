#!/usr/bin/env python3
"""Copy source/executable witnesses and create an immutable review checksum set."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
p=argparse.ArgumentParser(description=__doc__);p.add_argument('scratch',type=Path);p.add_argument('layer_sources',type=Path);p.add_argument('output',type=Path);p.add_argument('revision');a=p.parse_args()
assert(a.output/'raw').is_dir()
sources=a.output/'sources';sources.mkdir()
revs={'review':a.revision,'submitted-148':'a052a8ccfee0c463b176265c28392c755b02ee10','base-147':'0464774e7c5c63a398041ddaa131b9bd4f7035c7','merged-main':'91a13739b73f46213ecfe7083c77297c531fc8ad','original-timing-harness':'38129b882e0d105d527fac5407048e724ac77c16','ordered-harness':'d4006552','range-harness':'26eb088b','layer-preparation-tools':'f581cb7e'}
resolved={}
for name,rev in revs.items():
 rev=subprocess.check_output(['git','rev-parse',rev],text=True).strip();resolved[name]=rev
 with(sources/(name+'.tar')).open('wb')as f:subprocess.run(['git','archive',rev],stdout=f,check=True)
with tarfile.open(sources/'prepared-layer-layouts.tar','w')as archive:
 for f in sorted(a.layer_sources.rglob('*')):
  rel=f.relative_to(a.layer_sources)
  if any(x=='target'or x.endswith('-target')for x in rel.parts):continue
  if f.is_file():archive.add(f,arcname=str(rel),recursive=False)
executables=a.output/'executables';executables.mkdir()
for name in ['normal_range','analyze_normal','analyze_timing','analyze_policies','evaluate','contracts','nan_scan']:
 f=a.scratch/'target/release'/name
 if f.is_file():shutil.copy2(f,executables/name)
for profile in ['dev','release','thin']:
 for repeat in range(3):
  for mode in ['direct','facade','owned-compat']:
   f=a.scratch/'layer-builds'/f'{profile}-{repeat}-{mode}'/'target'/('debug'if profile=='dev'else profile)/'magemath-layer-consumer'
   assert f.is_file(),f
   shutil.copy2(f,executables/f'layer-{profile}-{repeat}-{mode}')
meta=dict(revisions=resolved,rustc=subprocess.check_output(['rustc','-Vv'],text=True),claim='frozen review evidence; stage outputs/logs retained; layer repeat-0 binaries reflect final math-edit rebuild')
(a.output/'IDENTITY.json').write_text(json.dumps(meta,indent=2)+'\n')
rows=[]
for f in sorted(a.output.rglob('*')):
 if f.is_file():rows.append(dict(path=str(f.relative_to(a.output)),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
manifest=a.output/'CORE-MANIFEST.json';manifest.write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(dict(files=len(rows),bytes=sum(r['bytes']for r in rows),manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())))

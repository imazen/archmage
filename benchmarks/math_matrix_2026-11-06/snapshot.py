#!/usr/bin/env python3
"""Freeze source and binaries beside each run; exclude caches and private instructions."""
from pathlib import Path
import hashlib,json,subprocess,sys,tarfile
root=Path.cwd();out=root/sys.argv[1];out.mkdir(parents=True,exist_ok=True)
files=[]
for folder in ['src','archmage-macros','magetypes','zenmath','magemath','benchmarks/math_matrix_2026-11-06']:
 for p in (root/folder).rglob('*'):
  if p.is_file() and not any(v in ['target','target-codegen','target-wasm','target-wasm-host']or v.startswith('target-')for v in p.relative_to(root).parts)and(p.suffix in ['.rs','.toml','.py','.sh']or p.name=='Cargo.lock'):files.append(p)
for name in ['Cargo.toml','Cargo.lock','token-registry.toml']:files.append(root/name)
files=sorted(set(files));manifest=''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(root))+'\n'for p in files)
(out/'source.sha256').write_text(manifest)
with tarfile.open(out/'source.tar.gz','w:gz')as tar:
 for p in files:tar.add(p,arcname=str(p.relative_to(root)),recursive=False)
bins=[]
for p in map(Path,sys.argv[2:]):
 b=p.read_bytes();bins.append(dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b)))
(out/'identity.json').write_text(json.dumps(dict(source_manifest_sha256=hashlib.sha256(manifest.encode()).hexdigest(),binaries=bins,rustc=subprocess.check_output(['rustc','-vV'],text=True),uname=subprocess.check_output(['uname','-a'],text=True)),indent=2)+'\n')
print((out/'identity.json').read_text())

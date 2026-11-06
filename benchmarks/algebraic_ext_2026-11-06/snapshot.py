#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,subprocess,sys,tarfile
r=Path.cwd();out=r/sys.argv[1];out.mkdir(parents=True,exist_ok=True);files=[]
for folder in ['src','archmage-macros','magetypes','benchmarks/algebraic_ext_2026-11-06']:
 for p in(r/folder).rglob('*'):
  if p.is_file()and not any(v.startswith('target')or v.startswith('outputs-')for v in p.relative_to(r).parts)and(p.suffix in['.rs','.toml','.py','.sh']or p.name=='Cargo.lock'):files.append(p)
for n in['Cargo.toml','Cargo.lock','token-registry.toml']:files.append(r/n)
files=sorted(set(files));m=''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(r))+'\n'for p in files);(out/'source.sha256').write_text(m)
with tarfile.open(out/'source.tar.gz','w:gz')as tar:
 for p in files:tar.add(p,arcname=str(p.relative_to(r)),recursive=False)
meta=dict(source_manifest_sha256=hashlib.sha256(m.encode()).hexdigest(),binaries=[dict(path=p,sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest(),bytes=Path(p).stat().st_size)for p in sys.argv[2:]],rustc=subprocess.check_output(['rustc','-vV'],text=True));(out/'identity.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta))

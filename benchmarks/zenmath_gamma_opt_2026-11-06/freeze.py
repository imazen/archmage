#!/usr/bin/env python3
"""Hash compiled inputs, binary and actual Cargo fingerprint into an immutable run."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('out',type=Path);args=parser.parse_args()
    here=Path(__file__).resolve().parent;root=here.parent.parent
    target=Path(os.environ['CARGO_TARGET_DIR']);binary=target/'release/zenmath-gamma-opt'
    args.out.mkdir(parents=True,exist_ok=False)
    files=set()
    for directory in [root/'src',root/'archmage-macros/src',root/'zenmath/src',here/'src']:
        files.update(directory.rglob('*.rs'))
    for directory in [root,root/'archmage-macros',root/'zenmath',here]:
        for name in ['Cargo.toml','Cargo.lock']:
            if (directory/name).exists():files.add(directory/name)
    files.update((root/'.cargo').rglob('*'))
    manifest=''.join(f'{sha(p)}  {p.relative_to(root)}\n' for p in sorted(files) if p.is_file())
    (args.out/'sources.sha256').write_text(manifest)
    shutil.copyfile(binary,args.out/'zenmath-gamma-opt')
    if sha(binary)!=sha(args.out/'zenmath-gamma-opt'):raise ValueError('binary retention copy mismatch')
    compiler=subprocess.check_output(['rustup','run','1.99.1','rustc','-vV'],text=True)
    (args.out/'rustc.txt').write_text(compiler)
    compiler_path=Path(os.environ['RUSTC'])
    fingerprints=[]
    for p in sorted((target/'release/.fingerprint').glob('*/bin-zenmath-gamma-opt.json')):
        data=json.loads(p.read_text())
        if data['rustflags']!=[]:raise ValueError('ambient compile flags: '+repr(data['rustflags']))
        fingerprints.append({'path':str(p),'sha256':sha(p),'data':data})
    if not fingerprints:raise ValueError('missing actual Cargo fingerprints')
    (args.out/'build.json').write_text(json.dumps(dict(source_manifest_sha256=sha(args.out/'sources.sha256'),
        binary=str(binary),binary_sha256=sha(binary),compiler=str(compiler_path),compiler_sha256=sha(compiler_path),
        compiler_version=compiler,cargo_fingerprints=fingerprints,profile='release thin LTO codegen-units=1',
        placement='local source/target/cargo/tmp/toolchain; NAS is artifact mirror only'),indent=2)+'\n')
    print((args.out/'build.json').read_text())

if __name__=='__main__':main()

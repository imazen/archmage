#!/usr/bin/env python3
"""Hash compiled inputs, binary and actual Cargo fingerprint into an immutable run."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import tarfile


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
    shutil.copy2(binary,args.out/'zenmath-gamma-opt')
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
    timed_match={}
    timed_path=os.environ.get('GAMMA_TIMED_IDENTITY')
    if timed_path:
        timed=Path(timed_path);old=dict((name,digest) for digest,name in (line.split('  ',1) for line in (timed/'sources.sha256').read_text().splitlines()))
        required=[p for p in files if p.is_file() and (not p.is_relative_to(here) or p.name in ['generated.rs','kernel.rs','piecewise.rs','piecewise_generated.rs','vector.rs','Cargo.toml','Cargo.lock'])]
        for p in required:
            name=str(p.relative_to(root));current=sha(p)
            if old.get(name)!=current:raise ValueError('timed numerical/build input changed: '+name)
            timed_match[name]=current
        if not all(any(name.endswith('/'+file) for name in timed_match) for file in ['kernel.rs','vector.rs','generated.rs','piecewise.rs','piecewise_generated.rs']):raise ValueError('incomplete numerical byte-match')
        (args.out/'timed-kernel-byte-match.json').write_text(json.dumps(dict(timed_identity=str(timed),timed_binary_sha256=sha(timed/'zenmath-gamma-opt'),files=timed_match),indent=2)+'\n')
    archive=args.out/'source.tar.gz'
    def included(member):
        if any(part in ['.git','.jj','.workongoing','.zenbench','__pycache__','target'] for part in Path(member.name).parts):return None
        return member
    with tarfile.open(archive,'w:gz') as tar:tar.add(root,arcname='source',filter=included)
    # Hash every manifested source member while streaming retained source archive.
    expected=dict((name,digest) for digest,name in (line.split('  ',1) for line in manifest.splitlines()));seen=set()
    with tarfile.open(archive,'r|gz') as tar:
        for member in tar:
            name=member.name.removeprefix('source/')
            if name in expected:
                if name in seen or not member.isfile():raise ValueError('archive member identity')
                stream=tar.extractfile(member);h=hashlib.sha256()
                while block:=stream.read(1<<20):h.update(block)
                if h.hexdigest()!=expected[name]:raise ValueError('source archive byte mismatch '+name)
                seen.add(name)
    if seen!=set(expected):raise ValueError('source archive missing required compiled inputs')
    (args.out/'build.json').write_text(json.dumps(dict(source_manifest_sha256=sha(args.out/'sources.sha256'),
        source_archive_sha256=sha(archive),timed_kernel_byte_match=timed_match,binary=str(binary),binary_sha256=sha(binary),compiler=str(compiler_path),compiler_sha256=sha(compiler_path),
        compiler_version=compiler,cargo_fingerprints=fingerprints,profile='release thin LTO codegen-units=1',
        placement='local source/target/cargo/tmp/toolchain; NAS is artifact mirror only'),indent=2)+'\n')
    print((args.out/'build.json').read_text())

if __name__=='__main__':main()

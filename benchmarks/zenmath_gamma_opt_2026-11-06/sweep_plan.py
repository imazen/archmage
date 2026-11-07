#!/usr/bin/env python3
"""Immutable eight-cell finalist plan; exact binary identity; durable failure and audit gate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import tarfile
import subprocess
import sys


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(1<<20):h.update(block)
    return h.hexdigest()


class ParentSignal(BaseException):
    def __init__(self,signum):self.signum=signum

def append_failure(directory,metadata):
    directory.mkdir(exist_ok=True)
    path=directory/'FAILURE.txt';original=path.read_bytes() if path.exists() else b''
    pending=directory/f'FAILURE.pending-{os.getpid()}'
    pending.write_bytes(original+(b'\n' if original else b'')+metadata.encode());os.replace(pending,path)

def run_logged(command,log_path,out):
    with log_path.open('xb') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        try:
            returncode=child.wait()
        except BaseException as error:
            if child.poll() is None:
                child.send_signal(error.signum if isinstance(error,ParentSignal) else signal.SIGTERM)
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
            append_failure(out,f'parent interruption={type(error).__name__}; signal={getattr(error,"signum",None)}; child returncode={child.returncode}\ncommand={json.dumps(command)}\n')
            raise
    if returncode:
        append_failure(out,f'child returncode={returncode}; signal={-returncode if returncode<0 else None}\ncommand={json.dumps(command)}\n')
        raise RuntimeError(f'child returncode={returncode}')

def validate_archive(identity,build):
    archive=identity/'source.tar.gz'
    if sha(archive)!=build['source_archive_sha256']:raise ValueError('retained source archive hash mismatch')
    expected=dict((name,digest) for digest,name in (line.split('  ',1) for line in (identity/'sources.sha256').read_text().splitlines()))
    seen=set()
    with tarfile.open(archive,'r|gz') as tar:
        for member in tar:
            name=member.name.removeprefix('source/')
            if name not in expected:continue
            if name in seen or not member.isfile():raise ValueError('archive source member identity')
            stream=tar.extractfile(member);h=hashlib.sha256()
            while block:=stream.read(1<<20):h.update(block)
            if h.hexdigest()!=expected[name]:raise ValueError('archived source bytes differ '+name)
            seen.add(name)
    if seen!=set(expected):raise ValueError('archived source missing')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('identity',type=Path);parser.add_argument('out',type=Path)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--execute-prepared',action='store_true');args=parser.parse_args()
    identity=args.identity.resolve();build=json.loads((identity/'build.json').read_text())
    binary=identity/'zenmath-gamma-opt';source=build['source_manifest_sha256'];binary_id=build['binary_sha256']
    if not os.access(binary,os.X_OK):raise ValueError('retained binary is not executable')
    if sha(binary)!=binary_id or sha(identity/'sources.sha256')!=source:raise ValueError('frozen identity mismatch')
    validate_archive(identity,build)
    if not build.get('timed_kernel_byte_match'):raise ValueError('missing timed numerical kernel byte-match evidence')
    cells=[dict(kind=k,family=f,degree=(13 if k==0 else 15) if f=='global' else 5,layout='estrin',width=w,
                p_bits='4019999a' if k==0 else '3ed55555',start=0x3db71074 if k==0 else 0x3b475041,end=0x3f800000,
                count=(0x3f800000-(0x3db71074 if k==0 else 0x3b475041)+1))
           for k in [0,1] for f,w in [('piece',0),('global',0),('global',4),('global',8)]]
    plan=dict(version=1,source=source,binary=binary_id,identity=str(identity),cells=cells)
    if args.execute_prepared:
        if json.loads((args.out/'plan.json').read_text())!=plan:raise ValueError('prepared plan differs')
        if any((args.out/f).exists() for f in ['SUCCESS.txt','FAILURE.txt']):raise ValueError('already executed plan')
        if any(p.is_dir() for p in args.out.iterdir()):raise ValueError('partial plan preserved; no implicit rerun')
    else:
        args.out.mkdir(exist_ok=False);(args.out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    if args.prepare_only:print(json.dumps(plan,indent=2));return
    rows=[]
    def terminated(signum,_frame):raise ParentSignal(signum)
    previous={s:signal.signal(s,terminated) for s in [signal.SIGTERM,signal.SIGINT]}
    try:
        for cell in cells:
            name=f"{cell['kind']}-{cell['family']}-e{cell['degree']}-w{cell['width']}";out=args.out/name
            cmd=[str(binary),'sweep',str(cell['kind']),cell['family'],str(cell['width']),str(out),source,binary_id]
            run_logged(cmd,args.out/(name+'.log'),out)
            cmd[1]='audit'
            run_logged(cmd,args.out/(name+'.audit.log'),out)
            if (out/'FAILURE.txt').exists():raise RuntimeError('failure marker after audit')
            summary=dict(line.split('=',1) for line in (out/'summary.txt').read_text().splitlines())
            rows.append(dict(cell,output=str(out),**summary))
        # Cross-path exact raw/aggregate comparison follows op/domain/source plan checks.
        for kind in [0,1]:
            group=[r for r in rows if r['kind']==kind]
            for key in ['raw_candidate','raw_reference','aggregate_candidate','aggregate_reference']:
                if len({r[key] for r in group})!=1:raise RuntimeError('cross-path digest disagreement '+key)
        (args.out/'index.json').write_text(json.dumps(dict(plan=plan,records=rows),indent=2)+'\n')
        (args.out/'SUCCESS.txt').write_text('8 AUDITED full-domain records; no FAILURE markers; all per-path and cross-path hashes equal\n')
    except BaseException as e:
        append_failure(args.out,f'{type(e).__name__}: signal={getattr(e,"signum",None)}; {e}\n')
        if isinstance(e,ParentSignal):raise SystemExit(128+e.signum)
        raise
    finally:
        for signum,handler in previous.items():signal.signal(signum,handler)
    print('COMPLETE eight full-domain finalist records; exact CORE output parity, audited coverage/checksums and cross-path hashes')

if __name__=='__main__':main()

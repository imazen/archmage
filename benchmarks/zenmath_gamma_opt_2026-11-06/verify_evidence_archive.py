#!/usr/bin/env python3
"""Compare every evidence archive member byte-for-byte by SHA256 against original files."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def sha(p):
    h=hashlib.sha256()
    with p.open('rb')as f:
        while block:=f.read(1<<20):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('archive',type=Path);p.add_argument('out',type=Path);a=p.parse_args();a.out.mkdir(exist_ok=False)
    expected={str(path.relative_to(a.source)):dict(sha256=sha(path),size=path.stat().st_size)for path in sorted(a.source.rglob('*'))if path.is_file()}
    seen=set()
    with tarfile.open(a.archive,'r|gz')as tar:
        for member in tar:
            if member.isdir():continue
            name=member.name.removeprefix('runs/')
            if name not in expected or name in seen or not member.isfile():raise ValueError('unexpected/duplicate/nonfile archive member '+name)
            stream=tar.extractfile(member);h=hashlib.sha256();size=0
            while block:=stream.read(1<<20):h.update(block);size+=len(block)
            if size!=expected[name]['size']or h.hexdigest()!=expected[name]['sha256']:raise ValueError('archived evidence byte mismatch '+name)
            seen.add(name)
    if seen!=set(expected):raise ValueError('archive missing original evidence')
    (a.out/'manifest.json').write_text(json.dumps(expected,indent=2)+'\n')
    report=dict(status='PASS',archive=str(a.archive),archive_sha256=sha(a.archive),source=str(a.source),verified_files=len(seen),verified_original_bytes=sum(v['size']for v in expected.values()),manifest_sha256=sha(a.out/'manifest.json'))
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()

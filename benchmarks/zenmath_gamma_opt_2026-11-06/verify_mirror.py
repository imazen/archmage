#!/usr/bin/env python3
"""Hash every retained evidence byte against mirror and each required archived source member."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while block:=f.read(1<<20):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('mirror',type=Path);p.add_argument('report',type=Path);p.add_argument('--skip',action='append',default=[]);a=p.parse_args()
    a.report.mkdir(exist_ok=False);rows=[]
    for path in sorted(a.source.rglob('*')):
        if not path.is_file():continue
        rel=path.relative_to(a.source)
        if any(str(rel)==s or str(rel).startswith(s+'/') for s in a.skip):continue
        mirror=a.mirror/rel;digest=sha(path)
        if not mirror.is_file() or mirror.stat().st_size!=path.stat().st_size or sha(mirror)!=digest:raise ValueError('mirror byte mismatch '+str(rel))
        rows.append(dict(path=str(rel),size=path.stat().st_size,sha256=digest))
    archives=[]
    for manifest in sorted(a.source.glob('*/identity/sources.sha256')):
        if any(str(manifest.relative_to(a.source)).startswith(s+'/') for s in a.skip):continue
        archive=manifest.parent/'source.tar.gz'
        if not archive.exists():archive=manifest.parent.parent/'source.tar.gz'
        if not archive.exists():raise ValueError('missing retained source archive '+str(manifest))
        expected={name:digest for digest,name in (line.split('  ',1) for line in manifest.read_text().splitlines())};seen=set()
        with tarfile.open(archive,'r|gz') as tar:
            for member in tar:
                name=member.name.removeprefix('./').removeprefix('source/')
                if name not in expected:continue
                if name in seen or not member.isfile():raise ValueError('archive duplicate/type')
                h=hashlib.sha256();stream=tar.extractfile(member)
                while block:=stream.read(1<<20):h.update(block)
                if h.hexdigest()!=expected[name]:raise ValueError('archived compiled source bytes '+name)
                seen.add(name)
        if seen!=set(expected):raise ValueError('archive missing required source')
        archives.append(dict(path=str(archive.relative_to(a.source)),sha256=sha(archive),verified_source_members=len(seen)))
    (a.report/'manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
    report=dict(status='PASS',source=str(a.source),mirror=str(a.mirror),files=len(rows),bytes=sum(r['size'] for r in rows),source_archives=archives,skipped=a.skip,manifest_sha256=sha(a.report/'manifest.json'))
    (a.report/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()

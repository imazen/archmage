#!/usr/bin/env python3
"""Verify complete archived chunks, numerical sources, binaries, index and log."""
import argparse
import hashlib
import gzip
import json
from pathlib import Path
import subprocess
import tarfile

p=argparse.ArgumentParser(description=__doc__)
for key in ['archive','root','report']:
    p.add_argument('--'+key,type=Path,required=True)
p.add_argument('--expected-archive-sha',required=True)
p.add_argument('--expected-index-sha',required=True)
a=p.parse_args()
if a.report.exists():
    raise ValueError('existing archive verification report; use a new path')
index_path=a.root/'final-audit/index.json'
if hashlib.sha256(index_path.read_bytes()).hexdigest()!=a.expected_index_sha:
    raise ValueError('index differs from accepted identity')
with a.archive.open('rb') as stream:
    archive_sha=hashlib.file_digest(stream,'sha256').hexdigest()
if archive_sha!=a.expected_archive_sha:
    raise ValueError('archive differs from accepted immutable identity')
index=json.loads(index_path.read_text())
if index['status']!='ALL-89-CELLS-AUDITED' or len(index['cells'])!=89 or (a.root/'final-audit/FAILURE.txt').exists():
    raise ValueError('matrix is not accepted')
base=str(a.root.relative_to(Path.home()))
required=set()
for cell in index['cells']:
    directory=str(Path(cell['path']).relative_to(Path.home()))
    required.update(directory+f'/chunk-{i:06}.tsv' for i in range(4096))
    required.update(directory+'/'+name for name in ['chunks.tsv','metadata.txt','summary.txt','identity.txt','invocations.jsonl'])
    logs=sorted(Path(cell['path']).parent.glob(Path(cell['path']).name+'.*.log'))
    if not logs:
        raise ValueError(f'missing original per-cell log: {cell["path"]}')
    required.update(str(log.relative_to(Path.home())) for log in logs)
chunk_count=sum('/chunk-' in name for name in required)
checks={base+'/final-audit/index.json':hashlib.sha256(index_path.read_bytes()).hexdigest(),base+'/final-audit/audit-comparison.log':hashlib.sha256((a.root/'final-audit/audit-comparison.log').read_bytes()).hexdigest()}
for directory,filename,expected in [
    ('binaries','zenmath-digests-x86_64-v3','cce89e4ff23448d76c042c07c499e5557a80bfa13552bc262b3db18a263f8a1c'),
    ('binaries','zenmath-digests-simd128.wasm','c9a3936ad6e193be0d868f2c803d2249962400ccf9d8ad014e5c917006255819'),
    ('incoming-arm/binaries','zenmath-digests-aarch64','0ec78aeeac918d3ea3ebabffcc5e756667d40a4325d162ccdba4ed13eb713572'),
    ('incoming-v4x/binaries','zenmath-digests-v4x','255c1bfe4a6b97584eb878c37b06f3a18b3309812b76f43843808360d9fdd368'),
    ('binaries','wasmtime49','04c48a2507d614209dfbaa45397e842a5c79f64fe75e5bd4d3b95cdf667fec5d')]:
    checks[base+'/'+directory+'/'+filename]=expected
for directory in ['build-x86-v3','build-wasm','incoming-arm/build-portable','incoming-v4x/build-v4x']:
    checks[base+'/'+directory+'/sources.sha256']=index['source_sha256']
    required.update(base+'/'+directory+'/'+name for name in ['build.json','rustc.txt'])
manifest=(a.root/'build-x86-v3/sources.sha256').read_text()
for line in manifest.splitlines():
    digest,relative=line.split('  ',1)
    checks['tmp/zenmath-digests-sol/source/'+relative]=digest
required.update(checks)
# Hash every originally audited chunk and auxiliary/log file, not presence alone.
for name in required:
    if name not in checks:
        original=Path.home()/name
        with original.open('rb') as stream:
            checks[name]=hashlib.file_digest(stream,'sha256').hexdigest()
seen=set()
verified={}
subprocess.run(['zstd','-t',str(a.archive)],check=True)
process=subprocess.Popen(['zstd','-qdc',str(a.archive)],stdout=subprocess.PIPE)
try:
    with tarfile.open(fileobj=process.stdout,mode='r|') as archive:
        for member in archive:
            name=member.name.removeprefix('./')
            if name in required:
                if not member.isfile():raise ValueError(f'required regular file missing: {name}')
                if name in seen:raise ValueError(f'duplicate archive entry {name}')
                seen.add(name)
            if name in checks:
                stream=archive.extractfile(member)
                digest=hashlib.file_digest(stream,'sha256').hexdigest()
                if digest!=checks[name]:raise ValueError(f'archived bytes differ: {name}')
                verified[name]=digest
    if process.wait()!=0:raise ValueError('zstd decompression failed')
except BaseException:
    if process.poll() is None:process.terminate()
    process.wait()
    raise
missing=required-seen
if missing:raise ValueError(f'missing archive entries: {sorted(missing)[:20]}')
manifest=a.report.with_suffix('.files.sha256.gz')
if manifest.exists():raise ValueError('existing per-file verification manifest')
manifest_hash=hashlib.sha256()
with gzip.open(manifest,'xb') as stream:
    for name,digest in sorted(verified.items()):
        line=f'{digest}  {name}\n'.encode()
        manifest_hash.update(line)
        stream.write(line)
with manifest.open('rb') as stream:compressed_manifest_sha=hashlib.file_digest(stream,'sha256').hexdigest()
report=dict(status='COMPLETE-ARCHIVE-BYTES-VERIFIED',archive_sha256=archive_sha,archive_bytes=a.archive.stat().st_size,index_sha256=a.expected_index_sha,cells=89,chunk_files=chunk_count,required_members=len(required),verified_file_hashes=len(verified),original_cell_logs=sum(name.endswith('.log') for name in required),verification='SHA256 of every required archived file equals original audited file',files_manifest=str(manifest),files_manifest_uncompressed_sha256=manifest_hash.hexdigest(),files_manifest_sha256=compressed_manifest_sha,missing=[])
a.report.write_text(json.dumps(report,indent=2)+'\n')
print(f'Archive bytes verified:89cells,{chunk_count}chunkfiles,{len(verified)}file hashes including original logs; SHA256={archive_sha}')

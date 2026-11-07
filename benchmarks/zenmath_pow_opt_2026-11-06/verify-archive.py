#!/usr/bin/env python3
"""Write or verify the full archive inventory, including exact path coverage."""
import hashlib
from pathlib import Path
import sys
root=Path(sys.argv[2]).resolve();manifest=root/'MANIFEST.sha256'
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(1024*1024):h.update(block)
    return h.hexdigest()
files=sorted(p for p in root.rglob('*')if p.is_file()and p!=manifest)
assert files and not any(p.is_symlink()for p in files)
if sys.argv[1]=='write':
    with manifest.open('x') as f:
        for p in files:f.write(digest(p)+'  '+str(p.relative_to(root))+'\n')
else:
    assert sys.argv[1]=='verify'
    expected={}
    for line in manifest.read_text().splitlines():
        h,name=line.split('  ',1);assert name not in expected;expected[name]=h
    assert set(expected)=={str(p.relative_to(root))for p in files}
    for p in files:assert digest(p)==expected[str(p.relative_to(root))],p
print(f'PASS {sys.argv[1]} {len(files)} files, {sum(p.stat().st_size for p in files)} bytes; manifest SHA256 {digest(manifest)}')

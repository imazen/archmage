#!/usr/bin/env python3
"""Seal immutable evidence files or verify their independent mirror."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('mode', choices=['create', 'verify'])
p.add_argument('directory', type=Path)
p.add_argument('manifest', type=Path)
a = p.parse_args()

def identity(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024*1024), b''):
            h.update(chunk)
    return {'bytes': path.stat().st_size, 'sha256': h.hexdigest()}

if a.mode == 'create':
    assert not a.manifest.exists()
    files = {}
    for path in sorted(a.directory.rglob('*')):
        if path.is_file():
            assert not path.is_symlink(), 'freeze regular reference files before sealing'
            files[str(path.relative_to(a.directory))] = identity(path)
    a.manifest.write_text(json.dumps({'files': files}, indent=2)+'\n')
else:
    files = json.loads(a.manifest.read_text())['files']
    for name, expected in files.items():
        relative = Path(name)
        assert not relative.is_absolute() and '..' not in relative.parts
        assert identity(a.directory/relative) == expected, name
print(json.dumps({'files': len(files), 'manifest_sha256': identity(a.manifest)['sha256'],
                  'mode': a.mode, 'result': 'PASS'}))

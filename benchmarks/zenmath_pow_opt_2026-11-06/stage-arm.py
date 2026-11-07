#!/usr/bin/env python3
"""Strip debug data from copied ARM executables; preserve original build identity."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

source, out = map(Path, sys.argv[1:])
out.mkdir(parents=True, exist_ok=False)
(out / 'binaries').mkdir()
identity = json.loads((source / 'identity.json').read_text())
for row in identity['binaries']:
    old = source / row['archived_as']
    assert hashlib.sha256(old.read_bytes()).hexdigest() == row['sha256']
    dest = out / 'binaries' / Path(row['path']).name
    subprocess.run(['strip', '--strip-debug', '-o', str(dest), str(old)], check=True)
    row['original_sha256'] = row['sha256']
    row['original_archived_as'] = row['archived_as']
    row['derivation'] = 'strip --strip-debug (copied executable; original retained)'
    row['sha256'] = hashlib.sha256(dest.read_bytes()).hexdigest()
    row['bytes'] = dest.stat().st_size
    row['archived_as'] = str(dest.relative_to(out))
(out / 'identity.json').write_text(json.dumps(identity, indent=2) + '\n')
for name in ['source.sha256', 'source.tar.gz']:
    shutil.copy2(source / name, out / name)
print(sum(row['bytes'] for row in identity['binaries']), 'total executable bytes')

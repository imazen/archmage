#!/usr/bin/env python3
"""Verify the independently frozen Rust baseline has not been reformatted/edited."""
from pathlib import Path
import hashlib
import json
root = Path(__file__).resolve().parent / 'baseline'
manifest = json.loads((root / 'PROVENANCE.json').read_text())
assert manifest['revision'] == '4ccaf1cf'
for record in manifest['files']:
    path = root / record['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], path
print(f"PASS {len(manifest['files'])} frozen baseline Rust files")

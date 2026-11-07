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

new=root/'zenmath-before-gamma'
record=json.loads((new/'PROVENANCE.json').read_text())
assert record['revision']=='9f466de6581a0837bf2fa1af8ce3538b929e31e6'
for row in record['files']:
    assert hashlib.sha256((new/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
print(f"PASS {len(record['files'])} exact pre-gamma Rust files")

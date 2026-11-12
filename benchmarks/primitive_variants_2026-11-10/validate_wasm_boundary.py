#!/usr/bin/env python3
"""Require strict module acceptance and relaxed-op rejection without the feature."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument('modules', type=Path)
a = p.parse_args()
print(subprocess.check_output(['wasmtime', '--version'], text=True), flush=True)
for kind in ('strict', 'relaxed'):
    module = a.modules / (kind + '.wasm')
    digest = hashlib.sha256(module.read_bytes()).hexdigest()
    command = ['wasmtime', 'run', '-Wrelaxed-simd=n', str(module)]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(json.dumps(dict(command=command, sha256=digest, returncode=result.returncode)), flush=True)
    print(result.stdout, flush=True)
    if kind == 'strict':
        assert result.returncode == 0, 'strict module requires an undeclared feature'
    else:
        assert result.returncode != 0 and 'relaxed SIMD support is not enabled' in result.stdout

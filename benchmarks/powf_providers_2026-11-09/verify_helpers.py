#!/usr/bin/env python3
"""Negative controls for frozen identities, executed code and malformed input."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from core_trace import prove
from external import load_manifest

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('native', type=Path)
p.add_argument('directory', type=Path)
a = p.parse_args()
a.directory.mkdir(parents=True, exist_ok=False)
native = a.native.resolve()
truncated = subprocess.run([str(native)], input='E system_C_powf 4 3f800000 40000000\n',
                           text=True, capture_output=True)
(a.directory/'truncated.stdout').write_text(truncated.stdout)
(a.directory/'truncated.stderr').write_text(truncated.stderr)
assert truncated.returncode != 0 and not truncated.stdout
assert 'truncated E input' in truncated.stderr
identity = a.directory/'identity'
identity.write_text('frozen\n')
manifest = {'workers': [{'binary': str(identity), 'sha256': hashlib.sha256(identity.read_bytes()).hexdigest()}],
            'dsos': {'POWF_CORE_MATH_DSO': {'path': str(identity), 'sha256': hashlib.sha256(identity.read_bytes()).hexdigest()}}}
path = a.directory/'manifest.json'
path.write_text(json.dumps(manifest))
os.environ['POWF_AOCL_DSO'] = 'stale selection'
load_manifest(path)
assert 'POWF_AOCL_DSO' not in os.environ and os.environ['POWF_CORE_MATH_DSO'] == str(identity)
identity.write_text('changed\n')
try:
    load_manifest(path)
except AssertionError as error:
    assert str(error) == 'frozen DSO changed'
else:
    raise AssertionError('changed DSO accepted')
identity.write_text('frozen\n')
manifest['dsos'] = {}
manifest['resolved_native_dsos'] = {str(identity): {'sha256': '0'*64}}
path.write_text(json.dumps(manifest))
try:
    load_manifest(path)
except AssertionError as error:
    assert str(error) == 'resolved native DSO changed'
else:
    raise AssertionError('changed resolved system DSO accepted')
trace = a.directory/'synthetic.trace'
trace.write_text('IN: \n0x1000: \nOBJD-T: c4e2d19901\nTrace 0: 0x123 [00000000/00001000/0/0]\n')
assert prove(trace)['executed_FMA_blocks'] == [0x1000]
trace = a.directory/'not-executed.trace'
trace.write_text('IN: \n0x1000: \nOBJD-T: c4e2d19901\nTrace 0: 0x123 [00000000/00002000/0/0]\n')
try:
    prove(trace)
except AssertionError as error:
    assert 'translation alone' in str(error)
else:
    raise AssertionError('unexecuted translation accepted')
print('PASS: malformed input, frozen DSO selection/hash and executed-FMA negative controls')

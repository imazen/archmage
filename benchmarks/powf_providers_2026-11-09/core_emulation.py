#!/usr/bin/env python3
"""Require actual FMA-without-AVX2 selection, then check the frozen corpus."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
from external import load_manifest
from core_trace import prove

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('manifest', type=Path)
p.add_argument('fixtures', type=Path)
p.add_argument('directory', type=Path)
p.add_argument('--qemu', default='qemu-x86_64')
a = p.parse_args()
a.directory.mkdir(parents=True, exist_ok=False)
manifest = load_manifest(a.manifest)
native = manifest['workers'][1]
command = [a.qemu, '-cpu', 'Haswell,-avx2', native['binary']]
listing = subprocess.run(command + ['--list'], capture_output=True, text=True, check=True)
(a.directory/'list.stdout').write_text(listing.stdout)
(a.directory/'list.stderr').write_text(listing.stderr)
assert 'CORE_MATH_sys_1_4_0_FMA_no_AVX2 1\n' in listing.stdout
assert 'glibc_2_43_libmvec_AVX2x8' not in listing.stdout
# These names are guarded by usable AVX+FMA and AVX2, respectively.
wrapper = (a.directory/'native-wrapper').resolve()
wrapper.write_text('#!/bin/sh\nexec ' + shlex.join(command) + ' "$@"\n')
wrapper.chmod(0o755)
manifest['workers'][1] = dict(native, binary=str(wrapper), sha256=hashlib.sha256(wrapper.read_bytes()).hexdigest())
manifest['providers'] = [r for r in manifest['providers'] if r['worker'] == 1]
manifest['emulation'] = {'command': command, 'underlying_worker': native,
                         'claim': 'correctness only; no speed evidence'}
frozen = a.directory/'manifest.json'
frozen.write_text(json.dumps(manifest, indent=2)+'\n')
subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('accuracy.py')),
                str(frozen), str(a.fixtures), str(a.directory/'accuracy')], check=True)
trace = a.directory/'execution.trace'
run = subprocess.run([a.qemu, '-cpu', 'Haswell,-avx2', '-d', 'in_asm,exec,nochain',
                      '-D', str(trace), native['binary']],
                     input='E CORE_MATH_sys_1_4_0_FMA_no_AVX2 1 3f800001 4019999a\n',
                     capture_output=True, text=True, check=True)
(a.directory/'trace.stdout').write_text(run.stdout)
(a.directory/'trace.stderr').write_text(run.stderr)
assert trace.stat().st_size < 64*1024*1024
proof = dict(prove(trace), usable_AVX_and_FMA=True, usable_AVX2=False,
             feature_evidence='guarded provider listing')
(a.directory/'proof.json').write_text(json.dumps(proof, indent=2)+'\n')
print(json.dumps(proof, indent=2))

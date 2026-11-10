#!/usr/bin/env python3
"""Freeze the exact standalone Arm C profiles and system powf comparison."""
import argparse
import json
import os
from pathlib import Path
from external import discover, ARM_C_ENV

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('rust',type=Path);p.add_argument('native',type=Path)
p.add_argument('build',type=Path);p.add_argument('output',type=Path)
p.add_argument('--baseline-only',action='store_true')
a=p.parse_args();assert not a.output.exists()
build=json.loads(a.build.read_text())
for name in list(os.environ):
    if (name.startswith('POWF_')and name.endswith('DSO'))or name in ARM_C_ENV:os.environ.pop(name)
for profile in build['builds']:
    if a.baseline_only and 'separate' not in profile['name']:continue
    os.environ[profile['environment']]=profile['binary']
if a.baseline_only:
    os.environ['GLIBC_TUNABLES']='glibc.cpu.hwcaps=-AVX2,-FMA'
else:os.environ.pop('GLIBC_TUNABLES',None)
x=discover(a.rust,a.native)
x['providers']=[p for p in x['providers']if p['method']in ['std','system_C_powf']or p['method'].startswith('Arm_powf_C_')]
assert any(p['method'].startswith('Arm_powf_C_')for p in x['providers'])
x.update(study='ARM-C-PLAN.md; additional pre-port comparison',build=build,
         native_identity=None if a.baseline_only else build['native_identity'],
         runtime_env={'GLIBC_TUNABLES':os.environ.get('GLIBC_TUNABLES')})
a.output.write_text(json.dumps(x,indent=2)+'\n')

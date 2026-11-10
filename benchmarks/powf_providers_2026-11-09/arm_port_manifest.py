#!/usr/bin/env python3
"""Select C/glibc and Rust graph arms for the additional port experiment."""
import argparse
import json
from pathlib import Path
from external import discover, digest, load_manifest

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('c_manifest',type=Path);p.add_argument('rust',type=Path);p.add_argument('output',type=Path)
p.add_argument('--native',type=Path)
p.add_argument('--gamma',action='store_true',help='include the explicitly restricted Gamma assumed-input arm')
a=p.parse_args();assert not a.output.exists()
c=load_manifest(a.c_manifest)
x=discover(a.rust,Path(c['workers'][1]['binary']))
x['providers']=[v for v in x['providers']if v['method']in ['std','system_C_powf']or v['method'].startswith('Arm_powf_C_')]
methods=['Arm_Rust_fused_cold','Arm_Rust_separate_cold','Arm_Rust_fused_warm','Arm_Rust_fused_unrolled']
if a.gamma:methods+=['Arm_Rust_fused_assume_Gamma','Arm_Rust_assume_Gamma_unrolled']
x['providers'] +=[{'label':m+'_Rust_baseline','method':m,'worker':0,'lanes':1}for m in methods]
if a.native:
    x['workers'].append({'name':'Rust_native_owner_diagnostic','binary':str(a.native.resolve()),'sha256':digest(a.native)})
    x['providers'] +=[{'label':m+'_Rust_native','method':m,'worker':2,'lanes':1}for m in ['std',*methods]]
x.update(study='ARM-C-PLAN.md; Rust port after completed C check',
         native_identity=c.get('native_identity'),runtime_env=c['runtime_env'],
         numerical_domain='Gamma checked by independent preflight'if a.gamma else'full f32 inputs; RNE and gradual underflow',
         accuracy_contract='zero non-NaN mismatches to matching C graph; compare MPFR separately',
         C_build=c['build'])
a.output.write_text(json.dumps(x,indent=2)+'\n')

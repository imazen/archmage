#!/usr/bin/env python3
"""Freeze explicitly selected portable evaluators alongside the physical C reference."""
import argparse
import json
from pathlib import Path
import sys
from external import digest, load_manifest

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('reference',type=Path);p.add_argument('contexts',type=Path);p.add_argument('directory',type=Path)
a=p.parse_args();a.directory.mkdir(parents=True,exist_ok=False)
x=load_manifest(a.reference)
x['providers']=[v for v in x['providers']if v['method']in ['std','system_C_powf','Arm_powf_C_AVX2_FMA_errno1','Arm_powf_C_x86_64_separate_errno1']and v['worker']<2]
methods=['Arm_Rust_fused_cold','Arm_Rust_separate_cold','Arm_Rust_fused_warm']
files={}
for context in json.loads(a.contexts.read_text()):
    command=context['command']
    assert command and all(isinstance(v,str)for v in command)
    for filename in context['files']:
        path=Path(filename).resolve();files[str(path)]={'sha256':digest(path)}
    wrapper=(a.directory/(context['name']+'.py')).resolve()
    wrapper.write_text('#!'+sys.executable+'\nimport os\nos.execv('+repr(command[0])+','+repr(command)+')\n')
    wrapper.chmod(0o755)
    index=len(x['workers'])
    x['workers'].append({'name':context['name'],'binary':str(wrapper),'sha256':digest(wrapper),'command':command})
    x['providers'] +=[{'label':m+'_'+context['name'],'method':m,'worker':index,'lanes':1}for m in methods]
x.update(referenced_files=files,contexts=json.loads(a.contexts.read_text()),claim='numerical execution only; emulation supplies no timing evidence')
(a.directory/'manifest.json').write_text(json.dumps(x,indent=2)+'\n')

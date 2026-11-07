#!/usr/bin/env python3
"""Extract exact linked functions from the measured binary, retaining full disassembly."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
binary=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=False)
with (out/'linked.disasm').open('w')as f:
    subprocess.run(['objdump','-Cd',str(binary)],stdout=f,check=True)
lines=(out/'linked.disasm').read_text().splitlines(keepends=True)
functions=[];start=None;symbol=None;address=None
patterns=['gamma_timing::fixed','gamma_timing::power','gamma_timing::whole','gamma_timing::context','zenmath::simd::__array::pow_f32x4','zenmath::simd::__scalar_transcendentals::powf','__attune_contexts_','zenmath::simd::__slice::pow_f32']
for i,line in enumerate(lines+['0 <END>:\n']):
    match=re.match(r'^([0-9a-f]+) <(.+)>:',line)
    if match:
        if start is not None and any(p in symbol for p in patterns):
            name=f'function-{len(functions):03}.s';(out/name).write_text(''.join(lines[start:i]))
            functions.append(dict(symbol=symbol,address=address,start_line=start+1,file=name))
        start=i;address=match[1];symbol=match[2]
assert functions,'required linked symbols not found'
meta=dict(binary=str(binary),binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),disassembly_sha256=hashlib.sha256((out/'linked.disasm').read_bytes()).hexdigest(),functions=functions,note='Exact linked binary. Functions include guards and complete rescue; instruction totals are not hot-path throughput estimates.')
if len(sys.argv)>3:
    timed=Path(sys.argv[3]).resolve()
    meta['timed_binary_sha256']=hashlib.sha256(timed.read_bytes()).hexdigest()
    meta['timed_binary']=str(timed)
    sections={}
    for section in ['.text','.rodata']:
        outputs=[]
        for label,source in [('linked',binary),('timed',timed)]:
            path=out/(label+section+'.bin')
            subprocess.run(['objcopy','--only-section='+section,'-O','binary',str(source),str(path)],check=True)
            outputs.append(path.read_bytes())
        assert outputs[0]==outputs[1],section
        sections[section]=dict(bytes=len(outputs[0]),sha256=hashlib.sha256(outputs[0]).hexdigest())
    meta['identical_sections']=sections
(out/'identity.json').write_text(json.dumps(meta,indent=2)+'\n')
print(f'PASS {len(functions)} exact linked function extracts')

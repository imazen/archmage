#!/usr/bin/env python3
"""Retain selected functions and line references from the exact timed module WAT."""
import hashlib
import json
from pathlib import Path
import sys
wat,module,out=map(Path,sys.argv[1:]);out.mkdir(parents=True,exist_ok=False)
lines=wat.read_text().splitlines(keepends=True);rows=[]
for i,line in enumerate(lines):
    if not line.startswith('  (func '):continue
    label=None
    if 'try_gamma' in line:label='scalar-certificate'
    if 'pow_f32x4_wasm128' in line and 'zenmath4simd' in line:label='packed-entry'
    if 'Wasm128Token' in line:
        for name,key in [('packed-certificate','gamma9candidate'),('decode-polynomial','polynomials6decode'),('encode-polynomial','polynomials6encode')]:
            if key in line:label=name
    if label is None:continue
    end=i+1
    while end<len(lines)and not lines[end].startswith(('  (func ','  (data ','  (elem ')):end+=1
    text=''.join(lines[i:end]);path=out/(label+'.wat');assert not path.exists();path.write_text(text)
    rows.append(dict(label=label,start_line=i+1,end_line=end,symbol=line.strip(),sha256=hashlib.sha256(text.encode()).hexdigest(),calls=sum('call 'in x for x in lines[i:end]),packed_mul=sum('f64x2.mul'in x for x in lines[i:end]),packed_add=sum('f64x2.add'in x for x in lines[i:end])))
assert len(rows)==5,rows
(out/'identity.json').write_text(json.dumps(dict(module_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),wat_sha256=hashlib.sha256(wat.read_bytes()).hexdigest(),functions=rows),indent=2)+'\n')
print('PASS five exact-module functions, with original WAT line references')

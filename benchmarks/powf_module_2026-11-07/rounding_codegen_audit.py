#!/usr/bin/env python3
"""Verify separate probe graphs contain no fused arithmetic instructions."""
import argparse
import importlib.util
import json
from pathlib import Path
import re

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('assembly',type=Path)
p.add_argument('output',type=Path)
p.add_argument('--macho',action='store_true')
p.add_argument('--wasm',action='store_true')
a=p.parse_args()
root=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('archmage_codegen',root/'xtask/codegen.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
source=a.assembly.read_text()
if a.macho:
    functions={}
    for name,body in re.findall(r'^([A-Za-z_][^:\s]*):[^\n]*\n(Lfunc_begin\d+:\n.*?)^Lfunc_end\d+:',source,re.M|re.S):
        lines=[]
        for line in body.splitlines():
            if re.match(r'^\t[a-z][a-z0-9_.]*(?:\s|$)',line):lines.append(line.strip().split(';',1)[0].rstrip())
            elif line.endswith(':'):lines.append('LABEL '+line[:-1])
            elif line.strip() and not line.lstrip().startswith(('.', ';', '#', '//')):raise ValueError((name,line))
        assert lines
        functions[name]=lines
else:
    functions=module.parse(source)
rows=[]
for name in functions:
    match=re.search(r'roundprobe_(lowp|midp)_(fused|separate)_x(4|8|16)_(scalar|v3|neon|wasm128)',name)
    if not match:continue
    body=module.expand(name,functions)
    instructions=[line for line in body if not line.startswith('LABEL ')]
    fused=[line for line in instructions if re.match(r'^(v?f(?:madd|msub|nmadd|nmsub)|fn?mla|fn?mls|f32x4\.relaxed_(?:madd|nmadd))',line)]
    multiply=[line for line in instructions if re.match(r'^(v?mul(?:ps|ss)|fmul|f32x4\.mul|f32\.mul)\b',line)]
    adds=[line for line in instructions if re.match(r'^(v?add(?:ps|ss)|fadd|f32x4\.add|f32\.add)\b',line)]
    row=dict(zip(['family','rounding','width','tier'],match.groups()))|dict(symbol=name,instructions=len(instructions),fused=fused,multiplications=len(multiply),additions=len(adds),body=body)
    if row['rounding']=='separate':
        assert not fused, ('unexpected fusion',name,fused)
        assert multiply and adds, ('no separate arithmetic found',name)
    elif row['tier'] in ['v3','neon']:
        assert fused, ('native fused probe has no FMA instructions',name)
    rows.append(row)
expected=24
assert len(rows)==expected, ('missing actual probe bodies',len(rows),expected)
assert len({(r['family'],r['rounding'],r['width'],r['tier'])for r in rows})==expected
assert not a.output.exists()
a.output.write_text(json.dumps(dict(assembly=str(a.assembly),rows=rows),indent=2)+'\n')
print('24 selected bodies checked; every separate graph has multiply/add and zero FMA instructions')

#!/usr/bin/env python3
"""Report normalized caller instructions/local callees; std externals excluded."""
import argparse
import importlib.util
import json
from pathlib import Path
import re
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('assembly',type=Path)
p.add_argument('output',type=Path)
a=p.parse_args()
root=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('archmage_codegen',root/'xtask/codegen.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
functions=module.parse(a.assembly.read_text())
rows=[]
for name in functions:
    match=re.search(r'probe_(const_caller|const_auto|runtime_literal_caller|runtime_literal_auto|runtime_dynamic|midp_unchecked|midp|lowp|std_f32|std_f64)_x(4|8|16)_(PositiveNormal|NonNegativeNormal|Gamma)_(literal|dynamic)_(scalar|v3|v4x|v4|neon|wasm128)',name)
    if not match:continue
    body=module.expand(name,functions)
    instructions=[line for line in body if not line.startswith('LABEL ')]
    external=[line for line in instructions if module.opcode(line)in['callq','call','bl','return_call'] and not line.startswith('INTERNAL_CALL')]
    branches=[line for line in instructions if re.match(r'^(j[a-z]+|b\.[a-z]+|cbnz|cbz|tbz|tbnz|br_if)\b',line)]
    rows.append(dict(zip(['candidate','width','domain','exponent','tier'],match.groups()))|{'symbol':name,'instructions_with_local_callees':len(instructions),'conditional_or_unconditional_branches':len(branches),'external_calls':external,'body':body})
assert rows,'no caller probes parsed; refusing vacuous report'
summary={'assembly':str(a.assembly),'instructions_include_local_callees':True,'external_library_instructions_unmeasured':True,'rows':rows}
a.output.write_text(json.dumps(summary,indent=2)+'\n')
print('caller probes',len(rows),'external library instruction work excluded')

#!/usr/bin/env python3
"""Preserve disassembly and instruction counts; counts are not runtime timings."""
import argparse
import json
from pathlib import Path
import re
import subprocess
p=argparse.ArgumentParser(description=__doc__);p.add_argument('builds',type=Path);a=p.parse_args();rows=[]
for profile in ['release','thin']:
 for mode in ['direct','facade','owned-compat']:
  root=a.builds/f'{profile}-1-{mode}';binary=root/'target'/profile/'magemath-layer-consumer'
  dis=subprocess.check_output(['objdump','-dC','--no-show-raw-insn',str(binary)],text=True)
  (root/'codegen.asm').write_text(dis)
  functions={};current=None
  for line in dis.splitlines():
   m=re.match(r'^[0-9a-f]+ <(.*)>:$',line)
   if m:
    name=m[1];current=name if ('magemath_layer_consumer::'in name or re.search(r'(?:magetypes|magemath)::(?:powf|expf|logf)::',name))else None
    if current:functions[current]=[]
   elif current:
    m=re.match(r'^\s+[0-9a-f]+:\s+(.+)$',line)
    if m and not re.match(r'(?:[a-z]+\s+)?nop|int3|data16|cs nop',m[1]):functions[current].append(m[1])
  row=dict(profile=profile,mode=mode,functions={n:dict(instructions=len(v),calls=sum(bool(re.match(r'call',i))for i in v))for n,v in functions.items()})
  row['total_instructions']=sum(f['instructions']for f in row['functions'].values());rows.append(row)
(a.builds/'codegen-summary.json').write_text(json.dumps(rows,indent=2)+'\n')
for r in rows:print(r['profile'],r['mode'],len(r['functions']),r['total_instructions'])

#!/usr/bin/env python3
"""Count actual whole-loop instructions; retain branch/call bodies and symbol IDs."""
import argparse,hashlib,json,re,subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('binary',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
text=subprocess.check_output(['objdump','--no-show-raw-insn','-dC',str(a.binary)],text=True)
(a.output/'disassembly.txt').write_text(text)
functions={};current=None
for line in text.splitlines():
 m=re.match(r'^([0-9a-f]+) <(.+)>:$',line)
 if m:current=int(m[1],16);functions[current]={'name':m[2],'lines':[]}
 elif current is not None and re.match(r'^\s*[0-9a-f]+:',line):functions[current]['lines'].append(line)
rows=[]
for line in subprocess.check_output(['nm','-C',str(a.binary)],text=True).splitlines():
 m=re.match(r'^([0-9a-f]+) [tT] (midlow_bench::__arcane_loop(4|8|16)_(v3|v4x)::<(\d+), (\d+)>)$',line)
 if not m:continue
 f=functions[int(m[1],16)];instructions=[x for x in f['lines']if not re.search(r':\s*(nop|data16|cs |int3)',x)]
 rows.append({'symbol':m[2],'width':int(m[3]),'context':m[4],'flags':int(m[5]),'arm':int(m[6]),'instructions':len(instructions),'calls':[x.strip()for x in instructions if re.search(r':\s*call',x)],'address':m[1]})
assert rows
(a.output/'counts.json').write_text(json.dumps({'binary':str(a.binary),'sha256':hashlib.sha256(a.binary.read_bytes()).hexdigest(),'scope':'complete stream/chain/tail loop, including all branches; not dynamic hot-path instruction counts','rows':rows},indent=2)+'\n')
for r in rows:
 if r['context']=='v3' and r['width']==8 and ((r['flags']in[38,46,166,174,182,190])or r['arm']!=0):print(json.dumps(r))

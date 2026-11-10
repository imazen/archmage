#!/usr/bin/env python3
"""Exact normalized assembly comparison against unchanged log/exp composition."""
import argparse,importlib.util,json,re
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('assembly',type=Path);p.add_argument('output',type=Path);p.add_argument('--macho',action='store_true');p.add_argument('--record-only',action='store_true');p.add_argument('--baseline',type=Path);a=p.parse_args()
spec=importlib.util.spec_from_file_location('codegen',Path(__file__).resolve().parents[2]/'xtask/codegen.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=re.sub(r"Cs[0-9a-zA-Z]+_", "CsHASH_",a.assembly.read_text());rows=[]
if not a.macho:funcs=m.parse(s)
else:
 funcs={}
 data={name:re.sub(r';[^\n]*','',body).strip()for name,body in re.findall(r'^(lCPI[^:\s]+):[^\n]*\n((?:[ \t]*\.(?:long|quad|word|xword|byte|short|zero)\s+[^\n]+\n)+)',s,re.M)}
 import hashlib
 for name,body in re.findall(r'^([A-Za-z_][^:\s]*):[^\n]*\n(\t.cfi_startproc.*?\t.cfi_endproc)',s,re.M|re.S):
  labels=re.findall(r'^(LBB\w+):',body,re.M);label_ids={label:f'BLOCK_{i}'for i,label in enumerate(labels)};lines=[]
  for line in body.splitlines():
   if line.strip().removesuffix(':')in label_ids:lines.append('LABEL '+label_ids[line.strip().removesuffix(':')])
   elif re.match(r'^\t[a-z][a-z0-9_.]*(?:\s|$)',line):
    line=line.strip().split(';',1)[0].rstrip()
    line=re.sub(r'\bLBB\w+\b',lambda m:label_ids[m[0]],line)
    line=re.sub(r'\blCPI[\w.]+',lambda m:'DATA_'+hashlib.sha256(data[m[0]].encode()).hexdigest(),line)
    lines.append(line)
  funcs[name]=lines
 aliases=dict(re.findall(r'^\s*(\S+)\s*=\s*(\S+)\s*$',s,re.M))
 def resolve(name,active=()):
  assert name not in active
  return funcs[name]if name in funcs else resolve(aliases[name],(*active,name))
 for name in aliases:funcs[name]=resolve(name)

for name,body in re.findall(r'^([^:\s]*REMAPTABLE_X(?:4|8|16)_[^:\s]*):[^\n]*\n((?:\s*\.(?:quad|xword|int32)\s+[^\n]+\n)+)',s,re.M):
 targets=re.findall(r'\.(?:quad|xword|int32)\s+(\S+)',body);assert len(targets)==2,(name,targets)
 before,after=[m.expand(t.rstrip(','),funcs)for t in targets]
 equal=before==after
 rows.append({'table':name,'targets':targets,'equal':equal,'original':before,'shared':after})
assert len(rows)in[36,72],('missing or unexpected pairs',len(rows))
if a.baseline:
 baseline=json.loads(a.baseline.read_text())
 lookup={r['table'].split('REMAPTABLE_',1)[1]:r for r in baseline['rows']}
 assert len(lookup)==len(rows)
 for r in rows:
  prior=lookup[r['table'].split('REMAPTABLE_',1)[1]]
  r['composition_equal']=r['equal']
  r['original']=prior['shared']
  r['equal']=r['original']==r['shared']
report={'pairs':len(rows),'equal':sum(r['equal']for r in rows),'baseline':str(a.baseline)if a.baseline else 'unchanged log/exp composition','normalization':'existing consumer codegen parser; constants, operands, branches, calls and instruction order retained','rows':rows}
a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'pairs':len(rows),'equal':report['equal'],'changed':[r['table']for r in rows if not r['equal']]}))
if not a.record_only:assert all(r['equal']for r in rows),'assembly differs; inspect retained report'

#!/usr/bin/env python3
"""Audit every public-policy probe and preserve aliases, ties and instruction counts."""
import argparse,importlib.util,json,re
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('assembly',type=Path);p.add_argument('output',type=Path);p.add_argument('--macho',action='store_true');p.add_argument('--wasm',action='store_true');a=p.parse_args()
spec=importlib.util.spec_from_file_location('codegen',Path(__file__).resolve().parents[2]/'xtask/codegen.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
source=a.assembly.read_text()
if a.macho:
 functions={}
 for name,body in re.findall(r'^([A-Za-z_][^:\s]*):[^\n]*\n(Lfunc_begin\d+:\n.*?)^Lfunc_end\d+:',source,re.M|re.S):
  lines=[]
  for line in body.splitlines():
   if re.match(r'^\t[a-z][a-z0-9_.]*(?:\s|$)',line):lines.append(line.strip().split(';',1)[0].rstrip())
   elif line.endswith(':'):lines.append('LABEL '+line[:-1])
   elif line.strip() and not line.lstrip().startswith(('.', ';', '#', '//')):raise ValueError((name,line))
  assert lines;functions[name]=lines
 aliases=dict(re.findall(r'^\s*(\S+)\s*=\s*(\S+)\s*$',source,re.M))
 def resolve(name,active=()):
  assert name not in active
  return functions[name]if name in functions else resolve(aliases[name],(*active,name))
 for name in aliases:functions[name]=resolve(name)
else:
 source=re.sub(r"Cs[0-9a-zA-Z]+_", "CsHASH_", source)
 functions=m.parse(source)
selected={};flags=[x for x in range(64)if x&3!=3]+[x for x in range(64,128)if x&16 and not x&8]
for name in functions:
 match=re.search(r'policyprobe_f(\d+)_x(4|8|16)_(scalar|v3|neon|wasm128)',name)
 if not match:continue
 f,n,t=match.groups();key=(int(f),int(n),t)
 if key not in selected or 'caller_policyprobe_'in name:selected[key]=name
# A used pointer table survives body merging without adding instructions to
# the probes. Its relocation order is the frozen option encoding order.
for name,body in re.findall(r'^([^:\s]*POLICYTABLE_X(?:4|8|16)_(?:SCALAR|V3|NEON|WASM128)[^:\s]*):[^\n]*\n((?:\s*\.(?:quad|xword|int32)\s+[^\n]+\n)+)',source,re.M):
 match=re.search(r'POLICYTABLE_X(4|8|16)_(SCALAR|V3|NEON|WASM128)',name)
 n,t=match.groups();targets=re.findall(r'\.(?:quad|xword|int32)\s+(\S+)',body)
 assert len(targets)==64,(name,len(targets))
 for f,target in zip(flags,targets):
  target=target.rstrip(',')
  assert target in functions,('missing retained table target',name,target)
  selected[(f,int(n),t.lower())]=target
expected={(f,n,t)for f in flags for n in [4,8,16]for t in ['scalar', 'wasm128'if a.wasm else 'neon'if a.macho else 'v3']}
assert set(selected)==expected,('missing probes',expected-set(selected),'extra',set(selected)-expected)
rows=[]
for(f,n,t),name in sorted(selected.items()):
 body=m.expand(name,functions);ins=[x for x in body if not x.startswith('LABEL ')]
 fmas=[x for x in ins if re.match(r'^(v?f(?:madd|msub|nmadd|nmsub)|fn?mla|fn?mls|f32x4\.relaxed_(?:madd|nmadd))',x)]
 branches=[x for x in ins if re.match(r'^(j(?!mp)[a-z]+|b\.[a-z]+|cbn?z|tbn?z|br_if)\b',x)]
 calls=[x for x in ins if re.match(r'^(callq?|bl|call)\b',x)]
 if not f&64 and f&16 and f&4 and t=='scalar':
  assert not fmas,('scalar BackendDependent Caller unexpectedly fused',name)
  assert not any('fmaf' in x or 'std_probe'in x or 'wasm_fma'in x for x in calls),(name,calls)
 if not f&64 and t in ['v3','neon']:assert fmas,('native approximation has no FMA',name)
 rows.append(dict(flags=f,width=n,tier=t,symbol=name,instructions=len(ins),conditional_branches=len(branches),fmas=len(fmas),calls=calls,body=body))
# Native LOWP's shorter Horner chain must survive the public delegates.
lookup={(r['flags'],r['width'],r['tier']):r for r in rows}
for(f,n,t),r in lookup.items():
 if f<64 and not f&8 and t in ['v3','neon']:
  assert lookup[(f|8,n,t)]['fmas']<r['fmas'],('LOWP did not shorten native fused graph',f,n,t)
assert not a.output.exists();a.output.write_text(json.dumps({'assembly':str(a.assembly),'rows':rows,'probe_count':len(rows),'claim':'static normalized counts exclude external callees; timing is separate'},indent=2)+'\n');print(f'{len(rows)} public policy probes audited, including compiler aliases')

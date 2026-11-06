#!/usr/bin/env python3
"""Retain reproducible size/build and normalized wrapper-codegen evidence."""
import json,re,sys
from pathlib import Path
root=Path(sys.argv[1]);report={}
def instructions(mode,name):
 text=(root/f'disassembly-{mode}.txt').read_text()
 m=re.search(r'^[0-9a-f]+ <codegen::'+name+r'>:\n(.*?)(?=\n\n)',text,re.M|re.S)
 if not m:return None
 result=[]
 for line in m[1].splitlines():
  parts=line.split('\t')
  if len(parts)<3:continue
  instruction=parts[-1]
  instruction=re.sub(r'[-]?0x[0-9a-f]+(?=\(%rip\))','RIP',instruction)
  instruction=instruction.replace(f'codegen::{name}','SELF')
  instruction=re.sub(r'\b[0-9a-f]+ (?=<)','ADDR ',instruction)
  result.append(instruction)
 return result
for mode in ['base','extra','o0']:
 binary=(root/mode).read_bytes();legacy=instructions(mode,'legacy');default=instructions(mode,'default_policy')
 assert legacy and default, (mode, 'missing legacy/default instructions')
 sections={}
 for line in (root/f'size-{mode}.txt').read_text().splitlines():
  fields=line.split()
  if len(fields)>=2 and fields[0] in ['.text','.rodata']:sections[fields[0]]=int(fields[1])
 fields=dict(re.findall(r'(\w+)=([0-9.]+)',(root/f'build-{mode}.txt').read_text()))
 report[mode]=dict(file_bytes=len(binary),sections=sections,build=fields,default_legacy_normalized_equal=legacy==default,debug_assertion_text_present=b'power input violates PositiveNormalPrecondition' in binary,debug_input_symbol_present='debug_input' in (root/f'disassembly-{mode}.txt').read_text())
 assert not report[mode]['debug_assertion_text_present']
 assert not report[mode]['debug_input_symbol_present']
 if mode!='o0':assert legacy==default
base_text=(root/'disassembly-base.txt').read_text()
references=[line.strip() for line in base_text.splitlines() if re.search(r'call\s+[0-9a-f]+ <codegen::precondition>',line)]
assert len(references)==2,references
assert '<codegen::asserted_precondition>:' not in base_text
report['optimized_debug_policy_equivalence']=dict(both_calls_share_precondition_body=True,call_sites=references)
report['one_extra_signed_policy']=dict(file_bytes=report['extra']['file_bytes']-report['base']['file_bytes'],text_bytes=report['extra']['sections']['.text']-report['base']['sections']['.text'])
(root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

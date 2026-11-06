#!/usr/bin/env python3
"""Extract reviewable functions using IR identities (ARM names inline into closures)."""
from pathlib import Path
import json,re,sys
root,out=map(Path,sys.argv[1:]);out.mkdir(parents=True,exist_ok=True);report=[]
for platform,name,match in [
('arm','arm-array-dot','; math_matrix_study::algebraic::cases_f32::{closure#15}'),
('arm','arm-auto-dot','; math_matrix_study::algebraic::cases_f32::{closure#2}'),
('arm','arm-fused-dot','; math_matrix_study::algebraic::cases_f32::{closure#19}'),
('arm','arm-array-polynomial','; math_matrix_study::algebraic::cases_f32::{closure#11}'),
('r7900x','v3-array-polynomial','poly_array_algebraic_f32_v3'),
('r7900x','v3-auto-dot','dot_algebraic_f32_v3'),
('r7900x','v3-ordinary-polynomial','poly_vector_f32_v3'),
('r7900x','v4x-auto-dot64','dot_algebraic_f64_v4x'),
('wasm','wasm-array-dot','; math_matrix_study::algebraic::cases_f32::{closure#15}'),
('wasm','wasm-array-polynomial','; math_matrix_study::algebraic::cases_f32::{closure#11}'),
]:
 ir=next((root/platform/'codegen-final').glob('*.ll'));asm=next(ir.parent.glob('*.s'));s=ir.read_text();a=asm.read_text()
 if match.startswith(';'):
  at=s.find(match+'\n');assert at>=0,(name,match);defstart=s.index('define ',at)
 else:
  m=next(m for m in re.finditer(r'^define .*$',s,re.M)if match in m[0]);defstart=m.start()
 stop=s.index('\n}',defstart)+2;body=s[defstart:stop];sym=re.search(r'@([^\s(]+)',body)[1]
 # Assembly uses target-specific end markers; only return this exact symbol.
 m=re.search(r'^'+re.escape(sym)+r':[^\n]*\n',a,re.M);assert m,(name,sym)
 end=re.search(r'^\.Lfunc_end\d+:' if platform!='wasm' else r'^\tend_function',a[m.end():],re.M);assert end,name
 end=m.end()+end.end();assembly=a[m.start():end];line=a[:m.start()].count('\n')+1
 (out/(name+'.s')).write_text(f'# Source {platform}/{asm.name}:{line}\n# Exact symbol {sym}\n'+assembly+'\n')
 arithmetic=[l.strip()for l in body.splitlines()if re.search(r'\b(fmul|fadd)|vector.reduce.fadd|llvm.fma',l)]
 (out/(name+'.ll')).write_text('; Selected arithmetic from '+platform+'/'+ir.name+'\n; Symbol '+sym+'\n'+'\n'.join(arithmetic)+'\n')
 report.append(dict(probe=name,platform=platform,source_asm=asm.name,line=line,symbol=sym,ir_packed=any(re.search(r'<\d+ x (float|double)>',l)for l in arithmetic),flags=sorted({word for l in arithmetic for word in ['reassoc','nsz','arcp','contract','afn','nnan','ninf']if re.search(r'\b'+word+r'\b',l)})))
for p in out.iterdir():assert p.stat().st_size<=30000,p
(out/'index.json').write_text(json.dumps(report,indent=2)+'\n')
print('extracted',len(report),'codegen probes')

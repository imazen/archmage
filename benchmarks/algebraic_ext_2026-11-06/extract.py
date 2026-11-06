#!/usr/bin/env python3
"""Extract real linked functions, matching IR by symbol identity, with exact hashes."""
from pathlib import Path
import re,json,sys
root,out=map(Path,sys.argv[1:]);out.mkdir(parents=True,exist_ok=True);index=[]
probes=[('native','v3-poly-index','poly_index_f32_v3'),('native','v3-poly-token','poly_token_f32_v3'),('native','v3-poly-map','poly_map_f32_v3'),('native','v3-dot-index','dot_index_f32_v3'),('native','v3-dot-ordinary-reduce','dot_index_ordinary_reduce_f32_v3'),('native','baseline-poly-index','generated::cases_f32::{closure#3}'),('arm','neon-poly-index','generated::cases_f32::{closure#3}'),('arm','neon-poly-token','generated::cases_f32::{closure#5}'),('arm','neon-poly-map','generated::cases_f32::{closure#7}'),('arm','neon-dot-index','generated::cases_f32::{closure#15}'),('arm','neon-dot-ordinary-reduce','generated::cases_f32::{closure#25}')]
for target,label,needle in probes:
 for mode in ['off','thin']:
  path=root/target/'timed-codegen'/f'{mode}.disasm';s=path.read_text();fs=list(re.finditer(r'^([0-9a-f]+) <(.+)>:\n',s,re.M));matches=[(i,m)for i,m in enumerate(fs)if needle in m[2]and'shim'not in m[2]];assert matches,(target,mode,needle);i,m=matches[0];body=s[m.start():fs[i+1].start()]if i+1<len(fs)else s[m.start():];name=f'{mode}-{label}';file=out/(name+'.txt');file.write_text(f'# {target}/timed-codegen/{mode}.disasm:{s[:m.start()].count(chr(10))+1}\n# Binary hash: '+(path.parent/f'{mode}.sha256').read_text()+body);assert file.stat().st_size<=30000,file
  ir=next((root/target/'codegen'/mode).glob('*.ll'));txt=ir.read_text();definition=None
  if 'closure#'in needle:
   comment='; algebraic_ext_consumer::'+needle;at=txt.find(comment+'\n')
   if at>=0:definition=txt[txt.index('define ',at):].split('\n}')[0]
  else:
   defs=[x for x in txt.split('\ndefine ')[1:]if needle in x.split('\n')[0]]
   if defs:definition=defs[0].split('\n}')[0]
  assert definition is not None,(target,mode,needle,'missing IR function (check alias explicitly)')
  lines=[l.strip()for l in definition.splitlines()if re.search(r'\bfadd\b|\bfmul\b|\bfdiv\b|\bfrem\b|llvm.fma|vector.reduce',l)]
  (out/(name+'.ll')).write_text('; '+str(ir.relative_to(root))+'\n; '+definition.split('\n')[0]+'\n'+'\n'.join(lines)+'\n')
  flags=sorted({f for l in lines for f in['reassoc','nsz','arcp','contract','afn','nnan','ninf']if re.search(r'\b'+f+r'\b',l)})
  index.append(dict(probe=name,address=m[1],symbol=m[2],flags=flags,packed_ir=any(re.search(r'<\d+ x (float|double)>',l)for l in lines),external_method_ir_references=definition.count('algebraic_magetypes_prototype'),thin_ir_note='pre-link IR; linked assembly is final lowering evidence'if mode=='thin'else'optimized no-LTO IR'))
(out/'index.json').write_text(json.dumps(index,indent=2)+'\n');print('linked functions extracted',len(index))

# Actual timed guest lowering, rather than the compiler's pre-link assembly.
wasm=[]
for mode in ['off','thin']:
 path=root/'wasm'/'timed-codegen'/(mode+'.wat');s=path.read_text();fs=list(re.finditer(r'^  \(func .*$',s,re.M))
 for form in ['index','token','map']:
  needle='poly_'+form+'_f32_wasm128';hits=[(i,m)for i,m in enumerate(fs)if needle in m[0]];assert len(hits)==1
  i,m=hits[0];body=s[m.start():fs[i+1].start()];line=s[:m.start()].count('\n')+1
  dest=out/f'{mode}-wasm-poly-{form}.wat';dest.write_text(';; wasm/timed-codegen/'+mode+'.wat:'+str(line)+'\n;; Module hash: '+(path.parent/(mode+'.sha256')).read_text().replace('\n','\n;; ')+'\n'+body);assert dest.stat().st_size<=30000
  wasm.append(dict(mode=mode,form=form,symbol=m[0],line=line,packed_mul=body.count('f32x4.mul'),packed_add=body.count('f32x4.add'),calls=re.findall(r'call (\S+)',body)))
(out/'wasm-index.json').write_text(json.dumps(wasm,indent=2)+'\n');print('timed WASM functions extracted',len(wasm))

"""Representative bounded feedback, separate from throughput matrix."""
from pathlib import Path
import csv
p=Path(__file__).resolve().parent
src=['use archmage::prelude::*;use crate::Case;'];regs=[]
rows=list(csv.DictReader((p/'api-matrix.csv').open()))
for r in rows:
 op=r['operation'];w=int(r['width']);label=r['candidate'];e=r['source_expression']
 if op not in ['exp','ln','log2','sqrt']:continue
 if w not in [1,8] or 'context' in label or 'dispatch' in label:continue
 if 'correct_proof' in label:
  e=f'f32x8::from_array_t(t,archmage::attuned!(magemath::portable::array::{op}_f32x8(a.to_array())))'
 elif ';' in e:continue
 n=f'fb{len(regs)}'
 if w==1:
  pre='a = -a.abs();' if op=='exp' else 'a=a.abs()+1.1;' if op in ['ln','log2'] else 'a=a.abs()+0.1;'
  src.append(f'fn {n}(x:&[f32],_y:&[f32],o:&mut[f32]){{assert_eq!(o.len(),1024);let mut a=x[0];for z in o{{{pre}let b=0.;let _=b;a={e};*z=a;}}}}')
  regs.append(f'v.push(Case{{op:"{op}",label:"{label}/baseline/x1",run:Box::new({n})}});')
 else:
  pre='a = -a.abs();' if op=='exp' else 'a=a.abs()+f32x8::splat_t(t,1.1);' if op in ['ln','log2'] else 'a=a.abs()+f32x8::splat_t(t,0.1);'
  src.append(f'#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]fn {n}(t:Token,x:&[f32],_y:&[f32],o:&mut[f32]){{assert_eq!(o.len(),1024);let mut a=f32x8::load_t(t,x[..8].try_into().unwrap());for z in o.as_chunks_mut::<8>().0{{{pre}a={e};a.store(z);}}}}')
  for tier,tok,cfg in [('scalar','ScalarToken',None),('v3','X64V3Token','target_arch="x86_64"'),('v4x','X64V4xToken','all(target_arch="x86_64",feature="avx512")'),('neon','NeonToken','target_arch="aarch64"'),('wasm128','Wasm128Token','all(target_arch="wasm32",target_feature="simd128")')]:
   prefix=f'#[cfg({cfg})]'if cfg else ''
   regs.append(f'{prefix}if let Some(t)={tok}::summon(){{v.push(Case{{op:"{op}",label:"{label}/{tier}/x8",run:Box::new(move|x,y,o|{n}_{tier}_t(t,x,y,o))}});}}')
src.append('pub fn cases()->Vec<Case<f32>>{let mut v=Vec::with_capacity(128);'+''.join(regs)+'v}')
(p/'src/feedback.rs').write_text('\n'.join(src))

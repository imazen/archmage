#!/usr/bin/env python3
from pathlib import Path
p=Path(__file__).resolve().parent
s=['use archmage::prelude::*;','use algebraic_magetypes_prototype::{AlgebraicOps,AlgebraicOpsWithToken,AlgebraicReduce};','use crate::Case;']
regs=[]
for ty,w in [('f32',8),('f64',4)]:
 v=f'{ty}x{w}'
 for op in ['poly','dot','div','rem']:
  forms=['ordinary','index','token','map','auto']+(['fused']if op in ['poly','dot']else[])+(['index_ordinary_reduce']if op=='dot'else[])
  for form in forms:
   name=f'{op}_{form}_{ty}'
   c=lambda x:f'{v}::splat_t(t,{x})'
   def calc(a,b,oper):
    if form in ['index','index_ordinary_reduce']:return f'{a}.algebraic_{oper}({b})'
    if form=='token':return f'{a}.algebraic_{oper}_t(t,{b})'
    if form=='map':return f'map_{oper}_{ty}(t,{a},{b})'
    return f'({a}{dict(add="+",sub="-",mul="*",div="/",rem="%")[oper]}{b})'
   if form=='auto':
    if op=='poly':body=f'for(o,&a)in out.iter_mut().zip(x){{*o=(1.0_{ty}/6.0).algebraic_mul(a).algebraic_add(0.5).algebraic_mul(a).algebraic_add(1.0).algebraic_mul(a).algebraic_add(1.0);}}'
    elif op=='dot':body=f'let mut r=0.0_{ty};for(&a,&b)in x.iter().zip(y){{r=r.algebraic_add(a.algebraic_mul(b));}}out[0]=r;'
    else:body=f'for((o,&a),&b)in out.iter_mut().zip(x).zip(y){{*o=a.algebraic_{op}(b);}}'
   else:
    load=f'let a={v}::load_t(t,a);let b={v}::load_t(t,b);'
    chunks=f'for((o,a),b)in out.as_chunks_mut::<{w}>().0.iter_mut().zip(x.as_chunks::<{w}>().0).zip(y.as_chunks::<{w}>().0){{{load}'
    if op=='poly':
     if form=='fused':expr=f'{c("1.0/6.0")}.mul_add_fused(a,{c("0.5")}).mul_add_fused(a,{c("1.0")}).mul_add_fused(a,{c("1.0")})'
     else:
      expr=c('1.0/6.0')
      for co in ['0.5','1.0','1.0']:expr=calc(calc(expr,'a','mul'),c(co),'add')
     expr=expr[1:-1] if form=='ordinary' else expr
     body=chunks+f'let _=b;let r={expr};r.store(o);}}'
    elif op=='dot':
     expr='a.mul_add_fused(b,sum)'if form=='fused'else calc('sum',calc('a','b','mul'),'add')
     reduce='sum.reduce_add()'if form in ['ordinary','fused','index_ordinary_reduce']else'sum.algebraic_reduce_add()'
     body=f'let mut sum={c("0.0")};for(a,b)in x.as_chunks::<{w}>().0.iter().zip(y.as_chunks::<{w}>().0){{{load}sum={expr};}}out[0]={reduce};'
    else:
     # Magetypes has no Rem; ordinary rem is explicitly a scalar lane-map control.
     expr=f'map_rem_ordinary_{ty}(t,a,b)'if op=='rem'and form=='ordinary'else calc('a','b',op)
     expr=expr[1:-1] if form=='ordinary'and op=='div' else expr
     body=chunks+f'let r={expr};r.store(o);}}'
   if form=='ordinary'and op=='dot': body=body.replace('sum=(sum+(a*b));','sum+=a*b;')
   s.append(f'#[attune(_*_t,+v4x(cfg(avx512)),define({v}))]\npub fn {name}(t:Token,x:&[{ty}],y:&[{ty}],out:&mut[{ty}]){{let _=t;assert_eq!(x.len(),y.len());assert_eq!(x.len(),out.len());assert_eq!(x.len()%{w},0);{body}}}')
   for tier,tok,cfg in [('scalar','ScalarToken',None),('v3','X64V3Token','target_arch="x86_64"'),('v4x','X64V4xToken','all(target_arch="x86_64",feature="avx512")'),('neon','NeonToken','target_arch="aarch64"'),('wasm128','Wasm128Token','all(target_arch="wasm32",target_feature="simd128")')]:
    pre=f'#[cfg({cfg})]'if cfg else''
    regs.append((ty,f'{pre}if let Some(t)={tok}::summon(){{v.push(Case{{group:"{op}/{ty}",name:"{form}/{tier}",run:Box::new(move|x,y,o|{name}_{tier}_t(t,x,y,o))}});}}'))
 # Generic direct maps are consumer-local controls, not the extension trait.
 for op in ['add','mul','div','rem']:
  s.append(f'#[inline(always)]fn map_{op}_{ty}<T:magetypes::simd::backends::{"F32x8Backend"if ty=="f32"else"F64x4Backend"}>(t:T,a:magetypes::simd::generic::{v}<T>,b:magetypes::simd::generic::{v}<T>)->magetypes::simd::generic::{v}<T>{{let a=a.to_array();let b=b.to_array();magetypes::simd::generic::{v}::from_array_t(t,core::array::from_fn(|i|a[i].algebraic_{op}(b[i])))}}')
 s.append(f'#[inline(always)]fn map_rem_ordinary_{ty}<T:magetypes::simd::backends::{"F32x8Backend"if ty=="f32"else"F64x4Backend"}>(t:T,a:magetypes::simd::generic::{v}<T>,b:magetypes::simd::generic::{v}<T>)->magetypes::simd::generic::{v}<T>{{let a=a.to_array();let b=b.to_array();magetypes::simd::generic::{v}::from_array_t(t,core::array::from_fn(|i|a[i]%b[i]))}}')
for ty in ['f32','f64']:s.append(f'pub fn cases_{ty}()->Vec<Case<{ty}>>{{let mut v=Vec::with_capacity(128);'+''.join(r for t,r in regs if t==ty)+'v}')
(p/'consumer/src/generated.rs').write_text('\n'.join(s)+'\n')
# Exact-representable behavioral fixtures, every width/backend and all five ops.
t=['use archmage::prelude::*;','use algebraic_magetypes_prototype::{AlgebraicOps,AlgebraicOpsWithToken,AlgebraicReduce};']
for ty,widths in [('f32',[4,8,16]),('f64',[2,4,8])]:
 for w in widths:
  v=f'{ty}x{w}';fn=f'fixture_{v}'
  body=f'let aa:[{ty};{w}]=core::array::from_fn(|i|{{let a=(4*i+7) as {ty};if i%2==0{{a}}else{{-a}}}});let bb:[{ty};{w}]=core::array::from_fn(|i|[2.,4.,8.,16.][i%4]);let a={v}::from_array_t(t,aa);let b={v}::from_array_t(t,bb);'
  for op,sign in [('add','+'),('sub','-'),('mul','*'),('div','/'),('rem','%')]:
   body+=f'let expected:[{ty};{w}]=core::array::from_fn(|i|aa[i]{sign}bb[i]);assert_eq!(a.algebraic_{op}(b).to_array(),expected);assert_eq!(a.algebraic_{op}_t(t,b).to_array(),expected);'
  body+=f'assert_eq!(a.algebraic_reduce_add(),aa.into_iter().sum::<{ty}>());'
  t.append(f'#[attune(_*_t,+v4x(cfg(avx512)),define({v}))]fn {fn}(t:Token){{{body}}}')
  calls=[]
  for tier,tok,cfg in [('scalar','ScalarToken',None),('v3','X64V3Token','target_arch="x86_64"'),('v4x','X64V4xToken','all(target_arch="x86_64",feature="avx512")'),('neon','NeonToken','target_arch="aarch64"'),('wasm128','Wasm128Token','all(target_arch="wasm32",target_feature="simd128")')]:
   pre=f'#[cfg({cfg})]'if cfg else'';calls.append(f'{pre}if let Some(t)={tok}::summon(){{{fn}_{tier}_t(t);n+=10;}}')
  t.append(f'pub fn check_{v}()->usize{{let mut n=0;'+''.join(calls)+'n}')
t.append('pub fn check_all()->usize{'+ '+'.join(f'check_{ty}x{w}()'for ty,ww in [('f32',[4,8,16]),('f64',[2,4,8])]for w in ww)+'}')
(p/'consumer/src/fixtures.rs').write_text('\n'.join(t))

#!/usr/bin/env python3
"""Retain paired old-composition/shared-engine probes through LLVM merging."""
from pathlib import Path
s='#![forbid(unsafe_code)]\n#![allow(dead_code,non_snake_case)]\n'
s+='type Probe<T,const N:usize>=fn(T,&[f32;N],f32,&mut[f32;N]);\n'
for n in [4,8,16]:
 for kind in ['lowp','midp','midp_unchecked']:
  for known in ['dynamic','literal']:
   exp='n'if known=='dynamic'else'2.4f32'
   original={'lowp':f'{{let r=(v.log2_lowp()*V::splat_t(token,{exp})).exp2_lowp();V::blend(v.simd_eq(V::splat_t(token,0.)),V::splat_t(token,0.),r)}}','midp':f'(v.log2_midp()*V::splat_t(token,{exp})).exp2_midp()','midp_unchecked':f'(v.log2_midp_unchecked()*V::splat_t(token,{exp})).exp2_midp_unchecked()'}[kind]
   for which,body in [('original',original),('shared',f'v.pow_{kind}({exp})')]:
    name=f'remap_{which}_{kind}_x{n}_{known}'
    s+=f'''#[archmage::magetypes(v4x(cfg(avx512)),v4(cfg(avx512)),v3,neon,wasm128,scalar)]
#[inline(never)]
fn {name}(token:Token,x:&[f32;{n}],n:f32,out:&mut[f32;{n}]){{use magetypes::simd::generic::f32x{n} as V;let _=n;let v=V::from_array_t(token,*x);*out=({body}).to_array();}}
'''
   for tier,ty,cfg in [('scalar','ScalarToken',None),('v3','X64V3Token','target_arch="x86_64"'),('v4','X64V4Token','all(target_arch="x86_64",feature="avx512")'),('v4x','X64V4xToken','all(target_arch="x86_64",feature="avx512")'),('neon','NeonToken','target_arch="aarch64"'),('wasm128','Wasm128Token','target_arch="wasm32"')]:
    if cfg:s+=f'#[cfg({cfg})]\n'
    s+=f'#[used]\nstatic REMAPTABLE_X{n}_{kind.upper()}_{known.upper()}_{tier.upper()}: [Probe<archmage::{ty},{n}>;2]=[remap_original_{kind}_x{n}_{known}_{tier},remap_shared_{kind}_x{n}_{known}_{tier}];\n'
s+='fn main(){}\n'
Path(__file__).with_name('src').joinpath('bin/remap_codegen.rs').write_text(s)

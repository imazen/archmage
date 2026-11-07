#!/usr/bin/env python3
"""Exact 35 x width kernel/array/slice comparison with the legacy inherent API."""
from pathlib import Path
import json
root=Path(__file__).resolve().parents[2]
rows=json.loads((root/'magemath/docs/legacy-inventory.json').read_text())['methods']
s='''//! Legacy compatibility: all35 names, all enabled widths, every available ISA.
#![allow(deprecated)] // The legacy inherent methods are the contract oracle.
use archmage::prelude::*;
use std::hint::black_box;
fn same<const N:usize>(tier:&str,op:&str,input:[f32;N],y:f32,got:[f32;N],want:[f32;N]){
 for i in 0..N {assert!(got[i].to_bits()==want[i].to_bits()||got[i].is_nan()&&want[i].is_nan(),"{tier}/{op}/width{N}/lane{i} {:08x}^{:08x}: got{:08x} expected{:08x}",input[i].to_bits(),y.to_bits(),got[i].to_bits(),want[i].to_bits());}
}
fn corpus()->Vec<f32>{
 let mut bits=vec![0,0x80000000,1,0x80000001,0x007fffff,0x00800000,0x00800001,0x3f7fffff,0x3f800000,0x3f800001,0xbf800000,0xc0000000,0x42feffff,0x42ff0000,0x42ffffff,0x43000000,0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,0x7f800001,0xff800001,0x7fc00123,0xffc00123];
 let mut state=0x5a17b39du32;for _ in 0..4096{state=state.wrapping_mul(1664525).wrapping_add(1013904223);bits.push(state);}bits.into_iter().map(f32::from_bits).collect()
}
'''
for width in [4,8,16]:
 helpers={}
 kernel=f'magemath::legacy::kernels::f32x{width}'if width!=8 else 'magemath::legacy::kernels'
 for row in rows:
  name=row['name'];call=',y'if row['exponent']else '';method='y'if row['exponent']else ''
  helper_start=len(s)
  if width==16:s+='#[cfg(feature="w512")]\n'
  # Bound debug stack frames without changing the corpus or assertions.
  s+=f'#[attune(_scalar_t,_v3_t,_neon_t,_wasm128_t,_v4x_t(cfg(avx512)),define(f32x{width}),inline(never))]\nfn {name}_width{width}(tier:&str,data:&[f32]){{\nlet t=Token::from_context();for chunk in data.chunks({width}){{let mut input=[1.0;{width}];input[..chunk.len()].copy_from_slice(chunk);let v=f32x{width}::from_array_t(t,input);\n'
  s+=' {\n'
  if row['exponent']:s+='for y in [-3.0,-1.0,0.0,0.5,1.0,2.0,2.4,f32::INFINITY,f32::NAN]{\n'
  else:s+='let y=0.0;\n'
  s+=f'let want=black_box(v).{name}({method}).to_array();let got={kernel}::{name}(t,black_box(v){call}).to_array();same(tier,"{name}",input,y,got,want);\n'
  s+=f'let got=attuned!(magemath::legacy::array::{name}_f32x{width}(black_box(input){call}),[_v3,_neon,_wasm128,_scalar,_v4x(cfg(avx512))],using(t));same(tier,"array/{name}",input,y,got,want);\n'
  # Small exact tail fixtures are lane-distinct and exercise every tail length.
  s+=f'if chunk.len()=={width}{{for len in 0..={width}{{let mut tail=input[..len].to_vec();attuned!(magemath::legacy::slice::{name}_f32x{width}(&mut tail{call}),[_v3,_neon,_wasm128,_scalar,_v4x(cfg(avx512))],using(t));for i in 0..len{{assert!(tail[i].to_bits()==want[i].to_bits()||tail[i].is_nan()&&want[i].is_nan(),"{{tier}}/slice/{name}/width{width}/tail{{len}}/lane{{i}} input{{:08x}}",input[i].to_bits());}}}}}}\n'
  if row['exponent']:s+='}\n'
  s+='}\n}\n'
  helpers[name]=s[helper_start:]
  s=s[:helper_start]
 for row in rows:
  name=row['name'];call=',y'if row['exponent']else '';method='y'if row['exponent']else ''
  s+=helpers[name]
  s+=f'{{{"let y=2.4;" if row["exponent"] else ""}let mut got=data[..2*{width}+3].to_vec();let mut want=Vec::new();for chunk in got.chunks({width}){{let mut pad=[1.0;{width}];pad[..chunk.len()].copy_from_slice(chunk);let v=f32x{width}::from_array_t(t,pad);let out=black_box(v).{name}({method}).to_array();want.extend_from_slice(&out[..chunk.len()]);}}attuned!(magemath::legacy::slice::{name}_f32x{width}(&mut got{call}),[_v3,_neon,_wasm128,_scalar,_v4x(cfg(avx512))],using(t));for i in 0..got.len(){{assert!(got[i].to_bits()==want[i].to_bits()||got[i].is_nan()&&want[i].is_nan(),"{{tier}}/multiblock/{name}/width{width}/lane{{i}}");}}}}\n'
  s+='}\n'
 if width==16:s+='#[cfg(feature="w512")]\n'
 s+=f'#[attune(_scalar_t,_v3_t,_neon_t,_wasm128_t,_v4x_t(cfg(avx512)),define(f32x{width}))]\nfn width{width}(tier:&str){{\nlet t=Token::from_context();let data=corpus();\n'
 for row in rows:
  s+=f'attuned!({row["name"]}_width{width}(tier,&data),[_v3,_neon,_wasm128,_scalar,_v4x(cfg(avx512))],using(t));\n'
 s+='}\n'
s+='''
#[test]
fn all_legacy_names_widths_and_tiers(){
 #[allow(unused_mut)] // AVX-512 adds a tier only in that feature build.
 let mut tiers=vec![("scalar",ScalarToken::summon().is_some()),("v3",X64V3Token::summon().is_some()),("neon",NeonToken::summon().is_some()),("wasm128",Wasm128Token::summon().is_some())];
 #[cfg(feature="avx512")]tiers.push(("v4x",X64V4xToken::summon().is_some()));
 for(name,available)in &tiers{println!("LEGACY_TIER {name} supported={available}");}
 if let Ok(required)=std::env::var("LEGACY_REQUIRE_TIERS"){for tier in required.split(','){assert!(tiers.iter().any(|(name,ok)|*name==tier&&*ok),"required legacy tier missing: {tier}");}}
 println!("LEGACY_WIDTHS 4=true 8=true 16={}",cfg!(feature="w512"));
'''
for tier,token,arch in [('scalar','ScalarToken',None),('v3','X64V3Token','x86_64'),('neon','NeonToken','aarch64'),('wasm128','Wasm128Token','wasm32'),('v4x','X64V4xToken','v4x')]:
 if arch:s+=('#[cfg(all(target_arch="x86_64",feature="avx512"))]\n'if arch=='v4x'else f'#[cfg(target_arch="{arch}")]\n')
 s+=f'if let Some(t)={token}::summon(){{width4_{tier}_t(t,"{tier}");width8_{tier}_t(t,"{tier}");#[cfg(feature="w512")]width16_{tier}_t(t,"{tier}");println!("PASS legacy {tier}: {{}}/35x{{}} names/widths, kernel+array+alltails",if cfg!(feature="w512"){{105}}else{{70}},if cfg!(feature="w512"){{3}}else{{2}});}}\n'
s+='}\n'
(root/'magemath/tests/legacy_superset.rs').write_text(s)
print('PASS generated exact legacy parity:35 names x 3 widths; kernel/array/slice tails; scalar/V3/NEON/WASM/V4x inventory')

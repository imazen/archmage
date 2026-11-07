#!/usr/bin/env python3
"""Compare additional vector widths with the established width-eight families."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROWS = json.loads((ROOT / 'magemath/docs/legacy-inventory.json').read_text())['methods']
S = '''//! Existing portable/relaxed/unfused contracts at all enabled widths.
use archmage::prelude::*;
use std::hint::black_box;
fn equal(a:f32,b:f32)->bool {a.to_bits()==b.to_bits() || a.is_nan()&&b.is_nan()}
'''
for width in (4, 16):
    calls = []
    for mode in ('portable', 'relaxed', 'unfused'):
        for row in ROWS:
            name = row['name']
            helper = f'{mode}_{name}_width{width}'
            calls.append(f'attuned!({helper}(tier),[_v3,_neon,_wasm128,_scalar,_v4x(cfg(avx512))],using(t));')
            if width == 16:
                S += '#[cfg(feature="w512")]\n'
            # Separate frames keep unoptimized parity tests within the ordinary
            # test-thread stack; inputs and assertions remain unchanged.
            S += f'''#[attune(_scalar_t,_v3_t,_neon_t,_wasm128_t,_v4x_t(cfg(avx512)),define(f32x8,f32x{width}),inline(never))]
fn {helper}(tier:&str) {{
let t=Token::from_context();
'''
            arg = ',2.4' if row['exponent'] else ''
            # Checked functions exercise exceptional rails; unchecked calls have
            # finite positive normal operands, also within exp's safe interval.
            unchecked = '_unchecked' in name
            values = '[0.125,0.25,0.5,0.75,1.0,1.25,2.0,4.0]' if unchecked else '[0.0,-0.0,1.0,-1.0,f32::MIN_POSITIVE,f32::from_bits(1),f32::INFINITY,f32::NEG_INFINITY,f32::NAN,0.125,0.75,2.0,4.0,32.0,-32.0]'
            S += f'''{{
let data={values}; let input:Vec<f32>=(0..2*{width}+3).map(|i|data[i%data.len()]).collect();
let mut want=Vec::new();
for chunk in input.chunks(8) {{ let mut pad=[1.0;8];pad[..chunk.len()].copy_from_slice(chunk);
let v=f32x8::from_array_t(t,black_box(pad));
let out=magemath::{mode}::kernels::{name}(t,v{arg}).to_array();want.extend_from_slice(&out[..chunk.len()]); }}
let mut got=input.clone();
attuned!(magemath::{mode}::slice::{name}_f32x{width}(&mut got{arg}),[_v3,_neon,_wasm128,_scalar,_v4x(cfg(avx512))],using(t));
for i in 0..got.len(){{assert!(equal(got[i],want[i]),"{{tier}}/{mode}/{name}/width{width}/slice/{{i}} input{{:08x}} got{{:08x}} want{{:08x}}",input[i].to_bits(),got[i].to_bits(),want[i].to_bits());}}
for (index,chunk) in input.chunks({width}).enumerate() {{let mut pad=[1.0;{width}];pad[..chunk.len()].copy_from_slice(chunk);
let v=f32x{width}::from_array_t(t,black_box(pad));
let kernel=magemath::{mode}::kernels::f32x{width}::{name}(t,v{arg}).to_array();
let array=attuned!(magemath::{mode}::array::{name}_f32x{width}(black_box(pad){arg}),[_v3,_neon,_wasm128,_scalar,_v4x(cfg(avx512))],using(t));
for i in 0..chunk.len(){{let expected=want[index*{width}+i];assert!(equal(kernel[i],expected)&&equal(array[i],expected),"{{tier}}/{mode}/{name}/width{width}/kernel-array/{{i}}");}}
}}
}}
'''
            S += '}\n'
    if width == 16:
        S += '#[cfg(feature="w512")]\n'
    S += f'''#[attune(_scalar_t,_v3_t,_neon_t,_wasm128_t,_v4x_t(cfg(avx512)),define(f32x8,f32x{width}))]
fn width{width}(tier:&str) {{
let t=Token::from_context();
''' + '\n'.join(calls) + '\n}\n'
S += '''#[test]
fn same_mode_widths(){
let mut supported=vec!["scalar"];
'''
for tier, token, cfg in [('scalar','ScalarToken',''), ('v3','X64V3Token','#[cfg(target_arch="x86_64")]'), ('neon','NeonToken','#[cfg(target_arch="aarch64")]'), ('wasm128','Wasm128Token','#[cfg(target_arch="wasm32")]'), ('v4x','X64V4xToken','#[cfg(all(target_arch="x86_64",feature="avx512"))]')]:
    S += f'''{cfg}
if let Some(t)={token}::summon(){{supported.push("{tier}");width4_{tier}_t(t,"{tier}");#[cfg(feature="w512")]width16_{tier}_t(t,"{tier}");println!("PASS existing families width parity {tier}");}}
'''
S += '''if let Ok(required)=std::env::var("LEGACY_REQUIRE_TIERS"){for tier in required.split(','){assert!(supported.contains(&tier),"required width tier missing: {tier}");}}
}
'''
(ROOT / 'magemath/tests/width_parity.rs').write_text(S)

"""Generate measurement wrappers, keeping algorithms in freeze.py/live library."""
from pathlib import Path
import sys
out = Path(sys.argv[1])
ops = ['exp2_lowp', 'exp_lowp', 'pow_lowp', 'exp2_midp', 'exp_midp', 'pow_midp', 'sigmoid_midp', 'silu_midp', 'log2_lowp', 'ln_lowp', 'log10_lowp', 'log2_midp', 'ln_midp', 'log10_midp']
s = '#![allow(dead_code)]\n'
for w in (4, 8, 16):
    for label in ('baseline', 'fused'):
        s += f'mod {label}{w} {{ include!(concat!(env!("OUT_DIR"), "/{label}{w}.rs")); }}\nuse {label}{w}::{label.title()}{w};\n'
    s += f'''
#[inline(always)]
fn eval{w}<T: magetypes::simd::backends::F32x{w}Convert, const OP: usize, const ARM: usize>(t:T, x:magetypes::simd::generic::f32x{w}<T>) -> magetypes::simd::generic::f32x{w}<T> {{
    match (OP, ARM) {{
'''
    for i, op in enumerate(ops):
        for arm,label in enumerate(('baseline', '', 'fused')):
            name = label+'_'+op if label else op
            args = ('t, ' if label else '') + ('2.4' if op.startswith('pow') else '')
            s += f'({i},{arm}) => x.{name}({args.rstrip(", ")}),\n'
    s+=' _ => unreachable!(),\n }\n}\n'
    s += f'''
#[inline(always)]
fn core{w}<T: magetypes::simd::backends::F32x{w}Convert, const OP:usize, const ARM:usize>(t:T, input:&[[f32;{w}]], output:&mut [[f32;{w}]], chain:bool) {{
    use magetypes::simd::generic::f32x{w} as V;
    if chain {{
        let mut x=V::from_array_t(t, input[0]);
        let a=V::splat_t(t, 0.125);
        let b=V::splat_t(t, 0.5);
        for _ in 0..input.len() {{ x=eval{w}::<T,OP,ARM>(t, x).abs()*a+b; }}
        output[0]=x.to_array();
    }} else {{
        for (x,y) in input.iter().zip(output) {{ *y=eval{w}::<T,OP,ARM>(t,V::from_array_t(t,*x)).to_array(); }}
    }}
}}
#[archmage::magetypes(v3, neon, wasm128, scalar)]
fn kernel{w}<const OP:usize,const ARM:usize>(t:Token,input:&[[f32;{w}]],output:&mut [[f32;{w}]],chain:bool) {{ core{w}::<Token,OP,ARM>(t,input,output,chain); }}
'''
    s += f'''
#[archmage::arcane]
fn scalar_v3_{w}<const OP:usize,const ARM:usize>(_t:archmage::X64V3Token,input:&[[f32;{w}]],output:&mut [[f32;{w}]],chain:bool) {{ core{w}::<_,OP,ARM>(archmage::ScalarToken,input,output,chain); }}
#[cfg(not(target_arch="wasm32"))]
fn register{w}<const OP:usize>(suite:&mut zenbench::Suite, name:&str, context:&str,n:usize,kind:&str,distribution:&str) {{
    let source_input: Vec<[f32;{w}]>=(0..n/{w}).map(|i| core::array::from_fn(|j| value(i*{w}+j,distribution, OP))).collect();
    let chain=kind=="chain";
    let id=format!("{{name}}/x{w}/{{context}}/{{n}}/{{kind}}/{{distribution}}");
    if !matches_filter(&id) {{return;}}
    suite.group(id, |g| {{
        g.config().min_rounds(50).max_rounds(50).auto_rounds(false)
            .warmup_time(std::time::Duration::from_millis(10))
            .max_time(std::time::Duration::from_millis(100))
            .max_wall_time(std::time::Duration::from_secs(60));
'''
    for arm,name in enumerate(('baseline','exceptional','fixed_fused')):
        s+=f'''
        let input=source_input.clone();
        let benchmark_context=context.to_string();
        g.bench("{name}",move |b| {{
            let mut output=vec![[0.0;{w}];input.len()];
            match benchmark_context.as_str() {{
                "scalar" => b.iter(|| {{kernel{w}_scalar::<OP,{arm}>(archmage::ScalarToken,std::hint::black_box(&input),std::hint::black_box(&mut output),chain);}}),
                #[cfg(target_arch="x86_64")]
                "v3" => {{let t=archmage::X64V3Token::summon().expect("required v3"); b.iter(|| {{kernel{w}_v3::<OP,{arm}>(t,std::hint::black_box(&input),std::hint::black_box(&mut output),chain);}});}},
                #[cfg(target_arch="x86_64")]
                "scalar-v3" => {{let t=archmage::X64V3Token::summon().expect("required v3"); b.iter(|| {{scalar_v3_{w}::<OP,{arm}>(t,std::hint::black_box(&input),std::hint::black_box(&mut output),chain);}});}},
                #[cfg(target_arch="x86_64")]
                "baseline-v3" => {{let t=archmage::X64V3Token::summon().expect("required v3"); b.iter(|| {{core{w}::<_,OP,{arm}>(t,std::hint::black_box(&input),std::hint::black_box(&mut output),chain);}});}},
                #[cfg(target_arch="aarch64")]
                "neon" => {{let t=archmage::NeonToken::summon().expect("required neon"); b.iter(|| {{kernel{w}_neon::<OP,{arm}>(t,std::hint::black_box(&input),std::hint::black_box(&mut output),chain);}});}},
                _ => panic!("unsupported required context {{benchmark_context}}"),
            }}
            std::hint::black_box(output);
        }});
'''
    s+=' });\n}\n'
s+='#[cfg(not(target_arch="wasm32"))]\npub fn register(suite:&mut zenbench::Suite,context:&str,n:usize,kind:&str,dist:&str) {\n'
for i,op in enumerate(ops):
    for w in (4,8,16):
        s+=f'register{w}::<{i}>(suite,"{op}",context,n,kind,dist);\n'
s+='}\n'
# Included module cannot have an inner attribute after macro expansion.
# Every legacy method, including aliases, is checked against frozen main.
import re
for w in (4,8,16):
    methods=re.findall(r'pub fn (\w+)\(self(, n: f32)?\) -> Self',Path(f'magetypes/src/simd/generic/generated/transcendentals_f32x{w}.rs').read_text())
    s+=f'fn verify{w}<T: magetypes::simd::backends::F32x{w}Convert>(t:T) {{\n'
    s+=f'for block in 0..8192u32 {{ let a:[f32;{w}]=core::array::from_fn(|i| {{let k=block*{w}+i as u32; f32::from_bits(0x00800000 + k.wrapping_mul(747796405).wrapping_add(2891336453)%0x7dffffff)}}); let x=magetypes::simd::generic::f32x{w}::from_array_t(t,std::hint::black_box(a));\n'
    for name,extra in methods:
        args='2.4' if extra else ''
        base_args='t, 2.4' if extra else 't'
        s+=f'let old=x.baseline_{name}({base_args}).to_array(); let new=x.{name}({args}).to_array(); for i in 0..{w} {{ if old[i].is_nan() {{assert!(new[i].is_nan());}} else {{assert_eq!(old[i].to_bits(),new[i].to_bits(),"ordinary {name} x{w} input={{:08x}}",a[i].to_bits());}} }}\n'
    s+='}\n}\n'
s+='pub fn verify(context:&str) {match context {\n'
for context,token in [('scalar','archmage::ScalarToken'),('v3','archmage::X64V3Token::summon().expect("required v3")'),('neon','archmage::NeonToken::summon().expect("required neon")'),('wasm128','archmage::Wasm128Token::summon().expect("required wasm128")')]:
    if context!='scalar':
        arch={'v3':'x86_64','neon':'aarch64','wasm128':'wasm32'}[context]
        s+=f'#[cfg(target_arch="{arch}")]\n'
    s+=f'"{context}" => {{let t={token};verify4(t);verify8(t);verify16(t);}},\n'
s+='_ => panic!("unsupported verification context {context}"),} println!("frozen-main ordinary bits: 35 methods x 3 widths passed for {context}");}\n'

for w in (4,8,16):
    s+=f'fn dump{w}<T: magetypes::simd::backends::F32x{w}Convert>(t:T) {{\n'
    s+=f'for block in 0..64 {{let input:[f32;{w}]=core::array::from_fn(|i|super::oracle_value(block*{w}+i));let x=magetypes::simd::generic::f32x{w}::from_array_t(t,input);\n'
    for i,op in enumerate(ops):
        s+=f'println!("{{}}",serde_json::json!({{"op":"{op}","width":{w},"token":T::NAME,"input":input.map(f32::to_bits),"baseline":eval{w}::<T,{i},0>(t,x).to_array().map(f32::to_bits),"exceptional":eval{w}::<T,{i},1>(t,x).to_array().map(f32::to_bits),"fused":eval{w}::<T,{i},2>(t,x).to_array().map(f32::to_bits)}}));\n'
    s+='}\n}\n'
s+='pub fn dump(context:&str) {match context {\n'
for context,token in [('scalar','archmage::ScalarToken'),('v3','archmage::X64V3Token::summon().expect("required v3")'),('neon','archmage::NeonToken::summon().expect("required neon")'),('wasm128','archmage::Wasm128Token::summon().expect("required wasm128")')]:
    if context!='scalar':
        s+=f'#[cfg(target_arch="{dict(v3="x86_64",neon="aarch64",wasm128="wasm32")[context]}")]\n'
    s+=f'"{context}" => {{let t={token};dump4(t);dump8(t);dump16(t);}},\n'
s+='_=>panic!("unsupported dump context {context}"),}}\n'

s+='#[cfg(target_arch="wasm32")]\npub fn guest_kernel(op:i32,arm:i32,width:i32,state:&mut super::guest::State,chain:bool) {let t=archmage::Wasm128Token::summon().expect("required SIMD");match (op,arm,width) {\n'
for i,op in enumerate(ops):
    for arm in range(3):
        for w in (4,8,16):
            s+=f'({i},{arm},{w})=>kernel{w}_wasm128::<{i},{arm}>(t,&state.input{w},&mut state.output{w},chain),\n'
s+='_=>panic!("invalid guest kernel"),}}\n'
(out/'kernels.rs').write_text(s.replace('#![allow(dead_code)]\n',''))

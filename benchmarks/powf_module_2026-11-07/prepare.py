"""Extract the real graph as a prevalidated performance control, not a rewrite."""
from pathlib import Path
import hashlib
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
graph = (ROOT / 'magetypes/src/powf/graph.rs').read_text().replace('//!', '//')
graph = graph.replace('use super::{PowfError, PowfErrorKind};',
                      'use magetypes::powf::{PowfError, PowfErrorKind};')
graph = graph.replace('fn $core<', 'pub(super) fn $core<')
graph = graph.replace('crate::simd', 'magetypes::simd')
graph = graph.replace('crate::soft_fma::', 'super::soft::')
graph = graph.replace('#[cfg(feature = "w512")]\n', '')
assert graph.count('if invalid.any_true() {') == 1
assert graph.count('if outside.bitcast_to_i32().any_true() {') == 1
graph = graph.replace('if invalid.any_true() {', 'if false {')
graph = graph.replace('if outside.bitcast_to_i32().any_true() {', 'if false {')
# WASM control keeps the existing fused primitive at each graph node. This
# compares primitive-level cached probing with the new whole-operation route.
a = graph.index('                    #[cfg(target_arch = "wasm32")]\n                    if ARITHMETIC == WASM_SOFTWARE')
b = graph.index('                    #[cfg(all(feature = "std", not(miri)))]', a)
graph = graph[:a] + graph[b:]
graph = graph.replace('a.mul_add_relaxed(b, c)', '{ #[cfg(target_arch="wasm32")] {a.mul_add_fused(b,c)} #[cfg(not(target_arch="wasm32"))] {a.mul_add_relaxed(b,c)} }')
(OUT / 'prevalidated.rs').write_text(graph)
source = (ROOT / 'magetypes/src/soft_fma.rs').read_text()
start = source.index('/// The 29 low f64 significand bits')
end = source.index('software_fma_or_test! {', start)
soft = source[start:end]
(OUT / 'soft.rs').write_text(soft)
print('prevalidated graph sha256', hashlib.sha256(graph.encode()).hexdigest())
print('exact software control sha256', hashlib.sha256(soft.encode()).hexdigest())

code = ''
for width in [4, 8, 16]:
    code += f'''
#[inline(always)]
fn core{width}<T: magetypes::simd::backends::F32x{width}Convert, const ARM:usize>(
    token:T, input:&[[f32;{width}]], output:&mut [[f32;{width}]], chain:bool, options:PowfOptions,
) {{
    use magetypes::simd::generic::f32x{width} as V;
    let mut acc=input[0];
    let exponent=std::hint::black_box(2.4);
    for (base, out) in input.iter().zip(output) {{
        let x=if chain {{acc}} else {{*base}};
        let y=match ARM {{
            0 => prevalidated_call{width}(token,x,exponent,options.domain),
            1 => V::from_array_t(token,x).try_powf_runtime(exponent,PowfOptions {{execution:PowfExecution::Caller,..options}}).unwrap().to_array(),
            2 => V::from_array_t(token,x).try_powf_runtime(exponent,PowfOptions {{execution:PowfExecution::Automatic,..options}}).unwrap().to_array(),
            _ => unreachable!(),
        }};
        let y=if chain {{y.map(|x|x.abs()*0.125+0.5)}} else {{y}};
        acc=y; *out=y;
    }}
}}
#[archmage::magetypes(v3,neon,wasm128,scalar)]
fn loop{width}<const ARM:usize>(token:Token,input:&[[f32;{width}]],output:&mut [[f32;{width}]],chain:bool,options:PowfOptions) {{
    core{width}::<Token,ARM>(token,input,output,chain,options);
}}

#[archmage::magetypes(v3,neon,wasm128,scalar)]
fn prevalidated_call{width}(token:Token,x:[f32;{width}],n:f32,domain:PowfDomain)->[f32;{width}] {{
    // Token specialization chooses exactly the same arithmetic family as Caller.
    let mode=if core::any::TypeId::of::<Token>()==core::any::TypeId::of::<archmage::ScalarToken>() {{prevalidated::SOFTWARE}} else {{prevalidated::NATIVE}};
    if mode==prevalidated::SOFTWARE {{prevalidated::core{width}::<Token,{{prevalidated::SOFTWARE}},false,false>(token,x,n).unwrap()}}
    else {{prevalidated::core{width}::<Token,{{prevalidated::NATIVE}},false,false>(token,x,n).unwrap()}}
}}
'''
    # The generic core calls a wrapper with the token as proof, so it needs a
    # generic control entry as well, not one of the feature-specific wrappers.
    a = code.rindex('#[archmage::magetypes(v3,neon,wasm128,scalar)]\nfn prevalidated_call')
    code = code[:a] + f'''
#[inline(always)]
fn prevalidated_call{width}<T:magetypes::simd::backends::F32x{width}Convert>(token:T,x:[f32;{width}],n:f32,domain:PowfDomain)->[f32;{width}] {{
    macro_rules! call {{($arithmetic:expr)=>{{match domain {{
        PowfDomain::PositiveNormal=>prevalidated::core{width}::<T,{{$arithmetic}},false,false>(token,x,n).unwrap(),
        PowfDomain::NonNegativeNormal=>prevalidated::core{width}::<T,{{$arithmetic}},true,false>(token,x,n).unwrap(),
        PowfDomain::Gamma=>prevalidated::core{width}::<T,{{$arithmetic}},true,true>(token,x,n).unwrap(),
    }}}};}}
    if core::any::TypeId::of::<T>()==core::any::TypeId::of::<archmage::ScalarToken>() {{call!(prevalidated::SOFTWARE)}}
    else {{call!(prevalidated::NATIVE)}}
}}
'''
    code += f'''
#[cfg(not(target_arch="wasm32"))]
fn register{width}(suite:&mut zenbench::Suite,context:&str,size:usize,chain:bool,domain:PowfDomain,distribution:&str) {{
    let name=format!("powf/x{width}/{{context}}/{{size}}/{{}}/{{domain:?}}/{{distribution}}",if chain {{"chain"}} else {{"stream"}});
    if !matches_filter(&name) {{return;}}
    let input:Vec<[f32;{width}]>=(0..size/{width}).map(|i|core::array::from_fn(|j|value(i*{width}+j,distribution))).collect();
    let options=PowfOptions {{domain,execution:PowfExecution::Caller}};
    // Independent equality gate before any timing, with every selected arm.
    let mut expected=vec![[0.0;{width}];input.len()];
    let mut checked=expected.clone();
    for arm in 0..3 {{
        execute{width}(arm,context,&input,&mut checked,chain,options);
        if arm==0 {{expected.clone_from(&checked);}} else {{assert_eq!(expected.iter().flatten().map(|x|x.to_bits()).collect::<Vec<_>>(),checked.iter().flatten().map(|x|x.to_bits()).collect::<Vec<_>>(),"pre-timing graph equality {{name}} arm{{arm}}");}}
    }}
    suite.group(name,|g| {{
        g.config().min_rounds(50).max_rounds(50).auto_rounds(false).warmup_time(std::time::Duration::from_millis(10)).max_time(std::time::Duration::from_millis(100)).max_wall_time(std::time::Duration::from_secs(60));
        for (arm,label) in ["prevalidated_graph","checked_caller","checked_auto"].into_iter().enumerate() {{
            let input=input.clone();let context=context.to_string();
            g.bench(label,move |b| {{
                let mut output=vec![[0.0;{width}];input.len()];
                execute{width}(arm,&context,&input,&mut output,chain,options);
                b.iter(||execute{width}(arm,&context,std::hint::black_box(&input),std::hint::black_box(&mut output),chain,options));
                std::hint::black_box(output);
            }});
        }}
    }});
}}

fn execute{width}(arm:usize,context:&str,input:&[[f32;{width}]],output:&mut[[f32;{width}]],chain:bool,options:PowfOptions) {{
    macro_rules! go {{($function:ident,$token:expr)=>{{match arm {{0=>$function::<0>($token,input,output,chain,options),1=>$function::<1>($token,input,output,chain,options),2=>$function::<2>($token,input,output,chain,options),_=>unreachable!()}}}};}}
    match context {{
        "scalar"=>go!(loop{width}_scalar,archmage::ScalarToken),
        #[cfg(target_arch="x86_64")]
        "v3"=>go!(loop{width}_v3,archmage::X64V3Token::summon().expect("required V3")),
        #[cfg(target_arch="aarch64")]
        "neon"=>go!(loop{width}_neon,archmage::NeonToken::summon().expect("required NEON")),
        #[cfg(target_arch="wasm32")]
        "wasm128"=>go!(loop{width}_wasm128,archmage::Wasm128Token::summon().expect("required WASM SIMD")),
        _=>panic!("unsupported required context {{context}}"),
    }}
}}
'''
for width in [4, 8, 16]:
    code += f'''
#[cfg(not(target_arch="wasm32"))]
#[archmage::magetypes(v3,neon,wasm128,scalar)]
fn reject{width}<const AUTO:bool>(token:Token,input:&[[f32;{width}]],output:&mut [u16]) {{
    let options=PowfOptions {{domain:PowfDomain::Gamma,execution:if AUTO {{PowfExecution::Automatic}} else {{PowfExecution::Caller}}}};
    for (x,out) in input.iter().zip(output) {{
        let result=magetypes::simd::generic::f32x{width}::from_array_t(token,*x).try_powf_runtime(std::hint::black_box(2.4),options);
        *out=match result {{Ok(_)=>0,Err(e)=>{{assert_eq!(e.kind,magetypes::powf::PowfErrorKind::InvalidBase);e.lane_mask}}}};
    }}
}}
#[cfg(not(target_arch="wasm32"))]
fn reject_execute{width}(automatic:bool,context:&str,input:&[[f32;{width}]],output:&mut[u16]) {{
    macro_rules! go {{($f:ident,$t:expr)=>{{if automatic {{$f::<true>($t,input,output)}} else {{$f::<false>($t,input,output)}}}};}}
    match context {{
        "scalar"=>go!(reject{width}_scalar,archmage::ScalarToken),
        #[cfg(target_arch="x86_64")] "v3"=>go!(reject{width}_v3,archmage::X64V3Token::summon().expect("required V3")),
        #[cfg(target_arch="aarch64")] "neon"=>go!(reject{width}_neon,archmage::NeonToken::summon().expect("required NEON")),
        #[cfg(target_arch="wasm32")] "wasm128"=>go!(reject{width}_wasm128,archmage::Wasm128Token::summon().expect("required WASM SIMD")),
        _=>panic!("unsupported required context"),
    }}
}}
#[cfg(not(target_arch="wasm32"))]
fn reject_register{width}(suite:&mut zenbench::Suite,context:&str,size:usize,dense:bool,subnormal:bool) {{
    let name=format!("reject/x{width}/{{context}}/{{size}}/{{}}/{{}}",if dense {{"dense"}} else {{"sparse"}},if subnormal {{"subnormal"}} else {{"nan"}});
    if !matches_filter(&name) {{return;}}
    let input:Vec<[f32;{width}]>=(0..size/{width}).map(|i|core::array::from_fn(|lane|if dense || (i%16==0 && lane==0) {{f32::from_bits(if subnormal {{1}} else {{0x7f80_0001}})}} else {{1.0}})).collect();
    let expected:Vec<u16>=input.iter().map(|v|v.iter().enumerate().fold(0,|mask,(lane,&x)|mask|if x!=1.0 {{1<<lane}} else {{0}})).collect();
    let mut output=vec![0;input.len()];
    for automatic in [false,true] {{reject_execute{width}(automatic,context,&input,&mut output);assert_eq!(output,expected);}}
    suite.group(name,|g| {{
        g.config().min_rounds(50).max_rounds(50).auto_rounds(false).warmup_time(std::time::Duration::from_millis(10)).max_time(std::time::Duration::from_millis(100)).max_wall_time(std::time::Duration::from_secs(60));
        for (automatic,label) in [(false,"checked_caller"),(true,"checked_auto")] {{
            let input=input.clone();let context=context.to_string();
            g.bench(label,move|b| {{let mut output=vec![0;input.len()];reject_execute{width}(automatic,&context,&input,&mut output);b.iter(||reject_execute{width}(automatic,&context,std::hint::black_box(&input),std::hint::black_box(&mut output)));std::hint::black_box(output);}});
        }}
    }});
}}
'''
(OUT / 'cost_kernels.rs').write_text(code)

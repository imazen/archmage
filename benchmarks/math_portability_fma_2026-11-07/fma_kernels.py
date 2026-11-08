"""Controls reuse the exact array software path and real public primitives."""
from pathlib import Path
import sys
out=Path(sys.argv[1])
source=Path('magetypes/src/soft_fma.rs').read_text()
# Make the already private software control available even on an FMA target.
needle='software_fma_or_test! {'
start=source.index(needle)
brace=source.index('{',start)
depth=1
end=brace+1
while depth:
    depth += (source[end]=='{')-(source[end]=='}')
    end+=1
source=source[:start]+source[brace+1:end-1]+source[end:]
macro_start=source.index('macro_rules! software_fma_or_test')
macro_brace=source.index('{',macro_start)
depth=1;macro_end=macro_brace+1
while depth:
    depth+=(source[macro_end]=='{')-(source[macro_end]=='}');macro_end+=1
source=source[:macro_start]+source[macro_end:]
source=source.replace('//!','//').replace('#![forbid(unsafe_code)]','')
source+='\npub fn software64<const N:usize>(a:[f64;N],b:[f64;N],c:[f64;N])->[f64;N] {f64_soft::fma_soft_lanes(a,b,c)}\n'
s='#[allow(dead_code)] mod soft {\n'+source+'\n}\n'
for f,widths in [('f32',[1,4,8,16]),('f64',[1,2,4,8])]:
    for n in widths:
        name=f'{f}x{n}'
        bound=f'{f.upper()}x{n}Backend'
        # x1 is a scalar wrapper, wider types use the selected token backend.
        if n==1:
            bound='archmage::SimdToken'
            current=f'[magetypes::simd::{name}(x[0]).mul_add_fused(magetypes::simd::{name}(b[0]),magetypes::simd::{name}(c[0])).0]'
        else:
            bound='magetypes::simd::backends::'+bound
            current=f'magetypes::simd::generic::{name}::from_array_t(t,x).mul_add_fused(magetypes::simd::generic::{name}::from_array_t(t,b),magetypes::simd::generic::{name}::from_array_t(t,c)).to_array()'
        software='soft::fmaf_soft_lanes(x,b,c)' if f=='f32' else 'soft::software64(x,b,c)'
        s+=f'''
#[inline(always)]
fn core_{name}<T:{bound},const ARM:usize>(t:T,input:&[[{f};{n}]],output:&mut [[{f};{n}]],chain:bool) {{
    let _=t;
    let b=[std::hint::black_box(0.999_91);{n}]; let c=[std::hint::black_box(0.000_13);{n}];
    let mut acc=input[0];
    for (a,y) in input.iter().zip(output.iter_mut()) {{
        let x=if chain {{acc}} else {{*a}};
        let z=match ARM {{0=>{current},1=>{software},2=>core::array::from_fn(|i|x[i].mul_add(b[i],c[i])),_=>unreachable!()}};
        if chain {{acc=z;}} else {{*y=z;}}
    }}
    if chain {{output[0]=acc;}}
}}
#[archmage::magetypes(v3,neon,scalar)]
fn kernel_{name}<const ARM:usize>(t:Token,input:&[[{f};{n}]],output:&mut [[{f};{n}]],chain:bool) {{core_{name}::<Token,ARM>(t,input,output,chain);}}
#[archmage::arcane]
fn scalar_v3_{name}<const ARM:usize>(_t:archmage::X64V3Token,input:&[[{f};{n}]],output:&mut [[{f};{n}]],chain:bool) {{core_{name}::<_,ARM>(archmage::ScalarToken,input,output,chain);}}
'''
        s+=f'''
#[cfg(target_arch="x86_64")]
fn dispatch_{name}(input:&[[{f};{n}]],output:&mut [[{f};{n}]],chain:bool) {{
    if let Some(t)=archmage::X64V3Token::summon() {{kernel_{name}_v3::<2>(t,input,output,chain);}}
    else {{kernel_{name}_scalar::<1>(archmage::ScalarToken,input,output,chain);}}
}}
fn register_{name}(suite:&mut zenbench::Suite,context:&str,size:usize,chain:bool,dist:&str) {{
    let id=format!("fma/{name}/{{context}}/{{size}}/{{}}/{{dist}}",if chain {{"chain"}} else {{"stream"}});
    if !super::matches_filter(&id) {{return;}}
    let source:Vec<[{f};{n}]>=(0..size/{n}).map(|i|core::array::from_fn(|j|super::fma_value_{f}(i*{n}+j,dist))).collect();
    suite.group(id,|g| {{
        g.config().min_rounds(50).max_rounds(50).auto_rounds(false).warmup_time(std::time::Duration::from_millis(10)).max_time(std::time::Duration::from_millis(100)).max_wall_time(std::time::Duration::from_secs(60));
'''
        for arm,label in enumerate(('current_fused','array_software','whole_buffer_hardware')):
            s+=f'''
        let input=source.clone(); let context=context.to_string();
        g.bench("{label}",move |bencher| {{ let mut output=vec![[0.0;{n}];input.len()];
            match context.as_str() {{
                "scalar" => {{
                    #[cfg(target_arch="x86_64")]
                    if {arm}==2 {{bencher.iter(||dispatch_{name}(std::hint::black_box(&input),std::hint::black_box(&mut output),chain));}}
                    else {{bencher.iter(||kernel_{name}_scalar::<{arm}>(archmage::ScalarToken,std::hint::black_box(&input),std::hint::black_box(&mut output),chain));}}
                    #[cfg(not(target_arch="x86_64"))]
                    bencher.iter(||kernel_{name}_scalar::<{arm}>(archmage::ScalarToken,std::hint::black_box(&input),std::hint::black_box(&mut output),chain));
                }},
                #[cfg(target_arch="x86_64")]
                "scalar-v3" => {{let t=archmage::X64V3Token::summon().expect("required v3");bencher.iter(||scalar_v3_{name}::<{arm}>(t,std::hint::black_box(&input),std::hint::black_box(&mut output),chain));}},
                #[cfg(target_arch="x86_64")]
                "v3" => {{let t=archmage::X64V3Token::summon().expect("required v3");bencher.iter(||kernel_{name}_v3::<{arm}>(t,std::hint::black_box(&input),std::hint::black_box(&mut output),chain));}},
                #[cfg(target_arch="x86_64")]
                "baseline-v3" => {{let t=archmage::X64V3Token::summon().expect("required v3");bencher.iter(||core_{name}::<_,{arm}>(t,std::hint::black_box(&input),std::hint::black_box(&mut output),chain));}},
                #[cfg(target_arch="aarch64")]
                "neon" => {{let t=archmage::NeonToken::summon().expect("required neon");bencher.iter(||kernel_{name}_neon::<{arm}>(t,std::hint::black_box(&input),std::hint::black_box(&mut output),chain));}},
                _=>panic!("unsupported context {{context}}"),
            }} std::hint::black_box(output);
        }});
'''
        # Shadow context moves a String after first arm; use unique local name.
        s=s.replace('let context=context.to_string();','let arm_context=context.to_string();').replace('match context.as_str()','match arm_context.as_str()').replace('unsupported context {context}','unsupported context {arm_context}')
        s+='});\n}\n'
s+='pub fn register(suite:&mut zenbench::Suite,context:&str,size:usize,kind:&str,dist:&str) {\n'
for f,widths in [('f32',[1,4,8,16]),('f64',[1,2,4,8])]:
    for n in widths:
        s+=f'register_{f}x{n}(suite,context,size,kind=="chain",dist);\n'
s+='}\n'
(out/'fma_kernels.rs').write_text(s)

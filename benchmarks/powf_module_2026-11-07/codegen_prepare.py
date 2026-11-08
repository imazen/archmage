"""Generate actual caller probes, with options/exponents visible or opaque."""
from pathlib import Path
import sys
out=Path(sys.argv[1])
code='use magetypes::powf::{PowfDomain,PowfExecution,PowfOptions};\n'
calls=[]
for n in [4,8,16]:
    for domain in ['PositiveNormal','NonNegativeNormal','Gamma']:
        for known in [False,True]:
            exp='f32::from_bits(0x4019999a)' if known else 'exponent'
            opts=f'PowfOptions::new().with_domain(PowfDomain::{domain})'
            bodies={
                'const_caller':f'v.try_powf_with::<{{{opts}.with_execution(PowfExecution::Caller).bits()}}>({exp}).unwrap().to_array()',
                'const_auto':f'v.try_powf_with::<{{{opts}.bits()}}>({exp}).unwrap().to_array()',
                'runtime_literal_caller':f'v.try_powf_runtime({exp},{opts}.with_execution(PowfExecution::Caller)).unwrap().to_array()',
                'runtime_literal_auto':f'v.try_powf_runtime({exp},{opts}).unwrap().to_array()',
                'runtime_dynamic':f'v.try_powf_runtime({exp},options).unwrap().to_array()',
                'midp':f'v.pow_midp({exp}).to_array()',
                'midp_unchecked':f'v.pow_midp_unchecked({exp}).to_array()',
                'lowp':f'v.pow_lowp({exp}).to_array()',
                'std_f32':f'(*input).map(|x|x.powf({exp}))',
                'std_f64':f'(*input).map(|x|f64::from(x).powf(f64::from({exp}))as f32)',
            }
            for label,body in bodies.items():
                name=f'probe_{label}_x{n}_{domain}_{"literal"if known else"dynamic"}'
                code+=f'''#[allow(non_snake_case,unused_variables)]
#[archmage::magetypes(v4x(cfg(avx512)),v4(cfg(avx512)),v3,neon,wasm128,scalar)]
#[inline(never)]
pub fn {name}(token:Token,input:&[f32;{n}],exponent:f32,options:PowfOptions,output:&mut[f32;{n}]){{
    let v=magetypes::simd::generic::f32x{n}::from_array_t(token,*input);
    *output={body};
}}
'''
                calls.append((name,n,domain))
code+='fn main(){use archmage::SimdToken;let n=std::hint::black_box(2.4_f32);\n'
for width in [4,8,16]:code+=f'let input{width}=std::hint::black_box([0.75;{width}]);let mut output{width}=[0.0;{width}];\n'
for tier,ty,cfg in [('scalar','ScalarToken',None),('v3','X64V3Token','target_arch="x86_64"'),('v4','X64V4Token','all(target_arch="x86_64",feature="avx512")'),('v4x','X64V4xToken','all(target_arch="x86_64",feature="avx512")'),('neon','NeonToken','target_arch="aarch64"'),('wasm128','Wasm128Token','target_arch="wasm32"')]:
    if cfg:code+=f'#[cfg({cfg})]\n'
    code+=f'if let Some(token)=archmage::{ty}::summon(){{\n'
    for name,width,domain in calls:
        code+=f'{name}_{tier}(token,&input{width},n,std::hint::black_box(PowfOptions::new().with_domain(PowfDomain::{domain}).with_execution(PowfExecution::Caller)),&mut output{width});std::hint::black_box(output{width});\n'
    code+='}\n'
code+='}\n'
(out/'replacement_probes.rs').write_text(code)

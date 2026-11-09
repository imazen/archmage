"""Public API permutations: literal options through the caller and feature kernel."""
from pathlib import Path
import sys
out=Path(sys.argv[1]);code='';probes='';calls=[]
flags=[x for x in range(64) if x&3!=3]+[x for x in range(64,128) if x&16 and not x&8]
assert len(flags)==64
labels=[]
for f in flags:
 labels.append(f'{"Backend" if f&16 else "BitExact"}_{"HIGHP" if f&64 else "LOWP" if f&8 else "MIDP"}_{["PositiveNormal","NonNegativeNormal","Gamma","Full"][f&3]}_{"AssumeValid" if f&32 else "Checked"}_{"Caller" if f&4 else "Automatic"}')
labels+=['legacy_pow_lowp','legacy_pow_midp','std_f32_powf','std_f64_powf_cast_f32','runtime_BitExact_MIDP_Gamma_Checked_Automatic','local_HIGHP_ResultMap_reference','direct_array_HIGHP_Full']
code+='const LABELS: &[&str] = &'+repr(labels).replace("'",'"')+';\n'
for n in [1,4,8,16]:
 vector=f'magetypes::simd::generic::f32x{n}'
 public=f'v.try_powf_with::<FLAGS>(n).unwrap().to_array()' if n!=1 else 'magetypes::powf::try_powf_array_with::<FLAGS,1>(x,n).unwrap()'
 runtime='v.try_powf_runtime(n,options).unwrap().to_array()' if n!=1 else 'magetypes::powf::try_powf_array_runtime(x,n,options).unwrap()'
 original=(f'highp_reference::run::<83,{n}>(v.to_array(),n).map(|a|{vector}::from_array_t(token,a)).unwrap().to_array()'if n!=1 else 'x.map(|x|{let v=magetypes::simd::generic::f32x4::splat_t(archmage::ScalarToken,x);highp_reference::run::<83,4>(v.to_array(),n).map(|a|magetypes::simd::generic::f32x4::from_array_t(archmage::ScalarToken,a)).unwrap().to_array()[0]})')
 legacy=[f'v.pow_lowp(n).to_array()',f'v.pow_midp(n).to_array()'] if n!=1 else ['x.map(|x|magetypes::simd::generic::f32x4::splat_t(archmage::ScalarToken,x).pow_lowp(n).to_array()[0])','x.map(|x|magetypes::simd::generic::f32x4::splat_t(archmage::ScalarToken,x).pow_midp(n).to_array()[0])']
 code+=f'''#[archmage::magetypes(v3,neon,scalar)]
#[inline(never)]
fn policy_loop{n}<const ARM:usize,const FLAGS:u32,const EXP:u32>(token:Token,input:&[[f32;{n}]],output:&mut[[f32;{n}]],chain:bool,options:magetypes::powf::PowfOptions){{
 let _=token; let n=f32::from_bits(EXP); let mut acc=input[0];
 for(base,out)in input.iter().zip(output){{
  let x=if chain{{acc}}else{{*base}};
'''
 if n!=1:code+=f'  let v={vector}::from_array_t(token,x);\n'
 code+=f'''  let y=match ARM {{0=>{public},1=>{legacy[0]},2=>{legacy[1]},3=>x.map(|x|x.powf(n)),4=>x.map(|x|f64::from(x).powf(f64::from(n)) as f32),5=>{runtime},6=>{original},7=>magetypes::powf::try_powf_array_with::<83,{n}>(x,n).unwrap(),_=>unreachable!()}};
  acc=if chain{{y.map(normalize)}}else{{y}};*out=acc;
 }}
}}
fn execute{n}<const EXP:u32>(arm:usize,context:&str,input:&[[f32;{n}]],output:&mut[[f32;{n}]],chain:bool){{
 let options=std::hint::black_box(magetypes::powf::PowfOptions::PORTABLE_GAMMA_MIDP);
 macro_rules! call{{($f:ident,$t:expr)=>{{match arm{{
'''
 for i,f in enumerate(flags):code+=f'  {i}=>$f::<0,{f},EXP>($t,input,output,chain,options),\n'
 for i in range(7):code+=f'  {64+i}=>$f::<{i+1},2,EXP>($t,input,output,chain,options),\n'
 code+='  _=>unreachable!(),\n }}}\n'
 code+=f''' match context{{
 "scalar"=>call!(policy_loop{n}_scalar,archmage::ScalarToken),
 #[cfg(target_arch="x86_64")] "v3"=>call!(policy_loop{n}_v3,archmage::X64V3Token::summon().expect("required V3")),
 #[cfg(target_arch="aarch64")] "neon"=>call!(policy_loop{n}_neon,archmage::NeonToken::summon().expect("required NEON")),
 _=>panic!("unsupported explicit context"),
 }}
}}
fn register{n}<const EXP:u32>(suite:Option<&mut zenbench::Suite>,cell:&Cell){{
 let input:Vec<[f32;{n}]>=(0..cell.size/{n}).map(|i|core::array::from_fn(|j|(((i*{n}+j)%255)+1)as f32/255.)).collect();
 let mut output=vec![[0.;{n}];input.len()];
 for arm in 0..LABELS.len(){{execute{n}::<EXP>(arm,cell.context,&input,&mut output,cell.chain);assert!(output.iter().flatten().all(|x|x.is_normal()));}}
 let Some(suite)=suite else{{return;}};
 suite.group(&cell.name,|g|{{
  g.config().min_rounds(50).max_rounds(50).auto_rounds(false).warmup_time(std::time::Duration::from_millis(10)).max_time(std::time::Duration::from_millis(100)).max_wall_time(std::time::Duration::from_secs(120));
  for(arm,label)in LABELS.iter().enumerate(){{let input=input.clone();let context=cell.context;let chain=cell.chain;
   g.bench(*label,move|b|{{let mut output=vec![[0.;{n}];input.len()];execute{n}::<EXP>(arm,context,&input,&mut output,chain);
    b.iter(||execute{n}::<EXP>(arm,context,std::hint::black_box(&input),std::hint::black_box(&mut output),chain));std::hint::black_box(output);
   }});
  }}
 }});
}}
'''
 if n!=1:
  for f in flags:
   name=f'policyprobe_f{f}_x{n}'
   probes+=f'''#[archmage::magetypes(v3,neon,wasm128,scalar)]
#[inline(never)]
pub fn {name}(token:Token,input:&[f32;{n}],n:f32,output:&mut[f32;{n}]){{
 *output={vector}::from_array_t(token,*input).try_powf_with::<{f}>(n).unwrap().to_array();
}}
'''
   calls.append((name,n))
   # Guaranteed feature targets can inline a feature wrapper; retain outer body.
   for tier,ty,cfg in [('neon','NeonToken','aarch64'),('wasm128','Wasm128Token','wasm32')]:
    probes+=f'''#[cfg(target_arch="{cfg}")]
#[inline(never)]
fn caller_{name}_{tier}(token:archmage::{ty},input:&[f32;{n}],n:f32,output:&mut[f32;{n}]){{{name}_{tier}(token,input,n,output);}}
'''
for n in [1,4,8,16]:
 for kind in ['array','std','original']:
  if kind=='original'and n==1:continue
  name=f'{kind}probe_x{n}'
  if kind=='array':expr=f'magetypes::powf::try_powf_array_with::<83,{n}>(*input,n).unwrap()'
  elif kind=='std':expr='input.map(|x|x.powf(n))'
  else:expr=f'highp_reference::run::<83,{n}>(magetypes::simd::generic::f32x{n}::from_array_t(token,*input).to_array(),n).map(|a|magetypes::simd::generic::f32x{n}::from_array_t(token,a)).unwrap().to_array()'
  probes+=f'#[archmage::magetypes(v3,neon,wasm128,scalar)]\n#[inline(never)]\npub fn {name}(token:Token,input:&[f32;{n}],n:f32,output:&mut[f32;{n}]){{let _=token;*output={expr};}}\n'
  for tier,ty,cfg in [('scalar','ScalarToken',None),('v3','X64V3Token','x86_64'),('neon','NeonToken','aarch64'),('wasm128','Wasm128Token','wasm32')]:
   attr=f'#[cfg(target_arch="{cfg}")]\n'if cfg else ''
   alias=f'{kind.title()}ProbeX{n}{tier.title()}'
   probes+=attr+f'type {alias}=fn(archmage::{ty},&[f32;{n}],f32,&mut[f32;{n}]);\n'
   probes+=attr+f'#[used]\npub static EXTRATABLE_{kind.upper()}_X{n}_{tier.upper()}: [{alias};1]=[{name}_{tier}];\n'
# Retained pointer tables expose every configuration even when LLVM merges
# identical probe bodies. The audit resolves the actual relocation targets.
for n in [4,8,16]:
 for tier,ty,cfg in [('scalar','ScalarToken',None),('v3','X64V3Token','x86_64'),('neon','NeonToken','aarch64'),('wasm128','Wasm128Token','wasm32')]:
  attr=f'#[cfg(target_arch="{cfg}")]\n'if cfg else ''
  alias=f'ProbeX{n}{tier.title()}'
  probes+=attr+f'type {alias}=fn(archmage::{ty},&[f32;{n}],f32,&mut[f32;{n}]);\n'
  probes+=attr+f'#[used]\npub static POLICYTABLE_X{n}_{tier.upper()}: [{alias};64]=[\n'
  probes+=''.join(f'policyprobe_f{f}_x{n}_{tier},\n'for f in flags)+'];\n'
probes+='fn main(){use archmage::SimdToken;let n=std::hint::black_box(2.4);\n'
for n in [4,8,16]:probes+=f'let input{n}=std::hint::black_box([0.75;{n}]);let mut output{n}=[0.;{n}];\n'
for tier,ty,cfg in [('scalar','ScalarToken',None),('v3','X64V3Token','x86_64'),('neon','NeonToken','aarch64'),('wasm128','Wasm128Token','wasm32')]:
 if cfg:probes+=f'#[cfg(target_arch="{cfg}")]\n'
 probes+=f'if let Some(token)=archmage::{ty}::summon(){{\n'
 for name,n in calls:
  caller=f'caller_{name}_{tier}'if tier in ['neon','wasm128']else f'{name}_{tier}'
  probes+=f'{caller}(token,&input{n},n,&mut output{n});std::hint::black_box(output{n});\n'
 probes+='}\n'
probes+='}\n'
(out/'policy_bench.rs').write_text(code)
(out/'policy_probes.rs').write_text(probes)

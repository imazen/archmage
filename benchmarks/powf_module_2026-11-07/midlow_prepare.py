#!/usr/bin/env python3
"""Generate literal const public API arms; preserve the token in scalar tails."""
from pathlib import Path
flags=[f for f in range(64) if f&3!=3]+[f for f in range(128,192) if f&3==2]
labels=[f'{"backend" if f&16 else "portable"}/{"lowp" if f&8 else "midp"}/{("positive-gamma" if f&128 else ["positive","nonnegative","gamma"][f&3])}/{"assume" if f&32 else "checked"}/{"caller" if f&4 else "auto"}'for f in flags]+['legacy/pow_lowp','legacy/pow_midp','legacy/pow_midp_unchecked']
s='const LABELS: &[&str] = &'+repr(labels).replace("'",'"')+';\n'
s += '#[inline(never)]\nfn loop1<const FLAGS:u32,const ARM:usize>(input:&[f32],output:&mut[f32],n:f32,chain:bool){let mut acc=0.75;for (&x,y)in input.iter().zip(output){let x=if chain{acc}else{x};let v=magetypes::simd::generic::f32x4::splat_t(archmage::ScalarToken,x);let z=match ARM{0=>magetypes::powf::try_powf_with::<FLAGS>(x,n).unwrap(),1=>v.pow_lowp(n).to_array()[0],2=>v.pow_midp(n).to_array()[0],3=>v.pow_midp_unchecked(n).to_array()[0],_=>unreachable!()};*y=z;if chain{acc=normalize(z);}}}\nfn execute1(arm:usize,_context:&str,input:&[f32],output:&mut[f32],n:f32,chain:bool){match arm{\n'+''.join(f'{i}=>loop1::<{f},0>(input,output,n,chain),\n'for i,f in enumerate(flags))+''.join(f'{64+i}=>loop1::<0,{i+1}>(input,output,n,chain),\n'for i in range(3))+'_=>unreachable!()}}\n'
for width in [4,8,16]:
 s+=f'''#[archmage::magetypes(v4x(cfg(avx512)),v3,neon,wasm128,scalar)]
#[inline(never)]
fn loop{width}<const FLAGS:u32,const ARM:usize>(token:Token,input:&[f32],output:&mut[f32],n:f32,chain:bool) {{
 use magetypes::simd::generic::{{f32x{width} as V, f32x4 as Tail}};
 let mut acc=V::splat_t(token,0.75);
 let mut chunks=input.chunks_exact({width});let mut out=output.chunks_exact_mut({width});
 for (x,y) in chunks.by_ref().zip(out.by_ref()){{
  let v=if chain{{acc}}else{{V::from_array_t(token,x.try_into().unwrap())}};
  let result=match ARM{{0=>v.try_powf_with::<FLAGS>(n).unwrap(),1=>v.pow_lowp(n),2=>v.pow_midp(n),3=>v.pow_midp_unchecked(n),_=>unreachable!()}};
  *<&mut [f32;{width}]>::try_from(y).unwrap()=result.to_array();
  if chain{{acc=V::from_array_t(token,result.to_array().map(normalize));}}
 }}
 for (&x,y) in chunks.remainder().iter().zip(out.into_remainder()){{
  let v=Tail::splat_t(token,x);
  *y=match ARM{{0=>v.try_powf_with::<FLAGS>(n).unwrap(),1=>v.pow_lowp(n),2=>v.pow_midp(n),3=>v.pow_midp_unchecked(n),_=>unreachable!()}}.to_array()[0];
 }}
}}
fn execute{width}(arm:usize,context:&str,input:&[f32],output:&mut[f32],n:f32,chain:bool){{
 macro_rules! call {{($fun:ident,$t:expr)=>{{match arm{{
'''
 for i,f in enumerate(flags):s+=f'{i}=>$fun::<{f},0>($t,input,output,n,chain),\n'
 for i in range(3):s+=f'{64+i}=>$fun::<0,{i+1}>($t,input,output,n,chain),\n'
 s+=f'''_=>unreachable!()}}}}}}
 match context{{
 "scalar"=>call!(loop{width}_scalar,archmage::ScalarToken),
 #[cfg(target_arch="x86_64")]"v3"=>call!(loop{width}_v3,archmage::X64V3Token::summon().unwrap()),
 #[cfg(all(target_arch="x86_64",feature="avx512"))]"v4x"=>call!(loop{width}_v4x,archmage::X64V4xToken::summon().unwrap()),
 #[cfg(target_arch="aarch64")]"neon"=>call!(loop{width}_neon,archmage::NeonToken::summon().unwrap()),
 #[cfg(target_arch="wasm32")]"wasm128"=>call!(loop{width}_wasm128,archmage::Wasm128Token::summon().unwrap()),
 _=>panic!("unsupported context")
 }}
}}
'''
s+='fn scalar_api(arm:usize,x:f32,n:f32)->f32{match arm{\n'+''.join(f'{i}=>magetypes::powf::try_powf_with::<{f}>(x,n).unwrap(),\n'for i,f in enumerate(flags))+'_=>unreachable!()}}\n'
s+='''fn execute(width:usize,arm:usize,context:&str,input:&[f32],output:&mut[f32],n:f32,chain:bool){match width{1=>execute1(arm,context,input,output,n,chain),4=>execute4(arm,context,input,output,n,chain),8=>execute8(arm,context,input,output,n,chain),16=>execute16(arm,context,input,output,n,chain),_=>unreachable!()}}
fn contexts()->Vec<&'static str>{let mut v=vec!["scalar"];#[cfg(target_arch="x86_64")]v.push("v3");#[cfg(all(target_arch="x86_64",feature="avx512"))]v.push("v4x");#[cfg(target_arch="aarch64")]v.push("neon");#[cfg(target_arch="wasm32")]v.push("wasm128");v}
fn verify(){
 let mut state=0x9e3779b9u32;
 for context in contexts(){for arm in 0..64{for n in [0.,-0.,0.5,2.4,4.]{
  let input:Vec<f32>=(0..513).map(|i|{state^=state<<13;state^=state>>17;state^=state<<5;
   if arm<48 && arm%3!=0 && i%17==0{0.}else{f32::from_bits(0x37800000+state%0x10000001)}}).collect();
  let mut reference=vec![0.;input.len()];
  for (x,y) in input.iter().zip(&mut reference){execute(4,arm,context,core::slice::from_ref(x),core::slice::from_mut(y),n,false);}
  for width in [4,8,16]{for length in (0..=33).chain([64,65,127,255,511,512,513]){let mut out=vec![0.;length];execute(width,arm,context,&input[..length],&mut out,n,false);assert_eq!(out.iter().map(|x|x.to_bits()).collect::<Vec<_>>(),reference[..length].iter().map(|x|x.to_bits()).collect::<Vec<_>>(),"{context} {} x{width} len{length} exponent{n}",LABELS[arm]);}}
  // Automatic scalar API must use the same selected graph as automatic vectors.
  if (arm<48 && arm%6<3)||(arm>=48 && arm%2==0) {for (&x,&expected) in input.iter().zip(&reference){assert_eq!(scalar_api(arm,x,n).to_bits(),expected.to_bits(),"public scalar {} {context}",LABELS[arm]);}}
 }}}
 println!("all 64 const policies: single-lane/vector/tails agree at lengths 0..33, 5 exponents, every selected context");
}
fn register(suite:&mut zenbench::Suite){
 let context=std::env::var("MIDLOW_CONTEXT").unwrap_or("v3".into());
 let width=std::env::var("MIDLOW_WIDTH").map_or(8,|s|s.parse().unwrap());
 let size=std::env::var("MIDLOW_SIZE").map_or(64,|s|s.parse().unwrap());
 let chain=std::env::var("MIDLOW_CHAIN").is_ok_and(|s|s=="1");
 let n=f32::from_bits(std::env::var("MIDLOW_EXP").map_or(0x4019999a,|s|u32::from_str_radix(&s,16).unwrap()));
 let arms:Vec<usize>=std::env::var("MIDLOW_ARMS").unwrap_or("29,35,41,47,57,59,61,63,64,65,66".into()).split(',').map(|s|s.parse().unwrap()).collect();
 let input:Vec<f32>=(0..size).map(|i|((i%255)+1)as f32/255.).collect();
 let name=format!("midlow/{context}/x{width}/{size}/{}/{:08x}",if chain{"chain"}else{"stream"},n.to_bits());
 suite.group(&name,|g|{
  g.config().min_rounds(50).max_rounds(50).auto_rounds(false).warmup_time(std::time::Duration::from_millis(10)).max_time(std::time::Duration::from_millis(100)).max_wall_time(std::time::Duration::from_secs(180));
  for arm in arms {let input=input.clone();let context=context.clone();g.bench(LABELS[arm],move|b|{let mut output=vec![0.;size];execute(width,arm,&context,&input,&mut output,n,chain);b.iter(||execute(width,arm,&context,std::hint::black_box(&input),std::hint::black_box(&mut output),std::hint::black_box(n),chain));std::hint::black_box(output);});}
 });
}
'''
Path(__file__).with_name('midlow_generated.rs').write_text(s)

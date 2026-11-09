const LABELS: &[&str] = &["portable/midp/positive/checked/auto", "portable/midp/nonnegative/checked/auto", "portable/midp/gamma/checked/auto", "portable/midp/positive/checked/caller", "portable/midp/nonnegative/checked/caller", "portable/midp/gamma/checked/caller", "portable/lowp/positive/checked/auto", "portable/lowp/nonnegative/checked/auto", "portable/lowp/gamma/checked/auto", "portable/lowp/positive/checked/caller", "portable/lowp/nonnegative/checked/caller", "portable/lowp/gamma/checked/caller", "backend/midp/positive/checked/auto", "backend/midp/nonnegative/checked/auto", "backend/midp/gamma/checked/auto", "backend/midp/positive/checked/caller", "backend/midp/nonnegative/checked/caller", "backend/midp/gamma/checked/caller", "backend/lowp/positive/checked/auto", "backend/lowp/nonnegative/checked/auto", "backend/lowp/gamma/checked/auto", "backend/lowp/positive/checked/caller", "backend/lowp/nonnegative/checked/caller", "backend/lowp/gamma/checked/caller", "portable/midp/positive/assume/auto", "portable/midp/nonnegative/assume/auto", "portable/midp/gamma/assume/auto", "portable/midp/positive/assume/caller", "portable/midp/nonnegative/assume/caller", "portable/midp/gamma/assume/caller", "portable/lowp/positive/assume/auto", "portable/lowp/nonnegative/assume/auto", "portable/lowp/gamma/assume/auto", "portable/lowp/positive/assume/caller", "portable/lowp/nonnegative/assume/caller", "portable/lowp/gamma/assume/caller", "backend/midp/positive/assume/auto", "backend/midp/nonnegative/assume/auto", "backend/midp/gamma/assume/auto", "backend/midp/positive/assume/caller", "backend/midp/nonnegative/assume/caller", "backend/midp/gamma/assume/caller", "backend/lowp/positive/assume/auto", "backend/lowp/nonnegative/assume/auto", "backend/lowp/gamma/assume/auto", "backend/lowp/positive/assume/caller", "backend/lowp/nonnegative/assume/caller", "backend/lowp/gamma/assume/caller", "legacy/pow_lowp", "legacy/pow_midp", "legacy/pow_midp_unchecked"];
#[archmage::magetypes(v4x(cfg(avx512)),v3,neon,wasm128,scalar)]
#[inline(never)]
fn loop4<const FLAGS:u32,const ARM:usize>(token:Token,input:&[f32],output:&mut[f32],n:f32,chain:bool) {
 use magetypes::simd::generic::{f32x4 as V, f32x4 as Tail};
 let mut acc=V::splat_t(token,0.75);
 let mut chunks=input.chunks_exact(4);let mut out=output.chunks_exact_mut(4);
 for (x,y) in chunks.by_ref().zip(out.by_ref()){
  let v=if chain{acc}else{V::from_array_t(token,x.try_into().unwrap())};
  let result=match ARM{0=>v.try_powf_with::<FLAGS>(n).unwrap(),1=>v.pow_lowp(n),2=>v.pow_midp(n),3=>v.pow_midp_unchecked(n),_=>unreachable!()};
  *<&mut [f32;4]>::try_from(y).unwrap()=result.to_array();
  if chain{acc=V::from_array_t(token,result.to_array().map(normalize));}
 }
 for (&x,y) in chunks.remainder().iter().zip(out.into_remainder()){
  let v=Tail::splat_t(token,x);
  *y=match ARM{0=>v.try_powf_with::<FLAGS>(n).unwrap(),1=>v.pow_lowp(n),2=>v.pow_midp(n),3=>v.pow_midp_unchecked(n),_=>unreachable!()}.to_array()[0];
 }
}
fn execute4(arm:usize,context:&str,input:&[f32],output:&mut[f32],n:f32,chain:bool){
 macro_rules! call {($fun:ident,$t:expr)=>{match arm{
0=>$fun::<0,0>($t,input,output,n,chain),
1=>$fun::<1,0>($t,input,output,n,chain),
2=>$fun::<2,0>($t,input,output,n,chain),
3=>$fun::<4,0>($t,input,output,n,chain),
4=>$fun::<5,0>($t,input,output,n,chain),
5=>$fun::<6,0>($t,input,output,n,chain),
6=>$fun::<8,0>($t,input,output,n,chain),
7=>$fun::<9,0>($t,input,output,n,chain),
8=>$fun::<10,0>($t,input,output,n,chain),
9=>$fun::<12,0>($t,input,output,n,chain),
10=>$fun::<13,0>($t,input,output,n,chain),
11=>$fun::<14,0>($t,input,output,n,chain),
12=>$fun::<16,0>($t,input,output,n,chain),
13=>$fun::<17,0>($t,input,output,n,chain),
14=>$fun::<18,0>($t,input,output,n,chain),
15=>$fun::<20,0>($t,input,output,n,chain),
16=>$fun::<21,0>($t,input,output,n,chain),
17=>$fun::<22,0>($t,input,output,n,chain),
18=>$fun::<24,0>($t,input,output,n,chain),
19=>$fun::<25,0>($t,input,output,n,chain),
20=>$fun::<26,0>($t,input,output,n,chain),
21=>$fun::<28,0>($t,input,output,n,chain),
22=>$fun::<29,0>($t,input,output,n,chain),
23=>$fun::<30,0>($t,input,output,n,chain),
24=>$fun::<32,0>($t,input,output,n,chain),
25=>$fun::<33,0>($t,input,output,n,chain),
26=>$fun::<34,0>($t,input,output,n,chain),
27=>$fun::<36,0>($t,input,output,n,chain),
28=>$fun::<37,0>($t,input,output,n,chain),
29=>$fun::<38,0>($t,input,output,n,chain),
30=>$fun::<40,0>($t,input,output,n,chain),
31=>$fun::<41,0>($t,input,output,n,chain),
32=>$fun::<42,0>($t,input,output,n,chain),
33=>$fun::<44,0>($t,input,output,n,chain),
34=>$fun::<45,0>($t,input,output,n,chain),
35=>$fun::<46,0>($t,input,output,n,chain),
36=>$fun::<48,0>($t,input,output,n,chain),
37=>$fun::<49,0>($t,input,output,n,chain),
38=>$fun::<50,0>($t,input,output,n,chain),
39=>$fun::<52,0>($t,input,output,n,chain),
40=>$fun::<53,0>($t,input,output,n,chain),
41=>$fun::<54,0>($t,input,output,n,chain),
42=>$fun::<56,0>($t,input,output,n,chain),
43=>$fun::<57,0>($t,input,output,n,chain),
44=>$fun::<58,0>($t,input,output,n,chain),
45=>$fun::<60,0>($t,input,output,n,chain),
46=>$fun::<61,0>($t,input,output,n,chain),
47=>$fun::<62,0>($t,input,output,n,chain),
48=>$fun::<0,1>($t,input,output,n,chain),
49=>$fun::<0,2>($t,input,output,n,chain),
50=>$fun::<0,3>($t,input,output,n,chain),
_=>unreachable!()}}}
 match context{
 "scalar"=>call!(loop4_scalar,archmage::ScalarToken),
 #[cfg(target_arch="x86_64")]"v3"=>call!(loop4_v3,archmage::X64V3Token::summon().unwrap()),
 #[cfg(all(target_arch="x86_64",feature="avx512"))]"v4x"=>call!(loop4_v4x,archmage::X64V4xToken::summon().unwrap()),
 #[cfg(target_arch="aarch64")]"neon"=>call!(loop4_neon,archmage::NeonToken::summon().unwrap()),
 #[cfg(target_arch="wasm32")]"wasm128"=>call!(loop4_wasm128,archmage::Wasm128Token::summon().unwrap()),
 _=>panic!("unsupported context")
 }
}
#[archmage::magetypes(v4x(cfg(avx512)),v3,neon,wasm128,scalar)]
#[inline(never)]
fn loop8<const FLAGS:u32,const ARM:usize>(token:Token,input:&[f32],output:&mut[f32],n:f32,chain:bool) {
 use magetypes::simd::generic::{f32x8 as V, f32x4 as Tail};
 let mut acc=V::splat_t(token,0.75);
 let mut chunks=input.chunks_exact(8);let mut out=output.chunks_exact_mut(8);
 for (x,y) in chunks.by_ref().zip(out.by_ref()){
  let v=if chain{acc}else{V::from_array_t(token,x.try_into().unwrap())};
  let result=match ARM{0=>v.try_powf_with::<FLAGS>(n).unwrap(),1=>v.pow_lowp(n),2=>v.pow_midp(n),3=>v.pow_midp_unchecked(n),_=>unreachable!()};
  *<&mut [f32;8]>::try_from(y).unwrap()=result.to_array();
  if chain{acc=V::from_array_t(token,result.to_array().map(normalize));}
 }
 for (&x,y) in chunks.remainder().iter().zip(out.into_remainder()){
  let v=Tail::splat_t(token,x);
  *y=match ARM{0=>v.try_powf_with::<FLAGS>(n).unwrap(),1=>v.pow_lowp(n),2=>v.pow_midp(n),3=>v.pow_midp_unchecked(n),_=>unreachable!()}.to_array()[0];
 }
}
fn execute8(arm:usize,context:&str,input:&[f32],output:&mut[f32],n:f32,chain:bool){
 macro_rules! call {($fun:ident,$t:expr)=>{match arm{
0=>$fun::<0,0>($t,input,output,n,chain),
1=>$fun::<1,0>($t,input,output,n,chain),
2=>$fun::<2,0>($t,input,output,n,chain),
3=>$fun::<4,0>($t,input,output,n,chain),
4=>$fun::<5,0>($t,input,output,n,chain),
5=>$fun::<6,0>($t,input,output,n,chain),
6=>$fun::<8,0>($t,input,output,n,chain),
7=>$fun::<9,0>($t,input,output,n,chain),
8=>$fun::<10,0>($t,input,output,n,chain),
9=>$fun::<12,0>($t,input,output,n,chain),
10=>$fun::<13,0>($t,input,output,n,chain),
11=>$fun::<14,0>($t,input,output,n,chain),
12=>$fun::<16,0>($t,input,output,n,chain),
13=>$fun::<17,0>($t,input,output,n,chain),
14=>$fun::<18,0>($t,input,output,n,chain),
15=>$fun::<20,0>($t,input,output,n,chain),
16=>$fun::<21,0>($t,input,output,n,chain),
17=>$fun::<22,0>($t,input,output,n,chain),
18=>$fun::<24,0>($t,input,output,n,chain),
19=>$fun::<25,0>($t,input,output,n,chain),
20=>$fun::<26,0>($t,input,output,n,chain),
21=>$fun::<28,0>($t,input,output,n,chain),
22=>$fun::<29,0>($t,input,output,n,chain),
23=>$fun::<30,0>($t,input,output,n,chain),
24=>$fun::<32,0>($t,input,output,n,chain),
25=>$fun::<33,0>($t,input,output,n,chain),
26=>$fun::<34,0>($t,input,output,n,chain),
27=>$fun::<36,0>($t,input,output,n,chain),
28=>$fun::<37,0>($t,input,output,n,chain),
29=>$fun::<38,0>($t,input,output,n,chain),
30=>$fun::<40,0>($t,input,output,n,chain),
31=>$fun::<41,0>($t,input,output,n,chain),
32=>$fun::<42,0>($t,input,output,n,chain),
33=>$fun::<44,0>($t,input,output,n,chain),
34=>$fun::<45,0>($t,input,output,n,chain),
35=>$fun::<46,0>($t,input,output,n,chain),
36=>$fun::<48,0>($t,input,output,n,chain),
37=>$fun::<49,0>($t,input,output,n,chain),
38=>$fun::<50,0>($t,input,output,n,chain),
39=>$fun::<52,0>($t,input,output,n,chain),
40=>$fun::<53,0>($t,input,output,n,chain),
41=>$fun::<54,0>($t,input,output,n,chain),
42=>$fun::<56,0>($t,input,output,n,chain),
43=>$fun::<57,0>($t,input,output,n,chain),
44=>$fun::<58,0>($t,input,output,n,chain),
45=>$fun::<60,0>($t,input,output,n,chain),
46=>$fun::<61,0>($t,input,output,n,chain),
47=>$fun::<62,0>($t,input,output,n,chain),
48=>$fun::<0,1>($t,input,output,n,chain),
49=>$fun::<0,2>($t,input,output,n,chain),
50=>$fun::<0,3>($t,input,output,n,chain),
_=>unreachable!()}}}
 match context{
 "scalar"=>call!(loop8_scalar,archmage::ScalarToken),
 #[cfg(target_arch="x86_64")]"v3"=>call!(loop8_v3,archmage::X64V3Token::summon().unwrap()),
 #[cfg(all(target_arch="x86_64",feature="avx512"))]"v4x"=>call!(loop8_v4x,archmage::X64V4xToken::summon().unwrap()),
 #[cfg(target_arch="aarch64")]"neon"=>call!(loop8_neon,archmage::NeonToken::summon().unwrap()),
 #[cfg(target_arch="wasm32")]"wasm128"=>call!(loop8_wasm128,archmage::Wasm128Token::summon().unwrap()),
 _=>panic!("unsupported context")
 }
}
#[archmage::magetypes(v4x(cfg(avx512)),v3,neon,wasm128,scalar)]
#[inline(never)]
fn loop16<const FLAGS:u32,const ARM:usize>(token:Token,input:&[f32],output:&mut[f32],n:f32,chain:bool) {
 use magetypes::simd::generic::{f32x16 as V, f32x4 as Tail};
 let mut acc=V::splat_t(token,0.75);
 let mut chunks=input.chunks_exact(16);let mut out=output.chunks_exact_mut(16);
 for (x,y) in chunks.by_ref().zip(out.by_ref()){
  let v=if chain{acc}else{V::from_array_t(token,x.try_into().unwrap())};
  let result=match ARM{0=>v.try_powf_with::<FLAGS>(n).unwrap(),1=>v.pow_lowp(n),2=>v.pow_midp(n),3=>v.pow_midp_unchecked(n),_=>unreachable!()};
  *<&mut [f32;16]>::try_from(y).unwrap()=result.to_array();
  if chain{acc=V::from_array_t(token,result.to_array().map(normalize));}
 }
 for (&x,y) in chunks.remainder().iter().zip(out.into_remainder()){
  let v=Tail::splat_t(token,x);
  *y=match ARM{0=>v.try_powf_with::<FLAGS>(n).unwrap(),1=>v.pow_lowp(n),2=>v.pow_midp(n),3=>v.pow_midp_unchecked(n),_=>unreachable!()}.to_array()[0];
 }
}
fn execute16(arm:usize,context:&str,input:&[f32],output:&mut[f32],n:f32,chain:bool){
 macro_rules! call {($fun:ident,$t:expr)=>{match arm{
0=>$fun::<0,0>($t,input,output,n,chain),
1=>$fun::<1,0>($t,input,output,n,chain),
2=>$fun::<2,0>($t,input,output,n,chain),
3=>$fun::<4,0>($t,input,output,n,chain),
4=>$fun::<5,0>($t,input,output,n,chain),
5=>$fun::<6,0>($t,input,output,n,chain),
6=>$fun::<8,0>($t,input,output,n,chain),
7=>$fun::<9,0>($t,input,output,n,chain),
8=>$fun::<10,0>($t,input,output,n,chain),
9=>$fun::<12,0>($t,input,output,n,chain),
10=>$fun::<13,0>($t,input,output,n,chain),
11=>$fun::<14,0>($t,input,output,n,chain),
12=>$fun::<16,0>($t,input,output,n,chain),
13=>$fun::<17,0>($t,input,output,n,chain),
14=>$fun::<18,0>($t,input,output,n,chain),
15=>$fun::<20,0>($t,input,output,n,chain),
16=>$fun::<21,0>($t,input,output,n,chain),
17=>$fun::<22,0>($t,input,output,n,chain),
18=>$fun::<24,0>($t,input,output,n,chain),
19=>$fun::<25,0>($t,input,output,n,chain),
20=>$fun::<26,0>($t,input,output,n,chain),
21=>$fun::<28,0>($t,input,output,n,chain),
22=>$fun::<29,0>($t,input,output,n,chain),
23=>$fun::<30,0>($t,input,output,n,chain),
24=>$fun::<32,0>($t,input,output,n,chain),
25=>$fun::<33,0>($t,input,output,n,chain),
26=>$fun::<34,0>($t,input,output,n,chain),
27=>$fun::<36,0>($t,input,output,n,chain),
28=>$fun::<37,0>($t,input,output,n,chain),
29=>$fun::<38,0>($t,input,output,n,chain),
30=>$fun::<40,0>($t,input,output,n,chain),
31=>$fun::<41,0>($t,input,output,n,chain),
32=>$fun::<42,0>($t,input,output,n,chain),
33=>$fun::<44,0>($t,input,output,n,chain),
34=>$fun::<45,0>($t,input,output,n,chain),
35=>$fun::<46,0>($t,input,output,n,chain),
36=>$fun::<48,0>($t,input,output,n,chain),
37=>$fun::<49,0>($t,input,output,n,chain),
38=>$fun::<50,0>($t,input,output,n,chain),
39=>$fun::<52,0>($t,input,output,n,chain),
40=>$fun::<53,0>($t,input,output,n,chain),
41=>$fun::<54,0>($t,input,output,n,chain),
42=>$fun::<56,0>($t,input,output,n,chain),
43=>$fun::<57,0>($t,input,output,n,chain),
44=>$fun::<58,0>($t,input,output,n,chain),
45=>$fun::<60,0>($t,input,output,n,chain),
46=>$fun::<61,0>($t,input,output,n,chain),
47=>$fun::<62,0>($t,input,output,n,chain),
48=>$fun::<0,1>($t,input,output,n,chain),
49=>$fun::<0,2>($t,input,output,n,chain),
50=>$fun::<0,3>($t,input,output,n,chain),
_=>unreachable!()}}}
 match context{
 "scalar"=>call!(loop16_scalar,archmage::ScalarToken),
 #[cfg(target_arch="x86_64")]"v3"=>call!(loop16_v3,archmage::X64V3Token::summon().unwrap()),
 #[cfg(all(target_arch="x86_64",feature="avx512"))]"v4x"=>call!(loop16_v4x,archmage::X64V4xToken::summon().unwrap()),
 #[cfg(target_arch="aarch64")]"neon"=>call!(loop16_neon,archmage::NeonToken::summon().unwrap()),
 #[cfg(target_arch="wasm32")]"wasm128"=>call!(loop16_wasm128,archmage::Wasm128Token::summon().unwrap()),
 _=>panic!("unsupported context")
 }
}
fn scalar_api(arm:usize,x:f32,n:f32)->f32{match arm{
0=>magetypes::powf::try_powf_with::<0>(x,n).unwrap(),
1=>magetypes::powf::try_powf_with::<1>(x,n).unwrap(),
2=>magetypes::powf::try_powf_with::<2>(x,n).unwrap(),
3=>magetypes::powf::try_powf_with::<4>(x,n).unwrap(),
4=>magetypes::powf::try_powf_with::<5>(x,n).unwrap(),
5=>magetypes::powf::try_powf_with::<6>(x,n).unwrap(),
6=>magetypes::powf::try_powf_with::<8>(x,n).unwrap(),
7=>magetypes::powf::try_powf_with::<9>(x,n).unwrap(),
8=>magetypes::powf::try_powf_with::<10>(x,n).unwrap(),
9=>magetypes::powf::try_powf_with::<12>(x,n).unwrap(),
10=>magetypes::powf::try_powf_with::<13>(x,n).unwrap(),
11=>magetypes::powf::try_powf_with::<14>(x,n).unwrap(),
12=>magetypes::powf::try_powf_with::<16>(x,n).unwrap(),
13=>magetypes::powf::try_powf_with::<17>(x,n).unwrap(),
14=>magetypes::powf::try_powf_with::<18>(x,n).unwrap(),
15=>magetypes::powf::try_powf_with::<20>(x,n).unwrap(),
16=>magetypes::powf::try_powf_with::<21>(x,n).unwrap(),
17=>magetypes::powf::try_powf_with::<22>(x,n).unwrap(),
18=>magetypes::powf::try_powf_with::<24>(x,n).unwrap(),
19=>magetypes::powf::try_powf_with::<25>(x,n).unwrap(),
20=>magetypes::powf::try_powf_with::<26>(x,n).unwrap(),
21=>magetypes::powf::try_powf_with::<28>(x,n).unwrap(),
22=>magetypes::powf::try_powf_with::<29>(x,n).unwrap(),
23=>magetypes::powf::try_powf_with::<30>(x,n).unwrap(),
24=>magetypes::powf::try_powf_with::<32>(x,n).unwrap(),
25=>magetypes::powf::try_powf_with::<33>(x,n).unwrap(),
26=>magetypes::powf::try_powf_with::<34>(x,n).unwrap(),
27=>magetypes::powf::try_powf_with::<36>(x,n).unwrap(),
28=>magetypes::powf::try_powf_with::<37>(x,n).unwrap(),
29=>magetypes::powf::try_powf_with::<38>(x,n).unwrap(),
30=>magetypes::powf::try_powf_with::<40>(x,n).unwrap(),
31=>magetypes::powf::try_powf_with::<41>(x,n).unwrap(),
32=>magetypes::powf::try_powf_with::<42>(x,n).unwrap(),
33=>magetypes::powf::try_powf_with::<44>(x,n).unwrap(),
34=>magetypes::powf::try_powf_with::<45>(x,n).unwrap(),
35=>magetypes::powf::try_powf_with::<46>(x,n).unwrap(),
36=>magetypes::powf::try_powf_with::<48>(x,n).unwrap(),
37=>magetypes::powf::try_powf_with::<49>(x,n).unwrap(),
38=>magetypes::powf::try_powf_with::<50>(x,n).unwrap(),
39=>magetypes::powf::try_powf_with::<52>(x,n).unwrap(),
40=>magetypes::powf::try_powf_with::<53>(x,n).unwrap(),
41=>magetypes::powf::try_powf_with::<54>(x,n).unwrap(),
42=>magetypes::powf::try_powf_with::<56>(x,n).unwrap(),
43=>magetypes::powf::try_powf_with::<57>(x,n).unwrap(),
44=>magetypes::powf::try_powf_with::<58>(x,n).unwrap(),
45=>magetypes::powf::try_powf_with::<60>(x,n).unwrap(),
46=>magetypes::powf::try_powf_with::<61>(x,n).unwrap(),
47=>magetypes::powf::try_powf_with::<62>(x,n).unwrap(),
_=>unreachable!()}}
fn execute(width:usize,arm:usize,context:&str,input:&[f32],output:&mut[f32],n:f32,chain:bool){match width{4=>execute4(arm,context,input,output,n,chain),8=>execute8(arm,context,input,output,n,chain),16=>execute16(arm,context,input,output,n,chain),_=>unreachable!()}}
fn contexts()->Vec<&'static str>{let mut v=vec!["scalar"];#[cfg(target_arch="x86_64")]v.push("v3");#[cfg(all(target_arch="x86_64",feature="avx512"))]v.push("v4x");#[cfg(target_arch="aarch64")]v.push("neon");#[cfg(target_arch="wasm32")]v.push("wasm128");v}
fn verify(){
 let mut state=0x9e3779b9u32;
 for context in contexts(){for arm in 0..48{for n in [0.,-0.,0.5,2.4,4.]{
  let input:Vec<f32>=(0..513).map(|i|{state^=state<<13;state^=state>>17;state^=state<<5;
   if arm%3!=0 && i%17==0{0.}else{f32::from_bits(0x37800000+state%0x10000001)}}).collect();
  let mut reference=vec![0.;input.len()];
  for (x,y) in input.iter().zip(&mut reference){execute(4,arm,context,core::slice::from_ref(x),core::slice::from_mut(y),n,false);}
  for width in [4,8,16]{for length in (0..=33).chain([64,65,127,255,511,512,513]){let mut out=vec![0.;length];execute(width,arm,context,&input[..length],&mut out,n,false);assert_eq!(out.iter().map(|x|x.to_bits()).collect::<Vec<_>>(),reference[..length].iter().map(|x|x.to_bits()).collect::<Vec<_>>(),"{context} {} x{width} len{length} exponent{n}",LABELS[arm]);}}
  // Automatic scalar API must use the same selected graph as automatic vectors.
  if arm%6<3 {for (&x,&expected) in input.iter().zip(&reference){assert_eq!(scalar_api(arm,x,n).to_bits(),expected.to_bits(),"public scalar {} {context}",LABELS[arm]);}}
 }}}
 println!("all 48 const policies: single-lane/vector/tails agree at lengths 0..33, 5 exponents, every selected context");
}
fn register(suite:&mut zenbench::Suite){
 let context=std::env::var("MIDLOW_CONTEXT").unwrap_or("v3".into());
 let width=std::env::var("MIDLOW_WIDTH").map_or(8,|s|s.parse().unwrap());
 let size=std::env::var("MIDLOW_SIZE").map_or(64,|s|s.parse().unwrap());
 let chain=std::env::var("MIDLOW_CHAIN").is_ok_and(|s|s=="1");
 let n=f32::from_bits(std::env::var("MIDLOW_EXP").map_or(0x4019999a,|s|u32::from_str_radix(&s,16).unwrap()));
 let arms:Vec<usize>=std::env::var("MIDLOW_ARMS").unwrap_or("29,35,41,47,48,49,50".into()).split(',').map(|s|s.parse().unwrap()).collect();
 let input:Vec<f32>=(0..size).map(|i|((i%255)+1)as f32/255.).collect();
 let name=format!("midlow/{context}/x{width}/{size}/{}/{:08x}",if chain{"chain"}else{"stream"},n.to_bits());
 suite.group(&name,|g|{
  g.config().min_rounds(50).max_rounds(50).auto_rounds(false).warmup_time(std::time::Duration::from_millis(10)).max_time(std::time::Duration::from_millis(100)).max_wall_time(std::time::Duration::from_secs(180));
  for arm in arms {let input=input.clone();let context=context.clone();g.bench(LABELS[arm],move|b|{let mut output=vec![0.;size];execute(width,arm,&context,&input,&mut output,n,chain);b.iter(||execute(width,arm,&context,std::hint::black_box(&input),std::hint::black_box(&mut output),std::hint::black_box(n),chain));std::hint::black_box(output);});}
 });
}

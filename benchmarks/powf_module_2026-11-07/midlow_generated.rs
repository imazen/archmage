const LABELS: &[&str] = &["portable/midp/positive/checked/auto", "portable/midp/nonnegative/checked/auto", "portable/midp/gamma/checked/auto", "portable/midp/positive/checked/caller", "portable/midp/nonnegative/checked/caller", "portable/midp/gamma/checked/caller", "portable/lowp/positive/checked/auto", "portable/lowp/nonnegative/checked/auto", "portable/lowp/gamma/checked/auto", "portable/lowp/positive/checked/caller", "portable/lowp/nonnegative/checked/caller", "portable/lowp/gamma/checked/caller", "backend/midp/positive/checked/auto", "backend/midp/nonnegative/checked/auto", "backend/midp/gamma/checked/auto", "backend/midp/positive/checked/caller", "backend/midp/nonnegative/checked/caller", "backend/midp/gamma/checked/caller", "backend/lowp/positive/checked/auto", "backend/lowp/nonnegative/checked/auto", "backend/lowp/gamma/checked/auto", "backend/lowp/positive/checked/caller", "backend/lowp/nonnegative/checked/caller", "backend/lowp/gamma/checked/caller", "portable/midp/positive/assume/auto", "portable/midp/nonnegative/assume/auto", "portable/midp/gamma/assume/auto", "portable/midp/positive/assume/caller", "portable/midp/nonnegative/assume/caller", "portable/midp/gamma/assume/caller", "portable/lowp/positive/assume/auto", "portable/lowp/nonnegative/assume/auto", "portable/lowp/gamma/assume/auto", "portable/lowp/positive/assume/caller", "portable/lowp/nonnegative/assume/caller", "portable/lowp/gamma/assume/caller", "backend/midp/positive/assume/auto", "backend/midp/nonnegative/assume/auto", "backend/midp/gamma/assume/auto", "backend/midp/positive/assume/caller", "backend/midp/nonnegative/assume/caller", "backend/midp/gamma/assume/caller", "backend/lowp/positive/assume/auto", "backend/lowp/nonnegative/assume/auto", "backend/lowp/gamma/assume/auto", "backend/lowp/positive/assume/caller", "backend/lowp/nonnegative/assume/caller", "backend/lowp/gamma/assume/caller", "portable/midp/positive-gamma/checked/auto", "portable/midp/positive-gamma/checked/caller", "portable/lowp/positive-gamma/checked/auto", "portable/lowp/positive-gamma/checked/caller", "backend/midp/positive-gamma/checked/auto", "backend/midp/positive-gamma/checked/caller", "backend/lowp/positive-gamma/checked/auto", "backend/lowp/positive-gamma/checked/caller", "portable/midp/positive-gamma/assume/auto", "portable/midp/positive-gamma/assume/caller", "portable/lowp/positive-gamma/assume/auto", "portable/lowp/positive-gamma/assume/caller", "backend/midp/positive-gamma/assume/auto", "backend/midp/positive-gamma/assume/caller", "backend/lowp/positive-gamma/assume/auto", "backend/lowp/positive-gamma/assume/caller", "legacy/pow_lowp", "legacy/pow_midp", "legacy/pow_midp_unchecked"];
#[inline(never)]
fn loop1<const FLAGS:u32,const ARM:usize>(input:&[f32],output:&mut[f32],n:f32,chain:bool){let mut acc=0.75;for (&x,y)in input.iter().zip(output){let x=if chain{acc}else{x};let v=magetypes::simd::generic::f32x4::splat_t(archmage::ScalarToken,x);let z=match ARM{0=>magetypes::powf::try_powf_with::<FLAGS>(x,n).unwrap(),1=>v.pow_lowp(n).to_array()[0],2=>v.pow_midp(n).to_array()[0],3=>v.pow_midp_unchecked(n).to_array()[0],_=>unreachable!()};*y=z;if chain{acc=normalize(z);}}}
fn execute1(arm:usize,_context:&str,input:&[f32],output:&mut[f32],n:f32,chain:bool){match arm{
0=>loop1::<0,0>(input,output,n,chain),
1=>loop1::<1,0>(input,output,n,chain),
2=>loop1::<2,0>(input,output,n,chain),
3=>loop1::<4,0>(input,output,n,chain),
4=>loop1::<5,0>(input,output,n,chain),
5=>loop1::<6,0>(input,output,n,chain),
6=>loop1::<8,0>(input,output,n,chain),
7=>loop1::<9,0>(input,output,n,chain),
8=>loop1::<10,0>(input,output,n,chain),
9=>loop1::<12,0>(input,output,n,chain),
10=>loop1::<13,0>(input,output,n,chain),
11=>loop1::<14,0>(input,output,n,chain),
12=>loop1::<16,0>(input,output,n,chain),
13=>loop1::<17,0>(input,output,n,chain),
14=>loop1::<18,0>(input,output,n,chain),
15=>loop1::<20,0>(input,output,n,chain),
16=>loop1::<21,0>(input,output,n,chain),
17=>loop1::<22,0>(input,output,n,chain),
18=>loop1::<24,0>(input,output,n,chain),
19=>loop1::<25,0>(input,output,n,chain),
20=>loop1::<26,0>(input,output,n,chain),
21=>loop1::<28,0>(input,output,n,chain),
22=>loop1::<29,0>(input,output,n,chain),
23=>loop1::<30,0>(input,output,n,chain),
24=>loop1::<32,0>(input,output,n,chain),
25=>loop1::<33,0>(input,output,n,chain),
26=>loop1::<34,0>(input,output,n,chain),
27=>loop1::<36,0>(input,output,n,chain),
28=>loop1::<37,0>(input,output,n,chain),
29=>loop1::<38,0>(input,output,n,chain),
30=>loop1::<40,0>(input,output,n,chain),
31=>loop1::<41,0>(input,output,n,chain),
32=>loop1::<42,0>(input,output,n,chain),
33=>loop1::<44,0>(input,output,n,chain),
34=>loop1::<45,0>(input,output,n,chain),
35=>loop1::<46,0>(input,output,n,chain),
36=>loop1::<48,0>(input,output,n,chain),
37=>loop1::<49,0>(input,output,n,chain),
38=>loop1::<50,0>(input,output,n,chain),
39=>loop1::<52,0>(input,output,n,chain),
40=>loop1::<53,0>(input,output,n,chain),
41=>loop1::<54,0>(input,output,n,chain),
42=>loop1::<56,0>(input,output,n,chain),
43=>loop1::<57,0>(input,output,n,chain),
44=>loop1::<58,0>(input,output,n,chain),
45=>loop1::<60,0>(input,output,n,chain),
46=>loop1::<61,0>(input,output,n,chain),
47=>loop1::<62,0>(input,output,n,chain),
48=>loop1::<130,0>(input,output,n,chain),
49=>loop1::<134,0>(input,output,n,chain),
50=>loop1::<138,0>(input,output,n,chain),
51=>loop1::<142,0>(input,output,n,chain),
52=>loop1::<146,0>(input,output,n,chain),
53=>loop1::<150,0>(input,output,n,chain),
54=>loop1::<154,0>(input,output,n,chain),
55=>loop1::<158,0>(input,output,n,chain),
56=>loop1::<162,0>(input,output,n,chain),
57=>loop1::<166,0>(input,output,n,chain),
58=>loop1::<170,0>(input,output,n,chain),
59=>loop1::<174,0>(input,output,n,chain),
60=>loop1::<178,0>(input,output,n,chain),
61=>loop1::<182,0>(input,output,n,chain),
62=>loop1::<186,0>(input,output,n,chain),
63=>loop1::<190,0>(input,output,n,chain),
64=>loop1::<0,1>(input,output,n,chain),
65=>loop1::<0,2>(input,output,n,chain),
66=>loop1::<0,3>(input,output,n,chain),
_=>unreachable!()}}
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
  *y=match ARM{0=>magetypes::powf::try_powf_with_token::<FLAGS,_>(token,x,n).unwrap(),1=>v.pow_lowp(n).to_array()[0],2=>v.pow_midp(n).to_array()[0],3=>v.pow_midp_unchecked(n).to_array()[0],_=>unreachable!()};
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
48=>$fun::<130,0>($t,input,output,n,chain),
49=>$fun::<134,0>($t,input,output,n,chain),
50=>$fun::<138,0>($t,input,output,n,chain),
51=>$fun::<142,0>($t,input,output,n,chain),
52=>$fun::<146,0>($t,input,output,n,chain),
53=>$fun::<150,0>($t,input,output,n,chain),
54=>$fun::<154,0>($t,input,output,n,chain),
55=>$fun::<158,0>($t,input,output,n,chain),
56=>$fun::<162,0>($t,input,output,n,chain),
57=>$fun::<166,0>($t,input,output,n,chain),
58=>$fun::<170,0>($t,input,output,n,chain),
59=>$fun::<174,0>($t,input,output,n,chain),
60=>$fun::<178,0>($t,input,output,n,chain),
61=>$fun::<182,0>($t,input,output,n,chain),
62=>$fun::<186,0>($t,input,output,n,chain),
63=>$fun::<190,0>($t,input,output,n,chain),
64=>$fun::<0,1>($t,input,output,n,chain),
65=>$fun::<0,2>($t,input,output,n,chain),
66=>$fun::<0,3>($t,input,output,n,chain),
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
  *y=match ARM{0=>magetypes::powf::try_powf_with_token::<FLAGS,_>(token,x,n).unwrap(),1=>v.pow_lowp(n).to_array()[0],2=>v.pow_midp(n).to_array()[0],3=>v.pow_midp_unchecked(n).to_array()[0],_=>unreachable!()};
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
48=>$fun::<130,0>($t,input,output,n,chain),
49=>$fun::<134,0>($t,input,output,n,chain),
50=>$fun::<138,0>($t,input,output,n,chain),
51=>$fun::<142,0>($t,input,output,n,chain),
52=>$fun::<146,0>($t,input,output,n,chain),
53=>$fun::<150,0>($t,input,output,n,chain),
54=>$fun::<154,0>($t,input,output,n,chain),
55=>$fun::<158,0>($t,input,output,n,chain),
56=>$fun::<162,0>($t,input,output,n,chain),
57=>$fun::<166,0>($t,input,output,n,chain),
58=>$fun::<170,0>($t,input,output,n,chain),
59=>$fun::<174,0>($t,input,output,n,chain),
60=>$fun::<178,0>($t,input,output,n,chain),
61=>$fun::<182,0>($t,input,output,n,chain),
62=>$fun::<186,0>($t,input,output,n,chain),
63=>$fun::<190,0>($t,input,output,n,chain),
64=>$fun::<0,1>($t,input,output,n,chain),
65=>$fun::<0,2>($t,input,output,n,chain),
66=>$fun::<0,3>($t,input,output,n,chain),
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
  *y=match ARM{0=>magetypes::powf::try_powf_with_token::<FLAGS,_>(token,x,n).unwrap(),1=>v.pow_lowp(n).to_array()[0],2=>v.pow_midp(n).to_array()[0],3=>v.pow_midp_unchecked(n).to_array()[0],_=>unreachable!()};
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
48=>$fun::<130,0>($t,input,output,n,chain),
49=>$fun::<134,0>($t,input,output,n,chain),
50=>$fun::<138,0>($t,input,output,n,chain),
51=>$fun::<142,0>($t,input,output,n,chain),
52=>$fun::<146,0>($t,input,output,n,chain),
53=>$fun::<150,0>($t,input,output,n,chain),
54=>$fun::<154,0>($t,input,output,n,chain),
55=>$fun::<158,0>($t,input,output,n,chain),
56=>$fun::<162,0>($t,input,output,n,chain),
57=>$fun::<166,0>($t,input,output,n,chain),
58=>$fun::<170,0>($t,input,output,n,chain),
59=>$fun::<174,0>($t,input,output,n,chain),
60=>$fun::<178,0>($t,input,output,n,chain),
61=>$fun::<182,0>($t,input,output,n,chain),
62=>$fun::<186,0>($t,input,output,n,chain),
63=>$fun::<190,0>($t,input,output,n,chain),
64=>$fun::<0,1>($t,input,output,n,chain),
65=>$fun::<0,2>($t,input,output,n,chain),
66=>$fun::<0,3>($t,input,output,n,chain),
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
48=>magetypes::powf::try_powf_with::<130>(x,n).unwrap(),
49=>magetypes::powf::try_powf_with::<134>(x,n).unwrap(),
50=>magetypes::powf::try_powf_with::<138>(x,n).unwrap(),
51=>magetypes::powf::try_powf_with::<142>(x,n).unwrap(),
52=>magetypes::powf::try_powf_with::<146>(x,n).unwrap(),
53=>magetypes::powf::try_powf_with::<150>(x,n).unwrap(),
54=>magetypes::powf::try_powf_with::<154>(x,n).unwrap(),
55=>magetypes::powf::try_powf_with::<158>(x,n).unwrap(),
56=>magetypes::powf::try_powf_with::<162>(x,n).unwrap(),
57=>magetypes::powf::try_powf_with::<166>(x,n).unwrap(),
58=>magetypes::powf::try_powf_with::<170>(x,n).unwrap(),
59=>magetypes::powf::try_powf_with::<174>(x,n).unwrap(),
60=>magetypes::powf::try_powf_with::<178>(x,n).unwrap(),
61=>magetypes::powf::try_powf_with::<182>(x,n).unwrap(),
62=>magetypes::powf::try_powf_with::<186>(x,n).unwrap(),
63=>magetypes::powf::try_powf_with::<190>(x,n).unwrap(),
_=>unreachable!()}}
fn execute(width:usize,arm:usize,context:&str,input:&[f32],output:&mut[f32],n:f32,chain:bool){match width{1=>execute1(arm,context,input,output,n,chain),4=>execute4(arm,context,input,output,n,chain),8=>execute8(arm,context,input,output,n,chain),16=>execute16(arm,context,input,output,n,chain),_=>unreachable!()}}
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

#![forbid(unsafe_code)]
#![allow(dead_code,non_snake_case)]
type Probe<T,const N:usize>=fn(T,&[f32;N],f32,&mut[f32;N]);
#[archmage::magetypes(v4x(cfg(avx512)),v4(cfg(avx512)),v3,neon,wasm128,scalar)]
#[inline(never)]
fn remap_original_lowp_x4_dynamic(token:Token,x:&[f32;4],n:f32,out:&mut[f32;4]){use magetypes::simd::generic::f32x4 as V;let _=n;let v=V::from_array_t(token,*x);*out=({let r=(v.log2_lowp()*V::splat_t(token,n)).exp2_lowp();V::blend(v.simd_eq(V::splat_t(token,0.)),V::splat_t(token,0.),r)}).to_array();}
#[archmage::magetypes(v4x(cfg(avx512)),v4(cfg(avx512)),v3,neon,wasm128,scalar)]
#[inline(never)]
fn remap_shared_lowp_x4_dynamic(token:Token,x:&[f32;4],n:f32,out:&mut[f32;4]){use magetypes::simd::generic::f32x4 as V;let _=n;let v=V::from_array_t(token,*x);*out=(v.pow_lowp(n)).to_array();}
#[used]
static REMAPTABLE_X4_LOWP_DYNAMIC_SCALAR: [Probe<archmage::ScalarToken,4>;2]=[remap_original_lowp_x4_dynamic_scalar,remap_shared_lowp_x4_dynamic_scalar];
#[cfg(target_arch="x86_64")]
#[used]
static REMAPTABLE_X4_LOWP_DYNAMIC_V3: [Probe<archmage::X64V3Token,4>;2]=[remap_original_lowp_x4_dynamic_v3,remap_shared_lowp_x4_dynamic_v3];
#[cfg(all(target_arch="x86_64",feature="avx512"))]
#[used]
static REMAPTABLE_X4_LOWP_DYNAMIC_V4: [Probe<archmage::X64V4Token,4>;2]=[remap_original_lowp_x4_dynamic_v4,remap_shared_lowp_x4_dynamic_v4];
#[cfg(all(target_arch="x86_64",feature="avx512"))]
#[used]
static REMAPTABLE_X4_LOWP_DYNAMIC_V4X: [Probe<archmage::X64V4xToken,4>;2]=[remap_original_lowp_x4_dynamic_v4x,remap_shared_lowp_x4_dynamic_v4x];
#[cfg(target_arch="aarch64")]
#[used]
static REMAPTABLE_X4_LOWP_DYNAMIC_NEON: [Probe<archmage::NeonToken,4>;2]=[remap_original_lowp_x4_dynamic_neon,remap_shared_lowp_x4_dynamic_neon];
#[cfg(target_arch="wasm32")]
#[used]
static REMAPTABLE_X4_LOWP_DYNAMIC_WASM128: [Probe<archmage::Wasm128Token,4>;2]=[remap_original_lowp_x4_dynamic_wasm128,remap_shared_lowp_x4_dynamic_wasm128];
#[archmage::magetypes(v4x(cfg(avx512)),v4(cfg(avx512)),v3,neon,wasm128,scalar)]
#[inline(never)]
fn remap_original_lowp_x4_literal(token:Token,x:&[f32;4],n:f32,out:&mut[f32;4]){use magetypes::simd::generic::f32x4 as V;let _=n;let v=V::from_array_t(token,*x);*out=({let r=(v.log2_lowp()*V::splat_t(token,2.4f32)).exp2_lowp();V::blend(v.simd_eq(V::splat_t(token,0.)),V::splat_t(token,0.),r)}).to_array();}
#[archmage::magetypes(v4x(cfg(avx512)),v4(cfg(avx512)),v3,neon,wasm128,scalar)]
#[inline(never)]
fn remap_shared_lowp_x4_literal(token:Token,x:&[f32;4],n:f32,out:&mut[f32;4]){use magetypes::simd::generic::f32x4 as V;let _=n;let v=V::from_array_t(token,*x);*out=(v.pow_lowp(2.4f32)).to_array();}
#[used]
static REMAPTABLE_X4_LOWP_LITERAL_SCALAR: [Probe<archmage::ScalarToken,4>;2]=[remap_original_lowp_x4_literal_scalar,remap_shared_lowp_x4_literal_scalar];
#[cfg(target_arch="x86_64")]
#[used]
static REMAPTABLE_X4_LOWP_LITERAL_V3: [Probe<archmage::X64V3Token,4>;2]=[remap_original_lowp_x4_literal_v3,remap_shared_lowp_x4_literal_v3];
#[cfg(all(target_arch="x86_64",feature="avx512"))]
#[used]
static REMAPTABLE_X4_LOWP_LITERAL_V4: [Probe<archmage::X64V4Token,4>;2]=[remap_original_lowp_x4_literal_v4,remap_shared_lowp_x4_literal_v4];
#[cfg(all(target_arch="x86_64",feature="avx512"))]
#[used]
static REMAPTABLE_X4_LOWP_LITERAL_V4X: [Probe<archmage::X64V4xToken,4>;2]=[remap_original_lowp_x4_literal_v4x,remap_shared_lowp_x4_literal_v4x];
#[cfg(target_arch="aarch64")]
#[used]
static REMAPTABLE_X4_LOWP_LITERAL_NEON: [Probe<archmage::NeonToken,4>;2]=[remap_original_lowp_x4_literal_neon,remap_shared_lowp_x4_literal_neon];
#[cfg(target_arch="wasm32")]
#[used]
static REMAPTABLE_X4_LOWP_LITERAL_WASM128: [Probe<archmage::Wasm128Token,4>;2]=[remap_original_lowp_x4_literal_wasm128,remap_shared_lowp_x4_literal_wasm128];
#[archmage::magetypes(v4x(cfg(avx512)),v4(cfg(avx512)),v3,neon,wasm128,scalar)]
#[inline(never)]
fn remap_original_midp_x4_dynamic(token:Token,x:&[f32;4],n:f32,out:&mut[f32;4]){use magetypes::simd::generic::f32x4 as V;let _=n;let v=V::from_array_t(token,*x);*out=((v.log2_midp()*V::splat_t(token,n)).exp2_midp()).to_array();}
#[archmage::magetypes(v4x(cfg(avx512)),v4(cfg(avx512)),v3,neon,wasm128,scalar)]
#[inline(never)]
fn remap_shared_midp_x4_dynamic(token:Token,x:&[f32;4],n:f32,out:&mut[f32;4]){use magetypes::simd::generic::f32x4 as V;let _=n;let v=V::from_array_t(token,*x);*out=(v.pow_midp(n)).to_array();}
#[used]
static REMAPTABLE_X4_MIDP_DYNAMIC_SCALAR: [Probe<archmage::ScalarToken,4>;2]=[remap_original_midp_x4_dynamic_scalar,remap_shared_midp_x4_dynamic_scalar];
#[cfg(target_arch="x86_64")]
#[used]
static REMAPTABLE_X4_MIDP_DYNAMIC_V3: [Probe<archmage::X64V3Token,4>;2]=[remap_original_midp_x4_dynamic_v3,remap_shared_midp_x4_dynamic_v3];
#[cfg(all(target_arch="x86_64",feature="avx512"))]
#[used]
static REMAPTABLE_X4_MIDP_DYNAMIC_V4: [Probe<archmage::X64V4Token,4>;2]=[remap_original_midp_x4_dynamic_v4,remap_shared_midp_x4_dynamic_v4];
#[cfg(all(target_arch="x86_64",feature="avx512"))]
#[used]
static REMAPTABLE_X4_MIDP_DYNAMIC_V4X: [Probe<archmage::X64V4xToken,4>;2]=[remap_original_midp_x4_dynamic_v4x,remap_shared_midp_x4_dynamic_v4x];
#[cfg(target_arch="aarch64")]
#[used]
static REMAPTABLE_X4_MIDP_DYNAMIC_NEON: [Probe<archmage::NeonToken,4>;2]=[remap_original_midp_x4_dynamic_neon,remap_shared_midp_x4_dynamic_neon];
#[cfg(target_arch="wasm32")]
#[used]
static REMAPTABLE_X4_MIDP_DYNAMIC_WASM128: [Probe<archmage::Wasm128Token,4>;2]=[remap_original_midp_x4_dynamic_wasm128,remap_shared_midp_x4_dynamic_wasm128];
#[archmage::magetypes(v4x(cfg(avx512)),v4(cfg(avx512)),v3,neon,wasm128,scalar)]
#[inline(never)]
fn remap_original_midp_x4_literal(token:Token,x:&[f32;4],n:f32,out:&mut[f32;4]){use magetypes::simd::generic::f32x4 as V;let _=n;let v=V::from_array_t(token,*x);*out=((v.log2_midp()*V::splat_t(token,2.4f32)).exp2_midp()).to_array();}
#[archmage::magetypes(v4x(cfg(avx512)),v4(cfg(avx512)),v3,neon,wasm128,scalar)]
#[inline(never)]
fn remap_shared_midp_x4_literal(token:Token,x:&[f32;4],n:f32,out:&mut[f32;4]){use magetypes::simd::generic::f32x4 as V;let _=n;let v=V::from_array_t(token,*x);*out=(v.pow_midp(2.4f32)).to_array();}
#[used]
static REMAPTABLE_X4_MIDP_LITERAL_SCALAR: [Probe<archmage::ScalarToken,4>;2]=[remap_original_midp_x4_literal_scalar,remap_shared_midp_x4_literal_scalar];
#[cfg(target_arch="x86_64")]
#[used]
static REMAPTABLE_X4_MIDP_LITERAL_V3: [Probe<archmage::X64V3Token,4>;2]=[remap_original_midp_x4_literal_v3,remap_shared_midp_x4_literal_v3];
#[cfg(all(target_arch="x86_64",feature="avx512"))]
#[used]
static REMAPTABLE_X4_MIDP_LITERAL_V4: [Probe<archmage::X64V4Token,4>;2]=[remap_original_midp_x4_literal_v4,remap_shared_midp_x4_literal_v4];
#[cfg(all(target_arch="x86_64",feature="avx512"))]
#[used]
static REMAPTABLE_X4_MIDP_LITERAL_V4X: [Probe<archmage::X64V4xToken,4>;2]=[remap_original_midp_x4_literal_v4x,remap_shared_midp_x4_literal_v4x];
#[cfg(target_arch="aarch64")]
#[used]
static REMAPTABLE_X4_MIDP_LITERAL_NEON: [Probe<archmage::NeonToken,4>;2]=[remap_original_midp_x4_literal_neon,remap_shared_midp_x4_literal_neon];
#[cfg(target_arch="wasm32")]
#[used]
static REMAPTABLE_X4_MIDP_LITERAL_WASM128: [Probe<archmage::Wasm128Token,4>;2]=[remap_original_midp_x4_literal_wasm128,remap_shared_midp_x4_literal_wasm128];
#[archmage::magetypes(v4x(cfg(avx512)),v4(cfg(avx512)),v3,neon,wasm128,scalar)]
#[inline(never)]
fn remap_original_midp_unchecked_x4_dynamic(token:Token,x:&[f32;4],n:f32,out:&mut[f32;4]){use magetypes::simd::generic::f32x4 as V;let _=n;let v=V::from_array_t(token,*x);*out=((v.log2_midp_unchecked()*V::splat_t(token,n)).exp2_midp_unchecked()).to_array();}
#[archmage::magetypes(v4x(cfg(avx512)),v4(cfg(avx512)),v3,neon,wasm128,scalar)]
#[inline(never)]
fn remap_shared_midp_unchecked_x4_dynamic(token:Token,x:&[f32;4],n:f32,out:&mut[f32;4]){use magetypes::simd::generic::f32x4 as V;let _=n;let v=V::from_array_t(token,*x);*out=(v.pow_midp_unchecked(n)).to_array();}
#[used]
static REMAPTABLE_X4_MIDP_UNCHECKED_DYNAMIC_SCALAR: [Probe<archmage::ScalarToken,4>;2]=[remap_original_midp_unchecked_x4_dynamic_scalar,remap_shared_midp_unchecked_x4_dynamic_scalar];
#[cfg(target_arch="x86_64")]
#[used]
static REMAPTABLE_X4_MIDP_UNCHECKED_DYNAMIC_V3: [Probe<archmage::X64V3Token,4>;2]=[remap_original_midp_unchecked_x4_dynamic_v3,remap_shared_midp_unchecked_x4_dynamic_v3];
#[cfg(all(target_arch="x86_64",feature="avx512"))]
#[used]
static REMAPTABLE_X4_MIDP_UNCHECKED_DYNAMIC_V4: [Probe<archmage::X64V4Token,4>;2]=[remap_original_midp_unchecked_x4_dynamic_v4,remap_shared_midp_unchecked_x4_dynamic_v4];
#[cfg(all(target_arch="x86_64",feature="avx512"))]
#[used]
static REMAPTABLE_X4_MIDP_UNCHECKED_DYNAMIC_V4X: [Probe<archmage::X64V4xToken,4>;2]=[remap_original_midp_unchecked_x4_dynamic_v4x,remap_shared_midp_unchecked_x4_dynamic_v4x];
#[cfg(target_arch="aarch64")]
#[used]
static REMAPTABLE_X4_MIDP_UNCHECKED_DYNAMIC_NEON: [Probe<archmage::NeonToken,4>;2]=[remap_original_midp_unchecked_x4_dynamic_neon,remap_shared_midp_unchecked_x4_dynamic_neon];
#[cfg(target_arch="wasm32")]
#[used]
static REMAPTABLE_X4_MIDP_UNCHECKED_DYNAMIC_WASM128: [Probe<archmage::Wasm128Token,4>;2]=[remap_original_midp_unchecked_x4_dynamic_wasm128,remap_shared_midp_unchecked_x4_dynamic_wasm128];
#[archmage::magetypes(v4x(cfg(avx512)),v4(cfg(avx512)),v3,neon,wasm128,scalar)]
#[inline(never)]
fn remap_original_midp_unchecked_x4_literal(token:Token,x:&[f32;4],n:f32,out:&mut[f32;4]){use magetypes::simd::generic::f32x4 as V;let _=n;let v=V::from_array_t(token,*x);*out=((v.log2_midp_unchecked()*V::splat_t(token,2.4f32)).exp2_midp_unchecked()).to_array();}
#[archmage::magetypes(v4x(cfg(avx512)),v4(cfg(avx512)),v3,neon,wasm128,scalar)]
#[inline(never)]
fn remap_shared_midp_unchecked_x4_literal(token:Token,x:&[f32;4],n:f32,out:&mut[f32;4]){use magetypes::simd::generic::f32x4 as V;let _=n;let v=V::from_array_t(token,*x);*out=(v.pow_midp_unchecked(2.4f32)).to_array();}
#[used]
static REMAPTABLE_X4_MIDP_UNCHECKED_LITERAL_SCALAR: [Probe<archmage::ScalarToken,4>;2]=[remap_original_midp_unchecked_x4_literal_scalar,remap_shared_midp_unchecked_x4_literal_scalar];
#[cfg(target_arch="x86_64")]
#[used]
static REMAPTABLE_X4_MIDP_UNCHECKED_LITERAL_V3: [Probe<archmage::X64V3Token,4>;2]=[remap_original_midp_unchecked_x4_literal_v3,remap_shared_midp_unchecked_x4_literal_v3];
#[cfg(all(target_arch="x86_64",feature="avx512"))]
#[used]
static REMAPTABLE_X4_MIDP_UNCHECKED_LITERAL_V4: [Probe<archmage::X64V4Token,4>;2]=[remap_original_midp_unchecked_x4_literal_v4,remap_shared_midp_unchecked_x4_literal_v4];
#[cfg(all(target_arch="x86_64",feature="avx512"))]
#[used]
static REMAPTABLE_X4_MIDP_UNCHECKED_LITERAL_V4X: [Probe<archmage::X64V4xToken,4>;2]=[remap_original_midp_unchecked_x4_literal_v4x,remap_shared_midp_unchecked_x4_literal_v4x];
#[cfg(target_arch="aarch64")]
#[used]
static REMAPTABLE_X4_MIDP_UNCHECKED_LITERAL_NEON: [Probe<archmage::NeonToken,4>;2]=[remap_original_midp_unchecked_x4_literal_neon,remap_shared_midp_unchecked_x4_literal_neon];
#[cfg(target_arch="wasm32")]
#[used]
static REMAPTABLE_X4_MIDP_UNCHECKED_LITERAL_WASM128: [Probe<archmage::Wasm128Token,4>;2]=[remap_original_midp_unchecked_x4_literal_wasm128,remap_shared_midp_unchecked_x4_literal_wasm128];
#[archmage::magetypes(v4x(cfg(avx512)),v4(cfg(avx512)),v3,neon,wasm128,scalar)]
#[inline(never)]
fn remap_original_lowp_x8_dynamic(token:Token,x:&[f32;8],n:f32,out:&mut[f32;8]){use magetypes::simd::generic::f32x8 as V;let _=n;let v=V::from_array_t(token,*x);*out=({let r=(v.log2_lowp()*V::splat_t(token,n)).exp2_lowp();V::blend(v.simd_eq(V::splat_t(token,0.)),V::splat_t(token,0.),r)}).to_array();}
#[archmage::magetypes(v4x(cfg(avx512)),v4(cfg(avx512)),v3,neon,wasm128,scalar)]
#[inline(never)]
fn remap_shared_lowp_x8_dynamic(token:Token,x:&[f32;8],n:f32,out:&mut[f32;8]){use magetypes::simd::generic::f32x8 as V;let _=n;let v=V::from_array_t(token,*x);*out=(v.pow_lowp(n)).to_array();}
#[used]
static REMAPTABLE_X8_LOWP_DYNAMIC_SCALAR: [Probe<archmage::ScalarToken,8>;2]=[remap_original_lowp_x8_dynamic_scalar,remap_shared_lowp_x8_dynamic_scalar];
#[cfg(target_arch="x86_64")]
#[used]
static REMAPTABLE_X8_LOWP_DYNAMIC_V3: [Probe<archmage::X64V3Token,8>;2]=[remap_original_lowp_x8_dynamic_v3,remap_shared_lowp_x8_dynamic_v3];
#[cfg(all(target_arch="x86_64",feature="avx512"))]
#[used]
static REMAPTABLE_X8_LOWP_DYNAMIC_V4: [Probe<archmage::X64V4Token,8>;2]=[remap_original_lowp_x8_dynamic_v4,remap_shared_lowp_x8_dynamic_v4];
#[cfg(all(target_arch="x86_64",feature="avx512"))]
#[used]
static REMAPTABLE_X8_LOWP_DYNAMIC_V4X: [Probe<archmage::X64V4xToken,8>;2]=[remap_original_lowp_x8_dynamic_v4x,remap_shared_lowp_x8_dynamic_v4x];
#[cfg(target_arch="aarch64")]
#[used]
static REMAPTABLE_X8_LOWP_DYNAMIC_NEON: [Probe<archmage::NeonToken,8>;2]=[remap_original_lowp_x8_dynamic_neon,remap_shared_lowp_x8_dynamic_neon];
#[cfg(target_arch="wasm32")]
#[used]
static REMAPTABLE_X8_LOWP_DYNAMIC_WASM128: [Probe<archmage::Wasm128Token,8>;2]=[remap_original_lowp_x8_dynamic_wasm128,remap_shared_lowp_x8_dynamic_wasm128];
#[archmage::magetypes(v4x(cfg(avx512)),v4(cfg(avx512)),v3,neon,wasm128,scalar)]
#[inline(never)]
fn remap_original_lowp_x8_literal(token:Token,x:&[f32;8],n:f32,out:&mut[f32;8]){use magetypes::simd::generic::f32x8 as V;let _=n;let v=V::from_array_t(token,*x);*out=({let r=(v.log2_lowp()*V::splat_t(token,2.4f32)).exp2_lowp();V::blend(v.simd_eq(V::splat_t(token,0.)),V::splat_t(token,0.),r)}).to_array();}
#[archmage::magetypes(v4x(cfg(avx512)),v4(cfg(avx512)),v3,neon,wasm128,scalar)]
#[inline(never)]
fn remap_shared_lowp_x8_literal(token:Token,x:&[f32;8],n:f32,out:&mut[f32;8]){use magetypes::simd::generic::f32x8 as V;let _=n;let v=V::from_array_t(token,*x);*out=(v.pow_lowp(2.4f32)).to_array();}
#[used]
static REMAPTABLE_X8_LOWP_LITERAL_SCALAR: [Probe<archmage::ScalarToken,8>;2]=[remap_original_lowp_x8_literal_scalar,remap_shared_lowp_x8_literal_scalar];
#[cfg(target_arch="x86_64")]
#[used]
static REMAPTABLE_X8_LOWP_LITERAL_V3: [Probe<archmage::X64V3Token,8>;2]=[remap_original_lowp_x8_literal_v3,remap_shared_lowp_x8_literal_v3];
#[cfg(all(target_arch="x86_64",feature="avx512"))]
#[used]
static REMAPTABLE_X8_LOWP_LITERAL_V4: [Probe<archmage::X64V4Token,8>;2]=[remap_original_lowp_x8_literal_v4,remap_shared_lowp_x8_literal_v4];
#[cfg(all(target_arch="x86_64",feature="avx512"))]
#[used]
static REMAPTABLE_X8_LOWP_LITERAL_V4X: [Probe<archmage::X64V4xToken,8>;2]=[remap_original_lowp_x8_literal_v4x,remap_shared_lowp_x8_literal_v4x];
#[cfg(target_arch="aarch64")]
#[used]
static REMAPTABLE_X8_LOWP_LITERAL_NEON: [Probe<archmage::NeonToken,8>;2]=[remap_original_lowp_x8_literal_neon,remap_shared_lowp_x8_literal_neon];
#[cfg(target_arch="wasm32")]
#[used]
static REMAPTABLE_X8_LOWP_LITERAL_WASM128: [Probe<archmage::Wasm128Token,8>;2]=[remap_original_lowp_x8_literal_wasm128,remap_shared_lowp_x8_literal_wasm128];
#[archmage::magetypes(v4x(cfg(avx512)),v4(cfg(avx512)),v3,neon,wasm128,scalar)]
#[inline(never)]
fn remap_original_midp_x8_dynamic(token:Token,x:&[f32;8],n:f32,out:&mut[f32;8]){use magetypes::simd::generic::f32x8 as V;let _=n;let v=V::from_array_t(token,*x);*out=((v.log2_midp()*V::splat_t(token,n)).exp2_midp()).to_array();}
#[archmage::magetypes(v4x(cfg(avx512)),v4(cfg(avx512)),v3,neon,wasm128,scalar)]
#[inline(never)]
fn remap_shared_midp_x8_dynamic(token:Token,x:&[f32;8],n:f32,out:&mut[f32;8]){use magetypes::simd::generic::f32x8 as V;let _=n;let v=V::from_array_t(token,*x);*out=(v.pow_midp(n)).to_array();}
#[used]
static REMAPTABLE_X8_MIDP_DYNAMIC_SCALAR: [Probe<archmage::ScalarToken,8>;2]=[remap_original_midp_x8_dynamic_scalar,remap_shared_midp_x8_dynamic_scalar];
#[cfg(target_arch="x86_64")]
#[used]
static REMAPTABLE_X8_MIDP_DYNAMIC_V3: [Probe<archmage::X64V3Token,8>;2]=[remap_original_midp_x8_dynamic_v3,remap_shared_midp_x8_dynamic_v3];
#[cfg(all(target_arch="x86_64",feature="avx512"))]
#[used]
static REMAPTABLE_X8_MIDP_DYNAMIC_V4: [Probe<archmage::X64V4Token,8>;2]=[remap_original_midp_x8_dynamic_v4,remap_shared_midp_x8_dynamic_v4];
#[cfg(all(target_arch="x86_64",feature="avx512"))]
#[used]
static REMAPTABLE_X8_MIDP_DYNAMIC_V4X: [Probe<archmage::X64V4xToken,8>;2]=[remap_original_midp_x8_dynamic_v4x,remap_shared_midp_x8_dynamic_v4x];
#[cfg(target_arch="aarch64")]
#[used]
static REMAPTABLE_X8_MIDP_DYNAMIC_NEON: [Probe<archmage::NeonToken,8>;2]=[remap_original_midp_x8_dynamic_neon,remap_shared_midp_x8_dynamic_neon];
#[cfg(target_arch="wasm32")]
#[used]
static REMAPTABLE_X8_MIDP_DYNAMIC_WASM128: [Probe<archmage::Wasm128Token,8>;2]=[remap_original_midp_x8_dynamic_wasm128,remap_shared_midp_x8_dynamic_wasm128];
#[archmage::magetypes(v4x(cfg(avx512)),v4(cfg(avx512)),v3,neon,wasm128,scalar)]
#[inline(never)]
fn remap_original_midp_x8_literal(token:Token,x:&[f32;8],n:f32,out:&mut[f32;8]){use magetypes::simd::generic::f32x8 as V;let _=n;let v=V::from_array_t(token,*x);*out=((v.log2_midp()*V::splat_t(token,2.4f32)).exp2_midp()).to_array();}
#[archmage::magetypes(v4x(cfg(avx512)),v4(cfg(avx512)),v3,neon,wasm128,scalar)]
#[inline(never)]
fn remap_shared_midp_x8_literal(token:Token,x:&[f32;8],n:f32,out:&mut[f32;8]){use magetypes::simd::generic::f32x8 as V;let _=n;let v=V::from_array_t(token,*x);*out=(v.pow_midp(2.4f32)).to_array();}
#[used]
static REMAPTABLE_X8_MIDP_LITERAL_SCALAR: [Probe<archmage::ScalarToken,8>;2]=[remap_original_midp_x8_literal_scalar,remap_shared_midp_x8_literal_scalar];
#[cfg(target_arch="x86_64")]
#[used]
static REMAPTABLE_X8_MIDP_LITERAL_V3: [Probe<archmage::X64V3Token,8>;2]=[remap_original_midp_x8_literal_v3,remap_shared_midp_x8_literal_v3];
#[cfg(all(target_arch="x86_64",feature="avx512"))]
#[used]
static REMAPTABLE_X8_MIDP_LITERAL_V4: [Probe<archmage::X64V4Token,8>;2]=[remap_original_midp_x8_literal_v4,remap_shared_midp_x8_literal_v4];
#[cfg(all(target_arch="x86_64",feature="avx512"))]
#[used]
static REMAPTABLE_X8_MIDP_LITERAL_V4X: [Probe<archmage::X64V4xToken,8>;2]=[remap_original_midp_x8_literal_v4x,remap_shared_midp_x8_literal_v4x];
#[cfg(target_arch="aarch64")]
#[used]
static REMAPTABLE_X8_MIDP_LITERAL_NEON: [Probe<archmage::NeonToken,8>;2]=[remap_original_midp_x8_literal_neon,remap_shared_midp_x8_literal_neon];
#[cfg(target_arch="wasm32")]
#[used]
static REMAPTABLE_X8_MIDP_LITERAL_WASM128: [Probe<archmage::Wasm128Token,8>;2]=[remap_original_midp_x8_literal_wasm128,remap_shared_midp_x8_literal_wasm128];
#[archmage::magetypes(v4x(cfg(avx512)),v4(cfg(avx512)),v3,neon,wasm128,scalar)]
#[inline(never)]
fn remap_original_midp_unchecked_x8_dynamic(token:Token,x:&[f32;8],n:f32,out:&mut[f32;8]){use magetypes::simd::generic::f32x8 as V;let _=n;let v=V::from_array_t(token,*x);*out=((v.log2_midp_unchecked()*V::splat_t(token,n)).exp2_midp_unchecked()).to_array();}
#[archmage::magetypes(v4x(cfg(avx512)),v4(cfg(avx512)),v3,neon,wasm128,scalar)]
#[inline(never)]
fn remap_shared_midp_unchecked_x8_dynamic(token:Token,x:&[f32;8],n:f32,out:&mut[f32;8]){use magetypes::simd::generic::f32x8 as V;let _=n;let v=V::from_array_t(token,*x);*out=(v.pow_midp_unchecked(n)).to_array();}
#[used]
static REMAPTABLE_X8_MIDP_UNCHECKED_DYNAMIC_SCALAR: [Probe<archmage::ScalarToken,8>;2]=[remap_original_midp_unchecked_x8_dynamic_scalar,remap_shared_midp_unchecked_x8_dynamic_scalar];
#[cfg(target_arch="x86_64")]
#[used]
static REMAPTABLE_X8_MIDP_UNCHECKED_DYNAMIC_V3: [Probe<archmage::X64V3Token,8>;2]=[remap_original_midp_unchecked_x8_dynamic_v3,remap_shared_midp_unchecked_x8_dynamic_v3];
#[cfg(all(target_arch="x86_64",feature="avx512"))]
#[used]
static REMAPTABLE_X8_MIDP_UNCHECKED_DYNAMIC_V4: [Probe<archmage::X64V4Token,8>;2]=[remap_original_midp_unchecked_x8_dynamic_v4,remap_shared_midp_unchecked_x8_dynamic_v4];
#[cfg(all(target_arch="x86_64",feature="avx512"))]
#[used]
static REMAPTABLE_X8_MIDP_UNCHECKED_DYNAMIC_V4X: [Probe<archmage::X64V4xToken,8>;2]=[remap_original_midp_unchecked_x8_dynamic_v4x,remap_shared_midp_unchecked_x8_dynamic_v4x];
#[cfg(target_arch="aarch64")]
#[used]
static REMAPTABLE_X8_MIDP_UNCHECKED_DYNAMIC_NEON: [Probe<archmage::NeonToken,8>;2]=[remap_original_midp_unchecked_x8_dynamic_neon,remap_shared_midp_unchecked_x8_dynamic_neon];
#[cfg(target_arch="wasm32")]
#[used]
static REMAPTABLE_X8_MIDP_UNCHECKED_DYNAMIC_WASM128: [Probe<archmage::Wasm128Token,8>;2]=[remap_original_midp_unchecked_x8_dynamic_wasm128,remap_shared_midp_unchecked_x8_dynamic_wasm128];
#[archmage::magetypes(v4x(cfg(avx512)),v4(cfg(avx512)),v3,neon,wasm128,scalar)]
#[inline(never)]
fn remap_original_midp_unchecked_x8_literal(token:Token,x:&[f32;8],n:f32,out:&mut[f32;8]){use magetypes::simd::generic::f32x8 as V;let _=n;let v=V::from_array_t(token,*x);*out=((v.log2_midp_unchecked()*V::splat_t(token,2.4f32)).exp2_midp_unchecked()).to_array();}
#[archmage::magetypes(v4x(cfg(avx512)),v4(cfg(avx512)),v3,neon,wasm128,scalar)]
#[inline(never)]
fn remap_shared_midp_unchecked_x8_literal(token:Token,x:&[f32;8],n:f32,out:&mut[f32;8]){use magetypes::simd::generic::f32x8 as V;let _=n;let v=V::from_array_t(token,*x);*out=(v.pow_midp_unchecked(2.4f32)).to_array();}
#[used]
static REMAPTABLE_X8_MIDP_UNCHECKED_LITERAL_SCALAR: [Probe<archmage::ScalarToken,8>;2]=[remap_original_midp_unchecked_x8_literal_scalar,remap_shared_midp_unchecked_x8_literal_scalar];
#[cfg(target_arch="x86_64")]
#[used]
static REMAPTABLE_X8_MIDP_UNCHECKED_LITERAL_V3: [Probe<archmage::X64V3Token,8>;2]=[remap_original_midp_unchecked_x8_literal_v3,remap_shared_midp_unchecked_x8_literal_v3];
#[cfg(all(target_arch="x86_64",feature="avx512"))]
#[used]
static REMAPTABLE_X8_MIDP_UNCHECKED_LITERAL_V4: [Probe<archmage::X64V4Token,8>;2]=[remap_original_midp_unchecked_x8_literal_v4,remap_shared_midp_unchecked_x8_literal_v4];
#[cfg(all(target_arch="x86_64",feature="avx512"))]
#[used]
static REMAPTABLE_X8_MIDP_UNCHECKED_LITERAL_V4X: [Probe<archmage::X64V4xToken,8>;2]=[remap_original_midp_unchecked_x8_literal_v4x,remap_shared_midp_unchecked_x8_literal_v4x];
#[cfg(target_arch="aarch64")]
#[used]
static REMAPTABLE_X8_MIDP_UNCHECKED_LITERAL_NEON: [Probe<archmage::NeonToken,8>;2]=[remap_original_midp_unchecked_x8_literal_neon,remap_shared_midp_unchecked_x8_literal_neon];
#[cfg(target_arch="wasm32")]
#[used]
static REMAPTABLE_X8_MIDP_UNCHECKED_LITERAL_WASM128: [Probe<archmage::Wasm128Token,8>;2]=[remap_original_midp_unchecked_x8_literal_wasm128,remap_shared_midp_unchecked_x8_literal_wasm128];
#[archmage::magetypes(v4x(cfg(avx512)),v4(cfg(avx512)),v3,neon,wasm128,scalar)]
#[inline(never)]
fn remap_original_lowp_x16_dynamic(token:Token,x:&[f32;16],n:f32,out:&mut[f32;16]){use magetypes::simd::generic::f32x16 as V;let _=n;let v=V::from_array_t(token,*x);*out=({let r=(v.log2_lowp()*V::splat_t(token,n)).exp2_lowp();V::blend(v.simd_eq(V::splat_t(token,0.)),V::splat_t(token,0.),r)}).to_array();}
#[archmage::magetypes(v4x(cfg(avx512)),v4(cfg(avx512)),v3,neon,wasm128,scalar)]
#[inline(never)]
fn remap_shared_lowp_x16_dynamic(token:Token,x:&[f32;16],n:f32,out:&mut[f32;16]){use magetypes::simd::generic::f32x16 as V;let _=n;let v=V::from_array_t(token,*x);*out=(v.pow_lowp(n)).to_array();}
#[used]
static REMAPTABLE_X16_LOWP_DYNAMIC_SCALAR: [Probe<archmage::ScalarToken,16>;2]=[remap_original_lowp_x16_dynamic_scalar,remap_shared_lowp_x16_dynamic_scalar];
#[cfg(target_arch="x86_64")]
#[used]
static REMAPTABLE_X16_LOWP_DYNAMIC_V3: [Probe<archmage::X64V3Token,16>;2]=[remap_original_lowp_x16_dynamic_v3,remap_shared_lowp_x16_dynamic_v3];
#[cfg(all(target_arch="x86_64",feature="avx512"))]
#[used]
static REMAPTABLE_X16_LOWP_DYNAMIC_V4: [Probe<archmage::X64V4Token,16>;2]=[remap_original_lowp_x16_dynamic_v4,remap_shared_lowp_x16_dynamic_v4];
#[cfg(all(target_arch="x86_64",feature="avx512"))]
#[used]
static REMAPTABLE_X16_LOWP_DYNAMIC_V4X: [Probe<archmage::X64V4xToken,16>;2]=[remap_original_lowp_x16_dynamic_v4x,remap_shared_lowp_x16_dynamic_v4x];
#[cfg(target_arch="aarch64")]
#[used]
static REMAPTABLE_X16_LOWP_DYNAMIC_NEON: [Probe<archmage::NeonToken,16>;2]=[remap_original_lowp_x16_dynamic_neon,remap_shared_lowp_x16_dynamic_neon];
#[cfg(target_arch="wasm32")]
#[used]
static REMAPTABLE_X16_LOWP_DYNAMIC_WASM128: [Probe<archmage::Wasm128Token,16>;2]=[remap_original_lowp_x16_dynamic_wasm128,remap_shared_lowp_x16_dynamic_wasm128];
#[archmage::magetypes(v4x(cfg(avx512)),v4(cfg(avx512)),v3,neon,wasm128,scalar)]
#[inline(never)]
fn remap_original_lowp_x16_literal(token:Token,x:&[f32;16],n:f32,out:&mut[f32;16]){use magetypes::simd::generic::f32x16 as V;let _=n;let v=V::from_array_t(token,*x);*out=({let r=(v.log2_lowp()*V::splat_t(token,2.4f32)).exp2_lowp();V::blend(v.simd_eq(V::splat_t(token,0.)),V::splat_t(token,0.),r)}).to_array();}
#[archmage::magetypes(v4x(cfg(avx512)),v4(cfg(avx512)),v3,neon,wasm128,scalar)]
#[inline(never)]
fn remap_shared_lowp_x16_literal(token:Token,x:&[f32;16],n:f32,out:&mut[f32;16]){use magetypes::simd::generic::f32x16 as V;let _=n;let v=V::from_array_t(token,*x);*out=(v.pow_lowp(2.4f32)).to_array();}
#[used]
static REMAPTABLE_X16_LOWP_LITERAL_SCALAR: [Probe<archmage::ScalarToken,16>;2]=[remap_original_lowp_x16_literal_scalar,remap_shared_lowp_x16_literal_scalar];
#[cfg(target_arch="x86_64")]
#[used]
static REMAPTABLE_X16_LOWP_LITERAL_V3: [Probe<archmage::X64V3Token,16>;2]=[remap_original_lowp_x16_literal_v3,remap_shared_lowp_x16_literal_v3];
#[cfg(all(target_arch="x86_64",feature="avx512"))]
#[used]
static REMAPTABLE_X16_LOWP_LITERAL_V4: [Probe<archmage::X64V4Token,16>;2]=[remap_original_lowp_x16_literal_v4,remap_shared_lowp_x16_literal_v4];
#[cfg(all(target_arch="x86_64",feature="avx512"))]
#[used]
static REMAPTABLE_X16_LOWP_LITERAL_V4X: [Probe<archmage::X64V4xToken,16>;2]=[remap_original_lowp_x16_literal_v4x,remap_shared_lowp_x16_literal_v4x];
#[cfg(target_arch="aarch64")]
#[used]
static REMAPTABLE_X16_LOWP_LITERAL_NEON: [Probe<archmage::NeonToken,16>;2]=[remap_original_lowp_x16_literal_neon,remap_shared_lowp_x16_literal_neon];
#[cfg(target_arch="wasm32")]
#[used]
static REMAPTABLE_X16_LOWP_LITERAL_WASM128: [Probe<archmage::Wasm128Token,16>;2]=[remap_original_lowp_x16_literal_wasm128,remap_shared_lowp_x16_literal_wasm128];
#[archmage::magetypes(v4x(cfg(avx512)),v4(cfg(avx512)),v3,neon,wasm128,scalar)]
#[inline(never)]
fn remap_original_midp_x16_dynamic(token:Token,x:&[f32;16],n:f32,out:&mut[f32;16]){use magetypes::simd::generic::f32x16 as V;let _=n;let v=V::from_array_t(token,*x);*out=((v.log2_midp()*V::splat_t(token,n)).exp2_midp()).to_array();}
#[archmage::magetypes(v4x(cfg(avx512)),v4(cfg(avx512)),v3,neon,wasm128,scalar)]
#[inline(never)]
fn remap_shared_midp_x16_dynamic(token:Token,x:&[f32;16],n:f32,out:&mut[f32;16]){use magetypes::simd::generic::f32x16 as V;let _=n;let v=V::from_array_t(token,*x);*out=(v.pow_midp(n)).to_array();}
#[used]
static REMAPTABLE_X16_MIDP_DYNAMIC_SCALAR: [Probe<archmage::ScalarToken,16>;2]=[remap_original_midp_x16_dynamic_scalar,remap_shared_midp_x16_dynamic_scalar];
#[cfg(target_arch="x86_64")]
#[used]
static REMAPTABLE_X16_MIDP_DYNAMIC_V3: [Probe<archmage::X64V3Token,16>;2]=[remap_original_midp_x16_dynamic_v3,remap_shared_midp_x16_dynamic_v3];
#[cfg(all(target_arch="x86_64",feature="avx512"))]
#[used]
static REMAPTABLE_X16_MIDP_DYNAMIC_V4: [Probe<archmage::X64V4Token,16>;2]=[remap_original_midp_x16_dynamic_v4,remap_shared_midp_x16_dynamic_v4];
#[cfg(all(target_arch="x86_64",feature="avx512"))]
#[used]
static REMAPTABLE_X16_MIDP_DYNAMIC_V4X: [Probe<archmage::X64V4xToken,16>;2]=[remap_original_midp_x16_dynamic_v4x,remap_shared_midp_x16_dynamic_v4x];
#[cfg(target_arch="aarch64")]
#[used]
static REMAPTABLE_X16_MIDP_DYNAMIC_NEON: [Probe<archmage::NeonToken,16>;2]=[remap_original_midp_x16_dynamic_neon,remap_shared_midp_x16_dynamic_neon];
#[cfg(target_arch="wasm32")]
#[used]
static REMAPTABLE_X16_MIDP_DYNAMIC_WASM128: [Probe<archmage::Wasm128Token,16>;2]=[remap_original_midp_x16_dynamic_wasm128,remap_shared_midp_x16_dynamic_wasm128];
#[archmage::magetypes(v4x(cfg(avx512)),v4(cfg(avx512)),v3,neon,wasm128,scalar)]
#[inline(never)]
fn remap_original_midp_x16_literal(token:Token,x:&[f32;16],n:f32,out:&mut[f32;16]){use magetypes::simd::generic::f32x16 as V;let _=n;let v=V::from_array_t(token,*x);*out=((v.log2_midp()*V::splat_t(token,2.4f32)).exp2_midp()).to_array();}
#[archmage::magetypes(v4x(cfg(avx512)),v4(cfg(avx512)),v3,neon,wasm128,scalar)]
#[inline(never)]
fn remap_shared_midp_x16_literal(token:Token,x:&[f32;16],n:f32,out:&mut[f32;16]){use magetypes::simd::generic::f32x16 as V;let _=n;let v=V::from_array_t(token,*x);*out=(v.pow_midp(2.4f32)).to_array();}
#[used]
static REMAPTABLE_X16_MIDP_LITERAL_SCALAR: [Probe<archmage::ScalarToken,16>;2]=[remap_original_midp_x16_literal_scalar,remap_shared_midp_x16_literal_scalar];
#[cfg(target_arch="x86_64")]
#[used]
static REMAPTABLE_X16_MIDP_LITERAL_V3: [Probe<archmage::X64V3Token,16>;2]=[remap_original_midp_x16_literal_v3,remap_shared_midp_x16_literal_v3];
#[cfg(all(target_arch="x86_64",feature="avx512"))]
#[used]
static REMAPTABLE_X16_MIDP_LITERAL_V4: [Probe<archmage::X64V4Token,16>;2]=[remap_original_midp_x16_literal_v4,remap_shared_midp_x16_literal_v4];
#[cfg(all(target_arch="x86_64",feature="avx512"))]
#[used]
static REMAPTABLE_X16_MIDP_LITERAL_V4X: [Probe<archmage::X64V4xToken,16>;2]=[remap_original_midp_x16_literal_v4x,remap_shared_midp_x16_literal_v4x];
#[cfg(target_arch="aarch64")]
#[used]
static REMAPTABLE_X16_MIDP_LITERAL_NEON: [Probe<archmage::NeonToken,16>;2]=[remap_original_midp_x16_literal_neon,remap_shared_midp_x16_literal_neon];
#[cfg(target_arch="wasm32")]
#[used]
static REMAPTABLE_X16_MIDP_LITERAL_WASM128: [Probe<archmage::Wasm128Token,16>;2]=[remap_original_midp_x16_literal_wasm128,remap_shared_midp_x16_literal_wasm128];
#[archmage::magetypes(v4x(cfg(avx512)),v4(cfg(avx512)),v3,neon,wasm128,scalar)]
#[inline(never)]
fn remap_original_midp_unchecked_x16_dynamic(token:Token,x:&[f32;16],n:f32,out:&mut[f32;16]){use magetypes::simd::generic::f32x16 as V;let _=n;let v=V::from_array_t(token,*x);*out=((v.log2_midp_unchecked()*V::splat_t(token,n)).exp2_midp_unchecked()).to_array();}
#[archmage::magetypes(v4x(cfg(avx512)),v4(cfg(avx512)),v3,neon,wasm128,scalar)]
#[inline(never)]
fn remap_shared_midp_unchecked_x16_dynamic(token:Token,x:&[f32;16],n:f32,out:&mut[f32;16]){use magetypes::simd::generic::f32x16 as V;let _=n;let v=V::from_array_t(token,*x);*out=(v.pow_midp_unchecked(n)).to_array();}
#[used]
static REMAPTABLE_X16_MIDP_UNCHECKED_DYNAMIC_SCALAR: [Probe<archmage::ScalarToken,16>;2]=[remap_original_midp_unchecked_x16_dynamic_scalar,remap_shared_midp_unchecked_x16_dynamic_scalar];
#[cfg(target_arch="x86_64")]
#[used]
static REMAPTABLE_X16_MIDP_UNCHECKED_DYNAMIC_V3: [Probe<archmage::X64V3Token,16>;2]=[remap_original_midp_unchecked_x16_dynamic_v3,remap_shared_midp_unchecked_x16_dynamic_v3];
#[cfg(all(target_arch="x86_64",feature="avx512"))]
#[used]
static REMAPTABLE_X16_MIDP_UNCHECKED_DYNAMIC_V4: [Probe<archmage::X64V4Token,16>;2]=[remap_original_midp_unchecked_x16_dynamic_v4,remap_shared_midp_unchecked_x16_dynamic_v4];
#[cfg(all(target_arch="x86_64",feature="avx512"))]
#[used]
static REMAPTABLE_X16_MIDP_UNCHECKED_DYNAMIC_V4X: [Probe<archmage::X64V4xToken,16>;2]=[remap_original_midp_unchecked_x16_dynamic_v4x,remap_shared_midp_unchecked_x16_dynamic_v4x];
#[cfg(target_arch="aarch64")]
#[used]
static REMAPTABLE_X16_MIDP_UNCHECKED_DYNAMIC_NEON: [Probe<archmage::NeonToken,16>;2]=[remap_original_midp_unchecked_x16_dynamic_neon,remap_shared_midp_unchecked_x16_dynamic_neon];
#[cfg(target_arch="wasm32")]
#[used]
static REMAPTABLE_X16_MIDP_UNCHECKED_DYNAMIC_WASM128: [Probe<archmage::Wasm128Token,16>;2]=[remap_original_midp_unchecked_x16_dynamic_wasm128,remap_shared_midp_unchecked_x16_dynamic_wasm128];
#[archmage::magetypes(v4x(cfg(avx512)),v4(cfg(avx512)),v3,neon,wasm128,scalar)]
#[inline(never)]
fn remap_original_midp_unchecked_x16_literal(token:Token,x:&[f32;16],n:f32,out:&mut[f32;16]){use magetypes::simd::generic::f32x16 as V;let _=n;let v=V::from_array_t(token,*x);*out=((v.log2_midp_unchecked()*V::splat_t(token,2.4f32)).exp2_midp_unchecked()).to_array();}
#[archmage::magetypes(v4x(cfg(avx512)),v4(cfg(avx512)),v3,neon,wasm128,scalar)]
#[inline(never)]
fn remap_shared_midp_unchecked_x16_literal(token:Token,x:&[f32;16],n:f32,out:&mut[f32;16]){use magetypes::simd::generic::f32x16 as V;let _=n;let v=V::from_array_t(token,*x);*out=(v.pow_midp_unchecked(2.4f32)).to_array();}
#[used]
static REMAPTABLE_X16_MIDP_UNCHECKED_LITERAL_SCALAR: [Probe<archmage::ScalarToken,16>;2]=[remap_original_midp_unchecked_x16_literal_scalar,remap_shared_midp_unchecked_x16_literal_scalar];
#[cfg(target_arch="x86_64")]
#[used]
static REMAPTABLE_X16_MIDP_UNCHECKED_LITERAL_V3: [Probe<archmage::X64V3Token,16>;2]=[remap_original_midp_unchecked_x16_literal_v3,remap_shared_midp_unchecked_x16_literal_v3];
#[cfg(all(target_arch="x86_64",feature="avx512"))]
#[used]
static REMAPTABLE_X16_MIDP_UNCHECKED_LITERAL_V4: [Probe<archmage::X64V4Token,16>;2]=[remap_original_midp_unchecked_x16_literal_v4,remap_shared_midp_unchecked_x16_literal_v4];
#[cfg(all(target_arch="x86_64",feature="avx512"))]
#[used]
static REMAPTABLE_X16_MIDP_UNCHECKED_LITERAL_V4X: [Probe<archmage::X64V4xToken,16>;2]=[remap_original_midp_unchecked_x16_literal_v4x,remap_shared_midp_unchecked_x16_literal_v4x];
#[cfg(target_arch="aarch64")]
#[used]
static REMAPTABLE_X16_MIDP_UNCHECKED_LITERAL_NEON: [Probe<archmage::NeonToken,16>;2]=[remap_original_midp_unchecked_x16_literal_neon,remap_shared_midp_unchecked_x16_literal_neon];
#[cfg(target_arch="wasm32")]
#[used]
static REMAPTABLE_X16_MIDP_UNCHECKED_LITERAL_WASM128: [Probe<archmage::Wasm128Token,16>;2]=[remap_original_midp_unchecked_x16_literal_wasm128,remap_shared_midp_unchecked_x16_literal_wasm128];
fn main(){}

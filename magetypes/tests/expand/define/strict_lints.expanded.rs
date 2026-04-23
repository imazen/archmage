#![deny(non_camel_case_types, non_snake_case, warnings)]
use archmage::magetypes;
#[doc(hidden)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn __arcane_kernel_v3(
    token: archmage::X64V3Token,
    data: &[f32; 8],
    bytes: &[u8; 16],
) -> f32 {
    #[allow(non_camel_case_types, dead_code)]
    type f32x8 = ::magetypes::simd::generic::f32x8<archmage::X64V3Token>;
    #[allow(non_camel_case_types, dead_code)]
    type u8x16 = ::magetypes::simd::generic::u8x16<archmage::X64V3Token>;
    let _ = u8x16::load(token, bytes);
    f32x8::load(token, data).reduce_add()
}
#[inline(always)]
fn kernel_v3(token: archmage::X64V3Token, data: &[f32; 8], bytes: &[u8; 16]) -> f32 {
    const _ARCHMAGE_TOKEN_MISMATCH: () = [
        (),
    ][!(<archmage::X64V3Token>::__ARCHMAGE_TIER_TAG == 4085983307u32) as usize];
    unsafe { __arcane_kernel_v3(token, data, bytes) }
}
fn kernel_scalar(
    token: archmage::ScalarToken,
    data: &[f32; 8],
    bytes: &[u8; 16],
) -> f32 {
    #[allow(non_camel_case_types, dead_code)]
    type f32x8 = ::magetypes::simd::generic::f32x8<archmage::ScalarToken>;
    #[allow(non_camel_case_types, dead_code)]
    type u8x16 = ::magetypes::simd::generic::u8x16<archmage::ScalarToken>;
    let _ = u8x16::load(token, bytes);
    f32x8::load(token, data).reduce_add()
}
fn main() {}

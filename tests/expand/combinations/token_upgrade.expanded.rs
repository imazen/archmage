use archmage::{arcane, SimdToken, X64V3Token, X64V4Token};
#[doc(hidden)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
)]
#[inline]
fn __arcane_v4_fast(_t: X64V4Token, x: f32) -> f32 {
    x * 4.0
}
#[inline(always)]
fn v4_fast(_t: X64V4Token, x: f32) -> f32 {
    {
        fn __archmage_verify(_: &::archmage::X64V4Token) {}
        __archmage_verify(&_t);
    }
    unsafe { __arcane_v4_fast(_t, x) }
}
#[doc(hidden)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn __arcane_v3_with_upgrade(_t: X64V3Token, x: f32) -> f32 {
    if let Some(v4) = X64V4Token::summon() { v4_fast(v4, x) } else { x * 2.0 }
}
#[inline(always)]
fn v3_with_upgrade(_t: X64V3Token, x: f32) -> f32 {
    {
        fn __archmage_verify(_: &::archmage::X64V3Token) {}
        __archmage_verify(&_t);
    }
    unsafe { __arcane_v3_with_upgrade(_t, x) }
}
fn main() {}

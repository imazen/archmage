use archmage::{arcane, X64V3Token};
#[doc(hidden)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn __arcane_process(token: X64V3Token, ptr: *const f32) -> f32 {
    unsafe { *ptr }
}
#[inline(always)]
unsafe fn process(token: X64V3Token, ptr: *const f32) -> f32 {
    {
        fn __archmage_verify(_: &::archmage::X64V3Token) {}
        __archmage_verify(&token);
    }
    unsafe { __arcane_process(token, ptr) }
}
fn main() {}

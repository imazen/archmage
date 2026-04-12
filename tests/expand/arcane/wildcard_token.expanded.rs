use archmage::{arcane, X64V3Token};
#[doc(hidden)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn __arcane_process(__archmage_arg_0: X64V3Token, a: f32) -> f32 {
    let _: X64V3Token = __archmage_arg_0;
    a * 2.0
}
#[inline(always)]
fn process(__archmage_arg_0: X64V3Token, a: f32) -> f32 {
    {
        fn __archmage_verify(_: &::archmage::X64V3Token) {}
        __archmage_verify(&__archmage_arg_0);
    }
    unsafe { __arcane_process(__archmage_arg_0, a) }
}
fn main() {}

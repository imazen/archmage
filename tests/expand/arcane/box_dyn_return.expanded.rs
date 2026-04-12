use archmage::{arcane, X64V3Token};
#[doc(hidden)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn __arcane_make_adder(token: X64V3Token, offset: f32) -> Box<dyn Fn(f32) -> f32> {
    let _ = token;
    Box::new(move |x| x + offset)
}
#[inline(always)]
fn make_adder(token: X64V3Token, offset: f32) -> Box<dyn Fn(f32) -> f32> {
    {
        fn __archmage_verify(_: &::archmage::X64V3Token) {}
        __archmage_verify(&token);
    }
    unsafe { __arcane_make_adder(token, offset) }
}
fn main() {}

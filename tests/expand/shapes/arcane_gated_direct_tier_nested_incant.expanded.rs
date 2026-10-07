use archmage::prelude::*;
fn inner_scalar(_: ScalarToken, x: u32) -> u32 {
    x
}
#[doc(hidden)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn __arcane_outer(t: X64V3Token, x: u32) -> u32 {
    '__incant_rewrite: {
        use archmage::SimdToken;
        inner_scalar(archmage::ScalarToken, x)
    }
}
#[inline(always)]
fn outer(t: X64V3Token, x: u32) -> u32 {
    let _: () = <X64V3Token>::__ARCHMAGE_ASSERT_TIER_F38B284B;
    unsafe { __arcane_outer(t, x) }
}
fn main() {}

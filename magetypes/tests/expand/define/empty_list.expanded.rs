use archmage::magetypes;
#[doc(hidden)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn __arcane_kernel_v3(_token: archmage::X64V3Token, x: f32) -> f32 {
    x * 2.0
}
#[inline(always)]
fn kernel_v3(_token: archmage::X64V3Token, x: f32) -> f32 {
    let _: () = <archmage::X64V3Token>::__ARCHMAGE_ASSERT_TIER_F38B284B;
    unsafe { __arcane_kernel_v3(_token, x) }
}
fn kernel_scalar(_token: archmage::ScalarToken, x: f32) -> f32 {
    x * 2.0
}
fn main() {}

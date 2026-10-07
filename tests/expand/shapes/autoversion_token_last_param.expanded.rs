use archmage::{ScalarToken, autoversion};
fn sum(x: u32, __archmage_arg_0: ScalarToken) -> u32 {
    #[allow(unused_imports)]
    use archmage::SimdToken;
    {
        if let Some(__t) = archmage::X64V3Token::summon() {
            return sum_v3(x, __t);
        }
    }
    sum_scalar(x, archmage::ScalarToken)
}
#[doc(hidden)]
#[allow(dead_code)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn __arcane_sum_v3(x: u32, __archmage_arg_0: archmage::X64V3Token) -> u32 {
    x
}
#[allow(dead_code)]
#[inline(always)]
fn sum_v3(x: u32, __archmage_arg_0: archmage::X64V3Token) -> u32 {
    let _: () = <archmage::X64V3Token>::__ARCHMAGE_ASSERT_TIER_F38B284B;
    unsafe { __arcane_sum_v3(x, __archmage_arg_0) }
}
#[allow(dead_code)]
fn sum_scalar(x: u32, _: archmage::ScalarToken) -> u32 {
    x
}
fn main() {
    let _ = sum(1, ScalarToken);
}

use archmage::autoversion;
fn sum(data: &[f32; 4]) -> f32 {
    #[allow(unused_imports)]
    use archmage::SimdToken;
    {
        if let Some(__t) = archmage::X64V3Token::summon() {
            return sum_v3(__t, data);
        }
    }
    sum_scalar(archmage::ScalarToken, data)
}
#[doc(hidden)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn __arcane_sum_v3(_token: archmage::X64V3Token, data: &[f32; 4]) -> f32 {
    data.iter().sum()
}
#[allow(dead_code)]
#[inline(always)]
#[allow(clippy::inline_always)]
fn sum_v3(_token: archmage::X64V3Token, data: &[f32; 4]) -> f32 {
    let _: () = <archmage::X64V3Token>::__ARCHMAGE_ASSERT_TIER_F38B284B;
    #[allow(clippy::used_underscore_binding)]
    { unsafe { __arcane_sum_v3(_token, data) } }
}
#[allow(dead_code)]
fn sum_scalar(_token: archmage::ScalarToken, data: &[f32; 4]) -> f32 {
    data.iter().sum()
}
fn main() {}

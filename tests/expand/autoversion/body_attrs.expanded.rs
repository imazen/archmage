use archmage::autoversion;
#[track_caller]
#[allow(unused_variables)]
#[must_use]
fn process(unused: f32) -> f32 {
    #[allow(unused_imports)]
    use archmage::SimdToken;
    {
        if let Some(__t) = archmage::X64V3Token::summon() {
            return process_v3(__t, unused);
        }
    }
    process_scalar(archmage::ScalarToken, unused)
}
#[doc(hidden)]
#[allow(dead_code)]
#[track_caller]
#[expect(unused_variables)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn __arcane_process_v3(_token: archmage::X64V3Token, unused: f32) -> f32 {
    1.0
}
#[allow(dead_code)]
#[track_caller]
#[allow(unused_variables)]
#[inline(always)]
fn process_v3(_token: archmage::X64V3Token, unused: f32) -> f32 {
    let _: () = <archmage::X64V3Token>::__ARCHMAGE_ASSERT_TIER_F38B284B;
    unsafe { __arcane_process_v3(_token, unused) }
}
#[allow(dead_code)]
#[track_caller]
#[expect(unused_variables)]
fn process_scalar(_token: archmage::ScalarToken, unused: f32) -> f32 {
    1.0
}
fn main() {}

use archmage::autoversion;
fn probe(__archmage_arg_0: f32, x: f32) -> f32 {
    #[allow(unused_imports)]
    use archmage::SimdToken;
    {
        if let Some(__t) = archmage::X64V3Token::summon() {
            return probe_v3(__t, __archmage_arg_0, x);
        }
    }
    probe_scalar(archmage::ScalarToken, __archmage_arg_0, x)
}
#[doc(hidden)]
#[allow(dead_code)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn __arcane_probe_v3(_token: archmage::X64V3Token, _: f32, x: f32) -> f32 {
    x
}
#[allow(dead_code)]
#[inline(always)]
fn probe_v3(_token: archmage::X64V3Token, __archmage_arg_0: f32, x: f32) -> f32 {
    let _: () = <archmage::X64V3Token>::__ARCHMAGE_ASSERT_TIER_F38B284B;
    unsafe { __arcane_probe_v3(_token, __archmage_arg_0, x) }
}
#[allow(dead_code)]
fn probe_scalar(_token: archmage::ScalarToken, _: f32, x: f32) -> f32 {
    x
}
fn main() {
    let _ = probe(1.0, 2.0);
}

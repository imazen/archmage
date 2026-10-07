use archmage::autoversion;
fn probe(n: u32) -> impl Iterator<Item = u32> {
    #[allow(unused_imports)]
    use archmage::SimdToken;
    {
        if let Some(__t) = archmage::X64V3Token::summon() {
            return probe_v3(__t, n);
        }
    }
    probe_scalar(archmage::ScalarToken, n)
}
#[doc(hidden)]
#[allow(dead_code)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn __arcane_probe_v3(_token: archmage::X64V3Token, n: u32) -> impl Iterator<Item = u32> {
    0..n
}
#[allow(dead_code)]
#[inline(always)]
fn probe_v3(_token: archmage::X64V3Token, n: u32) -> impl Iterator<Item = u32> {
    let _: () = <archmage::X64V3Token>::__ARCHMAGE_ASSERT_TIER_F38B284B;
    unsafe { __arcane_probe_v3(_token, n) }
}
#[allow(dead_code)]
fn probe_scalar(_token: archmage::ScalarToken, n: u32) -> impl Iterator<Item = u32> {
    0..n
}
fn main() {}

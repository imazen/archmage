use archmage::autoversion;
fn probe<'a, U: Copy, const N: usize>(xs: &'a [U; N]) -> &'a [U; N] {
    #[allow(unused_imports)]
    use archmage::SimdToken;
    {
        if let Some(__t) = archmage::X64V3Token::summon() {
            return probe_v3::<U, N>(__t, xs);
        }
    }
    probe_scalar::<U, N>(archmage::ScalarToken, xs)
}
#[doc(hidden)]
#[allow(dead_code)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn __arcane_probe_v3<'a, U: Copy, const N: usize>(
    _token: archmage::X64V3Token,
    xs: &'a [U; N],
) -> &'a [U; N] {
    xs
}
#[allow(dead_code)]
#[inline(always)]
fn probe_v3<'a, U: Copy, const N: usize>(
    _token: archmage::X64V3Token,
    xs: &'a [U; N],
) -> &'a [U; N] {
    let _: () = <archmage::X64V3Token>::__ARCHMAGE_ASSERT_TIER_F38B284B;
    unsafe { __arcane_probe_v3::<U, N>(_token, xs) }
}
#[allow(dead_code)]
fn probe_scalar<'a, U: Copy, const N: usize>(
    _token: archmage::ScalarToken,
    xs: &'a [U; N],
) -> &'a [U; N] {
    xs
}
fn main() {}

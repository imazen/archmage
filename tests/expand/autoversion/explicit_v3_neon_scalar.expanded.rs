use archmage::autoversion;
fn sum(data: &[f32; 4]) -> f32 {
    use archmage::SimdToken;
    {
        if let Some(__t) = archmage::X64V3Token::summon() {
            return sum_v3(__t, data);
        }
    }
    sum_scalar(archmage::ScalarToken, data)
}
#[doc(hidden)]
#[allow(dead_code)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn __arcane_sum_v3(_token: archmage::X64V3Token, data: &[f32; 4]) -> f32 {
    data.iter().sum()
}
#[allow(dead_code)]
#[inline(always)]
fn sum_v3(_token: archmage::X64V3Token, data: &[f32; 4]) -> f32 {
    {
        fn __archmage_verify(_: &::archmage::X64V3Token) {}
        __archmage_verify(&_token);
    }
    unsafe { __arcane_sum_v3(_token, data) }
}
#[allow(dead_code)]
fn sum_scalar(_token: archmage::ScalarToken, data: &[f32; 4]) -> f32 {
    data.iter().sum()
}
fn main() {}

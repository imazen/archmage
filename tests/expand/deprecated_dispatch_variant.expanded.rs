use archmage::{arcane, dispatch_variant, X64V3Token, ScalarToken};
#[doc(hidden)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn __arcane_compute_v3(_token: X64V3Token, x: f32) -> f32 {
    x * x
}
#[inline(always)]
fn compute_v3(_token: X64V3Token, x: f32) -> f32 {
    unsafe { __arcane_compute_v3(_token, x) }
}
fn compute_scalar(_token: ScalarToken, x: f32) -> f32 {
    x * x
}
fn dispatch(x: f32) -> f32 {
    '__incant: {
        use archmage::SimdToken;
        {
            if let Some(__t) = archmage::X64V3Token::summon() {
                break '__incant compute_v3(__t, x);
            }
        }
        compute_scalar(archmage::ScalarToken, x)
    }
}
fn main() {}

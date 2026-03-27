use archmage::{arcane, incant, X64V3Token, X64V4Token, NeonToken, ScalarToken};
#[doc(hidden)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
)]
#[inline]
fn __arcane_work_v4(_token: X64V4Token, x: f32) -> f32 {
    x * 2.0
}
#[inline(always)]
fn work_v4(_token: X64V4Token, x: f32) -> f32 {
    unsafe { __arcane_work_v4(_token, x) }
}
#[doc(hidden)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn __arcane_work_v3(_token: X64V3Token, x: f32) -> f32 {
    x * 2.0
}
#[inline(always)]
fn work_v3(_token: X64V3Token, x: f32) -> f32 {
    unsafe { __arcane_work_v3(_token, x) }
}
fn work_scalar(_token: ScalarToken, x: f32) -> f32 {
    x * 2.0
}
fn dispatch(x: f32) -> f32 {
    '__incant: {
        use archmage::SimdToken;
        {
            if let Some(__t) = archmage::X64V3Token::summon() {
                break '__incant work_v3(__t, x);
            }
        }
        work_scalar(archmage::ScalarToken, x)
    }
}
fn main() {}

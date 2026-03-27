use archmage::autoversion;
fn sum_squares(data: &[f32; 4]) -> f32 {
    use archmage::SimdToken;
    {
        if let Some(__t) = archmage::X64V4Token::summon() {
            return sum_squares_v4(__t, data);
        }
        if let Some(__t) = archmage::X64V3Token::summon() {
            return sum_squares_v3(__t, data);
        }
    }
    sum_squares_scalar(archmage::ScalarToken, data)
}
#[doc(hidden)]
#[allow(dead_code)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
)]
#[inline]
fn __arcane_sum_squares_v4(_token: archmage::X64V4Token, data: &[f32; 4]) -> f32 {
    let mut sum = 0.0f32;
    for &x in data {
        sum += x * x;
    }
    sum
}
#[allow(dead_code)]
#[inline(always)]
fn sum_squares_v4(_token: archmage::X64V4Token, data: &[f32; 4]) -> f32 {
    unsafe { __arcane_sum_squares_v4(_token, data) }
}
#[doc(hidden)]
#[allow(dead_code)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn __arcane_sum_squares_v3(_token: archmage::X64V3Token, data: &[f32; 4]) -> f32 {
    let mut sum = 0.0f32;
    for &x in data {
        sum += x * x;
    }
    sum
}
#[allow(dead_code)]
#[inline(always)]
fn sum_squares_v3(_token: archmage::X64V3Token, data: &[f32; 4]) -> f32 {
    unsafe { __arcane_sum_squares_v3(_token, data) }
}
#[allow(dead_code)]
fn sum_squares_scalar(_token: archmage::ScalarToken, data: &[f32; 4]) -> f32 {
    let mut sum = 0.0f32;
    for &x in data {
        sum += x * x;
    }
    sum
}
fn main() {}

use archmage::autoversion;
unsafe fn unsafe_sum(ptr: *const f32, len: usize) -> f32 {
    use archmage::SimdToken;
    {
        if let Some(__t) = archmage::X64V4Token::summon() {
            return unsafe { unsafe_sum_v4(__t, ptr, len) };
        }
        if let Some(__t) = archmage::X64V3Token::summon() {
            return unsafe { unsafe_sum_v3(__t, ptr, len) };
        }
    }
    unsafe { unsafe_sum_scalar(archmage::ScalarToken, ptr, len) }
}
#[doc(hidden)]
#[allow(dead_code)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
)]
#[inline]
fn __arcane_unsafe_sum_v4(
    _token: archmage::X64V4Token,
    ptr: *const f32,
    len: usize,
) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..len {
        sum += unsafe { *ptr.add(i) };
    }
    sum
}
#[allow(dead_code)]
#[inline(always)]
unsafe fn unsafe_sum_v4(
    _token: archmage::X64V4Token,
    ptr: *const f32,
    len: usize,
) -> f32 {
    unsafe { __arcane_unsafe_sum_v4(_token, ptr, len) }
}
#[doc(hidden)]
#[allow(dead_code)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn __arcane_unsafe_sum_v3(
    _token: archmage::X64V3Token,
    ptr: *const f32,
    len: usize,
) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..len {
        sum += unsafe { *ptr.add(i) };
    }
    sum
}
#[allow(dead_code)]
#[inline(always)]
unsafe fn unsafe_sum_v3(
    _token: archmage::X64V3Token,
    ptr: *const f32,
    len: usize,
) -> f32 {
    unsafe { __arcane_unsafe_sum_v3(_token, ptr, len) }
}
#[allow(dead_code)]
unsafe fn unsafe_sum_scalar(
    _token: archmage::ScalarToken,
    ptr: *const f32,
    len: usize,
) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..len {
        sum += unsafe { *ptr.add(i) };
    }
    sum
}
fn main() {}

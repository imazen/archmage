use archmage::autoversion;
unsafe fn process(ptr: *const f32, len: usize) -> f32 {
    #[allow(unused_imports)]
    use archmage::SimdToken;
    {
        if let Some(__t) = archmage::X64V4Token::summon() {
            return unsafe { process_v4(__t, ptr, len) };
        }
        if let Some(__t) = archmage::X64V3Token::summon() {
            return unsafe { process_v3(__t, ptr, len) };
        }
    }
    unsafe { process_scalar(archmage::ScalarToken, ptr, len) }
}
#[doc(hidden)]
#[allow(dead_code)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
)]
#[inline]
fn __arcane_process_v4(
    _token: archmage::X64V4Token,
    ptr: *const f32,
    len: usize,
) -> f32 {
    let mut s = 0.0f32;
    for i in 0..len {
        s += unsafe { *ptr.add(i) };
    }
    s
}
#[allow(dead_code)]
#[inline(always)]
unsafe fn process_v4(_token: archmage::X64V4Token, ptr: *const f32, len: usize) -> f32 {
    const _ARCHMAGE_TOKEN_MISMATCH: () = [
        (),
    ][!(<archmage::X64V4Token>::__ARCHMAGE_TIER_TAG == 4263219212u32) as usize];
    unsafe { __arcane_process_v4(_token, ptr, len) }
}
#[doc(hidden)]
#[allow(dead_code)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn __arcane_process_v3(
    _token: archmage::X64V3Token,
    ptr: *const f32,
    len: usize,
) -> f32 {
    let mut s = 0.0f32;
    for i in 0..len {
        s += unsafe { *ptr.add(i) };
    }
    s
}
#[allow(dead_code)]
#[inline(always)]
unsafe fn process_v3(_token: archmage::X64V3Token, ptr: *const f32, len: usize) -> f32 {
    const _ARCHMAGE_TOKEN_MISMATCH: () = [
        (),
    ][!(<archmage::X64V3Token>::__ARCHMAGE_TIER_TAG == 4085983307u32) as usize];
    unsafe { __arcane_process_v3(_token, ptr, len) }
}
#[allow(dead_code)]
unsafe fn process_scalar(
    _token: archmage::ScalarToken,
    ptr: *const f32,
    len: usize,
) -> f32 {
    let mut s = 0.0f32;
    for i in 0..len {
        s += unsafe { *ptr.add(i) };
    }
    s
}
fn main() {}

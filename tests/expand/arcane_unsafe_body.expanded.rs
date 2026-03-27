use archmage::{arcane, X64V3Token};
#[doc(hidden)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn __arcane_with_raw_ptr(token: X64V3Token, ptr: *const f32, len: usize) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..len {
        sum += unsafe { *ptr.add(i) };
    }
    sum
}
#[inline(always)]
fn with_raw_ptr(token: X64V3Token, ptr: *const f32, len: usize) -> f32 {
    unsafe { __arcane_with_raw_ptr(token, ptr, len) }
}
fn main() {}

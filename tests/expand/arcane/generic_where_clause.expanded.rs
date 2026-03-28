use archmage::{arcane, X64V3Token};
#[doc(hidden)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn __arcane_sum_slice<T>(token: X64V3Token, data: &[T]) -> f32
where
    T: Copy + Into<f32>,
{
    let _ = token;
    let mut s = 0.0f32;
    for &x in data {
        s += x.into();
    }
    s
}
#[inline(always)]
fn sum_slice<T>(token: X64V3Token, data: &[T]) -> f32
where
    T: Copy + Into<f32>,
{
    unsafe { __arcane_sum_slice::<T>(token, data) }
}
fn main() {}

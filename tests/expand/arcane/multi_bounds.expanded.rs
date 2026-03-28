use archmage::{arcane, X64V3Token};
use core::ops::{Add, Mul};
#[doc(hidden)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn __arcane_fma<T>(token: X64V3Token, a: T, b: T, c: T) -> T
where
    T: Copy + Mul<Output = T> + Add<Output = T>,
{
    let _ = token;
    a * b + c
}
#[inline(always)]
fn fma<T>(token: X64V3Token, a: T, b: T, c: T) -> T
where
    T: Copy + Mul<Output = T> + Add<Output = T>,
{
    unsafe { __arcane_fma::<T>(token, a, b, c) }
}
fn main() {}

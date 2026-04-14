use archmage::{arcane, rite, X64V3Token};
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn normalize(v: &[f32; 4]) -> [f32; 4] {
    let mut s = 0.0f32;
    for &x in v {
        s += x * x;
    }
    let inv = 1.0 / s.sqrt();
    [v[0] * inv, v[1] * inv, v[2] * inv, v[3] * inv]
}
#[doc(hidden)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn __arcane_process(token: X64V3Token, data: &[f32; 4]) -> [f32; 4] {
    let _ = token;
    normalize(data)
}
#[inline(always)]
fn process(token: X64V3Token, data: &[f32; 4]) -> [f32; 4] {
    const _ARCHMAGE_TOKEN_MISMATCH: () = [
        (),
    ][!(<X64V3Token>::__ARCHMAGE_TIER_TAG == 4085983307u32) as usize];
    unsafe { __arcane_process(token, data) }
}
fn main() {}

use archmage::{arcane, X64V3Token};
#[doc(hidden)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn __arcane_process(
    __archmage_arg_0: X64V3Token,
    __archmage_arg_1: f32,
    __archmage_arg_2: f32,
) -> f32 {
    let _: X64V3Token = __archmage_arg_0;
    let _: f32 = __archmage_arg_1;
    let _: f32 = __archmage_arg_2;
    0.0
}
#[inline(always)]
fn process(
    __archmage_arg_0: X64V3Token,
    __archmage_arg_1: f32,
    __archmage_arg_2: f32,
) -> f32 {
    const _ARCHMAGE_TOKEN_MISMATCH: () = [
        (),
    ][!(<X64V3Token>::__ARCHMAGE_TIER_TAG == 4085983307u32) as usize];
    unsafe { __arcane_process(__archmage_arg_0, __archmage_arg_1, __archmage_arg_2) }
}
fn main() {}

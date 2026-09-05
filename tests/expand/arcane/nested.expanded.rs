use archmage::{arcane, X64V3Token};
#[inline(always)]
fn process(token: X64V3Token, a: f32, b: f32) -> f32 {
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    fn __simd_inner_process(token: X64V3Token, a: f32, b: f32) -> f32 {
        a + b
    }
    let _: () = <X64V3Token>::__ARCHMAGE_ASSERT_TIER_F38B284B;
    unsafe { __simd_inner_process(token, a, b) }
}
fn main() {}

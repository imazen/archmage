use archmage::prelude::*;
struct S {
    k: f32,
}
impl S {
    #[doc(hidden)]
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    fn __arcane_probe(token: X64V3Token, x: f32) -> f32 {
        let _ = token;
        Self::offset() + x
    }
    #[inline(always)]
    fn probe(token: X64V3Token, x: f32) -> f32 {
        let _: () = <X64V3Token>::__ARCHMAGE_ASSERT_TIER_F38B284B;
        unsafe { Self::__arcane_probe(token, x) }
    }
    fn offset() -> f32 {
        1.0
    }
}
fn main() {}

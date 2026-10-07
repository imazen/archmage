use archmage::prelude::*;
struct S {
    k: f32,
}
trait Work {
    fn run(&self, token: X64V3Token, x: f32) -> f32;
}
impl Work for S {
    #[inline(always)]
    fn run(&self, token: X64V3Token, x: f32) -> f32 {
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        fn __simd_inner_run(_self: &S, token: X64V3Token, x: f32) -> f32 {
            let _ = token;
            _self.k + x
        }
        let _: () = <X64V3Token>::__ARCHMAGE_ASSERT_TIER_F38B284B;
        unsafe { __simd_inner_run(self, token, x) }
    }
}
fn main() {}

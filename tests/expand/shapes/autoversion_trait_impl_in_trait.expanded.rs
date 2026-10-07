use archmage::autoversion;
struct S {
    k: f32,
}
trait Work {
    fn run(&self, x: f32) -> f32;
}
impl Work for S {
    fn run(&self, x: f32) -> f32 {
        #[doc(hidden)]
        #[allow(dead_code)]
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        fn __arcane_run_v3(_self: &S, _token: archmage::X64V3Token, x: f32) -> f32 {
            _self.k + x
        }
        #[allow(dead_code)]
        #[inline(always)]
        fn run_v3(_self: &S, _token: archmage::X64V3Token, x: f32) -> f32 {
            let _: () = <archmage::X64V3Token>::__ARCHMAGE_ASSERT_TIER_F38B284B;
            unsafe { __arcane_run_v3(_self, _token, x) }
        }
        #[allow(dead_code)]
        fn run_scalar(_self: &S, _token: archmage::ScalarToken, x: f32) -> f32 {
            _self.k + x
        }
        #[allow(unused_imports)]
        use archmage::SimdToken;
        {
            if let Some(__t) = archmage::X64V3Token::summon() {
                return run_v3(self, __t, x);
            }
        }
        run_scalar(self, archmage::ScalarToken, x)
    }
}
fn main() {
    let _ = S { k: 1.0 }.run(2.0);
}

use archmage::autoversion;
struct S {
    k: f32,
}
trait Work {
    fn pick<'a>(&'a self, other: &'a f32) -> &'a f32;
}
impl Work for S {
    fn pick<'a>(&'a self, other: &'a f32) -> &'a f32 {
        #[doc(hidden)]
        #[allow(dead_code)]
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        fn __arcane_pick_v3<'a>(
            _self: &'a S,
            _token: archmage::X64V3Token,
            other: &'a f32,
        ) -> &'a f32 {
            if _self.k > *other { &_self.k } else { other }
        }
        #[allow(dead_code)]
        #[inline(always)]
        fn pick_v3<'a>(
            _self: &'a S,
            _token: archmage::X64V3Token,
            other: &'a f32,
        ) -> &'a f32 {
            let _: () = <archmage::X64V3Token>::__ARCHMAGE_ASSERT_TIER_F38B284B;
            unsafe { __arcane_pick_v3(_self, _token, other) }
        }
        #[allow(dead_code)]
        fn pick_scalar<'a>(
            _self: &'a S,
            _token: archmage::ScalarToken,
            other: &'a f32,
        ) -> &'a f32 {
            if _self.k > *other { &_self.k } else { other }
        }
        #[allow(unused_imports)]
        use archmage::SimdToken;
        {
            if let Some(__t) = archmage::X64V3Token::summon() {
                return pick_v3(self, __t, other);
            }
        }
        pick_scalar(self, archmage::ScalarToken, other)
    }
}
fn main() {
    let (s, t) = (S { k: 1.0 }, S { k: 2.0 });
    let _: f32 = *s.pick(&t.k);
}

use archmage::{arcane, HasX64V2};
#[doc(hidden)]
#[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b")]
#[inline]
fn __arcane_process(token: impl HasX64V2, a: f32) -> f32 {
    a + 1.0
}
#[inline(always)]
fn process(token: impl HasX64V2, a: f32) -> f32 {
    {
        #[inline(always)]
        const fn __archmage_assert_tier_trait<__T: ?Sized + ::archmage::HasX64V2>(
            _: &__T,
        ) {}
        __archmage_assert_tier_trait(&token);
    }
    unsafe { __arcane_process(token, a) }
}
fn main() {}

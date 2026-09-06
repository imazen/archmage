use archmage::{rite, HasX64V2};
#[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b")]
#[inline]
fn helper(token: impl HasX64V2, a: f32) -> f32 {
    {
        #[inline(always)]
        const fn __archmage_assert_tier_trait<__T: ?Sized + ::archmage::HasX64V2>(
            _: &__T,
        ) {}
        __archmage_assert_tier_trait(&token);
    }
    let _ = token;
    a
}
fn main() {}

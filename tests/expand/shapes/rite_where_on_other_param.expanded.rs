use archmage::prelude::*;
#[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b")]
#[inline]
fn probe<T: HasX64V2, U>(token: T, u: U) -> U
where
    U: Copy,
{
    {
        #[inline(always)]
        const fn __archmage_assert_tier_trait<__T: ?Sized + ::archmage::HasX64V2>(
            _: &__T,
        ) {}
        __archmage_assert_tier_trait(&token);
    }
    let _ = token;
    u
}
fn main() {}

use archmage::prelude::*;
#[doc(hidden)]
#[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b")]
#[inline]
fn __arcane_probe<T: HasX64V2, U>(token: T, u: U) -> U
where
    U: Copy,
{
    let _ = token;
    u
}
#[inline(always)]
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
    unsafe { __arcane_probe::<T, U>(token, u) }
}
fn main() {}

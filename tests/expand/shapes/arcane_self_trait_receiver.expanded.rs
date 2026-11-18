use archmage::{HasX64V2, arcane};
trait V2Op {
    fn by_val(self, x: u32) -> u32;
    fn by_ref(&self, x: u32) -> u32;
    fn by_mut(&mut self, x: u32) -> u32;
}
impl<T: HasX64V2> V2Op for T {
    #[inline(always)]
    fn by_val(self, x: u32) -> u32 {
        #[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b")]
        #[inline]
        fn __simd_inner_by_val(_self: impl HasX64V2, x: u32) -> u32 {
            let _ = _self;
            x + 1
        }
        {
            #[inline(always)]
            const fn __archmage_assert_tier_trait<__T: ?Sized + ::archmage::HasX64V2>(
                _: &__T,
            ) {}
            __archmage_assert_tier_trait(&self);
        }
        unsafe { __simd_inner_by_val(self, x) }
    }
    #[inline(always)]
    fn by_ref(&self, x: u32) -> u32 {
        #[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b")]
        #[inline]
        fn __simd_inner_by_ref(_self: &impl HasX64V2, x: u32) -> u32 {
            let _ = _self;
            x + 2
        }
        {
            #[inline(always)]
            const fn __archmage_assert_tier_trait<__T: ?Sized + ::archmage::HasX64V2>(
                _: &__T,
            ) {}
            __archmage_assert_tier_trait(&*self);
        }
        unsafe { __simd_inner_by_ref(self, x) }
    }
    #[inline(always)]
    fn by_mut(&mut self, x: u32) -> u32 {
        #[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b")]
        #[inline]
        fn __simd_inner_by_mut(_self: &mut impl HasX64V2, x: u32) -> u32 {
            let _ = _self;
            x + 3
        }
        {
            #[inline(always)]
            const fn __archmage_assert_tier_trait<__T: ?Sized + ::archmage::HasX64V2>(
                _: &__T,
            ) {}
            __archmage_assert_tier_trait(&*self);
        }
        unsafe { __simd_inner_by_mut(self, x) }
    }
}
fn main() {}

use super::{f32x4, f32x8};
#[cfg(feature="w512")]
use super::f32x16;
impl<M: crate::simd::generic::ConstructorMode, T: crate::simd::generic::F32x8FromHalves> f32x8<T, M> {
    /// Combine two `f32x4<T, M>` halves into one `f32x8<T, M>`.
    ///
    /// The wider backend determines the required CPU features.
    #[inline(always)]
    pub fn from_halves(token: T, lo: f32x4<T, M>, hi: f32x4<T, M>) -> Self {
        Self::from_repr_unchecked(
            token,
            <T as crate::simd::generic::F32x8FromHalves>::from_halves(token, lo.into_repr(), hi.into_repr()),
        )
    }
    /// Extract the low 128-bit half.
    #[inline(always)]
    pub fn low(self) -> f32x4<T, M> {
        f32x4::from_repr_unchecked(
            self.1,
            <T as crate::simd::generic::F32x8FromHalves>::low(self.1, self.into_repr()),
        )
    }
    /// Extract the high 128-bit half.
    #[inline(always)]
    pub fn high(self) -> f32x4<T, M> {
        f32x4::from_repr_unchecked(
            self.1,
            <T as crate::simd::generic::F32x8FromHalves>::high(self.1, self.into_repr()),
        )
    }
    /// Split into `(low, high)` halves.
    #[inline(always)]
    pub fn split(self) -> (f32x4<T, M>, f32x4<T, M>) {
        (self.low(), self.high())
    }
}

#[cfg(feature = "w512")]
impl<M: crate::simd::generic::ConstructorMode, T: crate::simd::generic::F32x16FromHalves> f32x16<T, M> {
    /// Combine two `f32x8<T, M>` halves into one `f32x16<T, M>`.
    #[inline(always)]
    pub fn from_halves(token: T, lo: f32x8<T, M>, hi: f32x8<T, M>) -> Self {
        Self::from_repr_unchecked(
            token,
            <T as crate::simd::generic::F32x16FromHalves>::from_halves(token, lo.into_repr(), hi.into_repr()),
        )
    }
    /// Extract the low 256-bit half.
    #[inline(always)]
    pub fn low(self) -> f32x8<T, M> {
        f32x8::from_repr_unchecked(
            self.1,
            <T as crate::simd::generic::F32x16FromHalves>::low(self.1, self.into_repr()),
        )
    }
    /// Extract the high 256-bit half.
    #[inline(always)]
    pub fn high(self) -> f32x8<T, M> {
        f32x8::from_repr_unchecked(
            self.1,
            <T as crate::simd::generic::F32x16FromHalves>::high(self.1, self.into_repr()),
        )
    }
    /// Split into `(low, high)` halves.
    #[inline(always)]
    pub fn split(self) -> (f32x8<T, M>, f32x8<T, M>) {
        (self.low(), self.high())
    }
}
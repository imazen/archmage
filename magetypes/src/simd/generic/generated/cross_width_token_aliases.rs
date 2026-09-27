// Generated explicit-token migration aliases. Do not edit.
impl<T: F32x8FromHalves> f32x8<T> {
    #[inline(always)]
    #[doc = "Explicit-token alias of [`Self::from_halves`], with identical arguments and behavior.\n\nThe `_t` spelling is intended for migration to magetypes 0.10.\nThe caller does not need a target-feature annotation."]
    #[forbid(unsafe_code)]
    pub fn from_halves_t(token: T, lo: f32x4<T>, hi: f32x4<T>) -> Self {
        Self::from_halves(token, lo, hi)
    }
}
#[cfg(feature = "w512")]
impl<T: F32x16FromHalves> f32x16<T> {
    #[inline(always)]
    #[doc = "Explicit-token alias of [`Self::from_halves`], with identical arguments and behavior.\n\nThe `_t` spelling is intended for migration to magetypes 0.10.\nThe caller does not need a target-feature annotation."]
    #[forbid(unsafe_code)]
    pub fn from_halves_t(token: T, lo: f32x8<T>, hi: f32x8<T>) -> Self {
        Self::from_halves(token, lo, hi)
    }
}

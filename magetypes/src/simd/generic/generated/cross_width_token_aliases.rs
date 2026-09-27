// Generated deprecated token-constructor forwarders. Do not edit.
impl<T: F32x8FromHalves> f32x8<T> {
    #[inline(always)]
    #[doc = "Deprecated token-taking spelling of [`Self::from_halves_t`].\n\nUse `from_halves_t` to keep explicit-token construction when `from_halves` becomes tokenless in magetypes 0.10."]
    #[deprecated(
        note = "Use from_halves_t(token, lo, hi); from_halves becomes tokenless in magetypes 0.10."
    )]
    #[forbid(unsafe_code)]
    pub fn from_halves(token: T, lo: f32x4<T>, hi: f32x4<T>) -> Self {
        Self::from_halves_t(token, lo, hi)
    }
}
#[cfg(feature = "w512")]
impl<T: F32x16FromHalves> f32x16<T> {
    #[inline(always)]
    #[doc = "Deprecated token-taking spelling of [`Self::from_halves_t`].\n\nUse `from_halves_t` to keep explicit-token construction when `from_halves` becomes tokenless in magetypes 0.10."]
    #[deprecated(
        note = "Use from_halves_t(token, lo, hi); from_halves becomes tokenless in magetypes 0.10."
    )]
    #[forbid(unsafe_code)]
    pub fn from_halves(token: T, lo: f32x8<T>, hi: f32x8<T>) -> Self {
        Self::from_halves_t(token, lo, hi)
    }
}

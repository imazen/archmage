#[cfg(feature = "w512")]
use super::f32x16;
use super::{f32x4, f32x8};
impl<M: crate::simd::generic::ConstructorMode, T: crate::simd::generic::F32x8FromHalves>
    f32x8<T, M>
{
    /// Combine two `f32x4<T, M>` halves into one `f32x8<T, M>`.
    ///
    /// The wider backend determines the required CPU features.
    #[inline(always)]
    pub(crate) fn from_halves_with_token(token: T, lo: f32x4<T, M>, hi: f32x4<T, M>) -> Self {
        Self::from_repr_unchecked(
            token,
            <T as crate::simd::generic::F32x8FromHalves>::from_halves(
                token,
                lo.into_repr(),
                hi.into_repr(),
            ),
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
impl<M: crate::simd::generic::ConstructorMode, T: crate::simd::generic::F32x16FromHalves>
    f32x16<T, M>
{
    /// Combine two `f32x8<T, M>` halves into one `f32x16<T, M>`.
    #[inline(always)]
    pub(crate) fn from_halves_with_token(token: T, lo: f32x8<T, M>, hi: f32x8<T, M>) -> Self {
        Self::from_repr_unchecked(
            token,
            <T as crate::simd::generic::F32x16FromHalves>::from_halves(
                token,
                lo.into_repr(),
                hi.into_repr(),
            ),
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
#[cfg(feature = "w512")]
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl f32x16<archmage::X64V4Token, crate::simd::generic::Context> {
    /// Combine two `f32x8<T, M>` halves into one `f32x16<T, M>`.
    #[forbid(unsafe_code)]
    #[archmage::rite(v4)]
    pub fn from_halves(
        lo: f32x8<archmage::X64V4Token, crate::simd::generic::Context>,
        hi: f32x8<archmage::X64V4Token, crate::simd::generic::Context>,
    ) -> Self {
        Self::from_halves_with_token(archmage::X64V4Token::from_context(), lo, hi)
    }
}

#[cfg(feature = "w512")]
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl f32x16<archmage::X64V4xToken, crate::simd::generic::Context> {
    /// Combine two `f32x8<T, M>` halves into one `f32x16<T, M>`.
    #[forbid(unsafe_code)]
    #[archmage::rite(v4x)]
    pub fn from_halves(
        lo: f32x8<archmage::X64V4xToken, crate::simd::generic::Context>,
        hi: f32x8<archmage::X64V4xToken, crate::simd::generic::Context>,
    ) -> Self {
        Self::from_halves_with_token(archmage::X64V4xToken::from_context(), lo, hi)
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "aarch64")]
impl f32x16<archmage::NeonToken, crate::simd::generic::Context> {
    /// Combine two `f32x8<T, M>` halves into one `f32x16<T, M>`.
    #[forbid(unsafe_code)]
    #[archmage::rite(neon)]
    pub fn from_halves(
        lo: f32x8<archmage::NeonToken, crate::simd::generic::Context>,
        hi: f32x8<archmage::NeonToken, crate::simd::generic::Context>,
    ) -> Self {
        Self::from_halves_with_token(archmage::NeonToken::from_context(), lo, hi)
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "wasm32")]
impl f32x16<archmage::Wasm128Token, crate::simd::generic::Context> {
    /// Combine two `f32x8<T, M>` halves into one `f32x16<T, M>`.
    #[forbid(unsafe_code)]
    #[archmage::rite(wasm128)]
    pub fn from_halves(
        lo: f32x8<archmage::Wasm128Token, crate::simd::generic::Context>,
        hi: f32x8<archmage::Wasm128Token, crate::simd::generic::Context>,
    ) -> Self {
        Self::from_halves_with_token(archmage::Wasm128Token::from_context(), lo, hi)
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "x86_64")]
impl f32x16<archmage::X64V3Token, crate::simd::generic::Context> {
    /// Combine two `f32x8<T, M>` halves into one `f32x16<T, M>`.
    #[forbid(unsafe_code)]
    #[archmage::rite(v3)]
    pub fn from_halves(
        lo: f32x8<archmage::X64V3Token, crate::simd::generic::Context>,
        hi: f32x8<archmage::X64V3Token, crate::simd::generic::Context>,
    ) -> Self {
        Self::from_halves_with_token(archmage::X64V3Token::from_context(), lo, hi)
    }
}

#[cfg(feature = "w512")]
impl<T: crate::simd::generic::F32x16FromHalves> f32x16<T, crate::simd::generic::Explicit> {
    /// Combine two `f32x8<T, M>` halves into one `f32x16<T, M>`.
    #[inline(always)]
    pub fn from_halves(
        token: T,
        lo: f32x8<T, crate::simd::generic::Explicit>,
        hi: f32x8<T, crate::simd::generic::Explicit>,
    ) -> Self {
        Self::from_halves_with_token(token, lo, hi)
    }
}

#[cfg(feature = "w512")]
impl f32x16<archmage::ScalarToken, crate::simd::generic::Context> {
    /// Combine two `f32x8<T, M>` halves into one `f32x16<T, M>`.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn from_halves(
        lo: f32x8<archmage::ScalarToken, crate::simd::generic::Context>,
        hi: f32x8<archmage::ScalarToken, crate::simd::generic::Context>,
    ) -> Self {
        Self::from_halves_with_token(archmage::ScalarToken, lo, hi)
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl f32x8<archmage::Avx512Fp16Token, crate::simd::generic::Context> {
    /// Combine two `f32x4<T, M>` halves into one `f32x8<T, M>`.
    /// The wider backend determines the required CPU features.
    #[forbid(unsafe_code)]
    #[archmage::rite(fp16)]
    pub fn from_halves(
        lo: f32x4<archmage::Avx512Fp16Token, crate::simd::generic::Context>,
        hi: f32x4<archmage::Avx512Fp16Token, crate::simd::generic::Context>,
    ) -> Self {
        Self::from_halves_with_token(archmage::Avx512Fp16Token::from_context(), lo, hi)
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl f32x8<archmage::X64V4Token, crate::simd::generic::Context> {
    /// Combine two `f32x4<T, M>` halves into one `f32x8<T, M>`.
    /// The wider backend determines the required CPU features.
    #[forbid(unsafe_code)]
    #[archmage::rite(v4)]
    pub fn from_halves(
        lo: f32x4<archmage::X64V4Token, crate::simd::generic::Context>,
        hi: f32x4<archmage::X64V4Token, crate::simd::generic::Context>,
    ) -> Self {
        Self::from_halves_with_token(archmage::X64V4Token::from_context(), lo, hi)
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl f32x8<archmage::X64V4xToken, crate::simd::generic::Context> {
    /// Combine two `f32x4<T, M>` halves into one `f32x8<T, M>`.
    /// The wider backend determines the required CPU features.
    #[forbid(unsafe_code)]
    #[archmage::rite(v4x)]
    pub fn from_halves(
        lo: f32x4<archmage::X64V4xToken, crate::simd::generic::Context>,
        hi: f32x4<archmage::X64V4xToken, crate::simd::generic::Context>,
    ) -> Self {
        Self::from_halves_with_token(archmage::X64V4xToken::from_context(), lo, hi)
    }
}

#[cfg(target_arch = "aarch64")]
impl f32x8<archmage::NeonToken, crate::simd::generic::Context> {
    /// Combine two `f32x4<T, M>` halves into one `f32x8<T, M>`.
    /// The wider backend determines the required CPU features.
    #[forbid(unsafe_code)]
    #[archmage::rite(neon)]
    pub fn from_halves(
        lo: f32x4<archmage::NeonToken, crate::simd::generic::Context>,
        hi: f32x4<archmage::NeonToken, crate::simd::generic::Context>,
    ) -> Self {
        Self::from_halves_with_token(archmage::NeonToken::from_context(), lo, hi)
    }
}

#[cfg(target_arch = "wasm32")]
impl f32x8<archmage::Wasm128Token, crate::simd::generic::Context> {
    /// Combine two `f32x4<T, M>` halves into one `f32x8<T, M>`.
    /// The wider backend determines the required CPU features.
    #[forbid(unsafe_code)]
    #[archmage::rite(wasm128)]
    pub fn from_halves(
        lo: f32x4<archmage::Wasm128Token, crate::simd::generic::Context>,
        hi: f32x4<archmage::Wasm128Token, crate::simd::generic::Context>,
    ) -> Self {
        Self::from_halves_with_token(archmage::Wasm128Token::from_context(), lo, hi)
    }
}

#[cfg(target_arch = "x86_64")]
impl f32x8<archmage::X64V3Token, crate::simd::generic::Context> {
    /// Combine two `f32x4<T, M>` halves into one `f32x8<T, M>`.
    /// The wider backend determines the required CPU features.
    #[forbid(unsafe_code)]
    #[archmage::rite(v3)]
    pub fn from_halves(
        lo: f32x4<archmage::X64V3Token, crate::simd::generic::Context>,
        hi: f32x4<archmage::X64V3Token, crate::simd::generic::Context>,
    ) -> Self {
        Self::from_halves_with_token(archmage::X64V3Token::from_context(), lo, hi)
    }
}

impl<T: crate::simd::generic::F32x8FromHalves> f32x8<T, crate::simd::generic::Explicit> {
    /// Combine two `f32x4<T, M>` halves into one `f32x8<T, M>`.
    /// The wider backend determines the required CPU features.
    #[inline(always)]
    pub fn from_halves(
        token: T,
        lo: f32x4<T, crate::simd::generic::Explicit>,
        hi: f32x4<T, crate::simd::generic::Explicit>,
    ) -> Self {
        Self::from_halves_with_token(token, lo, hi)
    }
}

impl f32x8<archmage::ScalarToken, crate::simd::generic::Context> {
    /// Combine two `f32x4<T, M>` halves into one `f32x8<T, M>`.
    /// The wider backend determines the required CPU features.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn from_halves(
        lo: f32x4<archmage::ScalarToken, crate::simd::generic::Context>,
        hi: f32x4<archmage::ScalarToken, crate::simd::generic::Context>,
    ) -> Self {
        Self::from_halves_with_token(archmage::ScalarToken, lo, hi)
    }
}

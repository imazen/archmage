//! Cross-width raising and lowering for paired generic SIMD types.
//!
//! Implements [`F32x8Halves`] (and the matching `f32x8::from_halves` /
//! `f32x8::low` / `f32x8::high` methods) so generic code parameterized on
//! `T: F32x8Halves` can pack two narrower 128-bit halves into a 256-bit
//! vector, or extract them back out, without naming the platform.
//!
//! Per-tier intent:
//!
//! - `X64V3Token` (AVX): `_mm256_insertf128_ps` / `_mm256_castps256_ps128`
//!   / `_mm256_extractf128_ps`.
//! - `NeonToken`: zero-cost — `f32x8<NeonToken>::Repr` is already
//!   `[float32x4_t; 2]`, so raising is array construction and lowering is
//!   array indexing.
//! - `Wasm128Token`: same shape as NEON — `Repr` is `[v128; 2]`.
//! - `ScalarToken`: array concat / split.
//!
//! The token is load-bearing on x86 even though it's free on the other tiers:
//! combining two `f32x4<X64V3Token>` (which only needs SSE) into an
//! `f32x8<X64V3Token>` emits AVX (`vinsertf128`), and the V3 token is what
//! proves AVX is available. The trait takes the token-shape uniformly to
//! match the other token-gated constructors (`splat`, `load`, etc.).
//!
//! Status: scoped to `f32x8` for [#36] (moxcms unblock). Extension to the
//! other 9 W256 pairs (`i16x16`, `i32x8`, `f64x4`, `u8x32`, `u16x16`,
//! `u32x8`, `i8x32`, `i64x4`, `u64x4`) follows the same template; expand
//! when those land downstream demand.
//!
//! [#36]: https://github.com/imazen/archmage/issues/36

use archmage::SimdToken;

use crate::simd::backends::sealed::Sealed;
use crate::simd::backends::{F32x4Backend, F32x8Backend};
use crate::simd::generic::{f32x4, f32x8};

/// Raise/lower between `f32x4<T>` and `f32x8<T>` for the same token `T`.
///
/// Implemented for every token that supplies both [`F32x4Backend`] and
/// [`F32x8Backend`]: `X64V3Token`, `NeonToken`, `Wasm128Token`, and
/// `ScalarToken` today.
///
/// Generic code rarely names this trait directly — write
/// `T: F32x8Backend + F32x4Backend` (the supertraits) and call
/// [`f32x8::from_halves`] / [`f32x8::low`] / [`f32x8::high`] instead.
pub trait F32x8Halves: F32x8Backend + F32x4Backend + SimdToken + Sealed + Copy + 'static {
    /// Combine two 128-bit halves into a 256-bit representation.
    fn from_halves(
        lo: <Self as F32x4Backend>::Repr,
        hi: <Self as F32x4Backend>::Repr,
    ) -> <Self as F32x8Backend>::Repr;

    /// Extract the low 128-bit half of a 256-bit representation.
    fn low(wide: <Self as F32x8Backend>::Repr) -> <Self as F32x4Backend>::Repr;

    /// Extract the high 128-bit half of a 256-bit representation.
    fn high(wide: <Self as F32x8Backend>::Repr) -> <Self as F32x4Backend>::Repr;
}

// ── X64V3Token (AVX) ────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
impl F32x8Halves for archmage::X64V3Token {
    #[inline(always)]
    fn from_halves(
        lo: core::arch::x86_64::__m128,
        hi: core::arch::x86_64::__m128,
    ) -> core::arch::x86_64::__m256 {
        // SAFETY: X64V3Token proves AVX (and SSE) are available at runtime.
        // _mm256_insertf128_ps / _mm256_castps128_ps256 are AVX intrinsics
        // expecting __m128 / __m256 inputs that we've just constructed.
        unsafe {
            core::arch::x86_64::_mm256_insertf128_ps(
                core::arch::x86_64::_mm256_castps128_ps256(lo),
                hi,
                1,
            )
        }
    }

    #[inline(always)]
    fn low(wide: core::arch::x86_64::__m256) -> core::arch::x86_64::__m128 {
        // SAFETY: X64V3Token proves AVX. _mm256_castps256_ps128 is a free
        // truncation of the low 128 bits.
        unsafe { core::arch::x86_64::_mm256_castps256_ps128(wide) }
    }

    #[inline(always)]
    fn high(wide: core::arch::x86_64::__m256) -> core::arch::x86_64::__m128 {
        // SAFETY: X64V3Token proves AVX. _mm256_extractf128_ps with imm=1
        // returns the high 128-bit lane.
        unsafe { core::arch::x86_64::_mm256_extractf128_ps(wide, 1) }
    }
}

// ── NeonToken (polyfilled — Repr is already a 2-element array) ──────────

#[cfg(target_arch = "aarch64")]
impl F32x8Halves for archmage::NeonToken {
    #[inline(always)]
    fn from_halves(
        lo: core::arch::aarch64::float32x4_t,
        hi: core::arch::aarch64::float32x4_t,
    ) -> [core::arch::aarch64::float32x4_t; 2] {
        [lo, hi]
    }

    #[inline(always)]
    fn low(wide: [core::arch::aarch64::float32x4_t; 2]) -> core::arch::aarch64::float32x4_t {
        wide[0]
    }

    #[inline(always)]
    fn high(wide: [core::arch::aarch64::float32x4_t; 2]) -> core::arch::aarch64::float32x4_t {
        wide[1]
    }
}

// ── Wasm128Token (polyfilled — Repr is already a 2-element array) ──────

#[cfg(target_arch = "wasm32")]
impl F32x8Halves for archmage::Wasm128Token {
    #[inline(always)]
    fn from_halves(
        lo: core::arch::wasm32::v128,
        hi: core::arch::wasm32::v128,
    ) -> [core::arch::wasm32::v128; 2] {
        [lo, hi]
    }

    #[inline(always)]
    fn low(wide: [core::arch::wasm32::v128; 2]) -> core::arch::wasm32::v128 {
        wide[0]
    }

    #[inline(always)]
    fn high(wide: [core::arch::wasm32::v128; 2]) -> core::arch::wasm32::v128 {
        wide[1]
    }
}

// ── ScalarToken (array concat / split) ──────────────────────────────────

impl F32x8Halves for archmage::ScalarToken {
    #[inline(always)]
    fn from_halves(lo: [f32; 4], hi: [f32; 4]) -> [f32; 8] {
        [lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]]
    }

    #[inline(always)]
    fn low(wide: [f32; 8]) -> [f32; 4] {
        [wide[0], wide[1], wide[2], wide[3]]
    }

    #[inline(always)]
    fn high(wide: [f32; 8]) -> [f32; 4] {
        [wide[4], wide[5], wide[6], wide[7]]
    }
}

// ── Generic exposure on f32x8<T> ────────────────────────────────────────

impl<T: F32x8Halves> f32x8<T> {
    /// Combine two `f32x4<T>` halves into one `f32x8<T>`.
    ///
    /// Token is load-bearing on x86 — passing it proves AVX is available so
    /// the wide vector can be constructed via `vinsertf128`. On NEON /
    /// Wasm128 / Scalar the token is redundant (the wider repr is just an
    /// array of two narrower reprs) but the API takes one uniformly.
    #[inline(always)]
    pub fn from_halves(_token: T, lo: f32x4<T>, hi: f32x4<T>) -> Self {
        Self::from_repr_unchecked(T::from_halves(lo.into_repr(), hi.into_repr()))
    }

    /// Extract the low 128-bit half.
    #[inline(always)]
    pub fn low(self) -> f32x4<T> {
        f32x4::from_repr_unchecked(T::low(self.into_repr()))
    }

    /// Extract the high 128-bit half.
    #[inline(always)]
    pub fn high(self) -> f32x4<T> {
        f32x4::from_repr_unchecked(T::high(self.into_repr()))
    }

    /// Split into `(low, high)` halves. Equivalent to `(self.low(), self.high())`.
    #[inline(always)]
    pub fn split(self) -> (f32x4<T>, f32x4<T>) {
        (self.low(), self.high())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip: `from_halves` then `split` should yield the same lanes.
    fn roundtrip<T: F32x8Halves>(token: T) {
        let lo = f32x4::<T>::from_array(token, [1.0, 2.0, 3.0, 4.0]);
        let hi = f32x4::<T>::from_array(token, [5.0, 6.0, 7.0, 8.0]);
        let wide = f32x8::<T>::from_halves(token, lo, hi);
        let (back_lo, back_hi) = wide.split();
        assert_eq!(back_lo.to_array(), [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(back_hi.to_array(), [5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn scalar_roundtrip() {
        roundtrip(archmage::ScalarToken::summon().expect("scalar always available"));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn x64v3_roundtrip() {
        if let Some(token) = archmage::X64V3Token::summon() {
            roundtrip(token);
        } else {
            eprintln!("X64V3Token not available — skipping");
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_roundtrip() {
        if let Some(token) = archmage::NeonToken::summon() {
            roundtrip(token);
        } else {
            eprintln!("NeonToken not available — skipping");
        }
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn wasm128_roundtrip() {
        if let Some(token) = archmage::Wasm128Token::summon() {
            roundtrip(token);
        }
    }

    #[test]
    fn from_halves_lane_order() {
        let token = archmage::ScalarToken::summon().unwrap();
        let lo = f32x4::<archmage::ScalarToken>::from_array(token, [10.0, 20.0, 30.0, 40.0]);
        let hi = f32x4::<archmage::ScalarToken>::from_array(token, [50.0, 60.0, 70.0, 80.0]);
        let wide = f32x8::<archmage::ScalarToken>::from_halves(token, lo, hi);
        // Lane 0 must come from `lo[0]`, lane 4 from `hi[0]`.
        assert_eq!(
            wide.to_array(),
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
        );
    }
}

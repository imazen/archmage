//! Fast single-instruction float min/max and in-bounds float-to-integer truncation.
//!
//! ## Fast pseudo-min / pseudo-max (`min_fast`, `max_fast`)
//!
//! Portable [`f32x4::min`] / [`f32x4::max`] follow IEEE-754 `minNum`/`maxNum` or
//! WebAssembly `f32x4.min`/`f32x4.max` (NaN-propagating), which on baseline WASM
//! (`+simd128` without `+relaxed-simd`) costs multiple instructions compared to
//! `f32x4.pmin` / `f32x4.pmax` and on Scalar uses `f32::min` / `f32::max`.
//!
//! `min_fast` and `max_fast` emit a single instruction on every hardware SIMD tier:
//! - **x86 (`X64V3Token`, `X64V4Token`, `X64V4xToken`, `Avx512Fp16Token`):** `vminps` / `vmaxps` / `vminpd` / `vmaxpd` (`a < b ? a : b` / `a > b ? a : b`; returns `b` when either input is `NaN`).
//! - **ARM64 (`NeonToken`):** `fminnm` / `fmaxnm` (`vminnmq_f32` / `vmaxnmq_f32`, `vminnmq_f64` / `vmaxnmq_f64`).
//! - **WASM (`Wasm128Token`):** `f32x4.pmin(b, a)` / `f32x4.pmax(b, a)` and `f64x2.pmin(b, a)` / `f64x2.pmax(b, a)` (matching `vminps(a, b)` / `vmaxps(a, b)` so `x.max_fast(zero)` clamps `NaN` to `0.0` when `zero` is the second operand).
//! - **WASM (`Wasm128RelaxedFmaToken`):** `f32x4.relaxed_min` / `f32x4.relaxed_max` and `f64x2.relaxed_min` / `f64x2.relaxed_max`.
//! - **Scalar (`ScalarToken`):** branchless `if a < b { a } else { b }` / `if a > b { a } else { b }`.
//!
//! For finite non-zero inputs (and finite inputs where `-0.0` vs `+0.0` sign does not
//! matter), all backends return identical results. On `NaN` or `(-0.0, +0.0)` ties,
//! behavior is backend-defined.
//!
//! ## Fast float-to-integer truncation (`to_i32_fast`, `to_u32_fast`)
//!
//! [`f32x4::to_i32`] (`to_i32_saturating` on x86) guarantees uniform saturation across
//! ISAs by emitting extra compare+blend instructions around `vcvttps2dq` on x86. When
//! inputs are already clamped or known finite and in-bounds (quantization after ReLU6 /
//! `clamp`, LUT index generation after range reduction, pixel packing), `to_i32_fast`
//! and `to_u32_fast` emit the fastest native truncation instruction directly without
//! overflow/NaN fixup masks.

#![forbid(unsafe_code)]

use archmage::SimdToken;

use crate::simd::backends::sealed::Sealed;
use crate::simd::backends::{
    F32x4Backend, F32x4Convert, F32x8Backend, F32x8Convert, F64x2Backend, F64x4Backend,
    U32x4Backend, U32x8Backend,
};
#[cfg(feature = "w512")]
use crate::simd::backends::{F32x16Backend, F32x16Convert, F64x8Backend, U32x16Backend};
use crate::simd::generic::{f32x4, f32x8, f64x2, f64x4, i32x4, i32x8, u32x4, u32x8};
#[cfg(feature = "w512")]
use crate::simd::generic::{f32x16, f64x8, i32x16, u32x16};

// ============================================================================
// min_fast / max_fast
// ============================================================================

impl<T: F32x4Backend> f32x4<T> {
    /// Fast single-instruction lane-wise minimum.
    ///
    /// Exact for finite inputs (up to `-0.0` vs `+0.0` sign); `NaN` handling is
    /// backend-specific (`vminps` / `f32x4.pmin(other, self)` / `if self < other`
    /// return `other` when `self` is `NaN`; `fminnm` returns the numeric operand).
    #[inline(always)]
    pub fn min_fast(self, other: Self) -> Self {
        Self(T::min_fast(self.1, self.0, other.0), self.1)
    }

    /// Fast single-instruction lane-wise maximum.
    ///
    /// Exact for finite inputs (up to `-0.0` vs `+0.0` sign); `NaN` handling is
    /// backend-specific (`vmaxps` / `f32x4.pmax(other, self)` / `if self > other`
    /// return `other` when `self` is `NaN`; `fmaxnm` returns the numeric operand).
    #[inline(always)]
    pub fn max_fast(self, other: Self) -> Self {
        Self(T::max_fast(self.1, self.0, other.0), self.1)
    }
}

impl<T: F32x8Backend> f32x8<T> {
    /// Fast single-instruction lane-wise minimum. See [`f32x4::min_fast`].
    #[inline(always)]
    pub fn min_fast(self, other: Self) -> Self {
        Self(T::min_fast(self.1, self.0, other.0), self.1)
    }

    /// Fast single-instruction lane-wise maximum. See [`f32x4::max_fast`].
    #[inline(always)]
    pub fn max_fast(self, other: Self) -> Self {
        Self(T::max_fast(self.1, self.0, other.0), self.1)
    }
}

#[cfg(feature = "w512")]
impl<T: F32x16Backend> f32x16<T> {
    /// Fast single-instruction lane-wise minimum. See [`f32x4::min_fast`].
    #[inline(always)]
    pub fn min_fast(self, other: Self) -> Self {
        Self(T::min_fast(self.1, self.0, other.0), self.1)
    }

    /// Fast single-instruction lane-wise maximum. See [`f32x4::max_fast`].
    #[inline(always)]
    pub fn max_fast(self, other: Self) -> Self {
        Self(T::max_fast(self.1, self.0, other.0), self.1)
    }
}

impl<T: F64x2Backend> f64x2<T> {
    /// Fast single-instruction lane-wise minimum. See [`f32x4::min_fast`].
    #[inline(always)]
    pub fn min_fast(self, other: Self) -> Self {
        Self(T::min_fast(self.1, self.0, other.0), self.1)
    }

    /// Fast single-instruction lane-wise maximum. See [`f32x4::max_fast`].
    #[inline(always)]
    pub fn max_fast(self, other: Self) -> Self {
        Self(T::max_fast(self.1, self.0, other.0), self.1)
    }
}

impl<T: F64x4Backend> f64x4<T> {
    /// Fast single-instruction lane-wise minimum. See [`f32x4::min_fast`].
    #[inline(always)]
    pub fn min_fast(self, other: Self) -> Self {
        Self(T::min_fast(self.1, self.0, other.0), self.1)
    }

    /// Fast single-instruction lane-wise maximum. See [`f32x4::max_fast`].
    #[inline(always)]
    pub fn max_fast(self, other: Self) -> Self {
        Self(T::max_fast(self.1, self.0, other.0), self.1)
    }
}

#[cfg(feature = "w512")]
impl<T: F64x8Backend> f64x8<T> {
    /// Fast single-instruction lane-wise minimum. See [`f32x4::min_fast`].
    #[inline(always)]
    pub fn min_fast(self, other: Self) -> Self {
        Self(T::min_fast(self.1, self.0, other.0), self.1)
    }

    /// Fast single-instruction lane-wise maximum. See [`f32x4::max_fast`].
    #[inline(always)]
    pub fn max_fast(self, other: Self) -> Self {
        Self(T::max_fast(self.1, self.0, other.0), self.1)
    }
}

// ============================================================================
// to_i32_fast
// ============================================================================

impl<T: F32x4Convert> f32x4<T> {
    /// Fast single-instruction truncation toward zero to `i32x4<T>` without
    /// out-of-range/NaN saturation fixups.
    ///
    /// Exact for finite values in `-2_147_483_520.0..=2_147_483_520.0` (every `f32`
    /// whose truncated integer fits in `i32`, plus `-2_147_483_648.0`). Values
    /// outside `i32` range or `NaN` return an unspecified `i32` (`i32::MIN` on x86,
    /// saturated/zero on NEON/WASM/Scalar, relaxed on `Wasm128RelaxedFmaToken`) —
    /// never UB.
    #[inline(always)]
    pub fn to_i32_fast(self) -> i32x4<T> {
        i32x4::from_repr_unchecked(self.1, T::convert_f32_to_i32_fast(self.1, self.0))
    }
}

impl<T: F32x8Convert> f32x8<T> {
    /// Fast single-instruction truncation toward zero to `i32x8<T>`.
    /// See [`f32x4::to_i32_fast`].
    #[inline(always)]
    pub fn to_i32_fast(self) -> i32x8<T> {
        i32x8::from_repr_unchecked(self.1, T::convert_f32_to_i32_fast(self.1, self.0))
    }
}

#[cfg(feature = "w512")]
impl<T: F32x16Convert> f32x16<T> {
    /// Fast single-instruction truncation toward zero to `i32x16<T>`.
    /// See [`f32x4::to_i32_fast`].
    #[inline(always)]
    pub fn to_i32_fast(self) -> i32x16<T> {
        i32x16::from_repr_unchecked(self.1, T::convert_f32_to_i32_fast(self.1, self.0))
    }
}

// ============================================================================
// F32x{4,8,16}ToU32 & to_u32_fast
// ============================================================================

/// Fast in-bounds `f32x4<T> -> u32x4<T>` truncation toward zero.
///
/// Sealed trait implemented by every magetypes backend token that supports
/// [`F32x4Backend`] and [`U32x4Backend`].
pub trait F32x4ToU32: F32x4Backend + U32x4Backend + SimdToken + Sealed + Copy + 'static {
    /// Truncate finite in-range `f32` lanes (`0.0..=4_294_967_040.0`) to `u32`.
    fn convert_f32_to_u32_fast(
        self,
        a: <Self as F32x4Backend>::Repr,
    ) -> <Self as U32x4Backend>::Repr;
}

/// Fast in-bounds `f32x8<T> -> u32x8<T>` truncation toward zero.
///
/// Sealed trait implemented by every magetypes backend token that supports
/// [`F32x8Backend`] and [`U32x8Backend`].
pub trait F32x8ToU32: F32x8Backend + U32x8Backend + SimdToken + Sealed + Copy + 'static {
    /// Truncate finite in-range `f32` lanes (`0.0..=4_294_967_040.0`) to `u32`.
    fn convert_f32_to_u32_fast(
        self,
        a: <Self as F32x8Backend>::Repr,
    ) -> <Self as U32x8Backend>::Repr;
}

/// Fast in-bounds `f32x16<T> -> u32x16<T>` truncation toward zero.
///
/// Sealed trait implemented by every magetypes backend token that supports
/// [`F32x16Backend`] and [`U32x16Backend`].
#[cfg(feature = "w512")]
pub trait F32x16ToU32: F32x16Backend + U32x16Backend + SimdToken + Sealed + Copy + 'static {
    /// Truncate finite in-range `f32` lanes (`0.0..=4_294_967_040.0`) to `u32`.
    fn convert_f32_to_u32_fast(
        self,
        a: <Self as F32x16Backend>::Repr,
    ) -> <Self as U32x16Backend>::Repr;
}

impl F32x4ToU32 for archmage::ScalarToken {
    #[inline(always)]
    fn convert_f32_to_u32_fast(self, a: [f32; 4]) -> [u32; 4] {
        let _ = self;
        core::array::from_fn(|i| a[i] as u32)
    }
}

impl F32x8ToU32 for archmage::ScalarToken {
    #[inline(always)]
    fn convert_f32_to_u32_fast(self, a: [f32; 8]) -> [u32; 8] {
        let _ = self;
        core::array::from_fn(|i| a[i] as u32)
    }
}

#[cfg(feature = "w512")]
impl F32x16ToU32 for archmage::ScalarToken {
    #[inline(always)]
    fn convert_f32_to_u32_fast(self, a: [f32; 16]) -> [u32; 16] {
        let _ = self;
        core::array::from_fn(|i| a[i] as u32)
    }
}

#[cfg(target_arch = "aarch64")]
impl F32x4ToU32 for archmage::NeonToken {
    #[archmage::arcane(suppress_const_test, _self = archmage::NeonToken)]
    fn convert_f32_to_u32_fast(
        self,
        a: core::arch::aarch64::float32x4_t,
    ) -> core::arch::aarch64::uint32x4_t {
        core::arch::aarch64::vcvtq_u32_f32(a)
    }
}

#[cfg(target_arch = "aarch64")]
impl F32x8ToU32 for archmage::NeonToken {
    #[inline(always)]
    fn convert_f32_to_u32_fast(
        self,
        a: [core::arch::aarch64::float32x4_t; 2],
    ) -> [core::arch::aarch64::uint32x4_t; 2] {
        [
            <Self as F32x4ToU32>::convert_f32_to_u32_fast(self, a[0]),
            <Self as F32x4ToU32>::convert_f32_to_u32_fast(self, a[1]),
        ]
    }
}

#[cfg(all(target_arch = "aarch64", feature = "w512"))]
impl F32x16ToU32 for archmage::NeonToken {
    #[inline(always)]
    fn convert_f32_to_u32_fast(
        self,
        a: [core::arch::aarch64::float32x4_t; 4],
    ) -> [core::arch::aarch64::uint32x4_t; 4] {
        core::array::from_fn(|i| <Self as F32x4ToU32>::convert_f32_to_u32_fast(self, a[i]))
    }
}

#[cfg(target_arch = "wasm32")]
impl F32x4ToU32 for archmage::Wasm128Token {
    #[inline(always)]
    fn convert_f32_to_u32_fast(self, a: core::arch::wasm32::v128) -> core::arch::wasm32::v128 {
        let _ = self;
        core::arch::wasm32::u32x4_trunc_sat_f32x4(a)
    }
}

#[cfg(target_arch = "wasm32")]
impl F32x8ToU32 for archmage::Wasm128Token {
    #[inline(always)]
    fn convert_f32_to_u32_fast(
        self,
        a: [core::arch::wasm32::v128; 2],
    ) -> [core::arch::wasm32::v128; 2] {
        [
            <Self as F32x4ToU32>::convert_f32_to_u32_fast(self, a[0]),
            <Self as F32x4ToU32>::convert_f32_to_u32_fast(self, a[1]),
        ]
    }
}

#[cfg(all(target_arch = "wasm32", feature = "w512"))]
impl F32x16ToU32 for archmage::Wasm128Token {
    #[inline(always)]
    fn convert_f32_to_u32_fast(
        self,
        a: [core::arch::wasm32::v128; 4],
    ) -> [core::arch::wasm32::v128; 4] {
        core::array::from_fn(|i| <Self as F32x4ToU32>::convert_f32_to_u32_fast(self, a[i]))
    }
}

#[cfg(all(
    target_arch = "wasm32",
    target_feature = "simd128",
    target_feature = "relaxed-simd"
))]
impl F32x4ToU32 for archmage::Wasm128RelaxedFmaToken {
    #[archmage::arcane(suppress_const_test, _self = archmage::Wasm128RelaxedFmaToken)]
    fn convert_f32_to_u32_fast(self, a: core::arch::wasm32::v128) -> core::arch::wasm32::v128 {
        core::arch::wasm32::u32x4_relaxed_trunc_f32x4(a)
    }
}

#[cfg(all(
    target_arch = "wasm32",
    target_feature = "simd128",
    target_feature = "relaxed-simd"
))]
impl F32x8ToU32 for archmage::Wasm128RelaxedFmaToken {
    #[inline(always)]
    fn convert_f32_to_u32_fast(
        self,
        a: [core::arch::wasm32::v128; 2],
    ) -> [core::arch::wasm32::v128; 2] {
        [
            <Self as F32x4ToU32>::convert_f32_to_u32_fast(self, a[0]),
            <Self as F32x4ToU32>::convert_f32_to_u32_fast(self, a[1]),
        ]
    }
}

#[cfg(all(
    target_arch = "wasm32",
    target_feature = "simd128",
    target_feature = "relaxed-simd",
    feature = "w512"
))]
impl F32x16ToU32 for archmage::Wasm128RelaxedFmaToken {
    #[inline(always)]
    fn convert_f32_to_u32_fast(
        self,
        a: [core::arch::wasm32::v128; 4],
    ) -> [core::arch::wasm32::v128; 4] {
        core::array::from_fn(|i| <Self as F32x4ToU32>::convert_f32_to_u32_fast(self, a[i]))
    }
}

#[cfg(target_arch = "x86_64")]
impl F32x4ToU32 for archmage::X64V3Token {
    #[archmage::arcane(suppress_const_test, _self = archmage::X64V3Token)]
    fn convert_f32_to_u32_fast(self, a: core::arch::x86_64::__m128) -> core::arch::x86_64::__m128i {
        use core::arch::x86_64::{
            _CMP_GE_OQ, _mm_and_ps, _mm_castps_si128, _mm_cmp_ps, _mm_cvttps_epi32, _mm_set1_ps,
            _mm_slli_epi32, _mm_sub_ps, _mm_xor_si128,
        };
        let two31 = _mm_set1_ps(2_147_483_648.0);
        let ge = _mm_cmp_ps::<_CMP_GE_OQ>(a, two31);
        let adj = _mm_sub_ps(a, _mm_and_ps(ge, two31));
        _mm_xor_si128(
            _mm_cvttps_epi32(adj),
            _mm_slli_epi32::<31>(_mm_castps_si128(ge)),
        )
    }
}

#[cfg(target_arch = "x86_64")]
impl F32x8ToU32 for archmage::X64V3Token {
    #[archmage::arcane(suppress_const_test, _self = archmage::X64V3Token)]
    fn convert_f32_to_u32_fast(self, a: core::arch::x86_64::__m256) -> core::arch::x86_64::__m256i {
        use core::arch::x86_64::{
            _CMP_GE_OQ, _mm256_and_ps, _mm256_castps_si256, _mm256_cmp_ps, _mm256_cvttps_epi32,
            _mm256_set1_ps, _mm256_slli_epi32, _mm256_sub_ps, _mm256_xor_si256,
        };
        let two31 = _mm256_set1_ps(2_147_483_648.0);
        let ge = _mm256_cmp_ps::<_CMP_GE_OQ>(a, two31);
        let adj = _mm256_sub_ps(a, _mm256_and_ps(ge, two31));
        _mm256_xor_si256(
            _mm256_cvttps_epi32(adj),
            _mm256_slli_epi32::<31>(_mm256_castps_si256(ge)),
        )
    }
}

#[cfg(all(target_arch = "x86_64", feature = "w512"))]
impl F32x16ToU32 for archmage::X64V3Token {
    #[inline(always)]
    fn convert_f32_to_u32_fast(
        self,
        a: [core::arch::x86_64::__m256; 2],
    ) -> [core::arch::x86_64::__m256i; 2] {
        [
            <Self as F32x8ToU32>::convert_f32_to_u32_fast(self, a[0]),
            <Self as F32x8ToU32>::convert_f32_to_u32_fast(self, a[1]),
        ]
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl F32x4ToU32 for archmage::X64V4Token {
    #[archmage::arcane(suppress_const_test, _self = archmage::X64V4Token)]
    fn convert_f32_to_u32_fast(self, a: core::arch::x86_64::__m128) -> core::arch::x86_64::__m128i {
        core::arch::x86_64::_mm_cvttps_epu32(a)
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl F32x4ToU32 for archmage::X64V4xToken {
    #[inline(always)]
    fn convert_f32_to_u32_fast(self, a: core::arch::x86_64::__m128) -> core::arch::x86_64::__m128i {
        <archmage::X64V4Token as F32x4ToU32>::convert_f32_to_u32_fast(self.v4(), a)
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl F32x4ToU32 for archmage::Avx512Fp16Token {
    #[inline(always)]
    fn convert_f32_to_u32_fast(self, a: core::arch::x86_64::__m128) -> core::arch::x86_64::__m128i {
        <archmage::X64V4Token as F32x4ToU32>::convert_f32_to_u32_fast(self.v4(), a)
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl F32x8ToU32 for archmage::X64V4Token {
    #[archmage::arcane(suppress_const_test, _self = archmage::X64V4Token)]
    fn convert_f32_to_u32_fast(self, a: core::arch::x86_64::__m256) -> core::arch::x86_64::__m256i {
        core::arch::x86_64::_mm256_cvttps_epu32(a)
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl F32x8ToU32 for archmage::X64V4xToken {
    #[inline(always)]
    fn convert_f32_to_u32_fast(self, a: core::arch::x86_64::__m256) -> core::arch::x86_64::__m256i {
        <archmage::X64V4Token as F32x8ToU32>::convert_f32_to_u32_fast(self.v4(), a)
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl F32x8ToU32 for archmage::Avx512Fp16Token {
    #[inline(always)]
    fn convert_f32_to_u32_fast(self, a: core::arch::x86_64::__m256) -> core::arch::x86_64::__m256i {
        <archmage::X64V4Token as F32x8ToU32>::convert_f32_to_u32_fast(self.v4(), a)
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512", feature = "w512"))]
impl F32x16ToU32 for archmage::X64V4Token {
    #[archmage::arcane(suppress_const_test, _self = archmage::X64V4Token)]
    fn convert_f32_to_u32_fast(self, a: core::arch::x86_64::__m512) -> core::arch::x86_64::__m512i {
        core::arch::x86_64::_mm512_cvttps_epu32(a)
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512", feature = "w512"))]
impl F32x16ToU32 for archmage::X64V4xToken {
    #[inline(always)]
    fn convert_f32_to_u32_fast(self, a: core::arch::x86_64::__m512) -> core::arch::x86_64::__m512i {
        <archmage::X64V4Token as F32x16ToU32>::convert_f32_to_u32_fast(self.v4(), a)
    }
}

impl<T: F32x4ToU32> f32x4<T> {
    /// Fast truncation toward zero to `u32x4<T>` for finite in-range values
    /// (`0.0..=4_294_967_040.0`).
    ///
    /// Emits a single instruction on AVX-512 (`vcvttps2udq`), ARM64 NEON (`fcvtzu`),
    /// WASM (`u32x4.trunc_sat_f32x4_u`), and `Wasm128RelaxedFmaToken`
    /// (`u32x4.relaxed_trunc_f32x4_u`), and an exact `2^31` range-split on AVX2.
    /// Negative, out-of-range (`>= 2^32`), and `NaN` inputs produce an unspecified
    /// `u32` — never UB.
    #[inline(always)]
    pub fn to_u32_fast(self) -> u32x4<T> {
        u32x4::from_repr_unchecked(self.1, T::convert_f32_to_u32_fast(self.1, self.0))
    }
}

impl<T: F32x8ToU32> f32x8<T> {
    /// Fast truncation toward zero to `u32x8<T>` for finite in-range values
    /// (`0.0..=4_294_967_040.0`). See [`f32x4::to_u32_fast`].
    #[inline(always)]
    pub fn to_u32_fast(self) -> u32x8<T> {
        u32x8::from_repr_unchecked(self.1, T::convert_f32_to_u32_fast(self.1, self.0))
    }
}

#[cfg(feature = "w512")]
impl<T: F32x16ToU32> f32x16<T> {
    /// Fast truncation toward zero to `u32x16<T>` for finite in-range values
    /// (`0.0..=4_294_967_040.0`). See [`f32x4::to_u32_fast`].
    #[inline(always)]
    pub fn to_u32_fast(self) -> u32x16<T> {
        u32x16::from_repr_unchecked(self.1, T::convert_f32_to_u32_fast(self.1, self.0))
    }
}

//! Fast single-instruction float min/max and in-bounds float-to-integer truncation.
//!
//! ## Fast pseudo-min / pseudo-max (`min_fast`, `max_fast`)
//!
//! Portable [`f32x4::min`] / [`f32x4::max`] follow IEEE-754 `minNum`/`maxNum` or
//! WebAssembly `f32x4.min`/`f32x4.max` (NaN-propagating), which on WebAssembly
//! SIMD128 costs multiple instructions compared to `f32x4.pmin` / `f32x4.pmax`,
//! and on Scalar uses `f32::min` / `f32::max`.
//!
//! `min_fast` and `max_fast` emit a single instruction on every hardware SIMD tier:
//! - **x86 (`X64V3Token`, `X64V4Token`, `X64V4xToken`, `Avx512Fp16Token`):** `vminps` / `vmaxps` (`a < b ? a : b` / `a > b ? a : b`; returns `b` when either input is `NaN`).
//! - **ARM64 (`NeonToken`):** `fminnm` / `fmaxnm` (`vminnmq_f32` / `vmaxnmq_f32`).
//! - **WASM (`Wasm128Token`):** `f32x4.pmin(b, a)` / `f32x4.pmax(b, a)` (matching `vminps(a, b)` / `vmaxps(a, b)` so `x.max_fast(zero)` clamps `NaN` to `0.0` when `zero` is the second operand).
//! - **Scalar (`ScalarToken`):** branchless `if a < b { a } else { b }` / `if a > b { a } else { b }`.
//!
//! For finite non-zero inputs (and finite inputs where `-0.0` vs `+0.0` sign does not
//! matter), all backends return identical results. On `NaN` or `(-0.0, +0.0)` ties,
//! behavior is backend-defined.
//!
//! ## Fast float-to-integer truncation (`to_i32_fast`)
//!
//! [`f32x4::to_i32`] (`to_i32_saturating` on x86) guarantees uniform saturation across
//! ISAs by emitting extra compare+blend instructions around `vcvttps2dq` on x86. When
//! inputs are already clamped or known finite and in-bounds (quantization after ReLU6 /
//! `clamp`, LUT index generation after range reduction, pixel packing), `to_i32_fast`
//! emits the native truncation instruction directly without overflow/NaN fixup masks
//! (`vcvttps2dq` on x86, `vcvtq_s32_f32` on NEON, `i32x4.trunc_sat_f32x4_s` on strict
//! WASM or `i32x4.relaxed_trunc_f32x4_s` when compiled with `relaxed-simd`).

#![forbid(unsafe_code)]

use crate::simd::backends::{F32x4Backend, F32x4Convert, F32x8Backend, F32x8Convert};
#[cfg(feature = "w512")]
use crate::simd::backends::{F32x16Backend, F32x16Convert};
use crate::simd::generic::{f32x4, f32x8, i32x4, i32x8};
#[cfg(feature = "w512")]
use crate::simd::generic::{f32x16, i32x16};

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
    /// saturated/zero on NEON/strict WASM/Scalar, engine-defined with `relaxed-simd`)
    /// — never UB.
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

//! IEEE 754-2019 float min/max scalar helpers, token-matched scalar-tail
//! helpers, and unchecked float-to-integer truncation (`to_i32_fast`).
//!
//! ## Three-tier float min / max semantics
//!
//! Every float vector type (`f32x4`, `f32x8`, `f32x16`, `f64x2`, `f64x4`, `f64x8`)
//! provides three explicit semantics with matching scalar helpers so SIMD chunk
//! loops and scalar tails agree:
//!
//! - [`f32x4::min`] / [`f32x4::max`]: Matches [`f32::min`] / [`f32::max`] (`f64::min` /
//!   `f64::max`): if one operand is `NaN` and the other is numeric (including `±inf`),
//!   returns the numeric operand; signed-zero (`±0.0`) tie-breaking is unspecified.
//!   Scalar tails use `a.min(b)` / `a.max(b)` or `Self::min_scalar` / `Self::max_scalar`.
//! - [`f32x4::minimum`] / [`f32x4::maximum`]: IEEE 754-2019 conformant: propagates `NaN`
//!   (returning quiet `NaN`) and strictly orders `-0.0 < +0.0`. Emits 1 instruction
//!   on NEON (`vminq`/`vmaxq`) and WASM (`f32x4.min`/`f32x4.max`), and exact compare +
//!   bitwise select on x86 and Scalar. Scalar tails use [`minimum_f32`] /
//!   [`maximum_f32`] / [`minimum_f64`] / [`maximum_f64`] or `Self::minimum_scalar` /
//!   `Self::maximum_scalar`.
//! - [`f32x4::min_fast`] / [`f32x4::max_fast`]: Fastest single-instruction lane-wise
//!   minimum/maximum on every platform, ignoring `NaN` and signed-zero ordering
//!   (`vminps`/`vminpd` on x86, `fminnm` on NEON, `pmin(b, a)` or `relaxed_min` on
//!   WASM, `if a < b { a } else { b }` on `ScalarToken`). Scalar tails must use
//!   `Self::min_fast_scalar(token, a, b)` / `Self::max_fast_scalar(token, a, b)`
//!   to match the active token's platform/engine behavior across edge cases.
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

use crate::simd::backends::{
    F32x4Backend, F32x4Convert, F32x8Backend, F32x8Convert, F64x2Backend, F64x4Backend,
};
#[cfg(feature = "w512")]
use crate::simd::backends::{F32x16Backend, F32x16Convert, F64x8Backend};
use crate::simd::generic::{f32x4, f32x8, f64x2, f64x4, i32x4, i32x8};
#[cfg(feature = "w512")]
use crate::simd::generic::{f32x16, f64x8, i32x16};

// ============================================================================
// IEEE 754-2019 scalar functions (for SIMD tails matching minimum/maximum)
// ============================================================================

/// IEEE 754-2019 `minimum` for `f32`: propagates `NaN` (returning quiet `NaN`)
/// and strictly orders `-0.0 < +0.0`.
///
/// Matches [`f32x4::minimum`], [`f32x8::minimum`], and `f32x16::minimum`.
#[inline(always)]
pub const fn minimum_f32(a: f32, b: f32) -> f32 {
    crate::nostd_math::minimumf(a, b)
}

/// IEEE 754-2019 `maximum` for `f32`: propagates `NaN` (returning quiet `NaN`)
/// and strictly orders `-0.0 < +0.0`.
///
/// Matches [`f32x4::maximum`], [`f32x8::maximum`], and `f32x16::maximum`.
#[inline(always)]
pub const fn maximum_f32(a: f32, b: f32) -> f32 {
    crate::nostd_math::maximumf(a, b)
}

/// IEEE 754-2019 `minimum` for `f64`: propagates `NaN` (returning quiet `NaN`)
/// and strictly orders `-0.0 < +0.0`.
///
/// Matches [`f64x2::minimum`], [`f64x4::minimum`], and `f64x8::minimum`.
#[inline(always)]
pub const fn minimum_f64(a: f64, b: f64) -> f64 {
    crate::nostd_math::minimum(a, b)
}

/// IEEE 754-2019 `maximum` for `f64`: propagates `NaN` (returning quiet `NaN`)
/// and strictly orders `-0.0 < +0.0`.
///
/// Matches [`f64x2::maximum`], [`f64x4::maximum`], and `f64x8::maximum`.
#[inline(always)]
pub const fn maximum_f64(a: f64, b: f64) -> f64 {
    crate::nostd_math::maximum(a, b)
}

// ============================================================================
// Token-matched scalar tail helpers on generic float vector types
// ============================================================================

macro_rules! impl_float_scalar_ops {
    ($vec:ident, $backend:ident, $elem:ty, $ieee_min:ident, $ieee_max:ident) => {
        impl<T: $backend> $vec<T> {
            /// Scalar equivalent of [`Self::min`] (`f32::min` / `f64::min` semantics).
            #[inline(always)]
            pub fn min_scalar(token: T, a: $elem, b: $elem) -> $elem {
                Self::splat_t(token, a)
                    .min(Self::splat_t(token, b))
                    .to_array()[0]
            }

            /// Scalar equivalent of [`Self::max`] (`f32::max` / `f64::max` semantics).
            #[inline(always)]
            pub fn max_scalar(token: T, a: $elem, b: $elem) -> $elem {
                Self::splat_t(token, a)
                    .max(Self::splat_t(token, b))
                    .to_array()[0]
            }

            /// Scalar equivalent of [`Self::minimum`] (IEEE 754-2019 `NaN`-propagating, `-0.0 < +0.0`).
            #[inline(always)]
            pub fn minimum_scalar(_token: T, a: $elem, b: $elem) -> $elem {
                $ieee_min(a, b)
            }

            /// Scalar equivalent of [`Self::maximum`] (IEEE 754-2019 `NaN`-propagating, `-0.0 < +0.0`).
            #[inline(always)]
            pub fn maximum_scalar(_token: T, a: $elem, b: $elem) -> $elem {
                $ieee_max(a, b)
            }

            /// Token-matched scalar equivalent of [`Self::min_fast`].
            ///
            /// Executes the active token's `min_fast` operation so SIMD tails match
            /// the vector loop bit-for-bit across architectures and WASM engines.
            #[inline(always)]
            pub fn min_fast_scalar(token: T, a: $elem, b: $elem) -> $elem {
                Self::splat_t(token, a)
                    .min_fast(Self::splat_t(token, b))
                    .to_array()[0]
            }

            /// Token-matched scalar equivalent of [`Self::max_fast`].
            ///
            /// Executes the active token's `max_fast` operation so SIMD tails match
            /// the vector loop bit-for-bit across architectures and WASM engines.
            #[inline(always)]
            pub fn max_fast_scalar(token: T, a: $elem, b: $elem) -> $elem {
                Self::splat_t(token, a)
                    .max_fast(Self::splat_t(token, b))
                    .to_array()[0]
            }
        }
    };
}

impl_float_scalar_ops!(f32x4, F32x4Backend, f32, minimum_f32, maximum_f32);
impl_float_scalar_ops!(f32x8, F32x8Backend, f32, minimum_f32, maximum_f32);
#[cfg(feature = "w512")]
impl_float_scalar_ops!(f32x16, F32x16Backend, f32, minimum_f32, maximum_f32);

impl_float_scalar_ops!(f64x2, F64x2Backend, f64, minimum_f64, maximum_f64);
impl_float_scalar_ops!(f64x4, F64x4Backend, f64, minimum_f64, maximum_f64);
#[cfg(feature = "w512")]
impl_float_scalar_ops!(f64x8, F64x8Backend, f64, minimum_f64, maximum_f64);

macro_rules! impl_scalar1_ops {
    ($vec:ty, $elem:ty, $ieee_min:ident, $ieee_max:ident) => {
        impl $vec {
            /// Scalar equivalent of [`Self::min`].
            #[inline(always)]
            pub fn min_scalar(_token: archmage::ScalarToken, a: $elem, b: $elem) -> $elem {
                a.min(b)
            }

            /// Scalar equivalent of [`Self::max`].
            #[inline(always)]
            pub fn max_scalar(_token: archmage::ScalarToken, a: $elem, b: $elem) -> $elem {
                a.max(b)
            }

            /// Scalar equivalent of [`Self::minimum`] (IEEE 754-2019).
            #[inline(always)]
            pub fn minimum_scalar(_token: archmage::ScalarToken, a: $elem, b: $elem) -> $elem {
                $ieee_min(a, b)
            }

            /// Scalar equivalent of [`Self::maximum`] (IEEE 754-2019).
            #[inline(always)]
            pub fn maximum_scalar(_token: archmage::ScalarToken, a: $elem, b: $elem) -> $elem {
                $ieee_max(a, b)
            }

            /// Scalar equivalent of [`Self::min_fast`].
            #[inline(always)]
            pub fn min_fast_scalar(_token: archmage::ScalarToken, a: $elem, b: $elem) -> $elem {
                if a < b { a } else { b }
            }

            /// Scalar equivalent of [`Self::max_fast`].
            #[inline(always)]
            pub fn max_fast_scalar(_token: archmage::ScalarToken, a: $elem, b: $elem) -> $elem {
                if a > b { a } else { b }
            }
        }
    };
}

impl_scalar1_ops!(crate::simd::scalar::f32x1, f32, minimum_f32, maximum_f32);
impl_scalar1_ops!(crate::simd::scalar::f64x1, f64, minimum_f64, maximum_f64);

impl crate::simd::scalar::f32x1 {
    /// Scalar equivalent of [`Self::to_i32_fast`].
    #[inline(always)]
    pub fn to_i32_fast_scalar(_token: archmage::ScalarToken, a: f32) -> i32 {
        a as i32
    }
}

// ============================================================================
// to_i32_fast / to_i32_fast_scalar
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

    /// Token-matched scalar equivalent of [`Self::to_i32_fast`].
    #[inline(always)]
    pub fn to_i32_fast_scalar(token: T, a: f32) -> i32 {
        Self::splat_t(token, a).to_i32_fast().to_array()[0]
    }
}

impl<T: F32x8Convert> f32x8<T> {
    /// Fast single-instruction truncation toward zero to `i32x8<T>`.
    /// See [`f32x4::to_i32_fast`].
    #[inline(always)]
    pub fn to_i32_fast(self) -> i32x8<T> {
        i32x8::from_repr_unchecked(self.1, T::convert_f32_to_i32_fast(self.1, self.0))
    }

    /// Token-matched scalar equivalent of [`Self::to_i32_fast`].
    #[inline(always)]
    pub fn to_i32_fast_scalar(token: T, a: f32) -> i32 {
        Self::splat_t(token, a).to_i32_fast().to_array()[0]
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

    /// Token-matched scalar equivalent of [`Self::to_i32_fast`].
    #[inline(always)]
    pub fn to_i32_fast_scalar(token: T, a: f32) -> i32 {
        Self::splat_t(token, a).to_i32_fast().to_array()[0]
    }
}

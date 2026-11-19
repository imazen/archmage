//! Scalar helpers matching [`crate::simd::generic`] vector semantics for SIMD
//! remainder loops and non-SIMD fallback paths.
//!
//! ## What you do NOT need a helper for
//!
//! Plain `core` Rust already compiles in `#![no_std]` (MSRV 1.89) and matches
//! `magetypes` portable vector semantics bit-for-bit:
//!
//! - Arithmetic: `a + b`, `a - b`, `a * b`, `a / b`, `-a`
//! - Magnitude / sign: `x.abs()`, `x.copysign(y)`
//! - Rust numeric min/max (`1.0` over a single quiet `NaN`): `a.min(b)`,
//!   `a.max(b)`. Signalling `NaN` handling is unspecified in both: NEON vectors
//!   return a quiet `NaN`, the other backends the numeric operand.
//!
//! `x.clamp(lo, hi)` is **not** vector `clamp`: `f32::clamp` keeps a `NaN` input
//! and panics on `lo > hi` or `NaN` bounds, while vector `clamp` is
//! `max_fast` then `min_fast` and turns a quiet `NaN` lane into `lo` on x86,
//! NEON, strict WASM, and `ScalarToken` (signalling `NaN` and WASM
//! `relaxed-simd` `NaN` lanes are unspecified per backend). Use
//! [`clamp_f32`] / [`clamp_f64`].
//! - Saturating float-to-integer (`to_i32_saturating`): `x as i32`
//!
//! ## What this module provides
//!
//! Only the operations that `core` 1.89 omits or where a vector operation is
//! intentionally backend-dependent:
//!
//! 1. **Portable `#![no_std]` helpers** (concrete, non-generic — zero token
//!    argument and zero monomorphization cost):
//!    - IEEE 754-2019 `NaN`-propagating min/max (`-0.0 < +0.0`): [`minimum_f32`],
//!      [`maximum_f32`], [`minimum_f64`], [`maximum_f64`]
//!    - Rounding (ties-to-even, matching `v.round()`) and `sqrt`: [`round_f32`],
//!      [`floor_f32`], [`ceil_f32`], [`sqrt_f32`], [`round_f64`], [`floor_f64`],
//!      [`ceil_f64`], [`sqrt_f64`]
//!    - Single-rounding FMA (`v.mul_add_fused` / `v.mul_sub_fused`):
//!      [`mul_add_fused_f32`], [`mul_sub_fused_f32`], [`mul_add_fused_f64`],
//!      [`mul_sub_fused_f64`]
//! 2. **Token-matched scalar primitives** (1 instruction on native tokens;
//!    supertraits of `F32x*Backend` / `F64x*Backend`, so generic `<T: F32x8Backend>`
//!    kernels call them with no extra trait bounds):
//!    - Fast platform-native min/max (`NaN` / `±0.0` unspecified): [`min_fast_f32`],
//!      [`max_fast_f32`], [`min_fast_f64`], [`max_fast_f64`]
//!    - Vector `clamp` (`max_fast` then `min_fast`): [`clamp_f32`], [`clamp_f64`]
//!    - Relaxed FMA (`v.mul_add_relaxed` / `v.mul_sub_relaxed`):
//!      [`mul_add_relaxed_f32`], [`mul_sub_relaxed_f32`], [`mul_add_relaxed_f64`],
//!      [`mul_sub_relaxed_f64`]
//!    - Native / unchecked float-to-integer truncation (`v.to_i32` / `v.to_i32_fast`):
//!      [`to_i32`], [`to_i32_fast`]
//! 3. **Compile-time policy-matched helpers** for vector remainders. Use the
//!    same [`crate::numerics`] policy bits in both paths.
//!    - Truncation and multiply-negate-add: [`trunc_f32`], [`trunc_f64`],
//!      [`mul_neg_add_fused_f32`], [`mul_neg_add_fused_f64`],
//!      [`mul_neg_add_relaxed_f32`], [`mul_neg_add_relaxed_f64`].
//!    - Integer and half conversions: [`from_u32`], [`from_bf16`],
//!      [`from_f16_with`], [`to_i32_with`], [`to_u32_with`], [`to_u16_with`],
//!      [`to_u8_with`], [`to_f16_bits_with`], [`to_bf16_with`].
//!    - Reciprocals: [`recip_f32_with`], [`recip_f64_with`],
//!      [`rsqrt_f32_with`], [`rsqrt_f64_with`].
//!    - Extrema: [`min_f32_with`], [`max_f32_with`], [`clamp_f32_with`],
//!      [`min_f64_with`], [`max_f64_with`], [`clamp_f64_with`].

#![forbid(unsafe_code)]

use crate::simd::backends::sealed::Sealed;
use archmage::{IntoConcreteToken, SimdToken};

// ============================================================================
// Portable no_std scalar helpers (zero token argument, non-generic)
// ============================================================================

/// IEEE 754-2019 `minimum` for `f32`: propagates `NaN` (returning quiet `NaN`)
/// and strictly orders `-0.0 < +0.0`. The vector forms return a quiet `NaN`
/// whose sign and payload differ by backend; this returns the canonical one.
///
/// Matches [`f32x4::minimum`](crate::simd::generic::f32x4::minimum),
/// [`f32x8::minimum`](crate::simd::generic::f32x8::minimum), and
/// `f32x16::minimum`.
#[inline(always)]
pub const fn minimum_f32(a: f32, b: f32) -> f32 {
    crate::nostd_math::minimumf(a, b)
}

/// IEEE 754-2019 `maximum` for `f32`: propagates `NaN` (returning quiet `NaN`)
/// and strictly orders `-0.0 < +0.0`. The vector forms return a quiet `NaN`
/// whose sign and payload differ by backend; this returns the canonical one.
///
/// Matches [`f32x4::maximum`](crate::simd::generic::f32x4::maximum),
/// [`f32x8::maximum`](crate::simd::generic::f32x8::maximum), and
/// `f32x16::maximum`.
#[inline(always)]
pub const fn maximum_f32(a: f32, b: f32) -> f32 {
    crate::nostd_math::maximumf(a, b)
}

/// IEEE 754-2019 `minimum` for `f64`: propagates `NaN` (returning quiet `NaN`)
/// and strictly orders `-0.0 < +0.0`. The vector forms return a quiet `NaN`
/// whose sign and payload differ by backend; this returns the canonical one.
///
/// Matches [`f64x2::minimum`](crate::simd::generic::f64x2::minimum),
/// [`f64x4::minimum`](crate::simd::generic::f64x4::minimum), and
/// `f64x8::minimum`.
#[inline(always)]
pub const fn minimum_f64(a: f64, b: f64) -> f64 {
    crate::nostd_math::minimum(a, b)
}

/// IEEE 754-2019 `maximum` for `f64`: propagates `NaN` (returning quiet `NaN`)
/// and strictly orders `-0.0 < +0.0`. The vector forms return a quiet `NaN`
/// whose sign and payload differ by backend; this returns the canonical one.
///
/// Matches [`f64x2::maximum`](crate::simd::generic::f64x2::maximum),
/// [`f64x4::maximum`](crate::simd::generic::f64x4::maximum), and
/// `f64x8::maximum`.
#[inline(always)]
pub const fn maximum_f64(a: f64, b: f64) -> f64 {
    crate::nostd_math::maximum(a, b)
}

/// Round `f32` to the nearest integer with ties to even (`#![no_std]`).
///
/// Matches `f32xN::round()` (unlike `f32::round()`, which breaks ties away from zero).
#[inline(always)]
pub fn round_f32(x: f32) -> f32 {
    crate::nostd_math::roundevenf(x)
}

/// Round `f32` toward `-inf` (`#![no_std]`), matching `f32xN::floor()`.
#[inline(always)]
pub fn floor_f32(x: f32) -> f32 {
    crate::nostd_math::floorf(x)
}

/// Round `f32` toward `+inf` (`#![no_std]`), matching `f32xN::ceil()`.
#[inline(always)]
pub fn ceil_f32(x: f32) -> f32 {
    crate::nostd_math::ceilf(x)
}

/// Square root of `f32` (`#![no_std]`), matching `f32xN::sqrt()`.
#[inline(always)]
pub fn sqrt_f32(x: f32) -> f32 {
    crate::nostd_math::sqrtf(x)
}

/// Round `f64` to the nearest integer with ties to even (`#![no_std]`).
///
/// Matches `f64xN::round()` (unlike `f64::round()`, which breaks ties away from zero).
#[inline(always)]
pub fn round_f64(x: f64) -> f64 {
    crate::nostd_math::roundeven(x)
}

/// Round `f64` toward `-inf` (`#![no_std]`), matching `f64xN::floor()`.
#[inline(always)]
pub fn floor_f64(x: f64) -> f64 {
    crate::nostd_math::floor(x)
}

/// Round `f64` toward `+inf` (`#![no_std]`), matching `f64xN::ceil()`.
#[inline(always)]
pub fn ceil_f64(x: f64) -> f64 {
    crate::nostd_math::ceil(x)
}

/// Square root of `f64` (`#![no_std]`), matching `f64xN::sqrt()`.
#[inline(always)]
pub fn sqrt_f64(x: f64) -> f64 {
    crate::nostd_math::sqrt(x)
}

/// Correctly rounded single-rounding `a * b + c` (`#![no_std]`), matching
/// `f32xN::mul_add_fused` / `mul_add_portable` on every backend.
#[inline(always)]
pub fn mul_add_fused_f32(a: f32, b: f32, c: f32) -> f32 {
    crate::soft_fma::fmaf_lanes([a], [b], [c])[0]
}

/// Correctly rounded single-rounding `a * b - c` (`#![no_std]`), matching
/// `f32xN::mul_sub_fused` / `mul_sub_portable` on every backend.
#[inline(always)]
pub fn mul_sub_fused_f32(a: f32, b: f32, c: f32) -> f32 {
    crate::soft_fma::fmaf_lanes([a], [b], [-c])[0]
}

/// Correctly rounded single-rounding `a * b + c` (`#![no_std]`), matching
/// `f64xN::mul_add_fused` / `mul_add_portable` on every backend.
#[inline(always)]
pub fn mul_add_fused_f64(a: f64, b: f64, c: f64) -> f64 {
    crate::soft_fma::fma_lanes([a], [b], [c])[0]
}

/// Correctly rounded single-rounding `a * b - c` (`#![no_std]`), matching
/// `f64xN::mul_sub_fused` / `mul_sub_portable` on every backend.
#[inline(always)]
pub fn mul_sub_fused_f64(a: f64, b: f64, c: f64) -> f64 {
    crate::soft_fma::fma_lanes([a], [b], [-c])[0]
}

// ============================================================================
// Token-matched backend-dependent scalar traits & functions
// ============================================================================

/// Token-gated scalar operations for backend-dependent `f32` primitives.
///
/// Sealed — implemented only for `archmage` SIMD tokens, and required as a
/// supertrait of `F32x4Backend`, `F32x8Backend`, and `F32x16Backend` so generic
/// kernels can call [`min_fast_f32`], [`max_fast_f32`], [`clamp_f32`],
/// [`mul_add_relaxed_f32`], [`mul_sub_relaxed_f32`], [`to_i32`], and
/// [`to_i32_fast`] directly in their scalar tails.
pub trait F32ScalarBackend: IntoConcreteToken + SimdToken + Sealed + Copy + 'static {
    /// Fast scalar minimum matching `f32xN::<Self>::min_fast`.
    #[doc(hidden)]
    fn min_fast_f32(self, a: f32, b: f32) -> f32;

    /// Fast scalar maximum matching `f32xN::<Self>::max_fast`.
    #[doc(hidden)]
    fn max_fast_f32(self, a: f32, b: f32) -> f32;

    /// Relaxed scalar multiply-add `a * b + c` matching `f32xN::<Self>::mul_add_relaxed`.
    #[doc(hidden)]
    fn mul_add_relaxed_f32(self, a: f32, b: f32, c: f32) -> f32;

    /// Relaxed scalar multiply-sub `a * b - c` matching `f32xN::<Self>::mul_sub_relaxed`.
    #[doc(hidden)]
    fn mul_sub_relaxed_f32(self, a: f32, b: f32, c: f32) -> f32;

    /// Scalar truncation matching `f32xN::<Self>::to_i32`.
    #[doc(hidden)]
    fn to_i32_scalar(self, a: f32) -> i32;

    /// Fast unchecked scalar truncation matching `f32xN::<Self>::to_i32_fast`.
    #[doc(hidden)]
    fn to_i32_fast_scalar(self, a: f32) -> i32;
}

/// Token-gated scalar operations for backend-dependent `f64` primitives.
///
/// Sealed — implemented only for `archmage` SIMD tokens, and required as a
/// supertrait of `F64x2Backend`, `F64x4Backend`, and `F64x8Backend` so generic
/// kernels can call [`min_fast_f64`], [`max_fast_f64`], [`clamp_f64`],
/// [`mul_add_relaxed_f64`], and [`mul_sub_relaxed_f64`] directly in their
/// scalar tails.
pub trait F64ScalarBackend: IntoConcreteToken + SimdToken + Sealed + Copy + 'static {
    /// Fast scalar minimum matching `f64xN::<Self>::min_fast`.
    #[doc(hidden)]
    fn min_fast_f64(self, a: f64, b: f64) -> f64;

    /// Fast scalar maximum matching `f64xN::<Self>::max_fast`.
    #[doc(hidden)]
    fn max_fast_f64(self, a: f64, b: f64) -> f64;

    /// Relaxed scalar multiply-add `a * b + c` matching `f64xN::<Self>::mul_add_relaxed`.
    #[doc(hidden)]
    fn mul_add_relaxed_f64(self, a: f64, b: f64, c: f64) -> f64;

    /// Relaxed scalar multiply-sub `a * b - c` matching `f64xN::<Self>::mul_sub_relaxed`.
    #[doc(hidden)]
    fn mul_sub_relaxed_f64(self, a: f64, b: f64, c: f64) -> f64;
}

impl F32ScalarBackend for archmage::ScalarToken {
    #[inline(always)]
    fn min_fast_f32(self, a: f32, b: f32) -> f32 {
        if a < b { a } else { b }
    }

    #[inline(always)]
    fn max_fast_f32(self, a: f32, b: f32) -> f32 {
        if a > b { a } else { b }
    }

    #[inline(always)]
    fn mul_add_relaxed_f32(self, a: f32, b: f32, c: f32) -> f32 {
        a * b + c
    }

    #[inline(always)]
    fn mul_sub_relaxed_f32(self, a: f32, b: f32, c: f32) -> f32 {
        a * b - c
    }

    #[inline(always)]
    fn to_i32_scalar(self, a: f32) -> i32 {
        a as i32
    }

    #[inline(always)]
    fn to_i32_fast_scalar(self, a: f32) -> i32 {
        a as i32
    }
}

impl F64ScalarBackend for archmage::ScalarToken {
    #[inline(always)]
    fn min_fast_f64(self, a: f64, b: f64) -> f64 {
        if a < b { a } else { b }
    }

    #[inline(always)]
    fn max_fast_f64(self, a: f64, b: f64) -> f64 {
        if a > b { a } else { b }
    }

    #[inline(always)]
    fn mul_add_relaxed_f64(self, a: f64, b: f64, c: f64) -> f64 {
        a * b + c
    }

    #[inline(always)]
    fn mul_sub_relaxed_f64(self, a: f64, b: f64, c: f64) -> f64 {
        a * b - c
    }
}

#[cfg(target_arch = "x86_64")]
mod x86_hw {
    use core::arch::x86_64::*;

    #[archmage::arcane]
    pub(super) fn min_ss(_token: archmage::X64V3Token, a: f32, b: f32) -> f32 {
        _mm_cvtss_f32(_mm_min_ss(_mm_set_ss(a), _mm_set_ss(b)))
    }

    #[archmage::arcane]
    pub(super) fn max_ss(_token: archmage::X64V3Token, a: f32, b: f32) -> f32 {
        _mm_cvtss_f32(_mm_max_ss(_mm_set_ss(a), _mm_set_ss(b)))
    }

    #[archmage::arcane]
    pub(super) fn fmadd_ss(_token: archmage::X64V3Token, a: f32, b: f32, c: f32) -> f32 {
        _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(a), _mm_set_ss(b), _mm_set_ss(c)))
    }

    #[archmage::arcane]
    pub(super) fn fmsub_ss(_token: archmage::X64V3Token, a: f32, b: f32, c: f32) -> f32 {
        _mm_cvtss_f32(_mm_fmsub_ss(_mm_set_ss(a), _mm_set_ss(b), _mm_set_ss(c)))
    }

    #[archmage::arcane]
    pub(super) fn cvttss_si32(_token: archmage::X64V3Token, a: f32) -> i32 {
        _mm_cvttss_si32(_mm_set_ss(a))
    }

    #[archmage::arcane]
    pub(super) fn min_sd(_token: archmage::X64V3Token, a: f64, b: f64) -> f64 {
        _mm_cvtsd_f64(_mm_min_sd(_mm_set_sd(a), _mm_set_sd(b)))
    }

    #[archmage::arcane]
    pub(super) fn max_sd(_token: archmage::X64V3Token, a: f64, b: f64) -> f64 {
        _mm_cvtsd_f64(_mm_max_sd(_mm_set_sd(a), _mm_set_sd(b)))
    }

    #[archmage::arcane]
    pub(super) fn fmadd_sd(_token: archmage::X64V3Token, a: f64, b: f64, c: f64) -> f64 {
        _mm_cvtsd_f64(_mm_fmadd_sd(_mm_set_sd(a), _mm_set_sd(b), _mm_set_sd(c)))
    }

    #[archmage::arcane]
    pub(super) fn fmsub_sd(_token: archmage::X64V3Token, a: f64, b: f64, c: f64) -> f64 {
        _mm_cvtsd_f64(_mm_fmsub_sd(_mm_set_sd(a), _mm_set_sd(b), _mm_set_sd(c)))
    }
}

#[cfg(target_arch = "x86_64")]
impl F32ScalarBackend for archmage::X64V3Token {
    #[inline(always)]
    fn min_fast_f32(self, a: f32, b: f32) -> f32 {
        x86_hw::min_ss(self, a, b)
    }

    #[inline(always)]
    fn max_fast_f32(self, a: f32, b: f32) -> f32 {
        x86_hw::max_ss(self, a, b)
    }

    #[inline(always)]
    fn mul_add_relaxed_f32(self, a: f32, b: f32, c: f32) -> f32 {
        x86_hw::fmadd_ss(self, a, b, c)
    }

    #[inline(always)]
    fn mul_sub_relaxed_f32(self, a: f32, b: f32, c: f32) -> f32 {
        x86_hw::fmsub_ss(self, a, b, c)
    }

    #[inline(always)]
    fn to_i32_scalar(self, a: f32) -> i32 {
        x86_hw::cvttss_si32(self, a)
    }

    #[inline(always)]
    fn to_i32_fast_scalar(self, a: f32) -> i32 {
        x86_hw::cvttss_si32(self, a)
    }
}

#[cfg(target_arch = "x86_64")]
impl F64ScalarBackend for archmage::X64V3Token {
    #[inline(always)]
    fn min_fast_f64(self, a: f64, b: f64) -> f64 {
        x86_hw::min_sd(self, a, b)
    }

    #[inline(always)]
    fn max_fast_f64(self, a: f64, b: f64) -> f64 {
        x86_hw::max_sd(self, a, b)
    }

    #[inline(always)]
    fn mul_add_relaxed_f64(self, a: f64, b: f64, c: f64) -> f64 {
        x86_hw::fmadd_sd(self, a, b, c)
    }

    #[inline(always)]
    fn mul_sub_relaxed_f64(self, a: f64, b: f64, c: f64) -> f64 {
        x86_hw::fmsub_sd(self, a, b, c)
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
macro_rules! impl_x86_v3_scalar_delegate {
    ($($token:ty),+ $(,)?) => {
        $(
            impl F32ScalarBackend for $token {
                #[inline(always)]
                fn min_fast_f32(self, a: f32, b: f32) -> f32 {
                    self.v3().min_fast_f32(a, b)
                }

                #[inline(always)]
                fn max_fast_f32(self, a: f32, b: f32) -> f32 {
                    self.v3().max_fast_f32(a, b)
                }

                #[inline(always)]
                fn mul_add_relaxed_f32(self, a: f32, b: f32, c: f32) -> f32 {
                    self.v3().mul_add_relaxed_f32(a, b, c)
                }

                #[inline(always)]
                fn mul_sub_relaxed_f32(self, a: f32, b: f32, c: f32) -> f32 {
                    self.v3().mul_sub_relaxed_f32(a, b, c)
                }

                #[inline(always)]
                fn to_i32_scalar(self, a: f32) -> i32 {
                    self.v3().to_i32_scalar(a)
                }

                #[inline(always)]
                fn to_i32_fast_scalar(self, a: f32) -> i32 {
                    self.v3().to_i32_fast_scalar(a)
                }
            }

            impl F64ScalarBackend for $token {
                #[inline(always)]
                fn min_fast_f64(self, a: f64, b: f64) -> f64 {
                    self.v3().min_fast_f64(a, b)
                }

                #[inline(always)]
                fn max_fast_f64(self, a: f64, b: f64) -> f64 {
                    self.v3().max_fast_f64(a, b)
                }

                #[inline(always)]
                fn mul_add_relaxed_f64(self, a: f64, b: f64, c: f64) -> f64 {
                    self.v3().mul_add_relaxed_f64(a, b, c)
                }

                #[inline(always)]
                fn mul_sub_relaxed_f64(self, a: f64, b: f64, c: f64) -> f64 {
                    self.v3().mul_sub_relaxed_f64(a, b, c)
                }
            }
        )+
    };
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_x86_v3_scalar_delegate!(
    archmage::X64V4Token,
    archmage::X64V4xToken,
    archmage::Avx512Fp16Token
);

#[cfg(target_arch = "aarch64")]
mod arm_hw {
    use core::arch::aarch64::*;

    #[archmage::arcane]
    pub(super) fn minnm_f32(_token: archmage::NeonToken, a: f32, b: f32) -> f32 {
        vget_lane_f32::<0>(vminnm_f32(vdup_n_f32(a), vdup_n_f32(b)))
    }

    #[archmage::arcane]
    pub(super) fn maxnm_f32(_token: archmage::NeonToken, a: f32, b: f32) -> f32 {
        vget_lane_f32::<0>(vmaxnm_f32(vdup_n_f32(a), vdup_n_f32(b)))
    }

    #[archmage::arcane]
    pub(super) fn fma_f32(_token: archmage::NeonToken, a: f32, b: f32, c: f32) -> f32 {
        vget_lane_f32::<0>(vfma_f32(vdup_n_f32(c), vdup_n_f32(a), vdup_n_f32(b)))
    }

    #[archmage::arcane]
    pub(super) fn minnm_f64(_token: archmage::NeonToken, a: f64, b: f64) -> f64 {
        vget_lane_f64::<0>(vminnm_f64(vdup_n_f64(a), vdup_n_f64(b)))
    }

    #[archmage::arcane]
    pub(super) fn maxnm_f64(_token: archmage::NeonToken, a: f64, b: f64) -> f64 {
        vget_lane_f64::<0>(vmaxnm_f64(vdup_n_f64(a), vdup_n_f64(b)))
    }

    #[archmage::arcane]
    pub(super) fn fma_f64(_token: archmage::NeonToken, a: f64, b: f64, c: f64) -> f64 {
        vget_lane_f64::<0>(vfma_f64(vdup_n_f64(c), vdup_n_f64(a), vdup_n_f64(b)))
    }
}

#[cfg(target_arch = "aarch64")]
impl F32ScalarBackend for archmage::NeonToken {
    #[inline(always)]
    fn min_fast_f32(self, a: f32, b: f32) -> f32 {
        arm_hw::minnm_f32(self, a, b)
    }

    #[inline(always)]
    fn max_fast_f32(self, a: f32, b: f32) -> f32 {
        arm_hw::maxnm_f32(self, a, b)
    }

    #[inline(always)]
    fn mul_add_relaxed_f32(self, a: f32, b: f32, c: f32) -> f32 {
        arm_hw::fma_f32(self, a, b, c)
    }

    #[inline(always)]
    fn mul_sub_relaxed_f32(self, a: f32, b: f32, c: f32) -> f32 {
        arm_hw::fma_f32(self, a, b, -c)
    }

    #[inline(always)]
    fn to_i32_scalar(self, a: f32) -> i32 {
        a as i32
    }

    #[inline(always)]
    fn to_i32_fast_scalar(self, a: f32) -> i32 {
        a as i32
    }
}

#[cfg(target_arch = "aarch64")]
impl F64ScalarBackend for archmage::NeonToken {
    #[inline(always)]
    fn min_fast_f64(self, a: f64, b: f64) -> f64 {
        arm_hw::minnm_f64(self, a, b)
    }

    #[inline(always)]
    fn max_fast_f64(self, a: f64, b: f64) -> f64 {
        arm_hw::maxnm_f64(self, a, b)
    }

    #[inline(always)]
    fn mul_add_relaxed_f64(self, a: f64, b: f64, c: f64) -> f64 {
        arm_hw::fma_f64(self, a, b, c)
    }

    #[inline(always)]
    fn mul_sub_relaxed_f64(self, a: f64, b: f64, c: f64) -> f64 {
        arm_hw::fma_f64(self, a, b, -c)
    }
}

#[cfg(target_arch = "wasm32")]
impl F32ScalarBackend for archmage::Wasm128Token {
    #[inline(always)]
    fn min_fast_f32(self, a: f32, b: f32) -> f32 {
        #[cfg(target_feature = "relaxed-simd")]
        {
            crate::wasm_fma::min_f32_fast_scalar(self, a, b)
        }
        #[cfg(not(target_feature = "relaxed-simd"))]
        {
            // Matches f32x4.pmin(b, a): returns `a` if `a < b`, else `b`.
            if a < b { a } else { b }
        }
    }

    #[inline(always)]
    fn max_fast_f32(self, a: f32, b: f32) -> f32 {
        #[cfg(target_feature = "relaxed-simd")]
        {
            crate::wasm_fma::max_f32_fast_scalar(self, a, b)
        }
        #[cfg(not(target_feature = "relaxed-simd"))]
        {
            // Matches f32x4.pmax(b, a): returns `a` if `a > b`, else `b`.
            if a > b { a } else { b }
        }
    }

    #[inline(always)]
    fn mul_add_relaxed_f32(self, a: f32, b: f32, c: f32) -> f32 {
        #[cfg(target_feature = "relaxed-simd")]
        {
            crate::wasm_fma::madd_f32_scalar(self, a, b, c)
        }
        #[cfg(not(target_feature = "relaxed-simd"))]
        {
            a * b + c
        }
    }

    #[inline(always)]
    fn mul_sub_relaxed_f32(self, a: f32, b: f32, c: f32) -> f32 {
        #[cfg(target_feature = "relaxed-simd")]
        {
            crate::wasm_fma::msub_f32_scalar(self, a, b, c)
        }
        #[cfg(not(target_feature = "relaxed-simd"))]
        {
            a * b - c
        }
    }

    #[inline(always)]
    fn to_i32_scalar(self, a: f32) -> i32 {
        a as i32
    }

    #[inline(always)]
    fn to_i32_fast_scalar(self, a: f32) -> i32 {
        #[cfg(target_feature = "relaxed-simd")]
        {
            crate::wasm_fma::trunc_f32_fast_scalar(self, a)
        }
        #[cfg(not(target_feature = "relaxed-simd"))]
        {
            a as i32
        }
    }
}

#[cfg(target_arch = "wasm32")]
impl F64ScalarBackend for archmage::Wasm128Token {
    #[inline(always)]
    fn min_fast_f64(self, a: f64, b: f64) -> f64 {
        #[cfg(target_feature = "relaxed-simd")]
        {
            crate::wasm_fma::min_f64_fast_scalar(self, a, b)
        }
        #[cfg(not(target_feature = "relaxed-simd"))]
        {
            if a < b { a } else { b }
        }
    }

    #[inline(always)]
    fn max_fast_f64(self, a: f64, b: f64) -> f64 {
        #[cfg(target_feature = "relaxed-simd")]
        {
            crate::wasm_fma::max_f64_fast_scalar(self, a, b)
        }
        #[cfg(not(target_feature = "relaxed-simd"))]
        {
            if a > b { a } else { b }
        }
    }

    #[inline(always)]
    fn mul_add_relaxed_f64(self, a: f64, b: f64, c: f64) -> f64 {
        #[cfg(target_feature = "relaxed-simd")]
        {
            crate::wasm_fma::madd_f64_scalar(self, a, b, c)
        }
        #[cfg(not(target_feature = "relaxed-simd"))]
        {
            a * b + c
        }
    }

    #[inline(always)]
    fn mul_sub_relaxed_f64(self, a: f64, b: f64, c: f64) -> f64 {
        #[cfg(target_feature = "relaxed-simd")]
        {
            crate::wasm_fma::msub_f64_scalar(self, a, b, c)
        }
        #[cfg(not(target_feature = "relaxed-simd"))]
        {
            a * b - c
        }
    }
}

/// Token-matched fast `f32` minimum (`NaN` and `±0.0` behavior matches the active
/// token's `min_fast`).
#[inline(always)]
pub fn min_fast_f32<T: F32ScalarBackend>(token: T, a: f32, b: f32) -> f32 {
    token.min_fast_f32(a, b)
}

/// Token-matched fast `f32` maximum (`NaN` and `±0.0` behavior matches the active
/// token's `max_fast`).
#[inline(always)]
pub fn max_fast_f32<T: F32ScalarBackend>(token: T, a: f32, b: f32) -> f32 {
    token.max_fast_f32(a, b)
}

/// Token-matched `f32` clamp, matching the active token's `f32xN::clamp`:
/// `min_fast_f32(token, max_fast_f32(token, x, lo), hi)`.
///
/// Exact when `lo <= hi` and nothing is `NaN`. A quiet `NaN` `x` becomes `lo`
/// on x86, NEON, strict WASM, and `ScalarToken`; signalling `NaN`, WASM
/// `relaxed-simd` `NaN` lanes, `NaN` bounds, and `lo > hi` are unspecified per
/// backend. Unlike `f32::clamp` it never panics.
#[inline(always)]
pub fn clamp_f32<T: F32ScalarBackend>(token: T, x: f32, lo: f32, hi: f32) -> f32 {
    token.min_fast_f32(token.max_fast_f32(x, lo), hi)
}

/// Token-matched relaxed `f32` multiply-add `a * b + c`, matching the active
/// token's `mul_add_relaxed`.
#[inline(always)]
pub fn mul_add_relaxed_f32<T: F32ScalarBackend>(token: T, a: f32, b: f32, c: f32) -> f32 {
    token.mul_add_relaxed_f32(a, b, c)
}

/// Token-matched relaxed `f32` multiply-sub `a * b - c`, matching the active
/// token's `mul_sub_relaxed`.
#[inline(always)]
pub fn mul_sub_relaxed_f32<T: F32ScalarBackend>(token: T, a: f32, b: f32, c: f32) -> f32 {
    token.mul_sub_relaxed_f32(a, b, c)
}

/// Token-matched scalar `f32` truncation toward zero, matching the active
/// token's `f32xN::to_i32` (returns `i32::MIN` on x86 for overflow/`NaN`,
/// saturates on NEON/WASM/`ScalarToken`).
#[inline(always)]
pub fn to_i32<T: F32ScalarBackend>(token: T, a: f32) -> i32 {
    token.to_i32_scalar(a)
}

/// Token-matched fast unchecked `f32` truncation toward zero, matching the
/// active token's `f32xN::to_i32_fast` (including `relaxed-simd` on WASM).
#[inline(always)]
pub fn to_i32_fast<T: F32ScalarBackend>(token: T, a: f32) -> i32 {
    token.to_i32_fast_scalar(a)
}

/// Token-matched fast `f64` minimum (`NaN` and `±0.0` behavior matches the active
/// token's `min_fast`).
#[inline(always)]
pub fn min_fast_f64<T: F64ScalarBackend>(token: T, a: f64, b: f64) -> f64 {
    token.min_fast_f64(a, b)
}

/// Token-matched fast `f64` maximum (`NaN` and `±0.0` behavior matches the active
/// token's `max_fast`).
#[inline(always)]
pub fn max_fast_f64<T: F64ScalarBackend>(token: T, a: f64, b: f64) -> f64 {
    token.max_fast_f64(a, b)
}

/// Token-matched `f64` clamp, matching the active token's `f64xN::clamp`:
/// `min_fast_f64(token, max_fast_f64(token, x, lo), hi)`.
///
/// Exact when `lo <= hi` and nothing is `NaN`. A quiet `NaN` `x` becomes `lo`
/// on x86, NEON, strict WASM, and `ScalarToken`; signalling `NaN`, WASM
/// `relaxed-simd` `NaN` lanes, `NaN` bounds, and `lo > hi` are unspecified per
/// backend. Unlike `f64::clamp` it never panics.
#[inline(always)]
pub fn clamp_f64<T: F64ScalarBackend>(token: T, x: f64, lo: f64, hi: f64) -> f64 {
    token.min_fast_f64(token.max_fast_f64(x, lo), hi)
}

/// Token-matched relaxed `f64` multiply-add `a * b + c`, matching the active
/// token's `mul_add_relaxed`.
#[inline(always)]
pub fn mul_add_relaxed_f64<T: F64ScalarBackend>(token: T, a: f64, b: f64, c: f64) -> f64 {
    token.mul_add_relaxed_f64(a, b, c)
}

/// Token-matched relaxed `f64` multiply-sub `a * b - c`, matching the active
/// token's `mul_sub_relaxed`.
#[inline(always)]
pub fn mul_sub_relaxed_f64<T: F64ScalarBackend>(token: T, a: f64, b: f64, c: f64) -> f64 {
    token.mul_sub_relaxed_f64(a, b, c)
}

#[path = "scalar_numerics.rs"]
mod numerics;
pub use numerics::*;

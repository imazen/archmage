//! Conversion traits between float and integer SIMD backends.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

use super::F32x4Backend;
use super::F32x8Backend;
#[cfg(feature = "w512")]
use super::F32x16Backend;
use super::F64x2Backend;
use super::F64x4Backend;
#[cfg(feature = "w512")]
use super::F64x8Backend;
use super::I32x4Backend;
use super::I32x8Backend;
#[cfg(feature = "w512")]
use super::I32x16Backend;
use super::I64x2Backend;
use super::I64x4Backend;
use super::U32x4Backend;
use super::U32x8Backend;
use super::sealed::Sealed;
use archmage::SimdToken;

/// Conversions between f32x4 and i32x4 representations.
///
/// Requires both `F32x4Backend` and `I32x4Backend` to be implemented.
pub trait F32x4Convert: F32x4Backend + I32x4Backend + SimdToken + Sealed + Copy + 'static {
    /// Bitcast f32x4 to i32x4 (reinterpret bits, no conversion).
    fn bitcast_f32_to_i32(self, a: <Self as F32x4Backend>::Repr) -> <Self as I32x4Backend>::Repr;

    /// Bitcast i32x4 to f32x4 (reinterpret bits, no conversion).
    fn bitcast_i32_to_f32(self, a: <Self as I32x4Backend>::Repr) -> <Self as F32x4Backend>::Repr;

    /// Convert f32x4 to i32x4 with truncation toward zero.
    fn convert_f32_to_i32(self, a: <Self as F32x4Backend>::Repr) -> <Self as I32x4Backend>::Repr;

    /// Convert with truncation and UNIFORM saturation: out-of-range
    /// lanes clamp to `i32::MIN`/`i32::MAX`, NaN lanes become 0 —
    /// identical on every backend (Rust `as` / NEON FCVTZS / WASM
    /// trunc_sat semantics). Default delegates to
    /// [`Self::convert_f32_to_i32`] (already conformant on NEON/WASM/
    /// scalar); x86 overrides with a cvttps + compare/blend fixup
    /// (the bare op yields the `i32::MIN` sentinel for overflow AND
    /// NaN there — issue #80).
    #[inline(always)]
    fn convert_f32_to_i32_saturating(
        self,
        a: <Self as F32x4Backend>::Repr,
    ) -> <Self as I32x4Backend>::Repr {
        <Self as F32x4Convert>::convert_f32_to_i32(self, a)
    }

    /// Fast single-instruction truncation toward zero for finite in-range
    /// values (`-2147483520.0..=2147483520.0`). Out-of-range and NaN lanes
    /// produce an unspecified `i32`. Defaults to [`convert_f32_to_i32`](Self::convert_f32_to_i32);
    /// `Wasm128RelaxedFmaToken` overrides with `i32x4.relaxed_trunc_f32x4_s`.
    #[inline(always)]
    fn convert_f32_to_i32_fast(
        self,
        a: <Self as F32x4Backend>::Repr,
    ) -> <Self as I32x4Backend>::Repr {
        <Self as F32x4Convert>::convert_f32_to_i32(self, a)
    }

    /// Convert f32x4 to i32x4 with rounding to nearest.
    fn convert_f32_to_i32_round(
        self,
        a: <Self as F32x4Backend>::Repr,
    ) -> <Self as I32x4Backend>::Repr;

    /// Convert i32x4 to f32x4.
    fn convert_i32_to_f32(self, a: <Self as I32x4Backend>::Repr) -> <Self as F32x4Backend>::Repr;
}

/// Conversions between f32x8 and i32x8 representations.
///
/// Requires both `F32x8Backend` and `I32x8Backend` to be implemented.
pub trait F32x8Convert: F32x8Backend + I32x8Backend + SimdToken + Sealed + Copy + 'static {
    /// Bitcast f32x8 to i32x8 (reinterpret bits, no conversion).
    fn bitcast_f32_to_i32(self, a: <Self as F32x8Backend>::Repr) -> <Self as I32x8Backend>::Repr;

    /// Bitcast i32x8 to f32x8 (reinterpret bits, no conversion).
    fn bitcast_i32_to_f32(self, a: <Self as I32x8Backend>::Repr) -> <Self as F32x8Backend>::Repr;

    /// Convert f32x8 to i32x8 with truncation toward zero.
    fn convert_f32_to_i32(self, a: <Self as F32x8Backend>::Repr) -> <Self as I32x8Backend>::Repr;

    /// Convert with truncation and UNIFORM saturation: out-of-range
    /// lanes clamp to `i32::MIN`/`i32::MAX`, NaN lanes become 0 —
    /// identical on every backend (Rust `as` / NEON FCVTZS / WASM
    /// trunc_sat semantics). Default delegates to
    /// [`Self::convert_f32_to_i32`] (already conformant on NEON/WASM/
    /// scalar); x86 overrides with a cvttps + compare/blend fixup
    /// (the bare op yields the `i32::MIN` sentinel for overflow AND
    /// NaN there — issue #80).
    #[inline(always)]
    fn convert_f32_to_i32_saturating(
        self,
        a: <Self as F32x8Backend>::Repr,
    ) -> <Self as I32x8Backend>::Repr {
        <Self as F32x8Convert>::convert_f32_to_i32(self, a)
    }

    /// Fast single-instruction truncation toward zero for finite in-range
    /// values (`-2147483520.0..=2147483520.0`). Out-of-range and NaN lanes
    /// produce an unspecified `i32`. Defaults to [`convert_f32_to_i32`](Self::convert_f32_to_i32);
    /// `Wasm128RelaxedFmaToken` overrides with `i32x4.relaxed_trunc_f32x4_s`.
    #[inline(always)]
    fn convert_f32_to_i32_fast(
        self,
        a: <Self as F32x8Backend>::Repr,
    ) -> <Self as I32x8Backend>::Repr {
        <Self as F32x8Convert>::convert_f32_to_i32(self, a)
    }

    /// Convert f32x8 to i32x8 with rounding to nearest.
    fn convert_f32_to_i32_round(
        self,
        a: <Self as F32x8Backend>::Repr,
    ) -> <Self as I32x8Backend>::Repr;

    /// Convert i32x8 to f32x8.
    fn convert_i32_to_f32(self, a: <Self as I32x8Backend>::Repr) -> <Self as F32x8Backend>::Repr;
}

/// Conversions between f32x16 and i32x16 representations.
///
/// Requires both `F32x16Backend` and `I32x16Backend` to be implemented.
#[cfg(feature = "w512")]
pub trait F32x16Convert:
    F32x16Backend + I32x16Backend + SimdToken + Sealed + Copy + 'static
{
    /// Bitcast f32x16 to i32x16 (reinterpret bits, no conversion).
    fn bitcast_f32_to_i32(self, a: <Self as F32x16Backend>::Repr) -> <Self as I32x16Backend>::Repr;

    /// Bitcast i32x16 to f32x16 (reinterpret bits, no conversion).
    fn bitcast_i32_to_f32(self, a: <Self as I32x16Backend>::Repr) -> <Self as F32x16Backend>::Repr;

    /// Convert f32x16 to i32x16 with truncation toward zero.
    fn convert_f32_to_i32(self, a: <Self as F32x16Backend>::Repr) -> <Self as I32x16Backend>::Repr;

    /// Convert with truncation and UNIFORM saturation: out-of-range
    /// lanes clamp to `i32::MIN`/`i32::MAX`, NaN lanes become 0 —
    /// identical on every backend (Rust `as` / NEON FCVTZS / WASM
    /// trunc_sat semantics). Default delegates to
    /// [`Self::convert_f32_to_i32`] (already conformant on NEON/WASM/
    /// scalar); x86 overrides with a cvttps + compare/blend fixup
    /// (the bare op yields the `i32::MIN` sentinel for overflow AND
    /// NaN there — issue #80).
    #[inline(always)]
    fn convert_f32_to_i32_saturating(
        self,
        a: <Self as F32x16Backend>::Repr,
    ) -> <Self as I32x16Backend>::Repr {
        <Self as F32x16Convert>::convert_f32_to_i32(self, a)
    }

    /// Fast single-instruction truncation toward zero for finite in-range
    /// values (`-2147483520.0..=2147483520.0`). Out-of-range and NaN lanes
    /// produce an unspecified `i32`. Defaults to [`convert_f32_to_i32`](Self::convert_f32_to_i32);
    /// `Wasm128RelaxedFmaToken` overrides with `i32x4.relaxed_trunc_f32x4_s`.
    #[inline(always)]
    fn convert_f32_to_i32_fast(
        self,
        a: <Self as F32x16Backend>::Repr,
    ) -> <Self as I32x16Backend>::Repr {
        <Self as F32x16Convert>::convert_f32_to_i32(self, a)
    }

    /// Convert f32x16 to i32x16 with rounding to nearest.
    fn convert_f32_to_i32_round(
        self,
        a: <Self as F32x16Backend>::Repr,
    ) -> <Self as I32x16Backend>::Repr;

    /// Convert i32x16 to f32x16.
    fn convert_i32_to_f32(self, a: <Self as I32x16Backend>::Repr) -> <Self as F32x16Backend>::Repr;
}

/// Bitcast conversions between u32x4 and i32x4 representations.
///
/// Requires both `U32x4Backend` and `I32x4Backend` to be implemented.
pub trait U32x4Bitcast: U32x4Backend + I32x4Backend + SimdToken + Sealed + Copy + 'static {
    /// Bitcast u32x4 to i32x4 (reinterpret bits, no conversion).
    fn bitcast_u32_to_i32(self, a: <Self as U32x4Backend>::Repr) -> <Self as I32x4Backend>::Repr;

    /// Bitcast i32x4 to u32x4 (reinterpret bits, no conversion).
    fn bitcast_i32_to_u32(self, a: <Self as I32x4Backend>::Repr) -> <Self as U32x4Backend>::Repr;
}

/// Bitcast conversions between u32x8 and i32x8 representations.
///
/// Requires both `U32x8Backend` and `I32x8Backend` to be implemented.
pub trait U32x8Bitcast: U32x8Backend + I32x8Backend + SimdToken + Sealed + Copy + 'static {
    /// Bitcast u32x8 to i32x8 (reinterpret bits, no conversion).
    fn bitcast_u32_to_i32(self, a: <Self as U32x8Backend>::Repr) -> <Self as I32x8Backend>::Repr;

    /// Bitcast i32x8 to u32x8 (reinterpret bits, no conversion).
    fn bitcast_i32_to_u32(self, a: <Self as I32x8Backend>::Repr) -> <Self as U32x8Backend>::Repr;
}

/// Bitcast conversions between i64x2 and f64x2 representations.
///
/// Requires both `I64x2Backend` and `F64x2Backend` to be implemented.
pub trait I64x2Bitcast: I64x2Backend + F64x2Backend + SimdToken + Sealed + Copy + 'static {
    /// Bitcast i64x2 to f64x2 (reinterpret bits, no conversion).
    fn bitcast_i64_to_f64(self, a: <Self as I64x2Backend>::Repr) -> <Self as F64x2Backend>::Repr;

    /// Bitcast f64x2 to i64x2 (reinterpret bits, no conversion).
    fn bitcast_f64_to_i64(self, a: <Self as F64x2Backend>::Repr) -> <Self as I64x2Backend>::Repr;
}

/// Bitcast conversions between i64x4 and f64x4 representations.
///
/// Requires both `I64x4Backend` and `F64x4Backend` to be implemented.
pub trait I64x4Bitcast: I64x4Backend + F64x4Backend + SimdToken + Sealed + Copy + 'static {
    /// Bitcast i64x4 to f64x4 (reinterpret bits, no conversion).
    fn bitcast_i64_to_f64(self, a: <Self as I64x4Backend>::Repr) -> <Self as F64x4Backend>::Repr;

    /// Bitcast f64x4 to i64x4 (reinterpret bits, no conversion).
    fn bitcast_f64_to_i64(self, a: <Self as F64x4Backend>::Repr) -> <Self as I64x4Backend>::Repr;
}

/// Conversions between f64x2 and i32x4 representations (uses low 2 i32 lanes).
///
/// Requires both `F64x2Backend` and `I32x4Backend` to be implemented.
pub trait F64x2Convert: F64x2Backend + I32x4Backend + SimdToken + Sealed + Copy + 'static {
    /// Convert f64x2 to the low 2 lanes of i32x4 with truncation toward zero
    /// (lanes 2 and 3 of the result are zero).
    fn convert_f64_to_i32_low(
        self,
        a: <Self as F64x2Backend>::Repr,
    ) -> <Self as I32x4Backend>::Repr;

    /// Convert f64x2 to the low 2 lanes of i32x4 with truncation and UNIFORM
    /// saturation (lanes 2 and 3 are zero): out-of-range lanes clamp to
    /// `i32::MIN`/`i32::MAX` and NaN lanes become 0 — identical on every
    /// backend (Rust `as` / NEON / WASM `trunc_sat_zero` semantics). Default
    /// delegates to [`Self::convert_f64_to_i32_low`]; x86 overrides with a
    /// cvttpd + compare/blend fixup (issue #80).
    #[inline(always)]
    fn convert_f64_to_i32_low_saturating(
        self,
        a: <Self as F64x2Backend>::Repr,
    ) -> <Self as I32x4Backend>::Repr {
        <Self as F64x2Convert>::convert_f64_to_i32_low(self, a)
    }

    /// Convert f64x2 to the low 2 lanes of i32x4 with rounding to nearest
    /// (lanes 2 and 3 of the result are zero).
    fn convert_f64_to_i32_low_round(
        self,
        a: <Self as F64x2Backend>::Repr,
    ) -> <Self as I32x4Backend>::Repr;

    /// Convert the low 2 lanes of i32x4 to f64x2.
    fn convert_i32_low_to_f64(
        self,
        a: <Self as I32x4Backend>::Repr,
    ) -> <Self as F64x2Backend>::Repr;
}

/// Conversions between f64x4 and i32x4 representations.
///
/// Requires both `F64x4Backend` and `I32x4Backend` to be implemented.
pub trait F64x4Convert: F64x4Backend + I32x4Backend + SimdToken + Sealed + Copy + 'static {
    /// Convert f64x4 to i32x4 with truncation toward zero.
    fn convert_f64_to_i32(self, a: <Self as F64x4Backend>::Repr) -> <Self as I32x4Backend>::Repr;

    /// Convert f64x4 to i32x4 with truncation and UNIFORM saturation:
    /// out-of-range lanes clamp to `i32::MIN`/`i32::MAX` and NaN lanes
    /// become 0 — identical on every backend (Rust `as` / NEON / WASM
    /// semantics). Default delegates to [`Self::convert_f64_to_i32`]; x86
    /// overrides with a cvttpd + compare/blend fixup (issue #80).
    #[inline(always)]
    fn convert_f64_to_i32_saturating(
        self,
        a: <Self as F64x4Backend>::Repr,
    ) -> <Self as I32x4Backend>::Repr {
        <Self as F64x4Convert>::convert_f64_to_i32(self, a)
    }

    /// Convert f64x4 to i32x4 with rounding to nearest.
    fn convert_f64_to_i32_round(
        self,
        a: <Self as F64x4Backend>::Repr,
    ) -> <Self as I32x4Backend>::Repr;

    /// Convert i32x4 to f64x4.
    fn convert_i32_to_f64(self, a: <Self as I32x4Backend>::Repr) -> <Self as F64x4Backend>::Repr;
}

/// Conversions between f64x8 and i32x8 representations.
///
/// Requires both `F64x8Backend` and `I32x8Backend` to be implemented.
#[cfg(feature = "w512")]
pub trait F64x8Convert: F64x8Backend + I32x8Backend + SimdToken + Sealed + Copy + 'static {
    /// Convert f64x8 to i32x8 with truncation toward zero.
    fn convert_f64_to_i32(self, a: <Self as F64x8Backend>::Repr) -> <Self as I32x8Backend>::Repr;

    /// Convert f64x8 to i32x8 with truncation and UNIFORM saturation:
    /// out-of-range lanes clamp to `i32::MIN`/`i32::MAX` and NaN lanes
    /// become 0 — identical on every backend (Rust `as` / NEON / WASM
    /// semantics). Default delegates to [`Self::convert_f64_to_i32`]; x86
    /// overrides with a cvttpd + compare/blend fixup (issue #80).
    #[inline(always)]
    fn convert_f64_to_i32_saturating(
        self,
        a: <Self as F64x8Backend>::Repr,
    ) -> <Self as I32x8Backend>::Repr {
        <Self as F64x8Convert>::convert_f64_to_i32(self, a)
    }

    /// Convert f64x8 to i32x8 with rounding to nearest.
    fn convert_f64_to_i32_round(
        self,
        a: <Self as F64x8Backend>::Repr,
    ) -> <Self as I32x8Backend>::Repr;

    /// Convert i32x8 to f64x8.
    fn convert_i32_to_f64(self, a: <Self as I32x8Backend>::Repr) -> <Self as F64x8Backend>::Repr;
}

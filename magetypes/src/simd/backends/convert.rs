//! Conversion traits between float and integer SIMD backends.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

use super::F32x4Backend;
use super::F32x8Backend;
use super::F64x2Backend;
use super::F64x4Backend;
use super::I32x4Backend;
use super::I32x8Backend;
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
    fn bitcast_f32_to_i32(a: <Self as F32x4Backend>::Repr) -> <Self as I32x4Backend>::Repr;

    /// Bitcast i32x4 to f32x4 (reinterpret bits, no conversion).
    fn bitcast_i32_to_f32(a: <Self as I32x4Backend>::Repr) -> <Self as F32x4Backend>::Repr;

    /// Convert f32x4 to i32x4 with truncation toward zero.
    fn convert_f32_to_i32(a: <Self as F32x4Backend>::Repr) -> <Self as I32x4Backend>::Repr;

    /// Convert f32x4 to i32x4 with rounding to nearest.
    fn convert_f32_to_i32_round(a: <Self as F32x4Backend>::Repr) -> <Self as I32x4Backend>::Repr;

    /// Convert i32x4 to f32x4.
    fn convert_i32_to_f32(a: <Self as I32x4Backend>::Repr) -> <Self as F32x4Backend>::Repr;
}

/// Conversions between f32x8 and i32x8 representations.
///
/// Requires both `F32x8Backend` and `I32x8Backend` to be implemented.
pub trait F32x8Convert: F32x8Backend + I32x8Backend + SimdToken + Sealed + Copy + 'static {
    /// Bitcast f32x8 to i32x8 (reinterpret bits, no conversion).
    fn bitcast_f32_to_i32(a: <Self as F32x8Backend>::Repr) -> <Self as I32x8Backend>::Repr;

    /// Bitcast i32x8 to f32x8 (reinterpret bits, no conversion).
    fn bitcast_i32_to_f32(a: <Self as I32x8Backend>::Repr) -> <Self as F32x8Backend>::Repr;

    /// Convert f32x8 to i32x8 with truncation toward zero.
    fn convert_f32_to_i32(a: <Self as F32x8Backend>::Repr) -> <Self as I32x8Backend>::Repr;

    /// Convert f32x8 to i32x8 with rounding to nearest.
    fn convert_f32_to_i32_round(a: <Self as F32x8Backend>::Repr) -> <Self as I32x8Backend>::Repr;

    /// Convert i32x8 to f32x8.
    fn convert_i32_to_f32(a: <Self as I32x8Backend>::Repr) -> <Self as F32x8Backend>::Repr;
}

/// Bitcast conversions between u32x4 and i32x4 representations.
///
/// Requires both `U32x4Backend` and `I32x4Backend` to be implemented.
pub trait U32x4Bitcast: U32x4Backend + I32x4Backend + SimdToken + Sealed + Copy + 'static {
    /// Bitcast u32x4 to i32x4 (reinterpret bits, no conversion).
    fn bitcast_u32_to_i32(a: <Self as U32x4Backend>::Repr) -> <Self as I32x4Backend>::Repr;

    /// Bitcast i32x4 to u32x4 (reinterpret bits, no conversion).
    fn bitcast_i32_to_u32(a: <Self as I32x4Backend>::Repr) -> <Self as U32x4Backend>::Repr;
}

/// Bitcast conversions between u32x8 and i32x8 representations.
///
/// Requires both `U32x8Backend` and `I32x8Backend` to be implemented.
pub trait U32x8Bitcast: U32x8Backend + I32x8Backend + SimdToken + Sealed + Copy + 'static {
    /// Bitcast u32x8 to i32x8 (reinterpret bits, no conversion).
    fn bitcast_u32_to_i32(a: <Self as U32x8Backend>::Repr) -> <Self as I32x8Backend>::Repr;

    /// Bitcast i32x8 to u32x8 (reinterpret bits, no conversion).
    fn bitcast_i32_to_u32(a: <Self as I32x8Backend>::Repr) -> <Self as U32x8Backend>::Repr;
}

/// Bitcast conversions between i64x2 and f64x2 representations.
///
/// Requires both `I64x2Backend` and `F64x2Backend` to be implemented.
pub trait I64x2Bitcast: I64x2Backend + F64x2Backend + SimdToken + Sealed + Copy + 'static {
    /// Bitcast i64x2 to f64x2 (reinterpret bits, no conversion).
    fn bitcast_i64_to_f64(a: <Self as I64x2Backend>::Repr) -> <Self as F64x2Backend>::Repr;

    /// Bitcast f64x2 to i64x2 (reinterpret bits, no conversion).
    fn bitcast_f64_to_i64(a: <Self as F64x2Backend>::Repr) -> <Self as I64x2Backend>::Repr;
}

/// Bitcast conversions between i64x4 and f64x4 representations.
///
/// Requires both `I64x4Backend` and `F64x4Backend` to be implemented.
pub trait I64x4Bitcast: I64x4Backend + F64x4Backend + SimdToken + Sealed + Copy + 'static {
    /// Bitcast i64x4 to f64x4 (reinterpret bits, no conversion).
    fn bitcast_i64_to_f64(a: <Self as I64x4Backend>::Repr) -> <Self as F64x4Backend>::Repr;

    /// Bitcast f64x4 to i64x4 (reinterpret bits, no conversion).
    fn bitcast_f64_to_i64(a: <Self as F64x4Backend>::Repr) -> <Self as I64x4Backend>::Repr;
}

//! Conversion traits between float and integer SIMD backends.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

use super::F32x4Backend;
use super::F32x8Backend;
use super::I32x4Backend;
use super::I32x8Backend;
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

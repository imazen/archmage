//! Bitcast conversion traits for integer types.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

use super::sealed::Sealed;
use archmage::SimdToken;

/// Bitcast conversions between i8x16 and u8x16 representations.
pub trait I8x16Bitcast:
    super::I8x16Backend + super::U8x16Backend + SimdToken + Sealed + Copy + 'static
{
    /// Bitcast i8x16 to u8x16 (reinterpret bits).
    fn bitcast_i8_to_u8(
        a: <Self as super::I8x16Backend>::Repr,
    ) -> <Self as super::U8x16Backend>::Repr;
    /// Bitcast u8x16 to i8x16 (reinterpret bits).
    fn bitcast_u8_to_i8(
        a: <Self as super::U8x16Backend>::Repr,
    ) -> <Self as super::I8x16Backend>::Repr;
}

/// Bitcast conversions between i8x32 and u8x32 representations.
pub trait I8x32Bitcast:
    super::I8x32Backend + super::U8x32Backend + SimdToken + Sealed + Copy + 'static
{
    /// Bitcast i8x32 to u8x32 (reinterpret bits).
    fn bitcast_i8_to_u8(
        a: <Self as super::I8x32Backend>::Repr,
    ) -> <Self as super::U8x32Backend>::Repr;
    /// Bitcast u8x32 to i8x32 (reinterpret bits).
    fn bitcast_u8_to_i8(
        a: <Self as super::U8x32Backend>::Repr,
    ) -> <Self as super::I8x32Backend>::Repr;
}

/// Bitcast conversions between i16x8 and u16x8 representations.
pub trait I16x8Bitcast:
    super::I16x8Backend + super::U16x8Backend + SimdToken + Sealed + Copy + 'static
{
    /// Bitcast i16x8 to u16x8 (reinterpret bits).
    fn bitcast_i16_to_u16(
        a: <Self as super::I16x8Backend>::Repr,
    ) -> <Self as super::U16x8Backend>::Repr;
    /// Bitcast u16x8 to i16x8 (reinterpret bits).
    fn bitcast_u16_to_i16(
        a: <Self as super::U16x8Backend>::Repr,
    ) -> <Self as super::I16x8Backend>::Repr;
}

/// Bitcast conversions between i16x16 and u16x16 representations.
pub trait I16x16Bitcast:
    super::I16x16Backend + super::U16x16Backend + SimdToken + Sealed + Copy + 'static
{
    /// Bitcast i16x16 to u16x16 (reinterpret bits).
    fn bitcast_i16_to_u16(
        a: <Self as super::I16x16Backend>::Repr,
    ) -> <Self as super::U16x16Backend>::Repr;
    /// Bitcast u16x16 to i16x16 (reinterpret bits).
    fn bitcast_u16_to_i16(
        a: <Self as super::U16x16Backend>::Repr,
    ) -> <Self as super::I16x16Backend>::Repr;
}

/// Bitcast conversions between u64x2, i64x2, and f64x2 representations.
pub trait U64x2Bitcast:
    super::U64x2Backend + super::I64x2Backend + SimdToken + Sealed + Copy + 'static
{
    /// Bitcast u64x2 to i64x2 (reinterpret bits).
    fn bitcast_u64_to_i64(
        a: <Self as super::U64x2Backend>::Repr,
    ) -> <Self as super::I64x2Backend>::Repr;
    /// Bitcast i64x2 to u64x2 (reinterpret bits).
    fn bitcast_i64_to_u64(
        a: <Self as super::I64x2Backend>::Repr,
    ) -> <Self as super::U64x2Backend>::Repr;
}

/// Bitcast conversions between u64x4, i64x4, and f64x4 representations.
pub trait U64x4Bitcast:
    super::U64x4Backend + super::I64x4Backend + SimdToken + Sealed + Copy + 'static
{
    /// Bitcast u64x4 to i64x4 (reinterpret bits).
    fn bitcast_u64_to_i64(
        a: <Self as super::U64x4Backend>::Repr,
    ) -> <Self as super::I64x4Backend>::Repr;
    /// Bitcast i64x4 to u64x4 (reinterpret bits).
    fn bitcast_i64_to_u64(
        a: <Self as super::I64x4Backend>::Repr,
    ) -> <Self as super::U64x4Backend>::Repr;
}

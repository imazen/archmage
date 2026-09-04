//! Widening and saturating-narrowing conversion traits for integer types.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

use super::sealed::Sealed;
use archmage::SimdToken;

/// Zero-extending widening from `u8x16` to `u16x8`.
///
/// Native at every tier and in natural lane order everywhere; see
/// `docs/CROSS-ISA-INT-PRIMITIVES.md` §3.
pub trait U8x16Widen:
    super::U8x16Backend + super::U16x8Backend + SimdToken + Sealed + Copy + 'static
{
    /// Zero-extend source lanes `0..8` to `u16`.
    ///
    /// Result lane `i` is `a[i] as u16` for `i` in `0..8`.
    fn widen_low_u8_to_u16(
        self,
        a: <Self as super::U8x16Backend>::Repr,
    ) -> <Self as super::U16x8Backend>::Repr;

    /// Zero-extend source lanes `8..16` to `u16`.
    ///
    /// Result lane `i` is `a[i + 8] as u16`.
    fn widen_high_u8_to_u16(
        self,
        a: <Self as super::U8x16Backend>::Repr,
    ) -> <Self as super::U16x8Backend>::Repr;
}

/// Zero-extending widening from `u16x8` to `u32x4`.
///
/// Native at every tier and in natural lane order everywhere; see
/// `docs/CROSS-ISA-INT-PRIMITIVES.md` §3.
pub trait U16x8Widen:
    super::U16x8Backend + super::U32x4Backend + SimdToken + Sealed + Copy + 'static
{
    /// Zero-extend source lanes `0..4` to `u32`.
    ///
    /// Result lane `i` is `a[i] as u32` for `i` in `0..4`.
    fn widen_low_u16_to_u32(
        self,
        a: <Self as super::U16x8Backend>::Repr,
    ) -> <Self as super::U32x4Backend>::Repr;

    /// Zero-extend source lanes `4..8` to `u32`.
    ///
    /// Result lane `i` is `a[i + 4] as u32`.
    fn widen_high_u16_to_u32(
        self,
        a: <Self as super::U16x8Backend>::Repr,
    ) -> <Self as super::U32x4Backend>::Repr;
}

/// Sign-extending widening from `i8x16` to `i16x8`.
///
/// Native at every tier and in natural lane order everywhere; see
/// `docs/CROSS-ISA-INT-PRIMITIVES.md` §3.
pub trait I8x16Widen:
    super::I8x16Backend + super::I16x8Backend + SimdToken + Sealed + Copy + 'static
{
    /// Sign-extend source lanes `0..8` to `i16`.
    ///
    /// Result lane `i` is `a[i] as i16` for `i` in `0..8`.
    fn widen_low_i8_to_i16(
        self,
        a: <Self as super::I8x16Backend>::Repr,
    ) -> <Self as super::I16x8Backend>::Repr;

    /// Sign-extend source lanes `8..16` to `i16`.
    ///
    /// Result lane `i` is `a[i + 8] as i16`.
    fn widen_high_i8_to_i16(
        self,
        a: <Self as super::I8x16Backend>::Repr,
    ) -> <Self as super::I16x8Backend>::Repr;
}

/// Sign-extending widening from `i16x8` to `i32x4`.
///
/// Native at every tier and in natural lane order everywhere; see
/// `docs/CROSS-ISA-INT-PRIMITIVES.md` §3.
pub trait I16x8Widen:
    super::I16x8Backend + super::I32x4Backend + SimdToken + Sealed + Copy + 'static
{
    /// Sign-extend source lanes `0..4` to `i32`.
    ///
    /// Result lane `i` is `a[i] as i32` for `i` in `0..4`.
    fn widen_low_i16_to_i32(
        self,
        a: <Self as super::I16x8Backend>::Repr,
    ) -> <Self as super::I32x4Backend>::Repr;

    /// Sign-extend source lanes `4..8` to `i32`.
    ///
    /// Result lane `i` is `a[i + 4] as i32`.
    fn widen_high_i16_to_i32(
        self,
        a: <Self as super::I16x8Backend>::Repr,
    ) -> <Self as super::I32x4Backend>::Repr;
}

/// Zero-extending widening from `u8x32` to `u16x16`.
///
/// Native at every tier and in natural lane order everywhere; see
/// `docs/CROSS-ISA-INT-PRIMITIVES.md` §3.
pub trait U8x32Widen:
    super::U8x32Backend + super::U16x16Backend + SimdToken + Sealed + Copy + 'static
{
    /// Zero-extend source lanes `0..16` to `u16`.
    ///
    /// Result lane `i` is `a[i] as u16` for `i` in `0..16`.
    fn widen_low_u8_to_u16(
        self,
        a: <Self as super::U8x32Backend>::Repr,
    ) -> <Self as super::U16x16Backend>::Repr;

    /// Zero-extend source lanes `16..32` to `u16`.
    ///
    /// Result lane `i` is `a[i + 16] as u16`.
    fn widen_high_u8_to_u16(
        self,
        a: <Self as super::U8x32Backend>::Repr,
    ) -> <Self as super::U16x16Backend>::Repr;
}

/// Zero-extending widening from `u16x16` to `u32x8`.
///
/// Native at every tier and in natural lane order everywhere; see
/// `docs/CROSS-ISA-INT-PRIMITIVES.md` §3.
pub trait U16x16Widen:
    super::U16x16Backend + super::U32x8Backend + SimdToken + Sealed + Copy + 'static
{
    /// Zero-extend source lanes `0..8` to `u32`.
    ///
    /// Result lane `i` is `a[i] as u32` for `i` in `0..8`.
    fn widen_low_u16_to_u32(
        self,
        a: <Self as super::U16x16Backend>::Repr,
    ) -> <Self as super::U32x8Backend>::Repr;

    /// Zero-extend source lanes `8..16` to `u32`.
    ///
    /// Result lane `i` is `a[i + 8] as u32`.
    fn widen_high_u16_to_u32(
        self,
        a: <Self as super::U16x16Backend>::Repr,
    ) -> <Self as super::U32x8Backend>::Repr;
}

/// Sign-extending widening from `i8x32` to `i16x16`.
///
/// Native at every tier and in natural lane order everywhere; see
/// `docs/CROSS-ISA-INT-PRIMITIVES.md` §3.
pub trait I8x32Widen:
    super::I8x32Backend + super::I16x16Backend + SimdToken + Sealed + Copy + 'static
{
    /// Sign-extend source lanes `0..16` to `i16`.
    ///
    /// Result lane `i` is `a[i] as i16` for `i` in `0..16`.
    fn widen_low_i8_to_i16(
        self,
        a: <Self as super::I8x32Backend>::Repr,
    ) -> <Self as super::I16x16Backend>::Repr;

    /// Sign-extend source lanes `16..32` to `i16`.
    ///
    /// Result lane `i` is `a[i + 16] as i16`.
    fn widen_high_i8_to_i16(
        self,
        a: <Self as super::I8x32Backend>::Repr,
    ) -> <Self as super::I16x16Backend>::Repr;
}

/// Sign-extending widening from `i16x16` to `i32x8`.
///
/// Native at every tier and in natural lane order everywhere; see
/// `docs/CROSS-ISA-INT-PRIMITIVES.md` §3.
pub trait I16x16Widen:
    super::I16x16Backend + super::I32x8Backend + SimdToken + Sealed + Copy + 'static
{
    /// Sign-extend source lanes `0..8` to `i32`.
    ///
    /// Result lane `i` is `a[i] as i32` for `i` in `0..8`.
    fn widen_low_i16_to_i32(
        self,
        a: <Self as super::I16x16Backend>::Repr,
    ) -> <Self as super::I32x8Backend>::Repr;

    /// Sign-extend source lanes `8..16` to `i32`.
    ///
    /// Result lane `i` is `a[i + 8] as i32`.
    fn widen_high_i16_to_i32(
        self,
        a: <Self as super::I16x16Backend>::Repr,
    ) -> <Self as super::I32x8Backend>::Repr;
}

/// Zero-extending widening from `u8x64` to `u16x32`.
///
/// Native at every tier and in natural lane order everywhere; see
/// `docs/CROSS-ISA-INT-PRIMITIVES.md` §3.
#[cfg(feature = "w512")]
pub trait U8x64Widen:
    super::U8x64Backend + super::U16x32Backend + SimdToken + Sealed + Copy + 'static
{
    /// Zero-extend source lanes `0..32` to `u16`.
    ///
    /// Result lane `i` is `a[i] as u16` for `i` in `0..32`.
    fn widen_low_u8_to_u16(
        self,
        a: <Self as super::U8x64Backend>::Repr,
    ) -> <Self as super::U16x32Backend>::Repr;

    /// Zero-extend source lanes `32..64` to `u16`.
    ///
    /// Result lane `i` is `a[i + 32] as u16`.
    fn widen_high_u8_to_u16(
        self,
        a: <Self as super::U8x64Backend>::Repr,
    ) -> <Self as super::U16x32Backend>::Repr;
}

/// Zero-extending widening from `u16x32` to `u32x16`.
///
/// Native at every tier and in natural lane order everywhere; see
/// `docs/CROSS-ISA-INT-PRIMITIVES.md` §3.
#[cfg(feature = "w512")]
pub trait U16x32Widen:
    super::U16x32Backend + super::U32x16Backend + SimdToken + Sealed + Copy + 'static
{
    /// Zero-extend source lanes `0..16` to `u32`.
    ///
    /// Result lane `i` is `a[i] as u32` for `i` in `0..16`.
    fn widen_low_u16_to_u32(
        self,
        a: <Self as super::U16x32Backend>::Repr,
    ) -> <Self as super::U32x16Backend>::Repr;

    /// Zero-extend source lanes `16..32` to `u32`.
    ///
    /// Result lane `i` is `a[i + 16] as u32`.
    fn widen_high_u16_to_u32(
        self,
        a: <Self as super::U16x32Backend>::Repr,
    ) -> <Self as super::U32x16Backend>::Repr;
}

/// Sign-extending widening from `i8x64` to `i16x32`.
///
/// Native at every tier and in natural lane order everywhere; see
/// `docs/CROSS-ISA-INT-PRIMITIVES.md` §3.
#[cfg(feature = "w512")]
pub trait I8x64Widen:
    super::I8x64Backend + super::I16x32Backend + SimdToken + Sealed + Copy + 'static
{
    /// Sign-extend source lanes `0..32` to `i16`.
    ///
    /// Result lane `i` is `a[i] as i16` for `i` in `0..32`.
    fn widen_low_i8_to_i16(
        self,
        a: <Self as super::I8x64Backend>::Repr,
    ) -> <Self as super::I16x32Backend>::Repr;

    /// Sign-extend source lanes `32..64` to `i16`.
    ///
    /// Result lane `i` is `a[i + 32] as i16`.
    fn widen_high_i8_to_i16(
        self,
        a: <Self as super::I8x64Backend>::Repr,
    ) -> <Self as super::I16x32Backend>::Repr;
}

/// Sign-extending widening from `i16x32` to `i32x16`.
///
/// Native at every tier and in natural lane order everywhere; see
/// `docs/CROSS-ISA-INT-PRIMITIVES.md` §3.
#[cfg(feature = "w512")]
pub trait I16x32Widen:
    super::I16x32Backend + super::I32x16Backend + SimdToken + Sealed + Copy + 'static
{
    /// Sign-extend source lanes `0..16` to `i32`.
    ///
    /// Result lane `i` is `a[i] as i32` for `i` in `0..16`.
    fn widen_low_i16_to_i32(
        self,
        a: <Self as super::I16x32Backend>::Repr,
    ) -> <Self as super::I32x16Backend>::Repr;

    /// Sign-extend source lanes `16..32` to `i32`.
    ///
    /// Result lane `i` is `a[i + 16] as i32`.
    fn widen_high_i16_to_i32(
        self,
        a: <Self as super::I16x32Backend>::Repr,
    ) -> <Self as super::I32x16Backend>::Repr;
}

/// Saturating narrowing from two `i16x8` operands.
///
/// Only the **signed-source** shape is offered: x86 and wasm have no
/// unsigned-source narrowing instruction, so a `u16 -> u8`
/// saturating narrow would disagree across ISAs above `0x7FFF`.
/// See `docs/CROSS-ISA-INT-PRIMITIVES.md` §3.
pub trait I16x8Narrow:
    super::I16x8Backend
    + super::I8x16Backend
    + super::U8x16Backend
    + SimdToken
    + Sealed
    + Copy
    + 'static
{
    /// Signed-saturating narrow to `i8x16`.
    ///
    /// Result lane `i` is `a[i].clamp(i8::MIN, i8::MAX)` for
    /// `i < 8` and `b[i - 8].clamp(..)` for `i >= 8`.
    fn narrow_saturating_i16_to_i8(
        self,
        a: <Self as super::I16x8Backend>::Repr,
        b: <Self as super::I16x8Backend>::Repr,
    ) -> <Self as super::I8x16Backend>::Repr;

    /// Unsigned-saturating narrow to `u8x16`.
    ///
    /// Result lane `i` is `a[i].clamp(0, u8::MAX as i16)` for
    /// `i < 8` and `b[i - 8].clamp(..)` for `i >= 8`.
    fn narrow_saturating_i16_to_u8(
        self,
        a: <Self as super::I16x8Backend>::Repr,
        b: <Self as super::I16x8Backend>::Repr,
    ) -> <Self as super::U8x16Backend>::Repr;
}

/// Saturating narrowing from two `i32x4` operands.
///
/// Only the **signed-source** shape is offered: x86 and wasm have no
/// unsigned-source narrowing instruction, so a `u32 -> u16`
/// saturating narrow would disagree across ISAs above `0x7FFFFFFF`.
/// See `docs/CROSS-ISA-INT-PRIMITIVES.md` §3.
pub trait I32x4Narrow:
    super::I32x4Backend
    + super::I16x8Backend
    + super::U16x8Backend
    + SimdToken
    + Sealed
    + Copy
    + 'static
{
    /// Signed-saturating narrow to `i16x8`.
    ///
    /// Result lane `i` is `a[i].clamp(i16::MIN, i16::MAX)` for
    /// `i < 4` and `b[i - 4].clamp(..)` for `i >= 4`.
    fn narrow_saturating_i32_to_i16(
        self,
        a: <Self as super::I32x4Backend>::Repr,
        b: <Self as super::I32x4Backend>::Repr,
    ) -> <Self as super::I16x8Backend>::Repr;

    /// Unsigned-saturating narrow to `u16x8`.
    ///
    /// Result lane `i` is `a[i].clamp(0, u16::MAX as i32)` for
    /// `i < 4` and `b[i - 4].clamp(..)` for `i >= 4`.
    fn narrow_saturating_i32_to_u16(
        self,
        a: <Self as super::I32x4Backend>::Repr,
        b: <Self as super::I32x4Backend>::Repr,
    ) -> <Self as super::U16x8Backend>::Repr;
}

/// Saturating narrowing from two `i16x16` operands.
///
/// Only the **signed-source** shape is offered: x86 and wasm have no
/// unsigned-source narrowing instruction, so a `u16 -> u8`
/// saturating narrow would disagree across ISAs above `0x7FFF`.
/// See `docs/CROSS-ISA-INT-PRIMITIVES.md` §3.
pub trait I16x16Narrow:
    super::I16x16Backend
    + super::I8x32Backend
    + super::U8x32Backend
    + SimdToken
    + Sealed
    + Copy
    + 'static
{
    /// Signed-saturating narrow to `i8x32`.
    ///
    /// Result lane `i` is `a[i].clamp(i8::MIN, i8::MAX)` for
    /// `i < 16` and `b[i - 16].clamp(..)` for `i >= 16`.
    fn narrow_saturating_i16_to_i8(
        self,
        a: <Self as super::I16x16Backend>::Repr,
        b: <Self as super::I16x16Backend>::Repr,
    ) -> <Self as super::I8x32Backend>::Repr;

    /// Unsigned-saturating narrow to `u8x32`.
    ///
    /// Result lane `i` is `a[i].clamp(0, u8::MAX as i16)` for
    /// `i < 16` and `b[i - 16].clamp(..)` for `i >= 16`.
    fn narrow_saturating_i16_to_u8(
        self,
        a: <Self as super::I16x16Backend>::Repr,
        b: <Self as super::I16x16Backend>::Repr,
    ) -> <Self as super::U8x32Backend>::Repr;
}

/// Saturating narrowing from two `i32x8` operands.
///
/// Only the **signed-source** shape is offered: x86 and wasm have no
/// unsigned-source narrowing instruction, so a `u32 -> u16`
/// saturating narrow would disagree across ISAs above `0x7FFFFFFF`.
/// See `docs/CROSS-ISA-INT-PRIMITIVES.md` §3.
pub trait I32x8Narrow:
    super::I32x8Backend
    + super::I16x16Backend
    + super::U16x16Backend
    + SimdToken
    + Sealed
    + Copy
    + 'static
{
    /// Signed-saturating narrow to `i16x16`.
    ///
    /// Result lane `i` is `a[i].clamp(i16::MIN, i16::MAX)` for
    /// `i < 8` and `b[i - 8].clamp(..)` for `i >= 8`.
    fn narrow_saturating_i32_to_i16(
        self,
        a: <Self as super::I32x8Backend>::Repr,
        b: <Self as super::I32x8Backend>::Repr,
    ) -> <Self as super::I16x16Backend>::Repr;

    /// Unsigned-saturating narrow to `u16x16`.
    ///
    /// Result lane `i` is `a[i].clamp(0, u16::MAX as i32)` for
    /// `i < 8` and `b[i - 8].clamp(..)` for `i >= 8`.
    fn narrow_saturating_i32_to_u16(
        self,
        a: <Self as super::I32x8Backend>::Repr,
        b: <Self as super::I32x8Backend>::Repr,
    ) -> <Self as super::U16x16Backend>::Repr;
}

/// Saturating narrowing from two `i16x32` operands.
///
/// Only the **signed-source** shape is offered: x86 and wasm have no
/// unsigned-source narrowing instruction, so a `u16 -> u8`
/// saturating narrow would disagree across ISAs above `0x7FFF`.
/// See `docs/CROSS-ISA-INT-PRIMITIVES.md` §3.
#[cfg(feature = "w512")]
pub trait I16x32Narrow:
    super::I16x32Backend
    + super::I8x64Backend
    + super::U8x64Backend
    + SimdToken
    + Sealed
    + Copy
    + 'static
{
    /// Signed-saturating narrow to `i8x64`.
    ///
    /// Result lane `i` is `a[i].clamp(i8::MIN, i8::MAX)` for
    /// `i < 32` and `b[i - 32].clamp(..)` for `i >= 32`.
    fn narrow_saturating_i16_to_i8(
        self,
        a: <Self as super::I16x32Backend>::Repr,
        b: <Self as super::I16x32Backend>::Repr,
    ) -> <Self as super::I8x64Backend>::Repr;

    /// Unsigned-saturating narrow to `u8x64`.
    ///
    /// Result lane `i` is `a[i].clamp(0, u8::MAX as i16)` for
    /// `i < 32` and `b[i - 32].clamp(..)` for `i >= 32`.
    fn narrow_saturating_i16_to_u8(
        self,
        a: <Self as super::I16x32Backend>::Repr,
        b: <Self as super::I16x32Backend>::Repr,
    ) -> <Self as super::U8x64Backend>::Repr;
}

/// Saturating narrowing from two `i32x16` operands.
///
/// Only the **signed-source** shape is offered: x86 and wasm have no
/// unsigned-source narrowing instruction, so a `u32 -> u16`
/// saturating narrow would disagree across ISAs above `0x7FFFFFFF`.
/// See `docs/CROSS-ISA-INT-PRIMITIVES.md` §3.
#[cfg(feature = "w512")]
pub trait I32x16Narrow:
    super::I32x16Backend
    + super::I16x32Backend
    + super::U16x32Backend
    + SimdToken
    + Sealed
    + Copy
    + 'static
{
    /// Signed-saturating narrow to `i16x32`.
    ///
    /// Result lane `i` is `a[i].clamp(i16::MIN, i16::MAX)` for
    /// `i < 16` and `b[i - 16].clamp(..)` for `i >= 16`.
    fn narrow_saturating_i32_to_i16(
        self,
        a: <Self as super::I32x16Backend>::Repr,
        b: <Self as super::I32x16Backend>::Repr,
    ) -> <Self as super::I16x32Backend>::Repr;

    /// Unsigned-saturating narrow to `u16x32`.
    ///
    /// Result lane `i` is `a[i].clamp(0, u16::MAX as i32)` for
    /// `i < 16` and `b[i - 16].clamp(..)` for `i >= 16`.
    fn narrow_saturating_i32_to_u16(
        self,
        a: <Self as super::I32x16Backend>::Repr,
        b: <Self as super::I32x16Backend>::Repr,
    ) -> <Self as super::U16x32Backend>::Repr;
}

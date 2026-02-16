//! Popcnt extension backend traits for W512 integer types.
//!
//! These traits extend the base integer backends with population count
//! operations, available on AVX-512 Modern (VPOPCNTDQ + BITALG).
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

use super::*;

/// Population count (popcnt) extension for i8x64.
///
/// Returns a vector where each lane contains the number of set bits
/// in the corresponding lane of the input.
///
/// Requires AVX-512 VPOPCNTDQ (32/64-bit) or BITALG (8/16-bit).
pub trait i8x64PopcntBackend: I8x64Backend {
    /// Count set bits in each lane.
    fn popcnt(a: Self::Repr) -> Self::Repr;
}

/// Population count (popcnt) extension for u8x64.
///
/// Returns a vector where each lane contains the number of set bits
/// in the corresponding lane of the input.
///
/// Requires AVX-512 VPOPCNTDQ (32/64-bit) or BITALG (8/16-bit).
pub trait u8x64PopcntBackend: U8x64Backend {
    /// Count set bits in each lane.
    fn popcnt(a: Self::Repr) -> Self::Repr;
}

/// Population count (popcnt) extension for i16x32.
///
/// Returns a vector where each lane contains the number of set bits
/// in the corresponding lane of the input.
///
/// Requires AVX-512 VPOPCNTDQ (32/64-bit) or BITALG (8/16-bit).
pub trait i16x32PopcntBackend: I16x32Backend {
    /// Count set bits in each lane.
    fn popcnt(a: Self::Repr) -> Self::Repr;
}

/// Population count (popcnt) extension for u16x32.
///
/// Returns a vector where each lane contains the number of set bits
/// in the corresponding lane of the input.
///
/// Requires AVX-512 VPOPCNTDQ (32/64-bit) or BITALG (8/16-bit).
pub trait u16x32PopcntBackend: U16x32Backend {
    /// Count set bits in each lane.
    fn popcnt(a: Self::Repr) -> Self::Repr;
}

/// Population count (popcnt) extension for i32x16.
///
/// Returns a vector where each lane contains the number of set bits
/// in the corresponding lane of the input.
///
/// Requires AVX-512 VPOPCNTDQ (32/64-bit) or BITALG (8/16-bit).
pub trait i32x16PopcntBackend: I32x16Backend {
    /// Count set bits in each lane.
    fn popcnt(a: Self::Repr) -> Self::Repr;
}

/// Population count (popcnt) extension for u32x16.
///
/// Returns a vector where each lane contains the number of set bits
/// in the corresponding lane of the input.
///
/// Requires AVX-512 VPOPCNTDQ (32/64-bit) or BITALG (8/16-bit).
pub trait u32x16PopcntBackend: U32x16Backend {
    /// Count set bits in each lane.
    fn popcnt(a: Self::Repr) -> Self::Repr;
}

/// Population count (popcnt) extension for i64x8.
///
/// Returns a vector where each lane contains the number of set bits
/// in the corresponding lane of the input.
///
/// Requires AVX-512 VPOPCNTDQ (32/64-bit) or BITALG (8/16-bit).
pub trait i64x8PopcntBackend: I64x8Backend {
    /// Count set bits in each lane.
    fn popcnt(a: Self::Repr) -> Self::Repr;
}

/// Population count (popcnt) extension for u64x8.
///
/// Returns a vector where each lane contains the number of set bits
/// in the corresponding lane of the input.
///
/// Requires AVX-512 VPOPCNTDQ (32/64-bit) or BITALG (8/16-bit).
pub trait u64x8PopcntBackend: U64x8Backend {
    /// Count set bits in each lane.
    fn popcnt(a: Self::Repr) -> Self::Repr;
}

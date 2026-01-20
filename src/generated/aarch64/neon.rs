//! Token-gated wrappers for `#[target_feature(enable = "neon")]` functions.
//!
//! This module contains 240 NEON load/store functions that are safe to call when you have a [`NeonToken`].
//!
//! **Auto-generated** from safe_unaligned_simd v0.2.3 - do not edit manually.
//! Run `cargo xtask generate` to regenerate.

// Guard against rare aarch64 targets without NEON (e.g., aarch64-unknown-none-softfloat)
#![cfg(target_feature = "neon")]

#![allow(unused_imports)]
#![allow(unused_macros)] // aarch64_load_store reserved for future SVE/SVE2
#![allow(clippy::too_many_arguments)]
#![allow(clippy::missing_safety_doc)]

use core::arch::aarch64::*;
use crate::tokens::arm::NeonToken;

// Macro for NEON functions - NEON is baseline on aarch64, so no inner function needed.
// The unsafe block is required because safe_unaligned_simd functions are marked with
// #[target_feature], but the operations themselves are safe (they use references).
// Token exists for API consistency with x86.
macro_rules! neon_load_store {
    (
        unsafe: $kind:ident;
        size: $size:ident;

        $(
            $(#[$meta:meta])* fn $intrinsic:ident(_: &[$base_ty:ty; $n:literal][..$len:literal] as $realty:ty) -> $ret:ty;
        )*
    ) => {
        $(
            neon_load_store!(@ $kind $(#[$meta])* $intrinsic [$realty] [$ret]);
        )*
    };

    (@ load $(#[$meta:meta])* $intrinsic:ident [$realty:ty] [$ret:ty]) => {
        $(#[$meta])*
        #[inline(always)]
        pub fn $intrinsic(_token: NeonToken, from: &$realty) -> $ret {
            // SAFETY: NEON is always available on aarch64. The function is safe
            // (uses references), the unsafe is only due to #[target_feature] annotation.
            unsafe { safe_unaligned_simd::aarch64::$intrinsic(from) }
        }
    };

    (@ store $(#[$meta:meta])* $intrinsic:ident [$realty:ty] [$ret:ty]) => {
        $(#[$meta])*
        #[inline(always)]
        pub fn $intrinsic(_token: NeonToken, into: &mut $realty, val: $ret) {
            // SAFETY: NEON is always available on aarch64. The function is safe
            // (uses references), the unsafe is only due to #[target_feature] annotation.
            unsafe { safe_unaligned_simd::aarch64::$intrinsic(into, val) }
        }
    };
}

// Macro for non-baseline features (SVE, SVE2) - requires #[target_feature] wrapper.
// Use this for any aarch64 feature that needs runtime detection.
macro_rules! aarch64_load_store {
    (
        token: $token:ty;
        feature: $feature:literal;
        unsafe: $kind:ident;
        size: $size:ident;

        $(
            $(#[$meta:meta])* fn $intrinsic:ident(_: &[$base_ty:ty; $n:literal][..$len:literal] as $realty:ty) -> $ret:ty;
        )*
    ) => {
        $(
            aarch64_load_store!(@ $kind $token $feature $(#[$meta])* $intrinsic [$realty] [$ret]);
        )*
    };

    (@ load $token:ty, $feature:literal, $(#[$meta:meta])* $intrinsic:ident [$realty:ty] [$ret:ty]) => {
        $(#[$meta])*
        #[inline(always)]
        pub fn $intrinsic(_token: $token, from: &$realty) -> $ret {
            #[inline]
            #[target_feature(enable = $feature)]
            unsafe fn inner(from: &$realty) -> $ret {
                safe_unaligned_simd::aarch64::$intrinsic(from)
            }
            unsafe { inner(from) }
        }
    };

    (@ store $token:ty, $feature:literal, $(#[$meta:meta])* $intrinsic:ident [$realty:ty] [$ret:ty]) => {
        $(#[$meta])*
        #[inline(always)]
        pub fn $intrinsic(_token: $token, into: &mut $realty, val: $ret) {
            #[inline]
            #[target_feature(enable = $feature)]
            unsafe fn inner(into: &mut $realty, val: $ret) {
                safe_unaligned_simd::aarch64::$intrinsic(into, val)
            }
            unsafe { inner(into, val) }
        }
    };
}

// ============================================================================
// Auto-extracted macro invocations from safe_unaligned_simd
// ============================================================================

neon_load_store! {
    unsafe: load;
    // Loads full registers, so 8 bytes per register
    size: assert_size_8bytes;

    /// Load an array of 8 `u8` values to one 8-byte register.
    fn vld1_u8(_: &[u8; 8][..1] as [u8; 8]) -> uint8x8_t;
    /// Load an array of 8 `i8` values to one 8-byte register.
    fn vld1_s8(_: &[i8; 8][..1] as [i8; 8]) -> int8x8_t;
    /// Load an array of 4 `u16` values to one 8-byte register.
    fn vld1_u16(_: &[u16; 4][..1] as [u16; 4]) -> uint16x4_t;
    /// Load an array of 4 `i16` values to one 8-byte register.
    fn vld1_s16(_: &[i16; 4][..1] as [i16; 4]) -> int16x4_t;
    /// Load an array of 2 `u32` values to one 8-byte register.
    fn vld1_u32(_: &[u32; 2][..1] as [u32; 2]) -> uint32x2_t;
    /// Load an array of 2 `i32` values to one 8-byte register.
    fn vld1_s32(_: &[i32; 2][..1] as [i32; 2]) -> int32x2_t;
    /// Load an array of 2 `f32` values to one 8-byte register.
    fn vld1_f32(_: &[f32; 2][..1] as [f32; 2]) -> float32x2_t;
    /// Load one `u64` value to one 8-byte register.
    fn vld1_u64(_: &[u64; 1][..1] as u64) -> uint64x1_t;
    /// Load one `i64` value to one 8-byte register.
    fn vld1_s64(_: &[i64; 1][..1] as i64) -> int64x1_t;
    /// Load one `f64` value to one 8-byte register.
    fn vld1_f64(_: &[f64; 1][..1] as f64) -> float64x1_t;

    /// Load arrays of 8 `u8` values to two 8-byte registers.
    fn vld1_u8_x2(_: &[u8; 8][..2] as [[u8; 8]; 2]) -> uint8x8x2_t;
    /// Load arrays of 8 `i8` values to two 8-byte registers.
    fn vld1_s8_x2(_: &[i8; 8][..2] as [[i8; 8]; 2]) -> int8x8x2_t;
    /// Load arrays of 4 `u16` values to two 8-byte registers.
    fn vld1_u16_x2(_: &[u16; 4][..2] as [[u16; 4]; 2]) -> uint16x4x2_t;
    /// Load arrays of 4 `i16` values to two 8-byte registers.
    fn vld1_s16_x2(_: &[i16; 4][..2] as [[i16; 4]; 2]) -> int16x4x2_t;
    /// Load arrays of 2 `u32` values to two 8-byte registers.
    fn vld1_u32_x2(_: &[u32; 2][..2] as [[u32; 2]; 2]) -> uint32x2x2_t;
    /// Load arrays of 2 `i32` values to two 8-byte registers.
    fn vld1_s32_x2(_: &[i32; 2][..2] as [[i32; 2]; 2]) -> int32x2x2_t;
    /// Load arrays of 2 `f32` values to two 8-byte registers.
    fn vld1_f32_x2(_: &[f32; 2][..2] as [[f32; 2]; 2]) -> float32x2x2_t;
    /// Load two `u64` values to two 8-byte registers.
    fn vld1_u64_x2(_: &[u64; 1][..2] as [u64; 2]) -> uint64x1x2_t;
    /// Load two `i64` values to two 8-byte registers.
    fn vld1_s64_x2(_: &[i64; 1][..2] as [i64; 2]) -> int64x1x2_t;
    /// Load two `f64` values to two 8-byte registers.
    fn vld1_f64_x2(_: &[f64; 1][..2] as [f64; 2]) -> float64x1x2_t;

    /// Load arrays of 8 `u8` values to three 8-byte registers.
    fn vld1_u8_x3(_: &[u8; 8][..3] as [[u8; 8]; 3]) -> uint8x8x3_t;
    /// Load arrays of 8 `i8` values to three 8-byte registers.
    fn vld1_s8_x3(_: &[i8; 8][..3] as [[i8; 8]; 3]) -> int8x8x3_t;
    /// Load arrays of 4 `u16` values to three 8-byte registers.
    fn vld1_u16_x3(_: &[u16; 4][..3] as [[u16; 4]; 3]) -> uint16x4x3_t;
    /// Load arrays of 4 `i16` values to three 8-byte registers.
    fn vld1_s16_x3(_: &[i16; 4][..3] as [[i16; 4]; 3]) -> int16x4x3_t;
    /// Load arrays of 2 `u32` values to three 8-byte registers.
    fn vld1_u32_x3(_: &[u32; 2][..3] as [[u32; 2]; 3]) -> uint32x2x3_t;
    /// Load arrays of 2 `i32` values to three 8-byte registers.
    fn vld1_s32_x3(_: &[i32; 2][..3] as [[i32; 2]; 3]) -> int32x2x3_t;
    /// Load arrays of 2 `f32` values to three 8-byte registers.
    fn vld1_f32_x3(_: &[f32; 2][..3] as [[f32; 2]; 3]) -> float32x2x3_t;
    /// Load two `u64` values to three 8-byte registers.
    fn vld1_u64_x3(_: &[u64; 1][..3] as [u64; 3]) -> uint64x1x3_t;
    /// Load two `i64` values to three 8-byte registers.
    fn vld1_s64_x3(_: &[i64; 1][..3] as [i64; 3]) -> int64x1x3_t;
    /// Load two `f64` values to three 8-byte registers.
    fn vld1_f64_x3(_: &[f64; 1][..3] as [f64; 3]) -> float64x1x3_t;

    /// Load arrays of 8 `u8` values to four 8-byte registers.
    fn vld1_u8_x4(_: &[u8; 8][..4] as [[u8; 8]; 4]) -> uint8x8x4_t;
    /// Load arrays of 8 `i8` values to four 8-byte registers.
    fn vld1_s8_x4(_: &[i8; 8][..4] as [[i8; 8]; 4]) -> int8x8x4_t;
    /// Load arrays of 4 `u16` values to four 8-byte registers.
    fn vld1_u16_x4(_: &[u16; 4][..4] as [[u16; 4]; 4]) -> uint16x4x4_t;
    /// Load arrays of 4 `i16` values to four 8-byte registers.
    fn vld1_s16_x4(_: &[i16; 4][..4] as [[i16; 4]; 4]) -> int16x4x4_t;
    /// Load arrays of 2 `u32` values to four 8-byte registers.
    fn vld1_u32_x4(_: &[u32; 2][..4] as [[u32; 2]; 4]) -> uint32x2x4_t;
    /// Load arrays of 2 `i32` values to four 8-byte registers.
    fn vld1_s32_x4(_: &[i32; 2][..4] as [[i32; 2]; 4]) -> int32x2x4_t;
    /// Load arrays of 2 `f32` values to four 8-byte registers.
    fn vld1_f32_x4(_: &[f32; 2][..4] as [[f32; 2]; 4]) -> float32x2x4_t;
    /// Load two `u64` values to four 8-byte registers.
    fn vld1_u64_x4(_: &[u64; 1][..4] as [u64; 4]) -> uint64x1x4_t;
    /// Load two `i64` values to four 8-byte registers.
    fn vld1_s64_x4(_: &[i64; 1][..4] as [i64; 4]) -> int64x1x4_t;
    /// Load two `f64` values to four 8-byte registers.
    fn vld1_f64_x4(_: &[f64; 1][..4] as [f64; 4]) -> float64x1x4_t;
}

neon_load_store! {
    unsafe: load;
    // Loads full registers, so 16 bytes per register
    size: assert_size_16bytes;

    /// Load an array of 16 `u8` values to one 16-byte register.
    fn vld1q_u8(_: &[u8; 16][..1] as [u8; 16]) -> uint8x16_t;
    /// Load an array of 16 `i8` values to one 16-byte register.
    fn vld1q_s8(_: &[i8; 16][..1] as [i8; 16]) -> int8x16_t;
    /// Load an array of 8 `u16` values to one 16-byte register.
    fn vld1q_u16(_: &[u16; 8][..1] as [u16; 8]) -> uint16x8_t;
    /// Load an array of 8 `i16` values to one 16-byte register.
    fn vld1q_s16(_: &[i16; 8][..1] as [i16; 8]) -> int16x8_t;
    /// Load an array of 4 `u32` values to one 16-byte register.
    fn vld1q_u32(_: &[u32; 4][..1] as [u32; 4]) -> uint32x4_t;
    /// Load an array of 4 `i32` values to one 16-byte register.
    fn vld1q_s32(_: &[i32; 4][..1] as [i32; 4]) -> int32x4_t;
    /// Load an array of 4 `f32` values to one 16-byte register.
    fn vld1q_f32(_: &[f32; 4][..1] as [f32; 4]) -> float32x4_t;
    /// Load an array of 2 `u64` value to one 16-byte register.
    fn vld1q_u64(_: &[u64; 2][..1] as [u64; 2]) -> uint64x2_t;
    /// Load an array of 2 `i64` value to one 16-byte register.
    fn vld1q_s64(_: &[i64; 2][..1] as [i64; 2]) -> int64x2_t;
    /// Load an array of 2 `f64` value to one 16-byte register.
    fn vld1q_f64(_: &[f64; 2][..1] as [f64; 2]) -> float64x2_t;

    /// Load two arrays of 16 `u8` values to two 16-byte registers.
    fn vld1q_u8_x2(_: &[u8; 16][..2] as [[u8; 16]; 2]) -> uint8x16x2_t;
    /// Load two arrays of 16 `i8` values to two 16-byte registers.
    fn vld1q_s8_x2(_: &[i8; 16][..2] as [[i8; 16]; 2]) -> int8x16x2_t;
    /// Load two arrays of 8 `u16` values to two 16-byte registers.
    fn vld1q_u16_x2(_: &[u16; 8][..2] as [[u16; 8]; 2]) -> uint16x8x2_t;
    /// Load two arrays of 8 `i16` values to two 16-byte registers.
    fn vld1q_s16_x2(_: &[i16; 8][..2] as [[i16; 8]; 2]) -> int16x8x2_t;
    /// Load two arrays of 4 `u32` values to two 16-byte registers.
    fn vld1q_u32_x2(_: &[u32; 4][..2] as [[u32; 4]; 2]) -> uint32x4x2_t;
    /// Load two arrays of 4 `i32` values to two 16-byte registers.
    fn vld1q_s32_x2(_: &[i32; 4][..2] as [[i32; 4]; 2]) -> int32x4x2_t;
    /// Load two arrays of 4 `f32` values to two 16-byte registers.
    fn vld1q_f32_x2(_: &[f32; 4][..2] as [[f32; 4]; 2]) -> float32x4x2_t;
    /// Load two arrays of 2 `u64` value to two 16-byte registers.
    fn vld1q_u64_x2(_: &[u64; 2][..2] as [[u64; 2]; 2]) -> uint64x2x2_t;
    /// Load two arrays of 2 `i64` value to two 16-byte registers.
    fn vld1q_s64_x2(_: &[i64; 2][..2] as [[i64; 2]; 2]) -> int64x2x2_t;
    /// Load two arrays of 2 `f64` value to two 16-byte registers.
    fn vld1q_f64_x2(_: &[f64; 2][..2] as [[f64; 2]; 2]) -> float64x2x2_t;

    /// Load three arrays of 16 `u8` values to three16-byte registers.
    fn vld1q_u8_x3(_: &[u8; 16][..3] as [[u8; 16]; 3]) -> uint8x16x3_t;
    /// Load three arrays of 16 `i8` values to three16-byte registers.
    fn vld1q_s8_x3(_: &[i8; 16][..3] as [[i8; 16]; 3]) -> int8x16x3_t;
    /// Load three arrays of 8 `u16` values to three16-byte registers.
    fn vld1q_u16_x3(_: &[u16; 8][..3] as [[u16; 8]; 3]) -> uint16x8x3_t;
    /// Load three arrays of 8 `i16` values to three16-byte registers.
    fn vld1q_s16_x3(_: &[i16; 8][..3] as [[i16; 8]; 3]) -> int16x8x3_t;
    /// Load three arrays of 4 `u32` values to three16-byte registers.
    fn vld1q_u32_x3(_: &[u32; 4][..3] as [[u32; 4]; 3]) -> uint32x4x3_t;
    /// Load three arrays of 4 `i32` values to three16-byte registers.
    fn vld1q_s32_x3(_: &[i32; 4][..3] as [[i32; 4]; 3]) -> int32x4x3_t;
    /// Load three arrays of 4 `f32` values to three16-byte registers.
    fn vld1q_f32_x3(_: &[f32; 4][..3] as [[f32; 4]; 3]) -> float32x4x3_t;
    /// Load three arrays of 2 `u64` value to three16-byte registers.
    fn vld1q_u64_x3(_: &[u64; 2][..3] as [[u64; 2]; 3]) -> uint64x2x3_t;
    /// Load three arrays of 2 `i64` value to three16-byte registers.
    fn vld1q_s64_x3(_: &[i64; 2][..3] as [[i64; 2]; 3]) -> int64x2x3_t;
    /// Load three arrays of 2 `f64` value to three16-byte registers.
    fn vld1q_f64_x3(_: &[f64; 2][..3] as [[f64; 2]; 3]) -> float64x2x3_t;

    /// Load four arrays of 16 `u8` values to four 16-byte registers.
    fn vld1q_u8_x4(_: &[u8; 16][..4] as [[u8; 16]; 4]) -> uint8x16x4_t;
    /// Load four arrays of 16 `i8` values to four 16-byte registers.
    fn vld1q_s8_x4(_: &[i8; 16][..4] as [[i8; 16]; 4]) -> int8x16x4_t;
    /// Load four arrays of 8 `u16` values to four 16-byte registers.
    fn vld1q_u16_x4(_: &[u16; 8][..4] as [[u16; 8]; 4]) -> uint16x8x4_t;
    /// Load four arrays of 8 `i16` values to four 16-byte registers.
    fn vld1q_s16_x4(_: &[i16; 8][..4] as [[i16; 8]; 4]) -> int16x8x4_t;
    /// Load four arrays of 4 `u32` values to four 16-byte registers.
    fn vld1q_u32_x4(_: &[u32; 4][..4] as [[u32; 4]; 4]) -> uint32x4x4_t;
    /// Load four arrays of 4 `i32` values to four 16-byte registers.
    fn vld1q_s32_x4(_: &[i32; 4][..4] as [[i32; 4]; 4]) -> int32x4x4_t;
    /// Load four arrays of 4 `f32` values to four 16-byte registers.
    fn vld1q_f32_x4(_: &[f32; 4][..4] as [[f32; 4]; 4]) -> float32x4x4_t;
    /// Load four arrays of 2 `u64` value to four 16-byte registers.
    fn vld1q_u64_x4(_: &[u64; 2][..4] as [[u64; 2]; 4]) -> uint64x2x4_t;
    /// Load four arrays of 2 `i64` value to four 16-byte registers.
    fn vld1q_s64_x4(_: &[i64; 2][..4] as [[i64; 2]; 4]) -> int64x2x4_t;
    /// Load four arrays of 2 `f64` value to four 16-byte registers.
    fn vld1q_f64_x4(_: &[f64; 2][..4] as [[f64; 2]; 4]) -> float64x2x4_t;
}

neon_load_store! {
    unsafe: store;
    // Stores full registers, so 8 bytes per register
    size: assert_size_8bytes;

    /// Store an array of 8 `u8` values from one 8-byte register.
    fn vst1_u8(_: &[u8; 8][..1] as [u8; 8]) -> uint8x8_t;
    /// Store an array of 8 `i8` values from one 8-byte register.
    fn vst1_s8(_: &[i8; 8][..1] as [i8; 8]) -> int8x8_t;
    /// Store an array of 4 `u16` values from one 8-byte register.
    fn vst1_u16(_: &[u16; 4][..1] as [u16; 4]) -> uint16x4_t;
    /// Store an array of 4 `i16` values from one 8-byte register.
    fn vst1_s16(_: &[i16; 4][..1] as [i16; 4]) -> int16x4_t;
    /// Store an array of 2 `u32` values from one 8-byte register.
    fn vst1_u32(_: &[u32; 2][..1] as [u32; 2]) -> uint32x2_t;
    /// Store an array of 2 `i32` values from one 8-byte register.
    fn vst1_s32(_: &[i32; 2][..1] as [i32; 2]) -> int32x2_t;
    /// Store an array of 2 `f32` values from one 8-byte register.
    fn vst1_f32(_: &[f32; 2][..1] as [f32; 2]) -> float32x2_t;
    /// Store one `u64` value from one 8-byte register.
    fn vst1_u64(_: &[u64; 1][..1] as u64) -> uint64x1_t;
    /// Store one `i64` value from one 8-byte register.
    fn vst1_s64(_: &[i64; 1][..1] as i64) -> int64x1_t;
    /// Store one `f64` value from one 8-byte register.
    fn vst1_f64(_: &[f64; 1][..1] as f64) -> float64x1_t;

    /// Store arrays of 8 `u8` values from two 8-byte registers.
    fn vst1_u8_x2(_: &[u8; 8][..2] as [[u8; 8]; 2]) -> uint8x8x2_t;
    /// Store arrays of 8 `i8` values from two 8-byte registers.
    fn vst1_s8_x2(_: &[i8; 8][..2] as [[i8; 8]; 2]) -> int8x8x2_t;
    /// Store arrays of 4 `u16` values from two 8-byte registers.
    fn vst1_u16_x2(_: &[u16; 4][..2] as [[u16; 4]; 2]) -> uint16x4x2_t;
    /// Store arrays of 4 `i16` values from two 8-byte registers.
    fn vst1_s16_x2(_: &[i16; 4][..2] as [[i16; 4]; 2]) -> int16x4x2_t;
    /// Store arrays of 2 `u32` values from two 8-byte registers.
    fn vst1_u32_x2(_: &[u32; 2][..2] as [[u32; 2]; 2]) -> uint32x2x2_t;
    /// Store arrays of 2 `i32` values from two 8-byte registers.
    fn vst1_s32_x2(_: &[i32; 2][..2] as [[i32; 2]; 2]) -> int32x2x2_t;
    /// Store arrays of 2 `f32` values from two 8-byte registers.
    fn vst1_f32_x2(_: &[f32; 2][..2] as [[f32; 2]; 2]) -> float32x2x2_t;
    /// Store two `u64` values from two 8-byte registers.
    fn vst1_u64_x2(_: &[u64; 1][..2] as [u64; 2]) -> uint64x1x2_t;
    /// Store two `i64` values from two 8-byte registers.
    fn vst1_s64_x2(_: &[i64; 1][..2] as [i64; 2]) -> int64x1x2_t;
    /// Store two `f64` values from two 8-byte registers.
    fn vst1_f64_x2(_: &[f64; 1][..2] as [f64; 2]) -> float64x1x2_t;

    /// Store arrays of 8 `u8` values from three 8-byte registers.
    fn vst1_u8_x3(_: &[u8; 8][..3] as [[u8; 8]; 3]) -> uint8x8x3_t;
    /// Store arrays of 8 `i8` values from three 8-byte registers.
    fn vst1_s8_x3(_: &[i8; 8][..3] as [[i8; 8]; 3]) -> int8x8x3_t;
    /// Store arrays of 4 `u16` values from three 8-byte registers.
    fn vst1_u16_x3(_: &[u16; 4][..3] as [[u16; 4]; 3]) -> uint16x4x3_t;
    /// Store arrays of 4 `i16` values from three 8-byte registers.
    fn vst1_s16_x3(_: &[i16; 4][..3] as [[i16; 4]; 3]) -> int16x4x3_t;
    /// Store arrays of 2 `u32` values from three 8-byte registers.
    fn vst1_u32_x3(_: &[u32; 2][..3] as [[u32; 2]; 3]) -> uint32x2x3_t;
    /// Store arrays of 2 `i32` values from three 8-byte registers.
    fn vst1_s32_x3(_: &[i32; 2][..3] as [[i32; 2]; 3]) -> int32x2x3_t;
    /// Store arrays of 2 `f32` values from three 8-byte registers.
    fn vst1_f32_x3(_: &[f32; 2][..3] as [[f32; 2]; 3]) -> float32x2x3_t;
    /// Store two `u64` values from three 8-byte registers.
    fn vst1_u64_x3(_: &[u64; 1][..3] as [u64; 3]) -> uint64x1x3_t;
    /// Store two `i64` values from three 8-byte registers.
    fn vst1_s64_x3(_: &[i64; 1][..3] as [i64; 3]) -> int64x1x3_t;
    /// Store two `f64` values from three 8-byte registers.
    fn vst1_f64_x3(_: &[f64; 1][..3] as [f64; 3]) -> float64x1x3_t;

    /// Store arrays of 8 `u8` values from four 8-byte registers.
    fn vst1_u8_x4(_: &[u8; 8][..4] as [[u8; 8]; 4]) -> uint8x8x4_t;
    /// Store arrays of 8 `i8` values from four 8-byte registers.
    fn vst1_s8_x4(_: &[i8; 8][..4] as [[i8; 8]; 4]) -> int8x8x4_t;
    /// Store arrays of 4 `u16` values from four 8-byte registers.
    fn vst1_u16_x4(_: &[u16; 4][..4] as [[u16; 4]; 4]) -> uint16x4x4_t;
    /// Store arrays of 4 `i16` values from four 8-byte registers.
    fn vst1_s16_x4(_: &[i16; 4][..4] as [[i16; 4]; 4]) -> int16x4x4_t;
    /// Store arrays of 2 `u32` values from four 8-byte registers.
    fn vst1_u32_x4(_: &[u32; 2][..4] as [[u32; 2]; 4]) -> uint32x2x4_t;
    /// Store arrays of 2 `i32` values from four 8-byte registers.
    fn vst1_s32_x4(_: &[i32; 2][..4] as [[i32; 2]; 4]) -> int32x2x4_t;
    /// Store arrays of 2 `f32` values from four 8-byte registers.
    fn vst1_f32_x4(_: &[f32; 2][..4] as [[f32; 2]; 4]) -> float32x2x4_t;
    /// Store two `u64` values from four 8-byte registers.
    fn vst1_u64_x4(_: &[u64; 1][..4] as [u64; 4]) -> uint64x1x4_t;
    /// Store two `i64` values from four 8-byte registers.
    fn vst1_s64_x4(_: &[i64; 1][..4] as [i64; 4]) -> int64x1x4_t;
    /// Store two `f64` values from four 8-byte registers.
    fn vst1_f64_x4(_: &[f64; 1][..4] as [f64; 4]) -> float64x1x4_t;
}

neon_load_store! {
    unsafe: store;
    // Stores full registers, so 16 bytes per register
    size: assert_size_16bytes;

    /// Store an array of 16 `u8` values to one 16-byte register.
    fn vst1q_u8(_: &[u8; 16][..1] as [u8; 16]) -> uint8x16_t;
    /// Store an array of 16 `i8` values to one 16-byte register.
    fn vst1q_s8(_: &[i8; 16][..1] as [i8; 16]) -> int8x16_t;
    /// Store an array of 8 `u16` values to one 16-byte register.
    fn vst1q_u16(_: &[u16; 8][..1] as [u16; 8]) -> uint16x8_t;
    /// Store an array of 8 `i16` values to one 16-byte register.
    fn vst1q_s16(_: &[i16; 8][..1] as [i16; 8]) -> int16x8_t;
    /// Store an array of 4 `u32` values to one 16-byte register.
    fn vst1q_u32(_: &[u32; 4][..1] as [u32; 4]) -> uint32x4_t;
    /// Store an array of 4 `i32` values to one 16-byte register.
    fn vst1q_s32(_: &[i32; 4][..1] as [i32; 4]) -> int32x4_t;
    /// Store an array of 4 `f32` values to one 16-byte register.
    fn vst1q_f32(_: &[f32; 4][..1] as [f32; 4]) -> float32x4_t;
    /// Store an array of 2 `u64` value to one 16-byte register.
    fn vst1q_u64(_: &[u64; 2][..1] as [u64; 2]) -> uint64x2_t;
    /// Store an array of 2 `i64` value to one 16-byte register.
    fn vst1q_s64(_: &[i64; 2][..1] as [i64; 2]) -> int64x2_t;
    /// Store an array of 2 `f64` value to one 16-byte register.
    fn vst1q_f64(_: &[f64; 2][..1] as [f64; 2]) -> float64x2_t;

    /// Store two arrays of 16 `u8` values from two 16-byte registers.
    fn vst1q_u8_x2(_: &[u8; 16][..2] as [[u8; 16]; 2]) -> uint8x16x2_t;
    /// Store two arrays of 16 `i8` values from two 16-byte registers.
    fn vst1q_s8_x2(_: &[i8; 16][..2] as [[i8; 16]; 2]) -> int8x16x2_t;
    /// Store two arrays of 8 `u16` values from two 16-byte registers.
    fn vst1q_u16_x2(_: &[u16; 8][..2] as [[u16; 8]; 2]) -> uint16x8x2_t;
    /// Store two arrays of 8 `i16` values from two 16-byte registers.
    fn vst1q_s16_x2(_: &[i16; 8][..2] as [[i16; 8]; 2]) -> int16x8x2_t;
    /// Store two arrays of 4 `u32` values from two 16-byte registers.
    fn vst1q_u32_x2(_: &[u32; 4][..2] as [[u32; 4]; 2]) -> uint32x4x2_t;
    /// Store two arrays of 4 `i32` values from two 16-byte registers.
    fn vst1q_s32_x2(_: &[i32; 4][..2] as [[i32; 4]; 2]) -> int32x4x2_t;
    /// Store two arrays of 4 `f32` values from two 16-byte registers.
    fn vst1q_f32_x2(_: &[f32; 4][..2] as [[f32; 4]; 2]) -> float32x4x2_t;
    /// Store two arrays of 2 `u64` value from two 16-byte registers.
    fn vst1q_u64_x2(_: &[u64; 2][..2] as [[u64; 2]; 2]) -> uint64x2x2_t;
    /// Store two arrays of 2 `i64` value from two 16-byte registers.
    fn vst1q_s64_x2(_: &[i64; 2][..2] as [[i64; 2]; 2]) -> int64x2x2_t;
    /// Store two arrays of 2 `f64` value from two 16-byte registers.
    fn vst1q_f64_x2(_: &[f64; 2][..2] as [[f64; 2]; 2]) -> float64x2x2_t;

    /// Store three arrays of 16 `u8` values from three16-byte registers.
    fn vst1q_u8_x3(_: &[u8; 16][..3] as [[u8; 16]; 3]) -> uint8x16x3_t;
    /// Store three arrays of 16 `i8` values from three16-byte registers.
    fn vst1q_s8_x3(_: &[i8; 16][..3] as [[i8; 16]; 3]) -> int8x16x3_t;
    /// Store three arrays of 8 `u16` values from three16-byte registers.
    fn vst1q_u16_x3(_: &[u16; 8][..3] as [[u16; 8]; 3]) -> uint16x8x3_t;
    /// Store three arrays of 8 `i16` values from three16-byte registers.
    fn vst1q_s16_x3(_: &[i16; 8][..3] as [[i16; 8]; 3]) -> int16x8x3_t;
    /// Store three arrays of 4 `u32` values from three16-byte registers.
    fn vst1q_u32_x3(_: &[u32; 4][..3] as [[u32; 4]; 3]) -> uint32x4x3_t;
    /// Store three arrays of 4 `i32` values from three16-byte registers.
    fn vst1q_s32_x3(_: &[i32; 4][..3] as [[i32; 4]; 3]) -> int32x4x3_t;
    /// Store three arrays of 4 `f32` values from three16-byte registers.
    fn vst1q_f32_x3(_: &[f32; 4][..3] as [[f32; 4]; 3]) -> float32x4x3_t;
    /// Store three arrays of 2 `u64` value from three16-byte registers.
    fn vst1q_u64_x3(_: &[u64; 2][..3] as [[u64; 2]; 3]) -> uint64x2x3_t;
    /// Store three arrays of 2 `i64` value from three16-byte registers.
    fn vst1q_s64_x3(_: &[i64; 2][..3] as [[i64; 2]; 3]) -> int64x2x3_t;
    /// Store three arrays of 2 `f64` value from three16-byte registers.
    fn vst1q_f64_x3(_: &[f64; 2][..3] as [[f64; 2]; 3]) -> float64x2x3_t;

    /// Store four arrays of 16 `u8` values from four 16-byte registers.
    fn vst1q_u8_x4(_: &[u8; 16][..4] as [[u8; 16]; 4]) -> uint8x16x4_t;
    /// Store four arrays of 16 `i8` values from four 16-byte registers.
    fn vst1q_s8_x4(_: &[i8; 16][..4] as [[i8; 16]; 4]) -> int8x16x4_t;
    /// Store four arrays of 8 `u16` values from four 16-byte registers.
    fn vst1q_u16_x4(_: &[u16; 8][..4] as [[u16; 8]; 4]) -> uint16x8x4_t;
    /// Store four arrays of 8 `i16` values from four 16-byte registers.
    fn vst1q_s16_x4(_: &[i16; 8][..4] as [[i16; 8]; 4]) -> int16x8x4_t;
    /// Store four arrays of 4 `u32` values from four 16-byte registers.
    fn vst1q_u32_x4(_: &[u32; 4][..4] as [[u32; 4]; 4]) -> uint32x4x4_t;
    /// Store four arrays of 4 `i32` values from four 16-byte registers.
    fn vst1q_s32_x4(_: &[i32; 4][..4] as [[i32; 4]; 4]) -> int32x4x4_t;
    /// Store four arrays of 4 `f32` values from four 16-byte registers.
    fn vst1q_f32_x4(_: &[f32; 4][..4] as [[f32; 4]; 4]) -> float32x4x4_t;
    /// Store four arrays of 2 `u64` value from four 16-byte registers.
    fn vst1q_u64_x4(_: &[u64; 2][..4] as [[u64; 2]; 4]) -> uint64x2x4_t;
    /// Store four arrays of 2 `i64` value from four 16-byte registers.
    fn vst1q_s64_x4(_: &[i64; 2][..4] as [[i64; 2]; 4]) -> int64x2x4_t;
    /// Store four arrays of 2 `f64` value from four 16-byte registers.
    fn vst1q_f64_x4(_: &[f64; 2][..4] as [[f64; 2]; 4]) -> float64x2x4_t;
}

neon_load_store! {
    unsafe: load;
    size: various_sizes;

    /// Load one single-element `i8` and replicate to all lanes.
    fn vld1_dup_s8(_: &[i8; 1][..1] as i8) -> int8x8_t;
    /// Load an array of two `i8` elements and replicate to lanes of two registers.
    fn vld2_dup_s8(_: &[i8; 2][..1] as [i8; 2]) -> int8x8x2_t;
    /// Load an array of three `i8` elements and replicate to lanes of three registers.
    fn vld3_dup_s8(_: &[i8; 3][..1] as [i8; 3]) -> int8x8x3_t;
    /// Load an array of four `i8` elements and replicate to lanes of four registers.
    fn vld4_dup_s8(_: &[i8; 4][..1] as [i8; 4]) -> int8x8x4_t;

    /// Load one single-element `u8` and replicate to all lanes.
    fn vld1_dup_u8(_: &[u8; 1][..1] as u8) -> uint8x8_t;
    /// Load an array of two `u8` elements and replicate to lanes of two registers.
    fn vld2_dup_u8(_: &[u8; 2][..1] as [u8; 2]) -> uint8x8x2_t;
    /// Load an array of three `u8` elements and replicate to lanes of three registers.
    fn vld3_dup_u8(_: &[u8; 3][..1] as [u8; 3]) -> uint8x8x3_t;
    /// Load an array of four `u8` elements and replicate to lanes of four registers.
    fn vld4_dup_u8(_: &[u8; 4][..1] as [u8; 4]) -> uint8x8x4_t;

    /// Load one single-element `i16` and replicate to all lanes.
    fn vld1_dup_s16(_: &[i16; 1][..1] as i16) -> int16x4_t;
    /// Load an array of two `i16` elements and replicate to lanes of two registers.
    fn vld2_dup_s16(_: &[i16; 2][..1] as [i16; 2]) -> int16x4x2_t;
    /// Load an array of three `i16` elements and replicate to lanes of three registers.
    fn vld3_dup_s16(_: &[i16; 3][..1] as [i16; 3]) -> int16x4x3_t;
    /// Load an array of four `i16` elements and replicate to lanes of four registers.
    fn vld4_dup_s16(_: &[i16; 4][..1] as [i16; 4]) -> int16x4x4_t;

    /// Load one single-element `u16` and replicate to all lanes.
    fn vld1_dup_u16(_: &[u16; 1][..1] as u16) -> uint16x4_t;
    /// Load an array of two `u16` elements and replicate to lanes of two registers.
    fn vld2_dup_u16(_: &[u16; 2][..1] as [u16; 2]) -> uint16x4x2_t;
    /// Load an array of three `u16` elements and replicate to lanes of three registers.
    fn vld3_dup_u16(_: &[u16; 3][..1] as [u16; 3]) -> uint16x4x3_t;
    /// Load an array of four `u16` elements and replicate to lanes of four registers.
    fn vld4_dup_u16(_: &[u16; 4][..1] as [u16; 4]) -> uint16x4x4_t;

    /// Load one single-element `i32` and replicate to all lanes.
    fn vld1_dup_s32(_: &[i32; 1][..1] as i32) -> int32x2_t;
    /// Load an array of two `i32` elements and replicate to lanes of two registers.
    fn vld2_dup_s32(_: &[i32; 2][..1] as [i32; 2]) -> int32x2x2_t;
    /// Load an array of three `i32` elements and replicate to lanes of three registers.
    fn vld3_dup_s32(_: &[i32; 3][..1] as [i32; 3]) -> int32x2x3_t;
    /// Load an array of four `i32` elements and replicate to lanes of four registers.
    fn vld4_dup_s32(_: &[i32; 4][..1] as [i32; 4]) -> int32x2x4_t;

    /// Load one single-element `u32` and replicate to all lanes.
    fn vld1_dup_u32(_: &[u32; 1][..1] as u32) -> uint32x2_t;
    /// Load an array of two `u32` elements and replicate to lanes of two registers.
    fn vld2_dup_u32(_: &[u32; 2][..1] as [u32; 2]) -> uint32x2x2_t;
    /// Load an array of three `u32` elements and replicate to lanes of three registers.
    fn vld3_dup_u32(_: &[u32; 3][..1] as [u32; 3]) -> uint32x2x3_t;
    /// Load an array of four `u32` elements and replicate to lanes of four registers.
    fn vld4_dup_u32(_: &[u32; 4][..1] as [u32; 4]) -> uint32x2x4_t;

    /// Load one single-element `f32` and replicate to all lanes.
    fn vld1_dup_f32(_: &[f32; 1][..1] as f32) -> float32x2_t;
    /// Load an array of two `f32` elements and replicate to lanes of two registers.
    fn vld2_dup_f32(_: &[f32; 2][..1] as [f32; 2]) -> float32x2x2_t;
    /// Load an array of three `f32` elements and replicate to lanes of three registers.
    fn vld3_dup_f32(_: &[f32; 3][..1] as [f32; 3]) -> float32x2x3_t;
    /// Load an array of four `f32` elements and replicate to lanes of four registers.
    fn vld4_dup_f32(_: &[f32; 4][..1] as [f32; 4]) -> float32x2x4_t;

    /// Load one single-element `i64` and replicate to all lanes.
    fn vld1_dup_s64(_: &[i64; 1][..1] as i64) -> int64x1_t;
    /// Load an array of two `i64` elements and replicate to lanes of two registers.
    fn vld2_dup_s64(_: &[i64; 2][..1] as [i64; 2]) -> int64x1x2_t;
    /// Load an array of three `i64` elements and replicate to lanes of three registers.
    fn vld3_dup_s64(_: &[i64; 3][..1] as [i64; 3]) -> int64x1x3_t;
    /// Load an array of four `i64` elements and replicate to lanes of four registers.
    fn vld4_dup_s64(_: &[i64; 4][..1] as [i64; 4]) -> int64x1x4_t;

    /// Load one single-element `u64` and replicate to all lanes.
    fn vld1_dup_u64(_: &[u64; 1][..1] as u64) -> uint64x1_t;
    /// Load an array of two `u64` elements and replicate to lanes of two registers.
    fn vld2_dup_u64(_: &[u64; 2][..1] as [u64; 2]) -> uint64x1x2_t;
    /// Load an array of three `u64` elements and replicate to lanes of three registers.
    fn vld3_dup_u64(_: &[u64; 3][..1] as [u64; 3]) -> uint64x1x3_t;
    /// Load an array of four `u64` elements and replicate to lanes of four registers.
    fn vld4_dup_u64(_: &[u64; 4][..1] as [u64; 4]) -> uint64x1x4_t;

    /// Load one single-element `f64` and replicate to all lanes.
    fn vld1_dup_f64(_: &[f64; 1][..1] as f64) -> float64x1_t;
    /// Load an array of two `f64` elements and replicate to lanes of two registers.
    fn vld2_dup_f64(_: &[f64; 2][..1] as [f64; 2]) -> float64x1x2_t;
    /// Load an array of three `f64` elements and replicate to lanes of three registers.
    fn vld3_dup_f64(_: &[f64; 3][..1] as [f64; 3]) -> float64x1x3_t;
    /// Load an array of four `f64` elements and replicate to lanes of four registers.
    fn vld4_dup_f64(_: &[f64; 4][..1] as [f64; 4]) -> float64x1x4_t;
}

neon_load_store! {
    unsafe: load;
    size: various_sizes;

    /// Load one single-element `i8` and replicate to all lanes.
    fn vld1q_dup_s8(_: &[i8; 1][..1] as i8) -> int8x16_t;
    /// Load an array of two `i8` elements and replicate to lanes of two registers.
    fn vld2q_dup_s8(_: &[i8; 2][..1] as [i8; 2]) -> int8x16x2_t;
    /// Load an array of three `i8` elements and replicate to lanes of three registers.
    fn vld3q_dup_s8(_: &[i8; 3][..1] as [i8; 3]) -> int8x16x3_t;
    /// Load an array of four `i8` elements and replicate to lanes of four registers.
    fn vld4q_dup_s8(_: &[i8; 4][..1] as [i8; 4]) -> int8x16x4_t;

    /// Load one single-element `u8` and replicate to all lanes.
    fn vld1q_dup_u8(_: &[u8; 1][..1] as u8) -> uint8x16_t;
    /// Load an array of two `u8` elements and replicate to lanes of two registers.
    fn vld2q_dup_u8(_: &[u8; 2][..1] as [u8; 2]) -> uint8x16x2_t;
    /// Load an array of three `u8` elements and replicate to lanes of three registers.
    fn vld3q_dup_u8(_: &[u8; 3][..1] as [u8; 3]) -> uint8x16x3_t;
    /// Load an array of four `u8` elements and replicate to lanes of four registers.
    fn vld4q_dup_u8(_: &[u8; 4][..1] as [u8; 4]) -> uint8x16x4_t;

    /// Load one single-element `i16` and replicate to all lanes.
    fn vld1q_dup_s16(_: &[i16; 1][..1] as i16) -> int16x8_t;
    /// Load an array of two `i16` elements and replicate to lanes of two registers.
    fn vld2q_dup_s16(_: &[i16; 2][..1] as [i16; 2]) -> int16x8x2_t;
    /// Load an array of three `i16` elements and replicate to lanes of three registers.
    fn vld3q_dup_s16(_: &[i16; 3][..1] as [i16; 3]) -> int16x8x3_t;
    /// Load an array of four `i16` elements and replicate to lanes of four registers.
    fn vld4q_dup_s16(_: &[i16; 4][..1] as [i16; 4]) -> int16x8x4_t;

    /// Load one single-element `u16` and replicate to all lanes.
    fn vld1q_dup_u16(_: &[u16; 1][..1] as u16) -> uint16x8_t;
    /// Load an array of two `u16` elements and replicate to lanes of two registers.
    fn vld2q_dup_u16(_: &[u16; 2][..1] as [u16; 2]) -> uint16x8x2_t;
    /// Load an array of three `u16` elements and replicate to lanes of three registers.
    fn vld3q_dup_u16(_: &[u16; 3][..1] as [u16; 3]) -> uint16x8x3_t;
    /// Load an array of four `u16` elements and replicate to lanes of four registers.
    fn vld4q_dup_u16(_: &[u16; 4][..1] as [u16; 4]) -> uint16x8x4_t;

    /// Load one single-element `i32` and replicate to all lanes.
    fn vld1q_dup_s32(_: &[i32; 1][..1] as i32) -> int32x4_t;
    /// Load an array of two `i32` elements and replicate to lanes of two registers.
    fn vld2q_dup_s32(_: &[i32; 2][..1] as [i32; 2]) -> int32x4x2_t;
    /// Load an array of three `i32` elements and replicate to lanes of three registers.
    fn vld3q_dup_s32(_: &[i32; 3][..1] as [i32; 3]) -> int32x4x3_t;
    /// Load an array of four `i32` elements and replicate to lanes of four registers.
    fn vld4q_dup_s32(_: &[i32; 4][..1] as [i32; 4]) -> int32x4x4_t;

    /// Load one single-element `u32` and replicate to all lanes.
    fn vld1q_dup_u32(_: &[u32; 1][..1] as u32) -> uint32x4_t;
    /// Load an array of two `u32` elements and replicate to lanes of two registers.
    fn vld2q_dup_u32(_: &[u32; 2][..1] as [u32; 2]) -> uint32x4x2_t;
    /// Load an array of three `u32` elements and replicate to lanes of three registers.
    fn vld3q_dup_u32(_: &[u32; 3][..1] as [u32; 3]) -> uint32x4x3_t;
    /// Load an array of four `u32` elements and replicate to lanes of four registers.
    fn vld4q_dup_u32(_: &[u32; 4][..1] as [u32; 4]) -> uint32x4x4_t;

    /// Load one single-element `f32` and replicate to all lanes.
    fn vld1q_dup_f32(_: &[f32; 1][..1] as f32) -> float32x4_t;
    /// Load an array of two `f32` elements and replicate to lanes of two registers.
    fn vld2q_dup_f32(_: &[f32; 2][..1] as [f32; 2]) -> float32x4x2_t;
    /// Load an array of three `f32` elements and replicate to lanes of three registers.
    fn vld3q_dup_f32(_: &[f32; 3][..1] as [f32; 3]) -> float32x4x3_t;
    /// Load an array of four `f32` elements and replicate to lanes of four registers.
    fn vld4q_dup_f32(_: &[f32; 4][..1] as [f32; 4]) -> float32x4x4_t;

    /// Load one single-element `i64` and replicate to all lanes.
    fn vld1q_dup_s64(_: &[i64; 1][..1] as i64) -> int64x2_t;
    /// Load an array of two `i64` elements and replicate to lanes of two registers.
    fn vld2q_dup_s64(_: &[i64; 2][..1] as [i64; 2]) -> int64x2x2_t;
    /// Load an array of three `i64` elements and replicate to lanes of three registers.
    fn vld3q_dup_s64(_: &[i64; 3][..1] as [i64; 3]) -> int64x2x3_t;
    /// Load an array of four `i64` elements and replicate to lanes of four registers.
    fn vld4q_dup_s64(_: &[i64; 4][..1] as [i64; 4]) -> int64x2x4_t;

    /// Load one single-element `u64` and replicate to all lanes.
    fn vld1q_dup_u64(_: &[u64; 1][..1] as u64) -> uint64x2_t;
    /// Load an array of two `u64` elements and replicate to lanes of two registers.
    fn vld2q_dup_u64(_: &[u64; 2][..1] as [u64; 2]) -> uint64x2x2_t;
    /// Load an array of three `u64` elements and replicate to lanes of three registers.
    fn vld3q_dup_u64(_: &[u64; 3][..1] as [u64; 3]) -> uint64x2x3_t;
    /// Load an array of four `u64` elements and replicate to lanes of four registers.
    fn vld4q_dup_u64(_: &[u64; 4][..1] as [u64; 4]) -> uint64x2x4_t;

    /// Load one single-element `f64` and replicate to all lanes.
    fn vld1q_dup_f64(_: &[f64; 1][..1] as f64) -> float64x2_t;
    /// Load an array of two `f64` elements and replicate to lanes of two registers.
    fn vld2q_dup_f64(_: &[f64; 2][..1] as [f64; 2]) -> float64x2x2_t;
    /// Load an array of three `f64` elements and replicate to lanes of three registers.
    fn vld3q_dup_f64(_: &[f64; 3][..1] as [f64; 3]) -> float64x2x3_t;
    /// Load an array of four `f64` elements and replicate to lanes of four registers.
    fn vld4q_dup_f64(_: &[f64; 4][..1] as [f64; 4]) -> float64x2x4_t;
}


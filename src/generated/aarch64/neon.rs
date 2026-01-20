//! Token-gated wrappers for `#[target_feature(enable = "neon")]` functions.
//!
//! This module contains NEON load/store functions that are safe to call when you have a [`NeonToken`].
//!
//! **Auto-generated** from safe_unaligned_simd v0.2.3 macro invocations - do not edit manually.
//! See `xtask/src/main.rs` for generation notes.

#![allow(unused_imports)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::missing_safety_doc)]

use core::arch::aarch64::*;
use crate::tokens::arm::NeonToken;

// Token-aware macro that generates wrappers calling safe_unaligned_simd::aarch64::*
// This has the same invocation syntax as safe_unaligned_simd's vld_n_replicate_k!
macro_rules! neon_load_store {
    (
        unsafe: $kind:ident;
        size: $size:ident;

        $(
            $(#[$meta:meta])* fn $intrinsic:ident(_: &[$base_ty:ty; $n:literal][..$len:literal] as $realty:ty) -> $ret:ty;
        )*
    ) => {
        $(
            neon_load_store!(
                @ $kind $(#[$meta])* $intrinsic [$realty] [$ret]
            );
        )*
    };

    // Load wrapper
    (@ load $(#[$meta:meta])* $intrinsic:ident [$realty:ty] [$ret:ty]) => {
        $(#[$meta])*
        #[inline(always)]
        pub fn $intrinsic(_token: NeonToken, from: &$realty) -> $ret {
            #[inline]
            #[target_feature(enable = "neon")]
            unsafe fn inner(from: &$realty) -> $ret {
                safe_unaligned_simd::aarch64::$intrinsic(from)
            }
            // SAFETY: Token proves the target features are available
            unsafe { inner(from) }
        }
    };

    // Store wrapper
    (@ store $(#[$meta:meta])* $intrinsic:ident [$realty:ty] [$ret:ty]) => {
        $(#[$meta])*
        #[inline(always)]
        pub fn $intrinsic(_token: NeonToken, into: &mut $realty, val: $ret) {
            #[inline]
            #[target_feature(enable = "neon")]
            unsafe fn inner(into: &mut $realty, val: $ret) {
                safe_unaligned_simd::aarch64::$intrinsic(into, val)
            }
            // SAFETY: Token proves the target features are available
            unsafe { inner(into, val) }
        }
    };
}

// ============================================================================
// 8-byte register loads (vld1_*)
// ============================================================================

neon_load_store! {
    unsafe: load;
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

// ============================================================================
// 16-byte register loads (vld1q_*)
// ============================================================================

neon_load_store! {
    unsafe: load;
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

    /// Load three arrays of 16 `u8` values to three 16-byte registers.
    fn vld1q_u8_x3(_: &[u8; 16][..3] as [[u8; 16]; 3]) -> uint8x16x3_t;
    /// Load three arrays of 16 `i8` values to three 16-byte registers.
    fn vld1q_s8_x3(_: &[i8; 16][..3] as [[i8; 16]; 3]) -> int8x16x3_t;
    /// Load three arrays of 8 `u16` values to three 16-byte registers.
    fn vld1q_u16_x3(_: &[u16; 8][..3] as [[u16; 8]; 3]) -> uint16x8x3_t;
    /// Load three arrays of 8 `i16` values to three 16-byte registers.
    fn vld1q_s16_x3(_: &[i16; 8][..3] as [[i16; 8]; 3]) -> int16x8x3_t;
    /// Load three arrays of 4 `u32` values to three 16-byte registers.
    fn vld1q_u32_x3(_: &[u32; 4][..3] as [[u32; 4]; 3]) -> uint32x4x3_t;
    /// Load three arrays of 4 `i32` values to three 16-byte registers.
    fn vld1q_s32_x3(_: &[i32; 4][..3] as [[i32; 4]; 3]) -> int32x4x3_t;
    /// Load three arrays of 4 `f32` values to three 16-byte registers.
    fn vld1q_f32_x3(_: &[f32; 4][..3] as [[f32; 4]; 3]) -> float32x4x3_t;
    /// Load three arrays of 2 `u64` value to three 16-byte registers.
    fn vld1q_u64_x3(_: &[u64; 2][..3] as [[u64; 2]; 3]) -> uint64x2x3_t;
    /// Load three arrays of 2 `i64` value to three 16-byte registers.
    fn vld1q_s64_x3(_: &[i64; 2][..3] as [[i64; 2]; 3]) -> int64x2x3_t;
    /// Load three arrays of 2 `f64` value to three 16-byte registers.
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

// ============================================================================
// 8-byte register stores (vst1_*)
// ============================================================================

neon_load_store! {
    unsafe: store;
    size: assert_size_8bytes;

    /// Store 8 `u8` values from one 8-byte register.
    fn vst1_u8(_: &[u8; 8][..1] as [u8; 8]) -> uint8x8_t;
    /// Store 8 `i8` values from one 8-byte register.
    fn vst1_s8(_: &[i8; 8][..1] as [i8; 8]) -> int8x8_t;
    /// Store 4 `u16` values from one 8-byte register.
    fn vst1_u16(_: &[u16; 4][..1] as [u16; 4]) -> uint16x4_t;
    /// Store 4 `i16` values from one 8-byte register.
    fn vst1_s16(_: &[i16; 4][..1] as [i16; 4]) -> int16x4_t;
    /// Store 2 `u32` values from one 8-byte register.
    fn vst1_u32(_: &[u32; 2][..1] as [u32; 2]) -> uint32x2_t;
    /// Store 2 `i32` values from one 8-byte register.
    fn vst1_s32(_: &[i32; 2][..1] as [i32; 2]) -> int32x2_t;
    /// Store 2 `f32` values from one 8-byte register.
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
    /// Store three `u64` values from three 8-byte registers.
    fn vst1_u64_x3(_: &[u64; 1][..3] as [u64; 3]) -> uint64x1x3_t;
    /// Store three `i64` values from three 8-byte registers.
    fn vst1_s64_x3(_: &[i64; 1][..3] as [i64; 3]) -> int64x1x3_t;
    /// Store three `f64` values from three 8-byte registers.
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
    /// Store four `u64` values from four 8-byte registers.
    fn vst1_u64_x4(_: &[u64; 1][..4] as [u64; 4]) -> uint64x1x4_t;
    /// Store four `i64` values from four 8-byte registers.
    fn vst1_s64_x4(_: &[i64; 1][..4] as [i64; 4]) -> int64x1x4_t;
    /// Store four `f64` values from four 8-byte registers.
    fn vst1_f64_x4(_: &[f64; 1][..4] as [f64; 4]) -> float64x1x4_t;
}

// ============================================================================
// 16-byte register stores (vst1q_*)
// ============================================================================

neon_load_store! {
    unsafe: store;
    size: assert_size_16bytes;

    /// Store 16 `u8` values from one 16-byte register.
    fn vst1q_u8(_: &[u8; 16][..1] as [u8; 16]) -> uint8x16_t;
    /// Store 16 `i8` values from one 16-byte register.
    fn vst1q_s8(_: &[i8; 16][..1] as [i8; 16]) -> int8x16_t;
    /// Store 8 `u16` values from one 16-byte register.
    fn vst1q_u16(_: &[u16; 8][..1] as [u16; 8]) -> uint16x8_t;
    /// Store 8 `i16` values from one 16-byte register.
    fn vst1q_s16(_: &[i16; 8][..1] as [i16; 8]) -> int16x8_t;
    /// Store 4 `u32` values from one 16-byte register.
    fn vst1q_u32(_: &[u32; 4][..1] as [u32; 4]) -> uint32x4_t;
    /// Store 4 `i32` values from one 16-byte register.
    fn vst1q_s32(_: &[i32; 4][..1] as [i32; 4]) -> int32x4_t;
    /// Store 4 `f32` values from one 16-byte register.
    fn vst1q_f32(_: &[f32; 4][..1] as [f32; 4]) -> float32x4_t;
    /// Store 2 `u64` values from one 16-byte register.
    fn vst1q_u64(_: &[u64; 2][..1] as [u64; 2]) -> uint64x2_t;
    /// Store 2 `i64` values from one 16-byte register.
    fn vst1q_s64(_: &[i64; 2][..1] as [i64; 2]) -> int64x2_t;
    /// Store 2 `f64` values from one 16-byte register.
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

    /// Store three arrays of 16 `u8` values from three 16-byte registers.
    fn vst1q_u8_x3(_: &[u8; 16][..3] as [[u8; 16]; 3]) -> uint8x16x3_t;
    /// Store three arrays of 16 `i8` values from three 16-byte registers.
    fn vst1q_s8_x3(_: &[i8; 16][..3] as [[i8; 16]; 3]) -> int8x16x3_t;
    /// Store three arrays of 8 `u16` values from three 16-byte registers.
    fn vst1q_u16_x3(_: &[u16; 8][..3] as [[u16; 8]; 3]) -> uint16x8x3_t;
    /// Store three arrays of 8 `i16` values from three 16-byte registers.
    fn vst1q_s16_x3(_: &[i16; 8][..3] as [[i16; 8]; 3]) -> int16x8x3_t;
    /// Store three arrays of 4 `u32` values from three 16-byte registers.
    fn vst1q_u32_x3(_: &[u32; 4][..3] as [[u32; 4]; 3]) -> uint32x4x3_t;
    /// Store three arrays of 4 `i32` values from three 16-byte registers.
    fn vst1q_s32_x3(_: &[i32; 4][..3] as [[i32; 4]; 3]) -> int32x4x3_t;
    /// Store three arrays of 4 `f32` values from three 16-byte registers.
    fn vst1q_f32_x3(_: &[f32; 4][..3] as [[f32; 4]; 3]) -> float32x4x3_t;
    /// Store three arrays of 2 `u64` value from three 16-byte registers.
    fn vst1q_u64_x3(_: &[u64; 2][..3] as [[u64; 2]; 3]) -> uint64x2x3_t;
    /// Store three arrays of 2 `i64` value from three 16-byte registers.
    fn vst1q_s64_x3(_: &[i64; 2][..3] as [[i64; 2]; 3]) -> int64x2x3_t;
    /// Store three arrays of 2 `f64` value from three 16-byte registers.
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

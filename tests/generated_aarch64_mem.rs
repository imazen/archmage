//! Auto-generated exhaustive tests for aarch64 mem module intrinsics.
//!
//! This file exercises every intrinsic in `archmage::mem::neon` to ensure they compile
//! and execute correctly on supported hardware.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

#![cfg(target_arch = "aarch64")]
#![allow(unused_variables)]

use std::hint::black_box;
use core::arch::aarch64::*;

use archmage::SimdToken;
use archmage::mem::neon;

/// Test all NEON load intrinsics
#[test]
fn test_neon_load_intrinsics_exhaustive() {
    use archmage::NeonToken;

    let Some(token) = NeonToken::try_new() else {
        eprintln!("NEON not available, skipping test");
        return;
    };

    // Test data for various element types and sizes
    let u8_8: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    let u8_16: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let u16_4: [u16; 4] = [1, 2, 3, 4];
    let u16_8: [u16; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    let u32_2: [u32; 2] = [1, 2];
    let u32_4: [u32; 4] = [1, 2, 3, 4];
    let u64_1: [u64; 1] = [1];
    let u64_2: [u64; 2] = [1, 2];
    let f32_2: [f32; 2] = [1.0, 2.0];
    let f32_4: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let f64_1: [f64; 1] = [1.0];
    let f64_2: [f64; 2] = [1.0, 2.0];

    let i8_8: [i8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    let i8_16: [i8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let i16_4: [i16; 4] = [1, 2, 3, 4];
    let i16_8: [i16; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    let i32_2: [i32; 2] = [1, 2];
    let i32_4: [i32; 4] = [1, 2, 3, 4];
    let i64_1: [i64; 1] = [1];
    let i64_2: [i64; 2] = [1, 2];

    // Exercise load intrinsics
    let v = neon::vld1_u8(token, &u8_8); black_box(v);
    let v = neon::vld1q_u8(token, &u8_16); black_box(v);
    let v = neon::vld1_u16(token, &u16_4); black_box(v);
    let v = neon::vld1q_u16(token, &u16_8); black_box(v);
    let v = neon::vld1_u32(token, &u32_2); black_box(v);
    let v = neon::vld1q_u32(token, &u32_4); black_box(v);
    let v = neon::vld1_u64(token, &u64_1); black_box(v);
    let v = neon::vld1q_u64(token, &u64_2); black_box(v);
    let v = neon::vld1_s8(token, &i8_8); black_box(v);
    let v = neon::vld1q_s8(token, &i8_16); black_box(v);
    let v = neon::vld1_s16(token, &i16_4); black_box(v);
    let v = neon::vld1q_s16(token, &i16_8); black_box(v);
    let v = neon::vld1_s32(token, &i32_2); black_box(v);
    let v = neon::vld1q_s32(token, &i32_4); black_box(v);
    let v = neon::vld1_s64(token, &i64_1); black_box(v);
    let v = neon::vld1q_s64(token, &i64_2); black_box(v);
    let v = neon::vld1_f32(token, &f32_2); black_box(v);
    let v = neon::vld1q_f32(token, &f32_4); black_box(v);
    let v = neon::vld1_f64(token, &f64_1); black_box(v);
    let v = neon::vld1q_f64(token, &f64_2); black_box(v);
}

/// Test all NEON store intrinsics
#[test]
fn test_neon_store_intrinsics_exhaustive() {
    use archmage::NeonToken;

    let Some(token) = NeonToken::try_new() else {
        eprintln!("NEON not available, skipping test");
        return;
    };

    // Input data
    let u8_8: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    let u8_16: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let u16_4: [u16; 4] = [1, 2, 3, 4];
    let u16_8: [u16; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    let u32_2: [u32; 2] = [1, 2];
    let u32_4: [u32; 4] = [1, 2, 3, 4];
    let u64_1: [u64; 1] = [1];
    let u64_2: [u64; 2] = [1, 2];
    let f32_2: [f32; 2] = [1.0, 2.0];
    let f32_4: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let f64_1: [f64; 1] = [1.0];
    let f64_2: [f64; 2] = [1.0, 2.0];

    let i8_8: [i8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    let i8_16: [i8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let i16_4: [i16; 4] = [1, 2, 3, 4];
    let i16_8: [i16; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    let i32_2: [i32; 2] = [1, 2];
    let i32_4: [i32; 4] = [1, 2, 3, 4];
    let i64_1: [i64; 1] = [1];
    let i64_2: [i64; 2] = [1, 2];

    // Output buffers
    let mut out_u8_8: [u8; 8] = [0; 8];
    let mut out_u8_16: [u8; 16] = [0; 16];
    let mut out_u16_4: [u16; 4] = [0; 4];
    let mut out_u16_8: [u16; 8] = [0; 8];
    let mut out_u32_2: [u32; 2] = [0; 2];
    let mut out_u32_4: [u32; 4] = [0; 4];
    let mut out_u64_1: [u64; 1] = [0; 1];
    let mut out_u64_2: [u64; 2] = [0; 2];
    let mut out_f32_2: [f32; 2] = [0.0; 2];
    let mut out_f32_4: [f32; 4] = [0.0; 4];
    let mut out_f64_1: [f64; 1] = [0.0; 1];
    let mut out_f64_2: [f64; 2] = [0.0; 2];

    let mut out_i8_8: [i8; 8] = [0; 8];
    let mut out_i8_16: [i8; 16] = [0; 16];
    let mut out_i16_4: [i16; 4] = [0; 4];
    let mut out_i16_8: [i16; 8] = [0; 8];
    let mut out_i32_2: [i32; 2] = [0; 2];
    let mut out_i32_4: [i32; 4] = [0; 4];
    let mut out_i64_1: [i64; 1] = [0; 1];
    let mut out_i64_2: [i64; 2] = [0; 2];

    // Load-store round trips
    let v = neon::vld1_u8(token, &u8_8); neon::vst1_u8(token, &mut out_u8_8, v); assert_eq!(u8_8, out_u8_8);
    let v = neon::vld1q_u8(token, &u8_16); neon::vst1q_u8(token, &mut out_u8_16, v); assert_eq!(u8_16, out_u8_16);
    let v = neon::vld1_u16(token, &u16_4); neon::vst1_u16(token, &mut out_u16_4, v); assert_eq!(u16_4, out_u16_4);
    let v = neon::vld1q_u16(token, &u16_8); neon::vst1q_u16(token, &mut out_u16_8, v); assert_eq!(u16_8, out_u16_8);
    let v = neon::vld1_u32(token, &u32_2); neon::vst1_u32(token, &mut out_u32_2, v); assert_eq!(u32_2, out_u32_2);
    let v = neon::vld1q_u32(token, &u32_4); neon::vst1q_u32(token, &mut out_u32_4, v); assert_eq!(u32_4, out_u32_4);
    let v = neon::vld1_u64(token, &u64_1); neon::vst1_u64(token, &mut out_u64_1, v); assert_eq!(u64_1, out_u64_1);
    let v = neon::vld1q_u64(token, &u64_2); neon::vst1q_u64(token, &mut out_u64_2, v); assert_eq!(u64_2, out_u64_2);
    let v = neon::vld1_s8(token, &i8_8); neon::vst1_s8(token, &mut out_i8_8, v); assert_eq!(i8_8, out_i8_8);
    let v = neon::vld1q_s8(token, &i8_16); neon::vst1q_s8(token, &mut out_i8_16, v); assert_eq!(i8_16, out_i8_16);
    let v = neon::vld1_s16(token, &i16_4); neon::vst1_s16(token, &mut out_i16_4, v); assert_eq!(i16_4, out_i16_4);
    let v = neon::vld1q_s16(token, &i16_8); neon::vst1q_s16(token, &mut out_i16_8, v); assert_eq!(i16_8, out_i16_8);
    let v = neon::vld1_s32(token, &i32_2); neon::vst1_s32(token, &mut out_i32_2, v); assert_eq!(i32_2, out_i32_2);
    let v = neon::vld1q_s32(token, &i32_4); neon::vst1q_s32(token, &mut out_i32_4, v); assert_eq!(i32_4, out_i32_4);
    let v = neon::vld1_s64(token, &i64_1); neon::vst1_s64(token, &mut out_i64_1, v); assert_eq!(i64_1, out_i64_1);
    let v = neon::vld1q_s64(token, &i64_2); neon::vst1q_s64(token, &mut out_i64_2, v); assert_eq!(i64_2, out_i64_2);
    let v = neon::vld1_f32(token, &f32_2); neon::vst1_f32(token, &mut out_f32_2, v); assert_eq!(f32_2, out_f32_2);
    let v = neon::vld1q_f32(token, &f32_4); neon::vst1q_f32(token, &mut out_f32_4, v); assert_eq!(f32_4, out_f32_4);
    let v = neon::vld1_f64(token, &f64_1); neon::vst1_f64(token, &mut out_f64_1, v); assert_eq!(f64_1, out_f64_1);
    let v = neon::vld1q_f64(token, &f64_2); neon::vst1q_f64(token, &mut out_f64_2, v); assert_eq!(f64_2, out_f64_2);
}

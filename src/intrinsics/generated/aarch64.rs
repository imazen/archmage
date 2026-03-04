//! Combined `core::arch` + `safe_unaligned_simd` intrinsics for `aarch64`.
//!
//! **Auto-generated** by `cargo xtask generate` — do not edit manually.
//!
//! This module glob-imports all of `core::arch::aarch64` (types, value intrinsics,
//! and unsafe memory ops), then explicitly re-exports the safe reference-based
//! memory operations from `safe_unaligned_simd`. Rust's name resolution rules
//! make explicit imports shadow glob imports, so `_mm256_loadu_ps` etc. resolve
//! to the safe versions automatically.

#[allow(unused_imports)]
pub use core::arch::aarch64::*;

#[allow(unused_imports)]
pub use safe_unaligned_simd::aarch64::{
    vld1_dup_f32, vld1_dup_f64, vld1_dup_s8, vld1_dup_s16, vld1_dup_s32, vld1_dup_s64, vld1_dup_u8,
    vld1_dup_u16, vld1_dup_u32, vld1_dup_u64, vld1_f32, vld1_f32_x2, vld1_f32_x3, vld1_f32_x4,
    vld1_f64, vld1_f64_x2, vld1_f64_x3, vld1_f64_x4, vld1_s8, vld1_s8_x2, vld1_s8_x3, vld1_s8_x4,
    vld1_s16, vld1_s16_x2, vld1_s16_x3, vld1_s16_x4, vld1_s32, vld1_s32_x2, vld1_s32_x3,
    vld1_s32_x4, vld1_s64, vld1_s64_x2, vld1_s64_x3, vld1_s64_x4, vld1_u8, vld1_u8_x2, vld1_u8_x3,
    vld1_u8_x4, vld1_u16, vld1_u16_x2, vld1_u16_x3, vld1_u16_x4, vld1_u32, vld1_u32_x2,
    vld1_u32_x3, vld1_u32_x4, vld1_u64, vld1_u64_x2, vld1_u64_x3, vld1_u64_x4, vld1q_dup_f32,
    vld1q_dup_f64, vld1q_dup_s8, vld1q_dup_s16, vld1q_dup_s32, vld1q_dup_s64, vld1q_dup_u8,
    vld1q_dup_u16, vld1q_dup_u32, vld1q_dup_u64, vld1q_f32, vld1q_f32_x2, vld1q_f32_x3,
    vld1q_f32_x4, vld1q_f64, vld1q_f64_x2, vld1q_f64_x3, vld1q_f64_x4, vld1q_s8, vld1q_s8_x2,
    vld1q_s8_x3, vld1q_s8_x4, vld1q_s16, vld1q_s16_x2, vld1q_s16_x3, vld1q_s16_x4, vld1q_s32,
    vld1q_s32_x2, vld1q_s32_x3, vld1q_s32_x4, vld1q_s64, vld1q_s64_x2, vld1q_s64_x3, vld1q_s64_x4,
    vld1q_u8, vld1q_u8_x2, vld1q_u8_x3, vld1q_u8_x4, vld1q_u16, vld1q_u16_x2, vld1q_u16_x3,
    vld1q_u16_x4, vld1q_u32, vld1q_u32_x2, vld1q_u32_x3, vld1q_u32_x4, vld1q_u64, vld1q_u64_x2,
    vld1q_u64_x3, vld1q_u64_x4, vld2_dup_f32, vld2_dup_f64, vld2_dup_s8, vld2_dup_s16,
    vld2_dup_s32, vld2_dup_s64, vld2_dup_u8, vld2_dup_u16, vld2_dup_u32, vld2_dup_u64,
    vld2q_dup_f32, vld2q_dup_f64, vld2q_dup_s8, vld2q_dup_s16, vld2q_dup_s32, vld2q_dup_s64,
    vld2q_dup_u8, vld2q_dup_u16, vld2q_dup_u32, vld2q_dup_u64, vld2q_f32, vld2q_f64, vld2q_s8,
    vld2q_s16, vld2q_s32, vld2q_s64, vld2q_u8, vld2q_u16, vld2q_u32, vld2q_u64, vld3_dup_f32,
    vld3_dup_f64, vld3_dup_s8, vld3_dup_s16, vld3_dup_s32, vld3_dup_s64, vld3_dup_u8, vld3_dup_u16,
    vld3_dup_u32, vld3_dup_u64, vld3q_dup_f32, vld3q_dup_f64, vld3q_dup_s8, vld3q_dup_s16,
    vld3q_dup_s32, vld3q_dup_s64, vld3q_dup_u8, vld3q_dup_u16, vld3q_dup_u32, vld3q_dup_u64,
    vld3q_f32, vld3q_f64, vld3q_s8, vld3q_s16, vld3q_s32, vld3q_s64, vld3q_u8, vld3q_u16,
    vld3q_u32, vld3q_u64, vld4_dup_f32, vld4_dup_f64, vld4_dup_s8, vld4_dup_s16, vld4_dup_s32,
    vld4_dup_s64, vld4_dup_u8, vld4_dup_u16, vld4_dup_u32, vld4_dup_u64, vld4q_dup_f32,
    vld4q_dup_f64, vld4q_dup_s8, vld4q_dup_s16, vld4q_dup_s32, vld4q_dup_s64, vld4q_dup_u8,
    vld4q_dup_u16, vld4q_dup_u32, vld4q_dup_u64, vld4q_f32, vld4q_f64, vld4q_s8, vld4q_s16,
    vld4q_s32, vld4q_s64, vld4q_u8, vld4q_u16, vld4q_u32, vld4q_u64, vst1_f32, vst1_f32_x2,
    vst1_f32_x3, vst1_f32_x4, vst1_f64, vst1_f64_x2, vst1_f64_x3, vst1_f64_x4, vst1_s8, vst1_s8_x2,
    vst1_s8_x3, vst1_s8_x4, vst1_s16, vst1_s16_x2, vst1_s16_x3, vst1_s16_x4, vst1_s32, vst1_s32_x2,
    vst1_s32_x3, vst1_s32_x4, vst1_s64, vst1_s64_x2, vst1_s64_x3, vst1_s64_x4, vst1_u8, vst1_u8_x2,
    vst1_u8_x3, vst1_u8_x4, vst1_u16, vst1_u16_x2, vst1_u16_x3, vst1_u16_x4, vst1_u32, vst1_u32_x2,
    vst1_u32_x3, vst1_u32_x4, vst1_u64, vst1_u64_x2, vst1_u64_x3, vst1_u64_x4, vst1q_f32,
    vst1q_f32_x2, vst1q_f32_x3, vst1q_f32_x4, vst1q_f64, vst1q_f64_x2, vst1q_f64_x3, vst1q_f64_x4,
    vst1q_s8, vst1q_s8_x2, vst1q_s8_x3, vst1q_s8_x4, vst1q_s16, vst1q_s16_x2, vst1q_s16_x3,
    vst1q_s16_x4, vst1q_s32, vst1q_s32_x2, vst1q_s32_x3, vst1q_s32_x4, vst1q_s64, vst1q_s64_x2,
    vst1q_s64_x3, vst1q_s64_x4, vst1q_u8, vst1q_u8_x2, vst1q_u8_x3, vst1q_u8_x4, vst1q_u16,
    vst1q_u16_x2, vst1q_u16_x3, vst1q_u16_x4, vst1q_u32, vst1q_u32_x2, vst1q_u32_x3, vst1q_u32_x4,
    vst1q_u64, vst1q_u64_x2, vst1q_u64_x3, vst1q_u64_x4, vst2q_f32, vst2q_f64, vst2q_s8, vst2q_s16,
    vst2q_s32, vst2q_s64, vst2q_u8, vst2q_u16, vst2q_u32, vst2q_u64, vst3q_f32, vst3q_f64,
    vst3q_s8, vst3q_s16, vst3q_s32, vst3q_s64, vst3q_u8, vst3q_u16, vst3q_u32, vst3q_u64,
    vst4q_f32, vst4q_f64, vst4q_s8, vst4q_s16, vst4q_s32, vst4q_s64, vst4q_u8, vst4q_u16,
    vst4q_u32, vst4q_u64,
};

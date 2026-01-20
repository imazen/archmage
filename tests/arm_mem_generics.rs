//! Tests that ARM mem module handles generic trait bounds properly.
//!
//! Verifies that:
//! - `mem::neon` functions accept `impl HasNeon`
//! - All tokens implementing `HasNeon` work with mem functions
//! - Generic functions can use any `HasNeon` token

#![cfg(target_arch = "aarch64")]
#![cfg(feature = "safe_unaligned_simd")]

use archmage::Arm64;
use archmage::mem::neon;
use archmage::tokens::{HasNeon, NeonToken, SimdToken, Sve2Token, SveToken};

// ============================================================================
// Test that NeonToken works with mem::neon
// ============================================================================

#[test]
fn test_neon_token_with_mem_load() {
    if let Some(token) = NeonToken::try_new() {
        let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let _v = neon::vld1q_f32(token, &data);
    }
}

#[test]
fn test_neon_token_with_mem_store() {
    if let Some(token) = NeonToken::try_new() {
        let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let v = neon::vld1q_f32(token, &data);
        let mut out: [f32; 4] = [0.0; 4];
        neon::vst1q_f32(token, &mut out, v);
        assert_eq!(out, data);
    }
}

// ============================================================================
// Test that Arm64 alias works (it's an alias for NeonToken)
// ============================================================================

#[test]
fn test_arm64_alias_with_mem() {
    if let Some(token) = Arm64::try_new() {
        let data: [i32; 4] = [1, 2, 3, 4];
        let v = neon::vld1q_s32(token, &data);
        let mut out: [i32; 4] = [0; 4];
        neon::vst1q_s32(token, &mut out, v);
        assert_eq!(out, data);
    }
}

// ============================================================================
// Test that SveToken works with mem::neon (SVE implies NEON)
// ============================================================================

#[test]
fn test_sve_token_with_mem_neon() {
    // SVE isn't available on all hardware, but if it is, it should work
    if let Some(token) = SveToken::try_new() {
        let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let v = neon::vld1q_f32(token, &data);
        let mut out: [f32; 4] = [0.0; 4];
        neon::vst1q_f32(token, &mut out, v);
        assert_eq!(out, data);
    }
}

// ============================================================================
// Test that Sve2Token works with mem::neon (SVE2 implies NEON)
// ============================================================================

#[test]
fn test_sve2_token_with_mem_neon() {
    // SVE2 isn't available on all hardware, but if it is, it should work
    if let Some(token) = Sve2Token::try_new() {
        let data: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let v = neon::vld1q_u8(token, &data);
        let mut out: [u8; 16] = [0; 16];
        neon::vst1q_u8(token, &mut out, v);
        assert_eq!(out, data);
    }
}

// ============================================================================
// Test generic functions that accept impl HasNeon
// ============================================================================

/// A generic function that accepts any token implementing HasNeon
fn generic_load_store<T: HasNeon + Copy>(token: T, data: &[f32; 4]) -> [f32; 4] {
    let v = neon::vld1q_f32(token, data);
    let mut out: [f32; 4] = [0.0; 4];
    neon::vst1q_f32(token, &mut out, v);
    out
}

#[test]
fn test_generic_function_with_neon_token() {
    if let Some(token) = NeonToken::try_new() {
        let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let result = generic_load_store(token, &data);
        assert_eq!(result, data);
    }
}

#[test]
fn test_generic_function_with_sve_token() {
    if let Some(token) = SveToken::try_new() {
        let data: [f32; 4] = [5.0, 6.0, 7.0, 8.0];
        let result = generic_load_store(token, &data);
        assert_eq!(result, data);
    }
}

#[test]
fn test_generic_function_with_sve2_token() {
    if let Some(token) = Sve2Token::try_new() {
        let data: [f32; 4] = [9.0, 10.0, 11.0, 12.0];
        let result = generic_load_store(token, &data);
        assert_eq!(result, data);
    }
}

// ============================================================================
// Test that token can be passed through multiple function calls
// ============================================================================

fn inner_load<T: HasNeon>(token: T, data: &[i16; 8]) -> core::arch::aarch64::int16x8_t {
    neon::vld1q_s16(token, data)
}

fn inner_store<T: HasNeon>(token: T, out: &mut [i16; 8], v: core::arch::aarch64::int16x8_t) {
    neon::vst1q_s16(token, out, v)
}

fn outer_process<T: HasNeon + Copy>(token: T, data: &[i16; 8]) -> [i16; 8] {
    let v = inner_load(token, data);
    let mut out: [i16; 8] = [0; 8];
    inner_store(token, &mut out, v);
    out
}

#[test]
fn test_token_passthrough_with_neon() {
    if let Some(token) = NeonToken::try_new() {
        let data: [i16; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        let result = outer_process(token, &data);
        assert_eq!(result, data);
    }
}

// ============================================================================
// Test various data types
// ============================================================================

#[test]
fn test_various_types_u8() {
    if let Some(token) = NeonToken::try_new() {
        let data: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        let v = neon::vld1_u8(token, &data);
        let mut out: [u8; 8] = [0; 8];
        neon::vst1_u8(token, &mut out, v);
        assert_eq!(out, data);
    }
}

#[test]
fn test_various_types_i8() {
    if let Some(token) = NeonToken::try_new() {
        let data: [i8; 16] = [
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16,
        ];
        let v = neon::vld1q_s8(token, &data);
        let mut out: [i8; 16] = [0; 16];
        neon::vst1q_s8(token, &mut out, v);
        assert_eq!(out, data);
    }
}

#[test]
fn test_various_types_u16() {
    if let Some(token) = NeonToken::try_new() {
        let data: [u16; 8] = [100, 200, 300, 400, 500, 600, 700, 800];
        let v = neon::vld1q_u16(token, &data);
        let mut out: [u16; 8] = [0; 8];
        neon::vst1q_u16(token, &mut out, v);
        assert_eq!(out, data);
    }
}

#[test]
fn test_various_types_u32() {
    if let Some(token) = NeonToken::try_new() {
        let data: [u32; 4] = [1000, 2000, 3000, 4000];
        let v = neon::vld1q_u32(token, &data);
        let mut out: [u32; 4] = [0; 4];
        neon::vst1q_u32(token, &mut out, v);
        assert_eq!(out, data);
    }
}

#[test]
fn test_various_types_f64() {
    if let Some(token) = NeonToken::try_new() {
        let data: [f64; 2] = [1.5, 2.5];
        let v = neon::vld1q_f64(token, &data);
        let mut out: [f64; 2] = [0.0; 2];
        neon::vst1q_f64(token, &mut out, v);
        assert_eq!(out, data);
    }
}

// ============================================================================
// Test multi-register loads (vld1_*_x2, vld1_*_x3, vld1_*_x4)
// ============================================================================

#[test]
fn test_multi_register_x2() {
    if let Some(token) = NeonToken::try_new() {
        let data: [[f32; 4]; 2] = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];
        let v = neon::vld1q_f32_x2(token, &data);
        let mut out: [[f32; 4]; 2] = [[0.0; 4]; 2];
        neon::vst1q_f32_x2(token, &mut out, v);
        assert_eq!(out, data);
    }
}

#[test]
fn test_multi_register_x3() {
    if let Some(token) = NeonToken::try_new() {
        let data: [[i32; 4]; 3] = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]];
        let v = neon::vld1q_s32_x3(token, &data);
        let mut out: [[i32; 4]; 3] = [[0; 4]; 3];
        neon::vst1q_s32_x3(token, &mut out, v);
        assert_eq!(out, data);
    }
}

#[test]
fn test_multi_register_x4() {
    if let Some(token) = NeonToken::try_new() {
        let data: [[u32; 4]; 4] = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ];
        let v = neon::vld1q_u32_x4(token, &data);
        let mut out: [[u32; 4]; 4] = [[0; 4]; 4];
        neon::vst1q_u32_x4(token, &mut out, v);
        assert_eq!(out, data);
    }
}

// ============================================================================
// Test 64-bit (8-byte) register operations
// ============================================================================

#[test]
fn test_64bit_registers() {
    if let Some(token) = NeonToken::try_new() {
        // f32x2
        let data_f32: [f32; 2] = [1.5, 2.5];
        let v_f32 = neon::vld1_f32(token, &data_f32);
        let mut out_f32: [f32; 2] = [0.0; 2];
        neon::vst1_f32(token, &mut out_f32, v_f32);
        assert_eq!(out_f32, data_f32);

        // i32x2
        let data_i32: [i32; 2] = [100, -200];
        let v_i32 = neon::vld1_s32(token, &data_i32);
        let mut out_i32: [i32; 2] = [0; 2];
        neon::vst1_s32(token, &mut out_i32, v_i32);
        assert_eq!(out_i32, data_i32);

        // u16x4
        let data_u16: [u16; 4] = [1, 2, 3, 4];
        let v_u16 = neon::vld1_u16(token, &data_u16);
        let mut out_u16: [u16; 4] = [0; 4];
        neon::vst1_u16(token, &mut out_u16, v_u16);
        assert_eq!(out_u16, data_u16);
    }
}

// ============================================================================
// Test with where clause syntax
// ============================================================================

fn where_clause_generic<T>(token: T, data: &[f32; 4]) -> [f32; 4]
where
    T: HasNeon + Copy,
{
    let v = neon::vld1q_f32(token, data);
    let mut out: [f32; 4] = [0.0; 4];
    neon::vst1q_f32(token, &mut out, v);
    out
}

#[test]
fn test_where_clause_syntax() {
    if let Some(token) = NeonToken::try_new() {
        let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let result = where_clause_generic(token, &data);
        assert_eq!(result, data);
    }
}

// ============================================================================
// Test trait object usage (compile-time check)
// ============================================================================

// This doesn't work because HasNeon is not object-safe (no Sized constraint on
// impl bounds), but we can verify that the generated wrappers work with any
// concrete type implementing HasNeon. The `impl HasNeon` pattern in the
// generated code is the right approach for zero-cost generic dispatch.

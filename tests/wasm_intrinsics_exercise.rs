//! Comprehensive WASM SIMD128 intrinsic exercise tests for Wasm128Token.
//!
//! Tests ~100 WASM SIMD128 intrinsics to verify they compile and execute correctly
//! with archmage's Wasm128Token.
//!
//! Run with: RUSTFLAGS="-C target-feature=+simd128" cargo test --target wasm32-wasip1 --test wasm_intrinsics_exercise

#![cfg(target_arch = "wasm32")]
#![cfg(target_feature = "simd128")]
#![allow(unused_imports, unused_variables, dead_code)]
#![allow(clippy::eq_op, clippy::identity_op)]

use archmage::{SimdToken, Wasm128Token, arcane};
use core::arch::wasm32::*;
use core::hint::black_box;

#[test]
fn test_wasm128_intrinsics() {
    if let Some(token) = Wasm128Token::summon() {
        exercise_integer_arithmetic(token);
        exercise_float_arithmetic(token);
        exercise_comparisons(token);
        exercise_bitwise(token);
        exercise_shifts(token);
        exercise_conversions(token);
        exercise_lane_ops(token);
        exercise_boolean_reductions(token);
        exercise_shuffle_swizzle(token);
        exercise_extended_multiply(token);
        exercise_saturating_arithmetic(token);
        println!("All Wasm128Token intrinsic tests passed!");
    } else {
        println!("Wasm128Token not available - skipping tests");
    }
}

// =============================================================================
// Integer Arithmetic
// =============================================================================

#[arcane]
fn exercise_integer_arithmetic(token: Wasm128Token) {
    // i8x16
    let a_i8 = i8x16_splat(1);
    let b_i8 = i8x16_splat(2);
    black_box(i8x16_add(a_i8, b_i8));
    black_box(i8x16_sub(a_i8, b_i8));
    black_box(i8x16_neg(a_i8));
    black_box(i8x16_abs(a_i8));
    black_box(i8x16_min(a_i8, b_i8));
    black_box(u8x16_min(a_i8, b_i8));
    black_box(i8x16_max(a_i8, b_i8));
    black_box(u8x16_max(a_i8, b_i8));
    black_box(u8x16_avgr(a_i8, b_i8));

    // i16x8
    let a_i16 = i16x8_splat(100);
    let b_i16 = i16x8_splat(200);
    black_box(i16x8_add(a_i16, b_i16));
    black_box(i16x8_sub(a_i16, b_i16));
    black_box(i16x8_mul(a_i16, b_i16));
    black_box(i16x8_neg(a_i16));
    black_box(i16x8_abs(a_i16));
    black_box(i16x8_min(a_i16, b_i16));
    black_box(u16x8_min(a_i16, b_i16));
    black_box(i16x8_max(a_i16, b_i16));
    black_box(u16x8_max(a_i16, b_i16));
    black_box(u16x8_avgr(a_i16, b_i16));
    black_box(i16x8_q15mulr_sat(a_i16, b_i16));

    // i32x4
    let a_i32 = i32x4_splat(1000);
    let b_i32 = i32x4_splat(2000);
    black_box(i32x4_add(a_i32, b_i32));
    black_box(i32x4_sub(a_i32, b_i32));
    black_box(i32x4_mul(a_i32, b_i32));
    black_box(i32x4_neg(a_i32));
    black_box(i32x4_abs(a_i32));
    black_box(i32x4_min(a_i32, b_i32));
    black_box(u32x4_min(a_i32, b_i32));
    black_box(i32x4_max(a_i32, b_i32));
    black_box(u32x4_max(a_i32, b_i32));

    // i64x2
    let a_i64 = i64x2_splat(100000);
    let b_i64 = i64x2_splat(200000);
    black_box(i64x2_add(a_i64, b_i64));
    black_box(i64x2_sub(a_i64, b_i64));
    black_box(i64x2_mul(a_i64, b_i64));
    black_box(i64x2_neg(a_i64));
    black_box(i64x2_abs(a_i64));
}

// =============================================================================
// Float Arithmetic
// =============================================================================

#[arcane]
fn exercise_float_arithmetic(token: Wasm128Token) {
    // f32x4
    let a_f32 = f32x4_splat(1.5);
    let b_f32 = f32x4_splat(2.5);
    black_box(f32x4_add(a_f32, b_f32));
    black_box(f32x4_sub(a_f32, b_f32));
    black_box(f32x4_mul(a_f32, b_f32));
    black_box(f32x4_div(a_f32, b_f32));
    black_box(f32x4_neg(a_f32));
    black_box(f32x4_abs(a_f32));
    black_box(f32x4_sqrt(a_f32));
    black_box(f32x4_min(a_f32, b_f32));
    black_box(f32x4_max(a_f32, b_f32));
    black_box(f32x4_pmin(a_f32, b_f32));
    black_box(f32x4_pmax(a_f32, b_f32));
    black_box(f32x4_ceil(a_f32));
    black_box(f32x4_floor(a_f32));
    black_box(f32x4_trunc(a_f32));
    black_box(f32x4_nearest(a_f32));

    // f64x2
    let a_f64 = f64x2_splat(1.5);
    let b_f64 = f64x2_splat(2.5);
    black_box(f64x2_add(a_f64, b_f64));
    black_box(f64x2_sub(a_f64, b_f64));
    black_box(f64x2_mul(a_f64, b_f64));
    black_box(f64x2_div(a_f64, b_f64));
    black_box(f64x2_neg(a_f64));
    black_box(f64x2_abs(a_f64));
    black_box(f64x2_sqrt(a_f64));
    black_box(f64x2_min(a_f64, b_f64));
    black_box(f64x2_max(a_f64, b_f64));
    black_box(f64x2_pmin(a_f64, b_f64));
    black_box(f64x2_pmax(a_f64, b_f64));
    black_box(f64x2_ceil(a_f64));
    black_box(f64x2_floor(a_f64));
    black_box(f64x2_trunc(a_f64));
    black_box(f64x2_nearest(a_f64));
}

// =============================================================================
// Comparisons
// =============================================================================

#[arcane]
fn exercise_comparisons(token: Wasm128Token) {
    // i8x16
    let a = i8x16_splat(1);
    let b = i8x16_splat(2);
    black_box(i8x16_eq(a, b));
    black_box(i8x16_ne(a, b));
    black_box(i8x16_lt(a, b));
    black_box(u8x16_lt(a, b));
    black_box(i8x16_gt(a, b));
    black_box(u8x16_gt(a, b));
    black_box(i8x16_le(a, b));
    black_box(u8x16_le(a, b));
    black_box(i8x16_ge(a, b));
    black_box(u8x16_ge(a, b));

    // i32x4
    let c = i32x4_splat(1);
    let d = i32x4_splat(2);
    black_box(i32x4_eq(c, d));
    black_box(i32x4_ne(c, d));
    black_box(i32x4_lt(c, d));
    black_box(u32x4_lt(c, d));
    black_box(i32x4_gt(c, d));
    black_box(u32x4_gt(c, d));
    black_box(i32x4_le(c, d));
    black_box(u32x4_le(c, d));
    black_box(i32x4_ge(c, d));
    black_box(u32x4_ge(c, d));

    // i64x2
    let e = i64x2_splat(1);
    let f = i64x2_splat(2);
    black_box(i64x2_eq(e, f));
    black_box(i64x2_ne(e, f));
    black_box(i64x2_lt(e, f));
    black_box(i64x2_gt(e, f));
    black_box(i64x2_le(e, f));
    black_box(i64x2_ge(e, f));

    // f32x4
    let g = f32x4_splat(1.0);
    let h = f32x4_splat(2.0);
    black_box(f32x4_eq(g, h));
    black_box(f32x4_ne(g, h));
    black_box(f32x4_lt(g, h));
    black_box(f32x4_gt(g, h));
    black_box(f32x4_le(g, h));
    black_box(f32x4_ge(g, h));

    // f64x2
    let i = f64x2_splat(1.0);
    let j = f64x2_splat(2.0);
    black_box(f64x2_eq(i, j));
    black_box(f64x2_ne(i, j));
    black_box(f64x2_lt(i, j));
    black_box(f64x2_gt(i, j));
    black_box(f64x2_le(i, j));
    black_box(f64x2_ge(i, j));
}

// =============================================================================
// Bitwise Operations
// =============================================================================

#[arcane]
fn exercise_bitwise(token: Wasm128Token) {
    let a = i32x4_splat(0x0F0F0F0F);
    let b = i32x4_splat(0x33333333);

    black_box(v128_and(a, b));
    black_box(v128_or(a, b));
    black_box(v128_xor(a, b));
    black_box(v128_andnot(a, b));
    black_box(v128_not(a));

    // Bitselect: for each bit, choose from a (if mask=1) or b (if mask=0)
    let mask = i32x4_splat(-1i32); // all ones
    black_box(v128_bitselect(a, b, mask));
}

// =============================================================================
// Shifts
// =============================================================================

#[arcane]
fn exercise_shifts(token: Wasm128Token) {
    let a_i8 = i8x16_splat(0x42);
    black_box(i8x16_shl(a_i8, 1));
    black_box(i8x16_shr(a_i8, 1));
    black_box(u8x16_shr(a_i8, 1));

    let a_i16 = i16x8_splat(0x4242);
    black_box(i16x8_shl(a_i16, 1));
    black_box(i16x8_shr(a_i16, 1));
    black_box(u16x8_shr(a_i16, 1));

    let a_i32 = i32x4_splat(0x42424242);
    black_box(i32x4_shl(a_i32, 1));
    black_box(i32x4_shr(a_i32, 1));
    black_box(u32x4_shr(a_i32, 1));

    let a_i64 = i64x2_splat(0x4242424242424242);
    black_box(i64x2_shl(a_i64, 1));
    black_box(i64x2_shr(a_i64, 1));
    black_box(u64x2_shr(a_i64, 1));
}

// =============================================================================
// Conversions
// =============================================================================

#[arcane]
fn exercise_conversions(token: Wasm128Token) {
    // Float <-> Int truncation
    let f = f32x4_splat(42.7);
    black_box(i32x4_trunc_sat_f32x4(f));
    black_box(u32x4_trunc_sat_f32x4(f));

    let d = f64x2_splat(42.7);
    black_box(i32x4_trunc_sat_f64x2_zero(d));
    black_box(u32x4_trunc_sat_f64x2_zero(d));

    // Int -> Float conversion
    let i = i32x4_splat(42);
    black_box(f32x4_convert_i32x4(i));
    black_box(f32x4_convert_u32x4(i));
    black_box(f64x2_convert_low_i32x4(i));
    black_box(f64x2_convert_low_u32x4(i));

    // Float demote/promote
    let dd = f64x2_splat(3.14);
    black_box(f32x4_demote_f64x2_zero(dd));
    let ff = f32x4_splat(3.14);
    black_box(f64x2_promote_low_f32x4(ff));

    // Extend (widen)
    let narrow = i8x16_splat(42);
    black_box(i16x8_extend_low_i8x16(narrow));
    black_box(i16x8_extend_high_i8x16(narrow));
    black_box(i16x8_extend_low_u8x16(narrow));
    black_box(i16x8_extend_high_u8x16(narrow));

    let narrow16 = i16x8_splat(42);
    black_box(i32x4_extend_low_i16x8(narrow16));
    black_box(i32x4_extend_high_i16x8(narrow16));
    black_box(i32x4_extend_low_u16x8(narrow16));
    black_box(i32x4_extend_high_u16x8(narrow16));

    let narrow32 = i32x4_splat(42);
    black_box(i64x2_extend_low_i32x4(narrow32));
    black_box(i64x2_extend_high_i32x4(narrow32));
    black_box(i64x2_extend_low_u32x4(narrow32));
    black_box(i64x2_extend_high_u32x4(narrow32));

    // Narrow (truncate)
    let wide_i16 = i16x8_splat(42);
    black_box(i8x16_narrow_i16x8(wide_i16, wide_i16));
    black_box(u8x16_narrow_i16x8(wide_i16, wide_i16));
    let wide_i32 = i32x4_splat(42);
    black_box(i16x8_narrow_i32x4(wide_i32, wide_i32));
    black_box(u16x8_narrow_i32x4(wide_i32, wide_i32));
}

// =============================================================================
// Lane Operations
// =============================================================================

#[arcane]
fn exercise_lane_ops(token: Wasm128Token) {
    // Extract lane
    let v_i8 = i8x16_splat(42);
    black_box(i8x16_extract_lane::<0>(v_i8));
    black_box(u8x16_extract_lane::<0>(v_i8));
    let v_i16 = i16x8_splat(42);
    black_box(i16x8_extract_lane::<0>(v_i16));
    black_box(u16x8_extract_lane::<0>(v_i16));
    let v_i32 = i32x4_splat(42);
    black_box(i32x4_extract_lane::<0>(v_i32));
    let v_i64 = i64x2_splat(42);
    black_box(i64x2_extract_lane::<0>(v_i64));
    let v_f32 = f32x4_splat(42.0);
    black_box(f32x4_extract_lane::<0>(v_f32));
    let v_f64 = f64x2_splat(42.0);
    black_box(f64x2_extract_lane::<0>(v_f64));

    // Replace lane
    black_box(i8x16_replace_lane::<0>(v_i8, 99));
    black_box(i16x8_replace_lane::<0>(v_i16, 99));
    black_box(i32x4_replace_lane::<0>(v_i32, 99));
    black_box(i64x2_replace_lane::<0>(v_i64, 99));
    black_box(f32x4_replace_lane::<0>(v_f32, 99.0));
    black_box(f64x2_replace_lane::<0>(v_f64, 99.0));

    // Splat (already used above but explicitly test)
    black_box(i8x16_splat(1));
    black_box(i16x8_splat(1));
    black_box(i32x4_splat(1));
    black_box(i64x2_splat(1));
    black_box(f32x4_splat(1.0));
    black_box(f64x2_splat(1.0));
}

// =============================================================================
// Boolean Reductions
// =============================================================================

#[arcane]
fn exercise_boolean_reductions(token: Wasm128Token) {
    let a = i32x4_splat(0);
    let b = i32x4_splat(-1);

    // v128_any_true: true if any bit is set
    assert!(!v128_any_true(a), "zero vector should not have any true");
    assert!(v128_any_true(b), "all-ones vector should have any true");

    // Per-lane all_true
    assert!(
        !i8x16_all_true(i8x16_splat(0)),
        "zero i8x16 should not be all true"
    );
    assert!(
        i8x16_all_true(i8x16_splat(1)),
        "nonzero i8x16 should be all true"
    );
    assert!(
        i32x4_all_true(i32x4_splat(1)),
        "nonzero i32x4 should be all true"
    );

    // Bitmask
    let mask_i8 = i8x16_bitmask(i8x16_splat(-1i8));
    assert_eq!(mask_i8, 0xFFFF, "all-ones i8x16 bitmask should be 0xFFFF");
    let mask_i32 = i32x4_bitmask(i32x4_splat(-1));
    assert_eq!(mask_i32, 0x0F, "all-ones i32x4 bitmask should be 0x0F");
    let mask_i64 = i64x2_bitmask(i64x2_splat(-1));
    assert_eq!(mask_i64, 0x03, "all-ones i64x2 bitmask should be 0x03");
}

// =============================================================================
// Shuffle and Swizzle
// =============================================================================

#[arcane]
fn exercise_shuffle_swizzle(token: Wasm128Token) {
    let a = i8x16_splat(1);
    let b = i8x16_splat(2);

    // i8x16_shuffle: select lanes from a (indices 0-15) or b (indices 16-31)
    let shuffled = i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23>(a, b);
    black_box(shuffled);

    // i8x16_swizzle: select lanes from a using indices in b
    let indices = i8x16_splat(0); // all lane 0
    let swizzled = i8x16_swizzle(a, indices);
    black_box(swizzled);
}

// =============================================================================
// Extended Multiply
// =============================================================================

#[arcane]
fn exercise_extended_multiply(token: Wasm128Token) {
    // i16x8 extended multiply from i8x16
    let a = i8x16_splat(10);
    let b = i8x16_splat(20);
    black_box(i16x8_extmul_low_i8x16(a, b));
    black_box(i16x8_extmul_high_i8x16(a, b));
    black_box(i16x8_extmul_low_u8x16(a, b));
    black_box(i16x8_extmul_high_u8x16(a, b));

    // i32x4 extended multiply from i16x8
    let c = i16x8_splat(10);
    let d = i16x8_splat(20);
    black_box(i32x4_extmul_low_i16x8(c, d));
    black_box(i32x4_extmul_high_i16x8(c, d));
    black_box(i32x4_extmul_low_u16x8(c, d));
    black_box(i32x4_extmul_high_u16x8(c, d));

    // i64x2 extended multiply from i32x4
    let e = i32x4_splat(10);
    let f = i32x4_splat(20);
    black_box(i64x2_extmul_low_i32x4(e, f));
    black_box(i64x2_extmul_high_i32x4(e, f));
    black_box(i64x2_extmul_low_u32x4(e, f));
    black_box(i64x2_extmul_high_u32x4(e, f));

    // Dot product
    let g = i16x8_splat(3);
    let h = i16x8_splat(4);
    let dot = i32x4_dot_i16x8(g, h);
    black_box(dot);
    // Each i32 lane = sum of 2 i16*i16 products = 3*4 + 3*4 = 24
    let lane0 = i32x4_extract_lane::<0>(dot);
    assert_eq!(lane0, 24, "dot product of [3,3] . [4,4] = 24");
}

// =============================================================================
// Saturating Arithmetic
// =============================================================================

#[arcane]
fn exercise_saturating_arithmetic(token: Wasm128Token) {
    // i8x16 saturating
    let a = i8x16_splat(100);
    let b = i8x16_splat(100);
    black_box(i8x16_add_sat(a, b)); // 100+100 = saturates to 127
    black_box(u8x16_add_sat(a, b)); // 100+100 = 200 (no saturation)
    black_box(i8x16_sub_sat(i8x16_splat(-100), b)); // -100-100 saturates to -128
    black_box(u8x16_sub_sat(i8x16_splat(0), b)); // 0-100 saturates to 0

    // Verify saturation
    let sum = i8x16_add_sat(i8x16_splat(120), i8x16_splat(120));
    let lane0 = i8x16_extract_lane::<0>(sum);
    assert_eq!(lane0, 127, "signed i8 saturation: 120+120 should be 127");

    // i16x8 saturating
    let c = i16x8_splat(30000);
    let d = i16x8_splat(30000);
    black_box(i16x8_add_sat(c, d));
    black_box(u16x8_add_sat(c, d));
    black_box(i16x8_sub_sat(i16x8_splat(-30000), d));
    black_box(u16x8_sub_sat(i16x8_splat(0), d));
}

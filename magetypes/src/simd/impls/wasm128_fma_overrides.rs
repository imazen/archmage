//! Direct single-rounding relaxed FMA overrides for [`archmage::Wasm128RelaxedFmaToken`]
//! (paired with the generated `wasm128_fma_delegated.rs`, which forwards every
//! other backend method to [`archmage::Wasm128Token`]).
//!
//! Holding a [`Wasm128RelaxedFmaToken`] proves both `+simd128,+relaxed-simd`
//! availability and that the executing WebAssembly engine fuses all lanes of
//! `f32x4.relaxed_madd` and `f64x2.relaxed_madd` to single-rounding IEEE-754
//! FMA. Both the fast (`mul_add` / `mul_sub`) and single-rounding
//! (`mul_add_portable` / `mul_sub_portable`) trait methods emit bare
//! `f32x4_relaxed_madd` / `f64x2_relaxed_madd` instructions with zero per-call
//! `AtomicU8` probe loads and zero software-FMA fallback branches.

#![cfg(all(
    target_arch = "wasm32",
    target_feature = "simd128",
    target_feature = "relaxed-simd"
))]

use archmage::Wasm128RelaxedFmaToken;
use core::arch::wasm32::{
    f32x4_neg, f32x4_relaxed_madd, f32x4_relaxed_max, f32x4_relaxed_min, f64x2_neg,
    f64x2_relaxed_madd, f64x2_relaxed_max, f64x2_relaxed_min, i16x8_relaxed_q15mulr,
    i32x4_relaxed_trunc_f32x4, i32x4_shuffle, v128,
};

// ============================================================================
// f32x4 / f32x8 / f32x16
// ============================================================================

#[archmage::attune(wrap, suppress_const_test)]
pub(super) fn mul_add_f32x4(token: Wasm128RelaxedFmaToken, a: v128, b: v128, c: v128) -> v128 {
    let _ = token;
    f32x4_relaxed_madd(a, b, c)
}

#[archmage::attune(wrap, suppress_const_test)]
pub(super) fn mul_sub_f32x4(token: Wasm128RelaxedFmaToken, a: v128, b: v128, c: v128) -> v128 {
    let _ = token;
    f32x4_relaxed_madd(a, b, f32x4_neg(c))
}

#[inline(always)]
pub(super) fn mul_add_f32x8(
    token: Wasm128RelaxedFmaToken,
    a: [v128; 2],
    b: [v128; 2],
    c: [v128; 2],
) -> [v128; 2] {
    [
        mul_add_f32x4(token, a[0], b[0], c[0]),
        mul_add_f32x4(token, a[1], b[1], c[1]),
    ]
}

#[inline(always)]
pub(super) fn mul_sub_f32x8(
    token: Wasm128RelaxedFmaToken,
    a: [v128; 2],
    b: [v128; 2],
    c: [v128; 2],
) -> [v128; 2] {
    [
        mul_sub_f32x4(token, a[0], b[0], c[0]),
        mul_sub_f32x4(token, a[1], b[1], c[1]),
    ]
}

#[cfg(feature = "w512")]
#[inline(always)]
pub(super) fn mul_add_f32x16(
    token: Wasm128RelaxedFmaToken,
    a: [v128; 4],
    b: [v128; 4],
    c: [v128; 4],
) -> [v128; 4] {
    [
        mul_add_f32x4(token, a[0], b[0], c[0]),
        mul_add_f32x4(token, a[1], b[1], c[1]),
        mul_add_f32x4(token, a[2], b[2], c[2]),
        mul_add_f32x4(token, a[3], b[3], c[3]),
    ]
}

#[cfg(feature = "w512")]
#[inline(always)]
pub(super) fn mul_sub_f32x16(
    token: Wasm128RelaxedFmaToken,
    a: [v128; 4],
    b: [v128; 4],
    c: [v128; 4],
) -> [v128; 4] {
    [
        mul_sub_f32x4(token, a[0], b[0], c[0]),
        mul_sub_f32x4(token, a[1], b[1], c[1]),
        mul_sub_f32x4(token, a[2], b[2], c[2]),
        mul_sub_f32x4(token, a[3], b[3], c[3]),
    ]
}

// ============================================================================
// f64x2 / f64x4 / f64x8
// ============================================================================

#[archmage::attune(wrap, suppress_const_test)]
pub(super) fn mul_add_f64x2(token: Wasm128RelaxedFmaToken, a: v128, b: v128, c: v128) -> v128 {
    let _ = token;
    f64x2_relaxed_madd(a, b, c)
}

#[archmage::attune(wrap, suppress_const_test)]
pub(super) fn mul_sub_f64x2(token: Wasm128RelaxedFmaToken, a: v128, b: v128, c: v128) -> v128 {
    let _ = token;
    f64x2_relaxed_madd(a, b, f64x2_neg(c))
}

#[inline(always)]
pub(super) fn mul_add_f64x4(
    token: Wasm128RelaxedFmaToken,
    a: [v128; 2],
    b: [v128; 2],
    c: [v128; 2],
) -> [v128; 2] {
    [
        mul_add_f64x2(token, a[0], b[0], c[0]),
        mul_add_f64x2(token, a[1], b[1], c[1]),
    ]
}

#[inline(always)]
pub(super) fn mul_sub_f64x4(
    token: Wasm128RelaxedFmaToken,
    a: [v128; 2],
    b: [v128; 2],
    c: [v128; 2],
) -> [v128; 2] {
    [
        mul_sub_f64x2(token, a[0], b[0], c[0]),
        mul_sub_f64x2(token, a[1], b[1], c[1]),
    ]
}

#[cfg(feature = "w512")]
#[inline(always)]
pub(super) fn mul_add_f64x8(
    token: Wasm128RelaxedFmaToken,
    a: [v128; 4],
    b: [v128; 4],
    c: [v128; 4],
) -> [v128; 4] {
    [
        mul_add_f64x2(token, a[0], b[0], c[0]),
        mul_add_f64x2(token, a[1], b[1], c[1]),
        mul_add_f64x2(token, a[2], b[2], c[2]),
        mul_add_f64x2(token, a[3], b[3], c[3]),
    ]
}

#[cfg(feature = "w512")]
#[inline(always)]
pub(super) fn mul_sub_f64x8(
    token: Wasm128RelaxedFmaToken,
    a: [v128; 4],
    b: [v128; 4],
    c: [v128; 4],
) -> [v128; 4] {
    [
        mul_sub_f64x2(token, a[0], b[0], c[0]),
        mul_sub_f64x2(token, a[1], b[1], c[1]),
        mul_sub_f64x2(token, a[2], b[2], c[2]),
        mul_sub_f64x2(token, a[3], b[3], c[3]),
    ]
}

// ============================================================================
// Fast min / max (f32 / f64)
// ============================================================================

#[archmage::attune(wrap, suppress_const_test)]
pub(super) fn min_fast_f32x4(token: Wasm128RelaxedFmaToken, a: v128, b: v128) -> v128 {
    let _ = token;
    f32x4_relaxed_min(a, b)
}

#[archmage::attune(wrap, suppress_const_test)]
pub(super) fn max_fast_f32x4(token: Wasm128RelaxedFmaToken, a: v128, b: v128) -> v128 {
    let _ = token;
    f32x4_relaxed_max(a, b)
}

#[inline(always)]
pub(super) fn min_fast_f32x8(
    token: Wasm128RelaxedFmaToken,
    a: [v128; 2],
    b: [v128; 2],
) -> [v128; 2] {
    [
        min_fast_f32x4(token, a[0], b[0]),
        min_fast_f32x4(token, a[1], b[1]),
    ]
}

#[inline(always)]
pub(super) fn max_fast_f32x8(
    token: Wasm128RelaxedFmaToken,
    a: [v128; 2],
    b: [v128; 2],
) -> [v128; 2] {
    [
        max_fast_f32x4(token, a[0], b[0]),
        max_fast_f32x4(token, a[1], b[1]),
    ]
}

#[cfg(feature = "w512")]
#[inline(always)]
pub(super) fn min_fast_f32x16(
    token: Wasm128RelaxedFmaToken,
    a: [v128; 4],
    b: [v128; 4],
) -> [v128; 4] {
    [
        min_fast_f32x4(token, a[0], b[0]),
        min_fast_f32x4(token, a[1], b[1]),
        min_fast_f32x4(token, a[2], b[2]),
        min_fast_f32x4(token, a[3], b[3]),
    ]
}

#[cfg(feature = "w512")]
#[inline(always)]
pub(super) fn max_fast_f32x16(
    token: Wasm128RelaxedFmaToken,
    a: [v128; 4],
    b: [v128; 4],
) -> [v128; 4] {
    [
        max_fast_f32x4(token, a[0], b[0]),
        max_fast_f32x4(token, a[1], b[1]),
        max_fast_f32x4(token, a[2], b[2]),
        max_fast_f32x4(token, a[3], b[3]),
    ]
}

#[archmage::attune(wrap, suppress_const_test)]
pub(super) fn min_fast_f64x2(token: Wasm128RelaxedFmaToken, a: v128, b: v128) -> v128 {
    let _ = token;
    f64x2_relaxed_min(a, b)
}

#[archmage::attune(wrap, suppress_const_test)]
pub(super) fn max_fast_f64x2(token: Wasm128RelaxedFmaToken, a: v128, b: v128) -> v128 {
    let _ = token;
    f64x2_relaxed_max(a, b)
}

#[inline(always)]
pub(super) fn min_fast_f64x4(
    token: Wasm128RelaxedFmaToken,
    a: [v128; 2],
    b: [v128; 2],
) -> [v128; 2] {
    [
        min_fast_f64x2(token, a[0], b[0]),
        min_fast_f64x2(token, a[1], b[1]),
    ]
}

#[inline(always)]
pub(super) fn max_fast_f64x4(
    token: Wasm128RelaxedFmaToken,
    a: [v128; 2],
    b: [v128; 2],
) -> [v128; 2] {
    [
        max_fast_f64x2(token, a[0], b[0]),
        max_fast_f64x2(token, a[1], b[1]),
    ]
}

#[cfg(feature = "w512")]
#[inline(always)]
pub(super) fn min_fast_f64x8(
    token: Wasm128RelaxedFmaToken,
    a: [v128; 4],
    b: [v128; 4],
) -> [v128; 4] {
    [
        min_fast_f64x2(token, a[0], b[0]),
        min_fast_f64x2(token, a[1], b[1]),
        min_fast_f64x2(token, a[2], b[2]),
        min_fast_f64x2(token, a[3], b[3]),
    ]
}

#[cfg(feature = "w512")]
#[inline(always)]
pub(super) fn max_fast_f64x8(
    token: Wasm128RelaxedFmaToken,
    a: [v128; 4],
    b: [v128; 4],
) -> [v128; 4] {
    [
        max_fast_f64x2(token, a[0], b[0]),
        max_fast_f64x2(token, a[1], b[1]),
        max_fast_f64x2(token, a[2], b[2]),
        max_fast_f64x2(token, a[3], b[3]),
    ]
}

// ============================================================================
// Lane-indexed FMA (f32x4 / f32x8)
// ============================================================================

#[archmage::attune(wrap, suppress_const_test)]
pub(super) fn mul_add_lane_f32x4<const LANE: i32>(
    token: Wasm128RelaxedFmaToken,
    a: v128,
    v_in: v128,
    acc: v128,
) -> v128 {
    let _ = token;
    const { assert!(LANE >= 0 && LANE < 4, "LANE must be in 0..4") };
    let splat = match LANE {
        0 => i32x4_shuffle::<0, 0, 0, 0>(v_in, v_in),
        1 => i32x4_shuffle::<1, 1, 1, 1>(v_in, v_in),
        2 => i32x4_shuffle::<2, 2, 2, 2>(v_in, v_in),
        3 => i32x4_shuffle::<3, 3, 3, 3>(v_in, v_in),
        _ => unreachable!(),
    };
    f32x4_relaxed_madd(a, splat, acc)
}

#[archmage::attune(wrap, suppress_const_test)]
pub(super) fn mul_add_lane_f32x8<const LANE: i32>(
    token: Wasm128RelaxedFmaToken,
    a: [v128; 2],
    v_in: [v128; 2],
    acc: [v128; 2],
) -> [v128; 2] {
    let _ = token;
    const { assert!(LANE >= 0 && LANE < 8, "LANE must be in 0..8") };
    let splat = match LANE {
        0 => i32x4_shuffle::<0, 0, 0, 0>(v_in[0], v_in[0]),
        1 => i32x4_shuffle::<1, 1, 1, 1>(v_in[0], v_in[0]),
        2 => i32x4_shuffle::<2, 2, 2, 2>(v_in[0], v_in[0]),
        3 => i32x4_shuffle::<3, 3, 3, 3>(v_in[0], v_in[0]),
        4 => i32x4_shuffle::<0, 0, 0, 0>(v_in[1], v_in[1]),
        5 => i32x4_shuffle::<1, 1, 1, 1>(v_in[1], v_in[1]),
        6 => i32x4_shuffle::<2, 2, 2, 2>(v_in[1], v_in[1]),
        7 => i32x4_shuffle::<3, 3, 3, 3>(v_in[1], v_in[1]),
        _ => unreachable!(),
    };
    [
        f32x4_relaxed_madd(a[0], splat, acc[0]),
        f32x4_relaxed_madd(a[1], splat, acc[1]),
    ]
}

// ============================================================================
// Fast float-to-i32 truncation & Q15 rounding multiply
// ============================================================================

#[archmage::attune(wrap, suppress_const_test)]
pub(super) fn convert_f32_to_i32_fast_x4(token: Wasm128RelaxedFmaToken, a: v128) -> v128 {
    let _ = token;
    i32x4_relaxed_trunc_f32x4(a)
}

#[inline(always)]
pub(super) fn convert_f32_to_i32_fast_x8(token: Wasm128RelaxedFmaToken, a: [v128; 2]) -> [v128; 2] {
    [
        convert_f32_to_i32_fast_x4(token, a[0]),
        convert_f32_to_i32_fast_x4(token, a[1]),
    ]
}

#[cfg(feature = "w512")]
#[inline(always)]
pub(super) fn convert_f32_to_i32_fast_x16(
    token: Wasm128RelaxedFmaToken,
    a: [v128; 4],
) -> [v128; 4] {
    [
        convert_f32_to_i32_fast_x4(token, a[0]),
        convert_f32_to_i32_fast_x4(token, a[1]),
        convert_f32_to_i32_fast_x4(token, a[2]),
        convert_f32_to_i32_fast_x4(token, a[3]),
    ]
}

#[archmage::attune(wrap, suppress_const_test)]
pub(super) fn mul_hrs_q15_x8(token: Wasm128RelaxedFmaToken, a: v128, b: v128) -> v128 {
    let _ = token;
    i16x8_relaxed_q15mulr(a, b)
}

#[inline(always)]
pub(super) fn mul_hrs_q15_x16(
    token: Wasm128RelaxedFmaToken,
    a: [v128; 2],
    b: [v128; 2],
) -> [v128; 2] {
    [
        mul_hrs_q15_x8(token, a[0], b[0]),
        mul_hrs_q15_x8(token, a[1], b[1]),
    ]
}

#[cfg(feature = "w512")]
#[inline(always)]
pub(super) fn mul_hrs_q15_x32(
    token: Wasm128RelaxedFmaToken,
    a: [v128; 4],
    b: [v128; 4],
) -> [v128; 4] {
    [
        mul_hrs_q15_x8(token, a[0], b[0]),
        mul_hrs_q15_x8(token, a[1], b[1]),
        mul_hrs_q15_x8(token, a[2], b[2]),
        mul_hrs_q15_x8(token, a[3], b[3]),
    ]
}

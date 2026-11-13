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
use core::arch::wasm32::{f32x4_neg, f32x4_relaxed_madd, f64x2_neg, f64x2_relaxed_madd, v128};

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

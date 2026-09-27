//! Software fusion for strict SIMD; direct engine arithmetic for relaxed SIMD.
#![forbid(unsafe_code)]

use archmage::Wasm128Token;
use core::arch::wasm32::{self as w, v128};

#[cfg(not(target_feature = "relaxed-simd"))]
#[archmage::rite]
pub(crate) fn f32x4(token: Wasm128Token, a: v128, b: v128, c: v128) -> v128 {
    let lo = wide_sum(token, a, b, c);
    let hi = wide_sum(
        token,
        w::i32x4_shuffle::<2, 3, 2, 3>(a, a),
        w::i32x4_shuffle::<2, 3, 2, 3>(b, b),
        w::i32x4_shuffle::<2, 3, 2, 3>(c, c),
    );
    w::i32x4_shuffle::<0, 1, 4, 5>(lo, hi)
}

/// Vector form of nostd_math::fmaf: exact f64 product, TwoSum, round to odd.
#[cfg(not(target_feature = "relaxed-simd"))]
#[archmage::rite]
fn wide_sum(_token: Wasm128Token, a: v128, b: v128, c: v128) -> v128 {
    let product = w::f64x2_mul(w::f64x2_promote_low_f32x4(a), w::f64x2_promote_low_f32x4(b));
    let addend = w::f64x2_promote_low_f32x4(c);
    let sum = w::f64x2_add(product, addend);
    let virtual_addend = w::f64x2_sub(sum, product);
    let error = w::f64x2_add(
        w::f64x2_sub(product, w::f64x2_sub(sum, virtual_addend)),
        w::f64x2_sub(addend, virtual_addend),
    );
    let zero = w::f64x2_splat(0.0);
    let one = w::i64x2_splat(1);
    let even = w::i64x2_eq(w::v128_and(sum, one), w::i64x2_splat(0));
    let finite = w::f64x2_lt(w::f64x2_abs(sum), w::f64x2_splat(f64::INFINITY));
    let adjust = w::v128_and(finite, w::v128_and(even, w::f64x2_ne(error, zero)));
    let same_sign = w::i64x2_eq(w::f64x2_gt(error, zero), w::f64x2_gt(sum, zero));
    let step = w::v128_bitselect(one, w::i64x2_splat(-1), same_sign);
    let odd = w::i64x2_add(sum, step);
    w::f32x4_demote_f64x2_zero(w::v128_bitselect(odd, sum, adjust))
}

#[cfg(not(target_feature = "relaxed-simd"))]
#[archmage::rite]
pub(crate) fn f64x2(_token: Wasm128Token, a: v128, b: v128, c: v128) -> v128 {
    w::f64x2(
        crate::nostd_math::fma(
            w::f64x2_extract_lane::<0>(a),
            w::f64x2_extract_lane::<0>(b),
            w::f64x2_extract_lane::<0>(c),
        ),
        crate::nostd_math::fma(
            w::f64x2_extract_lane::<1>(a),
            w::f64x2_extract_lane::<1>(b),
            w::f64x2_extract_lane::<1>(c),
        ),
    )
}

#[cfg(target_feature = "relaxed-simd")]
#[archmage::rite]
pub(crate) fn f32x4(_token: Wasm128Token, a: v128, b: v128, c: v128) -> v128 {
    w::f32x4_relaxed_madd(a, b, c)
}

#[cfg(target_feature = "relaxed-simd")]
#[archmage::rite]
pub(crate) fn f64x2(_token: Wasm128Token, a: v128, b: v128, c: v128) -> v128 {
    w::f64x2_relaxed_madd(a, b, c)
}

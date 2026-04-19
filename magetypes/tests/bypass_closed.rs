//! Bypass-closure tests (runtime sanctioned-form assertions).
//!
//! This file is the runtime-assert counterpart to the `compile_fail`
//! doctest grid in [`magetypes::bypass_adversarial`]. Each test here
//! exercises the **sanctioned form** of a representative backend trait
//! method (pass a real token, observe the expected output) across
//! the same ten categories the doctests cover: construction, memory,
//! arithmetic, math, comparison, reduction, bitwise, shift, boolean,
//! and convert/bitcast.
//!
//! The compile-fail grid proves the UFCS-tokenless form fails; these
//! tests prove the sanctioned (with-token) form still works correctly.
//! Together they guard both directions of the soundness contract.

use archmage::{ScalarToken, SimdToken};
use magetypes::simd::backends::{
    F32x4Backend, F32x8Backend, F32x8Convert, F64x2Backend, I32x4Backend, U8x16Backend,
    U32x4Backend,
};

// -------------------------------------------------------------------
// Original smoke test — pre-refactor PoC, kept for regression.
// -------------------------------------------------------------------

/// Pre-fix PoC — UFCS splat without a token — now fails to compile.
///
/// ```compile_fail
/// use archmage::X64V3Token;
/// use magetypes::simd::backends::F32x8Backend;
/// // Should fail: splat now requires `self`, can't be called UFCS-style
/// // without a token value.
/// let _ = <X64V3Token as F32x8Backend>::splat(7.0);
/// ```
///
/// Confirms construction methods (`splat` / `zero` / `load` / `from_array`)
/// all require a token value.
#[cfg(target_arch = "x86_64")]
#[test]
fn splat_with_token_works() {
    use archmage::X64V3Token;
    if let Some(t) = X64V3Token::summon() {
        // Sanctioned form: pass a real token.
        let r = <X64V3Token as F32x8Backend>::splat(t, 7.0);
        let mut out = [0.0f32; 8];
        <X64V3Token as F32x8Backend>::store(t, r, &mut out);
        assert_eq!(out, [7.0; 8]);
    }
}

// -------------------------------------------------------------------
// Sanctioned-form tests, one per category.
// Scalar backend is always available, so these run everywhere.
// -------------------------------------------------------------------

#[test]
fn sanctioned_construction_splat_zero() {
    let t = ScalarToken::summon().unwrap();
    let s = <ScalarToken as F32x8Backend>::splat(t, 3.5);
    let z = <ScalarToken as F32x8Backend>::zero(t);
    assert_eq!(<ScalarToken as F32x8Backend>::to_array(t, s), [3.5; 8]);
    assert_eq!(<ScalarToken as F32x8Backend>::to_array(t, z), [0.0; 8]);
}

#[test]
fn sanctioned_construction_load_from_array() {
    let t = ScalarToken::summon().unwrap();
    let src = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let r = <ScalarToken as F32x8Backend>::load(t, &src);
    assert_eq!(<ScalarToken as F32x8Backend>::to_array(t, r), src);
    let r2 = <ScalarToken as F32x8Backend>::from_array(t, src);
    assert_eq!(<ScalarToken as F32x8Backend>::to_array(t, r2), src);
}

#[test]
fn sanctioned_construction_u8x16_from_array() {
    let t = ScalarToken::summon().unwrap();
    let src: [u8; 16] = core::array::from_fn(|i| i as u8);
    let r = <ScalarToken as U8x16Backend>::from_array(t, src);
    assert_eq!(<ScalarToken as U8x16Backend>::to_array(t, r), src);
}

#[test]
fn sanctioned_memory_store_to_array() {
    let t = ScalarToken::summon().unwrap();
    let r = <ScalarToken as F32x8Backend>::splat(t, 2.5);
    let mut out = [0.0f32; 8];
    <ScalarToken as F32x8Backend>::store(t, r, &mut out);
    assert_eq!(out, [2.5; 8]);
    assert_eq!(<ScalarToken as F32x8Backend>::to_array(t, r), [2.5; 8]);

    // F64x2 flavor
    let r2 = <ScalarToken as F64x2Backend>::splat(t, -1.25);
    assert_eq!(<ScalarToken as F64x2Backend>::to_array(t, r2), [-1.25; 2]);
}

#[test]
fn sanctioned_arithmetic_add_neg() {
    let t = ScalarToken::summon().unwrap();
    let a = <ScalarToken as I32x4Backend>::from_array(t, [1, 2, 3, 4]);
    let b = <ScalarToken as I32x4Backend>::from_array(t, [10, 20, 30, 40]);
    let sum = <ScalarToken as I32x4Backend>::add(t, a, b);
    assert_eq!(
        <ScalarToken as I32x4Backend>::to_array(t, sum),
        [11, 22, 33, 44]
    );

    let v = <ScalarToken as F32x4Backend>::from_array(t, [1.0, -2.0, 3.0, -4.0]);
    let n = <ScalarToken as F32x4Backend>::neg(t, v);
    assert_eq!(
        <ScalarToken as F32x4Backend>::to_array(t, n),
        [-1.0, 2.0, -3.0, 4.0]
    );
}

#[test]
fn sanctioned_math_min_sqrt_mul_add() {
    let t = ScalarToken::summon().unwrap();
    let a = <ScalarToken as F32x8Backend>::from_array(t, [1.0, 4.0, 9.0, 16.0, 1.0, 2.0, 3.0, 4.0]);
    let b = <ScalarToken as F32x8Backend>::from_array(t, [2.0, 3.0, 5.0, 5.0, 0.5, 10.0, 1.5, 7.0]);
    let mn = <ScalarToken as F32x8Backend>::min(t, a, b);
    assert_eq!(
        <ScalarToken as F32x8Backend>::to_array(t, mn),
        [1.0, 3.0, 5.0, 5.0, 0.5, 2.0, 1.5, 4.0]
    );

    let s4 = <ScalarToken as F32x4Backend>::from_array(t, [1.0, 4.0, 9.0, 16.0]);
    let sr = <ScalarToken as F32x4Backend>::sqrt(t, s4);
    assert_eq!(
        <ScalarToken as F32x4Backend>::to_array(t, sr),
        [1.0, 2.0, 3.0, 4.0]
    );

    let fa = <ScalarToken as F32x8Backend>::splat(t, 2.0);
    let fb = <ScalarToken as F32x8Backend>::splat(t, 3.0);
    let fc = <ScalarToken as F32x8Backend>::splat(t, 1.0);
    let ma = <ScalarToken as F32x8Backend>::mul_add(t, fa, fb, fc);
    assert_eq!(<ScalarToken as F32x8Backend>::to_array(t, ma), [7.0; 8]);
}

#[test]
fn sanctioned_comparison_lt_blend() {
    let t = ScalarToken::summon().unwrap();
    let a = <ScalarToken as I32x4Backend>::from_array(t, [1, 2, 3, 4]);
    let b = <ScalarToken as I32x4Backend>::from_array(t, [2, 2, 2, 2]);
    let mask = <ScalarToken as I32x4Backend>::simd_lt(t, a, b);
    // Mask lanes are all-1s where true, all-0s where false.
    let out = <ScalarToken as I32x4Backend>::to_array(t, mask);
    assert_eq!(out[0], -1); // 1 < 2 -> true
    assert_eq!(out[1], 0); // 2 < 2 -> false
    assert_eq!(out[2], 0);
    assert_eq!(out[3], 0);

    // Blend: pick from 'tt' where mask lane is all-1s, 'ff' otherwise.
    let mask_all_true = <ScalarToken as F32x8Backend>::from_array(
        t,
        [f32::from_bits(!0u32); 8], // all ones bit-pattern
    );
    let tt = <ScalarToken as F32x8Backend>::splat(t, 1.0);
    let ff = <ScalarToken as F32x8Backend>::splat(t, 2.0);
    let blended = <ScalarToken as F32x8Backend>::blend(t, mask_all_true, tt, ff);
    assert_eq!(
        <ScalarToken as F32x8Backend>::to_array(t, blended),
        [1.0; 8]
    );
}

#[test]
fn sanctioned_reduction_reduce_add() {
    let t = ScalarToken::summon().unwrap();
    let v = <ScalarToken as F32x8Backend>::from_array(t, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let s = <ScalarToken as F32x8Backend>::reduce_add(t, v);
    assert_eq!(s, 36.0);
}

#[test]
fn sanctioned_bitwise_bitand_not() {
    let t = ScalarToken::summon().unwrap();
    let a = <ScalarToken as U32x4Backend>::from_array(t, [0xFF00, 0x00FF, 0xAAAA, 0x5555]);
    let b = <ScalarToken as U32x4Backend>::from_array(t, [0x0F0F, 0x0F0F, 0xFFFF, 0xFFFF]);
    let r = <ScalarToken as U32x4Backend>::bitand(t, a, b);
    assert_eq!(
        <ScalarToken as U32x4Backend>::to_array(t, r),
        [0x0F00, 0x000F, 0xAAAA, 0x5555]
    );

    let z = <ScalarToken as U32x4Backend>::zero(t);
    let n = <ScalarToken as U32x4Backend>::not(t, z);
    assert_eq!(<ScalarToken as U32x4Backend>::to_array(t, n), [u32::MAX; 4]);
}

#[test]
fn sanctioned_shift_shl_const() {
    let t = ScalarToken::summon().unwrap();
    let a = <ScalarToken as I32x4Backend>::from_array(t, [1, 2, 3, 4]);
    let r = <ScalarToken as I32x4Backend>::shl_const::<2>(t, a);
    assert_eq!(
        <ScalarToken as I32x4Backend>::to_array(t, r),
        [4, 8, 12, 16]
    );
}

#[test]
fn sanctioned_boolean_all_true_bitmask() {
    let t = ScalarToken::summon().unwrap();

    // all-true when every lane's high bit is set
    let all_ones = <ScalarToken as I32x4Backend>::splat(t, -1);
    assert!(<ScalarToken as I32x4Backend>::all_true(t, all_ones));
    let zero = <ScalarToken as I32x4Backend>::zero(t);
    assert!(!<ScalarToken as I32x4Backend>::all_true(t, zero));

    // bitmask: high bit of each u32 lane as a bitmask
    let mixed = <ScalarToken as U32x4Backend>::from_array(t, [0x8000_0000, 0x0, 0xFFFF_FFFF, 0x0]);
    let mask = <ScalarToken as U32x4Backend>::bitmask(t, mixed);
    assert_eq!(mask, 0b0101);
}

#[test]
fn sanctioned_bitcast_f32_to_i32() {
    let t = ScalarToken::summon().unwrap();
    let v = <ScalarToken as F32x8Backend>::splat(t, 1.0);
    let bits = <ScalarToken as F32x8Convert>::bitcast_f32_to_i32(t, v);
    // 1.0f32 bit pattern is 0x3F80_0000
    let i_arr = <ScalarToken as magetypes::simd::backends::I32x8Backend>::to_array(t, bits);
    assert_eq!(i_arr, [0x3F80_0000; 8]);
}

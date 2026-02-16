//! Tests for generic i32x4<T> and i32x8<T> wrapper types.

#![cfg(target_arch = "x86_64")]

use archmage::{ScalarToken, SimdToken, X64V3Token};
use magetypes::simd::backends::I32x4Backend;
use magetypes::simd::generic::{i32x4, i32x8};

fn get_x64v3() -> X64V3Token {
    X64V3Token::summon().expect("AVX2+FMA required for tests")
}

// ============================================================================
// i32x4 tests
// ============================================================================

#[test]
fn i32x4_size_matches_m128i() {
    assert_eq!(
        core::mem::size_of::<i32x4<X64V3Token>>(),
        core::mem::size_of::<core::arch::x86_64::__m128i>()
    );
}

#[test]
fn i32x4_scalar_size() {
    assert_eq!(
        core::mem::size_of::<i32x4<ScalarToken>>(),
        core::mem::size_of::<[i32; 4]>()
    );
}

#[test]
fn i32x4_basic_arithmetic() {
    let t = get_x64v3();
    let a = i32x4::<X64V3Token>::from_array(t, [1, 2, 3, 4]);
    let b = i32x4::<X64V3Token>::from_array(t, [10, 20, 30, 40]);

    assert_eq!((a + b).to_array(), [11, 22, 33, 44]);
    assert_eq!((b - a).to_array(), [9, 18, 27, 36]);
    assert_eq!((a * b).to_array(), [10, 40, 90, 160]);
    assert_eq!((-a).to_array(), [-1, -2, -3, -4]);
}

#[test]
fn i32x4_load_store_roundtrip() {
    let t = get_x64v3();
    let data = [100, -200, 300, -400];
    let v = i32x4::<X64V3Token>::load(t, &data);
    let mut out = [0i32; 4];
    v.store(&mut out);
    assert_eq!(out, data);
    assert_eq!(v.to_array(), data);
}

#[test]
fn i32x4_min_max_abs() {
    let t = get_x64v3();
    let a = i32x4::<X64V3Token>::from_array(t, [-5, 10, -15, 20]);
    let b = i32x4::<X64V3Token>::from_array(t, [5, -10, 15, -20]);

    assert_eq!(a.min(b).to_array(), [-5, -10, -15, -20]);
    assert_eq!(a.max(b).to_array(), [5, 10, 15, 20]);
    assert_eq!(a.abs().to_array(), [5, 10, 15, 20]);
}

#[test]
fn i32x4_clamp() {
    let t = get_x64v3();
    let v = i32x4::<X64V3Token>::from_array(t, [-100, 50, 200, 0]);
    let lo = i32x4::<X64V3Token>::splat(t, -10);
    let hi = i32x4::<X64V3Token>::splat(t, 100);
    assert_eq!(v.clamp(lo, hi).to_array(), [-10, 50, 100, 0]);
}

#[test]
fn i32x4_comparisons() {
    let t = get_x64v3();
    let a = i32x4::<X64V3Token>::from_array(t, [1, 2, 3, 4]);
    let b = i32x4::<X64V3Token>::from_array(t, [1, 3, 2, 4]);

    let eq = a.simd_eq(b);
    assert_eq!(eq.to_array(), [-1, 0, 0, -1]); // all-1s = -1 in i32

    let lt = a.simd_lt(b);
    assert_eq!(lt.to_array(), [0, -1, 0, 0]);

    let gt = a.simd_gt(b);
    assert_eq!(gt.to_array(), [0, 0, -1, 0]);
}

#[test]
fn i32x4_blend() {
    let t = get_x64v3();
    let a = i32x4::<X64V3Token>::from_array(t, [1, 2, 3, 4]);
    let b = i32x4::<X64V3Token>::from_array(t, [10, 20, 30, 40]);
    let mask = i32x4::<X64V3Token>::from_array(t, [-1, 0, -1, 0]); // pick a[0], b[1], a[2], b[3]

    let result = i32x4::blend(mask, a, b);
    assert_eq!(result.to_array(), [1, 20, 3, 40]);
}

#[test]
fn i32x4_reduce_add() {
    let t = get_x64v3();
    let v = i32x4::<X64V3Token>::from_array(t, [1, 2, 3, 4]);
    assert_eq!(v.reduce_add(), 10);
}

#[test]
fn i32x4_shifts() {
    let t = get_x64v3();
    let v = i32x4::<X64V3Token>::from_array(t, [1, 4, -8, 16]);

    assert_eq!(v.shl_const::<2>().to_array(), [4, 16, -32, 64]);
    assert_eq!(v.shr_arithmetic_const::<1>().to_array(), [0, 2, -4, 8]);
    assert_eq!(v.shr_logical_const::<1>().to_array()[0], 0);
    assert_eq!(v.shr_logical_const::<1>().to_array()[1], 2);
    // -8 >> 1 logical = large positive number (sign bit not extended)
    assert!(v.shr_logical_const::<1>().to_array()[2] > 0);
}

#[test]
fn i32x4_bitwise() {
    let t = get_x64v3();
    let a = i32x4::<X64V3Token>::from_array(t, [0xFF, 0x0F, 0xF0, 0x00]);
    let b = i32x4::<X64V3Token>::from_array(t, [0x0F, 0x0F, 0x0F, 0x0F]);

    assert_eq!((a & b).to_array(), [0x0F, 0x0F, 0x00, 0x00]);
    assert_eq!((a | b).to_array(), [0xFF, 0x0F, 0xFF, 0x0F]);
    assert_eq!((a ^ b).to_array(), [0xF0, 0x00, 0xFF, 0x0F]);
}

#[test]
fn i32x4_boolean_reductions() {
    let t = get_x64v3();
    let all_set = i32x4::<X64V3Token>::from_array(t, [-1, -1, -1, -1]);
    let none_set = i32x4::<X64V3Token>::from_array(t, [0, 0, 0, 0]);
    let some_set = i32x4::<X64V3Token>::from_array(t, [-1, 0, -1, 0]);

    assert!(all_set.all_true());
    assert!(all_set.any_true());
    assert!(!none_set.all_true());
    assert!(!none_set.any_true());
    assert!(!some_set.all_true());
    assert!(some_set.any_true());
}

#[test]
fn i32x4_bitmask() {
    let t = get_x64v3();
    let v = i32x4::<X64V3Token>::from_array(t, [-1, 0, -1, 0]);
    assert_eq!(v.bitmask(), 0b0101);

    let v2 = i32x4::<X64V3Token>::from_array(t, [0, -1, 0, -1]);
    assert_eq!(v2.bitmask(), 0b1010);
}

#[test]
fn i32x4_scalar_broadcast_ops() {
    let t = get_x64v3();
    let v = i32x4::<X64V3Token>::from_array(t, [10, 20, 30, 40]);
    assert_eq!((v + 5).to_array(), [15, 25, 35, 45]);
    assert_eq!((v - 5).to_array(), [5, 15, 25, 35]);
    assert_eq!((v * 2).to_array(), [20, 40, 60, 80]);
}

#[test]
fn i32x4_indexing() {
    let t = get_x64v3();
    let mut v = i32x4::<X64V3Token>::from_array(t, [1, 2, 3, 4]);
    assert_eq!(v[0], 1);
    assert_eq!(v[3], 4);
    v[2] = 99;
    assert_eq!(v[2], 99);
}

#[test]
fn i32x4_raw_m128i_roundtrip() {
    let t = get_x64v3();
    let v = i32x4::<X64V3Token>::from_array(t, [11, 22, 33, 44]);
    let raw = v.raw();
    let v2 = i32x4::from_m128i(t, raw);
    assert_eq!(v2.to_array(), [11, 22, 33, 44]);
}

// ============================================================================
// i32x8 tests
// ============================================================================

#[test]
fn i32x8_size_matches_m256i() {
    assert_eq!(
        core::mem::size_of::<i32x8<X64V3Token>>(),
        core::mem::size_of::<core::arch::x86_64::__m256i>()
    );
}

#[test]
fn i32x8_basic_arithmetic() {
    let t = get_x64v3();
    let a = i32x8::<X64V3Token>::from_array(t, [1, 2, 3, 4, 5, 6, 7, 8]);
    let b = i32x8::<X64V3Token>::from_array(t, [10, 20, 30, 40, 50, 60, 70, 80]);

    assert_eq!((a + b).to_array(), [11, 22, 33, 44, 55, 66, 77, 88]);
    assert_eq!((b - a).to_array(), [9, 18, 27, 36, 45, 54, 63, 72]);
    assert_eq!((-a).to_array(), [-1, -2, -3, -4, -5, -6, -7, -8]);
}

#[test]
fn i32x8_reduce_add() {
    let t = get_x64v3();
    let v = i32x8::<X64V3Token>::from_array(t, [1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(v.reduce_add(), 36);
}

#[test]
fn i32x8_shifts() {
    let t = get_x64v3();
    let v = i32x8::<X64V3Token>::from_array(t, [1, 2, 4, 8, -1, -2, -4, -8]);
    assert_eq!(
        v.shl_const::<1>().to_array(),
        [2, 4, 8, 16, -2, -4, -8, -16]
    );
}

#[test]
fn i32x8_raw_m256i_roundtrip() {
    let t = get_x64v3();
    let v = i32x8::<X64V3Token>::from_array(t, [1, 2, 3, 4, 5, 6, 7, 8]);
    let raw = v.raw();
    let v2 = i32x8::from_m256i(t, raw);
    assert_eq!(v2.to_array(), [1, 2, 3, 4, 5, 6, 7, 8]);
}

// ============================================================================
// Scalar backend tests
// ============================================================================

#[test]
fn i32x4_scalar_basic() {
    let t = ScalarToken;
    let a = i32x4::<ScalarToken>::from_array(t, [1, 2, 3, 4]);
    let b = i32x4::<ScalarToken>::from_array(t, [10, 20, 30, 40]);

    assert_eq!((a + b).to_array(), [11, 22, 33, 44]);
    assert_eq!(a.min(b).to_array(), [1, 2, 3, 4]);
    assert_eq!(a.max(b).to_array(), [10, 20, 30, 40]);
    assert_eq!(a.reduce_add(), 10);
    assert_eq!(a.shl_const::<1>().to_array(), [2, 4, 6, 8]);
}

#[test]
fn i32x8_scalar_basic() {
    let t = ScalarToken;
    let a = i32x8::<ScalarToken>::from_array(t, [1, -2, 3, -4, 5, -6, 7, -8]);
    assert_eq!(a.abs().to_array(), [1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(a.reduce_add(), -4);
}

// ============================================================================
// Generic function test
// ============================================================================

fn sum_generic<T: I32x4Backend>(token: T, data: &[i32; 4]) -> i32 {
    let v = i32x4::<T>::load(token, data);
    v.reduce_add()
}

#[test]
fn i32x4_generic_fn() {
    let data = [10, 20, 30, 40];

    // X64V3Token
    let t = get_x64v3();
    assert_eq!(sum_generic(t, &data), 100);

    // ScalarToken
    assert_eq!(sum_generic(ScalarToken, &data), 100);
}

// ============================================================================
// Cross-backend consistency
// ============================================================================

#[test]
fn i32x4_x86_scalar_agree() {
    let t_x86 = get_x64v3();
    let t_scalar = ScalarToken;

    let data = [-42, 17, i32::MAX, i32::MIN + 1]; // avoid abs(MIN) overflow
    let x86 = i32x4::<X64V3Token>::from_array(t_x86, data);
    let scalar = i32x4::<ScalarToken>::from_array(t_scalar, data);

    assert_eq!(x86.abs().to_array(), scalar.abs().to_array());
    assert_eq!(x86.reduce_add(), scalar.reduce_add());
    assert_eq!(
        x86.shl_const::<3>().to_array(),
        scalar.shl_const::<3>().to_array()
    );
    assert_eq!(
        x86.shr_arithmetic_const::<2>().to_array(),
        scalar.shr_arithmetic_const::<2>().to_array()
    );
}

// ============================================================================
// Cross-type conversions (f32 ↔ i32)
// ============================================================================

#[test]
fn f32x4_to_i32_truncate() {
    let t = get_x64v3();
    let f = magetypes::simd::generic::f32x4::<X64V3Token>::from_array(t, [1.9, -2.7, 3.1, -4.5]);
    let i = f.to_i32();
    assert_eq!(i.to_array(), [1, -2, 3, -4]);
}

#[test]
fn f32x4_to_i32_round() {
    let t = get_x64v3();
    let f = magetypes::simd::generic::f32x4::<X64V3Token>::from_array(t, [1.5, -2.5, 3.7, -4.3]);
    let i = f.to_i32_round();
    assert_eq!(i.to_array(), [2, -2, 4, -4]); // banker's rounding
}

#[test]
fn i32x4_to_f32() {
    let t = get_x64v3();
    let i = i32x4::<X64V3Token>::from_array(t, [1, -2, 3, -4]);
    let f = i.to_f32();
    assert_eq!(f.to_array(), [1.0, -2.0, 3.0, -4.0]);
}

#[test]
fn f32x4_i32x4_bitcast_roundtrip() {
    let t = get_x64v3();
    let f = magetypes::simd::generic::f32x4::<X64V3Token>::from_array(t, [1.0, -2.0, 3.0, -4.0]);
    let i = f.bitcast_to_i32();
    let f2 = i.bitcast_to_f32();
    assert_eq!(f2.to_array(), [1.0, -2.0, 3.0, -4.0]);
}

#[test]
fn f32x8_to_i32_truncate() {
    let t = get_x64v3();
    let f = magetypes::simd::generic::f32x8::<X64V3Token>::from_array(
        t,
        [1.9, -2.7, 3.1, -4.5, 5.5, -6.1, 7.9, -8.2],
    );
    let i = f.to_i32();
    assert_eq!(i.to_array(), [1, -2, 3, -4, 5, -6, 7, -8]);
}

#[test]
fn f32x8_i32x8_bitcast_roundtrip() {
    let t = get_x64v3();
    let f = magetypes::simd::generic::f32x8::<X64V3Token>::from_array(
        t,
        [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0],
    );
    let i = f.bitcast_to_i32();
    let f2 = i.bitcast_to_f32();
    assert_eq!(f2.to_array(), [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0]);
}

#[test]
fn i32x8_to_f32() {
    let t = get_x64v3();
    let i = i32x8::<X64V3Token>::from_array(t, [1, -2, 3, -4, 5, -6, 7, -8]);
    let f = i.to_f32();
    assert_eq!(f.to_array(), [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0]);
}

// Scalar backend conversions
#[test]
fn f32x4_i32x4_scalar_conversion() {
    let t = ScalarToken;
    let f = magetypes::simd::generic::f32x4::<ScalarToken>::from_array(t, [1.9, -2.7, 3.1, -4.5]);
    let i = f.to_i32();
    assert_eq!(i.to_array(), [1, -2, 3, -4]);

    let f2 = i.to_f32();
    assert_eq!(f2.to_array(), [1.0, -2.0, 3.0, -4.0]);
}

// ============================================================================
// Debug format
// ============================================================================

#[test]
fn i32x4_debug_format() {
    let t = get_x64v3();
    let v = i32x4::<X64V3Token>::from_array(t, [1, 2, 3, 4]);
    let s = format!("{v:?}");
    assert!(s.contains("i32x4"));
    assert!(s.contains("1"));
    assert!(s.contains("4"));
}

//! Auto-generated tests for SIMD types.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

#![cfg(target_arch = "x86_64")]
#![allow(unused)]

use archmage::SimdToken;
use archmage::simd::*;

#[test]
fn test_f32x8_basic() {
    let Some(token) = archmage::Avx2FmaToken::try_new() else {
        eprintln!("AVX2+FMA not available");
        return;
    };

    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let v = f32x8::load(token, &data);

    // Test round-trip
    let out = v.to_array();
    assert_eq!(data, out);

    // Test arithmetic
    let two = f32x8::splat(token, 2.0);
    let sum = v + two;
    let expected = [3.0f32, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    assert_eq!(sum.to_array(), expected);

    // Test min/max
    let a = f32x8::splat(token, 5.0);
    assert_eq!(v.min(a).to_array(), [1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0]);
    assert_eq!(v.max(a).to_array(), [5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 7.0, 8.0]);

    // Test FMA
    let b = f32x8::splat(token, 1.0);
    let fma = v.mul_add(two, b); // v * 2 + 1
    assert_eq!(fma.to_array(), [3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0]);
}

#[test]
fn test_f32x8_comparisons() {
    let Some(token) = archmage::Avx2FmaToken::try_new() else {
        eprintln!("AVX2+FMA not available");
        return;
    };

    let a = f32x8::load(token, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let b = f32x8::load(token, &[2.0, 2.0, 2.0, 2.0, 6.0, 6.0, 6.0, 6.0]);

    // Test simd_lt: lanes where a < b should be all-1s (as f32: NaN)
    let lt = a.simd_lt(b);
    let lt_arr = lt.to_array();
    assert!(lt_arr[0].is_nan()); // 1 < 2 = true (all-1s = NaN)
    assert_eq!(lt_arr[1].to_bits(), 0); // 2 < 2 = false
    assert_eq!(lt_arr[2].to_bits(), 0); // 3 < 2 = false

    // Test simd_eq
    let eq = a.simd_eq(b);
    let eq_arr = eq.to_array();
    assert_eq!(eq_arr[0].to_bits(), 0); // 1 == 2 = false
    assert!(eq_arr[1].is_nan()); // 2 == 2 = true

    // Test simd_gt
    let gt = a.simd_gt(b);
    let gt_arr = gt.to_array();
    assert_eq!(gt_arr[0].to_bits(), 0); // 1 > 2 = false
    assert_eq!(gt_arr[1].to_bits(), 0); // 2 > 2 = false
    assert!(gt_arr[2].is_nan()); // 3 > 2 = true
}

#[test]
fn test_f32x8_blend() {
    let Some(token) = archmage::Avx2FmaToken::try_new() else {
        eprintln!("AVX2+FMA not available");
        return;
    };

    let a = f32x8::load(token, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let b = f32x8::splat(token, 0.0);
    let threshold = f32x8::splat(token, 4.5);

    // Select a where a < threshold, else b
    let mask = a.simd_lt(threshold);
    let result = f32x8::blend(mask, a, b);

    assert_eq!(result.to_array(), [1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn test_f32x8_horizontal() {
    let Some(token) = archmage::Avx2FmaToken::try_new() else {
        eprintln!("AVX2+FMA not available");
        return;
    };

    let v = f32x8::load(token, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    // Sum: 1+2+3+4+5+6+7+8 = 36
    let sum = v.reduce_add();
    assert!((sum - 36.0).abs() < 0.001);

    // Min: 1.0
    let min = v.reduce_min();
    assert!((min - 1.0).abs() < 0.001);

    // Max: 8.0
    let max = v.reduce_max();
    assert!((max - 8.0).abs() < 0.001);
}

#[test]
fn test_f32x8_scalar_ops() {
    let Some(token) = archmage::Avx2FmaToken::try_new() else {
        eprintln!("AVX2+FMA not available");
        return;
    };

    let v = f32x8::load(token, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    // Test v + scalar
    let sum = v + 10.0;
    assert_eq!(sum.to_array(), [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]);

    // Test v * scalar
    let prod = v * 2.0;
    assert_eq!(prod.to_array(), [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);

    // Test v - scalar
    let diff = v - 0.5;
    assert_eq!(diff.to_array(), [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]);

    // Test v / scalar
    let quot = v / 2.0;
    assert_eq!(quot.to_array(), [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]);
}

#[test]
fn test_f32x8_conversions() {
    let Some(token) = archmage::Avx2FmaToken::try_new() else {
        eprintln!("AVX2+FMA not available");
        return;
    };

    // f32 -> i32 (truncate)
    let f = f32x8::load(token, &[1.9, -2.9, 3.1, -4.1, 5.5, -6.5, 7.0, -8.0]);
    let i = f.to_i32x8();
    assert_eq!(i.to_array(), [1, -2, 3, -4, 5, -6, 7, -8]);

    // f32 -> i32 (round)
    let rounded = f.to_i32x8_round();
    assert_eq!(rounded.to_array(), [2, -3, 3, -4, 6, -6, 7, -8]);

    // i32 -> f32
    let i2 = i32x8::load(token, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let f2 = f32x8::from_i32x8(i2);
    assert_eq!(f2.to_array(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn test_i32x8_basic() {
    let Some(token) = archmage::Avx2FmaToken::try_new() else {
        eprintln!("AVX2+FMA not available");
        return;
    };

    let data = [1i32, 2, 3, 4, 5, 6, 7, 8];
    let v = i32x8::load(token, &data);

    // Test round-trip
    assert_eq!(v.to_array(), data);

    // Test arithmetic
    let two = i32x8::splat(token, 2);
    let sum = v + two;
    assert_eq!(sum.to_array(), [3, 4, 5, 6, 7, 8, 9, 10]);

    // Test mul
    let prod = v * two;
    assert_eq!(prod.to_array(), [2, 4, 6, 8, 10, 12, 14, 16]);
}

#[test]
fn test_i32x8_comparisons() {
    let Some(token) = archmage::Avx2FmaToken::try_new() else {
        eprintln!("AVX2+FMA not available");
        return;
    };

    let a = i32x8::load(token, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let b = i32x8::load(token, &[2, 2, 2, 2, 6, 6, 6, 6]);

    // simd_eq: compare each lane
    let eq = a.simd_eq(b);
    let eq_arr = eq.to_array();
    assert_eq!(eq_arr[0], 0);  // 1 == 2 = false
    assert_eq!(eq_arr[1], -1); // 2 == 2 = true (all-1s = -1 as i32)
    assert_eq!(eq_arr[2], 0);  // 3 == 2 = false

    // simd_gt
    let gt = a.simd_gt(b);
    let gt_arr = gt.to_array();
    assert_eq!(gt_arr[0], 0);  // 1 > 2 = false
    assert_eq!(gt_arr[1], 0);  // 2 > 2 = false
    assert_eq!(gt_arr[2], -1); // 3 > 2 = true

    // simd_lt
    let lt = a.simd_lt(b);
    let lt_arr = lt.to_array();
    assert_eq!(lt_arr[0], -1); // 1 < 2 = true
    assert_eq!(lt_arr[1], 0);  // 2 < 2 = false
}

#[test]
fn test_i32x8_scalar_ops() {
    let Some(token) = archmage::Avx2FmaToken::try_new() else {
        eprintln!("AVX2+FMA not available");
        return;
    };

    let v = i32x8::load(token, &[1, 2, 3, 4, 5, 6, 7, 8]);

    // Test v + scalar
    let sum = v + 10;
    assert_eq!(sum.to_array(), [11, 12, 13, 14, 15, 16, 17, 18]);

    // Test v - scalar
    let diff = v - 1;
    assert_eq!(diff.to_array(), [0, 1, 2, 3, 4, 5, 6, 7]);
}

#[test]
fn test_u32x8_comparisons() {
    let Some(token) = archmage::Avx2FmaToken::try_new() else {
        eprintln!("AVX2+FMA not available");
        return;
    };

    // Test unsigned comparison (important: different from signed!)
    let a = u32x8::load(token, &[1, 2, 0xFFFF_FFFF, 4, 5, 6, 7, 8]);
    let b = u32x8::load(token, &[2, 2, 1, 4, 4, 4, 4, 4]);

    // simd_gt for unsigned: 0xFFFF_FFFF > 1 should be true
    let gt = a.simd_gt(b);
    let gt_arr = gt.to_array();
    assert_eq!(gt_arr[0], 0);           // 1 > 2 = false
    assert_eq!(gt_arr[1], 0);           // 2 > 2 = false
    assert_eq!(gt_arr[2], 0xFFFF_FFFF); // 0xFFFF_FFFF > 1 = true (unsigned!)
    assert_eq!(gt_arr[3], 0);           // 4 > 4 = false
    assert_eq!(gt_arr[4], 0xFFFF_FFFF); // 5 > 4 = true
}

#[test]
fn test_f32x4_basic() {
    let Some(token) = archmage::Sse41Token::try_new() else {
        eprintln!("SSE4.1 not available");
        return;
    };

    let data = [1.0f32, 2.0, 3.0, 4.0];
    let v = f32x4::load(token, &data);
    assert_eq!(v.to_array(), data);

    // Test horizontal sum
    let sum = v.reduce_add();
    assert!((sum - 10.0).abs() < 0.001);
}

#[test]
fn test_f32x4_comparisons() {
    let Some(token) = archmage::Sse41Token::try_new() else {
        eprintln!("SSE4.1 not available");
        return;
    };

    let a = f32x4::load(token, &[1.0, 2.0, 3.0, 4.0]);
    let b = f32x4::load(token, &[2.0, 2.0, 2.0, 2.0]);

    // Test simd_lt
    let lt = a.simd_lt(b);
    let lt_arr = lt.to_array();
    assert!(lt_arr[0].is_nan()); // 1 < 2 = true
    assert_eq!(lt_arr[1].to_bits(), 0); // 2 < 2 = false
    assert_eq!(lt_arr[2].to_bits(), 0); // 3 < 2 = false
}

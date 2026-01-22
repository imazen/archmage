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
fn test_f32x4_basic() {
    let Some(token) = archmage::Sse41Token::try_new() else {
        eprintln!("SSE4.1 not available");
        return;
    };

    let data = [1.0f32, 2.0, 3.0, 4.0];
    let v = f32x4::load(token, &data);
    assert_eq!(v.to_array(), data);
}

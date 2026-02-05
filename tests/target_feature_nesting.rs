//! Test: Nested #[target_feature] functions inline without unsafe
//!
//! As of Rust 1.85+, calling a #[target_feature] function from another
//! function with matching (or superset) features is safe - no unsafe needed.

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

// Outer function with target_feature
#[target_feature(enable = "avx2,fma")]
fn outer(data: &[f32; 8]) -> f32 {
    // Call inner helper - NO UNSAFE NEEDED
    // The compiler knows we have avx2,fma enabled here
    let a = inner_add(data);
    let b = inner_mul(data);
    a + b
}

// Inner helper - also has target_feature, but called from matching context
#[target_feature(enable = "avx2")]
#[inline]
fn inner_add(data: &[f32; 8]) -> f32 {
    unsafe {
        let v = _mm256_loadu_ps(data.as_ptr());
        let sum = _mm256_hadd_ps(v, v);
        let sum = _mm256_hadd_ps(sum, sum);
        _mm_cvtss_f32(_mm256_castps256_ps128(sum)) + _mm_cvtss_f32(_mm256_extractf128_ps::<1>(sum))
    }
}

#[target_feature(enable = "avx2")]
#[inline]
fn inner_mul(data: &[f32; 8]) -> f32 {
    unsafe {
        let v = _mm256_loadu_ps(data.as_ptr());
        let prod = _mm256_mul_ps(v, v);
        let sum = _mm256_hadd_ps(prod, prod);
        let sum = _mm256_hadd_ps(sum, sum);
        _mm_cvtss_f32(_mm256_castps256_ps128(sum)) + _mm_cvtss_f32(_mm256_extractf128_ps::<1>(sum))
    }
}

// Test that superset features can call subset features
#[target_feature(enable = "avx2,fma,bmi1,bmi2")]
fn v3_outer(data: &[f32; 8]) -> f32 {
    // v3 (avx2+fma+bmi) can call avx2-only functions safely
    inner_add(data)
}

// Test deeply nested calls
#[target_feature(enable = "avx2,fma")]
fn level1(data: &[f32; 8]) -> f32 {
    level2(data) * 2.0
}

#[target_feature(enable = "avx2,fma")]
#[inline]
fn level2(data: &[f32; 8]) -> f32 {
    level3(data) + 1.0
}

#[target_feature(enable = "avx2")]
#[inline]
fn level3(data: &[f32; 8]) -> f32 {
    inner_add(data)
}

#[test]
fn test_nested_target_feature_no_unsafe() {
    // Runtime check at entry point
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        // SAFETY: We checked for avx2+fma above
        let result = unsafe { outer(&data) };
        println!("outer result: {}", result);

        let result = unsafe { v3_outer(&data) };
        println!("v3_outer result: {}", result);

        let result = unsafe { level1(&data) };
        println!("level1 result: {}", result);
    }
}

// Verify this compiles: the key insight is that inner_add/inner_mul
// are called WITHOUT unsafe blocks from within outer/v3_outer/level2/level3
// because the caller has matching or superset features.

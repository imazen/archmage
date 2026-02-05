//! Tests for the #[rite] macro - inner SIMD helpers without wrapper overhead.

#![cfg(target_arch = "x86_64")]

use archmage::{arcane, rite, Desktop64, SimdToken, X64V3Token};
use std::arch::x86_64::*;

// Helper function using #[rite] - no wrapper, just target_feature annotation
#[rite]
fn add_vectors(_token: X64V3Token, a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
    unsafe {
        let va = _mm256_loadu_ps(a.as_ptr());
        let vb = _mm256_loadu_ps(b.as_ptr());
        let sum = _mm256_add_ps(va, vb);
        let mut out = [0.0f32; 8];
        _mm256_storeu_ps(out.as_mut_ptr(), sum);
        out
    }
}

#[rite]
fn mul_vectors(_token: X64V3Token, a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
    unsafe {
        let va = _mm256_loadu_ps(a.as_ptr());
        let vb = _mm256_loadu_ps(b.as_ptr());
        let prod = _mm256_mul_ps(va, vb);
        let mut out = [0.0f32; 8];
        _mm256_storeu_ps(out.as_mut_ptr(), prod);
        out
    }
}

#[rite]
fn horizontal_sum(_token: X64V3Token, v: __m256) -> f32 {
    // No unsafe needed - value-based intrinsics are safe inside #[target_feature] (Rust 1.85+)
    let sum = _mm256_hadd_ps(v, v);
    let sum = _mm256_hadd_ps(sum, sum);
    let low = _mm256_castps256_ps128(sum);
    let high = _mm256_extractf128_ps::<1>(sum);
    _mm_cvtss_f32(_mm_add_ss(low, high))
}

// Entry point using #[arcane] calls #[rite] helpers
#[arcane]
fn dot_product(token: X64V3Token, a: &[f32; 8], b: &[f32; 8]) -> f32 {
    // These calls should inline directly - no wrapper overhead
    let products = mul_vectors(token, a, b);
    unsafe {
        let v = _mm256_loadu_ps(products.as_ptr());
        horizontal_sum(token, v)
    }
}

// Complex example with multiple #[rite] calls
#[arcane]
fn weighted_sum(
    token: X64V3Token,
    a: &[f32; 8],
    b: &[f32; 8],
    weight_a: f32,
    weight_b: f32,
) -> f32 {
    // Scale a
    let scaled_a = {
        let weights = [weight_a; 8];
        mul_vectors(token, a, &weights)
    };
    // Scale b
    let scaled_b = {
        let weights = [weight_b; 8];
        mul_vectors(token, b, &weights)
    };
    // Add and sum
    let sum = add_vectors(token, &scaled_a, &scaled_b);
    unsafe {
        let v = _mm256_loadu_ps(sum.as_ptr());
        horizontal_sum(token, v)
    }
}

#[test]
fn test_rite_basic() {
    if let Some(token) = X64V3Token::summon() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [1.0f32; 8];

        // Direct call to #[rite] function requires unsafe
        // (Safe when called from #[arcane] context)
        let sum = unsafe { add_vectors(token, &a, &b) };
        assert_eq!(sum, [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }
}

#[test]
fn test_rite_from_arcane() {
    if let Some(token) = X64V3Token::summon() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [2.0f32; 8];

        let result = dot_product(token, &a, &b);
        // 1*2 + 2*2 + 3*2 + 4*2 + 5*2 + 6*2 + 7*2 + 8*2 = 2*(1+2+3+4+5+6+7+8) = 2*36 = 72
        assert_eq!(result, 72.0);
    }
}

#[test]
fn test_rite_complex() {
    if let Some(token) = X64V3Token::summon() {
        let a = [1.0f32; 8];
        let b = [2.0f32; 8];

        let result = weighted_sum(token, &a, &b, 0.5, 0.5);
        // 0.5 * 1.0 * 8 + 0.5 * 2.0 * 8 = 4 + 8 = 12
        assert_eq!(result, 12.0);
    }
}

#[test]
fn test_rite_with_desktop64_alias() {
    if let Some(token) = Desktop64::summon() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [1.0f32; 8];

        // Desktop64 = X64V3Token, so this works
        // Direct call requires unsafe (safe from #[arcane] context)
        let sum = unsafe { add_vectors(token, &a, &b) };
        assert_eq!(sum, [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }
}

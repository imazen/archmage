//! Test: Value intrinsics are SAFE inside #[arcane] (Rust 1.85+)
//!
//! Key insight: Inside #[target_feature], value-based intrinsics are safe.
//! Only pointer-based ops (load/store) still need unsafe.

#![cfg(target_arch = "x86_64")]

use archmage::{Desktop64, Has256BitSimd, SimdToken, arcane};
use std::arch::x86_64::*;

// Value intrinsics - ALL SAFE inside #[arcane], no unsafe needed!
#[arcane]
fn test_value_intrinsics(_token: impl Has256BitSimd, a: __m256, b: __m256) -> __m256 {
    let sum = _mm256_add_ps(a, b);
    let prod = _mm256_mul_ps(sum, sum);
    let blended = _mm256_blend_ps::<0b10101010>(prod, sum);
    _mm256_sqrt_ps(blended)
}

// Reference-based load via safe_unaligned_simd - also safe!
#[arcane]
fn test_safe_load(_token: impl Has256BitSimd, data: &[f32; 8]) -> __m256 {
    safe_unaligned_simd::x86_64::_mm256_loadu_ps(data)
}

#[test]
fn test_value_ops_are_safe() {
    if let Some(token) = Desktop64::summon() {
        let a = unsafe { _mm256_set1_ps(2.0) };
        let b = unsafe { _mm256_set1_ps(3.0) };
        let _ = test_value_intrinsics(token, a, b);
    }
}

#[test]
fn test_safe_simd_load() {
    if let Some(token) = Desktop64::summon() {
        let data = [1.0f32; 8];
        let _ = test_safe_load(token, &data);
    }
}

//! Integration test crate: archmage with NO feature flags.
//!
//! This crate depends on archmage with zero features enabled — no "avx512",
//! no "std". It exercises all dispatch macros to verify they generate valid,
//! warning-free code without any cargo feature gates.
//!
//! If a macro silently wraps variants in `#[cfg(feature = "avx512")]`, those
//! variants become dead code here (the feature doesn't exist in this crate),
//! and `#[deny(dead_code)]` will catch it.
#![deny(warnings)]
#![allow(deprecated)] // Legacy SimdToken usage in autoversion — will migrate to tokenless
#![no_std]

extern crate alloc;
use alloc::vec::Vec;

use archmage::prelude::*;

// ============================================================================
// #[autoversion] — default tiers (includes v4)
// ============================================================================

#[autoversion]
pub fn sum_squares(_token: SimdToken, data: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for &x in data {
        sum += x * x;
    }
    sum
}

#[autoversion]
pub fn scale_vec(_token: SimdToken, data: &[f32], factor: f32) -> Vec<f32> {
    data.iter().map(|&x| x * factor).collect()
}

// ============================================================================
// #[autoversion] — explicit tiers including v4
// ============================================================================

#[autoversion(v3, v4, neon)]
pub fn dot_product(_token: SimdToken, a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut sum = 0.0f32;
    for i in 0..n {
        sum += a[i] * b[i];
    }
    sum
}

// ============================================================================
// #[autoversion] with self receiver
// ============================================================================

pub struct Buffer {
    data: Vec<f32>,
}

impl Buffer {
    #[autoversion]
    pub fn total(&self, _token: SimdToken) -> f32 {
        self.data.iter().sum()
    }
}

// ============================================================================
// Smoke tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn autoversion_default_tiers() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let result = sum_squares(&data);
        assert!((result - 30.0).abs() < 1e-6);
    }

    #[test]
    fn autoversion_explicit_v4_tier() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0];
        let result = dot_product(&a, &b);
        assert!((result - 32.0).abs() < 1e-6);
    }

    #[test]
    fn autoversion_self_receiver() {
        let buf = Buffer {
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        assert!((buf.total() - 10.0).abs() < 1e-6);
    }

    #[test]
    fn autoversion_allocating_return() {
        let data = [1.0f32, 2.0, 3.0];
        let result = scale_vec(&data, 2.0);
        assert_eq!(result, vec![2.0, 4.0, 6.0]);
    }

    /// Verify v4 variant is callable on x86_64 (if CPU supports it).
    /// The point is that it COMPILES — the cfg guard doesn't eliminate it.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn v4_variant_reachable() {
        if let Some(token) = X64V4Token::summon() {
            let data = [1.0f32, 2.0, 3.0, 4.0];
            let result = sum_squares_v4(token, &data);
            assert!((result - 30.0).abs() < 1e-6);
        }
    }

    /// Verify v3 variant is callable on x86_64.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn v3_variant_reachable() {
        if let Some(token) = X64V3Token::summon() {
            let data = [1.0f32, 2.0, 3.0, 4.0];
            let result = sum_squares_v3(token, &data);
            assert!((result - 30.0).abs() < 1e-6);
        }
    }

    #[test]
    fn scalar_variant_reachable() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let result = sum_squares_scalar(ScalarToken, &data);
        assert!((result - 30.0).abs() < 1e-6);
    }
}

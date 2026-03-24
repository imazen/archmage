//! Compile-cost test crate: exercises incant!, #[rite], #[arcane], #[autoversion],
//! and optionally magetypes SIMD types.
//!
//! Feature matrix:
//!   (none)                    — archmage only, no avx512
//!   avx512                    — archmage + avx512
//!   use_magetypes             — archmage + magetypes, no avx512
//!   avx512,use_magetypes      — archmage + magetypes + avx512
#![allow(dead_code)]

use archmage::prelude::*;

// ============================================================================
// #[autoversion]: scalar auto-vectorization
// ============================================================================

#[autoversion]
fn sum_squares(data: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for &x in data {
        acc += x * x;
    }
    acc
}

#[autoversion(-wasm128, +v1)]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    let len = a.len().min(b.len());
    for i in 0..len {
        acc += a[i] * b[i];
    }
    acc
}

// ============================================================================
// #[arcane] + #[rite]: manual SIMD entry point + inlined helper
// ============================================================================

#[cfg(target_arch = "x86_64")]
mod x86 {
    use archmage::prelude::*;

    #[rite(v3)]
    fn add_chunk(a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
        let mut out = [0.0f32; 8];
        for i in 0..8 {
            out[i] = a[i] + b[i];
        }
        out
    }

    #[arcane]
    pub fn add_arrays(token: X64V3Token, a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
        add_chunk(a, b)
    }
}

// ============================================================================
// incant!: runtime dispatch
// ============================================================================

#[arcane]
fn scale_v3(_: X64V3Token, data: &mut [f32], factor: f32) {
    for x in data.iter_mut() {
        *x *= factor;
    }
}

#[arcane]
fn scale_neon(_: NeonToken, data: &mut [f32], factor: f32) {
    for x in data.iter_mut() {
        *x *= factor;
    }
}

fn scale_scalar(_: ScalarToken, data: &mut [f32], factor: f32) {
    for x in data.iter_mut() {
        *x *= factor;
    }
}

pub fn scale(data: &mut [f32], factor: f32) {
    incant!(scale(data, factor), [v3, neon, scalar])
}

// ============================================================================
// magetypes: SIMD types (only when use_magetypes feature is on)
// ============================================================================

#[cfg(feature = "use_magetypes")]
mod mage {
    use archmage::prelude::*;
    use magetypes::simd::backends::F32x8Backend;
    use magetypes::simd::generic::f32x8;

    #[inline(always)]
    fn sum_f32x8<T: F32x8Backend>(token: T, data: &[f32; 8]) -> f32 {
        f32x8::<T>::from_array(token, *data).reduce_add()
    }

    #[cfg(target_arch = "x86_64")]
    #[arcane]
    fn sum_simd_v3(token: X64V3Token, data: &[f32; 8]) -> f32 {
        sum_f32x8(token, data)
    }

    fn sum_simd_scalar(token: ScalarToken, data: &[f32; 8]) -> f32 {
        sum_f32x8(token, data)
    }

    pub fn sum_simd(data: &[f32; 8]) -> f32 {
        incant!(sum_simd(data), [v3, scalar])
    }
}

// ============================================================================
// Public API
// ============================================================================

pub fn api_sum_squares(data: &[f32]) -> f32 {
    sum_squares(data)
}

pub fn api_dot(a: &[f32], b: &[f32]) -> f32 {
    dot_product(a, b)
}

pub fn api_scale(data: &mut [f32], factor: f32) {
    scale(data, factor)
}

#[cfg(feature = "use_magetypes")]
pub fn api_sum_simd(data: &[f32; 8]) -> f32 {
    mage::sum_simd(data)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sum_squares_works() {
        assert!((api_sum_squares(&[1.0, 2.0, 3.0]) - 14.0).abs() < 1e-6);
    }

    #[test]
    fn dot_product_works() {
        assert!((api_dot(&[1.0, 2.0], &[3.0, 4.0]) - 11.0).abs() < 1e-6);
    }

    #[test]
    fn scale_works() {
        let mut data = [1.0f32, 2.0, 3.0];
        api_scale(&mut data, 2.0);
        assert_eq!(data, [2.0, 4.0, 6.0]);
    }

    #[cfg(feature = "use_magetypes")]
    #[test]
    fn sum_simd_works() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert!((api_sum_simd(&data) - 36.0).abs() < 0.01);
    }
}

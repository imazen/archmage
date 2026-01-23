//! Tests for the #[multiwidth] macro.
//!
//! This tests the macro's ability to generate width-specialized modules.

// Only run on x86_64 where we have SSE/AVX2/AVX512
#![cfg(target_arch = "x86_64")]

use archmage::multiwidth;

// Test basic multiwidth functionality - using fixed-size arrays that match each width
#[multiwidth]
#[allow(dead_code)] // Some generated functions may not be called in tests
mod basic_kernels {
    use archmage::simd::*;

    // Process a single vector's worth of data
    pub fn sum_vector(_token: Token, a: f32xN, b: f32xN) -> f32xN {
        a + b
    }

    // Create zero vector
    pub fn make_zero(token: Token) -> f32xN {
        f32xN::zero(token)
    }

    // Splat a value
    pub fn make_splat(token: Token, val: f32) -> f32xN {
        f32xN::splat(token, val)
    }
}

#[test]
fn test_sse_module_exists() {
    // Verify the sse module was generated
    use archmage::SimdToken;

    if let Some(token) = archmage::Sse41Token::try_new() {
        let a = archmage::simd::sse::f32xN::splat(token, 1.0);
        let b = archmage::simd::sse::f32xN::splat(token, 2.0);
        let sum = basic_kernels::sse::sum_vector(token, a, b);
        let result = sum.reduce_add();
        // 4 lanes * 3.0 = 12.0
        assert!(
            (result - 12.0).abs() < 0.001,
            "Expected 12.0, got {}",
            result
        );
    }
}

#[test]
fn test_avx2_module_exists() {
    // Verify the avx2 module was generated
    use archmage::SimdToken;

    if let Some(token) = archmage::Avx2FmaToken::try_new() {
        let a = archmage::simd::avx2::f32xN::splat(token, 1.0);
        let b = archmage::simd::avx2::f32xN::splat(token, 2.0);
        let sum = basic_kernels::avx2::sum_vector(token, a, b);
        let result = sum.reduce_add();
        // 8 lanes * 3.0 = 24.0
        assert!(
            (result - 24.0).abs() < 0.001,
            "Expected 24.0, got {}",
            result
        );
    }
}

#[cfg(feature = "avx512")]
#[test]
fn test_avx512_module_exists() {
    // Verify the avx512 module was generated
    use archmage::SimdToken;

    if let Some(token) = archmage::X64V4Token::try_new() {
        let a = archmage::simd::avx512::f32xN::splat(token, 1.0);
        let b = archmage::simd::avx512::f32xN::splat(token, 2.0);
        let sum = basic_kernels::avx512::sum_vector(token, a, b);
        let result = sum.reduce_add();
        // 16 lanes * 3.0 = 48.0
        assert!(
            (result - 48.0).abs() < 0.001,
            "Expected 48.0, got {}",
            result
        );
    }
}

#[test]
fn test_make_zero() {
    use archmage::SimdToken;

    if let Some(token) = archmage::Avx2FmaToken::try_new() {
        let zero = basic_kernels::avx2::make_zero(token);
        let sum = zero.reduce_add();
        assert!(sum.abs() < 0.001, "Expected 0.0, got {}", sum);
    }
}

#[test]
fn test_make_splat() {
    use archmage::SimdToken;

    if let Some(token) = archmage::Avx2FmaToken::try_new() {
        let splat = basic_kernels::avx2::make_splat(token, 5.0);
        let sum = splat.reduce_add();
        // 8 lanes * 5.0 = 40.0
        assert!((sum - 40.0).abs() < 0.001, "Expected 40.0, got {}", sum);
    }
}

// Test selective width generation
#[multiwidth(avx2)]
mod avx2_only {
    use archmage::simd::*;

    pub fn mul_vectors(_token: Token, a: f32xN, b: f32xN) -> f32xN {
        a * b
    }
}

#[test]
fn test_avx2_only_module() {
    use archmage::SimdToken;

    // avx2 module should exist
    if let Some(token) = archmage::Avx2FmaToken::try_new() {
        let a = archmage::simd::avx2::f32xN::splat(token, 3.0);
        let b = archmage::simd::avx2::f32xN::splat(token, 4.0);
        let result = avx2_only::avx2::mul_vectors(token, a, b);
        let sum = result.reduce_add();
        // 8 lanes * 12.0 = 96.0
        assert!((sum - 96.0).abs() < 0.001, "Expected 96.0, got {}", sum);
    }

    // sse module should NOT exist (compile error if uncommented)
    // avx2_only::sse::mul_vectors(...)
}

// Test multiple widths explicitly
#[multiwidth(sse, avx2)]
mod two_widths {
    use archmage::simd::*;

    pub fn neg_vector(_token: Token, v: f32xN) -> f32xN {
        -v
    }
}

#[test]
fn test_two_widths_sse() {
    use archmage::SimdToken;

    if let Some(token) = archmage::Sse41Token::try_new() {
        let v = archmage::simd::sse::f32xN::splat(token, 5.0);
        let neg = two_widths::sse::neg_vector(token, v);
        let sum = neg.reduce_add();
        // 4 lanes * -5.0 = -20.0
        assert!((sum - (-20.0)).abs() < 0.001, "Expected -20.0, got {}", sum);
    }
}

#[test]
fn test_two_widths_avx2() {
    use archmage::SimdToken;

    if let Some(token) = archmage::Avx2FmaToken::try_new() {
        let v = archmage::simd::avx2::f32xN::splat(token, 5.0);
        let neg = two_widths::avx2::neg_vector(token, v);
        let sum = neg.reduce_add();
        // 8 lanes * -5.0 = -40.0
        assert!((sum - (-40.0)).abs() < 0.001, "Expected -40.0, got {}", sum);
    }
}

// Test dispatcher generation for functions that take/return concrete types
#[multiwidth]
mod dispatchable_kernels {
    use archmage::simd::*;

    /// Sum all elements in a slice using SIMD
    pub fn sum_slice(token: Token, data: &[f32]) -> f32 {
        let mut acc = f32xN::zero(token);
        let chunks = data.chunks_exact(LANES_F32);
        let remainder = chunks.remainder();

        for chunk in chunks {
            // Convert chunk to array reference - this is safe because chunks_exact guarantees size
            let arr: &[f32; LANES_F32] = chunk.try_into().unwrap();
            let v = f32xN::load(token, arr);
            acc += v;
        }

        let mut sum = acc.reduce_add();

        // Handle remainder
        for &x in remainder {
            sum += x;
        }

        sum
    }

    /// Scale all elements in a slice by a factor
    pub fn scale_slice(token: Token, data: &mut [f32], factor: f32) {
        let factor_v = f32xN::splat(token, factor);
        let chunks = data.chunks_exact_mut(LANES_F32);

        for chunk in chunks {
            let arr: &[f32; LANES_F32] = (&*chunk).try_into().unwrap();
            let v = f32xN::load(token, arr);
            let scaled = v * factor_v;
            let out_arr: &mut [f32; LANES_F32] = chunk.try_into().unwrap();
            scaled.store(out_arr);
        }
    }
}

#[test]
fn test_dispatcher_sum_slice() {
    // This tests the auto-generated dispatcher function
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let sum = dispatchable_kernels::sum_slice(&data);
    // 1+2+3+4+5+6+7+8+9+10 = 55
    assert!((sum - 55.0).abs() < 0.001, "Expected 55.0, got {}", sum);
}

#[test]
fn test_dispatcher_scale_slice() {
    // Need at least 16 elements to have a full chunk on AVX-512
    let mut data = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    dispatchable_kernels::scale_slice(&mut data, 2.0);
    assert!((data[0] - 2.0).abs() < 0.001);
    assert!((data[15] - 32.0).abs() < 0.001);
}

#[test]
fn test_direct_sse_sum_slice() {
    use archmage::SimdToken;

    if let Some(token) = archmage::Sse41Token::try_new() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let sum = dispatchable_kernels::sse::sum_slice(token, &data);
        assert!((sum - 36.0).abs() < 0.001, "Expected 36.0, got {}", sum);
    }
}

#[test]
fn test_direct_avx2_sum_slice() {
    use archmage::SimdToken;

    if let Some(token) = archmage::Avx2FmaToken::try_new() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let sum = dispatchable_kernels::avx2::sum_slice(token, &data);
        // 1+2+3+4+5+6+7+8+9 = 45
        assert!((sum - 45.0).abs() < 0.001, "Expected 45.0, got {}", sum);
    }
}

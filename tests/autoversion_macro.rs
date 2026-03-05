//! Integration tests for the #[autoversion] macro.
//!
//! Tests variant generation, dispatch, explicit tiers, self receivers,
//! and correctness across all dispatch paths.

use archmage::prelude::*;

// ============================================================================
// Basic: free function with default tiers
// ============================================================================

#[autoversion]
fn sum_of_squares(_token: SimdToken, data: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for &x in data {
        sum += x * x;
    }
    sum
}

#[test]
fn dispatcher_returns_correct_result() {
    let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let expected: f32 = data.iter().map(|x| x * x).sum();
    let result = sum_of_squares(&data);
    assert!(
        (result - expected).abs() < 1e-3,
        "dispatcher returned {result}, expected {expected}"
    );
}

#[test]
fn scalar_variant_works() {
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let result = sum_of_squares_scalar(ScalarToken, &data);
    assert!((result - 30.0).abs() < 1e-6, "scalar: {result}");
}

#[cfg(target_arch = "x86_64")]
#[test]
fn v3_variant_works() {
    if let Some(token) = X64V3Token::summon() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let result = sum_of_squares_v3(token, &data);
        assert!((result - 30.0).abs() < 1e-6, "v3: {result}");
    }
}

// ============================================================================
// Explicit tiers
// ============================================================================

#[autoversion(v3, neon)]
fn dot_product(_token: SimdToken, a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut sum = 0.0f32;
    for i in 0..n {
        sum += a[i] * b[i];
    }
    sum
}

#[test]
fn explicit_tiers_dispatcher() {
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [4.0f32, 3.0, 2.0, 1.0];
    let result = dot_product(&a, &b);
    assert!((result - 20.0).abs() < 1e-6, "dot: {result}");
}

#[test]
fn explicit_tiers_scalar_variant() {
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [4.0f32, 3.0, 2.0, 1.0];
    let result = dot_product_scalar(ScalarToken, &a, &b);
    assert!((result - 20.0).abs() < 1e-6, "dot scalar: {result}");
}

// ============================================================================
// Multiple non-token parameters
// ============================================================================

#[autoversion]
fn scale_and_offset(_token: SimdToken, data: &[f32], scale: f32, offset: f32) -> Vec<f32> {
    data.iter().map(|&x| x * scale + offset).collect()
}

#[test]
fn multi_param_dispatcher() {
    let data = [1.0f32, 2.0, 3.0];
    let result = scale_and_offset(&data, 2.0, 10.0);
    assert_eq!(result, vec![12.0, 14.0, 16.0]);
}

// ============================================================================
// Mutable slice
// ============================================================================

#[autoversion]
fn normalize_inplace(_token: SimdToken, data: &mut [f32], scale: f32) {
    for x in data.iter_mut() {
        *x *= scale;
    }
}

#[test]
fn mutable_slice_dispatcher() {
    let mut data = vec![1.0f32, 2.0, 3.0, 4.0];
    normalize_inplace(&mut data, 0.5);
    assert_eq!(data, vec![0.5, 1.0, 1.5, 2.0]);
}

// ============================================================================
// Return type: Vec (allocating)
// ============================================================================

#[autoversion]
fn prefix_sums(_token: SimdToken, data: &[f32]) -> Vec<f32> {
    let mut result = Vec::with_capacity(data.len());
    let mut sum = 0.0f32;
    for &x in data {
        sum += x;
        result.push(sum);
    }
    result
}

#[test]
fn allocating_return_type() {
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let result = prefix_sums(&data);
    assert_eq!(result, vec![1.0, 3.0, 6.0, 10.0]);
}

// ============================================================================
// Self receiver: inherent method
// ============================================================================

struct Buffer {
    data: Vec<f32>,
}

impl Buffer {
    #[autoversion(_self = Buffer)]
    fn total(&self, _token: SimdToken) -> f32 {
        _self.data.iter().sum()
    }
}

#[test]
fn self_receiver_dispatcher() {
    let buf = Buffer {
        data: vec![1.0, 2.0, 3.0, 4.0],
    };
    let result = buf.total();
    assert!((result - 10.0).abs() < 1e-6, "total: {result}");
}

// ============================================================================
// Self receiver: &mut self
// ============================================================================

impl Buffer {
    #[autoversion(_self = Buffer)]
    fn scale_all(&mut self, _token: SimdToken, factor: f32) {
        for x in _self.data.iter_mut() {
            *x *= factor;
        }
    }
}

#[test]
fn mut_self_receiver_dispatcher() {
    let mut buf = Buffer {
        data: vec![1.0, 2.0, 3.0, 4.0],
    };
    buf.scale_all(3.0);
    assert_eq!(buf.data, vec![3.0, 6.0, 9.0, 12.0]);
}

// ============================================================================
// Wildcard token parameter
// ============================================================================

#[autoversion]
fn sum_wildcard(_: SimdToken, data: &[f32]) -> f32 {
    data.iter().sum()
}

#[test]
fn wildcard_token_param() {
    let data = [1.0f32, 2.0, 3.0];
    let result = sum_wildcard(&data);
    assert!((result - 6.0).abs() < 1e-6, "wildcard: {result}");
}

// ============================================================================
// Dispatch consistency: all reachable variants produce same result
// ============================================================================

#[test]
fn all_variants_consistent() {
    let data: Vec<f32> = (0..128).map(|i| (i as f32) * 0.1).collect();
    let expected = sum_of_squares_scalar(ScalarToken, &data);

    #[cfg(target_arch = "x86_64")]
    {
        if let Some(t) = X64V3Token::summon() {
            let v3 = sum_of_squares_v3(t, &data);
            assert!(
                (v3 - expected).abs() < 1e-1,
                "v3 ({v3}) != scalar ({expected})"
            );
        }
    }

    // Dispatcher should match scalar within floating-point tolerance
    let dispatched = sum_of_squares(&data);
    assert!(
        (dispatched - expected).abs() < 1e-1,
        "dispatched ({dispatched}) != scalar ({expected})"
    );
}

// ============================================================================
// Empty input edge case
// ============================================================================

#[test]
fn empty_input() {
    let empty: &[f32] = &[];
    assert_eq!(sum_of_squares(empty), 0.0);
    assert_eq!(dot_product(empty, empty), 0.0);
}

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
// Self receiver: &mut self (with _self = Type)
// ============================================================================

impl Buffer {
    #[autoversion(_self = Buffer)]
    fn scale_all(&mut self, _token: SimdToken, factor: f32) {
        for x in _self.data.iter_mut() {
            *x *= factor;
        }
    }
}

// ============================================================================
// Self receiver: plain self (no _self = Type needed for inherent methods)
// ============================================================================

struct Counter {
    values: Vec<f32>,
}

impl Counter {
    #[autoversion]
    fn sum(&self, _token: SimdToken) -> f32 {
        self.values.iter().sum()
    }

    #[autoversion]
    fn double_all(&mut self, _token: SimdToken) {
        for v in self.values.iter_mut() {
            *v *= 2.0;
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

#[test]
fn plain_self_receiver_ref() {
    let c = Counter {
        values: vec![1.0, 2.0, 3.0],
    };
    let result = c.sum();
    assert!((result - 6.0).abs() < 1e-6, "sum: {result}");
}

#[test]
fn plain_self_receiver_mut() {
    let mut c = Counter {
        values: vec![1.0, 2.0, 3.0],
    };
    c.double_all();
    assert_eq!(c.values, vec![2.0, 4.0, 6.0]);
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

// ============================================================================
// Plain self with explicit tiers (no _self = Type)
// ============================================================================

impl Counter {
    #[autoversion(v3, neon)]
    fn product(&self, _token: SimdToken) -> f32 {
        self.values.iter().product()
    }
}

#[test]
fn plain_self_with_explicit_tiers() {
    let c = Counter {
        values: vec![2.0, 3.0, 4.0],
    };
    assert!((c.product() - 24.0).abs() < 1e-6);
}

// ============================================================================
// Plain self with extra parameters (no _self = Type)
// ============================================================================

impl Counter {
    #[autoversion]
    fn weighted_sum(&self, _token: SimdToken, weight: f32) -> f32 {
        self.values.iter().map(|v| v * weight).sum()
    }
}

#[test]
fn plain_self_with_extra_params() {
    let c = Counter {
        values: vec![1.0, 2.0, 3.0],
    };
    assert!((c.weighted_sum(10.0) - 60.0).abs() < 1e-6);
}

// ============================================================================
// Owned self receiver (consumes self)
// ============================================================================

struct OwnedData {
    data: Vec<f32>,
}

impl OwnedData {
    #[autoversion]
    fn into_sum(self, _token: SimdToken) -> f32 {
        self.data.iter().sum()
    }
}

#[test]
fn owned_self_receiver() {
    let d = OwnedData {
        data: vec![1.0, 2.0, 3.0, 4.0],
    };
    assert!((d.into_sum() - 10.0).abs() < 1e-6);
}

// ============================================================================
// Token as first non-self parameter
// ============================================================================

// For methods, self is first and SimdToken is second — this works fine.
// (Already tested above: Counter::sum has &self first, _token second.)
// For free functions, SimdToken must be the first parameter.

// ============================================================================
// Multiple wildcard parameters
// ============================================================================

#[autoversion]
fn add_wildcards(_: SimdToken, _: &[f32], _: &[f32]) -> f32 {
    // all three non-self params are wildcards
    42.0
}

#[test]
fn multiple_wildcards() {
    let a = [1.0f32];
    let b = [2.0f32];
    assert_eq!(add_wildcards(&a, &b), 42.0);
}

// ============================================================================
// Return type: tuple
// ============================================================================

#[autoversion]
fn min_max(_token: SimdToken, data: &[f32]) -> (f32, f32) {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &x in data {
        if x < min {
            min = x;
        }
        if x > max {
            max = x;
        }
    }
    (min, max)
}

#[test]
fn tuple_return_type() {
    let data = [3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let (min, max) = min_max(&data);
    assert!((min - 1.0).abs() < 1e-6);
    assert!((max - 9.0).abs() < 1e-6);
}

// ============================================================================
// Return type: Option
// ============================================================================

#[autoversion]
fn find_first_negative(_token: SimdToken, data: &[f32]) -> Option<usize> {
    data.iter().position(|&x| x < 0.0)
}

#[test]
fn option_return_type() {
    assert_eq!(find_first_negative(&[1.0, -2.0, 3.0]), Some(1));
    assert_eq!(find_first_negative(&[1.0, 2.0, 3.0]), None);
}

// ============================================================================
// Integer data types (not just f32)
// ============================================================================

#[autoversion]
fn sum_i64(_token: SimdToken, data: &[i64]) -> i64 {
    data.iter().sum()
}

#[test]
fn integer_data() {
    let data: Vec<i64> = (1..=100).collect();
    assert_eq!(sum_i64(&data), 5050);
}

// ============================================================================
// Boolean return
// ============================================================================

#[autoversion]
fn all_positive(_token: SimdToken, data: &[f32]) -> bool {
    data.iter().all(|&x| x > 0.0)
}

#[test]
fn boolean_return() {
    assert!(all_positive(&[1.0, 2.0, 3.0]));
    assert!(!all_positive(&[1.0, -2.0, 3.0]));
}

// ============================================================================
// Scalar variant is directly callable
// ============================================================================

#[test]
fn scalar_variants_directly_callable() {
    let data = [1.0f32, 2.0, 3.0, 4.0];
    // Free function scalar variant
    let s = sum_of_squares_scalar(ScalarToken, &data);
    assert!((s - 30.0).abs() < 1e-6);

    // Method scalar variant
    let c = Counter {
        values: vec![1.0, 2.0, 3.0],
    };
    let s = c.sum_scalar(ScalarToken);
    assert!((s - 6.0).abs() < 1e-6);
}

// ============================================================================
// V3 variant (x86_64) is directly callable for methods
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[test]
fn v3_method_variant_directly_callable() {
    if let Some(token) = X64V3Token::summon() {
        let c = Counter {
            values: vec![10.0, 20.0, 30.0],
        };
        let result = c.sum_v3(token);
        assert!((result - 60.0).abs() < 1e-6);
    }
}

// ============================================================================
// Large data (exercise actual auto-vectorization)
// ============================================================================

#[autoversion]
fn sum_large(_token: SimdToken, data: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for &x in data {
        sum += x;
    }
    sum
}

#[test]
fn large_data_auto_vectorized() {
    let data: Vec<f32> = (0..4096).map(|i| i as f32).collect();
    let expected: f32 = (0..4096).map(|i| i as f32).sum();
    let result = sum_large(&data);
    assert!(
        (result - expected).abs() < 1.0,
        "large sum: got {result}, expected {expected}"
    );
}

// ============================================================================
// In-place mutation via mutable reference
// ============================================================================

#[autoversion]
fn clamp_inplace(_token: SimdToken, data: &mut [f32], lo: f32, hi: f32) {
    for x in data.iter_mut() {
        if *x < lo {
            *x = lo;
        }
        if *x > hi {
            *x = hi;
        }
    }
}

#[test]
fn inplace_mutation_with_bounds() {
    let mut data = vec![-5.0, 0.0, 5.0, 10.0, 15.0];
    clamp_inplace(&mut data, 0.0, 10.0);
    assert_eq!(data, vec![0.0, 0.0, 5.0, 10.0, 10.0]);
}

// ============================================================================
// Self receiver + return type that borrows from self
// ============================================================================

impl Counter {
    #[autoversion]
    fn values_ref(&self, _token: SimdToken) -> &[f32] {
        &self.values
    }
}

#[test]
fn self_receiver_borrowing_return() {
    let c = Counter {
        values: vec![1.0, 2.0, 3.0],
    };
    assert_eq!(c.values_ref(), &[1.0, 2.0, 3.0]);
}

// ============================================================================
// Unit return type (no return)
// ============================================================================

#[autoversion]
fn noop(_token: SimdToken, _data: &[f32]) {
    // intentionally empty
}

#[test]
fn unit_return_type() {
    noop(&[1.0, 2.0, 3.0]);
    // if it compiles and doesn't panic, it works
}

// ============================================================================
// Consistency: plain self and _self = Type produce same results
// ============================================================================

struct Accum {
    bias: f32,
}

impl Accum {
    // Plain self (sibling mode)
    #[autoversion]
    fn sum_plain(&self, _token: SimdToken, data: &[f32]) -> f32 {
        self.bias + data.iter().sum::<f32>()
    }

    // _self = Type (nested mode)
    #[autoversion(_self = Accum)]
    fn sum_nested(&self, _token: SimdToken, data: &[f32]) -> f32 {
        _self.bias + data.iter().sum::<f32>()
    }
}

#[test]
fn plain_vs_nested_self_consistent() {
    let a = Accum { bias: 100.0 };
    let data = [1.0f32, 2.0, 3.0];
    let plain = a.sum_plain(&data);
    let nested = a.sum_nested(&data);
    assert!(
        (plain - nested).abs() < 1e-6,
        "plain ({plain}) != nested ({nested})"
    );
    assert!((plain - 106.0).abs() < 1e-6);
}

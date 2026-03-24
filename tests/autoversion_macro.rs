//! Integration tests for the #[autoversion] macro.
//!
//! Tests variant generation, dispatch, explicit tiers, self receivers,
//! and correctness across all dispatch paths.
//!
//! Coverage matrix:
//! - Legacy form (explicit `_token: SimdToken`): full coverage from v0.5
//! - Tokenless form (no SimdToken): parity with legacy form
//! - incant! nesting: autoversioned fn as scalar fallback via bridge
//! - All self receiver types: &self, &mut self, self (owned)
//! - Const generics, type generics, lifetimes
//! - _self = Type nested mode

// Allow deprecated SimdToken usage — these tests intentionally exercise the legacy form
#![allow(deprecated)]

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

/// Verify that v4 variant is generated and callable even without the `avx512`
/// cargo feature. Before the fix, `#[autoversion]` wrapped v4 variants in
/// `#[cfg(feature = "avx512")]`, silently eliminating them in downstream crates.
#[cfg(target_arch = "x86_64")]
#[test]
fn v4_variant_exists_without_avx512_feature() {
    if let Some(token) = X64V4Token::summon() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let result = sum_of_squares_v4(token, &data);
        assert!((result - 30.0).abs() < 1e-6, "v4: {result}");
    }
    // If X64V4Token::summon() returns None, the CPU doesn't support AVX-512
    // — that's fine, the point is that the function EXISTS and compiles.
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
// Const generics
// ============================================================================

#[autoversion]
fn sum_array<const N: usize>(_token: SimdToken, data: &[f32; N]) -> f32 {
    let mut sum = 0.0f32;
    for &x in data {
        sum += x;
    }
    sum
}

#[test]
fn const_generic_basic() {
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let result = sum_array(&data);
    assert!((result - 10.0).abs() < 1e-6, "const generic: {result}");
}

#[test]
fn const_generic_scalar_variant() {
    let data = [1.0f32, 2.0, 3.0];
    let result = sum_array_scalar(ScalarToken, &data);
    assert!(
        (result - 6.0).abs() < 1e-6,
        "const generic scalar: {result}"
    );
}

// Const generic only in return type (not in args)
#[autoversion]
fn make_zeros<const N: usize>(_token: SimdToken) -> [f32; N] {
    [0.0f32; N]
}

#[test]
fn const_generic_return_only() {
    let result: [f32; 4] = make_zeros();
    assert_eq!(result, [0.0; 4]);
}

// Multiple const generics
#[autoversion]
fn reshape<const M: usize, const N: usize>(_token: SimdToken, data: &[f32; M]) -> [f32; N] {
    let mut out = [0.0f32; N];
    let len = M.min(N);
    // manual copy to avoid needing Copy bound shenanigans
    let mut i = 0;
    while i < len {
        out[i] = data[i];
        i += 1;
    }
    out
}

#[test]
fn const_generic_multiple() {
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let result: [f32; 2] = reshape(&data);
    assert_eq!(result, [1.0, 2.0]);
}

// Const generic + type generic
#[autoversion]
fn sum_generic<const N: usize, T: Default + Copy + core::ops::Add<Output = T>>(
    _token: SimdToken,
    data: &[T; N],
) -> T {
    let mut acc = T::default();
    let mut i = 0;
    while i < N {
        acc = acc + data[i];
        i += 1;
    }
    acc
}

#[test]
fn const_generic_plus_type_generic() {
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let result: f32 = sum_generic(&data);
    assert!((result - 10.0).abs() < 1e-6);
}

// Const generic not inferrable from args — only used in body
#[autoversion]
fn chunk_sum<const CHUNK: usize>(_token: SimdToken, data: &[f32]) -> f32 {
    let mut total = 0.0f32;
    for chunk in data.chunks(CHUNK) {
        for &x in chunk {
            total += x;
        }
    }
    total
}

#[test]
fn const_generic_body_only() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let result = chunk_sum::<4>(&data);
    assert!((result - 36.0).abs() < 1e-6, "chunk_sum: {result}");
}

// Const generic with self receiver
struct ConstGenericBuf {
    data: [f32; 4],
}

impl ConstGenericBuf {
    #[autoversion]
    fn extract<const N: usize>(&self, _token: SimdToken) -> [f32; N] {
        let mut out = [0.0f32; N];
        let len = N.min(4);
        let mut i = 0;
        while i < len {
            out[i] = self.data[i];
            i += 1;
        }
        out
    }

    #[autoversion(_self = ConstGenericBuf)]
    fn extract_nested<const N: usize>(&self, _token: SimdToken) -> [f32; N] {
        let mut out = [0.0f32; N];
        let len = N.min(4);
        let mut i = 0;
        while i < len {
            out[i] = _self.data[i];
            i += 1;
        }
        out
    }
}

#[test]
fn const_generic_self_receiver() {
    let buf = ConstGenericBuf {
        data: [1.0, 2.0, 3.0, 4.0],
    };
    let result: [f32; 2] = buf.extract();
    assert_eq!(result, [1.0, 2.0]);
}

#[test]
fn const_generic_nested_self() {
    let buf = ConstGenericBuf {
        data: [1.0, 2.0, 3.0, 4.0],
    };
    let result: [f32; 3] = buf.extract_nested();
    assert_eq!(result, [1.0, 2.0, 3.0]);
}

// Const generic with explicit tiers
#[autoversion(v3, neon)]
fn const_sum_explicit<const N: usize>(_token: SimdToken, data: &[f32; N]) -> f32 {
    let mut s = 0.0f32;
    let mut i = 0;
    while i < N {
        s += data[i];
        i += 1;
    }
    s
}

#[test]
fn const_generic_explicit_tiers() {
    let data = [1.0f32, 2.0, 3.0];
    let result = const_sum_explicit(&data);
    assert!((result - 6.0).abs() < 1e-6);
}

// Const generic with lifetime
#[autoversion]
fn first_n_sum<'a, const N: usize>(_token: SimdToken, data: &'a [f32]) -> f32 {
    let mut s = 0.0f32;
    let end = N.min(data.len());
    let mut i = 0;
    while i < end {
        s += data[i];
        i += 1;
    }
    s
}

#[test]
fn const_generic_with_lifetime() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let result = first_n_sum::<3>(&data);
    assert!((result - 6.0).abs() < 1e-6);
}

// Const generic body-only with self receiver (the BPP pattern from rav1d)
struct PixelRow {
    data: Vec<u8>,
}

impl PixelRow {
    #[autoversion]
    fn fill_row<const BPP: usize>(&self, _token: SimdToken, out: &mut Vec<u8>) {
        for chunk in self.data.chunks(BPP) {
            out.extend_from_slice(chunk);
        }
    }

    #[autoversion(_self = PixelRow)]
    fn fill_row_nested<const BPP: usize>(&self, _token: SimdToken, out: &mut Vec<u8>) {
        for chunk in _self.data.chunks(BPP) {
            out.extend_from_slice(chunk);
        }
    }
}

#[test]
fn const_generic_bpp_pattern() {
    let row = PixelRow {
        data: vec![1, 2, 3, 4, 5, 6],
    };
    let mut out = Vec::new();
    row.fill_row::<3>(&mut out);
    assert_eq!(out, vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn const_generic_bpp_pattern_nested() {
    let row = PixelRow {
        data: vec![1, 2, 3, 4, 5, 6],
    };
    let mut out = Vec::new();
    row.fill_row_nested::<2>(&mut out);
    assert_eq!(out, vec![1, 2, 3, 4, 5, 6]);
}

// Const generic scalar variant directly callable with turbofish
#[test]
fn const_generic_scalar_turbofish() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let result = chunk_sum_scalar::<4>(ScalarToken, &data);
    assert!((result - 36.0).abs() < 1e-6);
}

// Const generic v3 variant directly callable (x86_64)
#[cfg(target_arch = "x86_64")]
#[test]
fn const_generic_v3_turbofish() {
    if let Some(token) = X64V3Token::summon() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = chunk_sum_v3::<4>(token, &data);
        assert!((result - 36.0).abs() < 1e-6);
    }
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

// ============================================================================
// Tokenless: #[autoversion] without explicit SimdToken parameter
// ============================================================================

/// When no SimdToken parameter is present, #[autoversion] auto-injects one.
/// The dispatcher has the original signature (no token), callers unchanged.
#[autoversion]
fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let len = a.len().min(b.len());
    for i in 0..len {
        sum += a[i] * b[i];
    }
    sum
}

#[test]
fn tokenless_dispatcher_works() {
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [4.0f32, 3.0, 2.0, 1.0];
    let result = inner_product(&a, &b);
    assert!((result - 20.0).abs() < 1e-6, "tokenless: {result}");
}

#[test]
fn tokenless_scalar_variant_callable() {
    let a = [1.0f32, 2.0, 3.0];
    let b = [3.0f32, 2.0, 1.0];
    let result = inner_product_scalar(ScalarToken, &a, &b);
    assert!((result - 10.0).abs() < 1e-6, "scalar: {result}");
}

#[cfg(target_arch = "x86_64")]
#[test]
fn tokenless_v3_variant_callable() {
    if let Some(token) = X64V3Token::summon() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [4.0f32, 3.0, 2.0, 1.0];
        let result = inner_product_v3(token, &a, &b);
        assert!((result - 20.0).abs() < 1e-6, "v3: {result}");
    }
}

/// Tokenless with explicit tiers
#[autoversion(v3, neon)]
fn scale_sum(data: &[f32], factor: f32) -> f32 {
    let mut sum = 0.0f32;
    for &x in data {
        sum += x * factor;
    }
    sum
}

#[test]
fn tokenless_explicit_tiers() {
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let result = scale_sum(&data, 2.0);
    assert!((result - 20.0).abs() < 1e-6, "scale_sum: {result}");
}

/// Tokenless with const generics
#[autoversion]
fn fill_chunked<const N: usize>(data: &mut [f32], val: f32) {
    for chunk in data.chunks_mut(N) {
        for x in chunk {
            *x = val;
        }
    }
}

#[test]
fn tokenless_const_generic() {
    let mut data = [0.0f32; 16];
    fill_chunked::<4>(&mut data, 42.0);
    assert!(data.iter().all(|&x| (x - 42.0).abs() < 1e-6));
}

// ============================================================================
// Tokenless self receivers (parity with legacy SimdToken method tests)
// ============================================================================

struct TokenlessBuffer {
    data: Vec<f32>,
}

impl TokenlessBuffer {
    #[autoversion]
    fn total(&self) -> f32 {
        self.data.iter().sum()
    }

    #[autoversion]
    fn scale_all(&mut self, factor: f32) {
        for x in self.data.iter_mut() {
            *x *= factor;
        }
    }

    #[autoversion]
    fn into_total(self) -> f32 {
        self.data.iter().sum()
    }

    #[autoversion]
    fn values_ref(&self) -> &[f32] {
        &self.data
    }

    #[autoversion]
    fn weighted_sum(&self, weight: f32) -> f32 {
        self.data.iter().map(|v| v * weight).sum()
    }

    #[autoversion(v3, neon)]
    fn product(&self) -> f32 {
        self.data.iter().product()
    }
}

#[test]
fn tokenless_ref_self() {
    let buf = TokenlessBuffer {
        data: vec![1.0, 2.0, 3.0, 4.0],
    };
    assert!((buf.total() - 10.0).abs() < 1e-6);
}

#[test]
fn tokenless_mut_self() {
    let mut buf = TokenlessBuffer {
        data: vec![1.0, 2.0, 3.0],
    };
    buf.scale_all(3.0);
    assert_eq!(buf.data, vec![3.0, 6.0, 9.0]);
}

#[test]
fn tokenless_owned_self() {
    let buf = TokenlessBuffer {
        data: vec![1.0, 2.0, 3.0, 4.0],
    };
    assert!((buf.into_total() - 10.0).abs() < 1e-6);
}

#[test]
fn tokenless_borrowing_return() {
    let buf = TokenlessBuffer {
        data: vec![1.0, 2.0, 3.0],
    };
    assert_eq!(buf.values_ref(), &[1.0, 2.0, 3.0]);
}

#[test]
fn tokenless_self_with_extra_params() {
    let buf = TokenlessBuffer {
        data: vec![1.0, 2.0, 3.0],
    };
    assert!((buf.weighted_sum(10.0) - 60.0).abs() < 1e-6);
}

#[test]
fn tokenless_self_explicit_tiers() {
    let buf = TokenlessBuffer {
        data: vec![2.0, 3.0, 4.0],
    };
    assert!((buf.product() - 24.0).abs() < 1e-6);
}

#[test]
fn tokenless_scalar_method_variant() {
    let buf = TokenlessBuffer {
        data: vec![1.0, 2.0, 3.0],
    };
    let s = buf.total_scalar(ScalarToken);
    assert!((s - 6.0).abs() < 1e-6);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn tokenless_v3_method_variant() {
    if let Some(token) = X64V3Token::summon() {
        let buf = TokenlessBuffer {
            data: vec![10.0, 20.0, 30.0],
        };
        let result = buf.total_v3(token);
        assert!((result - 60.0).abs() < 1e-6);
    }
}

// ============================================================================
// Tokenless _self = Type (nested mode)
// ============================================================================

struct TokenlessNested {
    bias: f32,
}

impl TokenlessNested {
    #[autoversion(_self = TokenlessNested)]
    fn biased_sum(&self, data: &[f32]) -> f32 {
        _self.bias + data.iter().sum::<f32>()
    }

    #[autoversion(_self = TokenlessNested)]
    fn biased_scale(&mut self, data: &[f32], factor: f32) -> f32 {
        _self.bias * factor + data.iter().sum::<f32>()
    }
}

#[test]
fn tokenless_nested_ref_self() {
    let n = TokenlessNested { bias: 100.0 };
    let data = [1.0f32, 2.0, 3.0];
    assert!((n.biased_sum(&data) - 106.0).abs() < 1e-6);
}

#[test]
fn tokenless_nested_mut_self() {
    let mut n = TokenlessNested { bias: 10.0 };
    let data = [1.0f32, 2.0, 3.0];
    assert!((n.biased_scale(&data, 2.0) - 26.0).abs() < 1e-6);
}

// ============================================================================
// Tokenless parity: explicit vs tokenless produce identical results
// ============================================================================

#[autoversion]
fn parity_explicit(_token: SimdToken, data: &[f32]) -> f32 {
    data.iter().map(|x| x * x).sum()
}

#[autoversion]
fn parity_tokenless(data: &[f32]) -> f32 {
    data.iter().map(|x| x * x).sum()
}

#[test]
fn explicit_and_tokenless_produce_same_result() {
    let data: Vec<f32> = (0..256).map(|i| i as f32 * 0.1).collect();
    let explicit = parity_explicit(&data);
    let tokenless = parity_tokenless(&data);
    assert!(
        (explicit - tokenless).abs() < 1e-3,
        "explicit ({explicit}) != tokenless ({tokenless})"
    );
}

#[test]
fn explicit_and_tokenless_scalar_variants_match() {
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let explicit = parity_explicit_scalar(ScalarToken, &data);
    let tokenless = parity_tokenless_scalar(ScalarToken, &data);
    assert!(
        (explicit - tokenless).abs() < 1e-6,
        "scalar: explicit ({explicit}) != tokenless ({tokenless})"
    );
}

// ============================================================================
// incant! nesting: hand-written + autoversioned scalar fallback via bridge
// ============================================================================

/// Hand-written v3 — uses a distinctive multiplier so we can tell which path ran
#[cfg(target_arch = "x86_64")]
#[arcane]
fn nested_dispatch_v3(_token: X64V3Token, data: &[f32]) -> f32 {
    // Real code would use intrinsics. Multiplier identifies this path.
    data.iter().sum::<f32>() * 1000.0
}

/// Autoversioned fallback (inner) — tokenless, does its own dispatch
#[autoversion(v3, neon)]
fn nested_dispatch_fallback(data: &[f32]) -> f32 {
    data.iter().sum()
}

/// Bridge: incant! passes ScalarToken; bridge calls tokenless autoversion
fn nested_dispatch_scalar(_: ScalarToken, data: &[f32]) -> f32 {
    nested_dispatch_fallback(data)
}

/// Top-level: incant! dispatches to hand-written v3 or autoversioned fallback
fn nested_dispatch(data: &[f32]) -> f32 {
    incant!(nested_dispatch(data), [v3, scalar])
}

#[test]
fn incant_nesting_dispatches_correctly() {
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let result = nested_dispatch(&data);
    // Should produce a finite result regardless of which path
    assert!(result.is_finite(), "nested dispatch: {result}");
}

#[test]
fn incant_nesting_scalar_fallback_works() {
    let data = [1.0f32, 2.0, 3.0, 4.0];
    // Call the bridge directly — simulates what incant! does for scalar
    let result = nested_dispatch_scalar(ScalarToken, &data);
    assert!((result - 10.0).abs() < 1e-6, "scalar bridge: {result}");
}

#[test]
fn incant_nesting_autoversion_fallback_directly() {
    let data = [1.0f32, 2.0, 3.0, 4.0];
    // Call the autoversioned dispatcher directly
    let result = nested_dispatch_fallback(&data);
    assert!(
        (result - 10.0).abs() < 1e-6,
        "autoversion fallback: {result}"
    );
}

#[test]
fn incant_nesting_fallback_scalar_variant() {
    let data = [1.0f32, 2.0, 3.0, 4.0];
    // Call the scalar variant of the autoversioned fallback
    let result = nested_dispatch_fallback_scalar(ScalarToken, &data);
    assert!(
        (result - 10.0).abs() < 1e-6,
        "fallback scalar variant: {result}"
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn incant_nesting_v3_path_identified() {
    if let Some(token) = X64V3Token::summon() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        // The hand-written v3 multiplies by 1000
        let v3_result = nested_dispatch_v3(token, &data);
        assert!(
            (v3_result - 10000.0).abs() < 1e-3,
            "hand-written v3: {v3_result}"
        );
        // The top-level dispatcher should pick v3 on x86_64 with AVX2
        let dispatched = nested_dispatch(&data);
        assert!(
            (dispatched - 10000.0).abs() < 1e-3,
            "dispatched should pick v3: {dispatched}"
        );
    }
}

// ============================================================================
// incant! nesting with methods
// ============================================================================

struct NestedProcessor {
    scale: f32,
}

impl NestedProcessor {
    /// Top-level: manual dispatch via if/else, calling hand-written or bridge
    pub fn process(&self, data: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        if let Some(token) = X64V3Token::summon() {
            return self.process_v3(token, data);
        }
        self.process_scalar(ScalarToken, data)
    }

    #[cfg(target_arch = "x86_64")]
    #[arcane]
    fn process_v3(&self, _token: X64V3Token, data: &[f32]) -> f32 {
        data.iter().sum::<f32>() * self.scale * 100.0 // distinctive
    }

    fn process_scalar(&self, _: ScalarToken, data: &[f32]) -> f32 {
        self.process_auto(data)
    }

    #[autoversion(v3, neon)]
    fn process_auto(&self, data: &[f32]) -> f32 {
        data.iter().sum::<f32>() * self.scale
    }
}

#[test]
fn incant_nesting_method_dispatches() {
    let p = NestedProcessor { scale: 2.0 };
    let data = [1.0f32, 2.0, 3.0];
    let result = p.process(&data);
    assert!(result.is_finite(), "method nested dispatch: {result}");
}

#[test]
fn incant_nesting_method_scalar_bridge() {
    let p = NestedProcessor { scale: 2.0 };
    let data = [1.0f32, 2.0, 3.0];
    let result = p.process_scalar(ScalarToken, &data);
    assert!(
        (result - 12.0).abs() < 1e-6,
        "method scalar bridge: {result}"
    );
}

#[test]
fn incant_nesting_method_auto_directly() {
    let p = NestedProcessor { scale: 3.0 };
    let data = [1.0f32, 2.0, 3.0];
    let result = p.process_auto(&data);
    assert!(
        (result - 18.0).abs() < 1e-6,
        "method auto dispatch: {result}"
    );
}

// ============================================================================
// ScalarToken nesting: no bridge needed
// ============================================================================

/// Hand-written v3 for the bridgeless nesting test
#[cfg(target_arch = "x86_64")]
#[arcane]
fn bridgeless_v3(_token: X64V3Token, data: &[f32]) -> f32 {
    data.iter().sum::<f32>() * 1000.0 // distinctive
}

/// ScalarToken autoversion: the dispatcher IS the incant! scalar target.
/// No bridge function needed — ScalarToken is kept in the dispatcher signature.
#[autoversion(v3, neon)]
fn bridgeless_scalar(_: ScalarToken, data: &[f32]) -> f32 {
    data.iter().sum()
}

/// Top-level: incant! dispatches to hand-written v3 or ScalarToken autoversion
fn bridgeless(data: &[f32]) -> f32 {
    incant!(bridgeless(data), [v3, scalar])
}

#[test]
fn scalar_token_nesting_dispatches() {
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let result = bridgeless(&data);
    assert!(result.is_finite(), "bridgeless dispatch: {result}");
}

#[test]
fn scalar_token_nesting_scalar_directly() {
    let data = [1.0f32, 2.0, 3.0, 4.0];
    // incant! calls this with ScalarToken — and it works directly
    let result = bridgeless_scalar(ScalarToken, &data);
    assert!((result - 10.0).abs() < 1e-6, "scalar direct: {result}");
}

#[test]
fn scalar_token_nesting_scalar_variant() {
    let data = [1.0f32, 2.0, 3.0, 4.0];
    // The autoversion's own _scalar variant
    let result = bridgeless_scalar_scalar(ScalarToken, &data);
    assert!((result - 10.0).abs() < 1e-6, "scalar variant: {result}");
}

#[cfg(target_arch = "x86_64")]
#[test]
fn scalar_token_nesting_v3_path() {
    if let Some(token) = X64V3Token::summon() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        // Hand-written v3 has ×1000 multiplier
        let v3_result = bridgeless_v3(token, &data);
        assert!(
            (v3_result - 10000.0).abs() < 1e-3,
            "hand-written v3: {v3_result}"
        );
        // Top-level should pick v3
        let dispatched = bridgeless(&data);
        assert!(
            (dispatched - 10000.0).abs() < 1e-3,
            "should pick v3: {dispatched}"
        );
    }
}

// ScalarToken nesting with method
struct BridgelessProcessor {
    scale: f32,
}

impl BridgelessProcessor {
    pub fn process(&self, data: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        if let Some(token) = X64V3Token::summon() {
            return self.process_v3(token, data);
        }
        self.process_scalar(ScalarToken, data)
    }

    #[cfg(target_arch = "x86_64")]
    #[arcane]
    fn process_v3(&self, _token: X64V3Token, data: &[f32]) -> f32 {
        data.iter().sum::<f32>() * self.scale * 100.0
    }

    /// ScalarToken autoversion: dispatcher IS the scalar target
    #[autoversion(v3, neon)]
    fn process_scalar(&self, _: ScalarToken, data: &[f32]) -> f32 {
        data.iter().sum::<f32>() * self.scale
    }
}

#[test]
fn scalar_token_nesting_method() {
    let p = BridgelessProcessor { scale: 2.0 };
    let data = [1.0f32, 2.0, 3.0];
    let result = p.process(&data);
    assert!(result.is_finite(), "method bridgeless: {result}");
}

#[test]
fn scalar_token_nesting_method_scalar_directly() {
    let p = BridgelessProcessor { scale: 2.0 };
    let data = [1.0f32, 2.0, 3.0];
    let result = p.process_scalar(ScalarToken, &data);
    assert!(
        (result - 12.0).abs() < 1e-6,
        "method scalar direct: {result}"
    );
}

// ============================================================================
// `default` tier: tokenless fallback for incant! nesting
// ============================================================================

/// Hand-written v3
#[cfg(target_arch = "x86_64")]
#[arcane]
fn default_tier_v3(_token: X64V3Token, data: &[f32]) -> f32 {
    data.iter().sum::<f32>() * 1000.0
}

/// Autoversioned tokenless fallback — named _default, no token needed
#[autoversion(v3, neon)]
fn default_tier_default(data: &[f32]) -> f32 {
    data.iter().sum()
}

/// Top-level: incant! with `default` instead of `scalar`
fn default_tier(data: &[f32]) -> f32 {
    incant!(default_tier(data), [v3, default])
}

#[test]
fn default_tier_dispatches() {
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let result = default_tier(&data);
    assert!(result.is_finite(), "default tier: {result}");
}

#[test]
fn default_tier_fallback_directly() {
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let result = default_tier_default(&data);
    assert!((result - 10.0).abs() < 1e-6, "default fallback: {result}");
}

#[cfg(target_arch = "x86_64")]
#[test]
fn default_tier_picks_v3() {
    if let Some(token) = X64V3Token::summon() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let v3 = default_tier_v3(token, &data);
        assert!((v3 - 10000.0).abs() < 1e-3, "v3: {v3}");
        let dispatched = default_tier(&data);
        assert!(
            (dispatched - 10000.0).abs() < 1e-3,
            "dispatch: {dispatched}"
        );
    }
}

// default tier with autoversion(default) — autoversion generates _default variant
#[autoversion(v3, neon, default)]
fn auto_with_default(data: &[f32]) -> f32 {
    data.iter().sum()
}

#[test]
fn autoversion_default_tier() {
    let data = [1.0f32, 2.0, 3.0];
    let result = auto_with_default(&data);
    assert!((result - 6.0).abs() < 1e-6, "auto default: {result}");
}

#[test]
fn autoversion_default_variant_callable() {
    let data = [1.0f32, 2.0, 3.0];
    // _default variant is tokenless — call without any token
    let result = auto_with_default_default(&data);
    assert!((result - 6.0).abs() < 1e-6, "default variant: {result}");
}

// default tier with method
struct DefaultProcessor {
    scale: f32,
}

impl DefaultProcessor {
    pub fn process(&self, data: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        if let Some(token) = X64V3Token::summon() {
            return self.process_v3(token, data);
        }
        self.process_default(data)
    }

    #[cfg(target_arch = "x86_64")]
    #[arcane]
    fn process_v3(&self, _token: X64V3Token, data: &[f32]) -> f32 {
        data.iter().sum::<f32>() * self.scale * 100.0
    }

    #[autoversion(v3, neon)]
    fn process_default(&self, data: &[f32]) -> f32 {
        data.iter().sum::<f32>() * self.scale
    }
}

#[test]
fn default_tier_method() {
    let p = DefaultProcessor { scale: 2.0 };
    let data = [1.0f32, 2.0, 3.0];
    let result = p.process(&data);
    assert!(result.is_finite(), "method default: {result}");
}

#[test]
fn default_tier_method_fallback() {
    let p = DefaultProcessor { scale: 2.0 };
    let data = [1.0f32, 2.0, 3.0];
    let result = p.process_default(&data);
    assert!((result - 12.0).abs() < 1e-6, "method default: {result}");
}

// ============================================================================
// Nesting pitfall: autoversion on a _scalar function
//
// The natural mistake: use incant! with [v3, scalar], then autoversion
// the scalar fallback. The autoversion dispatcher strips the token,
// but incant! calls _scalar(ScalarToken, data) — signature mismatch.
//
// This section tests both WRONG patterns (that users might try) and
// the CORRECT solutions (ScalarToken param, default tier, bridge).
// ============================================================================

// --- Solution 1: ScalarToken param (autoversion keeps it) ---

#[cfg(target_arch = "x86_64")]
#[arcane]
fn resample_scalar_sol1_v3(_: X64V3Token, data: &[f32], factor: f32) -> f32 {
    data.iter().map(|x| x * factor).sum::<f32>() * 1000.0
}

#[autoversion(v3, neon)]
fn resample_scalar_sol1_scalar(_: ScalarToken, data: &[f32], factor: f32) -> f32 {
    data.iter().map(|x| x * factor).sum()
}

fn resample_scalar_sol1(data: &[f32], factor: f32) -> f32 {
    incant!(resample_scalar_sol1(data, factor), [v3, scalar])
}

#[test]
fn nesting_pitfall_scalar_token_solution() {
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let result = resample_scalar_sol1(&data, 2.0);
    assert!(result.is_finite(), "ScalarToken solution: {result}");
    // Direct call to the autoversioned scalar fallback — with ScalarToken
    let scalar = resample_scalar_sol1_scalar(ScalarToken, &data, 2.0);
    assert!((scalar - 20.0).abs() < 1e-6, "direct scalar: {scalar}");
}

// --- Solution 2: default tier (tokenless, cleanest) ---

#[cfg(target_arch = "x86_64")]
#[arcane]
fn resample_default_sol2_v3(_: X64V3Token, data: &[f32], factor: f32) -> f32 {
    data.iter().map(|x| x * factor).sum::<f32>() * 1000.0
}

#[autoversion(v3, neon)]
fn resample_default_sol2_default(data: &[f32], factor: f32) -> f32 {
    data.iter().map(|x| x * factor).sum()
}

fn resample_default_sol2(data: &[f32], factor: f32) -> f32 {
    incant!(resample_default_sol2(data, factor), [v3, default])
}

#[test]
fn nesting_pitfall_default_tier_solution() {
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let result = resample_default_sol2(&data, 2.0);
    assert!(result.is_finite(), "default tier solution: {result}");
    // Direct call — no token at all
    let fallback = resample_default_sol2_default(&data, 2.0);
    assert!((fallback - 20.0).abs() < 1e-6, "direct default: {fallback}");
}

// --- Solution 3: bridge function (works with plain scalar tier) ---

#[cfg(target_arch = "x86_64")]
#[arcane]
fn resample_bridge_sol3_v3(_: X64V3Token, data: &[f32], factor: f32) -> f32 {
    data.iter().map(|x| x * factor).sum::<f32>() * 1000.0
}

#[autoversion(v3, neon)]
fn resample_bridge_sol3_auto(data: &[f32], factor: f32) -> f32 {
    data.iter().map(|x| x * factor).sum()
}

// Bridge: incant! passes ScalarToken, bridge forwards tokenlessly
fn resample_bridge_sol3_scalar(_: ScalarToken, data: &[f32], factor: f32) -> f32 {
    resample_bridge_sol3_auto(data, factor)
}

fn resample_bridge_sol3(data: &[f32], factor: f32) -> f32 {
    incant!(resample_bridge_sol3(data, factor), [v3, scalar])
}

#[test]
fn nesting_pitfall_bridge_solution() {
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let result = resample_bridge_sol3(&data, 2.0);
    assert!(result.is_finite(), "bridge solution: {result}");
}

// --- Verify all three solutions produce the same scalar result ---

#[test]
fn nesting_pitfall_all_solutions_agree() {
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let factor = 2.0;
    let s1 = resample_scalar_sol1_scalar(ScalarToken, &data, factor);
    let s2 = resample_default_sol2_default(&data, factor);
    let s3 = resample_bridge_sol3_auto(&data, factor);
    assert!((s1 - s2).abs() < 1e-6, "sol1 ({s1}) != sol2 ({s2})");
    assert!((s2 - s3).abs() < 1e-6, "sol2 ({s2}) != sol3 ({s3})");
}

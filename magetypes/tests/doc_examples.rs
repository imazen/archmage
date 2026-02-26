//! Compiled tests for every code example in the magetypes documentation.
//!
//! Each module corresponds to a doc page under `docs/site/content/magetypes/`.
//! If you change a doc example, update the corresponding test here.
//! If you add a new doc page with code, add a test module.
//!
//! Run: `cargo test -p magetypes --test doc_examples`

#![allow(dead_code, unused_variables)]

use archmage::{ScalarToken, SimdToken};
use magetypes::simd::backends::{F32x8Backend, F32x8Convert, I32x8Backend};
use magetypes::simd::generic::{f32x8, i32x8};

#[cfg(target_arch = "x86_64")]
use archmage::X64V3Token;

// ============================================================================
// magetypes/_index.md — Introduction
// ============================================================================
mod index_page {
    use super::*;

    fn example<T: F32x8Backend>(token: T) {
        let a = f32x8::<T>::from_array(token, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let b = f32x8::<T>::splat(token, 2.0);
        let c = a * b;
        let sum = c.reduce_add();
        // (1+2+3+4+5+6+7+8) * 2 = 72
        assert!((sum - 72.0).abs() < 0.01);
    }

    #[test]
    fn intro_example_scalar() {
        example(ScalarToken);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn intro_example_x86() {
        if let Some(token) = X64V3Token::summon() {
            example(token);
        }
    }
}

// ============================================================================
// getting-started/installation.md — Verify It Works
// ============================================================================
mod installation {
    use super::*;

    fn verify<T: F32x8Backend>(token: T) {
        let v = f32x8::<T>::splat(token, 42.0);
        assert_eq!(v.to_array(), [42.0; 8]);
    }

    #[test]
    fn verify_scalar() {
        verify(ScalarToken);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn verify_x86() {
        if let Some(token) = X64V3Token::summon() {
            verify(token);
        }
    }
}

// ============================================================================
// getting-started/first-types.md — Your First Types
// ============================================================================
mod first_types {
    use super::*;

    // The Pattern: summon → construct → operate → extract
    fn the_pattern<T: F32x8Backend>(token: T) {
        let a = f32x8::<T>::from_array(token, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let b = f32x8::<T>::splat(token, 2.0);
        let c = a * b;
        let result: [f32; 8] = c.to_array();
        assert_eq!(result, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
    }

    // Summon once, pass to functions
    fn process_data<T: F32x8Backend>(token: T, input: &[f32; 8]) -> f32 {
        let a = f32x8::<T>::from_array(token, *input);
        let b = f32x8::<T>::splat(token, 0.5);
        let scaled = a * b;
        scaled.reduce_add()
    }

    // Type properties: Copy
    fn copy_demo<T: F32x8Backend>(token: T) {
        let a = f32x8::<T>::splat(token, 1.0);
        let b = a; // Copy
        let c = a + b; // Both still valid
        assert_eq!(c.to_array(), [2.0; 8]);
    }

    #[test]
    fn pattern_scalar() {
        the_pattern(ScalarToken);
    }

    #[test]
    fn process_data_scalar() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = process_data(ScalarToken, &data);
        assert!((result - 18.0).abs() < 0.01);
    }

    #[test]
    fn copy_demo_scalar() {
        copy_demo(ScalarToken);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn pattern_x86() {
        if let Some(token) = X64V3Token::summon() {
            the_pattern(token);
        }
    }
}

// ============================================================================
// types/overview.md — Type Overview
// ============================================================================
mod overview {
    use super::*;

    fn basic_usage<T: F32x8Backend>(token: T) {
        let a = f32x8::<T>::from_array(token, [1.0; 8]);
        let b = f32x8::<T>::splat(token, 2.0);
        let c = a + b;
        let d = c * c;
        let result: [f32; 8] = d.to_array();
        assert_eq!(result, [9.0; 8]);
    }

    #[test]
    fn basic_usage_scalar() {
        basic_usage(ScalarToken);
    }
}

// ============================================================================
// operations/construction.md — Construction & Extraction
// ============================================================================
mod construction {
    use super::*;

    fn constructors<T: F32x8Backend>(token: T) {
        // from_array
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let v = f32x8::<T>::from_array(token, data);
        assert_eq!(v.to_array(), data);

        // splat
        let v = f32x8::<T>::splat(token, 3.14159);
        assert!((v[0] - 3.14159).abs() < 1e-5);

        // zero
        let v = f32x8::<T>::zero(token);
        assert_eq!(v.to_array(), [0.0; 8]);

        // load from array reference
        let v = f32x8::<T>::load(token, &data);
        assert_eq!(v.to_array(), data);
    }

    fn extraction<T: F32x8Backend>(token: T) {
        let v = f32x8::<T>::from_array(token, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        // to_array
        let arr: [f32; 8] = v.to_array();
        assert_eq!(arr[0], 1.0);

        // store
        let mut buf = [0.0f32; 8];
        v.store(&mut buf);
        assert_eq!(buf[0], 1.0);

        // lane access
        assert_eq!(v[0], 1.0);
        assert_eq!(v[2], 3.0);
    }

    #[test]
    fn constructors_scalar() {
        constructors(ScalarToken);
    }

    #[test]
    fn extraction_scalar() {
        extraction(ScalarToken);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn constructors_x86() {
        if let Some(token) = X64V3Token::summon() {
            constructors(token);
        }
    }
}

// ============================================================================
// operations/operators.md — Arithmetic & Comparisons
// ============================================================================
mod operators {
    use super::*;

    fn arithmetic<T: F32x8Backend>(token: T) {
        let a = f32x8::<T>::splat(token, 2.0);
        let b = f32x8::<T>::splat(token, 3.0);

        let sum = a + b;
        assert_eq!(sum.to_array(), [5.0; 8]);

        let diff = a - b;
        assert_eq!(diff.to_array(), [-1.0; 8]);

        let prod = a * b;
        assert_eq!(prod.to_array(), [6.0; 8]);

        let neg = -a;
        assert_eq!(neg.to_array(), [-2.0; 8]);
    }

    fn fma<T: F32x8Backend>(token: T) {
        let a = f32x8::<T>::splat(token, 2.0);
        let b = f32x8::<T>::splat(token, 3.0);
        let c = f32x8::<T>::splat(token, 1.0);

        let result = a.mul_add(b, c);
        assert_eq!(result.to_array(), [7.0; 8]);
    }

    fn comparisons<T: F32x8Backend>(token: T) {
        let a = f32x8::<T>::from_array(token, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let b = f32x8::<T>::splat(token, 4.0);

        let mask = a.simd_lt(b);
        let result = f32x8::<T>::blend(mask, f32x8::<T>::splat(token, 0.0), a);
        // Where a < 4: 0.0; else: a
        assert_eq!(result[0], 0.0);
        assert_eq!(result[3], 4.0); // 4.0 is NOT < 4.0
    }

    fn min_max<T: F32x8Backend>(token: T) {
        let a = f32x8::<T>::from_array(token, [1.0, 5.0, 3.0, 7.0, 2.0, 6.0, 4.0, 8.0]);
        let b = f32x8::<T>::splat(token, 4.0);

        let min = a.min(b);
        assert_eq!(min[0], 1.0);
        assert_eq!(min[1], 4.0);

        let max = a.max(b);
        assert_eq!(max[0], 4.0);
        assert_eq!(max[1], 5.0);
    }

    fn dot_product<T: F32x8Backend>(token: T, a: &[f32; 8], b: &[f32; 8]) -> f32 {
        let va = f32x8::<T>::from_array(token, *a);
        let vb = f32x8::<T>::from_array(token, *b);
        (va * vb).reduce_add()
    }

    #[test]
    fn arithmetic_scalar() {
        arithmetic(ScalarToken);
    }

    #[test]
    fn fma_scalar() {
        fma(ScalarToken);
    }

    #[test]
    fn comparisons_scalar() {
        comparisons(ScalarToken);
    }

    #[test]
    fn min_max_scalar() {
        min_max(ScalarToken);
    }

    #[test]
    fn dot_product_scalar() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let result = dot_product(ScalarToken, &a, &b);
        assert!((result - 120.0).abs() < 0.01);
    }
}

// ============================================================================
// operations/reductions.md — Reductions
// ============================================================================
mod reductions {
    use super::*;

    fn basic_reductions<T: F32x8Backend>(token: T) {
        let v = f32x8::<T>::from_array(token, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        let sum = v.reduce_add();
        assert!((sum - 36.0).abs() < 0.01);

        let max = v.reduce_max();
        assert_eq!(max, 8.0);

        let min = v.reduce_min();
        assert_eq!(min, 1.0);
    }

    fn find_max<T: F32x8Backend>(token: T, data: &[f32]) -> f32 {
        let chunks = data.chunks_exact(8);
        let remainder = chunks.remainder();

        let mut max_v = f32x8::<T>::splat(token, f32::NEG_INFINITY);
        for chunk in chunks {
            let v = f32x8::<T>::load(token, chunk.try_into().unwrap());
            max_v = max_v.max(v);
        }

        let mut result = max_v.reduce_max();
        for &x in remainder {
            if x > result {
                result = x;
            }
        }
        result
    }

    #[test]
    fn basic_reductions_scalar() {
        basic_reductions(ScalarToken);
    }

    #[test]
    fn find_max_scalar() {
        let data: Vec<f32> = (1..=100).map(|i| i as f32).collect();
        let result = find_max(ScalarToken, &data);
        assert_eq!(result, 100.0);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn find_max_x86() {
        if let Some(token) = X64V3Token::summon() {
            let data: Vec<f32> = (1..=100).map(|i| i as f32).collect();
            let result = find_max(token, &data);
            assert_eq!(result, 100.0);
        }
    }
}

// ============================================================================
// operations/bitwise.md — Integer & Bitwise
// ============================================================================
mod bitwise {
    use super::*;

    fn integer_arithmetic<T: I32x8Backend>(token: T) {
        let a = i32x8::<T>::splat(token, 10);
        let b = i32x8::<T>::splat(token, 3);

        let sum = a + b;
        assert_eq!(sum.to_array(), [13; 8]);

        let diff = a - b;
        assert_eq!(diff.to_array(), [7; 8]);
    }

    fn bitwise_ops<T: I32x8Backend>(token: T) {
        let a = i32x8::<T>::splat(token, 0xFF);
        let b = i32x8::<T>::splat(token, 0x0F);

        let result = a & b;
        assert_eq!(result[0], 0x0F);

        let result = a | b;
        assert_eq!(result[0], 0xFF);

        let result = a ^ b;
        assert_eq!(result[0], 0xF0);

        let result = a.not();
        assert_eq!(result[0], !0xFF);
    }

    #[test]
    fn integer_arithmetic_scalar() {
        integer_arithmetic(ScalarToken);
    }

    #[test]
    fn bitwise_ops_scalar() {
        bitwise_ops(ScalarToken);
    }
}

// ============================================================================
// conversions/float-int.md — Float / Integer
// ============================================================================
mod float_int {
    use super::*;

    fn float_to_int<T: F32x8Convert>(token: T) {
        let floats = f32x8::<T>::from_array(token, [1.5, 2.7, -3.2, 4.0, 5.9, 6.1, 7.0, 8.5]);
        let ints = floats.to_i32();
        assert_eq!(ints.to_array(), [1, 2, -3, 4, 5, 6, 7, 8]);
    }

    fn int_to_float<T: F32x8Convert>(token: T) {
        let ints = i32x8::<T>::from_array(token, [1, 2, 3, 4, 5, 6, 7, 8]);
        let floats = f32x8::<T>::from_i32(token, ints);
        assert_eq!(
            floats.to_array(),
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        );
    }

    #[test]
    fn float_to_int_scalar() {
        float_to_int(ScalarToken);
    }

    #[test]
    fn int_to_float_scalar() {
        int_to_float(ScalarToken);
    }
}

// ============================================================================
// conversions/bitcast.md — Bitcast
// ============================================================================
mod bitcast {
    use super::*;

    fn float_to_int_bitcast<T: F32x8Convert>(token: T) {
        let floats = f32x8::<T>::splat(token, 1.0);
        let bits = floats.bitcast_to_i32();
        // IEEE 754: 1.0f32 = 0x3f800000
        assert_eq!(bits[0], 0x3f800000_i32);
    }

    #[test]
    fn float_to_int_bitcast_scalar() {
        float_to_int_bitcast(ScalarToken);
    }
}

// ============================================================================
// conversions/slice-casting.md — Slice Casting
// ============================================================================
mod slice_casting {
    use super::*;

    #[test]
    fn cast_slice_scalar() {
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        if let Some(vectors) = f32x8::<ScalarToken>::cast_slice(ScalarToken, &data) {
            assert_eq!(vectors.len(), 8);
            assert_eq!(
                vectors[0].to_array(),
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
            );
        }
    }

    #[test]
    fn bytes_roundtrip_scalar() {
        let v = f32x8::<ScalarToken>::splat(ScalarToken, 1.0);
        let bytes: &[u8; 32] = v.as_bytes();
        let restored = f32x8::<ScalarToken>::from_bytes(ScalarToken, bytes);
        assert_eq!(restored.to_array(), [1.0; 8]);
    }
}

// ============================================================================
// math/transcendentals.md — Transcendentals
// ============================================================================
mod transcendentals {
    use super::*;

    fn exp_and_log<T: F32x8Convert>(token: T) {
        let v = f32x8::<T>::splat(token, 3.0);
        let exp2 = v.exp2_midp();
        assert!((exp2[0] - 8.0).abs() < 0.01);

        let v = f32x8::<T>::splat(token, 8.0);
        let log2 = v.log2_midp();
        assert!((log2[0] - 3.0).abs() < 0.01);
    }

    fn sqrt_test<T: F32x8Backend>(token: T) {
        let v = f32x8::<T>::splat(token, 9.0);
        let result = v.sqrt();
        assert!((result[0] - 3.0).abs() < 0.01);
    }

    fn softmax<T: F32x8Convert>(token: T, logits: &[f32; 8]) -> [f32; 8] {
        let v = f32x8::<T>::load(token, logits);
        let max_val = f32x8::<T>::splat(token, v.reduce_max());
        let shifted = v - max_val;
        let exp = shifted.exp_midp();
        let sum = exp.reduce_add();
        let result = exp / f32x8::<T>::splat(token, sum);
        result.to_array()
    }

    #[test]
    fn exp_and_log_scalar() {
        exp_and_log(ScalarToken);
    }

    #[test]
    fn sqrt_scalar() {
        sqrt_test(ScalarToken);
    }

    #[test]
    fn softmax_scalar() {
        let logits = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];
        let probs = softmax(ScalarToken, &logits);
        for &p in &probs {
            assert!(p > 0.0);
        }
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.02);
    }
}

// ============================================================================
// math/precision.md — Precision Levels
// ============================================================================
mod precision {
    use super::*;

    fn precision_tiers<T: F32x8Convert>(token: T) {
        let v = f32x8::<T>::splat(token, 2.0);

        let fast = v.exp2_lowp();
        let balanced = v.exp2_midp();

        // Both should be approximately 4.0
        assert!((fast[0] - 4.0).abs() < 0.1);
        assert!((balanced[0] - 4.0).abs() < 0.01);
    }

    #[test]
    fn precision_tiers_scalar() {
        precision_tiers(ScalarToken);
    }
}

// ============================================================================
// math/approximations.md — Approximations
// ============================================================================
mod approximations {
    use super::*;

    fn rcp_and_rsqrt<T: F32x8Backend>(token: T) {
        let v = f32x8::<T>::splat(token, 4.0);

        let rcp = v.rcp_approx();
        assert!((rcp[0] - 0.25).abs() < 0.01);

        let rsqrt = v.rsqrt_approx();
        assert!((rsqrt[0] - 0.5).abs() < 0.01);

        let precise_rcp = v.recip();
        assert!((precise_rcp[0] - 0.25).abs() < 0.001);
    }

    fn newton_raphson_rsqrt<T: F32x8Backend>(token: T) {
        let v = f32x8::<T>::splat(token, 4.0);
        let approx = v.rsqrt_approx();
        let half = f32x8::<T>::splat(token, 0.5);
        let three_halves = f32x8::<T>::splat(token, 1.5);
        let refined = approx * (three_halves - half * v * approx * approx);
        assert!((refined[0] - 0.5).abs() < 0.001);
    }

    #[test]
    fn rcp_and_rsqrt_scalar() {
        rcp_and_rsqrt(ScalarToken);
    }

    #[test]
    fn newton_raphson_scalar() {
        newton_raphson_rsqrt(ScalarToken);
    }
}

// ============================================================================
// memory/load-store.md — Load & Store
// ============================================================================
mod load_store {
    use super::*;

    fn load_and_store<T: F32x8Backend>(token: T) {
        let arr = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        // from_array
        let v = f32x8::<T>::from_array(token, arr);
        assert_eq!(v.to_array(), arr);

        // store
        let v = f32x8::<T>::splat(token, 42.0);
        let mut buf = [0.0f32; 8];
        v.store(&mut buf);
        assert_eq!(buf, [42.0; 8]);
    }

    #[test]
    fn load_and_store_scalar() {
        load_and_store(ScalarToken);
    }
}

// ============================================================================
// memory/chunked.md — Chunked Processing
// ============================================================================
mod chunked {
    use super::*;

    fn process_large<T: F32x8Backend>(token: T, data: &mut [f32]) {
        let (chunks, remainder) = data.split_at_mut(data.len() - data.len() % 8);

        for chunk in chunks.chunks_exact_mut(8) {
            let chunk_arr: &mut [f32; 8] = chunk.try_into().unwrap();
            let v = f32x8::<T>::from_array(token, *chunk_arr);
            let result = v * v;
            result.store(chunk_arr);
        }

        for x in remainder {
            *x = *x * *x;
        }
    }

    fn sum_array<T: F32x8Backend>(token: T, data: &[f32]) -> f32 {
        let chunks = data.chunks_exact(8);
        let remainder = chunks.remainder();

        let mut acc = f32x8::<T>::zero(token);
        for chunk in chunks {
            let chunk_arr: &[f32; 8] = chunk.try_into().unwrap();
            let v = f32x8::<T>::from_array(token, *chunk_arr);
            acc = acc + v;
        }

        let mut total = acc.reduce_add();
        for &x in remainder {
            total += x;
        }
        total
    }

    #[test]
    fn process_large_scalar() {
        let mut data = vec![2.0_f32; 16];
        process_large(ScalarToken, &mut data);
        assert_eq!(data, vec![4.0_f32; 16]);
    }

    #[test]
    fn sum_array_scalar() {
        let data: Vec<f32> = (1..=100).map(|i| i as f32).collect();
        let result = sum_array(ScalarToken, &data);
        assert!((result - 5050.0).abs() < 0.01);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn sum_array_x86() {
        if let Some(token) = X64V3Token::summon() {
            let data: Vec<f32> = (1..=100).map(|i| i as f32).collect();
            let result = sum_array(token, &data);
            assert!((result - 5050.0).abs() < 0.01);
        }
    }
}

// ============================================================================
// cross-platform/polyfills.md — Polyfills
// ============================================================================
mod polyfills {
    use super::*;

    fn polyfill_demo<T: F32x8Backend>(token: T) {
        // f32x8 works with any backend — native or polyfilled
        let a = f32x8::<T>::splat(token, 1.0);
        let b = f32x8::<T>::splat(token, 2.0);
        let c = a + b;
        let sum = c.reduce_add();
        assert!((sum - 24.0).abs() < 0.01);
    }

    #[test]
    fn polyfill_scalar() {
        polyfill_demo(ScalarToken);
    }

    // implementation_name() is only available on platform-native types,
    // not on ScalarToken. Test with concrete platform types.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn impl_name_x86() {
        assert_eq!(
            f32x8::<X64V3Token>::implementation_name(),
            "x86::v3::f32x8"
        );
    }
}

// ============================================================================
// dispatch/types-and-dispatch.md — Types and Dispatch
// ============================================================================
mod dispatch {
    use super::*;

    // The generic kernel — works with any backend
    fn sum_kernel<T: F32x8Backend>(token: T, data: &[f32; 8]) -> f32 {
        f32x8::<T>::from_array(token, *data).reduce_add()
    }

    // Dispatch pattern
    fn sum_dispatch(data: &[f32; 8]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        if let Some(token) = X64V3Token::summon() {
            return sum_kernel(token, data);
        }

        sum_kernel(ScalarToken, data)
    }

    #[test]
    fn dispatch_works() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = sum_dispatch(&data);
        assert!((result - 36.0).abs() < 0.01);
    }
}

// ============================================================================
// operations/operators.md — Vector normalization example
// ============================================================================
mod normalization {
    use super::*;

    fn normalize<T: F32x8Backend>(token: T, v: &mut [f32; 8]) {
        let vec = f32x8::<T>::from_array(token, *v);
        let len_sq = (vec * vec).reduce_add();
        let len = len_sq.sqrt();

        if len > 0.0 {
            let inv_len = f32x8::<T>::splat(token, 1.0 / len);
            let normalized = vec * inv_len;
            *v = normalized.to_array();
        }
    }

    #[test]
    fn normalize_scalar() {
        let mut v = [3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        normalize(ScalarToken, &mut v);
        assert!((v[0] - 0.6).abs() < 1e-5);
        assert!((v[1] - 0.8).abs() < 1e-5);
    }
}

// ============================================================================
// math/transcendentals.md — Gaussian example
// ============================================================================
mod gaussian {
    use super::*;

    fn gaussian<T: F32x8Convert>(token: T, x: &[f32; 8], sigma: f32) -> [f32; 8] {
        let v = f32x8::<T>::from_array(token, *x);
        let sigma_v = f32x8::<T>::splat(token, sigma);
        let two = f32x8::<T>::splat(token, 2.0);

        let x_sq = v * v;
        let two_sigma_sq = two * sigma_v * sigma_v;
        let exponent = -(x_sq / two_sigma_sq);
        let result = exponent.exp_midp();
        result.to_array()
    }

    #[test]
    fn gaussian_scalar() {
        let x = [0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0];
        let result = gaussian(ScalarToken, &x, 1.0);
        // At x=0, gaussian(0, sigma=1) = exp(0) = 1.0
        assert!((result[0] - 1.0).abs() < 0.01);
        // At x=1, gaussian(1, sigma=1) = exp(-0.5) ≈ 0.606
        assert!((result[1] - 0.606).abs() < 0.02);
    }
}

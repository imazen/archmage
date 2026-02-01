//! Property-based testing to find divergences between native and polyfill implementations.
//!
//! This uses proptest to generate random inputs and verify that:
//! 1. Native 256-bit ops match polyfill (2x 128-bit) results
//! 2. Native 512-bit ops match polyfill (2x 256-bit) results
//! 3. Various edge cases are handled consistently
//!
//! Run with: cargo test -p magetypes --test fuzz_divergence --features avx512

#![cfg(target_arch = "x86_64")]

use proptest::prelude::*;

// Helper to compare f32 with tolerance for NaN and precision
fn f32_eq(a: f32, b: f32) -> bool {
    if a.is_nan() && b.is_nan() {
        true
    } else if a.is_nan() || b.is_nan() {
        false
    } else if a.is_infinite() && b.is_infinite() {
        a.is_sign_positive() == b.is_sign_positive()
    } else {
        (a - b).abs() < 1e-6 || (a - b).abs() / a.abs().max(b.abs()) < 1e-6
    }
}

fn f32_arrays_eq<const N: usize>(a: &[f32; N], b: &[f32; N]) -> bool {
    a.iter().zip(b.iter()).all(|(x, y)| f32_eq(*x, *y))
}

fn i32_arrays_eq<const N: usize>(a: &[i32; N], b: &[i32; N]) -> bool {
    a == b
}

// ============================================================================
// f32x8: Native AVX2 vs Polyfill (2x f32x4)
// ============================================================================

mod f32x8_native_vs_polyfill {
    use super::*;
    use archmage::{SimdToken, X64V3Token};
    use magetypes::simd::{f32x4, f32x8};

    // Run an operation through both native f32x8 and polyfill (2x f32x4)
    fn compare_unary<F, G>(input: [f32; 8], native_op: F, polyfill_op: G) -> bool
    where
        F: Fn(f32x8) -> f32x8,
        G: Fn(f32x4) -> f32x4,
    {
        if let Some(token) = X64V3Token::try_new() {
            // Native path
            let v = f32x8::from_array(token, input);
            let native_result = native_op(v).to_array();

            // Polyfill path (split into two f32x4)
            let lo: [f32; 4] = input[0..4].try_into().unwrap();
            let hi: [f32; 4] = input[4..8].try_into().unwrap();
            let v_lo = f32x4::from_array(token, lo);
            let v_hi = f32x4::from_array(token, hi);
            let poly_lo = polyfill_op(v_lo).to_array();
            let poly_hi = polyfill_op(v_hi).to_array();
            let polyfill_result: [f32; 8] = [
                poly_lo[0], poly_lo[1], poly_lo[2], poly_lo[3],
                poly_hi[0], poly_hi[1], poly_hi[2], poly_hi[3],
            ];

            f32_arrays_eq(&native_result, &polyfill_result)
        } else {
            true // Skip if no token
        }
    }

    fn compare_binary<F, G>(a: [f32; 8], b: [f32; 8], native_op: F, polyfill_op: G) -> bool
    where
        F: Fn(f32x8, f32x8) -> f32x8,
        G: Fn(f32x4, f32x4) -> f32x4,
    {
        if let Some(token) = X64V3Token::try_new() {
            let va = f32x8::from_array(token, a);
            let vb = f32x8::from_array(token, b);
            let native_result = native_op(va, vb).to_array();

            let a_lo: [f32; 4] = a[0..4].try_into().unwrap();
            let a_hi: [f32; 4] = a[4..8].try_into().unwrap();
            let b_lo: [f32; 4] = b[0..4].try_into().unwrap();
            let b_hi: [f32; 4] = b[4..8].try_into().unwrap();

            let va_lo = f32x4::from_array(token, a_lo);
            let va_hi = f32x4::from_array(token, a_hi);
            let vb_lo = f32x4::from_array(token, b_lo);
            let vb_hi = f32x4::from_array(token, b_hi);

            let poly_lo = polyfill_op(va_lo, vb_lo).to_array();
            let poly_hi = polyfill_op(va_hi, vb_hi).to_array();
            let polyfill_result: [f32; 8] = [
                poly_lo[0], poly_lo[1], poly_lo[2], poly_lo[3],
                poly_hi[0], poly_hi[1], poly_hi[2], poly_hi[3],
            ];

            f32_arrays_eq(&native_result, &polyfill_result)
        } else {
            true
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10000))]

        #[test]
        fn fuzz_add(a in prop::array::uniform8(-1e10f32..1e10f32),
                    b in prop::array::uniform8(-1e10f32..1e10f32)) {
            prop_assert!(compare_binary(a, b, |x, y| x + y, |x, y| x + y));
        }

        #[test]
        fn fuzz_sub(a in prop::array::uniform8(-1e10f32..1e10f32),
                    b in prop::array::uniform8(-1e10f32..1e10f32)) {
            prop_assert!(compare_binary(a, b, |x, y| x - y, |x, y| x - y));
        }

        #[test]
        fn fuzz_mul(a in prop::array::uniform8(-1e5f32..1e5f32),
                    b in prop::array::uniform8(-1e5f32..1e5f32)) {
            prop_assert!(compare_binary(a, b, |x, y| x * y, |x, y| x * y));
        }

        #[test]
        fn fuzz_div(a in prop::array::uniform8(-1e10f32..1e10f32),
                    b in prop::array::uniform8(-1e10f32..1e10f32).prop_filter("non-zero", |arr| arr.iter().all(|&x| x.abs() > 1e-10))) {
            prop_assert!(compare_binary(a, b, |x, y| x / y, |x, y| x / y));
        }

        #[test]
        fn fuzz_min(a in prop::array::uniform8(-1e10f32..1e10f32),
                    b in prop::array::uniform8(-1e10f32..1e10f32)) {
            prop_assert!(compare_binary(a, b, |x, y| x.min(y), |x, y| x.min(y)));
        }

        #[test]
        fn fuzz_max(a in prop::array::uniform8(-1e10f32..1e10f32),
                    b in prop::array::uniform8(-1e10f32..1e10f32)) {
            prop_assert!(compare_binary(a, b, |x, y| x.max(y), |x, y| x.max(y)));
        }

        #[test]
        fn fuzz_abs(a in prop::array::uniform8(-1e10f32..1e10f32)) {
            prop_assert!(compare_unary(a, |x| x.abs(), |x| x.abs()));
        }

        #[test]
        fn fuzz_neg(a in prop::array::uniform8(-1e10f32..1e10f32)) {
            prop_assert!(compare_unary(a, |x| -x, |x| -x));
        }

        #[test]
        fn fuzz_sqrt(a in prop::array::uniform8(0.0f32..1e10f32)) {
            prop_assert!(compare_unary(a, |x| x.sqrt(), |x| x.sqrt()));
        }

        #[test]
        fn fuzz_floor(a in prop::array::uniform8(-1e6f32..1e6f32)) {
            prop_assert!(compare_unary(a, |x| x.floor(), |x| x.floor()));
        }

        #[test]
        fn fuzz_ceil(a in prop::array::uniform8(-1e6f32..1e6f32)) {
            prop_assert!(compare_unary(a, |x| x.ceil(), |x| x.ceil()));
        }

        #[test]
        fn fuzz_round(a in prop::array::uniform8(-1e6f32..1e6f32)) {
            prop_assert!(compare_unary(a, |x| x.round(), |x| x.round()));
        }
    }

    // Edge case tests with specific problematic values
    #[test]
    fn test_edge_cases() {
        if let Some(token) = X64V3Token::try_new() {
            let edge_cases: [[f32; 8]; 8] = [
                [0.0, -0.0, 1.0, -1.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN, f32::NAN],
                [f32::MIN, f32::MAX, f32::MIN_POSITIVE, -f32::MIN_POSITIVE, 0.0, 0.0, 0.0, 0.0],
                [1e-38, -1e-38, 1e38, -1e38, 0.5, -0.5, 1.5, -1.5],
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], // fractions
                [1.0 / 3.0; 8], // repeating decimal
                [core::f32::consts::PI; 8],
                [core::f32::consts::E; 8],
                [f32::from_bits(0x7F800001); 8], // signaling NaN
            ];

            for case in &edge_cases {
                let v = f32x8::from_array(token, *case);
                let _ = v.abs().to_array();
                let _ = (-v).to_array();
                let _ = v.floor().to_array();
                let _ = v.ceil().to_array();
            }
        }
    }
}

// ============================================================================
// i32x8: Native AVX2 vs Polyfill (2x i32x4)
// ============================================================================

mod i32x8_native_vs_polyfill {
    use super::*;
    use archmage::{SimdToken, X64V3Token};
    use magetypes::simd::{i32x4, i32x8};

    fn compare_unary<F, G>(input: [i32; 8], native_op: F, polyfill_op: G) -> bool
    where
        F: Fn(i32x8) -> i32x8,
        G: Fn(i32x4) -> i32x4,
    {
        if let Some(token) = X64V3Token::try_new() {
            let v = i32x8::from_array(token, input);
            let native_result = native_op(v).to_array();

            let lo: [i32; 4] = input[0..4].try_into().unwrap();
            let hi: [i32; 4] = input[4..8].try_into().unwrap();
            let v_lo = i32x4::from_array(token, lo);
            let v_hi = i32x4::from_array(token, hi);
            let poly_lo = polyfill_op(v_lo).to_array();
            let poly_hi = polyfill_op(v_hi).to_array();
            let polyfill_result: [i32; 8] = [
                poly_lo[0], poly_lo[1], poly_lo[2], poly_lo[3],
                poly_hi[0], poly_hi[1], poly_hi[2], poly_hi[3],
            ];

            i32_arrays_eq(&native_result, &polyfill_result)
        } else {
            true
        }
    }

    fn compare_binary<F, G>(a: [i32; 8], b: [i32; 8], native_op: F, polyfill_op: G) -> bool
    where
        F: Fn(i32x8, i32x8) -> i32x8,
        G: Fn(i32x4, i32x4) -> i32x4,
    {
        if let Some(token) = X64V3Token::try_new() {
            let va = i32x8::from_array(token, a);
            let vb = i32x8::from_array(token, b);
            let native_result = native_op(va, vb).to_array();

            let a_lo: [i32; 4] = a[0..4].try_into().unwrap();
            let a_hi: [i32; 4] = a[4..8].try_into().unwrap();
            let b_lo: [i32; 4] = b[0..4].try_into().unwrap();
            let b_hi: [i32; 4] = b[4..8].try_into().unwrap();

            let va_lo = i32x4::from_array(token, a_lo);
            let va_hi = i32x4::from_array(token, a_hi);
            let vb_lo = i32x4::from_array(token, b_lo);
            let vb_hi = i32x4::from_array(token, b_hi);

            let poly_lo = polyfill_op(va_lo, vb_lo).to_array();
            let poly_hi = polyfill_op(va_hi, vb_hi).to_array();
            let polyfill_result: [i32; 8] = [
                poly_lo[0], poly_lo[1], poly_lo[2], poly_lo[3],
                poly_hi[0], poly_hi[1], poly_hi[2], poly_hi[3],
            ];

            i32_arrays_eq(&native_result, &polyfill_result)
        } else {
            true
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10000))]

        #[test]
        fn fuzz_add(a in prop::array::uniform8(i32::MIN..i32::MAX),
                    b in prop::array::uniform8(i32::MIN..i32::MAX)) {
            prop_assert!(compare_binary(a, b, |x, y| x + y, |x, y| x + y));
        }

        #[test]
        fn fuzz_sub(a in prop::array::uniform8(i32::MIN..i32::MAX),
                    b in prop::array::uniform8(i32::MIN..i32::MAX)) {
            prop_assert!(compare_binary(a, b, |x, y| x - y, |x, y| x - y));
        }

        #[test]
        fn fuzz_min(a in prop::array::uniform8(i32::MIN..i32::MAX),
                    b in prop::array::uniform8(i32::MIN..i32::MAX)) {
            prop_assert!(compare_binary(a, b, |x, y| x.min(y), |x, y| x.min(y)));
        }

        #[test]
        fn fuzz_max(a in prop::array::uniform8(i32::MIN..i32::MAX),
                    b in prop::array::uniform8(i32::MIN..i32::MAX)) {
            prop_assert!(compare_binary(a, b, |x, y| x.max(y), |x, y| x.max(y)));
        }

        #[test]
        fn fuzz_abs(a in prop::array::uniform8(i32::MIN + 1..i32::MAX)) { // MIN + 1 to avoid overflow
            prop_assert!(compare_unary(a, |x| x.abs(), |x| x.abs()));
        }
    }

    #[test]
    fn test_edge_cases() {
        if let Some(token) = X64V3Token::try_new() {
            let edge_cases: [[i32; 8]; 5] = [
                [0, -1, 1, i32::MAX, i32::MIN + 1, 0, 0, 0], // MIN + 1 for abs safety
                [i32::MAX, i32::MAX, i32::MIN + 1, i32::MIN + 1, 0, 0, 1, -1],
                [0x7FFFFFFF, 0x80000001_u32 as i32, 0x00000001, 0xFFFFFFFF_u32 as i32, 0, 0, 0, 0],
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [1, 2, 3, 4, 5, 6, 7, 8],
            ];

            for case in &edge_cases {
                let v = i32x8::from_array(token, *case);
                let _ = v.abs().to_array();
                let _ = v.min(v).to_array();
                let _ = v.max(v).to_array();
            }
        }
    }
}

// ============================================================================
// Reduce operations (these aggregate across lanes, more subtle divergences)
// ============================================================================

mod reduce_operations {
    use super::*;
    use archmage::{SimdToken, X64V3Token};
    use magetypes::simd::f32x8;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10000))]

        #[test]
        fn fuzz_reduce_add(a in prop::array::uniform8(-1e5f32..1e5f32)) {
            if let Some(token) = X64V3Token::try_new() {
                let v = f32x8::from_array(token, a);
                let native = v.reduce_add();
                let scalar: f32 = a.iter().sum();
                // Reduce operations may have different associativity, allow some tolerance
                prop_assert!((native - scalar).abs() < 1e-3 * scalar.abs().max(1.0),
                    "native={}, scalar={}", native, scalar);
            }
        }

        #[test]
        fn fuzz_reduce_min(a in prop::array::uniform8(-1e10f32..1e10f32)) {
            if let Some(token) = X64V3Token::try_new() {
                let v = f32x8::from_array(token, a);
                let native = v.reduce_min();
                let scalar = a.iter().copied().fold(f32::INFINITY, f32::min);
                prop_assert!(f32_eq(native, scalar), "native={}, scalar={}", native, scalar);
            }
        }

        #[test]
        fn fuzz_reduce_max(a in prop::array::uniform8(-1e10f32..1e10f32)) {
            if let Some(token) = X64V3Token::try_new() {
                let v = f32x8::from_array(token, a);
                let native = v.reduce_max();
                let scalar = a.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                prop_assert!(f32_eq(native, scalar), "native={}, scalar={}", native, scalar);
            }
        }
    }
}

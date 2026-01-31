//! Verification tests comparing polyfill results to native SIMD.
//!
//! These tests ensure polyfill implementations produce identical
//! results to native SIMD operations.

#![cfg(target_arch = "x86_64")]

use archmage::{SimdToken, X64V3Token};
use magetypes::simd::f32x8;
use magetypes::simd::polyfill::sse as poly;

// ============================================================================
// Helper: Compare polyfill f32x8 to native f32x8
// ============================================================================

fn arrays_equal(a: &[f32; 8], b: &[f32; 8]) -> bool {
    a.iter()
        .zip(b.iter())
        .all(|(x, y)| (x - y).abs() < f32::EPSILON || (x.is_nan() && y.is_nan()))
}

fn arrays_close(a: &[f32; 8], b: &[f32; 8], tolerance: f32) -> bool {
    a.iter().zip(b.iter()).all(|(x, y)| {
        if x.is_nan() && y.is_nan() {
            true
        } else if x.is_infinite() && y.is_infinite() {
            x.signum() == y.signum()
        } else if *x == 0.0 && *y == 0.0 {
            true
        } else {
            let rel_err = ((x - y) / x.abs().max(y.abs())).abs();
            rel_err < tolerance
        }
    })
}

// ============================================================================
// Arithmetic Operations
// ============================================================================

#[test]
fn verify_add() {
    let sse = X64V3Token::try_new().expect("SSE4.1 required");
    let avx = match X64V3Token::try_new() {
        Some(t) => t,
        None => {
            eprintln!("AVX2 not available, skipping native comparison");
            return;
        }
    };

    let data_a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let data_b = [8.0f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

    // Native AVX2
    let native_a = f32x8::load(avx, &data_a);
    let native_b = f32x8::load(avx, &data_b);
    let native_result = (native_a + native_b).to_array();

    // Polyfill
    let poly_a = poly::f32x8::load(sse, &data_a);
    let poly_b = poly::f32x8::load(sse, &data_b);
    let poly_result = (poly_a + poly_b).to_array();

    assert!(
        arrays_equal(&native_result, &poly_result),
        "Add mismatch:\n  native: {:?}\n  poly:   {:?}",
        native_result,
        poly_result
    );
}

#[test]
fn verify_sub() {
    let sse = X64V3Token::try_new().expect("SSE4.1 required");
    let avx = match X64V3Token::try_new() {
        Some(t) => t,
        None => return,
    };

    let data_a = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
    let data_b = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let native_result = (f32x8::load(avx, &data_a) - f32x8::load(avx, &data_b)).to_array();
    let poly_result =
        (poly::f32x8::load(sse, &data_a) - poly::f32x8::load(sse, &data_b)).to_array();

    assert!(
        arrays_equal(&native_result, &poly_result),
        "Sub mismatch:\n  native: {:?}\n  poly:   {:?}",
        native_result,
        poly_result
    );
}

#[test]
fn verify_mul() {
    let sse = X64V3Token::try_new().expect("SSE4.1 required");
    let avx = match X64V3Token::try_new() {
        Some(t) => t,
        None => return,
    };

    let data_a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let data_b = [2.0f32, 2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.5];

    let native_result = (f32x8::load(avx, &data_a) * f32x8::load(avx, &data_b)).to_array();
    let poly_result =
        (poly::f32x8::load(sse, &data_a) * poly::f32x8::load(sse, &data_b)).to_array();

    assert!(
        arrays_equal(&native_result, &poly_result),
        "Mul mismatch:\n  native: {:?}\n  poly:   {:?}",
        native_result,
        poly_result
    );
}

#[test]
fn verify_div() {
    let sse = X64V3Token::try_new().expect("SSE4.1 required");
    let avx = match X64V3Token::try_new() {
        Some(t) => t,
        None => return,
    };

    let data_a = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
    let data_b = [2.0f32, 4.0, 5.0, 8.0, 10.0, 12.0, 14.0, 16.0];

    let native_result = (f32x8::load(avx, &data_a) / f32x8::load(avx, &data_b)).to_array();
    let poly_result =
        (poly::f32x8::load(sse, &data_a) / poly::f32x8::load(sse, &data_b)).to_array();

    assert!(
        arrays_equal(&native_result, &poly_result),
        "Div mismatch:\n  native: {:?}\n  poly:   {:?}",
        native_result,
        poly_result
    );
}

#[test]
fn verify_neg() {
    let sse = X64V3Token::try_new().expect("SSE4.1 required");
    let avx = match X64V3Token::try_new() {
        Some(t) => t,
        None => return,
    };

    let data = [1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0];

    let native_result = (-f32x8::load(avx, &data)).to_array();
    let poly_result = (-poly::f32x8::load(sse, &data)).to_array();

    assert!(
        arrays_equal(&native_result, &poly_result),
        "Neg mismatch:\n  native: {:?}\n  poly:   {:?}",
        native_result,
        poly_result
    );
}

// ============================================================================
// Math Operations
// ============================================================================

#[test]
fn verify_min() {
    let sse = X64V3Token::try_new().expect("SSE4.1 required");
    let avx = match X64V3Token::try_new() {
        Some(t) => t,
        None => return,
    };

    let data_a = [1.0f32, 5.0, 3.0, 7.0, 2.0, 8.0, 4.0, 6.0];
    let data_b = [4.0f32, 2.0, 6.0, 1.0, 5.0, 3.0, 7.0, 2.0];

    let native_result = f32x8::load(avx, &data_a)
        .min(f32x8::load(avx, &data_b))
        .to_array();
    let poly_result = poly::f32x8::load(sse, &data_a)
        .min(poly::f32x8::load(sse, &data_b))
        .to_array();

    assert!(
        arrays_equal(&native_result, &poly_result),
        "Min mismatch:\n  native: {:?}\n  poly:   {:?}",
        native_result,
        poly_result
    );
}

#[test]
fn verify_max() {
    let sse = X64V3Token::try_new().expect("SSE4.1 required");
    let avx = match X64V3Token::try_new() {
        Some(t) => t,
        None => return,
    };

    let data_a = [1.0f32, 5.0, 3.0, 7.0, 2.0, 8.0, 4.0, 6.0];
    let data_b = [4.0f32, 2.0, 6.0, 1.0, 5.0, 3.0, 7.0, 2.0];

    let native_result = f32x8::load(avx, &data_a)
        .max(f32x8::load(avx, &data_b))
        .to_array();
    let poly_result = poly::f32x8::load(sse, &data_a)
        .max(poly::f32x8::load(sse, &data_b))
        .to_array();

    assert!(
        arrays_equal(&native_result, &poly_result),
        "Max mismatch:\n  native: {:?}\n  poly:   {:?}",
        native_result,
        poly_result
    );
}

#[test]
fn verify_abs() {
    let sse = X64V3Token::try_new().expect("SSE4.1 required");
    let avx = match X64V3Token::try_new() {
        Some(t) => t,
        None => return,
    };

    let data = [-1.0f32, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0];

    let native_result = f32x8::load(avx, &data).abs().to_array();
    let poly_result = poly::f32x8::load(sse, &data).abs().to_array();

    assert!(
        arrays_equal(&native_result, &poly_result),
        "Abs mismatch:\n  native: {:?}\n  poly:   {:?}",
        native_result,
        poly_result
    );
}

#[test]
fn verify_sqrt() {
    let sse = X64V3Token::try_new().expect("SSE4.1 required");
    let avx = match X64V3Token::try_new() {
        Some(t) => t,
        None => return,
    };

    let data = [1.0f32, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0];

    let native_result = f32x8::load(avx, &data).sqrt().to_array();
    let poly_result = poly::f32x8::load(sse, &data).sqrt().to_array();

    assert!(
        arrays_equal(&native_result, &poly_result),
        "Sqrt mismatch:\n  native: {:?}\n  poly:   {:?}",
        native_result,
        poly_result
    );
}

#[test]
fn verify_floor() {
    let sse = X64V3Token::try_new().expect("SSE4.1 required");
    let avx = match X64V3Token::try_new() {
        Some(t) => t,
        None => return,
    };

    let data = [1.1f32, 2.9, -1.1, -2.9, 0.5, -0.5, 3.0, -3.0];

    let native_result = f32x8::load(avx, &data).floor().to_array();
    let poly_result = poly::f32x8::load(sse, &data).floor().to_array();

    assert!(
        arrays_equal(&native_result, &poly_result),
        "Floor mismatch:\n  native: {:?}\n  poly:   {:?}",
        native_result,
        poly_result
    );
}

#[test]
fn verify_ceil() {
    let sse = X64V3Token::try_new().expect("SSE4.1 required");
    let avx = match X64V3Token::try_new() {
        Some(t) => t,
        None => return,
    };

    let data = [1.1f32, 2.9, -1.1, -2.9, 0.5, -0.5, 3.0, -3.0];

    let native_result = f32x8::load(avx, &data).ceil().to_array();
    let poly_result = poly::f32x8::load(sse, &data).ceil().to_array();

    assert!(
        arrays_equal(&native_result, &poly_result),
        "Ceil mismatch:\n  native: {:?}\n  poly:   {:?}",
        native_result,
        poly_result
    );
}

#[test]
fn verify_round() {
    let sse = X64V3Token::try_new().expect("SSE4.1 required");
    let avx = match X64V3Token::try_new() {
        Some(t) => t,
        None => return,
    };

    let data = [1.4f32, 1.5, 1.6, 2.5, -1.4, -1.5, -1.6, -2.5];

    let native_result = f32x8::load(avx, &data).round().to_array();
    let poly_result = poly::f32x8::load(sse, &data).round().to_array();

    assert!(
        arrays_equal(&native_result, &poly_result),
        "Round mismatch:\n  native: {:?}\n  poly:   {:?}",
        native_result,
        poly_result
    );
}

// ============================================================================
// Horizontal Operations
// ============================================================================

#[test]
fn verify_reduce_add() {
    let sse = X64V3Token::try_new().expect("SSE4.1 required");
    let avx = match X64V3Token::try_new() {
        Some(t) => t,
        None => return,
    };

    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let native_result = f32x8::load(avx, &data).reduce_add();
    let poly_result = poly::f32x8::load(sse, &data).reduce_add();

    assert!(
        (native_result - poly_result).abs() < 1e-5,
        "Reduce_add mismatch: native={}, poly={}",
        native_result,
        poly_result
    );
}

#[test]
fn verify_reduce_max() {
    let sse = X64V3Token::try_new().expect("SSE4.1 required");
    let avx = match X64V3Token::try_new() {
        Some(t) => t,
        None => return,
    };

    let data = [1.0f32, 8.0, 3.0, 6.0, 2.0, 7.0, 4.0, 5.0];

    let native_result = f32x8::load(avx, &data).reduce_max();
    let poly_result = poly::f32x8::load(sse, &data).reduce_max();

    assert!(
        (native_result - poly_result).abs() < f32::EPSILON,
        "Reduce_max mismatch: native={}, poly={}",
        native_result,
        poly_result
    );
}

#[test]
fn verify_reduce_min() {
    let sse = X64V3Token::try_new().expect("SSE4.1 required");
    let avx = match X64V3Token::try_new() {
        Some(t) => t,
        None => return,
    };

    let data = [5.0f32, 2.0, 7.0, 1.0, 8.0, 3.0, 6.0, 4.0];

    let native_result = f32x8::load(avx, &data).reduce_min();
    let poly_result = poly::f32x8::load(sse, &data).reduce_min();

    assert!(
        (native_result - poly_result).abs() < f32::EPSILON,
        "Reduce_min mismatch: native={}, poly={}",
        native_result,
        poly_result
    );
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn verify_edge_cases_infinity() {
    let sse = X64V3Token::try_new().expect("SSE4.1 required");
    let avx = match X64V3Token::try_new() {
        Some(t) => t,
        None => return,
    };

    let data = [
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0,
        -0.0,
        1.0,
        -1.0,
        f32::MAX,
        f32::MIN,
    ];

    // Add with itself
    let native_result = (f32x8::load(avx, &data) + f32x8::load(avx, &data)).to_array();
    let poly_result = (poly::f32x8::load(sse, &data) + poly::f32x8::load(sse, &data)).to_array();

    for i in 0..8 {
        let n = native_result[i];
        let p = poly_result[i];
        if n.is_nan() {
            assert!(p.is_nan(), "NaN mismatch at index {}", i);
        } else if n.is_infinite() {
            assert!(
                p.is_infinite() && n.signum() == p.signum(),
                "Infinity mismatch at index {}: native={}, poly={}",
                i,
                n,
                p
            );
        } else {
            assert!(
                (n - p).abs() < f32::EPSILON,
                "Value mismatch at index {}: native={}, poly={}",
                i,
                n,
                p
            );
        }
    }
}

#[test]
fn verify_edge_cases_denormals() {
    let sse = X64V3Token::try_new().expect("SSE4.1 required");
    let avx = match X64V3Token::try_new() {
        Some(t) => t,
        None => return,
    };

    // Smallest positive denormal
    let denormal = f32::from_bits(1);
    let data = [
        denormal,
        denormal * 2.0,
        denormal * 4.0,
        f32::MIN_POSITIVE,
        -denormal,
        -denormal * 2.0,
        0.0,
        1.0,
    ];

    let native_result = (f32x8::load(avx, &data) * f32x8::splat(avx, 2.0)).to_array();
    let poly_result = (poly::f32x8::load(sse, &data) * poly::f32x8::splat(sse, 2.0)).to_array();

    // Denormal handling may differ, so we use relative tolerance
    assert!(
        arrays_close(&native_result, &poly_result, 1e-6),
        "Denormal mismatch:\n  native: {:?}\n  poly:   {:?}",
        native_result,
        poly_result
    );
}

// ============================================================================
// mul_add (FMA)
// ============================================================================

#[test]
fn verify_mul_add() {
    let sse = X64V3Token::try_new().expect("SSE4.1 required");
    let avx = match X64V3Token::try_new() {
        Some(t) => t,
        None => return,
    };

    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = [2.0f32, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
    let c = [1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

    // Native AVX2 uses FMA instruction
    let va = f32x8::load(avx, &a);
    let vb = f32x8::load(avx, &b);
    let vc = f32x8::load(avx, &c);
    let native_result = va.mul_add(vb, vc).to_array();

    // Polyfill uses separate mul+add (not fused)
    let pa = poly::f32x8::load(sse, &a);
    let pb = poly::f32x8::load(sse, &b);
    let pc = poly::f32x8::load(sse, &c);
    let poly_result = pa.mul_add(pb, pc).to_array();

    // Results should be very close (within floating point tolerance)
    // FMA may have slightly different rounding than mul+add
    assert!(
        arrays_close(&native_result, &poly_result, 1e-6),
        "mul_add mismatch:\n  native: {:?}\n  poly:   {:?}",
        native_result,
        poly_result
    );
}

// ============================================================================
// Load/Store Roundtrip
// ============================================================================

#[test]
fn verify_load_store_roundtrip() {
    let sse = X64V3Token::try_new().expect("SSE4.1 required");

    let original = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let v = poly::f32x8::load(sse, &original);
    let result = v.to_array();

    assert_eq!(original, result, "Load/store roundtrip failed");
}

#[test]
fn verify_splat() {
    let sse = X64V3Token::try_new().expect("SSE4.1 required");
    let avx = match X64V3Token::try_new() {
        Some(t) => t,
        None => return,
    };

    let value = 42.0f32;

    let native_result = f32x8::splat(avx, value).to_array();
    let poly_result = poly::f32x8::splat(sse, value).to_array();

    assert_eq!(
        native_result, poly_result,
        "Splat mismatch:\n  native: {:?}\n  poly:   {:?}",
        native_result, poly_result
    );
}

#[test]
fn verify_zero() {
    let sse = X64V3Token::try_new().expect("SSE4.1 required");
    let avx = match X64V3Token::try_new() {
        Some(t) => t,
        None => return,
    };

    let native_result = f32x8::zero(avx).to_array();
    let poly_result = poly::f32x8::zero(sse).to_array();

    assert_eq!(
        native_result, poly_result,
        "Zero mismatch:\n  native: {:?}\n  poly:   {:?}",
        native_result, poly_result
    );
}

// ============================================================================
// Clamp
// ============================================================================

#[test]
fn verify_clamp() {
    let sse = X64V3Token::try_new().expect("SSE4.1 required");
    let avx = match X64V3Token::try_new() {
        Some(t) => t,
        None => return,
    };

    let data = [-1.0f32, 0.5, 1.5, 0.0, 0.25, 0.75, 2.0, -0.5];
    let lo = 0.0f32;
    let hi = 1.0f32;

    let native_result = f32x8::load(avx, &data)
        .clamp(f32x8::splat(avx, lo), f32x8::splat(avx, hi))
        .to_array();
    let poly_result = poly::f32x8::load(sse, &data)
        .clamp(poly::f32x8::splat(sse, lo), poly::f32x8::splat(sse, hi))
        .to_array();

    assert!(
        arrays_equal(&native_result, &poly_result),
        "Clamp mismatch:\n  native: {:?}\n  poly:   {:?}",
        native_result,
        poly_result
    );
}

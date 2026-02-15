//! Tests for generic f32x8<T> — strategy pattern SIMD types.
//!
//! Verifies that f32x8<X64V3Token> produces correct results and has
//! the same size/layout as the old platform-specific f32x8.

#![cfg(target_arch = "x86_64")]

use archmage::{ScalarToken, SimdToken, X64V3Token};
use magetypes::simd::backends::F32x8Backend;
use magetypes::simd::generic::f32x8;

// ============================================================================
// Size and layout verification
// ============================================================================

#[test]
fn size_of_generic_f32x8_equals_m256() {
    // f32x8<X64V3Token> must be exactly 32 bytes (same as __m256)
    assert_eq!(
        core::mem::size_of::<f32x8<X64V3Token>>(),
        32,
        "f32x8<X64V3Token> should be 32 bytes (same as __m256)"
    );
}

#[test]
fn size_of_generic_matches_old_type() {
    // Generic and old types must have identical size
    assert_eq!(
        core::mem::size_of::<f32x8<X64V3Token>>(),
        core::mem::size_of::<magetypes::simd::f32x8>(),
        "generic f32x8<X64V3Token> should have same size as old f32x8"
    );
}

#[test]
fn align_of_generic_f32x8() {
    // Must maintain proper SIMD alignment
    assert_eq!(
        core::mem::align_of::<f32x8<X64V3Token>>(),
        core::mem::align_of::<magetypes::simd::f32x8>(),
        "generic f32x8<X64V3Token> should have same alignment as old f32x8"
    );
}

// ============================================================================
// Basic operations
// ============================================================================

#[test]
fn basic_arithmetic() {
    if let Some(token) = X64V3Token::summon() {
        let a = f32x8::<X64V3Token>::splat(token, 3.0);
        let b = f32x8::<X64V3Token>::splat(token, 2.0);

        let sum = a + b;
        let diff = a - b;
        let prod = a * b;
        let quot = a / b;

        assert_eq!(sum.to_array(), [5.0; 8]);
        assert_eq!(diff.to_array(), [1.0; 8]);
        assert_eq!(prod.to_array(), [6.0; 8]);
        assert_eq!(quot.to_array(), [1.5; 8]);
    }
}

#[test]
fn negation() {
    if let Some(token) = X64V3Token::summon() {
        let a = f32x8::<X64V3Token>::splat(token, 3.0);
        let neg = -a;
        assert_eq!(neg.to_array(), [-3.0; 8]);
    }
}

#[test]
fn load_store_roundtrip() {
    if let Some(token) = X64V3Token::summon() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let v = f32x8::<X64V3Token>::load(token, &data);
        let mut out = [0.0f32; 8];
        v.store(&mut out);
        assert_eq!(data, out);
    }
}

#[test]
fn from_array_to_array() {
    if let Some(token) = X64V3Token::summon() {
        let arr = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let v = f32x8::<X64V3Token>::from_array(token, arr);
        assert_eq!(v.to_array(), arr);
    }
}

#[test]
fn from_slice() {
    if let Some(token) = X64V3Token::summon() {
        let slice = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let v = f32x8::<X64V3Token>::from_slice(token, slice);
        assert_eq!(v.to_array(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }
}

#[test]
fn zero() {
    if let Some(token) = X64V3Token::summon() {
        let v = f32x8::<X64V3Token>::zero(token);
        assert_eq!(v.to_array(), [0.0; 8]);
    }
}

// ============================================================================
// Math operations
// ============================================================================

#[test]
fn min_max() {
    if let Some(token) = X64V3Token::summon() {
        let a = f32x8::<X64V3Token>::from_array(token, [1.0, 4.0, 2.0, 8.0, 3.0, 6.0, 5.0, 7.0]);
        let b = f32x8::<X64V3Token>::from_array(token, [3.0, 2.0, 5.0, 1.0, 7.0, 4.0, 6.0, 8.0]);

        let mn = a.min(b);
        let mx = a.max(b);

        assert_eq!(mn.to_array(), [1.0, 2.0, 2.0, 1.0, 3.0, 4.0, 5.0, 7.0]);
        assert_eq!(mx.to_array(), [3.0, 4.0, 5.0, 8.0, 7.0, 6.0, 6.0, 8.0]);
    }
}

#[test]
fn sqrt_abs() {
    if let Some(token) = X64V3Token::summon() {
        let a = f32x8::<X64V3Token>::splat(token, 4.0);
        assert_eq!(a.sqrt().to_array(), [2.0; 8]);

        let b = f32x8::<X64V3Token>::splat(token, -3.0);
        assert_eq!(b.abs().to_array(), [3.0; 8]);
    }
}

#[test]
fn floor_ceil_round() {
    if let Some(token) = X64V3Token::summon() {
        let a = f32x8::<X64V3Token>::splat(token, 2.7);
        assert_eq!(a.floor().to_array(), [2.0; 8]);
        assert_eq!(a.ceil().to_array(), [3.0; 8]);
        assert_eq!(a.round().to_array(), [3.0; 8]);

        let b = f32x8::<X64V3Token>::splat(token, 2.3);
        assert_eq!(b.round().to_array(), [2.0; 8]);
    }
}

#[test]
fn mul_add_mul_sub() {
    if let Some(token) = X64V3Token::summon() {
        let a = f32x8::<X64V3Token>::splat(token, 2.0);
        let b = f32x8::<X64V3Token>::splat(token, 3.0);
        let c = f32x8::<X64V3Token>::splat(token, 1.0);

        // a * b + c = 2*3+1 = 7
        assert_eq!(a.mul_add(b, c).to_array(), [7.0; 8]);
        // a * b - c = 2*3-1 = 5
        assert_eq!(a.mul_sub(b, c).to_array(), [5.0; 8]);
    }
}

#[test]
fn clamp() {
    if let Some(token) = X64V3Token::summon() {
        let v = f32x8::<X64V3Token>::from_array(token, [-1.0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 10.0]);
        let lo = f32x8::<X64V3Token>::splat(token, 0.0);
        let hi = f32x8::<X64V3Token>::splat(token, 5.0);
        let clamped = v.clamp(lo, hi);
        assert_eq!(clamped.to_array(), [0.0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.0, 5.0]);
    }
}

// ============================================================================
// Comparisons and blend
// ============================================================================

#[test]
fn comparisons_and_blend() {
    if let Some(token) = X64V3Token::summon() {
        let a = f32x8::<X64V3Token>::from_array(token, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let b = f32x8::<X64V3Token>::splat(token, 4.0);

        // a < b: lanes 0,1,2 are true (1,2,3 < 4)
        let mask = a.simd_lt(b);
        let selected = f32x8::<X64V3Token>::blend(
            mask,
            f32x8::<X64V3Token>::splat(token, 10.0),
            f32x8::<X64V3Token>::splat(token, 20.0),
        );
        let result = selected.to_array();
        assert_eq!(result[0], 10.0); // 1 < 4 → 10
        assert_eq!(result[1], 10.0); // 2 < 4 → 10
        assert_eq!(result[2], 10.0); // 3 < 4 → 10
        assert_eq!(result[3], 20.0); // 4 < 4 → false → 20
        assert_eq!(result[4], 20.0); // 5 < 4 → false → 20
    }
}

// ============================================================================
// Reductions
// ============================================================================

#[test]
fn reduce_add() {
    if let Some(token) = X64V3Token::summon() {
        let v = f32x8::<X64V3Token>::from_array(token, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(v.reduce_add(), 36.0);
    }
}

#[test]
fn reduce_min_max() {
    if let Some(token) = X64V3Token::summon() {
        let v = f32x8::<X64V3Token>::from_array(token, [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]);
        assert_eq!(v.reduce_min(), 1.0);
        assert_eq!(v.reduce_max(), 9.0);
    }
}

// ============================================================================
// Approximations
// ============================================================================

#[test]
fn recip_approx() {
    if let Some(token) = X64V3Token::summon() {
        let v = f32x8::<X64V3Token>::splat(token, 4.0);
        let r = v.recip();
        for &val in &r.to_array() {
            assert!((val - 0.25).abs() < 1e-6, "recip(4.0) ≈ 0.25, got {val}");
        }
    }
}

// ============================================================================
// Scalar broadcast operators
// ============================================================================

#[test]
fn scalar_ops() {
    if let Some(token) = X64V3Token::summon() {
        let v = f32x8::<X64V3Token>::splat(token, 3.0);
        assert_eq!((v + 2.0).to_array(), [5.0; 8]);
        assert_eq!((v - 1.0).to_array(), [2.0; 8]);
        assert_eq!((v * 2.0).to_array(), [6.0; 8]);
        assert_eq!((v / 3.0).to_array(), [1.0; 8]);
    }
}

// ============================================================================
// Index
// ============================================================================

#[test]
fn indexing() {
    if let Some(token) = X64V3Token::summon() {
        let v = f32x8::<X64V3Token>::from_array(token, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[7], 8.0);

        let mut v = v;
        v[3] = 99.0;
        assert_eq!(v[3], 99.0);
    }
}

// ============================================================================
// Bitwise
// ============================================================================

#[test]
fn bitwise_not() {
    if let Some(token) = X64V3Token::summon() {
        let v = f32x8::<X64V3Token>::zero(token);
        let n = v.not();
        // NOT of all-zeros should be all-1s (NaN in f32)
        for &val in &n.to_array() {
            assert!(val.is_nan(), "NOT(0.0) should be NaN");
        }
    }
}

// ============================================================================
// Into<[f32; 8]>
// ============================================================================

#[test]
fn into_array() {
    if let Some(token) = X64V3Token::summon() {
        let v = f32x8::<X64V3Token>::splat(token, 42.0);
        let arr: [f32; 8] = v.into();
        assert_eq!(arr, [42.0; 8]);
    }
}

// ============================================================================
// Debug
// ============================================================================

#[test]
fn debug_format() {
    if let Some(token) = X64V3Token::summon() {
        let v = f32x8::<X64V3Token>::splat(token, 1.0);
        let s = format!("{v:?}");
        assert!(
            s.starts_with("f32x8("),
            "Debug should start with f32x8(, got: {s}"
        );
    }
}

// ============================================================================
// Platform-specific: raw access
// ============================================================================

#[test]
fn raw_m256_roundtrip() {
    if let Some(token) = X64V3Token::summon() {
        let v = f32x8::<X64V3Token>::splat(token, 5.0);
        let raw = v.raw();
        let v2 = f32x8::<X64V3Token>::from_m256(token, raw);
        assert_eq!(v2.to_array(), [5.0; 8]);
    }
}

// ============================================================================
// Generic function test — this is the whole point
// ============================================================================

fn generic_sum<T: F32x8Backend>(token: T, data: &[f32; 8]) -> f32 {
    let v = f32x8::<T>::load(token, data);
    v.reduce_add()
}

fn generic_dot<T: F32x8Backend>(token: T, a: &[f32; 8], b: &[f32; 8]) -> f32 {
    let va = f32x8::<T>::load(token, a);
    let vb = f32x8::<T>::load(token, b);
    (va * vb).reduce_add()
}

#[test]
fn generic_functions_work() {
    if let Some(token) = X64V3Token::summon() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert_eq!(generic_sum(token, &data), 36.0);

        let a = [1.0; 8];
        let b = [2.0; 8];
        assert_eq!(generic_dot(token, &a, &b), 16.0);
    }
}

// ============================================================================
// Assign operators
// ============================================================================

#[test]
fn assign_operators() {
    if let Some(token) = X64V3Token::summon() {
        let mut v = f32x8::<X64V3Token>::splat(token, 2.0);
        v += f32x8::<X64V3Token>::splat(token, 3.0);
        assert_eq!(v.to_array(), [5.0; 8]);

        v -= f32x8::<X64V3Token>::splat(token, 1.0);
        assert_eq!(v.to_array(), [4.0; 8]);

        v *= f32x8::<X64V3Token>::splat(token, 2.0);
        assert_eq!(v.to_array(), [8.0; 8]);

        v /= f32x8::<X64V3Token>::splat(token, 4.0);
        assert_eq!(v.to_array(), [2.0; 8]);
    }
}

// ============================================================================
// repr/from_repr roundtrip
// ============================================================================

#[test]
fn repr_roundtrip() {
    if let Some(token) = X64V3Token::summon() {
        let v = f32x8::<X64V3Token>::splat(token, 7.0);
        let repr = v.into_repr();
        let v2 = f32x8::<X64V3Token>::from_repr(token, repr);
        assert_eq!(v2.to_array(), [7.0; 8]);
    }
}

// ============================================================================
// ScalarToken backend tests
// ============================================================================

#[test]
fn scalar_basic_arithmetic() {
    let token = ScalarToken;
    let a = f32x8::<ScalarToken>::splat(token, 3.0);
    let b = f32x8::<ScalarToken>::splat(token, 2.0);

    assert_eq!((a + b).to_array(), [5.0; 8]);
    assert_eq!((a - b).to_array(), [1.0; 8]);
    assert_eq!((a * b).to_array(), [6.0; 8]);
    assert_eq!((a / b).to_array(), [1.5; 8]);
    assert_eq!((-a).to_array(), [-3.0; 8]);
}

#[test]
fn scalar_math() {
    let token = ScalarToken;

    let v = f32x8::<ScalarToken>::splat(token, 4.0);
    assert_eq!(v.sqrt().to_array(), [2.0; 8]);
    assert_eq!(v.abs().to_array(), [4.0; 8]);
    assert_eq!((-v).abs().to_array(), [4.0; 8]);

    let v = f32x8::<ScalarToken>::splat(token, 2.7);
    assert_eq!(v.floor().to_array(), [2.0; 8]);
    assert_eq!(v.ceil().to_array(), [3.0; 8]);

    let a = f32x8::<ScalarToken>::splat(token, 2.0);
    let b = f32x8::<ScalarToken>::splat(token, 3.0);
    let c = f32x8::<ScalarToken>::splat(token, 1.0);
    assert_eq!(a.mul_add(b, c).to_array(), [7.0; 8]);
    assert_eq!(a.mul_sub(b, c).to_array(), [5.0; 8]);
}

#[test]
fn scalar_reductions() {
    let token = ScalarToken;
    let v = f32x8::<ScalarToken>::from_array(token, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert_eq!(v.reduce_add(), 36.0);
    assert_eq!(v.reduce_min(), 1.0);
    assert_eq!(v.reduce_max(), 8.0);
}

#[test]
fn scalar_comparisons() {
    let token = ScalarToken;
    let a = f32x8::<ScalarToken>::from_array(token, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let b = f32x8::<ScalarToken>::splat(token, 4.0);
    let mask = a.simd_lt(b);
    let selected = f32x8::<ScalarToken>::blend(
        mask,
        f32x8::<ScalarToken>::splat(token, 10.0),
        f32x8::<ScalarToken>::splat(token, 20.0),
    );
    let result = selected.to_array();
    assert_eq!(result[0], 10.0); // 1 < 4 → true
    assert_eq!(result[2], 10.0); // 3 < 4 → true
    assert_eq!(result[3], 20.0); // 4 < 4 → false
    assert_eq!(result[7], 20.0); // 8 < 4 → false
}

#[test]
fn scalar_size() {
    // ScalarToken repr is [f32; 8] = 32 bytes, same as SIMD
    assert_eq!(core::mem::size_of::<f32x8<ScalarToken>>(), 32);
}

// ============================================================================
// Cross-backend: same generic function, both backends produce same results
// ============================================================================

#[test]
fn cross_backend_sum() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let scalar_result = generic_sum(ScalarToken, &data);

    if let Some(token) = X64V3Token::summon() {
        let simd_result = generic_sum(token, &data);
        assert_eq!(
            scalar_result, simd_result,
            "Scalar and AVX2 backends should produce identical results"
        );
    }
}

#[test]
fn cross_backend_dot() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

    let scalar_result = generic_dot(ScalarToken, &a, &b);

    if let Some(token) = X64V3Token::summon() {
        let simd_result = generic_dot(token, &a, &b);
        assert_eq!(
            scalar_result, simd_result,
            "Scalar and AVX2 backends should produce identical results"
        );
    }
}

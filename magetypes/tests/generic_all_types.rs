//! Comprehensive tests for ALL 30 generic SIMD types.
//!
//! Uses ScalarToken exclusively so tests work on every platform (x86, ARM, WASM).
//! For x86-specific backend tests, see generic_f32x8.rs, generic_int_types.rs, etc.
//!
//! Coverage: 6 float types + 24 integer types = 30 types, ~450 tests total.

#![allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]

use archmage::ScalarToken;

// ============================================================================
// Float type tests (f32x4, f32x8, f32x16, f64x2, f64x4, f64x8)
// ============================================================================

macro_rules! float_tests {
    ($mod:ident, $ty:ident, $elem:ty, $lanes:expr) => {
        mod $mod {
            use archmage::ScalarToken;
            use magetypes::simd::generic::$ty;

            #[test]
            fn construction() {
                let t = ScalarToken;
                assert_eq!(
                    $ty::<ScalarToken>::zero(t).to_array(),
                    [0.0 as $elem; $lanes]
                );
                assert_eq!(
                    $ty::<ScalarToken>::splat(t, 5.0 as $elem).to_array(),
                    [5.0 as $elem; $lanes]
                );
            }

            #[test]
            fn load_store_roundtrip() {
                let t = ScalarToken;
                let v = $ty::<ScalarToken>::splat(t, 42.0 as $elem);
                let arr = v.to_array();
                assert_eq!(arr, [42.0 as $elem; $lanes]);
                let v2 = $ty::<ScalarToken>::from_array(t, arr);
                let mut out = [0.0 as $elem; $lanes];
                v2.store(&mut out);
                assert_eq!(out, [42.0 as $elem; $lanes]);
            }

            #[test]
            fn from_slice() {
                let t = ScalarToken;
                let data: Vec<$elem> = (0..($lanes + 2)).map(|i| (i as $elem) + 1.0).collect();
                let v = $ty::<ScalarToken>::from_slice(t, &data);
                for i in 0..$lanes {
                    assert_eq!(v[i], (i as $elem) + 1.0);
                }
            }

            #[test]
            fn arithmetic() {
                let t = ScalarToken;
                let a = $ty::<ScalarToken>::splat(t, 6.0 as $elem);
                let b = $ty::<ScalarToken>::splat(t, 2.0 as $elem);
                assert_eq!((a + b).to_array(), [8.0 as $elem; $lanes]);
                assert_eq!((a - b).to_array(), [4.0 as $elem; $lanes]);
                assert_eq!((a * b).to_array(), [12.0 as $elem; $lanes]);
                assert_eq!((a / b).to_array(), [3.0 as $elem; $lanes]);
                assert_eq!((-a).to_array(), [-6.0 as $elem; $lanes]);
            }

            #[test]
            fn math() {
                let t = ScalarToken;
                let v = $ty::<ScalarToken>::splat(t, 4.0 as $elem);
                assert_eq!(v.sqrt().to_array(), [2.0 as $elem; $lanes]);
                assert_eq!(v.abs().to_array(), [4.0 as $elem; $lanes]);
                assert_eq!((-v).abs().to_array(), [4.0 as $elem; $lanes]);

                let w = $ty::<ScalarToken>::splat(t, 2.7 as $elem);
                assert_eq!(w.floor().to_array(), [2.0 as $elem; $lanes]);
                assert_eq!(w.ceil().to_array(), [3.0 as $elem; $lanes]);
            }

            #[test]
            fn mul_add_sub() {
                let t = ScalarToken;
                let a = $ty::<ScalarToken>::splat(t, 2.0 as $elem);
                let b = $ty::<ScalarToken>::splat(t, 3.0 as $elem);
                let c = $ty::<ScalarToken>::splat(t, 1.0 as $elem);
                assert_eq!(a.mul_add(b, c).to_array(), [7.0 as $elem; $lanes]);
                assert_eq!(a.mul_sub(b, c).to_array(), [5.0 as $elem; $lanes]);
            }

            #[test]
            fn comparisons_and_blend() {
                let t = ScalarToken;
                let a = $ty::<ScalarToken>::splat(t, 3.0 as $elem);
                let b = $ty::<ScalarToken>::splat(t, 5.0 as $elem);
                let mask = a.simd_lt(b);
                let one = $ty::<ScalarToken>::splat(t, 1.0 as $elem);
                let zero = $ty::<ScalarToken>::zero(t);
                // a < b is all-true → blend picks `one`
                assert_eq!(
                    $ty::blend(mask, one, zero).to_array(),
                    [1.0 as $elem; $lanes]
                );
                // a == a is all-true
                let eq_mask = a.simd_eq(a);
                assert_eq!(
                    $ty::blend(eq_mask, one, zero).to_array(),
                    [1.0 as $elem; $lanes]
                );
            }

            #[test]
            fn all_six_comparisons() {
                let t = ScalarToken;
                let lo = $ty::<ScalarToken>::splat(t, 1.0 as $elem);
                let hi = $ty::<ScalarToken>::splat(t, 2.0 as $elem);
                let one = $ty::<ScalarToken>::splat(t, 1.0 as $elem);
                let zero = $ty::<ScalarToken>::zero(t);

                // eq/ne
                assert_eq!(
                    $ty::blend(lo.simd_eq(lo), one, zero).to_array(),
                    [1.0 as $elem; $lanes]
                );
                assert_eq!(
                    $ty::blend(lo.simd_ne(hi), one, zero).to_array(),
                    [1.0 as $elem; $lanes]
                );
                // lt/le/gt/ge
                assert_eq!(
                    $ty::blend(lo.simd_lt(hi), one, zero).to_array(),
                    [1.0 as $elem; $lanes]
                );
                assert_eq!(
                    $ty::blend(lo.simd_le(lo), one, zero).to_array(),
                    [1.0 as $elem; $lanes]
                );
                assert_eq!(
                    $ty::blend(hi.simd_gt(lo), one, zero).to_array(),
                    [1.0 as $elem; $lanes]
                );
                assert_eq!(
                    $ty::blend(hi.simd_ge(hi), one, zero).to_array(),
                    [1.0 as $elem; $lanes]
                );
            }

            #[test]
            fn reductions() {
                let t = ScalarToken;
                let v = $ty::<ScalarToken>::splat(t, 1.0 as $elem);
                assert_eq!(v.reduce_add(), $lanes as $elem);
                assert_eq!(v.reduce_min(), 1.0 as $elem);
                assert_eq!(v.reduce_max(), 1.0 as $elem);
            }

            #[test]
            fn clamp() {
                let t = ScalarToken;
                let v = $ty::<ScalarToken>::splat(t, 10.0 as $elem);
                let lo = $ty::<ScalarToken>::splat(t, 0.0 as $elem);
                let hi = $ty::<ScalarToken>::splat(t, 5.0 as $elem);
                assert_eq!(v.clamp(lo, hi).to_array(), [5.0 as $elem; $lanes]);

                let v2 = $ty::<ScalarToken>::splat(t, -1.0 as $elem);
                assert_eq!(v2.clamp(lo, hi).to_array(), [0.0 as $elem; $lanes]);
            }

            #[test]
            fn indexing() {
                let t = ScalarToken;
                let mut v = $ty::<ScalarToken>::splat(t, 1.0 as $elem);
                assert_eq!(v[0], 1.0 as $elem);
                v[0] = 99.0 as $elem;
                assert_eq!(v[0], 99.0 as $elem);
                // Last lane
                assert_eq!(v[$lanes - 1], 1.0 as $elem);
            }

            #[test]
            fn assign_ops() {
                let t = ScalarToken;
                let mut v = $ty::<ScalarToken>::splat(t, 2.0 as $elem);
                v += $ty::<ScalarToken>::splat(t, 3.0 as $elem);
                assert_eq!(v.to_array(), [5.0 as $elem; $lanes]);
                v -= $ty::<ScalarToken>::splat(t, 1.0 as $elem);
                assert_eq!(v.to_array(), [4.0 as $elem; $lanes]);
                v *= $ty::<ScalarToken>::splat(t, 2.0 as $elem);
                assert_eq!(v.to_array(), [8.0 as $elem; $lanes]);
                v /= $ty::<ScalarToken>::splat(t, 4.0 as $elem);
                assert_eq!(v.to_array(), [2.0 as $elem; $lanes]);
            }

            #[test]
            fn scalar_broadcast() {
                let t = ScalarToken;
                let v = $ty::<ScalarToken>::splat(t, 10.0 as $elem);
                assert_eq!((v + 5.0 as $elem).to_array(), [15.0 as $elem; $lanes]);
                assert_eq!((v - 3.0 as $elem).to_array(), [7.0 as $elem; $lanes]);
                assert_eq!((v * 2.0 as $elem).to_array(), [20.0 as $elem; $lanes]);
                assert_eq!((v / 5.0 as $elem).to_array(), [2.0 as $elem; $lanes]);
            }

            #[test]
            fn into_array() {
                let t = ScalarToken;
                let v = $ty::<ScalarToken>::splat(t, 7.0 as $elem);
                let arr: [$elem; $lanes] = v.into();
                assert_eq!(arr, [7.0 as $elem; $lanes]);
            }

            #[test]
            fn debug_format() {
                let t = ScalarToken;
                let v = $ty::<ScalarToken>::splat(t, 1.0 as $elem);
                let s = format!("{v:?}");
                assert!(
                    s.contains(stringify!($ty)),
                    "Debug should contain '{}', got: {}",
                    stringify!($ty),
                    s
                );
            }

            #[test]
            fn not_bitwise() {
                let t = ScalarToken;
                let z = $ty::<ScalarToken>::zero(t);
                // NOT of all-zeros gives NaN (all bits set)
                for &val in &z.not().to_array() {
                    assert!(val.is_nan(), "NOT(0.0) should be NaN");
                }
            }
        }
    };
}

float_tests!(test_f32x4, f32x4, f32, 4);
float_tests!(test_f32x8, f32x8, f32, 8);
float_tests!(test_f32x16, f32x16, f32, 16);
float_tests!(test_f64x2, f64x2, f64, 2);
float_tests!(test_f64x4, f64x4, f64, 4);
float_tests!(test_f64x8, f64x8, f64, 8);

// ============================================================================
// Integer helper macros
// ============================================================================

/// Tests common to ALL integer types (signed and unsigned, with and without mul).
macro_rules! common_int_tests {
    ($ty:ident, $elem:ty, $lanes:expr) => {
        #[test]
        fn construction() {
            let t = ScalarToken;
            assert_eq!($ty::<ScalarToken>::zero(t).to_array(), [0 as $elem; $lanes]);
            assert_eq!(
                $ty::<ScalarToken>::splat(t, 5 as $elem).to_array(),
                [5 as $elem; $lanes]
            );
        }

        #[test]
        fn load_store_roundtrip() {
            let t = ScalarToken;
            let v = $ty::<ScalarToken>::splat(t, 42 as $elem);
            let arr = v.to_array();
            assert_eq!(arr, [42 as $elem; $lanes]);
            let v2 = $ty::<ScalarToken>::from_array(t, arr);
            let mut out = [0 as $elem; $lanes];
            v2.store(&mut out);
            assert_eq!(out, [42 as $elem; $lanes]);
        }

        #[test]
        fn add_sub() {
            let t = ScalarToken;
            let a = $ty::<ScalarToken>::splat(t, 10 as $elem);
            let b = $ty::<ScalarToken>::splat(t, 3 as $elem);
            assert_eq!((a + b).to_array(), [13 as $elem; $lanes]);
            assert_eq!((a - b).to_array(), [7 as $elem; $lanes]);
        }

        #[test]
        fn min_max_clamp() {
            let t = ScalarToken;
            let a = $ty::<ScalarToken>::splat(t, 3 as $elem);
            let b = $ty::<ScalarToken>::splat(t, 7 as $elem);
            assert_eq!(a.min(b).to_array(), [3 as $elem; $lanes]);
            assert_eq!(a.max(b).to_array(), [7 as $elem; $lanes]);
            let v = $ty::<ScalarToken>::splat(t, 10 as $elem);
            assert_eq!(v.clamp(a, b).to_array(), [7 as $elem; $lanes]);
        }

        #[test]
        fn comparisons() {
            let t = ScalarToken;
            let a = $ty::<ScalarToken>::splat(t, 3 as $elem);
            let b = $ty::<ScalarToken>::splat(t, 5 as $elem);
            // a < b → all true
            assert!(a.simd_lt(b).all_true());
            // a == a → all true
            assert!(a.simd_eq(a).all_true());
            // a != b → all true
            assert!(a.simd_ne(b).all_true());
            // a > b → all false
            assert!(!a.simd_gt(b).any_true());
            // b >= a → all true
            assert!(b.simd_ge(a).all_true());
            // a <= b → all true
            assert!(a.simd_le(b).all_true());
        }

        #[test]
        fn blend() {
            let t = ScalarToken;
            let a = $ty::<ScalarToken>::splat(t, 10 as $elem);
            let b = $ty::<ScalarToken>::splat(t, 20 as $elem);
            // All-1s mask → pick a
            let mask = $ty::<ScalarToken>::splat(t, !0 as $elem);
            assert_eq!($ty::blend(mask, a, b).to_array(), [10 as $elem; $lanes]);
            // All-0s mask → pick b
            let no_mask = $ty::<ScalarToken>::zero(t);
            assert_eq!($ty::blend(no_mask, a, b).to_array(), [20 as $elem; $lanes]);
        }

        #[test]
        fn reduce_add() {
            let t = ScalarToken;
            let v = $ty::<ScalarToken>::splat(t, 1 as $elem);
            assert_eq!(v.reduce_add(), $lanes as $elem);
        }

        #[test]
        fn shifts() {
            let t = ScalarToken;
            let v = $ty::<ScalarToken>::splat(t, 4 as $elem);
            assert_eq!(v.shl_const::<1>().to_array(), [8 as $elem; $lanes]);
            assert_eq!(v.shr_logical_const::<1>().to_array(), [2 as $elem; $lanes]);
        }

        #[test]
        fn bitwise_ops() {
            let t = ScalarToken;
            let a = $ty::<ScalarToken>::splat(t, 0x0F as $elem);
            let b = $ty::<ScalarToken>::splat(t, 0x03 as $elem);
            assert_eq!((a & b).to_array(), [0x03 as $elem; $lanes]);
            assert_eq!((a | b).to_array(), [0x0F as $elem; $lanes]);
            assert_eq!((a ^ b).to_array(), [0x0C as $elem; $lanes]);
        }

        #[test]
        fn not_op() {
            let t = ScalarToken;
            let z = $ty::<ScalarToken>::zero(t);
            assert_eq!(z.not().to_array(), [!0 as $elem; $lanes]);
        }

        #[test]
        fn boolean_reductions() {
            let t = ScalarToken;
            let all = $ty::<ScalarToken>::splat(t, !0 as $elem);
            let none = $ty::<ScalarToken>::zero(t);
            assert!(all.all_true());
            assert!(all.any_true());
            assert!(!none.all_true());
            assert!(!none.any_true());
        }

        #[test]
        fn bitmask() {
            let t = ScalarToken;
            let all = $ty::<ScalarToken>::splat(t, !0 as $elem);
            let none = $ty::<ScalarToken>::zero(t);
            let expected: u64 = if $lanes >= 64 {
                !0u64
            } else {
                (1u64 << $lanes) - 1
            };
            assert_eq!(all.bitmask() as u64, expected);
            assert_eq!(none.bitmask() as u64, 0);
        }

        #[test]
        fn indexing() {
            let t = ScalarToken;
            let mut v = $ty::<ScalarToken>::splat(t, 1 as $elem);
            assert_eq!(v[0], 1 as $elem);
            v[0] = 99 as $elem;
            assert_eq!(v[0], 99 as $elem);
            assert_eq!(v[$lanes - 1], 1 as $elem);
        }

        #[test]
        fn scalar_broadcast_add_sub() {
            let t = ScalarToken;
            let v = $ty::<ScalarToken>::splat(t, 10 as $elem);
            assert_eq!((v + 5 as $elem).to_array(), [15 as $elem; $lanes]);
            assert_eq!((v - 3 as $elem).to_array(), [7 as $elem; $lanes]);
        }

        #[test]
        fn assign_add_sub() {
            let t = ScalarToken;
            let mut v = $ty::<ScalarToken>::splat(t, 2 as $elem);
            v += $ty::<ScalarToken>::splat(t, 3 as $elem);
            assert_eq!(v.to_array(), [5 as $elem; $lanes]);
            v -= $ty::<ScalarToken>::splat(t, 1 as $elem);
            assert_eq!(v.to_array(), [4 as $elem; $lanes]);
        }

        #[test]
        fn debug_format() {
            let t = ScalarToken;
            let v = $ty::<ScalarToken>::splat(t, 1 as $elem);
            let s = format!("{v:?}");
            assert!(
                s.contains(stringify!($ty)),
                "Debug should contain '{}', got: {}",
                stringify!($ty),
                s
            );
        }

        #[test]
        fn into_array() {
            let t = ScalarToken;
            let v = $ty::<ScalarToken>::splat(t, 7 as $elem);
            let arr: [$elem; $lanes] = v.into();
            assert_eq!(arr, [7 as $elem; $lanes]);
        }
    };
}

/// Extra tests for signed integer types: neg, abs, shr_arithmetic.
macro_rules! signed_extras {
    ($ty:ident, $elem:ty, $lanes:expr) => {
        #[test]
        fn neg() {
            let t = ScalarToken;
            let v = $ty::<ScalarToken>::splat(t, 5 as $elem);
            assert_eq!((-v).to_array(), [-5 as $elem; $lanes]);
        }

        #[test]
        fn abs() {
            let t = ScalarToken;
            let v = $ty::<ScalarToken>::splat(t, -5 as $elem);
            assert_eq!(v.abs().to_array(), [5 as $elem; $lanes]);
        }

        #[test]
        fn shr_arithmetic() {
            let t = ScalarToken;
            let v = $ty::<ScalarToken>::splat(t, -8 as $elem);
            assert_eq!(
                v.shr_arithmetic_const::<1>().to_array(),
                [-4 as $elem; $lanes]
            );
        }
    };
}

/// Extra tests for types with hardware multiply.
macro_rules! mul_extras {
    ($ty:ident, $elem:ty, $lanes:expr) => {
        #[test]
        fn mul() {
            let t = ScalarToken;
            let a = $ty::<ScalarToken>::splat(t, 3 as $elem);
            let b = $ty::<ScalarToken>::splat(t, 4 as $elem);
            assert_eq!((a * b).to_array(), [12 as $elem; $lanes]);
        }

        #[test]
        fn mul_assign() {
            let t = ScalarToken;
            let mut v = $ty::<ScalarToken>::splat(t, 3 as $elem);
            v *= $ty::<ScalarToken>::splat(t, 2 as $elem);
            assert_eq!(v.to_array(), [6 as $elem; $lanes]);
        }

        #[test]
        fn scalar_mul() {
            let t = ScalarToken;
            let v = $ty::<ScalarToken>::splat(t, 5 as $elem);
            assert_eq!((v * 3 as $elem).to_array(), [15 as $elem; $lanes]);
        }
    };
}

// ============================================================================
// Integer type test instantiations
// ============================================================================

/// Master macro that combines the appropriate test sets for each integer variant.
macro_rules! int_type_tests {
    // Signed integer WITH hardware multiply (i16, i32)
    (signed_mul: $mod:ident, $ty:ident, $elem:ty, $lanes:expr) => {
        mod $mod {
            use archmage::ScalarToken;
            use magetypes::simd::generic::$ty;
            common_int_tests!($ty, $elem, $lanes);
            signed_extras!($ty, $elem, $lanes);
            mul_extras!($ty, $elem, $lanes);
        }
    };
    // Signed integer WITHOUT hardware multiply (i8, i64)
    (signed: $mod:ident, $ty:ident, $elem:ty, $lanes:expr) => {
        mod $mod {
            use archmage::ScalarToken;
            use magetypes::simd::generic::$ty;
            common_int_tests!($ty, $elem, $lanes);
            signed_extras!($ty, $elem, $lanes);
        }
    };
    // Unsigned integer WITH hardware multiply (u16, u32)
    (unsigned_mul: $mod:ident, $ty:ident, $elem:ty, $lanes:expr) => {
        mod $mod {
            use archmage::ScalarToken;
            use magetypes::simd::generic::$ty;
            common_int_tests!($ty, $elem, $lanes);
            mul_extras!($ty, $elem, $lanes);
        }
    };
    // Unsigned integer WITHOUT hardware multiply (u8, u64)
    (unsigned: $mod:ident, $ty:ident, $elem:ty, $lanes:expr) => {
        mod $mod {
            use archmage::ScalarToken;
            use magetypes::simd::generic::$ty;
            common_int_tests!($ty, $elem, $lanes);
        }
    };
}

// --- Signed integers with mul (i16, i32) ---
int_type_tests!(signed_mul: test_i16x8, i16x8, i16, 8);
int_type_tests!(signed_mul: test_i16x16, i16x16, i16, 16);
int_type_tests!(signed_mul: test_i16x32, i16x32, i16, 32);
int_type_tests!(signed_mul: test_i32x4, i32x4, i32, 4);
int_type_tests!(signed_mul: test_i32x8, i32x8, i32, 8);
int_type_tests!(signed_mul: test_i32x16, i32x16, i32, 16);

// --- Signed integers without mul (i8, i64) ---
int_type_tests!(signed: test_i8x16, i8x16, i8, 16);
int_type_tests!(signed: test_i8x32, i8x32, i8, 32);
int_type_tests!(signed: test_i8x64, i8x64, i8, 64);
int_type_tests!(signed: test_i64x2, i64x2, i64, 2);
int_type_tests!(signed: test_i64x4, i64x4, i64, 4);
int_type_tests!(signed: test_i64x8, i64x8, i64, 8);

// --- Unsigned integers with mul (u16, u32) ---
int_type_tests!(unsigned_mul: test_u16x8, u16x8, u16, 8);
int_type_tests!(unsigned_mul: test_u16x16, u16x16, u16, 16);
int_type_tests!(unsigned_mul: test_u16x32, u16x32, u16, 32);
int_type_tests!(unsigned_mul: test_u32x4, u32x4, u32, 4);
int_type_tests!(unsigned_mul: test_u32x8, u32x8, u32, 8);
int_type_tests!(unsigned_mul: test_u32x16, u32x16, u32, 16);

// --- Unsigned integers without mul (u8, u64) ---
int_type_tests!(unsigned: test_u8x16, u8x16, u8, 16);
int_type_tests!(unsigned: test_u8x32, u8x32, u8, 32);
int_type_tests!(unsigned: test_u8x64, u8x64, u8, 64);
int_type_tests!(unsigned: test_u64x2, u64x2, u64, 2);
int_type_tests!(unsigned: test_u64x4, u64x4, u64, 4);
int_type_tests!(unsigned: test_u64x8, u64x8, u64, 8);

// ============================================================================
// Generic function tests — the whole point of generic types
// ============================================================================

use magetypes::simd::backends::{F32x4Backend, F32x8Backend, I32x4Backend};
use magetypes::simd::generic::{f32x4, f32x8, i32x4};

fn generic_sum_f32x8<T: F32x8Backend>(token: T, data: &[f32; 8]) -> f32 {
    let v = f32x8::<T>::load(token, data);
    v.reduce_add()
}

fn generic_dot_f32x8<T: F32x8Backend>(token: T, a: &[f32; 8], b: &[f32; 8]) -> f32 {
    let va = f32x8::<T>::load(token, a);
    let vb = f32x8::<T>::load(token, b);
    (va * vb).reduce_add()
}

fn generic_sum_f32x4<T: F32x4Backend>(token: T, data: &[f32; 4]) -> f32 {
    let v = f32x4::<T>::load(token, data);
    v.reduce_add()
}

fn generic_sum_i32x4<T: I32x4Backend>(token: T, data: &[i32; 4]) -> i32 {
    let v = i32x4::<T>::load(token, data);
    v.reduce_add()
}

#[test]
fn generic_functions_scalar() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    assert_eq!(generic_sum_f32x8(ScalarToken, &data), 36.0);

    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = [2.0; 8];
    assert_eq!(generic_dot_f32x8(ScalarToken, &a, &b), 72.0);

    assert_eq!(generic_sum_f32x4(ScalarToken, &[1.0, 2.0, 3.0, 4.0]), 10.0);
    assert_eq!(generic_sum_i32x4(ScalarToken, &[10, 20, 30, 40]), 100);
}

/// Verify that a generic function called with different backends produces
/// the same results (within floating-point tolerance).
#[cfg(target_arch = "x86_64")]
#[test]
fn generic_functions_cross_backend() {
    use archmage::{SimdToken, X64V3Token};

    if let Some(t) = X64V3Token::summon() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let scalar = generic_sum_f32x8(ScalarToken, &data);
        let simd = generic_sum_f32x8(t, &data);
        assert_eq!(scalar, simd, "Scalar and AVX2 should agree");

        let a = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let b = [2.0, 7.0, 1.0, 8.0, 2.0, 8.0, 1.0, 8.0];
        let scalar_dot = generic_dot_f32x8(ScalarToken, &a, &b);
        let simd_dot = generic_dot_f32x8(t, &a, &b);
        assert_eq!(scalar_dot, simd_dot, "Scalar and AVX2 dot should agree");
    }
}

// ============================================================================
// Practical algorithm: generic normalize + distance
// ============================================================================

fn generic_euclidean_dist_sq<T: F32x8Backend>(token: T, a: &[f32; 8], b: &[f32; 8]) -> f32 {
    let va = f32x8::<T>::load(token, a);
    let vb = f32x8::<T>::load(token, b);
    let diff = va - vb;
    diff.mul_add(diff, f32x8::<T>::zero(token)).reduce_add()
}

#[test]
fn generic_distance_scalar() {
    let a = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let b = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let dist_sq = generic_euclidean_dist_sq(ScalarToken, &a, &b);
    assert!((dist_sq - 2.0).abs() < 1e-6);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn generic_distance_cross_backend() {
    use archmage::{SimdToken, X64V3Token};

    if let Some(t) = X64V3Token::summon() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let scalar = generic_euclidean_dist_sq(ScalarToken, &a, &b);
        let simd = generic_euclidean_dist_sq(t, &a, &b);
        assert!((scalar - simd).abs() < 1e-4, "scalar={scalar} simd={simd}");
    }
}

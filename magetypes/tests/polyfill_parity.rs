//! Polyfill vs native parity tests.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.
//!
//! These tests verify that polyfill implementations (using 2x W128 ops)
//! produce bit-identical results to native W256/W512 implementations.

#![cfg(target_arch = "x86_64")]
#![allow(unused_imports)]

use archmage::{SimdToken, X64V3Token};

// ============================================================================
// f32x8: polyfill::sse vs native w256
// ============================================================================

mod f32x8_parity {
    use super::*;
    use magetypes::simd::polyfill::sse::f32x8 as poly;
    use magetypes::simd::x86::w256::f32x8 as native;

    #[test]
    fn test_from_array_to_array() {
        if let Some(token) = X64V3Token::try_new() {
            let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let n = native::from_array(token, input);
            let p = poly::from_array(token, input);
            assert_eq!(n.to_array(), p.to_array(), "from_array/to_array mismatch");
        }
    }

    #[test]
    fn test_add() {
        if let Some(token) = X64V3Token::try_new() {
            let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let b = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

            let n = (native::from_array(token, a) + native::from_array(token, b)).to_array();
            let p = (poly::from_array(token, a) + poly::from_array(token, b)).to_array();
            assert_eq!(n, p, "add mismatch");
        }
    }

    #[test]
    fn test_sub() {
        if let Some(token) = X64V3Token::try_new() {
            let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let b = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

            let n = (native::from_array(token, a) - native::from_array(token, b)).to_array();
            let p = (poly::from_array(token, a) - poly::from_array(token, b)).to_array();
            assert_eq!(n, p, "sub mismatch");
        }
    }

    #[test]
    fn test_mul() {
        if let Some(token) = X64V3Token::try_new() {
            let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let b = [2.0f32, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];

            let n = (native::from_array(token, a) * native::from_array(token, b)).to_array();
            let p = (poly::from_array(token, a) * poly::from_array(token, b)).to_array();
            assert_eq!(n, p, "mul mismatch");
        }
    }

    #[test]
    fn test_div() {
        if let Some(token) = X64V3Token::try_new() {
            let a = [2.0f32, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
            let b = [2.0f32, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];

            let n = (native::from_array(token, a) / native::from_array(token, b)).to_array();
            let p = (poly::from_array(token, a) / poly::from_array(token, b)).to_array();
            assert_eq!(n, p, "div mismatch");
        }
    }

    #[test]
    fn test_min_max() {
        if let Some(token) = X64V3Token::try_new() {
            let a = [1.0f32, 5.0, 3.0, 7.0, 2.0, 6.0, 4.0, 8.0];
            let b = [2.0f32, 4.0, 6.0, 8.0, 1.0, 5.0, 3.0, 7.0];

            let n_min = native::from_array(token, a)
                .min(native::from_array(token, b))
                .to_array();
            let p_min = poly::from_array(token, a)
                .min(poly::from_array(token, b))
                .to_array();
            assert_eq!(n_min, p_min, "min mismatch");

            let n_max = native::from_array(token, a)
                .max(native::from_array(token, b))
                .to_array();
            let p_max = poly::from_array(token, a)
                .max(poly::from_array(token, b))
                .to_array();
            assert_eq!(n_max, p_max, "max mismatch");
        }
    }

    #[test]
    fn test_sqrt() {
        if let Some(token) = X64V3Token::try_new() {
            let a = [1.0f32, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0];

            let n = native::from_array(token, a).sqrt().to_array();
            let p = poly::from_array(token, a).sqrt().to_array();
            assert_eq!(n, p, "sqrt mismatch");
        }
    }

    #[test]
    fn test_floor_ceil_round() {
        if let Some(token) = X64V3Token::try_new() {
            let a = [1.3f32, 2.7, -1.3, -2.7, 0.5, 1.5, 2.5, 3.5];

            let n_floor = native::from_array(token, a).floor().to_array();
            let p_floor = poly::from_array(token, a).floor().to_array();
            assert_eq!(n_floor, p_floor, "floor mismatch");

            let n_ceil = native::from_array(token, a).ceil().to_array();
            let p_ceil = poly::from_array(token, a).ceil().to_array();
            assert_eq!(n_ceil, p_ceil, "ceil mismatch");

            let n_round = native::from_array(token, a).round().to_array();
            let p_round = poly::from_array(token, a).round().to_array();
            assert_eq!(n_round, p_round, "round mismatch");
        }
    }

    #[test]
    fn test_mul_add() {
        if let Some(token) = X64V3Token::try_new() {
            let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let b = [2.0f32, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
            let c = [0.5f32, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];

            let n = native::from_array(token, a)
                .mul_add(native::from_array(token, b), native::from_array(token, c))
                .to_array();
            let p = poly::from_array(token, a)
                .mul_add(poly::from_array(token, b), poly::from_array(token, c))
                .to_array();
            assert_eq!(n, p, "mul_add mismatch");
        }
    }

    #[test]
    fn test_abs_neg() {
        if let Some(token) = X64V3Token::try_new() {
            let a = [1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0];

            let n_abs = native::from_array(token, a).abs().to_array();
            let p_abs = poly::from_array(token, a).abs().to_array();
            assert_eq!(n_abs, p_abs, "abs mismatch");

            let n_neg = (-native::from_array(token, a)).to_array();
            let p_neg = (-poly::from_array(token, a)).to_array();
            assert_eq!(n_neg, p_neg, "neg mismatch");
        }
    }

    #[test]
    fn test_reduce() {
        if let Some(token) = X64V3Token::try_new() {
            let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

            let n_add = native::from_array(token, a).reduce_add();
            let p_add = poly::from_array(token, a).reduce_add();
            assert_eq!(n_add, p_add, "reduce_add mismatch");

            let n_min = native::from_array(token, a).reduce_min();
            let p_min = poly::from_array(token, a).reduce_min();
            assert_eq!(n_min, p_min, "reduce_min mismatch");

            let n_max = native::from_array(token, a).reduce_max();
            let p_max = poly::from_array(token, a).reduce_max();
            assert_eq!(n_max, p_max, "reduce_max mismatch");
        }
    }
}

// ============================================================================
// i32x8: polyfill::sse vs native w256
// ============================================================================

mod i32x8_parity {
    use super::*;
    use magetypes::simd::polyfill::sse::i32x8 as poly;
    use magetypes::simd::x86::w256::i32x8 as native;

    #[test]
    fn test_from_array_to_array() {
        if let Some(token) = X64V3Token::try_new() {
            let input = [1i32, -2, 3, -4, 5, -6, 7, -8];
            let n = native::from_array(token, input);
            let p = poly::from_array(token, input);
            assert_eq!(n.to_array(), p.to_array(), "from_array/to_array mismatch");
        }
    }

    #[test]
    fn test_add_sub() {
        if let Some(token) = X64V3Token::try_new() {
            let a = [10i32, 20, 30, 40, 50, 60, 70, 80];
            let b = [1i32, 2, 3, 4, 5, 6, 7, 8];

            let n_add = (native::from_array(token, a) + native::from_array(token, b)).to_array();
            let p_add = (poly::from_array(token, a) + poly::from_array(token, b)).to_array();
            assert_eq!(n_add, p_add, "add mismatch");

            let n_sub = (native::from_array(token, a) - native::from_array(token, b)).to_array();
            let p_sub = (poly::from_array(token, a) - poly::from_array(token, b)).to_array();
            assert_eq!(n_sub, p_sub, "sub mismatch");
        }
    }

    #[test]
    fn test_min_max() {
        if let Some(token) = X64V3Token::try_new() {
            let a = [1i32, -5, 3, -7, 2, -4, 6, -8];
            let b = [2i32, -4, 2, -8, 1, -5, 5, -9];

            let n_min = native::from_array(token, a)
                .min(native::from_array(token, b))
                .to_array();
            let p_min = poly::from_array(token, a)
                .min(poly::from_array(token, b))
                .to_array();
            assert_eq!(n_min, p_min, "min mismatch");

            let n_max = native::from_array(token, a)
                .max(native::from_array(token, b))
                .to_array();
            let p_max = poly::from_array(token, a)
                .max(poly::from_array(token, b))
                .to_array();
            assert_eq!(n_max, p_max, "max mismatch");
        }
    }

    #[test]
    fn test_abs() {
        if let Some(token) = X64V3Token::try_new() {
            let a = [1i32, -2, 3, -4, 5, -6, 7, -8];

            let n = native::from_array(token, a).abs().to_array();
            let p = poly::from_array(token, a).abs().to_array();
            assert_eq!(n, p, "abs mismatch");
        }
    }
}

// ============================================================================
// f64x4: polyfill::sse vs native w256
// ============================================================================

mod f64x4_parity {
    use super::*;
    use magetypes::simd::polyfill::sse::f64x4 as poly;
    use magetypes::simd::x86::w256::f64x4 as native;

    #[test]
    fn test_from_array_to_array() {
        if let Some(token) = X64V3Token::try_new() {
            let input = [1.0f64, 2.0, 3.0, 4.0];
            let n = native::from_array(token, input);
            let p = poly::from_array(token, input);
            assert_eq!(n.to_array(), p.to_array(), "from_array/to_array mismatch");
        }
    }

    #[test]
    fn test_add_sub_mul_div() {
        if let Some(token) = X64V3Token::try_new() {
            let a = [4.0f64, 8.0, 12.0, 16.0];
            let b = [2.0f64, 2.0, 2.0, 2.0];

            let n_add = (native::from_array(token, a) + native::from_array(token, b)).to_array();
            let p_add = (poly::from_array(token, a) + poly::from_array(token, b)).to_array();
            assert_eq!(n_add, p_add, "add mismatch");

            let n_sub = (native::from_array(token, a) - native::from_array(token, b)).to_array();
            let p_sub = (poly::from_array(token, a) - poly::from_array(token, b)).to_array();
            assert_eq!(n_sub, p_sub, "sub mismatch");

            let n_mul = (native::from_array(token, a) * native::from_array(token, b)).to_array();
            let p_mul = (poly::from_array(token, a) * poly::from_array(token, b)).to_array();
            assert_eq!(n_mul, p_mul, "mul mismatch");

            let n_div = (native::from_array(token, a) / native::from_array(token, b)).to_array();
            let p_div = (poly::from_array(token, a) / poly::from_array(token, b)).to_array();
            assert_eq!(n_div, p_div, "div mismatch");
        }
    }

    #[test]
    fn test_sqrt() {
        if let Some(token) = X64V3Token::try_new() {
            let a = [1.0f64, 4.0, 9.0, 16.0];

            let n = native::from_array(token, a).sqrt().to_array();
            let p = poly::from_array(token, a).sqrt().to_array();
            assert_eq!(n, p, "sqrt mismatch");
        }
    }
}

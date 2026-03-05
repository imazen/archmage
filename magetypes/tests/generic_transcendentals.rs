//! Tests for generic transcendental math functions on f32x4<T>, f32x8<T>, and f32x16<T>.
//!
//! Tests against std::f32 functions and known exact values.
//! Verifies both X64V3Token (AVX2+FMA) and ScalarToken backends.

#[cfg(target_arch = "x86_64")]
mod tests {
    use archmage::{ScalarToken, SimdToken, X64V3Token};
    use magetypes::simd::generic::{f32x4, f32x8, f32x16};

    // ====== Helpers ======

    /// Check that all lanes are within `tol` of expected (relative error for large values,
    /// absolute for small).
    fn assert_close(actual: &[f32], expected: &[f32], tol: f32, msg: &str) {
        assert_eq!(actual.len(), expected.len());
        for i in 0..actual.len() {
            let a = actual[i];
            let e = expected[i];
            let err = if e.abs() > 1.0 {
                ((a - e) / e).abs()
            } else {
                (a - e).abs()
            };
            assert!(
                err < tol,
                "{msg}: lane {i}: got {a}, expected {e}, err {err} (tol {tol})"
            );
        }
    }

    fn assert_close_8(actual: [f32; 8], expected: [f32; 8], tol: f32, msg: &str) {
        assert_close(&actual, &expected, tol, msg);
    }

    fn assert_close_4(actual: [f32; 4], expected: [f32; 4], tol: f32, msg: &str) {
        assert_close(&actual, &expected, tol, msg);
    }

    fn assert_close_16(actual: [f32; 16], expected: [f32; 16], tol: f32, msg: &str) {
        assert_close(&actual, &expected, tol, msg);
    }

    // ====== f32x8 Low-Precision Tests (X64V3Token) ======

    #[test]
    fn f32x8_log2_lowp_exact_powers() {
        if let Some(t) = X64V3Token::summon() {
            let input = f32x8::from_array(t, [1.0, 2.0, 4.0, 8.0, 0.5, 0.25, 16.0, 64.0]);
            let result = input.log2_lowp().to_array();
            let expected = [0.0, 1.0, 2.0, 3.0, -1.0, -2.0, 4.0, 6.0];
            assert_close_8(result, expected, 0.02, "log2_lowp powers of 2");
        }
    }

    #[test]
    fn f32x8_log2_lowp_general() {
        if let Some(t) = X64V3Token::summon() {
            let input = f32x8::from_array(t, [1.5, 3.0, 10.0, 100.0, 0.1, 0.01, 7.0, 42.0]);
            let result = input.log2_lowp().to_array();
            let expected: [f32; 8] = core::array::from_fn(|i| {
                [1.5_f32, 3.0, 10.0, 100.0, 0.1, 0.01, 7.0, 42.0][i].log2()
            });
            assert_close_8(result, expected, 0.02, "log2_lowp general");
        }
    }

    #[test]
    fn f32x8_exp2_lowp_exact() {
        if let Some(t) = X64V3Token::summon() {
            let input = f32x8::from_array(t, [0.0, 1.0, 2.0, 3.0, -1.0, -2.0, 4.0, 7.0]);
            let result = input.exp2_lowp().to_array();
            let expected = [1.0, 2.0, 4.0, 8.0, 0.5, 0.25, 16.0, 128.0];
            assert_close_8(result, expected, 0.02, "exp2_lowp exact");
        }
    }

    #[test]
    fn f32x8_exp2_lowp_fractional() {
        if let Some(t) = X64V3Token::summon() {
            let input = f32x8::from_array(t, [0.5, 1.5, -0.5, 2.5, 0.1, 0.9, -3.5, 10.0]);
            let result = input.exp2_lowp().to_array();
            let expected: [f32; 8] =
                core::array::from_fn(|i| [0.5_f32, 1.5, -0.5, 2.5, 0.1, 0.9, -3.5, 10.0][i].exp2());
            assert_close_8(result, expected, 0.02, "exp2_lowp fractional");
        }
    }

    #[test]
    fn f32x8_ln_lowp() {
        if let Some(t) = X64V3Token::summon() {
            let vals = [1.0, 2.718_281_8, 10.0, 0.5, 100.0, 0.1, 7.389, 20.0];
            let input = f32x8::from_array(t, vals);
            let result = input.ln_lowp().to_array();
            let expected: [f32; 8] = core::array::from_fn(|i| vals[i].ln());
            assert_close_8(result, expected, 0.02, "ln_lowp");
        }
    }

    #[test]
    fn f32x8_exp_lowp() {
        if let Some(t) = X64V3Token::summon() {
            let vals = [0.0, 1.0, -1.0, 2.0, 0.5, -2.0, 3.0, -0.5];
            let input = f32x8::from_array(t, vals);
            let result = input.exp_lowp().to_array();
            let expected: [f32; 8] = core::array::from_fn(|i| vals[i].exp());
            assert_close_8(result, expected, 0.02, "exp_lowp");
        }
    }

    #[test]
    fn f32x8_log10_lowp() {
        if let Some(t) = X64V3Token::summon() {
            let vals = [1.0, 10.0, 100.0, 1000.0, 0.1, 0.01, 50.0, 7.0];
            let input = f32x8::from_array(t, vals);
            let result = input.log10_lowp().to_array();
            let expected: [f32; 8] = core::array::from_fn(|i| vals[i].log10());
            assert_close_8(result, expected, 0.02, "log10_lowp");
        }
    }

    #[test]
    fn f32x8_pow_lowp() {
        if let Some(t) = X64V3Token::summon() {
            let vals = [1.0, 2.0, 4.0, 8.0, 3.0, 10.0, 0.5, 100.0];
            let input = f32x8::from_array(t, vals);
            let result = input.pow_lowp(0.5).to_array();
            let expected: [f32; 8] = core::array::from_fn(|i| vals[i].sqrt());
            assert_close_8(result, expected, 0.02, "pow_lowp(0.5) ≈ sqrt");
        }
    }

    // ====== f32x8 Mid-Precision Tests (X64V3Token) ======

    #[test]
    fn f32x8_log2_midp_exact_powers() {
        if let Some(t) = X64V3Token::summon() {
            let input = f32x8::from_array(t, [1.0, 2.0, 4.0, 8.0, 0.5, 0.25, 16.0, 64.0]);
            let result = input.log2_midp_unchecked().to_array();
            let expected = [0.0, 1.0, 2.0, 3.0, -1.0, -2.0, 4.0, 6.0];
            assert_close_8(result, expected, 1e-5, "log2_midp powers of 2");
        }
    }

    #[test]
    fn f32x8_log2_midp_general() {
        if let Some(t) = X64V3Token::summon() {
            let vals = [1.5, 3.0, 10.0, 100.0, 0.1, 0.01, 7.0, 42.0];
            let input = f32x8::from_array(t, vals);
            let result = input.log2_midp_unchecked().to_array();
            let expected: [f32; 8] = core::array::from_fn(|i| vals[i].log2());
            assert_close_8(result, expected, 5e-6, "log2_midp general");
        }
    }

    #[test]
    fn f32x8_log2_midp_edge_cases() {
        if let Some(t) = X64V3Token::summon() {
            let input = f32x8::from_array(t, [0.0, -1.0, 1.0, 2.0, 0.0, -0.5, 4.0, 8.0]);
            let result = input.log2_midp().to_array();

            // 0 -> -inf
            assert!(
                result[0].is_infinite() && result[0].is_sign_negative(),
                "log2(0) = -inf"
            );
            assert!(
                result[4].is_infinite() && result[4].is_sign_negative(),
                "log2(0) = -inf"
            );

            // negative -> NaN
            assert!(result[1].is_nan(), "log2(-1) = NaN");
            assert!(result[5].is_nan(), "log2(-0.5) = NaN");

            // normal values
            assert!((result[2] - 0.0).abs() < 1e-5, "log2(1) ~ 0");
            assert!((result[3] - 1.0).abs() < 1e-5, "log2(2) ~ 1");
        }
    }

    #[test]
    fn f32x8_exp2_midp_exact() {
        if let Some(t) = X64V3Token::summon() {
            let input = f32x8::from_array(t, [0.0, 1.0, 2.0, 3.0, -1.0, -2.0, 4.0, 7.0]);
            let result = input.exp2_midp().to_array();
            let expected = [1.0, 2.0, 4.0, 8.0, 0.5, 0.25, 16.0, 128.0];
            assert_close_8(result, expected, 1e-5, "exp2_midp exact");
        }
    }

    #[test]
    fn f32x8_exp2_midp_fractional() {
        if let Some(t) = X64V3Token::summon() {
            let vals = [0.5, 1.5, -0.5, 2.5, 0.1, 0.9, -3.5, 10.0];
            let input = f32x8::from_array(t, vals);
            let result = input.exp2_midp().to_array();
            let expected: [f32; 8] = core::array::from_fn(|i| vals[i].exp2());
            assert_close_8(result, expected, 5e-6, "exp2_midp fractional");
        }
    }

    #[test]
    fn f32x8_ln_midp() {
        if let Some(t) = X64V3Token::summon() {
            let vals = [
                1.0,
                core::f32::consts::E,
                10.0,
                0.5,
                100.0,
                0.1,
                7.389,
                20.0,
            ];
            let input = f32x8::from_array(t, vals);
            let result = input.ln_midp().to_array();
            let expected: [f32; 8] = core::array::from_fn(|i| vals[i].ln());
            assert_close_8(result, expected, 5e-6, "ln_midp");
        }
    }

    #[test]
    fn f32x8_exp_midp() {
        if let Some(t) = X64V3Token::summon() {
            let vals = [0.0, 1.0, -1.0, 2.0, 0.5, -2.0, 3.0, -0.5];
            let input = f32x8::from_array(t, vals);
            let result = input.exp_midp().to_array();
            let expected: [f32; 8] = core::array::from_fn(|i| vals[i].exp());
            assert_close_8(result, expected, 5e-6, "exp_midp");
        }
    }

    #[test]
    fn f32x8_log10_midp() {
        if let Some(t) = X64V3Token::summon() {
            let vals = [1.0, 10.0, 100.0, 1000.0, 0.1, 0.01, 50.0, 7.0];
            let input = f32x8::from_array(t, vals);
            let result = input.log10_midp().to_array();
            let expected: [f32; 8] = core::array::from_fn(|i| vals[i].log10());
            assert_close_8(result, expected, 5e-6, "log10_midp");
        }
    }

    #[test]
    fn f32x8_pow_midp() {
        if let Some(t) = X64V3Token::summon() {
            let vals = [1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0];
            let input = f32x8::from_array(t, vals);
            let result = input.pow_midp(0.5).to_array();
            let expected: [f32; 8] = core::array::from_fn(|i| vals[i].sqrt());
            assert_close_8(result, expected, 5e-6, "pow_midp(0.5)");
        }
    }

    #[test]
    fn f32x8_pow_midp_cube() {
        if let Some(t) = X64V3Token::summon() {
            let vals = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 0.5, 0.1];
            let input = f32x8::from_array(t, vals);
            let result = input.pow_midp(3.0).to_array();
            let expected: [f32; 8] = core::array::from_fn(|i| vals[i].powi(3));
            assert_close_8(result, expected, 5e-5, "pow_midp(3.0)");
        }
    }

    // ====== f32x8 cbrt Tests ======

    #[test]
    fn f32x8_cbrt_midp_perfect_cubes() {
        if let Some(t) = X64V3Token::summon() {
            let input = f32x8::from_array(t, [1.0, 8.0, 27.0, 64.0, 125.0, 216.0, 343.0, 512.0]);
            let result = input.cbrt_midp().to_array();
            let expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            assert_close_8(result, expected, 1e-5, "cbrt_midp perfect cubes");
        }
    }

    #[test]
    fn f32x8_cbrt_midp_negative() {
        if let Some(t) = X64V3Token::summon() {
            let input =
                f32x8::from_array(t, [-1.0, -8.0, -27.0, -64.0, -0.001, -1000.0, -0.125, -1e6]);
            let result = input.cbrt_midp().to_array();
            let expected: [f32; 8] = core::array::from_fn(|i| {
                let v = [-1.0_f32, -8.0, -27.0, -64.0, -0.001, -1000.0, -0.125, -1e6][i];
                -v.abs().cbrt()
            });
            assert_close_8(result, expected, 1e-5, "cbrt_midp negative");
        }
    }

    #[test]
    fn f32x8_cbrt_midp_general() {
        if let Some(t) = X64V3Token::summon() {
            let vals = [0.001, 0.1, 0.5, 2.0, 10.0, 100.0, 1000.0, 1e6];
            let input = f32x8::from_array(t, vals);
            let result = input.cbrt_midp().to_array();
            let expected: [f32; 8] = core::array::from_fn(|i| vals[i].cbrt());
            assert_close_8(result, expected, 1e-5, "cbrt_midp general");
        }
    }

    #[test]
    fn f32x8_cbrt_midp_precise_zero() {
        if let Some(t) = X64V3Token::summon() {
            let input = f32x8::from_array(t, [0.0, 0.0, 1.0, 8.0, -8.0, 0.0, 27.0, 0.0]);
            let result = input.cbrt_midp_precise().to_array();
            assert_eq!(result[0], 0.0, "cbrt_precise(0) = 0");
            assert_eq!(result[1], 0.0, "cbrt_precise(0) = 0");
            assert_eq!(result[5], 0.0, "cbrt_precise(0) = 0");
            assert_eq!(result[7], 0.0, "cbrt_precise(0) = 0");
            assert!((result[2] - 1.0).abs() < 1e-5, "cbrt_precise(1) ~ 1");
            assert!((result[3] - 2.0).abs() < 1e-5, "cbrt_precise(8) ~ 2");
        }
    }

    // ====== f32x8 Roundtrip Tests ======

    #[test]
    fn f32x8_log2_exp2_roundtrip() {
        if let Some(t) = X64V3Token::summon() {
            let vals = [0.5, 1.0, 2.0, 4.0, 10.0, 0.1, 50.0, 100.0];
            let input = f32x8::from_array(t, vals);
            let roundtrip = input.log2_midp_unchecked().exp2_midp().to_array();
            assert_close_8(roundtrip, vals, 1e-4, "log2 -> exp2 roundtrip");
        }
    }

    #[test]
    fn f32x8_ln_exp_roundtrip() {
        if let Some(t) = X64V3Token::summon() {
            let vals = [0.5, 1.0, 2.0, 4.0, 10.0, 0.1, 50.0, 100.0];
            let input = f32x8::from_array(t, vals);
            let roundtrip = input.ln_midp_unchecked().exp_midp().to_array();
            assert_close_8(roundtrip, vals, 1e-4, "ln -> exp roundtrip");
        }
    }

    // ====== f32x8 _unchecked Variants ======

    #[test]
    fn f32x8_unchecked_variants_match() {
        if let Some(t) = X64V3Token::summon() {
            let vals = [1.0, 2.0, 4.0, 8.0, 0.5, 0.25, 16.0, 64.0];
            let input = f32x8::from_array(t, vals);

            // _unchecked and normal should agree on valid inputs
            let log2_normal = input.log2_lowp().to_array();
            let log2_unchecked = input.log2_lowp_unchecked().to_array();
            assert_eq!(
                log2_normal, log2_unchecked,
                "log2_lowp == log2_lowp_unchecked on valid input"
            );

            let exp_vals = [0.0, 1.0, 2.0, -1.0, 0.5, -0.5, 3.0, -2.0];
            let exp_input = f32x8::from_array(t, exp_vals);
            let exp_normal = exp_input.exp2_lowp().to_array();
            let exp_unchecked = exp_input.exp2_lowp_unchecked().to_array();
            assert_eq!(
                exp_normal, exp_unchecked,
                "exp2_lowp == exp2_lowp_unchecked on valid input"
            );
        }
    }

    // ====== f32x4 Tests (X64V3Token) ======

    #[test]
    fn f32x4_log2_lowp() {
        if let Some(t) = X64V3Token::summon() {
            let input = f32x4::from_array(t, [1.0, 2.0, 4.0, 8.0]);
            let result = input.log2_lowp().to_array();
            let expected = [0.0, 1.0, 2.0, 3.0];
            assert_close_4(result, expected, 0.02, "f32x4 log2_lowp");
        }
    }

    #[test]
    fn f32x4_exp2_lowp() {
        if let Some(t) = X64V3Token::summon() {
            let input = f32x4::from_array(t, [0.0, 1.0, 2.0, -1.0]);
            let result = input.exp2_lowp().to_array();
            let expected = [1.0, 2.0, 4.0, 0.5];
            assert_close_4(result, expected, 0.02, "f32x4 exp2_lowp");
        }
    }

    #[test]
    fn f32x4_log2_midp() {
        if let Some(t) = X64V3Token::summon() {
            let vals = [1.5, 3.0, 10.0, 100.0];
            let input = f32x4::from_array(t, vals);
            let result = input.log2_midp_unchecked().to_array();
            let expected: [f32; 4] = core::array::from_fn(|i| vals[i].log2());
            assert_close_4(result, expected, 5e-6, "f32x4 log2_midp");
        }
    }

    #[test]
    fn f32x4_exp2_midp() {
        if let Some(t) = X64V3Token::summon() {
            let vals = [0.5, 1.5, -0.5, 2.5];
            let input = f32x4::from_array(t, vals);
            let result = input.exp2_midp().to_array();
            let expected: [f32; 4] = core::array::from_fn(|i| vals[i].exp2());
            assert_close_4(result, expected, 5e-6, "f32x4 exp2_midp");
        }
    }

    #[test]
    fn f32x4_ln_midp() {
        if let Some(t) = X64V3Token::summon() {
            let vals = [1.0, core::f32::consts::E, 10.0, 0.5];
            let input = f32x4::from_array(t, vals);
            let result = input.ln_midp().to_array();
            let expected: [f32; 4] = core::array::from_fn(|i| vals[i].ln());
            assert_close_4(result, expected, 5e-6, "f32x4 ln_midp");
        }
    }

    #[test]
    fn f32x4_cbrt_midp() {
        if let Some(t) = X64V3Token::summon() {
            let input = f32x4::from_array(t, [1.0, 8.0, 27.0, 64.0]);
            let result = input.cbrt_midp().to_array();
            let expected = [1.0, 2.0, 3.0, 4.0];
            assert_close_4(result, expected, 1e-5, "f32x4 cbrt_midp");
        }
    }

    #[test]
    fn f32x4_cbrt_midp_negative() {
        if let Some(t) = X64V3Token::summon() {
            let input = f32x4::from_array(t, [-1.0, -8.0, -27.0, -64.0]);
            let result = input.cbrt_midp().to_array();
            let expected = [-1.0, -2.0, -3.0, -4.0];
            assert_close_4(result, expected, 1e-5, "f32x4 cbrt_midp negative");
        }
    }

    #[test]
    fn f32x4_log2_midp_edge_cases() {
        if let Some(t) = X64V3Token::summon() {
            let input = f32x4::from_array(t, [0.0, -1.0, 1.0, 2.0]);
            let result = input.log2_midp().to_array();
            assert!(result[0].is_infinite() && result[0].is_sign_negative());
            assert!(result[1].is_nan());
            assert!((result[2] - 0.0).abs() < 1e-5);
            assert!((result[3] - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn f32x4_pow_midp() {
        if let Some(t) = X64V3Token::summon() {
            let input = f32x4::from_array(t, [4.0, 9.0, 16.0, 25.0]);
            let result = input.pow_midp(0.5).to_array();
            let expected = [2.0, 3.0, 4.0, 5.0];
            assert_close_4(result, expected, 5e-6, "f32x4 pow_midp(0.5)");
        }
    }

    // ====== Scalar Backend Tests ======

    #[test]
    fn f32x8_scalar_log2_lowp() {
        let t = ScalarToken;
        let input = f32x8::from_array(t, [1.0, 2.0, 4.0, 8.0, 0.5, 0.25, 16.0, 64.0]);
        let result = input.log2_lowp().to_array();
        let expected = [0.0, 1.0, 2.0, 3.0, -1.0, -2.0, 4.0, 6.0];
        assert_close_8(result, expected, 0.02, "scalar log2_lowp");
    }

    #[test]
    fn f32x8_scalar_exp2_lowp() {
        let t = ScalarToken;
        let input = f32x8::from_array(t, [0.0, 1.0, 2.0, 3.0, -1.0, -2.0, 4.0, 7.0]);
        let result = input.exp2_lowp().to_array();
        let expected = [1.0, 2.0, 4.0, 8.0, 0.5, 0.25, 16.0, 128.0];
        assert_close_8(result, expected, 0.02, "scalar exp2_lowp");
    }

    #[test]
    fn f32x8_scalar_log2_midp() {
        let t = ScalarToken;
        let vals = [1.5, 3.0, 10.0, 100.0, 0.1, 0.01, 7.0, 42.0];
        let input = f32x8::from_array(t, vals);
        let result = input.log2_midp_unchecked().to_array();
        let expected: [f32; 8] = core::array::from_fn(|i| vals[i].log2());
        assert_close_8(result, expected, 5e-6, "scalar log2_midp");
    }

    #[test]
    fn f32x8_scalar_exp2_midp() {
        let t = ScalarToken;
        let vals = [0.5, 1.5, -0.5, 2.5, 0.1, 0.9, -3.5, 10.0];
        let input = f32x8::from_array(t, vals);
        let result = input.exp2_midp().to_array();
        let expected: [f32; 8] = core::array::from_fn(|i| vals[i].exp2());
        assert_close_8(result, expected, 5e-6, "scalar exp2_midp");
    }

    #[test]
    fn f32x8_scalar_cbrt_midp() {
        let t = ScalarToken;
        let input = f32x8::from_array(t, [1.0, 8.0, 27.0, 64.0, 125.0, 216.0, 343.0, 512.0]);
        let result = input.cbrt_midp().to_array();
        let expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert_close_8(result, expected, 1e-5, "scalar cbrt_midp");
    }

    #[test]
    fn f32x4_scalar_log2_midp() {
        let t = ScalarToken;
        let vals = [1.5, 3.0, 10.0, 100.0];
        let input = f32x4::from_array(t, vals);
        let result = input.log2_midp_unchecked().to_array();
        let expected: [f32; 4] = core::array::from_fn(|i| vals[i].log2());
        assert_close_4(result, expected, 5e-6, "scalar f32x4 log2_midp");
    }

    #[test]
    fn f32x4_scalar_cbrt_midp() {
        let t = ScalarToken;
        let input = f32x4::from_array(t, [1.0, 8.0, 27.0, -8.0]);
        let result = input.cbrt_midp().to_array();
        assert!((result[0] - 1.0).abs() < 1e-5);
        assert!((result[1] - 2.0).abs() < 1e-5);
        assert!((result[2] - 3.0).abs() < 1e-5);
        assert!((result[3] - (-2.0)).abs() < 1e-5);
    }

    // ====== Cross-Backend Consistency ======

    #[test]
    fn f32x8_x86_scalar_log2_agree() {
        if let Some(x86_t) = X64V3Token::summon() {
            let scalar_t = ScalarToken;
            let vals = [1.5, 3.0, 10.0, 100.0, 0.1, 0.01, 7.0, 42.0];

            let x86_result = f32x8::from_array(x86_t, vals)
                .log2_midp_unchecked()
                .to_array();
            let scalar_result = f32x8::from_array(scalar_t, vals)
                .log2_midp_unchecked()
                .to_array();

            // Both should be close to std::f32::log2 (may differ by a few ULP due to FMA)
            let expected: [f32; 8] = core::array::from_fn(|i| vals[i].log2());
            assert_close_8(x86_result, expected, 5e-6, "x86 log2_midp");
            assert_close_8(scalar_result, expected, 5e-6, "scalar log2_midp");
        }
    }

    #[test]
    fn f32x8_x86_scalar_exp2_agree() {
        if let Some(x86_t) = X64V3Token::summon() {
            let scalar_t = ScalarToken;
            let vals = [0.5, 1.5, -0.5, 2.5, 0.1, 0.9, -3.5, 10.0];

            let x86_result = f32x8::from_array(x86_t, vals).exp2_midp().to_array();
            let scalar_result = f32x8::from_array(scalar_t, vals).exp2_midp().to_array();

            let expected: [f32; 8] = core::array::from_fn(|i| vals[i].exp2());
            assert_close_8(x86_result, expected, 5e-6, "x86 exp2_midp");
            assert_close_8(scalar_result, expected, 5e-6, "scalar exp2_midp");
        }
    }

    // ====== Generic Function Test ======

    /// Verify transcendentals work in a generic context.
    fn gamma_correct<T: magetypes::simd::backends::F32x8Convert>(
        token: T,
        linear: &[f32; 8],
        gamma: f32,
    ) -> [f32; 8] {
        f32x8::<T>::from_array(token, *linear)
            .pow_midp(1.0 / gamma)
            .to_array()
    }

    #[test]
    fn generic_gamma_correction() {
        if let Some(t) = X64V3Token::summon() {
            let linear = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0];
            let result = gamma_correct(t, &linear, 2.2);

            // pow(x, 1/2.2) should be in [0, 1] for inputs in [0, 1]
            for (i, &v) in result.iter().enumerate() {
                if linear[i] == 0.0 {
                    // pow(0, x) via log2->exp2 is undefined, skip
                    continue;
                }
                assert!(v >= 0.0 && v <= 1.0001, "gamma lane {i}: {v} out of range");
                let expected = linear[i].powf(1.0 / 2.2);
                assert!(
                    ((v - expected) / expected).abs() < 5e-5,
                    "gamma lane {i}: got {v}, expected {expected}"
                );
            }
        }
    }

    // ====== Large Range Test ======

    #[test]
    fn f32x8_exp2_midp_large_range() {
        if let Some(t) = X64V3Token::summon() {
            // Test across the representable range
            let vals = [-100.0, -50.0, -10.0, -1.0, 1.0, 10.0, 50.0, 100.0];
            let input = f32x8::from_array(t, vals);
            let result = input.exp2_midp().to_array();
            let expected: [f32; 8] = core::array::from_fn(|i| vals[i].exp2());
            assert_close_8(result, expected, 5e-5, "exp2_midp large range");
        }
    }

    #[test]
    fn f32x8_log2_midp_large_range() {
        if let Some(t) = X64V3Token::summon() {
            let vals = [1e-30, 1e-10, 0.001, 1.0, 1000.0, 1e10, 1e30, 1e38];
            let input = f32x8::from_array(t, vals);
            let result = input.log2_midp_unchecked().to_array();
            let expected: [f32; 8] = core::array::from_fn(|i| vals[i].log2());
            assert_close_8(result, expected, 5e-5, "log2_midp large range");
        }
    }

    // ====== Parity with Old Types ======

    /// Compare generic f32x8<X64V3Token> transcendentals against old per-platform types.
    #[test]
    fn f32x8_generic_vs_old_log2_lowp() {
        use magetypes::simd::f32x8 as OldF32x8;

        if let Some(t) = X64V3Token::summon() {
            let vals = [1.5, 3.0, 10.0, 100.0, 0.1, 0.01, 7.0, 42.0];

            let old_result = OldF32x8::from_array(t, vals).log2_lowp().to_array();
            let new_result = f32x8::<X64V3Token>::from_array(t, vals)
                .log2_lowp()
                .to_array();

            // Should be bit-identical since they use the same intrinsics
            for i in 0..8 {
                assert!(
                    (old_result[i] - new_result[i]).abs() < 1e-7,
                    "log2_lowp parity lane {i}: old={}, new={}",
                    old_result[i],
                    new_result[i]
                );
            }
        }
    }

    #[test]
    fn f32x8_generic_vs_old_exp2_lowp() {
        use magetypes::simd::f32x8 as OldF32x8;

        if let Some(t) = X64V3Token::summon() {
            let vals = [0.5, 1.5, -0.5, 2.5, 0.1, 0.9, -3.5, 10.0];

            let old_result = OldF32x8::from_array(t, vals).exp2_lowp().to_array();
            let new_result = f32x8::<X64V3Token>::from_array(t, vals)
                .exp2_lowp()
                .to_array();

            for i in 0..8 {
                assert!(
                    (old_result[i] - new_result[i]).abs() < 1e-5,
                    "exp2_lowp parity lane {i}: old={}, new={}",
                    old_result[i],
                    new_result[i]
                );
            }
        }
    }

    #[test]
    fn f32x8_generic_vs_old_log2_midp() {
        use magetypes::simd::f32x8 as OldF32x8;

        if let Some(t) = X64V3Token::summon() {
            let vals = [1.5, 3.0, 10.0, 100.0, 0.1, 0.01, 7.0, 42.0];

            let old_result = OldF32x8::from_array(t, vals)
                .log2_midp_unchecked()
                .to_array();
            let new_result = f32x8::<X64V3Token>::from_array(t, vals)
                .log2_midp_unchecked()
                .to_array();

            for i in 0..8 {
                assert!(
                    (old_result[i] - new_result[i]).abs() < 1e-7,
                    "log2_midp parity lane {i}: old={}, new={}",
                    old_result[i],
                    new_result[i]
                );
            }
        }
    }

    #[test]
    fn f32x8_generic_vs_old_cbrt_midp() {
        use magetypes::simd::f32x8 as OldF32x8;

        if let Some(t) = X64V3Token::summon() {
            let vals = [1.0, 8.0, 27.0, 64.0, 0.001, 1000.0, 0.5, 100.0];

            let old_result = OldF32x8::from_array(t, vals).cbrt_midp().to_array();
            let new_result = f32x8::<X64V3Token>::from_array(t, vals)
                .cbrt_midp()
                .to_array();

            for i in 0..8 {
                assert!(
                    (old_result[i] - new_result[i]).abs() < 1e-6,
                    "cbrt_midp parity lane {i}: old={}, new={}",
                    old_result[i],
                    new_result[i]
                );
            }
        }
    }

    // ====== f32x16 Low-Precision Tests (X64V3Token — 2×256-bit polyfill) ======

    const F16_VALS: [f32; 16] = [
        1.0, 2.0, 4.0, 8.0, 0.5, 0.25, 16.0, 64.0, 1.5, 3.0, 10.0, 100.0, 0.1, 0.01, 7.0, 42.0,
    ];

    #[test]
    fn f32x16_log2_lowp() {
        if let Some(t) = X64V3Token::summon() {
            let input = f32x16::from_array(t, F16_VALS);
            let result = input.log2_lowp().to_array();
            let expected: [f32; 16] = core::array::from_fn(|i| F16_VALS[i].log2());
            assert_close_16(result, expected, 0.02, "f32x16 log2_lowp");
        }
    }

    #[test]
    fn f32x16_exp2_lowp() {
        if let Some(t) = X64V3Token::summon() {
            let vals: [f32; 16] = [
                0.0, 1.0, 2.0, 3.0, -1.0, -2.0, 4.0, 7.0, 0.5, 1.5, -0.5, 2.5, 0.1, 0.9, -3.5, 10.0,
            ];
            let input = f32x16::from_array(t, vals);
            let result = input.exp2_lowp().to_array();
            let expected: [f32; 16] = core::array::from_fn(|i| vals[i].exp2());
            assert_close_16(result, expected, 0.02, "f32x16 exp2_lowp");
        }
    }

    #[test]
    fn f32x16_ln_lowp() {
        if let Some(t) = X64V3Token::summon() {
            let input = f32x16::from_array(t, F16_VALS);
            let result = input.ln_lowp().to_array();
            let expected: [f32; 16] = core::array::from_fn(|i| F16_VALS[i].ln());
            assert_close_16(result, expected, 0.02, "f32x16 ln_lowp");
        }
    }

    #[test]
    fn f32x16_exp_lowp() {
        if let Some(t) = X64V3Token::summon() {
            let vals: [f32; 16] = [
                0.0, 1.0, -1.0, 2.0, 0.5, -2.0, 3.0, -0.5, -3.0, 0.1, -0.1, 1.5, -1.5, 0.25, -0.25,
                4.0,
            ];
            let input = f32x16::from_array(t, vals);
            let result = input.exp_lowp().to_array();
            let expected: [f32; 16] = core::array::from_fn(|i| vals[i].exp());
            assert_close_16(result, expected, 0.02, "f32x16 exp_lowp");
        }
    }

    #[test]
    fn f32x16_log10_lowp() {
        if let Some(t) = X64V3Token::summon() {
            let input = f32x16::from_array(t, F16_VALS);
            let result = input.log10_lowp().to_array();
            let expected: [f32; 16] = core::array::from_fn(|i| F16_VALS[i].log10());
            assert_close_16(result, expected, 0.02, "f32x16 log10_lowp");
        }
    }

    #[test]
    fn f32x16_pow_lowp() {
        if let Some(t) = X64V3Token::summon() {
            let input = f32x16::from_array(t, F16_VALS);
            let result = input.pow_lowp(0.5).to_array();
            let expected: [f32; 16] = core::array::from_fn(|i| F16_VALS[i].sqrt());
            assert_close_16(result, expected, 0.02, "f32x16 pow_lowp(0.5)");
        }
    }

    // ====== f32x16 Mid-Precision Tests (X64V3Token) ======

    #[test]
    fn f32x16_log2_midp() {
        if let Some(t) = X64V3Token::summon() {
            let input = f32x16::from_array(t, F16_VALS);
            let result = input.log2_midp_unchecked().to_array();
            let expected: [f32; 16] = core::array::from_fn(|i| F16_VALS[i].log2());
            assert_close_16(result, expected, 5e-6, "f32x16 log2_midp");
        }
    }

    #[test]
    fn f32x16_log2_midp_edge_cases() {
        if let Some(t) = X64V3Token::summon() {
            let vals: [f32; 16] = [
                0.0, -1.0, 1.0, 2.0, 0.0, -0.5, 4.0, 8.0, 0.0, -2.0, 16.0, 32.0, 0.5, -10.0, 0.25,
                64.0,
            ];
            let input = f32x16::from_array(t, vals);
            let result = input.log2_midp().to_array();

            // 0 -> -inf
            for &i in &[0, 4, 8] {
                assert!(
                    result[i].is_infinite() && result[i].is_sign_negative(),
                    "log2(0) at lane {i}"
                );
            }
            // negative -> NaN
            for &i in &[1, 5, 9, 13] {
                assert!(result[i].is_nan(), "log2(negative) at lane {i}");
            }
            // normal values
            assert!((result[2] - 0.0).abs() < 1e-5, "log2(1) ~ 0");
            assert!((result[3] - 1.0).abs() < 1e-5, "log2(2) ~ 1");
            assert!((result[6] - 2.0).abs() < 1e-5, "log2(4) ~ 2");
        }
    }

    #[test]
    fn f32x16_exp2_midp() {
        if let Some(t) = X64V3Token::summon() {
            let vals: [f32; 16] = [
                0.0, 1.0, 2.0, 3.0, -1.0, -2.0, 4.0, 7.0, 0.5, 1.5, -0.5, 2.5, 0.1, 0.9, -3.5, 10.0,
            ];
            let input = f32x16::from_array(t, vals);
            let result = input.exp2_midp().to_array();
            let expected: [f32; 16] = core::array::from_fn(|i| vals[i].exp2());
            assert_close_16(result, expected, 5e-6, "f32x16 exp2_midp");
        }
    }

    #[test]
    fn f32x16_ln_midp() {
        if let Some(t) = X64V3Token::summon() {
            let input = f32x16::from_array(t, F16_VALS);
            let result = input.ln_midp().to_array();
            let expected: [f32; 16] = core::array::from_fn(|i| F16_VALS[i].ln());
            assert_close_16(result, expected, 5e-6, "f32x16 ln_midp");
        }
    }

    #[test]
    fn f32x16_exp_midp() {
        if let Some(t) = X64V3Token::summon() {
            let vals: [f32; 16] = [
                0.0, 1.0, -1.0, 2.0, 0.5, -2.0, 3.0, -0.5, -3.0, 0.1, -0.1, 1.5, -1.5, 0.25, -0.25,
                4.0,
            ];
            let input = f32x16::from_array(t, vals);
            let result = input.exp_midp().to_array();
            let expected: [f32; 16] = core::array::from_fn(|i| vals[i].exp());
            assert_close_16(result, expected, 5e-6, "f32x16 exp_midp");
        }
    }

    #[test]
    fn f32x16_log10_midp() {
        if let Some(t) = X64V3Token::summon() {
            let input = f32x16::from_array(t, F16_VALS);
            let result = input.log10_midp().to_array();
            let expected: [f32; 16] = core::array::from_fn(|i| F16_VALS[i].log10());
            assert_close_16(result, expected, 5e-6, "f32x16 log10_midp");
        }
    }

    #[test]
    fn f32x16_pow_midp() {
        if let Some(t) = X64V3Token::summon() {
            let vals: [f32; 16] = [
                1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0, 121.0, 144.0, 169.0,
                196.0, 225.0, 256.0,
            ];
            let input = f32x16::from_array(t, vals);
            let result = input.pow_midp(0.5).to_array();
            let expected: [f32; 16] = core::array::from_fn(|i| vals[i].sqrt());
            assert_close_16(result, expected, 5e-6, "f32x16 pow_midp(0.5)");
        }
    }

    #[test]
    fn f32x16_pow_midp_cube() {
        if let Some(t) = X64V3Token::summon() {
            let vals: [f32; 16] = [
                1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 0.5, 0.1, 0.2, 0.3, 6.0, 7.0, 8.0, 9.0, 0.25, 20.0,
            ];
            let input = f32x16::from_array(t, vals);
            let result = input.pow_midp(3.0).to_array();
            let expected: [f32; 16] = core::array::from_fn(|i| vals[i].powi(3));
            assert_close_16(result, expected, 5e-5, "f32x16 pow_midp(3.0)");
        }
    }

    // ====== f32x16 cbrt Tests ======

    #[test]
    fn f32x16_cbrt_midp() {
        if let Some(t) = X64V3Token::summon() {
            let vals: [f32; 16] = [
                1.0, 8.0, 27.0, 64.0, 125.0, 216.0, 343.0, 512.0, 0.001, 0.1, 0.5, 2.0, 10.0,
                100.0, 1000.0, 1e6,
            ];
            let input = f32x16::from_array(t, vals);
            let result = input.cbrt_midp().to_array();
            let expected: [f32; 16] = core::array::from_fn(|i| vals[i].cbrt());
            assert_close_16(result, expected, 1e-5, "f32x16 cbrt_midp");
        }
    }

    #[test]
    fn f32x16_cbrt_midp_negative() {
        if let Some(t) = X64V3Token::summon() {
            let vals: [f32; 16] = [
                -1.0, -8.0, -27.0, -64.0, -0.001, -1000.0, -0.125, -1e6, -125.0, -216.0, -343.0,
                -512.0, -0.5, -2.0, -10.0, -100.0,
            ];
            let input = f32x16::from_array(t, vals);
            let result = input.cbrt_midp().to_array();
            let expected: [f32; 16] = core::array::from_fn(|i| -vals[i].abs().cbrt());
            assert_close_16(result, expected, 1e-5, "f32x16 cbrt_midp negative");
        }
    }

    #[test]
    fn f32x16_cbrt_midp_precise_zero() {
        if let Some(t) = X64V3Token::summon() {
            let vals: [f32; 16] = [
                0.0, 0.0, 1.0, 8.0, -8.0, 0.0, 27.0, 0.0, 64.0, 0.0, -27.0, 125.0, 0.0, 216.0, 0.0,
                -1.0,
            ];
            let input = f32x16::from_array(t, vals);
            let result = input.cbrt_midp_precise().to_array();
            for &i in &[0, 1, 5, 7, 9, 12, 14] {
                assert_eq!(result[i], 0.0, "cbrt_precise(0) = 0 at lane {i}");
            }
            assert!((result[2] - 1.0).abs() < 1e-5, "cbrt_precise(1) ~ 1");
            assert!((result[3] - 2.0).abs() < 1e-5, "cbrt_precise(8) ~ 2");
        }
    }

    // ====== f32x16 Roundtrip Tests ======

    #[test]
    fn f32x16_log2_exp2_roundtrip() {
        if let Some(t) = X64V3Token::summon() {
            let input = f32x16::from_array(t, F16_VALS);
            let roundtrip = input.log2_midp_unchecked().exp2_midp().to_array();
            assert_close_16(roundtrip, F16_VALS, 1e-4, "f32x16 log2 -> exp2 roundtrip");
        }
    }

    #[test]
    fn f32x16_ln_exp_roundtrip() {
        if let Some(t) = X64V3Token::summon() {
            let input = f32x16::from_array(t, F16_VALS);
            let roundtrip = input.ln_midp_unchecked().exp_midp().to_array();
            assert_close_16(roundtrip, F16_VALS, 1e-4, "f32x16 ln -> exp roundtrip");
        }
    }

    // ====== f32x16 _unchecked Variants ======

    #[test]
    fn f32x16_unchecked_variants_match() {
        if let Some(t) = X64V3Token::summon() {
            let input = f32x16::from_array(t, F16_VALS);

            let log2_normal = input.log2_lowp().to_array();
            let log2_unchecked = input.log2_lowp_unchecked().to_array();
            assert_eq!(
                log2_normal, log2_unchecked,
                "f32x16 log2_lowp == log2_lowp_unchecked on valid input"
            );

            let pow_normal = input.pow_midp(2.0).to_array();
            let pow_precise = input.pow_midp_precise(2.0).to_array();
            assert_eq!(
                pow_normal, pow_precise,
                "f32x16 pow_midp == pow_midp_precise"
            );
        }
    }

    // ====== f32x16 Scalar Backend Tests ======

    #[test]
    fn f32x16_scalar_log2_midp() {
        let t = ScalarToken;
        let input = f32x16::from_array(t, F16_VALS);
        let result = input.log2_midp_unchecked().to_array();
        let expected: [f32; 16] = core::array::from_fn(|i| F16_VALS[i].log2());
        assert_close_16(result, expected, 5e-6, "scalar f32x16 log2_midp");
    }

    #[test]
    fn f32x16_scalar_exp2_midp() {
        let t = ScalarToken;
        let vals: [f32; 16] = [
            0.5, 1.5, -0.5, 2.5, 0.1, 0.9, -3.5, 10.0, 0.0, 1.0, 2.0, 3.0, -1.0, -2.0, 4.0, 7.0,
        ];
        let input = f32x16::from_array(t, vals);
        let result = input.exp2_midp().to_array();
        let expected: [f32; 16] = core::array::from_fn(|i| vals[i].exp2());
        assert_close_16(result, expected, 5e-6, "scalar f32x16 exp2_midp");
    }

    #[test]
    fn f32x16_scalar_pow_midp() {
        let t = ScalarToken;
        let vals: [f32; 16] = [
            1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0, 121.0, 144.0, 169.0, 196.0,
            225.0, 256.0,
        ];
        let input = f32x16::from_array(t, vals);
        let result = input.pow_midp(0.5).to_array();
        let expected: [f32; 16] = core::array::from_fn(|i| vals[i].sqrt());
        assert_close_16(result, expected, 5e-6, "scalar f32x16 pow_midp(0.5)");
    }

    #[test]
    fn f32x16_scalar_cbrt_midp() {
        let t = ScalarToken;
        let vals: [f32; 16] = [
            1.0, 8.0, 27.0, 64.0, 125.0, 216.0, 343.0, 512.0, 0.001, 0.1, 0.5, 2.0, 10.0, 100.0,
            1000.0, 1e6,
        ];
        let input = f32x16::from_array(t, vals);
        let result = input.cbrt_midp().to_array();
        let expected: [f32; 16] = core::array::from_fn(|i| vals[i].cbrt());
        assert_close_16(result, expected, 1e-5, "scalar f32x16 cbrt_midp");
    }

    // ====== f32x16 Cross-Backend Consistency ======

    #[test]
    fn f32x16_x86_scalar_log2_agree() {
        if let Some(x86_t) = X64V3Token::summon() {
            let scalar_t = ScalarToken;

            let x86_result = f32x16::from_array(x86_t, F16_VALS)
                .log2_midp_unchecked()
                .to_array();
            let scalar_result = f32x16::from_array(scalar_t, F16_VALS)
                .log2_midp_unchecked()
                .to_array();

            let expected: [f32; 16] = core::array::from_fn(|i| F16_VALS[i].log2());
            assert_close_16(x86_result, expected, 5e-6, "x86 f32x16 log2_midp");
            assert_close_16(scalar_result, expected, 5e-6, "scalar f32x16 log2_midp");
        }
    }

    #[test]
    fn f32x16_x86_scalar_exp2_agree() {
        if let Some(x86_t) = X64V3Token::summon() {
            let scalar_t = ScalarToken;
            let vals: [f32; 16] = [
                0.5, 1.5, -0.5, 2.5, 0.1, 0.9, -3.5, 10.0, 0.0, 1.0, 2.0, 3.0, -1.0, -2.0, 4.0, 7.0,
            ];

            let x86_result = f32x16::from_array(x86_t, vals).exp2_midp().to_array();
            let scalar_result = f32x16::from_array(scalar_t, vals).exp2_midp().to_array();

            let expected: [f32; 16] = core::array::from_fn(|i| vals[i].exp2());
            assert_close_16(x86_result, expected, 5e-6, "x86 f32x16 exp2_midp");
            assert_close_16(scalar_result, expected, 5e-6, "scalar f32x16 exp2_midp");
        }
    }

    #[test]
    fn f32x16_x86_scalar_pow_agree() {
        if let Some(x86_t) = X64V3Token::summon() {
            let scalar_t = ScalarToken;

            let x86_result = f32x16::from_array(x86_t, F16_VALS).pow_midp(0.5).to_array();
            let scalar_result = f32x16::from_array(scalar_t, F16_VALS)
                .pow_midp(0.5)
                .to_array();

            let expected: [f32; 16] = core::array::from_fn(|i| F16_VALS[i].sqrt());
            assert_close_16(x86_result, expected, 5e-6, "x86 f32x16 pow_midp(0.5)");
            assert_close_16(scalar_result, expected, 5e-6, "scalar f32x16 pow_midp(0.5)");
        }
    }

    // ====== f32x16 Generic Function Test ======

    fn gamma_correct_16<T: magetypes::simd::backends::F32x16Convert>(
        token: T,
        linear: &[f32; 16],
        gamma: f32,
    ) -> [f32; 16] {
        f32x16::<T>::from_array(token, *linear)
            .pow_midp(1.0 / gamma)
            .to_array()
    }

    #[test]
    fn f32x16_generic_gamma_correction() {
        if let Some(t) = X64V3Token::summon() {
            let linear: [f32; 16] = [
                0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9,
                1.0,
            ];
            let result = gamma_correct_16(t, &linear, 2.2);

            for (i, &v) in result.iter().enumerate() {
                assert!(v >= 0.0 && v <= 1.0001, "gamma lane {i}: {v} out of range");
                let expected = linear[i].powf(1.0 / 2.2);
                assert!(
                    ((v - expected) / expected).abs() < 5e-5,
                    "gamma lane {i}: got {v}, expected {expected}"
                );
            }
        }
    }

    // ====== f32x16 Large Range Test ======

    #[test]
    fn f32x16_exp2_midp_large_range() {
        if let Some(t) = X64V3Token::summon() {
            let vals: [f32; 16] = [
                -100.0, -50.0, -10.0, -1.0, 1.0, 10.0, 50.0, 100.0, -80.0, -20.0, -5.0, 0.0, 5.0,
                20.0, 80.0, 120.0,
            ];
            let input = f32x16::from_array(t, vals);
            let result = input.exp2_midp().to_array();
            let expected: [f32; 16] = core::array::from_fn(|i| vals[i].exp2());
            assert_close_16(result, expected, 5e-5, "f32x16 exp2_midp large range");
        }
    }

    #[test]
    fn f32x16_log2_midp_large_range() {
        if let Some(t) = X64V3Token::summon() {
            let vals: [f32; 16] = [
                1e-30, 1e-20, 1e-10, 0.001, 0.1, 1.0, 10.0, 1000.0, 1e6, 1e10, 1e15, 1e20, 1e25,
                1e30, 1e35, 1e38,
            ];
            let input = f32x16::from_array(t, vals);
            let result = input.log2_midp_unchecked().to_array();
            let expected: [f32; 16] = core::array::from_fn(|i| vals[i].log2());
            assert_close_16(result, expected, 5e-5, "f32x16 log2_midp large range");
        }
    }

    // ====== f32x16 Conversion Tests (F32x16Convert) ======

    #[test]
    fn f32x16_bitcast_roundtrip() {
        if let Some(t) = X64V3Token::summon() {
            let input = f32x16::from_array(t, F16_VALS);
            let as_int = input.bitcast_to_i32();
            let back = f32x16::from_i32_bitcast(t, as_int);
            assert_eq!(back.to_array(), F16_VALS, "bitcast roundtrip");
        }
    }

    #[test]
    fn f32x16_convert_roundtrip() {
        if let Some(t) = X64V3Token::summon() {
            // Use integer-valued floats so truncation is lossless
            let vals: [f32; 16] = [
                0.0, 1.0, -1.0, 2.0, -2.0, 100.0, -100.0, 42.0, 7.0, -7.0, 255.0, -128.0, 50.0,
                -50.0, 1000.0, -1000.0,
            ];
            let input = f32x16::from_array(t, vals);
            let as_int = input.to_i32();
            let back = f32x16::from_i32(t, as_int);
            assert_eq!(back.to_array(), vals, "convert roundtrip");
        }
    }

    #[test]
    fn f32x16_to_i32_round() {
        if let Some(t) = X64V3Token::summon() {
            let vals: [f32; 16] = [
                0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, -3.5, 0.4, 0.6, 1.4, 1.6, -0.4, -0.6, -1.4,
                -1.6,
            ];
            let input = f32x16::from_array(t, vals);
            let result = input.to_i32_round();
            let arr = result.to_array();
            // Banker's rounding (round to even)
            assert_eq!(arr[0], 0, "round(0.5) = 0 (banker's)");
            assert_eq!(arr[1], 2, "round(1.5) = 2 (banker's)");
            assert_eq!(arr[8], 0, "round(0.4) = 0");
            assert_eq!(arr[9], 1, "round(0.6) = 1");
        }
    }

    #[test]
    fn f32x16_scalar_bitcast_roundtrip() {
        let t = ScalarToken;
        let input = f32x16::from_array(t, F16_VALS);
        let as_int = input.bitcast_to_i32();
        let back = f32x16::from_i32_bitcast(t, as_int);
        assert_eq!(back.to_array(), F16_VALS, "scalar bitcast roundtrip");
    }
}

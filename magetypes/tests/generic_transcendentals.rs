//! Tests for generic transcendental math functions on f32x4<T> and f32x8<T>.
//!
//! Tests against std::f32 functions and known exact values.
//! Verifies both X64V3Token (AVX2+FMA) and ScalarToken backends.

#[cfg(target_arch = "x86_64")]
mod tests {
    use archmage::{ScalarToken, SimdToken, X64V3Token};
    use magetypes::simd::generic::{f32x4, f32x8};

    fn get_x64v3() -> X64V3Token {
        X64V3Token::summon().expect("test requires AVX2+FMA")
    }

    // ====== Helpers ======

    /// Check that all lanes are within `tol` of expected (relative error for large values,
    /// absolute for small).
    fn assert_close_8(actual: [f32; 8], expected: [f32; 8], tol: f32, msg: &str) {
        for i in 0..8 {
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

    fn assert_close_4(actual: [f32; 4], expected: [f32; 4], tol: f32, msg: &str) {
        for i in 0..4 {
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

    // ====== f32x8 Low-Precision Tests (X64V3Token) ======

    #[test]
    fn f32x8_log2_lowp_exact_powers() {
        let t = get_x64v3();
        let input = f32x8::from_array(t, [1.0, 2.0, 4.0, 8.0, 0.5, 0.25, 16.0, 64.0]);
        let result = input.log2_lowp().to_array();
        let expected = [0.0, 1.0, 2.0, 3.0, -1.0, -2.0, 4.0, 6.0];
        assert_close_8(result, expected, 0.02, "log2_lowp powers of 2");
    }

    #[test]
    fn f32x8_log2_lowp_general() {
        let t = get_x64v3();
        let input = f32x8::from_array(t, [1.5, 3.0, 10.0, 100.0, 0.1, 0.01, 7.0, 42.0]);
        let result = input.log2_lowp().to_array();
        let expected: [f32; 8] = core::array::from_fn(|i| {
            [1.5_f32, 3.0, 10.0, 100.0, 0.1, 0.01, 7.0, 42.0][i].log2()
        });
        assert_close_8(result, expected, 0.02, "log2_lowp general");
    }

    #[test]
    fn f32x8_exp2_lowp_exact() {
        let t = get_x64v3();
        let input = f32x8::from_array(t, [0.0, 1.0, 2.0, 3.0, -1.0, -2.0, 4.0, 7.0]);
        let result = input.exp2_lowp().to_array();
        let expected = [1.0, 2.0, 4.0, 8.0, 0.5, 0.25, 16.0, 128.0];
        assert_close_8(result, expected, 0.02, "exp2_lowp exact");
    }

    #[test]
    fn f32x8_exp2_lowp_fractional() {
        let t = get_x64v3();
        let input = f32x8::from_array(t, [0.5, 1.5, -0.5, 2.5, 0.1, 0.9, -3.5, 10.0]);
        let result = input.exp2_lowp().to_array();
        let expected: [f32; 8] = core::array::from_fn(|i| {
            [0.5_f32, 1.5, -0.5, 2.5, 0.1, 0.9, -3.5, 10.0][i].exp2()
        });
        assert_close_8(result, expected, 0.02, "exp2_lowp fractional");
    }

    #[test]
    fn f32x8_ln_lowp() {
        let t = get_x64v3();
        let vals = [1.0, 2.718_281_8, 10.0, 0.5, 100.0, 0.1, 7.389, 20.0];
        let input = f32x8::from_array(t, vals);
        let result = input.ln_lowp().to_array();
        let expected: [f32; 8] = core::array::from_fn(|i| vals[i].ln());
        assert_close_8(result, expected, 0.02, "ln_lowp");
    }

    #[test]
    fn f32x8_exp_lowp() {
        let t = get_x64v3();
        let vals = [0.0, 1.0, -1.0, 2.0, 0.5, -2.0, 3.0, -0.5];
        let input = f32x8::from_array(t, vals);
        let result = input.exp_lowp().to_array();
        let expected: [f32; 8] = core::array::from_fn(|i| vals[i].exp());
        assert_close_8(result, expected, 0.02, "exp_lowp");
    }

    #[test]
    fn f32x8_log10_lowp() {
        let t = get_x64v3();
        let vals = [1.0, 10.0, 100.0, 1000.0, 0.1, 0.01, 50.0, 7.0];
        let input = f32x8::from_array(t, vals);
        let result = input.log10_lowp().to_array();
        let expected: [f32; 8] = core::array::from_fn(|i| vals[i].log10());
        assert_close_8(result, expected, 0.02, "log10_lowp");
    }

    #[test]
    fn f32x8_pow_lowp() {
        let t = get_x64v3();
        let vals = [1.0, 2.0, 4.0, 8.0, 3.0, 10.0, 0.5, 100.0];
        let input = f32x8::from_array(t, vals);
        let result = input.pow_lowp(0.5).to_array();
        let expected: [f32; 8] = core::array::from_fn(|i| vals[i].sqrt());
        assert_close_8(result, expected, 0.02, "pow_lowp(0.5) ≈ sqrt");
    }

    // ====== f32x8 Mid-Precision Tests (X64V3Token) ======

    #[test]
    fn f32x8_log2_midp_exact_powers() {
        let t = get_x64v3();
        let input = f32x8::from_array(t, [1.0, 2.0, 4.0, 8.0, 0.5, 0.25, 16.0, 64.0]);
        let result = input.log2_midp_unchecked().to_array();
        let expected = [0.0, 1.0, 2.0, 3.0, -1.0, -2.0, 4.0, 6.0];
        assert_close_8(result, expected, 1e-5, "log2_midp powers of 2");
    }

    #[test]
    fn f32x8_log2_midp_general() {
        let t = get_x64v3();
        let vals = [1.5, 3.0, 10.0, 100.0, 0.1, 0.01, 7.0, 42.0];
        let input = f32x8::from_array(t, vals);
        let result = input.log2_midp_unchecked().to_array();
        let expected: [f32; 8] = core::array::from_fn(|i| vals[i].log2());
        assert_close_8(result, expected, 5e-6, "log2_midp general");
    }

    #[test]
    fn f32x8_log2_midp_edge_cases() {
        let t = get_x64v3();
        let input = f32x8::from_array(t, [0.0, -1.0, 1.0, 2.0, 0.0, -0.5, 4.0, 8.0]);
        let result = input.log2_midp().to_array();

        // 0 → -inf
        assert!(result[0].is_infinite() && result[0].is_sign_negative(), "log2(0) = -inf");
        assert!(result[4].is_infinite() && result[4].is_sign_negative(), "log2(0) = -inf");

        // negative → NaN
        assert!(result[1].is_nan(), "log2(-1) = NaN");
        assert!(result[5].is_nan(), "log2(-0.5) = NaN");

        // normal values
        assert!((result[2] - 0.0).abs() < 1e-5, "log2(1) ≈ 0");
        assert!((result[3] - 1.0).abs() < 1e-5, "log2(2) ≈ 1");
    }

    #[test]
    fn f32x8_exp2_midp_exact() {
        let t = get_x64v3();
        let input = f32x8::from_array(t, [0.0, 1.0, 2.0, 3.0, -1.0, -2.0, 4.0, 7.0]);
        let result = input.exp2_midp().to_array();
        let expected = [1.0, 2.0, 4.0, 8.0, 0.5, 0.25, 16.0, 128.0];
        assert_close_8(result, expected, 1e-5, "exp2_midp exact");
    }

    #[test]
    fn f32x8_exp2_midp_fractional() {
        let t = get_x64v3();
        let vals = [0.5, 1.5, -0.5, 2.5, 0.1, 0.9, -3.5, 10.0];
        let input = f32x8::from_array(t, vals);
        let result = input.exp2_midp().to_array();
        let expected: [f32; 8] = core::array::from_fn(|i| vals[i].exp2());
        assert_close_8(result, expected, 5e-6, "exp2_midp fractional");
    }

    #[test]
    fn f32x8_ln_midp() {
        let t = get_x64v3();
        let vals = [1.0, core::f32::consts::E, 10.0, 0.5, 100.0, 0.1, 7.389, 20.0];
        let input = f32x8::from_array(t, vals);
        let result = input.ln_midp().to_array();
        let expected: [f32; 8] = core::array::from_fn(|i| vals[i].ln());
        assert_close_8(result, expected, 5e-6, "ln_midp");
    }

    #[test]
    fn f32x8_exp_midp() {
        let t = get_x64v3();
        let vals = [0.0, 1.0, -1.0, 2.0, 0.5, -2.0, 3.0, -0.5];
        let input = f32x8::from_array(t, vals);
        let result = input.exp_midp().to_array();
        let expected: [f32; 8] = core::array::from_fn(|i| vals[i].exp());
        assert_close_8(result, expected, 5e-6, "exp_midp");
    }

    #[test]
    fn f32x8_log10_midp() {
        let t = get_x64v3();
        let vals = [1.0, 10.0, 100.0, 1000.0, 0.1, 0.01, 50.0, 7.0];
        let input = f32x8::from_array(t, vals);
        let result = input.log10_midp().to_array();
        let expected: [f32; 8] = core::array::from_fn(|i| vals[i].log10());
        assert_close_8(result, expected, 5e-6, "log10_midp");
    }

    #[test]
    fn f32x8_pow_midp() {
        let t = get_x64v3();
        let vals = [1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0];
        let input = f32x8::from_array(t, vals);
        let result = input.pow_midp(0.5).to_array();
        let expected: [f32; 8] = core::array::from_fn(|i| vals[i].sqrt());
        assert_close_8(result, expected, 5e-6, "pow_midp(0.5)");
    }

    #[test]
    fn f32x8_pow_midp_cube() {
        let t = get_x64v3();
        let vals = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 0.5, 0.1];
        let input = f32x8::from_array(t, vals);
        let result = input.pow_midp(3.0).to_array();
        let expected: [f32; 8] = core::array::from_fn(|i| vals[i].powi(3));
        assert_close_8(result, expected, 5e-5, "pow_midp(3.0)");
    }

    // ====== f32x8 cbrt Tests ======

    #[test]
    fn f32x8_cbrt_midp_perfect_cubes() {
        let t = get_x64v3();
        let input = f32x8::from_array(t, [1.0, 8.0, 27.0, 64.0, 125.0, 216.0, 343.0, 512.0]);
        let result = input.cbrt_midp().to_array();
        let expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert_close_8(result, expected, 1e-5, "cbrt_midp perfect cubes");
    }

    #[test]
    fn f32x8_cbrt_midp_negative() {
        let t = get_x64v3();
        let input = f32x8::from_array(t, [-1.0, -8.0, -27.0, -64.0, -0.001, -1000.0, -0.125, -1e6]);
        let result = input.cbrt_midp().to_array();
        let expected: [f32; 8] = core::array::from_fn(|i| {
            let v = [-1.0_f32, -8.0, -27.0, -64.0, -0.001, -1000.0, -0.125, -1e6][i];
            -v.abs().cbrt()
        });
        assert_close_8(result, expected, 1e-5, "cbrt_midp negative");
    }

    #[test]
    fn f32x8_cbrt_midp_general() {
        let t = get_x64v3();
        let vals = [0.001, 0.1, 0.5, 2.0, 10.0, 100.0, 1000.0, 1e6];
        let input = f32x8::from_array(t, vals);
        let result = input.cbrt_midp().to_array();
        let expected: [f32; 8] = core::array::from_fn(|i| vals[i].cbrt());
        assert_close_8(result, expected, 1e-5, "cbrt_midp general");
    }

    #[test]
    fn f32x8_cbrt_midp_precise_zero() {
        let t = get_x64v3();
        let input = f32x8::from_array(t, [0.0, 0.0, 1.0, 8.0, -8.0, 0.0, 27.0, 0.0]);
        let result = input.cbrt_midp_precise().to_array();
        assert_eq!(result[0], 0.0, "cbrt_precise(0) = 0");
        assert_eq!(result[1], 0.0, "cbrt_precise(0) = 0");
        assert_eq!(result[5], 0.0, "cbrt_precise(0) = 0");
        assert_eq!(result[7], 0.0, "cbrt_precise(0) = 0");
        assert!((result[2] - 1.0).abs() < 1e-5, "cbrt_precise(1) ≈ 1");
        assert!((result[3] - 2.0).abs() < 1e-5, "cbrt_precise(8) ≈ 2");
    }

    // ====== f32x8 Roundtrip Tests ======

    #[test]
    fn f32x8_log2_exp2_roundtrip() {
        let t = get_x64v3();
        let vals = [0.5, 1.0, 2.0, 4.0, 10.0, 0.1, 50.0, 100.0];
        let input = f32x8::from_array(t, vals);
        let roundtrip = input.log2_midp_unchecked().exp2_midp().to_array();
        assert_close_8(roundtrip, vals, 1e-4, "log2 → exp2 roundtrip");
    }

    #[test]
    fn f32x8_ln_exp_roundtrip() {
        let t = get_x64v3();
        let vals = [0.5, 1.0, 2.0, 4.0, 10.0, 0.1, 50.0, 100.0];
        let input = f32x8::from_array(t, vals);
        let roundtrip = input.ln_midp_unchecked().exp_midp().to_array();
        assert_close_8(roundtrip, vals, 1e-4, "ln → exp roundtrip");
    }

    // ====== f32x8 _unchecked Variants ======

    #[test]
    fn f32x8_unchecked_variants_match() {
        let t = get_x64v3();
        let vals = [1.0, 2.0, 4.0, 8.0, 0.5, 0.25, 16.0, 64.0];
        let input = f32x8::from_array(t, vals);

        // _unchecked and normal should agree on valid inputs
        let log2_normal = input.log2_lowp().to_array();
        let log2_unchecked = input.log2_lowp_unchecked().to_array();
        assert_eq!(log2_normal, log2_unchecked, "log2_lowp == log2_lowp_unchecked on valid input");

        let exp_vals = [0.0, 1.0, 2.0, -1.0, 0.5, -0.5, 3.0, -2.0];
        let exp_input = f32x8::from_array(t, exp_vals);
        let exp_normal = exp_input.exp2_lowp().to_array();
        let exp_unchecked = exp_input.exp2_lowp_unchecked().to_array();
        assert_eq!(exp_normal, exp_unchecked, "exp2_lowp == exp2_lowp_unchecked on valid input");
    }

    // ====== f32x4 Tests (X64V3Token) ======

    #[test]
    fn f32x4_log2_lowp() {
        let t = get_x64v3();
        let input = f32x4::from_array(t, [1.0, 2.0, 4.0, 8.0]);
        let result = input.log2_lowp().to_array();
        let expected = [0.0, 1.0, 2.0, 3.0];
        assert_close_4(result, expected, 0.02, "f32x4 log2_lowp");
    }

    #[test]
    fn f32x4_exp2_lowp() {
        let t = get_x64v3();
        let input = f32x4::from_array(t, [0.0, 1.0, 2.0, -1.0]);
        let result = input.exp2_lowp().to_array();
        let expected = [1.0, 2.0, 4.0, 0.5];
        assert_close_4(result, expected, 0.02, "f32x4 exp2_lowp");
    }

    #[test]
    fn f32x4_log2_midp() {
        let t = get_x64v3();
        let vals = [1.5, 3.0, 10.0, 100.0];
        let input = f32x4::from_array(t, vals);
        let result = input.log2_midp_unchecked().to_array();
        let expected: [f32; 4] = core::array::from_fn(|i| vals[i].log2());
        assert_close_4(result, expected, 5e-6, "f32x4 log2_midp");
    }

    #[test]
    fn f32x4_exp2_midp() {
        let t = get_x64v3();
        let vals = [0.5, 1.5, -0.5, 2.5];
        let input = f32x4::from_array(t, vals);
        let result = input.exp2_midp().to_array();
        let expected: [f32; 4] = core::array::from_fn(|i| vals[i].exp2());
        assert_close_4(result, expected, 5e-6, "f32x4 exp2_midp");
    }

    #[test]
    fn f32x4_ln_midp() {
        let t = get_x64v3();
        let vals = [1.0, core::f32::consts::E, 10.0, 0.5];
        let input = f32x4::from_array(t, vals);
        let result = input.ln_midp().to_array();
        let expected: [f32; 4] = core::array::from_fn(|i| vals[i].ln());
        assert_close_4(result, expected, 5e-6, "f32x4 ln_midp");
    }

    #[test]
    fn f32x4_cbrt_midp() {
        let t = get_x64v3();
        let input = f32x4::from_array(t, [1.0, 8.0, 27.0, 64.0]);
        let result = input.cbrt_midp().to_array();
        let expected = [1.0, 2.0, 3.0, 4.0];
        assert_close_4(result, expected, 1e-5, "f32x4 cbrt_midp");
    }

    #[test]
    fn f32x4_cbrt_midp_negative() {
        let t = get_x64v3();
        let input = f32x4::from_array(t, [-1.0, -8.0, -27.0, -64.0]);
        let result = input.cbrt_midp().to_array();
        let expected = [-1.0, -2.0, -3.0, -4.0];
        assert_close_4(result, expected, 1e-5, "f32x4 cbrt_midp negative");
    }

    #[test]
    fn f32x4_log2_midp_edge_cases() {
        let t = get_x64v3();
        let input = f32x4::from_array(t, [0.0, -1.0, 1.0, 2.0]);
        let result = input.log2_midp().to_array();
        assert!(result[0].is_infinite() && result[0].is_sign_negative());
        assert!(result[1].is_nan());
        assert!((result[2] - 0.0).abs() < 1e-5);
        assert!((result[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn f32x4_pow_midp() {
        let t = get_x64v3();
        let input = f32x4::from_array(t, [4.0, 9.0, 16.0, 25.0]);
        let result = input.pow_midp(0.5).to_array();
        let expected = [2.0, 3.0, 4.0, 5.0];
        assert_close_4(result, expected, 5e-6, "f32x4 pow_midp(0.5)");
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
        let x86_t = get_x64v3();
        let scalar_t = ScalarToken;
        let vals = [1.5, 3.0, 10.0, 100.0, 0.1, 0.01, 7.0, 42.0];

        let x86_result = f32x8::from_array(x86_t, vals).log2_midp_unchecked().to_array();
        let scalar_result = f32x8::from_array(scalar_t, vals).log2_midp_unchecked().to_array();

        // Both should be close to std::f32::log2 (may differ by a few ULP due to FMA)
        let expected: [f32; 8] = core::array::from_fn(|i| vals[i].log2());
        assert_close_8(x86_result, expected, 5e-6, "x86 log2_midp");
        assert_close_8(scalar_result, expected, 5e-6, "scalar log2_midp");
    }

    #[test]
    fn f32x8_x86_scalar_exp2_agree() {
        let x86_t = get_x64v3();
        let scalar_t = ScalarToken;
        let vals = [0.5, 1.5, -0.5, 2.5, 0.1, 0.9, -3.5, 10.0];

        let x86_result = f32x8::from_array(x86_t, vals).exp2_midp().to_array();
        let scalar_result = f32x8::from_array(scalar_t, vals).exp2_midp().to_array();

        let expected: [f32; 8] = core::array::from_fn(|i| vals[i].exp2());
        assert_close_8(x86_result, expected, 5e-6, "x86 exp2_midp");
        assert_close_8(scalar_result, expected, 5e-6, "scalar exp2_midp");
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
        let t = get_x64v3();
        let linear = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0];
        let result = gamma_correct(t, &linear, 2.2);

        // pow(x, 1/2.2) should be in [0, 1] for inputs in [0, 1]
        for (i, &v) in result.iter().enumerate() {
            if linear[i] == 0.0 {
                // pow(0, x) via log2→exp2 is undefined, skip
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

    // ====== Large Range Test ======

    #[test]
    fn f32x8_exp2_midp_large_range() {
        let t = get_x64v3();
        // Test across the representable range
        let vals = [-100.0, -50.0, -10.0, -1.0, 1.0, 10.0, 50.0, 100.0];
        let input = f32x8::from_array(t, vals);
        let result = input.exp2_midp().to_array();
        let expected: [f32; 8] = core::array::from_fn(|i| vals[i].exp2());
        assert_close_8(result, expected, 5e-5, "exp2_midp large range");
    }

    #[test]
    fn f32x8_log2_midp_large_range() {
        let t = get_x64v3();
        let vals = [1e-30, 1e-10, 0.001, 1.0, 1000.0, 1e10, 1e30, 1e38];
        let input = f32x8::from_array(t, vals);
        let result = input.log2_midp_unchecked().to_array();
        let expected: [f32; 8] = core::array::from_fn(|i| vals[i].log2());
        assert_close_8(result, expected, 5e-5, "log2_midp large range");
    }

    // ====== Parity with Old Types ======

    /// Compare generic f32x8<X64V3Token> transcendentals against old per-platform types.
    #[test]
    fn f32x8_generic_vs_old_log2_lowp() {
        use magetypes::simd::f32x8 as OldF32x8;

        let t = get_x64v3();
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

    #[test]
    fn f32x8_generic_vs_old_exp2_lowp() {
        use magetypes::simd::f32x8 as OldF32x8;

        let t = get_x64v3();
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

    #[test]
    fn f32x8_generic_vs_old_log2_midp() {
        use magetypes::simd::f32x8 as OldF32x8;

        let t = get_x64v3();
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

    #[test]
    fn f32x8_generic_vs_old_cbrt_midp() {
        use magetypes::simd::f32x8 as OldF32x8;

        let t = get_x64v3();
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

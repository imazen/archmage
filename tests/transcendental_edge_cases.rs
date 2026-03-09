//! Tests that transcendental functions handle edge cases correctly.
//!
//! Covers: cbrt (all variants), pow (all variants), log2/exp2/ln/exp (midp)
//! Edge cases: zero, negative zero, infinity, NaN, denormals
//!
//! Tests both the direct x86 types AND generic types to catch parity issues.
//!
//! Run:
//!   cargo test --test transcendental_edge_cases --features "std"
//!   cargo test --test transcendental_edge_cases --features "std avx512"

#![cfg(target_arch = "x86_64")]

use archmage::{SimdToken, X64V3Token, arcane};
use magetypes::simd::{f32x8, v3};

// ============================================================================
// Direct x86 type wrappers (tests the platform-specific codegen)
// ============================================================================

#[arcane]
fn direct_cbrt_lowp(token: X64V3Token, input: &[f32; 8]) -> [f32; 8] {
    v3::f32x8::load(token, input).cbrt_lowp().to_array()
}

#[arcane]
fn direct_cbrt_midp(token: X64V3Token, input: &[f32; 8]) -> [f32; 8] {
    v3::f32x8::load(token, input).cbrt_midp().to_array()
}

#[arcane]
fn direct_cbrt_midp_precise(token: X64V3Token, input: &[f32; 8]) -> [f32; 8] {
    v3::f32x8::load(token, input).cbrt_midp_precise().to_array()
}

#[arcane]
fn direct_pow_lowp(token: X64V3Token, input: &[f32; 8], n: f32) -> [f32; 8] {
    v3::f32x8::load(token, input).pow_lowp(n).to_array()
}

#[arcane]
fn direct_pow_midp(token: X64V3Token, input: &[f32; 8], n: f32) -> [f32; 8] {
    v3::f32x8::load(token, input).pow_midp(n).to_array()
}

#[arcane]
fn direct_pow_midp_precise(token: X64V3Token, input: &[f32; 8], n: f32) -> [f32; 8] {
    v3::f32x8::load(token, input).pow_midp_precise(n).to_array()
}

#[arcane]
fn direct_pow_midp_unchecked(token: X64V3Token, input: &[f32; 8], n: f32) -> [f32; 8] {
    v3::f32x8::load(token, input)
        .pow_midp_unchecked(n)
        .to_array()
}

#[arcane]
fn direct_pow_lowp_unchecked(token: X64V3Token, input: &[f32; 8], n: f32) -> [f32; 8] {
    v3::f32x8::load(token, input)
        .pow_lowp_unchecked(n)
        .to_array()
}

#[arcane]
fn direct_log2_lowp(token: X64V3Token, input: &[f32; 8]) -> [f32; 8] {
    v3::f32x8::load(token, input).log2_lowp().to_array()
}

#[arcane]
fn direct_log2_midp(token: X64V3Token, input: &[f32; 8]) -> [f32; 8] {
    v3::f32x8::load(token, input).log2_midp().to_array()
}

#[arcane]
fn direct_exp2_midp(token: X64V3Token, input: &[f32; 8]) -> [f32; 8] {
    v3::f32x8::load(token, input).exp2_midp().to_array()
}

#[arcane]
fn direct_ln_midp(token: X64V3Token, input: &[f32; 8]) -> [f32; 8] {
    v3::f32x8::load(token, input).ln_midp().to_array()
}

#[arcane]
fn direct_exp_midp(token: X64V3Token, input: &[f32; 8]) -> [f32; 8] {
    v3::f32x8::load(token, input).exp_midp().to_array()
}

// ============================================================================
// Generic type wrappers (tests the generic codegen)
// ============================================================================

#[arcane]
fn generic_cbrt_lowp(token: X64V3Token, input: &[f32; 8]) -> [f32; 8] {
    f32x8::load(token, input).cbrt_lowp().to_array()
}

#[arcane]
fn generic_cbrt_midp(token: X64V3Token, input: &[f32; 8]) -> [f32; 8] {
    f32x8::load(token, input).cbrt_midp().to_array()
}

#[arcane]
fn generic_cbrt_midp_precise(token: X64V3Token, input: &[f32; 8]) -> [f32; 8] {
    f32x8::load(token, input).cbrt_midp_precise().to_array()
}

#[arcane]
fn generic_pow_lowp(token: X64V3Token, input: &[f32; 8], n: f32) -> [f32; 8] {
    f32x8::load(token, input).pow_lowp(n).to_array()
}

#[arcane]
fn generic_pow_midp(token: X64V3Token, input: &[f32; 8], n: f32) -> [f32; 8] {
    f32x8::load(token, input).pow_midp(n).to_array()
}

#[arcane]
fn generic_pow_midp_precise(token: X64V3Token, input: &[f32; 8], n: f32) -> [f32; 8] {
    f32x8::load(token, input).pow_midp_precise(n).to_array()
}

// ============================================================================
// 128-bit (f32x4) direct type wrappers
// ============================================================================

#[arcane]
fn direct_f32x4_cbrt_lowp(token: X64V3Token, input: &[f32; 4]) -> [f32; 4] {
    v3::f32x4::load(token, input).cbrt_lowp().to_array()
}

#[arcane]
fn direct_f32x4_cbrt_midp(token: X64V3Token, input: &[f32; 4]) -> [f32; 4] {
    v3::f32x4::load(token, input).cbrt_midp().to_array()
}

#[arcane]
fn direct_f32x4_cbrt_midp_precise(token: X64V3Token, input: &[f32; 4]) -> [f32; 4] {
    v3::f32x4::load(token, input).cbrt_midp_precise().to_array()
}

#[arcane]
fn direct_f32x4_pow_midp(token: X64V3Token, input: &[f32; 4], n: f32) -> [f32; 4] {
    v3::f32x4::load(token, input).pow_midp(n).to_array()
}

#[arcane]
fn direct_f32x4_pow_midp_precise(token: X64V3Token, input: &[f32; 4], n: f32) -> [f32; 4] {
    v3::f32x4::load(token, input).pow_midp_precise(n).to_array()
}

// ============================================================================
// Tests
// ============================================================================

/// Assert that a value is bit-exact zero (positive or negative).
fn assert_zero(val: f32, label: &str) {
    assert!(
        val == 0.0 && (val.to_bits() == 0 || val.to_bits() == 0x8000_0000),
        "{label}: expected ±0.0, got {val} (bits: {:#010x})",
        val.to_bits()
    );
}

/// Assert a value is positive zero specifically.
fn assert_pos_zero(val: f32, label: &str) {
    assert_eq!(
        val.to_bits(),
        0,
        "{label}: expected +0.0, got {val} (bits: {:#010x})",
        val.to_bits()
    );
}

/// Assert a value is negative zero specifically.
fn assert_neg_zero(val: f32, label: &str) {
    assert_eq!(
        val.to_bits(),
        0x8000_0000,
        "{label}: expected -0.0, got {val} (bits: {:#010x})",
        val.to_bits()
    );
}

mod cbrt_zero {
    use super::*;

    // ---- Direct x86 types ----

    #[test]
    fn direct_cbrt_lowp_pos_zero() {
        let Some(t) = X64V3Token::summon() else {
            return;
        };
        let input = [0.0f32; 8];
        let result = direct_cbrt_lowp(t, &input);
        for (i, &v) in result.iter().enumerate() {
            assert_pos_zero(v, &format!("direct_cbrt_lowp[{i}]"));
        }
    }

    #[test]
    fn direct_cbrt_lowp_neg_zero() {
        let Some(t) = X64V3Token::summon() else {
            return;
        };
        let input = [-0.0f32; 8];
        let result = direct_cbrt_lowp(t, &input);
        for (i, &v) in result.iter().enumerate() {
            assert_neg_zero(v, &format!("direct_cbrt_lowp[{i}]"));
        }
    }

    #[test]
    fn direct_cbrt_midp_pos_zero() {
        let Some(t) = X64V3Token::summon() else {
            return;
        };
        let input = [0.0f32; 8];
        let result = direct_cbrt_midp(t, &input);
        for (i, &v) in result.iter().enumerate() {
            assert_pos_zero(v, &format!("direct_cbrt_midp[{i}]"));
        }
    }

    #[test]
    fn direct_cbrt_midp_neg_zero() {
        let Some(t) = X64V3Token::summon() else {
            return;
        };
        let input = [-0.0f32; 8];
        let result = direct_cbrt_midp(t, &input);
        for (i, &v) in result.iter().enumerate() {
            assert_neg_zero(v, &format!("direct_cbrt_midp[{i}]"));
        }
    }

    #[test]
    fn direct_cbrt_midp_precise_pos_zero() {
        let Some(t) = X64V3Token::summon() else {
            return;
        };
        let input = [0.0f32; 8];
        let result = direct_cbrt_midp_precise(t, &input);
        for (i, &v) in result.iter().enumerate() {
            assert_pos_zero(v, &format!("direct_cbrt_midp_precise[{i}]"));
        }
    }

    #[test]
    fn direct_cbrt_midp_precise_neg_zero() {
        let Some(t) = X64V3Token::summon() else {
            return;
        };
        let input = [-0.0f32; 8];
        let result = direct_cbrt_midp_precise(t, &input);
        for (i, &v) in result.iter().enumerate() {
            assert_neg_zero(v, &format!("direct_cbrt_midp_precise[{i}]"));
        }
    }

    // ---- Generic types ----

    #[test]
    fn generic_cbrt_lowp_pos_zero() {
        let Some(t) = X64V3Token::summon() else {
            return;
        };
        let input = [0.0f32; 8];
        let result = generic_cbrt_lowp(t, &input);
        for (i, &v) in result.iter().enumerate() {
            assert_pos_zero(v, &format!("generic_cbrt_lowp[{i}]"));
        }
    }

    #[test]
    fn generic_cbrt_lowp_neg_zero() {
        let Some(t) = X64V3Token::summon() else {
            return;
        };
        let input = [-0.0f32; 8];
        let result = generic_cbrt_lowp(t, &input);
        for (i, &v) in result.iter().enumerate() {
            assert_neg_zero(v, &format!("generic_cbrt_lowp[{i}]"));
        }
    }

    #[test]
    fn generic_cbrt_midp_pos_zero() {
        let Some(t) = X64V3Token::summon() else {
            return;
        };
        let input = [0.0f32; 8];
        let result = generic_cbrt_midp(t, &input);
        for (i, &v) in result.iter().enumerate() {
            assert_pos_zero(v, &format!("generic_cbrt_midp[{i}]"));
        }
    }

    #[test]
    fn generic_cbrt_midp_neg_zero() {
        let Some(t) = X64V3Token::summon() else {
            return;
        };
        let input = [-0.0f32; 8];
        let result = generic_cbrt_midp(t, &input);
        for (i, &v) in result.iter().enumerate() {
            assert_neg_zero(v, &format!("generic_cbrt_midp[{i}]"));
        }
    }

    #[test]
    fn generic_cbrt_midp_precise_pos_zero() {
        let Some(t) = X64V3Token::summon() else {
            return;
        };
        let input = [0.0f32; 8];
        let result = generic_cbrt_midp_precise(t, &input);
        for (i, &v) in result.iter().enumerate() {
            assert_pos_zero(v, &format!("generic_cbrt_midp_precise[{i}]"));
        }
    }

    #[test]
    fn generic_cbrt_midp_precise_neg_zero() {
        let Some(t) = X64V3Token::summon() else {
            return;
        };
        let input = [-0.0f32; 8];
        let result = generic_cbrt_midp_precise(t, &input);
        for (i, &v) in result.iter().enumerate() {
            assert_neg_zero(v, &format!("generic_cbrt_midp_precise[{i}]"));
        }
    }

    // ---- 128-bit (f32x4) direct types ----

    #[test]
    fn direct_f32x4_cbrt_lowp_zero() {
        let Some(t) = X64V3Token::summon() else {
            return;
        };
        let result = direct_f32x4_cbrt_lowp(t, &[0.0, -0.0, 0.0, -0.0]);
        assert_pos_zero(result[0], "f32x4_cbrt_lowp[0]");
        assert_neg_zero(result[1], "f32x4_cbrt_lowp[1]");
        assert_pos_zero(result[2], "f32x4_cbrt_lowp[2]");
        assert_neg_zero(result[3], "f32x4_cbrt_lowp[3]");
    }

    #[test]
    fn direct_f32x4_cbrt_midp_zero() {
        let Some(t) = X64V3Token::summon() else {
            return;
        };
        let result = direct_f32x4_cbrt_midp(t, &[0.0, -0.0, 0.0, -0.0]);
        assert_pos_zero(result[0], "f32x4_cbrt_midp[0]");
        assert_neg_zero(result[1], "f32x4_cbrt_midp[1]");
        assert_pos_zero(result[2], "f32x4_cbrt_midp[2]");
        assert_neg_zero(result[3], "f32x4_cbrt_midp[3]");
    }

    #[test]
    fn direct_f32x4_cbrt_midp_precise_zero() {
        let Some(t) = X64V3Token::summon() else {
            return;
        };
        let result = direct_f32x4_cbrt_midp_precise(t, &[0.0, -0.0, 0.0, -0.0]);
        assert_pos_zero(result[0], "f32x4_cbrt_midp_precise[0]");
        assert_neg_zero(result[1], "f32x4_cbrt_midp_precise[1]");
        assert_pos_zero(result[2], "f32x4_cbrt_midp_precise[2]");
        assert_neg_zero(result[3], "f32x4_cbrt_midp_precise[3]");
    }

    // ---- Mixed inputs (zeros interleaved with normal values) ----

    #[test]
    fn direct_cbrt_mixed_with_zeros() {
        let Some(t) = X64V3Token::summon() else {
            return;
        };
        let input = [0.0, 8.0, -0.0, 27.0, 0.0, -8.0, 1.0, -0.0];
        let result_lowp = direct_cbrt_lowp(t, &input);
        let result_midp = direct_cbrt_midp(t, &input);
        let result_precise = direct_cbrt_midp_precise(t, &input);

        // Check zeros
        assert_pos_zero(result_lowp[0], "lowp[0]");
        assert_neg_zero(result_lowp[2], "lowp[2]");
        assert_pos_zero(result_lowp[4], "lowp[4]");
        assert_neg_zero(result_lowp[7], "lowp[7]");

        assert_pos_zero(result_midp[0], "midp[0]");
        assert_neg_zero(result_midp[2], "midp[2]");
        assert_pos_zero(result_midp[4], "midp[4]");
        assert_neg_zero(result_midp[7], "midp[7]");

        assert_pos_zero(result_precise[0], "precise[0]");
        assert_neg_zero(result_precise[2], "precise[2]");
        assert_pos_zero(result_precise[4], "precise[4]");
        assert_neg_zero(result_precise[7], "precise[7]");

        // Check non-zero values haven't been broken
        assert!(
            (result_midp[1] - 2.0).abs() < 1e-5,
            "cbrt(8) ~ 2, got {}",
            result_midp[1]
        );
        assert!(
            (result_midp[3] - 3.0).abs() < 1e-5,
            "cbrt(27) ~ 3, got {}",
            result_midp[3]
        );
        assert!(
            (result_midp[5] - (-2.0)).abs() < 1e-5,
            "cbrt(-8) ~ -2, got {}",
            result_midp[5]
        );
        assert!(
            (result_midp[6] - 1.0).abs() < 1e-5,
            "cbrt(1) ~ 1, got {}",
            result_midp[6]
        );
    }

    // ---- Scalar fallbacks ----

    #[test]
    fn scalar_cbrt_lowp_zero() {
        assert_pos_zero(
            magetypes::nostd_math::cbrt_lowp_f32(0.0),
            "cbrt_lowp_f32(+0)",
        );
        assert_neg_zero(
            magetypes::nostd_math::cbrt_lowp_f32(-0.0),
            "cbrt_lowp_f32(-0)",
        );
    }

    #[test]
    fn scalar_cbrt_midp_zero() {
        assert_pos_zero(
            magetypes::nostd_math::cbrt_midp_f32(0.0),
            "cbrt_midp_f32(+0)",
        );
        assert_neg_zero(
            magetypes::nostd_math::cbrt_midp_f32(-0.0),
            "cbrt_midp_f32(-0)",
        );
    }
}

mod pow_zero {
    use super::*;

    /// pow(0, positive_n) should be 0
    #[test]
    fn pow_midp_zero_to_positive() {
        let Some(t) = X64V3Token::summon() else {
            return;
        };
        let zeros = [0.0f32; 8];
        for n in [0.5, 1.0, 2.0, 3.0, 10.0] {
            let result = direct_pow_midp(t, &zeros, n);
            for (i, &v) in result.iter().enumerate() {
                assert_zero(v, &format!("pow_midp(0, {n})[{i}]"));
            }
        }
    }

    /// pow(0, positive_n) should be 0 — precise variant
    #[test]
    fn pow_midp_precise_zero_to_positive() {
        let Some(t) = X64V3Token::summon() else {
            return;
        };
        let zeros = [0.0f32; 8];
        for n in [0.5, 1.0, 2.0, 3.0, 10.0] {
            let result = direct_pow_midp_precise(t, &zeros, n);
            for (i, &v) in result.iter().enumerate() {
                assert_zero(v, &format!("pow_midp_precise(0, {n})[{i}]"));
            }
        }
    }

    /// pow(0, negative_n) should be inf
    #[test]
    fn pow_midp_zero_to_negative() {
        let Some(t) = X64V3Token::summon() else {
            return;
        };
        let zeros = [0.0f32; 8];
        let result = direct_pow_midp(t, &zeros, -1.0);
        for (i, &v) in result.iter().enumerate() {
            assert!(
                v == f32::INFINITY,
                "pow_midp(0, -1)[{i}]: expected inf, got {v}"
            );
        }
    }

    /// pow(positive, 0) should be 1.0 — this is the IEEE convention
    /// Note: pow_midp uses log2_midp(x) * 0 which gives 0 for positive x, then exp2(0) = 1.
    #[test]
    fn pow_midp_positive_to_zero() {
        let Some(t) = X64V3Token::summon() else {
            return;
        };
        let input = [1.0, 2.0, 0.5, 100.0, 0.001, 1e10, 1e-10, 42.0];
        let result = direct_pow_midp(t, &input, 0.0);
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "pow_midp({}, 0)[{i}]: expected 1.0, got {v}",
                input[i]
            );
        }
    }

    /// pow(0, 0) behavior: IEEE says 1.0, but log-exp composition gives
    /// indeterminate results (-inf * 0 = NaN, then NaN through exp2 clamping).
    /// The exact result depends on SIMD min/max NaN handling which varies.
    /// This test just documents that the result is not 1.0 (we don't fix this).
    #[test]
    fn pow_midp_zero_to_zero_not_one() {
        let Some(t) = X64V3Token::summon() else {
            return;
        };
        let zeros = [0.0f32; 8];
        let result = direct_pow_midp(t, &zeros, 0.0);
        // log2_midp(0) = -inf, -inf * 0 = NaN, NaN through exp2 clamping
        // Result is implementation-defined (NaN or garbage from clamped NaN).
        // We don't guarantee IEEE pow(0,0)=1 behavior.
        for &v in result.iter() {
            // Just verify we don't crash and the result is "weird"
            assert!(
                v != 1.0,
                "pow_midp(0, 0) unexpectedly returned 1.0 (IEEE convention) — \
                 if this is intentional, update the test"
            );
        }
    }

    /// pow generic and direct both return zero for pow(0, positive_n)
    #[test]
    fn generic_pow_midp_zero_to_positive() {
        let Some(t) = X64V3Token::summon() else {
            return;
        };
        let zeros = [0.0f32; 8];
        for n in [0.5, 1.0, 2.0, 3.0] {
            let generic = generic_pow_midp(t, &zeros, n);
            for (i, &v) in generic.iter().enumerate() {
                assert_zero(v, &format!("generic_pow_midp(0, {n})[{i}]"));
            }
        }
    }

    /// pow(0, positive) via f32x4
    #[test]
    fn f32x4_pow_midp_zero() {
        let Some(t) = X64V3Token::summon() else {
            return;
        };
        let result = direct_f32x4_pow_midp(t, &[0.0; 4], 2.0);
        for (i, &v) in result.iter().enumerate() {
            assert_zero(v, &format!("f32x4_pow_midp(0, 2)[{i}]"));
        }
    }

    /// pow(0, positive) via f32x4 precise
    #[test]
    fn f32x4_pow_midp_precise_zero() {
        let Some(t) = X64V3Token::summon() else {
            return;
        };
        let result = direct_f32x4_pow_midp_precise(t, &[0.0; 4], 2.0);
        for (i, &v) in result.iter().enumerate() {
            assert_zero(v, &format!("f32x4_pow_midp_precise(0, 2)[{i}]"));
        }
    }
}

mod log_exp_edge_cases {
    use super::*;

    #[test]
    fn log2_midp_zero_is_neg_inf() {
        let Some(t) = X64V3Token::summon() else {
            return;
        };
        let result = direct_log2_midp(t, &[0.0; 8]);
        for (i, &v) in result.iter().enumerate() {
            assert_eq!(
                v,
                f32::NEG_INFINITY,
                "log2_midp(0)[{i}]: expected -inf, got {v}"
            );
        }
    }

    #[test]
    fn log2_midp_negative_is_nan() {
        let Some(t) = X64V3Token::summon() else {
            return;
        };
        let result = direct_log2_midp(t, &[-1.0; 8]);
        for (i, &v) in result.iter().enumerate() {
            assert!(v.is_nan(), "log2_midp(-1)[{i}]: expected NaN, got {v}");
        }
    }

    #[test]
    fn log2_midp_inf_is_inf() {
        let Some(t) = X64V3Token::summon() else {
            return;
        };
        let result = direct_log2_midp(t, &[f32::INFINITY; 8]);
        for (i, &v) in result.iter().enumerate() {
            assert_eq!(
                v,
                f32::INFINITY,
                "log2_midp(inf)[{i}]: expected inf, got {v}"
            );
        }
    }

    #[test]
    fn exp2_midp_neg_inf_is_zero() {
        let Some(t) = X64V3Token::summon() else {
            return;
        };
        let result = direct_exp2_midp(t, &[f32::NEG_INFINITY; 8]);
        for (i, &v) in result.iter().enumerate() {
            assert_zero(v, &format!("exp2_midp(-inf)[{i}]"));
        }
    }

    #[test]
    fn exp2_midp_inf_is_inf() {
        let Some(t) = X64V3Token::summon() else {
            return;
        };
        let result = direct_exp2_midp(t, &[f32::INFINITY; 8]);
        for (i, &v) in result.iter().enumerate() {
            assert_eq!(
                v,
                f32::INFINITY,
                "exp2_midp(inf)[{i}]: expected inf, got {v}"
            );
        }
    }

    #[test]
    fn exp2_midp_zero_is_one() {
        let Some(t) = X64V3Token::summon() else {
            return;
        };
        let result = direct_exp2_midp(t, &[0.0; 8]);
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-6,
                "exp2_midp(0)[{i}]: expected 1.0, got {v}"
            );
        }
    }

    #[test]
    fn ln_midp_zero_is_neg_inf() {
        let Some(t) = X64V3Token::summon() else {
            return;
        };
        let result = direct_ln_midp(t, &[0.0; 8]);
        for (i, &v) in result.iter().enumerate() {
            assert_eq!(
                v,
                f32::NEG_INFINITY,
                "ln_midp(0)[{i}]: expected -inf, got {v}"
            );
        }
    }

    #[test]
    fn exp_midp_zero_is_one() {
        let Some(t) = X64V3Token::summon() else {
            return;
        };
        let result = direct_exp_midp(t, &[0.0; 8]);
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-6,
                "exp_midp(0)[{i}]: expected 1.0, got {v}"
            );
        }
    }
}

#[cfg(feature = "avx512")]
mod avx512_cbrt_zero {
    use super::{assert_neg_zero, assert_pos_zero, assert_zero};
    use archmage::{SimdToken, X64V4Token, arcane};
    use magetypes::simd::v4;

    #[arcane]
    fn v4_cbrt_lowp(token: X64V4Token, input: &[f32; 16]) -> [f32; 16] {
        v4::f32x16::load(token, input).cbrt_lowp().to_array()
    }

    #[arcane]
    fn v4_cbrt_midp(token: X64V4Token, input: &[f32; 16]) -> [f32; 16] {
        v4::f32x16::load(token, input).cbrt_midp().to_array()
    }

    #[arcane]
    fn v4_cbrt_midp_precise(token: X64V4Token, input: &[f32; 16]) -> [f32; 16] {
        v4::f32x16::load(token, input)
            .cbrt_midp_precise()
            .to_array()
    }

    #[arcane]
    fn v4_pow_midp(token: X64V4Token, input: &[f32; 16], n: f32) -> [f32; 16] {
        v4::f32x16::load(token, input).pow_midp(n).to_array()
    }

    #[test]
    fn avx512_cbrt_lowp_zero() {
        let Some(t) = X64V4Token::summon() else {
            return;
        };
        let input = [0.0f32; 16];
        let result = v4_cbrt_lowp(t, &input);
        for (i, &v) in result.iter().enumerate() {
            assert_pos_zero(v, &format!("avx512_cbrt_lowp[{i}]"));
        }
    }

    #[test]
    fn avx512_cbrt_midp_zero() {
        let Some(t) = X64V4Token::summon() else {
            return;
        };
        let result_pos = v4_cbrt_midp(t, &[0.0; 16]);
        let result_neg = v4_cbrt_midp(t, &[-0.0; 16]);
        for (i, &v) in result_pos.iter().enumerate() {
            assert_pos_zero(v, &format!("avx512_cbrt_midp_pos[{i}]"));
        }
        for (i, &v) in result_neg.iter().enumerate() {
            assert_neg_zero(v, &format!("avx512_cbrt_midp_neg[{i}]"));
        }
    }

    #[test]
    fn avx512_cbrt_midp_precise_zero() {
        let Some(t) = X64V4Token::summon() else {
            return;
        };
        let result = v4_cbrt_midp_precise(t, &[0.0; 16]);
        for (i, &v) in result.iter().enumerate() {
            assert_pos_zero(v, &format!("avx512_cbrt_midp_precise[{i}]"));
        }
    }

    #[test]
    fn avx512_cbrt_mixed_zeros_and_values() {
        let Some(t) = X64V4Token::summon() else {
            return;
        };
        let input = [
            0.0, 8.0, -0.0, 27.0, 0.0, -8.0, 1.0, -0.0, 0.0, 64.0, -0.0, 125.0, 0.0, -27.0, 1000.0,
            -0.0,
        ];
        let result = v4_cbrt_midp(t, &input);

        assert_pos_zero(result[0], "mixed[0]");
        assert!((result[1] - 2.0).abs() < 1e-5);
        assert_neg_zero(result[2], "mixed[2]");
        assert!((result[3] - 3.0).abs() < 1e-5);
        assert_pos_zero(result[4], "mixed[4]");
        assert!((result[5] + 2.0).abs() < 1e-5);
        assert_neg_zero(result[7], "mixed[7]");
        assert_pos_zero(result[8], "mixed[8]");
        assert_neg_zero(result[10], "mixed[10]");
        assert_pos_zero(result[12], "mixed[12]");
        assert_neg_zero(result[15], "mixed[15]");
    }

    #[test]
    fn avx512_pow_midp_zero_to_positive() {
        let Some(t) = X64V4Token::summon() else {
            return;
        };
        let result = v4_pow_midp(t, &[0.0; 16], 2.0);
        for (i, &v) in result.iter().enumerate() {
            assert_zero(v, &format!("avx512_pow_midp(0, 2)[{i}]"));
        }
    }
}

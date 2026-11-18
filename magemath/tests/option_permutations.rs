//! Comprehensive verification of option constraints, out-of-bounds policies,
//! granular domain flags, polynomial zero-reduction ranges, and holding each
//! individual option constraint fixed while sweeping all other option axes.

use archmage::ScalarToken;
use magemath::{
    Execution, Portability, SimdMath, SliceError, Validation,
    cbrtf::{Cbrtf, CbrtfErrorKind, CbrtfOptions, CbrtfPrecision},
    expf::{Expf, ExpfErrorKind, ExpfOptions, ExpfPrecision, try_exp},
    logf::{Logf, LogfErrorKind, LogfOptions, LogfPrecision},
    powf::{MAX_EXP_BOUND, Powf, PowfErrorKind, PowfOptions, PowfPrecision},
};
#[cfg(feature = "w512")]
use magetypes::simd::generic::f32x16;
use magetypes::simd::generic::{f32x4, f32x8};

fn assert_bits_or_nan(actual: f32, expected: f32, ctx: &str) {
    if expected.is_nan() {
        assert!(actual.is_nan(), "{ctx}: expected NaN, got {actual:?}");
    } else {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{ctx}: actual={actual:?} ({:#010x}) != expected={expected:?} ({:#010x})",
            actual.to_bits(),
            expected.to_bits()
        );
    }
}

fn assert_close_rel(actual: f32, expected: f32, max_rel: f32, ctx: &str) {
    let diff = (actual - expected).abs();
    let scale = expected.abs().max(f32::MIN_POSITIVE);
    assert!(
        diff <= max_rel * scale,
        "{ctx}: actual={actual:?}, expected={expected:?}, rel_err={}",
        diff / scale
    );
}

// ============================================================================
// 1. Targeted regressions for review findings & edge boundaries
// ============================================================================

#[test]
fn expf_midp_127_5_upper_endpoint_stays_finite_and_neighbors_elide() {
    const AT_127_5: ExpfOptions = ExpfOptions::new().input_range(127.5, 127.5);
    const BELOW_127_5: ExpfOptions =
        ExpfOptions::new().input_range(-126.0, f32::from_bits(127.5_f32.to_bits() - 1));
    assert!(!AT_127_5.elides_clamp());
    assert!(BELOW_127_5.elides_clamp());

    let ref_val = <Expf>::exp2(127.5).unwrap();
    assert!(ref_val.is_finite() && ref_val > 0.0);
    assert_eq!(
        Expf::<{ AT_127_5.bits() }>::exp2(127.5).unwrap().to_bits(),
        ref_val.to_bits()
    );
    assert_eq!(
        AT_127_5.runtime().exp2(127.5).unwrap().to_bits(),
        ref_val.to_bits()
    );

    let pred = f32::from_bits(127.5_f32.to_bits() - 1);
    let pred_ref = <Expf>::exp2(pred).unwrap();
    assert!(pred_ref.is_finite());
    assert_eq!(
        Expf::<{ BELOW_127_5.bits() }>::exp2(pred)
            .unwrap()
            .to_bits(),
        pred_ref.to_bits()
    );

    // `0x42b0_c0a6` scales by LOG2_E to exact 127.5f32.
    let exp_edge = f32::from_bits(0x42b0_c0a6);
    assert_eq!(exp_edge * core::f32::consts::LOG2_E, 127.5);
    let exp_at = ExpfOptions::new().input_range(exp_edge, exp_edge);
    assert!(!exp_at.elides_exp_clamp());
    let exp_ref = try_exp(exp_edge).unwrap();
    assert!(exp_ref.is_finite());
    assert_eq!(
        exp_at.runtime().exp(exp_edge).unwrap().to_bits(),
        exp_ref.to_bits()
    );
    const EXP_AT_BITS: u128 = ExpfOptions::new()
        .input_range(f32::from_bits(0x42b0_c0a6), f32::from_bits(0x42b0_c0a6))
        .bits();
    assert_eq!(
        Expf::<EXP_AT_BITS>::exp(exp_edge).unwrap().to_bits(),
        exp_ref.to_bits()
    );

    let exp_pred = f32::from_bits(0x42b0_c0a5);
    assert!(exp_pred * core::f32::consts::LOG2_E < 127.5);
    assert!(
        ExpfOptions::new()
            .input_range(-87.0, exp_pred)
            .elides_exp_clamp()
    );
}

#[test]
fn expf_default_checked_accepts_large_finite_inputs_when_scaled_arg_overflows() {
    assert_eq!(try_exp(f32::MAX).unwrap(), f32::INFINITY);
    assert_eq!(try_exp(-f32::MAX).unwrap().to_bits(), 0.0_f32.to_bits());

    const RANGED_ALL_FINITE: u128 = ExpfOptions::new().input_range(-f32::MAX, f32::MAX).bits();
    assert_eq!(
        Expf::<RANGED_ALL_FINITE>::exp(f32::MAX).unwrap(),
        f32::INFINITY
    );
    assert_eq!(
        Expf::<RANGED_ALL_FINITE>::exp(-f32::MAX).unwrap().to_bits(),
        0.0_f32.to_bits()
    );

    const DEFAULT_LOWP: u128 = ExpfOptions::new().lowp().bits();
    const RANGED_LOWP: u128 = ExpfOptions::new()
        .lowp()
        .input_range(-f32::MAX, f32::MAX)
        .bits();
    assert_eq!(
        Expf::<DEFAULT_LOWP>::exp(f32::MAX).unwrap().to_bits(),
        Expf::<RANGED_LOWP>::exp(f32::MAX).unwrap().to_bits()
    );
    assert_eq!(
        Expf::<DEFAULT_LOWP>::exp(-f32::MAX).unwrap().to_bits(),
        Expf::<RANGED_LOWP>::exp(-f32::MAX).unwrap().to_bits()
    );
}

#[test]
fn highp_slice_reports_earliest_failing_index_across_mixed_error_stages() {
    const CHECKED_HIGHP: u128 = PowfOptions::new().highp().bits();
    // Lane 0 overflows normal output (`OutsideApproximationRange`), lane 1 fails base validation (`InvalidBase`).
    let chunk_a = [f32::MAX, f32::NAN, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0];
    // Lane 1 fails base validation before lane 2 overflows normal output.
    let chunk_b = [1.0, f32::NAN, f32::MAX, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0];
    let mut dst = [0.0; 9];

    assert_eq!(
        Powf::<CHECKED_HIGHP>::slice(&chunk_a, &mut dst, 2.0),
        Err(SliceError {
            kind: PowfErrorKind::OutsideApproximationRange,
            index: 0,
        })
    );
    assert_eq!(
        PowfOptions::new()
            .highp()
            .runtime()
            .slice(&chunk_a, &mut dst, 2.0),
        Err(SliceError {
            kind: PowfErrorKind::OutsideApproximationRange,
            index: 0,
        })
    );
    assert_eq!(
        Powf::<CHECKED_HIGHP>::slice(&chunk_b, &mut dst, 2.0),
        Err(SliceError {
            kind: PowfErrorKind::InvalidBase,
            index: 1,
        })
    );

    #[cfg(feature = "std")]
    {
        const STD_HIGHP: u128 = PowfOptions::new().highp().backend_dependent().bits();
        assert_eq!(
            Powf::<STD_HIGHP>::slice(&chunk_a, &mut dst, 2.0),
            Err(SliceError {
                kind: PowfErrorKind::OutsideApproximationRange,
                index: 0,
            })
        );
        assert_eq!(
            PowfOptions::new()
                .highp()
                .backend_dependent()
                .runtime()
                .slice(&chunk_a, &mut dst, 2.0),
            Err(SliceError {
                kind: PowfErrorKind::OutsideApproximationRange,
                index: 0,
            })
        );
        assert_eq!(
            Powf::<STD_HIGHP>::slice(&chunk_b, &mut dst, 2.0),
            Err(SliceError {
                kind: PowfErrorKind::InvalidBase,
                index: 1,
            })
        );
    }
}

#[test]
fn logf_granular_zero_vs_subnormals_vs_full_domain() {
    const ZERO_ONLY: u128 = LogfOptions::new().admit_zero().bits();
    const SUB_ONLY: u128 = LogfOptions::new().admit_subnormals().bits();
    const FULL_MIDP: u128 = LogfOptions::new().admit_all_floats().bits();
    const FULL_LOWP: u128 = LogfOptions::new().admit_all_floats().lowp().bits();

    // admit_zero accepts ±0.0 -> -inf and log1p(-1.0) -> -inf, while rejecting subnormals.
    assert_eq!(Logf::<ZERO_ONLY>::log2(0.0).unwrap(), f32::NEG_INFINITY);
    assert_eq!(Logf::<ZERO_ONLY>::log2(-0.0).unwrap(), f32::NEG_INFINITY);
    assert_eq!(Logf::<ZERO_ONLY>::log1p(-1.0).unwrap(), f32::NEG_INFINITY);
    assert!(Logf::<ZERO_ONLY>::log2(f32::from_bits(1)).is_err());

    // admit_subnormals normalizes positive subnormals accurately while rejecting ±0.0.
    assert!(Logf::<SUB_ONLY>::log2(0.0).is_err());
    assert_eq!(Logf::<SUB_ONLY>::log2(f32::from_bits(1)).unwrap(), -149.0);
    assert_eq!(Logf::<FULL_MIDP>::log2(f32::from_bits(1)).unwrap(), -149.0);
    assert!((Logf::<FULL_LOWP>::log2(f32::from_bits(1)).unwrap() - (-149.0)).abs() < 1e-4);

    // Non-power-of-two subnormals across all log variants.
    for shift in [0_u32, 1, 7, 15, 22] {
        let sub = f32::from_bits((1_u32 << shift) | (shift * 3));
        let exact_log2 = (sub as f64).log2() as f32;
        assert_close_rel(
            Logf::<FULL_MIDP>::log2(sub).unwrap(),
            exact_log2,
            1e-6,
            "midp subnorm log2",
        );
        assert_close_rel(
            Logf::<FULL_MIDP>::ln(sub).unwrap(),
            (sub as f64).ln() as f32,
            1e-6,
            "midp subnorm ln",
        );
        assert_close_rel(
            Logf::<FULL_MIDP>::log10(sub).unwrap(),
            (sub as f64).log10() as f32,
            1e-6,
            "midp subnorm log10",
        );
        assert!(
            (Logf::<FULL_LOWP>::log2(sub).unwrap() - exact_log2).abs() < 2e-5,
            "lowp subnorm log2"
        );
    }
}

#[test]
fn positive_gamma_clears_prior_zero_and_subnormal_flags() {
    let expected = PowfOptions::new().positive_gamma_domain();
    assert_eq!(PowfOptions::new().gamma().positive_gamma(), expected);
    assert_eq!(PowfOptions::new().admit_zero().positive_gamma(), expected);
    assert_eq!(
        PowfOptions::new()
            .admit_zero()
            .admit_subnormals()
            .positive_gamma_domain(),
        expected
    );
    const PG: u128 = PowfOptions::new().gamma().positive_gamma().bits();
    assert!(Powf::<PG>::scalar(0.0, 2.2).is_err());
    assert!(Powf::<PG>::scalar(f32::from_bits(1), 0.5).is_err());
}

#[test]
fn subnormal_only_intervals_rejected_until_admit_subnormals_enabled() {
    let sub_exp = f32::from_bits(0x80);
    let bad_pow = PowfOptions::new().exponent_range(sub_exp, sub_exp);
    assert!(!bad_pow.is_valid());
    assert_eq!(
        bad_pow.runtime().scalar(2.0, sub_exp).unwrap_err().kind,
        PowfErrorKind::UnsupportedOptions
    );

    let ok_pow = bad_pow.admit_subnormals();
    assert!(ok_pow.is_valid());
    assert_eq!(PowfOptions::from_bits(ok_pow.bits()), Some(ok_pow));
    assert_eq!(ok_pow.runtime().scalar(2.0, sub_exp).unwrap(), 1.0);

    // MAX_EXP_BOUND is accepted at both poles.
    let wide_exp = PowfOptions::new().exponent_range(-MAX_EXP_BOUND, MAX_EXP_BOUND);
    assert!(wide_exp.is_valid());
    assert_eq!(PowfOptions::from_bits(wide_exp.bits()), Some(wide_exp));

    let sub_x = f32::from_bits(0x100);
    let bad_cbrt = CbrtfOptions::new().input_range(f32::from_bits(1), sub_x);
    assert!(!bad_cbrt.is_valid());
    assert_eq!(
        bad_cbrt.runtime().scalar(sub_x).unwrap_err().kind,
        CbrtfErrorKind::UnsupportedOptions
    );

    let ok_cbrt = bad_cbrt.admit_subnormals();
    assert!(ok_cbrt.is_valid());
    assert_eq!(CbrtfOptions::from_bits(ok_cbrt.bits()), Some(ok_cbrt));
    // 2^-147 -> cbrt = 2^-49.
    let x_147 = f32::from_bits(1 << 2);
    assert_close_rel(
        ok_cbrt.runtime().scalar(x_147).unwrap(),
        f32::from_bits((127 - 49) << 23),
        1e-6,
        "subnormal cbrt",
    );
}

// ============================================================================
// 2. Polynomial fitting & zero-reduction range verification
// ============================================================================

#[test]
fn polynomial_single_octave_zero_reduction_matches_full_graph_bit_for_bit() {
    const SMALL_EXP2_M: u128 = ExpfOptions::new().small_exp2_range().bits();
    const SMALL_EXP_M: u128 = ExpfOptions::new().small_exp_range().bits();
    const SMALL_EXP2_L: u128 = ExpfOptions::new()
        .input_range(0.0, 0.999_999_94)
        .lowp()
        .bits();
    const UNCLAMP_EXP2: u128 = ExpfOptions::new().unclamped_exp2_range().bits();
    const UNCLAMP_EXP: u128 = ExpfOptions::new().unclamped_exp_range().bits();

    assert!(
        ExpfOptions::from_bits(SMALL_EXP2_M)
            .unwrap()
            .elides_reduction()
    );
    assert!(
        ExpfOptions::from_bits(SMALL_EXP_M)
            .unwrap()
            .elides_exp_reduction()
    );
    assert!(
        ExpfOptions::from_bits(SMALL_EXP2_L)
            .unwrap()
            .elides_reduction()
    );
    assert!(ExpfOptions::from_bits(UNCLAMP_EXP2).unwrap().elides_clamp());
    assert!(
        ExpfOptions::from_bits(UNCLAMP_EXP)
            .unwrap()
            .elides_exp_clamp()
    );

    for i in 0..=512 {
        let t = -0.5 + (i as f32) / 512.0;
        let x = if i == 256 { 0.0 } else { t };
        assert_eq!(
            Expf::<SMALL_EXP2_M>::exp2(x).unwrap().to_bits(),
            <Expf>::exp2(x).unwrap().to_bits(),
            "exp2 midp zero-reduction mismatch at {x}"
        );
        let xe = x * (core::f32::consts::LN_2 * 0.999_999_9);
        assert_eq!(
            Expf::<SMALL_EXP_M>::exp(xe).unwrap().to_bits(),
            <Expf>::exp(xe).unwrap().to_bits(),
            "exp midp zero-reduction mismatch at {xe}"
        );
    }
    assert_eq!(
        Expf::<SMALL_EXP2_M>::exp2(-0.0).unwrap().to_bits(),
        <Expf>::exp2(-0.0).unwrap().to_bits()
    );
    for i in 0..256 {
        let x = (i as f32) / 256.0;
        assert_eq!(
            Expf::<SMALL_EXP2_L>::exp2(x).unwrap().to_bits(),
            Expf::<{ ExpfOptions::new().lowp().bits() }>::exp2(x)
                .unwrap()
                .to_bits(),
            "exp2 lowp zero-reduction mismatch at {x}"
        );
    }

    const NEAR_ONE_M: u128 = LogfOptions::new().near_one_range().bits();
    const SMALL_LOG1P_M: u128 = LogfOptions::new().small_log1p_range().bits();
    const SMALL_LOG1P_L: u128 = LogfOptions::new().small_log1p_range().lowp().bits();
    assert!(
        LogfOptions::from_bits(NEAR_ONE_M)
            .unwrap()
            .elides_reduction()
    );
    assert!(
        LogfOptions::from_bits(SMALL_LOG1P_M)
            .unwrap()
            .elides_log1p_reduction()
    );
    assert!(
        LogfOptions::from_bits(SMALL_LOG1P_L)
            .unwrap()
            .elides_log1p_reduction()
    );

    let lo_b = 0x3f35_04f3_u32;
    let hi_b = 0x3fb5_04f2_u32;
    let step = (hi_b - lo_b) / 512;
    for k in 0..=512 {
        let bits = if k == 512 { hi_b } else { lo_b + k * step };
        let x = f32::from_bits(bits);
        assert_eq!(
            Logf::<NEAR_ONE_M>::log2(x).unwrap().to_bits(),
            <Logf>::log2(x).unwrap().to_bits(),
            "log2 midp near-one mismatch at {x}"
        );
        assert_eq!(
            Logf::<NEAR_ONE_M>::ln(x).unwrap().to_bits(),
            <Logf>::ln(x).unwrap().to_bits(),
            "ln midp near-one mismatch at {x}"
        );
    }
    assert_eq!(
        Logf::<NEAR_ONE_M>::log2(1.0).unwrap().to_bits(),
        <Logf>::log2(1.0).unwrap().to_bits()
    );

    for i in 0..=256 {
        let x = -0.25 + (i as f32) / 512.0;
        assert_eq!(
            Logf::<SMALL_LOG1P_M>::log1p(x).unwrap().to_bits(),
            <Logf>::log1p(x).unwrap().to_bits(),
            "log1p midp mismatch at {x}"
        );
        assert_eq!(
            Logf::<SMALL_LOG1P_L>::log1p(x).unwrap().to_bits(),
            Logf::<{ LogfOptions::new().lowp().bits() }>::log1p(x)
                .unwrap()
                .to_bits(),
            "log1p lowp mismatch at {x}"
        );
    }
    assert_eq!(
        Logf::<SMALL_LOG1P_M>::log1p(-0.0).unwrap().to_bits(),
        (-0.0_f32).to_bits()
    );

    // Sign-aware srgb_encode_domain and unit_gamma_domain elide range checks!
    const SRGB_ENC: PowfOptions = PowfOptions::new().srgb_encode_domain();
    const UNIT_GAMMA: PowfOptions = PowfOptions::new().unit_gamma_domain();
    const UNIT_CBRT: CbrtfOptions = CbrtfOptions::new().unit_range();
    assert!(SRGB_ENC.elides_range_check());
    assert!(UNIT_GAMMA.elides_range_check());
    assert!(UNIT_CBRT.elides_denorm_check());
    assert_close_rel(
        Cbrtf::<{ UNIT_CBRT.bits() }>::scalar(-0.125).unwrap(),
        -0.5,
        1e-6,
        "unit_range cbrt",
    );
    assert_close_rel(
        Powf::<{ SRGB_ENC.bits() }>::scalar(f32::MIN_POSITIVE, 1.0 / 2.4).unwrap(),
        (f32::MIN_POSITIVE as f64).powf(1.0 / 2.4) as f32,
        1e-5,
        "srgb encode min_positive",
    );
}

#[test]
fn infallible_clamp_out_of_bounds_works_inline_on_simd_math_chains() {
    const POW_CLAMP: u128 = PowfOptions::new()
        .unit_gamma_domain()
        .clamp_out_of_bounds()
        .bits();
    const EXP_CLAMP: u128 = ExpfOptions::new()
        .input_range(-2.0, 2.0)
        .clamp_out_of_bounds()
        .bits();
    const LOG_CLAMP: u128 = LogfOptions::new()
        .input_range(0.25, 4.0)
        .admit_zero()
        .clamp_out_of_bounds()
        .bits();
    const CBRT_CLAMP: u128 = CbrtfOptions::new()
        .unit_range()
        .clamp_out_of_bounds()
        .bits();

    let v = f32x4::from_array_t(ScalarToken, [-0.25, 0.25, 1.5, f32::NAN]);
    let p = v.powf::<POW_CLAMP>(0.5).to_array();
    assert_eq!(p[0], 0.0);
    assert_close_rel(p[1], 0.5, 1e-6, "clamped pow in-bounds");
    assert_eq!(p[2], 1.0);
    assert!(p[3].is_nan());

    let e = f32x4::from_array_t(ScalarToken, [-10.0, 0.0, 10.0, f32::NAN])
        .exp2::<EXP_CLAMP>()
        .to_array();
    assert_close_rel(e[0], 0.25, 1e-6, "clamped exp2 lo");
    assert_eq!(e[1], 1.0);
    assert_close_rel(e[2], 4.0, 1e-6, "clamped exp2 hi");
    assert!(e[3].is_nan());

    let l = f32x4::from_array_t(ScalarToken, [-5.0, 0.125, 16.0, f32::NAN])
        .log2::<LOG_CLAMP>()
        .to_array();
    assert_eq!(l[0], f32::NEG_INFINITY); // <= 0 clamps to +0.0 -> -inf under admit_zero
    assert_close_rel(l[1], -2.0, 1e-6, "0.125 clamps to 0.25 -> -2");
    assert_close_rel(l[2], 2.0, 1e-6, "16 clamps to 4 -> +2");
    assert!(l[3].is_nan());

    let c = f32x4::from_array_t(ScalarToken, [-8.0, -0.125, 8.0, f32::NAN])
        .cbrt::<CBRT_CLAMP>()
        .to_array();
    assert_close_rel(c[0], -1.0, 1e-6, "clamped cbrt lo");
    assert_close_rel(c[1], -0.5, 1e-6, "clamped cbrt mid");
    assert_close_rel(c[2], 1.0, 1e-6, "clamped cbrt hi");
    assert!(c[3].is_nan());
}

// ============================================================================
// 3. Single-constraint-held exhaustive permutation compliance across all 4 families
// ============================================================================

fn verify_powf_option_compliance(opts: PowfOptions, in_exp: f32) {
    if !opts.is_valid() {
        assert_eq!(PowfOptions::from_bits(opts.bits()), None);
        assert_eq!(
            opts.runtime().scalar(1.0, 1.0).unwrap_err().kind,
            PowfErrorKind::UnsupportedOptions
        );
        let mut dst = [0.0; 4];
        assert_eq!(
            opts.runtime()
                .slice(&[1.0; 4], &mut dst, 1.0)
                .unwrap_err()
                .kind,
            PowfErrorKind::UnsupportedOptions
        );
        return;
    }

    let encoded = opts.bits();
    assert_eq!(PowfOptions::from_bits(encoded), Some(opts));
    let rt = opts.runtime();

    let decoded = PowfOptions::from_bits(encoded).unwrap();
    let is_full = decoded == opts.admit_all_floats();
    let is_clamp_oob = decoded == opts.clamp_out_of_bounds();
    let is_assume = decoded == opts.assume_in_bounds();
    let is_checked = !is_full && !is_clamp_oob && !is_assume;

    // 1. In-bounds agreement across scalar, array<4>, array<8>, f32x4, f32x8, f32x16, and slice(19).
    let in_bases4 = [0.25_f32, 0.5, 0.75, 1.0];
    let n = in_exp;
    let s0 = rt.scalar(in_bases4[0], n).unwrap();
    let a4 = rt.array(in_bases4, n).unwrap();
    assert_bits_or_nan(a4[0], s0, "powf scalar vs array4");

    let v4 = f32x4::from_array_t(ScalarToken, in_bases4);
    let simd4 = rt.simd(v4, n).unwrap().to_array();
    for i in 0..4 {
        assert_bits_or_nan(simd4[i], a4[i], "powf simd4 vs array4");
    }

    let in_bases8: [f32; 8] = core::array::from_fn(|i| in_bases4[i % 4]);
    let a8 = rt.array(in_bases8, n).unwrap();
    let simd8 = rt
        .simd(f32x8::from_array_t(ScalarToken, in_bases8), n)
        .unwrap()
        .to_array();
    for i in 0..8 {
        assert_bits_or_nan(a8[i], a4[i % 4], "powf array8 vs array4");
        assert_bits_or_nan(simd8[i], a8[i], "powf simd8 vs array8");
    }

    #[cfg(feature = "w512")]
    {
        let in_bases16: [f32; 16] = core::array::from_fn(|i| in_bases4[i % 4]);
        let a16 = rt.array(in_bases16, n).unwrap();
        let simd16 = rt
            .simd(f32x16::from_array_t(ScalarToken, in_bases16), n)
            .unwrap()
            .to_array();
        for i in 0..16 {
            assert_bits_or_nan(a16[i], a4[i % 4], "powf array16 vs array4");
            assert_bits_or_nan(simd16[i], a16[i], "powf simd16 vs array16");
        }
    }

    let slice_in: Vec<f32> = (0..19).map(|i| in_bases4[i % 4]).collect();
    let mut slice_out = vec![-1.0; 19];
    rt.slice(&slice_in, &mut slice_out, n).unwrap();
    for i in 0..19 {
        assert_bits_or_nan(slice_out[i], a4[i % 4], "powf slice19 vs array4");
    }

    // 2. Out-of-bounds / special-value behavior according to Validation & Domain.
    if is_checked {
        let mut bad = [0.5_f32; 9];
        bad[7] = -1.0;
        let mut dst = [0.0; 9];
        assert_eq!(
            rt.slice(&bad, &mut dst, n),
            Err(SliceError {
                kind: PowfErrorKind::InvalidBase,
                index: 7,
            })
        );
        assert_eq!(
            rt.array([0.5, -1.0, f32::NAN, 0.5], n)
                .unwrap_err()
                .lane_mask,
            0b0110
        );
        assert_eq!(
            rt.scalar(0.5, f32::NAN).unwrap_err().kind,
            PowfErrorKind::InvalidExponent
        );
    } else if is_clamp_oob {
        // Must never return Err on any float input or exponent!
        for &exp in &[
            -10.0_f32,
            -0.0,
            0.0,
            f32::from_bits(1),
            0.5,
            10.0,
            f32::INFINITY,
            f32::NAN,
        ] {
            let probe = [
                f32::NEG_INFINITY,
                -2.0,
                -0.0,
                0.0,
                f32::from_bits(1),
                0.5,
                1e6,
                f32::NAN,
            ];
            let res8 = rt
                .array(probe, exp)
                .expect("clamp_out_of_bounds must be infallible");
            let mut dst = [0.0; 8];
            rt.slice(&probe, &mut dst, exp)
                .expect("clamp_out_of_bounds slice must be infallible");
            for i in 0..8 {
                assert_bits_or_nan(dst[i], res8[i], "powf clamp_oob slice vs array8");
            }
            if exp.is_nan() {
                if is_full {
                    // IEEE 754 pow(x, qNaN): 1.0^NaN == 1.0, other lanes NaN.
                    assert!(res8[5].is_nan());
                } else {
                    assert!(res8.iter().all(|v| v.is_nan()));
                }
            } else if is_full && exp == 0.0 {
                // IEEE 754 pow(qNaN, ±0.0) == 1.0 under full_domain Highp.
                assert_eq!(res8[7], 1.0);
            } else {
                assert!(res8[7].is_nan(), "NaN base must propagate NaN");
                assert!(!res8[5].is_nan(), "finite in-bounds base must not be NaN");
            }
        }
    } else if is_full {
        let probe = [
            -2.0_f32,
            -0.0,
            0.0,
            f32::from_bits(1),
            0.5,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ];
        let res = rt
            .array(probe, 2.0)
            .expect("full_domain highp is infallible");
        assert_eq!(res[0], 4.0);
        assert_eq!(res[2], 0.0);
        assert_eq!(res[5], f32::INFINITY);
        assert!(res[7].is_nan());
    }
}

#[test]
fn powf_hold_each_constraint_across_all_other_permutations() {
    let bases: [Option<(f32, f32)>; 3] = [
        None,
        Some((1.0 / 65_536.0, 65_536.0)),
        Some((f32::MIN_POSITIVE, 1.0)),
    ];
    let exps: [Option<(f32, f32)>; 4] = [
        None,
        Some((0.0, 0.99)),
        Some((-2.0, 4.0)),
        Some((f32::from_bits(0x80), f32::from_bits(0x80))),
    ];
    // 0=none, 1=zero, 2=subnorm, 3=zero+subnorm, 4=all_floats
    let doms = [0_u8, 1, 2, 3, 4];
    let clamps = [false, true];
    let precs = [
        PowfPrecision::Lowp,
        PowfPrecision::Midp,
        PowfPrecision::Highp,
    ];
    let ports = [Portability::BitExact, Portability::BackendDependent];
    let vals = [
        Validation::Checked,
        Validation::ClampOutOfBounds,
        Validation::AssumeValid,
    ];
    let execs = [Execution::Upgrade, Execution::Caller];

    let build = |b: Option<(f32, f32)>,
                 e: Option<(f32, f32)>,
                 d: u8,
                 c: bool,
                 pr: PowfPrecision,
                 po: Portability,
                 va: Validation,
                 ex: Execution| {
        let mut o = PowfOptions::new()
            .precision(pr)
            .portability(po)
            .validation(va)
            .execution(ex);
        if let Some((lo, hi)) = b {
            o = o.base_range(lo, hi);
        }
        if let Some((lo, hi)) = e {
            o = o.exponent_range(lo, hi);
        }
        match d {
            1 => o = o.admit_zero(),
            2 => o = o.admit_subnormals(),
            3 => o = o.admit_zero().admit_subnormals(),
            4 => o = o.admit_all_floats(),
            _ => {}
        }
        if c {
            o = o.saturate_output();
        }
        o
    };

    // Track how many valid permutations each held constraint exercised so no
    // constraint can vacuously pass with zero valid combinations.
    let mut valid_counts = [0_usize; 14];

    for &b in &bases {
        for &e in &exps {
            for &d in &doms {
                for &c in &clamps {
                    for &pr in &precs {
                        for &po in &ports {
                            for &va in &vals {
                                for &ex in &execs {
                                    let opts = build(b, e, d, c, pr, po, va, ex);
                                    let sample_exp = match e {
                                        Some((lo, hi)) if lo == hi => lo,
                                        _ => 0.5,
                                    };
                                    verify_powf_option_compliance(opts, sample_exp);
                                    if opts.is_valid() {
                                        if matches!(d, 1 | 3)
                                            && e.is_none_or(|(lo, hi)| lo <= 0.0 && hi >= 0.5)
                                        {
                                            valid_counts[0] += 1;
                                            assert_eq!(
                                                opts.runtime().scalar(0.0, 0.0).unwrap(),
                                                1.0
                                            );
                                            assert_eq!(
                                                opts.runtime().scalar(0.0, 0.5).unwrap(),
                                                0.0
                                            );
                                        }
                                        if matches!(d, 2 | 3) {
                                            valid_counts[1] += 1;
                                            let sub = f32::from_bits(1 << 8); // 2^-141
                                            let got =
                                                opts.runtime().scalar(sub, sample_exp).unwrap();
                                            assert!(got.is_normal() && got > 0.0);
                                        }
                                        if d == 4 {
                                            valid_counts[2] += 1;
                                        }
                                        if b.is_some() {
                                            valid_counts[3] += 1;
                                        }
                                        if e.is_some() {
                                            valid_counts[4] += 1;
                                        }
                                        if c {
                                            valid_counts[5] += 1;
                                        }
                                        if pr == PowfPrecision::Lowp {
                                            valid_counts[6] += 1;
                                        }
                                        if pr == PowfPrecision::Midp {
                                            valid_counts[7] += 1;
                                        }
                                        if pr == PowfPrecision::Highp {
                                            valid_counts[8] += 1;
                                        }
                                        if po == Portability::BitExact {
                                            valid_counts[9] += 1;
                                        }
                                        if va == Validation::Checked {
                                            valid_counts[10] += 1;
                                        }
                                        if va == Validation::ClampOutOfBounds {
                                            valid_counts[11] += 1;
                                        }
                                        if va == Validation::AssumeValid {
                                            valid_counts[12] += 1;
                                        }
                                        if ex == Execution::Caller {
                                            valid_counts[13] += 1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for (idx, &count) in valid_counts.iter().enumerate() {
        assert!(
            count > 0,
            "powf held constraint index {idx} had 0 valid permutations"
        );
    }
}

#[test]
fn expf_hold_each_constraint_across_all_other_permutations() {
    let ranges: [Option<(f32, f32)>; 4] = [
        None,
        Some((-0.3, 0.3)),
        Some((0.0, 0.5)),
        Some((-20.0, 20.0)),
    ];
    let fulls = [false, true];
    let precs = [ExpfPrecision::Midp, ExpfPrecision::Lowp];
    let ports = [Portability::BitExact, Portability::BackendDependent];
    let vals = [
        Validation::Checked,
        Validation::ClampOutOfBounds,
        Validation::AssumeValid,
    ];
    let execs = [Execution::Upgrade, Execution::Caller];

    let mut valid_counts = [0_usize; 9];
    let in_vals = [0.0_f32, 0.1, 0.2, 0.25];

    for &r in &ranges {
        for &full in &fulls {
            for &pr in &precs {
                for &po in &ports {
                    for &va in &vals {
                        for &ex in &execs {
                            let mut opts = ExpfOptions::new()
                                .precision(pr)
                                .portability(po)
                                .validation(va)
                                .execution(ex);
                            if let Some((lo, hi)) = r {
                                opts = opts.input_range(lo, hi);
                            }
                            if full {
                                opts = opts.admit_all_floats();
                            }

                            if !opts.is_valid() {
                                assert_eq!(ExpfOptions::from_bits(opts.bits()), None);
                                assert_eq!(
                                    opts.runtime().exp2(0.1).unwrap_err().kind,
                                    ExpfErrorKind::UnsupportedOptions
                                );
                                continue;
                            }

                            assert_eq!(ExpfOptions::from_bits(opts.bits()), Some(opts));
                            let rt = opts.runtime();
                            let a4_2 = rt.exp2_array(in_vals).unwrap();
                            let a4_e = rt.exp_array(in_vals).unwrap();
                            let v4 = f32x4::from_array_t(ScalarToken, in_vals);
                            let s4_2 = rt.exp2_simd(v4).unwrap().to_array();
                            let s4_e = rt.exp_simd(v4).unwrap().to_array();
                            let mut sl_2 = [0.0; 11];
                            let mut sl_e = [0.0; 11];
                            let sl_in: [f32; 11] = core::array::from_fn(|i| in_vals[i % 4]);
                            rt.exp2_slice(&sl_in, &mut sl_2).unwrap();
                            rt.exp_slice(&sl_in, &mut sl_e).unwrap();
                            for i in 0..11 {
                                assert_bits_or_nan(sl_2[i], a4_2[i % 4], "exp2 slice vs array");
                                assert_bits_or_nan(sl_e[i], a4_e[i % 4], "exp slice vs array");
                            }
                            for i in 0..4 {
                                assert_bits_or_nan(s4_2[i], a4_2[i], "exp2 simd vs array");
                                assert_bits_or_nan(s4_e[i], a4_e[i], "exp simd vs array");
                            }

                            if va == Validation::Checked && !full {
                                valid_counts[0] += 1;
                                assert!(rt.exp2(f32::NAN).is_err());
                                assert!(rt.exp(f32::INFINITY).is_err());
                                if let Some((_, hi)) = r {
                                    assert!(rt.exp2(hi + 1.0).is_err());
                                }
                            }
                            if va == Validation::ClampOutOfBounds {
                                valid_counts[1] += 1;
                                let oob = [f32::NEG_INFINITY, -500.0, 500.0, f32::NAN];
                                let out2 = rt.exp2_array(oob).unwrap();
                                let oute = rt.exp_array(oob).unwrap();
                                assert!(out2[3].is_nan() && oute[3].is_nan());
                                assert!(!out2[0].is_nan() && !oute[0].is_nan());
                            }
                            if va == Validation::AssumeValid {
                                valid_counts[2] += 1;
                            }
                            if full {
                                valid_counts[3] += 1;
                                assert_eq!(rt.exp2(f32::NEG_INFINITY).unwrap(), 0.0);
                                assert_eq!(rt.exp(f32::INFINITY).unwrap(), f32::INFINITY);
                                assert!(rt.exp2(f32::NAN).unwrap().is_nan());
                            }
                            if r.is_some() {
                                valid_counts[4] += 1;
                            }
                            if pr == ExpfPrecision::Lowp {
                                valid_counts[5] += 1;
                            }
                            if pr == ExpfPrecision::Midp {
                                valid_counts[6] += 1;
                            }
                            if po == Portability::BitExact {
                                valid_counts[7] += 1;
                            }
                            if ex == Execution::Caller {
                                valid_counts[8] += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    for (idx, &count) in valid_counts.iter().enumerate() {
        assert!(count > 0, "expf held constraint {idx} had 0 valid cases");
    }
}

#[test]
fn logf_hold_each_constraint_across_all_other_permutations() {
    let ranges: [Option<(f32, f32)>; 5] = [
        None,
        Some((0.75, 1.25)),
        Some((-0.25, 0.25)),
        Some((0.0, 1.0)),
        Some((-1.0, 1.0)),
    ];
    // 0=none, 1=zero, 2=subnorm, 3=zero+subnorm, 4=all_floats
    let doms = [0_u8, 1, 2, 3, 4];
    let precs = [LogfPrecision::Midp, LogfPrecision::Lowp];
    let ports = [Portability::BitExact, Portability::BackendDependent];
    let vals = [
        Validation::Checked,
        Validation::ClampOutOfBounds,
        Validation::AssumeValid,
    ];
    let execs = [Execution::Upgrade, Execution::Caller];

    let mut valid_counts = [0_usize; 11];
    let pos_in = [0.8_f32, 0.85, 0.9, 1.0];
    let log1p_in = [-0.1_f32, 0.0, 0.1, 0.2];

    for &r in &ranges {
        for &d in &doms {
            for &pr in &precs {
                for &po in &ports {
                    for &va in &vals {
                        for &ex in &execs {
                            let mut opts = LogfOptions::new()
                                .precision(pr)
                                .portability(po)
                                .validation(va)
                                .execution(ex);
                            if let Some((lo, hi)) = r {
                                opts = opts.input_range(lo, hi);
                            }
                            match d {
                                1 => opts = opts.admit_zero(),
                                2 => opts = opts.admit_subnormals(),
                                3 => opts = opts.admit_zero().admit_subnormals(),
                                4 => opts = opts.admit_all_floats(),
                                _ => {}
                            }

                            let rt = opts.runtime();
                            // Check log2 / ln / log10 (`op` 0..=2)
                            if let Ok(a4) = rt.log2_array(pos_in) {
                                let v4 = f32x4::from_array_t(ScalarToken, pos_in);
                                let s4 = rt.log2_simd(v4).unwrap().to_array();
                                let mut sl = [0.0; 10];
                                let sl_in: [f32; 10] = core::array::from_fn(|i| pos_in[i % 4]);
                                rt.log2_slice(&sl_in, &mut sl).unwrap();
                                for i in 0..10 {
                                    assert_bits_or_nan(sl[i], a4[i % 4], "log2 slice vs array");
                                }
                                for i in 0..4 {
                                    assert_bits_or_nan(s4[i], a4[i], "log2 simd vs array");
                                }

                                if matches!(d, 1 | 3 | 4) {
                                    valid_counts[0] += 1;
                                    assert_eq!(rt.log2(0.0).unwrap(), f32::NEG_INFINITY);
                                    assert_eq!(rt.ln(-0.0).unwrap(), f32::NEG_INFINITY);
                                }
                                if matches!(d, 2..=4)
                                    && r.is_none_or(|(lo, _)| lo <= f32::from_bits(1))
                                {
                                    valid_counts[1] += 1;
                                    let sub_res = rt.log2(f32::from_bits(1)).unwrap();
                                    assert!((sub_res - (-149.0)).abs() < 1e-3);
                                }
                                if d == 4 {
                                    valid_counts[2] += 1;
                                    assert!(rt.ln(-2.0).unwrap().is_nan());
                                    assert_eq!(rt.log10(f32::INFINITY).unwrap(), f32::INFINITY);
                                }
                                if va == Validation::Checked && d != 4 {
                                    valid_counts[3] += 1;
                                    assert!(rt.log2(-2.0).is_err());
                                    assert!(rt.log2(f32::NAN).is_err());
                                }
                                if va == Validation::ClampOutOfBounds {
                                    valid_counts[4] += 1;
                                    let oob = [f32::NEG_INFINITY, -1.0, f32::INFINITY, f32::NAN];
                                    let c_out = rt.log2_array(oob).unwrap();
                                    assert!(c_out[3].is_nan());
                                    if d == 4 {
                                        assert!(c_out[0].is_nan());
                                    } else {
                                        assert!(!c_out[0].is_nan());
                                    }
                                }
                                if va == Validation::AssumeValid {
                                    valid_counts[5] += 1;
                                }
                                if r.is_some() {
                                    valid_counts[6] += 1;
                                }
                                if pr == LogfPrecision::Lowp {
                                    valid_counts[7] += 1;
                                }
                                if pr == LogfPrecision::Midp {
                                    valid_counts[8] += 1;
                                }
                                if po == Portability::BitExact {
                                    valid_counts[9] += 1;
                                }
                                if ex == Execution::Caller {
                                    valid_counts[10] += 1;
                                }
                            } else {
                                assert_eq!(
                                    rt.log2(1.0).unwrap_err().kind,
                                    LogfErrorKind::UnsupportedOptions
                                );
                            }

                            // Check log1p (`op == 3`) when range covers `log1p_in`.
                            let covers_log1p = r.is_none_or(|(lo, hi)| lo <= -0.1 && hi >= 0.2);
                            if covers_log1p && let Ok(a4_1p) = rt.log1p_array(log1p_in) {
                                let mut sl = [0.0; 9];
                                let sl_in: [f32; 9] = core::array::from_fn(|i| log1p_in[i % 4]);
                                rt.log1p_slice(&sl_in, &mut sl).unwrap();
                                for i in 0..9 {
                                    assert_bits_or_nan(sl[i], a4_1p[i % 4], "log1p slice vs array");
                                }
                                if matches!(d, 1 | 3 | 4) {
                                    assert_eq!(rt.log1p(-1.0).unwrap(), f32::NEG_INFINITY);
                                }
                                if va == Validation::ClampOutOfBounds {
                                    let oob = [f32::NEG_INFINITY, -5.0, f32::INFINITY, f32::NAN];
                                    let c1p = rt.log1p_array(oob).unwrap();
                                    assert!(c1p[3].is_nan());
                                    if d == 4 {
                                        assert!(c1p[1].is_nan());
                                    } else {
                                        assert!(!c1p[1].is_nan());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for (idx, &count) in valid_counts.iter().enumerate() {
        assert!(count > 0, "logf held constraint {idx} had 0 valid cases");
    }
}

#[test]
fn cbrtf_hold_each_constraint_across_all_other_permutations() {
    let ranges: [Option<(f32, f32)>; 4] = [
        None,
        Some((-1.0, 1.0)),
        Some((0.125, 8.0)),
        Some((f32::from_bits(1), f32::from_bits(0x100))),
    ];
    // 0=none, 1=subnorm, 2=all_floats
    let doms = [0_u8, 1, 2];
    let precs = [CbrtfPrecision::Midp, CbrtfPrecision::Lowp];
    let ports = [Portability::BitExact, Portability::BackendDependent];
    let vals = [
        Validation::Checked,
        Validation::ClampOutOfBounds,
        Validation::AssumeValid,
    ];
    let execs = [Execution::Upgrade, Execution::Caller];

    let mut valid_counts = [0_usize; 10];
    let normal_in = [0.125_f32, 0.25, 0.5, 1.0];

    for &r in &ranges {
        for &d in &doms {
            for &pr in &precs {
                for &po in &ports {
                    for &va in &vals {
                        for &ex in &execs {
                            let mut opts = CbrtfOptions::new()
                                .precision(pr)
                                .portability(po)
                                .validation(va)
                                .execution(ex);
                            if let Some((lo, hi)) = r {
                                opts = opts.input_range(lo, hi);
                            }
                            match d {
                                1 => opts = opts.admit_subnormals(),
                                2 => opts = opts.admit_all_floats(),
                                _ => {}
                            }

                            if !opts.is_valid() {
                                assert_eq!(CbrtfOptions::from_bits(opts.bits()), None);
                                assert_eq!(
                                    opts.runtime().scalar(0.5).unwrap_err().kind,
                                    CbrtfErrorKind::UnsupportedOptions
                                );
                                continue;
                            }

                            assert_eq!(CbrtfOptions::from_bits(opts.bits()), Some(opts));
                            let rt = opts.runtime();

                            let sample: [f32; 4] =
                                if r == Some((f32::from_bits(1), f32::from_bits(0x100))) {
                                    [
                                        f32::from_bits(1),
                                        f32::from_bits(4),
                                        f32::from_bits(16),
                                        f32::from_bits(64),
                                    ]
                                } else {
                                    normal_in
                                };

                            let a4 = rt.array(sample).unwrap();
                            let v4 = f32x4::from_array_t(ScalarToken, sample);
                            let s4 = rt.simd(v4).unwrap().to_array();
                            let mut sl = [0.0; 11];
                            let sl_in: [f32; 11] = core::array::from_fn(|i| sample[i % 4]);
                            rt.slice(&sl_in, &mut sl).unwrap();
                            for i in 0..11 {
                                assert_bits_or_nan(sl[i], a4[i % 4], "cbrt slice vs array");
                            }
                            for i in 0..4 {
                                assert_bits_or_nan(s4[i], a4[i], "cbrt simd vs array");
                            }

                            if matches!(d, 1 | 2)
                                && r.is_none_or(|(lo, hi)| {
                                    lo <= f32::from_bits(4) && hi >= f32::from_bits(4)
                                })
                            {
                                valid_counts[0] += 1;
                                let c_sub = rt.scalar(f32::from_bits(4)).unwrap();
                                assert!(c_sub.is_normal() && c_sub > 0.0);
                            }
                            if d == 2 {
                                valid_counts[1] += 1;
                                assert_eq!(
                                    rt.scalar(f32::NEG_INFINITY).unwrap(),
                                    f32::NEG_INFINITY
                                );
                                assert!(rt.scalar(f32::NAN).unwrap().is_nan());
                            }
                            if va == Validation::Checked && d != 2 {
                                valid_counts[2] += 1;
                                assert!(rt.scalar(f32::NAN).is_err());
                                assert!(rt.scalar(f32::MAX).is_err());
                                if d == 0
                                    && r.is_none_or(|(lo, hi)| {
                                        lo < f32::MIN_POSITIVE && hi > -f32::MIN_POSITIVE
                                    })
                                {
                                    assert!(rt.scalar(f32::from_bits(1)).is_err());
                                }
                            }
                            if va == Validation::ClampOutOfBounds {
                                valid_counts[3] += 1;
                                let oob = [
                                    f32::NEG_INFINITY,
                                    -f32::from_bits(1),
                                    f32::INFINITY,
                                    f32::NAN,
                                ];
                                let out = rt.array(oob).unwrap();
                                assert!(out[3].is_nan());
                                assert!(!out[0].is_nan() && out[0].is_finite() || d == 2);
                            }
                            if va == Validation::AssumeValid {
                                valid_counts[4] += 1;
                            }
                            if r.is_some() {
                                valid_counts[5] += 1;
                            }
                            if pr == CbrtfPrecision::Lowp {
                                valid_counts[6] += 1;
                            }
                            if pr == CbrtfPrecision::Midp {
                                valid_counts[7] += 1;
                            }
                            if po == Portability::BitExact {
                                valid_counts[8] += 1;
                            }
                            if ex == Execution::Caller {
                                valid_counts[9] += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    for (idx, &count) in valid_counts.iter().enumerate() {
        assert!(count > 0, "cbrtf held constraint {idx} had 0 valid cases");
    }
}

#[test]
fn const_generic_specializations_match_runtime_across_new_constraints() {
    macro_rules! check_pow {
        ($opts:expr, $probe:expr, $exp:expr) => {{
            const OPTS: PowfOptions = $opts;
            const CFG: u128 = OPTS.bits();
            let rt = OPTS.runtime();
            let c_arr = Powf::<CFG>::array($probe, $exp);
            let r_arr = rt.array($probe, $exp);
            match (c_arr, r_arr) {
                (Ok(ca), Ok(ra)) => {
                    for i in 0..4 {
                        assert_bits_or_nan(ca[i], ra[i], "const vs runtime powf array");
                    }
                    let (mut cs, mut rs) = ([0.0; 9], [0.0; 9]);
                    let sl_in: [f32; 9] = core::array::from_fn(|i| $probe[i % 4]);
                    Powf::<CFG>::slice(&sl_in, &mut cs, $exp).unwrap();
                    rt.slice(&sl_in, &mut rs, $exp).unwrap();
                    for i in 0..9 {
                        assert_bits_or_nan(cs[i], rs[i], "const vs runtime powf slice");
                    }
                }
                (Err(ce), Err(re)) => assert_eq!(ce, re),
                other => panic!("const/runtime mismatch: {other:?}"),
            }
        }};
    }

    check_pow!(
        PowfOptions::new().admit_zero().admit_subnormals(),
        [0.0_f32, f32::from_bits(16), 0.25, 1.0],
        0.5
    );
    check_pow!(
        PowfOptions::new()
            .srgb_encode_domain()
            .clamp_out_of_bounds(),
        [-0.5_f32, 0.0, 0.5, 2.0],
        1.0 / 2.4
    );
    check_pow!(
        PowfOptions::new()
            .unit_gamma_domain()
            .highp()
            .clamp_out_of_bounds(),
        [-1.0_f32, 0.0, 0.25, f32::NAN],
        2.2
    );

    macro_rules! check_log {
        ($opts:expr, $probe:expr) => {{
            const OPTS: LogfOptions = $opts;
            const CFG: u128 = OPTS.bits();
            let rt = OPTS.runtime();
            let ca = Logf::<CFG>::log2_array($probe).unwrap();
            let ra = rt.log2_array($probe).unwrap();
            for i in 0..4 {
                assert_bits_or_nan(ca[i], ra[i], "const vs runtime log2");
            }
        }};
    }

    check_log!(LogfOptions::new().admit_zero(), [0.0_f32, -0.0, 0.5, 2.0]);
    check_log!(
        LogfOptions::new().admit_subnormals(),
        [f32::from_bits(1), f32::from_bits(1024), 0.5, 2.0]
    );
    check_log!(
        LogfOptions::new().near_one_range().clamp_out_of_bounds(),
        [0.1_f32, 0.9, 2.5, f32::NAN]
    );

    const CBRT_SUB: CbrtfOptions = CbrtfOptions::new().admit_subnormals();
    let c_sub =
        Cbrtf::<{ CBRT_SUB.bits() }>::array([-f32::from_bits(8), -0.0, f32::from_bits(8), 8.0])
            .unwrap();
    let r_sub = CBRT_SUB
        .runtime()
        .array([-f32::from_bits(8), -0.0, f32::from_bits(8), 8.0])
        .unwrap();
    for i in 0..4 {
        assert_bits_or_nan(c_sub[i], r_sub[i], "const vs runtime cbrt subnorm");
    }
}

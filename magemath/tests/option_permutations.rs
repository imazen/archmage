//! Comprehensive verification of discrete `*Domain` enums, `Subnormals`
//! (`Reject`, `FlushOrClamp`, `Normalize`), `bf16` custom range bounds,
//! `PowfOptions::const_exponent` (IEEE hardware algebraic + general fast-paths),
//! dedicated `log1pf`, and holding each individual option constraint fixed
//! while sweeping all other option axes.

use archmage::ScalarToken;
use magemath::{
    Execution, MAX_BF16, MAX_CBRT_BF16, Portability, SimdMath, SliceError, Subnormals, Validation,
    cbrtf::{Cbrtf, CbrtfDomain, CbrtfErrorKind, CbrtfOptions, CbrtfPrecision},
    expf::{Expf, ExpfDomain, ExpfErrorKind, ExpfOptions, ExpfPrecision, try_exp},
    log1pf::{Log1pf, Log1pfDomain, Log1pfErrorKind, Log1pfOptions, Log1pfPrecision},
    logf::{Logf, LogfDomain, LogfErrorKind, LogfOptions, LogfPrecision},
    powf::{MAX_EXP_BOUND, Powf, PowfDomain, PowfErrorKind, PowfOptions, PowfPrecision},
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
// 1. Targeted regressions for edge boundaries & bf16 bounds
// ============================================================================

#[test]
fn expf_midp_127_5_upper_endpoint_stays_finite_and_bf16_neighbors_elide() {
    // 127.5 (`0x42ff_0000`) and 127.0 (`0x42fe_0000`) are adjacent exact bf16 values.
    const AT_127_5: ExpfOptions = ExpfOptions::new().input_range(127.5, 127.5);
    const BELOW_127_5: ExpfOptions = ExpfOptions::new().input_range(-126.0, 127.0);
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

    let pred = 127.0_f32;
    let pred_ref = <Expf>::exp2(pred).unwrap();
    assert!(pred_ref.is_finite());
    assert_eq!(
        Expf::<{ BELOW_127_5.bits() }>::exp2(pred)
            .unwrap()
            .to_bits(),
        pred_ref.to_bits()
    );

    // `88.5` (`0x42b1_0000`, exact bf16) scales by LOG2_E to 127.678 >= 127.5
    // (still < 128, so exp(88.5) = 2.723e38 is finite, proving !elides_exp_clamp
    // kept the pre-round clamp active instead of overflowing round() at 127.5);
    // `88.0` (`0x42b0_0000`, exact bf16) scales by LOG2_E to 126.957 < 127.5.
    let exp_above = 88.5_f32;
    let exp_at = ExpfOptions::new().input_range(exp_above, exp_above);
    assert!(!exp_at.elides_exp_clamp());
    let exp_above_val = exp_at.runtime().exp(exp_above).unwrap();
    assert!(exp_above_val.is_finite() && exp_above_val > 2.7e38);
    assert_eq!(
        exp_above_val.to_bits(),
        try_exp(exp_above).unwrap().to_bits()
    );

    assert!(
        ExpfOptions::new()
            .input_range(-87.0, 88.0)
            .elides_exp_clamp()
    );
    assert!(
        ExpfOptions::new()
            .domain(ExpfDomain::NonPositive)
            .runtime()
            .exp(0.0)
            .unwrap()
            == 1.0
    );
    assert!(
        ExpfOptions::new()
            .domain(ExpfDomain::NonPositive)
            .runtime()
            .exp(0.1)
            .is_err()
    );
}

#[test]
fn expf_default_checked_accepts_large_finite_inputs_when_scaled_arg_overflows() {
    assert_eq!(try_exp(f32::MAX).unwrap(), f32::INFINITY);
    assert_eq!(try_exp(-f32::MAX).unwrap().to_bits(), 0.0_f32.to_bits());

    const RANGED_ALL_BF16: u128 = ExpfOptions::new().input_range(-MAX_BF16, MAX_BF16).bits();
    assert_eq!(
        Expf::<RANGED_ALL_BF16>::exp(MAX_BF16).unwrap(),
        f32::INFINITY
    );
    assert_eq!(
        Expf::<RANGED_ALL_BF16>::exp(-MAX_BF16).unwrap().to_bits(),
        0.0_f32.to_bits()
    );

    const DEFAULT_LOWP: u128 = ExpfOptions::new().lowp().bits();
    const RANGED_LOWP: u128 = ExpfOptions::new()
        .lowp()
        .input_range(-MAX_BF16, MAX_BF16)
        .bits();
    assert_eq!(
        Expf::<DEFAULT_LOWP>::exp(MAX_BF16).unwrap().to_bits(),
        Expf::<RANGED_LOWP>::exp(MAX_BF16).unwrap().to_bits()
    );
    assert_eq!(
        Expf::<DEFAULT_LOWP>::exp(-MAX_BF16).unwrap().to_bits(),
        Expf::<RANGED_LOWP>::exp(-MAX_BF16).unwrap().to_bits()
    );
}

#[test]
fn highp_slice_reports_earliest_failing_index_across_mixed_error_stages() {
    const CHECKED_HIGHP: u128 = PowfOptions::new().highp().bits();
    let chunk_a = [f32::MAX, f32::NAN, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0];
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
fn logf_and_log1pf_granular_zero_vs_subnormals_vs_full_domain() {
    const ZERO_ONLY: u128 = LogfOptions::new().admit_zero().reject_subnormals().bits();
    const SUB_ONLY: u128 = LogfOptions::new().admit_subnormals().bits();
    const FLUSH_ZERO: u128 = LogfOptions::new()
        .admit_zero()
        .flush_or_clamp_subnormals()
        .bits();
    const FULL_MIDP: u128 = LogfOptions::new().admit_all_floats().bits();
    const FULL_LOWP: u128 = LogfOptions::new().admit_all_floats().lowp().bits();
    const L1P_NEG_ONE: u128 = Log1pfOptions::new().domain(Log1pfDomain::FromNegOne).bits();

    assert_eq!(Logf::<ZERO_ONLY>::log2(0.0).unwrap(), f32::NEG_INFINITY);
    assert_eq!(Logf::<ZERO_ONLY>::log2(-0.0).unwrap(), f32::NEG_INFINITY);
    assert_eq!(
        Log1pf::<L1P_NEG_ONE>::scalar(-1.0).unwrap(),
        f32::NEG_INFINITY
    );
    assert!(Logf::<ZERO_ONLY>::log2(f32::from_bits(1)).is_err());
    assert_eq!(
        Logf::<FLUSH_ZERO>::log2(f32::from_bits(1)).unwrap(),
        f32::NEG_INFINITY
    );

    assert!(Logf::<SUB_ONLY>::log2(0.0).is_err());
    assert_eq!(Logf::<SUB_ONLY>::log2(f32::from_bits(1)).unwrap(), -149.0);
    assert_eq!(Logf::<FULL_MIDP>::log2(f32::from_bits(1)).unwrap(), -149.0);
    assert!((Logf::<FULL_LOWP>::log2(f32::from_bits(1)).unwrap() - (-149.0)).abs() < 1e-4);

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
    assert_eq!(
        Powf::<PG>::scalar(f32::from_bits(1), 0.5),
        Powf::<PG>::scalar(1.0 / 65_536.0, 0.5)
    );
}

#[test]
fn subnormal_only_intervals_rejected_until_subnormal_policy_enabled() {
    let sub_exp = f32::from_bits(0x0001_0000); // exact nonzero bf16 subnormal
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

    // Pure nonzero-subnormal interval remains invalid under FlushOrClamp (no non-subnormal
    // value in [sub_exp, sub_exp]), whereas [0.0, sub_exp] admits 0.0 under FlushOrClamp.
    assert!(!bad_pow.flush_or_clamp_subnormals().is_valid());
    let flush_pow = PowfOptions::new()
        .exponent_range(0.0, sub_exp)
        .flush_or_clamp_subnormals();
    assert!(flush_pow.is_valid());
    assert_eq!(PowfOptions::from_bits(flush_pow.bits()), Some(flush_pow));
    assert_eq!(flush_pow.runtime().scalar(2.0, sub_exp).unwrap(), 1.0);

    // MAX_EXP_BOUND and MAX_CBRT_BF16 are accepted at both poles.
    let wide_exp = PowfOptions::new().exponent_range(-MAX_EXP_BOUND, MAX_EXP_BOUND);
    assert!(wide_exp.is_valid());
    assert_eq!(PowfOptions::from_bits(wide_exp.bits()), Some(wide_exp));

    let wide_cbrt = CbrtfOptions::new().input_range(-MAX_CBRT_BF16, MAX_CBRT_BF16);
    assert!(wide_cbrt.is_valid());
    assert_eq!(CbrtfOptions::from_bits(wide_cbrt.bits()), Some(wide_cbrt));

    let sub_lo = f32::from_bits(0x0001_0000);
    let sub_hi = f32::from_bits(0x0004_0000);
    let bad_cbrt = CbrtfOptions::new().input_range(sub_lo, sub_hi);
    assert!(!bad_cbrt.is_valid());
    assert_eq!(
        bad_cbrt.runtime().scalar(sub_hi).unwrap_err().kind,
        CbrtfErrorKind::UnsupportedOptions
    );

    let ok_cbrt = bad_cbrt.admit_subnormals();
    assert!(ok_cbrt.is_valid());
    assert_eq!(CbrtfOptions::from_bits(ok_cbrt.bits()), Some(ok_cbrt));
    // 0x0004_0000 = 2^-126 * 2^-5 = 2^-131; test 0x0008_0000 = 2^-130? 0x0002_0000 = 2^-132 -> cbrt = 2^-44.
    let x_132 = f32::from_bits(0x0002_0000);
    assert_close_rel(
        ok_cbrt.runtime().scalar(x_132).unwrap(),
        f32::from_bits((127 - 44) << 23),
        1e-6,
        "subnormal cbrt",
    );
}

// ============================================================================
// 2. Polynomial fitting, discrete domains & zero-reduction range verification
// ============================================================================

#[test]
fn polynomial_single_octave_zero_reduction_matches_full_graph_bit_for_bit() {
    const SMALL_EXP2_M: u128 = ExpfOptions::new().domain(ExpfDomain::SingleOctave).bits();
    const SMALL_EXP_M: u128 = ExpfOptions::new().small_exp_range().bits();
    const SMALL_EXP2_L: u128 = ExpfOptions::new()
        .domain(ExpfDomain::PositiveOctave)
        .lowp()
        .bits();
    const UNCLAMP_EXP2: u128 = ExpfOptions::new().domain(ExpfDomain::NormalOutput).bits();
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
    assert!(
        ExpfOptions::from_bits(SMALL_EXP2_L)
            .unwrap()
            .elides_exp_reduction()
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

    const NEAR_ONE_M: u128 = LogfOptions::new().domain(LogfDomain::SingleOctave).bits();
    const NEAR_ONE_L: u128 = LogfOptions::new()
        .domain(LogfDomain::SingleOctave)
        .lowp()
        .bits();
    const SMALL_LOG1P_M: u128 = Log1pfOptions::new().domain(Log1pfDomain::AroundZero).bits();
    const SMALL_LOG1P_L: u128 = Log1pfOptions::new()
        .domain(Log1pfDomain::AroundZero)
        .lowp()
        .bits();
    const OCTAVE_LOG1P_M: u128 = Log1pfOptions::new()
        .domain(Log1pfDomain::SingleOctave)
        .bits();
    assert!(
        LogfOptions::from_bits(NEAR_ONE_M)
            .unwrap()
            .elides_reduction()
    );
    assert!(
        LogfOptions::from_bits(NEAR_ONE_L)
            .unwrap()
            .elides_reduction()
    );
    assert!(
        Log1pfOptions::from_bits(SMALL_LOG1P_M)
            .unwrap()
            .elides_reduction()
    );
    assert!(
        Log1pfOptions::from_bits(SMALL_LOG1P_L)
            .unwrap()
            .elides_reduction()
    );
    assert!(
        Log1pfOptions::from_bits(OCTAVE_LOG1P_M)
            .unwrap()
            .elides_reduction()
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

    for i in 0..=256 {
        let x = -0.25 + (i as f32) / 512.0;
        let x = if i == 128 { 0.0 } else { x };
        assert_eq!(
            Log1pf::<SMALL_LOG1P_M>::scalar(x).unwrap().to_bits(),
            <Log1pf>::scalar(x).unwrap().to_bits(),
            "log1p midp mismatch at {x}"
        );
        assert_eq!(
            Log1pf::<SMALL_LOG1P_L>::scalar(x).unwrap().to_bits(),
            Log1pf::<{ Log1pfOptions::new().lowp().bits() }>::scalar(x)
                .unwrap()
                .to_bits(),
            "log1p lowp mismatch at {x}"
        );
    }
    assert_eq!(
        Log1pf::<SMALL_LOG1P_M>::scalar(-0.0).unwrap().to_bits(),
        (-0.0_f32).to_bits()
    );

    // Discrete PowfDomain & CbrtfDomain subnormal opt-in verification.
    const SRGB_ENC: PowfOptions = PowfOptions::new().domain(PowfDomain::UnitEncode);
    const UNIT_GAMMA: PowfOptions = PowfOptions::new().domain(PowfDomain::SdrGamma);
    const PQ_DOM: PowfOptions = PowfOptions::new().domain(PowfDomain::PqRatio);
    const UNIT_CBRT: CbrtfOptions = CbrtfOptions::new().unit_range();
    const UNIT_CBRT_REJECT: CbrtfOptions = CbrtfOptions::new().unit_range().reject_subnormals();
    const UNIT_CBRT_NORM: CbrtfOptions = CbrtfOptions::new().unit_range().admit_subnormals();

    assert!(SRGB_ENC.elides_range_check());
    assert!(UNIT_GAMMA.elides_range_check());
    assert!(PQ_DOM.elides_range_check());
    // Unit domains flush by default and retain explicit reject and normalize policies.
    assert!(!UNIT_CBRT_REJECT.elides_denorm_check());
    assert!(UNIT_CBRT_NORM.elides_denorm_check());
    assert!(Cbrtf::<{ UNIT_CBRT_REJECT.bits() }>::scalar(f32::from_bits(1)).is_err());
    assert_eq!(
        Cbrtf::<{ UNIT_CBRT.bits() }>::scalar(-f32::from_bits(1))
            .unwrap()
            .to_bits(),
        (-0.0_f32).to_bits()
    );
    assert!(
        Cbrtf::<{ UNIT_CBRT_NORM.bits() }>::scalar(f32::from_bits(4))
            .unwrap()
            .is_normal()
    );
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
    assert_close_rel(
        Powf::<{ PQ_DOM.bits() }>::scalar(0.9, 78.84375).unwrap(),
        (0.9_f64).powf(78.84375) as f32,
        2e-5,
        "pq ratio m2",
    );
}

// ============================================================================
// 3. PowfOptions::const_exponent IEEE algebraic & general fast-paths
// ============================================================================

#[test]
fn powf_const_exponent_algebraic_and_general_fast_paths() {
    macro_rules! check_const_exp {
        ($opts:expr, $exp:literal, $inputs:expr, $max_rel:expr) => {{
            const OPTS: PowfOptions = $opts;
            const CFG: u128 = OPTS.bits();
            assert_eq!(PowfOptions::from_bits(CFG), Some(OPTS));
            let rt = OPTS.runtime();
            let xs: [f32; 8] = $inputs;

            let a8 = Powf::<CFG>::array_const(xs).unwrap();
            let ra8 = rt.array_const(xs).unwrap();
            let v8 = f32x8::from_array_t(ScalarToken, xs)
                .try_powf_const::<CFG>()
                .unwrap()
                .to_array();
            let rv8 = rt
                .simd_const(f32x8::from_array_t(ScalarToken, xs))
                .unwrap()
                .to_array();

            let sl_in: [f32; 19] = core::array::from_fn(|i| xs[i % 8]);
            let (mut sl_c, mut sl_r) = ([0.0_f32; 19], [0.0_f32; 19]);
            Powf::<CFG>::slice_const(&sl_in, &mut sl_c).unwrap();
            rt.slice_const(&sl_in, &mut sl_r).unwrap();

            for i in 0..8 {
                let sc = Powf::<CFG>::scalar_const(xs[i]).unwrap();
                let sct = Powf::<CFG>::scalar_const_with_token(ScalarToken, xs[i]).unwrap();
                let rsc = rt.scalar_const(xs[i]).unwrap();
                assert_bits_or_nan(sc, a8[i], "scalar_const vs array_const");
                assert_bits_or_nan(sct, a8[i], "scalar_const_with_token vs array_const");
                assert_bits_or_nan(rsc, a8[i], "runtime scalar_const vs array_const");
                assert_bits_or_nan(ra8[i], a8[i], "runtime array_const vs array_const");
                assert_bits_or_nan(v8[i], a8[i], "try_powf_const vs array_const");
                assert_bits_or_nan(rv8[i], a8[i], "runtime simd_const vs array_const");
                let want = (xs[i] as f64).powf($exp as f64) as f32;
                assert_close_rel(a8[i], want, $max_rel, concat!("exp=", stringify!($exp)));
            }
            for i in 0..19 {
                assert_bits_or_nan(sl_c[i], a8[i % 8], "slice_const vs array_const");
                assert_bits_or_nan(sl_r[i], a8[i % 8], "runtime slice_const vs array_const");
            }

            #[cfg(target_arch = "x86_64")]
            if let Some(tok) = <archmage::X64V3Token as archmage::SimdToken>::summon() {
                let v3_out = f32x8::from_array_t(tok, xs)
                    .try_powf_const::<CFG>()
                    .unwrap()
                    .to_array();
                for i in 0..8 {
                    if OPTS != OPTS.backend_dependent() {
                        assert_bits_or_nan(v3_out[i], a8[i], "V3 vs ScalarToken const_exponent");
                    }
                }
            }
        }};
    }

    let zero_pos = [0.0_f32, 0.0625, 0.25, 0.5, 0.75, 1.0, 4.0, 16.0];
    let strict_pos = [0.0625_f32, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 16.0];

    // All 11 IEEE hardware algebraic fast-paths across Midp, Lowp, and Highp!
    check_const_exp!(
        PowfOptions::new().admit_zero().const_exponent(0.0),
        0.0,
        zero_pos,
        0.0
    );
    check_const_exp!(
        PowfOptions::new().admit_zero().const_exponent(1.0),
        1.0,
        zero_pos,
        0.0
    );
    check_const_exp!(
        PowfOptions::new().const_exponent(-1.0),
        -1.0,
        strict_pos,
        1e-7
    );
    check_const_exp!(
        PowfOptions::new().gamma().const_exponent(2.0),
        2.0,
        zero_pos,
        1e-7
    );
    check_const_exp!(
        PowfOptions::new().const_exponent(-2.0),
        -2.0,
        strict_pos,
        2e-7
    );
    check_const_exp!(
        PowfOptions::new().gamma().const_exponent(3.0),
        3.0,
        zero_pos,
        2e-7
    );
    check_const_exp!(
        PowfOptions::new().gamma().const_exponent(4.0),
        4.0,
        zero_pos,
        2e-7
    );
    check_const_exp!(
        PowfOptions::new().admit_zero().const_exponent(0.5),
        0.5,
        zero_pos,
        1e-7
    );
    check_const_exp!(
        PowfOptions::new().const_exponent(-0.5),
        -0.5,
        strict_pos,
        2e-7
    );
    check_const_exp!(
        PowfOptions::new().gamma().const_exponent(1.5),
        1.5,
        zero_pos,
        2e-7
    );
    check_const_exp!(
        PowfOptions::new().admit_zero().const_exponent(0.25),
        0.25,
        zero_pos,
        2e-7
    );

    // Highp algebraic compound accuracy (<= 0.5 ULP via f64 intermediates) & Lowp parity.
    check_const_exp!(
        PowfOptions::new().gamma().highp().const_exponent(3.0),
        3.0,
        zero_pos,
        1e-7
    );
    check_const_exp!(
        PowfOptions::new().highp().const_exponent(-0.5),
        -0.5,
        strict_pos,
        1e-7
    );
    check_const_exp!(
        PowfOptions::new().gamma().highp().const_exponent(1.5),
        1.5,
        zero_pos,
        1e-7
    );
    check_const_exp!(
        PowfOptions::new().admit_zero().highp().const_exponent(0.25),
        0.25,
        zero_pos,
        1e-7
    );
    check_const_exp!(
        PowfOptions::new().gamma().lowp().const_exponent(0.5),
        0.5,
        zero_pos,
        1e-7
    );

    // General constant exponents (2.4, 1/2.4, PQ 78.84375, BT.709 0.45, -1.5).
    check_const_exp!(
        PowfOptions::new()
            .domain(PowfDomain::HdrGamma)
            .const_exponent(2.4),
        2.4,
        zero_pos,
        2e-5
    );
    check_const_exp!(
        PowfOptions::new()
            .domain(PowfDomain::Unit)
            .const_exponent(1.0 / 2.4),
        0.41666666,
        [0.0, 1e-6, 0.01, 0.1, 0.25, 0.5, 0.75, 1.0],
        2e-5
    );
    check_const_exp!(
        PowfOptions::new()
            .domain(PowfDomain::PqRatio)
            .const_exponent(78.84375),
        78.84375,
        [0.84, 0.86, 0.88, 0.90, 0.92, 0.95, 0.98, 1.0],
        2e-5
    );
    check_const_exp!(
        PowfOptions::new()
            .domain(PowfDomain::Unit)
            .lowp()
            .const_exponent(0.45),
        0.45,
        [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
        1e-2
    );
    check_const_exp!(
        PowfOptions::new().const_exponent(-1.5),
        -1.5,
        strict_pos,
        2e-5
    );

    // Infallible powf_const & scalar_const_infallible + Checked overflow rejection on algebraic path.
    const INFALLIBLE_SRGB: u128 = PowfOptions::new()
        .domain(PowfDomain::Unit)
        .const_exponent(2.4)
        .clamp_out_of_bounds()
        .bits();
    let v = f32x4::from_array_t(ScalarToken, [-1.0, f32::from_bits(1), 0.5, f32::NAN]);
    let out = v.powf_const::<INFALLIBLE_SRGB>().to_array();
    assert_eq!(out[0], 0.0);
    assert_eq!(out[1], 0.0);
    assert!(out[3].is_nan());
    assert_eq!(
        Powf::<INFALLIBLE_SRGB>::scalar_const_infallible(0.5).to_bits(),
        out[2].to_bits()
    );

    const CHECKED_SQ: u128 = PowfOptions::new().const_exponent(2.0).bits();
    assert_eq!(
        Powf::<CHECKED_SQ>::scalar_const(1.0e20).unwrap_err().kind,
        PowfErrorKind::OutsideApproximationRange
    );
    assert_eq!(
        Powf::<CHECKED_SQ>::scalar_const(1.0e-20).unwrap_err().kind,
        PowfErrorKind::OutsideApproximationRange
    );

    // Runtime API rejects mismatched const_exponent vs runtime-n calls with UnsupportedOptions.
    let const_rt = PowfOptions::new().const_exponent(2.0).runtime();
    assert_eq!(
        const_rt.scalar(2.0, 2.0).unwrap_err().kind,
        PowfErrorKind::UnsupportedOptions
    );
    let dyn_rt = PowfOptions::new().runtime();
    assert_eq!(
        dyn_rt.scalar_const(2.0).unwrap_err().kind,
        PowfErrorKind::UnsupportedOptions
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
    const LOG1P_CLAMP: u128 = Log1pfOptions::new()
        .domain(Log1pfDomain::AroundZero)
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
    assert_eq!(l[0], f32::NEG_INFINITY);
    assert_close_rel(l[1], -2.0, 1e-6, "0.125 clamps to 0.25 -> -2");
    assert_close_rel(l[2], 2.0, 1e-6, "16 clamps to 4 -> +2");
    assert!(l[3].is_nan());

    let l1p = f32x4::from_array_t(ScalarToken, [-2.0, f32::from_bits(1), 2.0, f32::NAN])
        .log1p::<LOG1P_CLAMP>()
        .to_array();
    assert_close_rel(l1p[0], (-0.25_f32).ln_1p(), 1e-6, "clamped log1p lo");
    assert_eq!(l1p[1], 0.0); // subnormal shoved to 0.0!
    assert_close_rel(l1p[2], (0.25_f32).ln_1p(), 1e-6, "clamped log1p hi");
    assert!(l1p[3].is_nan());

    let c = f32x4::from_array_t(ScalarToken, [-8.0, -0.125, 8.0, f32::NAN])
        .cbrt::<CBRT_CLAMP>()
        .to_array();
    assert_close_rel(c[0], -1.0, 1e-6, "clamped cbrt lo");
    assert_close_rel(c[1], -0.5, 1e-6, "clamped cbrt mid");
    assert_close_rel(c[2], 1.0, 1e-6, "clamped cbrt hi");
    assert!(c[3].is_nan());
}

// ============================================================================
// 4. Single-constraint-held exhaustive permutation compliance across all 5 families
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

    let is_full = opts.selected_domain() == PowfDomain::AllFloats;
    let is_clamp_oob = opts == opts.clamp_out_of_bounds() && !is_full;
    let is_assume = opts == opts.assume_in_bounds() && !is_full;
    let is_clamp_finite = opts == opts.clamp_finite() && !is_full;
    let is_checked = !is_full && !is_clamp_oob && !is_assume && !is_clamp_finite;

    let in_bases4 = if opts.selected_domain() == PowfDomain::PqRatio {
        [0.85_f32, 0.9, 0.95, 1.0]
    } else {
        [0.25_f32, 0.5, 0.75, 1.0]
    };
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

    if is_clamp_finite {
        let probe = [f32::NEG_INFINITY, f32::from_bits(1), -0.0, f32::INFINITY];
        let restoring = opts.clamp_out_of_bounds().runtime();
        for exp in [-10.0, -0.0, f32::from_bits(1), 0.5, f32::INFINITY] {
            assert_eq!(
                rt.array(probe, exp).unwrap().map(f32::to_bits),
                restoring.array(probe, exp).unwrap().map(f32::to_bits)
            );
        }
    } else if is_checked {
        let mut bad = [in_bases4[1]; 9];
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
            rt.array([in_bases4[1], -1.0, f32::NAN, in_bases4[1]], n)
                .unwrap_err()
                .lane_mask,
            0b0110
        );
        assert_eq!(
            rt.scalar(in_bases4[1], f32::NAN).unwrap_err().kind,
            PowfErrorKind::InvalidExponent
        );
    } else if is_clamp_oob {
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
                in_bases4[1],
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
                assert!(res8.iter().all(|v| v.is_nan()));
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
    let domains = [
        PowfDomain::PositiveNormals,
        PowfDomain::NonNegative,
        PowfDomain::Unit,
        PowfDomain::HdrGamma,
        PowfDomain::PqRatio,
        PowfDomain::AllFloats,
    ];
    let bases: [Option<(f32, f32)>; 2] = [None, Some((1.0 / 65_536.0, 65_536.0))];
    let exps: [Option<(f32, f32)>; 3] = [
        None,
        Some((-2.0, 4.0)),
        Some((f32::from_bits(0x0001_0000), f32::from_bits(0x0001_0000))),
    ];
    let subs = [
        Subnormals::Reject,
        Subnormals::FlushOrClamp,
        Subnormals::Normalize,
    ];
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
        Validation::ClampFinite,
        Validation::AssumeValid,
    ];
    let execs = [Execution::Upgrade, Execution::Caller];

    let mut valid_counts = [0_usize; 16];

    for &dom in &domains {
        for &b in &bases {
            for &e in &exps {
                for &sub in &subs {
                    for &c in &clamps {
                        for &pr in &precs {
                            for &po in &ports {
                                for &va in &vals {
                                    for &ex in &execs {
                                        let mut opts = PowfOptions::new()
                                            .domain(dom)
                                            .precision(pr)
                                            .portability(po)
                                            .validation(va)
                                            .execution(ex);
                                        if let Some((lo, hi)) = b {
                                            opts = opts.base_range(lo, hi);
                                        }
                                        if let Some((lo, hi)) = e {
                                            opts = opts.exponent_range(lo, hi);
                                        }
                                        opts = opts.subnormals(sub);
                                        if c {
                                            opts = opts.saturate_output();
                                        }

                                        let sample_exp = match e {
                                            Some((lo, hi)) if lo == hi => lo,
                                            _ => 0.5,
                                        };
                                        verify_powf_option_compliance(opts, sample_exp);
                                        if opts.is_valid() {
                                            if matches!(
                                                dom,
                                                PowfDomain::NonNegative
                                                    | PowfDomain::Unit
                                                    | PowfDomain::HdrGamma
                                            ) && e.is_none_or(|(lo, hi)| lo <= 0.0 && hi >= 0.5)
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
                                            if sub == Subnormals::Normalize {
                                                valid_counts[1] += 1;
                                                let s_in = f32::from_bits(1 << 8);
                                                let got = opts
                                                    .runtime()
                                                    .scalar(s_in, sample_exp)
                                                    .unwrap();
                                                assert!(got.is_normal() && got > 0.0);
                                            }
                                            if sub == Subnormals::FlushOrClamp {
                                                valid_counts[2] += 1;
                                                if va == Validation::Checked {
                                                    assert!(
                                                        opts.runtime()
                                                            .scalar(f32::from_bits(1), sample_exp)
                                                            .is_ok()
                                                    );
                                                }
                                            }
                                            if dom == PowfDomain::AllFloats {
                                                valid_counts[3] += 1;
                                            }
                                            if dom == PowfDomain::PqRatio {
                                                valid_counts[4] += 1;
                                            }
                                            if b.is_some() {
                                                valid_counts[5] += 1;
                                            }
                                            if e.is_some() {
                                                valid_counts[6] += 1;
                                            }
                                            if c {
                                                valid_counts[7] += 1;
                                            }
                                            if pr == PowfPrecision::Lowp {
                                                valid_counts[8] += 1;
                                            }
                                            if pr == PowfPrecision::Midp {
                                                valid_counts[9] += 1;
                                            }
                                            if pr == PowfPrecision::Highp {
                                                valid_counts[10] += 1;
                                            }
                                            if po == Portability::BitExact {
                                                valid_counts[11] += 1;
                                            }
                                            if va == Validation::Checked {
                                                valid_counts[12] += 1;
                                            }
                                            if va == Validation::ClampOutOfBounds {
                                                valid_counts[13] += 1;
                                            }
                                            if va == Validation::AssumeValid {
                                                valid_counts[14] += 1;
                                            }
                                            if ex == Execution::Caller {
                                                valid_counts[15] += 1;
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
    let domains = [
        ExpfDomain::Finite,
        ExpfDomain::NormalOutput,
        ExpfDomain::NonPositive,
        ExpfDomain::SignedUnit,
        ExpfDomain::SingleOctave,
        ExpfDomain::PositiveOctave,
        ExpfDomain::AllFloats,
    ];
    let ranges: [Option<(f32, f32)>; 2] = [None, Some((-20.0, 20.0))];
    let precs = [ExpfPrecision::Midp, ExpfPrecision::Lowp];
    let ports = [Portability::BitExact, Portability::BackendDependent];
    let vals = [
        Validation::Checked,
        Validation::ClampOutOfBounds,
        Validation::ClampFinite,
        Validation::AssumeValid,
    ];
    let execs = [Execution::Upgrade, Execution::Caller];

    let mut valid_counts = [0_usize; 11];

    for &dom in &domains {
        for &r in &ranges {
            for &pr in &precs {
                for &po in &ports {
                    for &va in &vals {
                        for &ex in &execs {
                            let mut opts = ExpfOptions::new()
                                .domain(dom)
                                .precision(pr)
                                .portability(po)
                                .validation(va)
                                .execution(ex);
                            if let Some((lo, hi)) = r {
                                opts = opts.input_range(lo, hi);
                            }

                            if !opts.is_valid() {
                                assert_eq!(ExpfOptions::from_bits(opts.bits()), None);
                                assert_eq!(
                                    opts.runtime().exp2(0.0).unwrap_err().kind,
                                    ExpfErrorKind::UnsupportedOptions
                                );
                                continue;
                            }

                            assert_eq!(ExpfOptions::from_bits(opts.bits()), Some(opts));
                            let rt = opts.runtime();
                            let in_vals = if dom == ExpfDomain::NonPositive && r.is_none() {
                                [-0.25_f32, -0.125, -0.0625, 0.0]
                            } else {
                                [0.0_f32, 0.1, 0.2, 0.25]
                            };
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

                            let full = dom == ExpfDomain::AllFloats;
                            if va == Validation::Checked && !full {
                                valid_counts[0] += 1;
                                assert!(rt.exp2(f32::NAN).is_err());
                                assert!(rt.exp(f32::INFINITY).is_err());
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
                            if dom == ExpfDomain::NonPositive && r.is_none() {
                                valid_counts[4] += 1;
                            }
                            if dom == ExpfDomain::SingleOctave && r.is_none() {
                                valid_counts[5] += 1;
                            }
                            if r.is_some() {
                                valid_counts[6] += 1;
                            }
                            if pr == ExpfPrecision::Lowp {
                                valid_counts[7] += 1;
                            }
                            if pr == ExpfPrecision::Midp {
                                valid_counts[8] += 1;
                            }
                            if po == Portability::BitExact {
                                valid_counts[9] += 1;
                            }
                            if ex == Execution::Caller {
                                valid_counts[10] += 1;
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
    let domains = [
        LogfDomain::PositiveNormals,
        LogfDomain::NonNegative,
        LogfDomain::Unit,
        LogfDomain::PositiveHdr,
        LogfDomain::SingleOctave,
        LogfDomain::AllFloats,
    ];
    let ranges: [Option<(f32, f32)>; 3] = [None, Some((0.75, 1.25)), Some((0.0, 1.0))];
    let subs = [
        Subnormals::Reject,
        Subnormals::FlushOrClamp,
        Subnormals::Normalize,
    ];
    let precs = [LogfPrecision::Midp, LogfPrecision::Lowp];
    let ports = [Portability::BitExact, Portability::BackendDependent];
    let vals = [
        Validation::Checked,
        Validation::ClampOutOfBounds,
        Validation::ClampFinite,
        Validation::AssumeValid,
    ];
    let execs = [Execution::Upgrade, Execution::Caller];

    let mut valid_counts = [0_usize; 12];
    let pos_in = [0.8_f32, 0.85, 0.9, 1.0];

    for &dom in &domains {
        for &r in &ranges {
            for &sub in &subs {
                for &pr in &precs {
                    for &po in &ports {
                        for &va in &vals {
                            for &ex in &execs {
                                let mut opts = LogfOptions::new()
                                    .domain(dom)
                                    .precision(pr)
                                    .portability(po)
                                    .validation(va)
                                    .execution(ex);
                                if let Some((lo, hi)) = r {
                                    opts = opts.input_range(lo, hi);
                                }
                                opts = opts.subnormals(sub);

                                if !opts.is_valid() {
                                    assert_eq!(LogfOptions::from_bits(opts.bits()), None);
                                    assert_eq!(
                                        opts.runtime().log2(1.0).unwrap_err().kind,
                                        LogfErrorKind::UnsupportedOptions
                                    );
                                    continue;
                                }

                                assert_eq!(LogfOptions::from_bits(opts.bits()), Some(opts));
                                let rt = opts.runtime();
                                let a4 = rt.log2_array(pos_in).unwrap();
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

                                if matches!(
                                    dom,
                                    LogfDomain::NonNegative
                                        | LogfDomain::Unit
                                        | LogfDomain::AllFloats
                                ) && r.is_none_or(|(lo, _)| lo == 0.0)
                                {
                                    valid_counts[0] += 1;
                                    assert_eq!(rt.log2(0.0).unwrap(), f32::NEG_INFINITY);
                                    assert_eq!(rt.ln(-0.0).unwrap(), f32::NEG_INFINITY);
                                }
                                if opts.subnormals_policy() == Subnormals::Normalize {
                                    valid_counts[1] += 1;
                                    let sub_res = rt.log2(f32::from_bits(1)).unwrap();
                                    assert!((sub_res - (-149.0)).abs() < 1e-3);
                                }
                                if sub == Subnormals::FlushOrClamp {
                                    valid_counts[2] += 1;
                                    if va == Validation::Checked {
                                        assert!(rt.log2(f32::from_bits(1)).is_ok());
                                    }
                                }
                                if dom == LogfDomain::AllFloats {
                                    valid_counts[3] += 1;
                                    assert!(rt.ln(-2.0).unwrap().is_nan());
                                    assert_eq!(rt.log10(f32::INFINITY).unwrap(), f32::INFINITY);
                                }
                                if va == Validation::Checked && dom != LogfDomain::AllFloats {
                                    valid_counts[4] += 1;
                                    assert!(rt.log2(-2.0).is_err());
                                    assert!(rt.log2(f32::NAN).is_err());
                                }
                                if va == Validation::ClampOutOfBounds {
                                    valid_counts[5] += 1;
                                    let oob = [f32::NEG_INFINITY, -1.0, f32::INFINITY, f32::NAN];
                                    let c_out = rt.log2_array(oob).unwrap();
                                    assert!(c_out[3].is_nan());
                                    if dom == LogfDomain::AllFloats {
                                        assert!(c_out[0].is_nan());
                                    } else {
                                        assert!(!c_out[0].is_nan());
                                    }
                                }
                                if va == Validation::AssumeValid {
                                    valid_counts[6] += 1;
                                }
                                if r.is_some() {
                                    valid_counts[7] += 1;
                                }
                                if pr == LogfPrecision::Lowp {
                                    valid_counts[8] += 1;
                                }
                                if pr == LogfPrecision::Midp {
                                    valid_counts[9] += 1;
                                }
                                if po == Portability::BitExact {
                                    valid_counts[10] += 1;
                                }
                                if ex == Execution::Caller {
                                    valid_counts[11] += 1;
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
fn log1pf_hold_each_constraint_across_all_other_permutations() {
    let domains = [
        Log1pfDomain::GreaterThanNegOne,
        Log1pfDomain::FromNegOne,
        Log1pfDomain::Unit,
        Log1pfDomain::SignedUnit,
        Log1pfDomain::AroundZero,
        Log1pfDomain::SingleOctave,
        Log1pfDomain::AllFloats,
    ];
    let ranges: [Option<(f32, f32)>; 3] = [None, Some((-0.25, 0.25)), Some((-1.0, 1.0))];
    let subs = [
        Subnormals::Reject,
        Subnormals::FlushOrClamp,
        Subnormals::Normalize,
    ];
    let precs = [Log1pfPrecision::Midp, Log1pfPrecision::Lowp];
    let ports = [Portability::BitExact, Portability::BackendDependent];
    let vals = [
        Validation::Checked,
        Validation::ClampOutOfBounds,
        Validation::ClampFinite,
        Validation::AssumeValid,
    ];
    let execs = [Execution::Upgrade, Execution::Caller];

    let mut valid_counts = [0_usize; 12];
    let sample_in = [0.0_f32, 0.0625, 0.125, 0.25];

    for &dom in &domains {
        for &r in &ranges {
            for &sub in &subs {
                for &pr in &precs {
                    for &po in &ports {
                        for &va in &vals {
                            for &ex in &execs {
                                let mut opts = Log1pfOptions::new()
                                    .domain(dom)
                                    .precision(pr)
                                    .portability(po)
                                    .validation(va)
                                    .execution(ex);
                                if let Some((lo, hi)) = r {
                                    opts = opts.input_range(lo, hi);
                                }
                                opts = opts.subnormals(sub);

                                if !opts.is_valid() {
                                    assert_eq!(Log1pfOptions::from_bits(opts.bits()), None);
                                    assert_eq!(
                                        opts.runtime().scalar(0.0).unwrap_err().kind,
                                        Log1pfErrorKind::UnsupportedOptions
                                    );
                                    continue;
                                }

                                assert_eq!(Log1pfOptions::from_bits(opts.bits()), Some(opts));
                                let rt = opts.runtime();
                                let a4 = rt.array(sample_in).unwrap();
                                let v4 = f32x4::from_array_t(ScalarToken, sample_in);
                                let s4 = rt.simd(v4).unwrap().to_array();
                                let mut sl = [0.0; 9];
                                let sl_in: [f32; 9] = core::array::from_fn(|i| sample_in[i % 4]);
                                rt.slice(&sl_in, &mut sl).unwrap();
                                for i in 0..9 {
                                    assert_bits_or_nan(sl[i], a4[i % 4], "log1pf slice vs array");
                                }
                                for i in 0..4 {
                                    assert_bits_or_nan(s4[i], a4[i], "log1pf simd vs array");
                                }

                                if matches!(
                                    dom,
                                    Log1pfDomain::FromNegOne
                                        | Log1pfDomain::SignedUnit
                                        | Log1pfDomain::AllFloats
                                ) && r.is_none_or(|(lo, _)| lo == -1.0)
                                {
                                    valid_counts[0] += 1;
                                    assert_eq!(rt.scalar(-1.0).unwrap(), f32::NEG_INFINITY);
                                }
                                if opts.subnormals_policy() == Subnormals::Normalize {
                                    valid_counts[1] += 1;
                                    assert_eq!(
                                        rt.scalar(f32::from_bits(1)).unwrap().to_bits(),
                                        1,
                                        "pos subnormal preservation failed on {opts:?}"
                                    );
                                    if dom != Log1pfDomain::Unit
                                        || r.is_some_and(|(lo, _)| lo < 0.0)
                                    {
                                        assert_eq!(
                                            rt.scalar(-f32::from_bits(1)).unwrap().to_bits(),
                                            0x8000_0001,
                                            "neg subnormal preservation failed on {opts:?}"
                                        );
                                    }
                                }
                                if sub == Subnormals::FlushOrClamp {
                                    valid_counts[2] += 1;
                                    if va == Validation::Checked {
                                        assert_eq!(
                                            rt.scalar(f32::from_bits(1)).unwrap().to_bits(),
                                            0,
                                            "failed on {opts:?}"
                                        );
                                    }
                                }
                                if dom == Log1pfDomain::AllFloats {
                                    valid_counts[3] += 1;
                                    assert!(rt.scalar(-2.0).unwrap().is_nan());
                                }
                                if va == Validation::Checked && dom != Log1pfDomain::AllFloats {
                                    valid_counts[4] += 1;
                                    assert!(rt.scalar(-2.0).is_err());
                                    assert!(rt.scalar(f32::NAN).is_err());
                                    if sub == Subnormals::Reject {
                                        assert!(rt.scalar(f32::from_bits(1)).is_err());
                                    }
                                }
                                if va == Validation::ClampOutOfBounds {
                                    valid_counts[5] += 1;
                                    let oob = [f32::NEG_INFINITY, -5.0, f32::INFINITY, f32::NAN];
                                    let c1p = rt.array(oob).unwrap();
                                    assert!(c1p[3].is_nan());
                                    if dom == Log1pfDomain::AllFloats {
                                        assert!(c1p[1].is_nan());
                                    } else {
                                        assert!(!c1p[1].is_nan());
                                    }
                                }
                                if va == Validation::AssumeValid {
                                    valid_counts[6] += 1;
                                }
                                if r.is_some() {
                                    valid_counts[7] += 1;
                                }
                                if pr == Log1pfPrecision::Lowp {
                                    valid_counts[8] += 1;
                                }
                                if pr == Log1pfPrecision::Midp {
                                    valid_counts[9] += 1;
                                }
                                if po == Portability::BitExact {
                                    valid_counts[10] += 1;
                                }
                                if ex == Execution::Caller {
                                    valid_counts[11] += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for (idx, &count) in valid_counts.iter().enumerate() {
        assert!(count > 0, "log1pf held constraint {idx} had 0 valid cases");
    }
}

#[test]
fn cbrtf_hold_each_constraint_across_all_other_permutations() {
    let domains = [
        CbrtfDomain::Bounded,
        CbrtfDomain::Unit,
        CbrtfDomain::SignedUnit,
        CbrtfDomain::PositiveNormals,
        CbrtfDomain::AllFloats,
    ];
    let ranges: [Option<(f32, f32)>; 3] = [
        None,
        Some((0.125, 8.0)),
        Some((f32::from_bits(0x0001_0000), f32::from_bits(0x0004_0000))),
    ];
    let subs = [
        Subnormals::Reject,
        Subnormals::FlushOrClamp,
        Subnormals::Normalize,
    ];
    let precs = [CbrtfPrecision::Midp, CbrtfPrecision::Lowp];
    let ports = [Portability::BitExact, Portability::BackendDependent];
    let vals = [
        Validation::Checked,
        Validation::ClampOutOfBounds,
        Validation::ClampFinite,
        Validation::AssumeValid,
    ];
    let execs = [Execution::Upgrade, Execution::Caller];

    let mut valid_counts = [0_usize; 12];
    let normal_in = [0.125_f32, 0.25, 0.5, 1.0];

    for &dom in &domains {
        for &r in &ranges {
            for &sub in &subs {
                for &pr in &precs {
                    for &po in &ports {
                        for &va in &vals {
                            for &ex in &execs {
                                let mut opts = CbrtfOptions::new()
                                    .domain(dom)
                                    .precision(pr)
                                    .portability(po)
                                    .validation(va)
                                    .execution(ex);
                                if let Some((lo, hi)) = r {
                                    opts = opts.input_range(lo, hi);
                                }
                                opts = opts.subnormals(sub);

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

                                let sample: [f32; 4] = if r
                                    == Some((
                                        f32::from_bits(0x0001_0000),
                                        f32::from_bits(0x0004_0000),
                                    )) {
                                    [
                                        f32::from_bits(0x0001_0000),
                                        f32::from_bits(0x0002_0000),
                                        f32::from_bits(0x0003_0000),
                                        f32::from_bits(0x0004_0000),
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

                                if opts.subnormals_policy() == Subnormals::Normalize {
                                    valid_counts[0] += 1;
                                    let c_sub = rt.scalar(f32::from_bits(0x0002_0000)).unwrap();
                                    assert!(c_sub.is_normal() && c_sub > 0.0);
                                }
                                if sub == Subnormals::FlushOrClamp {
                                    valid_counts[1] += 1;
                                    if va == Validation::Checked {
                                        assert!(rt.scalar(f32::from_bits(0x0002_0000)).is_ok());
                                    }
                                }
                                if dom == CbrtfDomain::AllFloats {
                                    valid_counts[2] += 1;
                                    assert_eq!(
                                        rt.scalar(f32::NEG_INFINITY).unwrap(),
                                        f32::NEG_INFINITY
                                    );
                                    assert!(rt.scalar(f32::NAN).unwrap().is_nan());
                                }
                                if dom == CbrtfDomain::PositiveNormals && r.is_none() {
                                    valid_counts[3] += 1;
                                }
                                if va == Validation::Checked && dom != CbrtfDomain::AllFloats {
                                    valid_counts[4] += 1;
                                    assert!(rt.scalar(f32::NAN).is_err());
                                    assert!(rt.scalar(f32::MAX).is_err());
                                }
                                if va == Validation::ClampOutOfBounds {
                                    valid_counts[5] += 1;
                                    let oob = [
                                        f32::NEG_INFINITY,
                                        -f32::from_bits(1),
                                        f32::INFINITY,
                                        f32::NAN,
                                    ];
                                    let out = rt.array(oob).unwrap();
                                    assert!(out[3].is_nan());
                                    assert!(
                                        (!out[0].is_nan() && out[0].is_finite())
                                            || dom == CbrtfDomain::AllFloats
                                    );
                                }
                                if va == Validation::AssumeValid {
                                    valid_counts[6] += 1;
                                }
                                if r.is_some() {
                                    valid_counts[7] += 1;
                                }
                                if pr == CbrtfPrecision::Lowp {
                                    valid_counts[8] += 1;
                                }
                                if pr == CbrtfPrecision::Midp {
                                    valid_counts[9] += 1;
                                }
                                if po == Portability::BitExact {
                                    valid_counts[10] += 1;
                                }
                                if ex == Execution::Caller {
                                    valid_counts[11] += 1;
                                }
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

    const EXP_SMALL_CLAMP: ExpfOptions = ExpfOptions::new().small_exp_range().clamp_out_of_bounds();
    let e_probe = [-5.0_f32, -0.0, 0.2, f32::NAN];
    let ce = Expf::<{ EXP_SMALL_CLAMP.bits() }>::exp_array(e_probe).unwrap();
    let re = EXP_SMALL_CLAMP.runtime().exp_array(e_probe).unwrap();
    for i in 0..4 {
        assert_bits_or_nan(ce[i], re[i], "const vs runtime exp small clamp");
    }

    const LOG1P_SMALL_CLAMP: Log1pfOptions = Log1pfOptions::new()
        .domain(Log1pfDomain::AroundZero)
        .clamp_out_of_bounds();
    let l1p_probe = [-2.0_f32, -0.0, 0.1, f32::NAN];
    let cl1p = Log1pf::<{ LOG1P_SMALL_CLAMP.bits() }>::array(l1p_probe).unwrap();
    let rl1p = LOG1P_SMALL_CLAMP.runtime().array(l1p_probe).unwrap();
    for i in 0..4 {
        assert_bits_or_nan(cl1p[i], rl1p[i], "const vs runtime log1p small clamp");
    }

    const POW_CHECKED_GAMMA: PowfOptions = PowfOptions::new().gamma_domain();
    let bad_slice = [0.5_f32, 0.25, -1.0, 0.75, f32::NAN, 0.5, 0.5, 0.5, -2.0];
    let (mut cs_bad, mut rs_bad) = ([0.0; 9], [0.0; 9]);
    assert_eq!(
        Powf::<{ POW_CHECKED_GAMMA.bits() }>::slice(&bad_slice, &mut cs_bad, 2.2).unwrap_err(),
        POW_CHECKED_GAMMA
            .runtime()
            .slice(&bad_slice, &mut rs_bad, 2.2)
            .unwrap_err()
    );
}

#[test]
fn powf_subnormal_straddling_exponent_intervals_clamp_to_admitted_endpoints() {
    const POS_OPTS: PowfOptions = PowfOptions::new()
        .admit_zero_base()
        .exponent_range(f32::MIN_POSITIVE / 2.0, f32::MIN_POSITIVE)
        .clamp_out_of_bounds()
        .caller();
    const POS_CFG: u128 = POS_OPTS.bits();
    const POS_HIGHP_CFG: u128 = POS_OPTS.highp().bits();
    #[cfg(feature = "std")]
    const POS_HIGHP_STD_CFG: u128 = POS_OPTS.highp().backend_dependent().bits();

    for exp_in in [
        f32::NEG_INFINITY,
        -1.0_f32,
        -0.0,
        0.0,
        f32::from_bits(1),
        f32::MIN_POSITIVE / 2.0,
    ] {
        assert_eq!(Powf::<POS_CFG>::scalar(0.0, exp_in).unwrap(), 0.0);
        assert_eq!(POS_OPTS.runtime().scalar(0.0, exp_in).unwrap(), 0.0);
        assert_eq!(Powf::<POS_HIGHP_CFG>::scalar(0.0, exp_in).unwrap(), 0.0);
        assert_eq!(POS_OPTS.highp().runtime().scalar(0.0, exp_in).unwrap(), 0.0);
        #[cfg(feature = "std")]
        {
            assert_eq!(Powf::<POS_HIGHP_STD_CFG>::scalar(0.0, exp_in).unwrap(), 0.0);
            assert_eq!(
                POS_OPTS
                    .highp()
                    .backend_dependent()
                    .runtime()
                    .scalar(0.0, exp_in)
                    .unwrap(),
                0.0
            );
        }

        let src = [0.0_f32, -0.0, 0.25, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0];
        let (mut c_dst, mut r_dst) = ([1.0_f32; 9], [1.0_f32; 9]);
        Powf::<POS_CFG>::slice(&src, &mut c_dst, exp_in).unwrap();
        POS_OPTS.runtime().slice(&src, &mut r_dst, exp_in).unwrap();
        assert_eq!(c_dst[0], 0.0);
        assert_eq!(c_dst[1], 0.0);
        assert_eq!(c_dst[8], 0.0);
        assert_eq!(c_dst, r_dst);

        let v4 = f32x4::from_array_t(ScalarToken, [0.0, -0.0, 0.5, 1.0]);
        let s4 = v4.powf::<POS_CFG>(exp_in).to_array();
        assert_eq!(s4[0], 0.0);
        assert_eq!(s4[1], 0.0);

        #[cfg(target_arch = "x86_64")]
        if let Some(tok) = <archmage::X64V3Token as archmage::SimdToken>::summon() {
            let v8 = magetypes::simd::generic::f32x8::from_array_t(
                tok,
                [0.0, -0.0, 0.25, 0.5, 0.0, 1.0, 2.0, 0.0],
            );
            let s8 = v8.powf::<POS_CFG>(exp_in).to_array();
            assert_eq!(s8[0], 0.0);
            assert_eq!(s8[1], 0.0);
            assert_eq!(s8[4], 0.0);
            assert_eq!(s8[7], 0.0);
        }
    }

    const NEG_OPTS: PowfOptions = PowfOptions::new()
        .exponent_range(-f32::MIN_POSITIVE, -f32::MIN_POSITIVE / 2.0)
        .clamp_out_of_bounds();
    const NEG_CFG: u128 = NEG_OPTS.bits();
    let ref_at_neg_min = Powf::<NEG_CFG>::scalar(4.0, -f32::MIN_POSITIVE).unwrap();
    for exp_in in [-f32::from_bits(1), -0.0_f32, 0.0, 1.0, f32::INFINITY] {
        assert_eq!(
            Powf::<NEG_CFG>::scalar(4.0, exp_in).unwrap().to_bits(),
            ref_at_neg_min.to_bits()
        );
        assert_eq!(
            NEG_OPTS.runtime().scalar(4.0, exp_in).unwrap().to_bits(),
            ref_at_neg_min.to_bits()
        );
    }
}

#[test]
fn ranged_subnormal_admission_and_flush_clamp_preserve_exact_endpoints() {
    let min_sub = f32::from_bits(1);
    let sub_bf16 = f32::from_bits(0x0001_0000);

    // PowfDomain::Unit + admit_subnormals + clamp_out_of_bounds preserves min_sub on Midp & Highp.
    let unit_midp = PowfOptions::new()
        .domain(PowfDomain::Unit)
        .admit_subnormals()
        .clamp_out_of_bounds();
    let unit_highp = unit_midp.highp();
    assert_close_rel(
        unit_midp.runtime().scalar(min_sub, 0.5).unwrap(),
        min_sub.sqrt(),
        1e-6,
        "midp unit subnorm",
    );
    assert_eq!(
        unit_highp.runtime().scalar(min_sub, 0.5).unwrap(),
        min_sub.sqrt()
    );

    // PowfDomain::PositiveUnit + admit_subnormals + clamp_out_of_bounds clamps <= 0 to min_sub (not MIN_POSITIVE).
    let pos_unit_midp = PowfOptions::new()
        .domain(PowfDomain::PositiveUnit)
        .admit_subnormals()
        .clamp_out_of_bounds();
    let pos_unit_highp = pos_unit_midp.highp();
    assert_close_rel(
        pos_unit_midp.runtime().scalar(-1.0, 0.5).unwrap(),
        min_sub.sqrt(),
        1e-6,
        "midp positive unit clamp to min_sub",
    );
    assert_eq!(
        pos_unit_highp.runtime().scalar(-1.0, 0.5).unwrap(),
        min_sub.sqrt()
    );

    // Logf custom [sub_bf16, 1.0] with FlushOrClamp shoves subnormals to MIN_POSITIVE (-126.0), not unscaled sub_bf16.
    let log_flush = LogfOptions::new()
        .input_range(sub_bf16, 1.0)
        .flush_or_clamp_subnormals();
    assert!(log_flush.is_valid());
    assert_eq!(log_flush.runtime().log2(sub_bf16).unwrap(), -126.0);
    assert_eq!(log_flush.runtime().log2(min_sub).unwrap(), -126.0);

    // CbrtfDomain::PositiveNormals + admit_subnormals scales subnormals properly.
    let cbrt_pos_sub = CbrtfOptions::new()
        .domain(CbrtfDomain::PositiveNormals)
        .admit_subnormals();
    assert!(cbrt_pos_sub.is_valid());
    let c = cbrt_pos_sub
        .runtime()
        .scalar(f32::from_bits(0x0002_0000))
        .unwrap();
    assert!((c - 2.0_f32.powi(-44)).abs() < 2.0_f32.powi(-60));
}

#[test]
fn round2_review_witnesses_cbrt_ceiling_lowp_log1p_subnorm_and_singleton_zero_flush() {
    let pos_sub = f32::from_bits(1);
    let neg_sub = -f32::from_bits(1);

    // 1. MAX_CBRT_BF16 (0x7eaa_0000 < f32::MAX / 3.0 = 0x7eaa_aaaa): finite signed results on Midp & Lowp,
    //    ±inf clamp to finite endpoints, and old oversized 0x7ed5 bf16 bound is rejected by from_bits.
    assert_eq!(MAX_CBRT_BF16.to_bits(), 0x7eaa_0000);
    assert!(core::hint::black_box(MAX_CBRT_BF16) < f32::MAX / 3.0);
    assert!(<Cbrtf>::scalar(f32::from_bits(0x7ed5_0000)).is_err());

    const WIDE_CBRT_MIDP: u128 = CbrtfOptions::new()
        .input_range(-MAX_CBRT_BF16, MAX_CBRT_BF16)
        .bits();
    const WIDE_CBRT_LOWP: u128 = CbrtfOptions::new()
        .input_range(-MAX_CBRT_BF16, MAX_CBRT_BF16)
        .lowp()
        .bits();
    const WIDE_CBRT_CLAMP: u128 = CbrtfOptions::new()
        .input_range(-MAX_CBRT_BF16, MAX_CBRT_BF16)
        .clamp_out_of_bounds()
        .bits();

    for (pos_v, neg_v) in [
        (
            Cbrtf::<WIDE_CBRT_MIDP>::scalar(MAX_CBRT_BF16).unwrap(),
            Cbrtf::<WIDE_CBRT_MIDP>::scalar(-MAX_CBRT_BF16).unwrap(),
        ),
        (
            Cbrtf::<WIDE_CBRT_LOWP>::scalar(MAX_CBRT_BF16).unwrap(),
            Cbrtf::<WIDE_CBRT_LOWP>::scalar(-MAX_CBRT_BF16).unwrap(),
        ),
        (
            CbrtfOptions::new()
                .input_range(-MAX_CBRT_BF16, MAX_CBRT_BF16)
                .runtime()
                .scalar(MAX_CBRT_BF16)
                .unwrap(),
            CbrtfOptions::new()
                .input_range(-MAX_CBRT_BF16, MAX_CBRT_BF16)
                .runtime()
                .scalar(-MAX_CBRT_BF16)
                .unwrap(),
        ),
    ] {
        let want = (MAX_CBRT_BF16 as f64).cbrt() as f32;
        assert!(pos_v.is_finite() && pos_v > 0.0);
        assert!(neg_v.is_finite() && neg_v < 0.0);
        assert_close_rel(pos_v, want, 2e-3, "cbrt +MAX_CBRT_BF16");
        assert_close_rel(neg_v, -want, 2e-3, "cbrt -MAX_CBRT_BF16");
    }
    assert_eq!(
        Cbrtf::<WIDE_CBRT_CLAMP>::scalar_infallible(f32::INFINITY).to_bits(),
        Cbrtf::<WIDE_CBRT_MIDP>::scalar(MAX_CBRT_BF16)
            .unwrap()
            .to_bits()
    );
    assert_eq!(
        Cbrtf::<WIDE_CBRT_CLAMP>::scalar_infallible(f32::NEG_INFINITY).to_bits(),
        Cbrtf::<WIDE_CBRT_MIDP>::scalar(-MAX_CBRT_BF16)
            .unwrap()
            .to_bits()
    );

    let bad_old_cbrt_word =
        (WIDE_CBRT_MIDP & !((0xffff_ffff_u128) << 64)) | (0xfed5_u128 << 64) | (0x7ed5_u128 << 80);
    assert_eq!(CbrtfOptions::from_bits(bad_old_cbrt_word), None);

    // 2. LOWP log1p preserves both signs of subnormals bit-for-bit under Normalize and AllFloats.
    const L1P_LOWP_NORM: u128 = Log1pfOptions::new().lowp().admit_subnormals().bits();
    const L1P_LOWP_FULL: u128 = Log1pfOptions::new().lowp().full_domain().bits();
    assert_eq!(
        Log1pf::<L1P_LOWP_NORM>::scalar(pos_sub).unwrap().to_bits(),
        1
    );
    assert_eq!(
        Log1pf::<L1P_LOWP_NORM>::scalar(neg_sub).unwrap().to_bits(),
        0x8000_0001
    );
    assert_eq!(
        Log1pf::<L1P_LOWP_FULL>::scalar(pos_sub).unwrap().to_bits(),
        1
    );
    assert_eq!(
        Log1pf::<L1P_LOWP_FULL>::scalar(neg_sub).unwrap().to_bits(),
        0x8000_0001
    );
    assert_eq!(
        Log1pfOptions::new()
            .lowp()
            .admit_subnormals()
            .runtime()
            .scalar(neg_sub)
            .unwrap()
            .to_bits(),
        0x8000_0001
    );

    // 3. Checked FlushOrClamp on singleton [0.0, 0.0] admits both +sub and -sub cleanly.
    const CBRT_ZERO_FLUSH: u128 = CbrtfOptions::new()
        .input_range(0.0, 0.0)
        .flush_or_clamp_subnormals()
        .bits();
    assert_eq!(
        Cbrtf::<CBRT_ZERO_FLUSH>::scalar(pos_sub).unwrap().to_bits(),
        0
    );
    assert_eq!(
        Cbrtf::<CBRT_ZERO_FLUSH>::scalar(neg_sub).unwrap().to_bits(),
        0x8000_0000
    );

    const L1P_ZERO_FLUSH: u128 = Log1pfOptions::new()
        .input_range(0.0, 0.0)
        .flush_or_clamp_subnormals()
        .bits();
    assert_eq!(
        Log1pf::<L1P_ZERO_FLUSH>::scalar(pos_sub).unwrap().to_bits(),
        0
    );
    assert_eq!(
        Log1pf::<L1P_ZERO_FLUSH>::scalar(neg_sub).unwrap().to_bits(),
        0x8000_0000
    );

    const POW_ZERO_EXP_FLUSH: u128 = PowfOptions::new()
        .exponent_range(0.0, 0.0)
        .flush_or_clamp_subnormals()
        .bits();
    assert_eq!(
        Powf::<POW_ZERO_EXP_FLUSH>::scalar(2.0, pos_sub).unwrap(),
        1.0
    );
    assert_eq!(
        Powf::<POW_ZERO_EXP_FLUSH>::scalar(2.0, neg_sub).unwrap(),
        1.0
    );
    assert_eq!(
        PowfOptions::new()
            .exponent_range(0.0, 0.0)
            .flush_or_clamp_subnormals()
            .runtime()
            .scalar(2.0, neg_sub)
            .unwrap(),
        1.0
    );

    // 4. Log1pfOptions::from_bits rejects redundant EXTRA_NEG_ONE without custom_range on open domains.
    assert_eq!(Log1pfOptions::from_bits(0x0080_000e), None); // GreaterThanNegOne + EXTRA_NEG_ONE
    let open_signed = Log1pfOptions::new()
        .domain(Log1pfDomain::OpenSignedUnit)
        .bits();
    assert_eq!(Log1pfOptions::from_bits(open_signed | (1 << 23)), None);

    // 5. All 33 algebraic (precision x exponent) pairs (11 exponents x {Midp, Lowp, Highp}).
    let alg_exps = [
        0.0_f32, 1.0, -1.0, 2.0, -2.0, 3.0, 4.0, 0.5, -0.5, 1.5, 0.25,
    ];
    let pos_bases = [0.0625_f32, 0.25, 0.5, 1.0, 2.0, 4.0, 9.0, 16.0];
    for &e in &alg_exps {
        for prec in [
            PowfPrecision::Midp,
            PowfPrecision::Lowp,
            PowfPrecision::Highp,
        ] {
            let opts = PowfOptions::new().precision(prec).const_exponent(e);
            assert!(opts.is_valid());
            let got = opts.runtime().array_const(pos_bases).unwrap();
            for (i, &b) in pos_bases.iter().enumerate() {
                let want = (b as f64).powf(e as f64) as f32;
                assert_close_rel(
                    got[i],
                    want,
                    3e-7,
                    "33 algebraic precision x exponent matrix",
                );
            }
        }
    }
}

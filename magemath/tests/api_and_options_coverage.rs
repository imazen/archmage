//! Independent-oracle coverage for Magemath options builders/validators,
//! const & runtime API entry points, `SimdMath` infallible methods, and error
//! `Display` implementations across all five families (`powf`, `expf`, `logf`,
//! `log1pf`, `cbrtf`).

use archmage::ScalarToken;
#[cfg(any(
    all(target_arch = "x86_64", target_feature = "sse2"),
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
use archmage::SimdToken;
use magemath::cbrtf::{
    Cbrtf, CbrtfDomain, CbrtfError, CbrtfErrorKind, CbrtfOptions, CbrtfPrecision, try_cbrt,
    try_cbrt_slice,
};
use magemath::expf::{
    Expf, ExpfDomain, ExpfError, ExpfErrorKind, ExpfOptions, ExpfPrecision, try_exp, try_exp_slice,
    try_exp2, try_exp2_slice,
};
use magemath::log1pf::{
    Log1pf, Log1pfDomain, Log1pfError, Log1pfErrorKind, Log1pfOptions, Log1pfPrecision, try_log1p,
    try_log1p_slice,
};
use magemath::logf::{
    Logf, LogfDomain, LogfError, LogfErrorKind, LogfOptions, LogfPrecision, try_ln, try_ln_slice,
    try_log2, try_log2_slice, try_log10, try_log10_slice,
};
use magemath::powf::{
    Powf, PowfDomain, PowfError, PowfErrorKind, PowfOptions, PowfPrecision, try_powf,
    try_powf_slice,
};
use magemath::{Execution, Portability, SimdMath, SliceError, Subnormals, Validation};
use magetypes::simd::generic::{f32x4, f32x8};

fn ulp_diff(a: f32, b: f32) -> u32 {
    if a == b {
        return 0;
    }
    assert!(
        a.is_finite() && b.is_finite() && a.is_sign_positive() == b.is_sign_positive(),
        "incompatible floats: a={a} ({:#010x}) b={b} ({:#010x})",
        a.to_bits(),
        b.to_bits()
    );
    a.to_bits().abs_diff(b.to_bits())
}

#[test]
fn all_option_builders_predicates_and_from_bits_roundtrips() {
    // PowfOptions
    assert_eq!(PowfOptions::default(), PowfOptions::new());
    let pow_presets = [
        PowfOptions::new().admit_zero().checked().upgrade(),
        PowfOptions::new()
            .admit_zero_base()
            .error_on_out_of_bounds(),
        PowfOptions::new().gamma_domain().midp().bit_exact(),
        PowfOptions::new()
            .positive_gamma_domain()
            .lowp()
            .backend_dependent(),
        PowfOptions::new().unit_gamma_domain().unchecked().caller(),
        PowfOptions::new()
            .srgb_encode_domain()
            .clamp_out_of_bounds(),
        PowfOptions::new().full_domain().highp().clamp_finite(),
        PowfOptions::new()
            .domain(PowfDomain::PositiveNormals)
            .base_range(f32::MIN_POSITIVE, 4.0)
            .exponent_range(-2.0, 2.0)
            .subnormals(Subnormals::Normalize)
            .precision(PowfPrecision::Midp)
            .portability(Portability::BitExact)
            .validation(Validation::Checked)
            .execution(Execution::Upgrade),
        PowfOptions::new().const_exponent(2.5).saturate_output(),
        PowfOptions::new()
            .const_exponent(0.5)
            .clamp_output()
            .assume_in_bounds(),
        PowfOptions::new()
            .reject_subnormals()
            .admit_subnormals()
            .flush_or_clamp_subnormals(),
    ];
    for opts in pow_presets {
        assert!(opts.is_valid(), "expected valid powf options: {opts:?}");
        opts.validate();
        let bits = opts.bits();
        assert_eq!(PowfOptions::from_bits(bits), Some(opts));
        let rt = opts.runtime();
        assert_eq!(rt.options(), opts);
    }
    assert_eq!(
        PowfOptions::new().reject_subnormals().subnormals_policy(),
        Subnormals::Reject
    );
    assert_eq!(
        PowfOptions::new()
            .const_exponent(2.2)
            .const_exponent_value(),
        Some(2.2)
    );
    assert_eq!(PowfOptions::new().const_exponent_value(), None);
    assert!(PowfOptions::new().clamp_out_of_bounds().is_infallible());
    assert!(!PowfOptions::new().checked().is_infallible());
    // Invalid PowfOptions bits (bad family tag / reserved bits)
    assert_eq!(PowfOptions::from_bits(0), None);
    assert_eq!(PowfOptions::from_bits(u128::MAX), None);
    assert!(!PowfOptions::new().full_domain().midp().is_valid());
    assert!(
        !PowfOptions::new()
            .base_range(0.5, 4.0)
            .admit_subnormals()
            .is_valid()
    );

    // ExpfOptions
    assert_eq!(ExpfOptions::default(), ExpfOptions::new());
    let exp_presets = [
        ExpfOptions::new()
            .small_exp2_range()
            .midp()
            .bit_exact()
            .checked(),
        ExpfOptions::new()
            .small_exp_range()
            .lowp()
            .backend_dependent()
            .error_on_out_of_bounds(),
        ExpfOptions::new()
            .unclamped_exp2_range()
            .clamp_out_of_bounds()
            .upgrade(),
        ExpfOptions::new()
            .unclamped_exp_range()
            .clamp_finite()
            .caller(),
        ExpfOptions::new().assume_in_bounds().unchecked(),
        ExpfOptions::new()
            .domain(ExpfDomain::NonPositive)
            .precision(ExpfPrecision::Midp),
    ];
    for opts in exp_presets {
        assert!(opts.is_valid(), "expected valid expf options: {opts:?}");
        opts.validate();
        let bits = opts.bits();
        assert_eq!(ExpfOptions::from_bits(bits), Some(opts));
        assert_eq!(opts.runtime().options(), opts);
    }
    assert_eq!(
        ExpfOptions::new()
            .domain(ExpfDomain::NonPositive)
            .selected_domain(),
        ExpfDomain::NonPositive
    );
    assert!(ExpfOptions::new().clamp_out_of_bounds().is_infallible());
    assert!(!ExpfOptions::new().checked().is_infallible());
    assert_eq!(ExpfOptions::from_bits(0), None);
    assert_eq!(ExpfOptions::from_bits(PowfOptions::new().bits()), None);

    // LogfOptions
    assert_eq!(LogfOptions::default(), LogfOptions::new());
    let log_presets = [
        LogfOptions::new()
            .near_one_range()
            .midp()
            .bit_exact()
            .checked(),
        LogfOptions::new()
            .reject_subnormals()
            .lowp()
            .backend_dependent()
            .error_on_out_of_bounds(),
        LogfOptions::new()
            .admit_subnormals()
            .clamp_out_of_bounds()
            .upgrade(),
        LogfOptions::new().admit_zero().clamp_finite().caller(),
        LogfOptions::new().assume_in_bounds().unchecked(),
        LogfOptions::new()
            .domain(LogfDomain::PositiveNormals)
            .precision(LogfPrecision::Midp),
    ];
    for opts in log_presets {
        assert!(opts.is_valid(), "expected valid logf options: {opts:?}");
        opts.validate();
        let bits = opts.bits();
        assert_eq!(LogfOptions::from_bits(bits), Some(opts));
        assert_eq!(opts.runtime().options(), opts);
    }
    assert_eq!(
        LogfOptions::new().admit_zero().selected_domain(),
        LogfDomain::NonNegative
    );
    assert!(LogfOptions::new().clamp_out_of_bounds().is_infallible());
    assert!(!LogfOptions::new().checked().is_infallible());
    assert_eq!(LogfOptions::from_bits(0), None);
    assert_eq!(LogfOptions::from_bits(ExpfOptions::new().bits()), None);

    // Log1pfOptions
    assert_eq!(Log1pfOptions::default(), Log1pfOptions::new());
    let log1p_presets = [
        Log1pfOptions::new().checked().upgrade().midp().bit_exact(),
        Log1pfOptions::new()
            .error_on_out_of_bounds()
            .caller()
            .lowp()
            .backend_dependent(),
        Log1pfOptions::new()
            .clamp_out_of_bounds()
            .reject_subnormals()
            .admit_neg_one(),
        Log1pfOptions::new()
            .clamp_finite()
            .admit_subnormals()
            .flush_or_clamp_subnormals(),
        Log1pfOptions::new().assume_in_bounds().unchecked(),
        Log1pfOptions::new().around_zero_range(),
        Log1pfOptions::new().small_range(),
        Log1pfOptions::new()
            .domain(Log1pfDomain::OpenSignedUnit)
            .admit_zero(),
        Log1pfOptions::new()
            .domain(Log1pfDomain::OpenSignedUnit)
            .input_range(-1.0, 0.5)
            .admit_neg_one(),
        Log1pfOptions::new()
            .domain(Log1pfDomain::GreaterThanNegOne)
            .input_range(-1.0, 0.5)
            .admit_neg_one(),
        Log1pfOptions::new()
            .domain(Log1pfDomain::Unit)
            .admit_neg_one(),
        Log1pfOptions::new().domain(Log1pfDomain::NonNegative),
        Log1pfOptions::new().domain(Log1pfDomain::SingleOctave),
        Log1pfOptions::new()
            .domain(Log1pfDomain::SingleOctave)
            .lowp(),
        Log1pfOptions::new().admit_all_floats(),
        Log1pfOptions::new().full_domain(),
        Log1pfOptions::new()
            .domain(Log1pfDomain::GreaterThanNegOne)
            .precision(Log1pfPrecision::Midp)
            .input_range(-0.25, 0.25),
    ];
    for opts in log1p_presets {
        assert!(opts.is_valid(), "expected valid log1pf options: {opts:?}");
        opts.validate();
        let bits = opts.bits();
        assert_eq!(Log1pfOptions::from_bits(bits), Some(opts));
        assert_eq!(opts.runtime().options(), opts);
    }
    assert_eq!(
        Log1pfOptions::new().selected_domain(),
        Log1pfDomain::GreaterThanNegOne
    );
    assert_eq!(
        Log1pfOptions::new().reject_subnormals().subnormals_policy(),
        Subnormals::Reject
    );
    assert_eq!(
        Log1pfOptions::new().full_domain().subnormals_policy(),
        Subnormals::Normalize
    );
    assert!(
        Log1pfOptions::new()
            .input_range(-0.25, 0.25)
            .elides_reduction()
    );
    assert!(Log1pfOptions::new().clamp_out_of_bounds().is_infallible());
    assert_eq!(Log1pfOptions::from_bits(0), None);
    assert_eq!(Log1pfOptions::from_bits(LogfOptions::new().bits()), None);
    // Invalid Log1pfOptions combinations (is_valid() == false and validate() panics)
    let invalid_log1p = [
        Log1pfOptions::new().full_domain().input_range(-0.5, 0.5),
        Log1pfOptions::new().full_domain().admit_subnormals(),
        Log1pfOptions::new().input_range(-1.0, 0.5),
        Log1pfOptions::new()
            .input_range(0.5, 2.0)
            .admit_subnormals(),
        Log1pfOptions::new()
            .input_range(0.0, 0.0)
            .admit_subnormals(),
    ];
    for opts in invalid_log1p {
        assert!(
            !opts.is_valid(),
            "expected invalid log1pf options: {opts:?}"
        );
        #[cfg(panic = "unwind")]
        assert!(std::panic::catch_unwind(|| opts.validate()).is_err());
    }
    // Invalid LogfOptions and PowfOptions combinations
    let invalid_log = [
        LogfOptions::new().full_domain().admit_subnormals(),
        LogfOptions::new().input_range(1.0, 2.0).admit_subnormals(),
    ];
    for opts in invalid_log {
        assert!(!opts.is_valid(), "expected invalid logf options: {opts:?}");
        #[cfg(panic = "unwind")]
        assert!(std::panic::catch_unwind(|| opts.validate()).is_err());
    }
    let invalid_pow = [
        PowfOptions::new().full_domain().midp(),
        PowfOptions::new().base_range(0.5, 4.0).admit_subnormals(),
        PowfOptions::new().full_domain().base_range(0.5, 4.0),
    ];
    for opts in invalid_pow {
        assert!(!opts.is_valid(), "expected invalid powf options: {opts:?}");
        #[cfg(panic = "unwind")]
        assert!(std::panic::catch_unwind(|| opts.validate()).is_err());
    }

    // CbrtfOptions
    assert_eq!(CbrtfOptions::default(), CbrtfOptions::new());
    let cbrt_presets = [
        CbrtfOptions::new()
            .unit_range()
            .lowp()
            .bit_exact()
            .checked(),
        CbrtfOptions::new()
            .reject_subnormals()
            .midp()
            .backend_dependent()
            .error_on_out_of_bounds(),
        CbrtfOptions::new()
            .flush_or_clamp_subnormals()
            .clamp_out_of_bounds()
            .upgrade(),
        CbrtfOptions::new()
            .admit_subnormals()
            .clamp_finite()
            .caller(),
        CbrtfOptions::new()
            .admit_all_floats()
            .assume_in_bounds()
            .unchecked(),
        CbrtfOptions::new()
            .full_domain()
            .domain(CbrtfDomain::AllFloats)
            .precision(CbrtfPrecision::Midp),
    ];
    for opts in cbrt_presets {
        assert!(opts.is_valid(), "expected valid cbrtf options: {opts:?}");
        opts.validate();
        let bits = opts.bits();
        assert_eq!(CbrtfOptions::from_bits(bits), Some(opts));
        assert_eq!(opts.runtime().options(), opts);
    }
    assert!(CbrtfOptions::new().clamp_out_of_bounds().is_infallible());
    assert_eq!(CbrtfOptions::from_bits(0), None);
    assert_eq!(CbrtfOptions::from_bits(Log1pfOptions::new().bits()), None);
}

#[test]
fn error_display_and_top_level_try_wrappers() {
    // Error Display formatting for all 5 error structs + SliceError
    assert_eq!(
        format!(
            "{}",
            PowfError {
                kind: PowfErrorKind::InvalidBase,
                lane_mask: 0x0003
            }
        ),
        "powf InvalidBase, lane mask 0x0003"
    );
    assert_eq!(
        format!(
            "{}",
            ExpfError {
                kind: ExpfErrorKind::InvalidInput,
                lane_mask: 0x0001
            }
        ),
        "expf InvalidInput, lane mask 0x0001"
    );
    assert_eq!(
        format!(
            "{}",
            LogfError {
                kind: LogfErrorKind::InvalidInput,
                lane_mask: 0x0002
            }
        ),
        "logf InvalidInput, lane mask 0x0002"
    );
    assert_eq!(
        format!(
            "{}",
            Log1pfError {
                kind: Log1pfErrorKind::InvalidInput,
                lane_mask: 0x0004
            }
        ),
        "log1pf InvalidInput, lane mask 0x0004"
    );
    assert_eq!(
        format!(
            "{}",
            CbrtfError {
                kind: CbrtfErrorKind::InvalidInput,
                lane_mask: 0x0008
            }
        ),
        "cbrtf InvalidInput, lane mask 0x0008"
    );
    assert_eq!(
        format!(
            "{}",
            SliceError {
                kind: CbrtfErrorKind::InvalidInput,
                index: 7
            }
        ),
        "InvalidInput at index 7"
    );

    // Top-level convenience wrappers: valid results vs f64 reference
    assert_eq!(try_powf(4.0, 0.5).unwrap().to_bits(), 2.0_f32.to_bits());
    assert_eq!(try_exp2(3.0).unwrap().to_bits(), 8.0_f32.to_bits());
    assert!(ulp_diff(try_exp(1.0).unwrap(), (1.0_f64).exp() as f32) <= 2);
    assert_eq!(try_log2(8.0).unwrap().to_bits(), 3.0_f32.to_bits());
    assert!(ulp_diff(try_ln(2.0).unwrap(), (2.0_f64).ln() as f32) <= 2);
    assert!(ulp_diff(try_log10(100.0).unwrap(), 2.0_f32) <= 2);
    assert!(ulp_diff(try_log1p(1.0).unwrap(), (1.0_f64).ln_1p() as f32) <= 2);
    assert_eq!(try_cbrt(8.0).unwrap().to_bits(), 2.0_f32.to_bits());

    let src = [1.0_f32, 2.0, 4.0, 8.0, 16.0];
    let mut dst = [0.0_f32; 5];
    try_powf_slice(&src, &mut dst, 0.5).unwrap();
    for (x, y) in src.into_iter().zip(dst) {
        assert!(ulp_diff(y, (x as f64).powf(0.5) as f32) <= 2);
    }
    try_exp2_slice(&[0.0, 1.0, 2.0, 3.0, 4.0], &mut dst).unwrap();
    assert_eq!(dst, [1.0, 2.0, 4.0, 8.0, 16.0]);
    try_exp_slice(&[0.0, 1.0, -1.0, 2.0, -2.0], &mut dst).unwrap();
    for (x, y) in [0.0_f32, 1.0, -1.0, 2.0, -2.0].into_iter().zip(dst) {
        assert!(ulp_diff(y, (x as f64).exp() as f32) <= 3);
    }
    try_log2_slice(&src, &mut dst).unwrap();
    assert_eq!(dst, [0.0, 1.0, 2.0, 3.0, 4.0]);
    try_ln_slice(&src, &mut dst).unwrap();
    for (x, y) in src.into_iter().zip(dst) {
        assert!(ulp_diff(y, (x as f64).ln() as f32) <= 3);
    }
    try_log10_slice(&[1.0, 10.0, 100.0, 1000.0, 10000.0], &mut dst).unwrap();
    for (x, y) in [1.0_f32, 10.0, 100.0, 1000.0, 10000.0].into_iter().zip(dst) {
        assert!(ulp_diff(y, (x as f64).log10() as f32) <= 3);
    }
    try_log1p_slice(&[0.0, 1.0, 3.0, 7.0, 15.0], &mut dst).unwrap();
    for (x, y) in [0.0_f32, 1.0, 3.0, 7.0, 15.0].into_iter().zip(dst) {
        assert!(ulp_diff(y, (x as f64).ln_1p() as f32) <= 2);
    }
    try_cbrt_slice(&[1.0, 8.0, -8.0, 27.0, 64.0], &mut dst).unwrap();
    for (x, y) in [1.0_f32, 8.0, -8.0, 27.0, 64.0].into_iter().zip(dst) {
        assert!(ulp_diff(y, (x as f64).cbrt() as f32) <= 2);
    }
}

#[test]
fn const_api_runtime_and_simd_math_infallible_entry_points() {
    const POW_INF: u128 = PowfOptions::new().clamp_out_of_bounds().bits();
    const POW_CONST_INF: u128 = PowfOptions::new()
        .const_exponent(2.0)
        .clamp_out_of_bounds()
        .bits();
    const POW_CONST_CHK: u128 = PowfOptions::new().const_exponent(0.5).bits();
    const EXP_INF: u128 = ExpfOptions::new().clamp_out_of_bounds().bits();
    const LOG_INF: u128 = LogfOptions::new().clamp_out_of_bounds().bits();
    const LOG1P_INF: u128 = Log1pfOptions::new().clamp_out_of_bounds().bits();
    const CBRT_INF: u128 = CbrtfOptions::new().clamp_out_of_bounds().bits();

    // Infallible scalar entry points
    assert_eq!(
        Powf::<POW_INF>::scalar_infallible(4.0, 0.5).to_bits(),
        2.0_f32.to_bits()
    );
    assert_eq!(
        Powf::<POW_CONST_INF>::scalar_const_infallible(3.0).to_bits(),
        9.0_f32.to_bits()
    );
    assert_eq!(
        Expf::<EXP_INF>::exp2_infallible(3.0).to_bits(),
        8.0_f32.to_bits()
    );
    assert!(ulp_diff(Expf::<EXP_INF>::exp_infallible(1.0), (1.0_f64).exp() as f32) <= 2);
    assert!(
        ulp_diff(
            Log1pf::<LOG1P_INF>::scalar_infallible(1.0),
            (1.0_f64).ln_1p() as f32
        ) <= 2
    );
    assert_eq!(
        Cbrtf::<CBRT_INF>::scalar_infallible(8.0).to_bits(),
        2.0_f32.to_bits()
    );

    // Array & unsupported-width error paths across all 5 const APIs
    assert_eq!(
        Powf::<POW_CONST_CHK>::array_const([4.0, 9.0, 16.0, 25.0]).unwrap(),
        [2.0, 3.0, 4.0, 5.0]
    );
    assert_eq!(
        <Powf>::array([1.0, 2.0, 3.0], 2.0).unwrap_err(),
        PowfError {
            kind: PowfErrorKind::UnsupportedWidth,
            lane_mask: 0
        }
    );
    assert_eq!(
        Powf::<POW_CONST_CHK>::array_const([1.0, 2.0, 3.0]).unwrap_err(),
        PowfError {
            kind: PowfErrorKind::UnsupportedWidth,
            lane_mask: 0
        }
    );
    assert_eq!(
        <Expf>::exp2_array([1.0, 2.0, 3.0]).unwrap_err(),
        ExpfError {
            kind: ExpfErrorKind::UnsupportedWidth,
            lane_mask: 0
        }
    );
    assert_eq!(
        <Logf>::log2_array([1.0, 2.0, 3.0]).unwrap_err(),
        LogfError {
            kind: LogfErrorKind::UnsupportedWidth,
            lane_mask: 0
        }
    );
    assert_eq!(<Log1pf>::array([0.0, 1.0, 2.0, 3.0]).unwrap().len(), 4);
    assert_eq!(
        <Log1pf>::array([0.0, 1.0, 2.0]).unwrap_err(),
        Log1pfError {
            kind: Log1pfErrorKind::UnsupportedWidth,
            lane_mask: 0
        }
    );
    assert_eq!(
        <Cbrtf>::array([1.0, 8.0, 27.0, 64.0])
            .unwrap()
            .map(|v| ulp_diff(v, v.round())),
        [0, 0, 0, 0]
    );
    assert_eq!(
        <Cbrtf>::array([1.0, 2.0, 3.0]).unwrap_err(),
        CbrtfError {
            kind: CbrtfErrorKind::UnsupportedWidth,
            lane_mask: 0
        }
    );

    // Const & runtime *_with_token + *_slice_with_token + *_simd across ScalarToken (and V3 on x86_64)
    let pow_rt = PowfOptions::new().runtime();
    let pow_rt_const = PowfOptions::new().const_exponent(0.5).runtime();
    let exp_rt = ExpfOptions::new().runtime();
    let log_rt = LogfOptions::new().runtime();
    let log1p_rt = Log1pfOptions::new().runtime();
    let cbrt_rt = CbrtfOptions::new().runtime();

    macro_rules! check_token_api {
        ($tok:expr) => {{
            let tok = $tok;
            assert_eq!(<Powf>::scalar_with_token(tok, 4.0, 0.5).unwrap(), 2.0);
            assert_eq!(
                Powf::<POW_CONST_CHK>::scalar_const_with_token(tok, 9.0).unwrap(),
                3.0
            );
            assert_eq!(pow_rt.scalar_with_token(tok, 4.0, 0.5).unwrap(), 2.0);
            assert_eq!(pow_rt_const.scalar_const_with_token(tok, 9.0).unwrap(), 3.0);
            assert_eq!(exp_rt.exp2_with_token(tok, 3.0).unwrap(), 8.0);
            assert!(
                ulp_diff(
                    exp_rt.exp_with_token(tok, 1.0).unwrap(),
                    (1.0_f64).exp() as f32
                ) <= 2
            );
            assert!(ulp_diff(exp_rt.exp(1.0).unwrap(), (1.0_f64).exp() as f32) <= 2);
            assert_eq!(log_rt.log2_with_token(tok, 8.0).unwrap(), 3.0);
            assert!(
                ulp_diff(
                    log_rt.ln_with_token(tok, 2.0).unwrap(),
                    (2.0_f64).ln() as f32
                ) <= 2
            );
            assert!(ulp_diff(log_rt.log10_with_token(tok, 100.0).unwrap(), 2.0) <= 2);
            assert!(
                ulp_diff(
                    log1p_rt.scalar_with_token(tok, 1.0).unwrap(),
                    (1.0_f64).ln_1p() as f32
                ) <= 2
            );
            assert_eq!(cbrt_rt.scalar(8.0).unwrap(), 2.0);
            assert_eq!(cbrt_rt.scalar_with_token(tok, 8.0).unwrap(), 2.0);

            let src = [1.0_f32, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0];
            let mut dst = [0.0_f32; 9];
            <Powf>::slice_with_token(tok, &src, &mut dst, 0.5).unwrap();
            for (x, y) in src.into_iter().zip(dst) {
                assert!(ulp_diff(y, (x as f64).sqrt() as f32) <= 2);
            }
            Powf::<POW_CONST_CHK>::slice_const_with_token(tok, &src, &mut dst).unwrap();
            assert_eq!(dst, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
            pow_rt.slice_with_token(tok, &src, &mut dst, 0.5).unwrap();
            for (x, y) in src.into_iter().zip(dst) {
                assert!(ulp_diff(y, (x as f64).sqrt() as f32) <= 2);
            }
            pow_rt_const
                .slice_const_with_token(tok, &src, &mut dst)
                .unwrap();
            assert_eq!(dst, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
            pow_rt_const.slice_const(&src, &mut dst).unwrap();
            assert_eq!(dst, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

            let exp_in = [0.0_f32, 1.0, 2.0, 3.0, 4.0, -1.0, -2.0, 0.5, 1.5];
            exp_rt
                .exp2_slice_with_token(tok, &exp_in, &mut dst)
                .unwrap();
            for (x, y) in exp_in.into_iter().zip(dst) {
                assert!(ulp_diff(y, (x as f64).exp2() as f32) <= 2);
            }
            exp_rt.exp_slice_with_token(tok, &exp_in, &mut dst).unwrap();
            for (x, y) in exp_in.into_iter().zip(dst) {
                assert!(ulp_diff(y, (x as f64).exp() as f32) <= 3);
            }
            log_rt.log2_slice_with_token(tok, &src, &mut dst).unwrap();
            for (x, y) in src.into_iter().zip(dst) {
                assert!(ulp_diff(y, (x as f64).log2() as f32) <= 2);
            }
            log_rt.ln_slice_with_token(tok, &src, &mut dst).unwrap();
            for (x, y) in src.into_iter().zip(dst) {
                assert!(ulp_diff(y, (x as f64).ln() as f32) <= 3);
            }
            log_rt.log10_slice_with_token(tok, &src, &mut dst).unwrap();
            for (x, y) in src.into_iter().zip(dst) {
                assert!(ulp_diff(y, (x as f64).log10() as f32) <= 3);
            }
            log1p_rt.slice_with_token(tok, &src, &mut dst).unwrap();
            for (x, y) in src.into_iter().zip(dst) {
                assert!(ulp_diff(y, (x as f64).ln_1p() as f32) <= 2);
            }
            cbrt_rt.slice_with_token(tok, &src, &mut dst).unwrap();
            for (x, y) in src.into_iter().zip(dst) {
                assert!(ulp_diff(y, (x as f64).cbrt() as f32) <= 2);
            }

            // Runtime slice error refinement: error inside full SIMD chunk (lane 3) AND in tail (index 8)
            for err_idx in [3usize, 8usize] {
                let mut bad = src;
                bad[err_idx] = -4.0;
                let err = pow_rt
                    .slice_with_token(tok, &bad, &mut dst, 0.5)
                    .unwrap_err();
                assert_eq!((err.kind, err.index), (PowfErrorKind::InvalidBase, err_idx));
                let err = pow_rt_const
                    .slice_const_with_token(tok, &bad, &mut dst)
                    .unwrap_err();
                assert_eq!((err.kind, err.index), (PowfErrorKind::InvalidBase, err_idx));
                let err = <Powf>::slice_with_token(tok, &bad, &mut dst, 0.5).unwrap_err();
                assert_eq!((err.kind, err.index), (PowfErrorKind::InvalidBase, err_idx));
                let err = Powf::<POW_CONST_CHK>::slice_const_with_token(tok, &bad, &mut dst)
                    .unwrap_err();
                assert_eq!((err.kind, err.index), (PowfErrorKind::InvalidBase, err_idx));
                let err = log_rt.ln_slice_with_token(tok, &bad, &mut dst).unwrap_err();
                assert_eq!(
                    (err.kind, err.index),
                    (LogfErrorKind::InvalidInput, err_idx)
                );
                let err = log1p_rt.slice_with_token(tok, &bad, &mut dst).unwrap_err();
                assert_eq!(
                    (err.kind, err.index),
                    (Log1pfErrorKind::InvalidInput, err_idx)
                );
                let mut bad_nan = src;
                bad_nan[err_idx] = f32::NAN;
                let err = exp_rt
                    .exp2_slice_with_token(tok, &bad_nan, &mut dst)
                    .unwrap_err();
                assert_eq!(
                    (err.kind, err.index),
                    (ExpfErrorKind::InvalidInput, err_idx)
                );
                let err = cbrt_rt
                    .slice_with_token(tok, &bad_nan, &mut dst)
                    .unwrap_err();
                assert_eq!(
                    (err.kind, err.index),
                    (CbrtfErrorKind::InvalidInput, err_idx)
                );
            }

            // Runtime & const array entry points at width 4 and width 8 + subnormal exponent handling
            assert_eq!(
                pow_rt.array([1.0, 4.0, 9.0, 16.0], 0.5).unwrap(),
                [1.0, 2.0, 3.0, 4.0]
            );
            assert_eq!(
                pow_rt_const.array_const([1.0, 4.0, 9.0, 16.0]).unwrap(),
                [1.0, 2.0, 3.0, 4.0]
            );
            let a8 = [1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0];
            assert_eq!(
                Powf::<POW_CONST_CHK>::array_const(a8).unwrap(),
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            );
            for (got, x) in <Powf>::array(a8, 0.5).unwrap().into_iter().zip(a8) {
                assert!(ulp_diff(got, (x as f64).sqrt() as f32) <= 2);
            }
            // Subnormal exponent flushing / rejection / clamping in prepare_exponent_with_bounds
            let subnorm = f32::from_bits(1);
            assert_eq!(
                PowfOptions::new()
                    .exponent_range(-1.0, 1.0)
                    .runtime()
                    .scalar(4.0, subnorm)
                    .unwrap(),
                1.0
            );
            assert_eq!(
                PowfOptions::new()
                    .exponent_range(0.5, 2.0)
                    .runtime()
                    .scalar(4.0, subnorm)
                    .unwrap(),
                2.0
            );
            assert_eq!(
                PowfOptions::new()
                    .exponent_range(-2.0, -0.5)
                    .runtime()
                    .scalar(4.0, -subnorm)
                    .unwrap(),
                0.5
            );
            assert!(
                PowfOptions::new()
                    .reject_subnormals()
                    .runtime()
                    .scalar(4.0, subnorm)
                    .is_err()
            );
            assert!(
                PowfOptions::new()
                    .clamp_out_of_bounds()
                    .runtime()
                    .scalar(4.0, f32::NAN)
                    .unwrap()
                    .is_nan()
            );
            assert_eq!(
                PowfOptions::new()
                    .exponent_range(0.5, 2.0)
                    .clamp_out_of_bounds()
                    .runtime()
                    .scalar(4.0, 10.0)
                    .unwrap(),
                16.0
            );

            // Const & runtime simd + SimdMath infallible trait methods on f32x4 and f32x8
            let v4 = f32x4::from_array_t(tok, [1.0, 4.0, 9.0, 16.0]);
            assert_eq!(
                <Powf>::simd(v4, 0.5).unwrap().to_array(),
                [1.0, 2.0, 3.0, 4.0]
            );
            assert_eq!(
                Powf::<POW_CONST_CHK>::simd_const(v4).unwrap().to_array(),
                [1.0, 2.0, 3.0, 4.0]
            );
            assert_eq!(
                <Expf>::exp2_simd(f32x4::from_array_t(tok, [0.0, 1.0, 2.0, 3.0]))
                    .unwrap()
                    .to_array(),
                [1.0, 2.0, 4.0, 8.0]
            );
            assert!(<Expf>::exp_simd(f32x4::from_array_t(tok, [0.0, 1.0, -1.0, 2.0])).is_ok());
            assert!(<Log1pf>::simd(v4).is_ok());
            assert!(<Cbrtf>::simd(v4).is_ok());

            let v8 = f32x8::from_array_t(tok, [1.0, 2.0, 4.0, 8.0, 1.0, 4.0, 9.0, 16.0]);
            assert_eq!(
                v8.powf::<POW_INF>(0.5).to_array()[4..],
                [1.0, 2.0, 3.0, 4.0]
            );
            assert_eq!(
                v8.powf_const::<POW_CONST_INF>().to_array()[..4],
                [1.0, 4.0, 16.0, 64.0]
            );
            assert_eq!(
                f32x8::from_array_t(tok, [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0])
                    .exp2::<EXP_INF>()
                    .to_array(),
                [1.0, 2.0, 4.0, 8.0, 1.0, 2.0, 4.0, 8.0]
            );
            for (x, y) in [0.0_f32, 1.0, -1.0, 2.0].into_iter().zip(
                f32x4::from_array_t(tok, [0.0, 1.0, -1.0, 2.0])
                    .exp::<EXP_INF>()
                    .to_array(),
            ) {
                assert!(ulp_diff(y, (x as f64).exp() as f32) <= 3);
            }
            assert_eq!(
                f32x4::from_array_t(tok, [1.0, 2.0, 4.0, 8.0])
                    .log2::<LOG_INF>()
                    .to_array(),
                [0.0, 1.0, 2.0, 3.0]
            );
            for (x, y) in [1.0_f32, 2.0, 4.0, 8.0].into_iter().zip(
                f32x4::from_array_t(tok, [1.0, 2.0, 4.0, 8.0])
                    .ln::<LOG_INF>()
                    .to_array(),
            ) {
                assert!(ulp_diff(y, (x as f64).ln() as f32) <= 3);
            }
            for (x, y) in [1.0_f32, 10.0, 100.0, 1000.0].into_iter().zip(
                f32x4::from_array_t(tok, [1.0, 10.0, 100.0, 1000.0])
                    .log10::<LOG_INF>()
                    .to_array(),
            ) {
                assert!(ulp_diff(y, (x as f64).log10() as f32) <= 3);
            }
            for (x, y) in [0.0_f32, 1.0, 3.0, 7.0].into_iter().zip(
                f32x4::from_array_t(tok, [0.0, 1.0, 3.0, 7.0])
                    .log1p::<LOG1P_INF>()
                    .to_array(),
            ) {
                assert!(ulp_diff(y, (x as f64).ln_1p() as f32) <= 2);
            }
            assert_eq!(
                f32x4::from_array_t(tok, [1.0, 8.0, 27.0, 64.0])
                    .cbrt::<CBRT_INF>()
                    .to_array()
                    .map(|v| ulp_diff(v, v.round())),
                [0, 0, 0, 0]
            );
        }};
    }

    check_token_api!(ScalarToken);
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    if let Some(v3) = archmage::X64V3Token::summon() {
        check_token_api!(v3);
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(neon) = archmage::NeonToken::summon() {
        check_token_api!(neon);
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(wasm) = archmage::Wasm128Token::summon() {
        check_token_api!(wasm);
    }
}

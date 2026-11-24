//! Independent-oracle coverage for `simd::scalar` (`*x1` types + deprecated
//! token forwarders), `numerics` option encoding/decoding/errors, and
//! `nostd_math` f64/f32 software math helpers.

use archmage::ScalarToken;
use magetypes::nostd_math;
use magetypes::numerics::{
    ExtremaDomain, ExtremaOptions, HalfConvert, HalfDomain, HalfRounding, HalfSubnormals,
    IntBounds, IntConvert, IntRounding, NumericsError, NumericsErrorKind, RecipDomain,
    RecipOptions, RecipPrecision,
};
use magetypes::simd::scalar::{f32x1, f64x1, i8x1, i16x1, i32x1, i64x1, u8x1, u16x1, u32x1, u64x1};

#[test]
fn scalar_f32x1_and_f64x1_match_ieee_and_rust_primitives() {
    let t = ScalarToken;

    // f32x1 constructors & storage
    assert_eq!(f32x1::LANES, 1);
    assert_eq!(f32x1::zero_t(t).to_array()[0].to_bits(), 0);
    assert_eq!(
        f32x1::splat_t(t, -0.0).to_array()[0].to_bits(),
        (-0.0f32).to_bits()
    );
    assert_eq!(f32x1::load_t(t, &[3.25]).to_array(), [3.25]);
    assert_eq!(f32x1::from_array_t(t, [-2.5]).to_array(), [-2.5]);
    let mut out32 = [0.0f32; 1];
    f32x1::splat_t(t, 7.5).store(&mut out32);
    assert_eq!(out32, [7.5]);

    // Deprecated token forwarders in scalar_token_aliases.rs
    #[allow(deprecated)]
    {
        assert_eq!(f32x1::splat(t, 1.5), f32x1(1.5));
        assert_eq!(f32x1::zero(t), f32x1(0.0));
        assert_eq!(f32x1::load(t, &[2.5]), f32x1(2.5));
        assert_eq!(f32x1::from_array(t, [3.5]), f32x1(3.5));
    }

    // Reductions, arithmetic, abs, rounding, trunc, sqrt, to_i32_fast
    let a = f32x1(6.5);
    let b = f32x1(-2.5);
    assert_eq!(a.reduce_add(), 6.5);
    assert_eq!(a.reduce_min(), 6.5);
    assert_eq!(a.reduce_max(), 6.5);
    assert_eq!((a + b).0, 4.0);
    assert_eq!((a - b).0, 9.0);
    assert_eq!((a * b).0, -16.25);
    assert_eq!((a / b).0, -2.6);
    assert_eq!((-a).0, -6.5);
    assert_eq!(f32x1(-0.0).abs().0.to_bits(), 0);
    assert_eq!(f32x1(-3.75).abs().0, 3.75);
    assert_eq!(f32x1(9.0).sqrt().0, 3.0);
    assert_eq!(f32x1(-2.5).floor().0, -3.0);
    assert_eq!(f32x1(-2.5).ceil().0, -2.0);
    assert_eq!(f32x1(2.5).round().0, 2.0);
    assert_eq!(f32x1(3.5).round().0, 4.0);
    assert_eq!(f32x1(-0.75).trunc().0.to_bits(), (-0.0f32).to_bits());
    assert_eq!(f32x1(-3.9).to_i32_fast(), i32x1(-3));

    // Extrema & clamp
    assert_eq!(f32x1(f32::NAN).min(f32x1(2.0)).0, 2.0);
    assert_eq!(f32x1(f32::NAN).max(f32x1(2.0)).0, 2.0);
    assert!(f32x1(f32::NAN).minimum(f32x1(2.0)).0.is_nan());
    assert!(f32x1(f32::NAN).maximum(f32x1(2.0)).0.is_nan());
    assert_eq!(
        f32x1(-0.0).minimum(f32x1(0.0)).0.to_bits(),
        (-0.0f32).to_bits()
    );
    assert_eq!(f32x1(-0.0).maximum(f32x1(0.0)).0.to_bits(), 0);
    assert_eq!(f32x1(3.0).min_fast(f32x1(1.0)).0, 1.0);
    assert_eq!(f32x1(3.0).max_fast(f32x1(5.0)).0, 5.0);
    assert_eq!(f32x1(10.0).clamp(f32x1(-1.0), f32x1(4.0)).0, 4.0);

    // FMA vs relaxed witness: (1 + 2^-23) * (1 - 2^-23) - 1 = -2^-46 fused, 0.0 relaxed
    let x = f32x1(1.0 + f32::EPSILON);
    let y = f32x1(1.0 - f32::EPSILON);
    let one = f32x1(1.0);
    let expected_fused = ((x.0 as f64) * (y.0 as f64) - 1.0_f64) as f32;
    assert_eq!(x.mul_sub_relaxed(y, one).0, x.0 * y.0 - 1.0);
    assert_eq!(x.mul_add_relaxed(y, -one).0, x.0 * y.0 - 1.0);
    assert_eq!(
        x.mul_sub_fused(y, one).0.to_bits(),
        expected_fused.to_bits()
    );
    assert_eq!(
        x.mul_add_fused(y, -one).0.to_bits(),
        expected_fused.to_bits()
    );
    assert_eq!(
        x.mul_add_portable(y, -one).0.to_bits(),
        expected_fused.to_bits()
    );
    assert_eq!(
        x.mul_sub_portable(y, one).0.to_bits(),
        expected_fused.to_bits()
    );
    assert_eq!(x.mul_neg_add_relaxed(y, one).0, (-x.0) * y.0 + 1.0);
    assert_eq!(
        x.mul_neg_add_fused(y, one).0.to_bits(),
        (-expected_fused).to_bits()
    );
    #[allow(deprecated)]
    {
        assert_eq!(x.mul_add(y, -one), x.mul_add_relaxed(y, -one));
        assert_eq!(x.mul_sub(y, one), x.mul_sub_relaxed(y, one));
    }

    // f64x1 complete surface
    assert_eq!(f64x1::LANES, 1);
    assert_eq!(f64x1::zero_t(t).to_array()[0].to_bits(), 0);
    assert_eq!(
        f64x1::splat_t(t, -0.0).to_array()[0].to_bits(),
        (-0.0f64).to_bits()
    );
    assert_eq!(f64x1::load_t(t, &[3.25]).to_array(), [3.25]);
    assert_eq!(f64x1::from_array_t(t, [-2.5]).to_array(), [-2.5]);
    let mut out64 = [0.0f64; 1];
    f64x1::splat_t(t, 7.5).store(&mut out64);
    assert_eq!(out64, [7.5]);
    #[allow(deprecated)]
    {
        assert_eq!(f64x1::splat(t, 1.5), f64x1(1.5));
        assert_eq!(f64x1::zero(t), f64x1(0.0));
        assert_eq!(f64x1::load(t, &[2.5]), f64x1(2.5));
        assert_eq!(f64x1::from_array(t, [3.5]), f64x1(3.5));
    }

    let d1 = f64x1(6.5);
    let d2 = f64x1(-2.5);
    assert_eq!(d1.reduce_add(), 6.5);
    assert_eq!(d1.reduce_min(), 6.5);
    assert_eq!(d1.reduce_max(), 6.5);
    assert_eq!((d1 + d2).0, 4.0);
    assert_eq!((d1 - d2).0, 9.0);
    assert_eq!((d1 * d2).0, -16.25);
    assert_eq!((d1 / d2).0, -2.6);
    assert_eq!((-d1).0, -6.5);
    assert_eq!(f64x1(-0.0).abs().0.to_bits(), 0);
    assert_eq!(f64x1(9.0).sqrt().0, 3.0);
    assert_eq!(f64x1(-0.75).trunc().0.to_bits(), (-0.0f64).to_bits());
    assert_eq!(f64x1(f64::NAN).min(f64x1(2.0)).0, 2.0);
    assert_eq!(f64x1(f64::NAN).max(f64x1(2.0)).0, 2.0);
    assert!(f64x1(f64::NAN).minimum(f64x1(2.0)).0.is_nan());
    assert!(f64x1(f64::NAN).maximum(f64x1(2.0)).0.is_nan());
    assert_eq!(
        f64x1(-0.0).minimum(f64x1(0.0)).0.to_bits(),
        (-0.0f64).to_bits()
    );
    assert_eq!(f64x1(-0.0).maximum(f64x1(0.0)).0.to_bits(), 0);
    assert_eq!(f64x1(3.0).min_fast(f64x1(1.0)).0, 1.0);
    assert_eq!(f64x1(3.0).max_fast(f64x1(5.0)).0, 5.0);
    assert_eq!(f64x1(-5.0).clamp(f64x1(0.0), f64x1(1.0)).0, 0.0);

    let dx = f64x1(1.0 + f64::EPSILON);
    let dy = f64x1(1.0 - f64::EPSILON);
    let done = f64x1(1.0);
    assert_eq!(dx.mul_add_relaxed(dy, -done).0, dx.0 * dy.0 - 1.0);
    assert_eq!(
        dx.mul_add_fused(dy, -done).0.to_bits(),
        dx.0.mul_add(dy.0, -1.0).to_bits()
    );
    assert_eq!(
        dx.mul_add_portable(dy, -done).0.to_bits(),
        dx.0.mul_add(dy.0, -1.0).to_bits()
    );
    assert_eq!(dx.mul_neg_add_relaxed(dy, done).0, (-dx.0) * dy.0 + 1.0);
    assert_eq!(
        dx.mul_neg_add_fused(dy, done).0.to_bits(),
        (-dx.0).mul_add(dy.0, 1.0).to_bits()
    );
    #[allow(deprecated)]
    {
        assert_eq!(dx.mul_add(dy, -done), dx.mul_add_relaxed(dy, -done));
    }
}

#[test]
fn scalar_integer_x1_types_and_deprecated_aliases() {
    let t = ScalarToken;

    macro_rules! check_int_x1 {
        ($ty:ident, $elem:ty, $a:expr, $b:expr) => {{
            assert_eq!($ty::LANES, 1);
            assert_eq!($ty::zero_t(t), $ty(0));
            assert_eq!($ty::splat_t(t, $a), $ty($a));
            assert_eq!($ty::from_array_t(t, [$b]).to_array(), [$b]);
            assert_eq!($ty($a) + $ty($b), $ty(($a as $elem).wrapping_add($b)));
            assert_eq!($ty($a) - $ty($b), $ty(($a as $elem).wrapping_sub($b)));
            #[allow(deprecated)]
            {
                assert_eq!($ty::splat(t, $a), $ty($a));
                assert_eq!($ty::zero(t), $ty(0));
                assert_eq!($ty::from_array(t, [$b]), $ty($b));
            }
        }};
    }

    check_int_x1!(i8x1, i8, i8::MAX, 2);
    check_int_x1!(u8x1, u8, 250, 20);
    check_int_x1!(i16x1, i16, i16::MAX, 7);
    check_int_x1!(u16x1, u16, u16::MAX, 7);
    check_int_x1!(i32x1, i32, i32::MAX, 5);
    check_int_x1!(u32x1, u32, u32::MAX, 5);
    check_int_x1!(i64x1, i64, i64::MAX, 9);
    check_int_x1!(u64x1, u64, u64::MAX, 9);

    // Extra methods on i32x1 and u32x1 (reduce_add, Mul)
    assert_eq!(i32x1(-123).reduce_add(), -123);
    assert_eq!(
        i32x1(100_000) * i32x1(100_000),
        i32x1(100_000i32.wrapping_mul(100_000))
    );
    assert_eq!(u32x1(456).reduce_add(), 456);
    assert_eq!(
        u32x1(100_000) * u32x1(100_000),
        u32x1(100_000u32.wrapping_mul(100_000))
    );
}

#[test]
fn numerics_options_roundtrip_reject_reserved_bits_and_format_errors() {
    // IntConvert: all 8 combinations + fluent aliases + reserved/foreign rejections
    assert_eq!(IntConvert::default(), IntConvert::new());
    for rounding in [IntRounding::NearestEven, IntRounding::Trunc] {
        for bounds in [
            IntBounds::Finite,
            IntBounds::InRange,
            IntBounds::NonNegativeFinite,
            IntBounds::SaturateAll,
        ] {
            let opts = IntConvert::new().rounding(rounding).bounds(bounds);
            let bits = opts.bits();
            assert_eq!(IntConvert::from_bits(bits), Some(opts));
            // Reserved payload bit rejected
            assert_eq!(IntConvert::from_bits(bits | 0x80), None);
        }
    }
    assert_eq!(
        IntConvert::new().trunc().saturate_all(),
        IntConvert::new()
            .rounding(IntRounding::Trunc)
            .bounds(IntBounds::SaturateAll)
    );
    assert_eq!(IntConvert::new().nearest_even().finite(), IntConvert::new());
    assert_eq!(
        IntConvert::new().in_range(),
        IntConvert::new().bounds(IntBounds::InRange)
    );
    assert_eq!(
        IntConvert::new().non_negative_finite(),
        IntConvert::new().bounds(IntBounds::NonNegativeFinite)
    );
    assert_eq!(IntConvert::from_bits(HalfConvert::new().bits()), None);

    // HalfConvert: all 8 combinations + fluent aliases + rejections
    assert_eq!(HalfConvert::default(), HalfConvert::new());
    for domain in [HalfDomain::Finite, HalfDomain::AllFloats] {
        for subnormals in [HalfSubnormals::FlushOrClamp, HalfSubnormals::Normalize] {
            for rounding in [HalfRounding::NearestEven, HalfRounding::Trunc] {
                let opts = HalfConvert::new()
                    .domain(domain)
                    .subnormals(subnormals)
                    .rounding(rounding);
                let bits = opts.bits();
                assert_eq!(HalfConvert::from_bits(bits), Some(opts));
                assert_eq!(HalfConvert::from_bits(bits | 0x8), None);
            }
        }
    }
    assert_eq!(
        HalfConvert::new().all_floats().normalize().trunc(),
        HalfConvert::new()
            .domain(HalfDomain::AllFloats)
            .subnormals(HalfSubnormals::Normalize)
            .rounding(HalfRounding::Trunc)
    );
    assert_eq!(HalfConvert::new().flush_or_clamp(), HalfConvert::new());
    assert_eq!(HalfConvert::from_bits(IntConvert::new().bits()), None);

    // RecipOptions: all 8 combinations + fluent aliases + rejections
    assert_eq!(RecipOptions::default(), RecipOptions::new());
    for precision in [
        RecipPrecision::Approx,
        RecipPrecision::Nr1,
        RecipPrecision::Nr1Ceil,
        RecipPrecision::Exact,
    ] {
        for domain in [RecipDomain::FiniteNormal, RecipDomain::AllFloats] {
            let opts = RecipOptions::new().precision(precision).domain(domain);
            let bits = opts.bits();
            assert_eq!(RecipOptions::from_bits(bits), Some(opts));
            assert_eq!(RecipOptions::from_bits(bits | 0x8), None);
        }
    }
    assert_eq!(
        RecipOptions::new().approx().all_floats(),
        RecipOptions::new()
            .precision(RecipPrecision::Approx)
            .domain(RecipDomain::AllFloats)
    );
    assert_eq!(RecipOptions::new().nr1(), RecipOptions::new());
    assert_eq!(
        RecipOptions::new().nr1_ceil(),
        RecipOptions::new().precision(RecipPrecision::Nr1Ceil)
    );
    assert_eq!(
        RecipOptions::new().exact(),
        RecipOptions::new().precision(RecipPrecision::Exact)
    );
    assert_eq!(RecipOptions::from_bits(ExtremaOptions::new().bits()), None);

    // ExtremaOptions: all 4 domains + fluent aliases + rejections
    assert_eq!(ExtremaOptions::default(), ExtremaOptions::new());
    for domain in [
        ExtremaDomain::Finite,
        ExtremaDomain::NonNanRhs,
        ExtremaDomain::DropNan,
        ExtremaDomain::PropagateNan,
    ] {
        let opts = ExtremaOptions::new().domain(domain);
        let bits = opts.bits();
        assert_eq!(ExtremaOptions::from_bits(bits), Some(opts));
        assert_eq!(ExtremaOptions::from_bits(bits | 0x4), None);
    }
    assert_eq!(
        ExtremaOptions::new().non_nan_rhs(),
        ExtremaOptions::new().domain(ExtremaDomain::NonNanRhs)
    );
    assert_eq!(
        ExtremaOptions::new().drop_nan(),
        ExtremaOptions::new().domain(ExtremaDomain::DropNan)
    );
    assert_eq!(
        ExtremaOptions::new().propagate_nan(),
        ExtremaOptions::new().domain(ExtremaDomain::PropagateNan)
    );
    assert_eq!(ExtremaOptions::from_bits(RecipOptions::new().bits()), None);

    // NumericsError Display
    let err = NumericsError {
        kind: NumericsErrorKind::IntegerBounds,
        lane_mask: 0x0005,
    };
    assert_eq!(format!("{err}"), "IntegerBounds (lane mask 0x0005)");
}

#[test]
fn nostd_math_f64_transcendentals_and_cbrt_match_stdlib_reference() {
    // Special values for log2_f64
    assert!(nostd_math::log2_f64(f64::NAN).is_nan());
    assert!(nostd_math::log2_f64(-1.0).is_nan());
    assert_eq!(nostd_math::log2_f64(0.0), f64::NEG_INFINITY);
    assert_eq!(nostd_math::log2_f64(-0.0), f64::NEG_INFINITY);
    assert_eq!(nostd_math::log2_f64(f64::INFINITY), f64::INFINITY);

    // Special values for exp2_f64
    assert!(nostd_math::exp2_f64(f64::NAN).is_nan());
    assert_eq!(nostd_math::exp2_f64(1024.0), f64::INFINITY);
    assert_eq!(nostd_math::exp2_f64(f64::INFINITY), f64::INFINITY);
    assert_eq!(nostd_math::exp2_f64(-1075.0), 0.0);
    assert_eq!(nostd_math::exp2_f64(f64::NEG_INFINITY), 0.0);

    // Accuracy checks against stdlib f64 across magnitudes (documented ~1% lowp tier;
    // actual rational poly is ~1e-5 relative error across normal inputs)
    for x in [1e-100_f64, 0.01, 0.5, 1.0, 1.5, 2.0, 10.0, 1234.5, 1e100] {
        let got_l2 = nostd_math::log2_f64(x);
        let ref_l2 = x.log2();
        let err_l2 = (got_l2 - ref_l2).abs() / ref_l2.abs().max(1.0);
        assert!(
            err_l2 < 1e-4,
            "log2_f64({x}): got={got_l2} ref={ref_l2} err={err_l2}"
        );

        let got_ln = nostd_math::ln_f64(x);
        let ref_ln = x.ln();
        let err_ln = (got_ln - ref_ln).abs() / ref_ln.abs().max(1.0);
        assert!(
            err_ln < 1e-4,
            "ln_f64({x}): got={got_ln} ref={ref_ln} err={err_ln}"
        );

        let got_l10 = nostd_math::log10_f64(x);
        let ref_l10 = x.log10();
        let err_l10 = (got_l10 - ref_l10).abs() / ref_l10.abs().max(1.0);
        assert!(
            err_l10 < 1e-4,
            "log10_f64({x}): got={got_l10} ref={ref_l10} err={err_l10}"
        );
    }

    // Degree-4 Taylor polynomial on xf in [0, 1) has max relative error ~6.7e-4 as xf -> 1.
    for x in [
        -500.0_f64, -10.5, -1.0, -0.05, 0.0, 0.5, 0.75, 0.95, 1.0, 10.5, 500.0,
    ] {
        let got_e2 = nostd_math::exp2_f64(x);
        let ref_e2 = x.exp2();
        let err_e2 = (got_e2 - ref_e2).abs() / ref_e2;
        assert!(
            err_e2 < 1e-3,
            "exp2_f64({x}): got={got_e2} ref={ref_e2} err={err_e2}"
        );
    }

    for x in [-100.0_f64, -1.0, 0.0, 1.0, 20.0, 100.0] {
        let got_e = nostd_math::exp_f64(x);
        let ref_e = x.exp();
        let err_e = (got_e - ref_e).abs() / ref_e;
        assert!(
            err_e < 1e-3,
            "exp_f64({x}): got={got_e} ref={ref_e} err={err_e}"
        );
    }

    for (x, n) in [(2.0_f64, 3.0), (0.5, 2.2), (10.0, -0.5), (3.5, 1.25)] {
        let got = nostd_math::powf_f64(x, n);
        let expected = x.powf(n);
        let err = (got - expected).abs() / expected;
        assert!(
            err < 1e-3,
            "powf_f64({x}, {n}): got={got} ref={expected} err={err}"
        );
    }

    // cbrt_lowp_f32 / cbrt_midp_f32 signed zero + accuracy vs stdlib f32::cbrt
    assert_eq!(nostd_math::cbrt_lowp_f32(0.0).to_bits(), 0);
    assert_eq!(
        nostd_math::cbrt_lowp_f32(-0.0).to_bits(),
        (-0.0f32).to_bits()
    );
    assert_eq!(nostd_math::cbrt_midp_f32(0.0).to_bits(), 0);
    assert_eq!(
        nostd_math::cbrt_midp_f32(-0.0).to_bits(),
        (-0.0f32).to_bits()
    );
    // Use exact mathematical cube roots (avoiding std::f32::cbrt which Miri
    // deliberately perturbs by non-deterministic ULPs).
    let cbrt_cases: &[(f32, f32)] = &[
        (-1e24_f32, -1e8_f32),
        (-27.0, -3.0),
        (-10.0, -2.154_434_7),
        (-8.0, -2.0),
        (-0.125, -0.5),
        (1e-24, 1e-8),
        (0.125, 0.5),
        (1.0, 1.0),
        (2.0, 1.259_921_1),
        (8.0, 2.0),
        (10.0, 2.154_434_7),
        (27.0, 3.0),
        (1e24, 1e8),
    ];
    for &(x, expected) in cbrt_cases {
        let low = nostd_math::cbrt_lowp_f32(x);
        let mid = nostd_math::cbrt_midp_f32(x);
        let low_ulps = low.to_bits().abs_diff(expected.to_bits());
        let mid_ulps = mid.to_bits().abs_diff(expected.to_bits());
        assert!(low_ulps <= 260, "cbrt_lowp_f32({x}): ulps={low_ulps}");
        assert!(mid_ulps <= 3, "cbrt_midp_f32({x}): ulps={mid_ulps}");
    }

    // fmaf_soft matches f64 exact product + sum
    let a = 1.0_f32 + f32::EPSILON;
    let b = 1.0_f32 - f32::EPSILON;
    let exact = ((a as f64) * (b as f64) - 1.0_f64) as f32;
    assert_eq!(nostd_math::fmaf_soft(a, b, -1.0).to_bits(), exact.to_bits());
    assert_eq!(nostd_math::fmaf(a, b, -1.0).to_bits(), exact.to_bits());
    assert_eq!(
        nostd_math::fma(1.0 + f64::EPSILON, 1.0 - f64::EPSILON, -1.0).to_bits(),
        (1.0 + f64::EPSILON)
            .mul_add(1.0 - f64::EPSILON, -1.0)
            .to_bits()
    );
}

#[test]
fn scalar_module_portable_and_token_matched_helpers() {
    #[cfg(any(
        all(target_arch = "x86_64", target_feature = "sse2"),
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ))]
    use archmage::SimdToken;
    use magetypes::scalar::{
        F32ScalarBackend, F64ScalarBackend, ceil_f32, ceil_f64, clamp_f32, clamp_f64, floor_f32,
        floor_f64, max_fast_f32, max_fast_f64, maximum_f32, maximum_f64, min_fast_f32,
        min_fast_f64, minimum_f32, minimum_f64, mul_add_fused_f32, mul_add_fused_f64,
        mul_add_relaxed_f32, mul_add_relaxed_f64, mul_sub_fused_f32, mul_sub_fused_f64,
        mul_sub_relaxed_f32, mul_sub_relaxed_f64, round_f32, round_f64, sqrt_f32, sqrt_f64, to_i32,
        to_i32_fast,
    };

    // Portable helpers (no token)
    assert!(minimum_f32(f32::NAN, 1.0).is_nan());
    assert!(maximum_f32(1.0, f32::NAN).is_nan());
    assert_eq!(minimum_f32(-0.0, 0.0).to_bits(), (-0.0f32).to_bits());
    assert_eq!(maximum_f32(-0.0, 0.0).to_bits(), 0.0f32.to_bits());
    assert!(minimum_f64(f64::NAN, 1.0).is_nan());
    assert!(maximum_f64(1.0, f64::NAN).is_nan());
    assert_eq!(minimum_f64(-0.0, 0.0).to_bits(), (-0.0f64).to_bits());
    assert_eq!(maximum_f64(-0.0, 0.0).to_bits(), 0.0f64.to_bits());

    assert_eq!(round_f32(2.5), 2.0);
    assert_eq!(round_f32(3.5), 4.0);
    assert_eq!(floor_f32(-1.25), -2.0);
    assert_eq!(ceil_f32(-1.25), -1.0);
    assert_eq!(sqrt_f32(25.0), 5.0);
    assert_eq!(round_f64(2.5), 2.0);
    assert_eq!(round_f64(3.5), 4.0);
    assert_eq!(floor_f64(-1.25), -2.0);
    assert_eq!(ceil_f64(-1.25), -1.0);
    assert_eq!(sqrt_f64(25.0), 5.0);

    let a32 = 1.0_f32 + f32::EPSILON;
    let b32 = 1.0_f32 - f32::EPSILON;
    let exact32 = ((a32 as f64) * (b32 as f64) - 1.0_f64) as f32;
    assert_eq!(
        mul_add_fused_f32(a32, b32, -1.0).to_bits(),
        exact32.to_bits()
    );
    assert_eq!(
        mul_sub_fused_f32(a32, b32, 1.0).to_bits(),
        exact32.to_bits()
    );

    let a64 = 1.0_f64 + f64::EPSILON;
    let b64 = 1.0_f64 - f64::EPSILON;
    let exact64 = a64.mul_add(b64, -1.0);
    assert_eq!(
        mul_add_fused_f64(a64, b64, -1.0).to_bits(),
        exact64.to_bits()
    );
    assert_eq!(
        mul_sub_fused_f64(a64, b64, 1.0).to_bits(),
        exact64.to_bits()
    );

    fn check_token_helpers<T: F32ScalarBackend + F64ScalarBackend>(t: T) {
        assert_eq!(min_fast_f32(t, 3.0, -2.0), -2.0);
        assert_eq!(max_fast_f32(t, 3.0, -2.0), 3.0);
        assert_eq!(clamp_f32(t, 10.0, -1.0, 4.0), 4.0);
        assert_eq!(clamp_f32(t, -5.0, -1.0, 4.0), -1.0);
        assert_eq!(mul_add_relaxed_f32(t, 3.0, 4.0, 5.0), 17.0);
        assert_eq!(mul_sub_relaxed_f32(t, 3.0, 4.0, 5.0), 7.0);
        assert_eq!(to_i32(t, -7.9), -7);
        assert_eq!(to_i32_fast(t, 12.7), 12);

        assert_eq!(min_fast_f64(t, 3.0, -2.0), -2.0);
        assert_eq!(max_fast_f64(t, 3.0, -2.0), 3.0);
        assert_eq!(clamp_f64(t, 10.0, -1.0, 4.0), 4.0);
        assert_eq!(clamp_f64(t, -5.0, -1.0, 4.0), -1.0);
        assert_eq!(mul_add_relaxed_f64(t, 3.0, 4.0, 5.0), 17.0);
        assert_eq!(mul_sub_relaxed_f64(t, 3.0, 4.0, 5.0), 7.0);
    }

    check_token_helpers(ScalarToken);
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    {
        if let Some(t) = archmage::X64V3Token::summon() {
            check_token_helpers(t);
        }
        if let Some(t) = archmage::X64V3VnniToken::summon() {
            check_token_helpers(t);
        }
        #[cfg(feature = "avx512")]
        {
            if let Some(t) = archmage::X64V4Token::summon() {
                check_token_helpers(t);
            }
            if let Some(t) = archmage::X64V4xToken::summon() {
                check_token_helpers(t);
            }
            if let Some(t) = archmage::Avx512Fp16Token::summon() {
                check_token_helpers(t);
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = archmage::NeonToken::summon() {
        check_token_helpers(t);
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = archmage::Wasm128Token::summon() {
        check_token_helpers(t);
    }
}

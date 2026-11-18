//! Fluent `const` options (`PowfOptions`, `ExpfOptions`, `LogfOptions`, `CbrtfOptions`),
//! `SimdMath` inline vector chaining (`v.powf::<CFG>`, `v.try_powf`, `v.token()`),
//! custom-range tail behavior, and `Execution::Upgrade` (`V3 -> V4/V4x`) vs `Execution::Caller`.
use archmage::ScalarToken;
use magemath::{
    Portability, SimdMath, SliceError,
    cbrtf::{self as c, Cbrtf, CbrtfOptions},
    expf::{self as e, Expf, ExpfOptions},
    logf::{self as l, Logf, LogfOptions},
    powf::{self as p, Powf, PowfOptions},
};
use magetypes::simd::generic::{f32x4, f32x8};

fn log_uniform(len: usize, lo: f32, hi: f32, seed: u32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (lo.ln() + (hi.ln() - lo.ln()) * (s >> 8) as f32 / 16_777_216.0).exp()
        })
        .collect()
}

fn uniform(len: usize, lo: f32, hi: f32, seed: u32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            lo + (hi - lo) * (s >> 8) as f32 / 16_777_216.0
        })
        .collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn powf_const_options_match_runtime_across_domains_including_highp_and_full() {
    fn same<const CFG: u128>(opts: PowfOptions, lo: f32, hi: f32, ns: &[f32], zero: bool) {
        assert_eq!(PowfOptions::from_bits(CFG), Some(opts));
        let rt = opts.runtime();
        for (k, &n) in ns.iter().enumerate() {
            let mut src = log_uniform(99, lo, hi, 77 + k as u32);
            if zero {
                src[5] = 0.0;
                src[98] = 0.0;
            }
            let (mut a, mut b) = (vec![0.0; 99], vec![0.0; 99]);
            Powf::<CFG>::slice(&src, &mut a, n).unwrap();
            rt.slice(&src, &mut b, n).unwrap();
            assert_eq!(bits(&a), bits(&b), "n={n}");
        }
    }

    const GM_O: PowfOptions = PowfOptions::new().gamma().midp();
    const GL_O: PowfOptions = PowfOptions::new().gamma().lowp();
    const PGM_O: PowfOptions = PowfOptions::new().positive_gamma().midp();
    const NM_O: PowfOptions = PowfOptions::new().midp();
    const GH_O: PowfOptions = PowfOptions::new().gamma().highp();
    const FH_O: PowfOptions = PowfOptions::new().full_domain().highp();
    const NN_O: PowfOptions = PowfOptions::new().admit_zero();

    same::<{ GM_O.bits() }>(
        GM_O,
        1.6e-5,
        65_000.0,
        &[0.0, 0.4545, 1.0, 2.2, 2.4, 4.0],
        true,
    );
    same::<{ GL_O.bits() }>(GL_O, 1.6e-5, 65_000.0, &[0.4545, 2.4, 4.0], true);
    same::<{ PGM_O.bits() }>(PGM_O, 1.6e-5, 65_000.0, &[0.4545, 2.4], false);
    same::<{ NM_O.bits() }>(NM_O, 1e-6, 1e6, &[-3.0, -0.5, 0.0, 1.5, 5.0], false);
    same::<{ GH_O.bits() }>(GH_O, 1.6e-5, 65_000.0, &[0.0, 0.5, 2.4], true);
    same::<{ FH_O.bits() }>(FH_O, 1e-4, 1e4, &[-2.5, 0.0, 2.4], true);
    same::<{ NN_O.bits() }>(NN_O, 1e-6, 1e6, &[0.0, 1.7], true);
}

const TONE_SPEC: PowfOptions = PowfOptions::new()
    .base_range(1.0 / 1024.0, 16.0)
    .exponent_range(-8.0, 8.0);
const TONE: u128 = TONE_SPEC.bits();
const TONE_LOWP: u128 = TONE_SPEC.lowp().bits();

#[test]
fn custom_powf_range_elides_check_and_stays_accurate() {
    const { assert!(TONE_SPEC.elides_range_check()) };
    let mut worst = (0f64, 0f64);
    for (k, n) in [-8.0f32, -7.99, -1.0, -0.001, 0.0, 0.3, 1.0, 2.2, 7.5, 8.0]
        .into_iter()
        .enumerate()
    {
        let mut src = log_uniform(512, 1.0 / 1024.0, 16.0, 11 + k as u32);
        src[0] = 1.0 / 1024.0;
        src[1] = 16.0;
        let (mut m, mut l) = (vec![0.0; 512], vec![0.0; 512]);
        Powf::<TONE>::slice(&src, &mut m, n).unwrap();
        Powf::<TONE_LOWP>::slice(&src, &mut l, n).unwrap();
        for i in 0..512 {
            let t = (src[i] as f64).powf(n as f64);
            assert!(m[i].is_finite() && l[i].is_finite(), "x={} n={n}", src[i]);
            worst.0 = worst.0.max(((m[i] as f64 - t) / t).abs());
            worst.1 = worst.1.max(((l[i] as f64 - t) / t).abs());
        }
    }
    assert!(worst.0 < 2e-5, "midp relative error {}", worst.0);
    assert!(worst.1 < 7e-3, "lowp relative error {}", worst.1);
}

#[test]
fn custom_powf_range_rejects_what_it_excludes() {
    let mut dst = [0.0; 12];
    let mut src = [1.0f32; 12];
    src[10] = 17.0;
    assert_eq!(
        Powf::<TONE>::slice(&src, &mut dst, 2.0),
        Err(SliceError {
            kind: p::PowfErrorKind::InvalidBase,
            index: 10
        })
    );
    src[10] = 0.0;
    assert_eq!(
        Powf::<TONE>::slice(&src, &mut dst, 2.0).unwrap_err().index,
        10
    );
    assert_eq!(
        Powf::<TONE>::slice(&[1.0; 12], &mut dst, 8.5),
        Err(SliceError {
            kind: p::PowfErrorKind::InvalidExponent,
            index: 0
        })
    );
}

/// Wide range keeping the range check while still enforcing `base_range` and `admit_zero`
/// in a single SIMD pass (no pre-scan).
const WIDE_SPEC: PowfOptions = PowfOptions::new()
    .base_range(1e-10, 1e10)
    .exponent_range(-10.0, 10.0)
    .admit_zero();
const WIDE: u128 = WIDE_SPEC.bits();

#[test]
fn wide_powf_range_single_pass_validation() {
    const { assert!(!WIDE_SPEC.elides_range_check()) };
    let mut dst = [0.0; 9];
    let mut src = [2.0f32; 9];
    src[3] = 0.0;
    Powf::<WIDE>::slice(&src, &mut dst, 3.0).unwrap();
    assert_eq!(dst[3], 0.0);
    assert_eq!(dst[0], 8.0);
    src[8] = 1e10;
    assert_eq!(
        Powf::<WIDE>::slice(&src, &mut dst, 10.0),
        Err(SliceError {
            kind: p::PowfErrorKind::OutsideApproximationRange,
            index: 8
        })
    );
    src[8] = 2.0;
    src[6] = 1e11;
    src[7] = -1.0;
    assert_eq!(
        Powf::<WIDE>::slice(&src, &mut dst, 1.0),
        Err(SliceError {
            kind: p::PowfErrorKind::InvalidBase,
            index: 6
        })
    );
    src[6] = 2.0;
    src[7] = 2.0;
    assert_eq!(
        Powf::<WIDE>::slice(&src, &mut dst, -1.0),
        Err(SliceError {
            kind: p::PowfErrorKind::InvalidBase,
            index: 3
        })
    );
}

#[test]
fn exponent_range_25bit_outward_rounding_and_bijection() {
    for (lo, hi) in [
        (0.0f32, 4.0f32),
        (-8.0, 8.0),
        (1.0 / 2.4, 2.4),
        (-0.333_333_34, 2.754_321_9),
    ] {
        let opts = PowfOptions::new().exponent_range(lo, hi);
        let decoded = PowfOptions::from_bits(opts.bits()).expect("valid u128 encoding");
        assert_eq!(decoded, opts);
        assert!(opts.runtime().scalar(2.0, lo).is_ok());
        assert!(opts.runtime().scalar(2.0, hi).is_ok());
    }
}

/// Ranges that deliberately exclude `1.0` verify `tail[0]` padding on tails < 8 and < 16.
const POW_EXCLUDING_ONE: u128 = PowfOptions::new()
    .base_range(2.0, 8.0)
    .exponent_range(0.5, 2.0)
    .bits();
const EXP_CLAMPED_SPEC: ExpfOptions = ExpfOptions::new().input_range(2.0, 8.0);
const EXP_CLAMPED: u128 = EXP_CLAMPED_SPEC.bits();
const EXP_TIGHT_SPEC: ExpfOptions = ExpfOptions::new().input_range(-10.0, 10.0);
const LOG_EXCLUDING_ONE: u128 = LogfOptions::new().input_range(2.0, 8.0).bits();
const LOG1P_SMALL: u128 = LogfOptions::new().input_range(-0.5, 0.5).bits();
const CBRT_EXCLUDING_ONE_SPEC: CbrtfOptions = CbrtfOptions::new().input_range(2.0, 8.0);
const CBRT_EXCLUDING_ONE: u128 = CBRT_EXCLUDING_ONE_SPEC.bits();
const CBRT_NEAR_ZERO_SPEC: CbrtfOptions = CbrtfOptions::new().input_range(-8.0, 8.0);
const CBRT_NEAR_ZERO: u128 = CBRT_NEAR_ZERO_SPEC.bits();

#[test]
fn custom_range_tails_excluding_one_accept_and_reject_exact_indices() {
    const { assert!(EXP_CLAMPED_SPEC.elides_clamp()) };
    const { assert!(EXP_TIGHT_SPEC.elides_exp_clamp()) };
    const { assert!(CBRT_EXCLUDING_ONE_SPEC.elides_denorm_check()) };
    const { assert!(!CBRT_NEAR_ZERO_SPEC.elides_denorm_check()) };

    for len in 1..=19 {
        let pow_src = uniform(len, 2.0, 8.0, 100 + len as u32);
        let mut pow_dst = vec![0.0; len];
        Powf::<POW_EXCLUDING_ONE>::slice(&pow_src, &mut pow_dst, 1.5).unwrap();
        for (i, &x) in pow_src.iter().enumerate() {
            assert_eq!(
                pow_dst[i].to_bits(),
                Powf::<POW_EXCLUDING_ONE>::scalar(x, 1.5).unwrap().to_bits()
            );
        }

        let mut exp_dst = vec![0.0; len];
        Expf::<EXP_CLAMPED>::exp2_slice(&pow_src, &mut exp_dst).unwrap();
        for (i, &x) in pow_src.iter().enumerate() {
            assert_eq!(
                exp_dst[i].to_bits(),
                Expf::<EXP_CLAMPED>::exp2(x).unwrap().to_bits()
            );
        }

        let mut log_dst = vec![0.0; len];
        Logf::<LOG_EXCLUDING_ONE>::log2_slice(&pow_src, &mut log_dst).unwrap();
        for (i, &x) in pow_src.iter().enumerate() {
            assert_eq!(
                log_dst[i].to_bits(),
                Logf::<LOG_EXCLUDING_ONE>::log2(x).unwrap().to_bits()
            );
        }

        let log1p_src = uniform(len, -0.5, 0.5, 200 + len as u32);
        let mut log1p_dst = vec![0.0; len];
        Logf::<LOG1P_SMALL>::log1p_slice(&log1p_src, &mut log1p_dst).unwrap();
        for (i, &x) in log1p_src.iter().enumerate() {
            assert_eq!(
                log1p_dst[i].to_bits(),
                Logf::<LOG1P_SMALL>::log1p(x).unwrap().to_bits()
            );
        }

        let mut cbrt_dst = vec![0.0; len];
        Cbrtf::<CBRT_EXCLUDING_ONE>::slice(&pow_src, &mut cbrt_dst).unwrap();
        for (i, &x) in pow_src.iter().enumerate() {
            assert_eq!(
                cbrt_dst[i].to_bits(),
                Cbrtf::<CBRT_EXCLUDING_ONE>::scalar(x).unwrap().to_bits()
            );
        }
    }

    // Verify rejection on a tail (< 8 and < 16) when `1.0` (outside [2.0, 8.0]) is placed in the slice.
    for len in [5usize, 13, 19] {
        for bad in 0..len {
            let mut src = vec![3.0f32; len];
            src[bad] = 1.0;
            let mut dst = vec![0.0; len];
            assert_eq!(
                Powf::<POW_EXCLUDING_ONE>::slice(&src, &mut dst, 1.5).unwrap_err(),
                SliceError {
                    kind: p::PowfErrorKind::InvalidBase,
                    index: bad
                }
            );
            assert_eq!(
                Expf::<EXP_CLAMPED>::exp2_slice(&src, &mut dst).unwrap_err(),
                SliceError {
                    kind: e::ExpfErrorKind::InvalidInput,
                    index: bad
                }
            );
            assert_eq!(
                Logf::<LOG_EXCLUDING_ONE>::ln_slice(&src, &mut dst).unwrap_err(),
                SliceError {
                    kind: l::LogfErrorKind::InvalidInput,
                    index: bad
                }
            );
            assert_eq!(
                Cbrtf::<CBRT_EXCLUDING_ONE>::slice(&src, &mut dst).unwrap_err(),
                SliceError {
                    kind: c::CbrtfErrorKind::InvalidInput,
                    index: bad
                }
            );
        }
    }

    // Bounded cbrtf crossing zero still rejects subnormals while accepting ±0.0.
    let cbrt_sub = [0.0, -0.0, f32::from_bits(1), 2.0, 4.0];
    let mut out = [0.0; 5];
    assert_eq!(
        Cbrtf::<CBRT_NEAR_ZERO>::slice(&cbrt_sub, &mut out).unwrap_err(),
        SliceError {
            kind: c::CbrtfErrorKind::InvalidInput,
            index: 2
        }
    );
}

#[test]
fn clamp_output_saturates_extreme_exponents_instead_of_rejecting() {
    const CLAMPED: u128 = PowfOptions::new().clamp_output().bits();
    assert_eq!(Powf::<CLAMPED>::scalar(1.0e20, 10.0), Ok(f32::INFINITY));
    assert_eq!(Powf::<CLAMPED>::scalar(1.0e-20, 10.0), Ok(0.0));
}

#[test]
fn simd_math_inline_chaining_and_token_extraction_across_all_families() {
    const P_CALLER: u128 = PowfOptions::new().gamma().caller().bits();
    const P_FAST: u128 = PowfOptions::new().gamma().unchecked().bits();
    const E_FAST: u128 = ExpfOptions::new().unchecked().bits();
    const L_FAST: u128 = LogfOptions::new().unchecked().bits();
    const C_FAST: u128 = CbrtfOptions::new().unchecked().bits();
    const EB: u128 = ExpfOptions::new().backend_dependent().caller().bits();
    const LB: u128 = LogfOptions::new().backend_dependent().caller().bits();
    const CB: u128 = CbrtfOptions::new().backend_dependent().caller().bits();

    const { assert!(Portability::BitExact as u8 != Portability::BackendDependent as u8) };

    let v4 = f32x4::from_array_t(ScalarToken, [0.25, 0.5, 1.0, 4.0]);
    let _: ScalarToken = v4.token();

    assert_eq!(
        v4.try_powf(2.0).unwrap().to_array(),
        v4.powf::<P_FAST>(2.0).to_array()
    );
    assert_eq!(
        v4.try_exp2().unwrap().to_array(),
        v4.exp2::<E_FAST>().to_array()
    );
    assert_eq!(
        v4.try_exp().unwrap().to_array(),
        v4.exp::<E_FAST>().to_array()
    );
    assert_eq!(
        v4.try_log2().unwrap().to_array(),
        v4.log2::<L_FAST>().to_array()
    );
    assert_eq!(
        v4.try_ln().unwrap().to_array(),
        v4.ln::<L_FAST>().to_array()
    );
    assert_eq!(
        v4.try_log10().unwrap().to_array(),
        v4.log10::<L_FAST>().to_array()
    );
    assert_eq!(
        v4.try_log1p().unwrap().to_array(),
        v4.log1p::<L_FAST>().to_array()
    );
    assert_eq!(
        v4.try_cbrt().unwrap().to_array(),
        v4.cbrt::<C_FAST>().to_array()
    );

    let s = Powf::<P_CALLER>::scalar(0.25, 2.0).unwrap();
    let st = Powf::<P_CALLER>::scalar_with_token(ScalarToken, 0.25, 2.0).unwrap();
    let a = Powf::<P_CALLER>::array::<4>([0.25; 4], 2.0).unwrap();
    let v = f32x8::splat_t(ScalarToken, 0.25)
        .try_powf_with::<P_CALLER>(2.0)
        .unwrap()
        .to_array();
    let mut sl = [0.0; 11];
    Powf::<P_CALLER>::slice_with_token(ScalarToken, &[0.25; 11], &mut sl, 2.0).unwrap();
    assert_eq!(s.to_bits(), st.to_bits());
    assert_eq!(a[0].to_bits(), s.to_bits());
    assert_eq!(v[0].to_bits(), s.to_bits());
    assert_eq!(sl[10].to_bits(), s.to_bits());

    // Unchecked accepts without returning InvalidBase error.
    assert!(Powf::<P_FAST>::scalar(-1.0, 2.0).is_ok());
    let _ = Powf::<P_FAST>::scalar_infallible(0.5, 2.2);

    let es = Expf::<EB>::exp_with_token(ScalarToken, 0.5).unwrap();
    let mut edst = [0.0; 9];
    Expf::<EB>::exp_slice_with_token(ScalarToken, &[0.5; 9], &mut edst).unwrap();
    assert_eq!(edst[8].to_bits(), es.to_bits());

    let ls = Logf::<LB>::log1p_with_token(ScalarToken, 0.5).unwrap();
    let mut ldst = [0.0; 9];
    Logf::<LB>::log1p_slice_with_token(ScalarToken, &[0.5; 9], &mut ldst).unwrap();
    assert_eq!(ldst[8].to_bits(), ls.to_bits());

    let cs = Cbrtf::<CB>::scalar_with_token(ScalarToken, 2.5).unwrap();
    let mut cdst = [0.0; 9];
    Cbrtf::<CB>::slice_with_token(ScalarToken, &[2.5; 9], &mut cdst).unwrap();
    assert_eq!(cdst[8].to_bits(), cs.to_bits());
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[test]
fn v3_token_upgrades_to_avx512_for_slices_w512_and_highp_while_caller_stays_v3() {
    use archmage::SimdToken;
    use magetypes::simd::generic::f32x16;

    const UP_GAMMA: u128 = PowfOptions::new().gamma().upgrade().bits();
    const CALL_GAMMA: u128 = PowfOptions::new().gamma().caller().bits();
    const UP_HIGHP: u128 = PowfOptions::new().gamma().highp().upgrade().bits();
    const CALL_HIGHP: u128 = PowfOptions::new().gamma().highp().caller().bits();
    const UP_EXP: u128 = ExpfOptions::new().upgrade().bits();
    const CALL_EXP: u128 = ExpfOptions::new().caller().bits();
    const UP_LOG: u128 = LogfOptions::new().upgrade().bits();
    const CALL_LOG: u128 = LogfOptions::new().caller().bits();
    const UP_CBRT: u128 = CbrtfOptions::new().upgrade().bits();
    const CALL_CBRT: u128 = CbrtfOptions::new().caller().bits();

    let src = log_uniform(35, 0.01, 100.0, 0x1234);

    let check_for_v3 = |v3: archmage::X64V3Token| {
        let (mut p_up, mut p_ca) = (vec![0.0; 35], vec![0.0; 35]);
        Powf::<UP_GAMMA>::slice_with_token(v3, &src, &mut p_up, 2.2).unwrap();
        Powf::<CALL_GAMMA>::slice_with_token(v3, &src, &mut p_ca, 2.2).unwrap();
        assert_eq!(bits(&p_up), bits(&p_ca));

        let (mut h_up, mut h_ca) = (vec![0.0; 35], vec![0.0; 35]);
        Powf::<UP_HIGHP>::slice_with_token(v3, &src, &mut h_up, 2.2).unwrap();
        Powf::<CALL_HIGHP>::slice_with_token(v3, &src, &mut h_ca, 2.2).unwrap();
        assert_eq!(bits(&h_up), bits(&h_ca));

        let (mut e_up, mut e_ca) = (vec![0.0; 35], vec![0.0; 35]);
        Expf::<UP_EXP>::exp2_slice_with_token(v3, &src, &mut e_up).unwrap();
        Expf::<CALL_EXP>::exp2_slice_with_token(v3, &src, &mut e_ca).unwrap();
        assert_eq!(bits(&e_up), bits(&e_ca));

        let (mut l_up, mut l_ca) = (vec![0.0; 35], vec![0.0; 35]);
        Logf::<UP_LOG>::ln_slice_with_token(v3, &src, &mut l_up).unwrap();
        Logf::<CALL_LOG>::ln_slice_with_token(v3, &src, &mut l_ca).unwrap();
        assert_eq!(bits(&l_up), bits(&l_ca));

        let (mut c_up, mut c_ca) = (vec![0.0; 35], vec![0.0; 35]);
        Cbrtf::<UP_CBRT>::slice_with_token(v3, &src, &mut c_up).unwrap();
        Cbrtf::<CALL_CBRT>::slice_with_token(v3, &src, &mut c_ca).unwrap();
        assert_eq!(bits(&c_up), bits(&c_ca));

        let arr16: [f32; 16] = core::array::from_fn(|i| src[i]);
        let v16 = f32x16::from_array_t(v3, arr16);
        assert_eq!(
            v16.try_powf_with::<UP_GAMMA>(2.2)
                .unwrap()
                .to_array()
                .map(f32::to_bits),
            v16.try_powf_with::<CALL_GAMMA>(2.2)
                .unwrap()
                .to_array()
                .map(f32::to_bits),
        );
        assert_eq!(
            v16.try_exp2_with::<UP_EXP>()
                .unwrap()
                .to_array()
                .map(f32::to_bits),
            v16.try_exp2_with::<CALL_EXP>()
                .unwrap()
                .to_array()
                .map(f32::to_bits),
        );
        assert_eq!(
            v16.try_ln_with::<UP_LOG>()
                .unwrap()
                .to_array()
                .map(f32::to_bits),
            v16.try_ln_with::<CALL_LOG>()
                .unwrap()
                .to_array()
                .map(f32::to_bits),
        );
        assert_eq!(
            v16.try_cbrt_with::<UP_CBRT>()
                .unwrap()
                .to_array()
                .map(f32::to_bits),
            v16.try_cbrt_with::<CALL_CBRT>()
                .unwrap()
                .to_array()
                .map(f32::to_bits),
        );

        let arr8: [f32; 8] = core::array::from_fn(|i| src[i]);
        let v8 = f32x8::from_array_t(v3, arr8);
        assert_eq!(
            v8.try_powf_with::<UP_HIGHP>(2.2)
                .unwrap()
                .to_array()
                .map(f32::to_bits),
            v8.try_powf_with::<CALL_HIGHP>(2.2)
                .unwrap()
                .to_array()
                .map(f32::to_bits),
        );
    };

    #[cfg(feature = "std")]
    {
        // Sweep across V4x, V4-only (V4x disabled), and V3-fallback (V4/V4x disabled) permutations.
        let _ = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |_| {
                if let Some(v3) = archmage::X64V3Token::summon() {
                    check_for_v3(v3);
                }
            },
        );
    }
    #[cfg(not(feature = "std"))]
    if let Some(v3) = archmage::X64V3Token::summon() {
        check_for_v3(v3);
    }
}

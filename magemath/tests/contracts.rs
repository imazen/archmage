//! Boundary, provider and scalar/vector-tail contracts for the migrated APIs.
#![forbid(unsafe_code)]
use magemath::{
    Execution, Portability, SimdMath, Validation,
    expf::{self as e, Expf, ExpfOptions, ExpfPrecision},
    log1pf::{Log1pf, Log1pfOptions, Log1pfPrecision},
    logf::{Logf, LogfOptions, LogfPrecision},
    powf::{self as p, Powf, PowfOptions, PowfPrecision},
};
#[path = "data/powf.rs"]
mod fixtures;

#[test]
fn canonical_power_matches_independent_fixtures() {
    const D0: u128 = PowfOptions::new().reject_subnormals().caller().bits();
    const D1: u128 = PowfOptions::new()
        .admit_zero()
        .reject_subnormals()
        .caller()
        .bits();
    const D2: u128 = PowfOptions::new()
        .gamma()
        .reject_subnormals()
        .caller()
        .bits();

    for (index, &[base, n, expected, stage, domain]) in fixtures::POWF.iter().enumerate() {
        // Same directed/default selection as Magetypes; full replay is opt-in.
        if !cfg!(feature = "_exhaustive-tests")
            && n != 0x4019999a
            && n != 0x3f000000
            && base != 0x3f800000
            && index % 97 != 0
        {
            continue;
        }
        let base = f32::from_bits(base);
        let n = f32::from_bits(n);
        let result = match domain {
            0 => Powf::<D0>::scalar(base, n),
            1 => Powf::<D1>::scalar(base, n),
            2 => Powf::<D2>::scalar(base, n),
            _ => panic!("fixture domain"),
        };
        if stage == 0 {
            assert_eq!(result.unwrap().to_bits(), expected, "fixture {index}");
        } else {
            let kind = match stage {
                1 => p::PowfErrorKind::InvalidExponent,
                2 => p::PowfErrorKind::InvalidBase,
                3 => p::PowfErrorKind::OutsideApproximationRange,
                _ => panic!("fixture stage"),
            };
            let error = result.unwrap_err();
            assert_eq!(error.kind, kind);
            assert_eq!(error.lane_mask, 1);
        }
    }
}

#[test]
fn every_exp_log_const_configuration_has_a_runtime_counterpart() {
    const fn exp_from_legacy(f: u32) -> ExpfOptions {
        let b = if f & 1 == 0 {
            ExpfOptions::new()
        } else {
            ExpfOptions::new().full_domain()
        };
        b.execution(if f & 4 == 0 {
            Execution::Upgrade
        } else {
            Execution::Caller
        })
        .precision(if f & 8 == 0 {
            ExpfPrecision::Midp
        } else {
            ExpfPrecision::Lowp
        })
        .portability(if f & 16 == 0 {
            Portability::BitExact
        } else {
            Portability::BackendDependent
        })
        .validation(if f & 32 == 0 {
            Validation::Checked
        } else {
            Validation::AssumeValid
        })
    }

    const fn log_from_legacy(f: u32) -> LogfOptions {
        let b = if f & 1 == 0 {
            LogfOptions::new()
        } else {
            LogfOptions::new().full_domain()
        };
        b.execution(if f & 4 == 0 {
            Execution::Upgrade
        } else {
            Execution::Caller
        })
        .precision(if f & 8 == 0 {
            LogfPrecision::Midp
        } else {
            LogfPrecision::Lowp
        })
        .portability(if f & 16 == 0 {
            Portability::BitExact
        } else {
            Portability::BackendDependent
        })
        .validation(if f & 32 == 0 {
            Validation::Checked
        } else {
            Validation::AssumeValid
        })
    }

    fn compare<const F: u32, const ECFG: u128, const LCFG: u128>() {
        let eo = exp_from_legacy(F);
        let lo = log_from_legacy(F);
        assert_eq!(ExpfOptions::from_bits(ECFG), Some(eo));
        assert_eq!(LogfOptions::from_bits(LCFG), Some(lo));
        for x in [0.5, 0.99, 1., 1.5, 127.25] {
            assert_eq!(
                eo.runtime().exp2(x).unwrap().to_bits(),
                Expf::<ECFG>::exp2(x).unwrap().to_bits()
            );
            assert_eq!(
                lo.runtime().log2(x).unwrap().to_bits(),
                Logf::<LCFG>::log2(x).unwrap().to_bits()
            );
        }
        if F & 1 != 0 {
            assert!(Expf::<ECFG>::exp2(f32::NAN).unwrap().is_nan());
            assert_eq!(Expf::<ECFG>::exp2(f32::NEG_INFINITY), Ok(0.));
            assert_eq!(Logf::<LCFG>::log2(-0.), Ok(f32::NEG_INFINITY));
            assert!(Logf::<LCFG>::log2(f32::NAN).unwrap().is_nan());
        }
    }
    macro_rules! run {
        ($($f:literal),*) => {$(
            compare::<$f, { exp_from_legacy($f).bits() }, { log_from_legacy($f).bits() }>();
        )*};
    }
    run!(
        0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29, 32, 33, 36, 37, 40, 41, 44, 45,
        48, 49, 52, 53, 56, 57, 60, 61
    );
}

#[test]
fn signed_zero_and_whole_vector_rejection() {
    const LOG1P_CALLER: u128 = Log1pfOptions::new().caller().bits();
    const LOG1P_DEFAULT: u128 = Log1pfOptions::new().bits();
    const LOG_DEFAULT: u128 = LogfOptions::new().bits();
    const EXP_DEFAULT: u128 = ExpfOptions::new().bits();
    for bits in [0, 0x80000000] {
        let x = f32::from_bits(bits);
        assert_eq!(Log1pf::<LOG1P_CALLER>::scalar(x).unwrap().to_bits(), bits);
        assert_eq!(
            Log1pf::<LOG1P_CALLER>::scalar_with_token(archmage::ScalarToken, x)
                .unwrap()
                .to_bits(),
            bits
        );
    }
    assert_eq!(
        Log1pf::<LOG1P_DEFAULT>::scalar_with_token(archmage::ScalarToken, -2.)
            .unwrap_err()
            .lane_mask,
        1
    );
    let error = Logf::<LOG_DEFAULT>::log2_array::<4>([1., 0., f32::NAN, -1.]).unwrap_err();
    assert_eq!(error.lane_mask, 0b1110);
    assert_eq!(
        Expf::<EXP_DEFAULT>::exp2_array::<3>([0.; 3])
            .unwrap_err()
            .kind,
        e::ExpfErrorKind::UnsupportedWidth
    );
}

#[archmage::magetypes(v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
fn check_tails(token: Token) {
    macro_rules! check {
        ($ecfg:expr, $pcfg:expr, $lcfg:expr, $l1pcfg:expr, $n:literal, $vec:ident) => {{
            const ECFG: u128 = $ecfg;
            const PCFG: u128 = $pcfg;
            const LCFG: u128 = $lcfg;
            const L1PCFG: u128 = $l1pcfg;
            use magetypes::simd::generic::$vec as Vec;
            let xs: [f32; $n] = core::array::from_fn(|i| 0.5 + (i % 8) as f32 / 16.);
            let input = Vec::from_array_t(token, xs);
            let got = input.try_exp2_with::<ECFG>().unwrap().to_array();
            let power = input.try_powf_with::<PCFG>(2.4).unwrap().to_array();
            let log = input.try_log2_with::<LCFG>().unwrap().to_array();
            let ln = input.try_ln_with::<LCFG>().unwrap().to_array();
            let log10 = input.try_log10_with::<LCFG>().unwrap().to_array();
            let log1p = input.try_log1p_with::<L1PCFG>().unwrap().to_array();
            for i in 0..$n {
                assert_eq!(
                    got[i].to_bits(),
                    Expf::<ECFG>::exp2_with_token(token, xs[i])
                        .unwrap()
                        .to_bits()
                );
                assert_eq!(
                    power[i].to_bits(),
                    Powf::<PCFG>::scalar_with_token(token, xs[i], 2.4)
                        .unwrap()
                        .to_bits()
                );
                assert_eq!(
                    log[i].to_bits(),
                    Logf::<LCFG>::log2_with_token(token, xs[i])
                        .unwrap()
                        .to_bits()
                );
                assert_eq!(
                    ln[i].to_bits(),
                    Logf::<LCFG>::ln_with_token(token, xs[i]).unwrap().to_bits()
                );
                assert_eq!(
                    log10[i].to_bits(),
                    Logf::<LCFG>::log10_with_token(token, xs[i])
                        .unwrap()
                        .to_bits()
                );
                assert_eq!(
                    log1p[i].to_bits(),
                    Log1pf::<L1PCFG>::scalar_with_token(token, xs[i])
                        .unwrap()
                        .to_bits()
                );
            }
        }};
    }
    macro_rules! profile {
        ($prec_e:expr, $prec_p:expr, $prec_l:expr, $prec_l1p:expr, $port:expr) => {
            check!(
                ExpfOptions::new()
                    .precision($prec_e)
                    .portability($port)
                    .caller()
                    .bits(),
                PowfOptions::new()
                    .precision($prec_p)
                    .portability($port)
                    .caller()
                    .bits(),
                LogfOptions::new()
                    .precision($prec_l)
                    .portability($port)
                    .caller()
                    .bits(),
                Log1pfOptions::new()
                    .precision($prec_l1p)
                    .portability($port)
                    .caller()
                    .bits(),
                4,
                f32x4
            );
            check!(
                ExpfOptions::new()
                    .precision($prec_e)
                    .portability($port)
                    .caller()
                    .bits(),
                PowfOptions::new()
                    .precision($prec_p)
                    .portability($port)
                    .caller()
                    .bits(),
                LogfOptions::new()
                    .precision($prec_l)
                    .portability($port)
                    .caller()
                    .bits(),
                Log1pfOptions::new()
                    .precision($prec_l1p)
                    .portability($port)
                    .caller()
                    .bits(),
                8,
                f32x8
            );
            #[cfg(feature = "w512")]
            check!(
                ExpfOptions::new()
                    .precision($prec_e)
                    .portability($port)
                    .caller()
                    .bits(),
                PowfOptions::new()
                    .precision($prec_p)
                    .portability($port)
                    .caller()
                    .bits(),
                LogfOptions::new()
                    .precision($prec_l)
                    .portability($port)
                    .caller()
                    .bits(),
                Log1pfOptions::new()
                    .precision($prec_l1p)
                    .portability($port)
                    .caller()
                    .bits(),
                16,
                f32x16
            );
        };
    }
    profile!(
        ExpfPrecision::Midp,
        PowfPrecision::Midp,
        LogfPrecision::Midp,
        Log1pfPrecision::Midp,
        Portability::BitExact
    );
    profile!(
        ExpfPrecision::Lowp,
        PowfPrecision::Lowp,
        LogfPrecision::Lowp,
        Log1pfPrecision::Lowp,
        Portability::BitExact
    );
    profile!(
        ExpfPrecision::Midp,
        PowfPrecision::Midp,
        LogfPrecision::Midp,
        Log1pfPrecision::Midp,
        Portability::BackendDependent
    );
    profile!(
        ExpfPrecision::Lowp,
        PowfPrecision::Lowp,
        LogfPrecision::Lowp,
        Log1pfPrecision::Lowp,
        Portability::BackendDependent
    );
}

#[test]
fn vectors_and_tails_reuse_the_same_caller_graph() {
    #[cfg(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32"
    ))]
    use archmage::SimdToken;
    check_tails_scalar(archmage::ScalarToken);
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = archmage::X64V3Token::summon() {
        check_tails_v3(t);
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    {
        if let Some(t) = archmage::X64V4Token::summon() {
            check_tails_v4(t);
        }
        if let Some(t) = archmage::X64V4xToken::summon() {
            check_tails_v4x(t);
        }
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = archmage::NeonToken::summon() {
        check_tails_neon(t);
    }
    #[cfg(target_arch = "wasm32")]
    if let Some(t) = archmage::Wasm128Token::summon() {
        check_tails_wasm128(t);
    }
}

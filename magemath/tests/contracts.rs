//! Boundary, provider and scalar/vector-tail contracts for the migrated APIs.
#![forbid(unsafe_code)]
use magemath::{expf as e, logf as l, powf as p};
#[path = "data/powf.rs"]
mod fixtures;

#[test]
fn canonical_power_matches_independent_fixtures() {
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
            0 => p::try_powf_with::<4>(base, n),
            1 => p::try_powf_with::<5>(base, n),
            2 => p::try_powf_with::<6>(base, n),
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
    fn compare<const F: u32>() {
        use e::{
            ExpfDomain as D, ExpfExecution as X, ExpfPortability as P, ExpfPrecision as Q,
            ExpfValidation as V,
        };
        let eo = e::ExpfOptions::new()
            .with_domain(if F & 1 == 0 { D::Bounded } else { D::Full })
            .with_execution(if F & 4 == 0 { X::Automatic } else { X::Caller })
            .with_precision(if F & 8 == 0 { Q::Midp } else { Q::Lowp })
            .with_portability(if F & 16 == 0 {
                P::BitExact
            } else {
                P::BackendDependent
            })
            .with_validation(if F & 32 == 0 {
                V::Checked
            } else {
                V::AssumeValid
            });
        use l::{
            LogfDomain as LD, LogfExecution as LX, LogfPortability as LP, LogfPrecision as LQ,
            LogfValidation as LV,
        };
        let lo = l::LogfOptions::new()
            .with_domain(if F & 1 == 0 {
                LD::Positive
            } else {
                LD::Specials
            })
            .with_execution(if F & 4 == 0 {
                LX::Automatic
            } else {
                LX::Caller
            })
            .with_precision(if F & 8 == 0 { LQ::Midp } else { LQ::Lowp })
            .with_portability(if F & 16 == 0 {
                LP::BitExact
            } else {
                LP::BackendDependent
            })
            .with_validation(if F & 32 == 0 {
                LV::Checked
            } else {
                LV::AssumeValid
            });
        assert_eq!(eo.bits(), F);
        assert_eq!(lo.bits(), F);
        for x in [0.5, 0.99, 1., 1.5, 127.25] {
            assert_eq!(
                e::try_exp2_runtime(x, eo).unwrap().to_bits(),
                e::try_exp2_with::<F>(x).unwrap().to_bits()
            );
            assert_eq!(
                l::try_log2_runtime(x, lo).unwrap().to_bits(),
                l::try_log2_with::<F>(x).unwrap().to_bits()
            );
        }
        if F & 1 != 0 {
            assert!(e::try_exp2_with::<F>(f32::NAN).unwrap().is_nan());
            assert_eq!(e::try_exp2_with::<F>(f32::NEG_INFINITY), Ok(0.));
            assert_eq!(l::try_log2_with::<F>(-0.), Ok(f32::NEG_INFINITY));
            assert!(l::try_log2_with::<F>(f32::NAN).unwrap().is_nan());
        }
    }
    macro_rules! run{($($f:literal),*)=>{$(compare::<$f>();)*};}
    run!(
        0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29, 32, 33, 36, 37, 40, 41, 44, 45,
        48, 49, 52, 53, 56, 57, 60, 61
    );
}

#[test]
fn signed_zero_and_whole_vector_rejection() {
    for bits in [0, 0x80000000] {
        let x = f32::from_bits(bits);
        assert_eq!(l::try_log1p_with::<4>(x).unwrap().to_bits(), bits);
        assert_eq!(
            l::try_log1p_with_token::<4, _>(archmage::ScalarToken, x)
                .unwrap()
                .to_bits(),
            bits
        );
    }
    assert_eq!(
        l::try_log1p_with_token::<0, _>(archmage::ScalarToken, -2.)
            .unwrap_err()
            .lane_mask,
        1
    );
    let error = l::try_log2_array_with::<0, 4>([1., 0., f32::NAN, -1.]).unwrap_err();
    assert_eq!(error.lane_mask, 0b1110);
    assert_eq!(
        e::try_exp2_array_with::<0, 3>([0.; 3]).unwrap_err().kind,
        e::ExpfErrorKind::UnsupportedWidth
    );
}

#[archmage::magetypes(v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
fn check_tails(token: Token) {
    macro_rules! check {
        ($f:literal,$n:literal,$vec:ident,$exp:ident,$pow:ident,$log:ident,$ln:ident,$log10:ident,$log1p:ident) => {{
            use magetypes::simd::generic::$vec as Vec;
            let xs: [f32; $n] = core::array::from_fn(|i| 0.5 + (i % 8) as f32 / 16.);
            let input = Vec::from_array_t(token, xs);
            let got = e::$exp::<$f, _>(token, input).unwrap().to_array();
            let power = p::$pow::<$f, _>(token, input, 2.4).unwrap().to_array();
            let log = l::$log::<$f, _>(token, input).unwrap().to_array();
            let ln = l::$ln::<$f, _>(token, input).unwrap().to_array();
            let log10 = l::$log10::<$f, _>(token, input).unwrap().to_array();
            let log1p = l::$log1p::<$f, _>(token, input).unwrap().to_array();
            for i in 0..$n {
                assert_eq!(
                    got[i].to_bits(),
                    e::try_exp2_with_token::<$f, _>(token, xs[i])
                        .unwrap()
                        .to_bits()
                );
                assert_eq!(
                    power[i].to_bits(),
                    p::try_powf_with_token::<$f, _>(token, xs[i], 2.4)
                        .unwrap()
                        .to_bits()
                );
                assert_eq!(
                    log[i].to_bits(),
                    l::try_log2_with_token::<$f, _>(token, xs[i])
                        .unwrap()
                        .to_bits()
                );
                assert_eq!(
                    ln[i].to_bits(),
                    l::try_ln_with_token::<$f, _>(token, xs[i])
                        .unwrap()
                        .to_bits()
                );
                assert_eq!(
                    log10[i].to_bits(),
                    l::try_log10_with_token::<$f, _>(token, xs[i])
                        .unwrap()
                        .to_bits()
                );
                assert_eq!(
                    log1p[i].to_bits(),
                    l::try_log1p_with_token::<$f, _>(token, xs[i])
                        .unwrap()
                        .to_bits()
                );
            }
        }};
    }
    macro_rules! profile {
        ($f:literal) => {
            check!(
                $f,
                4,
                f32x4,
                try_exp2_f32x4_with,
                try_powf_f32x4_with,
                try_log2_f32x4_with,
                try_ln_f32x4_with,
                try_log10_f32x4_with,
                try_log1p_f32x4_with
            );
            check!(
                $f,
                8,
                f32x8,
                try_exp2_f32x8_with,
                try_powf_f32x8_with,
                try_log2_f32x8_with,
                try_ln_f32x8_with,
                try_log10_f32x8_with,
                try_log1p_f32x8_with
            );
            #[cfg(feature = "w512")]
            check!(
                $f,
                16,
                f32x16,
                try_exp2_f32x16_with,
                try_powf_f32x16_with,
                try_log2_f32x16_with,
                try_ln_f32x16_with,
                try_log10_f32x16_with,
                try_log1p_f32x16_with
            );
        };
    }
    profile!(4);
    profile!(12);
    profile!(20);
    profile!(28);
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

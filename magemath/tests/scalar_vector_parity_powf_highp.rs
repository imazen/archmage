//! Exhaustive const-generic scalar vs vector parity for `magemath::powf` HIGHP
//! (all 20 portable encodings + all 20 `std` encodings, plus encoding completeness proof).
#![forbid(unsafe_code)]

#[macro_use]
#[path = "common/powf_parity.rs"]
mod powf_parity;

macro_rules! highp_flags {
    ($m:ident $(, $arg:expr)*) => {
        $m::<64>($($arg),*);
        $m::<65>($($arg),*);
        $m::<66>($($arg),*);
        $m::<67>($($arg),*);
        $m::<68>($($arg),*);
        $m::<69>($($arg),*);
        $m::<70>($($arg),*);
        $m::<71>($($arg),*);
        $m::<96>($($arg),*);
        $m::<97>($($arg),*);
        $m::<98>($($arg),*);
        $m::<99>($($arg),*);
        $m::<100>($($arg),*);
        $m::<101>($($arg),*);
        $m::<102>($($arg),*);
        $m::<103>($($arg),*);
        $m::<194>($($arg),*);
        $m::<198>($($arg),*);
        $m::<226>($($arg),*);
        $m::<230>($($arg),*);
        #[cfg(feature = "std")]
        {
            $m::<80>($($arg),*);
            $m::<81>($($arg),*);
            $m::<82>($($arg),*);
            $m::<83>($($arg),*);
            $m::<84>($($arg),*);
            $m::<85>($($arg),*);
            $m::<86>($($arg),*);
            $m::<87>($($arg),*);
            $m::<112>($($arg),*);
            $m::<113>($($arg),*);
            $m::<114>($($arg),*);
            $m::<115>($($arg),*);
            $m::<116>($($arg),*);
            $m::<117>($($arg),*);
            $m::<118>($($arg),*);
            $m::<119>($($arg),*);
            $m::<210>($($arg),*);
            $m::<214>($($arg),*);
            $m::<242>($($arg),*);
            $m::<246>($($arg),*);
        }
    };
}

#[archmage::magetypes(
    v4x(cfg(avx512)),
    v4(cfg(avx512)),
    v3,
    neon,
    wasm128_relaxed_fma,
    wasm128,
    scalar
)]
fn check_token(token: Token) {
    use powf_parity::*;
    define_powf_token_op!(tok_powf, Token, yes);
    highp_flags!(tok_powf, token);
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::arcane]
fn check_token_fp16(token: archmage::Avx512Fp16Token) {
    use powf_parity::*;
    define_powf_token_op!(fp16_powf, archmage::Avx512Fp16Token, no);
    highp_flags!(fp16_powf, token);
}

#[test]
fn powf_highp_scalar_and_vector_parity_across_all_configs() {
    use powf_parity::standalone_powf;
    highp_flags!(standalone_powf);
    run_all_summonable_tokens!();
}

#[test]
fn tested_powf_encodings_cover_every_supported_option_combination() {
    use magemath::powf::{
        PowfDomain, PowfErrorKind, PowfExecution, PowfOptions, PowfPortability, PowfPrecision,
        PowfValidation, try_powf_runtime,
    };

    const MIDP: &[u32] = &[
        0, 1, 2, 4, 5, 6, 16, 17, 18, 20, 21, 22, 32, 33, 34, 36, 37, 38, 48, 49, 50, 52, 53, 54,
        130, 134, 146, 150, 162, 166, 178, 182,
    ];
    const LOWP: &[u32] = &[
        8, 9, 10, 12, 13, 14, 24, 25, 26, 28, 29, 30, 40, 41, 42, 44, 45, 46, 56, 57, 58, 60, 61,
        62, 138, 142, 154, 158, 170, 174, 186, 190,
    ];
    const HIGHP_PORTABLE: &[u32] = &[
        64, 65, 66, 67, 68, 69, 70, 71, 96, 97, 98, 99, 100, 101, 102, 103, 194, 198, 226, 230,
    ];
    #[cfg(feature = "std")]
    const HIGHP_STD: &[u32] = &[
        80, 81, 82, 83, 84, 85, 86, 87, 112, 113, 114, 115, 116, 117, 118, 119, 210, 214, 242, 246,
    ];
    #[cfg(not(feature = "std"))]
    const HIGHP_STD: &[u32] = &[];

    let mut valid_count = 0usize;
    for d in [
        PowfDomain::PositiveNormal,
        PowfDomain::NonNegativeNormal,
        PowfDomain::Gamma,
        PowfDomain::PositiveGamma,
        PowfDomain::Full,
    ] {
        for x in [PowfExecution::Automatic, PowfExecution::Caller] {
            for q in [
                PowfPrecision::Midp,
                PowfPrecision::Lowp,
                PowfPrecision::Highp,
            ] {
                for p in [PowfPortability::BitExact, PowfPortability::BackendDependent] {
                    for v in [PowfValidation::Checked, PowfValidation::AssumeValid] {
                        let opts = PowfOptions::new()
                            .with_domain(d)
                            .with_execution(x)
                            .with_precision(q)
                            .with_portability(p)
                            .with_validation(v);
                        let bits = opts.bits();
                        let in_suite = MIDP.contains(&bits)
                            || LOWP.contains(&bits)
                            || HIGHP_PORTABLE.contains(&bits)
                            || HIGHP_STD.contains(&bits);
                        match try_powf_runtime(2.0, 0.5, opts) {
                            Ok(_) => {
                                assert!(
                                    in_suite,
                                    "supported PowfOptions encoding {bits} ({opts:?}) missing from parity suite"
                                );
                                valid_count += 1;
                            }
                            Err(err) => {
                                assert_eq!(err.kind, PowfErrorKind::UnsupportedOptions);
                                assert!(
                                    !in_suite,
                                    "unsupported PowfOptions encoding {bits} ({opts:?}) unexpectedly in suite"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    assert_eq!(
        valid_count,
        MIDP.len() + LOWP.len() + HIGHP_PORTABLE.len() + HIGHP_STD.len()
    );
}

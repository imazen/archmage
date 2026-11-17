//! Exhaustive const-generic scalar vs vector parity for `magemath::powf` HIGHP
//! (all 20 portable encodings + all 20 `std` encodings, plus encoding completeness proof).
#![forbid(unsafe_code)]

#[macro_use]
#[path = "common/powf_parity.rs"]
mod powf_parity;

macro_rules! highp_flags {
    ($m:ident $(, $arg:expr)*) => {
        macro_rules! call_one {
            ($f:literal) => {
                $m::<$f, { powf_parity::powf_options_from_bits($f).bits() }>($($arg),*);
            };
        }
        call_one!(64);
        call_one!(65);
        call_one!(66);
        call_one!(67);
        call_one!(68);
        call_one!(69);
        call_one!(70);
        call_one!(71);
        call_one!(96);
        call_one!(97);
        call_one!(98);
        call_one!(99);
        call_one!(100);
        call_one!(101);
        call_one!(102);
        call_one!(103);
        call_one!(194);
        call_one!(198);
        call_one!(226);
        call_one!(230);
        #[cfg(feature = "std")]
        {
            call_one!(80);
            call_one!(81);
            call_one!(82);
            call_one!(83);
            call_one!(84);
            call_one!(85);
            call_one!(86);
            call_one!(87);
            call_one!(112);
            call_one!(113);
            call_one!(114);
            call_one!(115);
            call_one!(116);
            call_one!(117);
            call_one!(118);
            call_one!(119);
            call_one!(210);
            call_one!(214);
            call_one!(242);
            call_one!(246);
        }
    };
}

#[archmage::magetypes(v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
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
    use magemath::{
        Execution, Portability, Validation,
        powf::{PowfErrorKind, PowfOptions, PowfPrecision},
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

    let suite: Vec<u128> = MIDP
        .iter()
        .chain(LOWP)
        .chain(HIGHP_PORTABLE)
        .chain(HIGHP_STD)
        .map(|&f| powf_parity::powf_options_from_bits(f).bits())
        .collect();

    let mut valid_count = 0usize;
    for base in [
        PowfOptions::new(),
        PowfOptions::new().admit_zero(),
        PowfOptions::new().gamma(),
        PowfOptions::new().positive_gamma(),
        PowfOptions::new().full_domain(),
    ] {
        for x in [Execution::Upgrade, Execution::Caller] {
            for q in [
                PowfPrecision::Midp,
                PowfPrecision::Lowp,
                PowfPrecision::Highp,
            ] {
                for p in [Portability::BitExact, Portability::BackendDependent] {
                    for v in [Validation::Checked, Validation::AssumeValid] {
                        let opts = base.execution(x).precision(q).portability(p).validation(v);
                        match opts.runtime().scalar(2.0, 0.5) {
                            Ok(_) => {
                                let bits = opts.bits();
                                assert!(
                                    suite.contains(&bits),
                                    "supported PowfOptions encoding ({opts:?}) missing from parity suite"
                                );
                                valid_count += 1;
                            }
                            Err(err) => {
                                assert_eq!(err.kind, PowfErrorKind::UnsupportedOptions);
                                assert!(!opts.is_valid());
                            }
                        }
                    }
                }
            }
        }
    }
    assert_eq!(valid_count, suite.len());
}

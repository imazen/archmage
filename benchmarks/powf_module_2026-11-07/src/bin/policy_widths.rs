//! Public const/runtime policy permutations versus independent rational fixtures.
#![forbid(unsafe_code)]
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
use archmage::SimdToken;
use magetypes::powf::PowfErrorKind;
use std::{
    collections::BTreeMap,
    io::{self, BufRead},
};

#[derive(Clone, Copy)]
struct Case {
    base: u32,
    expected: [u64; 4],
}

#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
fn relaxed_probe(token: archmage::Wasm128Token) -> u32 {
    use magetypes::simd::generic::f32x4;
    let v = |bits| f32x4::splat_t(token, std::hint::black_box(f32::from_bits(bits)));
    v(0x3f800001)
        .mul_add_relaxed(v(0x3f7ffffe), v(0xbf800000))
        .to_array()[0]
        .to_bits()
}

macro_rules! width {
    ($check:ident,$execute:ident,$kernel:ident,$vector:ident,$n:literal) => {
        #[archmage::magetypes(v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
        fn $check(
            token: Token,
            input: [f32; $n],
            exponent: f32,
            domain: u32,
            expected: [[u64; 4]; $n],
        ) {
            use magetypes::simd::generic::$vector as V;
            let v = V::from_array_t(token, input);
            let caller_fuses = core::any::TypeId::of::<Token>()
                != core::any::TypeId::of::<archmage::ScalarToken>()
                && backend_fuses();
            for flags in 0..64u32 {
                if flags & 3 != domain {
                    continue;
                }
                let fused = flags & 16 == 0
                    || if flags & 4 != 0 {
                        caller_fuses
                    } else {
                        backend_fuses()
                    };
                let method = usize::from(flags & 8 != 0) * 2 + usize::from(fused);
                let stage = expected
                    .iter()
                    .map(|r| r[method] >> 32)
                    .filter(|&x| x != 0)
                    .min()
                    .unwrap_or(0);
                if flags & 32 != 0 && stage != 0 {
                    continue;
                } // Caller only promises valid lanes.
                macro_rules! call {
                    ($flags:literal) => {
                        v.try_powf_with::<$flags>(exponent)
                    };
                }
                let actual = match flags {
                    0 => call!(0),
                    1 => call!(1),
                    2 => call!(2),
                    4 => call!(4),
                    5 => call!(5),
                    6 => call!(6),
                    8 => call!(8),
                    9 => call!(9),
                    10 => call!(10),
                    12 => call!(12),
                    13 => call!(13),
                    14 => call!(14),
                    16 => call!(16),
                    17 => call!(17),
                    18 => call!(18),
                    20 => call!(20),
                    21 => call!(21),
                    22 => call!(22),
                    24 => call!(24),
                    25 => call!(25),
                    26 => call!(26),
                    28 => call!(28),
                    29 => call!(29),
                    30 => call!(30),
                    32 => call!(32),
                    33 => call!(33),
                    34 => call!(34),
                    36 => call!(36),
                    37 => call!(37),
                    38 => call!(38),
                    40 => call!(40),
                    41 => call!(41),
                    42 => call!(42),
                    44 => call!(44),
                    45 => call!(45),
                    46 => call!(46),
                    48 => call!(48),
                    49 => call!(49),
                    50 => call!(50),
                    52 => call!(52),
                    53 => call!(53),
                    54 => call!(54),
                    56 => call!(56),
                    57 => call!(57),
                    58 => call!(58),
                    60 => call!(60),
                    61 => call!(61),
                    62 => call!(62),
                    _ => unreachable!(),
                };
                let opts = options(flags);
                let dynamic = v.try_powf_runtime(exponent, opts);
                assert_eq!(
                    actual.map(|v| v.to_array().map(f32::to_bits)),
                    dynamic.map(|v| v.to_array().map(f32::to_bits)),
                    "const/runtime flags={flags}"
                );
                let stage = expected
                    .iter()
                    .map(|r| r[method] >> 32)
                    .filter(|&x| x != 0)
                    .min()
                    .unwrap_or(0);
                match actual {
                    Ok(values) => {
                        assert_eq!(stage, 0, "expected rejection; width {} method {method}", $n);
                        for (lane, value) in values.to_array().iter().enumerate() {
                            assert_eq!(
                                u64::from(value.to_bits()),
                                expected[lane][method],
                                "width {} method {method} lane {lane} exponent {exponent:?}",
                                $n
                            );
                        }
                    }
                    Err(error) => {
                        let code = match error.kind {
                            PowfErrorKind::InvalidExponent => 1,
                            PowfErrorKind::InvalidBase => 2,
                            PowfErrorKind::OutsideApproximationRange => 3,
                            PowfErrorKind::UnsupportedWidth => 4,
                            _ => panic!("unknown rejection"),
                        };
                        assert_eq!(code, stage, "rejection stage width {} method {method}", $n);
                        let mask = expected.iter().enumerate().fold(0u16, |mask, (lane, r)| {
                            mask | if r[method] >> 32 == stage {
                                1 << lane
                            } else {
                                0
                            }
                        });
                        assert_eq!(
                            error.lane_mask, mask,
                            "rejection mask width {} method {method}",
                            $n
                        );
                    }
                }
            }
        }
        fn $execute(
            context: &str,
            input: [f32; $n],
            exponent: f32,
            domain: u32,
            expected: [[u64; 4]; $n],
        ) {
            if context == "scalar" {
                return $check::scalar(archmage::ScalarToken, input, exponent, domain, expected);
            }
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            if context == "v4" {
                return $check::v4(
                    archmage::X64V4Token::summon().expect("required V4"),
                    input,
                    exponent,
                    domain,
                    expected,
                );
            }
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            if context == "v4x" {
                return $check::v4x(
                    archmage::X64V4xToken::summon().expect("required V4x"),
                    input,
                    exponent,
                    domain,
                    expected,
                );
            }
            assert_eq!(context, "native", "explicit requested backend unavailable");
            #[cfg(target_arch = "x86_64")]
            return $check::v3(
                archmage::X64V3Token::summon().expect("required V3"),
                input,
                exponent,
                domain,
                expected,
            );
            #[cfg(target_arch = "aarch64")]
            return $check::neon(
                archmage::NeonToken::summon().expect("required NEON"),
                input,
                exponent,
                domain,
                expected,
            );
            #[cfg(target_arch = "wasm32")]
            return $check::wasm128(
                archmage::Wasm128Token::summon().expect("required SIMD128"),
                input,
                exponent,
                domain,
                expected,
            );
            #[cfg(not(any(
                target_arch = "x86_64",
                target_arch = "aarch64",
                target_arch = "wasm32"
            )))]
            panic!("native context unsupported");
        }
    };
}

// A module provides stable names for the macro-generated feature entries.
mod check4 {
    #[cfg(target_arch = "aarch64")]
    pub(super) use super::check4_neon as neon;
    pub(super) use super::check4_scalar as scalar;
    #[cfg(target_arch = "x86_64")]
    pub(super) use super::check4_v3 as v3;
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    pub(super) use super::check4_v4 as v4;
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    pub(super) use super::check4_v4x as v4x;
    #[cfg(target_arch = "wasm32")]
    pub(super) use super::check4_wasm128 as wasm128;
}
mod check8 {
    #[cfg(target_arch = "aarch64")]
    pub(super) use super::check8_neon as neon;
    pub(super) use super::check8_scalar as scalar;
    #[cfg(target_arch = "x86_64")]
    pub(super) use super::check8_v3 as v3;
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    pub(super) use super::check8_v4 as v4;
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    pub(super) use super::check8_v4x as v4x;
    #[cfg(target_arch = "wasm32")]
    pub(super) use super::check8_wasm128 as wasm128;
}
mod check16 {
    #[cfg(target_arch = "aarch64")]
    pub(super) use super::check16_neon as neon;
    pub(super) use super::check16_scalar as scalar;
    #[cfg(target_arch = "x86_64")]
    pub(super) use super::check16_v3 as v3;
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    pub(super) use super::check16_v4 as v4;
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    pub(super) use super::check16_v4x as v4x;
    #[cfg(target_arch = "wasm32")]
    pub(super) use super::check16_wasm128 as wasm128;
}
width!(check4, execute4, kernel4, f32x4, 4);
width!(check8, execute8, kernel8, f32x8, 8);
width!(check16, execute16, kernel16, f32x16, 16);

fn options(flags: u32) -> magetypes::powf::PowfOptions {
    use magetypes::powf::*;
    PowfOptions::new()
        .with_domain(match flags & 3 {
            0 => PowfDomain::PositiveNormal,
            1 => PowfDomain::NonNegativeNormal,
            2 => PowfDomain::Gamma,
            _ => unreachable!(),
        })
        .with_execution(if flags & 4 != 0 {
            PowfExecution::Caller
        } else {
            PowfExecution::Automatic
        })
        .with_precision(if flags & 8 != 0 {
            PowfPrecision::Lowp
        } else {
            PowfPrecision::Midp
        })
        .with_portability(if flags & 16 != 0 {
            PowfPortability::BackendDependent
        } else {
            PowfPortability::BitExact
        })
        .with_validation(if flags & 32 != 0 {
            PowfValidation::AssumeValid
        } else {
            PowfValidation::Checked
        })
}
fn backend_fuses() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        archmage::X64V3Token::summon().is_some()
    }
    #[cfg(target_arch = "aarch64")]
    {
        archmage::NeonToken::summon().is_some()
    }
    #[cfg(target_arch = "wasm32")]
    {
        relaxed_probe(archmage::Wasm128Token::summon().unwrap()) != 0
    }
    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32"
    )))]
    {
        false
    }
}

fn main() {
    #[cfg(target_arch = "wasm32")]
    {
        let expected = match std::env::var("POWF_EXPECT_RELAXED_FUSED").unwrap().as_str() {
            "true" => 0xa8800000,
            "false" => 0,
            _ => panic!("explicit engine rounding required"),
        };
        let actual = relaxed_probe(archmage::Wasm128Token::summon().unwrap());
        assert_eq!(actual, expected, "actual relaxed provider rounding");
        eprintln!("relaxed provider cancellation bits {actual:08x}");
    }
    let context = std::env::args().nth(1).expect("explicit context required");
    assert!(matches!(
        context.as_str(),
        "scalar" | "native" | "v4" | "v4x"
    ));
    let mut groups: BTreeMap<(u32, u32), Vec<Case>> = BTreeMap::new();
    let mut broadcasts = 0;
    let mut mixed = 0;
    let mut rows = 0;
    for line in io::stdin().lock().lines() {
        let line = line.unwrap();
        let fields: Vec<_> = line.split_whitespace().collect();
        assert_eq!(fields.len(), 7);
        let base = u32::from_str_radix(fields[0], 16).unwrap();
        let n = u32::from_str_radix(fields[1], 16).unwrap();
        let domain = fields[2].parse().unwrap();
        let expected = std::array::from_fn(|m| u64::from_str_radix(fields[3 + m], 16).unwrap());
        execute4(
            &context,
            [f32::from_bits(base); 4],
            f32::from_bits(n),
            domain,
            [expected; 4],
        );
        broadcasts += 1;
        execute8(
            &context,
            [f32::from_bits(base); 8],
            f32::from_bits(n),
            domain,
            [expected; 8],
        );
        broadcasts += 1;
        execute16(
            &context,
            [f32::from_bits(base); 16],
            f32::from_bits(n),
            domain,
            [expected; 16],
        );
        broadcasts += 1;
        groups
            .entry((n, domain))
            .or_default()
            .push(Case { base, expected });
        rows += 1;
    }
    for ((n, domain), cases) in groups {
        macro_rules! chunks {
            ($execute:ident,$width:literal) => {
                for chunk in cases.chunks($width) {
                    let input =
                        std::array::from_fn(|lane| f32::from_bits(chunk[lane % chunk.len()].base));
                    let expected = std::array::from_fn(|lane| chunk[lane % chunk.len()].expected);
                    $execute(&context, input, f32::from_bits(n), domain, expected);
                    mixed += 1;
                }
            };
        }
        chunks!(execute4, 4);
        chunks!(execute8, 8);
        chunks!(execute16, 16);
    }
    assert!(rows > 0);
    println!(
        "{}",
        serde_json::json!({"rows":rows,"context":context,"widths":[4,8,16],"broadcast_vectors":broadcasts,"mixed_vectors":mixed,"policy_encodings":48,"assume_valid":"accepted cases only","mismatches":0})
    );
}

//! Independent rational expectations across every width, lane and rejection mask.
#![forbid(unsafe_code)]
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
        #[archmage::magetypes(v3, neon, wasm128, scalar)]
        fn $check(
            token: Token,
            input: [f32; $n],
            exponent: f32,
            domain: u32,
            expected: [[u64; 4]; $n],
        ) {
            use magetypes::simd::generic::$vector as V;
            let v = V::from_array_t(token, input);
            for method in 0..4 {
                macro_rules! go {
                    ($lowp:literal,$fused:literal) => {
                        match domain {
                            0 => archmage_powf_study::rounding::$kernel::<
                                Token,
                                $lowp,
                                $fused,
                                true,
                                0,
                            >(token, v, exponent),
                            1 => archmage_powf_study::rounding::$kernel::<
                                Token,
                                $lowp,
                                $fused,
                                true,
                                1,
                            >(token, v, exponent),
                            2 => archmage_powf_study::rounding::$kernel::<
                                Token,
                                $lowp,
                                $fused,
                                true,
                                2,
                            >(token, v, exponent),
                            _ => panic!("unknown domain"),
                        }
                    };
                }
                let actual = match method {
                    0 => go!(false, false),
                    1 => go!(false, true),
                    2 => go!(true, false),
                    3 => go!(true, true),
                    _ => unreachable!(),
                };
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
            native: bool,
            input: [f32; $n],
            exponent: f32,
            domain: u32,
            expected: [[u64; 4]; $n],
        ) {
            if !native {
                return $check::scalar(archmage::ScalarToken, input, exponent, domain, expected);
            }
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
    #[cfg(target_arch = "wasm32")]
    pub(super) use super::check4_wasm128 as wasm128;
}
mod check8 {
    #[cfg(target_arch = "aarch64")]
    pub(super) use super::check8_neon as neon;
    pub(super) use super::check8_scalar as scalar;
    #[cfg(target_arch = "x86_64")]
    pub(super) use super::check8_v3 as v3;
    #[cfg(target_arch = "wasm32")]
    pub(super) use super::check8_wasm128 as wasm128;
}
mod check16 {
    #[cfg(target_arch = "aarch64")]
    pub(super) use super::check16_neon as neon;
    pub(super) use super::check16_scalar as scalar;
    #[cfg(target_arch = "x86_64")]
    pub(super) use super::check16_v3 as v3;
    #[cfg(target_arch = "wasm32")]
    pub(super) use super::check16_wasm128 as wasm128;
}
width!(check4, execute4, kernel4, f32x4, 4);
width!(check8, execute8, kernel8, f32x8, 8);
width!(check16, execute16, kernel16, f32x16, 16);

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
    let native = match std::env::args().nth(1).as_deref() {
        Some("scalar") => false,
        Some("native") => true,
        _ => panic!("explicit scalar or native context required"),
    };
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
            native,
            [f32::from_bits(base); 4],
            f32::from_bits(n),
            domain,
            [expected; 4],
        );
        broadcasts += 1;
        execute8(
            native,
            [f32::from_bits(base); 8],
            f32::from_bits(n),
            domain,
            [expected; 8],
        );
        broadcasts += 1;
        execute16(
            native,
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
                    $execute(native, input, f32::from_bits(n), domain, expected);
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
        serde_json::json!({"rows":rows,"native":native,"widths":[4,8,16],"broadcast_vectors":broadcasts,"mixed_vectors":mixed,"methods":4,"mismatches":0})
    );
}

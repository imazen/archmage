//! Exact fused rounding, or the engine result for relaxed WASM; NaNs are unspecified.
#![forbid(unsafe_code)]
use archmage::{ScalarToken, SimdToken, incant, magetypes};
use magetypes::nostd_math;
#[path = "common/fma_expected.rs"]
mod fma_expected;
use fma_expected::FmaExpected;

fn same32(actual: f32, expected: f32) {
    if expected.is_nan() {
        assert!(actual.is_nan());
    } else {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{actual:?} != {expected:?}"
        );
    }
}
fn same64(actual: f64, expected: f64) {
    if expected.is_nan() {
        assert!(actual.is_nan());
    } else {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{actual:?} != {expected:?}"
        );
    }
}
fn random(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn cases32() -> Vec<[f32; 3]> {
    let rails = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::MAX,
        f32::MIN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0x7f80_0001),
        f32::NAN,
    ];
    let mut cases = Vec::new();
    for a in rails {
        for b in rails {
            for c in rails {
                cases.push([a, b, c]);
            }
        }
    }
    // Double-rounding traps: the f64 sum lands on a midpoint, although the
    // exact result lies just below it. Vary the binade, parity, and sign.
    for exponent in 1..254u32 {
        let c = f32::from_bits(exponent << 23 | 1);
        let a = (f64::from(c) * (1.0 / 16777216.0)) as f32;
        for sign in [1.0, -1.0] {
            cases.push([sign * a, f32::from_bits(0x3f7f_fffe), sign * c]);
        }
    }
    cases.extend([
        [
            f32::from_bits(0x3380_0001),
            f32::from_bits(0x3f7f_fffe),
            f32::from_bits(0x3f80_0001),
        ],
        [
            f32::from_bits(0x3f80_0800),
            f32::from_bits(0x3f80_0800),
            f32::from_bits(0xbf80_1000),
        ],
        [f32::MAX, 2.0, -f32::MAX],
        [f32::from_bits(1), 0.5, -0.0],
    ]);
    let mut state = 0xa164_901b_36ec_f815;
    for _ in 0..4096 {
        cases.push(core::array::from_fn(|_| {
            f32::from_bits(random(&mut state) as u32)
        }));
    }
    cases
}
fn cases64() -> Vec<[f64; 3]> {
    let rails = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        f64::from_bits(1),
        -f64::from_bits(1),
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        f64::MAX,
        f64::MIN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::from_bits(0x7ff0_0000_0000_0001),
        f64::NAN,
    ];
    let mut cases = Vec::new();
    for a in rails {
        for b in rails {
            for c in rails {
                cases.push([a, b, c]);
            }
        }
    }
    cases.extend([
        [
            f64::from_bits(0x3ff0_0000_0200_0000),
            f64::from_bits(0x3ff0_0000_0200_0000),
            f64::from_bits(0xbff0_0000_0400_0000),
        ],
        [f64::MAX, 2.0, -f64::MAX],
        [f64::from_bits(1), 0.5, -0.0],
        [
            f64::from_bits(0x0010_0000_0000_0001),
            0.5,
            -f64::from_bits(1),
        ],
    ]);
    let mut state = 0x7438_736a_f231_6810;
    for _ in 0..4096 {
        cases.push(core::array::from_fn(|_| f64::from_bits(random(&mut state))));
    }
    cases
}

#[test]
fn software_rounding_against_std() {
    for [a, b, c] in cases32() {
        same32(nostd_math::fmaf(a, b, c), a.fused_expected(b, c));
    }
    for [a, b, c] in cases64() {
        same64(nostd_math::fma(a, b, c), a.fused_expected(b, c));
    }
}

#[test]
fn random_software_rounding_against_std() {
    let mut state = 0xe164_48b8_3902_1619;
    for _ in 0..1_000_000 {
        let [a, b, c] = core::array::from_fn(|_| f32::from_bits(random(&mut state) as u32));
        same32(nostd_math::fmaf(a, b, c), a.fused_expected(b, c));
        let [a, b, c] = core::array::from_fn(|_| f64::from_bits(random(&mut state)));
        same64(nostd_math::fma(a, b, c), a.fused_expected(b, c));
    }
}

#[test]
fn exact_counterexamples() {
    let [a, b, c] = [0x3380_0001, 0x3f7f_fffe, 0x3f80_0001].map(f32::from_bits);
    assert_eq!(
        ((f64::from(a) * f64::from(b) + f64::from(c)) as f32).to_bits(),
        0x3f80_0002
    );
    assert_eq!(nostd_math::fmaf(a, b, c).to_bits(), 0x3f80_0001);
    assert_eq!(nostd_math::fmaf(f32::MAX, 2.0, -f32::MAX), f32::MAX);
    assert_eq!(nostd_math::fma(f64::MAX, 2.0, -f64::MAX), f64::MAX);
    assert_eq!(
        nostd_math::fmaf(-f32::from_bits(1), 0.5, 0.0).to_bits(),
        (-0.0f32).to_bits()
    );
    // Exact negative products round to -0, including when the addend is +0.
    for (a, b) in [
        (f64::from_bits(1), -f64::from_bits(1)),
        (-f64::from_bits(1), 0.5),
    ] {
        assert_eq!(a.fused_expected(b, 0.0).to_bits(), (-0.0f64).to_bits());
        assert_eq!(nostd_math::fma(a, b, 0.0).to_bits(), (-0.0f64).to_bits());
    }
}

macro_rules! check {
    ($token:expr, $ty:ident, $n:expr, $inputs:expr, $same:ident) => {
        for start in (0..$inputs.len()).step_by($n) {
            let cases: [_; $n] = core::array::from_fn(|i| $inputs[(start + i) % $inputs.len()]);
            let a = magetypes::simd::generic::$ty::from_array_t($token, cases.map(|v| v[0]));
            let b = magetypes::simd::generic::$ty::from_array_t($token, cases.map(|v| v[1]));
            let c = magetypes::simd::generic::$ty::from_array_t($token, cases.map(|v| v[2]));
            let add = a.mul_add(b, c).to_array();
            let sub = a.mul_sub(b, c).to_array();
            for i in 0..$n {
                let [x, y, z] = cases[i];
                if !add[i].is_nan() && add[i].to_bits() != x.vector_expected(y, z, $token).to_bits()
                {
                    eprintln!("{} mul_add input {:?}", stringify!($ty), cases[i]);
                }
                if !sub[i].is_nan()
                    && sub[i].to_bits() != x.vector_expected(y, -z, $token).to_bits()
                {
                    eprintln!("{} mul_sub input {:?}", stringify!($ty), cases[i]);
                }
                $same(add[i], x.vector_expected(y, z, $token));
                $same(sub[i], x.vector_expected(y, -z, $token));
            }
        }
    };
}

// Each concrete variant exercises every enabled width with the same inputs.
#[magetypes(v3, v4, v4x, neon, wasm128, scalar)]
fn vectors(token: Token, inputs32: &[[f32; 3]], inputs64: &[[f64; 3]]) {
    check!(token, f32x4, 4, inputs32, same32);
    check!(token, f32x8, 8, inputs32, same32);
    incant!(narrow_f64(inputs64), [v3, neon, wasm128, scalar]);
    #[cfg(feature = "w512")]
    {
        check!(token, f32x16, 16, inputs32, same32);
        check!(token, f64x8, 8, inputs64, same64);
    }
}

#[magetypes(v3, neon, wasm128, scalar)]
fn narrow_f64(token: Token, inputs64: &[[f64; 3]]) {
    check!(token, f64x2, 2, inputs64, same64);
    check!(token, f64x4, 4, inputs64, same64);
}

#[test]
fn rounding_contract_across_widths_and_tokens() {
    let a = cases32();
    let b = cases64();
    vectors_scalar(ScalarToken, &a, &b);
    narrow_f64_scalar(ScalarToken, &b);
    for &[x, y, z] in &a {
        use magetypes::simd::f32x1 as V;
        same32(
            V::splat_t(ScalarToken, x)
                .mul_add(V::splat_t(ScalarToken, y), V::splat_t(ScalarToken, z))
                .to_array()[0],
            x.fused_expected(y, z),
        );
    }
    for &[x, y, z] in &b {
        use magetypes::simd::f64x1 as V;
        same64(
            V::splat_t(ScalarToken, x)
                .mul_add(V::splat_t(ScalarToken, y), V::splat_t(ScalarToken, z))
                .to_array()[0],
            x.fused_expected(y, z),
        );
    }
    match std::env::var("ARCHMAGE_TEST_TIER")
        .as_deref()
        .unwrap_or("auto")
    {
        "scalar" => {}
        #[cfg(target_arch = "x86_64")]
        "v3" => vectors_v3(
            archmage::X64V3Token::summon().expect("requested v3"),
            &a,
            &b,
        ),
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        "v4" => vectors_v4(
            archmage::X64V4Token::summon().expect("requested v4"),
            &a,
            &b,
        ),
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        "v4x" => vectors_v4x(
            archmage::X64V4xToken::summon().expect("requested v4x"),
            &a,
            &b,
        ),
        #[cfg(target_arch = "aarch64")]
        "neon" => vectors_neon(
            archmage::NeonToken::summon().expect("requested neon"),
            &a,
            &b,
        ),
        #[cfg(target_arch = "wasm32")]
        "wasm128" => vectors_wasm128(
            archmage::Wasm128Token::summon().expect("requested wasm128"),
            &a,
            &b,
        ),
        "auto" => incant!(vectors(&a, &b), [v4, v3, neon, wasm128, scalar]),
        tier => panic!("unsupported requested tier {tier}"),
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[test]
fn wasm_f32_widths() {
    let token = archmage::Wasm128Token::summon().expect("simd128 build");
    let inputs = cases32();
    check!(token, f32x4, 4, inputs, same32);
    check!(token, f32x8, 8, inputs, same32);
    #[cfg(feature = "w512")]
    check!(token, f32x16, 16, inputs, same32);
}

#[cfg(all(target_arch = "wasm32", target_feature = "relaxed-simd"))]
#[test]
fn relaxed_engine_projection() {
    #[archmage::arcane]
    fn probe(_token: archmage::Wasm128RelaxedToken) -> (u32, u64) {
        use core::arch::wasm32::*;
        let a = f32x4_splat(core::hint::black_box(f32::from_bits(0x3f80_0800)));
        let c = f32x4_splat(core::hint::black_box(f32::from_bits(0xbf80_1000)));
        let x = f32x4_extract_lane::<0>(f32x4_relaxed_madd(a, a, c)).to_bits();
        let a = f64x2_splat(core::hint::black_box(f64::from_bits(0x3ff0_0000_0200_0000)));
        let c = f64x2_splat(core::hint::black_box(f64::from_bits(0xbff0_0000_0400_0000)));
        (
            x,
            f64x2_extract_lane::<0>(f64x2_relaxed_madd(a, a, c)).to_bits(),
        )
    }
    let (x, y) = probe(archmage::Wasm128RelaxedToken::summon().expect("relaxed-simd build"));
    assert!(x == 0 || x == 0x3380_0000);
    assert!(y == 0 || y == 969u64 << 52);
    eprintln!(
        "raw relaxed FMA: f32 fused={}, f64 fused={}",
        x != 0,
        y != 0
    );
    if let Ok(expected) = std::env::var("ARCHMAGE_EXPECT_RELAXED_FUSION") {
        assert!(matches!(expected.as_str(), "yes" | "no"));
        assert_eq!((x != 0, y != 0), (expected == "yes", expected == "yes"));
    }
}

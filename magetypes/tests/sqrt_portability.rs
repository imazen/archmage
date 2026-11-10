//! Exact fixtures are independent of std sqrt and the retained software kernel.
//! All comparisons retain signed zeros; only NaN payloads are uncontracted.
#![forbid(unsafe_code)]
#[path = "data/sqrt_vectors.rs"]
mod fixtures;

fn same32(actual: f32, bits: u32) {
    let expected = f32::from_bits(bits);
    if expected.is_nan() {
        assert!(actual.is_nan(), "expected NaN, got {actual:?}");
    } else {
        assert_eq!(actual.to_bits(), bits, "sqrt result {actual:?}");
    }
}
fn same64(actual: f64, bits: u64) {
    let expected = f64::from_bits(bits);
    if expected.is_nan() {
        assert!(actual.is_nan(), "expected NaN, got {actual:?}");
    } else {
        assert_eq!(actual.to_bits(), bits, "sqrt result {actual:?}");
    }
}

#[archmage::magetypes(v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
fn widths(token: Token) {
    macro_rules! check {
        ($vec:ident, $float:ident, $data:ident, $same:ident, $n:literal) => {{
            for start in (0..fixtures::$data.len()).step_by($n) {
                let cases = core::array::from_fn::<_, $n, _>(|i| {
                    fixtures::$data[(start + i) % fixtures::$data.len()]
                });
                let input = cases.map(|(x, _)| $float::from_bits(x));
                let result = magetypes::simd::generic::$vec::from_array_t(token, input)
                    .sqrt()
                    .to_array();
                for (actual, (_, expected)) in result.into_iter().zip(cases) {
                    $same(actual, expected);
                }
            }
        }};
    }
    check!(f32x4, f32, F32, same32, 4);
    check!(f32x8, f32, F32, same32, 8);
    check!(f64x2, f64, F64, same64, 2);
    check!(f64x4, f64, F64, same64, 4);
    #[cfg(feature = "w512")]
    {
        check!(f32x16, f32, F32, same32, 16);
        check!(f64x8, f64, F64, same64, 8);
    }
}

#[test]
fn rational_fixtures_match_scalar_and_vector_widths() {
    for &(input, expected) in fixtures::F32 {
        let input = f32::from_bits(input);
        same32(magetypes::nostd_math::sqrtf(input), expected);
        same32(magetypes::simd::scalar::f32x1(input).sqrt().0, expected);
        same32(input.sqrt(), expected);
    }
    for &(input, expected) in fixtures::F64 {
        let input = f64::from_bits(input);
        same64(magetypes::nostd_math::sqrt(input), expected);
        same64(magetypes::simd::scalar::f64x1(input).sqrt().0, expected);
        same64(input.sqrt(), expected);
    }
    widths_scalar(archmage::ScalarToken);
    #[cfg(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32"
    ))]
    use archmage::SimdToken;
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(t) = archmage::X64V3Token::summon() {
            widths_v3(t);
        }
        #[cfg(feature = "avx512")]
        {
            if let Some(t) = archmage::X64V4Token::summon() {
                widths_v4(t);
            }
            if let Some(t) = archmage::X64V4xToken::summon() {
                widths_v4x(t);
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    widths_neon(archmage::NeonToken::summon().unwrap());
    #[cfg(target_arch = "wasm32")]
    if let Some(t) = archmage::Wasm128Token::summon() {
        widths_wasm128(t);
    }
}

#[test]
fn f32_stream_matches_hardware_and_retained_software() {
    // The caller explicitly chooses the opt-in full sweep (stride 1).
    // Keep memory constant: no corpus array or output buffer is materialized.
    let stride = std::env::var("ARCHMAGE_SQRT_F32_STRIDE").map_or(65_537, |s| {
        s.parse::<usize>().expect("positive sqrt stride")
    });
    assert!(stride > 0);
    for bits in (0..=u32::MAX).step_by(stride) {
        let input = f32::from_bits(bits);
        let expected = input.sqrt();
        same32(magetypes::nostd_math::sqrtf(input), expected.to_bits());
        same32(
            magetypes::simd::scalar::f32x1(input).sqrt().0,
            expected.to_bits(),
        );
    }
}

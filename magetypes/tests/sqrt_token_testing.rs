//! Explicitly selected by `just sqrt-token-testing`; one test per process.
//! Both tested libraries can be no_std; the test harness itself uses std.
#![forbid(unsafe_code)]

#[path = "data/sqrt_vectors.rs"]
mod fixtures;

#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
type Token = archmage::X64V1Token;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
type Token = archmage::NeonToken;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
type Token = archmage::Wasm128Token;

#[cfg(any(
    all(target_arch = "x86_64", target_feature = "sse2"),
    all(target_arch = "aarch64", target_feature = "neon"),
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[test]
fn exact_sqrt_with_baseline_token_enabled_and_disabled() {
    use archmage::SimdToken;
    // Fail loudly if the caller omitted the required testing feature.
    assert_eq!(
        Token::compiled_with(),
        None,
        "requires archmage/testable_dispatch"
    );
    for disabled in [false, true, false] {
        Token::dangerously_disable_token_process_wide(disabled).unwrap();
        assert_eq!(Token::summon().is_none(), disabled);
        for &(bits, expected) in fixtures::F32 {
            let x = f32::from_bits(bits);
            let got = magetypes::simd::scalar::f32x1(x).sqrt().0;
            if f32::from_bits(expected).is_nan() {
                assert!(got.is_nan());
            } else {
                assert_eq!(
                    got.to_bits(),
                    expected,
                    "f32 input={bits:x} disabled={disabled}"
                );
            }
        }
        for &(bits, expected) in fixtures::F64 {
            let x = f64::from_bits(bits);
            let got = magetypes::simd::scalar::f64x1(x).sqrt().0;
            if f64::from_bits(expected).is_nan() {
                assert!(got.is_nan());
            } else {
                assert_eq!(
                    got.to_bits(),
                    expected,
                    "f64 input={bits:x} disabled={disabled}"
                );
            }
        }
    }
}

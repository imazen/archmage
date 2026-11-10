//! WASM availability stays compile-time; testing suppresses selection only.
#![cfg(target_arch = "wasm32")]
#![cfg(all(feature = "std", feature = "testable_dispatch"))]
#![forbid(unsafe_code)]

use archmage::{SimdToken, Wasm128RelaxedToken, Wasm128Token};

#[test]
fn wasm_permutations_and_retained_proofs() {
    use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
    let mut observed = Vec::new();
    let report = for_each_token_permutation(CompileTimePolicy::Fail, |permutation| {
        assert_eq!(Wasm128Token::compiled_with(), None);
        assert_eq!(Wasm128RelaxedToken::compiled_with(), None);
        let strict = Wasm128Token::summon().is_some();
        let relaxed = Wasm128RelaxedToken::summon().is_some();
        assert_eq!(
            strict,
            cfg!(target_feature = "simd128") && !Wasm128Token::manually_disabled().unwrap()
        );
        assert_eq!(
            relaxed,
            cfg!(all(
                target_feature = "simd128",
                target_feature = "relaxed-simd"
            )) && !Wasm128RelaxedToken::manually_disabled().unwrap()
        );
        assert!(!relaxed || strict, "invalid cascade at {permutation}");
        observed.push((strict, relaxed));
    });
    let expected = if cfg!(target_feature = "relaxed-simd") {
        vec![(true, true), (true, false), (false, false)]
    } else if cfg!(target_feature = "simd128") {
        vec![(true, false), (false, false)]
    } else {
        vec![(false, false)]
    };
    assert_eq!(report.permutations_run, expected.len());
    observed.sort_unstable();
    let mut expected = expected;
    expected.sort_unstable();
    assert_eq!(observed, expected);
    assert!(!Wasm128Token::manually_disabled().unwrap());
    assert!(!Wasm128RelaxedToken::manually_disabled().unwrap());

    #[cfg(target_feature = "simd128")]
    {
        let _guard = archmage::testing::lock_token_testing();
        let retained = Wasm128Token::summon().unwrap();
        Wasm128Token::dangerously_disable_token_process_wide(true).unwrap();
        assert!(Wasm128Token::summon().is_none());
        assert!(Wasm128RelaxedToken::summon().is_none());
        // Disabling later selection does not revoke proofs already held.
        assert_eq!(sqrt_with_proof(retained, 4.0), 2.0);
        Wasm128Token::dangerously_disable_token_process_wide(false).unwrap();
    }
}

#[cfg(target_feature = "simd128")]
#[archmage::arcane]
fn sqrt_with_proof(_token: Wasm128Token, x: f32) -> f32 {
    use core::arch::wasm32::*;
    f32x4_extract_lane::<0>(f32x4_sqrt(f32x4_splat(x)))
}

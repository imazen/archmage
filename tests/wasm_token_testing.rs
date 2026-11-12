//! WASM feature availability is compile-time (`cfg`); `testable_dispatch`
//! suppresses selection only when the underlying target features are enabled.
#![cfg(target_arch = "wasm32")]
#![cfg(feature = "std")]
#![forbid(unsafe_code)]

use archmage::{SimdToken, Wasm128RelaxedFmaToken, Wasm128RelaxedToken, Wasm128Token};

#[cfg(not(feature = "testable_dispatch"))]
#[test]
fn wasm_compile_time_cfg_detection_and_safe_from_context() {
    let has_simd128 = cfg!(target_feature = "simd128");
    let has_relaxed = cfg!(all(
        target_feature = "simd128",
        target_feature = "relaxed-simd"
    ));

    assert_eq!(Wasm128Token::compiled_with(), Some(has_simd128));
    assert_eq!(Wasm128RelaxedToken::compiled_with(), Some(has_relaxed));
    assert_eq!(
        Wasm128RelaxedFmaToken::compiled_with(),
        if has_relaxed { None } else { Some(false) }
    );

    assert_eq!(Wasm128Token::summon().is_some(), has_simd128);
    assert_eq!(Wasm128RelaxedToken::summon().is_some(), has_relaxed);
    if !has_relaxed {
        assert!(Wasm128RelaxedFmaToken::summon().is_none());
    }

    #[cfg(target_feature = "simd128")]
    {
        let tok = Wasm128Token::summon().unwrap();
        assert_eq!(call_strict_context(tok), 1);
    }

    #[cfg(all(target_feature = "simd128", target_feature = "relaxed-simd"))]
    {
        let tok = Wasm128RelaxedToken::summon().unwrap();
        assert_eq!(call_relaxed_context(tok), 2);
        if let Some(fma) = Wasm128RelaxedFmaToken::summon() {
            assert_eq!(call_fma_context(fma), 3);
        }
    }

    assert_eq!(
        archmage::attuned!(
            dispatch_tier_name(),
            [_wasm128_relaxed_fma_t, _wasm128_t, _scalar_t]
        ),
        if Wasm128RelaxedFmaToken::summon().is_some() {
            Wasm128RelaxedFmaToken::NAME
        } else if has_simd128 {
            Wasm128Token::NAME
        } else {
            archmage::ScalarToken::NAME
        }
    );
}

#[cfg(not(feature = "testable_dispatch"))]
#[archmage::attune(wasm128)]
fn strict_from_context_helper() -> u32 {
    let _tok = Wasm128Token::from_context();
    1
}

#[cfg(not(feature = "testable_dispatch"))]
#[archmage::attune(wrap)]
fn call_strict_context(_token: Wasm128Token) -> u32 {
    strict_from_context_helper()
}

#[cfg(not(feature = "testable_dispatch"))]
#[archmage::attune(wasm128_relaxed)]
fn relaxed_from_context_helper() -> u32 {
    let _tok = Wasm128RelaxedToken::from_context();
    let _base = Wasm128Token::from_context();
    2
}

#[cfg(not(feature = "testable_dispatch"))]
#[archmage::attune(wrap)]
fn call_relaxed_context(_token: Wasm128RelaxedToken) -> u32 {
    relaxed_from_context_helper()
}

#[cfg(not(feature = "testable_dispatch"))]
#[archmage::attune(wasm128_relaxed_fma)]
fn fma_from_context_helper() -> u32 {
    let _tok = Wasm128RelaxedFmaToken::from_context();
    let _relaxed = Wasm128RelaxedToken::from_context();
    let _base = Wasm128Token::from_context();
    3
}

#[cfg(not(feature = "testable_dispatch"))]
#[archmage::attune(wrap)]
fn call_fma_context(_token: Wasm128RelaxedFmaToken) -> u32 {
    fma_from_context_helper()
}

#[cfg(not(feature = "testable_dispatch"))]
#[archmage::attune(_*_t, +wasm128_relaxed_fma)]
fn dispatch_tier_name(_token: Token) -> &'static str {
    <Token as SimdToken>::NAME
}

#[cfg(feature = "testable_dispatch")]
#[test]
fn wasm_permutations_and_retained_proofs() {
    use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

    let has_simd128 = cfg!(target_feature = "simd128");
    let has_relaxed = cfg!(all(
        target_feature = "simd128",
        target_feature = "relaxed-simd"
    ));

    assert_eq!(
        Wasm128Token::compiled_with(),
        if has_simd128 { None } else { Some(false) }
    );
    assert_eq!(
        Wasm128RelaxedToken::compiled_with(),
        if has_relaxed { None } else { Some(false) }
    );
    assert_eq!(
        Wasm128RelaxedFmaToken::compiled_with(),
        if has_relaxed { None } else { Some(false) }
    );

    let engine_fuses = Wasm128RelaxedFmaToken::summon().is_some();

    let mut observed = Vec::new();
    let report = for_each_token_permutation(CompileTimePolicy::Fail, |permutation| {
        let strict = Wasm128Token::summon().is_some();
        let relaxed = Wasm128RelaxedToken::summon().is_some();
        let fma = Wasm128RelaxedFmaToken::summon().is_some();
        assert_eq!(
            strict,
            has_simd128 && !Wasm128Token::manually_disabled().unwrap()
        );
        assert_eq!(
            relaxed,
            has_relaxed && !Wasm128RelaxedToken::manually_disabled().unwrap()
        );
        assert_eq!(
            fma,
            engine_fuses && !Wasm128RelaxedFmaToken::manually_disabled().unwrap()
        );
        assert!(!relaxed || strict, "invalid cascade at {permutation}");
        assert!(!fma || relaxed, "invalid FMA cascade at {permutation}");
        observed.push((strict, relaxed, fma));
    });

    let expected = if engine_fuses {
        vec![
            (true, true, true),
            (true, true, false),
            (true, false, false),
            (false, false, false),
        ]
    } else if has_relaxed {
        vec![
            (true, true, false),
            (true, false, false),
            (false, false, false),
        ]
    } else if has_simd128 {
        vec![(true, false, false), (false, false, false)]
    } else {
        vec![(false, false, false)]
    };
    assert_eq!(report.permutations_run, expected.len());
    observed.sort_unstable();
    let mut expected = expected;
    expected.sort_unstable();
    assert_eq!(observed, expected);
    assert!(!Wasm128Token::manually_disabled().unwrap());
    assert!(!Wasm128RelaxedToken::manually_disabled().unwrap());
    assert!(!Wasm128RelaxedFmaToken::manually_disabled().unwrap());

    #[cfg(target_feature = "simd128")]
    {
        let _guard = archmage::testing::lock_token_testing();
        let retained = Wasm128Token::summon().unwrap();
        Wasm128Token::dangerously_disable_token_process_wide(true).unwrap();
        assert!(Wasm128Token::summon().is_none());
        assert!(Wasm128RelaxedToken::summon().is_none());
        assert!(Wasm128RelaxedFmaToken::summon().is_none());
        // Disabling later selection does not revoke proofs already held.
        assert_eq!(sqrt_with_proof(retained, 4.0), 2.0);
        Wasm128Token::dangerously_disable_token_process_wide(false).unwrap();
    }
}

#[cfg(feature = "testable_dispatch")]
#[archmage::arcane]
fn sqrt_with_proof(_token: Wasm128Token, x: f32) -> f32 {
    use core::arch::wasm32::*;
    f32x4_extract_lane::<0>(f32x4_sqrt(f32x4_splat(x)))
}

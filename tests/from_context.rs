//! Compiler-checked token construction from an existing target-feature region.

#[test]
fn scalar_needs_no_features() {
    assert_eq!(archmage::ScalarToken::from_context(), archmage::ScalarToken);
}

#[cfg(target_arch = "x86_64")]
mod native {
    use archmage::{SimdToken, X64V1Token, X64V2Token, X64V3Token, arcane};

    // Explicit attribute, independent of arcane's feature generation.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    fn exact_context() -> X64V3Token {
        X64V3Token::from_context()
    }

    #[arcane]
    fn enter_context(_token: X64V3Token) {
        let _: X64V3Token = exact_context();
        // Supersets can create lower-tier proofs too, including SSE2 baseline.
        let _: X64V2Token = X64V2Token::from_context();
        let _: X64V1Token = X64V1Token::from_context();
    }

    #[test]
    fn matching_and_superset_contexts() {
        if let Some(token) = X64V3Token::summon() {
            enter_context(token);
        }
    }

    #[test]
    fn rejects_unproven_contexts() {
        let tests = trybuild::TestCases::new();
        for case in ["missing", "weaker", "pointer"] {
            tests.compile_fail(format!("tests/compile_fail/from_context_{case}.rs"));
        }
        // Enabling the public forge API changes rustc's constructor suggestions.
        // The generator test covers stub omission under every feature set.
        if !cfg!(feature = "forge-token-api") {
            tests.compile_fail("tests/compile_fail/from_context_wrong_arch.rs");
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod arm {
    use archmage::{Arm64V2Token, NeonToken, SimdToken, arcane};

    #[target_feature(enable = "neon")]
    fn exact_context() -> NeonToken {
        NeonToken::from_context()
    }

    #[arcane]
    fn enter_context(_token: Arm64V2Token) {
        let _: Arm64V2Token = Arm64V2Token::from_context();
        let _: NeonToken = exact_context();
    }

    #[test]
    fn matching_and_superset_contexts() {
        if let Some(token) = Arm64V2Token::summon() {
            enter_context(token);
        }
    }
}

#[cfg(target_arch = "wasm32")]
#[test]
fn wasm_allows_calls_from_any_context() {
    // WASM's target-feature call safety rules intentionally differ from native.
    let _: archmage::Wasm128Token = archmage::Wasm128Token::from_context();
    let _: archmage::Wasm128RelaxedToken = archmage::Wasm128RelaxedToken::from_context();
}

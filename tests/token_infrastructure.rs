//! Comprehensive tests for token infrastructure.
//!
//! Tests the token system: compiled_with(), summon(), disable mechanism,
//! cascading, CompileTimeGuaranteedError, deprecated aliases, IntoConcreteToken,
//! token extraction (downcast), and public API re-exports.
//!
//! Most tests here work on x86_64 using real tokens. Stub behavior is tested
//! via the ARM/WASM tokens which are stubs on x86_64.

use archmage::{
    // Aliases
    Arm64,
    CompileTimeGuaranteedError,
    Desktop64,
    IntoConcreteToken,
    // ARM tokens (stubs on x86_64)
    NeonAesToken,
    NeonCrcToken,
    NeonSha3Token,
    NeonToken,
    ScalarToken,
    SimdToken,
    // WASM tokens (stubs on x86_64)
    Wasm128Token,
    // x86 tokens
    X64V1Token,
    X64V2Token,
    X64V3Token,
};

#[cfg(feature = "avx512")]
use archmage::{Avx512Fp16Token, X64V4xToken, Avx512Token, Server64, X64V4Token};

// ============================================================================
// ScalarToken: always available
// ============================================================================

#[test]
fn scalar_compiled_with_always_true() {
    assert_eq!(ScalarToken::compiled_with(), Some(true));
}

#[test]
fn scalar_summon_always_some() {
    assert!(ScalarToken::summon().is_some());
}

#[test]
fn scalar_disable_always_errors() {
    let result = ScalarToken::dangerously_disable_token_process_wide(true);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.token_name, "Scalar");
}

#[test]
fn scalar_manually_disabled_always_errors() {
    let result = ScalarToken::manually_disabled();
    assert!(result.is_err());
}

// ============================================================================
// Stub tokens: ARM and WASM on x86_64
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[test]
fn stub_compiled_with_returns_some_false() {
    // ARM tokens are stubs on x86_64 → compiled_with() = Some(false)
    assert_eq!(NeonToken::compiled_with(), Some(false));
    assert_eq!(NeonAesToken::compiled_with(), Some(false));
    assert_eq!(NeonSha3Token::compiled_with(), Some(false));
    assert_eq!(NeonCrcToken::compiled_with(), Some(false));

    // WASM token is stub on x86_64
    assert_eq!(Wasm128Token::compiled_with(), Some(false));
}

#[cfg(target_arch = "x86_64")]
#[test]
fn stub_summon_returns_none() {
    assert!(NeonToken::summon().is_none());
    assert!(NeonAesToken::summon().is_none());
    assert!(NeonSha3Token::summon().is_none());
    assert!(NeonCrcToken::summon().is_none());
    assert!(Wasm128Token::summon().is_none());
}

#[cfg(target_arch = "x86_64")]
#[test]
fn stub_disable_returns_err() {
    // Stubs can't be disabled — returns CompileTimeGuaranteedError
    assert!(NeonToken::dangerously_disable_token_process_wide(true).is_err());
    assert!(NeonAesToken::dangerously_disable_token_process_wide(true).is_err());
    assert!(NeonSha3Token::dangerously_disable_token_process_wide(true).is_err());
    assert!(NeonCrcToken::dangerously_disable_token_process_wide(true).is_err());
    assert!(Wasm128Token::dangerously_disable_token_process_wide(true).is_err());
}

#[cfg(target_arch = "x86_64")]
#[test]
fn stub_manually_disabled_returns_err() {
    assert!(NeonToken::manually_disabled().is_err());
    assert!(NeonAesToken::manually_disabled().is_err());
    assert!(NeonSha3Token::manually_disabled().is_err());
    assert!(NeonCrcToken::manually_disabled().is_err());
    assert!(Wasm128Token::manually_disabled().is_err());
}

// ============================================================================
// x86 tokens: compiled_with() and summon() basics
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[test]
fn x86_compiled_with_returns_option() {
    // On x86_64, compiled_with() returns either Some(true) or None,
    // depending on whether the target features are compile-time enabled.
    let v2 = X64V2Token::compiled_with();
    assert!(v2 == Some(true) || v2.is_none());

    let v3 = X64V3Token::compiled_with();
    assert!(v3 == Some(true) || v3.is_none());
}

#[test]
fn x86_summon_returns_option() {
    // On a modern x86_64 CPU, V2 should almost always succeed.
    // We don't assert Some because some CI environments may be odd.
    let _v2 = X64V2Token::summon();
    let _v3 = X64V3Token::summon();
}

// ============================================================================
// Disable mechanism (process-wide)
// ============================================================================

// NOTE: These tests mutate global state (atomic flags). They MUST be run
// with `--test-threads=1` or in isolation. cargo test does NOT do this by
// default, but the individual tests clean up after themselves.

#[cfg(target_arch = "x86_64")]
#[test]
fn disable_v3_makes_summon_return_none() {
    // Skip if compiled with target features (disable returns Err)
    if X64V3Token::compiled_with() == Some(true) {
        return;
    }

    // Save initial state
    let was_available = X64V3Token::summon().is_some();

    // Disable
    let result = X64V3Token::dangerously_disable_token_process_wide(true);
    assert!(
        result.is_ok(),
        "disable should succeed when not compiled with features"
    );
    assert!(
        X64V3Token::summon().is_none(),
        "summon() should return None when disabled"
    );

    // manually_disabled() should reflect the state
    assert!(
        X64V3Token::manually_disabled().unwrap(),
        "manually_disabled should be true"
    );

    // Re-enable
    let result = X64V3Token::dangerously_disable_token_process_wide(false);
    assert!(result.is_ok());
    assert!(
        !X64V3Token::manually_disabled().unwrap(),
        "manually_disabled should be false after re-enable"
    );

    // After re-enable, summon() should be able to detect again
    // (cache was reset to 0 = unknown)
    if was_available {
        assert!(
            X64V3Token::summon().is_some(),
            "summon() should detect features again after re-enable"
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn disable_v2_makes_summon_return_none() {
    if X64V2Token::compiled_with() == Some(true) {
        return;
    }

    let was_available = X64V2Token::summon().is_some();

    let result = X64V2Token::dangerously_disable_token_process_wide(true);
    assert!(result.is_ok());
    assert!(X64V2Token::summon().is_none());
    assert!(X64V2Token::manually_disabled().unwrap());

    // Re-enable
    X64V2Token::dangerously_disable_token_process_wide(false).unwrap();
    assert!(!X64V2Token::manually_disabled().unwrap());

    if was_available {
        assert!(X64V2Token::summon().is_some());
    }
}

#[test]
fn default_is_not_disabled() {
    // Fresh process: manually_disabled() should be false (or Err for stubs/compiled)
    if let Ok(disabled) = X64V2Token::manually_disabled() {
        assert!(!disabled, "default should not be disabled");
    } // compiled_with or stub — that's fine
    if let Ok(disabled) = X64V3Token::manually_disabled() {
        assert!(!disabled);
    }
}

// ============================================================================
// Cascading: disabling parent affects children
// ============================================================================

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[test]
fn disable_v3_cascades_to_v4() {
    if X64V3Token::compiled_with() == Some(true) {
        return;
    }

    // Record whether V4 was available before
    let v4_was_available = X64V4Token::summon().is_some();

    // Disable V3 — should cascade to V4, V4x, Fp16
    X64V3Token::dangerously_disable_token_process_wide(true).unwrap();

    assert!(X64V3Token::summon().is_none(), "V3 should be disabled");
    assert!(
        X64V4Token::summon().is_none(),
        "V4 should be disabled (cascade from V3)"
    );
    assert!(
        X64V4xToken::summon().is_none(),
        "V4x should be disabled (cascade from V3)"
    );
    assert!(
        Avx512Fp16Token::summon().is_none(),
        "Fp16 should be disabled (cascade from V3)"
    );

    // Re-enable V3 — should also re-enable cascaded children
    X64V3Token::dangerously_disable_token_process_wide(false).unwrap();

    // V3 should be detectable again
    if X64V3Token::summon().is_some() {
        // V4 should also be detectable again (if CPU supports it)
        if v4_was_available {
            assert!(
                X64V4Token::summon().is_some(),
                "V4 should be re-detectable after parent re-enabled"
            );
        }
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[test]
fn disable_v2_cascades_to_v3_v4() {
    if X64V2Token::compiled_with() == Some(true) {
        return;
    }

    X64V2Token::dangerously_disable_token_process_wide(true).unwrap();

    assert!(X64V2Token::summon().is_none());
    assert!(
        X64V3Token::summon().is_none(),
        "V3 should be disabled (cascade from V2)"
    );
    assert!(
        X64V4Token::summon().is_none(),
        "V4 should be disabled (cascade from V2)"
    );

    // Re-enable
    X64V2Token::dangerously_disable_token_process_wide(false).unwrap();
}

#[cfg(feature = "avx512")]
#[test]
fn disable_child_does_not_affect_parent() {
    if X64V4Token::compiled_with() == Some(true) || X64V3Token::compiled_with() == Some(true) {
        return;
    }

    let v3_was_available = X64V3Token::summon().is_some();

    // Disable V4 — should NOT affect V3
    if X64V4Token::dangerously_disable_token_process_wide(true).is_ok() {
        assert!(X64V4Token::summon().is_none(), "V4 should be disabled");

        if v3_was_available {
            assert!(
                X64V3Token::summon().is_some(),
                "V3 should NOT be affected by disabling V4"
            );
        }

        // Cleanup
        X64V4Token::dangerously_disable_token_process_wide(false).unwrap();
    }
}

// ============================================================================
// CompileTimeGuaranteedError
// ============================================================================

#[test]
fn compile_time_guaranteed_error_display_with_features() {
    let err = CompileTimeGuaranteedError {
        token_name: "X64V3Token",
        target_features: "avx2,fma,bmi1,bmi2,f16c,lzcnt",
        disable_flags: "-Ctarget-feature=-avx2,-fma,-bmi1,-bmi2,-f16c,-lzcnt",
    };
    let msg = format!("{err}");
    assert!(
        msg.contains("X64V3Token"),
        "error message should contain token name"
    );
    assert!(
        msg.contains("avx2,fma"),
        "error message should contain feature list"
    );
    assert!(
        msg.contains("-Ctarget-feature=-avx2"),
        "error message should contain disable flags"
    );
    assert!(
        msg.contains("compile-time"),
        "error message should mention compile-time"
    );
    assert!(
        msg.contains("target-cpu"),
        "error message should mention target-cpu"
    );
}

#[test]
fn compile_time_guaranteed_error_display_empty_features() {
    let err = CompileTimeGuaranteedError {
        token_name: "Scalar",
        target_features: "",
        disable_flags: "",
    };
    let msg = format!("{err}");
    assert!(msg.contains("Scalar"));
    assert!(
        msg.contains("always available"),
        "scalar error should say always available"
    );
}

#[test]
fn compile_time_guaranteed_error_debug() {
    let err = CompileTimeGuaranteedError {
        token_name: "X64V3Token",
        target_features: "avx2,fma",
        disable_flags: "-Ctarget-feature=-avx2,-fma",
    };
    let debug = format!("{err:?}");
    assert!(debug.contains("X64V3Token"));
}

#[cfg(feature = "std")]
#[test]
fn compile_time_guaranteed_error_is_std_error() {
    let err = CompileTimeGuaranteedError {
        token_name: "TestToken",
        target_features: "avx2",
        disable_flags: "-Ctarget-feature=-avx2",
    };
    // Verify it implements std::error::Error
    let _: &dyn std::error::Error = &err;
}

// ============================================================================
// Deprecated aliases
// ============================================================================

#[test]
#[allow(deprecated)]
fn guaranteed_delegates_to_compiled_with() {
    assert_eq!(ScalarToken::guaranteed(), ScalarToken::compiled_with());
    assert_eq!(X64V2Token::guaranteed(), X64V2Token::compiled_with());
    assert_eq!(X64V3Token::guaranteed(), X64V3Token::compiled_with());
    assert_eq!(NeonToken::guaranteed(), NeonToken::compiled_with());
    assert_eq!(Wasm128Token::guaranteed(), Wasm128Token::compiled_with());
}

#[test]
fn try_new_delegates_to_summon() {
    #[allow(deprecated)]
    {
        assert_eq!(
            ScalarToken::try_new().is_some(),
            ScalarToken::summon().is_some()
        );
        assert_eq!(
            X64V2Token::try_new().is_some(),
            X64V2Token::summon().is_some()
        );
        assert_eq!(
            X64V3Token::try_new().is_some(),
            X64V3Token::summon().is_some()
        );
        assert_eq!(
            NeonToken::try_new().is_some(),
            NeonToken::summon().is_some()
        );
    }
}

#[test]
fn attempt_delegates_to_summon() {
    assert_eq!(
        ScalarToken::attempt().is_some(),
        ScalarToken::summon().is_some()
    );
    assert_eq!(
        X64V2Token::attempt().is_some(),
        X64V2Token::summon().is_some()
    );
    assert_eq!(
        X64V3Token::attempt().is_some(),
        X64V3Token::summon().is_some()
    );
}

// ============================================================================
// IntoConcreteToken
// ============================================================================

#[test]
fn scalar_into_concrete_token() {
    let token = ScalarToken;
    assert!(token.as_scalar().is_some());
    assert!(token.as_x64v2().is_none());
    assert!(token.as_x64v3().is_none());
    assert!(token.as_neon().is_none());
    assert!(token.as_wasm128().is_none());
}

#[test]
fn v2_into_concrete_token() {
    if let Some(token) = X64V2Token::summon() {
        assert!(token.as_x64v2().is_some());
        assert!(token.as_x64v3().is_none());
        assert!(token.as_neon().is_none());
        assert!(token.as_scalar().is_none());
    }
}

#[test]
fn v3_into_concrete_token() {
    if let Some(token) = X64V3Token::summon() {
        assert!(token.as_x64v3().is_some());
        assert!(token.as_x64v2().is_none(), "V3 is not V2 (different type)");
        assert!(token.as_neon().is_none());
        assert!(token.as_wasm128().is_none());
    }
}

#[cfg(feature = "avx512")]
#[test]
fn v4_into_concrete_token() {
    if let Some(token) = X64V4Token::summon() {
        assert!(token.as_x64v4().is_some());
        assert!(token.as_x64v3().is_none());
        assert!(token.as_neon().is_none());
    }
}

// ============================================================================
// Token extraction (downcast: higher → lower)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[test]
fn v3_extracts_to_v2() {
    if let Some(token) = X64V3Token::summon() {
        let _v2: X64V2Token = token.v2();
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[test]
fn v4_extracts_to_v3_and_v2() {
    if let Some(token) = X64V4Token::summon() {
        let _v3: X64V3Token = token.v3();
        let _v2: X64V2Token = token.v2();
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[test]
fn x64v4x_extracts_to_v4_v3_v2() {
    if let Some(token) = X64V4xToken::summon() {
        let _v4: X64V4Token = token.v4();
        let _avx512: X64V4Token = token.avx512();
        let _v3: X64V3Token = token.v3();
        let _v2: X64V2Token = token.v2();
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[test]
fn avx512fp16_extracts_to_v4_v3_v2() {
    if let Some(token) = Avx512Fp16Token::summon() {
        let _v4: X64V4Token = token.v4();
        let _avx512: X64V4Token = token.avx512();
        let _v3: X64V3Token = token.v3();
        let _v2: X64V2Token = token.v2();
    }
}

// ============================================================================
// Public API re-exports (type aliases)
// ============================================================================

#[test]
fn desktop64_is_x64v3() {
    // Desktop64 is a type alias for X64V3Token
    fn _takes_desktop64(_t: Desktop64) {}
    fn _takes_x64v3(t: X64V3Token) {
        _takes_desktop64(t);
    }

    // They should have the same compiled_with() and summon()
    assert_eq!(Desktop64::compiled_with(), X64V3Token::compiled_with());
    assert_eq!(
        Desktop64::summon().is_some(),
        X64V3Token::summon().is_some()
    );
}

#[cfg(feature = "avx512")]
#[test]
fn server64_is_x64v4() {
    assert_eq!(Server64::compiled_with(), X64V4Token::compiled_with());
    assert_eq!(Server64::summon().is_some(), X64V4Token::summon().is_some());
}

#[cfg(feature = "avx512")]
#[test]
fn avx512token_is_x64v4() {
    assert_eq!(Avx512Token::compiled_with(), X64V4Token::compiled_with());
    assert_eq!(
        Avx512Token::summon().is_some(),
        X64V4Token::summon().is_some()
    );
}

#[test]
fn arm64_is_neon() {
    assert_eq!(Arm64::compiled_with(), NeonToken::compiled_with());
    assert_eq!(Arm64::summon().is_some(), NeonToken::summon().is_some());
}

// ============================================================================
// SimdToken::NAME constants
// ============================================================================

#[test]
fn token_names_are_nonempty() {
    assert!(!X64V2Token::NAME.is_empty());
    assert!(!X64V3Token::NAME.is_empty());
    assert!(!NeonToken::NAME.is_empty());
    assert!(!Wasm128Token::NAME.is_empty());
    assert!(!ScalarToken::NAME.is_empty());
}

#[cfg(feature = "avx512")]
#[test]
fn avx512_token_names_are_nonempty() {
    assert!(!X64V4Token::NAME.is_empty());
    assert!(!X64V4xToken::NAME.is_empty());
    assert!(!Avx512Fp16Token::NAME.is_empty());
}

// ============================================================================
// Token traits: Copy, Clone, Send, Sync
// ============================================================================

#[test]
fn tokens_are_copy_clone_send_sync() {
    fn assert_token_traits<T: Copy + Clone + Send + Sync + 'static>() {}

    assert_token_traits::<ScalarToken>();
    assert_token_traits::<X64V2Token>();
    assert_token_traits::<X64V3Token>();
    assert_token_traits::<NeonToken>();
    assert_token_traits::<NeonAesToken>();
    assert_token_traits::<NeonSha3Token>();
    assert_token_traits::<NeonCrcToken>();
    assert_token_traits::<Wasm128Token>();
}

#[cfg(feature = "avx512")]
#[test]
fn avx512_tokens_are_copy_clone_send_sync() {
    fn assert_token_traits<T: Copy + Clone + Send + Sync + 'static>() {}

    assert_token_traits::<X64V4Token>();
    assert_token_traits::<X64V4xToken>();
    assert_token_traits::<Avx512Fp16Token>();
}

// ============================================================================
// Summon caching: second call should be fast
// ============================================================================

#[test]
fn summon_is_idempotent() {
    // Multiple calls should return the same result (caching should work)
    let a = X64V3Token::summon().is_some();
    let b = X64V3Token::summon().is_some();
    let c = X64V3Token::summon().is_some();
    assert_eq!(a, b);
    assert_eq!(b, c);
}

#[test]
fn summon_v2_is_idempotent() {
    let a = X64V2Token::summon().is_some();
    let b = X64V2Token::summon().is_some();
    assert_eq!(a, b);
}

// ============================================================================
// Feature flag strings (TARGET_FEATURES, ENABLE/DISABLE_TARGET_FEATURES)
// ============================================================================

#[test]
fn scalar_feature_strings_are_empty() {
    assert_eq!(ScalarToken::TARGET_FEATURES, "");
    assert_eq!(ScalarToken::ENABLE_TARGET_FEATURES, "");
    assert_eq!(ScalarToken::DISABLE_TARGET_FEATURES, "");
}

#[test]
fn x64v1_target_features_content() {
    let features = X64V1Token::TARGET_FEATURES;
    assert!(features.contains("sse"), "V1 features should contain sse");
    assert!(features.contains("sse2"), "V1 features should contain sse2");
}

#[test]
fn x64v2_target_features_content() {
    let features = X64V2Token::TARGET_FEATURES;
    // Must contain sse/sse2 (baseline) and v2-specific features
    assert!(features.contains("sse,"), "V2 features should contain sse");
    assert!(features.contains("sse2"), "V2 features should contain sse2");
    assert!(features.contains("sse3"), "V2 features should contain sse3");
    assert!(
        features.contains("ssse3"),
        "V2 features should contain ssse3"
    );
    assert!(
        features.contains("sse4.1"),
        "V2 features should contain sse4.1"
    );
    assert!(
        features.contains("sse4.2"),
        "V2 features should contain sse4.2"
    );
    assert!(
        features.contains("popcnt"),
        "V2 features should contain popcnt"
    );
}

#[test]
fn x64v3_target_features_content() {
    let features = X64V3Token::TARGET_FEATURES;
    assert!(features.contains("avx2"), "V3 features should contain avx2");
    assert!(features.contains("fma"), "V3 features should contain fma");
    assert!(features.contains("bmi1"), "V3 features should contain bmi1");
    assert!(features.contains("bmi2"), "V3 features should contain bmi2");
    assert!(features.contains("f16c"), "V3 features should contain f16c");
    assert!(
        features.contains("lzcnt"),
        "V3 features should contain lzcnt"
    );
}

#[test]
fn enable_flags_format() {
    // All non-scalar tokens should have proper ENABLE format
    let enable = X64V3Token::ENABLE_TARGET_FEATURES;
    assert!(
        enable.starts_with("-Ctarget-feature=+"),
        "enable flags should start with -Ctarget-feature=+, got: {enable}"
    );
    assert!(
        enable.contains("+avx2"),
        "enable flags should contain +avx2"
    );
    assert!(enable.contains("+fma"), "enable flags should contain +fma");
}

#[test]
fn disable_flags_format() {
    let disable = X64V3Token::DISABLE_TARGET_FEATURES;
    assert!(
        disable.starts_with("-Ctarget-feature=-"),
        "disable flags should start with -Ctarget-feature=-, got: {disable}"
    );
    assert!(
        disable.contains("-avx2"),
        "disable flags should contain -avx2"
    );
    assert!(
        disable.contains("-fma"),
        "disable flags should contain -fma"
    );
}

#[test]
fn stub_tokens_have_same_feature_strings_as_real() {
    // ARM tokens are stubs on x86_64, but should still carry feature strings
    assert!(
        !NeonToken::TARGET_FEATURES.is_empty(),
        "NeonToken stub should have non-empty TARGET_FEATURES"
    );
    assert!(
        NeonToken::TARGET_FEATURES.contains("neon"),
        "NeonToken stub should list neon"
    );
    assert!(
        NeonToken::ENABLE_TARGET_FEATURES.contains("+neon"),
        "NeonToken stub should have enable flags"
    );
    assert!(
        NeonToken::DISABLE_TARGET_FEATURES.contains("-neon"),
        "NeonToken stub should have disable flags"
    );

    // WASM token
    assert!(
        Wasm128Token::TARGET_FEATURES.contains("simd128"),
        "Wasm128Token stub should list simd128"
    );
}

#[cfg(feature = "avx512")]
#[test]
fn avx512_feature_strings() {
    let features = X64V4Token::TARGET_FEATURES;
    assert!(features.contains("avx512f"));
    assert!(features.contains("avx512bw"));
    assert!(features.contains("avx512cd"));
    assert!(features.contains("avx512dq"));
    assert!(features.contains("avx512vl"));

    let v4x_features = X64V4xToken::TARGET_FEATURES;
    assert!(v4x_features.contains("avx512vpopcntdq"));
    assert!(v4x_features.contains("gfni"));
    assert!(v4x_features.contains("vaes"));
}

#[test]
// ============================================================================
// testable_dispatch feature flag
// ============================================================================
#[cfg(feature = "testable_dispatch")]
#[test]
fn dct_compiled_with_returns_none() {
    // With testable_dispatch, compiled_with() should return None
    // even though we're on x86_64 and these features ARE compile-time available.
    assert_eq!(
        X64V2Token::compiled_with(),
        None,
        "V2 compiled_with() should be None with testable_dispatch"
    );
    assert_eq!(
        X64V3Token::compiled_with(),
        None,
        "V3 compiled_with() should be None with testable_dispatch"
    );
}

#[cfg(feature = "testable_dispatch")]
#[test]
fn dct_disable_returns_ok() {
    // With testable_dispatch, disable should succeed even when
    // compiled with -Ctarget-cpu=native.
    let result = X64V3Token::dangerously_disable_token_process_wide(true);
    assert!(
        result.is_ok(),
        "disable should succeed with testable_dispatch"
    );
    assert!(
        X64V3Token::summon().is_none(),
        "summon should return None after disable"
    );

    // Re-enable
    X64V3Token::dangerously_disable_token_process_wide(false).unwrap();
    assert!(
        X64V3Token::summon().is_some(),
        "summon should succeed after re-enable"
    );
}

#[test]
fn disable_all_tokens_except_wasm() {
    use archmage::dangerously_disable_tokens_except_wasm;

    let result = dangerously_disable_tokens_except_wasm(true);

    // On x86 without -Ctarget-cpu, this should succeed
    #[cfg(all(
        any(target_arch = "x86_64", target_arch = "x86"),
        not(all(
            target_feature = "sse3",
            target_feature = "ssse3",
            target_feature = "sse4.1",
            target_feature = "sse4.2",
            target_feature = "popcnt",
            not(feature = "testable_dispatch")
        ))
    ))]
    {
        assert!(result.is_ok(), "disable_all should succeed: {result:?}");
        assert!(
            X64V2Token::summon().is_none(),
            "V2 summon should be None after disable_all"
        );
        assert!(
            X64V3Token::summon().is_none(),
            "V3 summon should be None after disable_all"
        );
        #[cfg(feature = "avx512")]
        assert!(
            archmage::X64V4Token::summon().is_none(),
            "V4 summon should be None after disable_all"
        );

        // Re-enable
        let result = dangerously_disable_tokens_except_wasm(false);
        assert!(result.is_ok(), "re-enable should succeed: {result:?}");
        // Tokens that were previously available should come back
        // (can't assert summon().is_some() since CPU may not actually support them)
    }

    // If compile-time enabled, we get errors
    #[cfg(all(
        any(target_arch = "x86_64", target_arch = "x86"),
        all(
            target_feature = "sse3",
            target_feature = "ssse3",
            target_feature = "sse4.1",
            target_feature = "sse4.2",
            target_feature = "popcnt",
            not(feature = "testable_dispatch")
        )
    ))]
    {
        let err = result.unwrap_err();
        assert!(
            !err.errors.is_empty(),
            "should have at least one error for compile-time-enabled tokens"
        );
        assert!(
            err.to_string().contains("Failed to disable"),
            "Display should mention failure"
        );
    }
}

#[test]
fn compile_time_error_from_disable_contains_features() {
    // ScalarToken always returns Err with empty features
    let err = ScalarToken::dangerously_disable_token_process_wide(true).unwrap_err();
    assert_eq!(err.target_features, "");
    assert_eq!(err.disable_flags, "");

    // Stub tokens also return Err with populated features
    let err = NeonToken::dangerously_disable_token_process_wide(true).unwrap_err();
    assert!(
        err.target_features.contains("neon"),
        "neon stub error should have neon in target_features"
    );
    assert!(
        err.disable_flags.contains("-neon"),
        "neon stub error should have -neon in disable_flags"
    );
}

// ============================================================================
// Coverage: forge_token_dangerously + IntoConcreteToken for stubs
// ============================================================================

#[allow(deprecated)]
#[test]
fn stub_forge_and_into_concrete_token() {
    // ARM/WASM stubs exist on x86_64 and can be forged.
    // Their IntoConcreteToken impls should return Some(self) for their own type.
    unsafe {
        let neon = NeonToken::forge_token_dangerously();
        assert!(neon.as_neon().is_some());
        assert!(neon.as_x64v3().is_none());

        let neon_aes = NeonAesToken::forge_token_dangerously();
        assert!(neon_aes.as_neon_aes().is_some());

        let neon_sha3 = NeonSha3Token::forge_token_dangerously();
        assert!(neon_sha3.as_neon_sha3().is_some());

        let neon_crc = NeonCrcToken::forge_token_dangerously();
        assert!(neon_crc.as_neon_crc().is_some());

        let wasm = Wasm128Token::forge_token_dangerously();
        assert!(wasm.as_wasm128().is_some());

        let scalar = ScalarToken::forge_token_dangerously();
        assert!(scalar.as_scalar().is_some());
    }
}

#[cfg(feature = "avx512")]
#[allow(deprecated)]
#[test]
fn avx512fp16_forge_and_into_concrete_token() {
    unsafe {
        let fp16 = Avx512Fp16Token::forge_token_dangerously();
        assert!(fp16.as_avx512_fp16().is_some());
        assert!(fp16.as_x64v3().is_none());
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[allow(deprecated)]
#[test]
fn avx512fp16_forge_and_downcast() {
    // Test the downcast extraction methods on a forged Avx512Fp16Token.
    unsafe {
        let fp16 = Avx512Fp16Token::forge_token_dangerously();
        let _v4: X64V4Token = fp16.v4();
        let _avx512: X64V4Token = fp16.avx512();
        let _v3: X64V3Token = fp16.v3();
        let _v2: X64V2Token = fp16.v2();
    }
}

// ============================================================================
// Coverage: DisableAllSimdError Display
// ============================================================================

#[test]
fn disable_all_simd_error_display() {
    use archmage::DisableAllSimdError;
    let err = DisableAllSimdError {
        errors: vec![
            CompileTimeGuaranteedError {
                token_name: "X64V3Token",
                target_features: "avx2,fma",
                disable_flags: "-Ctarget-feature=-avx2,-fma",
            },
            CompileTimeGuaranteedError {
                token_name: "X64V2Token",
                target_features: "sse4.2,popcnt",
                disable_flags: "-Ctarget-feature=-sse4.2,-popcnt",
            },
        ],
    };
    let msg = format!("{err}");
    assert!(msg.contains("2"), "should mention error count");
    assert!(msg.contains("X64V3Token"), "should list first token");
    assert!(msg.contains("X64V2Token"), "should list second token");
}

// ============================================================================
// Coverage: X64V4xToken + Avx512Fp16Token disable/manually_disabled
// ============================================================================

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[test]
fn x64v4x_disable_and_manually_disabled() {
    if X64V4xToken::compiled_with() == Some(true) {
        return;
    }
    let result = X64V4xToken::dangerously_disable_token_process_wide(true);
    assert!(result.is_ok());
    assert!(X64V4xToken::manually_disabled().unwrap());

    X64V4xToken::dangerously_disable_token_process_wide(false).unwrap();
    assert!(!X64V4xToken::manually_disabled().unwrap());
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[test]
fn avx512fp16_disable_and_manually_disabled() {
    if Avx512Fp16Token::compiled_with() == Some(true) {
        return;
    }
    let result = Avx512Fp16Token::dangerously_disable_token_process_wide(true);
    assert!(result.is_ok());
    assert!(Avx512Fp16Token::manually_disabled().unwrap());

    Avx512Fp16Token::dangerously_disable_token_process_wide(false).unwrap();
    assert!(!Avx512Fp16Token::manually_disabled().unwrap());
}

#[cfg(feature = "avx512")]
#[test]
fn v4_manually_disabled_default() {
    if let Ok(disabled) = X64V4Token::manually_disabled() {
        assert!(!disabled, "default should not be disabled");
    } // compiled_with — that's fine
}

// ============================================================================
// Coverage: detect.rs convenience functions
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[test]
fn detect_convenience_functions() {
    // These are asm-inspection helpers; just verify they return bools.
    let _avx2 = archmage::detect::check_avx2_available();
    let _fma = archmage::detect::check_fma_available();
    let _avx512 = archmage::detect::check_avx512f_available();
}

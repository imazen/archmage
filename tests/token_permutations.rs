//! Tests for the token permutation test helper.

use archmage::SimdToken;
use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

/// Verify that permutations run and the "all enabled" state is always included.
#[test]
fn permutations_include_all_enabled() {
    let mut saw_all_enabled = false;
    let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        if perm.disabled.is_empty() {
            saw_all_enabled = true;
        }
    });
    assert!(
        saw_all_enabled,
        "should always include 'all enabled' permutation"
    );
    assert!(
        report.permutations_run >= 1,
        "should run at least the 'all enabled' permutation"
    );
}

/// On x86_64, if V2 and V3 are runtime-detected (not compile-time), we should
/// get at least 3 permutations: all enabled, V3 disabled, V2+V3 disabled.
#[cfg(target_arch = "x86_64")]
#[test]
fn x86_has_multiple_permutations() {
    let report = for_each_token_permutation(CompileTimePolicy::Warn, |_perm| {
        // Just count permutations
    });
    // Even if V2 is compile-time guaranteed, V3 should give us at least 2
    // (unless both are compile-time guaranteed, in which case warnings tell us)
    if report.warnings.is_empty() {
        // No compile-time tokens — should have at least 3 permutations (all, V3 off, V2+V3 off)
        assert!(
            report.permutations_run >= 3,
            "expected >=3 permutations without compile-time tokens, got {}",
            report.permutations_run
        );
    }
}

/// Verify that disabling actually makes summon() return None.
#[cfg(target_arch = "x86_64")]
#[test]
fn disabled_tokens_fail_summon() {
    let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        for &name in &perm.disabled {
            // Verify each disabled token's summon() returns None
            if name == archmage::X64V2Token::NAME {
                assert!(
                    archmage::X64V2Token::summon().is_none(),
                    "V2 should be None when disabled"
                );
            }
            if name == archmage::X64V3Token::NAME {
                assert!(
                    archmage::X64V3Token::summon().is_none(),
                    "V3 should be None when disabled"
                );
            }
        }
    });
    assert!(report.permutations_run >= 1);
}

/// Verify that tokens are re-enabled after each permutation.
#[cfg(target_arch = "x86_64")]
#[test]
fn tokens_reenabled_between_permutations() {
    let v2_available = archmage::X64V2Token::summon().is_some();
    let v3_available = archmage::X64V3Token::summon().is_some();

    let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        if perm.disabled.is_empty() {
            // In the "all enabled" permutation, tokens should match initial state.
            // (This permutation might not run first, but tokens are re-enabled
            // between permutations, so it should still match.)
            if v2_available {
                assert!(
                    archmage::X64V2Token::summon().is_some(),
                    "V2 should be available in 'all enabled' permutation"
                );
            }
            if v3_available {
                assert!(
                    archmage::X64V3Token::summon().is_some(),
                    "V3 should be available in 'all enabled' permutation"
                );
            }
        }
    });

    // After all permutations, tokens should be back to normal
    assert_eq!(archmage::X64V2Token::summon().is_some(), v2_available);
    assert_eq!(archmage::X64V3Token::summon().is_some(), v3_available);
    assert!(report.permutations_run >= 1);
}

/// Verify cascade: disabling V3 should also disable V4.
#[cfg(target_arch = "x86_64")]
#[test]
fn cascade_disabling_works() {
    let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        if perm.disabled.contains(&archmage::X64V3Token::NAME) {
            // V3 disabled → V4 must also be None (cascade)
            assert!(
                archmage::X64V4Token::summon().is_none(),
                "V4 should be None when V3 is disabled (cascade)"
            );
        }
    });
}

/// Verify no duplicate effective states by checking that every permutation
/// produces a distinct set of available tokens.
#[cfg(target_arch = "x86_64")]
#[test]
fn no_duplicate_effective_states() {
    let mut states: Vec<(bool, bool, bool, bool, bool, bool)> = Vec::new();

    let _ = for_each_token_permutation(CompileTimePolicy::Warn, |_perm| {
        let state = (
            archmage::X64V2Token::summon().is_some(),
            archmage::X64V3Token::summon().is_some(),
            archmage::X64V3CryptoToken::summon().is_some(),
            archmage::X64V4Token::summon().is_some(),
            archmage::X64V4xToken::summon().is_some(),
            archmage::Avx512Fp16Token::summon().is_some(),
        );
        assert!(
            !states.contains(&state),
            "duplicate effective state: {state:?}"
        );
        states.push(state);
    });
}

/// Verify the Warn policy doesn't panic, even if tokens are compile-time guaranteed.
#[test]
fn warn_policy_does_not_panic() {
    let report = for_each_token_permutation(CompileTimePolicy::Warn, |_perm| {});
    // Should complete without panicking
    assert!(report.permutations_run >= 1);
}

/// Verify the report Display impl includes warnings.
#[test]
fn report_display() {
    let report = for_each_token_permutation(CompileTimePolicy::Warn, |_perm| {});
    let display = format!("{report}");
    assert!(
        display.contains("permutations run"),
        "display should mention permutation count"
    );
    for w in &report.warnings {
        assert!(display.contains(w), "display should include warning: {w}");
    }
}

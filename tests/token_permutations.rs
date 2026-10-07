//! Tests for the token permutation test helper.
//!
//! Requires `archmage::testing`, which is gated on the `std` feature.
#![cfg(feature = "std")]

use archmage::SimdToken;
use archmage::testing::{CompileTimePolicy, for_each_token_permutation, lock_token_testing};

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
            if name == archmage::X64V3GfniCryptoToken::NAME {
                assert!(
                    archmage::X64V3GfniCryptoToken::summon().is_none(),
                    "V3 GFNI Crypto should be None when disabled"
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
    // Hold the lock for the entire test so no parallel test can disable
    // tokens between our initial capture and the post-permutation check.
    let _lock = lock_token_testing();

    let v2_available = archmage::X64V2Token::summon().is_some();
    let v3_available = archmage::X64V3Token::summon().is_some();

    let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        if perm.disabled.is_empty() {
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
    assert_eq!(
        archmage::X64V2Token::summon().is_some(),
        v2_available,
        "V2 should be re-enabled after permutations"
    );
    assert_eq!(
        archmage::X64V3Token::summon().is_some(),
        v3_available,
        "V3 should be re-enabled after permutations"
    );
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
        if perm.disabled.contains(&archmage::X64V3CryptoToken::NAME) {
            assert!(
                archmage::X64V3GfniCryptoToken::summon().is_none(),
                "V3 GFNI Crypto should be None when V3 Crypto is disabled (cascade)"
            );
        }
        if perm
            .disabled
            .contains(&archmage::X64V3GfniCryptoToken::NAME)
        {
            assert!(
                archmage::X64V4xToken::summon().is_none(),
                "V4x should be None when V3 GFNI Crypto is disabled (cascade)"
            );
        }
    });
}

/// Verify no duplicate effective states by checking that every permutation
/// produces a distinct set of available tokens.
#[cfg(target_arch = "x86_64")]
#[test]
fn no_duplicate_effective_states() {
    let mut states: Vec<(bool, bool, bool, bool, bool, bool, bool, bool, bool)> = Vec::new();

    let _ = for_each_token_permutation(CompileTimePolicy::Warn, |_perm| {
        let state = (
            archmage::X64V1Token::summon().is_some(),
            archmage::X64V2Token::summon().is_some(),
            archmage::X64CryptoToken::summon().is_some(),
            archmage::X64V3Token::summon().is_some(),
            archmage::X64V3CryptoToken::summon().is_some(),
            archmage::X64V3GfniCryptoToken::summon().is_some(),
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

/// With `testable_dispatch`, `CompileTimePolicy::Fail` must succeed —
/// no token should be compile-time guaranteed when the feature is active.
#[cfg(feature = "testable_dispatch")]
#[test]
fn fail_policy_succeeds_with_testable_dispatch() {
    let report = for_each_token_permutation(CompileTimePolicy::Fail, |perm| {
        // Verify the permutation actually ran.
        // In the "all enabled" state, at least one token should be available.
        if perm.disabled.is_empty() {
            #[cfg(target_arch = "x86_64")]
            assert!(
                archmage::X64V1Token::summon().is_some(),
                "baseline x86_64 token should be available"
            );
            #[cfg(target_arch = "aarch64")]
            assert!(
                archmage::NeonToken::summon().is_some(),
                "NEON token should be available on aarch64"
            );
        }

        // Disabled tokens must return None from summon().
        for &name in &perm.disabled {
            #[cfg(target_arch = "x86_64")]
            {
                if name == archmage::X64V2Token::NAME {
                    assert!(archmage::X64V2Token::summon().is_none());
                }
                if name == archmage::X64V3Token::NAME {
                    assert!(archmage::X64V3Token::summon().is_none());
                }
            }
        }
    });

    // No warnings — testable_dispatch should prevent all compile-time guarantees.
    assert!(
        report.warnings.is_empty(),
        "testable_dispatch should prevent compile-time guarantees, but got warnings: {:?}",
        report.warnings
    );

    // Should have run multiple permutations on any real CPU.
    assert!(
        report.permutations_run >= 2,
        "expected at least 2 permutations, got {}",
        report.permutations_run
    );
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

/// Regression test for the token-detect lost-update race.
///
/// `summon()` caches probe results in a per-token u8 (0 = undetected,
/// 1 = unavailable/disabled, 2 = available). A `summon()` that enters the
/// cold `*_detect()` path while the cache is 0 used to publish its result
/// with an unconditional store — so a detect started in an enabled
/// window could overwrite a `disable(true)` that landed mid-probe,
/// leaving the cache at 2 and `summon()` returning `Some` for a token
/// that was disabled process-wide (observed as `dispatch liveness`
/// flakes in downstream test suites that permute tokens in parallel
/// with unrelated `summon()` callers).
///
/// The fixed detect publishes with `compare_exchange(0, ..)` and
/// rechecks the disabled flag, so after `disable(true)` returns the
/// cache can never again hold a stale 2 until re-enabled: every summon
/// must observe 1 and return `None`.
///
/// Two threads: the toggler cycles enable→disable and then asserts on
/// its own summons, while the summoner spins `summon()` to keep a
/// detect probe in flight across the disable boundary. Run under
/// `testable_dispatch` or the disable calls are no-ops.
#[cfg(all(feature = "testable_dispatch", target_arch = "x86_64"))]
#[test]
fn detect_cannot_resurrect_a_disabled_token() {
    use archmage::X64V3Token;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    // The toggler disables a real token process-wide, so it must hold the
    // same lock `for_each_token_permutation` runs under — otherwise our
    // disables race sibling permutation tests (and theirs race ours).
    // The summoner thread deliberately stays unlocked: it models the
    // unlocked `summon()` callers the ecosystem can't prevent, which is
    // exactly the hazard the CAS'd detect store makes harmless.
    let _lock = lock_token_testing();

    if X64V3Token::summon().is_none() {
        // No AVX2 on this CPU — the token is statically absent, so the
        // lost-update path this test exercises cannot run.
        return;
    }
    assert!(
        X64V3Token::dangerously_disable_token_process_wide(false).is_ok(),
        "V3 must be runtime-detectable under testable_dispatch"
    );

    let stop = Arc::new(AtomicBool::new(false));
    let violations = Arc::new(AtomicUsize::new(0));

    let summoner = {
        let stop = Arc::clone(&stop);
        let violations = Arc::clone(&violations);
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                // While the cache reads 0 this call runs the full CPUID
                // probe — the in-flight window where a stale store used
                // to land. If it returns Some while the disabled flag is
                // already visible, the stale store won.
                let got = X64V3Token::summon();
                if got.is_some()
                    && X64V3Token::manually_disabled().unwrap_or(false)
                    && X64V3Token::summon().is_some()
                {
                    // A second summon returning Some means this isn't a
                    // probe that merely raced the flag store — the cache
                    // itself reads "available" under a committed disable.
                    violations.fetch_add(1, Ordering::Relaxed);
                }
            }
        })
    };

    let mut toggler_violations = 0usize;
    for _ in 0..2000 {
        // Re-enable: cache drops to 0, so the summoner's next call enters
        // the slow detect path — the window the bug needs.
        let _ = X64V3Token::dangerously_disable_token_process_wide(false);
        std::thread::yield_now();
        X64V3Token::dangerously_disable_token_process_wide(true).expect("V3 is runtime-detectable");
        // After disable() returns, the cache must read "disabled" for
        // every summon; a resurrected 2 fails outright.
        for _ in 0..32 {
            if X64V3Token::summon().is_some() {
                toggler_violations += 1;
            }
        }
        if toggler_violations > 0 || violations.load(Ordering::Relaxed) > 0 {
            break;
        }
    }

    stop.store(true, Ordering::Relaxed);
    let _ = X64V3Token::dangerously_disable_token_process_wide(false);
    summoner.join().unwrap();

    assert_eq!(
        toggler_violations, 0,
        "summon() returned Some while V3 was disabled process-wide \
         (detect() overwrote a concurrent disable — lost update)"
    );
    assert_eq!(
        violations.load(Ordering::Relaxed),
        0,
        "summon() returned Some while V3 was disabled process-wide \
         (detect() overwrote a concurrent disable — lost update)"
    );
}

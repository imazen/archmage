//! Test utilities for exhaustive SIMD token permutation testing.
//!
//! [`for_each_token_permutation`] runs a closure once for every unique
//! combination of SIMD tokens disabled on the current CPU. This verifies
//! that dispatch code (via [`crate::attuned!`], [`#[attune(dispatch)]`](crate::attune), or manual
//! `summon()` checks) correctly handles all fallback tiers.
//!
//! Token disabling is process-wide, so a mutex serializes all permutation
//! runs and manual disable/enable calls. Use [`lock_token_testing`] when
//! calling `dangerously_disable_token_process_wide` directly, or when
//! you need to observe stable `summon()` results alongside parallel
//! permutation tests.
//!
//! # Example
//!
//! ```rust,ignore
//! use archmage::testing::{for_each_token_permutation, CompileTimePolicy};
//!
//! #[test]
//! fn dispatch_works_at_all_tiers() {
//!     let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
//!         let result = my_simd_function(&data);
//!         assert_eq!(result, expected, "failed at: {perm}");
//!     });
//!     eprintln!("{report}");
//! }
//! ```
//!
//! # Compile-time guaranteed tokens
//!
//! When compiled with `-Ctarget-cpu=native` (or similar), some tokens become
//! compile-time guaranteed and can't be disabled at runtime. The
//! [`CompileTimePolicy`] parameter controls behavior:
//!
//! - [`Warn`](CompileTimePolicy::Warn): exclude those tokens silently (warnings in report)
//! - [`WarnStderr`](CompileTimePolicy::WarnStderr): same, but also prints to stderr
//! - [`Fail`](CompileTimePolicy::Fail): panic — use in CI with the
//!   `testable_dispatch` feature enabled for full coverage
//!
//! WASM includes SIMD128 and its relaxed descendant. Availability is fixed by
//! module features; testing only suppresses selection, preserving module load
//! requirements and the validity of previously obtained proof tokens.

use std::sync::{Mutex, MutexGuard};

use crate::CompileTimeGuaranteedError;

static TOKEN_PERMUTATION_MUTEX: Mutex<()> = Mutex::new(());

/// Tracks which thread holds the token testing lock, enabling reentrance
/// from `for_each_token_permutation` when the caller already holds it.
static OWNER_THREAD: Mutex<Option<std::thread::ThreadId>> = Mutex::new(None);

/// RAII guard for the token testing lock. While held, no other thread can
/// manipulate token disable state via [`for_each_token_permutation`] or
/// another [`lock_token_testing`] call.
///
/// Use this when you need to call `dangerously_disable_token_process_wide`
/// manually, or when you need to observe stable token state (via `summon()`)
/// alongside permutation tests running in parallel.
///
/// # Example
///
/// ```rust,ignore
/// use archmage::testing::lock_token_testing;
/// use archmage::{X64V3Token, SimdToken};
///
/// #[test]
/// fn manual_disable_test() {
///     let _lock = lock_token_testing();
///     let baseline = my_function(&data);
///     X64V3Token::dangerously_disable_token_process_wide(true).unwrap();
///     let fallback = my_function(&data);
///     X64V3Token::dangerously_disable_token_process_wide(false).unwrap();
///     assert_eq!(baseline, fallback);
/// }
/// ```
#[derive(Debug)]
pub struct TokenTestGuard {
    _inner: MutexGuard<'static, ()>,
}

impl Drop for TokenTestGuard {
    fn drop(&mut self) {
        // Clear ownership before releasing the main mutex.
        if let Ok(mut owner) = OWNER_THREAD.lock() {
            *owner = None;
        }
    }
}

/// Acquire the process-wide token testing lock.
///
/// This is the same lock used internally by [`for_each_token_permutation`].
/// Holding it ensures that no permutation test or other `lock_token_testing`
/// caller on another thread can manipulate token disable state concurrently.
///
/// `for_each_token_permutation` is reentrant — if called while the current
/// thread holds this lock, it skips locking and proceeds directly.
///
/// # Panics
///
/// Panics if the current thread already holds this lock (non-reentrant for
/// direct callers — use `for_each_token_permutation` for reentrance).
pub fn lock_token_testing() -> TokenTestGuard {
    let current = std::thread::current().id();

    // Check for same-thread reentrance (programming error).
    let already_held = OWNER_THREAD
        .lock()
        .ok()
        .is_some_and(|owner| *owner == Some(current));
    assert!(
        !already_held,
        "lock_token_testing: already held by the current thread"
    );

    let guard = TOKEN_PERMUTATION_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());

    // Record ownership for reentrance detection.
    if let Ok(mut owner) = OWNER_THREAD.lock() {
        *owner = Some(current);
    }

    TokenTestGuard { _inner: guard }
}

/// What to do when a token's features are compile-time guaranteed (can't be disabled).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompileTimePolicy {
    /// Exclude the token from permutations and collect a warning in the
    /// report. Tests still run but with reduced tier coverage.
    Warn,
    /// Same as [`Warn`](Self::Warn), but also prints each warning to stderr.
    WarnStderr,
    /// Panic. Use in CI with the `testable_dispatch` feature
    /// enabled, where full permutation coverage is expected.
    Fail,
}

/// Describes one permutation's disabled-token state.
#[derive(Debug, Clone)]
pub struct TokenPermutation {
    /// Human-readable label, e.g., `"all enabled"` or `"x86-64-v3, AVX-512 disabled"`.
    pub label: String,
    /// Names of tokens disabled in this permutation.
    pub disabled: Vec<&'static str>,
}

impl core::fmt::Display for TokenPermutation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.label)
    }
}

/// Summary of a complete permutation run.
#[must_use]
#[derive(Debug, Clone)]
pub struct PermutationReport {
    /// Warnings about compile-time guaranteed tokens excluded from permutations.
    pub warnings: Vec<String>,
    /// Number of permutations executed.
    pub permutations_run: usize,
}

impl core::fmt::Display for PermutationReport {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} permutations run", self.permutations_run)?;
        for w in &self.warnings {
            write!(f, "\n  warning: {w}")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Internal machinery
// ---------------------------------------------------------------------------

pub(crate) struct TokenSlot {
    name: &'static str,
    /// RUSTFLAGS to disable these features (e.g., `"-Ctarget-feature=-avx2,-fma,..."`).
    disable_flags: &'static str,
    /// Indices of all transitive descendants in the cascade tree.
    descendants: &'static [usize],
    disable: fn(bool) -> Result<(), CompileTimeGuaranteedError>,
    compiled_with: fn() -> Option<bool>,
    check_available: fn() -> bool,
}

impl TokenSlot {
    #[allow(dead_code)] // Unused on targets without generated token slots (e.g., i686, armv7).
    pub(crate) fn new<T: crate::SimdToken>(
        descendants: &'static [usize],
        disable: fn(bool) -> Result<(), CompileTimeGuaranteedError>,
    ) -> Self {
        fn check_avail<T: crate::SimdToken>() -> bool {
            T::summon().is_some()
        }
        Self {
            name: T::NAME,
            disable_flags: T::DISABLE_TARGET_FEATURES,
            descendants,
            disable,
            compiled_with: T::compiled_with,
            check_available: check_avail::<T>,
        }
    }
}

fn build_token_slots() -> Vec<TokenSlot> {
    crate::tokens::build_token_slots()
}

/// Generate all downward-closed subsets of `candidates` within the hierarchy.
///
/// A subset S is *downward-closed* if: for every token in S, all its
/// descendants that are also candidates are in S. This matches cascade
/// behavior — disabling a parent implicitly disables all descendants.
fn valid_disabled_subsets(slots: &[TokenSlot], candidates: &[usize]) -> Vec<Vec<usize>> {
    let n = candidates.len();
    let mut result = Vec::with_capacity(1 << n);

    for mask in 0u32..(1u32 << n) {
        let subset: Vec<usize> = (0..n)
            .filter(|&i| mask & (1 << i) != 0)
            .map(|i| candidates[i])
            .collect();

        // Check: if a token is disabled, all its candidate-descendants must be too.
        let valid = subset.iter().all(|&idx| {
            slots[idx]
                .descendants
                .iter()
                .all(|&desc| !candidates.contains(&desc) || subset.contains(&desc))
        });

        if valid {
            result.push(subset);
        }
    }

    result
}

/// Drop guard that re-enables disabled tokens even if the test closure panics.
struct ReenableOnDrop<'a> {
    slots: &'a [TokenSlot],
    indices: &'a [usize],
}

impl Drop for ReenableOnDrop<'_> {
    fn drop(&mut self) {
        for &idx in self.indices {
            let _ = (self.slots[idx].disable)(false);
        }
    }
}

/// Run `f` once for every unique combination of available SIMD tokens disabled.
///
/// Acquires a process-wide mutex to prevent concurrent token manipulation.
/// Re-enables all tokens after each invocation, even if `f` panics.
///
/// # Permutation logic
///
/// 1. Resets any stale disabled state from previous tests
/// 2. Discovers which tokens are available on this CPU (`summon()` → `Some`)
/// 3. Excludes tokens that can't be disabled (compile-time guaranteed),
///    applying `policy`
/// 4. Skips tokens the CPU doesn't have — they'd produce duplicate states
/// 5. Generates all valid combinations respecting cascade hierarchy
///    (disabling a parent implies its descendants are disabled too)
/// 6. Runs `f` for each unique effective state
///
/// # Panics
///
/// - If `policy` is [`CompileTimePolicy::Fail`] and any available token is
///   compile-time guaranteed.
/// - If `f` panics (after re-enabling tokens and releasing the mutex).
pub fn for_each_token_permutation(
    policy: CompileTimePolicy,
    mut f: impl FnMut(&TokenPermutation),
) -> PermutationReport {
    // If the current thread already holds the lock (via lock_token_testing),
    // skip locking to avoid deadlock. Otherwise acquire it.
    let already_held = OWNER_THREAD
        .lock()
        .ok()
        .is_some_and(|owner| *owner == Some(std::thread::current().id()));
    let _guard = if already_held {
        None
    } else {
        let g = TOKEN_PERMUTATION_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Ok(mut owner) = OWNER_THREAD.lock() {
            *owner = Some(std::thread::current().id());
        }
        Some(g)
    };
    // When we acquired the lock ourselves, clear ownership on drop.
    // (If the caller holds it via lock_token_testing, they own cleanup.)
    struct ClearOwnerOnDrop(bool);
    impl Drop for ClearOwnerOnDrop {
        fn drop(&mut self) {
            if self.0
                && let Ok(mut owner) = OWNER_THREAD.lock()
            {
                *owner = None;
            }
        }
    }
    let _clear_owner = ClearOwnerOnDrop(!already_held && _guard.is_some());

    let slots = build_token_slots();

    // Reset any stale disabled state from previous tests.
    // Compile-time guaranteed tokens fail here — that's fine, their
    // summon() bypasses the cache anyway.
    for slot in &slots {
        let _ = (slot.disable)(false);
    }

    // Probe availability and categorize.
    let mut warnings = Vec::new();
    let mut candidates = Vec::new();

    for (i, slot) in slots.iter().enumerate() {
        if !(slot.check_available)() {
            // CPU doesn't have this feature — already effectively disabled.
            continue;
        }
        if (slot.compiled_with)() == Some(true) {
            let msg = format!(
                "{}: compile-time guaranteed, excluded from permutations. \
                 To include it, either:\n\
                 \x20 1. Add `testable_dispatch` to archmage features in Cargo.toml\n\
                 \x20 2. Remove `-Ctarget-cpu` from RUSTFLAGS\n\
                 \x20 3. Compile with RUSTFLAGS=\"{}\"",
                slot.name, slot.disable_flags,
            );
            match policy {
                CompileTimePolicy::Warn => {
                    warnings.push(msg);
                }
                CompileTimePolicy::WarnStderr => {
                    eprintln!("warning: {msg}");
                    warnings.push(msg);
                }
                CompileTimePolicy::Fail => {
                    panic!("{msg}");
                }
            }
            continue;
        }
        candidates.push(i);
    }

    let subsets = valid_disabled_subsets(&slots, &candidates);
    let mut permutations_run = 0;

    for subset in &subsets {
        let disabled_names: Vec<&'static str> = subset.iter().map(|&i| slots[i].name).collect();

        let label = if subset.is_empty() {
            String::from("all enabled")
        } else {
            format!("{} disabled", disabled_names.join(", "))
        };

        let perm = TokenPermutation {
            label,
            disabled: disabled_names,
        };

        // Disable the tokens in this permutation.
        for &idx in subset {
            let _ = (slots[idx].disable)(true);
        }

        // Run the test. The drop guard re-enables tokens on both normal
        // return and panic, so subsequent permutations (or tests) start clean.
        {
            let _reenable = ReenableOnDrop {
                slots: &slots,
                indices: subset,
            };
            f(&perm);
        }

        permutations_run += 1;
    }

    PermutationReport {
        warnings,
        permutations_run,
    }
}

#[cfg(all(test, panic = "unwind"))]
mod tests {
    use super::*;

    #[test]
    fn caught_recursive_lock_panic_preserves_permutation_reentrancy() {
        let lock = lock_token_testing();
        let error = std::panic::catch_unwind(lock_token_testing).unwrap_err();
        assert_eq!(
            error.downcast_ref::<&str>().copied(),
            Some("lock_token_testing: already held by the current thread")
        );

        // Check ownership before calling a function that would deadlock if
        // it could no longer recognize this thread's already-held lock.
        assert_eq!(
            *OWNER_THREAD
                .lock()
                .expect("ownership survives a caught panic"),
            Some(std::thread::current().id())
        );
        let mut calls = 0;
        let report = for_each_token_permutation(CompileTimePolicy::Warn, |_| calls += 1);
        assert!(calls > 0);
        assert_eq!(report.permutations_run, calls);
        drop(lock);
        drop(lock_token_testing());
    }

    #[test]
    fn permutation_callback_panic_restores_available_tokens() {
        let _lock = lock_token_testing();
        let slots = build_token_slots();
        let availability = || {
            slots
                .iter()
                .map(|slot| (slot.check_available)())
                .collect::<Vec<_>>()
        };
        let before = availability();
        let has_disableable_tokens = slots
            .iter()
            .zip(&before)
            .any(|(slot, &available)| available && (slot.compiled_with)() != Some(true));
        let result = std::panic::catch_unwind(|| {
            for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
                if !perm.disabled.is_empty() || !has_disableable_tokens {
                    for slot in &slots {
                        if perm.disabled.contains(&slot.name) {
                            assert!(
                                !(slot.check_available)(),
                                "{} was never disabled",
                                slot.name
                            );
                        }
                    }
                    panic!("callback failed");
                }
            })
        });
        assert_eq!(
            result.unwrap_err().downcast_ref::<&str>().copied(),
            Some("callback failed")
        );
        assert_eq!(availability(), before);
    }
}

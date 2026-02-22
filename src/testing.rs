//! Test utilities for exhaustive SIMD token permutation testing.
//!
//! [`for_each_token_permutation`] runs a closure once for every unique
//! combination of SIMD tokens disabled on the current CPU. This verifies
//! that dispatch code (via `incant!` or manual `summon()` checks) correctly
//! handles all fallback tiers.
//!
//! Token disabling is process-wide, so a mutex serializes all permutation
//! runs. For fully correct results, run tests single-threaded:
//!
//! ```sh
//! cargo test -- --test-threads=1
//! ```
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

use alloc::{format, string::String, vec::Vec};
use std::sync::Mutex;

use crate::CompileTimeGuaranteedError;

static TOKEN_PERMUTATION_MUTEX: Mutex<()> = Mutex::new(());

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

struct TokenSlot {
    name: &'static str,
    /// RUSTFLAGS to disable these features (e.g., `"-Ctarget-feature=-avx2,-fma,..."`).
    disable_flags: &'static str,
    /// Indices of all transitive descendants in the cascade tree.
    descendants: &'static [usize],
    disable: fn(bool) -> Result<(), CompileTimeGuaranteedError>,
    compiled_with: fn() -> Option<bool>,
    check_available: fn() -> bool,
}

/// Builds a `TokenSlot` from a token type. Generates a named `fn` for the
/// availability check to avoid closure-to-fn-pointer coercion issues.
macro_rules! token_slot {
    ($token:ty, $desc:expr) => {{
        fn check_avail() -> bool {
            <$token as crate::SimdToken>::summon().is_some()
        }
        TokenSlot {
            name: <$token as crate::SimdToken>::NAME,
            disable_flags: <$token as crate::SimdToken>::DISABLE_TARGET_FEATURES,
            descendants: $desc,
            disable: <$token>::dangerously_disable_token_process_wide,
            compiled_with: <$token as crate::SimdToken>::compiled_with,
            check_available: check_avail,
        }
    }};
}

fn build_token_slots() -> Vec<TokenSlot> {
    let mut slots = Vec::new();

    // x86/x86_64 hierarchy:
    //   V1(0) → V2(1) → V3(2) → V3Crypto(3)
    //                          → V4(4) → V4x(5)
    //                                  → Fp16(6)
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        slots.push(token_slot!(crate::X64V1Token, &[1, 2, 3, 4, 5, 6]));
        slots.push(token_slot!(crate::X64V2Token, &[2, 3, 4, 5, 6]));
        slots.push(token_slot!(crate::X64V3Token, &[3, 4, 5, 6]));
        slots.push(token_slot!(crate::X64V3CryptoToken, &[]));
        slots.push(token_slot!(crate::X64V4Token, &[5, 6]));
        slots.push(token_slot!(crate::X64V4xToken, &[]));
        slots.push(token_slot!(crate::Avx512Fp16Token, &[]));
    }

    // AArch64 hierarchy:
    //   Neon(0) → Aes(1), Sha3(2), Crc(3)
    #[cfg(target_arch = "aarch64")]
    {
        slots.push(token_slot!(crate::NeonToken, &[1, 2, 3]));
        slots.push(token_slot!(crate::NeonAesToken, &[]));
        slots.push(token_slot!(crate::NeonSha3Token, &[]));
        slots.push(token_slot!(crate::NeonCrcToken, &[]));
    }

    // WASM: Wasm128Token is compile-time only (can't be disabled).
    // ScalarToken: always available (can't be disabled).
    // Neither contributes to permutations.

    slots
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
    let _guard = TOKEN_PERMUTATION_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());

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

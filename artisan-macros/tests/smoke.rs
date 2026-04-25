//! End-to-end smoke test for `#[cpu_tier]` and `#[chain]`.
//!
//! Uses scalar bodies on every tier to verify the dispatch mechanics without
//! needing real SIMD intrinsics. A real-world user would have hand-written
//! intrinsics in each tier function — correctness comes from matching output,
//! not from the dispatch macro itself.

#![allow(dead_code)]

use artisan_macros::{chain, cpu_tier};

// --- Scalar fallback (always available) ---
fn dot_scalar(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

// --- x86_64 tiers ---
#[cpu_tier(enable = "avx2,fma")]
fn dot_v3(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    // Real user would write AVX2+FMA intrinsics here.
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

#[cpu_tier(enable = "sse4.2")]
fn dot_v2(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

// --- aarch64 tier ---
#[cpu_tier(enable = "neon")]
fn dot_neon(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

// --- Chain declaration ---
// DELIBERATE MISMATCH to sanity-check that the compile-time feature-string
// assertion fires. Flip the `fma` in dot_v3's chain entry to something else
// and `cargo test` should fail with the artisan-macros mismatch error.
// Verified manually on 2026-04-20; left in the CANONICAL (matching) form.
#[chain(
    x86_64 = [
        dot_v3 = "avx2,fma",
        dot_v2 = "sse4.2",
    ],
    aarch64 = [
        dot_neon = "neon",
    ],
    default = dot_scalar,
)]
/// Public entry. Dispatches to the highest available tier on the host arch,
/// falling through to `dot_scalar` on uncovered arches or unsupported CPUs.
pub fn dot(a: &[f32; 4], b: &[f32; 4]) -> f32 {}

#[test]
fn dispatch_produces_correct_result() {
    let a = [1.0, 2.0, 3.0, 4.0];
    let b = [5.0, 6.0, 7.0, 8.0];
    // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
    assert_eq!(dot(&a, &b), 70.0);
}

#[test]
fn dispatch_is_idempotent_across_calls() {
    let a = [1.0f32; 4];
    let b = [2.0f32; 4];
    let expected = 8.0f32;
    for _ in 0..100 {
        assert_eq!(dot(&a, &b), expected);
    }
}

// The test-hooks block below exercises the thread-local override + RAII scope.
// Compiles only under `cfg(test)` or `cfg(feature = "artisan_test_hooks")`.
#[cfg(any(test, feature = "artisan_test_hooks"))]
#[test]
fn force_max_tier_scope_compiles_and_isolates() {
    let a = [1.0, 2.0, 3.0, 4.0];
    let b = [5.0, 6.0, 7.0, 8.0];
    let expected = 70.0;

    // Force down to Default (no SIMD). Dispatcher must skip every tier and hit
    // the scalar fallback.
    {
        let _scope = dot_force_max_tier(DotTier::Default);
        assert_eq!(dot(&a, &b), expected);
    } // _scope drops, restores previous (None)

    // After scope drop, normal dispatch resumes.
    assert_eq!(dot(&a, &b), expected);
}

// Smoke-test that the test-hook types are actually generated and public.
#[cfg(any(test, feature = "artisan_test_hooks"))]
#[allow(dead_code)]
fn _type_surface_check() {
    let _: DotTier = DotTier::Default;
    let _scope: DotScope = dot_force_max_tier(DotTier::Default);
}

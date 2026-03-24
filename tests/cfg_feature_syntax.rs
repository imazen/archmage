//! Tests for the `cfg(feature)` syntax across all macros.
//!
//! Tests `tier(feature)` in incant!, `cfg(feature)` in #[arcane], #[rite], #[autoversion].
#![allow(deprecated)] // Legacy SimdToken usage in autoversion tests

use archmage::prelude::*;

// ============================================================================
// incant! with tier(feature) syntax
// ============================================================================

#[arcane]
fn add_v3(_token: X64V3Token, a: f32, b: f32) -> f32 {
    a + b
}

fn add_scalar(_token: ScalarToken, a: f32, b: f32) -> f32 {
    a + b
}

#[arcane]
fn add_neon(_token: NeonToken, a: f32, b: f32) -> f32 {
    a + b
}

#[cfg(target_arch = "wasm32")]
fn add_wasm128(_token: Wasm128Token, a: f32, b: f32) -> f32 {
    a + b
}

// v4(avx512) — conditional dispatch. When this crate's avx512 feature is off
// (which it always is in this test), v4 dispatch is eliminated.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[arcane]
fn add_v4(_token: X64V4Token, a: f32, b: f32) -> f32 {
    a + b
}

/// Explicit tier list with feature-gated v4 (shorthand)
#[test]
fn incant_explicit_v4_feature_gate() {
    let result = incant!(add(1.0, 2.0), [v4(avx512), v3, scalar]);
    assert!((result - 3.0).abs() < 1e-6);
}

/// Same with canonical cfg() syntax
#[test]
fn incant_explicit_v4_cfg_gate() {
    let result = incant!(add(1.0, 2.0), [v4(cfg(avx512)), v3, scalar]);
    assert!((result - 3.0).abs() < 1e-6);
}

/// Default tiers (v4 auto-gated on avx512)
#[test]
fn incant_default_tiers() {
    let result = incant!(add(1.0, 2.0));
    assert!((result - 3.0).abs() < 1e-6);
}

// ============================================================================
// #[arcane(cfg(feature))] — combined arch + feature guard
// ============================================================================

// This function only exists when avx512 feature is on (via cfg(avx512) on arcane).
// No manual #[cfg(...)] needed — arcane generates it.
#[arcane(cfg(avx512))]
fn guarded_v4(_token: X64V4Token, x: f32) -> f32 {
    x * 2.0
}

// Same without cfg — always exists on x86_64
#[arcane]
fn always_v3(_token: X64V3Token, x: f32) -> f32 {
    x * 2.0
}

#[cfg(target_arch = "x86_64")]
#[test]
fn arcane_cfg_feature() {
    // always_v3 should be callable
    if let Some(token) = X64V3Token::summon() {
        let result = always_v3(token, 5.0);
        assert!((result - 10.0).abs() < 1e-6);
    }
    // guarded_v4 only exists with avx512 feature — can't call it here
    // but it should compile without errors
}

// ============================================================================
// #[rite(cfg(feature))] — single tier with feature guard
// ============================================================================

#[rite(v3, cfg(avx512))]
fn rite_guarded() -> f32 {
    42.0
}

// Without cfg — always exists
#[rite(v3)]
fn rite_always() -> f32 {
    42.0
}

// ============================================================================
// #[autoversion(cfg(feature))] — conditional dispatch
// ============================================================================

#[autoversion(cfg(avx512))]
fn auto_sum(_token: SimdToken, data: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for &x in data {
        sum += x;
    }
    sum
}

#[test]
fn autoversion_cfg_feature() {
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let result = auto_sum(&data);
    assert!((result - 10.0).abs() < 1e-6);
}

// ============================================================================
// #[autoversion] with per-tier feature gates
// ============================================================================

// Shorthand: v4(avx512)
#[autoversion(v4(avx512), v3, neon)]
fn auto_with_tier_gate(_token: SimdToken, data: &[f32]) -> f32 {
    data.iter().sum()
}

#[test]
fn autoversion_tier_feature_gate() {
    let data = [1.0f32, 2.0, 3.0];
    let result = auto_with_tier_gate(&data);
    assert!((result - 6.0).abs() < 1e-6);
}

// Canonical: v4(cfg(avx512))
#[autoversion(v4(cfg(avx512)), v3, neon)]
fn auto_with_cfg_gate(_token: SimdToken, data: &[f32]) -> f32 {
    data.iter().sum()
}

#[test]
fn autoversion_cfg_tier_gate() {
    let data = [1.0f32, 2.0, 3.0];
    let result = auto_with_cfg_gate(&data);
    assert!((result - 6.0).abs() < 1e-6);
}

// ============================================================================
// #[autoversion] inside macro_rules! — hygiene fix
// ============================================================================

macro_rules! define_autoversioned {
    ($name:ident) => {
        #[autoversion]
        fn $name(_token: SimdToken, data: &[f32]) -> f32 {
            let mut sum = 0.0f32;
            for &x in data {
                sum += x;
            }
            sum
        }
    };
}

define_autoversioned!(macro_sum);

#[test]
fn autoversion_in_macro_rules() {
    let data = [1.0f32, 2.0, 3.0];
    let result = macro_sum(&data);
    assert!((result - 6.0).abs() < 1e-6);
}

// Combined: cfg(feature) + macro_rules!
macro_rules! define_conditional_autoversioned {
    ($name:ident) => {
        #[autoversion(cfg(avx512))]
        fn $name(_token: SimdToken, data: &[f32]) -> f32 {
            data.iter().sum()
        }
    };
}

define_conditional_autoversioned!(conditional_macro_sum);

#[test]
fn autoversion_cfg_in_macro_rules() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let result = conditional_macro_sum(&data);
    assert!((result - 15.0).abs() < 1e-6);
}

//! Tier modifiers with avx512 feature enabled.
//!
//! Tests: +tier, -tier, +default, +v4 (unconditional), tier(cfg(feature))
#![cfg(target_arch = "x86_64")]
#![allow(dead_code)]

use archmage::prelude::*;

// ============================================================================
// +v1: add v1 to defaults (v4, v3, neon, wasm128, scalar)
// ============================================================================

#[autoversion(+v1)]
fn add_v1(data: &[f32]) -> f32 {
    data.iter().sum()
}

#[test]
fn additive_v1() {
    assert_eq!(add_v1(&[1.0, 2.0]), 3.0);
}

#[test]
fn additive_v1_variant_exists() {
    let t = X64V1Token::summon().expect("v1 always available on x86_64");
    assert_eq!(add_v1_v1(t, &[10.0]), 10.0);
}

// ============================================================================
// +v4: override v4(avx512) → unconditional v4
// ============================================================================

#[autoversion(+v4)]
fn unconditional_v4(data: &[f32]) -> f32 {
    data.iter().sum()
}

#[test]
fn plus_v4_dispatches() {
    assert_eq!(unconditional_v4(&[1.0, 2.0, 3.0]), 6.0);
}

#[test]
fn plus_v4_variant_exists() {
    if let Some(t) = X64V4Token::summon() {
        assert_eq!(unconditional_v4_v4(t, &[10.0]), 10.0);
    }
}

// ============================================================================
// +default: replace scalar with tokenless default
// ============================================================================

#[autoversion(+default)]
fn with_default(data: &[f32]) -> f32 {
    data.iter().sum()
}

#[test]
fn plus_default_dispatches() {
    assert_eq!(with_default(&[1.0, 2.0]), 3.0);
}

#[test]
fn plus_default_variant_exists() {
    // _default takes no token
    assert_eq!(with_default_default(&[5.0]), 5.0);
}

// ============================================================================
// -neon, -wasm128: remove tiers not needed on x86_64
// ============================================================================

#[autoversion(-neon, -wasm128)]
fn x86_only(data: &[f32]) -> f32 {
    data.iter().sum()
}

#[test]
fn minus_neon_wasm_dispatches() {
    assert_eq!(x86_only(&[1.0, 2.0, 3.0]), 6.0);
}

// ============================================================================
// incant! with modifiers
// ============================================================================

#[arcane]
fn inc_v3(_: X64V3Token, x: f32) -> f32 {
    x + 3.0
}

fn inc_v1(_: X64V1Token, x: f32) -> f32 {
    x + 1.0
}

fn inc_scalar(_: ScalarToken, x: f32) -> f32 {
    x + 0.0
}

fn inc_dispatch(x: f32) -> f32 {
    incant!(inc(x), [-neon, -wasm128, -v4, +v1])
}

#[test]
fn incant_modifiers() {
    let result = inc_dispatch(10.0);
    if X64V3Token::summon().is_some() {
        assert_eq!(result, 13.0);
    } else {
        assert!(result >= 10.0);
    }
}

// ============================================================================
// magetypes with modifiers
// ============================================================================

#[magetypes(-neon, -wasm128, +v1)]
fn mt_mod(_token: Token, x: f32) -> f32 {
    x + 1.0
}

#[test]
fn magetypes_modifier_scalar() {
    assert_eq!(mt_mod_scalar(ScalarToken, 1.0), 2.0);
}

#[test]
fn magetypes_modifier_v1_exists() {
    if let Some(t) = X64V1Token::summon() {
        assert_eq!(mt_mod_v1(t, 1.0), 2.0);
    }
}

#[test]
fn magetypes_modifier_v3_still_exists() {
    if let Some(t) = X64V3Token::summon() {
        assert_eq!(mt_mod_v3(t, 1.0), 2.0);
    }
}

// ============================================================================
// cfg(feature) syntax in tier gate: v4(cfg(avx512))
// ============================================================================

#[cfg(feature = "avx512")]
#[arcane]
fn cfg_v4(_: X64V4Token, x: f32) -> f32 {
    x + 4.0
}

#[arcane]
fn cfg_v3(_: X64V3Token, x: f32) -> f32 {
    x + 3.0
}

fn cfg_scalar(_: ScalarToken, x: f32) -> f32 {
    x + 0.0
}

fn cfg_dispatch(x: f32) -> f32 {
    incant!(cfg(x), [v4(cfg(avx512)), v3, scalar])
}

#[test]
fn cfg_syntax_in_tier_gate() {
    let result = cfg_dispatch(10.0);
    // avx512 feature is on, so v4 dispatch exists.
    // Actual result depends on CPU: v4 → 14.0, v3 → 13.0, scalar → 10.0
    assert!(result >= 10.0);
}

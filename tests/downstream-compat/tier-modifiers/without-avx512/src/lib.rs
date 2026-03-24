//! Tier modifiers WITHOUT avx512 feature.
//!
//! v4 tiers from defaults are cfg-gated on avx512 and eliminated.
//! No _v4 functions needed.
#![cfg(target_arch = "x86_64")]
#![allow(dead_code)]

use archmage::prelude::*;

// ============================================================================
// Default tiers without avx512: v4 is cfg-gated out, only v3/scalar on x86
// ============================================================================

#[autoversion]
fn defaults_no_avx512(data: &[f32]) -> f32 {
    data.iter().sum()
}

#[test]
fn default_tiers_without_avx512() {
    assert_eq!(defaults_no_avx512(&[1.0, 2.0, 3.0]), 6.0);
}

// ============================================================================
// +v1: add v1, v4 still cfg-gated out
// ============================================================================

#[autoversion(+v1)]
fn plus_v1_no_avx512(data: &[f32]) -> f32 {
    data.iter().sum()
}

#[test]
fn additive_v1_without_avx512() {
    assert_eq!(plus_v1_no_avx512(&[1.0, 2.0]), 3.0);
}

#[test]
fn additive_v1_variant() {
    let t = X64V1Token::summon().expect("v1 always available");
    assert_eq!(plus_v1_no_avx512_v1(t, &[10.0]), 10.0);
}

// ============================================================================
// +default: replace scalar with default (tokenless), no avx512
// ============================================================================

#[autoversion(+default)]
fn default_fallback_no_avx512(data: &[f32]) -> f32 {
    data.iter().sum()
}

#[test]
fn plus_default_without_avx512() {
    assert_eq!(default_fallback_no_avx512(&[1.0, 2.0, 3.0]), 6.0);
}

#[test]
fn default_variant_exists() {
    assert_eq!(default_fallback_no_avx512_default(&[5.0]), 5.0);
}

// ============================================================================
// -neon, -wasm128: strip non-x86 tiers (no avx512, so v4 also absent)
// ============================================================================

#[autoversion(-neon, -wasm128)]
fn stripped_no_avx512(data: &[f32]) -> f32 {
    data.iter().sum()
}

#[test]
fn stripped_dispatches() {
    assert_eq!(stripped_no_avx512(&[2.0, 3.0]), 5.0);
}

// ============================================================================
// -v4: explicitly remove v4 even when avx512 could be unified
// ============================================================================

#[autoversion(-v4)]
fn minus_v4_explicit(data: &[f32]) -> f32 {
    data.iter().sum()
}

#[test]
fn explicit_minus_v4() {
    assert_eq!(minus_v4_explicit(&[1.0, 1.0, 1.0, 1.0]), 4.0);
}

// ============================================================================
// incant! with modifiers, no avx512
// ============================================================================

#[arcane]
fn noavx_v3(_: X64V3Token, x: f32) -> f32 {
    x + 3.0
}

fn noavx_scalar(_: ScalarToken, x: f32) -> f32 {
    x + 0.0
}

fn noavx_dispatch(x: f32) -> f32 {
    // v4 is in defaults but cfg-gated on avx512 (which is off)
    // Only v3 and scalar are effective
    incant!(noavx(x), [-neon, -wasm128])
}

#[test]
fn incant_modifiers_no_avx512() {
    let result = noavx_dispatch(10.0);
    if X64V3Token::summon().is_some() {
        assert_eq!(result, 13.0);
    } else {
        assert_eq!(result, 10.0);
    }
}

// ============================================================================
// magetypes with modifiers, no avx512
// ============================================================================

#[magetypes(-neon, -wasm128)]
fn mt_noavx(_token: Token, x: f32) -> f32 {
    x + 1.0
}

#[test]
fn magetypes_no_avx512_scalar() {
    assert_eq!(mt_noavx_scalar(ScalarToken, 1.0), 2.0);
}

#[test]
fn magetypes_no_avx512_v3() {
    if let Some(t) = X64V3Token::summon() {
        assert_eq!(mt_noavx_v3(t, 1.0), 2.0);
    }
}

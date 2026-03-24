//! Tests for tier list modifiers: +tier (add/override), -tier (remove).
//!
//! Default tiers: v4(avx512), v3, neon, wasm128, scalar
//!
//! These tests verify:
//! - +tier appends new tiers to defaults
//! - +tier overrides existing defaults (e.g., +v4 makes v4 unconditional)
//! - +default replaces scalar fallback with tokenless default
//! - -tier removes a tier from defaults
//! - +tier(cfg(feature)) adds a cfg gate to a default tier
//! - Mixing +/- with plain tiers is a compile error (tested via compile_fail)
#![allow(deprecated)] // SimdToken in autoversion
#![cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]

use archmage::prelude::*;

// ============================================================================
// +tier: append new tiers to defaults
// ============================================================================

mod additive_append {
    use super::*;

    // +v1 adds v1 to defaults (v4, v3, neon, wasm128, scalar)
    #[autoversion(+v1)]
    fn sum_plus_v1(data: &[f32]) -> f32 {
        data.iter().sum()
    }

    #[test]
    fn autoversion_plus_v1_dispatches() {
        assert_eq!(sum_plus_v1(&[1.0, 2.0, 3.0]), 6.0);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn autoversion_plus_v1_generates_v1_variant() {
        // v1 (SSE2) is always available on x86_64
        let t = X64V1Token::summon().expect("v1 always available");
        assert_eq!(sum_plus_v1_v1(t, &[10.0, 20.0]), 30.0);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn autoversion_plus_v1_still_has_v3() {
        // v3 from defaults should still be present
        if let Some(t) = X64V3Token::summon() {
            assert_eq!(sum_plus_v1_v3(t, &[1.0, 2.0]), 3.0);
        }
    }

    // +v2 via magetypes
    #[magetypes(+v2)]
    fn mt_plus_v2(_token: Token, x: f32) -> f32 {
        x + 1.0
    }

    #[test]
    fn magetypes_plus_v2_scalar_exists() {
        assert_eq!(mt_plus_v2_scalar(ScalarToken, 1.0), 2.0);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn magetypes_plus_v2_generates_v2_variant() {
        if let Some(t) = X64V2Token::summon() {
            assert_eq!(mt_plus_v2_v2(t, 1.0), 2.0);
        }
    }
}

// ============================================================================
// +tier: override existing default tier
// ============================================================================

#[cfg(target_arch = "x86_64")]
mod additive_override {
    use super::*;

    // +v4 overrides v4(avx512) from defaults → v4 is now UNCONDITIONAL.
    // The variant is always compiled (no cfg(feature = "avx512") guard).
    // This means a _v4 function MUST exist without feature gating.
    #[autoversion(+v4)]
    fn sum_unconditional_v4(data: &[f32]) -> f32 {
        data.iter().sum()
    }

    #[test]
    fn autoversion_unconditional_v4_dispatches() {
        assert_eq!(sum_unconditional_v4(&[1.0, 2.0, 3.0]), 6.0);
    }

    // The _v4 variant exists unconditionally (no avx512 feature gate)
    #[test]
    fn autoversion_unconditional_v4_variant_exists() {
        if let Some(t) = X64V4Token::summon() {
            assert_eq!(sum_unconditional_v4_v4(t, &[10.0]), 10.0);
        }
    }
}

// ============================================================================
// +default: replace scalar fallback with tokenless default
// ============================================================================

mod additive_default_fallback {
    use super::*;

    // +default replaces scalar in defaults → dispatcher calls _default() instead of _scalar()
    #[autoversion(+default)]
    fn sum_with_default(data: &[f32]) -> f32 {
        data.iter().sum()
    }

    #[test]
    fn autoversion_plus_default_dispatches() {
        assert_eq!(sum_with_default(&[1.0, 2.0, 3.0]), 6.0);
    }

    // The _default variant should exist (tokenless)
    #[test]
    fn autoversion_plus_default_generates_default_variant() {
        assert_eq!(sum_with_default_default(&[5.0, 5.0]), 10.0);
    }
}

// ============================================================================
// -tier: remove a tier from defaults
// ============================================================================

mod subtractive {
    use super::*;

    // -wasm128 removes wasm128 from defaults. We don't need a _wasm128 variant.
    #[autoversion(-wasm128)]
    fn sum_no_wasm(data: &[f32]) -> f32 {
        data.iter().sum()
    }

    #[test]
    fn autoversion_minus_wasm_dispatches() {
        assert_eq!(sum_no_wasm(&[1.0, 2.0, 3.0]), 6.0);
    }

    // -neon removes neon. On aarch64 this means only scalar fallback.
    #[autoversion(-neon)]
    fn sum_no_neon(data: &[f32]) -> f32 {
        data.iter().sum()
    }

    #[test]
    fn autoversion_minus_neon_dispatches() {
        assert_eq!(sum_no_neon(&[1.0, 2.0]), 3.0);
    }

    // Combined: -wasm128, -neon, +v1 — remove two, add one
    #[autoversion(-wasm128, -neon, +v1)]
    fn sum_custom(data: &[f32]) -> f32 {
        data.iter().sum()
    }

    #[test]
    fn autoversion_combined_modifiers() {
        assert_eq!(sum_custom(&[1.0, 2.0, 3.0, 4.0]), 10.0);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn autoversion_combined_has_v1() {
        let t = X64V1Token::summon().expect("v1 always available");
        assert_eq!(sum_custom_v1(t, &[10.0]), 10.0);
    }
}

// ============================================================================
// +tier(cfg(feature)): add a cfg gate to a default tier
// ============================================================================

#[cfg(target_arch = "x86_64")]
mod additive_cfg_gate {
    use super::*;

    // incant! with +v3(cfg(nonexistent_feature)):
    // v3 is gated on a feature that doesn't exist → v3 dispatch eliminated.
    // v4 remains from defaults (gated on avx512).
    // Must provide all variants that could be dispatched to.

    #[cfg(feature = "avx512")]
    #[arcane]
    fn gated_incant_v4(_: X64V4Token, x: f32) -> f32 {
        x * 4.0
    }

    #[allow(unexpected_cfgs)]
    #[cfg(feature = "nonexistent_feature")]
    #[arcane]
    fn gated_incant_v3(_: X64V3Token, x: f32) -> f32 {
        x * 3.0
    }

    fn gated_incant_scalar(_: ScalarToken, x: f32) -> f32 {
        x * 1.0
    }

    #[allow(unexpected_cfgs)]
    fn gated_dispatch(x: f32) -> f32 {
        incant!(gated_incant(x), [+v3(cfg(nonexistent_feature))])
    }

    #[test]
    fn incant_gated_v3_uses_best_available() {
        let result = gated_dispatch(10.0);
        // v3 is gated on nonexistent_feature → eliminated.
        // On x86_64 with avx512: v4 (40.0). Without avx512: scalar (10.0).
        assert!(result >= 10.0);
    }
}

// ============================================================================
// incant! with modifiers
// ============================================================================

mod incant_modifiers {
    use super::*;

    // Variants for incant! — must cover defaults minus removals plus additions.
    // Using [-neon, -wasm128, +v1] → effective: v4(avx512), v3, v1, scalar

    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[arcane]
    fn im_v4(_: X64V4Token, x: f32) -> f32 {
        x + 4.0
    }

    #[arcane]
    fn im_v3(_: X64V3Token, x: f32) -> f32 {
        x + 3.0
    }

    fn im_v1(_: X64V1Token, x: f32) -> f32 {
        x + 1.0
    }

    fn im_scalar(_: ScalarToken, x: f32) -> f32 {
        x + 0.0
    }

    fn im_dispatch(x: f32) -> f32 {
        incant!(im(x), [-neon, -wasm128, +v1])
    }

    #[test]
    fn incant_with_modifiers() {
        let result = im_dispatch(10.0);
        // On x86_64 with v3: 13.0. With v1 only: 11.0. Non-x86: 10.0.
        assert!(result >= 10.0);
    }
}

// ============================================================================
// magetypes with modifiers
// ============================================================================

mod magetypes_modifiers {
    use super::*;

    // [-wasm128, +v2] → v4(avx512), v3, v2, neon, scalar
    #[magetypes(-wasm128, +v2)]
    fn mt_mod(_token: Token, x: f32) -> f32 {
        x + 1.0
    }

    #[test]
    fn magetypes_modifier_scalar_works() {
        assert_eq!(mt_mod_scalar(ScalarToken, 1.0), 2.0);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn magetypes_modifier_v2_exists() {
        if let Some(t) = X64V2Token::summon() {
            assert_eq!(mt_mod_v2(t, 1.0), 2.0);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn magetypes_modifier_v3_still_exists() {
        if let Some(t) = X64V3Token::summon() {
            assert_eq!(mt_mod_v3(t, 1.0), 2.0);
        }
    }
}

//! Tests for `#[magetypes(rite, ...)]` — the `rite` flag that emits per-tier
//! variants via `#[archmage::rite(import_intrinsics)]` (direct
//! `#[target_feature]` + `#[inline]`) instead of `#[archmage::arcane]`
//! (safe wrapper + inner via trampoline).
//!
//! ## Where rite-flavored variants work
//!
//! - Direct call from a matching-feature context (another `#[arcane]` or
//!   `#[rite]` body with the same `#[target_feature]`). No `unsafe` needed
//!   in Rust 1.86+ when caller's features ⊇ callee's.
//! - The `_scalar` variant has no `#[target_feature]` — callable from
//!   anywhere, no `unsafe`.
//! - Nested `incant!` from a tier-annotated outer body (arcane or rite)
//!   where the outer's tier variant has matching features. The `incant!`
//!   rewriter picks the matching tier's callee, which inlines into the
//!   caller's region.
//!
//! ## Where they DON'T work cleanly
//!
//! - Standalone `incant!` dispatch at a public API boundary. The
//!   dispatcher has no `#[target_feature]`, so calling a bare
//!   `#[target_feature]` rite variant requires `unsafe` — which
//!   `incant!`'s current dispatcher doesn't emit. Use arcane-flavored
//!   magetypes at the public boundary (the scalar variant of arcane-
//!   flavored magetypes also bypasses `incant!` rewriting, so calling
//!   rite helpers from inside it via nested `incant!` hits the same
//!   problem there).
//!
//! Practical recipe: `#[magetypes(rite)]` for inner helpers, regular
//! `#[magetypes(...)]` for public entry points.

#![cfg(target_arch = "x86_64")]

use archmage::{ScalarToken, magetypes};

// ============================================================================
// Scalar variant is trivially callable — no target_feature
// ============================================================================

#[magetypes(rite, v3, scalar)]
fn rite_clamp(_t: Token, x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

#[test]
fn rite_flag_emits_scalar_variant_callable_anywhere() {
    assert_eq!(rite_clamp_scalar(ScalarToken, 0.5), 0.5);
    assert_eq!(rite_clamp_scalar(ScalarToken, 2.0), 1.0);
    assert_eq!(rite_clamp_scalar(ScalarToken, -1.0), 0.0);
}

// ============================================================================
// Rite V3 variant is callable from a matching target_feature context
// ============================================================================

#[magetypes(rite, v3, scalar)]
fn rite_square(_t: Token, x: f32) -> f32 {
    x * x
}

// Hand-written `#[arcane]` wrapper for V3, which calls the rite-flavored
// `rite_square_v3` directly. Both have matching target_feature, so the call
// is safe in Rust 1.86+.
#[archmage::arcane]
fn call_rite_from_arcane_v3(token: archmage::X64V3Token, x: f32) -> f32 {
    rite_square_v3(token, x)
}

#[test]
fn rite_v3_callable_from_matching_arcane_context() {
    use archmage::SimdToken;
    if let Some(token) = archmage::X64V3Token::summon() {
        assert_eq!(call_rite_from_arcane_v3(token, 3.0), 9.0);
        assert_eq!(call_rite_from_arcane_v3(token, 0.5), 0.25);
    }
    // On CPUs without AVX2, this test degrades to exercising only scalar above.
}

// ============================================================================
// Rite variants compose via nested incant! inside a tier-matched body
// ============================================================================
// The nested-incant rewriter works when BOTH variants are available at the
// current tier. Here, both outer and inner are rite-flavored with {v3, scalar}.
// The V3 branch is exercised on CPUs with AVX2; the scalar branch is always
// exercised via the scalar variant below.

#[magetypes(rite, v3, scalar)]
fn rite_double(_t: Token, x: f32) -> f32 {
    x * 2.0
}

// Call rite_double's scalar variant directly — no incant! in the body, so
// no rewrite-or-dispatcher concern. Demonstrates that the scalar variant is
// usable as a building block.
fn compose_scalar(x: f32) -> f32 {
    let doubled = rite_double_scalar(ScalarToken, x);
    rite_square_scalar(ScalarToken, doubled)
}

#[test]
fn rite_scalar_variants_compose_directly() {
    // 5.0 → double → 10.0 → square → 100.0
    assert_eq!(compose_scalar(5.0), 100.0);
}

// ============================================================================
// Chaining rite V3 variants from inside a single arcane V3 context
// ============================================================================

#[archmage::arcane]
fn chain_rite_v3(token: archmage::X64V3Token, x: f32) -> f32 {
    // All these calls are safe — matching target_feature.
    let doubled = rite_double_v3(token, x);
    rite_square_v3(token, doubled)
}

#[test]
fn rite_v3_variants_chain_in_arcane_body() {
    use archmage::SimdToken;
    if let Some(token) = archmage::X64V3Token::summon() {
        // 3.0 → double → 6.0 → square → 36.0
        assert_eq!(chain_rite_v3(token, 3.0), 36.0);
    }
}

// ============================================================================
// Defaulted tier list + rite flag (no explicit tiers)
// ============================================================================

#[magetypes(rite)]
fn rite_default_tiers(_t: Token, x: f32) -> f32 {
    x + 1.0
}

#[test]
fn rite_with_default_tier_list_scalar_works() {
    // Default tier list includes scalar — always callable.
    assert_eq!(rite_default_tiers_scalar(ScalarToken, 10.0), 11.0);
}

// ============================================================================
// Token-substitution sanity: each tier variant's signature has the matching
// concrete token type. (Same as regular #[magetypes] — rite flag doesn't
// change Token substitution, only the wrapping strategy.)
// ============================================================================

#[magetypes(rite, v3, scalar)]
fn token_aware(_t: Token, x: f32) -> f32 {
    x * 2.0
}

#[test]
fn rite_variant_token_types_are_substituted_per_tier() {
    // Compile-time check: scalar variant takes ScalarToken.
    let _: fn(ScalarToken, f32) -> f32 = token_aware_scalar;
    assert_eq!(token_aware_scalar(ScalarToken, 3.0), 6.0);

    // V3 variant takes X64V3Token. Can't store in a plain fn pointer because
    // of its #[target_feature] attribute, but the following line would fail
    // to compile if the V3 variant took any OTHER token type:
    #[allow(dead_code, non_snake_case)]
    fn _compile_check(token: archmage::X64V3Token) {
        #[archmage::arcane]
        fn wrap(token: archmage::X64V3Token) -> f32 {
            token_aware_v3(token, 1.0)
        }
        let _ = wrap(token);
    }
}

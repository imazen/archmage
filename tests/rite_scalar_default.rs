//! Tests for `#[rite(scalar)]` and `#[rite(default)]` tiers.
//!
//! These are the two tokenless/feature-less tiers for `#[rite]`:
//!
//! - `scalar` — tokenful. Signature must take `ScalarToken` (or be `_token: ScalarToken`).
//!   Generates a variant with `#[inline]` but no `#[target_feature]`. Slots into
//!   `incant!`'s suffix convention as `_scalar`, receives the token pass-through
//!   that `incant!` rewriting emits from inside a `#[magetypes]`/`#[autoversion]`
//!   scalar variant.
//!
//! - `default` — tokenless. Signature takes no token. Generates a variant with
//!   `#[inline]` but no `#[target_feature]`, no cfg-gating, no imports. Slots in
//!   as `_default` for the fallback path when `incant!` doesn't need to pass a
//!   token (e.g., top-level public dispatch).
//!
//! Both tiers:
//! - Omit `#[target_feature]` (no features to enable)
//! - Omit cfg-gating (always compiled on every architecture)
//! - Share the `_<suffix>` naming convention so `incant!` can resolve by suffix

use archmage::{ScalarToken, rite};

// ============================================================================
// Single-tier #[rite(scalar)] — tokenful
// ============================================================================

#[rite(scalar)]
fn scalar_only_sum(_t: ScalarToken, data: &[f32]) -> f32 {
    data.iter().sum()
}

#[rite(_scalar)]
fn scalar_underscore_sum(_t: ScalarToken, data: &[f32]) -> f32 {
    data.iter().sum()
}

#[test]
fn single_scalar_tokenful() {
    let result = scalar_only_sum(ScalarToken, &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(result, 10.0);
}

#[test]
fn single_scalar_underscore_accepted() {
    // `_scalar` should be equivalent to `scalar` — the parser strips the prefix.
    let result = scalar_underscore_sum(ScalarToken, &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(result, 10.0);
}

// ============================================================================
// Single-tier #[rite(default)] — tokenless
// ============================================================================

#[rite(default)]
fn default_only_sum(data: &[f32]) -> f32 {
    data.iter().sum()
}

#[rite(_default)]
fn default_underscore_sum(data: &[f32]) -> f32 {
    data.iter().sum()
}

#[test]
fn single_default_tokenless() {
    let result = default_only_sum(&[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(result, 10.0);
}

#[test]
fn single_default_underscore_accepted() {
    let result = default_underscore_sum(&[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(result, 10.0);
}

// ============================================================================
// Multi-tier with scalar fallback
// ============================================================================

// A helper that's usable in both tier contexts and scalar fallbacks.
// In a multi-tier #[rite] with scalar, each variant has the SAME signature —
// rite doesn't substitute `Token`. So the signature must already accept
// `ScalarToken` (which all the other tiers' tokens also... don't implement).
// Practically: use tokenless multi-tier with a separate scalar variant that
// takes ScalarToken. Test that both exist and are callable.

#[rite(v3, neon, wasm128)]
fn multi_tokenless_square(data: &[f32; 4]) -> f32 {
    data.iter().map(|x| x * x).sum()
}

#[rite(scalar)]
fn multi_tokenless_square_scalar(_t: ScalarToken, data: &[f32; 4]) -> f32 {
    data.iter().map(|x| x * x).sum()
}

#[test]
fn tokenless_multi_tier_scalar_variant_callable() {
    let result = multi_tokenless_square_scalar(ScalarToken, &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(result, 30.0);
}

// ============================================================================
// Multi-tier WITH scalar in tier list — the key new capability
// ============================================================================

// This is the form the user's correction enables: `scalar` in a multi-tier rite
// list. All variants share the signature; the scalar variant just omits the
// `#[target_feature]` attribute and cfg-guard.

#[rite(v3, neon, wasm128, scalar)]
fn multi_with_scalar(token: ScalarToken, data: &[f32; 4]) -> f32 {
    let _ = token; // silence unused on non-scalar tiers
    data.iter().sum()
}

#[test]
fn multi_with_scalar_variant_callable() {
    // Only the _scalar variant is callable without a matching target_feature
    // context. The other tiers' variants exist but are cfg-gated.
    let result = multi_with_scalar_scalar(ScalarToken, &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(result, 10.0);
}

// ============================================================================
// Multi-tier WITH default in tier list
// ============================================================================

#[rite(v3, neon, wasm128, default)]
fn multi_with_default(data: &[f32; 4]) -> f32 {
    data.iter().sum()
}

#[test]
fn multi_with_default_variant_callable() {
    // _default variant is tokenless and callable from any context.
    let result = multi_with_default_default(&[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(result, 10.0);
}

// Note: the `_v3`, `_neon`, etc. variants are `#[target_feature]`-annotated,
// which makes them a different type from plain `fn` pointers — you can't cast
// or store them in `fn(...) -> ...`. That they compile at all is the check;
// they're called via direct name matching or via `incant!` rewriting, not via
// fn-pointer indirection.

// ============================================================================
// Multi-tier with BOTH scalar and default
// ============================================================================

// Scalar (tokenful) + default (tokenless) are distinct variants. Both should
// coexist. The signature must take ScalarToken because scalar requires it —
// default will ignore it since it's tokenless... wait, no, default takes the
// SAME signature. Since rite doesn't substitute, the signature is fixed across
// all tiers. So if the signature has `_t: ScalarToken`, default will also have
// `_t: ScalarToken` in its variant. That's fine — default is tokenless only in
// the sense of "no target_feature required to call me," not signature-wise.

#[rite(v3, neon, wasm128, scalar, default)]
fn multi_with_both(_t: ScalarToken, data: &[f32; 4]) -> f32 {
    data.iter().sum()
}

#[test]
fn multi_with_both_scalar_callable() {
    let result = multi_with_both_scalar(ScalarToken, &[5.0, 10.0, 15.0, 20.0]);
    assert_eq!(result, 50.0);
}

#[test]
fn multi_with_both_default_callable() {
    // _default has same signature here (takes ScalarToken because that's what
    // the user wrote). Difference is the variant has no target_feature.
    let result = multi_with_both_default(ScalarToken, &[5.0, 10.0, 15.0, 20.0]);
    assert_eq!(result, 50.0);
}

// ============================================================================
// Order of tiers shouldn't matter for correctness
// ============================================================================

#[rite(scalar, v3, neon)]
fn order_scalar_first(_t: ScalarToken, x: i32) -> i32 {
    x * 2
}

#[test]
fn tier_order_does_not_matter() {
    assert_eq!(order_scalar_first_scalar(ScalarToken, 21), 42);
}

// ============================================================================
// Stable suffix convention — the key claim for incant! routing
// ============================================================================

// If this compiles, the naming is `fn_scalar` and `fn_default` — exactly what
// `incant!`'s suffix matcher expects.

#[rite(v3, neon, wasm128, scalar, default)]
fn suffix_convention_test(_t: ScalarToken, x: f32) -> f32 {
    x
}

#[test]
fn suffix_names_match_convention() {
    // Calling by the expected suffixed name is a compile-time check that the
    // macro emitted the right identifiers. Direct calls work for scalar and
    // default (no target-feature context needed); the x86/neon/wasm variants
    // are cfg-gated and/or target_feature-guarded.
    assert_eq!(suffix_convention_test_scalar(ScalarToken, 1.0), 1.0);
    assert_eq!(suffix_convention_test_default(ScalarToken, 2.0), 2.0);
}

// ============================================================================
// Return types, references, and mutability all work through scalar/default
// ============================================================================

#[rite(scalar)]
fn scalar_mutates(_t: ScalarToken, out: &mut Vec<f32>, n: f32) {
    out.push(n);
}

#[rite(default)]
fn default_mutates(out: &mut Vec<f32>, n: f32) {
    out.push(n);
}

#[test]
fn scalar_and_default_support_mutation() {
    let mut v = Vec::new();
    scalar_mutates(ScalarToken, &mut v, 1.0);
    default_mutates(&mut v, 2.0);
    assert_eq!(v, vec![1.0, 2.0]);
}

// ============================================================================
// Generic scalar rite — works with traits
// ============================================================================

#[rite(scalar)]
fn scalar_generic<T: Copy + core::ops::Add<Output = T> + Default>(
    _t: ScalarToken,
    data: &[T],
) -> T {
    let mut acc = T::default();
    for &x in data {
        acc = acc + x;
    }
    acc
}

#[test]
fn scalar_rite_with_generics() {
    assert_eq!(scalar_generic::<f32>(ScalarToken, &[1.0, 2.0, 3.0]), 6.0);
    assert_eq!(scalar_generic::<i32>(ScalarToken, &[10, 20, 30]), 60);
}

// ============================================================================
// Note: nested-incant routing through multi-tier #[rite]
// ============================================================================
// Multi-tier `#[rite]` copies the user's signature verbatim across tiers (no
// `Token` substitution per tier). That limits its use with `incant!` rewriting
// to tokenless helpers or direct-name calls from matching-feature contexts.
// For nested-incant routing through a per-tier family, use `#[magetypes]` —
// or `#[magetypes(rite)]` (see tests/magetypes_rite_flag.rs) which combines
// magetypes' per-tier Token substitution with rite-style direct
// `#[target_feature]` wrapping.

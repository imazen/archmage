//! Combinatorial `incant!` call-chain tests across every macro kind.
//!
//! Threads `incant!` through chains that mix `#[arcane]`, `#[rite]`,
//! `#[autoversion]`, `#[magetypes]`, and plain functions — to surface
//! ergonomics gaps (does it compile? what token plumbing is required?) and
//! perf characteristics (does the chain inline, or hit a `#[target_feature]`
//! boundary?).
//!
//! Lives in `magetypes/tests` (not `archmage/tests`) because `#[magetypes]`
//! with the `f32x8` types needs the `magetypes` crate, and `magetypes`
//! depends on `archmage` — the reverse dependency can't exist.
//!
//! Every chain computes `sum(INPUT) == 36.0`, so one `for_each_token_permutation`
//! run proves correctness on every tier the host supports.
//!
//! ## Findings
//!
//! ### `incant!` dispatch TARGETS — what can be a `_v3`/`_neon`/`_scalar` variant?
//! `incant!` calls `variant(Token, args…)` from a **cold** (non-`target_feature`)
//! context — it prepends the matching concrete token. So a target must both
//! *accept* that token AND be *safe to call from cold code*.
//!
//! | Variant kind | Works as cold `incant!` target? | Why |
//! |---|---|---|
//! | `#[arcane] fn f_v3(_: X64V3Token, …)` | ✅ | safe wrapper + token-based, name carries the suffix |
//! | `#[magetypes(v3,neon,…)] fn f(token: Token,…)` | ✅ | generates safe-wrapped `f_v3(X64V3Token,…)` etc. |
//! | plain `fn f_scalar(_: ScalarToken,…)` | ✅ | no target_feature → always safe; the scalar fallback |
//! | `#[autoversion] fn f_default(…)` | ✅ (as `default` tier) | its own dispatcher; nest tokenless via `default` |
//! | `#[rite] fn f_v3(_: X64V3Token,…)` (per-tier) | ❌ | rite is raw `#[target_feature]` with **no safe wrapper** — calling from cold code requires `unsafe`, which `incant!` does not emit. **GAP**: use `#[arcane]` for cold dispatch targets. |
//! | `#[rite(v3, neon, …)] fn f(…)` (multi-tier) | ❌ | variants are **tokenless** `f_v3(…)`; `incant!` passes a token → arity mismatch (and the unsafe issue above). |
//!
//! Takeaway: **`incant!` cold-dispatch targets must be `#[arcane]`, `#[magetypes]`,
//! `#[autoversion]`(default), or plain.** `#[rite]` is an *inner* helper — reach
//! it by calling it from inside an `#[arcane]`/`#[magetypes]` variant, not via
//! `incant!`.
//!
//! ### `incant!` CALLERS — `incant!` nested inside a kind's body
//! Rewritten to a comptime direct call (no runtime summon) in any
//! `#[target_feature]` context: `#[arcane]` ✅, `#[rite]` ✅, `#[autoversion]`
//! variant ✅, `#[magetypes]` variant ✅. A plain fn body keeps runtime dispatch.
//!
//! ### Cross-kind direct chains (perf)
//! `#[arcane]`→`#[rite]`(same tier)→plain inlines into one `#[target_feature]`
//! region — no boundary. Downcast (v4→v3) is free. See `cross_kind_chains`.

#![allow(deprecated)]
#![allow(clippy::let_and_return)]
#![cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]

use archmage::prelude::*;
use magetypes::simd::f32x8;

const INPUT: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
const EXPECT: f32 = 36.0;

#[inline(always)]
fn leaf_sum(d: &[f32; 8]) -> f32 {
    d.iter().sum()
}

// ============================================================================
// 1. incant! dispatch targets — each tier variant is a different macro kind
// ============================================================================

mod dispatch_targets {
    use super::*;

    // ---- arcane variants (the canonical token-based, safe-wrapped target) ----
    #[arcane]
    fn mixed_v3(token: X64V3Token, d: &[f32; 8]) -> f32 {
        f32x8::from_array(token, *d).reduce_add()
    }
    #[arcane]
    fn mixed_neon(token: NeonToken, d: &[f32; 8]) -> f32 {
        f32x8::from_array(token, *d).reduce_add()
    }
    #[arcane]
    fn mixed_wasm128(token: Wasm128Token, d: &[f32; 8]) -> f32 {
        f32x8::from_array(token, *d).reduce_add()
    }
    fn mixed_scalar(_: ScalarToken, d: &[f32; 8]) -> f32 {
        leaf_sum(d)
    }
    pub fn mixed(d: &[f32; 8]) -> f32 {
        incant!(mixed(d), [v3, neon, wasm128, scalar])
    }

    // ---- all variants from one #[magetypes] (token-based, safe-wrapped) ----
    #[magetypes(define(f32x8), v3, neon, wasm128, scalar)]
    fn mt(token: Token, d: &[f32; 8]) -> f32 {
        f32x8::from_array(token, *d).reduce_add()
    }
    pub fn via_magetypes(d: &[f32; 8]) -> f32 {
        incant!(mt(d), [v3, neon, wasm128, scalar])
    }

    // ---- #[autoversion] nested as the `default` (tokenless) tier ----
    #[arcane]
    fn av_top_v4(token: X64V4Token, d: &[f32; 8]) -> f32 {
        let _ = token;
        leaf_sum(d)
    }
    #[autoversion(v3, neon, wasm128)]
    fn av_top_default(d: &[f32; 8]) -> f32 {
        d.iter().sum()
    }
    pub fn via_autoversion_default(d: &[f32; 8]) -> f32 {
        incant!(av_top(d), [v4(cfg(avx512)), default])
    }

    #[test]
    fn dispatch_targets_agree() {
        assert_eq!(mixed(&INPUT), EXPECT);
        assert_eq!(via_magetypes(&INPUT), EXPECT);
        assert_eq!(via_autoversion_default(&INPUT), EXPECT);
    }

    // ---- GAP DEMONSTRATION: #[rite] per-tier variants are NOT valid cold
    // incant! targets. The block below is the natural-looking attempt; it does
    // not compile (E0133: calling a #[target_feature] rite fn from cold code is
    // unsafe, and incant! emits no unsafe). Kept as documentation.
    //
    //   #[rite(import_intrinsics)]
    //   fn r_v3(token: X64V3Token, d: &[f32; 8]) -> f32 {
    //       f32x8::from_array(token, *d).reduce_add()
    //   }
    //   fn r_scalar(_: ScalarToken, d: &[f32; 8]) -> f32 { super::leaf_sum(d) }
    //   pub fn via_rite(d: &[f32; 8]) -> f32 {
    //       incant!(r(d), [v3, scalar])   // ← E0133: r_v3 call requires unsafe
    //   }
    //
    // The supported way to reach a #[rite] helper is from inside an #[arcane]
    // (or #[magetypes]) variant — see `cross_kind_chains::chain_arcane_to_rite`.
}

// ============================================================================
// 2. incant! nested inside each kind's body (comptime rewrite to direct call)
// ============================================================================

mod incant_nested_in_each {
    use super::*;

    #[arcane]
    fn leaf_v3(token: X64V3Token, d: &[f32; 8]) -> f32 {
        f32x8::from_array(token, *d).reduce_add()
    }
    #[arcane]
    fn leaf_neon(token: NeonToken, d: &[f32; 8]) -> f32 {
        f32x8::from_array(token, *d).reduce_add()
    }
    #[arcane]
    fn leaf_wasm128(token: Wasm128Token, d: &[f32; 8]) -> f32 {
        f32x8::from_array(token, *d).reduce_add()
    }
    fn leaf_scalar(_: ScalarToken, d: &[f32; 8]) -> f32 {
        leaf_sum(d)
    }

    // incant! inside #[arcane] → comptime direct call.
    #[arcane]
    pub fn caller_arcane_v3(_token: X64V3Token, d: &[f32; 8]) -> f32 {
        incant!(leaf(d), [v3, neon, wasm128, scalar])
    }

    // incant! inside token-based #[rite].
    #[rite(import_intrinsics)]
    pub fn caller_rite_v3(_token: X64V3Token, d: &[f32; 8]) -> f32 {
        incant!(leaf(d), [v3, neon, wasm128, scalar])
    }

    // incant! inside #[autoversion] body.
    #[autoversion(v3, neon, wasm128)]
    pub fn caller_autoversion(d: &[f32; 8]) -> f32 {
        incant!(leaf(d), [v3, neon, wasm128, scalar])
    }

    // incant! inside a #[magetypes] variant body.
    #[magetypes(v3, neon, wasm128, scalar)]
    fn caller_magetypes(token: Token, d: &[f32; 8]) -> f32 {
        let _ = token;
        incant!(leaf(d), [v3, neon, wasm128, scalar])
    }
    pub fn caller_magetypes_dispatch(d: &[f32; 8]) -> f32 {
        incant!(caller_magetypes(d), [v3, neon, wasm128, scalar])
    }

    #[test]
    fn nested_incant_in_each_kind() {
        // arcane/rite with X64V3Token only exist on x86_64.
        #[cfg(target_arch = "x86_64")]
        if let Some(t) = X64V3Token::summon() {
            assert_eq!(caller_arcane_v3(t, &INPUT), EXPECT);
            // rite has no safe wrapper → entering it from cold code is unsafe.
            // The summon() check above proves the v3 features are present.
            assert_eq!(unsafe { caller_rite_v3(t, &INPUT) }, EXPECT);
        }
        assert_eq!(caller_autoversion(&INPUT), EXPECT);
        assert_eq!(caller_magetypes_dispatch(&INPUT), EXPECT);
    }
}

// ============================================================================
// 3. Cross-kind direct call chains (perf: same-tier calls inline, no boundary)
// ============================================================================

mod cross_kind_chains {
    use super::*;

    #[inline(always)]
    fn plain_helper(d: &[f32; 8]) -> f32 {
        leaf_sum(d)
    }

    // rite(v3) token-based → plain. The plain helper has no target_feature, so
    // it adopts the rite caller's features and inlines.
    #[rite(import_intrinsics)]
    fn chain_rite_v3(token: X64V3Token, d: &[f32; 8]) -> f32 {
        let v = f32x8::from_array(token, *d).reduce_add();
        let p = plain_helper(d); // rite → plain, inlines
        v + p - leaf_sum(d) // == v
    }

    // arcane → rite(v3) (same tier) → plain. One #[target_feature] region.
    #[arcane]
    pub fn chain_arcane_to_rite(token: X64V3Token, d: &[f32; 8]) -> f32 {
        chain_rite_v3(token, d)
    }

    // arcane(v4) → downcast to v3 → rite(v3). Downcast is free.
    #[cfg(feature = "avx512")]
    #[arcane]
    pub fn chain_v4_downcast_to_rite(token: X64V4Token, d: &[f32; 8]) -> f32 {
        chain_rite_v3(token.v3(), d)
    }

    // #[magetypes] variant → plain helper (the variant threads its concrete
    // token for SIMD work, then bottoms out in a plain fn).
    #[magetypes(define(f32x8), v3, neon, wasm128, scalar)]
    fn chain_mt(token: Token, d: &[f32; 8]) -> f32 {
        let _ = f32x8::splat(token, 0.0);
        plain_helper(d)
    }
    pub fn chain_mt_dispatch(d: &[f32; 8]) -> f32 {
        incant!(chain_mt(d), [v3, neon, wasm128, scalar])
    }

    #[test]
    fn cross_kind_chains_run() {
        #[cfg(target_arch = "x86_64")]
        if let Some(t) = X64V3Token::summon() {
            assert_eq!(chain_arcane_to_rite(t, &INPUT), EXPECT);
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        if let Some(t) = X64V4Token::summon() {
            assert_eq!(chain_v4_downcast_to_rite(t, &INPUT), EXPECT);
        }
        assert_eq!(chain_mt_dispatch(&INPUT), EXPECT);
    }
}

// ============================================================================
// 4. Permutation sweep — every cold dispatcher correct on every host tier
// ============================================================================

#[test]
fn all_chains_correct_across_token_permutations() {
    use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

    let report = for_each_token_permutation(CompileTimePolicy::Warn, |_perm| {
        assert_eq!(dispatch_targets::mixed(&INPUT), EXPECT);
        assert_eq!(dispatch_targets::via_magetypes(&INPUT), EXPECT);
        assert_eq!(dispatch_targets::via_autoversion_default(&INPUT), EXPECT);
        assert_eq!(incant_nested_in_each::caller_autoversion(&INPUT), EXPECT);
        assert_eq!(
            incant_nested_in_each::caller_magetypes_dispatch(&INPUT),
            EXPECT
        );
        assert_eq!(cross_kind_chains::chain_mt_dispatch(&INPUT), EXPECT);
    });

    assert!(report.permutations_run >= 1);
}

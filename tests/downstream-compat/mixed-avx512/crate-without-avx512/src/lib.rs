//! Crate that does NOT enable avx512 on archmage.
//! The _v4 function is behind #[cfg(feature = "avx512")] — it only exists
//! when THIS crate's avx512 feature is enabled.
//!
//! The bug: cargo unifies archmage-macros features across the workspace.
//! crate-with-avx512 enables archmage/avx512 → archmage-macros/avx512.
//! So archmage-macros sees avx512=true for ALL expansions, including ours.
//! incant! generates a v4 dispatch arm, but our _v4 function doesn't exist.
#![deny(warnings)]

use archmage::prelude::*;

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[arcane]
fn mul_v4(_token: X64V4Token, a: f32, b: f32) -> f32 { a * b }

#[cfg(target_arch = "x86_64")]
#[arcane]
fn mul_v3(_token: X64V3Token, a: f32, b: f32) -> f32 { a * b }

fn mul_scalar(_token: ScalarToken, a: f32, b: f32) -> f32 { a * b }

/// v4(avx512) means "feature-gated" — dispatch wrapped in #[cfg(feature = "avx512")].
/// If this crate's avx512 feature is off, v4 dispatch silently eliminated.
pub fn mul(a: f32, b: f32) -> f32 {
    incant!(mul(a, b), [v4(avx512), v3, scalar])
}

/// Default tiers: v4 is automatically optional (has cfg_feature).
pub fn mul_default(a: f32, b: f32) -> f32 {
    incant!(mul(a, b))
}

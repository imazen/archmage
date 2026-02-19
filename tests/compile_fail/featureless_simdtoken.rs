//! Test that SimdToken produces a clear error in #[arcane].
//!
//! SimdToken has no CPU features, so the macro can't determine which
//! #[target_feature] to enable.

use archmage::arcane;

#[arcane]
fn bad_simdtoken(token: impl archmage::SimdToken, data: &[f32]) -> f32 {
    0.0
}

fn main() {}

//! Test that unknown trait bounds produce clear errors.
//!
//! HasAvx2 was removed in 0.3.0. The macro should not silently accept it.

use archmage::arcane;

#[arcane]
fn bad_trait(token: impl HasAvx2, data: &[f32]) -> f32 {
    0.0
}

fn main() {}

//! Test that unknown generic bounds produce clear errors.
//!
//! HasFma was removed in 0.3.0. The macro should not silently accept it.

use archmage::arcane;

#[arcane]
fn bad_generic<T: HasFma>(token: T, data: &[f32]) -> f32 {
    0.0
}

fn main() {}

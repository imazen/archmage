// #[token_target_features_boundary] — descriptive alias for #[arcane]
use archmage::{token_target_features_boundary, X64V3Token};

#[token_target_features_boundary]
fn descriptive_process(token: X64V3Token, a: f32, b: f32) -> f32 {
    a + b
}

fn main() {}

// #[token_target_features] — descriptive alias for #[rite]
use archmage::{token_target_features, X64V3Token};

#[token_target_features]
fn descriptive_helper(token: X64V3Token, a: f32, b: f32) -> f32 {
    a + b
}

fn main() {}

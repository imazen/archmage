// #[rite] single-tier with concrete token — adds target_feature + inline directly
use archmage::{rite, X64V3Token};

#[rite]
fn helper(token: X64V3Token, a: f32, b: f32) -> f32 {
    a * b + a
}

fn main() {}

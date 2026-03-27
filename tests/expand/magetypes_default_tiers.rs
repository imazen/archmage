// #[magetypes] with default tier list (no explicit tiers)
use archmage::{magetypes, SimdToken};

#[magetypes]
fn process_default(token: Token, data: &[f32; 4]) -> f32 {
    let _ = token;
    data.iter().sum()
}

fn main() {}

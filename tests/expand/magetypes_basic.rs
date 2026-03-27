// #[magetypes] basic — Token replaced with concrete types per tier
use archmage::{magetypes, SimdToken};

#[magetypes(v3, neon, scalar)]
fn process(token: Token, data: &[f32; 4]) -> f32 {
    let _ = token;
    data.iter().sum()
}

fn main() {}

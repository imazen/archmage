// #[magetypes] with feature-gated tier
use archmage::{magetypes, SimdToken};

#[magetypes(v4(cfg(avx512)), v3, scalar)]
fn gated_process(token: Token, data: &[f32; 4]) -> f32 {
    let _ = token;
    data.iter().sum()
}

fn main() {}

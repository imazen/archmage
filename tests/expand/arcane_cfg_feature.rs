// #[arcane] with cfg feature gate
use archmage::{arcane, X64V3Token};

#[arcane(cfg(my_feature))]
fn gated_process(token: X64V3Token, data: &[f32; 4]) -> f32 {
    data.iter().sum()
}

fn main() {}

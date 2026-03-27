// #[arcane] with ARM NeonToken — cfg-gated to aarch64
use archmage::{arcane, NeonToken};

#[arcane]
fn process_arm(token: NeonToken, data: &[f32; 4]) -> f32 {
    let _ = token;
    data.iter().sum()
}

fn main() {}

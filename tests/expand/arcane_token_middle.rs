// #[arcane] with token in the MIDDLE of parameters
use archmage::{arcane, X64V3Token};

#[arcane]
fn process(data: &[f32; 4], token: X64V3Token, scale: f32) -> f32 {
    let _ = token;
    data.iter().map(|&x| x * scale).sum()
}

fn main() {}

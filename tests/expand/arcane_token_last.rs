// #[arcane] with token as the LAST parameter
use archmage::{arcane, X64V3Token};

#[arcane]
fn process(data: &[f32; 4], scale: f32, token: X64V3Token) -> f32 {
    let _ = token;
    data.iter().map(|&x| x * scale).sum()
}

fn main() {}

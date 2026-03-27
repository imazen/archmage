// #[arcane] sibling mode with concrete token (default expansion)
use archmage::{arcane, X64V3Token};

#[arcane]
fn process(token: X64V3Token, data: &[f32; 4]) -> f32 {
    data.iter().sum()
}

fn main() {}

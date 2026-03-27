// #[arcane] with stub — generates unreachable stub on wrong architecture
use archmage::{arcane, X64V3Token, NeonToken};

#[arcane(stub)]
fn process_x86(token: X64V3Token, data: &[f32; 4]) -> f32 {
    data.iter().sum()
}

#[arcane(stub)]
fn process_arm(token: NeonToken, data: &[f32; 4]) -> f32 {
    data.iter().sum()
}

fn main() {}

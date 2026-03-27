// #[arcane] sibling mode with trait bound (impl HasX64V2)
use archmage::{arcane, HasX64V2};

#[arcane]
fn process_v2(token: impl HasX64V2, data: &[f32; 4]) -> f32 {
    data.iter().sum()
}

fn main() {}

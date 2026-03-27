// #[arcane] with wildcard token param (`_: X64V3Token`)
use archmage::{arcane, X64V3Token};

#[arcane]
fn process(_: X64V3Token, a: f32, b: f32) -> f32 {
    a * b
}

fn main() {}

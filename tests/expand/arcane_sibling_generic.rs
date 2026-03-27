// #[arcane] sibling mode with generic type parameter
use archmage::{arcane, HasX64V2};

#[arcane]
fn process_generic<T: HasX64V2>(token: T, a: f32, b: f32) -> f32 {
    a + b
}

fn main() {}

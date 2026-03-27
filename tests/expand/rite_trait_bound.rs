// #[rite] single-tier with impl Trait bound
use archmage::{rite, HasX64V2};

#[rite]
fn helper_v2(token: impl HasX64V2, a: f32, b: f32) -> f32 {
    let _ = token;
    a + b
}

fn main() {}

// #[rite] with ARM token — cfg-gated to aarch64
use archmage::{rite, NeonToken};

#[rite]
fn arm_helper(token: NeonToken, a: f32, b: f32) -> f32 {
    let _ = token;
    a + b
}

fn main() {}

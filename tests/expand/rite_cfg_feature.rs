// #[rite] with cfg feature gate
use archmage::{rite, X64V3Token};

#[rite(cfg(my_feature))]
fn gated_helper(token: X64V3Token, a: f32) -> f32 {
    a * 2.0
}

fn main() {}

// #[autoversion] should reject concrete tokens like X64V3Token
use archmage::prelude::*;

#[archmage::autoversion]
fn process(_token: X64V3Token, data: &[f32]) -> f32 {
    data.iter().sum()
}

fn main() {
    let _ = process(&[1.0]);
}

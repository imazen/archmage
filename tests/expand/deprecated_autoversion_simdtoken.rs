// #[autoversion] with SimdToken param — deprecated, stripped from dispatcher
#[allow(deprecated)]
use archmage::{autoversion, SimdToken};

#[allow(deprecated)]
#[autoversion]
fn legacy_sum(_token: SimdToken, data: &[f32; 4]) -> f32 {
    data.iter().sum()
}

fn main() {}

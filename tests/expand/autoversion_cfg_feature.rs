// #[autoversion] with cfg feature gate — full dispatch under feature, scalar-only without
use archmage::autoversion;

#[autoversion(cfg(simd_opt))]
fn gated_sum(data: &[f32; 4]) -> f32 {
    data.iter().sum()
}

fn main() {}

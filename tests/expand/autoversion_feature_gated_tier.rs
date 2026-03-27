// #[autoversion] with feature-gated tier: v4(cfg(avx512))
use archmage::autoversion;

#[autoversion(v4(cfg(avx512)), v3, neon, scalar)]
fn gated_tier_sum(data: &[f32; 4]) -> f32 {
    data.iter().sum()
}

fn main() {}

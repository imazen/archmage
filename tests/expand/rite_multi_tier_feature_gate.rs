// #[rite] multi-tier with cfg feature gate on the whole function
use archmage::rite;

#[rite(v3, neon, cfg(simd_opt))]
fn gated_compute(data: &[f32; 4]) -> f32 {
    data.iter().sum()
}

fn main() {}

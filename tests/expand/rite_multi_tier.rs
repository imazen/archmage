// #[rite(v3, v4, neon)] multi-tier — generates suffixed variants
use archmage::rite;

#[rite(v3, v4, neon)]
fn compute(data: &[f32; 4]) -> f32 {
    data.iter().sum()
}

fn main() {}

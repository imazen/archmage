// #[rite(v3, neon, stub)] multi-tier with stubs on wrong arch
use archmage::rite;

#[rite(v3, neon, stub)]
fn compute_stub(data: &[f32; 4]) -> f32 {
    data.iter().sum()
}

fn main() {}

// #[autoversion(v3, neon, scalar)] — explicit tier list
use archmage::autoversion;

#[autoversion(v3, neon, scalar)]
fn process_explicit(data: &[f32; 4]) -> f32 {
    data.iter().sum()
}

fn main() {}

// #[autoversion] with tier modifiers — add/remove from defaults
use archmage::autoversion;

#[autoversion(+arm_v2, -wasm128)]
fn with_modifiers(data: &[f32; 4]) -> f32 {
    data.iter().sum()
}

fn main() {}

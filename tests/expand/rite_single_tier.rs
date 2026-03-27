// #[rite(v3)] single-tier by name — tokenless, target_feature from tier name
use archmage::rite;

#[rite(v3)]
fn helper_v3(a: f32, b: f32) -> f32 {
    a * b + a
}

fn main() {}

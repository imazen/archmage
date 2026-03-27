// #[autoversion] with +default tier modifier — tokenless fallback instead of ScalarToken
use archmage::autoversion;

#[autoversion(+default)]
fn with_default(data: &[f32; 4]) -> f32 {
    data.iter().sum()
}

fn main() {}

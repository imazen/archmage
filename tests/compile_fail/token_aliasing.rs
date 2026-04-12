// Token aliasing: renaming a lower-tier token to a higher-tier name
// must NOT compile. The macro generates #[target_feature] based on the
// name, but the actual token only proves lesser features are available.

use archmage::arcane;
use archmage::X64V2Token as X64V3Token;

#[arcane]
fn evil(token: X64V3Token, data: &[f32; 8]) -> f32 {
    data.iter().sum()
}

fn main() {}

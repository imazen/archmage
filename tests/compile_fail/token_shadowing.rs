// Token shadowing: a local struct with the same name as an archmage token
// must NOT compile when used with #[arcane]. The assert_archmage_token check
// catches this because the shadowed type doesn't implement SimdToken (sealed).

use archmage::arcane;

// Shadow the real X64V3Token with a local imposter
#[derive(Clone, Copy)]
struct X64V3Token;

#[arcane]
fn evil(_token: X64V3Token, data: &[f32; 8]) -> f32 {
    data.iter().sum()
}

fn main() {}

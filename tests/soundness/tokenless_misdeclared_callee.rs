use archmage::{incant, rite, X64V2Token};
// The name claims V2 but the actual function requires AVX2. The generated
// constructor cannot make a direct call to this stronger function safe.
#[target_feature(enable = "avx2")]
fn helper_v2(_token: X64V2Token) {}
#[rite(v2)]
fn outer() { incant!(helper(), [v2, -scalar]); }
fn main() {}

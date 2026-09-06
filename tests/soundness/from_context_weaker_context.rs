// A caller whose feature context is a *subset* of the tier cannot construct
// the proof: v1 (SSE/SSE2) does not imply v3 (AVX2/FMA/BMI).
use archmage::{X64V3Token, rite};

#[rite(v1)]
fn weaker() -> X64V3Token {
    X64V3Token::from_context()
}

fn main() {}

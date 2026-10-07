//! #[rite] in token-parameter mode takes its features from one token. With
//! two, name the tier instead (issue #122).
#![forbid(unsafe_code)]
use archmage::prelude::*;

#[rite]
fn probe(_s: ScalarToken, _v: X64V3Token) -> X64V3Token {
    X64V3Token::from_context()
}

fn main() {}

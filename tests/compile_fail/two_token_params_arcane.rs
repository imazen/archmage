//! #[arcane] takes its features from one token parameter. With two, the old
//! rule picked whichever came first; now it says so (issue #122).
#![forbid(unsafe_code)]
use archmage::prelude::*;

#[arcane]
fn probe(_s: ScalarToken, _v: X64V3Token) -> X64V3Token {
    X64V3Token::from_context()
}

fn main() {}

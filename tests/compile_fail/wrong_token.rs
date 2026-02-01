//! Test that using the wrong token type fails to compile.

use archmage::SimdToken;
use archmage::tokens::x86::{X64V2Token, X64V3Token};

fn requires_x64v3(_token: X64V3Token) {}

fn main() {
    // This should fail: X64V2Token cannot be used where X64V3Token is required
    if let Some(v2) = X64V2Token::try_new() {
        requires_x64v3(v2);
    }
}

//! Test that using the wrong token type fails to compile.

use archmage::SimdToken;
use archmage::tokens::x86::{Sse42Token, Avx2Token};

fn requires_avx2(_token: Avx2Token) {}

fn main() {
    // This should fail: SSE4.2 token cannot be used where AVX2 is required
    if let Some(sse42) = Sse42Token::try_new() {
        requires_avx2(sse42);
    }
}

//! Test that safe-simd wrappers reject wrong tokens at compile time.

use archmage::SimdToken;
use archmage::tokens::x86::{Sse2Token, AvxToken};

fn main() {
    // This should fail: SSE2 token cannot be used for AVX functions
    if let Some(sse2) = Sse2Token::try_new() {
        let data = [1.0f32; 8];
        // _mm256_loadu_ps requires AvxToken, not Sse2Token
        let _ = archmage::mem::avx::_mm256_loadu_ps(sse2, &data);
    }
}

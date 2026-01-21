//! Test that safe-simd wrappers reject wrong tokens at compile time.

use archmage::SimdToken;
use archmage::tokens::x86::Sse42Token;

fn main() {
    // This should fail: SSE4.2 token cannot be used for AVX functions
    if let Some(sse42) = Sse42Token::try_new() {
        let data = [1.0f32; 8];
        // _mm256_loadu_ps requires AvxToken, not Sse42Token
        let _ = archmage::mem::avx::_mm256_loadu_ps(sse42, &data);
    }
}

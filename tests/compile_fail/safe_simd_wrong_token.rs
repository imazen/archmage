//! Test that safe-simd wrappers reject wrong tokens at compile time.

use archmage::SimdToken;
use archmage::tokens::x86::Sse41Token;

fn main() {
    // This should fail: SSE4.1 token cannot be used for AVX functions
    if let Some(sse41) = Sse41Token::try_new() {
        let data = [1.0f32; 8];
        // _mm256_loadu_ps requires an AVX-capable token (Has256BitSimd), not Sse41Token
        let _ = archmage::mem::avx::_mm256_loadu_ps(sse41, &data);
    }
}

//! Test: V4 code WITHOUT import_intrinsics, archmage WITHOUT avx512 feature.
//!
//! Uses only value-based intrinsics from core::arch (safe in #[target_feature] context).
//! No safe memory ops from safe_unaligned_simd needed.
//! Expected: compiles and works fine.
#![deny(warnings)]

use archmage::prelude::*;

/// V4 function using only value intrinsics — no memory ops.
/// Uses raw core::arch intrinsics (unsafe for memory, safe for values).
#[cfg(target_arch = "x86_64")]
#[arcane]
pub fn v4_add_values(_token: X64V4Token, a: core::arch::x86_64::__m512, b: core::arch::x86_64::__m512) -> core::arch::x86_64::__m512 {
    core::arch::x86_64::_mm512_add_ps(a, b)
}

/// V4 function that demonstrates the token + arcane work without any feature.
#[cfg(target_arch = "x86_64")]
#[arcane]
pub fn v4_setzero(_token: X64V4Token) -> core::arch::x86_64::__m512 {
    // _mm512_setzero_ps is a value intrinsic — safe in #[target_feature] context
    core::arch::x86_64::_mm512_setzero_ps()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_summon_works() {
        // V4 token exists and summon works regardless of avx512 feature
        let _result = X64V4Token::summon();
        // May be None if CPU doesn't support AVX-512, that's fine
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn v4_setzero_works() {
        if let Some(token) = X64V4Token::summon() {
            let z = v4_setzero(token);
            // Can't easily inspect __m512 without memory ops, but it compiled and ran
            let _ = z;
        }
    }
}

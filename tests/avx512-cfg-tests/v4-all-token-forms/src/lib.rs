//! Test: every way of spelling a V4/V4x/FP16 token triggers the avx512 error
//! when used with import_intrinsics AND archmage lacks the avx512 feature.
//!
//! Each test module is behind a cfg flag so we can test them individually.
//! Run: RUSTFLAGS='--cfg test_NAME' cargo check
//! Expected: compile_error about avx512 feature for each.
//!
//! Also tests that V3 and below do NOT trigger the error (they shouldn't).
#![deny(warnings)]
#![allow(dead_code, unexpected_cfgs)]

use archmage::prelude::*;

// ============================================================================
// These SHOULD error (AVX-512 tokens with import_intrinsics, no avx512 feature)
// ============================================================================

// --- Concrete token: X64V4Token ---
#[cfg(test_x64v4token)]
mod test_x64v4token {
    use super::*;
    #[arcane(import_intrinsics)]
    pub fn f(_token: X64V4Token) -> core::arch::x86_64::__m512 {
        _mm512_setzero_ps()
    }
}

// --- Alias: Avx512Token ---
#[cfg(test_avx512token)]
mod test_avx512token {
    use super::*;
    #[arcane(import_intrinsics)]
    pub fn f(_token: Avx512Token) -> core::arch::x86_64::__m512 {
        _mm512_setzero_ps()
    }
}

// --- Alias: Server64 ---
#[cfg(test_server64)]
mod test_server64 {
    use super::*;
    #[arcane(import_intrinsics)]
    pub fn f(_token: Server64) -> core::arch::x86_64::__m512 {
        _mm512_setzero_ps()
    }
}

// --- Concrete: X64V4xToken ---
#[cfg(test_x64v4xtoken)]
mod test_x64v4xtoken {
    use super::*;
    #[arcane(import_intrinsics)]
    pub fn f(_token: X64V4xToken) -> core::arch::x86_64::__m512 {
        _mm512_setzero_ps()
    }
}

// --- Alias: Avx512ModernToken (if it exists in prelude) ---
// Note: Avx512ModernToken may not be re-exported; skip if not available.

// --- Concrete: Avx512Fp16Token ---
#[cfg(test_avx512fp16)]
mod test_avx512fp16 {
    use super::*;
    #[arcane(import_intrinsics)]
    pub fn f(_token: Avx512Fp16Token) -> core::arch::x86_64::__m512 {
        _mm512_setzero_ps()
    }
}

// --- Trait bound: impl HasX64V4 ---
#[cfg(test_impl_hasx64v4)]
mod test_impl_hasx64v4 {
    use super::*;
    #[arcane(import_intrinsics)]
    pub fn f(_token: impl HasX64V4) -> core::arch::x86_64::__m512 {
        _mm512_setzero_ps()
    }
}

// --- Generic bound: T: HasX64V4 ---
#[cfg(test_generic_hasx64v4)]
mod test_generic_hasx64v4 {
    use super::*;
    #[arcane(import_intrinsics)]
    pub fn f<T: HasX64V4>(_token: T) -> core::arch::x86_64::__m512 {
        _mm512_setzero_ps()
    }
}

// --- #[rite(v4, import_intrinsics)] (tier-based) ---
#[cfg(test_rite_v4)]
mod test_rite_v4 {
    use super::*;
    #[rite(v4, import_intrinsics)]
    pub fn f() -> core::arch::x86_64::__m512 {
        _mm512_setzero_ps()
    }
}

// --- #[rite(v4, v3, import_intrinsics)] (multi-tier) ---
#[cfg(test_rite_multi)]
mod test_rite_multi {
    use super::*;
    #[rite(v4, v3, import_intrinsics)]
    pub fn f() -> f32 {
        0.0
    }
}

// ============================================================================
// These should NOT error (non-AVX-512 tokens)
// ============================================================================

// --- V3 with import_intrinsics: always fine ---
#[cfg(target_arch = "x86_64")]
#[arcane(import_intrinsics)]
pub fn v3_import(_token: X64V3Token) -> core::arch::x86_64::__m256 {
    _mm256_setzero_ps()
}

// --- V4 WITHOUT import_intrinsics: always fine ---
#[cfg(target_arch = "x86_64")]
#[arcane]
pub fn v4_no_import(_token: X64V4Token) -> core::arch::x86_64::__m512 {
    core::arch::x86_64::_mm512_setzero_ps()
}

// --- #[rite(v3, import_intrinsics)]: always fine ---
#[cfg(target_arch = "x86_64")]
#[rite(v3, import_intrinsics)]
pub fn v3_rite() -> core::arch::x86_64::__m256 {
    _mm256_setzero_ps()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn v3_import_works() {
        if let Some(token) = X64V3Token::summon() {
            let _ = v3_import(token);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn v4_no_import_works() {
        if let Some(token) = X64V4Token::summon() {
            let _ = v4_no_import(token);
        }
    }
}

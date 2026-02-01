//! Operation trait implementations for x86 tokens.
//!
//! This module provides implementations of `simd_ops` and `scalar_ops` traits
//! for x86 SIMD tokens. Only tier-level tokens are supported:
//! - `X64V2Token` (scalar fallback only, no 256-bit SIMD)
//! - `X64V3Token` (AVX2+FMA, aliases: `Desktop64`, `Avx2FmaToken`)
//! - `X64V4Token` (AVX-512, aliases: `Avx512Token`, `Server64`) [requires "avx512" feature]
//! - `Avx512ModernToken`, `Avx512Fp16Token` [requires "avx512" feature]

use super::scalar_ops::{DotProductOrScalar, HorizontalOpsOrScalar, Transpose8x8OrScalar};
use super::simd_ops::{DotProduct, HorizontalOps, Transpose8x8};
use crate::tokens::*;

// ============================================================================
// SIMD Operation Trait Implementations (simd_ops)
// ============================================================================

// Transpose8x8 for X64V3Token (AVX2+FMA)
// Note: Avx2FmaToken and Desktop64 are type aliases for X64V3Token,
// so this impl covers all three.
impl Transpose8x8 for X64V3Token {
    #[inline(always)]
    fn transpose_8x8(&self, block: &mut [f32; 64]) {
        super::transpose::transpose_8x8(*self, block)
    }

    #[inline(always)]
    fn transpose_8x8_copy(&self, input: &[f32; 64], output: &mut [f32; 64]) {
        super::transpose::transpose_8x8_copy(*self, input, output)
    }
}

// DotProduct for X64V3Token (uses FMA)
impl DotProduct for X64V3Token {
    #[inline(always)]
    fn dot_product_f32(&self, a: &[f32], b: &[f32]) -> f32 {
        super::dot_product::dot_product_f32(*self, a, b)
    }
}

// HorizontalOps for X64V3Token
impl HorizontalOps for X64V3Token {
    #[inline(always)]
    fn sum_f32(&self, data: &[f32]) -> f32 {
        super::horizontal::sum_f32_slice(*self, data)
    }

    #[inline(always)]
    fn max_f32(&self, data: &[f32]) -> f32 {
        super::horizontal::max_f32_slice(*self, data)
    }

    #[inline(always)]
    fn min_f32(&self, data: &[f32]) -> f32 {
        super::horizontal::min_f32_slice(*self, data)
    }
}

// AVX-512 tier tokens (require "avx512" feature)
#[cfg(feature = "avx512")]
mod avx512_impls {
    use super::*;

    // Transpose8x8 for X64V4Token (delegates to v3 implementation)
    // Note: Avx512Token and Server64 are type aliases for X64V4Token.
    impl Transpose8x8 for X64V4Token {
        #[inline(always)]
        fn transpose_8x8(&self, block: &mut [f32; 64]) {
            self.v3().transpose_8x8(block)
        }

        #[inline(always)]
        fn transpose_8x8_copy(&self, input: &[f32; 64], output: &mut [f32; 64]) {
            self.v3().transpose_8x8_copy(input, output)
        }
    }

    // DotProduct for X64V4Token (delegates to v3, which has FMA)
    impl DotProduct for X64V4Token {
        #[inline(always)]
        fn dot_product_f32(&self, a: &[f32], b: &[f32]) -> f32 {
            super::DotProduct::dot_product_f32(&self.v3(), a, b)
        }
    }

    // HorizontalOps for X64V4Token (delegates to v3)
    impl HorizontalOps for X64V4Token {
        #[inline(always)]
        fn sum_f32(&self, data: &[f32]) -> f32 {
            super::HorizontalOps::sum_f32(&self.v3(), data)
        }

        #[inline(always)]
        fn max_f32(&self, data: &[f32]) -> f32 {
            super::HorizontalOps::max_f32(&self.v3(), data)
        }

        #[inline(always)]
        fn min_f32(&self, data: &[f32]) -> f32 {
            super::HorizontalOps::min_f32(&self.v3(), data)
        }
    }
}

// ============================================================================
// Scalar Fallback Trait Implementations (scalar_ops)
// ============================================================================
// Tokens without optimized SIMD implementations use default scalar methods.
// Tokens with SIMD override to use the optimized implementations.

// X64V2Token: no 256-bit SIMD, uses scalar defaults
impl Transpose8x8OrScalar for X64V2Token {}
impl DotProductOrScalar for X64V2Token {}
impl HorizontalOpsOrScalar for X64V2Token {}

// X64V3Token: has 256-bit SIMD, overrides with optimized code
// (also covers Avx2FmaToken and Desktop64 aliases)
impl Transpose8x8OrScalar for X64V3Token {
    #[inline(always)]
    fn transpose_8x8_or_scalar(&self, block: &mut [f32; 64]) {
        Transpose8x8::transpose_8x8(self, block)
    }
    #[inline(always)]
    fn transpose_8x8_copy_or_scalar(&self, input: &[f32; 64], output: &mut [f32; 64]) {
        Transpose8x8::transpose_8x8_copy(self, input, output)
    }
}

impl DotProductOrScalar for X64V3Token {
    #[inline(always)]
    fn dot_product_f32_or_scalar(&self, a: &[f32], b: &[f32]) -> f32 {
        DotProduct::dot_product_f32(self, a, b)
    }
}

impl HorizontalOpsOrScalar for X64V3Token {
    #[inline(always)]
    fn sum_f32_or_scalar(&self, data: &[f32]) -> f32 {
        HorizontalOps::sum_f32(self, data)
    }
    #[inline(always)]
    fn max_f32_or_scalar(&self, data: &[f32]) -> f32 {
        HorizontalOps::max_f32(self, data)
    }
    #[inline(always)]
    fn min_f32_or_scalar(&self, data: &[f32]) -> f32 {
        HorizontalOps::min_f32(self, data)
    }
}

// AVX-512 scalar_ops implementations (require "avx512" feature)
#[cfg(feature = "avx512")]
mod avx512_scalar_impls {
    use super::*;

    // X64V4Token: has 256-bit SIMD (and more), overrides with optimized code
    // (also covers Avx512Token and Server64 aliases)
    impl Transpose8x8OrScalar for X64V4Token {
        #[inline(always)]
        fn transpose_8x8_or_scalar(&self, block: &mut [f32; 64]) {
            Transpose8x8::transpose_8x8(self, block)
        }
        #[inline(always)]
        fn transpose_8x8_copy_or_scalar(&self, input: &[f32; 64], output: &mut [f32; 64]) {
            Transpose8x8::transpose_8x8_copy(self, input, output)
        }
    }

    impl DotProductOrScalar for X64V4Token {
        #[inline(always)]
        fn dot_product_f32_or_scalar(&self, a: &[f32], b: &[f32]) -> f32 {
            DotProduct::dot_product_f32(self, a, b)
        }
    }

    impl HorizontalOpsOrScalar for X64V4Token {
        #[inline(always)]
        fn sum_f32_or_scalar(&self, data: &[f32]) -> f32 {
            HorizontalOps::sum_f32(self, data)
        }
        #[inline(always)]
        fn max_f32_or_scalar(&self, data: &[f32]) -> f32 {
            HorizontalOps::max_f32(self, data)
        }
        #[inline(always)]
        fn min_f32_or_scalar(&self, data: &[f32]) -> f32 {
            HorizontalOps::min_f32(self, data)
        }
    }

    // Avx512ModernToken: delegates to v3 for all operations
    impl Transpose8x8OrScalar for Avx512ModernToken {
        #[inline(always)]
        fn transpose_8x8_or_scalar(&self, block: &mut [f32; 64]) {
            Transpose8x8::transpose_8x8(&self.v3(), block)
        }
        #[inline(always)]
        fn transpose_8x8_copy_or_scalar(&self, input: &[f32; 64], output: &mut [f32; 64]) {
            Transpose8x8::transpose_8x8_copy(&self.v3(), input, output)
        }
    }

    impl DotProductOrScalar for Avx512ModernToken {
        #[inline(always)]
        fn dot_product_f32_or_scalar(&self, a: &[f32], b: &[f32]) -> f32 {
            DotProduct::dot_product_f32(&self.v3(), a, b)
        }
    }

    impl HorizontalOpsOrScalar for Avx512ModernToken {
        #[inline(always)]
        fn sum_f32_or_scalar(&self, data: &[f32]) -> f32 {
            HorizontalOps::sum_f32(&self.v3(), data)
        }
        #[inline(always)]
        fn max_f32_or_scalar(&self, data: &[f32]) -> f32 {
            HorizontalOps::max_f32(&self.v3(), data)
        }
        #[inline(always)]
        fn min_f32_or_scalar(&self, data: &[f32]) -> f32 {
            HorizontalOps::min_f32(&self.v3(), data)
        }
    }

    // Avx512Fp16Token: delegates to v3 for all operations
    impl Transpose8x8OrScalar for Avx512Fp16Token {
        #[inline(always)]
        fn transpose_8x8_or_scalar(&self, block: &mut [f32; 64]) {
            Transpose8x8::transpose_8x8(&self.v3(), block)
        }
        #[inline(always)]
        fn transpose_8x8_copy_or_scalar(&self, input: &[f32; 64], output: &mut [f32; 64]) {
            Transpose8x8::transpose_8x8_copy(&self.v3(), input, output)
        }
    }

    impl DotProductOrScalar for Avx512Fp16Token {
        #[inline(always)]
        fn dot_product_f32_or_scalar(&self, a: &[f32], b: &[f32]) -> f32 {
            DotProduct::dot_product_f32(&self.v3(), a, b)
        }
    }

    impl HorizontalOpsOrScalar for Avx512Fp16Token {
        #[inline(always)]
        fn sum_f32_or_scalar(&self, data: &[f32]) -> f32 {
            HorizontalOps::sum_f32(&self.v3(), data)
        }
        #[inline(always)]
        fn max_f32_or_scalar(&self, data: &[f32]) -> f32 {
            HorizontalOps::max_f32(&self.v3(), data)
        }
        #[inline(always)]
        fn min_f32_or_scalar(&self, data: &[f32]) -> f32 {
            HorizontalOps::min_f32(&self.v3(), data)
        }
    }
}

//! Operation trait implementations for x86 tokens.
//!
//! This module provides implementations of `simd_ops` and `scalar_ops` traits
//! for x86 SIMD tokens.

use super::scalar_ops::{DotProductOrScalar, HorizontalOpsOrScalar, Transpose8x8OrScalar};
use super::simd_ops::{DotProduct, HorizontalOps, Transpose8x8};
use crate::tokens::x86::*;

// ============================================================================
// SIMD Operation Trait Implementations (simd_ops)
// ============================================================================

// Transpose8x8 for Avx2Token
impl Transpose8x8 for Avx2Token {
    #[inline(always)]
    fn transpose_8x8(&self, block: &mut [f32; 64]) {
        super::transpose::transpose_8x8(*self, block)
    }

    #[inline(always)]
    fn transpose_8x8_copy(&self, input: &[f32; 64], output: &mut [f32; 64]) {
        super::transpose::transpose_8x8_copy(*self, input, output)
    }
}

// Transpose8x8 for Avx2FmaToken (delegates to Avx2)
impl Transpose8x8 for Avx2FmaToken {
    #[inline(always)]
    fn transpose_8x8(&self, block: &mut [f32; 64]) {
        self.avx2().transpose_8x8(block)
    }

    #[inline(always)]
    fn transpose_8x8_copy(&self, input: &[f32; 64], output: &mut [f32; 64]) {
        self.avx2().transpose_8x8_copy(input, output)
    }
}

// Transpose8x8 for X64V3Token (delegates to Avx2)
impl Transpose8x8 for X64V3Token {
    #[inline(always)]
    fn transpose_8x8(&self, block: &mut [f32; 64]) {
        self.avx2().transpose_8x8(block)
    }

    #[inline(always)]
    fn transpose_8x8_copy(&self, input: &[f32; 64], output: &mut [f32; 64]) {
        self.avx2().transpose_8x8_copy(input, output)
    }
}

// Transpose8x8 for X64V4Token (delegates to Avx2 for now, could use AVX-512)
impl Transpose8x8 for X64V4Token {
    #[inline(always)]
    fn transpose_8x8(&self, block: &mut [f32; 64]) {
        self.avx2().transpose_8x8(block)
    }

    #[inline(always)]
    fn transpose_8x8_copy(&self, input: &[f32; 64], output: &mut [f32; 64]) {
        self.avx2().transpose_8x8_copy(input, output)
    }
}

// Transpose8x8 for Avx512Token (delegates to Avx2)
impl Transpose8x8 for Avx512Token {
    #[inline(always)]
    fn transpose_8x8(&self, block: &mut [f32; 64]) {
        self.avx2().transpose_8x8(block)
    }

    #[inline(always)]
    fn transpose_8x8_copy(&self, input: &[f32; 64], output: &mut [f32; 64]) {
        self.avx2().transpose_8x8_copy(input, output)
    }
}

// Transpose8x8 for Avx512ModernToken (delegates to Avx2)
impl Transpose8x8 for Avx512ModernToken {
    #[inline(always)]
    fn transpose_8x8(&self, block: &mut [f32; 64]) {
        self.avx2().transpose_8x8(block)
    }

    #[inline(always)]
    fn transpose_8x8_copy(&self, input: &[f32; 64], output: &mut [f32; 64]) {
        self.avx2().transpose_8x8_copy(input, output)
    }
}

// Transpose8x8 for Avx512Fp16Token (delegates to Avx2)
impl Transpose8x8 for Avx512Fp16Token {
    #[inline(always)]
    fn transpose_8x8(&self, block: &mut [f32; 64]) {
        self.avx2().transpose_8x8(block)
    }

    #[inline(always)]
    fn transpose_8x8_copy(&self, input: &[f32; 64], output: &mut [f32; 64]) {
        self.avx2().transpose_8x8_copy(input, output)
    }
}

// DotProduct for Avx2FmaToken (uses FMA)
impl DotProduct for Avx2FmaToken {
    #[inline(always)]
    fn dot_product_f32(&self, a: &[f32], b: &[f32]) -> f32 {
        super::dot_product::dot_product_f32(*self, a, b)
    }
}

// DotProduct for X64V3Token (has FMA)
impl DotProduct for X64V3Token {
    #[inline(always)]
    fn dot_product_f32(&self, a: &[f32], b: &[f32]) -> f32 {
        super::dot_product::dot_product_f32(self.avx2_fma(), a, b)
    }
}

// DotProduct for X64V4Token (has FMA)
impl DotProduct for X64V4Token {
    #[inline(always)]
    fn dot_product_f32(&self, a: &[f32], b: &[f32]) -> f32 {
        super::dot_product::dot_product_f32(self.avx2_fma(), a, b)
    }
}

// DotProduct for Avx512Token (has FMA)
impl DotProduct for Avx512Token {
    #[inline(always)]
    fn dot_product_f32(&self, a: &[f32], b: &[f32]) -> f32 {
        super::dot_product::dot_product_f32(self.avx2_fma(), a, b)
    }
}

// DotProduct for Avx512ModernToken (has FMA)
impl DotProduct for Avx512ModernToken {
    #[inline(always)]
    fn dot_product_f32(&self, a: &[f32], b: &[f32]) -> f32 {
        super::dot_product::dot_product_f32(self.avx2_fma(), a, b)
    }
}

// DotProduct for Avx512Fp16Token (has FMA)
impl DotProduct for Avx512Fp16Token {
    #[inline(always)]
    fn dot_product_f32(&self, a: &[f32], b: &[f32]) -> f32 {
        super::dot_product::dot_product_f32(self.avx2_fma(), a, b)
    }
}

// DotProduct for Avx2Token (no FMA, uses multiply+add)
impl DotProduct for Avx2Token {
    #[inline(always)]
    fn dot_product_f32(&self, a: &[f32], b: &[f32]) -> f32 {
        super::dot_product::dot_product_f32_no_fma(*self, a, b)
    }
}

// HorizontalOps for Avx2Token
impl HorizontalOps for Avx2Token {
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

// HorizontalOps for Avx2FmaToken (delegates to Avx2)
impl HorizontalOps for Avx2FmaToken {
    #[inline(always)]
    fn sum_f32(&self, data: &[f32]) -> f32 {
        self.avx2().sum_f32(data)
    }

    #[inline(always)]
    fn max_f32(&self, data: &[f32]) -> f32 {
        self.avx2().max_f32(data)
    }

    #[inline(always)]
    fn min_f32(&self, data: &[f32]) -> f32 {
        self.avx2().min_f32(data)
    }
}

// HorizontalOps for X64V3Token (delegates to Avx2)
impl HorizontalOps for X64V3Token {
    #[inline(always)]
    fn sum_f32(&self, data: &[f32]) -> f32 {
        self.avx2().sum_f32(data)
    }

    #[inline(always)]
    fn max_f32(&self, data: &[f32]) -> f32 {
        self.avx2().max_f32(data)
    }

    #[inline(always)]
    fn min_f32(&self, data: &[f32]) -> f32 {
        self.avx2().min_f32(data)
    }
}

// HorizontalOps for X64V4Token (delegates to Avx2 for now)
impl HorizontalOps for X64V4Token {
    #[inline(always)]
    fn sum_f32(&self, data: &[f32]) -> f32 {
        self.avx2().sum_f32(data)
    }

    #[inline(always)]
    fn max_f32(&self, data: &[f32]) -> f32 {
        self.avx2().max_f32(data)
    }

    #[inline(always)]
    fn min_f32(&self, data: &[f32]) -> f32 {
        self.avx2().min_f32(data)
    }
}

// HorizontalOps for Avx512Token (delegates to Avx2 for now)
impl HorizontalOps for Avx512Token {
    #[inline(always)]
    fn sum_f32(&self, data: &[f32]) -> f32 {
        self.avx2().sum_f32(data)
    }

    #[inline(always)]
    fn max_f32(&self, data: &[f32]) -> f32 {
        self.avx2().max_f32(data)
    }

    #[inline(always)]
    fn min_f32(&self, data: &[f32]) -> f32 {
        self.avx2().min_f32(data)
    }
}

// HorizontalOps for Avx512ModernToken (delegates to Avx2 for now)
impl HorizontalOps for Avx512ModernToken {
    #[inline(always)]
    fn sum_f32(&self, data: &[f32]) -> f32 {
        self.avx2().sum_f32(data)
    }

    #[inline(always)]
    fn max_f32(&self, data: &[f32]) -> f32 {
        self.avx2().max_f32(data)
    }

    #[inline(always)]
    fn min_f32(&self, data: &[f32]) -> f32 {
        self.avx2().min_f32(data)
    }
}

// HorizontalOps for Avx512Fp16Token (delegates to Avx2 for now)
impl HorizontalOps for Avx512Fp16Token {
    #[inline(always)]
    fn sum_f32(&self, data: &[f32]) -> f32 {
        self.avx2().sum_f32(data)
    }

    #[inline(always)]
    fn max_f32(&self, data: &[f32]) -> f32 {
        self.avx2().max_f32(data)
    }

    #[inline(always)]
    fn min_f32(&self, data: &[f32]) -> f32 {
        self.avx2().min_f32(data)
    }
}

// ============================================================================
// Scalar Fallback Trait Implementations (scalar_ops)
// ============================================================================
// Tokens without optimized SIMD implementations use default scalar methods.
// Tokens with SIMD override to use the optimized implementations.

// Tokens WITHOUT 256-bit SIMD use scalar defaults (SSE4.2 is baseline)
impl Transpose8x8OrScalar for Sse42Token {}
impl DotProductOrScalar for Sse42Token {}
impl HorizontalOpsOrScalar for Sse42Token {}

// AVX token (has 256-bit float but not integer ops needed for full transpose)
impl Transpose8x8OrScalar for AvxToken {}
impl DotProductOrScalar for AvxToken {}
impl HorizontalOpsOrScalar for AvxToken {}

// Tokens WITH 256-bit SIMD override with optimized code
impl Transpose8x8OrScalar for Avx2Token {
    #[inline(always)]
    fn transpose_8x8_or_scalar(&self, block: &mut [f32; 64]) {
        Transpose8x8::transpose_8x8(self, block)
    }
    #[inline(always)]
    fn transpose_8x8_copy_or_scalar(&self, input: &[f32; 64], output: &mut [f32; 64]) {
        Transpose8x8::transpose_8x8_copy(self, input, output)
    }
}

impl Transpose8x8OrScalar for Avx2FmaToken {
    #[inline(always)]
    fn transpose_8x8_or_scalar(&self, block: &mut [f32; 64]) {
        Transpose8x8::transpose_8x8(self, block)
    }
    #[inline(always)]
    fn transpose_8x8_copy_or_scalar(&self, input: &[f32; 64], output: &mut [f32; 64]) {
        Transpose8x8::transpose_8x8_copy(self, input, output)
    }
}

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

impl DotProductOrScalar for Avx2Token {
    #[inline(always)]
    fn dot_product_f32_or_scalar(&self, a: &[f32], b: &[f32]) -> f32 {
        DotProduct::dot_product_f32(self, a, b)
    }
}

impl DotProductOrScalar for Avx2FmaToken {
    #[inline(always)]
    fn dot_product_f32_or_scalar(&self, a: &[f32], b: &[f32]) -> f32 {
        DotProduct::dot_product_f32(self, a, b)
    }
}

impl DotProductOrScalar for X64V3Token {
    #[inline(always)]
    fn dot_product_f32_or_scalar(&self, a: &[f32], b: &[f32]) -> f32 {
        DotProduct::dot_product_f32(self, a, b)
    }
}

impl DotProductOrScalar for X64V4Token {
    #[inline(always)]
    fn dot_product_f32_or_scalar(&self, a: &[f32], b: &[f32]) -> f32 {
        DotProduct::dot_product_f32(self, a, b)
    }
}

impl HorizontalOpsOrScalar for Avx2Token {
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

impl HorizontalOpsOrScalar for Avx2FmaToken {
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

// AVX-512 tokens (delegate to AVX2 implementation)
impl Transpose8x8OrScalar for Avx512Token {
    #[inline(always)]
    fn transpose_8x8_or_scalar(&self, block: &mut [f32; 64]) {
        Transpose8x8::transpose_8x8(&self.avx2(), block)
    }
    #[inline(always)]
    fn transpose_8x8_copy_or_scalar(&self, input: &[f32; 64], output: &mut [f32; 64]) {
        Transpose8x8::transpose_8x8_copy(&self.avx2(), input, output)
    }
}

impl DotProductOrScalar for Avx512Token {
    #[inline(always)]
    fn dot_product_f32_or_scalar(&self, a: &[f32], b: &[f32]) -> f32 {
        DotProduct::dot_product_f32(&self.avx2_fma(), a, b)
    }
}

impl HorizontalOpsOrScalar for Avx512Token {
    #[inline(always)]
    fn sum_f32_or_scalar(&self, data: &[f32]) -> f32 {
        HorizontalOps::sum_f32(&self.avx2(), data)
    }
    #[inline(always)]
    fn max_f32_or_scalar(&self, data: &[f32]) -> f32 {
        HorizontalOps::max_f32(&self.avx2(), data)
    }
    #[inline(always)]
    fn min_f32_or_scalar(&self, data: &[f32]) -> f32 {
        HorizontalOps::min_f32(&self.avx2(), data)
    }
}

// AVX-512 Modern tokens (delegate to AVX2 implementation)
impl Transpose8x8OrScalar for Avx512ModernToken {
    #[inline(always)]
    fn transpose_8x8_or_scalar(&self, block: &mut [f32; 64]) {
        Transpose8x8::transpose_8x8(&self.avx2(), block)
    }
    #[inline(always)]
    fn transpose_8x8_copy_or_scalar(&self, input: &[f32; 64], output: &mut [f32; 64]) {
        Transpose8x8::transpose_8x8_copy(&self.avx2(), input, output)
    }
}

impl DotProductOrScalar for Avx512ModernToken {
    #[inline(always)]
    fn dot_product_f32_or_scalar(&self, a: &[f32], b: &[f32]) -> f32 {
        DotProduct::dot_product_f32(&self.avx2_fma(), a, b)
    }
}

impl HorizontalOpsOrScalar for Avx512ModernToken {
    #[inline(always)]
    fn sum_f32_or_scalar(&self, data: &[f32]) -> f32 {
        HorizontalOps::sum_f32(&self.avx2(), data)
    }
    #[inline(always)]
    fn max_f32_or_scalar(&self, data: &[f32]) -> f32 {
        HorizontalOps::max_f32(&self.avx2(), data)
    }
    #[inline(always)]
    fn min_f32_or_scalar(&self, data: &[f32]) -> f32 {
        HorizontalOps::min_f32(&self.avx2(), data)
    }
}

// AVX-512 FP16 tokens (delegate to AVX2 implementation)
impl Transpose8x8OrScalar for Avx512Fp16Token {
    #[inline(always)]
    fn transpose_8x8_or_scalar(&self, block: &mut [f32; 64]) {
        Transpose8x8::transpose_8x8(&self.avx2(), block)
    }
    #[inline(always)]
    fn transpose_8x8_copy_or_scalar(&self, input: &[f32; 64], output: &mut [f32; 64]) {
        Transpose8x8::transpose_8x8_copy(&self.avx2(), input, output)
    }
}

impl DotProductOrScalar for Avx512Fp16Token {
    #[inline(always)]
    fn dot_product_f32_or_scalar(&self, a: &[f32], b: &[f32]) -> f32 {
        DotProduct::dot_product_f32(&self.avx2_fma(), a, b)
    }
}

impl HorizontalOpsOrScalar for Avx512Fp16Token {
    #[inline(always)]
    fn sum_f32_or_scalar(&self, data: &[f32]) -> f32 {
        HorizontalOps::sum_f32(&self.avx2(), data)
    }
    #[inline(always)]
    fn max_f32_or_scalar(&self, data: &[f32]) -> f32 {
        HorizontalOps::max_f32(&self.avx2(), data)
    }
    #[inline(always)]
    fn min_f32_or_scalar(&self, data: &[f32]) -> f32 {
        HorizontalOps::min_f32(&self.avx2(), data)
    }
}

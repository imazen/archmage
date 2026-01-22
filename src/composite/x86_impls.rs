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

// ============================================================================
// Scalar Fallback Trait Implementations (scalar_ops)
// ============================================================================
// Tokens without optimized SIMD implementations use default scalar methods.
// Tokens with SIMD override to use the optimized implementations.

// Tokens WITHOUT 256-bit SIMD use scalar defaults
impl Transpose8x8OrScalar for Sse41Token {}
impl Transpose8x8OrScalar for Sse42Token {}
impl Transpose8x8OrScalar for X64V2Token {}
impl DotProductOrScalar for Sse41Token {}
impl DotProductOrScalar for Sse42Token {}
impl DotProductOrScalar for X64V2Token {}
impl HorizontalOpsOrScalar for Sse41Token {}
impl HorizontalOpsOrScalar for Sse42Token {}
impl HorizontalOpsOrScalar for X64V2Token {}

// FMA token (no 256-bit guarantee, use scalar)
impl Transpose8x8OrScalar for FmaToken {}
impl DotProductOrScalar for FmaToken {}
impl HorizontalOpsOrScalar for FmaToken {}

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
impl Transpose8x8OrScalar for Avx512fToken {
    #[inline(always)]
    fn transpose_8x8_or_scalar(&self, block: &mut [f32; 64]) {
        Transpose8x8::transpose_8x8(&self.avx2(), block)
    }
    #[inline(always)]
    fn transpose_8x8_copy_or_scalar(&self, input: &[f32; 64], output: &mut [f32; 64]) {
        Transpose8x8::transpose_8x8_copy(&self.avx2(), input, output)
    }
}

impl Transpose8x8OrScalar for Avx512bwToken {
    #[inline(always)]
    fn transpose_8x8_or_scalar(&self, block: &mut [f32; 64]) {
        Transpose8x8::transpose_8x8(&self.avx512f().avx2(), block)
    }
    #[inline(always)]
    fn transpose_8x8_copy_or_scalar(&self, input: &[f32; 64], output: &mut [f32; 64]) {
        Transpose8x8::transpose_8x8_copy(&self.avx512f().avx2(), input, output)
    }
}

impl DotProductOrScalar for Avx512fToken {
    #[inline(always)]
    fn dot_product_f32_or_scalar(&self, a: &[f32], b: &[f32]) -> f32 {
        DotProduct::dot_product_f32(&self.avx2_fma(), a, b)
    }
}

impl DotProductOrScalar for Avx512bwToken {
    #[inline(always)]
    fn dot_product_f32_or_scalar(&self, a: &[f32], b: &[f32]) -> f32 {
        DotProduct::dot_product_f32(&self.avx512f().avx2_fma(), a, b)
    }
}

impl HorizontalOpsOrScalar for Avx512fToken {
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

impl HorizontalOpsOrScalar for Avx512bwToken {
    #[inline(always)]
    fn sum_f32_or_scalar(&self, data: &[f32]) -> f32 {
        HorizontalOps::sum_f32(&self.avx512f().avx2(), data)
    }
    #[inline(always)]
    fn max_f32_or_scalar(&self, data: &[f32]) -> f32 {
        HorizontalOps::max_f32(&self.avx512f().avx2(), data)
    }
    #[inline(always)]
    fn min_f32_or_scalar(&self, data: &[f32]) -> f32 {
        HorizontalOps::min_f32(&self.avx512f().avx2(), data)
    }
}

// AVX-512 + VL tokens
impl Transpose8x8OrScalar for Avx512fVlToken {
    #[inline(always)]
    fn transpose_8x8_or_scalar(&self, block: &mut [f32; 64]) {
        Transpose8x8::transpose_8x8(&self.avx2(), block)
    }
    #[inline(always)]
    fn transpose_8x8_copy_or_scalar(&self, input: &[f32; 64], output: &mut [f32; 64]) {
        Transpose8x8::transpose_8x8_copy(&self.avx2(), input, output)
    }
}

impl DotProductOrScalar for Avx512fVlToken {
    #[inline(always)]
    fn dot_product_f32_or_scalar(&self, a: &[f32], b: &[f32]) -> f32 {
        DotProduct::dot_product_f32(&self.avx512f().avx2_fma(), a, b)
    }
}

impl HorizontalOpsOrScalar for Avx512fVlToken {
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

impl Transpose8x8OrScalar for Avx512bwVlToken {
    #[inline(always)]
    fn transpose_8x8_or_scalar(&self, block: &mut [f32; 64]) {
        Transpose8x8::transpose_8x8(&self.avx512f_vl().avx2(), block)
    }
    #[inline(always)]
    fn transpose_8x8_copy_or_scalar(&self, input: &[f32; 64], output: &mut [f32; 64]) {
        Transpose8x8::transpose_8x8_copy(&self.avx512f_vl().avx2(), input, output)
    }
}

impl DotProductOrScalar for Avx512bwVlToken {
    #[inline(always)]
    fn dot_product_f32_or_scalar(&self, a: &[f32], b: &[f32]) -> f32 {
        DotProduct::dot_product_f32(&self.avx512f_vl().avx512f().avx2_fma(), a, b)
    }
}

impl HorizontalOpsOrScalar for Avx512bwVlToken {
    #[inline(always)]
    fn sum_f32_or_scalar(&self, data: &[f32]) -> f32 {
        HorizontalOps::sum_f32(&self.avx512f_vl().avx2(), data)
    }
    #[inline(always)]
    fn max_f32_or_scalar(&self, data: &[f32]) -> f32 {
        HorizontalOps::max_f32(&self.avx512f_vl().avx2(), data)
    }
    #[inline(always)]
    fn min_f32_or_scalar(&self, data: &[f32]) -> f32 {
        HorizontalOps::min_f32(&self.avx512f_vl().avx2(), data)
    }
}

// AVX-512 VBMI2 tokens
impl Transpose8x8OrScalar for Avx512Vbmi2Token {
    #[inline(always)]
    fn transpose_8x8_or_scalar(&self, block: &mut [f32; 64]) {
        Transpose8x8::transpose_8x8(&self.avx512bw().avx512f().avx2(), block)
    }
    #[inline(always)]
    fn transpose_8x8_copy_or_scalar(&self, input: &[f32; 64], output: &mut [f32; 64]) {
        Transpose8x8::transpose_8x8_copy(&self.avx512bw().avx512f().avx2(), input, output)
    }
}

impl DotProductOrScalar for Avx512Vbmi2Token {
    #[inline(always)]
    fn dot_product_f32_or_scalar(&self, a: &[f32], b: &[f32]) -> f32 {
        DotProduct::dot_product_f32(&self.avx512bw().avx512f().avx2_fma(), a, b)
    }
}

impl HorizontalOpsOrScalar for Avx512Vbmi2Token {
    #[inline(always)]
    fn sum_f32_or_scalar(&self, data: &[f32]) -> f32 {
        HorizontalOps::sum_f32(&self.avx512bw().avx512f().avx2(), data)
    }
    #[inline(always)]
    fn max_f32_or_scalar(&self, data: &[f32]) -> f32 {
        HorizontalOps::max_f32(&self.avx512bw().avx512f().avx2(), data)
    }
    #[inline(always)]
    fn min_f32_or_scalar(&self, data: &[f32]) -> f32 {
        HorizontalOps::min_f32(&self.avx512bw().avx512f().avx2(), data)
    }
}

impl Transpose8x8OrScalar for Avx512Vbmi2VlToken {
    #[inline(always)]
    fn transpose_8x8_or_scalar(&self, block: &mut [f32; 64]) {
        Transpose8x8::transpose_8x8(&self.avx512bw_vl().avx512f_vl().avx2(), block)
    }
    #[inline(always)]
    fn transpose_8x8_copy_or_scalar(&self, input: &[f32; 64], output: &mut [f32; 64]) {
        Transpose8x8::transpose_8x8_copy(&self.avx512bw_vl().avx512f_vl().avx2(), input, output)
    }
}

impl DotProductOrScalar for Avx512Vbmi2VlToken {
    #[inline(always)]
    fn dot_product_f32_or_scalar(&self, a: &[f32], b: &[f32]) -> f32 {
        DotProduct::dot_product_f32(&self.avx512bw_vl().avx512f_vl().avx512f().avx2_fma(), a, b)
    }
}

impl HorizontalOpsOrScalar for Avx512Vbmi2VlToken {
    #[inline(always)]
    fn sum_f32_or_scalar(&self, data: &[f32]) -> f32 {
        HorizontalOps::sum_f32(&self.avx512bw_vl().avx512f_vl().avx2(), data)
    }
    #[inline(always)]
    fn max_f32_or_scalar(&self, data: &[f32]) -> f32 {
        HorizontalOps::max_f32(&self.avx512bw_vl().avx512f_vl().avx2(), data)
    }
    #[inline(always)]
    fn min_f32_or_scalar(&self, data: &[f32]) -> f32 {
        HorizontalOps::min_f32(&self.avx512bw_vl().avx512f_vl().avx2(), data)
    }
}

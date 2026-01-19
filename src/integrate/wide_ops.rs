//! Integration with the `wide` crate
//!
//! Provides token-gated operations on `wide` types like `f32x8`.

#[cfg(target_arch = "x86_64")]
use crate::tokens::x86::*;

#[cfg(feature = "wide")]
use wide::f32x8;

// ============================================================================
// Token-gated wide::f32x8 operations
// ============================================================================

#[cfg(all(feature = "wide", target_arch = "x86_64"))]
impl Avx2Token {
    /// Load wide::f32x8 from slice - token proves AVX2 generates SIMD
    #[inline(always)]
    pub fn load_f32x8_wide(self, data: &[f32; 8]) -> f32x8 {
        f32x8::from(*data)
    }

    /// Store wide::f32x8 to slice - token proves AVX2 generates SIMD
    #[inline(always)]
    pub fn store_f32x8_wide(self, data: &mut [f32; 8], v: f32x8) {
        *data = v.to_array();
    }

    /// Zero wide::f32x8
    #[inline(always)]
    pub fn zero_f32x8_wide(self) -> f32x8 {
        f32x8::ZERO
    }

    /// Splat value to all lanes
    #[inline(always)]
    pub fn splat_f32x8_wide(self, value: f32) -> f32x8 {
        f32x8::splat(value)
    }

    /// Add two wide::f32x8
    #[inline(always)]
    pub fn add_f32x8_wide(self, a: f32x8, b: f32x8) -> f32x8 {
        a + b
    }

    /// Subtract two wide::f32x8
    #[inline(always)]
    pub fn sub_f32x8_wide(self, a: f32x8, b: f32x8) -> f32x8 {
        a - b
    }

    /// Multiply two wide::f32x8
    #[inline(always)]
    pub fn mul_f32x8_wide(self, a: f32x8, b: f32x8) -> f32x8 {
        a * b
    }

    /// Divide two wide::f32x8
    #[inline(always)]
    pub fn div_f32x8_wide(self, a: f32x8, b: f32x8) -> f32x8 {
        a / b
    }

    /// Minimum of two wide::f32x8
    #[inline(always)]
    pub fn min_f32x8_wide(self, a: f32x8, b: f32x8) -> f32x8 {
        a.min(b)
    }

    /// Maximum of two wide::f32x8
    #[inline(always)]
    pub fn max_f32x8_wide(self, a: f32x8, b: f32x8) -> f32x8 {
        a.max(b)
    }

    /// Square root
    #[inline(always)]
    pub fn sqrt_f32x8_wide(self, a: f32x8) -> f32x8 {
        a.sqrt()
    }

    /// Absolute value
    #[inline(always)]
    pub fn abs_f32x8_wide(self, a: f32x8) -> f32x8 {
        a.abs()
    }

    /// Round to nearest
    #[inline(always)]
    pub fn round_f32x8_wide(self, a: f32x8) -> f32x8 {
        a.round()
    }

    /// Floor
    #[inline(always)]
    pub fn floor_f32x8_wide(self, a: f32x8) -> f32x8 {
        a.floor()
    }

    /// Ceiling
    #[inline(always)]
    pub fn ceil_f32x8_wide(self, a: f32x8) -> f32x8 {
        a.ceil()
    }

    /// Truncate towards zero
    #[inline(always)]
    pub fn trunc_f32x8_wide(self, a: f32x8) -> f32x8 {
        a.trunc()
    }
}

#[cfg(all(feature = "wide", target_arch = "x86_64"))]
impl Avx2FmaToken {
    /// Fused multiply-add on wide types: a * b + c
    ///
    /// When AVX2+FMA is available (proven by this token), generates
    /// a single `vfmadd` instruction instead of separate mul and add.
    #[inline(always)]
    pub fn fma_f32x8_wide(self, a: f32x8, b: f32x8, c: f32x8) -> f32x8 {
        a.mul_add(b, c)
    }

    /// Fused multiply-subtract on wide types: a * b - c
    #[inline(always)]
    pub fn fms_f32x8_wide(self, a: f32x8, b: f32x8, c: f32x8) -> f32x8 {
        a.mul_sub(b, c)
    }

    /// Fused negated multiply-add: -(a * b) + c
    #[inline(always)]
    pub fn fnma_f32x8_wide(self, a: f32x8, b: f32x8, c: f32x8) -> f32x8 {
        a.mul_neg_add(b, c)
    }

    /// Fused negated multiply-subtract: -(a * b) - c
    #[inline(always)]
    pub fn fnms_f32x8_wide(self, a: f32x8, b: f32x8, c: f32x8) -> f32x8 {
        a.mul_neg_sub(b, c)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(all(test, feature = "wide", target_arch = "x86_64"))]
mod tests {
    use super::*;

    #[test]
    fn test_wide_arithmetic() {
        if let Some(token) = Avx2Token::try_new() {
            let a = token.splat_f32x8_wide(2.0);
            let b = token.splat_f32x8_wide(3.0);

            let sum = token.add_f32x8_wide(a, b);
            assert_eq!(sum.to_array(), [5.0f32; 8]);

            let product = token.mul_f32x8_wide(a, b);
            assert_eq!(product.to_array(), [6.0f32; 8]);
        }
    }

    #[test]
    fn test_wide_fma() {
        if let Some(token) = Avx2FmaToken::try_new() {
            let a = token.avx2().splat_f32x8_wide(2.0);
            let b = token.avx2().splat_f32x8_wide(3.0);
            let c = token.avx2().splat_f32x8_wide(1.0);

            // 2 * 3 + 1 = 7
            let result = token.fma_f32x8_wide(a, b, c);
            assert_eq!(result.to_array(), [7.0f32; 8]);
        }
    }

    #[test]
    fn test_wide_load_store() {
        if let Some(token) = Avx2Token::try_new() {
            let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let v = token.load_f32x8_wide(&data);

            let mut out = [0.0f32; 8];
            token.store_f32x8_wide(&mut out, v);

            assert_eq!(data, out);
        }
    }
}

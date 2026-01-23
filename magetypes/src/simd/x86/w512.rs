//! 512-bit (AVX-512) SIMD types.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

use core::arch::x86_64::*;
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign,
    Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};


// ============================================================================
// f32x16 - 16 x f32 (512-bit)
// ============================================================================

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct f32x16(__m512);

#[cfg(feature = "bytemuck")]
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
unsafe impl bytemuck::Zeroable for f32x16 {}
#[cfg(feature = "bytemuck")]
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
unsafe impl bytemuck::Pod for f32x16 {}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl f32x16 {
    pub const LANES: usize = 16;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::X64V4Token, data: &[f32; 16]) -> Self {
        Self(unsafe { _mm512_loadu_ps(data.as_ptr()) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::X64V4Token, v: f32) -> Self {
        Self(unsafe { _mm512_set1_ps(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::X64V4Token) -> Self {
        Self(unsafe { _mm512_setzero_ps() })
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::X64V4Token, arr: [f32; 16]) -> Self {
        // SAFETY: [f32; 16] and __m512 have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [f32; 16]) {
        unsafe { _mm512_storeu_ps(out.as_mut_ptr(), self.0) };
    }

    /// Convert to array
    #[inline(always)]
    pub fn to_array(self) -> [f32; 16] {
        let mut out = [0.0f32; 16];
        self.store(&mut out);
        out
    }

    /// Get reference to underlying array
    #[inline(always)]
    pub fn as_array(&self) -> &[f32; 16] {
        unsafe { &*(self as *const Self as *const [f32; 16]) }
    }

    /// Get mutable reference to underlying array
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut [f32; 16] {
        unsafe { &mut *(self as *mut Self as *mut [f32; 16]) }
    }

    /// Get raw intrinsic type
    #[inline(always)]
    pub fn raw(self) -> __m512 {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports the required SIMD features.
    /// Use token-gated constructors (`load`, `splat`, `zero`) for safe construction.
    #[inline(always)]
    pub unsafe fn from_raw(v: __m512) -> Self {
        Self(v)
    }

    // ========== Token-gated bytemuck replacements ==========

    /// Reinterpret a slice of scalars as a slice of SIMD vectors (token-gated).
    ///
    /// Returns `None` if the slice length is not a multiple of 16, or
    /// if the slice is not properly aligned.
    ///
    /// This is a safe, token-gated replacement for `bytemuck::cast_slice`.
    #[inline(always)]
    pub fn cast_slice(_: archmage::X64V4Token, slice: &[f32]) -> Option<&[Self]> {
        if slice.len() % 16 != 0 {
            return None;
        }
        let ptr = slice.as_ptr();
        if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {
            return None;
        }
        let len = slice.len() / 16;
        // SAFETY: alignment and length checked, layout is compatible
        Some(unsafe { core::slice::from_raw_parts(ptr as *const Self, len) })
    }

    /// Reinterpret a mutable slice of scalars as a slice of SIMD vectors (token-gated).
    ///
    /// Returns `None` if the slice length is not a multiple of 16, or
    /// if the slice is not properly aligned.
    ///
    /// This is a safe, token-gated replacement for `bytemuck::cast_slice_mut`.
    #[inline(always)]
    pub fn cast_slice_mut(_: archmage::X64V4Token, slice: &mut [f32]) -> Option<&mut [Self]> {
        if slice.len() % 16 != 0 {
            return None;
        }
        let ptr = slice.as_mut_ptr();
        if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {
            return None;
        }
        let len = slice.len() / 16;
        // SAFETY: alignment and length checked, layout is compatible
        Some(unsafe { core::slice::from_raw_parts_mut(ptr as *mut Self, len) })
    }

    /// View this vector as a byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of`.
    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8; 64] {
        // SAFETY: Self is repr(transparent) over __m512 which is 64 bytes
        unsafe { &*(self as *const Self as *const [u8; 64]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 64] {
        // SAFETY: Self is repr(transparent) over __m512 which is 64 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 64]) }
    }

    /// Create from a byte array (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::X64V4Token, bytes: &[u8; 64]) -> Self {
        // SAFETY: [u8; 64] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Element-wise minimum
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(unsafe { _mm512_min_ps(self.0, other.0) })
    }
    /// Element-wise maximum
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(unsafe { _mm512_max_ps(self.0, other.0) })
    }
    /// Clamp values between lo and hi
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }
    /// Square root
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        Self(unsafe { _mm512_sqrt_ps(self.0) })
    }
    /// Absolute value
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(unsafe {
            let mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFF_FFFFu32 as i32));
            _mm512_and_ps(self.0, mask)
        })
    }
    /// Round toward negative infinity
    #[inline(always)]
    pub fn floor(self) -> Self {
        Self(unsafe { _mm512_roundscale_ps::<0x01>(self.0) })
    }
    /// Round toward positive infinity
    #[inline(always)]
    pub fn ceil(self) -> Self {
        Self(unsafe { _mm512_roundscale_ps::<0x02>(self.0) })
    }
    /// Round to nearest integer
    #[inline(always)]
    pub fn round(self) -> Self {
        Self(unsafe { _mm512_roundscale_ps::<0x00>(self.0) })
    }
    /// Fused multiply-add: self * a + b
    #[inline(always)]
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        Self(unsafe { _mm512_fmadd_ps(self.0, a.0, b.0) })
    }
    /// Fused multiply-sub: self * a - b
    #[inline(always)]
    pub fn mul_sub(self, a: Self, b: Self) -> Self {
        Self(unsafe { _mm512_fmsub_ps(self.0, a.0, b.0) })
    }
    // ========== Comparisons ==========
    // These return a mask where each lane is all-1s (true) or all-0s (false).
    // Use with `blend()` to select values based on the comparison result.

    /// Lane-wise equality comparison.
    ///
    /// Returns a mask where each lane is all-1s if equality, all-0s otherwise.
    /// Use with `blend(mask, if_true, if_false)` to select values.
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_ps_mask::<_CMP_EQ_OQ>(self.0, other.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_castsi512_ps(_mm512_maskz_set1_epi32(mask, -1))
        })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if inequality, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_ps_mask::<_CMP_NEQ_OQ>(self.0, other.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_castsi512_ps(_mm512_maskz_set1_epi32(mask, -1))
        })
    }

    /// Lane-wise less-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(self.0, other.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_castsi512_ps(_mm512_maskz_set1_epi32(mask, -1))
        })
    }

    /// Lane-wise less-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than-or-equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_ps_mask::<_CMP_LE_OQ>(self.0, other.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_castsi512_ps(_mm512_maskz_set1_epi32(mask, -1))
        })
    }

    /// Lane-wise greater-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than, all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_ps_mask::<_CMP_GT_OQ>(self.0, other.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_castsi512_ps(_mm512_maskz_set1_epi32(mask, -1))
        })
    }

    /// Lane-wise greater-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than-or-equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_ps_mask::<_CMP_GE_OQ>(self.0, other.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_castsi512_ps(_mm512_maskz_set1_epi32(mask, -1))
        })
    }

    // ========== Blending/Selection ==========

    /// Select lanes from `if_true` where mask is all-1s, `if_false` where mask is all-0s.
    ///
    /// The mask should come from a comparison operation like `simd_lt()`.
    ///
    /// # Example
    /// ```ignore
    /// let a = f32x16::splat(token, 1.0);
    /// let b = f32x16::splat(token, 2.0);
    /// let mask = a.simd_lt(b);  // all true
    /// let result = f32x16::blend(mask, a, b);  // selects a
    /// ```
    #[inline(always)]
    pub fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self(unsafe {
            // Convert vector mask to mask register
            let m = _mm512_cmpneq_epi32_mask(_mm512_castps_si512(mask.0), _mm512_setzero_si512());
            _mm512_mask_blend_ps(m, if_false.0, if_true.0)
        })
    }
    // ========== Horizontal Operations ==========

    /// Sum all lanes horizontally.
    ///
    /// Returns a scalar containing the sum of all 16 lanes.
    #[inline(always)]
    pub fn reduce_add(self) -> f32 {
        unsafe { _mm512_reduce_add_ps(self.0) }
    }

    /// Find the minimum value across all lanes.
    #[inline(always)]
    pub fn reduce_min(self) -> f32 {
        unsafe { _mm512_reduce_min_ps(self.0) }
    }

    /// Find the maximum value across all lanes.
    #[inline(always)]
    pub fn reduce_max(self) -> f32 {
        unsafe { _mm512_reduce_max_ps(self.0) }
    }

    // ========== Type Conversions ==========

    /// Convert to signed 32-bit integers, rounding toward zero (truncation).
    ///
    /// Values outside the representable range become `i32::MIN` (0x80000000).
    #[inline(always)]
    pub fn to_i32x16(self) -> i32x16 {
        i32x16(unsafe { _mm512_cvttps_epi32(self.0) })
    }

    /// Convert to signed 32-bit integers, rounding to nearest even.
    ///
    /// Values outside the representable range become `i32::MIN` (0x80000000).
    #[inline(always)]
    pub fn to_i32x16_round(self) -> i32x16 {
        i32x16(unsafe { _mm512_cvtps_epi32(self.0) })
    }

    /// Create from signed 32-bit integers.
    #[inline(always)]
    pub fn from_i32x16(v: i32x16) -> Self {
        Self(unsafe { _mm512_cvtepi32_ps(v.0) })
    }

    // ========== Approximation Operations ==========

    /// Fast reciprocal approximation (1/x) with ~14-bit precision.
    ///
    /// For full precision, use `recip()` which applies Newton-Raphson refinement.
    #[inline(always)]
    pub fn rcp_approx(self) -> Self {
        Self(unsafe { _mm512_rcp14_ps(self.0) })
    }

    /// Precise reciprocal (1/x) using Newton-Raphson refinement.
    ///
    /// More accurate than `rcp_approx()` but slower. For maximum speed
    /// with acceptable precision loss, use `rcp_approx()`.
    #[inline(always)]
    pub fn recip(self) -> Self {
        // Newton-Raphson: x' = x * (2 - a*x)
        let approx = self.rcp_approx();
        let two = Self(unsafe { _mm512_set1_ps(2.0) });
        // One iteration gives ~24-bit precision from ~12-bit
        approx * (two - self * approx)
    }

    /// Fast reciprocal square root approximation (1/sqrt(x)) with ~14-bit precision.
    ///
    /// For full precision, use `sqrt().recip()` or apply Newton-Raphson manually.
    #[inline(always)]
    pub fn rsqrt_approx(self) -> Self {
        Self(unsafe { _mm512_rsqrt14_ps(self.0) })
    }

    /// Precise reciprocal square root (1/sqrt(x)) using Newton-Raphson refinement.
    #[inline(always)]
    pub fn rsqrt(self) -> Self {
        // Newton-Raphson for rsqrt: y' = 0.5 * y * (3 - x * y * y)
        let approx = self.rsqrt_approx();
        let half = Self(unsafe { _mm512_set1_ps(0.5) });
        let three = Self(unsafe { _mm512_set1_ps(3.0) });
        half * approx * (three - self * approx * approx)
    }

    // ========== Bitwise Unary Operations ==========
    /// Bitwise NOT (complement): flips all bits.
    #[inline(always)]
    pub fn not(self) -> Self {
        Self(unsafe {
            let ones = _mm512_set1_epi32(-1);
            let as_int = _mm512_castps_si512(self.0);
            _mm512_castsi512_ps(_mm512_xor_si512(as_int, ones))
        })
    }
    // ========== Transcendental Operations ==========

    /// Low-precision base-2 logarithm (~7.7e-5 max relative error).
    ///
    /// Uses rational polynomial approximation. Fast but not suitable for color-accurate work.
    /// For higher precision, use `log2_midp()`.
    #[inline(always)]
    pub fn log2_lowp(self) -> Self {
        // Rational polynomial coefficients from butteraugli/jpegli
        const P0: f32 = -1.850_383_34e-6;
        const P1: f32 = 1.428_716_05;
        const P2: f32 = 0.742_458_73;
        const Q0: f32 = 0.990_328_14;
        const Q1: f32 = 1.009_671_86;
        const Q2: f32 = 0.174_093_43;

        unsafe {
            let x_bits = _mm512_castps_si512(self.0);
            let offset = _mm512_set1_epi32(0x3f2aaaab_u32 as i32);
            let exp_bits = _mm512_sub_epi32(x_bits, offset);
            let exp_shifted = _mm512_srai_epi32::<23>(exp_bits);

            let mantissa_bits = _mm512_sub_epi32(x_bits, _mm512_slli_epi32::<23>(exp_shifted));
            let mantissa = _mm512_castsi512_ps(mantissa_bits);
            let exp_val = _mm512_cvtepi32_ps(exp_shifted);

            let one = _mm512_set1_ps(1.0);
            let m = _mm512_sub_ps(mantissa, one);

            // Horner's for numerator: P2*m^2 + P1*m + P0
            let yp = _mm512_fmadd_ps(_mm512_set1_ps(P2), m, _mm512_set1_ps(P1));
            let yp = _mm512_fmadd_ps(yp, m, _mm512_set1_ps(P0));

            // Horner's for denominator: Q2*m^2 + Q1*m + Q0
            let yq = _mm512_fmadd_ps(_mm512_set1_ps(Q2), m, _mm512_set1_ps(Q1));
            let yq = _mm512_fmadd_ps(yq, m, _mm512_set1_ps(Q0));

            Self(_mm512_add_ps(_mm512_div_ps(yp, yq), exp_val))
        }
    }

    /// Low-precision base-2 exponential (~5.5e-3 max relative error).
    ///
    /// Uses degree-3 polynomial approximation. Fast but not suitable for color-accurate work.
    /// For higher precision, use `exp2_midp()`.
    #[inline(always)]
    pub fn exp2_lowp(self) -> Self {
        // Polynomial coefficients
        const C0: f32 = 1.0;
        const C1: f32 = core::f32::consts::LN_2;
        const C2: f32 = 0.240_226_5;
        const C3: f32 = 0.055_504_11;

        unsafe {
            // Clamp to safe range
            let x = _mm512_max_ps(self.0, _mm512_set1_ps(-126.0));
            let x = _mm512_min_ps(x, _mm512_set1_ps(126.0));

            // Split into integer and fractional parts
            let xi = _mm512_roundscale_ps::<0x01>(x); // floor
            let xf = _mm512_sub_ps(x, xi);

            // Polynomial for 2^frac
            let poly = _mm512_fmadd_ps(_mm512_set1_ps(C3), xf, _mm512_set1_ps(C2));
            let poly = _mm512_fmadd_ps(poly, xf, _mm512_set1_ps(C1));
            let poly = _mm512_fmadd_ps(poly, xf, _mm512_set1_ps(C0));

            // Scale by 2^integer using bit manipulation
            let xi_i32 = _mm512_cvtps_epi32(xi);
            let bias = _mm512_set1_epi32(127);
            let scale_bits = _mm512_slli_epi32::<23>(_mm512_add_epi32(xi_i32, bias));
            let scale = _mm512_castsi512_ps(scale_bits);

            Self(_mm512_mul_ps(poly, scale))
        }
    }

    /// Low-precision natural logarithm.
    ///
    /// Computed as `log2_lowp(x) * ln(2)`. For higher precision, use `ln_midp()`.
    #[inline(always)]
    pub fn ln_lowp(self) -> Self {
        const LN2: f32 = core::f32::consts::LN_2;
        unsafe {
            Self(_mm512_mul_ps(self.log2_lowp().0, _mm512_set1_ps(LN2)))
        }
    }

    /// Low-precision natural exponential (e^x).
    ///
    /// Computed as `exp2_lowp(x * log2(e))`. For higher precision, use `exp_midp()`.
    #[inline(always)]
    pub fn exp_lowp(self) -> Self {
        const LOG2_E: f32 = core::f32::consts::LOG2_E;
        unsafe {
            Self(_mm512_mul_ps(self.0, _mm512_set1_ps(LOG2_E))).exp2_lowp()
        }
    }

    /// Low-precision base-10 logarithm.
    ///
    /// Computed as `log2_lowp(x) / log2(10)`.
    #[inline(always)]
    pub fn log10_lowp(self) -> Self {
        const LOG10_2: f32 = core::f32::consts::LOG10_2; // 1/log2(10)
        unsafe {
            Self(_mm512_mul_ps(self.log2_lowp().0, _mm512_set1_ps(LOG10_2)))
        }
    }

    /// Low-precision power function (self^n).
    ///
    /// Computed as `exp2_lowp(n * log2_lowp(self))`. For higher precision, use `pow_midp()`.
    /// Note: Only valid for positive self values.
    #[inline(always)]
    pub fn pow_lowp(self, n: f32) -> Self {
        unsafe {
            Self(_mm512_mul_ps(self.log2_lowp().0, _mm512_set1_ps(n))).exp2_lowp()
        }
    }

    // ========== Mid-Precision Transcendental Operations ==========

    /// Mid-precision base-2 logarithm (~3 ULP max error) - unchecked variant.
    ///
    /// Uses (a-1)/(a+1) transform with degree-6 odd polynomial.
    /// **Warning**: Does not handle edge cases (0, negative, inf, NaN).
    /// Use `log2_midp()` for correct IEEE behavior on edge cases.
    #[inline(always)]
    pub fn log2_midp_unchecked(self) -> Self {
        // Constants for range reduction
        const SQRT2_OVER_2: u32 = 0x3f3504f3; // sqrt(2)/2 in f32 bits
        const ONE: u32 = 0x3f800000;          // 1.0 in f32 bits
        const MANTISSA_MASK: i32 = 0x007fffff_u32 as i32;
        const EXPONENT_BIAS: i32 = 127;

        // Coefficients for odd polynomial on y = (a-1)/(a+1)
        const C0: f32 = 2.885_390_08;  // 2/ln(2)
        const C1: f32 = 0.961_800_76;  // y^2 coefficient
        const C2: f32 = 0.576_974_45;  // y^4 coefficient
        const C3: f32 = 0.434_411_97;  // y^6 coefficient

        unsafe {
            let x_bits = _mm512_castps_si512(self.0);

            // Normalize mantissa to [sqrt(2)/2, sqrt(2)]
            let offset = _mm512_set1_epi32((ONE - SQRT2_OVER_2) as i32);
            let adjusted = _mm512_add_epi32(x_bits, offset);

            // Extract exponent
            let exp_raw = _mm512_srai_epi32::<23>(adjusted);
            let exp_biased = _mm512_sub_epi32(exp_raw, _mm512_set1_epi32(EXPONENT_BIAS));
            let n = _mm512_cvtepi32_ps(exp_biased);

            // Reconstruct normalized mantissa
            let mantissa_bits = _mm512_and_si512(adjusted, _mm512_set1_epi32(MANTISSA_MASK));
            let a_bits = _mm512_add_epi32(mantissa_bits, _mm512_set1_epi32(SQRT2_OVER_2 as i32));
            let a = _mm512_castsi512_ps(a_bits);

            // y = (a - 1) / (a + 1)
            let one = _mm512_set1_ps(1.0);
            let a_minus_1 = _mm512_sub_ps(a, one);
            let a_plus_1 = _mm512_add_ps(a, one);
            let y = _mm512_div_ps(a_minus_1, a_plus_1);

            // y^2
            let y2 = _mm512_mul_ps(y, y);

            // Polynomial: c0 + y^2*(c1 + y^2*(c2 + y^2*c3))
            let poly = _mm512_fmadd_ps(_mm512_set1_ps(C3), y2, _mm512_set1_ps(C2));
            let poly = _mm512_fmadd_ps(poly, y2, _mm512_set1_ps(C1));
            let poly = _mm512_fmadd_ps(poly, y2, _mm512_set1_ps(C0));

            // Result: y * poly + n
            Self(_mm512_fmadd_ps(y, poly, n))
        }
    }

    /// Mid-precision base-2 logarithm (~3 ULP max error).
    ///
    /// Uses (a-1)/(a+1) transform with degree-6 odd polynomial.
    /// Suitable for 8-bit, 10-bit, and 12-bit color processing.
    ///
    /// Handles edge cases correctly: log2(0) = -inf, log2(negative) = NaN,
    /// log2(+inf) = +inf, log2(NaN) = NaN.
    #[inline(always)]
    pub fn log2_midp(self) -> Self {
        unsafe {
            let result = self.log2_midp_unchecked();

            // Edge case masks (AVX-512 uses mask registers)
            let zero = _mm512_setzero_ps();
            let is_zero = _mm512_cmp_ps_mask::<_CMP_EQ_OQ>(self.0, zero);
            let is_neg = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(self.0, zero);
            let is_inf = _mm512_cmp_ps_mask::<_CMP_EQ_OQ>(self.0, _mm512_set1_ps(f32::INFINITY));
            let is_nan = _mm512_cmp_ps_mask::<_CMP_UNORD_Q>(self.0, self.0);

            // Apply corrections using mask blend
            let neg_inf = _mm512_set1_ps(f32::NEG_INFINITY);
            let pos_inf = _mm512_set1_ps(f32::INFINITY);
            let nan = _mm512_set1_ps(f32::NAN);

            let r = _mm512_mask_blend_ps(is_zero, result.0, neg_inf);
            let r = _mm512_mask_blend_ps(is_neg, r, nan);
            let r = _mm512_mask_blend_ps(is_inf, r, pos_inf);
            let r = _mm512_mask_blend_ps(is_nan, r, nan);
            Self(r)
        }
    }

    /// Mid-precision base-2 exponential (~140 ULP, ~8e-6 max relative error) - unchecked variant.
    ///
    /// Uses degree-6 minimax polynomial.
    ///
    /// **Warning**: Clamps output to finite range. Does not return infinity for overflow.
    /// Use `exp2_midp()` for correct IEEE behavior on edge cases.
    #[inline(always)]
    pub fn exp2_midp_unchecked(self) -> Self {
        // Degree-6 minimax polynomial for 2^x on [0, 1]
        const C0: f32 = 1.0;
        const C1: f32 = 0.693_147_180_559_945;
        const C2: f32 = 0.240_226_506_959_101;
        const C3: f32 = 0.055_504_108_664_822;
        const C4: f32 = 0.009_618_129_107_629;
        const C5: f32 = 0.001_333_355_814_497;
        const C6: f32 = 0.000_154_035_303_933;

        unsafe {
            // Clamp to safe range
            let x = _mm512_max_ps(self.0, _mm512_set1_ps(-126.0));
            let x = _mm512_min_ps(x, _mm512_set1_ps(126.0));

            let xi = _mm512_roundscale_ps::<0x01>(x); // floor
            let xf = _mm512_sub_ps(x, xi);

            // Horner's method with 6 coefficients
            let poly = _mm512_fmadd_ps(_mm512_set1_ps(C6), xf, _mm512_set1_ps(C5));
            let poly = _mm512_fmadd_ps(poly, xf, _mm512_set1_ps(C4));
            let poly = _mm512_fmadd_ps(poly, xf, _mm512_set1_ps(C3));
            let poly = _mm512_fmadd_ps(poly, xf, _mm512_set1_ps(C2));
            let poly = _mm512_fmadd_ps(poly, xf, _mm512_set1_ps(C1));
            let poly = _mm512_fmadd_ps(poly, xf, _mm512_set1_ps(C0));

            // Scale by 2^integer
            let xi_i32 = _mm512_cvtps_epi32(xi);
            let bias = _mm512_set1_epi32(127);
            let scale_bits = _mm512_slli_epi32::<23>(_mm512_add_epi32(xi_i32, bias));
            let scale = _mm512_castsi512_ps(scale_bits);

            Self(_mm512_mul_ps(poly, scale))
        }
    }

    /// Mid-precision base-2 exponential (~140 ULP, ~8e-6 max relative error).
    ///
    /// Uses degree-6 minimax polynomial.
    /// Suitable for 8-bit, 10-bit, and 12-bit color processing.
    ///
    /// Handles edge cases correctly: exp2(x > 128) = +inf, exp2(x < -150) = 0,
    /// exp2(NaN) = NaN.
    #[inline(always)]
    pub fn exp2_midp(self) -> Self {
        unsafe {
            let result = self.exp2_midp_unchecked();

            // Edge case masks (AVX-512 uses mask registers)
            let is_overflow = _mm512_cmp_ps_mask::<_CMP_GE_OQ>(self.0, _mm512_set1_ps(128.0));
            let is_underflow = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(self.0, _mm512_set1_ps(-150.0));
            let is_nan = _mm512_cmp_ps_mask::<_CMP_UNORD_Q>(self.0, self.0);

            // Apply corrections using mask blend
            let pos_inf = _mm512_set1_ps(f32::INFINITY);
            let zero = _mm512_setzero_ps();
            let nan = _mm512_set1_ps(f32::NAN);

            let r = _mm512_mask_blend_ps(is_overflow, result.0, pos_inf);
            let r = _mm512_mask_blend_ps(is_underflow, r, zero);
            let r = _mm512_mask_blend_ps(is_nan, r, nan);
            Self(r)
        }
    }

    /// Mid-precision power function (self^n) - unchecked variant.
    ///
    /// Computed as `exp2_midp_unchecked(n * log2_midp_unchecked(self))`.
    ///
    /// **Warning**: Does not handle edge cases (0, negative, inf, NaN).
    /// Use `pow_midp()` for correct IEEE behavior on edge cases.
    #[inline(always)]
    pub fn pow_midp_unchecked(self, n: f32) -> Self {
        unsafe {
            Self(_mm512_mul_ps(self.log2_midp_unchecked().0, _mm512_set1_ps(n))).exp2_midp_unchecked()
        }
    }

    /// Mid-precision power function (self^n).
    ///
    /// Computed as `exp2_midp(n * log2_midp(self))`.
    /// Achieves 100% exact round-trips for 8-bit, 10-bit, and 12-bit values.
    ///
    /// Handles edge cases: pow(0, n) = 0 (n>0), pow(inf, n) = inf (n>0).
    /// Note: Only valid for positive self values.
    #[inline(always)]
    pub fn pow_midp(self, n: f32) -> Self {
        unsafe {
            Self(_mm512_mul_ps(self.log2_midp().0, _mm512_set1_ps(n))).exp2_midp()
        }
    }

    /// Mid-precision natural logarithm - unchecked variant.
    ///
    /// Computed as `log2_midp_unchecked(x) * ln(2)`.
    ///
    /// **Warning**: Does not handle edge cases (0, negative, inf, NaN).
    /// Use `ln_midp()` for correct IEEE behavior on edge cases.
    #[inline(always)]
    pub fn ln_midp_unchecked(self) -> Self {
        const LN2: f32 = core::f32::consts::LN_2;
        unsafe {
            Self(_mm512_mul_ps(self.log2_midp_unchecked().0, _mm512_set1_ps(LN2)))
        }
    }

    /// Mid-precision natural logarithm.
    ///
    /// Computed as `log2_midp(x) * ln(2)`.
    ///
    /// Handles edge cases: ln(0) = -inf, ln(negative) = NaN, ln(inf) = inf.
    #[inline(always)]
    pub fn ln_midp(self) -> Self {
        const LN2: f32 = core::f32::consts::LN_2;
        unsafe {
            Self(_mm512_mul_ps(self.log2_midp().0, _mm512_set1_ps(LN2)))
        }
    }

    /// Mid-precision natural exponential (e^x) - unchecked variant.
    ///
    /// Computed as `exp2_midp_unchecked(x * log2(e))`.
    ///
    /// **Warning**: Clamps output to finite range. Does not return infinity for overflow.
    /// Use `exp_midp()` for correct IEEE behavior on edge cases.
    #[inline(always)]
    pub fn exp_midp_unchecked(self) -> Self {
        const LOG2_E: f32 = core::f32::consts::LOG2_E;
        unsafe {
            Self(_mm512_mul_ps(self.0, _mm512_set1_ps(LOG2_E))).exp2_midp_unchecked()
        }
    }

    /// Mid-precision natural exponential (e^x).
    ///
    /// Computed as `exp2_midp(x * log2(e))`.
    ///
    /// Handles edge cases: exp(x>88) = inf, exp(x<-103) = 0, exp(NaN) = NaN.
    #[inline(always)]
    pub fn exp_midp(self) -> Self {
        const LOG2_E: f32 = core::f32::consts::LOG2_E;
        unsafe {
            Self(_mm512_mul_ps(self.0, _mm512_set1_ps(LOG2_E))).exp2_midp()
        }
    }

    // ========== Cube Root ==========

    /// Mid-precision cube root (x^(1/3)) - unchecked variant.
    ///
    /// Uses scalar extraction for initial guess + Newton-Raphson.
    /// Handles negative values correctly (returns -cbrt(|x|)).
    ///
    /// **Warning**: Does not handle edge cases (0, inf, NaN, denormals).
    /// Use `cbrt_midp()` for correct IEEE behavior on edge cases.
    #[inline(always)]
    pub fn cbrt_midp_unchecked(self) -> Self {
        // B1 magic constant for cube root initial approximation
        // B1 = (127 - 127.0/3 - 0.03306235651) * 2^23 = 709958130
        const B1: u32 = 709_958_130;
        const ONE_THIRD: f32 = 1.0 / 3.0;

        unsafe {
            // Extract to array for initial approximation (scalar division by 3)
            let x_arr: [f32; 16] = core::mem::transmute(self.0);
            let mut y_arr = [0.0f32; 16];

            for i in 0..16 {
                let xi = x_arr[i];
                let ui = xi.to_bits();
                let hx = ui & 0x7FFF_FFFF; // abs bits
                // Initial approximation: bits/3 + B1 (always positive)
                let approx = hx / 3 + B1;
                y_arr[i] = f32::from_bits(approx);
            }

            let abs_x = _mm512_andnot_ps(_mm512_set1_ps(-0.0), self.0);
            let sign_bits = _mm512_and_ps(self.0, _mm512_set1_ps(-0.0));
            let mut y = core::mem::transmute::<_, _>(y_arr);

            // Newton-Raphson: y = y * (2*x + y^3) / (x + 2*y^3)
            // Two iterations for full f32 precision
            let two = _mm512_set1_ps(2.0);

            // Iteration 1
            let y3 = _mm512_mul_ps(_mm512_mul_ps(y, y), y);
            let num = _mm512_fmadd_ps(two, abs_x, y3);
            let den = _mm512_fmadd_ps(two, y3, abs_x);
            y = _mm512_mul_ps(y, _mm512_div_ps(num, den));

            // Iteration 2
            let y3 = _mm512_mul_ps(_mm512_mul_ps(y, y), y);
            let num = _mm512_fmadd_ps(two, abs_x, y3);
            let den = _mm512_fmadd_ps(two, y3, abs_x);
            y = _mm512_mul_ps(y, _mm512_div_ps(num, den));

            // Restore sign
            Self(_mm512_or_ps(y, sign_bits))
        }
    }

    /// Mid-precision cube root (x^(1/3)).
    ///
    /// Uses scalar extraction for initial guess + Newton-Raphson.
    /// Handles negative values correctly (returns -cbrt(|x|)).
    ///
    /// Handles edge cases: cbrt(0) = 0, cbrt(±inf) = ±inf, cbrt(NaN) = NaN.
    /// Does not handle denormals (use `cbrt_midp_precise()` for full IEEE compliance).
    #[inline(always)]
    pub fn cbrt_midp(self) -> Self {
        unsafe {
            let result = self.cbrt_midp_unchecked();

            // Edge case masks (AVX-512 uses mask registers)
            let zero = _mm512_setzero_ps();
            let is_zero = _mm512_cmp_ps_mask::<_CMP_EQ_OQ>(self.0, zero);
            let abs_x = _mm512_andnot_ps(_mm512_set1_ps(-0.0), self.0);
            let is_inf = _mm512_cmp_ps_mask::<_CMP_EQ_OQ>(abs_x, _mm512_set1_ps(f32::INFINITY));
            let is_nan = _mm512_cmp_ps_mask::<_CMP_UNORD_Q>(self.0, self.0);

            // Apply corrections using mask blend (use self.0 for zero to preserve sign)
            let r = _mm512_mask_blend_ps(is_zero, result.0, self.0);  // ±0 -> ±0
            let r = _mm512_mask_blend_ps(is_inf, r, self.0);  // ±inf -> ±inf
            let r = _mm512_mask_blend_ps(is_nan, r, _mm512_set1_ps(f32::NAN));
            Self(r)
        }
    }

    /// Precise cube root (x^(1/3)) with full IEEE compliance.
    ///
    /// Uses scalar extraction for initial guess + Newton-Raphson.
    /// Handles negative values correctly (returns -cbrt(|x|)).
    ///
    /// Handles all edge cases including denormals. About 67% slower than `cbrt_midp()`.
    /// Use `cbrt_midp()` if denormal support is not needed (most image processing).
    #[inline(always)]
    pub fn cbrt_midp_precise(self) -> Self {
        unsafe {
            // Scale factor for denormals: 2^24
            const SCALE_UP: f32 = 16777216.0;  // 2^24
            const SCALE_DOWN: f32 = 0.00390625;  // 2^(-8) = cbrt(2^(-24))
            const DENORM_LIMIT: f32 = 1.17549435e-38;  // Smallest normal f32

            let abs_x = _mm512_andnot_ps(_mm512_set1_ps(-0.0), self.0);
            let is_denorm = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(abs_x, _mm512_set1_ps(DENORM_LIMIT));

            // Scale up denormals
            let scaled_x = _mm512_mul_ps(self.0, _mm512_set1_ps(SCALE_UP));
            let x_for_cbrt = _mm512_mask_blend_ps(is_denorm, self.0, scaled_x);

            // Compute cbrt with edge case handling
            let result = Self(x_for_cbrt).cbrt_midp();

            // Scale down results from denormal inputs
            let scaled_result = _mm512_mul_ps(result.0, _mm512_set1_ps(SCALE_DOWN));
            Self(_mm512_mask_blend_ps(is_denorm, result.0, scaled_result))
        }
    }

// ========== Matrix Transpose ==========

/// Transpose an 8x8 matrix using AVX-512.
///
/// Takes 8 f32x16 vectors where only the lower 8 elements are used.
#[inline]
pub fn transpose_8x8(rows: &mut [Self; 8]) {
    unsafe {
        let idx_lo = _mm512_setr_epi32(0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29);
        let idx_hi = _mm512_setr_epi32(2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31);

        let r0i = _mm512_castps_si512(rows[0].0);
        let r1i = _mm512_castps_si512(rows[1].0);
        let r2i = _mm512_castps_si512(rows[2].0);
        let r3i = _mm512_castps_si512(rows[3].0);
        let r4i = _mm512_castps_si512(rows[4].0);
        let r5i = _mm512_castps_si512(rows[5].0);
        let r6i = _mm512_castps_si512(rows[6].0);
        let r7i = _mm512_castps_si512(rows[7].0);

        let t0 = _mm512_castsi512_ps(_mm512_permutex2var_epi32(r0i, idx_lo, r1i));
        let t1 = _mm512_castsi512_ps(_mm512_permutex2var_epi32(r0i, idx_hi, r1i));
        let t2 = _mm512_castsi512_ps(_mm512_permutex2var_epi32(r2i, idx_lo, r3i));
        let t3 = _mm512_castsi512_ps(_mm512_permutex2var_epi32(r2i, idx_hi, r3i));
        let t4 = _mm512_castsi512_ps(_mm512_permutex2var_epi32(r4i, idx_lo, r5i));
        let t5 = _mm512_castsi512_ps(_mm512_permutex2var_epi32(r4i, idx_hi, r5i));
        let t6 = _mm512_castsi512_ps(_mm512_permutex2var_epi32(r6i, idx_lo, r7i));
        let t7 = _mm512_castsi512_ps(_mm512_permutex2var_epi32(r6i, idx_hi, r7i));

        let idx2_lo = _mm512_setr_epi64(0, 8, 1, 9, 4, 12, 5, 13);
        let idx2_hi = _mm512_setr_epi64(2, 10, 3, 11, 6, 14, 7, 15);

        let t0i = _mm512_castps_si512(t0);
        let t2i = _mm512_castps_si512(t2);
        let t1i = _mm512_castps_si512(t1);
        let t3i = _mm512_castps_si512(t3);
        let t4i = _mm512_castps_si512(t4);
        let t6i = _mm512_castps_si512(t6);
        let t5i = _mm512_castps_si512(t5);
        let t7i = _mm512_castps_si512(t7);

        let s0 = _mm512_castsi512_ps(_mm512_permutex2var_epi64(t0i, idx2_lo, t2i));
        let s1 = _mm512_castsi512_ps(_mm512_permutex2var_epi64(t0i, idx2_hi, t2i));
        let s2 = _mm512_castsi512_ps(_mm512_permutex2var_epi64(t1i, idx2_lo, t3i));
        let s3 = _mm512_castsi512_ps(_mm512_permutex2var_epi64(t1i, idx2_hi, t3i));
        let s4 = _mm512_castsi512_ps(_mm512_permutex2var_epi64(t4i, idx2_lo, t6i));
        let s5 = _mm512_castsi512_ps(_mm512_permutex2var_epi64(t4i, idx2_hi, t6i));
        let s6 = _mm512_castsi512_ps(_mm512_permutex2var_epi64(t5i, idx2_lo, t7i));
        let s7 = _mm512_castsi512_ps(_mm512_permutex2var_epi64(t5i, idx2_hi, t7i));

        let idx3_lo = _mm512_setr_epi32(0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, 24, 25, 26, 27);
        let idx3_hi = _mm512_setr_epi32(4, 5, 6, 7, 20, 21, 22, 23, 12, 13, 14, 15, 28, 29, 30, 31);

        let s0i = _mm512_castps_si512(s0);
        let s4i = _mm512_castps_si512(s4);
        let s1i = _mm512_castps_si512(s1);
        let s5i = _mm512_castps_si512(s5);
        let s2i = _mm512_castps_si512(s2);
        let s6i = _mm512_castps_si512(s6);
        let s3i = _mm512_castps_si512(s3);
        let s7i = _mm512_castps_si512(s7);

        rows[0] = Self(_mm512_castsi512_ps(_mm512_permutex2var_epi32(s0i, idx3_lo, s4i)));
        rows[1] = Self(_mm512_castsi512_ps(_mm512_permutex2var_epi32(s1i, idx3_lo, s5i)));
        rows[2] = Self(_mm512_castsi512_ps(_mm512_permutex2var_epi32(s2i, idx3_lo, s6i)));
        rows[3] = Self(_mm512_castsi512_ps(_mm512_permutex2var_epi32(s3i, idx3_lo, s7i)));
        rows[4] = Self(_mm512_castsi512_ps(_mm512_permutex2var_epi32(s0i, idx3_hi, s4i)));
        rows[5] = Self(_mm512_castsi512_ps(_mm512_permutex2var_epi32(s1i, idx3_hi, s5i)));
        rows[6] = Self(_mm512_castsi512_ps(_mm512_permutex2var_epi32(s2i, idx3_hi, s6i)));
        rows[7] = Self(_mm512_castsi512_ps(_mm512_permutex2var_epi32(s3i, idx3_hi, s7i)));
    }
}

/// Transpose an 8x8 matrix, returning the transposed rows.
#[inline]
pub fn transpose_8x8_copy(rows: [Self; 8]) -> [Self; 8] {
    let mut result = rows;
    Self::transpose_8x8(&mut result);
    result
}

/// Load an 8x8 f32 block into 8 f32x16 vectors (lower 8 elements used).
#[inline]
pub fn load_8x8(block: &[f32; 64]) -> [Self; 8] {
    unsafe {
        [
            Self(_mm512_castps256_ps512(_mm256_loadu_ps(block.as_ptr()))),
            Self(_mm512_castps256_ps512(_mm256_loadu_ps(block.as_ptr().add(8)))),
            Self(_mm512_castps256_ps512(_mm256_loadu_ps(block.as_ptr().add(16)))),
            Self(_mm512_castps256_ps512(_mm256_loadu_ps(block.as_ptr().add(24)))),
            Self(_mm512_castps256_ps512(_mm256_loadu_ps(block.as_ptr().add(32)))),
            Self(_mm512_castps256_ps512(_mm256_loadu_ps(block.as_ptr().add(40)))),
            Self(_mm512_castps256_ps512(_mm256_loadu_ps(block.as_ptr().add(48)))),
            Self(_mm512_castps256_ps512(_mm256_loadu_ps(block.as_ptr().add(56)))),
        ]
    }
}

/// Store 8 f32x16 vectors (lower 8 elements) to a contiguous 8x8 f32 block.
#[inline]
pub fn store_8x8(rows: &[Self; 8], block: &mut [f32; 64]) {
    unsafe {
        _mm256_storeu_ps(block.as_mut_ptr(), _mm512_castps512_ps256(rows[0].0));
        _mm256_storeu_ps(block.as_mut_ptr().add(8), _mm512_castps512_ps256(rows[1].0));
        _mm256_storeu_ps(block.as_mut_ptr().add(16), _mm512_castps512_ps256(rows[2].0));
        _mm256_storeu_ps(block.as_mut_ptr().add(24), _mm512_castps512_ps256(rows[3].0));
        _mm256_storeu_ps(block.as_mut_ptr().add(32), _mm512_castps512_ps256(rows[4].0));
        _mm256_storeu_ps(block.as_mut_ptr().add(40), _mm512_castps512_ps256(rows[5].0));
        _mm256_storeu_ps(block.as_mut_ptr().add(48), _mm512_castps512_ps256(rows[6].0));
        _mm256_storeu_ps(block.as_mut_ptr().add(56), _mm512_castps512_ps256(rows[7].0));
    }
}

}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
crate::impl_arithmetic_ops!(f32x16, _mm512_add_ps, _mm512_sub_ps, _mm512_mul_ps, _mm512_div_ps);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
crate::impl_float_assign_ops!(f32x16);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
crate::impl_neg!(f32x16, _mm512_sub_ps, _mm512_setzero_ps);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
crate::impl_bitwise_ops!(f32x16, __m512, _mm512_and_ps, _mm512_or_ps, _mm512_xor_ps);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
crate::impl_index!(f32x16, f32, 16);

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl From<[f32; 16]> for f32x16 {
    #[inline(always)]
    fn from(arr: [f32; 16]) -> Self {
        // SAFETY: [f32; 16] and __m512 have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl From<f32x16> for [f32; 16] {
    #[inline(always)]
    fn from(v: f32x16) -> Self {
        // SAFETY: __m512 and [f32; 16] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}


// Scalar broadcast operations for f32x16
// These allow `v + 2.0` instead of `v + f32x16::splat(token, 2.0)`

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl Add<f32> for f32x16 {
    type Output = Self;
    /// Add a scalar to all lanes: `v + 2.0`
    ///
    /// Broadcasts the scalar to all lanes, then adds.
    #[inline(always)]
    fn add(self, rhs: f32) -> Self {
        self + Self(unsafe { _mm512_set1_ps(rhs) })
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl Sub<f32> for f32x16 {
    type Output = Self;
    /// Subtract a scalar from all lanes: `v - 2.0`
    #[inline(always)]
    fn sub(self, rhs: f32) -> Self {
        self - Self(unsafe { _mm512_set1_ps(rhs) })
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl Mul<f32> for f32x16 {
    type Output = Self;
    /// Multiply all lanes by a scalar: `v * 2.0`
    #[inline(always)]
    fn mul(self, rhs: f32) -> Self {
        self * Self(unsafe { _mm512_set1_ps(rhs) })
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl Div<f32> for f32x16 {
    type Output = Self;
    /// Divide all lanes by a scalar: `v / 2.0`
    #[inline(always)]
    fn div(self, rhs: f32) -> Self {
        self / Self(unsafe { _mm512_set1_ps(rhs) })
    }
}


// ============================================================================
// f64x8 - 8 x f64 (512-bit)
// ============================================================================

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct f64x8(__m512d);

#[cfg(feature = "bytemuck")]
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
unsafe impl bytemuck::Zeroable for f64x8 {}
#[cfg(feature = "bytemuck")]
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
unsafe impl bytemuck::Pod for f64x8 {}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl f64x8 {
    pub const LANES: usize = 8;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::X64V4Token, data: &[f64; 8]) -> Self {
        Self(unsafe { _mm512_loadu_pd(data.as_ptr()) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::X64V4Token, v: f64) -> Self {
        Self(unsafe { _mm512_set1_pd(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::X64V4Token) -> Self {
        Self(unsafe { _mm512_setzero_pd() })
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::X64V4Token, arr: [f64; 8]) -> Self {
        // SAFETY: [f64; 8] and __m512d have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [f64; 8]) {
        unsafe { _mm512_storeu_pd(out.as_mut_ptr(), self.0) };
    }

    /// Convert to array
    #[inline(always)]
    pub fn to_array(self) -> [f64; 8] {
        let mut out = [0.0f64; 8];
        self.store(&mut out);
        out
    }

    /// Get reference to underlying array
    #[inline(always)]
    pub fn as_array(&self) -> &[f64; 8] {
        unsafe { &*(self as *const Self as *const [f64; 8]) }
    }

    /// Get mutable reference to underlying array
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut [f64; 8] {
        unsafe { &mut *(self as *mut Self as *mut [f64; 8]) }
    }

    /// Get raw intrinsic type
    #[inline(always)]
    pub fn raw(self) -> __m512d {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports the required SIMD features.
    /// Use token-gated constructors (`load`, `splat`, `zero`) for safe construction.
    #[inline(always)]
    pub unsafe fn from_raw(v: __m512d) -> Self {
        Self(v)
    }

    // ========== Token-gated bytemuck replacements ==========

    /// Reinterpret a slice of scalars as a slice of SIMD vectors (token-gated).
    ///
    /// Returns `None` if the slice length is not a multiple of 8, or
    /// if the slice is not properly aligned.
    ///
    /// This is a safe, token-gated replacement for `bytemuck::cast_slice`.
    #[inline(always)]
    pub fn cast_slice(_: archmage::X64V4Token, slice: &[f64]) -> Option<&[Self]> {
        if slice.len() % 8 != 0 {
            return None;
        }
        let ptr = slice.as_ptr();
        if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {
            return None;
        }
        let len = slice.len() / 8;
        // SAFETY: alignment and length checked, layout is compatible
        Some(unsafe { core::slice::from_raw_parts(ptr as *const Self, len) })
    }

    /// Reinterpret a mutable slice of scalars as a slice of SIMD vectors (token-gated).
    ///
    /// Returns `None` if the slice length is not a multiple of 8, or
    /// if the slice is not properly aligned.
    ///
    /// This is a safe, token-gated replacement for `bytemuck::cast_slice_mut`.
    #[inline(always)]
    pub fn cast_slice_mut(_: archmage::X64V4Token, slice: &mut [f64]) -> Option<&mut [Self]> {
        if slice.len() % 8 != 0 {
            return None;
        }
        let ptr = slice.as_mut_ptr();
        if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {
            return None;
        }
        let len = slice.len() / 8;
        // SAFETY: alignment and length checked, layout is compatible
        Some(unsafe { core::slice::from_raw_parts_mut(ptr as *mut Self, len) })
    }

    /// View this vector as a byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of`.
    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8; 64] {
        // SAFETY: Self is repr(transparent) over __m512d which is 64 bytes
        unsafe { &*(self as *const Self as *const [u8; 64]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 64] {
        // SAFETY: Self is repr(transparent) over __m512d which is 64 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 64]) }
    }

    /// Create from a byte array (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::X64V4Token, bytes: &[u8; 64]) -> Self {
        // SAFETY: [u8; 64] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Element-wise minimum
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(unsafe { _mm512_min_pd(self.0, other.0) })
    }
    /// Element-wise maximum
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(unsafe { _mm512_max_pd(self.0, other.0) })
    }
    /// Clamp values between lo and hi
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }
    /// Square root
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        Self(unsafe { _mm512_sqrt_pd(self.0) })
    }
    /// Absolute value
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(unsafe {
            let mask = _mm512_castsi512_pd(_mm512_set1_epi64(0x7FFF_FFFF_FFFF_FFFFu64 as i64));
            _mm512_and_pd(self.0, mask)
        })
    }
    /// Round toward negative infinity
    #[inline(always)]
    pub fn floor(self) -> Self {
        Self(unsafe { _mm512_roundscale_pd::<0x01>(self.0) })
    }
    /// Round toward positive infinity
    #[inline(always)]
    pub fn ceil(self) -> Self {
        Self(unsafe { _mm512_roundscale_pd::<0x02>(self.0) })
    }
    /// Round to nearest integer
    #[inline(always)]
    pub fn round(self) -> Self {
        Self(unsafe { _mm512_roundscale_pd::<0x00>(self.0) })
    }
    /// Fused multiply-add: self * a + b
    #[inline(always)]
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        Self(unsafe { _mm512_fmadd_pd(self.0, a.0, b.0) })
    }
    /// Fused multiply-sub: self * a - b
    #[inline(always)]
    pub fn mul_sub(self, a: Self, b: Self) -> Self {
        Self(unsafe { _mm512_fmsub_pd(self.0, a.0, b.0) })
    }
    // ========== Comparisons ==========
    // These return a mask where each lane is all-1s (true) or all-0s (false).
    // Use with `blend()` to select values based on the comparison result.

    /// Lane-wise equality comparison.
    ///
    /// Returns a mask where each lane is all-1s if equality, all-0s otherwise.
    /// Use with `blend(mask, if_true, if_false)` to select values.
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_pd_mask::<_CMP_EQ_OQ>(self.0, other.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_castsi512_pd(_mm512_maskz_set1_epi64(mask, -1))
        })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if inequality, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_pd_mask::<_CMP_NEQ_OQ>(self.0, other.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_castsi512_pd(_mm512_maskz_set1_epi64(mask, -1))
        })
    }

    /// Lane-wise less-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_pd_mask::<_CMP_LT_OQ>(self.0, other.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_castsi512_pd(_mm512_maskz_set1_epi64(mask, -1))
        })
    }

    /// Lane-wise less-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than-or-equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_pd_mask::<_CMP_LE_OQ>(self.0, other.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_castsi512_pd(_mm512_maskz_set1_epi64(mask, -1))
        })
    }

    /// Lane-wise greater-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than, all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_pd_mask::<_CMP_GT_OQ>(self.0, other.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_castsi512_pd(_mm512_maskz_set1_epi64(mask, -1))
        })
    }

    /// Lane-wise greater-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than-or-equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_pd_mask::<_CMP_GE_OQ>(self.0, other.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_castsi512_pd(_mm512_maskz_set1_epi64(mask, -1))
        })
    }

    // ========== Blending/Selection ==========

    /// Select lanes from `if_true` where mask is all-1s, `if_false` where mask is all-0s.
    ///
    /// The mask should come from a comparison operation like `simd_lt()`.
    ///
    /// # Example
    /// ```ignore
    /// let a = f64x8::splat(token, 1.0);
    /// let b = f64x8::splat(token, 2.0);
    /// let mask = a.simd_lt(b);  // all true
    /// let result = f64x8::blend(mask, a, b);  // selects a
    /// ```
    #[inline(always)]
    pub fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self(unsafe {
            // Convert vector mask to mask register
            let m = _mm512_cmpneq_epi64_mask(_mm512_castpd_si512(mask.0), _mm512_setzero_si512());
            _mm512_mask_blend_pd(m, if_false.0, if_true.0)
        })
    }
    // ========== Horizontal Operations ==========

    /// Sum all lanes horizontally.
    ///
    /// Returns a scalar containing the sum of all 8 lanes.
    #[inline(always)]
    pub fn reduce_add(self) -> f64 {
        unsafe { _mm512_reduce_add_pd(self.0) }
    }

    /// Find the minimum value across all lanes.
    #[inline(always)]
    pub fn reduce_min(self) -> f64 {
        unsafe { _mm512_reduce_min_pd(self.0) }
    }

    /// Find the maximum value across all lanes.
    #[inline(always)]
    pub fn reduce_max(self) -> f64 {
        unsafe { _mm512_reduce_max_pd(self.0) }
    }

    // ========== Approximation Operations ==========

    /// Fast reciprocal approximation (1/x) with ~14-bit precision.
    #[inline(always)]
    pub fn rcp_approx(self) -> Self {
        Self(unsafe { _mm512_rcp14_pd(self.0) })
    }

    /// Precise reciprocal (1/x) using Newton-Raphson refinement.
    #[inline(always)]
    pub fn recip(self) -> Self {
        let approx = self.rcp_approx();
        let two = Self(unsafe { _mm512_set1_pd(2.0) });
        approx * (two - self * approx)
    }

    /// Fast reciprocal square root approximation (1/sqrt(x)) with ~14-bit precision.
    #[inline(always)]
    pub fn rsqrt_approx(self) -> Self {
        Self(unsafe { _mm512_rsqrt14_pd(self.0) })
    }

    /// Precise reciprocal square root (1/sqrt(x)) using Newton-Raphson refinement.
    #[inline(always)]
    pub fn rsqrt(self) -> Self {
        let approx = self.rsqrt_approx();
        let half = Self(unsafe { _mm512_set1_pd(0.5) });
        let three = Self(unsafe { _mm512_set1_pd(3.0) });
        half * approx * (three - self * approx * approx)
    }

    // ========== Bitwise Unary Operations ==========
    /// Bitwise NOT (complement): flips all bits.
    #[inline(always)]
    pub fn not(self) -> Self {
        Self(unsafe {
            let ones = _mm512_set1_epi64(-1);
            let as_int = _mm512_castpd_si512(self.0);
            _mm512_castsi512_pd(_mm512_xor_si512(as_int, ones))
        })
    }
    // ========== Transcendental Operations ==========

    /// Low-precision base-2 logarithm.
    ///
    /// Uses polynomial approximation. For natural log, use `ln_lowp()`.
    #[inline(always)]
    pub fn log2_lowp(self) -> Self {
        // Polynomial coefficients for f64
        const P0: f64 = -1.850_383_340_051_831e-6;
        const P1: f64 = 1.428_716_047_008_376;
        const P2: f64 = 0.742_458_733_278_206;
        const Q0: f64 = 0.990_328_142_775_907;
        const Q1: f64 = 1.009_671_857_224_115;
        const Q2: f64 = 0.174_093_430_036_669;
        const OFFSET: i64 = 0x3fe6a09e667f3bcd_u64 as i64; // 2/3 in f64 bits

        unsafe {
            let x_bits = _mm512_castpd_si512(self.0);
            let offset = _mm512_set1_epi64(OFFSET);
            let exp_bits = _mm512_sub_epi64(x_bits, offset);
            let exp_shifted = _mm512_srai_epi64::<52>(exp_bits);

            let mantissa_bits = _mm512_sub_epi64(x_bits, _mm512_slli_epi64::<52>(exp_shifted));
            let mantissa = _mm512_castsi512_pd(mantissa_bits);
            // Convert exponent to f64
            let exp_arr: [i64; 8] = core::mem::transmute(exp_shifted);
            let exp_f64: [f64; 8] = [
exp_arr[0] as f64, exp_arr[1] as f64, exp_arr[2] as f64, exp_arr[3] as f64, exp_arr[4] as f64, exp_arr[5] as f64, exp_arr[6] as f64, exp_arr[7] as f64];
            let exp_val = _mm512_loadu_pd(exp_f64.as_ptr());

            let one = _mm512_set1_pd(1.0);
            let m = _mm512_sub_pd(mantissa, one);

            // Horner's for numerator
            let yp = _mm512_fmadd_pd(_mm512_set1_pd(P2), m, _mm512_set1_pd(P1));
            let yp = _mm512_fmadd_pd(yp, m, _mm512_set1_pd(P0));

            // Horner's for denominator
            let yq = _mm512_fmadd_pd(_mm512_set1_pd(Q2), m, _mm512_set1_pd(Q1));
            let yq = _mm512_fmadd_pd(yq, m, _mm512_set1_pd(Q0));

            Self(_mm512_add_pd(_mm512_div_pd(yp, yq), exp_val))
        }
    }

    /// Low-precision base-2 exponential (2^x).
    ///
    /// Uses polynomial approximation. For natural exp, use `exp_lowp()`.
    #[inline(always)]
    pub fn exp2_lowp(self) -> Self {
        const C0: f64 = 1.0;
        const C1: f64 = core::f64::consts::LN_2;
        const C2: f64 = 0.240_226_506_959_101;
        const C3: f64 = 0.055_504_108_664_822;
        const C4: f64 = 0.009_618_129_107_629;

        unsafe {
            // Clamp to safe range
            let x = _mm512_max_pd(self.0, _mm512_set1_pd(-1022.0));
            let x = _mm512_min_pd(x, _mm512_set1_pd(1022.0));

            let xi = _mm512_roundscale_pd::<0x01>(x); // floor
            let xf = _mm512_sub_pd(x, xi);

            // Polynomial for 2^frac
            let poly = _mm512_fmadd_pd(_mm512_set1_pd(C4), xf, _mm512_set1_pd(C3));
            let poly = _mm512_fmadd_pd(poly, xf, _mm512_set1_pd(C2));
            let poly = _mm512_fmadd_pd(poly, xf, _mm512_set1_pd(C1));
            let poly = _mm512_fmadd_pd(poly, xf, _mm512_set1_pd(C0));

            // Scale by 2^integer - extract, convert, scale
            let xi_arr: [f64; 8] = core::mem::transmute(xi);
            let scale_arr: [f64; 8] = [
f64::from_bits(((xi_arr[0] as i64 + 1023) << 52) as u64), f64::from_bits(((xi_arr[1] as i64 + 1023) << 52) as u64), f64::from_bits(((xi_arr[2] as i64 + 1023) << 52) as u64), f64::from_bits(((xi_arr[3] as i64 + 1023) << 52) as u64), f64::from_bits(((xi_arr[4] as i64 + 1023) << 52) as u64), f64::from_bits(((xi_arr[5] as i64 + 1023) << 52) as u64), f64::from_bits(((xi_arr[6] as i64 + 1023) << 52) as u64), f64::from_bits(((xi_arr[7] as i64 + 1023) << 52) as u64)];
            let scale = _mm512_loadu_pd(scale_arr.as_ptr());

            Self(_mm512_mul_pd(poly, scale))
        }
    }

    /// Low-precision natural logarithm.
    #[inline(always)]
    pub fn ln_lowp(self) -> Self {
        const LN2: f64 = core::f64::consts::LN_2;
        unsafe {
            Self(_mm512_mul_pd(self.log2_lowp().0, _mm512_set1_pd(LN2)))
        }
    }

    /// Low-precision natural exponential (e^x).
    #[inline(always)]
    pub fn exp_lowp(self) -> Self {
        const LOG2_E: f64 = core::f64::consts::LOG2_E;
        unsafe {
            Self(_mm512_mul_pd(self.0, _mm512_set1_pd(LOG2_E))).exp2_lowp()
        }
    }

    /// Low-precision base-10 logarithm.
    #[inline(always)]
    pub fn log10_lowp(self) -> Self {
        const LOG10_2: f64 = core::f64::consts::LOG10_2;
        unsafe {
            Self(_mm512_mul_pd(self.log2_lowp().0, _mm512_set1_pd(LOG10_2)))
        }
    }

    /// Low-precision power function (self^n).
    #[inline(always)]
    pub fn pow_lowp(self, n: f64) -> Self {
        unsafe {
            Self(_mm512_mul_pd(self.log2_lowp().0, _mm512_set1_pd(n))).exp2_lowp()
        }
    }

}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
crate::impl_arithmetic_ops!(f64x8, _mm512_add_pd, _mm512_sub_pd, _mm512_mul_pd, _mm512_div_pd);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
crate::impl_float_assign_ops!(f64x8);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
crate::impl_neg!(f64x8, _mm512_sub_pd, _mm512_setzero_pd);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
crate::impl_bitwise_ops!(f64x8, __m512d, _mm512_and_pd, _mm512_or_pd, _mm512_xor_pd);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
crate::impl_index!(f64x8, f64, 8);

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl From<[f64; 8]> for f64x8 {
    #[inline(always)]
    fn from(arr: [f64; 8]) -> Self {
        // SAFETY: [f64; 8] and __m512d have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl From<f64x8> for [f64; 8] {
    #[inline(always)]
    fn from(v: f64x8) -> Self {
        // SAFETY: __m512d and [f64; 8] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}


// Scalar broadcast operations for f64x8
// These allow `v + 2.0` instead of `v + f64x8::splat(token, 2.0)`

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl Add<f64> for f64x8 {
    type Output = Self;
    /// Add a scalar to all lanes: `v + 2.0`
    ///
    /// Broadcasts the scalar to all lanes, then adds.
    #[inline(always)]
    fn add(self, rhs: f64) -> Self {
        self + Self(unsafe { _mm512_set1_pd(rhs) })
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl Sub<f64> for f64x8 {
    type Output = Self;
    /// Subtract a scalar from all lanes: `v - 2.0`
    #[inline(always)]
    fn sub(self, rhs: f64) -> Self {
        self - Self(unsafe { _mm512_set1_pd(rhs) })
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl Mul<f64> for f64x8 {
    type Output = Self;
    /// Multiply all lanes by a scalar: `v * 2.0`
    #[inline(always)]
    fn mul(self, rhs: f64) -> Self {
        self * Self(unsafe { _mm512_set1_pd(rhs) })
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl Div<f64> for f64x8 {
    type Output = Self;
    /// Divide all lanes by a scalar: `v / 2.0`
    #[inline(always)]
    fn div(self, rhs: f64) -> Self {
        self / Self(unsafe { _mm512_set1_pd(rhs) })
    }
}


// ============================================================================
// i8x64 - 64 x i8 (512-bit)
// ============================================================================

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct i8x64(__m512i);

#[cfg(feature = "bytemuck")]
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
unsafe impl bytemuck::Zeroable for i8x64 {}
#[cfg(feature = "bytemuck")]
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
unsafe impl bytemuck::Pod for i8x64 {}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl i8x64 {
    pub const LANES: usize = 64;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::X64V4Token, data: &[i8; 64]) -> Self {
        Self(unsafe { _mm512_loadu_si512(data.as_ptr() as *const __m512i) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::X64V4Token, v: i8) -> Self {
        Self(unsafe { _mm512_set1_epi8(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::X64V4Token) -> Self {
        Self(unsafe { _mm512_setzero_si512() })
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::X64V4Token, arr: [i8; 64]) -> Self {
        // SAFETY: [i8; 64] and __m512i have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [i8; 64]) {
        unsafe { _mm512_storeu_si512(out.as_mut_ptr() as *mut __m512i, self.0) };
    }

    /// Convert to array
    #[inline(always)]
    pub fn to_array(self) -> [i8; 64] {
        let mut out = [0i8; 64];
        self.store(&mut out);
        out
    }

    /// Get reference to underlying array
    #[inline(always)]
    pub fn as_array(&self) -> &[i8; 64] {
        unsafe { &*(self as *const Self as *const [i8; 64]) }
    }

    /// Get mutable reference to underlying array
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut [i8; 64] {
        unsafe { &mut *(self as *mut Self as *mut [i8; 64]) }
    }

    /// Get raw intrinsic type
    #[inline(always)]
    pub fn raw(self) -> __m512i {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports the required SIMD features.
    /// Use token-gated constructors (`load`, `splat`, `zero`) for safe construction.
    #[inline(always)]
    pub unsafe fn from_raw(v: __m512i) -> Self {
        Self(v)
    }

    // ========== Token-gated bytemuck replacements ==========

    /// Reinterpret a slice of scalars as a slice of SIMD vectors (token-gated).
    ///
    /// Returns `None` if the slice length is not a multiple of 64, or
    /// if the slice is not properly aligned.
    ///
    /// This is a safe, token-gated replacement for `bytemuck::cast_slice`.
    #[inline(always)]
    pub fn cast_slice(_: archmage::X64V4Token, slice: &[i8]) -> Option<&[Self]> {
        if slice.len() % 64 != 0 {
            return None;
        }
        let ptr = slice.as_ptr();
        if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {
            return None;
        }
        let len = slice.len() / 64;
        // SAFETY: alignment and length checked, layout is compatible
        Some(unsafe { core::slice::from_raw_parts(ptr as *const Self, len) })
    }

    /// Reinterpret a mutable slice of scalars as a slice of SIMD vectors (token-gated).
    ///
    /// Returns `None` if the slice length is not a multiple of 64, or
    /// if the slice is not properly aligned.
    ///
    /// This is a safe, token-gated replacement for `bytemuck::cast_slice_mut`.
    #[inline(always)]
    pub fn cast_slice_mut(_: archmage::X64V4Token, slice: &mut [i8]) -> Option<&mut [Self]> {
        if slice.len() % 64 != 0 {
            return None;
        }
        let ptr = slice.as_mut_ptr();
        if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {
            return None;
        }
        let len = slice.len() / 64;
        // SAFETY: alignment and length checked, layout is compatible
        Some(unsafe { core::slice::from_raw_parts_mut(ptr as *mut Self, len) })
    }

    /// View this vector as a byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of`.
    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8; 64] {
        // SAFETY: Self is repr(transparent) over __m512i which is 64 bytes
        unsafe { &*(self as *const Self as *const [u8; 64]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 64] {
        // SAFETY: Self is repr(transparent) over __m512i which is 64 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 64]) }
    }

    /// Create from a byte array (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::X64V4Token, bytes: &[u8; 64]) -> Self {
        // SAFETY: [u8; 64] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Element-wise minimum
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(unsafe { _mm512_min_epi8(self.0, other.0) })
    }
    /// Element-wise maximum
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(unsafe { _mm512_max_epi8(self.0, other.0) })
    }
    /// Clamp values between lo and hi
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }
    /// Absolute value
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(unsafe { _mm512_abs_epi8(self.0) })
    }
    // ========== Comparisons ==========
    // These return a mask where each lane is all-1s (true) or all-0s (false).
    // Use with `blend()` to select values based on the comparison result.

    /// Lane-wise equality comparison.
    ///
    /// Returns a mask where each lane is all-1s if equality, all-0s otherwise.
    /// Use with `blend(mask, if_true, if_false)` to select values.
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_epi8_mask::<_MM_CMPINT_EQ>(self.0, other.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_maskz_set1_epi8(mask, -1)
        })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if inequality, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_epi8_mask::<_MM_CMPINT_NE>(self.0, other.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_maskz_set1_epi8(mask, -1)
        })
    }

    /// Lane-wise less-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_epi8_mask::<_MM_CMPINT_LT>(self.0, other.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_maskz_set1_epi8(mask, -1)
        })
    }

    /// Lane-wise less-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than-or-equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_epi8_mask::<_MM_CMPINT_LE>(self.0, other.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_maskz_set1_epi8(mask, -1)
        })
    }

    /// Lane-wise greater-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than, all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_epi8_mask::<_MM_CMPINT_LT>(other.0, self.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_maskz_set1_epi8(mask, -1)
        })
    }

    /// Lane-wise greater-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than-or-equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_epi8_mask::<_MM_CMPINT_LE>(other.0, self.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_maskz_set1_epi8(mask, -1)
        })
    }

    // ========== Blending/Selection ==========

    /// Select lanes from `if_true` where mask is all-1s, `if_false` where mask is all-0s.
    ///
    /// The mask should come from a comparison operation like `simd_lt()`.
    ///
    /// # Example
    /// ```ignore
    /// let a = i8x64::splat(token, 1.0);
    /// let b = i8x64::splat(token, 2.0);
    /// let mask = a.simd_lt(b);  // all true
    /// let result = i8x64::blend(mask, a, b);  // selects a
    /// ```
    #[inline(always)]
    pub fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self(unsafe {
            // Convert vector mask to mask register
            let m = _mm512_cmpneq_epi8_mask(mask.0, _mm512_setzero_si512());
            _mm512_mask_blend_epi8(m, if_false.0, if_true.0)
        })
    }
    // ========== Horizontal Operations ==========

    /// Sum all lanes horizontally.
    ///
    /// Returns a scalar containing the sum of all 64 lanes.
    /// Note: This uses a scalar loop. For performance-critical code,
    /// consider keeping values in SIMD until the final reduction.
    #[inline(always)]
    pub fn reduce_add(self) -> i8 {
        self.as_array().iter().copied().fold(0_i8, i8::wrapping_add)
    }

    // ========== Bitwise Unary Operations ==========
    /// Bitwise NOT (complement): flips all bits.
    #[inline(always)]
    pub fn not(self) -> Self {
        Self(unsafe {
            let ones = _mm512_set1_epi8(-1);
            _mm512_xor_si512(self.0, ones)
        })
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
crate::impl_int_arithmetic_ops!(i8x64, _mm512_add_epi8, _mm512_sub_epi8);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
crate::impl_assign_ops!(i8x64);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
crate::impl_bitwise_ops!(i8x64, __m512i, _mm512_and_si512, _mm512_or_si512, _mm512_xor_si512);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
crate::impl_index!(i8x64, i8, 64);

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl From<[i8; 64]> for i8x64 {
    #[inline(always)]
    fn from(arr: [i8; 64]) -> Self {
        // SAFETY: [i8; 64] and __m512i have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl From<i8x64> for [i8; 64] {
    #[inline(always)]
    fn from(v: i8x64) -> Self {
        // SAFETY: __m512i and [i8; 64] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}


// Scalar broadcast operations for i8x64
// These allow `v + 2.0` instead of `v + i8x64::splat(token, 2.0)`

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl Add<i8> for i8x64 {
    type Output = Self;
    /// Add a scalar to all lanes.
    #[inline(always)]
    fn add(self, rhs: i8) -> Self {
        self + Self(unsafe { _mm512_set1_epi8(rhs) })
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl Sub<i8> for i8x64 {
    type Output = Self;
    /// Subtract a scalar from all lanes.
    #[inline(always)]
    fn sub(self, rhs: i8) -> Self {
        self - Self(unsafe { _mm512_set1_epi8(rhs) })
    }
}


// ============================================================================
// u8x64 - 64 x u8 (512-bit)
// ============================================================================

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct u8x64(__m512i);

#[cfg(feature = "bytemuck")]
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
unsafe impl bytemuck::Zeroable for u8x64 {}
#[cfg(feature = "bytemuck")]
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
unsafe impl bytemuck::Pod for u8x64 {}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl u8x64 {
    pub const LANES: usize = 64;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::X64V4Token, data: &[u8; 64]) -> Self {
        Self(unsafe { _mm512_loadu_si512(data.as_ptr() as *const __m512i) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::X64V4Token, v: u8) -> Self {
        Self(unsafe { _mm512_set1_epi8(v as i8) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::X64V4Token) -> Self {
        Self(unsafe { _mm512_setzero_si512() })
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::X64V4Token, arr: [u8; 64]) -> Self {
        // SAFETY: [u8; 64] and __m512i have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [u8; 64]) {
        unsafe { _mm512_storeu_si512(out.as_mut_ptr() as *mut __m512i, self.0) };
    }

    /// Convert to array
    #[inline(always)]
    pub fn to_array(self) -> [u8; 64] {
        let mut out = [0u8; 64];
        self.store(&mut out);
        out
    }

    /// Get reference to underlying array
    #[inline(always)]
    pub fn as_array(&self) -> &[u8; 64] {
        unsafe { &*(self as *const Self as *const [u8; 64]) }
    }

    /// Get mutable reference to underlying array
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut [u8; 64] {
        unsafe { &mut *(self as *mut Self as *mut [u8; 64]) }
    }

    /// Get raw intrinsic type
    #[inline(always)]
    pub fn raw(self) -> __m512i {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports the required SIMD features.
    /// Use token-gated constructors (`load`, `splat`, `zero`) for safe construction.
    #[inline(always)]
    pub unsafe fn from_raw(v: __m512i) -> Self {
        Self(v)
    }

    // ========== Token-gated bytemuck replacements ==========

    /// Reinterpret a slice of scalars as a slice of SIMD vectors (token-gated).
    ///
    /// Returns `None` if the slice length is not a multiple of 64, or
    /// if the slice is not properly aligned.
    ///
    /// This is a safe, token-gated replacement for `bytemuck::cast_slice`.
    #[inline(always)]
    pub fn cast_slice(_: archmage::X64V4Token, slice: &[u8]) -> Option<&[Self]> {
        if slice.len() % 64 != 0 {
            return None;
        }
        let ptr = slice.as_ptr();
        if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {
            return None;
        }
        let len = slice.len() / 64;
        // SAFETY: alignment and length checked, layout is compatible
        Some(unsafe { core::slice::from_raw_parts(ptr as *const Self, len) })
    }

    /// Reinterpret a mutable slice of scalars as a slice of SIMD vectors (token-gated).
    ///
    /// Returns `None` if the slice length is not a multiple of 64, or
    /// if the slice is not properly aligned.
    ///
    /// This is a safe, token-gated replacement for `bytemuck::cast_slice_mut`.
    #[inline(always)]
    pub fn cast_slice_mut(_: archmage::X64V4Token, slice: &mut [u8]) -> Option<&mut [Self]> {
        if slice.len() % 64 != 0 {
            return None;
        }
        let ptr = slice.as_mut_ptr();
        if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {
            return None;
        }
        let len = slice.len() / 64;
        // SAFETY: alignment and length checked, layout is compatible
        Some(unsafe { core::slice::from_raw_parts_mut(ptr as *mut Self, len) })
    }

    /// View this vector as a byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of`.
    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8; 64] {
        // SAFETY: Self is repr(transparent) over __m512i which is 64 bytes
        unsafe { &*(self as *const Self as *const [u8; 64]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 64] {
        // SAFETY: Self is repr(transparent) over __m512i which is 64 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 64]) }
    }

    /// Create from a byte array (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::X64V4Token, bytes: &[u8; 64]) -> Self {
        // SAFETY: [u8; 64] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Element-wise minimum
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(unsafe { _mm512_min_epu8(self.0, other.0) })
    }
    /// Element-wise maximum
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(unsafe { _mm512_max_epu8(self.0, other.0) })
    }
    /// Clamp values between lo and hi
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }
    // ========== Comparisons ==========
    // These return a mask where each lane is all-1s (true) or all-0s (false).
    // Use with `blend()` to select values based on the comparison result.

    /// Lane-wise equality comparison.
    ///
    /// Returns a mask where each lane is all-1s if equality, all-0s otherwise.
    /// Use with `blend(mask, if_true, if_false)` to select values.
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_epu8_mask::<_MM_CMPINT_EQ>(self.0, other.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_maskz_set1_epi8(mask, -1)
        })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if inequality, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_epu8_mask::<_MM_CMPINT_NE>(self.0, other.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_maskz_set1_epi8(mask, -1)
        })
    }

    /// Lane-wise less-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_epu8_mask::<_MM_CMPINT_LT>(self.0, other.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_maskz_set1_epi8(mask, -1)
        })
    }

    /// Lane-wise less-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than-or-equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_epu8_mask::<_MM_CMPINT_LE>(self.0, other.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_maskz_set1_epi8(mask, -1)
        })
    }

    /// Lane-wise greater-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than, all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_epu8_mask::<_MM_CMPINT_LT>(other.0, self.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_maskz_set1_epi8(mask, -1)
        })
    }

    /// Lane-wise greater-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than-or-equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_epu8_mask::<_MM_CMPINT_LE>(other.0, self.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_maskz_set1_epi8(mask, -1)
        })
    }

    // ========== Blending/Selection ==========

    /// Select lanes from `if_true` where mask is all-1s, `if_false` where mask is all-0s.
    ///
    /// The mask should come from a comparison operation like `simd_lt()`.
    ///
    /// # Example
    /// ```ignore
    /// let a = u8x64::splat(token, 1.0);
    /// let b = u8x64::splat(token, 2.0);
    /// let mask = a.simd_lt(b);  // all true
    /// let result = u8x64::blend(mask, a, b);  // selects a
    /// ```
    #[inline(always)]
    pub fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self(unsafe {
            // Convert vector mask to mask register
            let m = _mm512_cmpneq_epi8_mask(mask.0, _mm512_setzero_si512());
            _mm512_mask_blend_epi8(m, if_false.0, if_true.0)
        })
    }
    // ========== Horizontal Operations ==========

    /// Sum all lanes horizontally.
    ///
    /// Returns a scalar containing the sum of all 64 lanes.
    /// Note: This uses a scalar loop. For performance-critical code,
    /// consider keeping values in SIMD until the final reduction.
    #[inline(always)]
    pub fn reduce_add(self) -> u8 {
        self.as_array().iter().copied().fold(0_u8, u8::wrapping_add)
    }

    // ========== Bitwise Unary Operations ==========
    /// Bitwise NOT (complement): flips all bits.
    #[inline(always)]
    pub fn not(self) -> Self {
        Self(unsafe {
            let ones = _mm512_set1_epi8(-1);
            _mm512_xor_si512(self.0, ones)
        })
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
crate::impl_int_arithmetic_ops!(u8x64, _mm512_add_epi8, _mm512_sub_epi8);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
crate::impl_assign_ops!(u8x64);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
crate::impl_bitwise_ops!(u8x64, __m512i, _mm512_and_si512, _mm512_or_si512, _mm512_xor_si512);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
crate::impl_index!(u8x64, u8, 64);

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl From<[u8; 64]> for u8x64 {
    #[inline(always)]
    fn from(arr: [u8; 64]) -> Self {
        // SAFETY: [u8; 64] and __m512i have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl From<u8x64> for [u8; 64] {
    #[inline(always)]
    fn from(v: u8x64) -> Self {
        // SAFETY: __m512i and [u8; 64] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}


// Scalar broadcast operations for u8x64
// These allow `v + 2.0` instead of `v + u8x64::splat(token, 2.0)`

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl Add<u8> for u8x64 {
    type Output = Self;
    /// Add a scalar to all lanes.
    #[inline(always)]
    fn add(self, rhs: u8) -> Self {
        self + Self(unsafe { _mm512_set1_epi8(rhs as i8) })
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl Sub<u8> for u8x64 {
    type Output = Self;
    /// Subtract a scalar from all lanes.
    #[inline(always)]
    fn sub(self, rhs: u8) -> Self {
        self - Self(unsafe { _mm512_set1_epi8(rhs as i8) })
    }
}


// ============================================================================
// i16x32 - 32 x i16 (512-bit)
// ============================================================================

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct i16x32(__m512i);

#[cfg(feature = "bytemuck")]
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
unsafe impl bytemuck::Zeroable for i16x32 {}
#[cfg(feature = "bytemuck")]
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
unsafe impl bytemuck::Pod for i16x32 {}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl i16x32 {
    pub const LANES: usize = 32;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::X64V4Token, data: &[i16; 32]) -> Self {
        Self(unsafe { _mm512_loadu_si512(data.as_ptr() as *const __m512i) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::X64V4Token, v: i16) -> Self {
        Self(unsafe { _mm512_set1_epi16(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::X64V4Token) -> Self {
        Self(unsafe { _mm512_setzero_si512() })
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::X64V4Token, arr: [i16; 32]) -> Self {
        // SAFETY: [i16; 32] and __m512i have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [i16; 32]) {
        unsafe { _mm512_storeu_si512(out.as_mut_ptr() as *mut __m512i, self.0) };
    }

    /// Convert to array
    #[inline(always)]
    pub fn to_array(self) -> [i16; 32] {
        let mut out = [0i16; 32];
        self.store(&mut out);
        out
    }

    /// Get reference to underlying array
    #[inline(always)]
    pub fn as_array(&self) -> &[i16; 32] {
        unsafe { &*(self as *const Self as *const [i16; 32]) }
    }

    /// Get mutable reference to underlying array
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut [i16; 32] {
        unsafe { &mut *(self as *mut Self as *mut [i16; 32]) }
    }

    /// Get raw intrinsic type
    #[inline(always)]
    pub fn raw(self) -> __m512i {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports the required SIMD features.
    /// Use token-gated constructors (`load`, `splat`, `zero`) for safe construction.
    #[inline(always)]
    pub unsafe fn from_raw(v: __m512i) -> Self {
        Self(v)
    }

    // ========== Token-gated bytemuck replacements ==========

    /// Reinterpret a slice of scalars as a slice of SIMD vectors (token-gated).
    ///
    /// Returns `None` if the slice length is not a multiple of 32, or
    /// if the slice is not properly aligned.
    ///
    /// This is a safe, token-gated replacement for `bytemuck::cast_slice`.
    #[inline(always)]
    pub fn cast_slice(_: archmage::X64V4Token, slice: &[i16]) -> Option<&[Self]> {
        if slice.len() % 32 != 0 {
            return None;
        }
        let ptr = slice.as_ptr();
        if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {
            return None;
        }
        let len = slice.len() / 32;
        // SAFETY: alignment and length checked, layout is compatible
        Some(unsafe { core::slice::from_raw_parts(ptr as *const Self, len) })
    }

    /// Reinterpret a mutable slice of scalars as a slice of SIMD vectors (token-gated).
    ///
    /// Returns `None` if the slice length is not a multiple of 32, or
    /// if the slice is not properly aligned.
    ///
    /// This is a safe, token-gated replacement for `bytemuck::cast_slice_mut`.
    #[inline(always)]
    pub fn cast_slice_mut(_: archmage::X64V4Token, slice: &mut [i16]) -> Option<&mut [Self]> {
        if slice.len() % 32 != 0 {
            return None;
        }
        let ptr = slice.as_mut_ptr();
        if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {
            return None;
        }
        let len = slice.len() / 32;
        // SAFETY: alignment and length checked, layout is compatible
        Some(unsafe { core::slice::from_raw_parts_mut(ptr as *mut Self, len) })
    }

    /// View this vector as a byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of`.
    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8; 64] {
        // SAFETY: Self is repr(transparent) over __m512i which is 64 bytes
        unsafe { &*(self as *const Self as *const [u8; 64]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 64] {
        // SAFETY: Self is repr(transparent) over __m512i which is 64 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 64]) }
    }

    /// Create from a byte array (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::X64V4Token, bytes: &[u8; 64]) -> Self {
        // SAFETY: [u8; 64] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Element-wise minimum
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(unsafe { _mm512_min_epi16(self.0, other.0) })
    }
    /// Element-wise maximum
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(unsafe { _mm512_max_epi16(self.0, other.0) })
    }
    /// Clamp values between lo and hi
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }
    /// Absolute value
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(unsafe { _mm512_abs_epi16(self.0) })
    }
    // ========== Comparisons ==========
    // These return a mask where each lane is all-1s (true) or all-0s (false).
    // Use with `blend()` to select values based on the comparison result.

    /// Lane-wise equality comparison.
    ///
    /// Returns a mask where each lane is all-1s if equality, all-0s otherwise.
    /// Use with `blend(mask, if_true, if_false)` to select values.
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_epi16_mask::<_MM_CMPINT_EQ>(self.0, other.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_maskz_set1_epi16(mask, -1)
        })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if inequality, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_epi16_mask::<_MM_CMPINT_NE>(self.0, other.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_maskz_set1_epi16(mask, -1)
        })
    }

    /// Lane-wise less-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_epi16_mask::<_MM_CMPINT_LT>(self.0, other.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_maskz_set1_epi16(mask, -1)
        })
    }

    /// Lane-wise less-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than-or-equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_epi16_mask::<_MM_CMPINT_LE>(self.0, other.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_maskz_set1_epi16(mask, -1)
        })
    }

    /// Lane-wise greater-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than, all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_epi16_mask::<_MM_CMPINT_LT>(other.0, self.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_maskz_set1_epi16(mask, -1)
        })
    }

    /// Lane-wise greater-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than-or-equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_epi16_mask::<_MM_CMPINT_LE>(other.0, self.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_maskz_set1_epi16(mask, -1)
        })
    }

    // ========== Blending/Selection ==========

    /// Select lanes from `if_true` where mask is all-1s, `if_false` where mask is all-0s.
    ///
    /// The mask should come from a comparison operation like `simd_lt()`.
    ///
    /// # Example
    /// ```ignore
    /// let a = i16x32::splat(token, 1.0);
    /// let b = i16x32::splat(token, 2.0);
    /// let mask = a.simd_lt(b);  // all true
    /// let result = i16x32::blend(mask, a, b);  // selects a
    /// ```
    #[inline(always)]
    pub fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self(unsafe {
            // Convert vector mask to mask register
            let m = _mm512_cmpneq_epi16_mask(mask.0, _mm512_setzero_si512());
            _mm512_mask_blend_epi16(m, if_false.0, if_true.0)
        })
    }
    // ========== Horizontal Operations ==========

    /// Sum all lanes horizontally.
    ///
    /// Returns a scalar containing the sum of all 32 lanes.
    /// Note: This uses a scalar loop. For performance-critical code,
    /// consider keeping values in SIMD until the final reduction.
    #[inline(always)]
    pub fn reduce_add(self) -> i16 {
        self.as_array().iter().copied().fold(0_i16, i16::wrapping_add)
    }

    // ========== Bitwise Unary Operations ==========
    /// Bitwise NOT (complement): flips all bits.
    #[inline(always)]
    pub fn not(self) -> Self {
        Self(unsafe {
            let ones = _mm512_set1_epi16(-1);
            _mm512_xor_si512(self.0, ones)
        })
    }
    // ========== Shift Operations ==========
    /// Shift each lane left by `N` bits.
    ///
    /// Bits shifted out are lost; zeros are shifted in.
    #[inline(always)]
    pub fn shl<const N: u32>(self) -> Self {
        Self(unsafe { _mm512_slli_epi16::<N>(self.0) })
    }
    /// Shift each lane right by `N` bits (logical/unsigned shift).
    ///
    /// Bits shifted out are lost; zeros are shifted in.
    #[inline(always)]
    pub fn shr<const N: u32>(self) -> Self {
        Self(unsafe { _mm512_srli_epi16::<N>(self.0) })
    }
    /// Arithmetic shift right by `N` bits (sign-extending).
    ///
    /// The sign bit is replicated into the vacated positions.
    #[inline(always)]
    pub fn shr_arithmetic<const N: u32>(self) -> Self {
        Self(unsafe { _mm512_srai_epi16::<N>(self.0) })
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
crate::impl_int_arithmetic_ops!(i16x32, _mm512_add_epi16, _mm512_sub_epi16);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
crate::impl_int_mul_op!(i16x32, _mm512_mullo_epi16);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
crate::impl_assign_ops!(i16x32);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
crate::impl_bitwise_ops!(i16x32, __m512i, _mm512_and_si512, _mm512_or_si512, _mm512_xor_si512);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
crate::impl_index!(i16x32, i16, 32);

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl From<[i16; 32]> for i16x32 {
    #[inline(always)]
    fn from(arr: [i16; 32]) -> Self {
        // SAFETY: [i16; 32] and __m512i have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl From<i16x32> for [i16; 32] {
    #[inline(always)]
    fn from(v: i16x32) -> Self {
        // SAFETY: __m512i and [i16; 32] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}


// Scalar broadcast operations for i16x32
// These allow `v + 2.0` instead of `v + i16x32::splat(token, 2.0)`

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl Add<i16> for i16x32 {
    type Output = Self;
    /// Add a scalar to all lanes.
    #[inline(always)]
    fn add(self, rhs: i16) -> Self {
        self + Self(unsafe { _mm512_set1_epi16(rhs) })
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl Sub<i16> for i16x32 {
    type Output = Self;
    /// Subtract a scalar from all lanes.
    #[inline(always)]
    fn sub(self, rhs: i16) -> Self {
        self - Self(unsafe { _mm512_set1_epi16(rhs) })
    }
}


// ============================================================================
// u16x32 - 32 x u16 (512-bit)
// ============================================================================

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct u16x32(__m512i);

#[cfg(feature = "bytemuck")]
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
unsafe impl bytemuck::Zeroable for u16x32 {}
#[cfg(feature = "bytemuck")]
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
unsafe impl bytemuck::Pod for u16x32 {}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl u16x32 {
    pub const LANES: usize = 32;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::X64V4Token, data: &[u16; 32]) -> Self {
        Self(unsafe { _mm512_loadu_si512(data.as_ptr() as *const __m512i) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::X64V4Token, v: u16) -> Self {
        Self(unsafe { _mm512_set1_epi16(v as i16) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::X64V4Token) -> Self {
        Self(unsafe { _mm512_setzero_si512() })
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::X64V4Token, arr: [u16; 32]) -> Self {
        // SAFETY: [u16; 32] and __m512i have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [u16; 32]) {
        unsafe { _mm512_storeu_si512(out.as_mut_ptr() as *mut __m512i, self.0) };
    }

    /// Convert to array
    #[inline(always)]
    pub fn to_array(self) -> [u16; 32] {
        let mut out = [0u16; 32];
        self.store(&mut out);
        out
    }

    /// Get reference to underlying array
    #[inline(always)]
    pub fn as_array(&self) -> &[u16; 32] {
        unsafe { &*(self as *const Self as *const [u16; 32]) }
    }

    /// Get mutable reference to underlying array
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut [u16; 32] {
        unsafe { &mut *(self as *mut Self as *mut [u16; 32]) }
    }

    /// Get raw intrinsic type
    #[inline(always)]
    pub fn raw(self) -> __m512i {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports the required SIMD features.
    /// Use token-gated constructors (`load`, `splat`, `zero`) for safe construction.
    #[inline(always)]
    pub unsafe fn from_raw(v: __m512i) -> Self {
        Self(v)
    }

    // ========== Token-gated bytemuck replacements ==========

    /// Reinterpret a slice of scalars as a slice of SIMD vectors (token-gated).
    ///
    /// Returns `None` if the slice length is not a multiple of 32, or
    /// if the slice is not properly aligned.
    ///
    /// This is a safe, token-gated replacement for `bytemuck::cast_slice`.
    #[inline(always)]
    pub fn cast_slice(_: archmage::X64V4Token, slice: &[u16]) -> Option<&[Self]> {
        if slice.len() % 32 != 0 {
            return None;
        }
        let ptr = slice.as_ptr();
        if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {
            return None;
        }
        let len = slice.len() / 32;
        // SAFETY: alignment and length checked, layout is compatible
        Some(unsafe { core::slice::from_raw_parts(ptr as *const Self, len) })
    }

    /// Reinterpret a mutable slice of scalars as a slice of SIMD vectors (token-gated).
    ///
    /// Returns `None` if the slice length is not a multiple of 32, or
    /// if the slice is not properly aligned.
    ///
    /// This is a safe, token-gated replacement for `bytemuck::cast_slice_mut`.
    #[inline(always)]
    pub fn cast_slice_mut(_: archmage::X64V4Token, slice: &mut [u16]) -> Option<&mut [Self]> {
        if slice.len() % 32 != 0 {
            return None;
        }
        let ptr = slice.as_mut_ptr();
        if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {
            return None;
        }
        let len = slice.len() / 32;
        // SAFETY: alignment and length checked, layout is compatible
        Some(unsafe { core::slice::from_raw_parts_mut(ptr as *mut Self, len) })
    }

    /// View this vector as a byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of`.
    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8; 64] {
        // SAFETY: Self is repr(transparent) over __m512i which is 64 bytes
        unsafe { &*(self as *const Self as *const [u8; 64]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 64] {
        // SAFETY: Self is repr(transparent) over __m512i which is 64 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 64]) }
    }

    /// Create from a byte array (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::X64V4Token, bytes: &[u8; 64]) -> Self {
        // SAFETY: [u8; 64] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Element-wise minimum
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(unsafe { _mm512_min_epu16(self.0, other.0) })
    }
    /// Element-wise maximum
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(unsafe { _mm512_max_epu16(self.0, other.0) })
    }
    /// Clamp values between lo and hi
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }
    // ========== Comparisons ==========
    // These return a mask where each lane is all-1s (true) or all-0s (false).
    // Use with `blend()` to select values based on the comparison result.

    /// Lane-wise equality comparison.
    ///
    /// Returns a mask where each lane is all-1s if equality, all-0s otherwise.
    /// Use with `blend(mask, if_true, if_false)` to select values.
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_epu16_mask::<_MM_CMPINT_EQ>(self.0, other.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_maskz_set1_epi16(mask, -1)
        })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if inequality, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_epu16_mask::<_MM_CMPINT_NE>(self.0, other.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_maskz_set1_epi16(mask, -1)
        })
    }

    /// Lane-wise less-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_epu16_mask::<_MM_CMPINT_LT>(self.0, other.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_maskz_set1_epi16(mask, -1)
        })
    }

    /// Lane-wise less-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than-or-equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_epu16_mask::<_MM_CMPINT_LE>(self.0, other.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_maskz_set1_epi16(mask, -1)
        })
    }

    /// Lane-wise greater-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than, all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_epu16_mask::<_MM_CMPINT_LT>(other.0, self.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_maskz_set1_epi16(mask, -1)
        })
    }

    /// Lane-wise greater-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than-or-equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_epu16_mask::<_MM_CMPINT_LE>(other.0, self.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_maskz_set1_epi16(mask, -1)
        })
    }

    // ========== Blending/Selection ==========

    /// Select lanes from `if_true` where mask is all-1s, `if_false` where mask is all-0s.
    ///
    /// The mask should come from a comparison operation like `simd_lt()`.
    ///
    /// # Example
    /// ```ignore
    /// let a = u16x32::splat(token, 1.0);
    /// let b = u16x32::splat(token, 2.0);
    /// let mask = a.simd_lt(b);  // all true
    /// let result = u16x32::blend(mask, a, b);  // selects a
    /// ```
    #[inline(always)]
    pub fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self(unsafe {
            // Convert vector mask to mask register
            let m = _mm512_cmpneq_epi16_mask(mask.0, _mm512_setzero_si512());
            _mm512_mask_blend_epi16(m, if_false.0, if_true.0)
        })
    }
    // ========== Horizontal Operations ==========

    /// Sum all lanes horizontally.
    ///
    /// Returns a scalar containing the sum of all 32 lanes.
    /// Note: This uses a scalar loop. For performance-critical code,
    /// consider keeping values in SIMD until the final reduction.
    #[inline(always)]
    pub fn reduce_add(self) -> u16 {
        self.as_array().iter().copied().fold(0_u16, u16::wrapping_add)
    }

    // ========== Bitwise Unary Operations ==========
    /// Bitwise NOT (complement): flips all bits.
    #[inline(always)]
    pub fn not(self) -> Self {
        Self(unsafe {
            let ones = _mm512_set1_epi16(-1);
            _mm512_xor_si512(self.0, ones)
        })
    }
    // ========== Shift Operations ==========
    /// Shift each lane left by `N` bits.
    ///
    /// Bits shifted out are lost; zeros are shifted in.
    #[inline(always)]
    pub fn shl<const N: u32>(self) -> Self {
        Self(unsafe { _mm512_slli_epi16::<N>(self.0) })
    }
    /// Shift each lane right by `N` bits (logical/unsigned shift).
    ///
    /// Bits shifted out are lost; zeros are shifted in.
    #[inline(always)]
    pub fn shr<const N: u32>(self) -> Self {
        Self(unsafe { _mm512_srli_epi16::<N>(self.0) })
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
crate::impl_int_arithmetic_ops!(u16x32, _mm512_add_epi16, _mm512_sub_epi16);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
crate::impl_int_mul_op!(u16x32, _mm512_mullo_epi16);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
crate::impl_assign_ops!(u16x32);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
crate::impl_bitwise_ops!(u16x32, __m512i, _mm512_and_si512, _mm512_or_si512, _mm512_xor_si512);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
crate::impl_index!(u16x32, u16, 32);

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl From<[u16; 32]> for u16x32 {
    #[inline(always)]
    fn from(arr: [u16; 32]) -> Self {
        // SAFETY: [u16; 32] and __m512i have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl From<u16x32> for [u16; 32] {
    #[inline(always)]
    fn from(v: u16x32) -> Self {
        // SAFETY: __m512i and [u16; 32] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}


// Scalar broadcast operations for u16x32
// These allow `v + 2.0` instead of `v + u16x32::splat(token, 2.0)`

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl Add<u16> for u16x32 {
    type Output = Self;
    /// Add a scalar to all lanes.
    #[inline(always)]
    fn add(self, rhs: u16) -> Self {
        self + Self(unsafe { _mm512_set1_epi16(rhs as i16) })
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl Sub<u16> for u16x32 {
    type Output = Self;
    /// Subtract a scalar from all lanes.
    #[inline(always)]
    fn sub(self, rhs: u16) -> Self {
        self - Self(unsafe { _mm512_set1_epi16(rhs as i16) })
    }
}


// ============================================================================
// i32x16 - 16 x i32 (512-bit)
// ============================================================================

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct i32x16(__m512i);

#[cfg(feature = "bytemuck")]
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
unsafe impl bytemuck::Zeroable for i32x16 {}
#[cfg(feature = "bytemuck")]
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
unsafe impl bytemuck::Pod for i32x16 {}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl i32x16 {
    pub const LANES: usize = 16;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::X64V4Token, data: &[i32; 16]) -> Self {
        Self(unsafe { _mm512_loadu_si512(data.as_ptr() as *const __m512i) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::X64V4Token, v: i32) -> Self {
        Self(unsafe { _mm512_set1_epi32(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::X64V4Token) -> Self {
        Self(unsafe { _mm512_setzero_si512() })
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::X64V4Token, arr: [i32; 16]) -> Self {
        // SAFETY: [i32; 16] and __m512i have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [i32; 16]) {
        unsafe { _mm512_storeu_si512(out.as_mut_ptr() as *mut __m512i, self.0) };
    }

    /// Convert to array
    #[inline(always)]
    pub fn to_array(self) -> [i32; 16] {
        let mut out = [0i32; 16];
        self.store(&mut out);
        out
    }

    /// Get reference to underlying array
    #[inline(always)]
    pub fn as_array(&self) -> &[i32; 16] {
        unsafe { &*(self as *const Self as *const [i32; 16]) }
    }

    /// Get mutable reference to underlying array
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut [i32; 16] {
        unsafe { &mut *(self as *mut Self as *mut [i32; 16]) }
    }

    /// Get raw intrinsic type
    #[inline(always)]
    pub fn raw(self) -> __m512i {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports the required SIMD features.
    /// Use token-gated constructors (`load`, `splat`, `zero`) for safe construction.
    #[inline(always)]
    pub unsafe fn from_raw(v: __m512i) -> Self {
        Self(v)
    }

    // ========== Token-gated bytemuck replacements ==========

    /// Reinterpret a slice of scalars as a slice of SIMD vectors (token-gated).
    ///
    /// Returns `None` if the slice length is not a multiple of 16, or
    /// if the slice is not properly aligned.
    ///
    /// This is a safe, token-gated replacement for `bytemuck::cast_slice`.
    #[inline(always)]
    pub fn cast_slice(_: archmage::X64V4Token, slice: &[i32]) -> Option<&[Self]> {
        if slice.len() % 16 != 0 {
            return None;
        }
        let ptr = slice.as_ptr();
        if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {
            return None;
        }
        let len = slice.len() / 16;
        // SAFETY: alignment and length checked, layout is compatible
        Some(unsafe { core::slice::from_raw_parts(ptr as *const Self, len) })
    }

    /// Reinterpret a mutable slice of scalars as a slice of SIMD vectors (token-gated).
    ///
    /// Returns `None` if the slice length is not a multiple of 16, or
    /// if the slice is not properly aligned.
    ///
    /// This is a safe, token-gated replacement for `bytemuck::cast_slice_mut`.
    #[inline(always)]
    pub fn cast_slice_mut(_: archmage::X64V4Token, slice: &mut [i32]) -> Option<&mut [Self]> {
        if slice.len() % 16 != 0 {
            return None;
        }
        let ptr = slice.as_mut_ptr();
        if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {
            return None;
        }
        let len = slice.len() / 16;
        // SAFETY: alignment and length checked, layout is compatible
        Some(unsafe { core::slice::from_raw_parts_mut(ptr as *mut Self, len) })
    }

    /// View this vector as a byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of`.
    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8; 64] {
        // SAFETY: Self is repr(transparent) over __m512i which is 64 bytes
        unsafe { &*(self as *const Self as *const [u8; 64]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 64] {
        // SAFETY: Self is repr(transparent) over __m512i which is 64 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 64]) }
    }

    /// Create from a byte array (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::X64V4Token, bytes: &[u8; 64]) -> Self {
        // SAFETY: [u8; 64] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Element-wise minimum
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(unsafe { _mm512_min_epi32(self.0, other.0) })
    }
    /// Element-wise maximum
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(unsafe { _mm512_max_epi32(self.0, other.0) })
    }
    /// Clamp values between lo and hi
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }
    /// Absolute value
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(unsafe { _mm512_abs_epi32(self.0) })
    }
    // ========== Comparisons ==========
    // These return a mask where each lane is all-1s (true) or all-0s (false).
    // Use with `blend()` to select values based on the comparison result.

    /// Lane-wise equality comparison.
    ///
    /// Returns a mask where each lane is all-1s if equality, all-0s otherwise.
    /// Use with `blend(mask, if_true, if_false)` to select values.
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_epi32_mask::<_MM_CMPINT_EQ>(self.0, other.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_maskz_set1_epi32(mask, -1)
        })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if inequality, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_epi32_mask::<_MM_CMPINT_NE>(self.0, other.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_maskz_set1_epi32(mask, -1)
        })
    }

    /// Lane-wise less-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_epi32_mask::<_MM_CMPINT_LT>(self.0, other.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_maskz_set1_epi32(mask, -1)
        })
    }

    /// Lane-wise less-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than-or-equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_epi32_mask::<_MM_CMPINT_LE>(self.0, other.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_maskz_set1_epi32(mask, -1)
        })
    }

    /// Lane-wise greater-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than, all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_epi32_mask::<_MM_CMPINT_LT>(other.0, self.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_maskz_set1_epi32(mask, -1)
        })
    }

    /// Lane-wise greater-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than-or-equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_epi32_mask::<_MM_CMPINT_LE>(other.0, self.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_maskz_set1_epi32(mask, -1)
        })
    }

    // ========== Blending/Selection ==========

    /// Select lanes from `if_true` where mask is all-1s, `if_false` where mask is all-0s.
    ///
    /// The mask should come from a comparison operation like `simd_lt()`.
    ///
    /// # Example
    /// ```ignore
    /// let a = i32x16::splat(token, 1.0);
    /// let b = i32x16::splat(token, 2.0);
    /// let mask = a.simd_lt(b);  // all true
    /// let result = i32x16::blend(mask, a, b);  // selects a
    /// ```
    #[inline(always)]
    pub fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self(unsafe {
            // Convert vector mask to mask register
            let m = _mm512_cmpneq_epi32_mask(mask.0, _mm512_setzero_si512());
            _mm512_mask_blend_epi32(m, if_false.0, if_true.0)
        })
    }
    // ========== Horizontal Operations ==========

    /// Sum all lanes horizontally.
    ///
    /// Returns a scalar containing the sum of all 16 lanes.
    /// Note: This uses a scalar loop. For performance-critical code,
    /// consider keeping values in SIMD until the final reduction.
    #[inline(always)]
    pub fn reduce_add(self) -> i32 {
        self.as_array().iter().copied().fold(0_i32, i32::wrapping_add)
    }

    // ========== Type Conversions ==========

    /// Convert to single-precision floats.
    #[inline(always)]
    pub fn to_f32x16(self) -> f32x16 {
        f32x16(unsafe { _mm512_cvtepi32_ps(self.0) })
    }

    // ========== Bitwise Unary Operations ==========
    /// Bitwise NOT (complement): flips all bits.
    #[inline(always)]
    pub fn not(self) -> Self {
        Self(unsafe {
            let ones = _mm512_set1_epi32(-1);
            _mm512_xor_si512(self.0, ones)
        })
    }
    // ========== Shift Operations ==========
    /// Shift each lane left by `N` bits.
    ///
    /// Bits shifted out are lost; zeros are shifted in.
    #[inline(always)]
    pub fn shl<const N: u32>(self) -> Self {
        Self(unsafe { _mm512_slli_epi32::<N>(self.0) })
    }
    /// Shift each lane right by `N` bits (logical/unsigned shift).
    ///
    /// Bits shifted out are lost; zeros are shifted in.
    #[inline(always)]
    pub fn shr<const N: u32>(self) -> Self {
        Self(unsafe { _mm512_srli_epi32::<N>(self.0) })
    }
    /// Arithmetic shift right by `N` bits (sign-extending).
    ///
    /// The sign bit is replicated into the vacated positions.
    #[inline(always)]
    pub fn shr_arithmetic<const N: u32>(self) -> Self {
        Self(unsafe { _mm512_srai_epi32::<N>(self.0) })
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
crate::impl_int_arithmetic_ops!(i32x16, _mm512_add_epi32, _mm512_sub_epi32);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
crate::impl_int_mul_op!(i32x16, _mm512_mullo_epi32);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
crate::impl_assign_ops!(i32x16);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
crate::impl_bitwise_ops!(i32x16, __m512i, _mm512_and_si512, _mm512_or_si512, _mm512_xor_si512);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
crate::impl_index!(i32x16, i32, 16);

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl From<[i32; 16]> for i32x16 {
    #[inline(always)]
    fn from(arr: [i32; 16]) -> Self {
        // SAFETY: [i32; 16] and __m512i have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl From<i32x16> for [i32; 16] {
    #[inline(always)]
    fn from(v: i32x16) -> Self {
        // SAFETY: __m512i and [i32; 16] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}


// Scalar broadcast operations for i32x16
// These allow `v + 2.0` instead of `v + i32x16::splat(token, 2.0)`

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl Add<i32> for i32x16 {
    type Output = Self;
    /// Add a scalar to all lanes.
    #[inline(always)]
    fn add(self, rhs: i32) -> Self {
        self + Self(unsafe { _mm512_set1_epi32(rhs) })
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl Sub<i32> for i32x16 {
    type Output = Self;
    /// Subtract a scalar from all lanes.
    #[inline(always)]
    fn sub(self, rhs: i32) -> Self {
        self - Self(unsafe { _mm512_set1_epi32(rhs) })
    }
}


// ============================================================================
// u32x16 - 16 x u32 (512-bit)
// ============================================================================

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct u32x16(__m512i);

#[cfg(feature = "bytemuck")]
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
unsafe impl bytemuck::Zeroable for u32x16 {}
#[cfg(feature = "bytemuck")]
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
unsafe impl bytemuck::Pod for u32x16 {}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl u32x16 {
    pub const LANES: usize = 16;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::X64V4Token, data: &[u32; 16]) -> Self {
        Self(unsafe { _mm512_loadu_si512(data.as_ptr() as *const __m512i) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::X64V4Token, v: u32) -> Self {
        Self(unsafe { _mm512_set1_epi32(v as i32) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::X64V4Token) -> Self {
        Self(unsafe { _mm512_setzero_si512() })
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::X64V4Token, arr: [u32; 16]) -> Self {
        // SAFETY: [u32; 16] and __m512i have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [u32; 16]) {
        unsafe { _mm512_storeu_si512(out.as_mut_ptr() as *mut __m512i, self.0) };
    }

    /// Convert to array
    #[inline(always)]
    pub fn to_array(self) -> [u32; 16] {
        let mut out = [0u32; 16];
        self.store(&mut out);
        out
    }

    /// Get reference to underlying array
    #[inline(always)]
    pub fn as_array(&self) -> &[u32; 16] {
        unsafe { &*(self as *const Self as *const [u32; 16]) }
    }

    /// Get mutable reference to underlying array
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut [u32; 16] {
        unsafe { &mut *(self as *mut Self as *mut [u32; 16]) }
    }

    /// Get raw intrinsic type
    #[inline(always)]
    pub fn raw(self) -> __m512i {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports the required SIMD features.
    /// Use token-gated constructors (`load`, `splat`, `zero`) for safe construction.
    #[inline(always)]
    pub unsafe fn from_raw(v: __m512i) -> Self {
        Self(v)
    }

    // ========== Token-gated bytemuck replacements ==========

    /// Reinterpret a slice of scalars as a slice of SIMD vectors (token-gated).
    ///
    /// Returns `None` if the slice length is not a multiple of 16, or
    /// if the slice is not properly aligned.
    ///
    /// This is a safe, token-gated replacement for `bytemuck::cast_slice`.
    #[inline(always)]
    pub fn cast_slice(_: archmage::X64V4Token, slice: &[u32]) -> Option<&[Self]> {
        if slice.len() % 16 != 0 {
            return None;
        }
        let ptr = slice.as_ptr();
        if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {
            return None;
        }
        let len = slice.len() / 16;
        // SAFETY: alignment and length checked, layout is compatible
        Some(unsafe { core::slice::from_raw_parts(ptr as *const Self, len) })
    }

    /// Reinterpret a mutable slice of scalars as a slice of SIMD vectors (token-gated).
    ///
    /// Returns `None` if the slice length is not a multiple of 16, or
    /// if the slice is not properly aligned.
    ///
    /// This is a safe, token-gated replacement for `bytemuck::cast_slice_mut`.
    #[inline(always)]
    pub fn cast_slice_mut(_: archmage::X64V4Token, slice: &mut [u32]) -> Option<&mut [Self]> {
        if slice.len() % 16 != 0 {
            return None;
        }
        let ptr = slice.as_mut_ptr();
        if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {
            return None;
        }
        let len = slice.len() / 16;
        // SAFETY: alignment and length checked, layout is compatible
        Some(unsafe { core::slice::from_raw_parts_mut(ptr as *mut Self, len) })
    }

    /// View this vector as a byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of`.
    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8; 64] {
        // SAFETY: Self is repr(transparent) over __m512i which is 64 bytes
        unsafe { &*(self as *const Self as *const [u8; 64]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 64] {
        // SAFETY: Self is repr(transparent) over __m512i which is 64 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 64]) }
    }

    /// Create from a byte array (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::X64V4Token, bytes: &[u8; 64]) -> Self {
        // SAFETY: [u8; 64] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Element-wise minimum
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(unsafe { _mm512_min_epu32(self.0, other.0) })
    }
    /// Element-wise maximum
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(unsafe { _mm512_max_epu32(self.0, other.0) })
    }
    /// Clamp values between lo and hi
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }
    // ========== Comparisons ==========
    // These return a mask where each lane is all-1s (true) or all-0s (false).
    // Use with `blend()` to select values based on the comparison result.

    /// Lane-wise equality comparison.
    ///
    /// Returns a mask where each lane is all-1s if equality, all-0s otherwise.
    /// Use with `blend(mask, if_true, if_false)` to select values.
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_epu32_mask::<_MM_CMPINT_EQ>(self.0, other.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_maskz_set1_epi32(mask, -1)
        })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if inequality, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_epu32_mask::<_MM_CMPINT_NE>(self.0, other.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_maskz_set1_epi32(mask, -1)
        })
    }

    /// Lane-wise less-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_epu32_mask::<_MM_CMPINT_LT>(self.0, other.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_maskz_set1_epi32(mask, -1)
        })
    }

    /// Lane-wise less-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than-or-equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_epu32_mask::<_MM_CMPINT_LE>(self.0, other.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_maskz_set1_epi32(mask, -1)
        })
    }

    /// Lane-wise greater-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than, all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_epu32_mask::<_MM_CMPINT_LT>(other.0, self.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_maskz_set1_epi32(mask, -1)
        })
    }

    /// Lane-wise greater-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than-or-equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_epu32_mask::<_MM_CMPINT_LE>(other.0, self.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_maskz_set1_epi32(mask, -1)
        })
    }

    // ========== Blending/Selection ==========

    /// Select lanes from `if_true` where mask is all-1s, `if_false` where mask is all-0s.
    ///
    /// The mask should come from a comparison operation like `simd_lt()`.
    ///
    /// # Example
    /// ```ignore
    /// let a = u32x16::splat(token, 1.0);
    /// let b = u32x16::splat(token, 2.0);
    /// let mask = a.simd_lt(b);  // all true
    /// let result = u32x16::blend(mask, a, b);  // selects a
    /// ```
    #[inline(always)]
    pub fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self(unsafe {
            // Convert vector mask to mask register
            let m = _mm512_cmpneq_epi32_mask(mask.0, _mm512_setzero_si512());
            _mm512_mask_blend_epi32(m, if_false.0, if_true.0)
        })
    }
    // ========== Horizontal Operations ==========

    /// Sum all lanes horizontally.
    ///
    /// Returns a scalar containing the sum of all 16 lanes.
    /// Note: This uses a scalar loop. For performance-critical code,
    /// consider keeping values in SIMD until the final reduction.
    #[inline(always)]
    pub fn reduce_add(self) -> u32 {
        self.as_array().iter().copied().fold(0_u32, u32::wrapping_add)
    }

    // ========== Bitwise Unary Operations ==========
    /// Bitwise NOT (complement): flips all bits.
    #[inline(always)]
    pub fn not(self) -> Self {
        Self(unsafe {
            let ones = _mm512_set1_epi32(-1);
            _mm512_xor_si512(self.0, ones)
        })
    }
    // ========== Shift Operations ==========
    /// Shift each lane left by `N` bits.
    ///
    /// Bits shifted out are lost; zeros are shifted in.
    #[inline(always)]
    pub fn shl<const N: u32>(self) -> Self {
        Self(unsafe { _mm512_slli_epi32::<N>(self.0) })
    }
    /// Shift each lane right by `N` bits (logical/unsigned shift).
    ///
    /// Bits shifted out are lost; zeros are shifted in.
    #[inline(always)]
    pub fn shr<const N: u32>(self) -> Self {
        Self(unsafe { _mm512_srli_epi32::<N>(self.0) })
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
crate::impl_int_arithmetic_ops!(u32x16, _mm512_add_epi32, _mm512_sub_epi32);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
crate::impl_int_mul_op!(u32x16, _mm512_mullo_epi32);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
crate::impl_assign_ops!(u32x16);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
crate::impl_bitwise_ops!(u32x16, __m512i, _mm512_and_si512, _mm512_or_si512, _mm512_xor_si512);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
crate::impl_index!(u32x16, u32, 16);

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl From<[u32; 16]> for u32x16 {
    #[inline(always)]
    fn from(arr: [u32; 16]) -> Self {
        // SAFETY: [u32; 16] and __m512i have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl From<u32x16> for [u32; 16] {
    #[inline(always)]
    fn from(v: u32x16) -> Self {
        // SAFETY: __m512i and [u32; 16] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}


// Scalar broadcast operations for u32x16
// These allow `v + 2.0` instead of `v + u32x16::splat(token, 2.0)`

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl Add<u32> for u32x16 {
    type Output = Self;
    /// Add a scalar to all lanes.
    #[inline(always)]
    fn add(self, rhs: u32) -> Self {
        self + Self(unsafe { _mm512_set1_epi32(rhs as i32) })
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl Sub<u32> for u32x16 {
    type Output = Self;
    /// Subtract a scalar from all lanes.
    #[inline(always)]
    fn sub(self, rhs: u32) -> Self {
        self - Self(unsafe { _mm512_set1_epi32(rhs as i32) })
    }
}


// ============================================================================
// i64x8 - 8 x i64 (512-bit)
// ============================================================================

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct i64x8(__m512i);

#[cfg(feature = "bytemuck")]
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
unsafe impl bytemuck::Zeroable for i64x8 {}
#[cfg(feature = "bytemuck")]
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
unsafe impl bytemuck::Pod for i64x8 {}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl i64x8 {
    pub const LANES: usize = 8;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::X64V4Token, data: &[i64; 8]) -> Self {
        Self(unsafe { _mm512_loadu_si512(data.as_ptr() as *const __m512i) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::X64V4Token, v: i64) -> Self {
        Self(unsafe { _mm512_set1_epi64(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::X64V4Token) -> Self {
        Self(unsafe { _mm512_setzero_si512() })
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::X64V4Token, arr: [i64; 8]) -> Self {
        // SAFETY: [i64; 8] and __m512i have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [i64; 8]) {
        unsafe { _mm512_storeu_si512(out.as_mut_ptr() as *mut __m512i, self.0) };
    }

    /// Convert to array
    #[inline(always)]
    pub fn to_array(self) -> [i64; 8] {
        let mut out = [0i64; 8];
        self.store(&mut out);
        out
    }

    /// Get reference to underlying array
    #[inline(always)]
    pub fn as_array(&self) -> &[i64; 8] {
        unsafe { &*(self as *const Self as *const [i64; 8]) }
    }

    /// Get mutable reference to underlying array
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut [i64; 8] {
        unsafe { &mut *(self as *mut Self as *mut [i64; 8]) }
    }

    /// Get raw intrinsic type
    #[inline(always)]
    pub fn raw(self) -> __m512i {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports the required SIMD features.
    /// Use token-gated constructors (`load`, `splat`, `zero`) for safe construction.
    #[inline(always)]
    pub unsafe fn from_raw(v: __m512i) -> Self {
        Self(v)
    }

    // ========== Token-gated bytemuck replacements ==========

    /// Reinterpret a slice of scalars as a slice of SIMD vectors (token-gated).
    ///
    /// Returns `None` if the slice length is not a multiple of 8, or
    /// if the slice is not properly aligned.
    ///
    /// This is a safe, token-gated replacement for `bytemuck::cast_slice`.
    #[inline(always)]
    pub fn cast_slice(_: archmage::X64V4Token, slice: &[i64]) -> Option<&[Self]> {
        if slice.len() % 8 != 0 {
            return None;
        }
        let ptr = slice.as_ptr();
        if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {
            return None;
        }
        let len = slice.len() / 8;
        // SAFETY: alignment and length checked, layout is compatible
        Some(unsafe { core::slice::from_raw_parts(ptr as *const Self, len) })
    }

    /// Reinterpret a mutable slice of scalars as a slice of SIMD vectors (token-gated).
    ///
    /// Returns `None` if the slice length is not a multiple of 8, or
    /// if the slice is not properly aligned.
    ///
    /// This is a safe, token-gated replacement for `bytemuck::cast_slice_mut`.
    #[inline(always)]
    pub fn cast_slice_mut(_: archmage::X64V4Token, slice: &mut [i64]) -> Option<&mut [Self]> {
        if slice.len() % 8 != 0 {
            return None;
        }
        let ptr = slice.as_mut_ptr();
        if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {
            return None;
        }
        let len = slice.len() / 8;
        // SAFETY: alignment and length checked, layout is compatible
        Some(unsafe { core::slice::from_raw_parts_mut(ptr as *mut Self, len) })
    }

    /// View this vector as a byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of`.
    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8; 64] {
        // SAFETY: Self is repr(transparent) over __m512i which is 64 bytes
        unsafe { &*(self as *const Self as *const [u8; 64]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 64] {
        // SAFETY: Self is repr(transparent) over __m512i which is 64 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 64]) }
    }

    /// Create from a byte array (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::X64V4Token, bytes: &[u8; 64]) -> Self {
        // SAFETY: [u8; 64] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Element-wise minimum
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(unsafe { _mm512_min_epi64(self.0, other.0) })
    }
    /// Element-wise maximum
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(unsafe { _mm512_max_epi64(self.0, other.0) })
    }
    /// Clamp values between lo and hi
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }
    /// Absolute value
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(unsafe { _mm512_abs_epi64(self.0) })
    }
    // ========== Comparisons ==========
    // These return a mask where each lane is all-1s (true) or all-0s (false).
    // Use with `blend()` to select values based on the comparison result.

    /// Lane-wise equality comparison.
    ///
    /// Returns a mask where each lane is all-1s if equality, all-0s otherwise.
    /// Use with `blend(mask, if_true, if_false)` to select values.
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_epi64_mask::<_MM_CMPINT_EQ>(self.0, other.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_maskz_set1_epi64(mask, -1)
        })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if inequality, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_epi64_mask::<_MM_CMPINT_NE>(self.0, other.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_maskz_set1_epi64(mask, -1)
        })
    }

    /// Lane-wise less-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_epi64_mask::<_MM_CMPINT_LT>(self.0, other.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_maskz_set1_epi64(mask, -1)
        })
    }

    /// Lane-wise less-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than-or-equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_epi64_mask::<_MM_CMPINT_LE>(self.0, other.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_maskz_set1_epi64(mask, -1)
        })
    }

    /// Lane-wise greater-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than, all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_epi64_mask::<_MM_CMPINT_LT>(other.0, self.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_maskz_set1_epi64(mask, -1)
        })
    }

    /// Lane-wise greater-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than-or-equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_epi64_mask::<_MM_CMPINT_LE>(other.0, self.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_maskz_set1_epi64(mask, -1)
        })
    }

    // ========== Blending/Selection ==========

    /// Select lanes from `if_true` where mask is all-1s, `if_false` where mask is all-0s.
    ///
    /// The mask should come from a comparison operation like `simd_lt()`.
    ///
    /// # Example
    /// ```ignore
    /// let a = i64x8::splat(token, 1.0);
    /// let b = i64x8::splat(token, 2.0);
    /// let mask = a.simd_lt(b);  // all true
    /// let result = i64x8::blend(mask, a, b);  // selects a
    /// ```
    #[inline(always)]
    pub fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self(unsafe {
            // Convert vector mask to mask register
            let m = _mm512_cmpneq_epi64_mask(mask.0, _mm512_setzero_si512());
            _mm512_mask_blend_epi64(m, if_false.0, if_true.0)
        })
    }
    // ========== Horizontal Operations ==========

    /// Sum all lanes horizontally.
    ///
    /// Returns a scalar containing the sum of all 8 lanes.
    /// Note: This uses a scalar loop. For performance-critical code,
    /// consider keeping values in SIMD until the final reduction.
    #[inline(always)]
    pub fn reduce_add(self) -> i64 {
        self.as_array().iter().copied().fold(0_i64, i64::wrapping_add)
    }

    // ========== Bitwise Unary Operations ==========
    /// Bitwise NOT (complement): flips all bits.
    #[inline(always)]
    pub fn not(self) -> Self {
        Self(unsafe {
            let ones = _mm512_set1_epi64(-1);
            _mm512_xor_si512(self.0, ones)
        })
    }
    // ========== Shift Operations ==========
    /// Shift each lane left by `N` bits.
    ///
    /// Bits shifted out are lost; zeros are shifted in.
    #[inline(always)]
    pub fn shl<const N: u32>(self) -> Self {
        Self(unsafe { _mm512_slli_epi64::<N>(self.0) })
    }
    /// Shift each lane right by `N` bits (logical/unsigned shift).
    ///
    /// Bits shifted out are lost; zeros are shifted in.
    #[inline(always)]
    pub fn shr<const N: u32>(self) -> Self {
        Self(unsafe { _mm512_srli_epi64::<N>(self.0) })
    }
    /// Arithmetic shift right by `N` bits (sign-extending).
    ///
    /// The sign bit is replicated into the vacated positions.
    #[inline(always)]
    pub fn shr_arithmetic<const N: u32>(self) -> Self {
        Self(unsafe { _mm512_srai_epi64::<N>(self.0) })
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
crate::impl_int_arithmetic_ops!(i64x8, _mm512_add_epi64, _mm512_sub_epi64);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
crate::impl_assign_ops!(i64x8);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
crate::impl_bitwise_ops!(i64x8, __m512i, _mm512_and_si512, _mm512_or_si512, _mm512_xor_si512);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
crate::impl_index!(i64x8, i64, 8);

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl From<[i64; 8]> for i64x8 {
    #[inline(always)]
    fn from(arr: [i64; 8]) -> Self {
        // SAFETY: [i64; 8] and __m512i have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl From<i64x8> for [i64; 8] {
    #[inline(always)]
    fn from(v: i64x8) -> Self {
        // SAFETY: __m512i and [i64; 8] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}


// Scalar broadcast operations for i64x8
// These allow `v + 2.0` instead of `v + i64x8::splat(token, 2.0)`

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl Add<i64> for i64x8 {
    type Output = Self;
    /// Add a scalar to all lanes.
    #[inline(always)]
    fn add(self, rhs: i64) -> Self {
        self + Self(unsafe { _mm512_set1_epi64(rhs) })
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl Sub<i64> for i64x8 {
    type Output = Self;
    /// Subtract a scalar from all lanes.
    #[inline(always)]
    fn sub(self, rhs: i64) -> Self {
        self - Self(unsafe { _mm512_set1_epi64(rhs) })
    }
}


// ============================================================================
// u64x8 - 8 x u64 (512-bit)
// ============================================================================

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct u64x8(__m512i);

#[cfg(feature = "bytemuck")]
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
unsafe impl bytemuck::Zeroable for u64x8 {}
#[cfg(feature = "bytemuck")]
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
unsafe impl bytemuck::Pod for u64x8 {}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl u64x8 {
    pub const LANES: usize = 8;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::X64V4Token, data: &[u64; 8]) -> Self {
        Self(unsafe { _mm512_loadu_si512(data.as_ptr() as *const __m512i) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::X64V4Token, v: u64) -> Self {
        Self(unsafe { _mm512_set1_epi64(v as i64) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::X64V4Token) -> Self {
        Self(unsafe { _mm512_setzero_si512() })
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::X64V4Token, arr: [u64; 8]) -> Self {
        // SAFETY: [u64; 8] and __m512i have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [u64; 8]) {
        unsafe { _mm512_storeu_si512(out.as_mut_ptr() as *mut __m512i, self.0) };
    }

    /// Convert to array
    #[inline(always)]
    pub fn to_array(self) -> [u64; 8] {
        let mut out = [0u64; 8];
        self.store(&mut out);
        out
    }

    /// Get reference to underlying array
    #[inline(always)]
    pub fn as_array(&self) -> &[u64; 8] {
        unsafe { &*(self as *const Self as *const [u64; 8]) }
    }

    /// Get mutable reference to underlying array
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut [u64; 8] {
        unsafe { &mut *(self as *mut Self as *mut [u64; 8]) }
    }

    /// Get raw intrinsic type
    #[inline(always)]
    pub fn raw(self) -> __m512i {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports the required SIMD features.
    /// Use token-gated constructors (`load`, `splat`, `zero`) for safe construction.
    #[inline(always)]
    pub unsafe fn from_raw(v: __m512i) -> Self {
        Self(v)
    }

    // ========== Token-gated bytemuck replacements ==========

    /// Reinterpret a slice of scalars as a slice of SIMD vectors (token-gated).
    ///
    /// Returns `None` if the slice length is not a multiple of 8, or
    /// if the slice is not properly aligned.
    ///
    /// This is a safe, token-gated replacement for `bytemuck::cast_slice`.
    #[inline(always)]
    pub fn cast_slice(_: archmage::X64V4Token, slice: &[u64]) -> Option<&[Self]> {
        if slice.len() % 8 != 0 {
            return None;
        }
        let ptr = slice.as_ptr();
        if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {
            return None;
        }
        let len = slice.len() / 8;
        // SAFETY: alignment and length checked, layout is compatible
        Some(unsafe { core::slice::from_raw_parts(ptr as *const Self, len) })
    }

    /// Reinterpret a mutable slice of scalars as a slice of SIMD vectors (token-gated).
    ///
    /// Returns `None` if the slice length is not a multiple of 8, or
    /// if the slice is not properly aligned.
    ///
    /// This is a safe, token-gated replacement for `bytemuck::cast_slice_mut`.
    #[inline(always)]
    pub fn cast_slice_mut(_: archmage::X64V4Token, slice: &mut [u64]) -> Option<&mut [Self]> {
        if slice.len() % 8 != 0 {
            return None;
        }
        let ptr = slice.as_mut_ptr();
        if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {
            return None;
        }
        let len = slice.len() / 8;
        // SAFETY: alignment and length checked, layout is compatible
        Some(unsafe { core::slice::from_raw_parts_mut(ptr as *mut Self, len) })
    }

    /// View this vector as a byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of`.
    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8; 64] {
        // SAFETY: Self is repr(transparent) over __m512i which is 64 bytes
        unsafe { &*(self as *const Self as *const [u8; 64]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 64] {
        // SAFETY: Self is repr(transparent) over __m512i which is 64 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 64]) }
    }

    /// Create from a byte array (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::X64V4Token, bytes: &[u8; 64]) -> Self {
        // SAFETY: [u8; 64] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Element-wise minimum
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(unsafe { _mm512_min_epu64(self.0, other.0) })
    }
    /// Element-wise maximum
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(unsafe { _mm512_max_epu64(self.0, other.0) })
    }
    /// Clamp values between lo and hi
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }
    // ========== Comparisons ==========
    // These return a mask where each lane is all-1s (true) or all-0s (false).
    // Use with `blend()` to select values based on the comparison result.

    /// Lane-wise equality comparison.
    ///
    /// Returns a mask where each lane is all-1s if equality, all-0s otherwise.
    /// Use with `blend(mask, if_true, if_false)` to select values.
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_epu64_mask::<_MM_CMPINT_EQ>(self.0, other.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_maskz_set1_epi64(mask, -1)
        })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if inequality, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_epu64_mask::<_MM_CMPINT_NE>(self.0, other.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_maskz_set1_epi64(mask, -1)
        })
    }

    /// Lane-wise less-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_epu64_mask::<_MM_CMPINT_LT>(self.0, other.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_maskz_set1_epi64(mask, -1)
        })
    }

    /// Lane-wise less-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than-or-equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_epu64_mask::<_MM_CMPINT_LE>(self.0, other.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_maskz_set1_epi64(mask, -1)
        })
    }

    /// Lane-wise greater-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than, all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_epu64_mask::<_MM_CMPINT_LT>(other.0, self.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_maskz_set1_epi64(mask, -1)
        })
    }

    /// Lane-wise greater-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than-or-equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(unsafe {
            let mask = _mm512_cmp_epu64_mask::<_MM_CMPINT_LE>(other.0, self.0);
            // Expand mask to vector: -1 where true, 0 where false
            _mm512_maskz_set1_epi64(mask, -1)
        })
    }

    // ========== Blending/Selection ==========

    /// Select lanes from `if_true` where mask is all-1s, `if_false` where mask is all-0s.
    ///
    /// The mask should come from a comparison operation like `simd_lt()`.
    ///
    /// # Example
    /// ```ignore
    /// let a = u64x8::splat(token, 1.0);
    /// let b = u64x8::splat(token, 2.0);
    /// let mask = a.simd_lt(b);  // all true
    /// let result = u64x8::blend(mask, a, b);  // selects a
    /// ```
    #[inline(always)]
    pub fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self(unsafe {
            // Convert vector mask to mask register
            let m = _mm512_cmpneq_epi64_mask(mask.0, _mm512_setzero_si512());
            _mm512_mask_blend_epi64(m, if_false.0, if_true.0)
        })
    }
    // ========== Horizontal Operations ==========

    /// Sum all lanes horizontally.
    ///
    /// Returns a scalar containing the sum of all 8 lanes.
    /// Note: This uses a scalar loop. For performance-critical code,
    /// consider keeping values in SIMD until the final reduction.
    #[inline(always)]
    pub fn reduce_add(self) -> u64 {
        self.as_array().iter().copied().fold(0_u64, u64::wrapping_add)
    }

    // ========== Bitwise Unary Operations ==========
    /// Bitwise NOT (complement): flips all bits.
    #[inline(always)]
    pub fn not(self) -> Self {
        Self(unsafe {
            let ones = _mm512_set1_epi64(-1);
            _mm512_xor_si512(self.0, ones)
        })
    }
    // ========== Shift Operations ==========
    /// Shift each lane left by `N` bits.
    ///
    /// Bits shifted out are lost; zeros are shifted in.
    #[inline(always)]
    pub fn shl<const N: u32>(self) -> Self {
        Self(unsafe { _mm512_slli_epi64::<N>(self.0) })
    }
    /// Shift each lane right by `N` bits (logical/unsigned shift).
    ///
    /// Bits shifted out are lost; zeros are shifted in.
    #[inline(always)]
    pub fn shr<const N: u32>(self) -> Self {
        Self(unsafe { _mm512_srli_epi64::<N>(self.0) })
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
crate::impl_int_arithmetic_ops!(u64x8, _mm512_add_epi64, _mm512_sub_epi64);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
crate::impl_assign_ops!(u64x8);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
crate::impl_bitwise_ops!(u64x8, __m512i, _mm512_and_si512, _mm512_or_si512, _mm512_xor_si512);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
crate::impl_index!(u64x8, u64, 8);

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl From<[u64; 8]> for u64x8 {
    #[inline(always)]
    fn from(arr: [u64; 8]) -> Self {
        // SAFETY: [u64; 8] and __m512i have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl From<u64x8> for [u64; 8] {
    #[inline(always)]
    fn from(v: u64x8) -> Self {
        // SAFETY: __m512i and [u64; 8] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}


// Scalar broadcast operations for u64x8
// These allow `v + 2.0` instead of `v + u64x8::splat(token, 2.0)`

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl Add<u64> for u64x8 {
    type Output = Self;
    /// Add a scalar to all lanes.
    #[inline(always)]
    fn add(self, rhs: u64) -> Self {
        self + Self(unsafe { _mm512_set1_epi64(rhs as i64) })
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl Sub<u64> for u64x8 {
    type Output = Self;
    /// Subtract a scalar from all lanes.
    #[inline(always)]
    fn sub(self, rhs: u64) -> Self {
        self - Self(unsafe { _mm512_set1_epi64(rhs as i64) })
    }
}


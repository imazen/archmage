//! 256-bit (AVX/AVX2) SIMD types.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

use core::arch::x86_64::*;
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

// ============================================================================
// f32x8 - 8 x f32 (256-bit)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct f32x8(__m256);

#[cfg(target_arch = "x86_64")]
impl f32x8 {
    pub const LANES: usize = 8;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::X64V3Token, data: &[f32; 8]) -> Self {
        Self(unsafe { _mm256_loadu_ps(data.as_ptr()) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::X64V3Token, v: f32) -> Self {
        Self(unsafe { _mm256_set1_ps(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::X64V3Token) -> Self {
        Self(unsafe { _mm256_setzero_ps() })
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::X64V3Token, arr: [f32; 8]) -> Self {
        // SAFETY: [f32; 8] and __m256 have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [f32; 8]) {
        unsafe { _mm256_storeu_ps(out.as_mut_ptr(), self.0) };
    }

    /// Convert to array
    #[inline(always)]
    pub fn to_array(self) -> [f32; 8] {
        let mut out = [0.0f32; 8];
        self.store(&mut out);
        out
    }

    /// Get reference to underlying array
    #[inline(always)]
    pub fn as_array(&self) -> &[f32; 8] {
        unsafe { &*(self as *const Self as *const [f32; 8]) }
    }

    /// Get mutable reference to underlying array
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut [f32; 8] {
        unsafe { &mut *(self as *mut Self as *mut [f32; 8]) }
    }

    /// Get raw intrinsic type
    #[inline(always)]
    pub fn raw(self) -> __m256 {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports the required SIMD features.
    /// Use token-gated constructors (`load`, `splat`, `zero`) for safe construction.
    #[inline(always)]
    pub unsafe fn from_raw(v: __m256) -> Self {
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
    pub fn cast_slice(_: archmage::X64V3Token, slice: &[f32]) -> Option<&[Self]> {
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
    pub fn cast_slice_mut(_: archmage::X64V3Token, slice: &mut [f32]) -> Option<&mut [Self]> {
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
    pub fn as_bytes(&self) -> &[u8; 32] {
        // SAFETY: Self is repr(transparent) over __m256 which is 32 bytes
        unsafe { &*(self as *const Self as *const [u8; 32]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 32] {
        // SAFETY: Self is repr(transparent) over __m256 which is 32 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 32]) }
    }

    /// Create from a byte array reference (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::X64V3Token, bytes: &[u8; 32]) -> Self {
        // SAFETY: [u8; 32] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Create from an owned byte array (token-gated, zero-cost).
    ///
    /// This is a zero-cost transmute from an owned byte array.
    #[inline(always)]
    pub fn from_bytes_owned(_: archmage::X64V3Token, bytes: [u8; 32]) -> Self {
        // SAFETY: [u8; 32] and Self have identical size
        Self(unsafe { core::mem::transmute(bytes) })
    }

    /// Element-wise minimum
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(unsafe { _mm256_min_ps(self.0, other.0) })
    }
    /// Element-wise maximum
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(unsafe { _mm256_max_ps(self.0, other.0) })
    }
    /// Clamp values between lo and hi
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }

    /// Square root
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        Self(unsafe { _mm256_sqrt_ps(self.0) })
    }
    /// Absolute value
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(unsafe {
            let mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFFu32 as i32));
            _mm256_and_ps(self.0, mask)
        })
    }
    /// Round toward negative infinity
    #[inline(always)]
    pub fn floor(self) -> Self {
        Self(unsafe { _mm256_floor_ps(self.0) })
    }
    /// Round toward positive infinity
    #[inline(always)]
    pub fn ceil(self) -> Self {
        Self(unsafe { _mm256_ceil_ps(self.0) })
    }
    /// Round to nearest integer
    #[inline(always)]
    pub fn round(self) -> Self {
        Self(unsafe {
            _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(self.0)
        })
    }
    /// Fused multiply-add: self * a + b
    #[inline(always)]
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        Self(unsafe { _mm256_fmadd_ps(self.0, a.0, b.0) })
    }

    /// Fused multiply-sub: self * a - b
    #[inline(always)]
    pub fn mul_sub(self, a: Self, b: Self) -> Self {
        Self(unsafe { _mm256_fmsub_ps(self.0, a.0, b.0) })
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
        Self(unsafe { _mm256_cmp_ps::<_CMP_EQ_OQ>(self.0, other.0) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if inequality, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(unsafe { _mm256_cmp_ps::<_CMP_NEQ_OQ>(self.0, other.0) })
    }

    /// Lane-wise less-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(unsafe { _mm256_cmp_ps::<_CMP_LT_OQ>(self.0, other.0) })
    }

    /// Lane-wise less-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than-or-equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(unsafe { _mm256_cmp_ps::<_CMP_LE_OQ>(self.0, other.0) })
    }

    /// Lane-wise greater-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than, all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe { _mm256_cmp_ps::<_CMP_GT_OQ>(self.0, other.0) })
    }

    /// Lane-wise greater-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than-or-equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(unsafe { _mm256_cmp_ps::<_CMP_GE_OQ>(self.0, other.0) })
    }

    // ========== Blending/Selection ==========

    /// Select lanes from `if_true` where mask is all-1s, `if_false` where mask is all-0s.
    ///
    /// The mask should come from a comparison operation like `simd_lt()`.
    ///
    /// # Example
    /// ```ignore
    /// let a = f32x8::splat(token, 1.0);
    /// let b = f32x8::splat(token, 2.0);
    /// let mask = a.simd_lt(b);  // all true
    /// let result = f32x8::blend(mask, a, b);  // selects a
    /// ```
    #[inline(always)]
    pub fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self(unsafe { _mm256_blendv_ps(if_false.0, if_true.0, mask.0) })
    }

    // ========== Horizontal Operations ==========

    /// Sum all lanes horizontally.
    ///
    /// Returns a scalar containing the sum of all 8 lanes.
    #[inline(always)]
    pub fn reduce_add(self) -> f32 {
        unsafe {
            let hi = _mm256_extractf128_ps::<1>(self.0);
            let lo = _mm256_castps256_ps128(self.0);
            let sum = _mm_add_ps(lo, hi);
            let h1 = _mm_hadd_ps(sum, sum);
            let h2 = _mm_hadd_ps(h1, h1);
            _mm_cvtss_f32(h2)
        }
    }

    /// Find the minimum value across all lanes.
    #[inline(always)]
    pub fn reduce_min(self) -> f32 {
        unsafe {
            let hi = _mm256_extractf128_ps::<1>(self.0);
            let lo = _mm256_castps256_ps128(self.0);
            let m = _mm_min_ps(lo, hi);
            let shuf = _mm_shuffle_ps::<0b10_11_00_01>(m, m);
            let m1 = _mm_min_ps(m, shuf);
            let shuf2 = _mm_shuffle_ps::<0b00_00_10_10>(m1, m1);
            let m2 = _mm_min_ps(m1, shuf2);
            _mm_cvtss_f32(m2)
        }
    }

    /// Find the maximum value across all lanes.
    #[inline(always)]
    pub fn reduce_max(self) -> f32 {
        unsafe {
            let hi = _mm256_extractf128_ps::<1>(self.0);
            let lo = _mm256_castps256_ps128(self.0);
            let m = _mm_max_ps(lo, hi);
            let shuf = _mm_shuffle_ps::<0b10_11_00_01>(m, m);
            let m1 = _mm_max_ps(m, shuf);
            let shuf2 = _mm_shuffle_ps::<0b00_00_10_10>(m1, m1);
            let m2 = _mm_max_ps(m1, shuf2);
            _mm_cvtss_f32(m2)
        }
    }

    // ========== Type Conversions ==========

    /// Convert to signed 32-bit integers, rounding toward zero (truncation).
    ///
    /// Values outside the representable range become `i32::MIN` (0x80000000).
    #[inline(always)]
    pub fn to_i32x8(self) -> i32x8 {
        i32x8(unsafe { _mm256_cvttps_epi32(self.0) })
    }

    /// Convert to signed 32-bit integers, rounding to nearest even.
    ///
    /// Values outside the representable range become `i32::MIN` (0x80000000).
    #[inline(always)]
    pub fn to_i32x8_round(self) -> i32x8 {
        i32x8(unsafe { _mm256_cvtps_epi32(self.0) })
    }

    /// Create from signed 32-bit integers.
    #[inline(always)]
    pub fn from_i32x8(v: i32x8) -> Self {
        Self(unsafe { _mm256_cvtepi32_ps(v.0) })
    }

    // ========== Approximation Operations ==========

    /// Fast reciprocal approximation (1/x) with ~12-bit precision.
    ///
    /// For full precision, use `recip()` which applies Newton-Raphson refinement.
    #[inline(always)]
    pub fn rcp_approx(self) -> Self {
        Self(unsafe { _mm256_rcp_ps(self.0) })
    }

    /// Precise reciprocal (1/x) using Newton-Raphson refinement.
    ///
    /// More accurate than `rcp_approx()` but slower. For maximum speed
    /// with acceptable precision loss, use `rcp_approx()`.
    #[inline(always)]
    pub fn recip(self) -> Self {
        // Newton-Raphson: x' = x * (2 - a*x)
        let approx = self.rcp_approx();
        let two = Self(unsafe { _mm256_set1_ps(2.0) });
        // One iteration gives ~24-bit precision from ~12-bit
        approx * (two - self * approx)
    }

    /// Fast reciprocal square root approximation (1/sqrt(x)) with ~12-bit precision.
    ///
    /// For full precision, use `sqrt().recip()` or apply Newton-Raphson manually.
    #[inline(always)]
    pub fn rsqrt_approx(self) -> Self {
        Self(unsafe { _mm256_rsqrt_ps(self.0) })
    }

    /// Precise reciprocal square root (1/sqrt(x)) using Newton-Raphson refinement.
    #[inline(always)]
    pub fn rsqrt(self) -> Self {
        // Newton-Raphson for rsqrt: y' = 0.5 * y * (3 - x * y * y)
        let approx = self.rsqrt_approx();
        let half = Self(unsafe { _mm256_set1_ps(0.5) });
        let three = Self(unsafe { _mm256_set1_ps(3.0) });
        half * approx * (three - self * approx * approx)
    }

    // ========== Bitwise Unary Operations ==========

    /// Bitwise NOT (complement): flips all bits.
    #[inline(always)]
    pub fn not(self) -> Self {
        Self(unsafe {
            let ones = _mm256_set1_epi32(-1);
            let as_int = _mm256_castps_si256(self.0);
            _mm256_castsi256_ps(_mm256_xor_si256(as_int, ones))
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
            let x_bits = _mm256_castps_si256(self.0);
            let offset = _mm256_set1_epi32(0x3f2aaaab_u32 as i32);
            let exp_bits = _mm256_sub_epi32(x_bits, offset);
            let exp_shifted = _mm256_srai_epi32::<23>(exp_bits);

            let mantissa_bits = _mm256_sub_epi32(x_bits, _mm256_slli_epi32::<23>(exp_shifted));
            let mantissa = _mm256_castsi256_ps(mantissa_bits);
            let exp_val = _mm256_cvtepi32_ps(exp_shifted);

            let one = _mm256_set1_ps(1.0);
            let m = _mm256_sub_ps(mantissa, one);

            // Horner's for numerator: P2*m^2 + P1*m + P0
            let yp = _mm256_fmadd_ps(_mm256_set1_ps(P2), m, _mm256_set1_ps(P1));
            let yp = _mm256_fmadd_ps(yp, m, _mm256_set1_ps(P0));

            // Horner's for denominator: Q2*m^2 + Q1*m + Q0
            let yq = _mm256_fmadd_ps(_mm256_set1_ps(Q2), m, _mm256_set1_ps(Q1));
            let yq = _mm256_fmadd_ps(yq, m, _mm256_set1_ps(Q0));

            Self(_mm256_add_ps(_mm256_div_ps(yp, yq), exp_val))
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
            let x = _mm256_max_ps(self.0, _mm256_set1_ps(-126.0));
            let x = _mm256_min_ps(x, _mm256_set1_ps(126.0));

            // Split into integer and fractional parts
            let xi = _mm256_floor_ps(x);
            let xf = _mm256_sub_ps(x, xi);

            // Polynomial for 2^frac
            let poly = _mm256_fmadd_ps(_mm256_set1_ps(C3), xf, _mm256_set1_ps(C2));
            let poly = _mm256_fmadd_ps(poly, xf, _mm256_set1_ps(C1));
            let poly = _mm256_fmadd_ps(poly, xf, _mm256_set1_ps(C0));

            // Scale by 2^integer using bit manipulation
            let xi_i32 = _mm256_cvtps_epi32(xi);
            let bias = _mm256_set1_epi32(127);
            let scale_bits = _mm256_slli_epi32::<23>(_mm256_add_epi32(xi_i32, bias));
            let scale = _mm256_castsi256_ps(scale_bits);

            Self(_mm256_mul_ps(poly, scale))
        }
    }

    /// Low-precision natural logarithm.
    ///
    /// Computed as `log2_lowp(x) * ln(2)`. For higher precision, use `ln_midp()`.
    #[inline(always)]
    pub fn ln_lowp(self) -> Self {
        const LN2: f32 = core::f32::consts::LN_2;
        unsafe { Self(_mm256_mul_ps(self.log2_lowp().0, _mm256_set1_ps(LN2))) }
    }

    /// Low-precision natural exponential (e^x).
    ///
    /// Computed as `exp2_lowp(x * log2(e))`. For higher precision, use `exp_midp()`.
    #[inline(always)]
    pub fn exp_lowp(self) -> Self {
        const LOG2_E: f32 = core::f32::consts::LOG2_E;
        unsafe { Self(_mm256_mul_ps(self.0, _mm256_set1_ps(LOG2_E))).exp2_lowp() }
    }

    /// Low-precision base-10 logarithm.
    ///
    /// Computed as `log2_lowp(x) / log2(10)`.
    #[inline(always)]
    pub fn log10_lowp(self) -> Self {
        const LOG10_2: f32 = core::f32::consts::LOG10_2; // 1/log2(10)
        unsafe { Self(_mm256_mul_ps(self.log2_lowp().0, _mm256_set1_ps(LOG10_2))) }
    }

    /// Low-precision power function (self^n).
    ///
    /// Computed as `exp2_lowp(n * log2_lowp(self))`. For higher precision, use `pow_midp()`.
    /// Note: Only valid for positive self values.
    #[inline(always)]
    pub fn pow_lowp(self, n: f32) -> Self {
        unsafe { Self(_mm256_mul_ps(self.log2_lowp().0, _mm256_set1_ps(n))).exp2_lowp() }
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
        const ONE: u32 = 0x3f800000; // 1.0 in f32 bits
        const MANTISSA_MASK: i32 = 0x007fffff_u32 as i32;
        const EXPONENT_BIAS: i32 = 127;

        // Coefficients for odd polynomial on y = (a-1)/(a+1)
        const C0: f32 = 2.885_390_08; // 2/ln(2)
        const C1: f32 = 0.961_800_76; // y^2 coefficient
        const C2: f32 = 0.576_974_45; // y^4 coefficient
        const C3: f32 = 0.434_411_97; // y^6 coefficient

        unsafe {
            let x_bits = _mm256_castps_si256(self.0);

            // Normalize mantissa to [sqrt(2)/2, sqrt(2)]
            let offset = _mm256_set1_epi32((ONE - SQRT2_OVER_2) as i32);
            let adjusted = _mm256_add_epi32(x_bits, offset);

            // Extract exponent
            let exp_raw = _mm256_srai_epi32::<23>(adjusted);
            let exp_biased = _mm256_sub_epi32(exp_raw, _mm256_set1_epi32(EXPONENT_BIAS));
            let n = _mm256_cvtepi32_ps(exp_biased);

            // Reconstruct normalized mantissa
            let mantissa_bits = _mm256_and_si256(adjusted, _mm256_set1_epi32(MANTISSA_MASK));
            let a_bits = _mm256_add_epi32(mantissa_bits, _mm256_set1_epi32(SQRT2_OVER_2 as i32));
            let a = _mm256_castsi256_ps(a_bits);

            // y = (a - 1) / (a + 1)
            let one = _mm256_set1_ps(1.0);
            let a_minus_1 = _mm256_sub_ps(a, one);
            let a_plus_1 = _mm256_add_ps(a, one);
            let y = _mm256_div_ps(a_minus_1, a_plus_1);

            // y^2
            let y2 = _mm256_mul_ps(y, y);

            // Polynomial: c0 + y^2*(c1 + y^2*(c2 + y^2*c3))
            let poly = _mm256_fmadd_ps(_mm256_set1_ps(C3), y2, _mm256_set1_ps(C2));
            let poly = _mm256_fmadd_ps(poly, y2, _mm256_set1_ps(C1));
            let poly = _mm256_fmadd_ps(poly, y2, _mm256_set1_ps(C0));

            // Result: y * poly + n
            Self(_mm256_fmadd_ps(y, poly, n))
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

            // Edge case masks
            let zero = _mm256_setzero_ps();
            let is_zero = _mm256_cmp_ps::<_CMP_EQ_OQ>(self.0, zero);
            let is_neg = _mm256_cmp_ps::<_CMP_LT_OQ>(self.0, zero);
            let is_inf = _mm256_cmp_ps::<_CMP_EQ_OQ>(self.0, _mm256_set1_ps(f32::INFINITY));
            let is_nan = _mm256_cmp_ps::<_CMP_UNORD_Q>(self.0, self.0);

            // Apply corrections
            let neg_inf = _mm256_set1_ps(f32::NEG_INFINITY);
            let pos_inf = _mm256_set1_ps(f32::INFINITY);
            let nan = _mm256_set1_ps(f32::NAN);

            let r = _mm256_blendv_ps(result.0, neg_inf, is_zero);
            let r = _mm256_blendv_ps(r, nan, is_neg);
            let r = _mm256_blendv_ps(r, pos_inf, is_inf);
            let r = _mm256_blendv_ps(r, nan, is_nan);
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
            let x = _mm256_max_ps(self.0, _mm256_set1_ps(-126.0));
            let x = _mm256_min_ps(x, _mm256_set1_ps(126.0));

            let xi = _mm256_floor_ps(x);
            let xf = _mm256_sub_ps(x, xi);

            // Horner's method with 6 coefficients
            let poly = _mm256_fmadd_ps(_mm256_set1_ps(C6), xf, _mm256_set1_ps(C5));
            let poly = _mm256_fmadd_ps(poly, xf, _mm256_set1_ps(C4));
            let poly = _mm256_fmadd_ps(poly, xf, _mm256_set1_ps(C3));
            let poly = _mm256_fmadd_ps(poly, xf, _mm256_set1_ps(C2));
            let poly = _mm256_fmadd_ps(poly, xf, _mm256_set1_ps(C1));
            let poly = _mm256_fmadd_ps(poly, xf, _mm256_set1_ps(C0));

            // Scale by 2^integer
            let xi_i32 = _mm256_cvtps_epi32(xi);
            let bias = _mm256_set1_epi32(127);
            let scale_bits = _mm256_slli_epi32::<23>(_mm256_add_epi32(xi_i32, bias));
            let scale = _mm256_castsi256_ps(scale_bits);

            Self(_mm256_mul_ps(poly, scale))
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

            // Edge case masks
            let is_overflow = _mm256_cmp_ps::<_CMP_GE_OQ>(self.0, _mm256_set1_ps(128.0));
            let is_underflow = _mm256_cmp_ps::<_CMP_LT_OQ>(self.0, _mm256_set1_ps(-150.0));
            let is_nan = _mm256_cmp_ps::<_CMP_UNORD_Q>(self.0, self.0);

            // Apply corrections
            let pos_inf = _mm256_set1_ps(f32::INFINITY);
            let zero = _mm256_setzero_ps();
            let nan = _mm256_set1_ps(f32::NAN);

            let r = _mm256_blendv_ps(result.0, pos_inf, is_overflow);
            let r = _mm256_blendv_ps(r, zero, is_underflow);
            let r = _mm256_blendv_ps(r, nan, is_nan);
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
            Self(_mm256_mul_ps(
                self.log2_midp_unchecked().0,
                _mm256_set1_ps(n),
            ))
            .exp2_midp_unchecked()
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
        unsafe { Self(_mm256_mul_ps(self.log2_midp().0, _mm256_set1_ps(n))).exp2_midp() }
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
            Self(_mm256_mul_ps(
                self.log2_midp_unchecked().0,
                _mm256_set1_ps(LN2),
            ))
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
        unsafe { Self(_mm256_mul_ps(self.log2_midp().0, _mm256_set1_ps(LN2))) }
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
        unsafe { Self(_mm256_mul_ps(self.0, _mm256_set1_ps(LOG2_E))).exp2_midp_unchecked() }
    }

    /// Mid-precision natural exponential (e^x).
    ///
    /// Computed as `exp2_midp(x * log2(e))`.
    ///
    /// Handles edge cases: exp(x>88) = inf, exp(x<-103) = 0, exp(NaN) = NaN.
    #[inline(always)]
    pub fn exp_midp(self) -> Self {
        const LOG2_E: f32 = core::f32::consts::LOG2_E;
        unsafe { Self(_mm256_mul_ps(self.0, _mm256_set1_ps(LOG2_E))).exp2_midp() }
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
            let x_arr: [f32; 8] = core::mem::transmute(self.0);
            let mut y_arr = [0.0f32; 8];

            for i in 0..8 {
                let xi = x_arr[i];
                let ui = xi.to_bits();
                let hx = ui & 0x7FFF_FFFF; // abs bits
                // Initial approximation: bits/3 + B1 (always positive)
                let approx = hx / 3 + B1;
                y_arr[i] = f32::from_bits(approx);
            }

            let abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0), self.0);
            let sign_bits = _mm256_and_ps(self.0, _mm256_set1_ps(-0.0));
            let mut y = core::mem::transmute::<_, _>(y_arr);

            // Newton-Raphson: y = y * (2*x + y^3) / (x + 2*y^3)
            // Two iterations for full f32 precision
            let two = _mm256_set1_ps(2.0);

            // Iteration 1
            let y3 = _mm256_mul_ps(_mm256_mul_ps(y, y), y);
            let num = _mm256_fmadd_ps(two, abs_x, y3);
            let den = _mm256_fmadd_ps(two, y3, abs_x);
            y = _mm256_mul_ps(y, _mm256_div_ps(num, den));

            // Iteration 2
            let y3 = _mm256_mul_ps(_mm256_mul_ps(y, y), y);
            let num = _mm256_fmadd_ps(two, abs_x, y3);
            let den = _mm256_fmadd_ps(two, y3, abs_x);
            y = _mm256_mul_ps(y, _mm256_div_ps(num, den));

            // Restore sign
            Self(_mm256_or_ps(y, sign_bits))
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

            // Edge case masks
            let zero = _mm256_setzero_ps();
            let is_zero = _mm256_cmp_ps::<_CMP_EQ_OQ>(self.0, zero);
            let abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0), self.0);
            let is_inf = _mm256_cmp_ps::<_CMP_EQ_OQ>(abs_x, _mm256_set1_ps(f32::INFINITY));
            let is_nan = _mm256_cmp_ps::<_CMP_UNORD_Q>(self.0, self.0);

            // Apply corrections (use self.0 for zero to preserve sign)
            let r = _mm256_blendv_ps(result.0, self.0, is_zero); // ±0 -> ±0
            let r = _mm256_blendv_ps(r, self.0, is_inf); // ±inf -> ±inf
            let r = _mm256_blendv_ps(r, _mm256_set1_ps(f32::NAN), is_nan);
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
            const SCALE_UP: f32 = 16777216.0; // 2^24
            const SCALE_DOWN: f32 = 0.00390625; // 2^(-8) = cbrt(2^(-24))
            const DENORM_LIMIT: f32 = 1.17549435e-38; // Smallest normal f32

            let abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0), self.0);
            let is_denorm = _mm256_cmp_ps::<_CMP_LT_OQ>(abs_x, _mm256_set1_ps(DENORM_LIMIT));

            // Scale up denormals
            let scaled_x = _mm256_mul_ps(self.0, _mm256_set1_ps(SCALE_UP));
            let x_for_cbrt = _mm256_blendv_ps(self.0, scaled_x, is_denorm);

            // Compute cbrt with edge case handling
            let result = Self(x_for_cbrt).cbrt_midp();

            // Scale down results from denormal inputs
            let scaled_result = _mm256_mul_ps(result.0, _mm256_set1_ps(SCALE_DOWN));
            Self(_mm256_blendv_ps(result.0, scaled_result, is_denorm))
        }
    }

    // ========== Load and Convert ==========

    /// Load 8 u8 values and convert to f32x8.
    ///
    /// Useful for image processing: load pixel values directly to float.
    #[inline(always)]
    pub fn from_u8(bytes: &[u8; 8]) -> Self {
        unsafe {
            // Load 8 bytes into low part of XMM register
            let b = _mm_loadl_epi64(bytes.as_ptr() as *const __m128i);
            let i32s = _mm256_cvtepu8_epi32(b);
            Self(_mm256_cvtepi32_ps(i32s))
        }
    }

    /// Convert to 8 u8 values with saturation.
    ///
    /// Values are clamped to [0, 255] and rounded.
    #[inline(always)]
    pub fn to_u8(self) -> [u8; 8] {
        unsafe {
            // Convert to i32
            let i32s = _mm256_cvtps_epi32(self.0);
            // Pack i32 to i16 (within lanes, then combine)
            let lo = _mm256_castsi256_si128(i32s);
            let hi = _mm256_extracti128_si256::<1>(i32s);
            let i16s = _mm_packs_epi32(lo, hi);
            // Pack i16 to u8
            let u8s = _mm_packus_epi16(i16s, i16s);
            let mut result = [0u8; 8];
            _mm_storel_epi64(result.as_mut_ptr() as *mut __m128i, u8s);
            result
        }
    }

    // ========== Interleave Operations ==========

    /// Interleave low elements within 128-bit lanes.
    ///
    /// [a0,a1,a2,a3,a4,a5,a6,a7] + [b0,b1,b2,b3,b4,b5,b6,b7]
    /// → [a0,b0,a1,b1,a4,b4,a5,b5]
    #[inline(always)]
    pub fn interleave_lo(self, other: Self) -> Self {
        Self(unsafe { _mm256_unpacklo_ps(self.0, other.0) })
    }

    /// Interleave high elements within 128-bit lanes.
    ///
    /// [a0,a1,a2,a3,a4,a5,a6,a7] + [b0,b1,b2,b3,b4,b5,b6,b7]
    /// → [a2,b2,a3,b3,a6,b6,a7,b7]
    #[inline(always)]
    pub fn interleave_hi(self, other: Self) -> Self {
        Self(unsafe { _mm256_unpackhi_ps(self.0, other.0) })
    }

    /// Interleave two vectors: returns (interleave_lo, interleave_hi)
    #[inline(always)]
    pub fn interleave(self, other: Self) -> (Self, Self) {
        (self.interleave_lo(other), self.interleave_hi(other))
    }

    // ========== 4-Channel Interleave/Deinterleave ==========

    /// Deinterleave 8 RGBA pixels from AoS to SoA format.
    ///
    /// Input: 4 f32x8 vectors, where pairs of adjacent elements form RGBA pixels.
    /// Each input vector contains 2 complete RGBA pixels:
    /// - `rgba[0]` = [R0, G0, B0, A0, R1, G1, B1, A1]
    /// - `rgba[1]` = [R2, G2, B2, A2, R3, G3, B3, A3]
    /// - `rgba[2]` = [R4, G4, B4, A4, R5, G5, B5, A5]
    /// - `rgba[3]` = [R6, G6, B6, A6, R7, G7, B7, A7]
    ///
    /// Output: 4 f32x8 vectors, one per channel:
    /// - `[0]` = [R0, R1, R2, R3, R4, R5, R6, R7]
    /// - `[1]` = [G0, G1, G2, G3, G4, G5, G6, G7]
    /// - `[2]` = [B0, B1, B2, B3, B4, B5, B6, B7]
    /// - `[3]` = [A0, A1, A2, A3, A4, A5, A6, A7]
    #[inline]
    pub fn deinterleave_4ch(rgba: [Self; 4]) -> [Self; 4] {
        unsafe {
            // Stage 1: Unpack pairs
            // unpacklo: [a0,b0,a1,b1, a4,b4,a5,b5]
            // unpackhi: [a2,b2,a3,b3, a6,b6,a7,b7]
            let rg_lo = _mm256_unpacklo_ps(rgba[0].0, rgba[1].0); // [R0,R2,G0,G2, R1,R3,G1,G3]
            let rg_hi = _mm256_unpackhi_ps(rgba[0].0, rgba[1].0); // [B0,B2,A0,A2, B1,B3,A1,A3]
            let rg_lo2 = _mm256_unpacklo_ps(rgba[2].0, rgba[3].0); // [R4,R6,G4,G6, R5,R7,G5,G7]
            let rg_hi2 = _mm256_unpackhi_ps(rgba[2].0, rgba[3].0); // [B4,B6,A4,A6, B5,B7,A5,A7]

            // Stage 2: Shuffle to separate R,G and B,A
            let r_g_01 = _mm256_unpacklo_ps(rg_lo, rg_lo2); // [R0,R4,R2,R6, R1,R5,R3,R7]
            let r_g_23 = _mm256_unpackhi_ps(rg_lo, rg_lo2); // [G0,G4,G2,G6, G1,G5,G3,G7]
            let b_a_01 = _mm256_unpacklo_ps(rg_hi, rg_hi2); // [B0,B4,B2,B6, B1,B5,B3,B7]
            let b_a_23 = _mm256_unpackhi_ps(rg_hi, rg_hi2); // [A0,A4,A2,A6, A1,A5,A3,A7]

            // Stage 3: Final permute to get contiguous channels
            // Need to reorder: [0,4,2,6,1,5,3,7] → [0,1,2,3,4,5,6,7]
            let perm = _mm256_setr_epi32(0, 4, 2, 6, 1, 5, 3, 7);
            let r = _mm256_permutevar8x32_ps(r_g_01, perm);
            let g = _mm256_permutevar8x32_ps(r_g_23, perm);
            let b = _mm256_permutevar8x32_ps(b_a_01, perm);
            let a = _mm256_permutevar8x32_ps(b_a_23, perm);

            [Self(r), Self(g), Self(b), Self(a)]
        }
    }

    /// Interleave 4 channels from SoA to AoS format.
    ///
    /// Input: 4 f32x8 vectors, one per channel (R, G, B, A).
    /// Output: 4 f32x8 vectors in interleaved AoS format.
    ///
    /// This is the inverse of `deinterleave_4ch`.
    #[inline]
    pub fn interleave_4ch(channels: [Self; 4]) -> [Self; 4] {
        unsafe {
            let r = channels[0].0;
            let g = channels[1].0;
            let b = channels[2].0;
            let a = channels[3].0;

            // Interleave R with G: [R0,G0,R1,G1, R4,G4,R5,G5]
            let rg_lo = _mm256_unpacklo_ps(r, g);
            // [R2,G2,R3,G3, R6,G6,R7,G7]
            let rg_hi = _mm256_unpackhi_ps(r, g);

            // Interleave B with A
            let ba_lo = _mm256_unpacklo_ps(b, a);
            let ba_hi = _mm256_unpackhi_ps(b, a);

            // Combine RG with BA: [R0,G0,B0,A0, R4,G4,B4,A4]
            let rgba_0 = _mm256_shuffle_ps::<0x44>(rg_lo, ba_lo);
            let rgba_1 = _mm256_shuffle_ps::<0xEE>(rg_lo, ba_lo);
            let rgba_2 = _mm256_shuffle_ps::<0x44>(rg_hi, ba_hi);
            let rgba_3 = _mm256_shuffle_ps::<0xEE>(rg_hi, ba_hi);

            // Permute to get final layout
            let out0 = _mm256_permute2f128_ps::<0x20>(rgba_0, rgba_1);
            let out1 = _mm256_permute2f128_ps::<0x20>(rgba_2, rgba_3);
            let out2 = _mm256_permute2f128_ps::<0x31>(rgba_0, rgba_1);
            let out3 = _mm256_permute2f128_ps::<0x31>(rgba_2, rgba_3);

            [Self(out0), Self(out1), Self(out2), Self(out3)]
        }
    }

    /// Load 8 RGBA u8 pixels and deinterleave to 4 f32x8 channel vectors.
    ///
    /// Input: 32 bytes = 8 RGBA pixels in interleaved format.
    /// Output: (R, G, B, A) where each is f32x8 with values in [0.0, 255.0].
    #[inline]
    pub fn load_8_rgba_u8(rgba: &[u8; 32]) -> (Self, Self, Self, Self) {
        unsafe {
            // Load 32 bytes
            let v = _mm256_loadu_si256(rgba.as_ptr() as *const __m256i);

            // Use vpshufb to gather channels within each 128-bit lane
            // Lane 0: pixels 0-3, Lane 1: pixels 4-7
            let r_mask = _mm256_setr_epi8(
                0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 4, 8, 12, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            );
            let g_mask = _mm256_setr_epi8(
                1, 5, 9, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 9, 13, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            );
            let b_mask = _mm256_setr_epi8(
                2, 6, 10, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 6, 10, 14, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            );
            let a_mask = _mm256_setr_epi8(
                3, 7, 11, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, 7, 11, 15, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            );

            // Gather each channel's bytes into low 4 bytes of each lane
            let r_bytes = _mm256_shuffle_epi8(v, r_mask);
            let g_bytes = _mm256_shuffle_epi8(v, g_mask);
            let b_bytes = _mm256_shuffle_epi8(v, b_mask);
            let a_bytes = _mm256_shuffle_epi8(v, a_mask);

            // Extract low 128-bit and high 128-bit lanes, combine low 4 bytes of each
            // to get 8 consecutive bytes, then extend to f32x8
            let r_lo = _mm256_castsi256_si128(r_bytes);
            let r_hi = _mm256_extracti128_si256::<1>(r_bytes);
            let r_combined = _mm_unpacklo_epi32(r_lo, r_hi); // [R0-3, R4-7, ...]
            let r_f32 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(r_combined));

            let g_lo = _mm256_castsi256_si128(g_bytes);
            let g_hi = _mm256_extracti128_si256::<1>(g_bytes);
            let g_combined = _mm_unpacklo_epi32(g_lo, g_hi);
            let g_f32 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(g_combined));

            let b_lo = _mm256_castsi256_si128(b_bytes);
            let b_hi = _mm256_extracti128_si256::<1>(b_bytes);
            let b_combined = _mm_unpacklo_epi32(b_lo, b_hi);
            let b_f32 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b_combined));

            let a_lo = _mm256_castsi256_si128(a_bytes);
            let a_hi = _mm256_extracti128_si256::<1>(a_bytes);
            let a_combined = _mm_unpacklo_epi32(a_lo, a_hi);
            let a_f32 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(a_combined));

            (Self(r_f32), Self(g_f32), Self(b_f32), Self(a_f32))
        }
    }

    /// Interleave 4 f32x8 channels and store as 8 RGBA u8 pixels.
    ///
    /// Input: (R, G, B, A) channel vectors with values that will be clamped to [0, 255].
    /// Output: 32 bytes = 8 RGBA pixels in interleaved format.
    #[inline]
    pub fn store_8_rgba_u8(r: Self, g: Self, b: Self, a: Self) -> [u8; 32] {
        unsafe {
            // Convert f32 to i32
            let ri = _mm256_cvtps_epi32(r.0);
            let gi = _mm256_cvtps_epi32(g.0);
            let bi = _mm256_cvtps_epi32(b.0);
            let ai = _mm256_cvtps_epi32(a.0);

            // Pack to i16 (need to handle AVX2's lane-wise packing)
            // _mm256_packs_epi32 packs within lanes: [lo0-3, hi0-3] + [lo4-7, hi4-7]
            // → [lo0-3 as i16, lo4-7 as i16, hi0-3 as i16, hi4-7 as i16]

            // Pack R,G and B,A together
            let rg = _mm256_packs_epi32(ri, gi); // [R0-3,G0-3, R4-7,G4-7] as i16
            let ba = _mm256_packs_epi32(bi, ai); // [B0-3,A0-3, B4-7,A4-7] as i16

            // Pack i16 to u8
            let rgba = _mm256_packus_epi16(rg, ba); // [R0-3,G0-3,B0-3,A0-3, R4-7,G4-7,B4-7,A4-7]

            // Shuffle within each lane to get RGBA order
            let shuf = _mm256_setr_epi8(
                0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 0, 4, 8, 12, 1, 5, 9, 13, 2,
                6, 10, 14, 3, 7, 11, 15,
            );
            let shuffled = _mm256_shuffle_epi8(rgba, shuf);

            let mut out = [0u8; 32];
            _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, shuffled);
            out
        }
    }

    // ========== Matrix Transpose ==========

    /// Transpose an 8x8 matrix represented as 8 row vectors.
    ///
    /// Uses the Highway-style 3-stage transpose:
    /// 1. `unpacklo/hi` - interleave pairs within 128-bit lanes
    /// 2. `shuffle` - reorder within lanes
    /// 3. `permute2f128` - exchange 128-bit halves
    #[inline]
    pub fn transpose_8x8(rows: &mut [Self; 8]) {
        unsafe {
            let t0 = _mm256_unpacklo_ps(rows[0].0, rows[1].0);
            let t1 = _mm256_unpackhi_ps(rows[0].0, rows[1].0);
            let t2 = _mm256_unpacklo_ps(rows[2].0, rows[3].0);
            let t3 = _mm256_unpackhi_ps(rows[2].0, rows[3].0);
            let t4 = _mm256_unpacklo_ps(rows[4].0, rows[5].0);
            let t5 = _mm256_unpackhi_ps(rows[4].0, rows[5].0);
            let t6 = _mm256_unpacklo_ps(rows[6].0, rows[7].0);
            let t7 = _mm256_unpackhi_ps(rows[6].0, rows[7].0);

            let s0 = _mm256_shuffle_ps::<0x44>(t0, t2);
            let s1 = _mm256_shuffle_ps::<0xEE>(t0, t2);
            let s2 = _mm256_shuffle_ps::<0x44>(t1, t3);
            let s3 = _mm256_shuffle_ps::<0xEE>(t1, t3);
            let s4 = _mm256_shuffle_ps::<0x44>(t4, t6);
            let s5 = _mm256_shuffle_ps::<0xEE>(t4, t6);
            let s6 = _mm256_shuffle_ps::<0x44>(t5, t7);
            let s7 = _mm256_shuffle_ps::<0xEE>(t5, t7);

            rows[0] = Self(_mm256_permute2f128_ps::<0x20>(s0, s4));
            rows[1] = Self(_mm256_permute2f128_ps::<0x20>(s1, s5));
            rows[2] = Self(_mm256_permute2f128_ps::<0x20>(s2, s6));
            rows[3] = Self(_mm256_permute2f128_ps::<0x20>(s3, s7));
            rows[4] = Self(_mm256_permute2f128_ps::<0x31>(s0, s4));
            rows[5] = Self(_mm256_permute2f128_ps::<0x31>(s1, s5));
            rows[6] = Self(_mm256_permute2f128_ps::<0x31>(s2, s6));
            rows[7] = Self(_mm256_permute2f128_ps::<0x31>(s3, s7));
        }
    }

    /// Transpose an 8x8 matrix, returning the transposed rows.
    #[inline]
    pub fn transpose_8x8_copy(rows: [Self; 8]) -> [Self; 8] {
        let mut result = rows;
        Self::transpose_8x8(&mut result);
        result
    }

    /// Load an 8x8 f32 block from a contiguous array.
    #[inline]
    pub fn load_8x8(block: &[f32; 64]) -> [Self; 8] {
        unsafe {
            [
                Self(_mm256_loadu_ps(block.as_ptr())),
                Self(_mm256_loadu_ps(block.as_ptr().add(8))),
                Self(_mm256_loadu_ps(block.as_ptr().add(16))),
                Self(_mm256_loadu_ps(block.as_ptr().add(24))),
                Self(_mm256_loadu_ps(block.as_ptr().add(32))),
                Self(_mm256_loadu_ps(block.as_ptr().add(40))),
                Self(_mm256_loadu_ps(block.as_ptr().add(48))),
                Self(_mm256_loadu_ps(block.as_ptr().add(56))),
            ]
        }
    }

    /// Store 8 row vectors to a contiguous 8x8 f32 block.
    #[inline]
    pub fn store_8x8(rows: &[Self; 8], block: &mut [f32; 64]) {
        unsafe {
            _mm256_storeu_ps(block.as_mut_ptr(), rows[0].0);
            _mm256_storeu_ps(block.as_mut_ptr().add(8), rows[1].0);
            _mm256_storeu_ps(block.as_mut_ptr().add(16), rows[2].0);
            _mm256_storeu_ps(block.as_mut_ptr().add(24), rows[3].0);
            _mm256_storeu_ps(block.as_mut_ptr().add(32), rows[4].0);
            _mm256_storeu_ps(block.as_mut_ptr().add(40), rows[5].0);
            _mm256_storeu_ps(block.as_mut_ptr().add(48), rows[6].0);
            _mm256_storeu_ps(block.as_mut_ptr().add(56), rows[7].0);
        }
    }

    // ========== Bitcast ==========
    /// Reinterpret bits as `i32x8` (zero-cost).
    #[inline(always)]
    pub fn bitcast_i32x8(self) -> i32x8 {
        unsafe { core::mem::transmute(_mm256_castps_si256(self.0)) }
    }

    /// Reinterpret bits as `&i32x8` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_ref_i32x8(&self) -> &i32x8 {
        unsafe { &*(self as *const Self as *const i32x8) }
    }

    /// Reinterpret bits as `&mut i32x8` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_mut_i32x8(&mut self) -> &mut i32x8 {
        unsafe { &mut *(self as *mut Self as *mut i32x8) }
    }
    /// Reinterpret bits as `u32x8` (zero-cost).
    #[inline(always)]
    pub fn bitcast_u32x8(self) -> u32x8 {
        unsafe { core::mem::transmute(_mm256_castps_si256(self.0)) }
    }

    /// Reinterpret bits as `&u32x8` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_ref_u32x8(&self) -> &u32x8 {
        unsafe { &*(self as *const Self as *const u32x8) }
    }

    /// Reinterpret bits as `&mut u32x8` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_mut_u32x8(&mut self) -> &mut u32x8 {
        unsafe { &mut *(self as *mut Self as *mut u32x8) }
    }
}

#[cfg(target_arch = "x86_64")]
crate::impl_arithmetic_ops!(
    f32x8,
    _mm256_add_ps,
    _mm256_sub_ps,
    _mm256_mul_ps,
    _mm256_div_ps
);
#[cfg(target_arch = "x86_64")]
crate::impl_float_assign_ops!(f32x8);
#[cfg(target_arch = "x86_64")]
crate::impl_neg!(f32x8, _mm256_sub_ps, _mm256_setzero_ps);
#[cfg(target_arch = "x86_64")]
crate::impl_bitwise_ops!(f32x8, __m256, _mm256_and_ps, _mm256_or_ps, _mm256_xor_ps);
#[cfg(target_arch = "x86_64")]
crate::impl_index!(f32x8, f32, 8);

#[cfg(target_arch = "x86_64")]
impl From<[f32; 8]> for f32x8 {
    #[inline(always)]
    fn from(arr: [f32; 8]) -> Self {
        // SAFETY: [f32; 8] and __m256 have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}
#[cfg(target_arch = "x86_64")]
impl From<f32x8> for [f32; 8] {
    #[inline(always)]
    fn from(v: f32x8) -> Self {
        // SAFETY: __m256 and [f32; 8] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}

// Scalar broadcast operations for f32x8
// These allow `v + 2.0` instead of `v + f32x8::splat(token, 2.0)`

#[cfg(target_arch = "x86_64")]
impl Add<f32> for f32x8 {
    type Output = Self;
    /// Add a scalar to all lanes: `v + 2.0`
    ///
    /// Broadcasts the scalar to all lanes, then adds.
    #[inline(always)]
    fn add(self, rhs: f32) -> Self {
        self + Self(unsafe { _mm256_set1_ps(rhs) })
    }
}

#[cfg(target_arch = "x86_64")]
impl Sub<f32> for f32x8 {
    type Output = Self;
    /// Subtract a scalar from all lanes: `v - 2.0`
    #[inline(always)]
    fn sub(self, rhs: f32) -> Self {
        self - Self(unsafe { _mm256_set1_ps(rhs) })
    }
}

#[cfg(target_arch = "x86_64")]
impl Mul<f32> for f32x8 {
    type Output = Self;
    /// Multiply all lanes by a scalar: `v * 2.0`
    #[inline(always)]
    fn mul(self, rhs: f32) -> Self {
        self * Self(unsafe { _mm256_set1_ps(rhs) })
    }
}

#[cfg(target_arch = "x86_64")]
impl Div<f32> for f32x8 {
    type Output = Self;
    /// Divide all lanes by a scalar: `v / 2.0`
    #[inline(always)]
    fn div(self, rhs: f32) -> Self {
        self / Self(unsafe { _mm256_set1_ps(rhs) })
    }
}

// ============================================================================
// f64x4 - 4 x f64 (256-bit)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct f64x4(__m256d);

#[cfg(target_arch = "x86_64")]
impl f64x4 {
    pub const LANES: usize = 4;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::X64V3Token, data: &[f64; 4]) -> Self {
        Self(unsafe { _mm256_loadu_pd(data.as_ptr()) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::X64V3Token, v: f64) -> Self {
        Self(unsafe { _mm256_set1_pd(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::X64V3Token) -> Self {
        Self(unsafe { _mm256_setzero_pd() })
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::X64V3Token, arr: [f64; 4]) -> Self {
        // SAFETY: [f64; 4] and __m256d have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [f64; 4]) {
        unsafe { _mm256_storeu_pd(out.as_mut_ptr(), self.0) };
    }

    /// Convert to array
    #[inline(always)]
    pub fn to_array(self) -> [f64; 4] {
        let mut out = [0.0f64; 4];
        self.store(&mut out);
        out
    }

    /// Get reference to underlying array
    #[inline(always)]
    pub fn as_array(&self) -> &[f64; 4] {
        unsafe { &*(self as *const Self as *const [f64; 4]) }
    }

    /// Get mutable reference to underlying array
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut [f64; 4] {
        unsafe { &mut *(self as *mut Self as *mut [f64; 4]) }
    }

    /// Get raw intrinsic type
    #[inline(always)]
    pub fn raw(self) -> __m256d {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports the required SIMD features.
    /// Use token-gated constructors (`load`, `splat`, `zero`) for safe construction.
    #[inline(always)]
    pub unsafe fn from_raw(v: __m256d) -> Self {
        Self(v)
    }

    // ========== Token-gated bytemuck replacements ==========

    /// Reinterpret a slice of scalars as a slice of SIMD vectors (token-gated).
    ///
    /// Returns `None` if the slice length is not a multiple of 4, or
    /// if the slice is not properly aligned.
    ///
    /// This is a safe, token-gated replacement for `bytemuck::cast_slice`.
    #[inline(always)]
    pub fn cast_slice(_: archmage::X64V3Token, slice: &[f64]) -> Option<&[Self]> {
        if slice.len() % 4 != 0 {
            return None;
        }
        let ptr = slice.as_ptr();
        if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {
            return None;
        }
        let len = slice.len() / 4;
        // SAFETY: alignment and length checked, layout is compatible
        Some(unsafe { core::slice::from_raw_parts(ptr as *const Self, len) })
    }

    /// Reinterpret a mutable slice of scalars as a slice of SIMD vectors (token-gated).
    ///
    /// Returns `None` if the slice length is not a multiple of 4, or
    /// if the slice is not properly aligned.
    ///
    /// This is a safe, token-gated replacement for `bytemuck::cast_slice_mut`.
    #[inline(always)]
    pub fn cast_slice_mut(_: archmage::X64V3Token, slice: &mut [f64]) -> Option<&mut [Self]> {
        if slice.len() % 4 != 0 {
            return None;
        }
        let ptr = slice.as_mut_ptr();
        if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {
            return None;
        }
        let len = slice.len() / 4;
        // SAFETY: alignment and length checked, layout is compatible
        Some(unsafe { core::slice::from_raw_parts_mut(ptr as *mut Self, len) })
    }

    /// View this vector as a byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of`.
    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8; 32] {
        // SAFETY: Self is repr(transparent) over __m256d which is 32 bytes
        unsafe { &*(self as *const Self as *const [u8; 32]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 32] {
        // SAFETY: Self is repr(transparent) over __m256d which is 32 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 32]) }
    }

    /// Create from a byte array reference (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::X64V3Token, bytes: &[u8; 32]) -> Self {
        // SAFETY: [u8; 32] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Create from an owned byte array (token-gated, zero-cost).
    ///
    /// This is a zero-cost transmute from an owned byte array.
    #[inline(always)]
    pub fn from_bytes_owned(_: archmage::X64V3Token, bytes: [u8; 32]) -> Self {
        // SAFETY: [u8; 32] and Self have identical size
        Self(unsafe { core::mem::transmute(bytes) })
    }

    /// Element-wise minimum
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(unsafe { _mm256_min_pd(self.0, other.0) })
    }
    /// Element-wise maximum
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(unsafe { _mm256_max_pd(self.0, other.0) })
    }
    /// Clamp values between lo and hi
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }

    /// Square root
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        Self(unsafe { _mm256_sqrt_pd(self.0) })
    }
    /// Absolute value
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(unsafe {
            let mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFF_FFFF_FFFF_FFFFu64 as i64));
            _mm256_and_pd(self.0, mask)
        })
    }
    /// Round toward negative infinity
    #[inline(always)]
    pub fn floor(self) -> Self {
        Self(unsafe { _mm256_floor_pd(self.0) })
    }
    /// Round toward positive infinity
    #[inline(always)]
    pub fn ceil(self) -> Self {
        Self(unsafe { _mm256_ceil_pd(self.0) })
    }
    /// Round to nearest integer
    #[inline(always)]
    pub fn round(self) -> Self {
        Self(unsafe {
            _mm256_round_pd::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(self.0)
        })
    }
    /// Fused multiply-add: self * a + b
    #[inline(always)]
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        Self(unsafe { _mm256_fmadd_pd(self.0, a.0, b.0) })
    }

    /// Fused multiply-sub: self * a - b
    #[inline(always)]
    pub fn mul_sub(self, a: Self, b: Self) -> Self {
        Self(unsafe { _mm256_fmsub_pd(self.0, a.0, b.0) })
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
        Self(unsafe { _mm256_cmp_pd::<_CMP_EQ_OQ>(self.0, other.0) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if inequality, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(unsafe { _mm256_cmp_pd::<_CMP_NEQ_OQ>(self.0, other.0) })
    }

    /// Lane-wise less-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(unsafe { _mm256_cmp_pd::<_CMP_LT_OQ>(self.0, other.0) })
    }

    /// Lane-wise less-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than-or-equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(unsafe { _mm256_cmp_pd::<_CMP_LE_OQ>(self.0, other.0) })
    }

    /// Lane-wise greater-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than, all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe { _mm256_cmp_pd::<_CMP_GT_OQ>(self.0, other.0) })
    }

    /// Lane-wise greater-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than-or-equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(unsafe { _mm256_cmp_pd::<_CMP_GE_OQ>(self.0, other.0) })
    }

    // ========== Blending/Selection ==========

    /// Select lanes from `if_true` where mask is all-1s, `if_false` where mask is all-0s.
    ///
    /// The mask should come from a comparison operation like `simd_lt()`.
    ///
    /// # Example
    /// ```ignore
    /// let a = f64x4::splat(token, 1.0);
    /// let b = f64x4::splat(token, 2.0);
    /// let mask = a.simd_lt(b);  // all true
    /// let result = f64x4::blend(mask, a, b);  // selects a
    /// ```
    #[inline(always)]
    pub fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self(unsafe { _mm256_blendv_pd(if_false.0, if_true.0, mask.0) })
    }

    // ========== Horizontal Operations ==========

    /// Sum all lanes horizontally.
    ///
    /// Returns a scalar containing the sum of all 4 lanes.
    #[inline(always)]
    pub fn reduce_add(self) -> f64 {
        unsafe {
            let hi = _mm256_extractf128_pd::<1>(self.0);
            let lo = _mm256_castpd256_pd128(self.0);
            let sum = _mm_add_pd(lo, hi);
            let h = _mm_hadd_pd(sum, sum);
            _mm_cvtsd_f64(h)
        }
    }

    /// Find the minimum value across all lanes.
    #[inline(always)]
    pub fn reduce_min(self) -> f64 {
        unsafe {
            let hi = _mm256_extractf128_pd::<1>(self.0);
            let lo = _mm256_castpd256_pd128(self.0);
            let m = _mm_min_pd(lo, hi);
            let shuf = _mm_shuffle_pd::<0b01>(m, m);
            let m2 = _mm_min_pd(m, shuf);
            _mm_cvtsd_f64(m2)
        }
    }

    /// Find the maximum value across all lanes.
    #[inline(always)]
    pub fn reduce_max(self) -> f64 {
        unsafe {
            let hi = _mm256_extractf128_pd::<1>(self.0);
            let lo = _mm256_castpd256_pd128(self.0);
            let m = _mm_max_pd(lo, hi);
            let shuf = _mm_shuffle_pd::<0b01>(m, m);
            let m2 = _mm_max_pd(m, shuf);
            _mm_cvtsd_f64(m2)
        }
    }

    // ========== Bitwise Unary Operations ==========

    /// Bitwise NOT (complement): flips all bits.
    #[inline(always)]
    pub fn not(self) -> Self {
        Self(unsafe {
            let ones = _mm256_set1_epi64x(-1);
            let as_int = _mm256_castpd_si256(self.0);
            _mm256_castsi256_pd(_mm256_xor_si256(as_int, ones))
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
            let x_bits = _mm256_castpd_si256(self.0);
            let offset = _mm256_set1_epi64x(OFFSET);
            let exp_bits = _mm256_sub_epi64(x_bits, offset);
            let exp_arr_raw: [i64; 4] = core::mem::transmute(exp_bits);
            let exp_shifted = _mm256_set_epi64x(
                exp_arr_raw[3] >> 52,
                exp_arr_raw[2] >> 52,
                exp_arr_raw[1] >> 52,
                exp_arr_raw[0] >> 52,
            );

            let mantissa_bits = _mm256_sub_epi64(x_bits, _mm256_slli_epi64::<52>(exp_shifted));
            let mantissa = _mm256_castsi256_pd(mantissa_bits);
            // Convert exponent to f64
            let exp_arr: [i64; 4] = core::mem::transmute(exp_shifted);
            let exp_f64: [f64; 4] = [
                exp_arr[0] as f64,
                exp_arr[1] as f64,
                exp_arr[2] as f64,
                exp_arr[3] as f64,
            ];
            let exp_val = _mm256_loadu_pd(exp_f64.as_ptr());

            let one = _mm256_set1_pd(1.0);
            let m = _mm256_sub_pd(mantissa, one);

            // Horner's for numerator
            let yp = _mm256_fmadd_pd(_mm256_set1_pd(P2), m, _mm256_set1_pd(P1));
            let yp = _mm256_fmadd_pd(yp, m, _mm256_set1_pd(P0));

            // Horner's for denominator
            let yq = _mm256_fmadd_pd(_mm256_set1_pd(Q2), m, _mm256_set1_pd(Q1));
            let yq = _mm256_fmadd_pd(yq, m, _mm256_set1_pd(Q0));

            Self(_mm256_add_pd(_mm256_div_pd(yp, yq), exp_val))
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
            let x = _mm256_max_pd(self.0, _mm256_set1_pd(-1022.0));
            let x = _mm256_min_pd(x, _mm256_set1_pd(1022.0));

            let xi = _mm256_floor_pd(x);
            let xf = _mm256_sub_pd(x, xi);

            // Polynomial for 2^frac
            let poly = _mm256_fmadd_pd(_mm256_set1_pd(C4), xf, _mm256_set1_pd(C3));
            let poly = _mm256_fmadd_pd(poly, xf, _mm256_set1_pd(C2));
            let poly = _mm256_fmadd_pd(poly, xf, _mm256_set1_pd(C1));
            let poly = _mm256_fmadd_pd(poly, xf, _mm256_set1_pd(C0));

            // Scale by 2^integer - extract, convert, scale
            let xi_arr: [f64; 4] = core::mem::transmute(xi);
            let scale_arr: [f64; 4] = [
                f64::from_bits(((xi_arr[0] as i64 + 1023) << 52) as u64),
                f64::from_bits(((xi_arr[1] as i64 + 1023) << 52) as u64),
                f64::from_bits(((xi_arr[2] as i64 + 1023) << 52) as u64),
                f64::from_bits(((xi_arr[3] as i64 + 1023) << 52) as u64),
            ];
            let scale = _mm256_loadu_pd(scale_arr.as_ptr());

            Self(_mm256_mul_pd(poly, scale))
        }
    }

    /// Low-precision natural logarithm.
    #[inline(always)]
    pub fn ln_lowp(self) -> Self {
        const LN2: f64 = core::f64::consts::LN_2;
        unsafe { Self(_mm256_mul_pd(self.log2_lowp().0, _mm256_set1_pd(LN2))) }
    }

    /// Low-precision natural exponential (e^x).
    #[inline(always)]
    pub fn exp_lowp(self) -> Self {
        const LOG2_E: f64 = core::f64::consts::LOG2_E;
        unsafe { Self(_mm256_mul_pd(self.0, _mm256_set1_pd(LOG2_E))).exp2_lowp() }
    }

    /// Low-precision base-10 logarithm.
    #[inline(always)]
    pub fn log10_lowp(self) -> Self {
        const LOG10_2: f64 = core::f64::consts::LOG10_2;
        unsafe { Self(_mm256_mul_pd(self.log2_lowp().0, _mm256_set1_pd(LOG10_2))) }
    }

    /// Low-precision power function (self^n).
    #[inline(always)]
    pub fn pow_lowp(self, n: f64) -> Self {
        unsafe { Self(_mm256_mul_pd(self.log2_lowp().0, _mm256_set1_pd(n))).exp2_lowp() }
    }

    // ========== Bitcast ==========
    /// Reinterpret bits as `i64x4` (zero-cost).
    #[inline(always)]
    pub fn bitcast_i64x4(self) -> i64x4 {
        unsafe { core::mem::transmute(_mm256_castpd_si256(self.0)) }
    }

    /// Reinterpret bits as `&i64x4` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_ref_i64x4(&self) -> &i64x4 {
        unsafe { &*(self as *const Self as *const i64x4) }
    }

    /// Reinterpret bits as `&mut i64x4` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_mut_i64x4(&mut self) -> &mut i64x4 {
        unsafe { &mut *(self as *mut Self as *mut i64x4) }
    }
    /// Reinterpret bits as `u64x4` (zero-cost).
    #[inline(always)]
    pub fn bitcast_u64x4(self) -> u64x4 {
        unsafe { core::mem::transmute(_mm256_castpd_si256(self.0)) }
    }

    /// Reinterpret bits as `&u64x4` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_ref_u64x4(&self) -> &u64x4 {
        unsafe { &*(self as *const Self as *const u64x4) }
    }

    /// Reinterpret bits as `&mut u64x4` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_mut_u64x4(&mut self) -> &mut u64x4 {
        unsafe { &mut *(self as *mut Self as *mut u64x4) }
    }
}

#[cfg(target_arch = "x86_64")]
crate::impl_arithmetic_ops!(
    f64x4,
    _mm256_add_pd,
    _mm256_sub_pd,
    _mm256_mul_pd,
    _mm256_div_pd
);
#[cfg(target_arch = "x86_64")]
crate::impl_float_assign_ops!(f64x4);
#[cfg(target_arch = "x86_64")]
crate::impl_neg!(f64x4, _mm256_sub_pd, _mm256_setzero_pd);
#[cfg(target_arch = "x86_64")]
crate::impl_bitwise_ops!(f64x4, __m256d, _mm256_and_pd, _mm256_or_pd, _mm256_xor_pd);
#[cfg(target_arch = "x86_64")]
crate::impl_index!(f64x4, f64, 4);

#[cfg(target_arch = "x86_64")]
impl From<[f64; 4]> for f64x4 {
    #[inline(always)]
    fn from(arr: [f64; 4]) -> Self {
        // SAFETY: [f64; 4] and __m256d have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}
#[cfg(target_arch = "x86_64")]
impl From<f64x4> for [f64; 4] {
    #[inline(always)]
    fn from(v: f64x4) -> Self {
        // SAFETY: __m256d and [f64; 4] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}

// Scalar broadcast operations for f64x4
// These allow `v + 2.0` instead of `v + f64x4::splat(token, 2.0)`

#[cfg(target_arch = "x86_64")]
impl Add<f64> for f64x4 {
    type Output = Self;
    /// Add a scalar to all lanes: `v + 2.0`
    ///
    /// Broadcasts the scalar to all lanes, then adds.
    #[inline(always)]
    fn add(self, rhs: f64) -> Self {
        self + Self(unsafe { _mm256_set1_pd(rhs) })
    }
}

#[cfg(target_arch = "x86_64")]
impl Sub<f64> for f64x4 {
    type Output = Self;
    /// Subtract a scalar from all lanes: `v - 2.0`
    #[inline(always)]
    fn sub(self, rhs: f64) -> Self {
        self - Self(unsafe { _mm256_set1_pd(rhs) })
    }
}

#[cfg(target_arch = "x86_64")]
impl Mul<f64> for f64x4 {
    type Output = Self;
    /// Multiply all lanes by a scalar: `v * 2.0`
    #[inline(always)]
    fn mul(self, rhs: f64) -> Self {
        self * Self(unsafe { _mm256_set1_pd(rhs) })
    }
}

#[cfg(target_arch = "x86_64")]
impl Div<f64> for f64x4 {
    type Output = Self;
    /// Divide all lanes by a scalar: `v / 2.0`
    #[inline(always)]
    fn div(self, rhs: f64) -> Self {
        self / Self(unsafe { _mm256_set1_pd(rhs) })
    }
}

// ============================================================================
// i8x32 - 32 x i8 (256-bit)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct i8x32(__m256i);

#[cfg(target_arch = "x86_64")]
impl i8x32 {
    pub const LANES: usize = 32;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::X64V3Token, data: &[i8; 32]) -> Self {
        Self(unsafe { _mm256_loadu_si256(data.as_ptr() as *const __m256i) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::X64V3Token, v: i8) -> Self {
        Self(unsafe { _mm256_set1_epi8(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::X64V3Token) -> Self {
        Self(unsafe { _mm256_setzero_si256() })
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::X64V3Token, arr: [i8; 32]) -> Self {
        // SAFETY: [i8; 32] and __m256i have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [i8; 32]) {
        unsafe { _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, self.0) };
    }

    /// Convert to array
    #[inline(always)]
    pub fn to_array(self) -> [i8; 32] {
        let mut out = [0i8; 32];
        self.store(&mut out);
        out
    }

    /// Get reference to underlying array
    #[inline(always)]
    pub fn as_array(&self) -> &[i8; 32] {
        unsafe { &*(self as *const Self as *const [i8; 32]) }
    }

    /// Get mutable reference to underlying array
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut [i8; 32] {
        unsafe { &mut *(self as *mut Self as *mut [i8; 32]) }
    }

    /// Get raw intrinsic type
    #[inline(always)]
    pub fn raw(self) -> __m256i {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports the required SIMD features.
    /// Use token-gated constructors (`load`, `splat`, `zero`) for safe construction.
    #[inline(always)]
    pub unsafe fn from_raw(v: __m256i) -> Self {
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
    pub fn cast_slice(_: archmage::X64V3Token, slice: &[i8]) -> Option<&[Self]> {
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
    pub fn cast_slice_mut(_: archmage::X64V3Token, slice: &mut [i8]) -> Option<&mut [Self]> {
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
    pub fn as_bytes(&self) -> &[u8; 32] {
        // SAFETY: Self is repr(transparent) over __m256i which is 32 bytes
        unsafe { &*(self as *const Self as *const [u8; 32]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 32] {
        // SAFETY: Self is repr(transparent) over __m256i which is 32 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 32]) }
    }

    /// Create from a byte array reference (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::X64V3Token, bytes: &[u8; 32]) -> Self {
        // SAFETY: [u8; 32] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Create from an owned byte array (token-gated, zero-cost).
    ///
    /// This is a zero-cost transmute from an owned byte array.
    #[inline(always)]
    pub fn from_bytes_owned(_: archmage::X64V3Token, bytes: [u8; 32]) -> Self {
        // SAFETY: [u8; 32] and Self have identical size
        Self(unsafe { core::mem::transmute(bytes) })
    }

    /// Element-wise minimum
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(unsafe { _mm256_min_epi8(self.0, other.0) })
    }
    /// Element-wise maximum
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(unsafe { _mm256_max_epi8(self.0, other.0) })
    }
    /// Clamp values between lo and hi
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }

    /// Absolute value
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(unsafe { _mm256_abs_epi8(self.0) })
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
        Self(unsafe { _mm256_cmpeq_epi8(self.0, other.0) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if inequality, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(unsafe {
            let eq = _mm256_cmpeq_epi8(self.0, other.0);
            let ones = _mm256_set1_epi8(-1);
            _mm256_xor_si256(eq, ones)
        })
    }

    /// Lane-wise greater-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than, all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe { _mm256_cmpgt_epi8(self.0, other.0) })
    }

    /// Lane-wise less-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(unsafe { _mm256_cmpgt_epi8(other.0, self.0) })
    }

    /// Lane-wise greater-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than-or-equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(unsafe {
            let lt = _mm256_cmpgt_epi8(other.0, self.0);
            let ones = _mm256_set1_epi8(-1);
            _mm256_xor_si256(lt, ones)
        })
    }

    /// Lane-wise less-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than-or-equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(unsafe {
            let gt = _mm256_cmpgt_epi8(self.0, other.0);
            let ones = _mm256_set1_epi8(-1);
            _mm256_xor_si256(gt, ones)
        })
    }

    // ========== Blending/Selection ==========

    /// Select lanes from `if_true` where mask is all-1s, `if_false` where mask is all-0s.
    ///
    /// The mask should come from a comparison operation like `simd_lt()`.
    ///
    /// # Example
    /// ```ignore
    /// let a = i8x32::splat(token, 1.0);
    /// let b = i8x32::splat(token, 2.0);
    /// let mask = a.simd_lt(b);  // all true
    /// let result = i8x32::blend(mask, a, b);  // selects a
    /// ```
    #[inline(always)]
    pub fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self(unsafe { _mm256_blendv_epi8(if_false.0, if_true.0, mask.0) })
    }

    // ========== Boolean Reductions ==========

    /// Returns true if all lanes are non-zero (truthy).
    ///
    /// Typically used with comparison results where true lanes are all-1s.
    #[inline(always)]
    pub fn all_true(self) -> bool {
        unsafe { _mm256_movemask_epi8(self.0) == -1_i32 }
    }

    /// Returns true if any lane is non-zero (truthy).
    #[inline(always)]
    pub fn any_true(self) -> bool {
        unsafe { _mm256_movemask_epi8(self.0) != 0 }
    }

    /// Extract the high bit of each lane as a bitmask.
    ///
    /// Returns a u32 where bit N corresponds to the sign bit of lane N.
    #[inline(always)]
    pub fn bitmask(self) -> u32 {
        unsafe { _mm256_movemask_epi8(self.0) as u32 }
    }

    // ========== Horizontal Operations ==========

    /// Sum all lanes horizontally.
    ///
    /// Returns a scalar containing the sum of all 32 lanes.
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
            let ones = _mm256_set1_epi8(-1);
            _mm256_xor_si256(self.0, ones)
        })
    }
    // ========== Bitcast ==========
    /// Reinterpret bits as `u8x32` (zero-cost).
    #[inline(always)]
    pub fn bitcast_u8x32(self) -> u8x32 {
        unsafe { core::mem::transmute(self) }
    }

    /// Reinterpret bits as `&u8x32` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_ref_u8x32(&self) -> &u8x32 {
        unsafe { &*(self as *const Self as *const u8x32) }
    }

    /// Reinterpret bits as `&mut u8x32` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_mut_u8x32(&mut self) -> &mut u8x32 {
        unsafe { &mut *(self as *mut Self as *mut u8x32) }
    }
}

#[cfg(target_arch = "x86_64")]
crate::impl_int_arithmetic_ops!(i8x32, _mm256_add_epi8, _mm256_sub_epi8);
#[cfg(target_arch = "x86_64")]
crate::impl_assign_ops!(i8x32);
#[cfg(target_arch = "x86_64")]
crate::impl_bitwise_ops!(
    i8x32,
    __m256i,
    _mm256_and_si256,
    _mm256_or_si256,
    _mm256_xor_si256
);
#[cfg(target_arch = "x86_64")]
crate::impl_index!(i8x32, i8, 32);

#[cfg(target_arch = "x86_64")]
impl From<[i8; 32]> for i8x32 {
    #[inline(always)]
    fn from(arr: [i8; 32]) -> Self {
        // SAFETY: [i8; 32] and __m256i have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}
#[cfg(target_arch = "x86_64")]
impl From<i8x32> for [i8; 32] {
    #[inline(always)]
    fn from(v: i8x32) -> Self {
        // SAFETY: __m256i and [i8; 32] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}

// Scalar broadcast operations for i8x32
// These allow `v + 2.0` instead of `v + i8x32::splat(token, 2.0)`

#[cfg(target_arch = "x86_64")]
impl Add<i8> for i8x32 {
    type Output = Self;
    /// Add a scalar to all lanes.
    #[inline(always)]
    fn add(self, rhs: i8) -> Self {
        self + Self(unsafe { _mm256_set1_epi8(rhs) })
    }
}

#[cfg(target_arch = "x86_64")]
impl Sub<i8> for i8x32 {
    type Output = Self;
    /// Subtract a scalar from all lanes.
    #[inline(always)]
    fn sub(self, rhs: i8) -> Self {
        self - Self(unsafe { _mm256_set1_epi8(rhs) })
    }
}

// ============================================================================
// u8x32 - 32 x u8 (256-bit)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct u8x32(__m256i);

#[cfg(target_arch = "x86_64")]
impl u8x32 {
    pub const LANES: usize = 32;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::X64V3Token, data: &[u8; 32]) -> Self {
        Self(unsafe { _mm256_loadu_si256(data.as_ptr() as *const __m256i) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::X64V3Token, v: u8) -> Self {
        Self(unsafe { _mm256_set1_epi8(v as i8) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::X64V3Token) -> Self {
        Self(unsafe { _mm256_setzero_si256() })
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::X64V3Token, arr: [u8; 32]) -> Self {
        // SAFETY: [u8; 32] and __m256i have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [u8; 32]) {
        unsafe { _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, self.0) };
    }

    /// Convert to array
    #[inline(always)]
    pub fn to_array(self) -> [u8; 32] {
        let mut out = [0u8; 32];
        self.store(&mut out);
        out
    }

    /// Get reference to underlying array
    #[inline(always)]
    pub fn as_array(&self) -> &[u8; 32] {
        unsafe { &*(self as *const Self as *const [u8; 32]) }
    }

    /// Get mutable reference to underlying array
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut [u8; 32] {
        unsafe { &mut *(self as *mut Self as *mut [u8; 32]) }
    }

    /// Get raw intrinsic type
    #[inline(always)]
    pub fn raw(self) -> __m256i {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports the required SIMD features.
    /// Use token-gated constructors (`load`, `splat`, `zero`) for safe construction.
    #[inline(always)]
    pub unsafe fn from_raw(v: __m256i) -> Self {
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
    pub fn cast_slice(_: archmage::X64V3Token, slice: &[u8]) -> Option<&[Self]> {
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
    pub fn cast_slice_mut(_: archmage::X64V3Token, slice: &mut [u8]) -> Option<&mut [Self]> {
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
    pub fn as_bytes(&self) -> &[u8; 32] {
        // SAFETY: Self is repr(transparent) over __m256i which is 32 bytes
        unsafe { &*(self as *const Self as *const [u8; 32]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 32] {
        // SAFETY: Self is repr(transparent) over __m256i which is 32 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 32]) }
    }

    /// Create from a byte array reference (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::X64V3Token, bytes: &[u8; 32]) -> Self {
        // SAFETY: [u8; 32] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Create from an owned byte array (token-gated, zero-cost).
    ///
    /// This is a zero-cost transmute from an owned byte array.
    #[inline(always)]
    pub fn from_bytes_owned(_: archmage::X64V3Token, bytes: [u8; 32]) -> Self {
        // SAFETY: [u8; 32] and Self have identical size
        Self(unsafe { core::mem::transmute(bytes) })
    }

    /// Element-wise minimum
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(unsafe { _mm256_min_epu8(self.0, other.0) })
    }
    /// Element-wise maximum
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(unsafe { _mm256_max_epu8(self.0, other.0) })
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
        Self(unsafe { _mm256_cmpeq_epi8(self.0, other.0) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if inequality, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(unsafe {
            let eq = _mm256_cmpeq_epi8(self.0, other.0);
            let ones = _mm256_set1_epi8(-1);
            _mm256_xor_si256(eq, ones)
        })
    }

    /// Lane-wise greater-than (unsigned) comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than (unsigned), all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe {
            // Flip sign bit to convert unsigned to signed comparison
            let bias = _mm256_set1_epi8(0x80u8 as i8);
            let a = _mm256_xor_si256(self.0, bias);
            let b = _mm256_xor_si256(other.0, bias);
            _mm256_cmpgt_epi8(a, b)
        })
    }

    /// Lane-wise less-than (unsigned) comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than (unsigned), all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        other.simd_gt(self)
    }

    /// Lane-wise greater-than-or-equal (unsigned) comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than-or-equal (unsigned), all-0s otherwise.
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(unsafe {
            let lt = other.simd_gt(self);
            let ones = _mm256_set1_epi8(-1);
            _mm256_xor_si256(lt.0, ones)
        })
    }

    /// Lane-wise less-than-or-equal (unsigned) comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than-or-equal (unsigned), all-0s otherwise.
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(unsafe {
            let gt = self.simd_gt(other);
            let ones = _mm256_set1_epi8(-1);
            _mm256_xor_si256(gt.0, ones)
        })
    }

    // ========== Blending/Selection ==========

    /// Select lanes from `if_true` where mask is all-1s, `if_false` where mask is all-0s.
    ///
    /// The mask should come from a comparison operation like `simd_lt()`.
    ///
    /// # Example
    /// ```ignore
    /// let a = u8x32::splat(token, 1.0);
    /// let b = u8x32::splat(token, 2.0);
    /// let mask = a.simd_lt(b);  // all true
    /// let result = u8x32::blend(mask, a, b);  // selects a
    /// ```
    #[inline(always)]
    pub fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self(unsafe { _mm256_blendv_epi8(if_false.0, if_true.0, mask.0) })
    }

    // ========== Boolean Reductions ==========

    /// Returns true if all lanes are non-zero (truthy).
    ///
    /// Typically used with comparison results where true lanes are all-1s.
    #[inline(always)]
    pub fn all_true(self) -> bool {
        unsafe { _mm256_movemask_epi8(self.0) == -1_i32 }
    }

    /// Returns true if any lane is non-zero (truthy).
    #[inline(always)]
    pub fn any_true(self) -> bool {
        unsafe { _mm256_movemask_epi8(self.0) != 0 }
    }

    /// Extract the high bit of each lane as a bitmask.
    ///
    /// Returns a u32 where bit N corresponds to the sign bit of lane N.
    #[inline(always)]
    pub fn bitmask(self) -> u32 {
        unsafe { _mm256_movemask_epi8(self.0) as u32 }
    }

    // ========== Horizontal Operations ==========

    /// Sum all lanes horizontally.
    ///
    /// Returns a scalar containing the sum of all 32 lanes.
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
            let ones = _mm256_set1_epi8(-1);
            _mm256_xor_si256(self.0, ones)
        })
    }
    // ========== Extend/Widen Operations ==========

    /// Zero-extend low 16 u8 values to i16x16.
    ///
    /// Takes the lower 16 bytes and zero-extends each to 16 bits.
    #[inline(always)]
    pub fn extend_lo_i16(self) -> i16x16 {
        i16x16(unsafe {
            let lo128 = _mm256_castsi256_si128(self.0);
            _mm256_cvtepu8_epi16(lo128)
        })
    }

    /// Zero-extend high 16 u8 values to i16x16.
    ///
    /// Takes the upper 16 bytes and zero-extends each to 16 bits.
    #[inline(always)]
    pub fn extend_hi_i16(self) -> i16x16 {
        i16x16(unsafe {
            let hi128 = _mm256_extracti128_si256::<1>(self.0);
            _mm256_cvtepu8_epi16(hi128)
        })
    }

    /// Zero-extend all 32 u8 values to two i16x16 vectors.
    ///
    /// Returns (low 16 as i16x16, high 16 as i16x16).
    #[inline(always)]
    pub fn extend_i16(self) -> (i16x16, i16x16) {
        (self.extend_lo_i16(), self.extend_hi_i16())
    }

    /// Zero-extend low 8 u8 values to i32x8.
    #[inline(always)]
    pub fn extend_lo_i32(self) -> i32x8 {
        i32x8(unsafe {
            let lo128 = _mm256_castsi256_si128(self.0);
            _mm256_cvtepu8_epi32(lo128)
        })
    }

    /// Zero-extend low 8 u8 values to f32x8.
    ///
    /// Useful for image processing: load 8 pixel values and convert to float.
    #[inline(always)]
    pub fn extend_lo_f32(self) -> f32x8 {
        f32x8(unsafe {
            let lo128 = _mm256_castsi256_si128(self.0);
            let i32s = _mm256_cvtepu8_epi32(lo128);
            _mm256_cvtepi32_ps(i32s)
        })
    }

    /// Zero-extend all 32 u8 values to four f32x8 vectors.
    ///
    /// Returns [bytes 0-7, bytes 8-15, bytes 16-23, bytes 24-31] as f32x8.
    /// Useful for processing 32 pixels as floats.
    #[inline(always)]
    pub fn extend_f32(self) -> [f32x8; 4] {
        unsafe {
            let lo128 = _mm256_castsi256_si128(self.0);
            let hi128 = _mm256_extracti128_si256::<1>(self.0);

            // bytes 0-7
            let i0 = _mm256_cvtepu8_epi32(lo128);
            let f0 = _mm256_cvtepi32_ps(i0);

            // bytes 8-15: shift lo128 right by 8 bytes
            let lo_hi = _mm_srli_si128::<8>(lo128);
            let i1 = _mm256_cvtepu8_epi32(lo_hi);
            let f1 = _mm256_cvtepi32_ps(i1);

            // bytes 16-23
            let i2 = _mm256_cvtepu8_epi32(hi128);
            let f2 = _mm256_cvtepi32_ps(i2);

            // bytes 24-31: shift hi128 right by 8 bytes
            let hi_hi = _mm_srli_si128::<8>(hi128);
            let i3 = _mm256_cvtepu8_epi32(hi_hi);
            let f3 = _mm256_cvtepi32_ps(i3);

            [f32x8(f0), f32x8(f1), f32x8(f2), f32x8(f3)]
        }
    }

    // ========== Bitcast ==========
    /// Reinterpret bits as `i8x32` (zero-cost).
    #[inline(always)]
    pub fn bitcast_i8x32(self) -> i8x32 {
        unsafe { core::mem::transmute(self) }
    }

    /// Reinterpret bits as `&i8x32` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_ref_i8x32(&self) -> &i8x32 {
        unsafe { &*(self as *const Self as *const i8x32) }
    }

    /// Reinterpret bits as `&mut i8x32` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_mut_i8x32(&mut self) -> &mut i8x32 {
        unsafe { &mut *(self as *mut Self as *mut i8x32) }
    }
}

#[cfg(target_arch = "x86_64")]
crate::impl_int_arithmetic_ops!(u8x32, _mm256_add_epi8, _mm256_sub_epi8);
#[cfg(target_arch = "x86_64")]
crate::impl_assign_ops!(u8x32);
#[cfg(target_arch = "x86_64")]
crate::impl_bitwise_ops!(
    u8x32,
    __m256i,
    _mm256_and_si256,
    _mm256_or_si256,
    _mm256_xor_si256
);
#[cfg(target_arch = "x86_64")]
crate::impl_index!(u8x32, u8, 32);

#[cfg(target_arch = "x86_64")]
impl From<[u8; 32]> for u8x32 {
    #[inline(always)]
    fn from(arr: [u8; 32]) -> Self {
        // SAFETY: [u8; 32] and __m256i have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}
#[cfg(target_arch = "x86_64")]
impl From<u8x32> for [u8; 32] {
    #[inline(always)]
    fn from(v: u8x32) -> Self {
        // SAFETY: __m256i and [u8; 32] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}

// Scalar broadcast operations for u8x32
// These allow `v + 2.0` instead of `v + u8x32::splat(token, 2.0)`

#[cfg(target_arch = "x86_64")]
impl Add<u8> for u8x32 {
    type Output = Self;
    /// Add a scalar to all lanes.
    #[inline(always)]
    fn add(self, rhs: u8) -> Self {
        self + Self(unsafe { _mm256_set1_epi8(rhs as i8) })
    }
}

#[cfg(target_arch = "x86_64")]
impl Sub<u8> for u8x32 {
    type Output = Self;
    /// Subtract a scalar from all lanes.
    #[inline(always)]
    fn sub(self, rhs: u8) -> Self {
        self - Self(unsafe { _mm256_set1_epi8(rhs as i8) })
    }
}

// ============================================================================
// i16x16 - 16 x i16 (256-bit)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct i16x16(__m256i);

#[cfg(target_arch = "x86_64")]
impl i16x16 {
    pub const LANES: usize = 16;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::X64V3Token, data: &[i16; 16]) -> Self {
        Self(unsafe { _mm256_loadu_si256(data.as_ptr() as *const __m256i) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::X64V3Token, v: i16) -> Self {
        Self(unsafe { _mm256_set1_epi16(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::X64V3Token) -> Self {
        Self(unsafe { _mm256_setzero_si256() })
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::X64V3Token, arr: [i16; 16]) -> Self {
        // SAFETY: [i16; 16] and __m256i have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [i16; 16]) {
        unsafe { _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, self.0) };
    }

    /// Convert to array
    #[inline(always)]
    pub fn to_array(self) -> [i16; 16] {
        let mut out = [0i16; 16];
        self.store(&mut out);
        out
    }

    /// Get reference to underlying array
    #[inline(always)]
    pub fn as_array(&self) -> &[i16; 16] {
        unsafe { &*(self as *const Self as *const [i16; 16]) }
    }

    /// Get mutable reference to underlying array
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut [i16; 16] {
        unsafe { &mut *(self as *mut Self as *mut [i16; 16]) }
    }

    /// Get raw intrinsic type
    #[inline(always)]
    pub fn raw(self) -> __m256i {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports the required SIMD features.
    /// Use token-gated constructors (`load`, `splat`, `zero`) for safe construction.
    #[inline(always)]
    pub unsafe fn from_raw(v: __m256i) -> Self {
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
    pub fn cast_slice(_: archmage::X64V3Token, slice: &[i16]) -> Option<&[Self]> {
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
    pub fn cast_slice_mut(_: archmage::X64V3Token, slice: &mut [i16]) -> Option<&mut [Self]> {
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
    pub fn as_bytes(&self) -> &[u8; 32] {
        // SAFETY: Self is repr(transparent) over __m256i which is 32 bytes
        unsafe { &*(self as *const Self as *const [u8; 32]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 32] {
        // SAFETY: Self is repr(transparent) over __m256i which is 32 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 32]) }
    }

    /// Create from a byte array reference (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::X64V3Token, bytes: &[u8; 32]) -> Self {
        // SAFETY: [u8; 32] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Create from an owned byte array (token-gated, zero-cost).
    ///
    /// This is a zero-cost transmute from an owned byte array.
    #[inline(always)]
    pub fn from_bytes_owned(_: archmage::X64V3Token, bytes: [u8; 32]) -> Self {
        // SAFETY: [u8; 32] and Self have identical size
        Self(unsafe { core::mem::transmute(bytes) })
    }

    /// Element-wise minimum
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(unsafe { _mm256_min_epi16(self.0, other.0) })
    }
    /// Element-wise maximum
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(unsafe { _mm256_max_epi16(self.0, other.0) })
    }
    /// Clamp values between lo and hi
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }

    /// Absolute value
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(unsafe { _mm256_abs_epi16(self.0) })
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
        Self(unsafe { _mm256_cmpeq_epi16(self.0, other.0) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if inequality, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(unsafe {
            let eq = _mm256_cmpeq_epi16(self.0, other.0);
            let ones = _mm256_set1_epi16(-1);
            _mm256_xor_si256(eq, ones)
        })
    }

    /// Lane-wise greater-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than, all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe { _mm256_cmpgt_epi16(self.0, other.0) })
    }

    /// Lane-wise less-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(unsafe { _mm256_cmpgt_epi16(other.0, self.0) })
    }

    /// Lane-wise greater-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than-or-equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(unsafe {
            let lt = _mm256_cmpgt_epi16(other.0, self.0);
            let ones = _mm256_set1_epi16(-1);
            _mm256_xor_si256(lt, ones)
        })
    }

    /// Lane-wise less-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than-or-equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(unsafe {
            let gt = _mm256_cmpgt_epi16(self.0, other.0);
            let ones = _mm256_set1_epi16(-1);
            _mm256_xor_si256(gt, ones)
        })
    }

    // ========== Blending/Selection ==========

    /// Select lanes from `if_true` where mask is all-1s, `if_false` where mask is all-0s.
    ///
    /// The mask should come from a comparison operation like `simd_lt()`.
    ///
    /// # Example
    /// ```ignore
    /// let a = i16x16::splat(token, 1.0);
    /// let b = i16x16::splat(token, 2.0);
    /// let mask = a.simd_lt(b);  // all true
    /// let result = i16x16::blend(mask, a, b);  // selects a
    /// ```
    #[inline(always)]
    pub fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self(unsafe { _mm256_blendv_epi8(if_false.0, if_true.0, mask.0) })
    }

    // ========== Boolean Reductions ==========

    /// Returns true if all lanes are non-zero (truthy).
    ///
    /// Typically used with comparison results where true lanes are all-1s.
    #[inline(always)]
    pub fn all_true(self) -> bool {
        unsafe { _mm256_movemask_epi8(self.0) == -1_i32 }
    }

    /// Returns true if any lane is non-zero (truthy).
    #[inline(always)]
    pub fn any_true(self) -> bool {
        unsafe { _mm256_movemask_epi8(self.0) != 0 }
    }

    /// Extract the high bit of each lane as a bitmask.
    ///
    /// Returns a u32 where bit N corresponds to the sign bit of lane N.
    #[inline(always)]
    pub fn bitmask(self) -> u32 {
        unsafe {
            let shifted = _mm256_srai_epi16::<15>(self.0);
            let packed = _mm256_packs_epi16(shifted, shifted);
            // packs interleaves, need to extract
            let lo = _mm256_castsi256_si128(packed);
            let hi = _mm256_extracti128_si256::<1>(packed);
            ((_mm_movemask_epi8(lo) & 0xFF) | ((_mm_movemask_epi8(hi) & 0xFF) << 8)) as u32
        }
    }

    // ========== Horizontal Operations ==========

    /// Sum all lanes horizontally.
    ///
    /// Returns a scalar containing the sum of all 16 lanes.
    /// Note: This uses a scalar loop. For performance-critical code,
    /// consider keeping values in SIMD until the final reduction.
    #[inline(always)]
    pub fn reduce_add(self) -> i16 {
        self.as_array()
            .iter()
            .copied()
            .fold(0_i16, i16::wrapping_add)
    }

    // ========== Bitwise Unary Operations ==========

    /// Bitwise NOT (complement): flips all bits.
    #[inline(always)]
    pub fn not(self) -> Self {
        Self(unsafe {
            let ones = _mm256_set1_epi16(-1);
            _mm256_xor_si256(self.0, ones)
        })
    }
    // ========== Shift Operations ==========

    /// Shift each lane left by `N` bits.
    ///
    /// Bits shifted out are lost; zeros are shifted in.
    #[inline(always)]
    pub fn shl<const N: i32>(self) -> Self {
        Self(unsafe { _mm256_slli_epi16::<N>(self.0) })
    }

    /// Shift each lane right by `N` bits (logical/unsigned shift).
    ///
    /// Bits shifted out are lost; zeros are shifted in.
    #[inline(always)]
    pub fn shr<const N: i32>(self) -> Self {
        Self(unsafe { _mm256_srli_epi16::<N>(self.0) })
    }

    /// Arithmetic shift right by `N` bits (sign-extending).
    ///
    /// The sign bit is replicated into the vacated positions.
    #[inline(always)]
    pub fn shr_arithmetic<const N: i32>(self) -> Self {
        Self(unsafe { _mm256_srai_epi16::<N>(self.0) })
    }

    // ========== Extend/Widen Operations ==========

    /// Sign-extend low 8 i16 values to i32x8.
    #[inline(always)]
    pub fn extend_lo_i32(self) -> i32x8 {
        i32x8(unsafe {
            let lo128 = _mm256_castsi256_si128(self.0);
            _mm256_cvtepi16_epi32(lo128)
        })
    }

    /// Sign-extend high 8 i16 values to i32x8.
    #[inline(always)]
    pub fn extend_hi_i32(self) -> i32x8 {
        i32x8(unsafe {
            let hi128 = _mm256_extracti128_si256::<1>(self.0);
            _mm256_cvtepi16_epi32(hi128)
        })
    }

    /// Sign-extend all 16 i16 values to two i32x8 vectors.
    #[inline(always)]
    pub fn extend_i32(self) -> (i32x8, i32x8) {
        (self.extend_lo_i32(), self.extend_hi_i32())
    }

    /// Sign-extend low 8 i16 values to f32x8.
    #[inline(always)]
    pub fn extend_lo_f32(self) -> f32x8 {
        f32x8(unsafe {
            let lo128 = _mm256_castsi256_si128(self.0);
            let i32s = _mm256_cvtepi16_epi32(lo128);
            _mm256_cvtepi32_ps(i32s)
        })
    }

    /// Sign-extend all 16 i16 values to two f32x8 vectors.
    #[inline(always)]
    pub fn extend_f32(self) -> (f32x8, f32x8) {
        unsafe {
            let lo128 = _mm256_castsi256_si128(self.0);
            let hi128 = _mm256_extracti128_si256::<1>(self.0);

            let i32_lo = _mm256_cvtepi16_epi32(lo128);
            let i32_hi = _mm256_cvtepi16_epi32(hi128);

            (
                f32x8(_mm256_cvtepi32_ps(i32_lo)),
                f32x8(_mm256_cvtepi32_ps(i32_hi)),
            )
        }
    }

    // ========== Pack/Narrow Operations ==========

    /// Pack two i16x16 vectors to u8x32 with unsigned saturation.
    ///
    /// Values below 0 become 0, values above 255 become 255.
    /// Note: AVX2 pack works within 128-bit lanes, so results are:
    /// [self_lo[0-7], other_lo[0-7], self_hi[0-7], other_hi[0-7]]
    #[inline(always)]
    pub fn pack_u8(self, other: Self) -> u8x32 {
        u8x32(unsafe { _mm256_packus_epi16(self.0, other.0) })
    }

    /// Pack two i16x16 vectors to i8x32 with signed saturation.
    ///
    /// Values are clamped to [-128, 127].
    #[inline(always)]
    pub fn pack_i8(self, other: Self) -> i8x32 {
        i8x32(unsafe { _mm256_packs_epi16(self.0, other.0) })
    }

    // ========== Bitcast ==========
    /// Reinterpret bits as `u16x16` (zero-cost).
    #[inline(always)]
    pub fn bitcast_u16x16(self) -> u16x16 {
        unsafe { core::mem::transmute(self) }
    }

    /// Reinterpret bits as `&u16x16` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_ref_u16x16(&self) -> &u16x16 {
        unsafe { &*(self as *const Self as *const u16x16) }
    }

    /// Reinterpret bits as `&mut u16x16` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_mut_u16x16(&mut self) -> &mut u16x16 {
        unsafe { &mut *(self as *mut Self as *mut u16x16) }
    }
}

#[cfg(target_arch = "x86_64")]
crate::impl_int_arithmetic_ops!(i16x16, _mm256_add_epi16, _mm256_sub_epi16);
#[cfg(target_arch = "x86_64")]
crate::impl_int_mul_op!(i16x16, _mm256_mullo_epi16);
#[cfg(target_arch = "x86_64")]
crate::impl_assign_ops!(i16x16);
#[cfg(target_arch = "x86_64")]
crate::impl_bitwise_ops!(
    i16x16,
    __m256i,
    _mm256_and_si256,
    _mm256_or_si256,
    _mm256_xor_si256
);
#[cfg(target_arch = "x86_64")]
crate::impl_index!(i16x16, i16, 16);

#[cfg(target_arch = "x86_64")]
impl From<[i16; 16]> for i16x16 {
    #[inline(always)]
    fn from(arr: [i16; 16]) -> Self {
        // SAFETY: [i16; 16] and __m256i have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}
#[cfg(target_arch = "x86_64")]
impl From<i16x16> for [i16; 16] {
    #[inline(always)]
    fn from(v: i16x16) -> Self {
        // SAFETY: __m256i and [i16; 16] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}

// Scalar broadcast operations for i16x16
// These allow `v + 2.0` instead of `v + i16x16::splat(token, 2.0)`

#[cfg(target_arch = "x86_64")]
impl Add<i16> for i16x16 {
    type Output = Self;
    /// Add a scalar to all lanes.
    #[inline(always)]
    fn add(self, rhs: i16) -> Self {
        self + Self(unsafe { _mm256_set1_epi16(rhs) })
    }
}

#[cfg(target_arch = "x86_64")]
impl Sub<i16> for i16x16 {
    type Output = Self;
    /// Subtract a scalar from all lanes.
    #[inline(always)]
    fn sub(self, rhs: i16) -> Self {
        self - Self(unsafe { _mm256_set1_epi16(rhs) })
    }
}

// ============================================================================
// u16x16 - 16 x u16 (256-bit)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct u16x16(__m256i);

#[cfg(target_arch = "x86_64")]
impl u16x16 {
    pub const LANES: usize = 16;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::X64V3Token, data: &[u16; 16]) -> Self {
        Self(unsafe { _mm256_loadu_si256(data.as_ptr() as *const __m256i) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::X64V3Token, v: u16) -> Self {
        Self(unsafe { _mm256_set1_epi16(v as i16) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::X64V3Token) -> Self {
        Self(unsafe { _mm256_setzero_si256() })
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::X64V3Token, arr: [u16; 16]) -> Self {
        // SAFETY: [u16; 16] and __m256i have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [u16; 16]) {
        unsafe { _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, self.0) };
    }

    /// Convert to array
    #[inline(always)]
    pub fn to_array(self) -> [u16; 16] {
        let mut out = [0u16; 16];
        self.store(&mut out);
        out
    }

    /// Get reference to underlying array
    #[inline(always)]
    pub fn as_array(&self) -> &[u16; 16] {
        unsafe { &*(self as *const Self as *const [u16; 16]) }
    }

    /// Get mutable reference to underlying array
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut [u16; 16] {
        unsafe { &mut *(self as *mut Self as *mut [u16; 16]) }
    }

    /// Get raw intrinsic type
    #[inline(always)]
    pub fn raw(self) -> __m256i {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports the required SIMD features.
    /// Use token-gated constructors (`load`, `splat`, `zero`) for safe construction.
    #[inline(always)]
    pub unsafe fn from_raw(v: __m256i) -> Self {
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
    pub fn cast_slice(_: archmage::X64V3Token, slice: &[u16]) -> Option<&[Self]> {
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
    pub fn cast_slice_mut(_: archmage::X64V3Token, slice: &mut [u16]) -> Option<&mut [Self]> {
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
    pub fn as_bytes(&self) -> &[u8; 32] {
        // SAFETY: Self is repr(transparent) over __m256i which is 32 bytes
        unsafe { &*(self as *const Self as *const [u8; 32]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 32] {
        // SAFETY: Self is repr(transparent) over __m256i which is 32 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 32]) }
    }

    /// Create from a byte array reference (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::X64V3Token, bytes: &[u8; 32]) -> Self {
        // SAFETY: [u8; 32] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Create from an owned byte array (token-gated, zero-cost).
    ///
    /// This is a zero-cost transmute from an owned byte array.
    #[inline(always)]
    pub fn from_bytes_owned(_: archmage::X64V3Token, bytes: [u8; 32]) -> Self {
        // SAFETY: [u8; 32] and Self have identical size
        Self(unsafe { core::mem::transmute(bytes) })
    }

    /// Element-wise minimum
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(unsafe { _mm256_min_epu16(self.0, other.0) })
    }
    /// Element-wise maximum
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(unsafe { _mm256_max_epu16(self.0, other.0) })
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
        Self(unsafe { _mm256_cmpeq_epi16(self.0, other.0) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if inequality, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(unsafe {
            let eq = _mm256_cmpeq_epi16(self.0, other.0);
            let ones = _mm256_set1_epi16(-1);
            _mm256_xor_si256(eq, ones)
        })
    }

    /// Lane-wise greater-than (unsigned) comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than (unsigned), all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe {
            // Flip sign bit to convert unsigned to signed comparison
            let bias = _mm256_set1_epi16(0x8000u16 as i16);
            let a = _mm256_xor_si256(self.0, bias);
            let b = _mm256_xor_si256(other.0, bias);
            _mm256_cmpgt_epi16(a, b)
        })
    }

    /// Lane-wise less-than (unsigned) comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than (unsigned), all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        other.simd_gt(self)
    }

    /// Lane-wise greater-than-or-equal (unsigned) comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than-or-equal (unsigned), all-0s otherwise.
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(unsafe {
            let lt = other.simd_gt(self);
            let ones = _mm256_set1_epi16(-1);
            _mm256_xor_si256(lt.0, ones)
        })
    }

    /// Lane-wise less-than-or-equal (unsigned) comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than-or-equal (unsigned), all-0s otherwise.
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(unsafe {
            let gt = self.simd_gt(other);
            let ones = _mm256_set1_epi16(-1);
            _mm256_xor_si256(gt.0, ones)
        })
    }

    // ========== Blending/Selection ==========

    /// Select lanes from `if_true` where mask is all-1s, `if_false` where mask is all-0s.
    ///
    /// The mask should come from a comparison operation like `simd_lt()`.
    ///
    /// # Example
    /// ```ignore
    /// let a = u16x16::splat(token, 1.0);
    /// let b = u16x16::splat(token, 2.0);
    /// let mask = a.simd_lt(b);  // all true
    /// let result = u16x16::blend(mask, a, b);  // selects a
    /// ```
    #[inline(always)]
    pub fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self(unsafe { _mm256_blendv_epi8(if_false.0, if_true.0, mask.0) })
    }

    // ========== Boolean Reductions ==========

    /// Returns true if all lanes are non-zero (truthy).
    ///
    /// Typically used with comparison results where true lanes are all-1s.
    #[inline(always)]
    pub fn all_true(self) -> bool {
        unsafe { _mm256_movemask_epi8(self.0) == -1_i32 }
    }

    /// Returns true if any lane is non-zero (truthy).
    #[inline(always)]
    pub fn any_true(self) -> bool {
        unsafe { _mm256_movemask_epi8(self.0) != 0 }
    }

    /// Extract the high bit of each lane as a bitmask.
    ///
    /// Returns a u32 where bit N corresponds to the sign bit of lane N.
    #[inline(always)]
    pub fn bitmask(self) -> u32 {
        unsafe {
            let shifted = _mm256_srai_epi16::<15>(self.0);
            let packed = _mm256_packs_epi16(shifted, shifted);
            // packs interleaves, need to extract
            let lo = _mm256_castsi256_si128(packed);
            let hi = _mm256_extracti128_si256::<1>(packed);
            ((_mm_movemask_epi8(lo) & 0xFF) | ((_mm_movemask_epi8(hi) & 0xFF) << 8)) as u32
        }
    }

    // ========== Horizontal Operations ==========

    /// Sum all lanes horizontally.
    ///
    /// Returns a scalar containing the sum of all 16 lanes.
    /// Note: This uses a scalar loop. For performance-critical code,
    /// consider keeping values in SIMD until the final reduction.
    #[inline(always)]
    pub fn reduce_add(self) -> u16 {
        self.as_array()
            .iter()
            .copied()
            .fold(0_u16, u16::wrapping_add)
    }

    // ========== Bitwise Unary Operations ==========

    /// Bitwise NOT (complement): flips all bits.
    #[inline(always)]
    pub fn not(self) -> Self {
        Self(unsafe {
            let ones = _mm256_set1_epi16(-1);
            _mm256_xor_si256(self.0, ones)
        })
    }
    // ========== Shift Operations ==========

    /// Shift each lane left by `N` bits.
    ///
    /// Bits shifted out are lost; zeros are shifted in.
    #[inline(always)]
    pub fn shl<const N: i32>(self) -> Self {
        Self(unsafe { _mm256_slli_epi16::<N>(self.0) })
    }

    /// Shift each lane right by `N` bits (logical/unsigned shift).
    ///
    /// Bits shifted out are lost; zeros are shifted in.
    #[inline(always)]
    pub fn shr<const N: i32>(self) -> Self {
        Self(unsafe { _mm256_srli_epi16::<N>(self.0) })
    }

    // ========== Extend/Widen Operations ==========

    /// Zero-extend low 8 u16 values to i32x8.
    #[inline(always)]
    pub fn extend_lo_i32(self) -> i32x8 {
        i32x8(unsafe {
            let lo128 = _mm256_castsi256_si128(self.0);
            _mm256_cvtepu16_epi32(lo128)
        })
    }

    /// Zero-extend high 8 u16 values to i32x8.
    #[inline(always)]
    pub fn extend_hi_i32(self) -> i32x8 {
        i32x8(unsafe {
            let hi128 = _mm256_extracti128_si256::<1>(self.0);
            _mm256_cvtepu16_epi32(hi128)
        })
    }

    /// Zero-extend all 16 u16 values to two i32x8 vectors.
    #[inline(always)]
    pub fn extend_i32(self) -> (i32x8, i32x8) {
        (self.extend_lo_i32(), self.extend_hi_i32())
    }

    /// Zero-extend low 8 u16 values to f32x8.
    #[inline(always)]
    pub fn extend_lo_f32(self) -> f32x8 {
        f32x8(unsafe {
            let lo128 = _mm256_castsi256_si128(self.0);
            let i32s = _mm256_cvtepu16_epi32(lo128);
            _mm256_cvtepi32_ps(i32s)
        })
    }

    /// Zero-extend all 16 u16 values to two f32x8 vectors.
    #[inline(always)]
    pub fn extend_f32(self) -> (f32x8, f32x8) {
        unsafe {
            let lo128 = _mm256_castsi256_si128(self.0);
            let hi128 = _mm256_extracti128_si256::<1>(self.0);

            let i32_lo = _mm256_cvtepu16_epi32(lo128);
            let i32_hi = _mm256_cvtepu16_epi32(hi128);

            (
                f32x8(_mm256_cvtepi32_ps(i32_lo)),
                f32x8(_mm256_cvtepi32_ps(i32_hi)),
            )
        }
    }

    // ========== Bitcast ==========
    /// Reinterpret bits as `i16x16` (zero-cost).
    #[inline(always)]
    pub fn bitcast_i16x16(self) -> i16x16 {
        unsafe { core::mem::transmute(self) }
    }

    /// Reinterpret bits as `&i16x16` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_ref_i16x16(&self) -> &i16x16 {
        unsafe { &*(self as *const Self as *const i16x16) }
    }

    /// Reinterpret bits as `&mut i16x16` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_mut_i16x16(&mut self) -> &mut i16x16 {
        unsafe { &mut *(self as *mut Self as *mut i16x16) }
    }
}

#[cfg(target_arch = "x86_64")]
crate::impl_int_arithmetic_ops!(u16x16, _mm256_add_epi16, _mm256_sub_epi16);
#[cfg(target_arch = "x86_64")]
crate::impl_int_mul_op!(u16x16, _mm256_mullo_epi16);
#[cfg(target_arch = "x86_64")]
crate::impl_assign_ops!(u16x16);
#[cfg(target_arch = "x86_64")]
crate::impl_bitwise_ops!(
    u16x16,
    __m256i,
    _mm256_and_si256,
    _mm256_or_si256,
    _mm256_xor_si256
);
#[cfg(target_arch = "x86_64")]
crate::impl_index!(u16x16, u16, 16);

#[cfg(target_arch = "x86_64")]
impl From<[u16; 16]> for u16x16 {
    #[inline(always)]
    fn from(arr: [u16; 16]) -> Self {
        // SAFETY: [u16; 16] and __m256i have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}
#[cfg(target_arch = "x86_64")]
impl From<u16x16> for [u16; 16] {
    #[inline(always)]
    fn from(v: u16x16) -> Self {
        // SAFETY: __m256i and [u16; 16] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}

// Scalar broadcast operations for u16x16
// These allow `v + 2.0` instead of `v + u16x16::splat(token, 2.0)`

#[cfg(target_arch = "x86_64")]
impl Add<u16> for u16x16 {
    type Output = Self;
    /// Add a scalar to all lanes.
    #[inline(always)]
    fn add(self, rhs: u16) -> Self {
        self + Self(unsafe { _mm256_set1_epi16(rhs as i16) })
    }
}

#[cfg(target_arch = "x86_64")]
impl Sub<u16> for u16x16 {
    type Output = Self;
    /// Subtract a scalar from all lanes.
    #[inline(always)]
    fn sub(self, rhs: u16) -> Self {
        self - Self(unsafe { _mm256_set1_epi16(rhs as i16) })
    }
}

// ============================================================================
// i32x8 - 8 x i32 (256-bit)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct i32x8(__m256i);

#[cfg(target_arch = "x86_64")]
impl i32x8 {
    pub const LANES: usize = 8;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::X64V3Token, data: &[i32; 8]) -> Self {
        Self(unsafe { _mm256_loadu_si256(data.as_ptr() as *const __m256i) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::X64V3Token, v: i32) -> Self {
        Self(unsafe { _mm256_set1_epi32(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::X64V3Token) -> Self {
        Self(unsafe { _mm256_setzero_si256() })
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::X64V3Token, arr: [i32; 8]) -> Self {
        // SAFETY: [i32; 8] and __m256i have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [i32; 8]) {
        unsafe { _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, self.0) };
    }

    /// Convert to array
    #[inline(always)]
    pub fn to_array(self) -> [i32; 8] {
        let mut out = [0i32; 8];
        self.store(&mut out);
        out
    }

    /// Get reference to underlying array
    #[inline(always)]
    pub fn as_array(&self) -> &[i32; 8] {
        unsafe { &*(self as *const Self as *const [i32; 8]) }
    }

    /// Get mutable reference to underlying array
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut [i32; 8] {
        unsafe { &mut *(self as *mut Self as *mut [i32; 8]) }
    }

    /// Get raw intrinsic type
    #[inline(always)]
    pub fn raw(self) -> __m256i {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports the required SIMD features.
    /// Use token-gated constructors (`load`, `splat`, `zero`) for safe construction.
    #[inline(always)]
    pub unsafe fn from_raw(v: __m256i) -> Self {
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
    pub fn cast_slice(_: archmage::X64V3Token, slice: &[i32]) -> Option<&[Self]> {
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
    pub fn cast_slice_mut(_: archmage::X64V3Token, slice: &mut [i32]) -> Option<&mut [Self]> {
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
    pub fn as_bytes(&self) -> &[u8; 32] {
        // SAFETY: Self is repr(transparent) over __m256i which is 32 bytes
        unsafe { &*(self as *const Self as *const [u8; 32]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 32] {
        // SAFETY: Self is repr(transparent) over __m256i which is 32 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 32]) }
    }

    /// Create from a byte array reference (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::X64V3Token, bytes: &[u8; 32]) -> Self {
        // SAFETY: [u8; 32] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Create from an owned byte array (token-gated, zero-cost).
    ///
    /// This is a zero-cost transmute from an owned byte array.
    #[inline(always)]
    pub fn from_bytes_owned(_: archmage::X64V3Token, bytes: [u8; 32]) -> Self {
        // SAFETY: [u8; 32] and Self have identical size
        Self(unsafe { core::mem::transmute(bytes) })
    }

    /// Element-wise minimum
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(unsafe { _mm256_min_epi32(self.0, other.0) })
    }
    /// Element-wise maximum
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(unsafe { _mm256_max_epi32(self.0, other.0) })
    }
    /// Clamp values between lo and hi
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }

    /// Absolute value
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(unsafe { _mm256_abs_epi32(self.0) })
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
        Self(unsafe { _mm256_cmpeq_epi32(self.0, other.0) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if inequality, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(unsafe {
            let eq = _mm256_cmpeq_epi32(self.0, other.0);
            let ones = _mm256_set1_epi32(-1);
            _mm256_xor_si256(eq, ones)
        })
    }

    /// Lane-wise greater-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than, all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe { _mm256_cmpgt_epi32(self.0, other.0) })
    }

    /// Lane-wise less-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(unsafe { _mm256_cmpgt_epi32(other.0, self.0) })
    }

    /// Lane-wise greater-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than-or-equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(unsafe {
            let lt = _mm256_cmpgt_epi32(other.0, self.0);
            let ones = _mm256_set1_epi32(-1);
            _mm256_xor_si256(lt, ones)
        })
    }

    /// Lane-wise less-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than-or-equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(unsafe {
            let gt = _mm256_cmpgt_epi32(self.0, other.0);
            let ones = _mm256_set1_epi32(-1);
            _mm256_xor_si256(gt, ones)
        })
    }

    // ========== Blending/Selection ==========

    /// Select lanes from `if_true` where mask is all-1s, `if_false` where mask is all-0s.
    ///
    /// The mask should come from a comparison operation like `simd_lt()`.
    ///
    /// # Example
    /// ```ignore
    /// let a = i32x8::splat(token, 1.0);
    /// let b = i32x8::splat(token, 2.0);
    /// let mask = a.simd_lt(b);  // all true
    /// let result = i32x8::blend(mask, a, b);  // selects a
    /// ```
    #[inline(always)]
    pub fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self(unsafe { _mm256_blendv_epi8(if_false.0, if_true.0, mask.0) })
    }

    // ========== Boolean Reductions ==========

    /// Returns true if all lanes are non-zero (truthy).
    ///
    /// Typically used with comparison results where true lanes are all-1s.
    #[inline(always)]
    pub fn all_true(self) -> bool {
        unsafe { _mm256_movemask_epi8(self.0) == -1_i32 }
    }

    /// Returns true if any lane is non-zero (truthy).
    #[inline(always)]
    pub fn any_true(self) -> bool {
        unsafe { _mm256_movemask_epi8(self.0) != 0 }
    }

    /// Extract the high bit of each lane as a bitmask.
    ///
    /// Returns a u32 where bit N corresponds to the sign bit of lane N.
    #[inline(always)]
    pub fn bitmask(self) -> u32 {
        unsafe { _mm256_movemask_ps(_mm256_castsi256_ps(self.0)) as u32 }
    }

    // ========== Horizontal Operations ==========

    /// Sum all lanes horizontally.
    ///
    /// Returns a scalar containing the sum of all 8 lanes.
    /// Note: This uses a scalar loop. For performance-critical code,
    /// consider keeping values in SIMD until the final reduction.
    #[inline(always)]
    pub fn reduce_add(self) -> i32 {
        self.as_array()
            .iter()
            .copied()
            .fold(0_i32, i32::wrapping_add)
    }

    // ========== Type Conversions ==========

    /// Convert to single-precision floats.
    #[inline(always)]
    pub fn to_f32x8(self) -> f32x8 {
        f32x8(unsafe { _mm256_cvtepi32_ps(self.0) })
    }

    // ========== Bitwise Unary Operations ==========

    /// Bitwise NOT (complement): flips all bits.
    #[inline(always)]
    pub fn not(self) -> Self {
        Self(unsafe {
            let ones = _mm256_set1_epi32(-1);
            _mm256_xor_si256(self.0, ones)
        })
    }
    // ========== Shift Operations ==========

    /// Shift each lane left by `N` bits.
    ///
    /// Bits shifted out are lost; zeros are shifted in.
    #[inline(always)]
    pub fn shl<const N: i32>(self) -> Self {
        Self(unsafe { _mm256_slli_epi32::<N>(self.0) })
    }

    /// Shift each lane right by `N` bits (logical/unsigned shift).
    ///
    /// Bits shifted out are lost; zeros are shifted in.
    #[inline(always)]
    pub fn shr<const N: i32>(self) -> Self {
        Self(unsafe { _mm256_srli_epi32::<N>(self.0) })
    }

    /// Arithmetic shift right by `N` bits (sign-extending).
    ///
    /// The sign bit is replicated into the vacated positions.
    #[inline(always)]
    pub fn shr_arithmetic<const N: i32>(self) -> Self {
        Self(unsafe { _mm256_srai_epi32::<N>(self.0) })
    }

    // ========== Extend/Widen Operations ==========

    /// Convert to f32x8.
    #[inline(always)]
    pub fn to_f32(self) -> f32x8 {
        f32x8(unsafe { _mm256_cvtepi32_ps(self.0) })
    }

    // ========== Pack/Narrow Operations ==========

    /// Pack two i32x8 vectors to i16x16 with signed saturation.
    ///
    /// Note: AVX2 pack works within 128-bit lanes.
    #[inline(always)]
    pub fn pack_i16(self, other: Self) -> i16x16 {
        i16x16(unsafe { _mm256_packs_epi32(self.0, other.0) })
    }

    /// Pack two i32x8 vectors to u16x16 with unsigned saturation.
    #[inline(always)]
    pub fn pack_u16(self, other: Self) -> u16x16 {
        u16x16(unsafe { _mm256_packus_epi32(self.0, other.0) })
    }

    // ========== Bitcast ==========
    /// Reinterpret bits as `f32x8` (zero-cost).
    #[inline(always)]
    pub fn bitcast_f32x8(self) -> f32x8 {
        unsafe { core::mem::transmute(_mm256_castsi256_ps(self.0)) }
    }

    /// Reinterpret bits as `&f32x8` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_ref_f32x8(&self) -> &f32x8 {
        unsafe { &*(self as *const Self as *const f32x8) }
    }

    /// Reinterpret bits as `&mut f32x8` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_mut_f32x8(&mut self) -> &mut f32x8 {
        unsafe { &mut *(self as *mut Self as *mut f32x8) }
    }
    /// Reinterpret bits as `u32x8` (zero-cost).
    #[inline(always)]
    pub fn bitcast_u32x8(self) -> u32x8 {
        unsafe { core::mem::transmute(self) }
    }

    /// Reinterpret bits as `&u32x8` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_ref_u32x8(&self) -> &u32x8 {
        unsafe { &*(self as *const Self as *const u32x8) }
    }

    /// Reinterpret bits as `&mut u32x8` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_mut_u32x8(&mut self) -> &mut u32x8 {
        unsafe { &mut *(self as *mut Self as *mut u32x8) }
    }
}

#[cfg(target_arch = "x86_64")]
crate::impl_int_arithmetic_ops!(i32x8, _mm256_add_epi32, _mm256_sub_epi32);
#[cfg(target_arch = "x86_64")]
crate::impl_int_mul_op!(i32x8, _mm256_mullo_epi32);
#[cfg(target_arch = "x86_64")]
crate::impl_assign_ops!(i32x8);
#[cfg(target_arch = "x86_64")]
crate::impl_bitwise_ops!(
    i32x8,
    __m256i,
    _mm256_and_si256,
    _mm256_or_si256,
    _mm256_xor_si256
);
#[cfg(target_arch = "x86_64")]
crate::impl_index!(i32x8, i32, 8);

#[cfg(target_arch = "x86_64")]
impl From<[i32; 8]> for i32x8 {
    #[inline(always)]
    fn from(arr: [i32; 8]) -> Self {
        // SAFETY: [i32; 8] and __m256i have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}
#[cfg(target_arch = "x86_64")]
impl From<i32x8> for [i32; 8] {
    #[inline(always)]
    fn from(v: i32x8) -> Self {
        // SAFETY: __m256i and [i32; 8] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}

// Scalar broadcast operations for i32x8
// These allow `v + 2.0` instead of `v + i32x8::splat(token, 2.0)`

#[cfg(target_arch = "x86_64")]
impl Add<i32> for i32x8 {
    type Output = Self;
    /// Add a scalar to all lanes.
    #[inline(always)]
    fn add(self, rhs: i32) -> Self {
        self + Self(unsafe { _mm256_set1_epi32(rhs) })
    }
}

#[cfg(target_arch = "x86_64")]
impl Sub<i32> for i32x8 {
    type Output = Self;
    /// Subtract a scalar from all lanes.
    #[inline(always)]
    fn sub(self, rhs: i32) -> Self {
        self - Self(unsafe { _mm256_set1_epi32(rhs) })
    }
}

// ============================================================================
// u32x8 - 8 x u32 (256-bit)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct u32x8(__m256i);

#[cfg(target_arch = "x86_64")]
impl u32x8 {
    pub const LANES: usize = 8;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::X64V3Token, data: &[u32; 8]) -> Self {
        Self(unsafe { _mm256_loadu_si256(data.as_ptr() as *const __m256i) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::X64V3Token, v: u32) -> Self {
        Self(unsafe { _mm256_set1_epi32(v as i32) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::X64V3Token) -> Self {
        Self(unsafe { _mm256_setzero_si256() })
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::X64V3Token, arr: [u32; 8]) -> Self {
        // SAFETY: [u32; 8] and __m256i have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [u32; 8]) {
        unsafe { _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, self.0) };
    }

    /// Convert to array
    #[inline(always)]
    pub fn to_array(self) -> [u32; 8] {
        let mut out = [0u32; 8];
        self.store(&mut out);
        out
    }

    /// Get reference to underlying array
    #[inline(always)]
    pub fn as_array(&self) -> &[u32; 8] {
        unsafe { &*(self as *const Self as *const [u32; 8]) }
    }

    /// Get mutable reference to underlying array
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut [u32; 8] {
        unsafe { &mut *(self as *mut Self as *mut [u32; 8]) }
    }

    /// Get raw intrinsic type
    #[inline(always)]
    pub fn raw(self) -> __m256i {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports the required SIMD features.
    /// Use token-gated constructors (`load`, `splat`, `zero`) for safe construction.
    #[inline(always)]
    pub unsafe fn from_raw(v: __m256i) -> Self {
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
    pub fn cast_slice(_: archmage::X64V3Token, slice: &[u32]) -> Option<&[Self]> {
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
    pub fn cast_slice_mut(_: archmage::X64V3Token, slice: &mut [u32]) -> Option<&mut [Self]> {
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
    pub fn as_bytes(&self) -> &[u8; 32] {
        // SAFETY: Self is repr(transparent) over __m256i which is 32 bytes
        unsafe { &*(self as *const Self as *const [u8; 32]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 32] {
        // SAFETY: Self is repr(transparent) over __m256i which is 32 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 32]) }
    }

    /// Create from a byte array reference (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::X64V3Token, bytes: &[u8; 32]) -> Self {
        // SAFETY: [u8; 32] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Create from an owned byte array (token-gated, zero-cost).
    ///
    /// This is a zero-cost transmute from an owned byte array.
    #[inline(always)]
    pub fn from_bytes_owned(_: archmage::X64V3Token, bytes: [u8; 32]) -> Self {
        // SAFETY: [u8; 32] and Self have identical size
        Self(unsafe { core::mem::transmute(bytes) })
    }

    /// Element-wise minimum
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(unsafe { _mm256_min_epu32(self.0, other.0) })
    }
    /// Element-wise maximum
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(unsafe { _mm256_max_epu32(self.0, other.0) })
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
        Self(unsafe { _mm256_cmpeq_epi32(self.0, other.0) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if inequality, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(unsafe {
            let eq = _mm256_cmpeq_epi32(self.0, other.0);
            let ones = _mm256_set1_epi32(-1);
            _mm256_xor_si256(eq, ones)
        })
    }

    /// Lane-wise greater-than (unsigned) comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than (unsigned), all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe {
            // Flip sign bit to convert unsigned to signed comparison
            let bias = _mm256_set1_epi32(0x8000_0000u32 as i32);
            let a = _mm256_xor_si256(self.0, bias);
            let b = _mm256_xor_si256(other.0, bias);
            _mm256_cmpgt_epi32(a, b)
        })
    }

    /// Lane-wise less-than (unsigned) comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than (unsigned), all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        other.simd_gt(self)
    }

    /// Lane-wise greater-than-or-equal (unsigned) comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than-or-equal (unsigned), all-0s otherwise.
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(unsafe {
            let lt = other.simd_gt(self);
            let ones = _mm256_set1_epi32(-1);
            _mm256_xor_si256(lt.0, ones)
        })
    }

    /// Lane-wise less-than-or-equal (unsigned) comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than-or-equal (unsigned), all-0s otherwise.
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(unsafe {
            let gt = self.simd_gt(other);
            let ones = _mm256_set1_epi32(-1);
            _mm256_xor_si256(gt.0, ones)
        })
    }

    // ========== Blending/Selection ==========

    /// Select lanes from `if_true` where mask is all-1s, `if_false` where mask is all-0s.
    ///
    /// The mask should come from a comparison operation like `simd_lt()`.
    ///
    /// # Example
    /// ```ignore
    /// let a = u32x8::splat(token, 1.0);
    /// let b = u32x8::splat(token, 2.0);
    /// let mask = a.simd_lt(b);  // all true
    /// let result = u32x8::blend(mask, a, b);  // selects a
    /// ```
    #[inline(always)]
    pub fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self(unsafe { _mm256_blendv_epi8(if_false.0, if_true.0, mask.0) })
    }

    // ========== Boolean Reductions ==========

    /// Returns true if all lanes are non-zero (truthy).
    ///
    /// Typically used with comparison results where true lanes are all-1s.
    #[inline(always)]
    pub fn all_true(self) -> bool {
        unsafe { _mm256_movemask_epi8(self.0) == -1_i32 }
    }

    /// Returns true if any lane is non-zero (truthy).
    #[inline(always)]
    pub fn any_true(self) -> bool {
        unsafe { _mm256_movemask_epi8(self.0) != 0 }
    }

    /// Extract the high bit of each lane as a bitmask.
    ///
    /// Returns a u32 where bit N corresponds to the sign bit of lane N.
    #[inline(always)]
    pub fn bitmask(self) -> u32 {
        unsafe { _mm256_movemask_ps(_mm256_castsi256_ps(self.0)) as u32 }
    }

    // ========== Horizontal Operations ==========

    /// Sum all lanes horizontally.
    ///
    /// Returns a scalar containing the sum of all 8 lanes.
    /// Note: This uses a scalar loop. For performance-critical code,
    /// consider keeping values in SIMD until the final reduction.
    #[inline(always)]
    pub fn reduce_add(self) -> u32 {
        self.as_array()
            .iter()
            .copied()
            .fold(0_u32, u32::wrapping_add)
    }

    // ========== Bitwise Unary Operations ==========

    /// Bitwise NOT (complement): flips all bits.
    #[inline(always)]
    pub fn not(self) -> Self {
        Self(unsafe {
            let ones = _mm256_set1_epi32(-1);
            _mm256_xor_si256(self.0, ones)
        })
    }
    // ========== Shift Operations ==========

    /// Shift each lane left by `N` bits.
    ///
    /// Bits shifted out are lost; zeros are shifted in.
    #[inline(always)]
    pub fn shl<const N: i32>(self) -> Self {
        Self(unsafe { _mm256_slli_epi32::<N>(self.0) })
    }

    /// Shift each lane right by `N` bits (logical/unsigned shift).
    ///
    /// Bits shifted out are lost; zeros are shifted in.
    #[inline(always)]
    pub fn shr<const N: i32>(self) -> Self {
        Self(unsafe { _mm256_srli_epi32::<N>(self.0) })
    }

    // ========== Bitcast ==========
    /// Reinterpret bits as `f32x8` (zero-cost).
    #[inline(always)]
    pub fn bitcast_f32x8(self) -> f32x8 {
        unsafe { core::mem::transmute(_mm256_castsi256_ps(self.0)) }
    }

    /// Reinterpret bits as `&f32x8` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_ref_f32x8(&self) -> &f32x8 {
        unsafe { &*(self as *const Self as *const f32x8) }
    }

    /// Reinterpret bits as `&mut f32x8` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_mut_f32x8(&mut self) -> &mut f32x8 {
        unsafe { &mut *(self as *mut Self as *mut f32x8) }
    }
    /// Reinterpret bits as `i32x8` (zero-cost).
    #[inline(always)]
    pub fn bitcast_i32x8(self) -> i32x8 {
        unsafe { core::mem::transmute(self) }
    }

    /// Reinterpret bits as `&i32x8` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_ref_i32x8(&self) -> &i32x8 {
        unsafe { &*(self as *const Self as *const i32x8) }
    }

    /// Reinterpret bits as `&mut i32x8` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_mut_i32x8(&mut self) -> &mut i32x8 {
        unsafe { &mut *(self as *mut Self as *mut i32x8) }
    }
}

#[cfg(target_arch = "x86_64")]
crate::impl_int_arithmetic_ops!(u32x8, _mm256_add_epi32, _mm256_sub_epi32);
#[cfg(target_arch = "x86_64")]
crate::impl_int_mul_op!(u32x8, _mm256_mullo_epi32);
#[cfg(target_arch = "x86_64")]
crate::impl_assign_ops!(u32x8);
#[cfg(target_arch = "x86_64")]
crate::impl_bitwise_ops!(
    u32x8,
    __m256i,
    _mm256_and_si256,
    _mm256_or_si256,
    _mm256_xor_si256
);
#[cfg(target_arch = "x86_64")]
crate::impl_index!(u32x8, u32, 8);

#[cfg(target_arch = "x86_64")]
impl From<[u32; 8]> for u32x8 {
    #[inline(always)]
    fn from(arr: [u32; 8]) -> Self {
        // SAFETY: [u32; 8] and __m256i have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}
#[cfg(target_arch = "x86_64")]
impl From<u32x8> for [u32; 8] {
    #[inline(always)]
    fn from(v: u32x8) -> Self {
        // SAFETY: __m256i and [u32; 8] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}

// Scalar broadcast operations for u32x8
// These allow `v + 2.0` instead of `v + u32x8::splat(token, 2.0)`

#[cfg(target_arch = "x86_64")]
impl Add<u32> for u32x8 {
    type Output = Self;
    /// Add a scalar to all lanes.
    #[inline(always)]
    fn add(self, rhs: u32) -> Self {
        self + Self(unsafe { _mm256_set1_epi32(rhs as i32) })
    }
}

#[cfg(target_arch = "x86_64")]
impl Sub<u32> for u32x8 {
    type Output = Self;
    /// Subtract a scalar from all lanes.
    #[inline(always)]
    fn sub(self, rhs: u32) -> Self {
        self - Self(unsafe { _mm256_set1_epi32(rhs as i32) })
    }
}

// ============================================================================
// i64x4 - 4 x i64 (256-bit)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct i64x4(__m256i);

#[cfg(target_arch = "x86_64")]
impl i64x4 {
    pub const LANES: usize = 4;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::X64V3Token, data: &[i64; 4]) -> Self {
        Self(unsafe { _mm256_loadu_si256(data.as_ptr() as *const __m256i) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::X64V3Token, v: i64) -> Self {
        Self(unsafe { _mm256_set1_epi64x(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::X64V3Token) -> Self {
        Self(unsafe { _mm256_setzero_si256() })
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::X64V3Token, arr: [i64; 4]) -> Self {
        // SAFETY: [i64; 4] and __m256i have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [i64; 4]) {
        unsafe { _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, self.0) };
    }

    /// Convert to array
    #[inline(always)]
    pub fn to_array(self) -> [i64; 4] {
        let mut out = [0i64; 4];
        self.store(&mut out);
        out
    }

    /// Get reference to underlying array
    #[inline(always)]
    pub fn as_array(&self) -> &[i64; 4] {
        unsafe { &*(self as *const Self as *const [i64; 4]) }
    }

    /// Get mutable reference to underlying array
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut [i64; 4] {
        unsafe { &mut *(self as *mut Self as *mut [i64; 4]) }
    }

    /// Get raw intrinsic type
    #[inline(always)]
    pub fn raw(self) -> __m256i {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports the required SIMD features.
    /// Use token-gated constructors (`load`, `splat`, `zero`) for safe construction.
    #[inline(always)]
    pub unsafe fn from_raw(v: __m256i) -> Self {
        Self(v)
    }

    // ========== Token-gated bytemuck replacements ==========

    /// Reinterpret a slice of scalars as a slice of SIMD vectors (token-gated).
    ///
    /// Returns `None` if the slice length is not a multiple of 4, or
    /// if the slice is not properly aligned.
    ///
    /// This is a safe, token-gated replacement for `bytemuck::cast_slice`.
    #[inline(always)]
    pub fn cast_slice(_: archmage::X64V3Token, slice: &[i64]) -> Option<&[Self]> {
        if slice.len() % 4 != 0 {
            return None;
        }
        let ptr = slice.as_ptr();
        if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {
            return None;
        }
        let len = slice.len() / 4;
        // SAFETY: alignment and length checked, layout is compatible
        Some(unsafe { core::slice::from_raw_parts(ptr as *const Self, len) })
    }

    /// Reinterpret a mutable slice of scalars as a slice of SIMD vectors (token-gated).
    ///
    /// Returns `None` if the slice length is not a multiple of 4, or
    /// if the slice is not properly aligned.
    ///
    /// This is a safe, token-gated replacement for `bytemuck::cast_slice_mut`.
    #[inline(always)]
    pub fn cast_slice_mut(_: archmage::X64V3Token, slice: &mut [i64]) -> Option<&mut [Self]> {
        if slice.len() % 4 != 0 {
            return None;
        }
        let ptr = slice.as_mut_ptr();
        if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {
            return None;
        }
        let len = slice.len() / 4;
        // SAFETY: alignment and length checked, layout is compatible
        Some(unsafe { core::slice::from_raw_parts_mut(ptr as *mut Self, len) })
    }

    /// View this vector as a byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of`.
    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8; 32] {
        // SAFETY: Self is repr(transparent) over __m256i which is 32 bytes
        unsafe { &*(self as *const Self as *const [u8; 32]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 32] {
        // SAFETY: Self is repr(transparent) over __m256i which is 32 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 32]) }
    }

    /// Create from a byte array reference (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::X64V3Token, bytes: &[u8; 32]) -> Self {
        // SAFETY: [u8; 32] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Create from an owned byte array (token-gated, zero-cost).
    ///
    /// This is a zero-cost transmute from an owned byte array.
    #[inline(always)]
    pub fn from_bytes_owned(_: archmage::X64V3Token, bytes: [u8; 32]) -> Self {
        // SAFETY: [u8; 32] and Self have identical size
        Self(unsafe { core::mem::transmute(bytes) })
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
        Self(unsafe { _mm256_cmpeq_epi64(self.0, other.0) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if inequality, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(unsafe {
            let eq = _mm256_cmpeq_epi64(self.0, other.0);
            let ones = _mm256_set1_epi64x(-1);
            _mm256_xor_si256(eq, ones)
        })
    }

    /// Lane-wise greater-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than, all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe { _mm256_cmpgt_epi64(self.0, other.0) })
    }

    /// Lane-wise less-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(unsafe { _mm256_cmpgt_epi64(other.0, self.0) })
    }

    /// Lane-wise greater-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than-or-equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(unsafe {
            let lt = _mm256_cmpgt_epi64(other.0, self.0);
            let ones = _mm256_set1_epi64x(-1);
            _mm256_xor_si256(lt, ones)
        })
    }

    /// Lane-wise less-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than-or-equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(unsafe {
            let gt = _mm256_cmpgt_epi64(self.0, other.0);
            let ones = _mm256_set1_epi64x(-1);
            _mm256_xor_si256(gt, ones)
        })
    }

    // ========== Blending/Selection ==========

    /// Select lanes from `if_true` where mask is all-1s, `if_false` where mask is all-0s.
    ///
    /// The mask should come from a comparison operation like `simd_lt()`.
    ///
    /// # Example
    /// ```ignore
    /// let a = i64x4::splat(token, 1.0);
    /// let b = i64x4::splat(token, 2.0);
    /// let mask = a.simd_lt(b);  // all true
    /// let result = i64x4::blend(mask, a, b);  // selects a
    /// ```
    #[inline(always)]
    pub fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self(unsafe { _mm256_blendv_epi8(if_false.0, if_true.0, mask.0) })
    }

    // ========== Boolean Reductions ==========

    /// Returns true if all lanes are non-zero (truthy).
    ///
    /// Typically used with comparison results where true lanes are all-1s.
    #[inline(always)]
    pub fn all_true(self) -> bool {
        unsafe { _mm256_movemask_epi8(self.0) == -1_i32 }
    }

    /// Returns true if any lane is non-zero (truthy).
    #[inline(always)]
    pub fn any_true(self) -> bool {
        unsafe { _mm256_movemask_epi8(self.0) != 0 }
    }

    /// Extract the high bit of each lane as a bitmask.
    ///
    /// Returns a u32 where bit N corresponds to the sign bit of lane N.
    #[inline(always)]
    pub fn bitmask(self) -> u32 {
        unsafe { _mm256_movemask_pd(_mm256_castsi256_pd(self.0)) as u32 }
    }

    // ========== Horizontal Operations ==========

    /// Sum all lanes horizontally.
    ///
    /// Returns a scalar containing the sum of all 4 lanes.
    /// Note: This uses a scalar loop. For performance-critical code,
    /// consider keeping values in SIMD until the final reduction.
    #[inline(always)]
    pub fn reduce_add(self) -> i64 {
        self.as_array()
            .iter()
            .copied()
            .fold(0_i64, i64::wrapping_add)
    }

    // ========== Bitwise Unary Operations ==========

    /// Bitwise NOT (complement): flips all bits.
    #[inline(always)]
    pub fn not(self) -> Self {
        Self(unsafe {
            let ones = _mm256_set1_epi64x(-1);
            _mm256_xor_si256(self.0, ones)
        })
    }
    // ========== Shift Operations ==========

    /// Shift each lane left by `N` bits.
    ///
    /// Bits shifted out are lost; zeros are shifted in.
    #[inline(always)]
    pub fn shl<const N: i32>(self) -> Self {
        Self(unsafe { _mm256_slli_epi64::<N>(self.0) })
    }

    /// Shift each lane right by `N` bits (logical/unsigned shift).
    ///
    /// Bits shifted out are lost; zeros are shifted in.
    #[inline(always)]
    pub fn shr<const N: i32>(self) -> Self {
        Self(unsafe { _mm256_srli_epi64::<N>(self.0) })
    }

    // ========== Bitcast ==========
    /// Reinterpret bits as `f64x4` (zero-cost).
    #[inline(always)]
    pub fn bitcast_f64x4(self) -> f64x4 {
        unsafe { core::mem::transmute(_mm256_castsi256_pd(self.0)) }
    }

    /// Reinterpret bits as `&f64x4` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_ref_f64x4(&self) -> &f64x4 {
        unsafe { &*(self as *const Self as *const f64x4) }
    }

    /// Reinterpret bits as `&mut f64x4` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_mut_f64x4(&mut self) -> &mut f64x4 {
        unsafe { &mut *(self as *mut Self as *mut f64x4) }
    }
    /// Reinterpret bits as `u64x4` (zero-cost).
    #[inline(always)]
    pub fn bitcast_u64x4(self) -> u64x4 {
        unsafe { core::mem::transmute(self) }
    }

    /// Reinterpret bits as `&u64x4` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_ref_u64x4(&self) -> &u64x4 {
        unsafe { &*(self as *const Self as *const u64x4) }
    }

    /// Reinterpret bits as `&mut u64x4` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_mut_u64x4(&mut self) -> &mut u64x4 {
        unsafe { &mut *(self as *mut Self as *mut u64x4) }
    }
}

#[cfg(target_arch = "x86_64")]
crate::impl_int_arithmetic_ops!(i64x4, _mm256_add_epi64, _mm256_sub_epi64);
#[cfg(target_arch = "x86_64")]
crate::impl_assign_ops!(i64x4);
#[cfg(target_arch = "x86_64")]
crate::impl_bitwise_ops!(
    i64x4,
    __m256i,
    _mm256_and_si256,
    _mm256_or_si256,
    _mm256_xor_si256
);
#[cfg(target_arch = "x86_64")]
crate::impl_index!(i64x4, i64, 4);

#[cfg(target_arch = "x86_64")]
impl From<[i64; 4]> for i64x4 {
    #[inline(always)]
    fn from(arr: [i64; 4]) -> Self {
        // SAFETY: [i64; 4] and __m256i have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}
#[cfg(target_arch = "x86_64")]
impl From<i64x4> for [i64; 4] {
    #[inline(always)]
    fn from(v: i64x4) -> Self {
        // SAFETY: __m256i and [i64; 4] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}

// Scalar broadcast operations for i64x4
// These allow `v + 2.0` instead of `v + i64x4::splat(token, 2.0)`

#[cfg(target_arch = "x86_64")]
impl Add<i64> for i64x4 {
    type Output = Self;
    /// Add a scalar to all lanes.
    #[inline(always)]
    fn add(self, rhs: i64) -> Self {
        self + Self(unsafe { _mm256_set1_epi64x(rhs) })
    }
}

#[cfg(target_arch = "x86_64")]
impl Sub<i64> for i64x4 {
    type Output = Self;
    /// Subtract a scalar from all lanes.
    #[inline(always)]
    fn sub(self, rhs: i64) -> Self {
        self - Self(unsafe { _mm256_set1_epi64x(rhs) })
    }
}

// ============================================================================
// u64x4 - 4 x u64 (256-bit)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct u64x4(__m256i);

#[cfg(target_arch = "x86_64")]
impl u64x4 {
    pub const LANES: usize = 4;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::X64V3Token, data: &[u64; 4]) -> Self {
        Self(unsafe { _mm256_loadu_si256(data.as_ptr() as *const __m256i) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::X64V3Token, v: u64) -> Self {
        Self(unsafe { _mm256_set1_epi64x(v as i64) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::X64V3Token) -> Self {
        Self(unsafe { _mm256_setzero_si256() })
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::X64V3Token, arr: [u64; 4]) -> Self {
        // SAFETY: [u64; 4] and __m256i have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [u64; 4]) {
        unsafe { _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, self.0) };
    }

    /// Convert to array
    #[inline(always)]
    pub fn to_array(self) -> [u64; 4] {
        let mut out = [0u64; 4];
        self.store(&mut out);
        out
    }

    /// Get reference to underlying array
    #[inline(always)]
    pub fn as_array(&self) -> &[u64; 4] {
        unsafe { &*(self as *const Self as *const [u64; 4]) }
    }

    /// Get mutable reference to underlying array
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut [u64; 4] {
        unsafe { &mut *(self as *mut Self as *mut [u64; 4]) }
    }

    /// Get raw intrinsic type
    #[inline(always)]
    pub fn raw(self) -> __m256i {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports the required SIMD features.
    /// Use token-gated constructors (`load`, `splat`, `zero`) for safe construction.
    #[inline(always)]
    pub unsafe fn from_raw(v: __m256i) -> Self {
        Self(v)
    }

    // ========== Token-gated bytemuck replacements ==========

    /// Reinterpret a slice of scalars as a slice of SIMD vectors (token-gated).
    ///
    /// Returns `None` if the slice length is not a multiple of 4, or
    /// if the slice is not properly aligned.
    ///
    /// This is a safe, token-gated replacement for `bytemuck::cast_slice`.
    #[inline(always)]
    pub fn cast_slice(_: archmage::X64V3Token, slice: &[u64]) -> Option<&[Self]> {
        if slice.len() % 4 != 0 {
            return None;
        }
        let ptr = slice.as_ptr();
        if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {
            return None;
        }
        let len = slice.len() / 4;
        // SAFETY: alignment and length checked, layout is compatible
        Some(unsafe { core::slice::from_raw_parts(ptr as *const Self, len) })
    }

    /// Reinterpret a mutable slice of scalars as a slice of SIMD vectors (token-gated).
    ///
    /// Returns `None` if the slice length is not a multiple of 4, or
    /// if the slice is not properly aligned.
    ///
    /// This is a safe, token-gated replacement for `bytemuck::cast_slice_mut`.
    #[inline(always)]
    pub fn cast_slice_mut(_: archmage::X64V3Token, slice: &mut [u64]) -> Option<&mut [Self]> {
        if slice.len() % 4 != 0 {
            return None;
        }
        let ptr = slice.as_mut_ptr();
        if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {
            return None;
        }
        let len = slice.len() / 4;
        // SAFETY: alignment and length checked, layout is compatible
        Some(unsafe { core::slice::from_raw_parts_mut(ptr as *mut Self, len) })
    }

    /// View this vector as a byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of`.
    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8; 32] {
        // SAFETY: Self is repr(transparent) over __m256i which is 32 bytes
        unsafe { &*(self as *const Self as *const [u8; 32]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 32] {
        // SAFETY: Self is repr(transparent) over __m256i which is 32 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 32]) }
    }

    /// Create from a byte array reference (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::X64V3Token, bytes: &[u8; 32]) -> Self {
        // SAFETY: [u8; 32] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Create from an owned byte array (token-gated, zero-cost).
    ///
    /// This is a zero-cost transmute from an owned byte array.
    #[inline(always)]
    pub fn from_bytes_owned(_: archmage::X64V3Token, bytes: [u8; 32]) -> Self {
        // SAFETY: [u8; 32] and Self have identical size
        Self(unsafe { core::mem::transmute(bytes) })
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
        Self(unsafe { _mm256_cmpeq_epi64(self.0, other.0) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if inequality, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(unsafe {
            let eq = _mm256_cmpeq_epi64(self.0, other.0);
            let ones = _mm256_set1_epi64x(-1);
            _mm256_xor_si256(eq, ones)
        })
    }

    /// Lane-wise greater-than (unsigned) comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than (unsigned), all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe {
            // Flip sign bit to convert unsigned to signed comparison
            let bias = _mm256_set1_epi64x(0x8000_0000_0000_0000u64 as i64);
            let a = _mm256_xor_si256(self.0, bias);
            let b = _mm256_xor_si256(other.0, bias);
            _mm256_cmpgt_epi64(a, b)
        })
    }

    /// Lane-wise less-than (unsigned) comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than (unsigned), all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        other.simd_gt(self)
    }

    /// Lane-wise greater-than-or-equal (unsigned) comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than-or-equal (unsigned), all-0s otherwise.
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(unsafe {
            let lt = other.simd_gt(self);
            let ones = _mm256_set1_epi64x(-1);
            _mm256_xor_si256(lt.0, ones)
        })
    }

    /// Lane-wise less-than-or-equal (unsigned) comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than-or-equal (unsigned), all-0s otherwise.
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(unsafe {
            let gt = self.simd_gt(other);
            let ones = _mm256_set1_epi64x(-1);
            _mm256_xor_si256(gt.0, ones)
        })
    }

    // ========== Blending/Selection ==========

    /// Select lanes from `if_true` where mask is all-1s, `if_false` where mask is all-0s.
    ///
    /// The mask should come from a comparison operation like `simd_lt()`.
    ///
    /// # Example
    /// ```ignore
    /// let a = u64x4::splat(token, 1.0);
    /// let b = u64x4::splat(token, 2.0);
    /// let mask = a.simd_lt(b);  // all true
    /// let result = u64x4::blend(mask, a, b);  // selects a
    /// ```
    #[inline(always)]
    pub fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self(unsafe { _mm256_blendv_epi8(if_false.0, if_true.0, mask.0) })
    }

    // ========== Boolean Reductions ==========

    /// Returns true if all lanes are non-zero (truthy).
    ///
    /// Typically used with comparison results where true lanes are all-1s.
    #[inline(always)]
    pub fn all_true(self) -> bool {
        unsafe { _mm256_movemask_epi8(self.0) == -1_i32 }
    }

    /// Returns true if any lane is non-zero (truthy).
    #[inline(always)]
    pub fn any_true(self) -> bool {
        unsafe { _mm256_movemask_epi8(self.0) != 0 }
    }

    /// Extract the high bit of each lane as a bitmask.
    ///
    /// Returns a u32 where bit N corresponds to the sign bit of lane N.
    #[inline(always)]
    pub fn bitmask(self) -> u32 {
        unsafe { _mm256_movemask_pd(_mm256_castsi256_pd(self.0)) as u32 }
    }

    // ========== Horizontal Operations ==========

    /// Sum all lanes horizontally.
    ///
    /// Returns a scalar containing the sum of all 4 lanes.
    /// Note: This uses a scalar loop. For performance-critical code,
    /// consider keeping values in SIMD until the final reduction.
    #[inline(always)]
    pub fn reduce_add(self) -> u64 {
        self.as_array()
            .iter()
            .copied()
            .fold(0_u64, u64::wrapping_add)
    }

    // ========== Bitwise Unary Operations ==========

    /// Bitwise NOT (complement): flips all bits.
    #[inline(always)]
    pub fn not(self) -> Self {
        Self(unsafe {
            let ones = _mm256_set1_epi64x(-1);
            _mm256_xor_si256(self.0, ones)
        })
    }
    // ========== Shift Operations ==========

    /// Shift each lane left by `N` bits.
    ///
    /// Bits shifted out are lost; zeros are shifted in.
    #[inline(always)]
    pub fn shl<const N: i32>(self) -> Self {
        Self(unsafe { _mm256_slli_epi64::<N>(self.0) })
    }

    /// Shift each lane right by `N` bits (logical/unsigned shift).
    ///
    /// Bits shifted out are lost; zeros are shifted in.
    #[inline(always)]
    pub fn shr<const N: i32>(self) -> Self {
        Self(unsafe { _mm256_srli_epi64::<N>(self.0) })
    }

    // ========== Bitcast ==========
    /// Reinterpret bits as `f64x4` (zero-cost).
    #[inline(always)]
    pub fn bitcast_f64x4(self) -> f64x4 {
        unsafe { core::mem::transmute(_mm256_castsi256_pd(self.0)) }
    }

    /// Reinterpret bits as `&f64x4` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_ref_f64x4(&self) -> &f64x4 {
        unsafe { &*(self as *const Self as *const f64x4) }
    }

    /// Reinterpret bits as `&mut f64x4` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_mut_f64x4(&mut self) -> &mut f64x4 {
        unsafe { &mut *(self as *mut Self as *mut f64x4) }
    }
    /// Reinterpret bits as `i64x4` (zero-cost).
    #[inline(always)]
    pub fn bitcast_i64x4(self) -> i64x4 {
        unsafe { core::mem::transmute(self) }
    }

    /// Reinterpret bits as `&i64x4` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_ref_i64x4(&self) -> &i64x4 {
        unsafe { &*(self as *const Self as *const i64x4) }
    }

    /// Reinterpret bits as `&mut i64x4` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_mut_i64x4(&mut self) -> &mut i64x4 {
        unsafe { &mut *(self as *mut Self as *mut i64x4) }
    }
}

#[cfg(target_arch = "x86_64")]
crate::impl_int_arithmetic_ops!(u64x4, _mm256_add_epi64, _mm256_sub_epi64);
#[cfg(target_arch = "x86_64")]
crate::impl_assign_ops!(u64x4);
#[cfg(target_arch = "x86_64")]
crate::impl_bitwise_ops!(
    u64x4,
    __m256i,
    _mm256_and_si256,
    _mm256_or_si256,
    _mm256_xor_si256
);
#[cfg(target_arch = "x86_64")]
crate::impl_index!(u64x4, u64, 4);

#[cfg(target_arch = "x86_64")]
impl From<[u64; 4]> for u64x4 {
    #[inline(always)]
    fn from(arr: [u64; 4]) -> Self {
        // SAFETY: [u64; 4] and __m256i have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}
#[cfg(target_arch = "x86_64")]
impl From<u64x4> for [u64; 4] {
    #[inline(always)]
    fn from(v: u64x4) -> Self {
        // SAFETY: __m256i and [u64; 4] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}

// Scalar broadcast operations for u64x4
// These allow `v + 2.0` instead of `v + u64x4::splat(token, 2.0)`

#[cfg(target_arch = "x86_64")]
impl Add<u64> for u64x4 {
    type Output = Self;
    /// Add a scalar to all lanes.
    #[inline(always)]
    fn add(self, rhs: u64) -> Self {
        self + Self(unsafe { _mm256_set1_epi64x(rhs as i64) })
    }
}

#[cfg(target_arch = "x86_64")]
impl Sub<u64> for u64x4 {
    type Output = Self;
    /// Subtract a scalar from all lanes.
    #[inline(always)]
    fn sub(self, rhs: u64) -> Self {
        self - Self(unsafe { _mm256_set1_epi64x(rhs as i64) })
    }
}

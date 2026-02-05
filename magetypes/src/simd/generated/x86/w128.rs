//! 128-bit (SSE) SIMD types.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

use core::arch::x86_64::*;
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

// ============================================================================
// f32x4 - 4 x f32 (128-bit)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct f32x4(__m128);

#[cfg(target_arch = "x86_64")]
impl f32x4 {
    pub const LANES: usize = 4;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::X64V3Token, data: &[f32; 4]) -> Self {
        Self(unsafe { _mm_loadu_ps(data.as_ptr()) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::X64V3Token, v: f32) -> Self {
        Self(unsafe { _mm_set1_ps(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::X64V3Token) -> Self {
        Self(unsafe { _mm_setzero_ps() })
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::X64V3Token, arr: [f32; 4]) -> Self {
        // SAFETY: [f32; 4] and __m128 have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [f32; 4]) {
        unsafe { _mm_storeu_ps(out.as_mut_ptr(), self.0) };
    }

    /// Convert to array
    #[inline(always)]
    pub fn to_array(self) -> [f32; 4] {
        let mut out = [0.0f32; 4];
        self.store(&mut out);
        out
    }

    /// Get reference to underlying array
    #[inline(always)]
    pub fn as_array(&self) -> &[f32; 4] {
        unsafe { &*(self as *const Self as *const [f32; 4]) }
    }

    /// Get mutable reference to underlying array
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut [f32; 4] {
        unsafe { &mut *(self as *mut Self as *mut [f32; 4]) }
    }

    /// Get raw intrinsic type
    #[inline(always)]
    pub fn raw(self) -> __m128 {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports the required SIMD features.
    /// Use token-gated constructors (`load`, `splat`, `zero`) for safe construction.
    #[inline(always)]
    pub unsafe fn from_raw(v: __m128) -> Self {
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
    pub fn cast_slice(_: archmage::X64V3Token, slice: &[f32]) -> Option<&[Self]> {
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
    pub fn cast_slice_mut(_: archmage::X64V3Token, slice: &mut [f32]) -> Option<&mut [Self]> {
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
    pub fn as_bytes(&self) -> &[u8; 16] {
        // SAFETY: Self is repr(transparent) over __m128 which is 16 bytes
        unsafe { &*(self as *const Self as *const [u8; 16]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 16] {
        // SAFETY: Self is repr(transparent) over __m128 which is 16 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 16]) }
    }

    /// Create from a byte array reference (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::X64V3Token, bytes: &[u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Create from an owned byte array (token-gated, zero-cost).
    ///
    /// This is a zero-cost transmute from an owned byte array.
    #[inline(always)]
    pub fn from_bytes_owned(_: archmage::X64V3Token, bytes: [u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(bytes) })
    }

    // ========== Implementation identification ==========

    /// Returns a string identifying this type's implementation.
    ///
    /// This is useful for verifying that the correct implementation is being used
    /// at compile time or at runtime (via `#[magetypes]` dispatch).
    #[inline(always)]
    pub const fn implementation_name() -> &'static str {
        "x86::v3::f32x4"
    }

    /// Element-wise minimum
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(unsafe { _mm_min_ps(self.0, other.0) })
    }
    /// Element-wise maximum
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(unsafe { _mm_max_ps(self.0, other.0) })
    }
    /// Clamp values between lo and hi
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }

    /// Square root
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        Self(unsafe { _mm_sqrt_ps(self.0) })
    }
    /// Absolute value
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(unsafe {
            let mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFF_FFFFu32 as i32));
            _mm_and_ps(self.0, mask)
        })
    }
    /// Round toward negative infinity
    #[inline(always)]
    pub fn floor(self) -> Self {
        Self(unsafe { _mm_floor_ps(self.0) })
    }
    /// Round toward positive infinity
    #[inline(always)]
    pub fn ceil(self) -> Self {
        Self(unsafe { _mm_ceil_ps(self.0) })
    }
    /// Round to nearest integer
    #[inline(always)]
    pub fn round(self) -> Self {
        Self(unsafe { _mm_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(self.0) })
    }
    /// Fused multiply-add: self * a + b
    #[inline(always)]
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        Self(unsafe { _mm_fmadd_ps(self.0, a.0, b.0) })
    }

    /// Fused multiply-sub: self * a - b
    #[inline(always)]
    pub fn mul_sub(self, a: Self, b: Self) -> Self {
        Self(unsafe { _mm_fmsub_ps(self.0, a.0, b.0) })
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
        Self(unsafe { _mm_cmpeq_ps(self.0, other.0) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if inequality, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(unsafe { _mm_cmpneq_ps(self.0, other.0) })
    }

    /// Lane-wise less-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(unsafe { _mm_cmplt_ps(self.0, other.0) })
    }

    /// Lane-wise less-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than-or-equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(unsafe { _mm_cmple_ps(self.0, other.0) })
    }

    /// Lane-wise greater-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than, all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe { _mm_cmpgt_ps(self.0, other.0) })
    }

    /// Lane-wise greater-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than-or-equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(unsafe { _mm_cmpge_ps(self.0, other.0) })
    }

    // ========== Blending/Selection ==========

    /// Select lanes from `if_true` where mask is all-1s, `if_false` where mask is all-0s.
    ///
    /// The mask should come from a comparison operation like `simd_lt()`.
    ///
    /// # Example
    /// ```ignore
    /// let a = f32x4::splat(token, 1.0);
    /// let b = f32x4::splat(token, 2.0);
    /// let mask = a.simd_lt(b);  // all true
    /// let result = f32x4::blend(mask, a, b);  // selects a
    /// ```
    #[inline(always)]
    pub fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self(unsafe { _mm_blendv_ps(if_false.0, if_true.0, mask.0) })
    }

    // ========== Horizontal Operations ==========

    /// Sum all lanes horizontally.
    ///
    /// Returns a scalar containing the sum of all 4 lanes.
    #[inline(always)]
    pub fn reduce_add(self) -> f32 {
        unsafe {
            let h1 = _mm_hadd_ps(self.0, self.0);
            let h2 = _mm_hadd_ps(h1, h1);
            _mm_cvtss_f32(h2)
        }
    }

    /// Find the minimum value across all lanes.
    #[inline(always)]
    pub fn reduce_min(self) -> f32 {
        unsafe {
            let shuf = _mm_shuffle_ps::<0b10_11_00_01>(self.0, self.0);
            let m1 = _mm_min_ps(self.0, shuf);
            let shuf2 = _mm_shuffle_ps::<0b00_00_10_10>(m1, m1);
            let m2 = _mm_min_ps(m1, shuf2);
            _mm_cvtss_f32(m2)
        }
    }

    /// Find the maximum value across all lanes.
    #[inline(always)]
    pub fn reduce_max(self) -> f32 {
        unsafe {
            let shuf = _mm_shuffle_ps::<0b10_11_00_01>(self.0, self.0);
            let m1 = _mm_max_ps(self.0, shuf);
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
    pub fn to_i32x4(self) -> i32x4 {
        i32x4(unsafe { _mm_cvttps_epi32(self.0) })
    }

    /// Convert to signed 32-bit integers, rounding to nearest even.
    ///
    /// Values outside the representable range become `i32::MIN` (0x80000000).
    #[inline(always)]
    pub fn to_i32x4_round(self) -> i32x4 {
        i32x4(unsafe { _mm_cvtps_epi32(self.0) })
    }

    /// Create from signed 32-bit integers.
    #[inline(always)]
    pub fn from_i32x4(v: i32x4) -> Self {
        Self(unsafe { _mm_cvtepi32_ps(v.0) })
    }

    // ========== Approximation Operations ==========

    /// Fast reciprocal approximation (1/x) with ~12-bit precision.
    ///
    /// For full precision, use `recip()` which applies Newton-Raphson refinement.
    #[inline(always)]
    pub fn rcp_approx(self) -> Self {
        Self(unsafe { _mm_rcp_ps(self.0) })
    }

    /// Precise reciprocal (1/x) using Newton-Raphson refinement.
    ///
    /// More accurate than `rcp_approx()` but slower. For maximum speed
    /// with acceptable precision loss, use `rcp_approx()`.
    #[inline(always)]
    pub fn recip(self) -> Self {
        // Newton-Raphson: x' = x * (2 - a*x)
        let approx = self.rcp_approx();
        let two = Self(unsafe { _mm_set1_ps(2.0) });
        // One iteration gives ~24-bit precision from ~12-bit
        approx * (two - self * approx)
    }

    /// Fast reciprocal square root approximation (1/sqrt(x)) with ~12-bit precision.
    ///
    /// For full precision, use `sqrt().recip()` or apply Newton-Raphson manually.
    #[inline(always)]
    pub fn rsqrt_approx(self) -> Self {
        Self(unsafe { _mm_rsqrt_ps(self.0) })
    }

    /// Precise reciprocal square root (1/sqrt(x)) using Newton-Raphson refinement.
    #[inline(always)]
    pub fn rsqrt(self) -> Self {
        // Newton-Raphson for rsqrt: y' = 0.5 * y * (3 - x * y * y)
        let approx = self.rsqrt_approx();
        let half = Self(unsafe { _mm_set1_ps(0.5) });
        let three = Self(unsafe { _mm_set1_ps(3.0) });
        half * approx * (three - self * approx * approx)
    }

    // ========== Bitwise Unary Operations ==========

    /// Bitwise NOT (complement): flips all bits.
    #[inline(always)]
    pub fn not(self) -> Self {
        Self(unsafe {
            let ones = _mm_set1_epi32(-1);
            let as_int = _mm_castps_si128(self.0);
            _mm_castsi128_ps(_mm_xor_si128(as_int, ones))
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
            let x_bits = _mm_castps_si128(self.0);
            let offset = _mm_set1_epi32(0x3f2aaaab_u32 as i32);
            let exp_bits = _mm_sub_epi32(x_bits, offset);
            let exp_shifted = _mm_srai_epi32::<23>(exp_bits);

            let mantissa_bits = _mm_sub_epi32(x_bits, _mm_slli_epi32::<23>(exp_shifted));
            let mantissa = _mm_castsi128_ps(mantissa_bits);
            let exp_val = _mm_cvtepi32_ps(exp_shifted);

            let one = _mm_set1_ps(1.0);
            let m = _mm_sub_ps(mantissa, one);

            // Horner's for numerator: P2*m^2 + P1*m + P0
            let yp = _mm_fmadd_ps(_mm_set1_ps(P2), m, _mm_set1_ps(P1));
            let yp = _mm_fmadd_ps(yp, m, _mm_set1_ps(P0));

            // Horner's for denominator: Q2*m^2 + Q1*m + Q0
            let yq = _mm_fmadd_ps(_mm_set1_ps(Q2), m, _mm_set1_ps(Q1));
            let yq = _mm_fmadd_ps(yq, m, _mm_set1_ps(Q0));

            Self(_mm_add_ps(_mm_div_ps(yp, yq), exp_val))
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
            let x = _mm_max_ps(self.0, _mm_set1_ps(-126.0));
            let x = _mm_min_ps(x, _mm_set1_ps(126.0));

            // Split into integer and fractional parts
            let xi = _mm_floor_ps(x);
            let xf = _mm_sub_ps(x, xi);

            // Polynomial for 2^frac
            let poly = _mm_fmadd_ps(_mm_set1_ps(C3), xf, _mm_set1_ps(C2));
            let poly = _mm_fmadd_ps(poly, xf, _mm_set1_ps(C1));
            let poly = _mm_fmadd_ps(poly, xf, _mm_set1_ps(C0));

            // Scale by 2^integer using bit manipulation
            let xi_i32 = _mm_cvtps_epi32(xi);
            let bias = _mm_set1_epi32(127);
            let scale_bits = _mm_slli_epi32::<23>(_mm_add_epi32(xi_i32, bias));
            let scale = _mm_castsi128_ps(scale_bits);

            Self(_mm_mul_ps(poly, scale))
        }
    }

    /// Low-precision natural logarithm.
    ///
    /// Computed as `log2_lowp(x) * ln(2)`. For higher precision, use `ln_midp()`.
    #[inline(always)]
    pub fn ln_lowp(self) -> Self {
        const LN2: f32 = core::f32::consts::LN_2;
        unsafe { Self(_mm_mul_ps(self.log2_lowp().0, _mm_set1_ps(LN2))) }
    }

    /// Low-precision natural exponential (e^x).
    ///
    /// Computed as `exp2_lowp(x * log2(e))`. For higher precision, use `exp_midp()`.
    #[inline(always)]
    pub fn exp_lowp(self) -> Self {
        const LOG2_E: f32 = core::f32::consts::LOG2_E;
        unsafe { Self(_mm_mul_ps(self.0, _mm_set1_ps(LOG2_E))).exp2_lowp() }
    }

    /// Low-precision base-10 logarithm.
    ///
    /// Computed as `log2_lowp(x) / log2(10)`.
    #[inline(always)]
    pub fn log10_lowp(self) -> Self {
        const LOG10_2: f32 = core::f32::consts::LOG10_2; // 1/log2(10)
        unsafe { Self(_mm_mul_ps(self.log2_lowp().0, _mm_set1_ps(LOG10_2))) }
    }

    /// Low-precision power function (self^n).
    ///
    /// Computed as `exp2_lowp(n * log2_lowp(self))`. For higher precision, use `pow_midp()`.
    /// Note: Only valid for positive self values.
    #[inline(always)]
    pub fn pow_lowp(self, n: f32) -> Self {
        unsafe { Self(_mm_mul_ps(self.log2_lowp().0, _mm_set1_ps(n))).exp2_lowp() }
    }

    /// Low-precision base-2 logarithm - unchecked variant.
    ///
    /// Identical to `log2_lowp()` (lowp already skips edge case handling).
    #[inline(always)]
    pub fn log2_lowp_unchecked(self) -> Self {
        self.log2_lowp()
    }

    /// Low-precision base-2 exponential - unchecked variant.
    ///
    /// Identical to `exp2_lowp()` (lowp already clamps to safe range).
    #[inline(always)]
    pub fn exp2_lowp_unchecked(self) -> Self {
        self.exp2_lowp()
    }

    /// Low-precision natural logarithm - unchecked variant.
    ///
    /// Identical to `ln_lowp()` (lowp already skips edge case handling).
    #[inline(always)]
    pub fn ln_lowp_unchecked(self) -> Self {
        self.ln_lowp()
    }

    /// Low-precision natural exponential - unchecked variant.
    ///
    /// Identical to `exp_lowp()` (lowp already clamps to safe range).
    #[inline(always)]
    pub fn exp_lowp_unchecked(self) -> Self {
        self.exp_lowp()
    }

    /// Low-precision base-10 logarithm - unchecked variant.
    ///
    /// Identical to `log10_lowp()` (lowp already skips edge case handling).
    #[inline(always)]
    pub fn log10_lowp_unchecked(self) -> Self {
        self.log10_lowp()
    }

    /// Low-precision power function - unchecked variant.
    ///
    /// Identical to `pow_lowp()` (lowp already skips edge case handling).
    #[inline(always)]
    pub fn pow_lowp_unchecked(self, n: f32) -> Self {
        self.pow_lowp(n)
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
            let x_bits = _mm_castps_si128(self.0);

            // Normalize mantissa to [sqrt(2)/2, sqrt(2)]
            let offset = _mm_set1_epi32((ONE - SQRT2_OVER_2) as i32);
            let adjusted = _mm_add_epi32(x_bits, offset);

            // Extract exponent
            let exp_raw = _mm_srai_epi32::<23>(adjusted);
            let exp_biased = _mm_sub_epi32(exp_raw, _mm_set1_epi32(EXPONENT_BIAS));
            let n = _mm_cvtepi32_ps(exp_biased);

            // Reconstruct normalized mantissa
            let mantissa_bits = _mm_and_si128(adjusted, _mm_set1_epi32(MANTISSA_MASK));
            let a_bits = _mm_add_epi32(mantissa_bits, _mm_set1_epi32(SQRT2_OVER_2 as i32));
            let a = _mm_castsi128_ps(a_bits);

            // y = (a - 1) / (a + 1)
            let one = _mm_set1_ps(1.0);
            let y = _mm_div_ps(_mm_sub_ps(a, one), _mm_add_ps(a, one));
            let y2 = _mm_mul_ps(y, y);

            // Polynomial: C0*y + C1*y^3 + C2*y^5 + C3*y^7 = y*(C0 + y^2*(C1 + y^2*(C2 + C3*y^2)))
            let poly = _mm_fmadd_ps(_mm_set1_ps(C3), y2, _mm_set1_ps(C2));
            let poly = _mm_fmadd_ps(poly, y2, _mm_set1_ps(C1));
            let poly = _mm_fmadd_ps(poly, y2, _mm_set1_ps(C0));

            Self(_mm_fmadd_ps(poly, y, n))
        }
    }

    /// Mid-precision base-2 logarithm (~3 ULP max error) with edge case handling.
    ///
    /// Returns -inf for 0, NaN for negative, correct results for inf/NaN.
    #[inline(always)]
    pub fn log2_midp(self) -> Self {
        unsafe {
            let x = self.0;
            let zero = _mm_setzero_ps();
            let neg_inf = _mm_set1_ps(f32::NEG_INFINITY);
            let nan = _mm_set1_ps(f32::NAN);
            let inf = _mm_set1_ps(f32::INFINITY);

            // Handle special cases
            let is_zero = _mm_cmp_ps::<_CMP_EQ_OQ>(x, zero);
            let is_negative = _mm_cmp_ps::<_CMP_LT_OQ>(x, zero);
            let is_inf = _mm_cmp_ps::<_CMP_EQ_OQ>(x, inf);

            // Compute log2 for normal values
            let log_result = self.log2_midp_unchecked().0;

            // Apply special case results
            let result = _mm_blendv_ps(log_result, neg_inf, is_zero);
            let result = _mm_blendv_ps(result, nan, is_negative);
            let result = _mm_blendv_ps(result, inf, is_inf);

            Self(result)
        }
    }

    /// Mid-precision base-2 exponential (~2 ULP max error) - unchecked variant.
    ///
    /// Uses degree-6 polynomial approximation.
    /// **Warning**: Does not handle edge cases (underflow, overflow).
    /// Use `exp2_midp()` for correct IEEE behavior on edge cases.
    #[inline(always)]
    pub fn exp2_midp_unchecked(self) -> Self {
        // Polynomial coefficients (degree 6 Remez)
        const C0: f32 = 1.0;
        const C1: f32 = 0.693_147_182;
        const C2: f32 = 0.240_226_463;
        const C3: f32 = 0.055_504_545;
        const C4: f32 = 0.009_618_055;
        const C5: f32 = 0.001_333_37;
        const C6: f32 = 0.000_154_47;

        unsafe {
            let x = self.0;

            // Split into integer and fractional parts
            let xi = _mm_floor_ps(x);
            let xf = _mm_sub_ps(x, xi);

            // Polynomial for 2^frac
            let poly = _mm_fmadd_ps(_mm_set1_ps(C6), xf, _mm_set1_ps(C5));
            let poly = _mm_fmadd_ps(poly, xf, _mm_set1_ps(C4));
            let poly = _mm_fmadd_ps(poly, xf, _mm_set1_ps(C3));
            let poly = _mm_fmadd_ps(poly, xf, _mm_set1_ps(C2));
            let poly = _mm_fmadd_ps(poly, xf, _mm_set1_ps(C1));
            let poly = _mm_fmadd_ps(poly, xf, _mm_set1_ps(C0));

            // Scale by 2^integer using bit manipulation
            let xi_i32 = _mm_cvtps_epi32(xi);
            let bias = _mm_set1_epi32(127);
            let scale_bits = _mm_slli_epi32::<23>(_mm_add_epi32(xi_i32, bias));
            let scale = _mm_castsi128_ps(scale_bits);

            Self(_mm_mul_ps(poly, scale))
        }
    }

    /// Mid-precision base-2 exponential (~2 ULP max error) with edge case handling.
    ///
    /// Returns 0 for large negative, inf for large positive, correct results for inf/NaN.
    #[inline(always)]
    pub fn exp2_midp(self) -> Self {
        unsafe {
            let x = self.0;
            let zero = _mm_setzero_ps();
            let inf = _mm_set1_ps(f32::INFINITY);

            // Clamp to prevent overflow in intermediate calculations
            let x_clamped = _mm_max_ps(x, _mm_set1_ps(-150.0));
            let x_clamped = _mm_min_ps(x_clamped, _mm_set1_ps(128.0));

            // Compute exp2 for clamped values
            let exp_result = Self(x_clamped).exp2_midp_unchecked().0;

            // Handle edge cases
            let underflow = _mm_cmp_ps::<_CMP_LT_OQ>(x, _mm_set1_ps(-150.0));
            let overflow = _mm_cmp_ps::<_CMP_GT_OQ>(x, _mm_set1_ps(128.0));

            let result = _mm_blendv_ps(exp_result, zero, underflow);
            let result = _mm_blendv_ps(result, inf, overflow);

            Self(result)
        }
    }

    /// Mid-precision natural logarithm - unchecked variant.
    ///
    /// Computed as `log2_midp_unchecked(x) * ln(2)`.
    #[inline(always)]
    pub fn ln_midp_unchecked(self) -> Self {
        const LN2: f32 = core::f32::consts::LN_2;
        unsafe { Self(_mm_mul_ps(self.log2_midp_unchecked().0, _mm_set1_ps(LN2))) }
    }

    /// Mid-precision natural logarithm with edge case handling.
    ///
    /// Returns -inf for 0, NaN for negative, correct results for inf/NaN.
    #[inline(always)]
    pub fn ln_midp(self) -> Self {
        const LN2: f32 = core::f32::consts::LN_2;
        unsafe { Self(_mm_mul_ps(self.log2_midp().0, _mm_set1_ps(LN2))) }
    }

    /// Mid-precision natural exponential (e^x) - unchecked variant.
    ///
    /// Computed as `exp2_midp_unchecked(x * log2(e))`.
    #[inline(always)]
    pub fn exp_midp_unchecked(self) -> Self {
        const LOG2_E: f32 = core::f32::consts::LOG2_E;
        unsafe { Self(_mm_mul_ps(self.0, _mm_set1_ps(LOG2_E))).exp2_midp_unchecked() }
    }

    /// Mid-precision natural exponential (e^x) with edge case handling.
    ///
    /// Returns 0 for large negative, inf for large positive, correct results for inf/NaN.
    #[inline(always)]
    pub fn exp_midp(self) -> Self {
        const LOG2_E: f32 = core::f32::consts::LOG2_E;
        unsafe { Self(_mm_mul_ps(self.0, _mm_set1_ps(LOG2_E))).exp2_midp() }
    }

    /// Mid-precision power function (self^n) - unchecked variant.
    ///
    /// Computed as `exp2_midp_unchecked(n * log2_midp_unchecked(self))`.
    /// Note: Only valid for positive self values.
    #[inline(always)]
    pub fn pow_midp_unchecked(self, n: f32) -> Self {
        unsafe {
            Self(_mm_mul_ps(self.log2_midp_unchecked().0, _mm_set1_ps(n))).exp2_midp_unchecked()
        }
    }

    /// Mid-precision power function (self^n) with edge case handling.
    ///
    /// Handles 0, negative, inf, and NaN in base. Only valid for positive self values.
    #[inline(always)]
    pub fn pow_midp(self, n: f32) -> Self {
        unsafe { Self(_mm_mul_ps(self.log2_midp().0, _mm_set1_ps(n))).exp2_midp() }
    }

    /// Mid-precision base-2 logarithm with full IEEE compliance.
    ///
    /// Handles all edge cases including denormals.
    /// About 50% slower than `log2_midp()` due to denormal scaling.
    #[inline(always)]
    pub fn log2_midp_precise(self) -> Self {
        unsafe {
            const SCALE_UP: f32 = 16777216.0; // 2^24
            const SCALE_ADJUST: f32 = 24.0; // log2(2^24)
            const DENORM_LIMIT: f32 = 1.17549435e-38;

            let zero = _mm_setzero_ps();
            let is_positive = _mm_cmp_ps::<_CMP_GT_OQ>(self.0, zero);
            let abs_x = _mm_andnot_ps(_mm_set1_ps(-0.0), self.0);
            let is_small = _mm_cmp_ps::<_CMP_LT_OQ>(abs_x, _mm_set1_ps(DENORM_LIMIT));
            let is_denorm = _mm_and_ps(is_positive, is_small);

            let scaled_x = _mm_mul_ps(self.0, _mm_set1_ps(SCALE_UP));
            let x_for_log = _mm_blendv_ps(self.0, scaled_x, is_denorm);

            let result = Self(x_for_log).log2_midp();

            let adjusted = _mm_sub_ps(result.0, _mm_set1_ps(SCALE_ADJUST));
            Self(_mm_blendv_ps(result.0, adjusted, is_denorm))
        }
    }

    /// Mid-precision natural logarithm with full IEEE compliance.
    ///
    /// Handles all edge cases including denormals.
    /// About 50% slower than `ln_midp()` due to denormal scaling.
    #[inline(always)]
    pub fn ln_midp_precise(self) -> Self {
        const LN2: f32 = core::f32::consts::LN_2;
        unsafe { Self(_mm_mul_ps(self.log2_midp_precise().0, _mm_set1_ps(LN2))) }
    }

    /// Mid-precision power function with denormal handling.
    ///
    /// Uses `log2_midp_precise()` to handle denormal inputs correctly.
    /// Note: Only valid for positive self values.
    #[inline(always)]
    pub fn pow_midp_precise(self, n: f32) -> Self {
        unsafe { Self(_mm_mul_ps(self.log2_midp_precise().0, _mm_set1_ps(n))).exp2_midp() }
    }

    /// Mid-precision base-10 logarithm - unchecked variant.
    ///
    /// Computed as `log2_midp_unchecked(x) * log10(2)`.
    /// **Warning**: Does not handle edge cases (0, negative, inf, NaN, denormals).
    #[inline(always)]
    pub fn log10_midp_unchecked(self) -> Self {
        const LOG10_2: f32 = core::f32::consts::LOG10_2;
        unsafe {
            Self(_mm_mul_ps(
                self.log2_midp_unchecked().0,
                _mm_set1_ps(LOG10_2),
            ))
        }
    }

    /// Mid-precision base-10 logarithm with edge case handling.
    ///
    /// Computed as `log2_midp(x) * log10(2)`.
    /// Returns -inf for 0, NaN for negative, correct results for inf/NaN.
    #[inline(always)]
    pub fn log10_midp(self) -> Self {
        const LOG10_2: f32 = core::f32::consts::LOG10_2;
        unsafe { Self(_mm_mul_ps(self.log2_midp().0, _mm_set1_ps(LOG10_2))) }
    }

    /// Mid-precision base-10 logarithm with full IEEE compliance.
    ///
    /// Handles all edge cases including denormals.
    /// About 50% slower than `log10_midp()` due to denormal scaling.
    #[inline(always)]
    pub fn log10_midp_precise(self) -> Self {
        const LOG10_2: f32 = core::f32::consts::LOG10_2;
        unsafe { Self(_mm_mul_ps(self.log2_midp_precise().0, _mm_set1_ps(LOG10_2))) }
    }

    /// Mid-precision cube root (~1 ULP max error).
    ///
    /// Uses scalar extraction for initial guess + Newton-Raphson.
    /// Handles negative values correctly (returns -cbrt(|x|)).
    ///
    /// Does not handle denormals. Use `cbrt_midp_precise()` if denormal support is needed.
    #[inline(always)]
    pub fn cbrt_midp(self) -> Self {
        // Kahan's magic constant for initial approximation
        const KAHAN_CBRT: f32 = 0.333_333_313;
        const TWO_THIRDS: f32 = 0.666_666_627;

        unsafe {
            let x = self.0;

            // Save sign and work with absolute value
            let sign_mask = _mm_set1_ps(-0.0);
            let sign = _mm_and_ps(x, sign_mask);
            let abs_x = _mm_andnot_ps(sign_mask, x);

            // Extract to scalar for initial approximation
            let arr: [f32; 4] = core::mem::transmute(abs_x);
            let approx: [f32; 4] = [
                f32::from_bits((arr[0].to_bits() / 3) + 0x2a508c2d),
                f32::from_bits((arr[1].to_bits() / 3) + 0x2a508c2d),
                f32::from_bits((arr[2].to_bits() / 3) + 0x2a508c2d),
                f32::from_bits((arr[3].to_bits() / 3) + 0x2a508c2d),
            ];
            let mut y = _mm_loadu_ps(approx.as_ptr());

            // Newton-Raphson iterations: y = y * (2/3 + x/(3*y^3))
            for _ in 0..3 {
                let y2 = _mm_mul_ps(y, y);
                let y3 = _mm_mul_ps(y2, y);
                let term = _mm_div_ps(abs_x, _mm_mul_ps(_mm_set1_ps(3.0), y3));
                y = _mm_mul_ps(y, _mm_add_ps(_mm_set1_ps(TWO_THIRDS), term));
            }

            // Restore sign
            Self(_mm_or_ps(y, sign))
        }
    }

    /// Mid-precision cube root with denormal handling (~1 ULP max error).
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

            let abs_x = _mm_andnot_ps(_mm_set1_ps(-0.0), self.0);
            let is_denorm = _mm_cmp_ps::<_CMP_LT_OQ>(abs_x, _mm_set1_ps(DENORM_LIMIT));

            // Scale up denormals
            let scaled_x = _mm_mul_ps(self.0, _mm_set1_ps(SCALE_UP));
            let x_for_cbrt = _mm_blendv_ps(self.0, scaled_x, is_denorm);

            // Compute cbrt with edge case handling
            let result = Self(x_for_cbrt).cbrt_midp();

            // Scale down results from denormal inputs
            let scaled_result = _mm_mul_ps(result.0, _mm_set1_ps(SCALE_DOWN));
            Self(_mm_blendv_ps(result.0, scaled_result, is_denorm))
        }
    }

    // ========== Load and Convert ==========

    /// Load 4 u8 values and convert to f32x4.
    ///
    /// Useful for image processing: load pixel values directly to float.
    #[inline(always)]
    pub fn from_u8(bytes: &[u8; 4]) -> Self {
        unsafe {
            // Load 4 bytes into low part of XMM register
            let b = _mm_cvtsi32_si128(i32::from_ne_bytes(*bytes));
            let i32s = _mm_cvtepu8_epi32(b);
            Self(_mm_cvtepi32_ps(i32s))
        }
    }

    /// Convert to 4 u8 values with saturation.
    ///
    /// Values are clamped to [0, 255] and rounded.
    #[inline(always)]
    pub fn to_u8(self) -> [u8; 4] {
        unsafe {
            // Convert to i32, pack to i16, pack to u8
            let i32s = _mm_cvtps_epi32(self.0);
            let i16s = _mm_packs_epi32(i32s, i32s);
            let u8s = _mm_packus_epi16(i16s, i16s);
            let val = _mm_cvtsi128_si32(u8s) as u32;
            val.to_ne_bytes()
        }
    }

    // ========== Interleave Operations ==========

    /// Interleave low elements: [a0,a1,a2,a3] + [b0,b1,b2,b3] → [a0,b0,a1,b1]
    #[inline(always)]
    pub fn interleave_lo(self, other: Self) -> Self {
        Self(unsafe { _mm_unpacklo_ps(self.0, other.0) })
    }

    /// Interleave high elements: [a0,a1,a2,a3] + [b0,b1,b2,b3] → [a2,b2,a3,b3]
    #[inline(always)]
    pub fn interleave_hi(self, other: Self) -> Self {
        Self(unsafe { _mm_unpackhi_ps(self.0, other.0) })
    }

    /// Interleave two vectors: returns (interleave_lo, interleave_hi)
    #[inline(always)]
    pub fn interleave(self, other: Self) -> (Self, Self) {
        (self.interleave_lo(other), self.interleave_hi(other))
    }

    // ========== 4-Channel Interleave/Deinterleave ==========

    /// Deinterleave 4 RGBA pixels from AoS to SoA format.
    ///
    /// Input: 4 vectors where each contains one pixel `[R, G, B, A]`.
    /// Output: 4 vectors where each contains one channel across all pixels.
    ///
    /// ```text
    /// Input:  rgba[0] = [R0, G0, B0, A0]  (pixel 0)
    ///         rgba[1] = [R1, G1, B1, A1]  (pixel 1)
    ///         rgba[2] = [R2, G2, B2, A2]  (pixel 2)
    ///         rgba[3] = [R3, G3, B3, A3]  (pixel 3)
    ///
    /// Output: [0] = [R0, R1, R2, R3]  (red channel)
    ///         [1] = [G0, G1, G2, G3]  (green channel)
    ///         [2] = [B0, B1, B2, B3]  (blue channel)
    ///         [3] = [A0, A1, A2, A3]  (alpha channel)
    /// ```
    #[inline]
    pub fn deinterleave_4ch(rgba: [Self; 4]) -> [Self; 4] {
        Self::transpose_4x4_copy(rgba)
    }

    /// Interleave 4 channels from SoA to AoS format.
    ///
    /// Input: 4 vectors where each contains one channel across pixels.
    /// Output: 4 vectors where each contains one complete RGBA pixel.
    ///
    /// This is the inverse of `deinterleave_4ch`.
    #[inline]
    pub fn interleave_4ch(channels: [Self; 4]) -> [Self; 4] {
        Self::transpose_4x4_copy(channels)
    }

    /// Load 4 RGBA u8 pixels and deinterleave to 4 f32x4 channel vectors.
    ///
    /// Input: 16 bytes = 4 RGBA pixels in interleaved format.
    /// Output: (R, G, B, A) where each is f32x4 with values in [0.0, 255.0].
    #[inline]
    pub fn load_4_rgba_u8(rgba: &[u8; 16]) -> (Self, Self, Self, Self) {
        unsafe {
            let v = _mm_loadu_si128(rgba.as_ptr() as *const __m128i);

            // Shuffle masks to gather each channel
            // R: bytes 0, 4, 8, 12 → positions 0, 1, 2, 3
            let r_mask = _mm_setr_epi8(0, -1, -1, -1, 4, -1, -1, -1, 8, -1, -1, -1, 12, -1, -1, -1);
            // G: bytes 1, 5, 9, 13
            let g_mask = _mm_setr_epi8(1, -1, -1, -1, 5, -1, -1, -1, 9, -1, -1, -1, 13, -1, -1, -1);
            // B: bytes 2, 6, 10, 14
            let b_mask =
                _mm_setr_epi8(2, -1, -1, -1, 6, -1, -1, -1, 10, -1, -1, -1, 14, -1, -1, -1);
            // A: bytes 3, 7, 11, 15
            let a_mask =
                _mm_setr_epi8(3, -1, -1, -1, 7, -1, -1, -1, 11, -1, -1, -1, 15, -1, -1, -1);

            // Shuffle and convert to f32
            let r_i32 = _mm_shuffle_epi8(v, r_mask);
            let g_i32 = _mm_shuffle_epi8(v, g_mask);
            let b_i32 = _mm_shuffle_epi8(v, b_mask);
            let a_i32 = _mm_shuffle_epi8(v, a_mask);

            (
                Self(_mm_cvtepi32_ps(r_i32)),
                Self(_mm_cvtepi32_ps(g_i32)),
                Self(_mm_cvtepi32_ps(b_i32)),
                Self(_mm_cvtepi32_ps(a_i32)),
            )
        }
    }

    /// Interleave 4 f32x4 channels and store as 4 RGBA u8 pixels.
    ///
    /// Input: (R, G, B, A) channel vectors with values that will be clamped to [0, 255].
    /// Output: 16 bytes = 4 RGBA pixels in interleaved format.
    #[inline]
    pub fn store_4_rgba_u8(r: Self, g: Self, b: Self, a: Self) -> [u8; 16] {
        unsafe {
            // Convert to i32 with rounding
            let ri = _mm_cvtps_epi32(r.0);
            let gi = _mm_cvtps_epi32(g.0);
            let bi = _mm_cvtps_epi32(b.0);
            let ai = _mm_cvtps_epi32(a.0);

            // Pack i32 to i16 (saturating)
            let rg = _mm_packs_epi32(ri, gi); // [R0,R1,R2,R3,G0,G1,G2,G3]
            let ba = _mm_packs_epi32(bi, ai); // [B0,B1,B2,B3,A0,A1,A2,A3]

            // Pack i16 to u8 (saturating)
            let rgba_packed = _mm_packus_epi16(rg, ba); // [R0-3,G0-3,B0-3,A0-3]

            // Shuffle to interleaved RGBA format
            let shuffle = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
            let result = _mm_shuffle_epi8(rgba_packed, shuffle);

            let mut out = [0u8; 16];
            _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, result);
            out
        }
    }

    // ========== Matrix Transpose ==========

    /// Transpose a 4x4 matrix represented as 4 row vectors.
    ///
    /// After transpose, `rows[i][j]` becomes `rows[j][i]`.
    #[inline]
    pub fn transpose_4x4(rows: &mut [Self; 4]) {
        unsafe {
            let t0 = _mm_unpacklo_ps(rows[0].0, rows[1].0);
            let t1 = _mm_unpackhi_ps(rows[0].0, rows[1].0);
            let t2 = _mm_unpacklo_ps(rows[2].0, rows[3].0);
            let t3 = _mm_unpackhi_ps(rows[2].0, rows[3].0);

            rows[0] = Self(_mm_movelh_ps(t0, t2));
            rows[1] = Self(_mm_movehl_ps(t2, t0));
            rows[2] = Self(_mm_movelh_ps(t1, t3));
            rows[3] = Self(_mm_movehl_ps(t3, t1));
        }
    }

    /// Transpose a 4x4 matrix, returning the transposed rows.
    #[inline]
    pub fn transpose_4x4_copy(rows: [Self; 4]) -> [Self; 4] {
        let mut result = rows;
        Self::transpose_4x4(&mut result);
        result
    }

    // ========== Bitcast ==========
    /// Reinterpret bits as `i32x4` (zero-cost).
    #[inline(always)]
    pub fn bitcast_i32x4(self) -> i32x4 {
        unsafe { core::mem::transmute(_mm_castps_si128(self.0)) }
    }

    /// Reinterpret bits as `&i32x4` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_ref_i32x4(&self) -> &i32x4 {
        unsafe { &*(self as *const Self as *const i32x4) }
    }

    /// Reinterpret bits as `&mut i32x4` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_mut_i32x4(&mut self) -> &mut i32x4 {
        unsafe { &mut *(self as *mut Self as *mut i32x4) }
    }
    /// Reinterpret bits as `u32x4` (zero-cost).
    #[inline(always)]
    pub fn bitcast_u32x4(self) -> u32x4 {
        unsafe { core::mem::transmute(_mm_castps_si128(self.0)) }
    }

    /// Reinterpret bits as `&u32x4` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_ref_u32x4(&self) -> &u32x4 {
        unsafe { &*(self as *const Self as *const u32x4) }
    }

    /// Reinterpret bits as `&mut u32x4` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_mut_u32x4(&mut self) -> &mut u32x4 {
        unsafe { &mut *(self as *mut Self as *mut u32x4) }
    }
}

#[cfg(target_arch = "x86_64")]
crate::impl_arithmetic_ops!(f32x4, _mm_add_ps, _mm_sub_ps, _mm_mul_ps, _mm_div_ps);
#[cfg(target_arch = "x86_64")]
crate::impl_float_assign_ops!(f32x4);
#[cfg(target_arch = "x86_64")]
crate::impl_neg!(f32x4, _mm_sub_ps, _mm_setzero_ps);
#[cfg(target_arch = "x86_64")]
crate::impl_bitwise_ops!(f32x4, __m128, _mm_and_ps, _mm_or_ps, _mm_xor_ps);
#[cfg(target_arch = "x86_64")]
crate::impl_index!(f32x4, f32, 4);

#[cfg(target_arch = "x86_64")]
impl From<[f32; 4]> for f32x4 {
    #[inline(always)]
    fn from(arr: [f32; 4]) -> Self {
        // SAFETY: [f32; 4] and __m128 have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}
#[cfg(target_arch = "x86_64")]
impl From<f32x4> for [f32; 4] {
    #[inline(always)]
    fn from(v: f32x4) -> Self {
        // SAFETY: __m128 and [f32; 4] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}

// Scalar broadcast operations for f32x4
// These allow `v + 2.0` instead of `v + f32x4::splat(token, 2.0)`

#[cfg(target_arch = "x86_64")]
impl Add<f32> for f32x4 {
    type Output = Self;
    /// Add a scalar to all lanes: `v + 2.0`
    ///
    /// Broadcasts the scalar to all lanes, then adds.
    #[inline(always)]
    fn add(self, rhs: f32) -> Self {
        self + Self(unsafe { _mm_set1_ps(rhs) })
    }
}

#[cfg(target_arch = "x86_64")]
impl Sub<f32> for f32x4 {
    type Output = Self;
    /// Subtract a scalar from all lanes: `v - 2.0`
    #[inline(always)]
    fn sub(self, rhs: f32) -> Self {
        self - Self(unsafe { _mm_set1_ps(rhs) })
    }
}

#[cfg(target_arch = "x86_64")]
impl Mul<f32> for f32x4 {
    type Output = Self;
    /// Multiply all lanes by a scalar: `v * 2.0`
    #[inline(always)]
    fn mul(self, rhs: f32) -> Self {
        self * Self(unsafe { _mm_set1_ps(rhs) })
    }
}

#[cfg(target_arch = "x86_64")]
impl Div<f32> for f32x4 {
    type Output = Self;
    /// Divide all lanes by a scalar: `v / 2.0`
    #[inline(always)]
    fn div(self, rhs: f32) -> Self {
        self / Self(unsafe { _mm_set1_ps(rhs) })
    }
}

// ============================================================================
// f64x2 - 2 x f64 (128-bit)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct f64x2(__m128d);

#[cfg(target_arch = "x86_64")]
impl f64x2 {
    pub const LANES: usize = 2;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::X64V3Token, data: &[f64; 2]) -> Self {
        Self(unsafe { _mm_loadu_pd(data.as_ptr()) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::X64V3Token, v: f64) -> Self {
        Self(unsafe { _mm_set1_pd(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::X64V3Token) -> Self {
        Self(unsafe { _mm_setzero_pd() })
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::X64V3Token, arr: [f64; 2]) -> Self {
        // SAFETY: [f64; 2] and __m128d have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [f64; 2]) {
        unsafe { _mm_storeu_pd(out.as_mut_ptr(), self.0) };
    }

    /// Convert to array
    #[inline(always)]
    pub fn to_array(self) -> [f64; 2] {
        let mut out = [0.0f64; 2];
        self.store(&mut out);
        out
    }

    /// Get reference to underlying array
    #[inline(always)]
    pub fn as_array(&self) -> &[f64; 2] {
        unsafe { &*(self as *const Self as *const [f64; 2]) }
    }

    /// Get mutable reference to underlying array
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut [f64; 2] {
        unsafe { &mut *(self as *mut Self as *mut [f64; 2]) }
    }

    /// Get raw intrinsic type
    #[inline(always)]
    pub fn raw(self) -> __m128d {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports the required SIMD features.
    /// Use token-gated constructors (`load`, `splat`, `zero`) for safe construction.
    #[inline(always)]
    pub unsafe fn from_raw(v: __m128d) -> Self {
        Self(v)
    }

    // ========== Token-gated bytemuck replacements ==========

    /// Reinterpret a slice of scalars as a slice of SIMD vectors (token-gated).
    ///
    /// Returns `None` if the slice length is not a multiple of 2, or
    /// if the slice is not properly aligned.
    ///
    /// This is a safe, token-gated replacement for `bytemuck::cast_slice`.
    #[inline(always)]
    pub fn cast_slice(_: archmage::X64V3Token, slice: &[f64]) -> Option<&[Self]> {
        if slice.len() % 2 != 0 {
            return None;
        }
        let ptr = slice.as_ptr();
        if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {
            return None;
        }
        let len = slice.len() / 2;
        // SAFETY: alignment and length checked, layout is compatible
        Some(unsafe { core::slice::from_raw_parts(ptr as *const Self, len) })
    }

    /// Reinterpret a mutable slice of scalars as a slice of SIMD vectors (token-gated).
    ///
    /// Returns `None` if the slice length is not a multiple of 2, or
    /// if the slice is not properly aligned.
    ///
    /// This is a safe, token-gated replacement for `bytemuck::cast_slice_mut`.
    #[inline(always)]
    pub fn cast_slice_mut(_: archmage::X64V3Token, slice: &mut [f64]) -> Option<&mut [Self]> {
        if slice.len() % 2 != 0 {
            return None;
        }
        let ptr = slice.as_mut_ptr();
        if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {
            return None;
        }
        let len = slice.len() / 2;
        // SAFETY: alignment and length checked, layout is compatible
        Some(unsafe { core::slice::from_raw_parts_mut(ptr as *mut Self, len) })
    }

    /// View this vector as a byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of`.
    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8; 16] {
        // SAFETY: Self is repr(transparent) over __m128d which is 16 bytes
        unsafe { &*(self as *const Self as *const [u8; 16]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 16] {
        // SAFETY: Self is repr(transparent) over __m128d which is 16 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 16]) }
    }

    /// Create from a byte array reference (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::X64V3Token, bytes: &[u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Create from an owned byte array (token-gated, zero-cost).
    ///
    /// This is a zero-cost transmute from an owned byte array.
    #[inline(always)]
    pub fn from_bytes_owned(_: archmage::X64V3Token, bytes: [u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(bytes) })
    }

    // ========== Implementation identification ==========

    /// Returns a string identifying this type's implementation.
    ///
    /// This is useful for verifying that the correct implementation is being used
    /// at compile time or at runtime (via `#[magetypes]` dispatch).
    #[inline(always)]
    pub const fn implementation_name() -> &'static str {
        "x86::v3::f64x2"
    }

    /// Element-wise minimum
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(unsafe { _mm_min_pd(self.0, other.0) })
    }
    /// Element-wise maximum
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(unsafe { _mm_max_pd(self.0, other.0) })
    }
    /// Clamp values between lo and hi
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }

    /// Square root
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        Self(unsafe { _mm_sqrt_pd(self.0) })
    }
    /// Absolute value
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(unsafe {
            let mask = _mm_castsi128_pd(_mm_set1_epi64x(0x7FFF_FFFF_FFFF_FFFFu64 as i64));
            _mm_and_pd(self.0, mask)
        })
    }
    /// Round toward negative infinity
    #[inline(always)]
    pub fn floor(self) -> Self {
        Self(unsafe { _mm_floor_pd(self.0) })
    }
    /// Round toward positive infinity
    #[inline(always)]
    pub fn ceil(self) -> Self {
        Self(unsafe { _mm_ceil_pd(self.0) })
    }
    /// Round to nearest integer
    #[inline(always)]
    pub fn round(self) -> Self {
        Self(unsafe { _mm_round_pd::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(self.0) })
    }
    /// Fused multiply-add: self * a + b
    #[inline(always)]
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        Self(unsafe { _mm_fmadd_pd(self.0, a.0, b.0) })
    }

    /// Fused multiply-sub: self * a - b
    #[inline(always)]
    pub fn mul_sub(self, a: Self, b: Self) -> Self {
        Self(unsafe { _mm_fmsub_pd(self.0, a.0, b.0) })
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
        Self(unsafe { _mm_cmpeq_pd(self.0, other.0) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if inequality, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(unsafe { _mm_cmpneq_pd(self.0, other.0) })
    }

    /// Lane-wise less-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(unsafe { _mm_cmplt_pd(self.0, other.0) })
    }

    /// Lane-wise less-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than-or-equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(unsafe { _mm_cmple_pd(self.0, other.0) })
    }

    /// Lane-wise greater-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than, all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe { _mm_cmpgt_pd(self.0, other.0) })
    }

    /// Lane-wise greater-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than-or-equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(unsafe { _mm_cmpge_pd(self.0, other.0) })
    }

    // ========== Blending/Selection ==========

    /// Select lanes from `if_true` where mask is all-1s, `if_false` where mask is all-0s.
    ///
    /// The mask should come from a comparison operation like `simd_lt()`.
    ///
    /// # Example
    /// ```ignore
    /// let a = f64x2::splat(token, 1.0);
    /// let b = f64x2::splat(token, 2.0);
    /// let mask = a.simd_lt(b);  // all true
    /// let result = f64x2::blend(mask, a, b);  // selects a
    /// ```
    #[inline(always)]
    pub fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self(unsafe { _mm_blendv_pd(if_false.0, if_true.0, mask.0) })
    }

    // ========== Horizontal Operations ==========

    /// Sum all lanes horizontally.
    ///
    /// Returns a scalar containing the sum of all 2 lanes.
    #[inline(always)]
    pub fn reduce_add(self) -> f64 {
        unsafe {
            let h = _mm_hadd_pd(self.0, self.0);
            _mm_cvtsd_f64(h)
        }
    }

    /// Find the minimum value across all lanes.
    #[inline(always)]
    pub fn reduce_min(self) -> f64 {
        unsafe {
            let shuf = _mm_shuffle_pd::<0b01>(self.0, self.0);
            let m = _mm_min_pd(self.0, shuf);
            _mm_cvtsd_f64(m)
        }
    }

    /// Find the maximum value across all lanes.
    #[inline(always)]
    pub fn reduce_max(self) -> f64 {
        unsafe {
            let shuf = _mm_shuffle_pd::<0b01>(self.0, self.0);
            let m = _mm_max_pd(self.0, shuf);
            _mm_cvtsd_f64(m)
        }
    }

    // ========== Type Conversions ==========

    /// Convert to signed 32-bit integers (2 lanes), rounding toward zero.
    ///
    /// Returns an `i32x4` where only the lower 2 lanes are valid.
    #[inline(always)]
    pub fn to_i32x4_low(self) -> i32x4 {
        i32x4(unsafe { _mm_cvttpd_epi32(self.0) })
    }

    // ========== Bitwise Unary Operations ==========

    /// Bitwise NOT (complement): flips all bits.
    #[inline(always)]
    pub fn not(self) -> Self {
        Self(unsafe {
            let ones = _mm_set1_epi64x(-1);
            let as_int = _mm_castpd_si128(self.0);
            _mm_castsi128_pd(_mm_xor_si128(as_int, ones))
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
            let x_bits = _mm_castpd_si128(self.0);
            let offset = _mm_set1_epi64x(OFFSET);
            let exp_bits = _mm_sub_epi64(x_bits, offset);
            let exp_arr_raw: [i64; 2] = core::mem::transmute(exp_bits);
            let exp_shifted = _mm_set_epi64x(exp_arr_raw[1] >> 52, exp_arr_raw[0] >> 52);

            let mantissa_bits = _mm_sub_epi64(x_bits, _mm_slli_epi64::<52>(exp_shifted));
            let mantissa = _mm_castsi128_pd(mantissa_bits);
            // Convert exponent to f64
            let exp_arr: [i64; 2] = core::mem::transmute(exp_shifted);
            let exp_f64: [f64; 2] = [exp_arr[0] as f64, exp_arr[1] as f64];
            let exp_val = _mm_loadu_pd(exp_f64.as_ptr());

            let one = _mm_set1_pd(1.0);
            let m = _mm_sub_pd(mantissa, one);

            // Horner's for numerator
            let yp = _mm_fmadd_pd(_mm_set1_pd(P2), m, _mm_set1_pd(P1));
            let yp = _mm_fmadd_pd(yp, m, _mm_set1_pd(P0));

            // Horner's for denominator
            let yq = _mm_fmadd_pd(_mm_set1_pd(Q2), m, _mm_set1_pd(Q1));
            let yq = _mm_fmadd_pd(yq, m, _mm_set1_pd(Q0));

            Self(_mm_add_pd(_mm_div_pd(yp, yq), exp_val))
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
            let x = _mm_max_pd(self.0, _mm_set1_pd(-1022.0));
            let x = _mm_min_pd(x, _mm_set1_pd(1022.0));

            let xi = _mm_floor_pd(x);
            let xf = _mm_sub_pd(x, xi);

            // Polynomial for 2^frac
            let poly = _mm_fmadd_pd(_mm_set1_pd(C4), xf, _mm_set1_pd(C3));
            let poly = _mm_fmadd_pd(poly, xf, _mm_set1_pd(C2));
            let poly = _mm_fmadd_pd(poly, xf, _mm_set1_pd(C1));
            let poly = _mm_fmadd_pd(poly, xf, _mm_set1_pd(C0));

            // Scale by 2^integer - extract, convert, scale
            let xi_arr: [f64; 2] = core::mem::transmute(xi);
            let scale_arr: [f64; 2] = [
                f64::from_bits(((xi_arr[0] as i64 + 1023) << 52) as u64),
                f64::from_bits(((xi_arr[1] as i64 + 1023) << 52) as u64),
            ];
            let scale = _mm_loadu_pd(scale_arr.as_ptr());

            Self(_mm_mul_pd(poly, scale))
        }
    }

    /// Low-precision natural logarithm.
    #[inline(always)]
    pub fn ln_lowp(self) -> Self {
        const LN2: f64 = core::f64::consts::LN_2;
        unsafe { Self(_mm_mul_pd(self.log2_lowp().0, _mm_set1_pd(LN2))) }
    }

    /// Low-precision natural exponential (e^x).
    #[inline(always)]
    pub fn exp_lowp(self) -> Self {
        const LOG2_E: f64 = core::f64::consts::LOG2_E;
        unsafe { Self(_mm_mul_pd(self.0, _mm_set1_pd(LOG2_E))).exp2_lowp() }
    }

    /// Low-precision base-10 logarithm.
    #[inline(always)]
    pub fn log10_lowp(self) -> Self {
        const LOG10_2: f64 = core::f64::consts::LOG10_2;
        unsafe { Self(_mm_mul_pd(self.log2_lowp().0, _mm_set1_pd(LOG10_2))) }
    }

    /// Low-precision power function (self^n).
    #[inline(always)]
    pub fn pow_lowp(self, n: f64) -> Self {
        unsafe { Self(_mm_mul_pd(self.log2_lowp().0, _mm_set1_pd(n))).exp2_lowp() }
    }

    // ========== Bitcast ==========
    /// Reinterpret bits as `i64x2` (zero-cost).
    #[inline(always)]
    pub fn bitcast_i64x2(self) -> i64x2 {
        unsafe { core::mem::transmute(_mm_castpd_si128(self.0)) }
    }

    /// Reinterpret bits as `&i64x2` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_ref_i64x2(&self) -> &i64x2 {
        unsafe { &*(self as *const Self as *const i64x2) }
    }

    /// Reinterpret bits as `&mut i64x2` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_mut_i64x2(&mut self) -> &mut i64x2 {
        unsafe { &mut *(self as *mut Self as *mut i64x2) }
    }
    /// Reinterpret bits as `u64x2` (zero-cost).
    #[inline(always)]
    pub fn bitcast_u64x2(self) -> u64x2 {
        unsafe { core::mem::transmute(_mm_castpd_si128(self.0)) }
    }

    /// Reinterpret bits as `&u64x2` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_ref_u64x2(&self) -> &u64x2 {
        unsafe { &*(self as *const Self as *const u64x2) }
    }

    /// Reinterpret bits as `&mut u64x2` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_mut_u64x2(&mut self) -> &mut u64x2 {
        unsafe { &mut *(self as *mut Self as *mut u64x2) }
    }
}

#[cfg(target_arch = "x86_64")]
crate::impl_arithmetic_ops!(f64x2, _mm_add_pd, _mm_sub_pd, _mm_mul_pd, _mm_div_pd);
#[cfg(target_arch = "x86_64")]
crate::impl_float_assign_ops!(f64x2);
#[cfg(target_arch = "x86_64")]
crate::impl_neg!(f64x2, _mm_sub_pd, _mm_setzero_pd);
#[cfg(target_arch = "x86_64")]
crate::impl_bitwise_ops!(f64x2, __m128d, _mm_and_pd, _mm_or_pd, _mm_xor_pd);
#[cfg(target_arch = "x86_64")]
crate::impl_index!(f64x2, f64, 2);

#[cfg(target_arch = "x86_64")]
impl From<[f64; 2]> for f64x2 {
    #[inline(always)]
    fn from(arr: [f64; 2]) -> Self {
        // SAFETY: [f64; 2] and __m128d have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}
#[cfg(target_arch = "x86_64")]
impl From<f64x2> for [f64; 2] {
    #[inline(always)]
    fn from(v: f64x2) -> Self {
        // SAFETY: __m128d and [f64; 2] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}

// Scalar broadcast operations for f64x2
// These allow `v + 2.0` instead of `v + f64x2::splat(token, 2.0)`

#[cfg(target_arch = "x86_64")]
impl Add<f64> for f64x2 {
    type Output = Self;
    /// Add a scalar to all lanes: `v + 2.0`
    ///
    /// Broadcasts the scalar to all lanes, then adds.
    #[inline(always)]
    fn add(self, rhs: f64) -> Self {
        self + Self(unsafe { _mm_set1_pd(rhs) })
    }
}

#[cfg(target_arch = "x86_64")]
impl Sub<f64> for f64x2 {
    type Output = Self;
    /// Subtract a scalar from all lanes: `v - 2.0`
    #[inline(always)]
    fn sub(self, rhs: f64) -> Self {
        self - Self(unsafe { _mm_set1_pd(rhs) })
    }
}

#[cfg(target_arch = "x86_64")]
impl Mul<f64> for f64x2 {
    type Output = Self;
    /// Multiply all lanes by a scalar: `v * 2.0`
    #[inline(always)]
    fn mul(self, rhs: f64) -> Self {
        self * Self(unsafe { _mm_set1_pd(rhs) })
    }
}

#[cfg(target_arch = "x86_64")]
impl Div<f64> for f64x2 {
    type Output = Self;
    /// Divide all lanes by a scalar: `v / 2.0`
    #[inline(always)]
    fn div(self, rhs: f64) -> Self {
        self / Self(unsafe { _mm_set1_pd(rhs) })
    }
}

// ============================================================================
// i8x16 - 16 x i8 (128-bit)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct i8x16(__m128i);

#[cfg(target_arch = "x86_64")]
impl i8x16 {
    pub const LANES: usize = 16;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::X64V3Token, data: &[i8; 16]) -> Self {
        Self(unsafe { _mm_loadu_si128(data.as_ptr() as *const __m128i) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::X64V3Token, v: i8) -> Self {
        Self(unsafe { _mm_set1_epi8(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::X64V3Token) -> Self {
        Self(unsafe { _mm_setzero_si128() })
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::X64V3Token, arr: [i8; 16]) -> Self {
        // SAFETY: [i8; 16] and __m128i have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [i8; 16]) {
        unsafe { _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, self.0) };
    }

    /// Convert to array
    #[inline(always)]
    pub fn to_array(self) -> [i8; 16] {
        let mut out = [0i8; 16];
        self.store(&mut out);
        out
    }

    /// Get reference to underlying array
    #[inline(always)]
    pub fn as_array(&self) -> &[i8; 16] {
        unsafe { &*(self as *const Self as *const [i8; 16]) }
    }

    /// Get mutable reference to underlying array
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut [i8; 16] {
        unsafe { &mut *(self as *mut Self as *mut [i8; 16]) }
    }

    /// Get raw intrinsic type
    #[inline(always)]
    pub fn raw(self) -> __m128i {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports the required SIMD features.
    /// Use token-gated constructors (`load`, `splat`, `zero`) for safe construction.
    #[inline(always)]
    pub unsafe fn from_raw(v: __m128i) -> Self {
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
    pub fn cast_slice(_: archmage::X64V3Token, slice: &[i8]) -> Option<&[Self]> {
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
    pub fn cast_slice_mut(_: archmage::X64V3Token, slice: &mut [i8]) -> Option<&mut [Self]> {
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
    pub fn as_bytes(&self) -> &[u8; 16] {
        // SAFETY: Self is repr(transparent) over __m128i which is 16 bytes
        unsafe { &*(self as *const Self as *const [u8; 16]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 16] {
        // SAFETY: Self is repr(transparent) over __m128i which is 16 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 16]) }
    }

    /// Create from a byte array reference (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::X64V3Token, bytes: &[u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Create from an owned byte array (token-gated, zero-cost).
    ///
    /// This is a zero-cost transmute from an owned byte array.
    #[inline(always)]
    pub fn from_bytes_owned(_: archmage::X64V3Token, bytes: [u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(bytes) })
    }

    // ========== Implementation identification ==========

    /// Returns a string identifying this type's implementation.
    ///
    /// This is useful for verifying that the correct implementation is being used
    /// at compile time or at runtime (via `#[magetypes]` dispatch).
    #[inline(always)]
    pub const fn implementation_name() -> &'static str {
        "x86::v3::i8x16"
    }

    /// Element-wise minimum
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(unsafe { _mm_min_epi8(self.0, other.0) })
    }
    /// Element-wise maximum
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(unsafe { _mm_max_epi8(self.0, other.0) })
    }
    /// Clamp values between lo and hi
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }

    /// Absolute value
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(unsafe { _mm_abs_epi8(self.0) })
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
        Self(unsafe { _mm_cmpeq_epi8(self.0, other.0) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if inequality, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(unsafe {
            let eq = _mm_cmpeq_epi8(self.0, other.0);
            let ones = _mm_set1_epi8(-1);
            _mm_xor_si128(eq, ones)
        })
    }

    /// Lane-wise greater-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than, all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe { _mm_cmpgt_epi8(self.0, other.0) })
    }

    /// Lane-wise less-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(unsafe { _mm_cmpgt_epi8(other.0, self.0) })
    }

    /// Lane-wise greater-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than-or-equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(unsafe {
            let lt = _mm_cmpgt_epi8(other.0, self.0);
            let ones = _mm_set1_epi8(-1);
            _mm_xor_si128(lt, ones)
        })
    }

    /// Lane-wise less-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than-or-equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(unsafe {
            let gt = _mm_cmpgt_epi8(self.0, other.0);
            let ones = _mm_set1_epi8(-1);
            _mm_xor_si128(gt, ones)
        })
    }

    // ========== Blending/Selection ==========

    /// Select lanes from `if_true` where mask is all-1s, `if_false` where mask is all-0s.
    ///
    /// The mask should come from a comparison operation like `simd_lt()`.
    ///
    /// # Example
    /// ```ignore
    /// let a = i8x16::splat(token, 1.0);
    /// let b = i8x16::splat(token, 2.0);
    /// let mask = a.simd_lt(b);  // all true
    /// let result = i8x16::blend(mask, a, b);  // selects a
    /// ```
    #[inline(always)]
    pub fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self(unsafe { _mm_blendv_epi8(if_false.0, if_true.0, mask.0) })
    }

    // ========== Boolean Reductions ==========

    /// Returns true if all lanes are non-zero (truthy).
    ///
    /// Typically used with comparison results where true lanes are all-1s.
    #[inline(always)]
    pub fn all_true(self) -> bool {
        unsafe { _mm_movemask_epi8(self.0) == 0xFFFF_u32 as i32 }
    }

    /// Returns true if any lane is non-zero (truthy).
    #[inline(always)]
    pub fn any_true(self) -> bool {
        unsafe { _mm_movemask_epi8(self.0) != 0 }
    }

    /// Extract the high bit of each lane as a bitmask.
    ///
    /// Returns a u32 where bit N corresponds to the sign bit of lane N.
    #[inline(always)]
    pub fn bitmask(self) -> u32 {
        unsafe { _mm_movemask_epi8(self.0) as u32 }
    }

    // ========== Horizontal Operations ==========

    /// Sum all lanes horizontally.
    ///
    /// Returns a scalar containing the sum of all 16 lanes.
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
            let ones = _mm_set1_epi8(-1);
            _mm_xor_si128(self.0, ones)
        })
    }
    // ========== Shift Operations (polyfill) ==========
    // x86 has no native 8-bit shift-by-immediate; uses 16-bit shift + mask.

    /// Shift each byte left by `N` bits.
    ///
    /// Bits shifted out are lost; zeros are shifted in.
    /// Implemented via 16-bit shift + byte mask (no native 8-bit shift in x86).
    #[inline(always)]
    pub fn shl<const N: i32>(self) -> Self {
        unsafe {
            let shifted = _mm_slli_epi16::<N>(self.0);
            let mask = _mm_set1_epi8((0xFFu8.wrapping_shl(N as u32)) as i8);
            Self(_mm_and_si128(shifted, mask))
        }
    }

    /// Shift each byte right by `N` bits (logical/unsigned shift).
    ///
    /// Bits shifted out are lost; zeros are shifted in.
    /// Implemented via 16-bit shift + byte mask (no native 8-bit shift in x86).
    #[inline(always)]
    pub fn shr<const N: i32>(self) -> Self {
        unsafe {
            let shifted = _mm_srli_epi16::<N>(self.0);
            let mask = _mm_set1_epi8((0xFFu8.wrapping_shr(N as u32)) as i8);
            Self(_mm_and_si128(shifted, mask))
        }
    }

    /// Arithmetic shift right by `N` bits (sign-extending).
    ///
    /// The sign bit is replicated into the vacated positions.
    /// Implemented via logical shift + sign bit fill (no native 8-bit shift in x86).
    #[inline(always)]
    pub fn shr_arithmetic<const N: i32>(self) -> Self {
        unsafe {
            let shifted = _mm_srli_epi16::<N>(self.0);
            let byte_mask = _mm_set1_epi8((0xFFu8.wrapping_shr(N as u32)) as i8);
            let logical = _mm_and_si128(shifted, byte_mask);
            let zero = _mm_setzero_si128();
            let sign = _mm_cmpgt_epi8(zero, self.0);
            let fill = _mm_set1_epi8((0xFFu8.wrapping_shl(8u32.wrapping_sub(N as u32))) as i8);
            Self(_mm_or_si128(logical, _mm_and_si128(sign, fill)))
        }
    }

    // ========== Bitcast ==========
    /// Reinterpret bits as `u8x16` (zero-cost).
    #[inline(always)]
    pub fn bitcast_u8x16(self) -> u8x16 {
        unsafe { core::mem::transmute(self) }
    }

    /// Reinterpret bits as `&u8x16` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_ref_u8x16(&self) -> &u8x16 {
        unsafe { &*(self as *const Self as *const u8x16) }
    }

    /// Reinterpret bits as `&mut u8x16` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_mut_u8x16(&mut self) -> &mut u8x16 {
        unsafe { &mut *(self as *mut Self as *mut u8x16) }
    }
}

#[cfg(target_arch = "x86_64")]
crate::impl_int_arithmetic_ops!(i8x16, _mm_add_epi8, _mm_sub_epi8);
#[cfg(target_arch = "x86_64")]
crate::impl_assign_ops!(i8x16);
#[cfg(target_arch = "x86_64")]
crate::impl_bitwise_ops!(i8x16, __m128i, _mm_and_si128, _mm_or_si128, _mm_xor_si128);
#[cfg(target_arch = "x86_64")]
crate::impl_index!(i8x16, i8, 16);

#[cfg(target_arch = "x86_64")]
impl From<[i8; 16]> for i8x16 {
    #[inline(always)]
    fn from(arr: [i8; 16]) -> Self {
        // SAFETY: [i8; 16] and __m128i have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}
#[cfg(target_arch = "x86_64")]
impl From<i8x16> for [i8; 16] {
    #[inline(always)]
    fn from(v: i8x16) -> Self {
        // SAFETY: __m128i and [i8; 16] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}

// Scalar broadcast operations for i8x16
// These allow `v + 2.0` instead of `v + i8x16::splat(token, 2.0)`

#[cfg(target_arch = "x86_64")]
impl Add<i8> for i8x16 {
    type Output = Self;
    /// Add a scalar to all lanes.
    #[inline(always)]
    fn add(self, rhs: i8) -> Self {
        self + Self(unsafe { _mm_set1_epi8(rhs) })
    }
}

#[cfg(target_arch = "x86_64")]
impl Sub<i8> for i8x16 {
    type Output = Self;
    /// Subtract a scalar from all lanes.
    #[inline(always)]
    fn sub(self, rhs: i8) -> Self {
        self - Self(unsafe { _mm_set1_epi8(rhs) })
    }
}

// ============================================================================
// u8x16 - 16 x u8 (128-bit)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct u8x16(__m128i);

#[cfg(target_arch = "x86_64")]
impl u8x16 {
    pub const LANES: usize = 16;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::X64V3Token, data: &[u8; 16]) -> Self {
        Self(unsafe { _mm_loadu_si128(data.as_ptr() as *const __m128i) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::X64V3Token, v: u8) -> Self {
        Self(unsafe { _mm_set1_epi8(v as i8) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::X64V3Token) -> Self {
        Self(unsafe { _mm_setzero_si128() })
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::X64V3Token, arr: [u8; 16]) -> Self {
        // SAFETY: [u8; 16] and __m128i have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [u8; 16]) {
        unsafe { _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, self.0) };
    }

    /// Convert to array
    #[inline(always)]
    pub fn to_array(self) -> [u8; 16] {
        let mut out = [0u8; 16];
        self.store(&mut out);
        out
    }

    /// Get reference to underlying array
    #[inline(always)]
    pub fn as_array(&self) -> &[u8; 16] {
        unsafe { &*(self as *const Self as *const [u8; 16]) }
    }

    /// Get mutable reference to underlying array
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut [u8; 16] {
        unsafe { &mut *(self as *mut Self as *mut [u8; 16]) }
    }

    /// Get raw intrinsic type
    #[inline(always)]
    pub fn raw(self) -> __m128i {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports the required SIMD features.
    /// Use token-gated constructors (`load`, `splat`, `zero`) for safe construction.
    #[inline(always)]
    pub unsafe fn from_raw(v: __m128i) -> Self {
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
    pub fn cast_slice(_: archmage::X64V3Token, slice: &[u8]) -> Option<&[Self]> {
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
    pub fn cast_slice_mut(_: archmage::X64V3Token, slice: &mut [u8]) -> Option<&mut [Self]> {
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
    pub fn as_bytes(&self) -> &[u8; 16] {
        // SAFETY: Self is repr(transparent) over __m128i which is 16 bytes
        unsafe { &*(self as *const Self as *const [u8; 16]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 16] {
        // SAFETY: Self is repr(transparent) over __m128i which is 16 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 16]) }
    }

    /// Create from a byte array reference (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::X64V3Token, bytes: &[u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Create from an owned byte array (token-gated, zero-cost).
    ///
    /// This is a zero-cost transmute from an owned byte array.
    #[inline(always)]
    pub fn from_bytes_owned(_: archmage::X64V3Token, bytes: [u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(bytes) })
    }

    // ========== Implementation identification ==========

    /// Returns a string identifying this type's implementation.
    ///
    /// This is useful for verifying that the correct implementation is being used
    /// at compile time or at runtime (via `#[magetypes]` dispatch).
    #[inline(always)]
    pub const fn implementation_name() -> &'static str {
        "x86::v3::u8x16"
    }

    /// Element-wise minimum
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(unsafe { _mm_min_epu8(self.0, other.0) })
    }
    /// Element-wise maximum
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(unsafe { _mm_max_epu8(self.0, other.0) })
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
        Self(unsafe { _mm_cmpeq_epi8(self.0, other.0) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if inequality, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(unsafe {
            let eq = _mm_cmpeq_epi8(self.0, other.0);
            let ones = _mm_set1_epi8(-1);
            _mm_xor_si128(eq, ones)
        })
    }

    /// Lane-wise greater-than (unsigned) comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than (unsigned), all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe {
            // Flip sign bit to convert unsigned to signed comparison
            let bias = _mm_set1_epi8(0x80u8 as i8);
            let a = _mm_xor_si128(self.0, bias);
            let b = _mm_xor_si128(other.0, bias);
            _mm_cmpgt_epi8(a, b)
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
            let ones = _mm_set1_epi8(-1);
            _mm_xor_si128(lt.0, ones)
        })
    }

    /// Lane-wise less-than-or-equal (unsigned) comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than-or-equal (unsigned), all-0s otherwise.
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(unsafe {
            let gt = self.simd_gt(other);
            let ones = _mm_set1_epi8(-1);
            _mm_xor_si128(gt.0, ones)
        })
    }

    // ========== Blending/Selection ==========

    /// Select lanes from `if_true` where mask is all-1s, `if_false` where mask is all-0s.
    ///
    /// The mask should come from a comparison operation like `simd_lt()`.
    ///
    /// # Example
    /// ```ignore
    /// let a = u8x16::splat(token, 1.0);
    /// let b = u8x16::splat(token, 2.0);
    /// let mask = a.simd_lt(b);  // all true
    /// let result = u8x16::blend(mask, a, b);  // selects a
    /// ```
    #[inline(always)]
    pub fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self(unsafe { _mm_blendv_epi8(if_false.0, if_true.0, mask.0) })
    }

    // ========== Boolean Reductions ==========

    /// Returns true if all lanes are non-zero (truthy).
    ///
    /// Typically used with comparison results where true lanes are all-1s.
    #[inline(always)]
    pub fn all_true(self) -> bool {
        unsafe { _mm_movemask_epi8(self.0) == 0xFFFF_u32 as i32 }
    }

    /// Returns true if any lane is non-zero (truthy).
    #[inline(always)]
    pub fn any_true(self) -> bool {
        unsafe { _mm_movemask_epi8(self.0) != 0 }
    }

    /// Extract the high bit of each lane as a bitmask.
    ///
    /// Returns a u32 where bit N corresponds to the sign bit of lane N.
    #[inline(always)]
    pub fn bitmask(self) -> u32 {
        unsafe { _mm_movemask_epi8(self.0) as u32 }
    }

    // ========== Horizontal Operations ==========

    /// Sum all lanes horizontally.
    ///
    /// Returns a scalar containing the sum of all 16 lanes.
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
            let ones = _mm_set1_epi8(-1);
            _mm_xor_si128(self.0, ones)
        })
    }
    // ========== Shift Operations (polyfill) ==========
    // x86 has no native 8-bit shift-by-immediate; uses 16-bit shift + mask.

    /// Shift each byte left by `N` bits.
    ///
    /// Bits shifted out are lost; zeros are shifted in.
    /// Implemented via 16-bit shift + byte mask (no native 8-bit shift in x86).
    #[inline(always)]
    pub fn shl<const N: i32>(self) -> Self {
        unsafe {
            let shifted = _mm_slli_epi16::<N>(self.0);
            let mask = _mm_set1_epi8((0xFFu8.wrapping_shl(N as u32)) as i8);
            Self(_mm_and_si128(shifted, mask))
        }
    }

    /// Shift each byte right by `N` bits (logical/unsigned shift).
    ///
    /// Bits shifted out are lost; zeros are shifted in.
    /// Implemented via 16-bit shift + byte mask (no native 8-bit shift in x86).
    #[inline(always)]
    pub fn shr<const N: i32>(self) -> Self {
        unsafe {
            let shifted = _mm_srli_epi16::<N>(self.0);
            let mask = _mm_set1_epi8((0xFFu8.wrapping_shr(N as u32)) as i8);
            Self(_mm_and_si128(shifted, mask))
        }
    }

    // ========== Extend/Widen Operations ==========

    /// Zero-extend low 8 u8 values to i16x8.
    ///
    /// Takes the lower 8 bytes and zero-extends each to 16 bits.
    #[inline(always)]
    pub fn extend_lo_i16(self) -> i16x8 {
        i16x8(unsafe { _mm_cvtepu8_epi16(self.0) })
    }

    /// Zero-extend high 8 u8 values to i16x8.
    ///
    /// Takes the upper 8 bytes and zero-extends each to 16 bits.
    #[inline(always)]
    pub fn extend_hi_i16(self) -> i16x8 {
        i16x8(unsafe {
            // Shift right by 8 bytes to get high half into low position
            let hi = _mm_srli_si128::<8>(self.0);
            _mm_cvtepu8_epi16(hi)
        })
    }

    /// Zero-extend all 16 u8 values to two i16x8 vectors.
    ///
    /// Returns (low 8 as i16x8, high 8 as i16x8).
    #[inline(always)]
    pub fn extend_i16(self) -> (i16x8, i16x8) {
        (self.extend_lo_i16(), self.extend_hi_i16())
    }

    /// Zero-extend low 4 u8 values to i32x4.
    #[inline(always)]
    pub fn extend_lo_i32(self) -> i32x4 {
        i32x4(unsafe { _mm_cvtepu8_epi32(self.0) })
    }

    /// Zero-extend low 4 u8 values to f32x4.
    #[inline(always)]
    pub fn extend_lo_f32(self) -> f32x4 {
        f32x4(unsafe {
            let i32s = _mm_cvtepu8_epi32(self.0);
            _mm_cvtepi32_ps(i32s)
        })
    }

    // ========== Bitcast ==========
    /// Reinterpret bits as `i8x16` (zero-cost).
    #[inline(always)]
    pub fn bitcast_i8x16(self) -> i8x16 {
        unsafe { core::mem::transmute(self) }
    }

    /// Reinterpret bits as `&i8x16` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_ref_i8x16(&self) -> &i8x16 {
        unsafe { &*(self as *const Self as *const i8x16) }
    }

    /// Reinterpret bits as `&mut i8x16` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_mut_i8x16(&mut self) -> &mut i8x16 {
        unsafe { &mut *(self as *mut Self as *mut i8x16) }
    }
}

#[cfg(target_arch = "x86_64")]
crate::impl_int_arithmetic_ops!(u8x16, _mm_add_epi8, _mm_sub_epi8);
#[cfg(target_arch = "x86_64")]
crate::impl_assign_ops!(u8x16);
#[cfg(target_arch = "x86_64")]
crate::impl_bitwise_ops!(u8x16, __m128i, _mm_and_si128, _mm_or_si128, _mm_xor_si128);
#[cfg(target_arch = "x86_64")]
crate::impl_index!(u8x16, u8, 16);

#[cfg(target_arch = "x86_64")]
impl From<[u8; 16]> for u8x16 {
    #[inline(always)]
    fn from(arr: [u8; 16]) -> Self {
        // SAFETY: [u8; 16] and __m128i have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}
#[cfg(target_arch = "x86_64")]
impl From<u8x16> for [u8; 16] {
    #[inline(always)]
    fn from(v: u8x16) -> Self {
        // SAFETY: __m128i and [u8; 16] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}

// Scalar broadcast operations for u8x16
// These allow `v + 2.0` instead of `v + u8x16::splat(token, 2.0)`

#[cfg(target_arch = "x86_64")]
impl Add<u8> for u8x16 {
    type Output = Self;
    /// Add a scalar to all lanes.
    #[inline(always)]
    fn add(self, rhs: u8) -> Self {
        self + Self(unsafe { _mm_set1_epi8(rhs as i8) })
    }
}

#[cfg(target_arch = "x86_64")]
impl Sub<u8> for u8x16 {
    type Output = Self;
    /// Subtract a scalar from all lanes.
    #[inline(always)]
    fn sub(self, rhs: u8) -> Self {
        self - Self(unsafe { _mm_set1_epi8(rhs as i8) })
    }
}

// ============================================================================
// i16x8 - 8 x i16 (128-bit)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct i16x8(__m128i);

#[cfg(target_arch = "x86_64")]
impl i16x8 {
    pub const LANES: usize = 8;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::X64V3Token, data: &[i16; 8]) -> Self {
        Self(unsafe { _mm_loadu_si128(data.as_ptr() as *const __m128i) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::X64V3Token, v: i16) -> Self {
        Self(unsafe { _mm_set1_epi16(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::X64V3Token) -> Self {
        Self(unsafe { _mm_setzero_si128() })
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::X64V3Token, arr: [i16; 8]) -> Self {
        // SAFETY: [i16; 8] and __m128i have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [i16; 8]) {
        unsafe { _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, self.0) };
    }

    /// Convert to array
    #[inline(always)]
    pub fn to_array(self) -> [i16; 8] {
        let mut out = [0i16; 8];
        self.store(&mut out);
        out
    }

    /// Get reference to underlying array
    #[inline(always)]
    pub fn as_array(&self) -> &[i16; 8] {
        unsafe { &*(self as *const Self as *const [i16; 8]) }
    }

    /// Get mutable reference to underlying array
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut [i16; 8] {
        unsafe { &mut *(self as *mut Self as *mut [i16; 8]) }
    }

    /// Get raw intrinsic type
    #[inline(always)]
    pub fn raw(self) -> __m128i {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports the required SIMD features.
    /// Use token-gated constructors (`load`, `splat`, `zero`) for safe construction.
    #[inline(always)]
    pub unsafe fn from_raw(v: __m128i) -> Self {
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
    pub fn cast_slice(_: archmage::X64V3Token, slice: &[i16]) -> Option<&[Self]> {
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
    pub fn cast_slice_mut(_: archmage::X64V3Token, slice: &mut [i16]) -> Option<&mut [Self]> {
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
    pub fn as_bytes(&self) -> &[u8; 16] {
        // SAFETY: Self is repr(transparent) over __m128i which is 16 bytes
        unsafe { &*(self as *const Self as *const [u8; 16]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 16] {
        // SAFETY: Self is repr(transparent) over __m128i which is 16 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 16]) }
    }

    /// Create from a byte array reference (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::X64V3Token, bytes: &[u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Create from an owned byte array (token-gated, zero-cost).
    ///
    /// This is a zero-cost transmute from an owned byte array.
    #[inline(always)]
    pub fn from_bytes_owned(_: archmage::X64V3Token, bytes: [u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(bytes) })
    }

    // ========== Implementation identification ==========

    /// Returns a string identifying this type's implementation.
    ///
    /// This is useful for verifying that the correct implementation is being used
    /// at compile time or at runtime (via `#[magetypes]` dispatch).
    #[inline(always)]
    pub const fn implementation_name() -> &'static str {
        "x86::v3::i16x8"
    }

    /// Element-wise minimum
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(unsafe { _mm_min_epi16(self.0, other.0) })
    }
    /// Element-wise maximum
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(unsafe { _mm_max_epi16(self.0, other.0) })
    }
    /// Clamp values between lo and hi
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }

    /// Absolute value
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(unsafe { _mm_abs_epi16(self.0) })
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
        Self(unsafe { _mm_cmpeq_epi16(self.0, other.0) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if inequality, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(unsafe {
            let eq = _mm_cmpeq_epi16(self.0, other.0);
            let ones = _mm_set1_epi16(-1);
            _mm_xor_si128(eq, ones)
        })
    }

    /// Lane-wise greater-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than, all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe { _mm_cmpgt_epi16(self.0, other.0) })
    }

    /// Lane-wise less-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(unsafe { _mm_cmpgt_epi16(other.0, self.0) })
    }

    /// Lane-wise greater-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than-or-equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(unsafe {
            let lt = _mm_cmpgt_epi16(other.0, self.0);
            let ones = _mm_set1_epi16(-1);
            _mm_xor_si128(lt, ones)
        })
    }

    /// Lane-wise less-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than-or-equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(unsafe {
            let gt = _mm_cmpgt_epi16(self.0, other.0);
            let ones = _mm_set1_epi16(-1);
            _mm_xor_si128(gt, ones)
        })
    }

    // ========== Blending/Selection ==========

    /// Select lanes from `if_true` where mask is all-1s, `if_false` where mask is all-0s.
    ///
    /// The mask should come from a comparison operation like `simd_lt()`.
    ///
    /// # Example
    /// ```ignore
    /// let a = i16x8::splat(token, 1.0);
    /// let b = i16x8::splat(token, 2.0);
    /// let mask = a.simd_lt(b);  // all true
    /// let result = i16x8::blend(mask, a, b);  // selects a
    /// ```
    #[inline(always)]
    pub fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self(unsafe { _mm_blendv_epi8(if_false.0, if_true.0, mask.0) })
    }

    // ========== Boolean Reductions ==========

    /// Returns true if all lanes are non-zero (truthy).
    ///
    /// Typically used with comparison results where true lanes are all-1s.
    #[inline(always)]
    pub fn all_true(self) -> bool {
        unsafe { _mm_movemask_epi8(self.0) == 0xFFFF_u32 as i32 }
    }

    /// Returns true if any lane is non-zero (truthy).
    #[inline(always)]
    pub fn any_true(self) -> bool {
        unsafe { _mm_movemask_epi8(self.0) != 0 }
    }

    /// Extract the high bit of each lane as a bitmask.
    ///
    /// Returns a u32 where bit N corresponds to the sign bit of lane N.
    #[inline(always)]
    pub fn bitmask(self) -> u32 {
        unsafe {
            // Shift right to get sign bit in LSB, pack to bytes
            let shifted = _mm_srai_epi16::<15>(self.0);
            let packed = _mm_packs_epi16(shifted, shifted);
            (_mm_movemask_epi8(packed) & 0xFF) as u32
        }
    }

    // ========== Horizontal Operations ==========

    /// Sum all lanes horizontally.
    ///
    /// Returns a scalar containing the sum of all 8 lanes.
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
            let ones = _mm_set1_epi16(-1);
            _mm_xor_si128(self.0, ones)
        })
    }
    // ========== Shift Operations ==========

    /// Shift each lane left by `N` bits.
    ///
    /// Bits shifted out are lost; zeros are shifted in.
    #[inline(always)]
    pub fn shl<const N: i32>(self) -> Self {
        Self(unsafe { _mm_slli_epi16::<N>(self.0) })
    }

    /// Shift each lane right by `N` bits (logical/unsigned shift).
    ///
    /// Bits shifted out are lost; zeros are shifted in.
    #[inline(always)]
    pub fn shr<const N: i32>(self) -> Self {
        Self(unsafe { _mm_srli_epi16::<N>(self.0) })
    }

    /// Arithmetic shift right by `N` bits (sign-extending).
    ///
    /// The sign bit is replicated into the vacated positions.
    #[inline(always)]
    pub fn shr_arithmetic<const N: i32>(self) -> Self {
        Self(unsafe { _mm_srai_epi16::<N>(self.0) })
    }

    // ========== Extend/Widen Operations ==========

    /// Sign-extend low 4 i16 values to i32x4.
    #[inline(always)]
    pub fn extend_lo_i32(self) -> i32x4 {
        i32x4(unsafe { _mm_cvtepi16_epi32(self.0) })
    }

    /// Sign-extend high 4 i16 values to i32x4.
    #[inline(always)]
    pub fn extend_hi_i32(self) -> i32x4 {
        i32x4(unsafe {
            let hi = _mm_srli_si128::<8>(self.0);
            _mm_cvtepi16_epi32(hi)
        })
    }

    /// Sign-extend all 8 i16 values to two i32x4 vectors.
    #[inline(always)]
    pub fn extend_i32(self) -> (i32x4, i32x4) {
        (self.extend_lo_i32(), self.extend_hi_i32())
    }

    /// Sign-extend low 4 i16 values to f32x4.
    #[inline(always)]
    pub fn extend_lo_f32(self) -> f32x4 {
        f32x4(unsafe {
            let i32s = _mm_cvtepi16_epi32(self.0);
            _mm_cvtepi32_ps(i32s)
        })
    }

    // ========== Pack/Narrow Operations ==========

    /// Pack two i16x8 vectors to u8x16 with unsigned saturation.
    ///
    /// Values below 0 become 0, values above 255 become 255.
    /// `self` provides low 8 bytes, `other` provides high 8 bytes.
    #[inline(always)]
    pub fn pack_u8(self, other: Self) -> u8x16 {
        u8x16(unsafe { _mm_packus_epi16(self.0, other.0) })
    }

    /// Pack two i16x8 vectors to i8x16 with signed saturation.
    ///
    /// Values are clamped to [-128, 127].
    #[inline(always)]
    pub fn pack_i8(self, other: Self) -> i8x16 {
        i8x16(unsafe { _mm_packs_epi16(self.0, other.0) })
    }

    // ========== Bitcast ==========
    /// Reinterpret bits as `u16x8` (zero-cost).
    #[inline(always)]
    pub fn bitcast_u16x8(self) -> u16x8 {
        unsafe { core::mem::transmute(self) }
    }

    /// Reinterpret bits as `&u16x8` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_ref_u16x8(&self) -> &u16x8 {
        unsafe { &*(self as *const Self as *const u16x8) }
    }

    /// Reinterpret bits as `&mut u16x8` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_mut_u16x8(&mut self) -> &mut u16x8 {
        unsafe { &mut *(self as *mut Self as *mut u16x8) }
    }
}

#[cfg(target_arch = "x86_64")]
crate::impl_int_arithmetic_ops!(i16x8, _mm_add_epi16, _mm_sub_epi16);
#[cfg(target_arch = "x86_64")]
crate::impl_int_mul_op!(i16x8, _mm_mullo_epi16);
#[cfg(target_arch = "x86_64")]
crate::impl_assign_ops!(i16x8);
#[cfg(target_arch = "x86_64")]
crate::impl_bitwise_ops!(i16x8, __m128i, _mm_and_si128, _mm_or_si128, _mm_xor_si128);
#[cfg(target_arch = "x86_64")]
crate::impl_index!(i16x8, i16, 8);

#[cfg(target_arch = "x86_64")]
impl From<[i16; 8]> for i16x8 {
    #[inline(always)]
    fn from(arr: [i16; 8]) -> Self {
        // SAFETY: [i16; 8] and __m128i have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}
#[cfg(target_arch = "x86_64")]
impl From<i16x8> for [i16; 8] {
    #[inline(always)]
    fn from(v: i16x8) -> Self {
        // SAFETY: __m128i and [i16; 8] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}

// Scalar broadcast operations for i16x8
// These allow `v + 2.0` instead of `v + i16x8::splat(token, 2.0)`

#[cfg(target_arch = "x86_64")]
impl Add<i16> for i16x8 {
    type Output = Self;
    /// Add a scalar to all lanes.
    #[inline(always)]
    fn add(self, rhs: i16) -> Self {
        self + Self(unsafe { _mm_set1_epi16(rhs) })
    }
}

#[cfg(target_arch = "x86_64")]
impl Sub<i16> for i16x8 {
    type Output = Self;
    /// Subtract a scalar from all lanes.
    #[inline(always)]
    fn sub(self, rhs: i16) -> Self {
        self - Self(unsafe { _mm_set1_epi16(rhs) })
    }
}

// ============================================================================
// u16x8 - 8 x u16 (128-bit)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct u16x8(__m128i);

#[cfg(target_arch = "x86_64")]
impl u16x8 {
    pub const LANES: usize = 8;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::X64V3Token, data: &[u16; 8]) -> Self {
        Self(unsafe { _mm_loadu_si128(data.as_ptr() as *const __m128i) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::X64V3Token, v: u16) -> Self {
        Self(unsafe { _mm_set1_epi16(v as i16) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::X64V3Token) -> Self {
        Self(unsafe { _mm_setzero_si128() })
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::X64V3Token, arr: [u16; 8]) -> Self {
        // SAFETY: [u16; 8] and __m128i have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [u16; 8]) {
        unsafe { _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, self.0) };
    }

    /// Convert to array
    #[inline(always)]
    pub fn to_array(self) -> [u16; 8] {
        let mut out = [0u16; 8];
        self.store(&mut out);
        out
    }

    /// Get reference to underlying array
    #[inline(always)]
    pub fn as_array(&self) -> &[u16; 8] {
        unsafe { &*(self as *const Self as *const [u16; 8]) }
    }

    /// Get mutable reference to underlying array
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut [u16; 8] {
        unsafe { &mut *(self as *mut Self as *mut [u16; 8]) }
    }

    /// Get raw intrinsic type
    #[inline(always)]
    pub fn raw(self) -> __m128i {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports the required SIMD features.
    /// Use token-gated constructors (`load`, `splat`, `zero`) for safe construction.
    #[inline(always)]
    pub unsafe fn from_raw(v: __m128i) -> Self {
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
    pub fn cast_slice(_: archmage::X64V3Token, slice: &[u16]) -> Option<&[Self]> {
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
    pub fn cast_slice_mut(_: archmage::X64V3Token, slice: &mut [u16]) -> Option<&mut [Self]> {
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
    pub fn as_bytes(&self) -> &[u8; 16] {
        // SAFETY: Self is repr(transparent) over __m128i which is 16 bytes
        unsafe { &*(self as *const Self as *const [u8; 16]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 16] {
        // SAFETY: Self is repr(transparent) over __m128i which is 16 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 16]) }
    }

    /// Create from a byte array reference (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::X64V3Token, bytes: &[u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Create from an owned byte array (token-gated, zero-cost).
    ///
    /// This is a zero-cost transmute from an owned byte array.
    #[inline(always)]
    pub fn from_bytes_owned(_: archmage::X64V3Token, bytes: [u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(bytes) })
    }

    // ========== Implementation identification ==========

    /// Returns a string identifying this type's implementation.
    ///
    /// This is useful for verifying that the correct implementation is being used
    /// at compile time or at runtime (via `#[magetypes]` dispatch).
    #[inline(always)]
    pub const fn implementation_name() -> &'static str {
        "x86::v3::u16x8"
    }

    /// Element-wise minimum
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(unsafe { _mm_min_epu16(self.0, other.0) })
    }
    /// Element-wise maximum
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(unsafe { _mm_max_epu16(self.0, other.0) })
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
        Self(unsafe { _mm_cmpeq_epi16(self.0, other.0) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if inequality, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(unsafe {
            let eq = _mm_cmpeq_epi16(self.0, other.0);
            let ones = _mm_set1_epi16(-1);
            _mm_xor_si128(eq, ones)
        })
    }

    /// Lane-wise greater-than (unsigned) comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than (unsigned), all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe {
            // Flip sign bit to convert unsigned to signed comparison
            let bias = _mm_set1_epi16(0x8000u16 as i16);
            let a = _mm_xor_si128(self.0, bias);
            let b = _mm_xor_si128(other.0, bias);
            _mm_cmpgt_epi16(a, b)
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
            let ones = _mm_set1_epi16(-1);
            _mm_xor_si128(lt.0, ones)
        })
    }

    /// Lane-wise less-than-or-equal (unsigned) comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than-or-equal (unsigned), all-0s otherwise.
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(unsafe {
            let gt = self.simd_gt(other);
            let ones = _mm_set1_epi16(-1);
            _mm_xor_si128(gt.0, ones)
        })
    }

    // ========== Blending/Selection ==========

    /// Select lanes from `if_true` where mask is all-1s, `if_false` where mask is all-0s.
    ///
    /// The mask should come from a comparison operation like `simd_lt()`.
    ///
    /// # Example
    /// ```ignore
    /// let a = u16x8::splat(token, 1.0);
    /// let b = u16x8::splat(token, 2.0);
    /// let mask = a.simd_lt(b);  // all true
    /// let result = u16x8::blend(mask, a, b);  // selects a
    /// ```
    #[inline(always)]
    pub fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self(unsafe { _mm_blendv_epi8(if_false.0, if_true.0, mask.0) })
    }

    // ========== Boolean Reductions ==========

    /// Returns true if all lanes are non-zero (truthy).
    ///
    /// Typically used with comparison results where true lanes are all-1s.
    #[inline(always)]
    pub fn all_true(self) -> bool {
        unsafe { _mm_movemask_epi8(self.0) == 0xFFFF_u32 as i32 }
    }

    /// Returns true if any lane is non-zero (truthy).
    #[inline(always)]
    pub fn any_true(self) -> bool {
        unsafe { _mm_movemask_epi8(self.0) != 0 }
    }

    /// Extract the high bit of each lane as a bitmask.
    ///
    /// Returns a u32 where bit N corresponds to the sign bit of lane N.
    #[inline(always)]
    pub fn bitmask(self) -> u32 {
        unsafe {
            // Shift right to get sign bit in LSB, pack to bytes
            let shifted = _mm_srai_epi16::<15>(self.0);
            let packed = _mm_packs_epi16(shifted, shifted);
            (_mm_movemask_epi8(packed) & 0xFF) as u32
        }
    }

    // ========== Horizontal Operations ==========

    /// Sum all lanes horizontally.
    ///
    /// Returns a scalar containing the sum of all 8 lanes.
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
            let ones = _mm_set1_epi16(-1);
            _mm_xor_si128(self.0, ones)
        })
    }
    // ========== Shift Operations ==========

    /// Shift each lane left by `N` bits.
    ///
    /// Bits shifted out are lost; zeros are shifted in.
    #[inline(always)]
    pub fn shl<const N: i32>(self) -> Self {
        Self(unsafe { _mm_slli_epi16::<N>(self.0) })
    }

    /// Shift each lane right by `N` bits (logical/unsigned shift).
    ///
    /// Bits shifted out are lost; zeros are shifted in.
    #[inline(always)]
    pub fn shr<const N: i32>(self) -> Self {
        Self(unsafe { _mm_srli_epi16::<N>(self.0) })
    }

    // ========== Extend/Widen Operations ==========

    /// Zero-extend low 4 u16 values to i32x4.
    #[inline(always)]
    pub fn extend_lo_i32(self) -> i32x4 {
        i32x4(unsafe { _mm_cvtepu16_epi32(self.0) })
    }

    /// Zero-extend high 4 u16 values to i32x4.
    #[inline(always)]
    pub fn extend_hi_i32(self) -> i32x4 {
        i32x4(unsafe {
            let hi = _mm_srli_si128::<8>(self.0);
            _mm_cvtepu16_epi32(hi)
        })
    }

    /// Zero-extend all 8 u16 values to two i32x4 vectors.
    #[inline(always)]
    pub fn extend_i32(self) -> (i32x4, i32x4) {
        (self.extend_lo_i32(), self.extend_hi_i32())
    }

    /// Zero-extend low 4 u16 values to f32x4.
    #[inline(always)]
    pub fn extend_lo_f32(self) -> f32x4 {
        f32x4(unsafe {
            let i32s = _mm_cvtepu16_epi32(self.0);
            _mm_cvtepi32_ps(i32s)
        })
    }

    // ========== Bitcast ==========
    /// Reinterpret bits as `i16x8` (zero-cost).
    #[inline(always)]
    pub fn bitcast_i16x8(self) -> i16x8 {
        unsafe { core::mem::transmute(self) }
    }

    /// Reinterpret bits as `&i16x8` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_ref_i16x8(&self) -> &i16x8 {
        unsafe { &*(self as *const Self as *const i16x8) }
    }

    /// Reinterpret bits as `&mut i16x8` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_mut_i16x8(&mut self) -> &mut i16x8 {
        unsafe { &mut *(self as *mut Self as *mut i16x8) }
    }
}

#[cfg(target_arch = "x86_64")]
crate::impl_int_arithmetic_ops!(u16x8, _mm_add_epi16, _mm_sub_epi16);
#[cfg(target_arch = "x86_64")]
crate::impl_int_mul_op!(u16x8, _mm_mullo_epi16);
#[cfg(target_arch = "x86_64")]
crate::impl_assign_ops!(u16x8);
#[cfg(target_arch = "x86_64")]
crate::impl_bitwise_ops!(u16x8, __m128i, _mm_and_si128, _mm_or_si128, _mm_xor_si128);
#[cfg(target_arch = "x86_64")]
crate::impl_index!(u16x8, u16, 8);

#[cfg(target_arch = "x86_64")]
impl From<[u16; 8]> for u16x8 {
    #[inline(always)]
    fn from(arr: [u16; 8]) -> Self {
        // SAFETY: [u16; 8] and __m128i have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}
#[cfg(target_arch = "x86_64")]
impl From<u16x8> for [u16; 8] {
    #[inline(always)]
    fn from(v: u16x8) -> Self {
        // SAFETY: __m128i and [u16; 8] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}

// Scalar broadcast operations for u16x8
// These allow `v + 2.0` instead of `v + u16x8::splat(token, 2.0)`

#[cfg(target_arch = "x86_64")]
impl Add<u16> for u16x8 {
    type Output = Self;
    /// Add a scalar to all lanes.
    #[inline(always)]
    fn add(self, rhs: u16) -> Self {
        self + Self(unsafe { _mm_set1_epi16(rhs as i16) })
    }
}

#[cfg(target_arch = "x86_64")]
impl Sub<u16> for u16x8 {
    type Output = Self;
    /// Subtract a scalar from all lanes.
    #[inline(always)]
    fn sub(self, rhs: u16) -> Self {
        self - Self(unsafe { _mm_set1_epi16(rhs as i16) })
    }
}

// ============================================================================
// i32x4 - 4 x i32 (128-bit)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct i32x4(__m128i);

#[cfg(target_arch = "x86_64")]
impl i32x4 {
    pub const LANES: usize = 4;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::X64V3Token, data: &[i32; 4]) -> Self {
        Self(unsafe { _mm_loadu_si128(data.as_ptr() as *const __m128i) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::X64V3Token, v: i32) -> Self {
        Self(unsafe { _mm_set1_epi32(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::X64V3Token) -> Self {
        Self(unsafe { _mm_setzero_si128() })
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::X64V3Token, arr: [i32; 4]) -> Self {
        // SAFETY: [i32; 4] and __m128i have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [i32; 4]) {
        unsafe { _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, self.0) };
    }

    /// Convert to array
    #[inline(always)]
    pub fn to_array(self) -> [i32; 4] {
        let mut out = [0i32; 4];
        self.store(&mut out);
        out
    }

    /// Get reference to underlying array
    #[inline(always)]
    pub fn as_array(&self) -> &[i32; 4] {
        unsafe { &*(self as *const Self as *const [i32; 4]) }
    }

    /// Get mutable reference to underlying array
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut [i32; 4] {
        unsafe { &mut *(self as *mut Self as *mut [i32; 4]) }
    }

    /// Get raw intrinsic type
    #[inline(always)]
    pub fn raw(self) -> __m128i {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports the required SIMD features.
    /// Use token-gated constructors (`load`, `splat`, `zero`) for safe construction.
    #[inline(always)]
    pub unsafe fn from_raw(v: __m128i) -> Self {
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
    pub fn cast_slice(_: archmage::X64V3Token, slice: &[i32]) -> Option<&[Self]> {
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
    pub fn cast_slice_mut(_: archmage::X64V3Token, slice: &mut [i32]) -> Option<&mut [Self]> {
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
    pub fn as_bytes(&self) -> &[u8; 16] {
        // SAFETY: Self is repr(transparent) over __m128i which is 16 bytes
        unsafe { &*(self as *const Self as *const [u8; 16]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 16] {
        // SAFETY: Self is repr(transparent) over __m128i which is 16 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 16]) }
    }

    /// Create from a byte array reference (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::X64V3Token, bytes: &[u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Create from an owned byte array (token-gated, zero-cost).
    ///
    /// This is a zero-cost transmute from an owned byte array.
    #[inline(always)]
    pub fn from_bytes_owned(_: archmage::X64V3Token, bytes: [u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(bytes) })
    }

    // ========== Implementation identification ==========

    /// Returns a string identifying this type's implementation.
    ///
    /// This is useful for verifying that the correct implementation is being used
    /// at compile time or at runtime (via `#[magetypes]` dispatch).
    #[inline(always)]
    pub const fn implementation_name() -> &'static str {
        "x86::v3::i32x4"
    }

    /// Element-wise minimum
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(unsafe { _mm_min_epi32(self.0, other.0) })
    }
    /// Element-wise maximum
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(unsafe { _mm_max_epi32(self.0, other.0) })
    }
    /// Clamp values between lo and hi
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }

    /// Absolute value
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(unsafe { _mm_abs_epi32(self.0) })
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
        Self(unsafe { _mm_cmpeq_epi32(self.0, other.0) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if inequality, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(unsafe {
            let eq = _mm_cmpeq_epi32(self.0, other.0);
            let ones = _mm_set1_epi32(-1);
            _mm_xor_si128(eq, ones)
        })
    }

    /// Lane-wise greater-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than, all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe { _mm_cmpgt_epi32(self.0, other.0) })
    }

    /// Lane-wise less-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(unsafe { _mm_cmpgt_epi32(other.0, self.0) })
    }

    /// Lane-wise greater-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than-or-equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(unsafe {
            let lt = _mm_cmpgt_epi32(other.0, self.0);
            let ones = _mm_set1_epi32(-1);
            _mm_xor_si128(lt, ones)
        })
    }

    /// Lane-wise less-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than-or-equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(unsafe {
            let gt = _mm_cmpgt_epi32(self.0, other.0);
            let ones = _mm_set1_epi32(-1);
            _mm_xor_si128(gt, ones)
        })
    }

    // ========== Blending/Selection ==========

    /// Select lanes from `if_true` where mask is all-1s, `if_false` where mask is all-0s.
    ///
    /// The mask should come from a comparison operation like `simd_lt()`.
    ///
    /// # Example
    /// ```ignore
    /// let a = i32x4::splat(token, 1.0);
    /// let b = i32x4::splat(token, 2.0);
    /// let mask = a.simd_lt(b);  // all true
    /// let result = i32x4::blend(mask, a, b);  // selects a
    /// ```
    #[inline(always)]
    pub fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self(unsafe { _mm_blendv_epi8(if_false.0, if_true.0, mask.0) })
    }

    // ========== Boolean Reductions ==========

    /// Returns true if all lanes are non-zero (truthy).
    ///
    /// Typically used with comparison results where true lanes are all-1s.
    #[inline(always)]
    pub fn all_true(self) -> bool {
        unsafe { _mm_movemask_epi8(self.0) == 0xFFFF_u32 as i32 }
    }

    /// Returns true if any lane is non-zero (truthy).
    #[inline(always)]
    pub fn any_true(self) -> bool {
        unsafe { _mm_movemask_epi8(self.0) != 0 }
    }

    /// Extract the high bit of each lane as a bitmask.
    ///
    /// Returns a u32 where bit N corresponds to the sign bit of lane N.
    #[inline(always)]
    pub fn bitmask(self) -> u32 {
        unsafe { _mm_movemask_ps(_mm_castsi128_ps(self.0)) as u32 }
    }

    // ========== Horizontal Operations ==========

    /// Sum all lanes horizontally.
    ///
    /// Returns a scalar containing the sum of all 4 lanes.
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
    pub fn to_f32x4(self) -> f32x4 {
        f32x4(unsafe { _mm_cvtepi32_ps(self.0) })
    }

    // ========== Bitwise Unary Operations ==========

    /// Bitwise NOT (complement): flips all bits.
    #[inline(always)]
    pub fn not(self) -> Self {
        Self(unsafe {
            let ones = _mm_set1_epi32(-1);
            _mm_xor_si128(self.0, ones)
        })
    }
    // ========== Shift Operations ==========

    /// Shift each lane left by `N` bits.
    ///
    /// Bits shifted out are lost; zeros are shifted in.
    #[inline(always)]
    pub fn shl<const N: i32>(self) -> Self {
        Self(unsafe { _mm_slli_epi32::<N>(self.0) })
    }

    /// Shift each lane right by `N` bits (logical/unsigned shift).
    ///
    /// Bits shifted out are lost; zeros are shifted in.
    #[inline(always)]
    pub fn shr<const N: i32>(self) -> Self {
        Self(unsafe { _mm_srli_epi32::<N>(self.0) })
    }

    /// Arithmetic shift right by `N` bits (sign-extending).
    ///
    /// The sign bit is replicated into the vacated positions.
    #[inline(always)]
    pub fn shr_arithmetic<const N: i32>(self) -> Self {
        Self(unsafe { _mm_srai_epi32::<N>(self.0) })
    }

    // ========== Extend/Widen Operations ==========

    /// Convert to f32x4.
    #[inline(always)]
    pub fn to_f32(self) -> f32x4 {
        f32x4(unsafe { _mm_cvtepi32_ps(self.0) })
    }

    // ========== Pack/Narrow Operations ==========

    /// Pack two i32x4 vectors to i16x8 with signed saturation.
    ///
    /// `self` provides low 4 values, `other` provides high 4 values.
    #[inline(always)]
    pub fn pack_i16(self, other: Self) -> i16x8 {
        i16x8(unsafe { _mm_packs_epi32(self.0, other.0) })
    }

    /// Pack two i32x4 vectors to u16x8 with unsigned saturation.
    ///
    /// Requires SSE4.1.
    #[inline(always)]
    pub fn pack_u16(self, other: Self) -> u16x8 {
        u16x8(unsafe { _mm_packus_epi32(self.0, other.0) })
    }

    // ========== Bitcast ==========
    /// Reinterpret bits as `f32x4` (zero-cost).
    #[inline(always)]
    pub fn bitcast_f32x4(self) -> f32x4 {
        unsafe { core::mem::transmute(_mm_castsi128_ps(self.0)) }
    }

    /// Reinterpret bits as `&f32x4` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_ref_f32x4(&self) -> &f32x4 {
        unsafe { &*(self as *const Self as *const f32x4) }
    }

    /// Reinterpret bits as `&mut f32x4` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_mut_f32x4(&mut self) -> &mut f32x4 {
        unsafe { &mut *(self as *mut Self as *mut f32x4) }
    }
    /// Reinterpret bits as `u32x4` (zero-cost).
    #[inline(always)]
    pub fn bitcast_u32x4(self) -> u32x4 {
        unsafe { core::mem::transmute(self) }
    }

    /// Reinterpret bits as `&u32x4` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_ref_u32x4(&self) -> &u32x4 {
        unsafe { &*(self as *const Self as *const u32x4) }
    }

    /// Reinterpret bits as `&mut u32x4` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_mut_u32x4(&mut self) -> &mut u32x4 {
        unsafe { &mut *(self as *mut Self as *mut u32x4) }
    }
}

#[cfg(target_arch = "x86_64")]
crate::impl_int_arithmetic_ops!(i32x4, _mm_add_epi32, _mm_sub_epi32);
#[cfg(target_arch = "x86_64")]
crate::impl_int_mul_op!(i32x4, _mm_mullo_epi32);
#[cfg(target_arch = "x86_64")]
crate::impl_assign_ops!(i32x4);
#[cfg(target_arch = "x86_64")]
crate::impl_bitwise_ops!(i32x4, __m128i, _mm_and_si128, _mm_or_si128, _mm_xor_si128);
#[cfg(target_arch = "x86_64")]
crate::impl_index!(i32x4, i32, 4);

#[cfg(target_arch = "x86_64")]
impl From<[i32; 4]> for i32x4 {
    #[inline(always)]
    fn from(arr: [i32; 4]) -> Self {
        // SAFETY: [i32; 4] and __m128i have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}
#[cfg(target_arch = "x86_64")]
impl From<i32x4> for [i32; 4] {
    #[inline(always)]
    fn from(v: i32x4) -> Self {
        // SAFETY: __m128i and [i32; 4] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}

// Scalar broadcast operations for i32x4
// These allow `v + 2.0` instead of `v + i32x4::splat(token, 2.0)`

#[cfg(target_arch = "x86_64")]
impl Add<i32> for i32x4 {
    type Output = Self;
    /// Add a scalar to all lanes.
    #[inline(always)]
    fn add(self, rhs: i32) -> Self {
        self + Self(unsafe { _mm_set1_epi32(rhs) })
    }
}

#[cfg(target_arch = "x86_64")]
impl Sub<i32> for i32x4 {
    type Output = Self;
    /// Subtract a scalar from all lanes.
    #[inline(always)]
    fn sub(self, rhs: i32) -> Self {
        self - Self(unsafe { _mm_set1_epi32(rhs) })
    }
}

// ============================================================================
// u32x4 - 4 x u32 (128-bit)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct u32x4(__m128i);

#[cfg(target_arch = "x86_64")]
impl u32x4 {
    pub const LANES: usize = 4;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::X64V3Token, data: &[u32; 4]) -> Self {
        Self(unsafe { _mm_loadu_si128(data.as_ptr() as *const __m128i) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::X64V3Token, v: u32) -> Self {
        Self(unsafe { _mm_set1_epi32(v as i32) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::X64V3Token) -> Self {
        Self(unsafe { _mm_setzero_si128() })
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::X64V3Token, arr: [u32; 4]) -> Self {
        // SAFETY: [u32; 4] and __m128i have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [u32; 4]) {
        unsafe { _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, self.0) };
    }

    /// Convert to array
    #[inline(always)]
    pub fn to_array(self) -> [u32; 4] {
        let mut out = [0u32; 4];
        self.store(&mut out);
        out
    }

    /// Get reference to underlying array
    #[inline(always)]
    pub fn as_array(&self) -> &[u32; 4] {
        unsafe { &*(self as *const Self as *const [u32; 4]) }
    }

    /// Get mutable reference to underlying array
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut [u32; 4] {
        unsafe { &mut *(self as *mut Self as *mut [u32; 4]) }
    }

    /// Get raw intrinsic type
    #[inline(always)]
    pub fn raw(self) -> __m128i {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports the required SIMD features.
    /// Use token-gated constructors (`load`, `splat`, `zero`) for safe construction.
    #[inline(always)]
    pub unsafe fn from_raw(v: __m128i) -> Self {
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
    pub fn cast_slice(_: archmage::X64V3Token, slice: &[u32]) -> Option<&[Self]> {
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
    pub fn cast_slice_mut(_: archmage::X64V3Token, slice: &mut [u32]) -> Option<&mut [Self]> {
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
    pub fn as_bytes(&self) -> &[u8; 16] {
        // SAFETY: Self is repr(transparent) over __m128i which is 16 bytes
        unsafe { &*(self as *const Self as *const [u8; 16]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 16] {
        // SAFETY: Self is repr(transparent) over __m128i which is 16 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 16]) }
    }

    /// Create from a byte array reference (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::X64V3Token, bytes: &[u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Create from an owned byte array (token-gated, zero-cost).
    ///
    /// This is a zero-cost transmute from an owned byte array.
    #[inline(always)]
    pub fn from_bytes_owned(_: archmage::X64V3Token, bytes: [u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(bytes) })
    }

    // ========== Implementation identification ==========

    /// Returns a string identifying this type's implementation.
    ///
    /// This is useful for verifying that the correct implementation is being used
    /// at compile time or at runtime (via `#[magetypes]` dispatch).
    #[inline(always)]
    pub const fn implementation_name() -> &'static str {
        "x86::v3::u32x4"
    }

    /// Element-wise minimum
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(unsafe { _mm_min_epu32(self.0, other.0) })
    }
    /// Element-wise maximum
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(unsafe { _mm_max_epu32(self.0, other.0) })
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
        Self(unsafe { _mm_cmpeq_epi32(self.0, other.0) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if inequality, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(unsafe {
            let eq = _mm_cmpeq_epi32(self.0, other.0);
            let ones = _mm_set1_epi32(-1);
            _mm_xor_si128(eq, ones)
        })
    }

    /// Lane-wise greater-than (unsigned) comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than (unsigned), all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe {
            // Flip sign bit to convert unsigned to signed comparison
            let bias = _mm_set1_epi32(0x8000_0000u32 as i32);
            let a = _mm_xor_si128(self.0, bias);
            let b = _mm_xor_si128(other.0, bias);
            _mm_cmpgt_epi32(a, b)
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
            let ones = _mm_set1_epi32(-1);
            _mm_xor_si128(lt.0, ones)
        })
    }

    /// Lane-wise less-than-or-equal (unsigned) comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than-or-equal (unsigned), all-0s otherwise.
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(unsafe {
            let gt = self.simd_gt(other);
            let ones = _mm_set1_epi32(-1);
            _mm_xor_si128(gt.0, ones)
        })
    }

    // ========== Blending/Selection ==========

    /// Select lanes from `if_true` where mask is all-1s, `if_false` where mask is all-0s.
    ///
    /// The mask should come from a comparison operation like `simd_lt()`.
    ///
    /// # Example
    /// ```ignore
    /// let a = u32x4::splat(token, 1.0);
    /// let b = u32x4::splat(token, 2.0);
    /// let mask = a.simd_lt(b);  // all true
    /// let result = u32x4::blend(mask, a, b);  // selects a
    /// ```
    #[inline(always)]
    pub fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self(unsafe { _mm_blendv_epi8(if_false.0, if_true.0, mask.0) })
    }

    // ========== Boolean Reductions ==========

    /// Returns true if all lanes are non-zero (truthy).
    ///
    /// Typically used with comparison results where true lanes are all-1s.
    #[inline(always)]
    pub fn all_true(self) -> bool {
        unsafe { _mm_movemask_epi8(self.0) == 0xFFFF_u32 as i32 }
    }

    /// Returns true if any lane is non-zero (truthy).
    #[inline(always)]
    pub fn any_true(self) -> bool {
        unsafe { _mm_movemask_epi8(self.0) != 0 }
    }

    /// Extract the high bit of each lane as a bitmask.
    ///
    /// Returns a u32 where bit N corresponds to the sign bit of lane N.
    #[inline(always)]
    pub fn bitmask(self) -> u32 {
        unsafe { _mm_movemask_ps(_mm_castsi128_ps(self.0)) as u32 }
    }

    // ========== Horizontal Operations ==========

    /// Sum all lanes horizontally.
    ///
    /// Returns a scalar containing the sum of all 4 lanes.
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
            let ones = _mm_set1_epi32(-1);
            _mm_xor_si128(self.0, ones)
        })
    }
    // ========== Shift Operations ==========

    /// Shift each lane left by `N` bits.
    ///
    /// Bits shifted out are lost; zeros are shifted in.
    #[inline(always)]
    pub fn shl<const N: i32>(self) -> Self {
        Self(unsafe { _mm_slli_epi32::<N>(self.0) })
    }

    /// Shift each lane right by `N` bits (logical/unsigned shift).
    ///
    /// Bits shifted out are lost; zeros are shifted in.
    #[inline(always)]
    pub fn shr<const N: i32>(self) -> Self {
        Self(unsafe { _mm_srli_epi32::<N>(self.0) })
    }

    // ========== Bitcast ==========
    /// Reinterpret bits as `f32x4` (zero-cost).
    #[inline(always)]
    pub fn bitcast_f32x4(self) -> f32x4 {
        unsafe { core::mem::transmute(_mm_castsi128_ps(self.0)) }
    }

    /// Reinterpret bits as `&f32x4` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_ref_f32x4(&self) -> &f32x4 {
        unsafe { &*(self as *const Self as *const f32x4) }
    }

    /// Reinterpret bits as `&mut f32x4` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_mut_f32x4(&mut self) -> &mut f32x4 {
        unsafe { &mut *(self as *mut Self as *mut f32x4) }
    }
    /// Reinterpret bits as `i32x4` (zero-cost).
    #[inline(always)]
    pub fn bitcast_i32x4(self) -> i32x4 {
        unsafe { core::mem::transmute(self) }
    }

    /// Reinterpret bits as `&i32x4` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_ref_i32x4(&self) -> &i32x4 {
        unsafe { &*(self as *const Self as *const i32x4) }
    }

    /// Reinterpret bits as `&mut i32x4` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_mut_i32x4(&mut self) -> &mut i32x4 {
        unsafe { &mut *(self as *mut Self as *mut i32x4) }
    }
}

#[cfg(target_arch = "x86_64")]
crate::impl_int_arithmetic_ops!(u32x4, _mm_add_epi32, _mm_sub_epi32);
#[cfg(target_arch = "x86_64")]
crate::impl_int_mul_op!(u32x4, _mm_mullo_epi32);
#[cfg(target_arch = "x86_64")]
crate::impl_assign_ops!(u32x4);
#[cfg(target_arch = "x86_64")]
crate::impl_bitwise_ops!(u32x4, __m128i, _mm_and_si128, _mm_or_si128, _mm_xor_si128);
#[cfg(target_arch = "x86_64")]
crate::impl_index!(u32x4, u32, 4);

#[cfg(target_arch = "x86_64")]
impl From<[u32; 4]> for u32x4 {
    #[inline(always)]
    fn from(arr: [u32; 4]) -> Self {
        // SAFETY: [u32; 4] and __m128i have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}
#[cfg(target_arch = "x86_64")]
impl From<u32x4> for [u32; 4] {
    #[inline(always)]
    fn from(v: u32x4) -> Self {
        // SAFETY: __m128i and [u32; 4] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}

// Scalar broadcast operations for u32x4
// These allow `v + 2.0` instead of `v + u32x4::splat(token, 2.0)`

#[cfg(target_arch = "x86_64")]
impl Add<u32> for u32x4 {
    type Output = Self;
    /// Add a scalar to all lanes.
    #[inline(always)]
    fn add(self, rhs: u32) -> Self {
        self + Self(unsafe { _mm_set1_epi32(rhs as i32) })
    }
}

#[cfg(target_arch = "x86_64")]
impl Sub<u32> for u32x4 {
    type Output = Self;
    /// Subtract a scalar from all lanes.
    #[inline(always)]
    fn sub(self, rhs: u32) -> Self {
        self - Self(unsafe { _mm_set1_epi32(rhs as i32) })
    }
}

// ============================================================================
// i64x2 - 2 x i64 (128-bit)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct i64x2(__m128i);

#[cfg(target_arch = "x86_64")]
impl i64x2 {
    pub const LANES: usize = 2;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::X64V3Token, data: &[i64; 2]) -> Self {
        Self(unsafe { _mm_loadu_si128(data.as_ptr() as *const __m128i) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::X64V3Token, v: i64) -> Self {
        Self(unsafe { _mm_set1_epi64x(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::X64V3Token) -> Self {
        Self(unsafe { _mm_setzero_si128() })
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::X64V3Token, arr: [i64; 2]) -> Self {
        // SAFETY: [i64; 2] and __m128i have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [i64; 2]) {
        unsafe { _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, self.0) };
    }

    /// Convert to array
    #[inline(always)]
    pub fn to_array(self) -> [i64; 2] {
        let mut out = [0i64; 2];
        self.store(&mut out);
        out
    }

    /// Get reference to underlying array
    #[inline(always)]
    pub fn as_array(&self) -> &[i64; 2] {
        unsafe { &*(self as *const Self as *const [i64; 2]) }
    }

    /// Get mutable reference to underlying array
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut [i64; 2] {
        unsafe { &mut *(self as *mut Self as *mut [i64; 2]) }
    }

    /// Get raw intrinsic type
    #[inline(always)]
    pub fn raw(self) -> __m128i {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports the required SIMD features.
    /// Use token-gated constructors (`load`, `splat`, `zero`) for safe construction.
    #[inline(always)]
    pub unsafe fn from_raw(v: __m128i) -> Self {
        Self(v)
    }

    // ========== Token-gated bytemuck replacements ==========

    /// Reinterpret a slice of scalars as a slice of SIMD vectors (token-gated).
    ///
    /// Returns `None` if the slice length is not a multiple of 2, or
    /// if the slice is not properly aligned.
    ///
    /// This is a safe, token-gated replacement for `bytemuck::cast_slice`.
    #[inline(always)]
    pub fn cast_slice(_: archmage::X64V3Token, slice: &[i64]) -> Option<&[Self]> {
        if slice.len() % 2 != 0 {
            return None;
        }
        let ptr = slice.as_ptr();
        if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {
            return None;
        }
        let len = slice.len() / 2;
        // SAFETY: alignment and length checked, layout is compatible
        Some(unsafe { core::slice::from_raw_parts(ptr as *const Self, len) })
    }

    /// Reinterpret a mutable slice of scalars as a slice of SIMD vectors (token-gated).
    ///
    /// Returns `None` if the slice length is not a multiple of 2, or
    /// if the slice is not properly aligned.
    ///
    /// This is a safe, token-gated replacement for `bytemuck::cast_slice_mut`.
    #[inline(always)]
    pub fn cast_slice_mut(_: archmage::X64V3Token, slice: &mut [i64]) -> Option<&mut [Self]> {
        if slice.len() % 2 != 0 {
            return None;
        }
        let ptr = slice.as_mut_ptr();
        if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {
            return None;
        }
        let len = slice.len() / 2;
        // SAFETY: alignment and length checked, layout is compatible
        Some(unsafe { core::slice::from_raw_parts_mut(ptr as *mut Self, len) })
    }

    /// View this vector as a byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of`.
    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8; 16] {
        // SAFETY: Self is repr(transparent) over __m128i which is 16 bytes
        unsafe { &*(self as *const Self as *const [u8; 16]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 16] {
        // SAFETY: Self is repr(transparent) over __m128i which is 16 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 16]) }
    }

    /// Create from a byte array reference (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::X64V3Token, bytes: &[u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Create from an owned byte array (token-gated, zero-cost).
    ///
    /// This is a zero-cost transmute from an owned byte array.
    #[inline(always)]
    pub fn from_bytes_owned(_: archmage::X64V3Token, bytes: [u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(bytes) })
    }

    // ========== Implementation identification ==========

    /// Returns a string identifying this type's implementation.
    ///
    /// This is useful for verifying that the correct implementation is being used
    /// at compile time or at runtime (via `#[magetypes]` dispatch).
    #[inline(always)]
    pub const fn implementation_name() -> &'static str {
        "x86::v3::i64x2"
    }

    /// Element-wise minimum (polyfill via compare+select)
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        unsafe {
            let mask = _mm_cmpgt_epi64(self.0, other.0);
            Self(_mm_blendv_epi8(self.0, other.0, mask))
        }
    }

    /// Element-wise maximum (polyfill via compare+select)
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        unsafe {
            let mask = _mm_cmpgt_epi64(self.0, other.0);
            Self(_mm_blendv_epi8(other.0, self.0, mask))
        }
    }

    /// Clamp values between lo and hi
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }

    #[cfg(feature = "avx512")]
    /// Element-wise minimum using AVX-512VL native intrinsic.
    ///
    /// Single instruction, faster than the polyfill used by `min()`.
    #[inline(always)]
    pub fn min_fast(self, other: Self, _: archmage::X64V4Token) -> Self {
        Self(unsafe { _mm_min_epi64(self.0, other.0) })
    }

    #[cfg(feature = "avx512")]
    /// Element-wise maximum using AVX-512VL native intrinsic.
    ///
    /// Single instruction, faster than the polyfill used by `max()`.
    #[inline(always)]
    pub fn max_fast(self, other: Self, _: archmage::X64V4Token) -> Self {
        Self(unsafe { _mm_max_epi64(self.0, other.0) })
    }

    /// Absolute value (polyfill via conditional negate)
    #[inline(always)]
    pub fn abs(self) -> Self {
        unsafe {
            let zero = _mm_setzero_si128();
            let sign = _mm_cmpgt_epi64(zero, self.0);
            Self(_mm_sub_epi64(_mm_xor_si128(self.0, sign), sign))
        }
    }

    #[cfg(feature = "avx512")]
    /// Absolute value using AVX-512VL native intrinsic.
    ///
    /// Single instruction, faster than the polyfill used by `abs()`.
    #[inline(always)]
    pub fn abs_fast(self, _: archmage::X64V4Token) -> Self {
        Self(unsafe { _mm_abs_epi64(self.0) })
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
        Self(unsafe { _mm_cmpeq_epi64(self.0, other.0) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if inequality, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(unsafe {
            let eq = _mm_cmpeq_epi64(self.0, other.0);
            let ones = _mm_set1_epi64x(-1);
            _mm_xor_si128(eq, ones)
        })
    }

    /// Lane-wise greater-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than, all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe { _mm_cmpgt_epi64(self.0, other.0) })
    }

    /// Lane-wise less-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(unsafe { _mm_cmpgt_epi64(other.0, self.0) })
    }

    /// Lane-wise greater-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than-or-equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(unsafe {
            let lt = _mm_cmpgt_epi64(other.0, self.0);
            let ones = _mm_set1_epi64x(-1);
            _mm_xor_si128(lt, ones)
        })
    }

    /// Lane-wise less-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than-or-equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(unsafe {
            let gt = _mm_cmpgt_epi64(self.0, other.0);
            let ones = _mm_set1_epi64x(-1);
            _mm_xor_si128(gt, ones)
        })
    }

    // ========== Blending/Selection ==========

    /// Select lanes from `if_true` where mask is all-1s, `if_false` where mask is all-0s.
    ///
    /// The mask should come from a comparison operation like `simd_lt()`.
    ///
    /// # Example
    /// ```ignore
    /// let a = i64x2::splat(token, 1.0);
    /// let b = i64x2::splat(token, 2.0);
    /// let mask = a.simd_lt(b);  // all true
    /// let result = i64x2::blend(mask, a, b);  // selects a
    /// ```
    #[inline(always)]
    pub fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self(unsafe { _mm_blendv_epi8(if_false.0, if_true.0, mask.0) })
    }

    // ========== Boolean Reductions ==========

    /// Returns true if all lanes are non-zero (truthy).
    ///
    /// Typically used with comparison results where true lanes are all-1s.
    #[inline(always)]
    pub fn all_true(self) -> bool {
        unsafe { _mm_movemask_epi8(self.0) == 0xFFFF_u32 as i32 }
    }

    /// Returns true if any lane is non-zero (truthy).
    #[inline(always)]
    pub fn any_true(self) -> bool {
        unsafe { _mm_movemask_epi8(self.0) != 0 }
    }

    /// Extract the high bit of each lane as a bitmask.
    ///
    /// Returns a u32 where bit N corresponds to the sign bit of lane N.
    #[inline(always)]
    pub fn bitmask(self) -> u32 {
        unsafe { _mm_movemask_pd(_mm_castsi128_pd(self.0)) as u32 }
    }

    // ========== Horizontal Operations ==========

    /// Sum all lanes horizontally.
    ///
    /// Returns a scalar containing the sum of all 2 lanes.
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
            let ones = _mm_set1_epi64x(-1);
            _mm_xor_si128(self.0, ones)
        })
    }
    // ========== Shift Operations ==========

    /// Shift each lane left by `N` bits.
    ///
    /// Bits shifted out are lost; zeros are shifted in.
    #[inline(always)]
    pub fn shl<const N: i32>(self) -> Self {
        Self(unsafe { _mm_slli_epi64::<N>(self.0) })
    }

    /// Shift each lane right by `N` bits (logical/unsigned shift).
    ///
    /// Bits shifted out are lost; zeros are shifted in.
    #[inline(always)]
    pub fn shr<const N: i32>(self) -> Self {
        Self(unsafe { _mm_srli_epi64::<N>(self.0) })
    }

    // ========== Bitcast ==========
    /// Reinterpret bits as `f64x2` (zero-cost).
    #[inline(always)]
    pub fn bitcast_f64x2(self) -> f64x2 {
        unsafe { core::mem::transmute(_mm_castsi128_pd(self.0)) }
    }

    /// Reinterpret bits as `&f64x2` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_ref_f64x2(&self) -> &f64x2 {
        unsafe { &*(self as *const Self as *const f64x2) }
    }

    /// Reinterpret bits as `&mut f64x2` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_mut_f64x2(&mut self) -> &mut f64x2 {
        unsafe { &mut *(self as *mut Self as *mut f64x2) }
    }
    /// Reinterpret bits as `u64x2` (zero-cost).
    #[inline(always)]
    pub fn bitcast_u64x2(self) -> u64x2 {
        unsafe { core::mem::transmute(self) }
    }

    /// Reinterpret bits as `&u64x2` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_ref_u64x2(&self) -> &u64x2 {
        unsafe { &*(self as *const Self as *const u64x2) }
    }

    /// Reinterpret bits as `&mut u64x2` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_mut_u64x2(&mut self) -> &mut u64x2 {
        unsafe { &mut *(self as *mut Self as *mut u64x2) }
    }
}

#[cfg(target_arch = "x86_64")]
crate::impl_int_arithmetic_ops!(i64x2, _mm_add_epi64, _mm_sub_epi64);
#[cfg(target_arch = "x86_64")]
crate::impl_assign_ops!(i64x2);
#[cfg(target_arch = "x86_64")]
crate::impl_bitwise_ops!(i64x2, __m128i, _mm_and_si128, _mm_or_si128, _mm_xor_si128);
#[cfg(target_arch = "x86_64")]
crate::impl_index!(i64x2, i64, 2);

#[cfg(target_arch = "x86_64")]
impl From<[i64; 2]> for i64x2 {
    #[inline(always)]
    fn from(arr: [i64; 2]) -> Self {
        // SAFETY: [i64; 2] and __m128i have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}
#[cfg(target_arch = "x86_64")]
impl From<i64x2> for [i64; 2] {
    #[inline(always)]
    fn from(v: i64x2) -> Self {
        // SAFETY: __m128i and [i64; 2] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}

// Scalar broadcast operations for i64x2
// These allow `v + 2.0` instead of `v + i64x2::splat(token, 2.0)`

#[cfg(target_arch = "x86_64")]
impl Add<i64> for i64x2 {
    type Output = Self;
    /// Add a scalar to all lanes.
    #[inline(always)]
    fn add(self, rhs: i64) -> Self {
        self + Self(unsafe { _mm_set1_epi64x(rhs) })
    }
}

#[cfg(target_arch = "x86_64")]
impl Sub<i64> for i64x2 {
    type Output = Self;
    /// Subtract a scalar from all lanes.
    #[inline(always)]
    fn sub(self, rhs: i64) -> Self {
        self - Self(unsafe { _mm_set1_epi64x(rhs) })
    }
}

// ============================================================================
// u64x2 - 2 x u64 (128-bit)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct u64x2(__m128i);

#[cfg(target_arch = "x86_64")]
impl u64x2 {
    pub const LANES: usize = 2;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::X64V3Token, data: &[u64; 2]) -> Self {
        Self(unsafe { _mm_loadu_si128(data.as_ptr() as *const __m128i) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::X64V3Token, v: u64) -> Self {
        Self(unsafe { _mm_set1_epi64x(v as i64) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::X64V3Token) -> Self {
        Self(unsafe { _mm_setzero_si128() })
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::X64V3Token, arr: [u64; 2]) -> Self {
        // SAFETY: [u64; 2] and __m128i have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [u64; 2]) {
        unsafe { _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, self.0) };
    }

    /// Convert to array
    #[inline(always)]
    pub fn to_array(self) -> [u64; 2] {
        let mut out = [0u64; 2];
        self.store(&mut out);
        out
    }

    /// Get reference to underlying array
    #[inline(always)]
    pub fn as_array(&self) -> &[u64; 2] {
        unsafe { &*(self as *const Self as *const [u64; 2]) }
    }

    /// Get mutable reference to underlying array
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut [u64; 2] {
        unsafe { &mut *(self as *mut Self as *mut [u64; 2]) }
    }

    /// Get raw intrinsic type
    #[inline(always)]
    pub fn raw(self) -> __m128i {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports the required SIMD features.
    /// Use token-gated constructors (`load`, `splat`, `zero`) for safe construction.
    #[inline(always)]
    pub unsafe fn from_raw(v: __m128i) -> Self {
        Self(v)
    }

    // ========== Token-gated bytemuck replacements ==========

    /// Reinterpret a slice of scalars as a slice of SIMD vectors (token-gated).
    ///
    /// Returns `None` if the slice length is not a multiple of 2, or
    /// if the slice is not properly aligned.
    ///
    /// This is a safe, token-gated replacement for `bytemuck::cast_slice`.
    #[inline(always)]
    pub fn cast_slice(_: archmage::X64V3Token, slice: &[u64]) -> Option<&[Self]> {
        if slice.len() % 2 != 0 {
            return None;
        }
        let ptr = slice.as_ptr();
        if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {
            return None;
        }
        let len = slice.len() / 2;
        // SAFETY: alignment and length checked, layout is compatible
        Some(unsafe { core::slice::from_raw_parts(ptr as *const Self, len) })
    }

    /// Reinterpret a mutable slice of scalars as a slice of SIMD vectors (token-gated).
    ///
    /// Returns `None` if the slice length is not a multiple of 2, or
    /// if the slice is not properly aligned.
    ///
    /// This is a safe, token-gated replacement for `bytemuck::cast_slice_mut`.
    #[inline(always)]
    pub fn cast_slice_mut(_: archmage::X64V3Token, slice: &mut [u64]) -> Option<&mut [Self]> {
        if slice.len() % 2 != 0 {
            return None;
        }
        let ptr = slice.as_mut_ptr();
        if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {
            return None;
        }
        let len = slice.len() / 2;
        // SAFETY: alignment and length checked, layout is compatible
        Some(unsafe { core::slice::from_raw_parts_mut(ptr as *mut Self, len) })
    }

    /// View this vector as a byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of`.
    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8; 16] {
        // SAFETY: Self is repr(transparent) over __m128i which is 16 bytes
        unsafe { &*(self as *const Self as *const [u8; 16]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 16] {
        // SAFETY: Self is repr(transparent) over __m128i which is 16 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 16]) }
    }

    /// Create from a byte array reference (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::X64V3Token, bytes: &[u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Create from an owned byte array (token-gated, zero-cost).
    ///
    /// This is a zero-cost transmute from an owned byte array.
    #[inline(always)]
    pub fn from_bytes_owned(_: archmage::X64V3Token, bytes: [u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(bytes) })
    }

    // ========== Implementation identification ==========

    /// Returns a string identifying this type's implementation.
    ///
    /// This is useful for verifying that the correct implementation is being used
    /// at compile time or at runtime (via `#[magetypes]` dispatch).
    #[inline(always)]
    pub const fn implementation_name() -> &'static str {
        "x86::v3::u64x2"
    }

    /// Element-wise minimum (polyfill via unsigned compare+select)
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        unsafe {
            let bias = _mm_set1_epi64x(i64::MIN);
            let a_biased = _mm_xor_si128(self.0, bias);
            let b_biased = _mm_xor_si128(other.0, bias);
            let mask = _mm_cmpgt_epi64(a_biased, b_biased);
            Self(_mm_blendv_epi8(self.0, other.0, mask))
        }
    }

    /// Element-wise maximum (polyfill via unsigned compare+select)
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        unsafe {
            let bias = _mm_set1_epi64x(i64::MIN);
            let a_biased = _mm_xor_si128(self.0, bias);
            let b_biased = _mm_xor_si128(other.0, bias);
            let mask = _mm_cmpgt_epi64(a_biased, b_biased);
            Self(_mm_blendv_epi8(other.0, self.0, mask))
        }
    }

    /// Clamp values between lo and hi
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }

    #[cfg(feature = "avx512")]
    /// Element-wise minimum using AVX-512VL native intrinsic.
    ///
    /// Single instruction, faster than the polyfill used by `min()`.
    #[inline(always)]
    pub fn min_fast(self, other: Self, _: archmage::X64V4Token) -> Self {
        Self(unsafe { _mm_min_epu64(self.0, other.0) })
    }

    #[cfg(feature = "avx512")]
    /// Element-wise maximum using AVX-512VL native intrinsic.
    ///
    /// Single instruction, faster than the polyfill used by `max()`.
    #[inline(always)]
    pub fn max_fast(self, other: Self, _: archmage::X64V4Token) -> Self {
        Self(unsafe { _mm_max_epu64(self.0, other.0) })
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
        Self(unsafe { _mm_cmpeq_epi64(self.0, other.0) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if inequality, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(unsafe {
            let eq = _mm_cmpeq_epi64(self.0, other.0);
            let ones = _mm_set1_epi64x(-1);
            _mm_xor_si128(eq, ones)
        })
    }

    /// Lane-wise greater-than (unsigned) comparison.
    ///
    /// Returns a mask where each lane is all-1s if greater-than (unsigned), all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe {
            // Flip sign bit to convert unsigned to signed comparison
            let bias = _mm_set1_epi64x(0x8000_0000_0000_0000u64 as i64);
            let a = _mm_xor_si128(self.0, bias);
            let b = _mm_xor_si128(other.0, bias);
            _mm_cmpgt_epi64(a, b)
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
            let ones = _mm_set1_epi64x(-1);
            _mm_xor_si128(lt.0, ones)
        })
    }

    /// Lane-wise less-than-or-equal (unsigned) comparison.
    ///
    /// Returns a mask where each lane is all-1s if less-than-or-equal (unsigned), all-0s otherwise.
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(unsafe {
            let gt = self.simd_gt(other);
            let ones = _mm_set1_epi64x(-1);
            _mm_xor_si128(gt.0, ones)
        })
    }

    // ========== Blending/Selection ==========

    /// Select lanes from `if_true` where mask is all-1s, `if_false` where mask is all-0s.
    ///
    /// The mask should come from a comparison operation like `simd_lt()`.
    ///
    /// # Example
    /// ```ignore
    /// let a = u64x2::splat(token, 1.0);
    /// let b = u64x2::splat(token, 2.0);
    /// let mask = a.simd_lt(b);  // all true
    /// let result = u64x2::blend(mask, a, b);  // selects a
    /// ```
    #[inline(always)]
    pub fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self(unsafe { _mm_blendv_epi8(if_false.0, if_true.0, mask.0) })
    }

    // ========== Boolean Reductions ==========

    /// Returns true if all lanes are non-zero (truthy).
    ///
    /// Typically used with comparison results where true lanes are all-1s.
    #[inline(always)]
    pub fn all_true(self) -> bool {
        unsafe { _mm_movemask_epi8(self.0) == 0xFFFF_u32 as i32 }
    }

    /// Returns true if any lane is non-zero (truthy).
    #[inline(always)]
    pub fn any_true(self) -> bool {
        unsafe { _mm_movemask_epi8(self.0) != 0 }
    }

    /// Extract the high bit of each lane as a bitmask.
    ///
    /// Returns a u32 where bit N corresponds to the sign bit of lane N.
    #[inline(always)]
    pub fn bitmask(self) -> u32 {
        unsafe { _mm_movemask_pd(_mm_castsi128_pd(self.0)) as u32 }
    }

    // ========== Horizontal Operations ==========

    /// Sum all lanes horizontally.
    ///
    /// Returns a scalar containing the sum of all 2 lanes.
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
            let ones = _mm_set1_epi64x(-1);
            _mm_xor_si128(self.0, ones)
        })
    }
    // ========== Shift Operations ==========

    /// Shift each lane left by `N` bits.
    ///
    /// Bits shifted out are lost; zeros are shifted in.
    #[inline(always)]
    pub fn shl<const N: i32>(self) -> Self {
        Self(unsafe { _mm_slli_epi64::<N>(self.0) })
    }

    /// Shift each lane right by `N` bits (logical/unsigned shift).
    ///
    /// Bits shifted out are lost; zeros are shifted in.
    #[inline(always)]
    pub fn shr<const N: i32>(self) -> Self {
        Self(unsafe { _mm_srli_epi64::<N>(self.0) })
    }

    // ========== Bitcast ==========
    /// Reinterpret bits as `f64x2` (zero-cost).
    #[inline(always)]
    pub fn bitcast_f64x2(self) -> f64x2 {
        unsafe { core::mem::transmute(_mm_castsi128_pd(self.0)) }
    }

    /// Reinterpret bits as `&f64x2` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_ref_f64x2(&self) -> &f64x2 {
        unsafe { &*(self as *const Self as *const f64x2) }
    }

    /// Reinterpret bits as `&mut f64x2` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_mut_f64x2(&mut self) -> &mut f64x2 {
        unsafe { &mut *(self as *mut Self as *mut f64x2) }
    }
    /// Reinterpret bits as `i64x2` (zero-cost).
    #[inline(always)]
    pub fn bitcast_i64x2(self) -> i64x2 {
        unsafe { core::mem::transmute(self) }
    }

    /// Reinterpret bits as `&i64x2` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_ref_i64x2(&self) -> &i64x2 {
        unsafe { &*(self as *const Self as *const i64x2) }
    }

    /// Reinterpret bits as `&mut i64x2` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_mut_i64x2(&mut self) -> &mut i64x2 {
        unsafe { &mut *(self as *mut Self as *mut i64x2) }
    }
}

#[cfg(target_arch = "x86_64")]
crate::impl_int_arithmetic_ops!(u64x2, _mm_add_epi64, _mm_sub_epi64);
#[cfg(target_arch = "x86_64")]
crate::impl_assign_ops!(u64x2);
#[cfg(target_arch = "x86_64")]
crate::impl_bitwise_ops!(u64x2, __m128i, _mm_and_si128, _mm_or_si128, _mm_xor_si128);
#[cfg(target_arch = "x86_64")]
crate::impl_index!(u64x2, u64, 2);

#[cfg(target_arch = "x86_64")]
impl From<[u64; 2]> for u64x2 {
    #[inline(always)]
    fn from(arr: [u64; 2]) -> Self {
        // SAFETY: [u64; 2] and __m128i have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}
#[cfg(target_arch = "x86_64")]
impl From<u64x2> for [u64; 2] {
    #[inline(always)]
    fn from(v: u64x2) -> Self {
        // SAFETY: __m128i and [u64; 2] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}

// Scalar broadcast operations for u64x2
// These allow `v + 2.0` instead of `v + u64x2::splat(token, 2.0)`

#[cfg(target_arch = "x86_64")]
impl Add<u64> for u64x2 {
    type Output = Self;
    /// Add a scalar to all lanes.
    #[inline(always)]
    fn add(self, rhs: u64) -> Self {
        self + Self(unsafe { _mm_set1_epi64x(rhs as i64) })
    }
}

#[cfg(target_arch = "x86_64")]
impl Sub<u64> for u64x2 {
    type Output = Self;
    /// Subtract a scalar from all lanes.
    #[inline(always)]
    fn sub(self, rhs: u64) -> Self {
        self - Self(unsafe { _mm_set1_epi64x(rhs as i64) })
    }
}

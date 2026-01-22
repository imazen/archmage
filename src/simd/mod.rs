//! Token-gated SIMD types with natural operators.
//!
//! Provides `wide`-like ergonomics with token-gated construction.
//! There is NO way to construct these types without proving CPU support.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

#![allow(dead_code)]
#![allow(non_camel_case_types)]
#![allow(missing_docs)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign,
    Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};


// ============================================================================
// Comparison Traits (return masks, not bool)
// ============================================================================

/// SIMD equality comparison (returns mask)
pub trait SimdEq<Rhs = Self> {
    type Output;
    fn simd_eq(self, rhs: Rhs) -> Self::Output;
}

/// SIMD inequality comparison (returns mask)
pub trait SimdNe<Rhs = Self> {
    type Output;
    fn simd_ne(self, rhs: Rhs) -> Self::Output;
}

/// SIMD less-than comparison (returns mask)
pub trait SimdLt<Rhs = Self> {
    type Output;
    fn simd_lt(self, rhs: Rhs) -> Self::Output;
}

/// SIMD less-than-or-equal comparison (returns mask)
pub trait SimdLe<Rhs = Self> {
    type Output;
    fn simd_le(self, rhs: Rhs) -> Self::Output;
}

/// SIMD greater-than comparison (returns mask)
pub trait SimdGt<Rhs = Self> {
    type Output;
    fn simd_gt(self, rhs: Rhs) -> Self::Output;
}

/// SIMD greater-than-or-equal comparison (returns mask)
pub trait SimdGe<Rhs = Self> {
    type Output;
    fn simd_ge(self, rhs: Rhs) -> Self::Output;
}


// ============================================================================
// Implementation Macros
// ============================================================================

macro_rules! impl_arithmetic_ops {
    ($t:ty, $add:path, $sub:path, $mul:path, $div:path) => {
        impl Add for $t {
            type Output = Self;
            #[inline(always)]
            fn add(self, rhs: Self) -> Self {
                Self(unsafe { $add(self.0, rhs.0) })
            }
        }
        impl Sub for $t {
            type Output = Self;
            #[inline(always)]
            fn sub(self, rhs: Self) -> Self {
                Self(unsafe { $sub(self.0, rhs.0) })
            }
        }
        impl Mul for $t {
            type Output = Self;
            #[inline(always)]
            fn mul(self, rhs: Self) -> Self {
                Self(unsafe { $mul(self.0, rhs.0) })
            }
        }
        impl Div for $t {
            type Output = Self;
            #[inline(always)]
            fn div(self, rhs: Self) -> Self {
                Self(unsafe { $div(self.0, rhs.0) })
            }
        }
    };
}

macro_rules! impl_int_arithmetic_ops {
    ($t:ty, $add:path, $sub:path) => {
        impl Add for $t {
            type Output = Self;
            #[inline(always)]
            fn add(self, rhs: Self) -> Self {
                Self(unsafe { $add(self.0, rhs.0) })
            }
        }
        impl Sub for $t {
            type Output = Self;
            #[inline(always)]
            fn sub(self, rhs: Self) -> Self {
                Self(unsafe { $sub(self.0, rhs.0) })
            }
        }
    };
}

macro_rules! impl_int_mul_op {
    ($t:ty, $mul:path) => {
        impl Mul for $t {
            type Output = Self;
            #[inline(always)]
            fn mul(self, rhs: Self) -> Self {
                Self(unsafe { $mul(self.0, rhs.0) })
            }
        }
    };
}

macro_rules! impl_bitwise_ops {
    ($t:ty, $inner:ty, $and:path, $or:path, $xor:path) => {
        impl BitAnd for $t {
            type Output = Self;
            #[inline(always)]
            fn bitand(self, rhs: Self) -> Self {
                Self(unsafe { $and(self.0, rhs.0) })
            }
        }
        impl BitOr for $t {
            type Output = Self;
            #[inline(always)]
            fn bitor(self, rhs: Self) -> Self {
                Self(unsafe { $or(self.0, rhs.0) })
            }
        }
        impl BitXor for $t {
            type Output = Self;
            #[inline(always)]
            fn bitxor(self, rhs: Self) -> Self {
                Self(unsafe { $xor(self.0, rhs.0) })
            }
        }
    };
}

macro_rules! impl_assign_ops {
    ($t:ty) => {
        impl AddAssign for $t {
            #[inline(always)]
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs;
            }
        }
        impl SubAssign for $t {
            #[inline(always)]
            fn sub_assign(&mut self, rhs: Self) {
                *self = *self - rhs;
            }
        }
        impl BitAndAssign for $t {
            #[inline(always)]
            fn bitand_assign(&mut self, rhs: Self) {
                *self = *self & rhs;
            }
        }
        impl BitOrAssign for $t {
            #[inline(always)]
            fn bitor_assign(&mut self, rhs: Self) {
                *self = *self | rhs;
            }
        }
        impl BitXorAssign for $t {
            #[inline(always)]
            fn bitxor_assign(&mut self, rhs: Self) {
                *self = *self ^ rhs;
            }
        }
    };
}

macro_rules! impl_float_assign_ops {
    ($t:ty) => {
        impl_assign_ops!($t);
        impl MulAssign for $t {
            #[inline(always)]
            fn mul_assign(&mut self, rhs: Self) {
                *self = *self * rhs;
            }
        }
        impl DivAssign for $t {
            #[inline(always)]
            fn div_assign(&mut self, rhs: Self) {
                *self = *self / rhs;
            }
        }
    };
}

macro_rules! impl_neg {
    ($t:ty, $sub:path, $zero:path) => {
        impl Neg for $t {
            type Output = Self;
            #[inline(always)]
            fn neg(self) -> Self {
                Self(unsafe { $sub($zero(), self.0) })
            }
        }
    };
}

macro_rules! impl_index {
    ($t:ty, $elem:ty, $lanes:expr) => {
        impl Index<usize> for $t {
            type Output = $elem;
            #[inline(always)]
            fn index(&self, i: usize) -> &Self::Output {
                assert!(i < $lanes, "index out of bounds");
                unsafe { &*(self as *const Self as *const $elem).add(i) }
            }
        }
        impl IndexMut<usize> for $t {
            #[inline(always)]
            fn index_mut(&mut self, i: usize) -> &mut Self::Output {
                assert!(i < $lanes, "index out of bounds");
                unsafe { &mut *(self as *mut Self as *mut $elem).add(i) }
            }
        }
    };
}


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
    pub fn load(_: crate::Sse41Token, data: &[f32; 4]) -> Self {
        Self(unsafe { _mm_loadu_ps(data.as_ptr()) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: crate::Sse41Token, v: f32) -> Self {
        Self(unsafe { _mm_set1_ps(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: crate::Sse41Token) -> Self {
        Self(unsafe { _mm_setzero_ps() })
    }

    /// Create from array (token-gated)
    #[inline(always)]
    pub fn from_array(token: crate::Sse41Token, arr: [f32; 4]) -> Self {
        Self::load(token, &arr)
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
    /// Returns a mask where each lane is all-1s if equal, all-0s otherwise.
    /// Use with `blend(mask, if_true, if_false)` to select values.
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> Self {
        Self(unsafe { _mm_cmpeq_ps(self.0, other.0) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if not equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(unsafe { _mm_cmpneq_ps(self.0, other.0) })
    }

    /// Lane-wise less-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if self < other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(unsafe { _mm_cmplt_ps(self.0, other.0) })
    }

    /// Lane-wise less-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if self <= other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(unsafe { _mm_cmple_ps(self.0, other.0) })
    }

    /// Lane-wise greater-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if self > other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe { _mm_cmpgt_ps(self.0, other.0) })
    }

    /// Lane-wise greater-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if self >= other, all-0s otherwise.
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
            _mm_castsi128_ps (_mm_xor_si128(as_int, ones))
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
        unsafe {
            Self(_mm_mul_ps(self.log2_lowp().0, _mm_set1_ps(LN2)))
        }
    }

    /// Low-precision natural exponential (e^x).
    ///
    /// Computed as `exp2_lowp(x * log2(e))`. For higher precision, use `exp_midp()`.
    #[inline(always)]
    pub fn exp_lowp(self) -> Self {
        const LOG2_E: f32 = core::f32::consts::LOG2_E;
        unsafe {
            Self(_mm_mul_ps(self.0, _mm_set1_ps(LOG2_E))).exp2_lowp()
        }
    }

    /// Low-precision base-10 logarithm.
    ///
    /// Computed as `log2_lowp(x) / log2(10)`.
    #[inline(always)]
    pub fn log10_lowp(self) -> Self {
        const LOG10_2: f32 = core::f32::consts::LOG10_2; // 1/log2(10)
        unsafe {
            Self(_mm_mul_ps(self.log2_lowp().0, _mm_set1_ps(LOG10_2)))
        }
    }

    /// Low-precision power function (self^n).
    ///
    /// Computed as `exp2_lowp(n * log2_lowp(self))`. For higher precision, use `pow_midp()`.
    /// Note: Only valid for positive self values.
    #[inline(always)]
    pub fn pow_lowp(self, n: f32) -> Self {
        unsafe {
            Self(_mm_mul_ps(self.log2_lowp().0, _mm_set1_ps(n))).exp2_lowp()
        }
    }

    // ========== Mid-Precision Transcendental Operations ==========

    /// Mid-precision base-2 logarithm (~3 ULP max error).
    ///
    /// Uses (a-1)/(a+1) transform with degree-6 odd polynomial.
    /// Suitable for 8-bit, 10-bit, and 12-bit color processing.
    #[inline(always)]
    pub fn log2_midp(self) -> Self {
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
            let a_minus_1 = _mm_sub_ps(a, one);
            let a_plus_1 = _mm_add_ps(a, one);
            let y = _mm_div_ps(a_minus_1, a_plus_1);

            // y^2
            let y2 = _mm_mul_ps(y, y);

            // Polynomial: c0 + y^2*(c1 + y^2*(c2 + y^2*c3))
            let poly = _mm_fmadd_ps(_mm_set1_ps(C3), y2, _mm_set1_ps(C2));
            let poly = _mm_fmadd_ps(poly, y2, _mm_set1_ps(C1));
            let poly = _mm_fmadd_ps(poly, y2, _mm_set1_ps(C0));

            // Result: y * poly + n
            Self(_mm_fmadd_ps(y, poly, n))
        }
    }

    /// Mid-precision base-2 exponential (~140 ULP, ~8e-6 max relative error).
    ///
    /// Uses degree-6 minimax polynomial.
    /// Suitable for 8-bit, 10-bit, and 12-bit color processing.
    #[inline(always)]
    pub fn exp2_midp(self) -> Self {
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
            let x = _mm_max_ps(self.0, _mm_set1_ps(-126.0));
            let x = _mm_min_ps(x, _mm_set1_ps(126.0));

            let xi = _mm_floor_ps(x);
            let xf = _mm_sub_ps(x, xi);

            // Horner's method with 6 coefficients
            let poly = _mm_fmadd_ps(_mm_set1_ps(C6), xf, _mm_set1_ps(C5));
            let poly = _mm_fmadd_ps(poly, xf, _mm_set1_ps(C4));
            let poly = _mm_fmadd_ps(poly, xf, _mm_set1_ps(C3));
            let poly = _mm_fmadd_ps(poly, xf, _mm_set1_ps(C2));
            let poly = _mm_fmadd_ps(poly, xf, _mm_set1_ps(C1));
            let poly = _mm_fmadd_ps(poly, xf, _mm_set1_ps(C0));

            // Scale by 2^integer
            let xi_i32 = _mm_cvtps_epi32(xi);
            let bias = _mm_set1_epi32(127);
            let scale_bits = _mm_slli_epi32::<23>(_mm_add_epi32(xi_i32, bias));
            let scale = _mm_castsi128_ps(scale_bits);

            Self(_mm_mul_ps(poly, scale))
        }
    }

    /// Mid-precision power function (self^n).
    ///
    /// Computed as `exp2_midp(n * log2_midp(self))`.
    /// Achieves 100% exact round-trips for 8-bit, 10-bit, and 12-bit values.
    /// Note: Only valid for positive self values.
    #[inline(always)]
    pub fn pow_midp(self, n: f32) -> Self {
        unsafe {
            Self(_mm_mul_ps(self.log2_midp().0, _mm_set1_ps(n))).exp2_midp()
        }
    }

    /// Mid-precision natural logarithm.
    ///
    /// Computed as `log2_midp(x) * ln(2)`.
    #[inline(always)]
    pub fn ln_midp(self) -> Self {
        const LN2: f32 = core::f32::consts::LN_2;
        unsafe {
            Self(_mm_mul_ps(self.log2_midp().0, _mm_set1_ps(LN2)))
        }
    }

    /// Mid-precision natural exponential (e^x).
    ///
    /// Computed as `exp2_midp(x * log2(e))`.
    #[inline(always)]
    pub fn exp_midp(self) -> Self {
        const LOG2_E: f32 = core::f32::consts::LOG2_E;
        unsafe {
            Self(_mm_mul_ps(self.0, _mm_set1_ps(LOG2_E))).exp2_midp()
        }
    }

    // ========== Cube Root ==========

    /// Low-precision cube root (x^(1/3)).
    ///
    /// Computed via `pow_lowp(x, 1/3)`. For negative inputs, returns NaN.
    /// For higher precision, use `cbrt_midp()`.
    #[inline(always)]
    pub fn cbrt_lowp(self) -> Self {
        self.pow_lowp(1.0 / 3.0)
    }

    /// Mid-precision cube root (x^(1/3)).
    ///
    /// Uses pow_midp with scalar extraction for initial guess + Newton-Raphson.
    /// Handles negative values correctly (returns -cbrt(|x|)).
    #[inline(always)]
    pub fn cbrt_midp(self) -> Self {
        // B1 magic constant for cube root initial approximation
        // B1 = (127 - 127.0/3 - 0.03306235651) * 2^23 = 709958130
        const B1: u32 = 709_958_130;
        const ONE_THIRD: f32 = 1.0 / 3.0;

        unsafe {
            // Extract to array for initial approximation (scalar division by 3)
            let x_arr: [f32; 4] = core::mem::transmute(self.0);
            let mut y_arr = [0.0f32; 4];

            for i in 0..4 {
                let xi = x_arr[i];
                let ui = xi.to_bits();
                let hx = ui & 0x7FFF_FFFF; // abs bits
                // Initial approximation: bits/3 + B1 (always positive)
                let approx = hx / 3 + B1;
                y_arr[i] = f32::from_bits(approx);
            }

            let abs_x = _mm_andnot_ps(_mm_set1_ps(-0.0), self.0);
            let sign_bits = _mm_and_ps(self.0, _mm_set1_ps(-0.0));
            let mut y = core::mem::transmute::<_, _>(y_arr);

            // Newton-Raphson: y = y * (2*x + y^3) / (x + 2*y^3)
            // Two iterations for full f32 precision
            let two = _mm_set1_ps(2.0);

            // Iteration 1
            let y3 = _mm_mul_ps(_mm_mul_ps(y, y), y);
            let num = _mm_fmadd_ps(two, abs_x, y3);
            let den = _mm_fmadd_ps(two, y3, abs_x);
            y = _mm_mul_ps(y, _mm_div_ps(num, den));

            // Iteration 2
            let y3 = _mm_mul_ps(_mm_mul_ps(y, y), y);
            let num = _mm_fmadd_ps(two, abs_x, y3);
            let den = _mm_fmadd_ps(two, y3, abs_x);
            y = _mm_mul_ps(y, _mm_div_ps(num, den));

            // Restore sign
            Self(_mm_or_ps(y, sign_bits))
        }
    }

}

#[cfg(target_arch = "x86_64")]
impl_arithmetic_ops!(f32x4, _mm_add_ps, _mm_sub_ps, _mm_mul_ps, _mm_div_ps);
#[cfg(target_arch = "x86_64")]
impl_float_assign_ops!(f32x4);
#[cfg(target_arch = "x86_64")]
impl_neg!(f32x4, _mm_sub_ps, _mm_setzero_ps);
#[cfg(target_arch = "x86_64")]
impl_bitwise_ops!(f32x4, __m128, _mm_and_ps, _mm_or_ps, _mm_xor_ps);
#[cfg(target_arch = "x86_64")]
impl_index!(f32x4, f32, 4);


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
    pub fn load(_: crate::Sse41Token, data: &[f64; 2]) -> Self {
        Self(unsafe { _mm_loadu_pd(data.as_ptr()) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: crate::Sse41Token, v: f64) -> Self {
        Self(unsafe { _mm_set1_pd(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: crate::Sse41Token) -> Self {
        Self(unsafe { _mm_setzero_pd() })
    }

    /// Create from array (token-gated)
    #[inline(always)]
    pub fn from_array(token: crate::Sse41Token, arr: [f64; 2]) -> Self {
        Self::load(token, &arr)
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
    /// Returns a mask where each lane is all-1s if equal, all-0s otherwise.
    /// Use with `blend(mask, if_true, if_false)` to select values.
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> Self {
        Self(unsafe { _mm_cmpeq_pd(self.0, other.0) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if not equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(unsafe { _mm_cmpneq_pd(self.0, other.0) })
    }

    /// Lane-wise less-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if self < other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(unsafe { _mm_cmplt_pd(self.0, other.0) })
    }

    /// Lane-wise less-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if self <= other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(unsafe { _mm_cmple_pd(self.0, other.0) })
    }

    /// Lane-wise greater-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if self > other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe { _mm_cmpgt_pd(self.0, other.0) })
    }

    /// Lane-wise greater-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if self >= other, all-0s otherwise.
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
            _mm_castsi128_pd (_mm_xor_si128(as_int, ones))
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
            let exp_shifted = _mm_srai_epi64::<52>(exp_bits);

            let mantissa_bits = _mm_sub_epi64(x_bits, _mm_slli_epi64::<52>(exp_shifted));
            let mantissa = _mm_castsi128_pd(mantissa_bits);
            // Convert exponent to f64
            let exp_arr: [i64; 2] = core::mem::transmute(exp_shifted);
            let exp_f64: [f64; 2] = [
exp_arr[0] as f64, exp_arr[1] as f64];
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
f64::from_bits(((xi_arr[0] as i64 + 1023) << 52) as u64), f64::from_bits(((xi_arr[1] as i64 + 1023) << 52) as u64)];
            let scale = _mm_loadu_pd(scale_arr.as_ptr());

            Self(_mm_mul_pd(poly, scale))
        }
    }

    /// Low-precision natural logarithm.
    #[inline(always)]
    pub fn ln_lowp(self) -> Self {
        const LN2: f64 = core::f64::consts::LN_2;
        unsafe {
            Self(_mm_mul_pd(self.log2_lowp().0, _mm_set1_pd(LN2)))
        }
    }

    /// Low-precision natural exponential (e^x).
    #[inline(always)]
    pub fn exp_lowp(self) -> Self {
        const LOG2_E: f64 = core::f64::consts::LOG2_E;
        unsafe {
            Self(_mm_mul_pd(self.0, _mm_set1_pd(LOG2_E))).exp2_lowp()
        }
    }

    /// Low-precision base-10 logarithm.
    #[inline(always)]
    pub fn log10_lowp(self) -> Self {
        const LOG10_2: f64 = core::f64::consts::LOG10_2;
        unsafe {
            Self(_mm_mul_pd(self.log2_lowp().0, _mm_set1_pd(LOG10_2)))
        }
    }

    /// Low-precision power function (self^n).
    #[inline(always)]
    pub fn pow_lowp(self, n: f64) -> Self {
        unsafe {
            Self(_mm_mul_pd(self.log2_lowp().0, _mm_set1_pd(n))).exp2_lowp()
        }
    }

}

#[cfg(target_arch = "x86_64")]
impl_arithmetic_ops!(f64x2, _mm_add_pd, _mm_sub_pd, _mm_mul_pd, _mm_div_pd);
#[cfg(target_arch = "x86_64")]
impl_float_assign_ops!(f64x2);
#[cfg(target_arch = "x86_64")]
impl_neg!(f64x2, _mm_sub_pd, _mm_setzero_pd);
#[cfg(target_arch = "x86_64")]
impl_bitwise_ops!(f64x2, __m128d, _mm_and_pd, _mm_or_pd, _mm_xor_pd);
#[cfg(target_arch = "x86_64")]
impl_index!(f64x2, f64, 2);


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
    pub fn load(_: crate::Sse41Token, data: &[i8; 16]) -> Self {
        Self(unsafe { _mm_loadu_si128(data.as_ptr() as *const __m128i) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: crate::Sse41Token, v: i8) -> Self {
        Self(unsafe { _mm_set1_epi8(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: crate::Sse41Token) -> Self {
        Self(unsafe { _mm_setzero_si128() })
    }

    /// Create from array (token-gated)
    #[inline(always)]
    pub fn from_array(token: crate::Sse41Token, arr: [i8; 16]) -> Self {
        Self::load(token, &arr)
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
    /// Returns a mask where each lane is all-1s if equal, all-0s otherwise.
    /// Use with `blend(mask, if_true, if_false)` to select values.
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> Self {
        Self(unsafe { _mm_cmpeq_epi8(self.0, other.0) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if not equal, all-0s otherwise.
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
    /// Returns a mask where each lane is all-1s if self > other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe { _mm_cmpgt_epi8(self.0, other.0) })
    }

    /// Lane-wise less-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if self < other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(unsafe { _mm_cmpgt_epi8(other.0, self.0) })
    }

    /// Lane-wise greater-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if self >= other, all-0s otherwise.
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
    /// Returns a mask where each lane is all-1s if self <= other, all-0s otherwise.
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

    // ========== Horizontal Operations ==========

    /// Sum all lanes horizontally.
    ///
    /// Returns a scalar containing the sum of all 16 lanes.
    /// Note: This uses a scalar loop. For performance-critical code,
    /// consider keeping values in SIMD until the final reduction.
    #[inline(always)]
    pub fn reduce_add(self) -> i8 {
        self.to_array().iter().copied().fold(0_i8, i8::wrapping_add)
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

}

#[cfg(target_arch = "x86_64")]
impl_int_arithmetic_ops!(i8x16, _mm_add_epi8, _mm_sub_epi8);
#[cfg(target_arch = "x86_64")]
impl_assign_ops!(i8x16);
#[cfg(target_arch = "x86_64")]
impl_bitwise_ops!(i8x16, __m128i, _mm_and_si128, _mm_or_si128, _mm_xor_si128);
#[cfg(target_arch = "x86_64")]
impl_index!(i8x16, i8, 16);


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
    pub fn load(_: crate::Sse41Token, data: &[u8; 16]) -> Self {
        Self(unsafe { _mm_loadu_si128(data.as_ptr() as *const __m128i) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: crate::Sse41Token, v: u8) -> Self {
        Self(unsafe { _mm_set1_epi8(v as i8) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: crate::Sse41Token) -> Self {
        Self(unsafe { _mm_setzero_si128() })
    }

    /// Create from array (token-gated)
    #[inline(always)]
    pub fn from_array(token: crate::Sse41Token, arr: [u8; 16]) -> Self {
        Self::load(token, &arr)
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
    /// Returns a mask where each lane is all-1s if equal, all-0s otherwise.
    /// Use with `blend(mask, if_true, if_false)` to select values.
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> Self {
        Self(unsafe { _mm_cmpeq_epi8(self.0, other.0) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if not equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(unsafe {
            let eq = _mm_cmpeq_epi8(self.0, other.0);
            let ones = _mm_set1_epi8(-1);
            _mm_xor_si128(eq, ones)
        })
    }

    /// Lane-wise greater-than comparison (unsigned).
    ///
    /// Returns a mask where each lane is all-1s if self > other, all-0s otherwise.
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

    /// Lane-wise less-than comparison (unsigned).
    ///
    /// Returns a mask where each lane is all-1s if self < other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        other.simd_gt(self)
    }

    /// Lane-wise greater-than-or-equal comparison (unsigned).
    ///
    /// Returns a mask where each lane is all-1s if self >= other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(unsafe {
            let lt = other.simd_gt(self);
            let ones = _mm_set1_epi8(-1);
            _mm_xor_si128(lt.0, ones)
        })
    }

    /// Lane-wise less-than-or-equal comparison (unsigned).
    ///
    /// Returns a mask where each lane is all-1s if self <= other, all-0s otherwise.
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

    // ========== Horizontal Operations ==========

    /// Sum all lanes horizontally.
    ///
    /// Returns a scalar containing the sum of all 16 lanes.
    /// Note: This uses a scalar loop. For performance-critical code,
    /// consider keeping values in SIMD until the final reduction.
    #[inline(always)]
    pub fn reduce_add(self) -> u8 {
        self.to_array().iter().copied().fold(0_u8, u8::wrapping_add)
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

}

#[cfg(target_arch = "x86_64")]
impl_int_arithmetic_ops!(u8x16, _mm_add_epi8, _mm_sub_epi8);
#[cfg(target_arch = "x86_64")]
impl_assign_ops!(u8x16);
#[cfg(target_arch = "x86_64")]
impl_bitwise_ops!(u8x16, __m128i, _mm_and_si128, _mm_or_si128, _mm_xor_si128);
#[cfg(target_arch = "x86_64")]
impl_index!(u8x16, u8, 16);


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
    pub fn load(_: crate::Sse41Token, data: &[i16; 8]) -> Self {
        Self(unsafe { _mm_loadu_si128(data.as_ptr() as *const __m128i) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: crate::Sse41Token, v: i16) -> Self {
        Self(unsafe { _mm_set1_epi16(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: crate::Sse41Token) -> Self {
        Self(unsafe { _mm_setzero_si128() })
    }

    /// Create from array (token-gated)
    #[inline(always)]
    pub fn from_array(token: crate::Sse41Token, arr: [i16; 8]) -> Self {
        Self::load(token, &arr)
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
    /// Returns a mask where each lane is all-1s if equal, all-0s otherwise.
    /// Use with `blend(mask, if_true, if_false)` to select values.
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> Self {
        Self(unsafe { _mm_cmpeq_epi16(self.0, other.0) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if not equal, all-0s otherwise.
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
    /// Returns a mask where each lane is all-1s if self > other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe { _mm_cmpgt_epi16(self.0, other.0) })
    }

    /// Lane-wise less-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if self < other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(unsafe { _mm_cmpgt_epi16(other.0, self.0) })
    }

    /// Lane-wise greater-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if self >= other, all-0s otherwise.
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
    /// Returns a mask where each lane is all-1s if self <= other, all-0s otherwise.
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

    // ========== Horizontal Operations ==========

    /// Sum all lanes horizontally.
    ///
    /// Returns a scalar containing the sum of all 8 lanes.
    /// Note: This uses a scalar loop. For performance-critical code,
    /// consider keeping values in SIMD until the final reduction.
    #[inline(always)]
    pub fn reduce_add(self) -> i16 {
        self.to_array().iter().copied().fold(0_i16, i16::wrapping_add)
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

}

#[cfg(target_arch = "x86_64")]
impl_int_arithmetic_ops!(i16x8, _mm_add_epi16, _mm_sub_epi16);
#[cfg(target_arch = "x86_64")]
impl_int_mul_op!(i16x8, _mm_mullo_epi16);
#[cfg(target_arch = "x86_64")]
impl_assign_ops!(i16x8);
#[cfg(target_arch = "x86_64")]
impl_bitwise_ops!(i16x8, __m128i, _mm_and_si128, _mm_or_si128, _mm_xor_si128);
#[cfg(target_arch = "x86_64")]
impl_index!(i16x8, i16, 8);


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
    pub fn load(_: crate::Sse41Token, data: &[u16; 8]) -> Self {
        Self(unsafe { _mm_loadu_si128(data.as_ptr() as *const __m128i) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: crate::Sse41Token, v: u16) -> Self {
        Self(unsafe { _mm_set1_epi16(v as i16) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: crate::Sse41Token) -> Self {
        Self(unsafe { _mm_setzero_si128() })
    }

    /// Create from array (token-gated)
    #[inline(always)]
    pub fn from_array(token: crate::Sse41Token, arr: [u16; 8]) -> Self {
        Self::load(token, &arr)
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
    /// Returns a mask where each lane is all-1s if equal, all-0s otherwise.
    /// Use with `blend(mask, if_true, if_false)` to select values.
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> Self {
        Self(unsafe { _mm_cmpeq_epi16(self.0, other.0) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if not equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(unsafe {
            let eq = _mm_cmpeq_epi16(self.0, other.0);
            let ones = _mm_set1_epi16(-1);
            _mm_xor_si128(eq, ones)
        })
    }

    /// Lane-wise greater-than comparison (unsigned).
    ///
    /// Returns a mask where each lane is all-1s if self > other, all-0s otherwise.
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

    /// Lane-wise less-than comparison (unsigned).
    ///
    /// Returns a mask where each lane is all-1s if self < other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        other.simd_gt(self)
    }

    /// Lane-wise greater-than-or-equal comparison (unsigned).
    ///
    /// Returns a mask where each lane is all-1s if self >= other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(unsafe {
            let lt = other.simd_gt(self);
            let ones = _mm_set1_epi16(-1);
            _mm_xor_si128(lt.0, ones)
        })
    }

    /// Lane-wise less-than-or-equal comparison (unsigned).
    ///
    /// Returns a mask where each lane is all-1s if self <= other, all-0s otherwise.
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

    // ========== Horizontal Operations ==========

    /// Sum all lanes horizontally.
    ///
    /// Returns a scalar containing the sum of all 8 lanes.
    /// Note: This uses a scalar loop. For performance-critical code,
    /// consider keeping values in SIMD until the final reduction.
    #[inline(always)]
    pub fn reduce_add(self) -> u16 {
        self.to_array().iter().copied().fold(0_u16, u16::wrapping_add)
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

}

#[cfg(target_arch = "x86_64")]
impl_int_arithmetic_ops!(u16x8, _mm_add_epi16, _mm_sub_epi16);
#[cfg(target_arch = "x86_64")]
impl_int_mul_op!(u16x8, _mm_mullo_epi16);
#[cfg(target_arch = "x86_64")]
impl_assign_ops!(u16x8);
#[cfg(target_arch = "x86_64")]
impl_bitwise_ops!(u16x8, __m128i, _mm_and_si128, _mm_or_si128, _mm_xor_si128);
#[cfg(target_arch = "x86_64")]
impl_index!(u16x8, u16, 8);


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
    pub fn load(_: crate::Sse41Token, data: &[i32; 4]) -> Self {
        Self(unsafe { _mm_loadu_si128(data.as_ptr() as *const __m128i) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: crate::Sse41Token, v: i32) -> Self {
        Self(unsafe { _mm_set1_epi32(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: crate::Sse41Token) -> Self {
        Self(unsafe { _mm_setzero_si128() })
    }

    /// Create from array (token-gated)
    #[inline(always)]
    pub fn from_array(token: crate::Sse41Token, arr: [i32; 4]) -> Self {
        Self::load(token, &arr)
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
    /// Returns a mask where each lane is all-1s if equal, all-0s otherwise.
    /// Use with `blend(mask, if_true, if_false)` to select values.
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> Self {
        Self(unsafe { _mm_cmpeq_epi32(self.0, other.0) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if not equal, all-0s otherwise.
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
    /// Returns a mask where each lane is all-1s if self > other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe { _mm_cmpgt_epi32(self.0, other.0) })
    }

    /// Lane-wise less-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if self < other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(unsafe { _mm_cmpgt_epi32(other.0, self.0) })
    }

    /// Lane-wise greater-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if self >= other, all-0s otherwise.
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
    /// Returns a mask where each lane is all-1s if self <= other, all-0s otherwise.
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

    // ========== Horizontal Operations ==========

    /// Sum all lanes horizontally.
    ///
    /// Returns a scalar containing the sum of all 4 lanes.
    /// Note: This uses a scalar loop. For performance-critical code,
    /// consider keeping values in SIMD until the final reduction.
    #[inline(always)]
    pub fn reduce_add(self) -> i32 {
        self.to_array().iter().copied().fold(0_i32, i32::wrapping_add)
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

}

#[cfg(target_arch = "x86_64")]
impl_int_arithmetic_ops!(i32x4, _mm_add_epi32, _mm_sub_epi32);
#[cfg(target_arch = "x86_64")]
impl_int_mul_op!(i32x4, _mm_mullo_epi32);
#[cfg(target_arch = "x86_64")]
impl_assign_ops!(i32x4);
#[cfg(target_arch = "x86_64")]
impl_bitwise_ops!(i32x4, __m128i, _mm_and_si128, _mm_or_si128, _mm_xor_si128);
#[cfg(target_arch = "x86_64")]
impl_index!(i32x4, i32, 4);


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
    pub fn load(_: crate::Sse41Token, data: &[u32; 4]) -> Self {
        Self(unsafe { _mm_loadu_si128(data.as_ptr() as *const __m128i) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: crate::Sse41Token, v: u32) -> Self {
        Self(unsafe { _mm_set1_epi32(v as i32) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: crate::Sse41Token) -> Self {
        Self(unsafe { _mm_setzero_si128() })
    }

    /// Create from array (token-gated)
    #[inline(always)]
    pub fn from_array(token: crate::Sse41Token, arr: [u32; 4]) -> Self {
        Self::load(token, &arr)
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
    /// Returns a mask where each lane is all-1s if equal, all-0s otherwise.
    /// Use with `blend(mask, if_true, if_false)` to select values.
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> Self {
        Self(unsafe { _mm_cmpeq_epi32(self.0, other.0) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if not equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(unsafe {
            let eq = _mm_cmpeq_epi32(self.0, other.0);
            let ones = _mm_set1_epi32(-1);
            _mm_xor_si128(eq, ones)
        })
    }

    /// Lane-wise greater-than comparison (unsigned).
    ///
    /// Returns a mask where each lane is all-1s if self > other, all-0s otherwise.
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

    /// Lane-wise less-than comparison (unsigned).
    ///
    /// Returns a mask where each lane is all-1s if self < other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        other.simd_gt(self)
    }

    /// Lane-wise greater-than-or-equal comparison (unsigned).
    ///
    /// Returns a mask where each lane is all-1s if self >= other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(unsafe {
            let lt = other.simd_gt(self);
            let ones = _mm_set1_epi32(-1);
            _mm_xor_si128(lt.0, ones)
        })
    }

    /// Lane-wise less-than-or-equal comparison (unsigned).
    ///
    /// Returns a mask where each lane is all-1s if self <= other, all-0s otherwise.
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

    // ========== Horizontal Operations ==========

    /// Sum all lanes horizontally.
    ///
    /// Returns a scalar containing the sum of all 4 lanes.
    /// Note: This uses a scalar loop. For performance-critical code,
    /// consider keeping values in SIMD until the final reduction.
    #[inline(always)]
    pub fn reduce_add(self) -> u32 {
        self.to_array().iter().copied().fold(0_u32, u32::wrapping_add)
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

}

#[cfg(target_arch = "x86_64")]
impl_int_arithmetic_ops!(u32x4, _mm_add_epi32, _mm_sub_epi32);
#[cfg(target_arch = "x86_64")]
impl_int_mul_op!(u32x4, _mm_mullo_epi32);
#[cfg(target_arch = "x86_64")]
impl_assign_ops!(u32x4);
#[cfg(target_arch = "x86_64")]
impl_bitwise_ops!(u32x4, __m128i, _mm_and_si128, _mm_or_si128, _mm_xor_si128);
#[cfg(target_arch = "x86_64")]
impl_index!(u32x4, u32, 4);


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
    pub fn load(_: crate::Sse41Token, data: &[i64; 2]) -> Self {
        Self(unsafe { _mm_loadu_si128(data.as_ptr() as *const __m128i) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: crate::Sse41Token, v: i64) -> Self {
        Self(unsafe { _mm_set1_epi64x(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: crate::Sse41Token) -> Self {
        Self(unsafe { _mm_setzero_si128() })
    }

    /// Create from array (token-gated)
    #[inline(always)]
    pub fn from_array(token: crate::Sse41Token, arr: [i64; 2]) -> Self {
        Self::load(token, &arr)
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

    // ========== Comparisons ==========
    // These return a mask where each lane is all-1s (true) or all-0s (false).
    // Use with `blend()` to select values based on the comparison result.

    /// Lane-wise equality comparison.
    ///
    /// Returns a mask where each lane is all-1s if equal, all-0s otherwise.
    /// Use with `blend(mask, if_true, if_false)` to select values.
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> Self {
        Self(unsafe { _mm_cmpeq_epi64(self.0, other.0) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if not equal, all-0s otherwise.
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
    /// Returns a mask where each lane is all-1s if self > other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe { _mm_cmpgt_epi64(self.0, other.0) })
    }

    /// Lane-wise less-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if self < other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(unsafe { _mm_cmpgt_epi64(other.0, self.0) })
    }

    /// Lane-wise greater-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if self >= other, all-0s otherwise.
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
    /// Returns a mask where each lane is all-1s if self <= other, all-0s otherwise.
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

    // ========== Horizontal Operations ==========

    /// Sum all lanes horizontally.
    ///
    /// Returns a scalar containing the sum of all 2 lanes.
    /// Note: This uses a scalar loop. For performance-critical code,
    /// consider keeping values in SIMD until the final reduction.
    #[inline(always)]
    pub fn reduce_add(self) -> i64 {
        self.to_array().iter().copied().fold(0_i64, i64::wrapping_add)
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

}

#[cfg(target_arch = "x86_64")]
impl_int_arithmetic_ops!(i64x2, _mm_add_epi64, _mm_sub_epi64);
#[cfg(target_arch = "x86_64")]
impl_assign_ops!(i64x2);
#[cfg(target_arch = "x86_64")]
impl_bitwise_ops!(i64x2, __m128i, _mm_and_si128, _mm_or_si128, _mm_xor_si128);
#[cfg(target_arch = "x86_64")]
impl_index!(i64x2, i64, 2);


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
    pub fn load(_: crate::Sse41Token, data: &[u64; 2]) -> Self {
        Self(unsafe { _mm_loadu_si128(data.as_ptr() as *const __m128i) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: crate::Sse41Token, v: u64) -> Self {
        Self(unsafe { _mm_set1_epi64x(v as i64) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: crate::Sse41Token) -> Self {
        Self(unsafe { _mm_setzero_si128() })
    }

    /// Create from array (token-gated)
    #[inline(always)]
    pub fn from_array(token: crate::Sse41Token, arr: [u64; 2]) -> Self {
        Self::load(token, &arr)
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

    // ========== Comparisons ==========
    // These return a mask where each lane is all-1s (true) or all-0s (false).
    // Use with `blend()` to select values based on the comparison result.

    /// Lane-wise equality comparison.
    ///
    /// Returns a mask where each lane is all-1s if equal, all-0s otherwise.
    /// Use with `blend(mask, if_true, if_false)` to select values.
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> Self {
        Self(unsafe { _mm_cmpeq_epi64(self.0, other.0) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if not equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(unsafe {
            let eq = _mm_cmpeq_epi64(self.0, other.0);
            let ones = _mm_set1_epi64x(-1);
            _mm_xor_si128(eq, ones)
        })
    }

    /// Lane-wise greater-than comparison (unsigned).
    ///
    /// Returns a mask where each lane is all-1s if self > other, all-0s otherwise.
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

    /// Lane-wise less-than comparison (unsigned).
    ///
    /// Returns a mask where each lane is all-1s if self < other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        other.simd_gt(self)
    }

    /// Lane-wise greater-than-or-equal comparison (unsigned).
    ///
    /// Returns a mask where each lane is all-1s if self >= other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(unsafe {
            let lt = other.simd_gt(self);
            let ones = _mm_set1_epi64x(-1);
            _mm_xor_si128(lt.0, ones)
        })
    }

    /// Lane-wise less-than-or-equal comparison (unsigned).
    ///
    /// Returns a mask where each lane is all-1s if self <= other, all-0s otherwise.
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

    // ========== Horizontal Operations ==========

    /// Sum all lanes horizontally.
    ///
    /// Returns a scalar containing the sum of all 2 lanes.
    /// Note: This uses a scalar loop. For performance-critical code,
    /// consider keeping values in SIMD until the final reduction.
    #[inline(always)]
    pub fn reduce_add(self) -> u64 {
        self.to_array().iter().copied().fold(0_u64, u64::wrapping_add)
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

}

#[cfg(target_arch = "x86_64")]
impl_int_arithmetic_ops!(u64x2, _mm_add_epi64, _mm_sub_epi64);
#[cfg(target_arch = "x86_64")]
impl_assign_ops!(u64x2);
#[cfg(target_arch = "x86_64")]
impl_bitwise_ops!(u64x2, __m128i, _mm_and_si128, _mm_or_si128, _mm_xor_si128);
#[cfg(target_arch = "x86_64")]
impl_index!(u64x2, u64, 2);


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
    pub fn load(_: crate::Avx2FmaToken, data: &[f32; 8]) -> Self {
        Self(unsafe { _mm256_loadu_ps(data.as_ptr()) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: crate::Avx2FmaToken, v: f32) -> Self {
        Self(unsafe { _mm256_set1_ps(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: crate::Avx2FmaToken) -> Self {
        Self(unsafe { _mm256_setzero_ps() })
    }

    /// Create from array (token-gated)
    #[inline(always)]
    pub fn from_array(token: crate::Avx2FmaToken, arr: [f32; 8]) -> Self {
        Self::load(token, &arr)
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
        Self(unsafe { _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(self.0) })
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
    /// Returns a mask where each lane is all-1s if equal, all-0s otherwise.
    /// Use with `blend(mask, if_true, if_false)` to select values.
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> Self {
        Self(unsafe { _mm256_cmp_ps::<_CMP_EQ_OQ>(self.0, other.0) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if not equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(unsafe { _mm256_cmp_ps::<_CMP_NEQ_OQ>(self.0, other.0) })
    }

    /// Lane-wise less-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if self < other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(unsafe { _mm256_cmp_ps::<_CMP_LT_OQ>(self.0, other.0) })
    }

    /// Lane-wise less-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if self <= other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(unsafe { _mm256_cmp_ps::<_CMP_LE_OQ>(self.0, other.0) })
    }

    /// Lane-wise greater-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if self > other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe { _mm256_cmp_ps::<_CMP_GT_OQ>(self.0, other.0) })
    }

    /// Lane-wise greater-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if self >= other, all-0s otherwise.
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
            _mm256_castsi256_ps (_mm256_xor_si256(as_int, ones))
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
        unsafe {
            Self(_mm256_mul_ps(self.log2_lowp().0, _mm256_set1_ps(LN2)))
        }
    }

    /// Low-precision natural exponential (e^x).
    ///
    /// Computed as `exp2_lowp(x * log2(e))`. For higher precision, use `exp_midp()`.
    #[inline(always)]
    pub fn exp_lowp(self) -> Self {
        const LOG2_E: f32 = core::f32::consts::LOG2_E;
        unsafe {
            Self(_mm256_mul_ps(self.0, _mm256_set1_ps(LOG2_E))).exp2_lowp()
        }
    }

    /// Low-precision base-10 logarithm.
    ///
    /// Computed as `log2_lowp(x) / log2(10)`.
    #[inline(always)]
    pub fn log10_lowp(self) -> Self {
        const LOG10_2: f32 = core::f32::consts::LOG10_2; // 1/log2(10)
        unsafe {
            Self(_mm256_mul_ps(self.log2_lowp().0, _mm256_set1_ps(LOG10_2)))
        }
    }

    /// Low-precision power function (self^n).
    ///
    /// Computed as `exp2_lowp(n * log2_lowp(self))`. For higher precision, use `pow_midp()`.
    /// Note: Only valid for positive self values.
    #[inline(always)]
    pub fn pow_lowp(self, n: f32) -> Self {
        unsafe {
            Self(_mm256_mul_ps(self.log2_lowp().0, _mm256_set1_ps(n))).exp2_lowp()
        }
    }

    // ========== Mid-Precision Transcendental Operations ==========

    /// Mid-precision base-2 logarithm (~3 ULP max error).
    ///
    /// Uses (a-1)/(a+1) transform with degree-6 odd polynomial.
    /// Suitable for 8-bit, 10-bit, and 12-bit color processing.
    #[inline(always)]
    pub fn log2_midp(self) -> Self {
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

    /// Mid-precision base-2 exponential (~140 ULP, ~8e-6 max relative error).
    ///
    /// Uses degree-6 minimax polynomial.
    /// Suitable for 8-bit, 10-bit, and 12-bit color processing.
    #[inline(always)]
    pub fn exp2_midp(self) -> Self {
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

    /// Mid-precision power function (self^n).
    ///
    /// Computed as `exp2_midp(n * log2_midp(self))`.
    /// Achieves 100% exact round-trips for 8-bit, 10-bit, and 12-bit values.
    /// Note: Only valid for positive self values.
    #[inline(always)]
    pub fn pow_midp(self, n: f32) -> Self {
        unsafe {
            Self(_mm256_mul_ps(self.log2_midp().0, _mm256_set1_ps(n))).exp2_midp()
        }
    }

    /// Mid-precision natural logarithm.
    ///
    /// Computed as `log2_midp(x) * ln(2)`.
    #[inline(always)]
    pub fn ln_midp(self) -> Self {
        const LN2: f32 = core::f32::consts::LN_2;
        unsafe {
            Self(_mm256_mul_ps(self.log2_midp().0, _mm256_set1_ps(LN2)))
        }
    }

    /// Mid-precision natural exponential (e^x).
    ///
    /// Computed as `exp2_midp(x * log2(e))`.
    #[inline(always)]
    pub fn exp_midp(self) -> Self {
        const LOG2_E: f32 = core::f32::consts::LOG2_E;
        unsafe {
            Self(_mm256_mul_ps(self.0, _mm256_set1_ps(LOG2_E))).exp2_midp()
        }
    }

    // ========== Cube Root ==========

    /// Low-precision cube root (x^(1/3)).
    ///
    /// Computed via `pow_lowp(x, 1/3)`. For negative inputs, returns NaN.
    /// For higher precision, use `cbrt_midp()`.
    #[inline(always)]
    pub fn cbrt_lowp(self) -> Self {
        self.pow_lowp(1.0 / 3.0)
    }

    /// Mid-precision cube root (x^(1/3)).
    ///
    /// Uses pow_midp with scalar extraction for initial guess + Newton-Raphson.
    /// Handles negative values correctly (returns -cbrt(|x|)).
    #[inline(always)]
    pub fn cbrt_midp(self) -> Self {
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

}

#[cfg(target_arch = "x86_64")]
impl_arithmetic_ops!(f32x8, _mm256_add_ps, _mm256_sub_ps, _mm256_mul_ps, _mm256_div_ps);
#[cfg(target_arch = "x86_64")]
impl_float_assign_ops!(f32x8);
#[cfg(target_arch = "x86_64")]
impl_neg!(f32x8, _mm256_sub_ps, _mm256_setzero_ps);
#[cfg(target_arch = "x86_64")]
impl_bitwise_ops!(f32x8, __m256, _mm256_and_ps, _mm256_or_ps, _mm256_xor_ps);
#[cfg(target_arch = "x86_64")]
impl_index!(f32x8, f32, 8);


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
    pub fn load(_: crate::Avx2FmaToken, data: &[f64; 4]) -> Self {
        Self(unsafe { _mm256_loadu_pd(data.as_ptr()) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: crate::Avx2FmaToken, v: f64) -> Self {
        Self(unsafe { _mm256_set1_pd(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: crate::Avx2FmaToken) -> Self {
        Self(unsafe { _mm256_setzero_pd() })
    }

    /// Create from array (token-gated)
    #[inline(always)]
    pub fn from_array(token: crate::Avx2FmaToken, arr: [f64; 4]) -> Self {
        Self::load(token, &arr)
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
        Self(unsafe { _mm256_round_pd::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(self.0) })
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
    /// Returns a mask where each lane is all-1s if equal, all-0s otherwise.
    /// Use with `blend(mask, if_true, if_false)` to select values.
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> Self {
        Self(unsafe { _mm256_cmp_pd::<_CMP_EQ_OQ>(self.0, other.0) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if not equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(unsafe { _mm256_cmp_pd::<_CMP_NEQ_OQ>(self.0, other.0) })
    }

    /// Lane-wise less-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if self < other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(unsafe { _mm256_cmp_pd::<_CMP_LT_OQ>(self.0, other.0) })
    }

    /// Lane-wise less-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if self <= other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(unsafe { _mm256_cmp_pd::<_CMP_LE_OQ>(self.0, other.0) })
    }

    /// Lane-wise greater-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if self > other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe { _mm256_cmp_pd::<_CMP_GT_OQ>(self.0, other.0) })
    }

    /// Lane-wise greater-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if self >= other, all-0s otherwise.
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
            _mm256_castsi256_pd (_mm256_xor_si256(as_int, ones))
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
            let exp_shifted = _mm256_srai_epi64::<52>(exp_bits);

            let mantissa_bits = _mm256_sub_epi64(x_bits, _mm256_slli_epi64::<52>(exp_shifted));
            let mantissa = _mm256_castsi256_pd(mantissa_bits);
            // Convert exponent to f64
            let exp_arr: [i64; 4] = core::mem::transmute(exp_shifted);
            let exp_f64: [f64; 4] = [
exp_arr[0] as f64, exp_arr[1] as f64, exp_arr[2] as f64, exp_arr[3] as f64];
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
f64::from_bits(((xi_arr[0] as i64 + 1023) << 52) as u64), f64::from_bits(((xi_arr[1] as i64 + 1023) << 52) as u64), f64::from_bits(((xi_arr[2] as i64 + 1023) << 52) as u64), f64::from_bits(((xi_arr[3] as i64 + 1023) << 52) as u64)];
            let scale = _mm256_loadu_pd(scale_arr.as_ptr());

            Self(_mm256_mul_pd(poly, scale))
        }
    }

    /// Low-precision natural logarithm.
    #[inline(always)]
    pub fn ln_lowp(self) -> Self {
        const LN2: f64 = core::f64::consts::LN_2;
        unsafe {
            Self(_mm256_mul_pd(self.log2_lowp().0, _mm256_set1_pd(LN2)))
        }
    }

    /// Low-precision natural exponential (e^x).
    #[inline(always)]
    pub fn exp_lowp(self) -> Self {
        const LOG2_E: f64 = core::f64::consts::LOG2_E;
        unsafe {
            Self(_mm256_mul_pd(self.0, _mm256_set1_pd(LOG2_E))).exp2_lowp()
        }
    }

    /// Low-precision base-10 logarithm.
    #[inline(always)]
    pub fn log10_lowp(self) -> Self {
        const LOG10_2: f64 = core::f64::consts::LOG10_2;
        unsafe {
            Self(_mm256_mul_pd(self.log2_lowp().0, _mm256_set1_pd(LOG10_2)))
        }
    }

    /// Low-precision power function (self^n).
    #[inline(always)]
    pub fn pow_lowp(self, n: f64) -> Self {
        unsafe {
            Self(_mm256_mul_pd(self.log2_lowp().0, _mm256_set1_pd(n))).exp2_lowp()
        }
    }

}

#[cfg(target_arch = "x86_64")]
impl_arithmetic_ops!(f64x4, _mm256_add_pd, _mm256_sub_pd, _mm256_mul_pd, _mm256_div_pd);
#[cfg(target_arch = "x86_64")]
impl_float_assign_ops!(f64x4);
#[cfg(target_arch = "x86_64")]
impl_neg!(f64x4, _mm256_sub_pd, _mm256_setzero_pd);
#[cfg(target_arch = "x86_64")]
impl_bitwise_ops!(f64x4, __m256d, _mm256_and_pd, _mm256_or_pd, _mm256_xor_pd);
#[cfg(target_arch = "x86_64")]
impl_index!(f64x4, f64, 4);


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
    pub fn load(_: crate::Avx2FmaToken, data: &[i8; 32]) -> Self {
        Self(unsafe { _mm256_loadu_si256(data.as_ptr() as *const __m256i) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: crate::Avx2FmaToken, v: i8) -> Self {
        Self(unsafe { _mm256_set1_epi8(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: crate::Avx2FmaToken) -> Self {
        Self(unsafe { _mm256_setzero_si256() })
    }

    /// Create from array (token-gated)
    #[inline(always)]
    pub fn from_array(token: crate::Avx2FmaToken, arr: [i8; 32]) -> Self {
        Self::load(token, &arr)
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
    /// Returns a mask where each lane is all-1s if equal, all-0s otherwise.
    /// Use with `blend(mask, if_true, if_false)` to select values.
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> Self {
        Self(unsafe { _mm256_cmpeq_epi8(self.0, other.0) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if not equal, all-0s otherwise.
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
    /// Returns a mask where each lane is all-1s if self > other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe { _mm256_cmpgt_epi8(self.0, other.0) })
    }

    /// Lane-wise less-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if self < other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(unsafe { _mm256_cmpgt_epi8(other.0, self.0) })
    }

    /// Lane-wise greater-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if self >= other, all-0s otherwise.
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
    /// Returns a mask where each lane is all-1s if self <= other, all-0s otherwise.
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

    // ========== Horizontal Operations ==========

    /// Sum all lanes horizontally.
    ///
    /// Returns a scalar containing the sum of all 32 lanes.
    /// Note: This uses a scalar loop. For performance-critical code,
    /// consider keeping values in SIMD until the final reduction.
    #[inline(always)]
    pub fn reduce_add(self) -> i8 {
        self.to_array().iter().copied().fold(0_i8, i8::wrapping_add)
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

}

#[cfg(target_arch = "x86_64")]
impl_int_arithmetic_ops!(i8x32, _mm256_add_epi8, _mm256_sub_epi8);
#[cfg(target_arch = "x86_64")]
impl_assign_ops!(i8x32);
#[cfg(target_arch = "x86_64")]
impl_bitwise_ops!(i8x32, __m256i, _mm256_and_si256, _mm256_or_si256, _mm256_xor_si256);
#[cfg(target_arch = "x86_64")]
impl_index!(i8x32, i8, 32);


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
    pub fn load(_: crate::Avx2FmaToken, data: &[u8; 32]) -> Self {
        Self(unsafe { _mm256_loadu_si256(data.as_ptr() as *const __m256i) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: crate::Avx2FmaToken, v: u8) -> Self {
        Self(unsafe { _mm256_set1_epi8(v as i8) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: crate::Avx2FmaToken) -> Self {
        Self(unsafe { _mm256_setzero_si256() })
    }

    /// Create from array (token-gated)
    #[inline(always)]
    pub fn from_array(token: crate::Avx2FmaToken, arr: [u8; 32]) -> Self {
        Self::load(token, &arr)
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
    /// Returns a mask where each lane is all-1s if equal, all-0s otherwise.
    /// Use with `blend(mask, if_true, if_false)` to select values.
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> Self {
        Self(unsafe { _mm256_cmpeq_epi8(self.0, other.0) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if not equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(unsafe {
            let eq = _mm256_cmpeq_epi8(self.0, other.0);
            let ones = _mm256_set1_epi8(-1);
            _mm256_xor_si256(eq, ones)
        })
    }

    /// Lane-wise greater-than comparison (unsigned).
    ///
    /// Returns a mask where each lane is all-1s if self > other, all-0s otherwise.
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

    /// Lane-wise less-than comparison (unsigned).
    ///
    /// Returns a mask where each lane is all-1s if self < other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        other.simd_gt(self)
    }

    /// Lane-wise greater-than-or-equal comparison (unsigned).
    ///
    /// Returns a mask where each lane is all-1s if self >= other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(unsafe {
            let lt = other.simd_gt(self);
            let ones = _mm256_set1_epi8(-1);
            _mm256_xor_si256(lt.0, ones)
        })
    }

    /// Lane-wise less-than-or-equal comparison (unsigned).
    ///
    /// Returns a mask where each lane is all-1s if self <= other, all-0s otherwise.
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

    // ========== Horizontal Operations ==========

    /// Sum all lanes horizontally.
    ///
    /// Returns a scalar containing the sum of all 32 lanes.
    /// Note: This uses a scalar loop. For performance-critical code,
    /// consider keeping values in SIMD until the final reduction.
    #[inline(always)]
    pub fn reduce_add(self) -> u8 {
        self.to_array().iter().copied().fold(0_u8, u8::wrapping_add)
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

}

#[cfg(target_arch = "x86_64")]
impl_int_arithmetic_ops!(u8x32, _mm256_add_epi8, _mm256_sub_epi8);
#[cfg(target_arch = "x86_64")]
impl_assign_ops!(u8x32);
#[cfg(target_arch = "x86_64")]
impl_bitwise_ops!(u8x32, __m256i, _mm256_and_si256, _mm256_or_si256, _mm256_xor_si256);
#[cfg(target_arch = "x86_64")]
impl_index!(u8x32, u8, 32);


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
    pub fn load(_: crate::Avx2FmaToken, data: &[i16; 16]) -> Self {
        Self(unsafe { _mm256_loadu_si256(data.as_ptr() as *const __m256i) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: crate::Avx2FmaToken, v: i16) -> Self {
        Self(unsafe { _mm256_set1_epi16(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: crate::Avx2FmaToken) -> Self {
        Self(unsafe { _mm256_setzero_si256() })
    }

    /// Create from array (token-gated)
    #[inline(always)]
    pub fn from_array(token: crate::Avx2FmaToken, arr: [i16; 16]) -> Self {
        Self::load(token, &arr)
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
    /// Returns a mask where each lane is all-1s if equal, all-0s otherwise.
    /// Use with `blend(mask, if_true, if_false)` to select values.
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> Self {
        Self(unsafe { _mm256_cmpeq_epi16(self.0, other.0) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if not equal, all-0s otherwise.
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
    /// Returns a mask where each lane is all-1s if self > other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe { _mm256_cmpgt_epi16(self.0, other.0) })
    }

    /// Lane-wise less-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if self < other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(unsafe { _mm256_cmpgt_epi16(other.0, self.0) })
    }

    /// Lane-wise greater-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if self >= other, all-0s otherwise.
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
    /// Returns a mask where each lane is all-1s if self <= other, all-0s otherwise.
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

    // ========== Horizontal Operations ==========

    /// Sum all lanes horizontally.
    ///
    /// Returns a scalar containing the sum of all 16 lanes.
    /// Note: This uses a scalar loop. For performance-critical code,
    /// consider keeping values in SIMD until the final reduction.
    #[inline(always)]
    pub fn reduce_add(self) -> i16 {
        self.to_array().iter().copied().fold(0_i16, i16::wrapping_add)
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

}

#[cfg(target_arch = "x86_64")]
impl_int_arithmetic_ops!(i16x16, _mm256_add_epi16, _mm256_sub_epi16);
#[cfg(target_arch = "x86_64")]
impl_int_mul_op!(i16x16, _mm256_mullo_epi16);
#[cfg(target_arch = "x86_64")]
impl_assign_ops!(i16x16);
#[cfg(target_arch = "x86_64")]
impl_bitwise_ops!(i16x16, __m256i, _mm256_and_si256, _mm256_or_si256, _mm256_xor_si256);
#[cfg(target_arch = "x86_64")]
impl_index!(i16x16, i16, 16);


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
    pub fn load(_: crate::Avx2FmaToken, data: &[u16; 16]) -> Self {
        Self(unsafe { _mm256_loadu_si256(data.as_ptr() as *const __m256i) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: crate::Avx2FmaToken, v: u16) -> Self {
        Self(unsafe { _mm256_set1_epi16(v as i16) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: crate::Avx2FmaToken) -> Self {
        Self(unsafe { _mm256_setzero_si256() })
    }

    /// Create from array (token-gated)
    #[inline(always)]
    pub fn from_array(token: crate::Avx2FmaToken, arr: [u16; 16]) -> Self {
        Self::load(token, &arr)
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
    /// Returns a mask where each lane is all-1s if equal, all-0s otherwise.
    /// Use with `blend(mask, if_true, if_false)` to select values.
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> Self {
        Self(unsafe { _mm256_cmpeq_epi16(self.0, other.0) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if not equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(unsafe {
            let eq = _mm256_cmpeq_epi16(self.0, other.0);
            let ones = _mm256_set1_epi16(-1);
            _mm256_xor_si256(eq, ones)
        })
    }

    /// Lane-wise greater-than comparison (unsigned).
    ///
    /// Returns a mask where each lane is all-1s if self > other, all-0s otherwise.
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

    /// Lane-wise less-than comparison (unsigned).
    ///
    /// Returns a mask where each lane is all-1s if self < other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        other.simd_gt(self)
    }

    /// Lane-wise greater-than-or-equal comparison (unsigned).
    ///
    /// Returns a mask where each lane is all-1s if self >= other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(unsafe {
            let lt = other.simd_gt(self);
            let ones = _mm256_set1_epi16(-1);
            _mm256_xor_si256(lt.0, ones)
        })
    }

    /// Lane-wise less-than-or-equal comparison (unsigned).
    ///
    /// Returns a mask where each lane is all-1s if self <= other, all-0s otherwise.
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

    // ========== Horizontal Operations ==========

    /// Sum all lanes horizontally.
    ///
    /// Returns a scalar containing the sum of all 16 lanes.
    /// Note: This uses a scalar loop. For performance-critical code,
    /// consider keeping values in SIMD until the final reduction.
    #[inline(always)]
    pub fn reduce_add(self) -> u16 {
        self.to_array().iter().copied().fold(0_u16, u16::wrapping_add)
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

}

#[cfg(target_arch = "x86_64")]
impl_int_arithmetic_ops!(u16x16, _mm256_add_epi16, _mm256_sub_epi16);
#[cfg(target_arch = "x86_64")]
impl_int_mul_op!(u16x16, _mm256_mullo_epi16);
#[cfg(target_arch = "x86_64")]
impl_assign_ops!(u16x16);
#[cfg(target_arch = "x86_64")]
impl_bitwise_ops!(u16x16, __m256i, _mm256_and_si256, _mm256_or_si256, _mm256_xor_si256);
#[cfg(target_arch = "x86_64")]
impl_index!(u16x16, u16, 16);


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
    pub fn load(_: crate::Avx2FmaToken, data: &[i32; 8]) -> Self {
        Self(unsafe { _mm256_loadu_si256(data.as_ptr() as *const __m256i) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: crate::Avx2FmaToken, v: i32) -> Self {
        Self(unsafe { _mm256_set1_epi32(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: crate::Avx2FmaToken) -> Self {
        Self(unsafe { _mm256_setzero_si256() })
    }

    /// Create from array (token-gated)
    #[inline(always)]
    pub fn from_array(token: crate::Avx2FmaToken, arr: [i32; 8]) -> Self {
        Self::load(token, &arr)
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
    /// Returns a mask where each lane is all-1s if equal, all-0s otherwise.
    /// Use with `blend(mask, if_true, if_false)` to select values.
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> Self {
        Self(unsafe { _mm256_cmpeq_epi32(self.0, other.0) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if not equal, all-0s otherwise.
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
    /// Returns a mask where each lane is all-1s if self > other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe { _mm256_cmpgt_epi32(self.0, other.0) })
    }

    /// Lane-wise less-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if self < other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(unsafe { _mm256_cmpgt_epi32(other.0, self.0) })
    }

    /// Lane-wise greater-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if self >= other, all-0s otherwise.
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
    /// Returns a mask where each lane is all-1s if self <= other, all-0s otherwise.
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

    // ========== Horizontal Operations ==========

    /// Sum all lanes horizontally.
    ///
    /// Returns a scalar containing the sum of all 8 lanes.
    /// Note: This uses a scalar loop. For performance-critical code,
    /// consider keeping values in SIMD until the final reduction.
    #[inline(always)]
    pub fn reduce_add(self) -> i32 {
        self.to_array().iter().copied().fold(0_i32, i32::wrapping_add)
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

}

#[cfg(target_arch = "x86_64")]
impl_int_arithmetic_ops!(i32x8, _mm256_add_epi32, _mm256_sub_epi32);
#[cfg(target_arch = "x86_64")]
impl_int_mul_op!(i32x8, _mm256_mullo_epi32);
#[cfg(target_arch = "x86_64")]
impl_assign_ops!(i32x8);
#[cfg(target_arch = "x86_64")]
impl_bitwise_ops!(i32x8, __m256i, _mm256_and_si256, _mm256_or_si256, _mm256_xor_si256);
#[cfg(target_arch = "x86_64")]
impl_index!(i32x8, i32, 8);


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
    pub fn load(_: crate::Avx2FmaToken, data: &[u32; 8]) -> Self {
        Self(unsafe { _mm256_loadu_si256(data.as_ptr() as *const __m256i) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: crate::Avx2FmaToken, v: u32) -> Self {
        Self(unsafe { _mm256_set1_epi32(v as i32) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: crate::Avx2FmaToken) -> Self {
        Self(unsafe { _mm256_setzero_si256() })
    }

    /// Create from array (token-gated)
    #[inline(always)]
    pub fn from_array(token: crate::Avx2FmaToken, arr: [u32; 8]) -> Self {
        Self::load(token, &arr)
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
    /// Returns a mask where each lane is all-1s if equal, all-0s otherwise.
    /// Use with `blend(mask, if_true, if_false)` to select values.
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> Self {
        Self(unsafe { _mm256_cmpeq_epi32(self.0, other.0) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if not equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(unsafe {
            let eq = _mm256_cmpeq_epi32(self.0, other.0);
            let ones = _mm256_set1_epi32(-1);
            _mm256_xor_si256(eq, ones)
        })
    }

    /// Lane-wise greater-than comparison (unsigned).
    ///
    /// Returns a mask where each lane is all-1s if self > other, all-0s otherwise.
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

    /// Lane-wise less-than comparison (unsigned).
    ///
    /// Returns a mask where each lane is all-1s if self < other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        other.simd_gt(self)
    }

    /// Lane-wise greater-than-or-equal comparison (unsigned).
    ///
    /// Returns a mask where each lane is all-1s if self >= other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(unsafe {
            let lt = other.simd_gt(self);
            let ones = _mm256_set1_epi32(-1);
            _mm256_xor_si256(lt.0, ones)
        })
    }

    /// Lane-wise less-than-or-equal comparison (unsigned).
    ///
    /// Returns a mask where each lane is all-1s if self <= other, all-0s otherwise.
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

    // ========== Horizontal Operations ==========

    /// Sum all lanes horizontally.
    ///
    /// Returns a scalar containing the sum of all 8 lanes.
    /// Note: This uses a scalar loop. For performance-critical code,
    /// consider keeping values in SIMD until the final reduction.
    #[inline(always)]
    pub fn reduce_add(self) -> u32 {
        self.to_array().iter().copied().fold(0_u32, u32::wrapping_add)
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

}

#[cfg(target_arch = "x86_64")]
impl_int_arithmetic_ops!(u32x8, _mm256_add_epi32, _mm256_sub_epi32);
#[cfg(target_arch = "x86_64")]
impl_int_mul_op!(u32x8, _mm256_mullo_epi32);
#[cfg(target_arch = "x86_64")]
impl_assign_ops!(u32x8);
#[cfg(target_arch = "x86_64")]
impl_bitwise_ops!(u32x8, __m256i, _mm256_and_si256, _mm256_or_si256, _mm256_xor_si256);
#[cfg(target_arch = "x86_64")]
impl_index!(u32x8, u32, 8);


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
    pub fn load(_: crate::Avx2FmaToken, data: &[i64; 4]) -> Self {
        Self(unsafe { _mm256_loadu_si256(data.as_ptr() as *const __m256i) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: crate::Avx2FmaToken, v: i64) -> Self {
        Self(unsafe { _mm256_set1_epi64x(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: crate::Avx2FmaToken) -> Self {
        Self(unsafe { _mm256_setzero_si256() })
    }

    /// Create from array (token-gated)
    #[inline(always)]
    pub fn from_array(token: crate::Avx2FmaToken, arr: [i64; 4]) -> Self {
        Self::load(token, &arr)
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

    // ========== Comparisons ==========
    // These return a mask where each lane is all-1s (true) or all-0s (false).
    // Use with `blend()` to select values based on the comparison result.

    /// Lane-wise equality comparison.
    ///
    /// Returns a mask where each lane is all-1s if equal, all-0s otherwise.
    /// Use with `blend(mask, if_true, if_false)` to select values.
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> Self {
        Self(unsafe { _mm256_cmpeq_epi64(self.0, other.0) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if not equal, all-0s otherwise.
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
    /// Returns a mask where each lane is all-1s if self > other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe { _mm256_cmpgt_epi64(self.0, other.0) })
    }

    /// Lane-wise less-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if self < other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(unsafe { _mm256_cmpgt_epi64(other.0, self.0) })
    }

    /// Lane-wise greater-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if self >= other, all-0s otherwise.
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
    /// Returns a mask where each lane is all-1s if self <= other, all-0s otherwise.
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

    // ========== Horizontal Operations ==========

    /// Sum all lanes horizontally.
    ///
    /// Returns a scalar containing the sum of all 4 lanes.
    /// Note: This uses a scalar loop. For performance-critical code,
    /// consider keeping values in SIMD until the final reduction.
    #[inline(always)]
    pub fn reduce_add(self) -> i64 {
        self.to_array().iter().copied().fold(0_i64, i64::wrapping_add)
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

}

#[cfg(target_arch = "x86_64")]
impl_int_arithmetic_ops!(i64x4, _mm256_add_epi64, _mm256_sub_epi64);
#[cfg(target_arch = "x86_64")]
impl_assign_ops!(i64x4);
#[cfg(target_arch = "x86_64")]
impl_bitwise_ops!(i64x4, __m256i, _mm256_and_si256, _mm256_or_si256, _mm256_xor_si256);
#[cfg(target_arch = "x86_64")]
impl_index!(i64x4, i64, 4);


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
    pub fn load(_: crate::Avx2FmaToken, data: &[u64; 4]) -> Self {
        Self(unsafe { _mm256_loadu_si256(data.as_ptr() as *const __m256i) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: crate::Avx2FmaToken, v: u64) -> Self {
        Self(unsafe { _mm256_set1_epi64x(v as i64) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: crate::Avx2FmaToken) -> Self {
        Self(unsafe { _mm256_setzero_si256() })
    }

    /// Create from array (token-gated)
    #[inline(always)]
    pub fn from_array(token: crate::Avx2FmaToken, arr: [u64; 4]) -> Self {
        Self::load(token, &arr)
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

    // ========== Comparisons ==========
    // These return a mask where each lane is all-1s (true) or all-0s (false).
    // Use with `blend()` to select values based on the comparison result.

    /// Lane-wise equality comparison.
    ///
    /// Returns a mask where each lane is all-1s if equal, all-0s otherwise.
    /// Use with `blend(mask, if_true, if_false)` to select values.
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> Self {
        Self(unsafe { _mm256_cmpeq_epi64(self.0, other.0) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if not equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(unsafe {
            let eq = _mm256_cmpeq_epi64(self.0, other.0);
            let ones = _mm256_set1_epi64x(-1);
            _mm256_xor_si256(eq, ones)
        })
    }

    /// Lane-wise greater-than comparison (unsigned).
    ///
    /// Returns a mask where each lane is all-1s if self > other, all-0s otherwise.
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

    /// Lane-wise less-than comparison (unsigned).
    ///
    /// Returns a mask where each lane is all-1s if self < other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        other.simd_gt(self)
    }

    /// Lane-wise greater-than-or-equal comparison (unsigned).
    ///
    /// Returns a mask where each lane is all-1s if self >= other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(unsafe {
            let lt = other.simd_gt(self);
            let ones = _mm256_set1_epi64x(-1);
            _mm256_xor_si256(lt.0, ones)
        })
    }

    /// Lane-wise less-than-or-equal comparison (unsigned).
    ///
    /// Returns a mask where each lane is all-1s if self <= other, all-0s otherwise.
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

    // ========== Horizontal Operations ==========

    /// Sum all lanes horizontally.
    ///
    /// Returns a scalar containing the sum of all 4 lanes.
    /// Note: This uses a scalar loop. For performance-critical code,
    /// consider keeping values in SIMD until the final reduction.
    #[inline(always)]
    pub fn reduce_add(self) -> u64 {
        self.to_array().iter().copied().fold(0_u64, u64::wrapping_add)
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

}

#[cfg(target_arch = "x86_64")]
impl_int_arithmetic_ops!(u64x4, _mm256_add_epi64, _mm256_sub_epi64);
#[cfg(target_arch = "x86_64")]
impl_assign_ops!(u64x4);
#[cfg(target_arch = "x86_64")]
impl_bitwise_ops!(u64x4, __m256i, _mm256_and_si256, _mm256_or_si256, _mm256_xor_si256);
#[cfg(target_arch = "x86_64")]
impl_index!(u64x4, u64, 4);


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


// ============================================================================
// f32x16 - 16 x f32 (512-bit)
// ============================================================================

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct f32x16(__m512);

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl f32x16 {
    pub const LANES: usize = 16;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: crate::X64V4Token, data: &[f32; 16]) -> Self {
        Self(unsafe { _mm512_loadu_ps(data.as_ptr()) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: crate::X64V4Token, v: f32) -> Self {
        Self(unsafe { _mm512_set1_ps(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: crate::X64V4Token) -> Self {
        Self(unsafe { _mm512_setzero_ps() })
    }

    /// Create from array (token-gated)
    #[inline(always)]
    pub fn from_array(token: crate::X64V4Token, arr: [f32; 16]) -> Self {
        Self::load(token, &arr)
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
            _mm512_castsi512_ps (_mm512_xor_si512(as_int, ones))
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

    /// Mid-precision base-2 logarithm (~3 ULP max error).
    ///
    /// Uses (a-1)/(a+1) transform with degree-6 odd polynomial.
    /// Suitable for 8-bit, 10-bit, and 12-bit color processing.
    #[inline(always)]
    pub fn log2_midp(self) -> Self {
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

    /// Mid-precision base-2 exponential (~140 ULP, ~8e-6 max relative error).
    ///
    /// Uses degree-6 minimax polynomial.
    /// Suitable for 8-bit, 10-bit, and 12-bit color processing.
    #[inline(always)]
    pub fn exp2_midp(self) -> Self {
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

    /// Mid-precision power function (self^n).
    ///
    /// Computed as `exp2_midp(n * log2_midp(self))`.
    /// Achieves 100% exact round-trips for 8-bit, 10-bit, and 12-bit values.
    /// Note: Only valid for positive self values.
    #[inline(always)]
    pub fn pow_midp(self, n: f32) -> Self {
        unsafe {
            Self(_mm512_mul_ps(self.log2_midp().0, _mm512_set1_ps(n))).exp2_midp()
        }
    }

    /// Mid-precision natural logarithm.
    ///
    /// Computed as `log2_midp(x) * ln(2)`.
    #[inline(always)]
    pub fn ln_midp(self) -> Self {
        const LN2: f32 = core::f32::consts::LN_2;
        unsafe {
            Self(_mm512_mul_ps(self.log2_midp().0, _mm512_set1_ps(LN2)))
        }
    }

    /// Mid-precision natural exponential (e^x).
    ///
    /// Computed as `exp2_midp(x * log2(e))`.
    #[inline(always)]
    pub fn exp_midp(self) -> Self {
        const LOG2_E: f32 = core::f32::consts::LOG2_E;
        unsafe {
            Self(_mm512_mul_ps(self.0, _mm512_set1_ps(LOG2_E))).exp2_midp()
        }
    }

    // ========== Cube Root ==========

    /// Low-precision cube root (x^(1/3)).
    ///
    /// Computed via `pow_lowp(x, 1/3)`. For negative inputs, returns NaN.
    /// For higher precision, use `cbrt_midp()`.
    #[inline(always)]
    pub fn cbrt_lowp(self) -> Self {
        self.pow_lowp(1.0 / 3.0)
    }

    /// Mid-precision cube root (x^(1/3)).
    ///
    /// Uses pow_midp with scalar extraction for initial guess + Newton-Raphson.
    /// Handles negative values correctly (returns -cbrt(|x|)).
    #[inline(always)]
    pub fn cbrt_midp(self) -> Self {
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

}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_arithmetic_ops!(f32x16, _mm512_add_ps, _mm512_sub_ps, _mm512_mul_ps, _mm512_div_ps);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_float_assign_ops!(f32x16);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_neg!(f32x16, _mm512_sub_ps, _mm512_setzero_ps);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_bitwise_ops!(f32x16, __m512, _mm512_and_ps, _mm512_or_ps, _mm512_xor_ps);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_index!(f32x16, f32, 16);


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

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl f64x8 {
    pub const LANES: usize = 8;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: crate::X64V4Token, data: &[f64; 8]) -> Self {
        Self(unsafe { _mm512_loadu_pd(data.as_ptr()) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: crate::X64V4Token, v: f64) -> Self {
        Self(unsafe { _mm512_set1_pd(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: crate::X64V4Token) -> Self {
        Self(unsafe { _mm512_setzero_pd() })
    }

    /// Create from array (token-gated)
    #[inline(always)]
    pub fn from_array(token: crate::X64V4Token, arr: [f64; 8]) -> Self {
        Self::load(token, &arr)
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
            _mm512_castsi512_pd (_mm512_xor_si512(as_int, ones))
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
impl_arithmetic_ops!(f64x8, _mm512_add_pd, _mm512_sub_pd, _mm512_mul_pd, _mm512_div_pd);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_float_assign_ops!(f64x8);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_neg!(f64x8, _mm512_sub_pd, _mm512_setzero_pd);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_bitwise_ops!(f64x8, __m512d, _mm512_and_pd, _mm512_or_pd, _mm512_xor_pd);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_index!(f64x8, f64, 8);


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

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl i8x64 {
    pub const LANES: usize = 64;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: crate::X64V4Token, data: &[i8; 64]) -> Self {
        Self(unsafe { _mm512_loadu_si512(data.as_ptr() as *const __m512i) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: crate::X64V4Token, v: i8) -> Self {
        Self(unsafe { _mm512_set1_epi8(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: crate::X64V4Token) -> Self {
        Self(unsafe { _mm512_setzero_si512() })
    }

    /// Create from array (token-gated)
    #[inline(always)]
    pub fn from_array(token: crate::X64V4Token, arr: [i8; 64]) -> Self {
        Self::load(token, &arr)
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
        self.to_array().iter().copied().fold(0_i8, i8::wrapping_add)
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
impl_int_arithmetic_ops!(i8x64, _mm512_add_epi8, _mm512_sub_epi8);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_assign_ops!(i8x64);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_bitwise_ops!(i8x64, __m512i, _mm512_and_si512, _mm512_or_si512, _mm512_xor_si512);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_index!(i8x64, i8, 64);


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

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl u8x64 {
    pub const LANES: usize = 64;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: crate::X64V4Token, data: &[u8; 64]) -> Self {
        Self(unsafe { _mm512_loadu_si512(data.as_ptr() as *const __m512i) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: crate::X64V4Token, v: u8) -> Self {
        Self(unsafe { _mm512_set1_epi8(v as i8) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: crate::X64V4Token) -> Self {
        Self(unsafe { _mm512_setzero_si512() })
    }

    /// Create from array (token-gated)
    #[inline(always)]
    pub fn from_array(token: crate::X64V4Token, arr: [u8; 64]) -> Self {
        Self::load(token, &arr)
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
        self.to_array().iter().copied().fold(0_u8, u8::wrapping_add)
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
impl_int_arithmetic_ops!(u8x64, _mm512_add_epi8, _mm512_sub_epi8);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_assign_ops!(u8x64);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_bitwise_ops!(u8x64, __m512i, _mm512_and_si512, _mm512_or_si512, _mm512_xor_si512);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_index!(u8x64, u8, 64);


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

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl i16x32 {
    pub const LANES: usize = 32;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: crate::X64V4Token, data: &[i16; 32]) -> Self {
        Self(unsafe { _mm512_loadu_si512(data.as_ptr() as *const __m512i) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: crate::X64V4Token, v: i16) -> Self {
        Self(unsafe { _mm512_set1_epi16(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: crate::X64V4Token) -> Self {
        Self(unsafe { _mm512_setzero_si512() })
    }

    /// Create from array (token-gated)
    #[inline(always)]
    pub fn from_array(token: crate::X64V4Token, arr: [i16; 32]) -> Self {
        Self::load(token, &arr)
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
        self.to_array().iter().copied().fold(0_i16, i16::wrapping_add)
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
impl_int_arithmetic_ops!(i16x32, _mm512_add_epi16, _mm512_sub_epi16);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_int_mul_op!(i16x32, _mm512_mullo_epi16);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_assign_ops!(i16x32);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_bitwise_ops!(i16x32, __m512i, _mm512_and_si512, _mm512_or_si512, _mm512_xor_si512);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_index!(i16x32, i16, 32);


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

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl u16x32 {
    pub const LANES: usize = 32;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: crate::X64V4Token, data: &[u16; 32]) -> Self {
        Self(unsafe { _mm512_loadu_si512(data.as_ptr() as *const __m512i) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: crate::X64V4Token, v: u16) -> Self {
        Self(unsafe { _mm512_set1_epi16(v as i16) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: crate::X64V4Token) -> Self {
        Self(unsafe { _mm512_setzero_si512() })
    }

    /// Create from array (token-gated)
    #[inline(always)]
    pub fn from_array(token: crate::X64V4Token, arr: [u16; 32]) -> Self {
        Self::load(token, &arr)
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
        self.to_array().iter().copied().fold(0_u16, u16::wrapping_add)
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
impl_int_arithmetic_ops!(u16x32, _mm512_add_epi16, _mm512_sub_epi16);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_int_mul_op!(u16x32, _mm512_mullo_epi16);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_assign_ops!(u16x32);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_bitwise_ops!(u16x32, __m512i, _mm512_and_si512, _mm512_or_si512, _mm512_xor_si512);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_index!(u16x32, u16, 32);


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

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl i32x16 {
    pub const LANES: usize = 16;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: crate::X64V4Token, data: &[i32; 16]) -> Self {
        Self(unsafe { _mm512_loadu_si512(data.as_ptr() as *const __m512i) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: crate::X64V4Token, v: i32) -> Self {
        Self(unsafe { _mm512_set1_epi32(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: crate::X64V4Token) -> Self {
        Self(unsafe { _mm512_setzero_si512() })
    }

    /// Create from array (token-gated)
    #[inline(always)]
    pub fn from_array(token: crate::X64V4Token, arr: [i32; 16]) -> Self {
        Self::load(token, &arr)
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
        self.to_array().iter().copied().fold(0_i32, i32::wrapping_add)
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
impl_int_arithmetic_ops!(i32x16, _mm512_add_epi32, _mm512_sub_epi32);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_int_mul_op!(i32x16, _mm512_mullo_epi32);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_assign_ops!(i32x16);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_bitwise_ops!(i32x16, __m512i, _mm512_and_si512, _mm512_or_si512, _mm512_xor_si512);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_index!(i32x16, i32, 16);


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

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl u32x16 {
    pub const LANES: usize = 16;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: crate::X64V4Token, data: &[u32; 16]) -> Self {
        Self(unsafe { _mm512_loadu_si512(data.as_ptr() as *const __m512i) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: crate::X64V4Token, v: u32) -> Self {
        Self(unsafe { _mm512_set1_epi32(v as i32) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: crate::X64V4Token) -> Self {
        Self(unsafe { _mm512_setzero_si512() })
    }

    /// Create from array (token-gated)
    #[inline(always)]
    pub fn from_array(token: crate::X64V4Token, arr: [u32; 16]) -> Self {
        Self::load(token, &arr)
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
        self.to_array().iter().copied().fold(0_u32, u32::wrapping_add)
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
impl_int_arithmetic_ops!(u32x16, _mm512_add_epi32, _mm512_sub_epi32);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_int_mul_op!(u32x16, _mm512_mullo_epi32);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_assign_ops!(u32x16);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_bitwise_ops!(u32x16, __m512i, _mm512_and_si512, _mm512_or_si512, _mm512_xor_si512);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_index!(u32x16, u32, 16);


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

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl i64x8 {
    pub const LANES: usize = 8;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: crate::X64V4Token, data: &[i64; 8]) -> Self {
        Self(unsafe { _mm512_loadu_si512(data.as_ptr() as *const __m512i) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: crate::X64V4Token, v: i64) -> Self {
        Self(unsafe { _mm512_set1_epi64(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: crate::X64V4Token) -> Self {
        Self(unsafe { _mm512_setzero_si512() })
    }

    /// Create from array (token-gated)
    #[inline(always)]
    pub fn from_array(token: crate::X64V4Token, arr: [i64; 8]) -> Self {
        Self::load(token, &arr)
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
        self.to_array().iter().copied().fold(0_i64, i64::wrapping_add)
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
impl_int_arithmetic_ops!(i64x8, _mm512_add_epi64, _mm512_sub_epi64);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_assign_ops!(i64x8);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_bitwise_ops!(i64x8, __m512i, _mm512_and_si512, _mm512_or_si512, _mm512_xor_si512);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_index!(i64x8, i64, 8);


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

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl u64x8 {
    pub const LANES: usize = 8;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: crate::X64V4Token, data: &[u64; 8]) -> Self {
        Self(unsafe { _mm512_loadu_si512(data.as_ptr() as *const __m512i) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: crate::X64V4Token, v: u64) -> Self {
        Self(unsafe { _mm512_set1_epi64(v as i64) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: crate::X64V4Token) -> Self {
        Self(unsafe { _mm512_setzero_si512() })
    }

    /// Create from array (token-gated)
    #[inline(always)]
    pub fn from_array(token: crate::X64V4Token, arr: [u64; 8]) -> Self {
        Self::load(token, &arr)
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
        self.to_array().iter().copied().fold(0_u64, u64::wrapping_add)
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
impl_int_arithmetic_ops!(u64x8, _mm512_add_epi64, _mm512_sub_epi64);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_assign_ops!(u64x8);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_bitwise_ops!(u64x8, __m512i, _mm512_and_si512, _mm512_or_si512, _mm512_xor_si512);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_index!(u64x8, u64, 8);


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


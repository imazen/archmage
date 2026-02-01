//! 128-bit (WASM SIMD) types.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

use core::arch::wasm32::*;

// ============================================================================
// f32x4 - 4 x f32 (128-bit WASM SIMD)
// ============================================================================

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct f32x4(v128);

impl f32x4 {
    pub const LANES: usize = 4;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::Simd128Token, data: &[f32; 4]) -> Self {
        Self(unsafe { v128_load(data.as_ptr() as *const v128) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::Simd128Token, v: f32) -> Self {
        Self(f32x4_splat(v))
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::Simd128Token) -> Self {
        Self(f32x4_splat(0.0f32))
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::Simd128Token, arr: [f32; 4]) -> Self {
        // SAFETY: [f32; 4] and v128 have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [f32; 4]) {
        unsafe { v128_store(out.as_mut_ptr() as *mut v128, self.0) };
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
    pub fn raw(self) -> v128 {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports WASM SIMD128.
    #[inline(always)]
    pub unsafe fn from_raw(v: v128) -> Self {
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
    pub fn cast_slice(_: archmage::Simd128Token, slice: &[f32]) -> Option<&[Self]> {
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
    pub fn cast_slice_mut(_: archmage::Simd128Token, slice: &mut [f32]) -> Option<&mut [Self]> {
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
        // SAFETY: Self is repr(transparent) over v128 which is 16 bytes
        unsafe { &*(self as *const Self as *const [u8; 16]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 16] {
        // SAFETY: Self is repr(transparent) over v128 which is 16 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 16]) }
    }

    /// Create from a byte array reference (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::Simd128Token, bytes: &[u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Create from an owned byte array (token-gated, zero-cost).
    ///
    /// This is a zero-cost transmute from an owned byte array.
    #[inline(always)]
    pub fn from_bytes_owned(_: archmage::Simd128Token, bytes: [u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(bytes) })
    }

    /// Element-wise minimum
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(f32x4_min(self.0, other.0))
    }

    /// Element-wise maximum
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(f32x4_max(self.0, other.0))
    }

    /// Clamp values between lo and hi
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }

    /// Square root
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        Self(f32x4_sqrt(self.0))
    }

    /// Absolute value
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(f32x4_abs(self.0))
    }

    /// Floor
    #[inline(always)]
    pub fn floor(self) -> Self {
        Self(f32x4_floor(self.0))
    }

    /// Ceil
    #[inline(always)]
    pub fn ceil(self) -> Self {
        Self(f32x4_ceil(self.0))
    }

    /// Round to nearest
    #[inline(always)]
    pub fn round(self) -> Self {
        Self(f32x4_nearest(self.0))
    }

    /// Fused multiply-add: self * a + b
    ///
    /// Note: WASM doesn't have native FMA in stable SIMD,
    /// this is emulated with separate mul and add.
    #[inline(always)]
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        Self(f32x4_add(f32x4_mul(self.0, a.0), b.0))
    }

    /// Reduce: sum all lanes
    #[inline(always)]
    pub fn reduce_add(self) -> f32 {
        f32x4_extract_lane::<0>(self.0)
            + f32x4_extract_lane::<1>(self.0)
            + f32x4_extract_lane::<2>(self.0)
            + f32x4_extract_lane::<3>(self.0)
    }

    /// Reduce: max of all lanes
    #[inline(always)]
    pub fn reduce_max(self) -> f32 {
        let arr = self.to_array();
        arr.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    }

    /// Reduce: min of all lanes
    #[inline(always)]
    pub fn reduce_min(self) -> f32 {
        let arr = self.to_array();
        arr.iter().copied().fold(f32::INFINITY, f32::min)
    }

    /// Element-wise equality comparison (returns mask)
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> Self {
        Self(f32x4_eq(self.0, other.0))
    }

    /// Element-wise inequality comparison (returns mask)
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(v128_not(f32x4_eq(self.0, other.0)))
    }

    /// Element-wise less-than comparison (returns mask)
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(f32x4_lt(self.0, other.0))
    }

    /// Element-wise less-than-or-equal comparison (returns mask)
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(f32x4_le(self.0, other.0))
    }

    /// Element-wise greater-than comparison (returns mask)
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(f32x4_gt(self.0, other.0))
    }

    /// Element-wise greater-than-or-equal comparison (returns mask)
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(f32x4_ge(self.0, other.0))
    }

    /// Blend two vectors based on a mask
    ///
    /// For each lane, selects from `self` if the corresponding mask lane is all-ones,
    /// otherwise selects from `other`.
    ///
    /// The mask should come from a comparison operation like `simd_lt()`.
    ///
    /// # Example
    /// ```ignore
    /// let a = f32x4::splat(token, 1.0);
    /// let b = f32x4::splat(token, 2.0);
    /// let mask = a.simd_lt(b);  // all true
    /// let result = a.blend(b, mask);  // selects from a (all ones in mask)
    /// ```
    #[inline(always)]
    pub fn blend(self, other: Self, mask: Self) -> Self {
        Self(v128_bitselect(self.0, other.0, mask.0))
    }

    /// Bitwise NOT
    #[inline(always)]
    pub fn not(self) -> Self {
        Self(v128_not(self.0))
    }

    // ========== Transcendental Operations (Polynomial Approximations) ==========
    //
    // WASM SIMD has no native transcendental intrinsics.
    // Provides _unchecked (fast) and checked (handles edge cases) variants.

    /// Low-precision base-2 logarithm - unchecked variant.
    ///
    /// **Warning**: Does not handle edge cases (0, negative, inf, NaN).
    /// Use `log2_lowp()` for correct IEEE behavior.
    #[inline(always)]
    pub fn log2_lowp_unchecked(self) -> Self {
        // Rational polynomial coefficients from butteraugli/jpegli
        const P0: f32 = -1.850_383_34e-6;
        const P1: f32 = 1.428_716_05;
        const P2: f32 = 0.742_458_73;
        const Q0: f32 = 0.990_328_14;
        const Q1: f32 = 1.009_671_86;
        const Q2: f32 = 0.174_093_43;
        const OFFSET: u32 = 0x3f2aaaab;

        let x = self.0;

        // Extract exponent via bit manipulation
        let offset = u32x4_splat(OFFSET);
        let exp_bits = i32x4_sub(x, offset);
        let exp_shifted = i32x4_shr(exp_bits, 23);

        // Reconstruct mantissa in [1, 2) range
        let mantissa_bits = i32x4_sub(x, i32x4_shl(exp_shifted, 23));
        let exp_val = f32x4_convert_i32x4(exp_shifted);

        // m = mantissa - 1
        let one = f32x4_splat(1.0);
        let m = f32x4_sub(mantissa_bits, one);

        // Horner's for numerator: P2*m^2 + P1*m + P0
        let m2 = f32x4_mul(m, m);
        let p2_m2 = f32x4_mul(f32x4_splat(P2), m2);
        let p1_m = f32x4_mul(f32x4_splat(P1), m);
        let yp = f32x4_add(p2_m2, f32x4_add(p1_m, f32x4_splat(P0)));

        // Horner's for denominator: Q2*m^2 + Q1*m + Q0
        let q2_m2 = f32x4_mul(f32x4_splat(Q2), m2);
        let q1_m = f32x4_mul(f32x4_splat(Q1), m);
        let yq = f32x4_add(q2_m2, f32x4_add(q1_m, f32x4_splat(Q0)));

        Self(f32x4_add(f32x4_div(yp, yq), exp_val))
    }

    /// Low-precision base-2 logarithm (~7.7e-5 max relative error).
    ///
    /// Handles edge cases: log2(0) = -inf, log2(negative) = NaN,
    /// log2(+inf) = +inf, log2(NaN) = NaN.
    #[inline(always)]
    pub fn log2_lowp(self) -> Self {
        let result = self.log2_lowp_unchecked();

        // Edge case masks
        let zero = f32x4_splat(0.0);
        let is_zero = f32x4_eq(self.0, zero);
        let is_neg = f32x4_lt(self.0, zero);
        let is_inf = f32x4_eq(self.0, f32x4_splat(f32::INFINITY));
        // NaN: x != x
        let is_nan = v128_not(f32x4_eq(self.0, self.0));

        // Apply corrections using bitselect
        let neg_inf = f32x4_splat(f32::NEG_INFINITY);
        let pos_inf = f32x4_splat(f32::INFINITY);
        let nan = f32x4_splat(f32::NAN);

        let r = v128_bitselect(neg_inf, result.0, is_zero); // 0 -> -inf
        let r = v128_bitselect(nan, r, is_neg); // neg -> NaN
        let r = v128_bitselect(pos_inf, r, is_inf); // inf -> inf
        let r = v128_bitselect(nan, r, is_nan); // NaN -> NaN
        Self(r)
    }

    /// Low-precision base-2 exponential - unchecked variant.
    ///
    /// **Warning**: Clamps to [-126, 126]. Does not return inf for overflow.
    #[inline(always)]
    pub fn exp2_lowp_unchecked(self) -> Self {
        const C0: f32 = 1.0;
        const C1: f32 = core::f32::consts::LN_2;
        const C2: f32 = 0.240_226_5;
        const C3: f32 = 0.055_504_11;

        // Clamp to safe range
        let x = f32x4_pmax(self.0, f32x4_splat(-126.0));
        let x = f32x4_pmin(x, f32x4_splat(126.0));

        let xi = f32x4_floor(x);
        let xf = f32x4_sub(x, xi);

        // Polynomial for 2^frac
        let poly = f32x4_add(f32x4_mul(f32x4_splat(C3), xf), f32x4_splat(C2));
        let poly = f32x4_add(f32x4_mul(poly, xf), f32x4_splat(C1));
        let poly = f32x4_add(f32x4_mul(poly, xf), f32x4_splat(C0));

        // Scale by 2^integer
        let xi_i32 = i32x4_trunc_sat_f32x4(xi);
        let bias = i32x4_splat(127);
        let scale_bits = i32x4_shl(i32x4_add(xi_i32, bias), 23);

        Self(f32x4_mul(poly, scale_bits))
    }

    /// Low-precision base-2 exponential (~5.5e-3 max relative error).
    ///
    /// Handles edge cases: exp2(x > 128) = +inf, exp2(x < -150) = 0,
    /// exp2(NaN) = NaN.
    #[inline(always)]
    pub fn exp2_lowp(self) -> Self {
        let result = self.exp2_lowp_unchecked();

        // Edge case masks
        let is_overflow = f32x4_ge(self.0, f32x4_splat(128.0));
        let is_underflow = f32x4_lt(self.0, f32x4_splat(-150.0));
        let is_nan = v128_not(f32x4_eq(self.0, self.0));

        let pos_inf = f32x4_splat(f32::INFINITY);
        let zero = f32x4_splat(0.0);
        let nan = f32x4_splat(f32::NAN);

        let r = v128_bitselect(pos_inf, result.0, is_overflow);
        let r = v128_bitselect(zero, r, is_underflow);
        let r = v128_bitselect(nan, r, is_nan);
        Self(r)
    }

    /// Low-precision natural logarithm - unchecked variant.
    ///
    /// **Warning**: Does not handle edge cases (0, negative, inf, NaN).
    #[inline(always)]
    pub fn ln_lowp_unchecked(self) -> Self {
        const LN2: f32 = core::f32::consts::LN_2;
        Self(f32x4_mul(self.log2_lowp_unchecked().0, f32x4_splat(LN2)))
    }

    /// Low-precision natural logarithm.
    ///
    /// Computed as `log2_lowp(x) * ln(2)`.
    /// Handles edge cases correctly.
    #[inline(always)]
    pub fn ln_lowp(self) -> Self {
        const LN2: f32 = core::f32::consts::LN_2;
        Self(f32x4_mul(self.log2_lowp().0, f32x4_splat(LN2)))
    }

    /// Low-precision natural exponential - unchecked variant.
    ///
    /// **Warning**: Clamps to finite range. Does not return inf for overflow.
    #[inline(always)]
    pub fn exp_lowp_unchecked(self) -> Self {
        const LOG2_E: f32 = core::f32::consts::LOG2_E;
        Self(f32x4_mul(self.0, f32x4_splat(LOG2_E))).exp2_lowp_unchecked()
    }

    /// Low-precision natural exponential (e^x).
    ///
    /// Computed as `exp2_lowp(x * log2(e))`.
    /// Handles edge cases correctly.
    #[inline(always)]
    pub fn exp_lowp(self) -> Self {
        const LOG2_E: f32 = core::f32::consts::LOG2_E;
        Self(f32x4_mul(self.0, f32x4_splat(LOG2_E))).exp2_lowp()
    }

    /// Low-precision base-10 logarithm - unchecked variant.
    ///
    /// **Warning**: Does not handle edge cases (0, negative, inf, NaN).
    #[inline(always)]
    pub fn log10_lowp_unchecked(self) -> Self {
        const LOG10_2: f32 = core::f32::consts::LOG10_2;
        Self(f32x4_mul(
            self.log2_lowp_unchecked().0,
            f32x4_splat(LOG10_2),
        ))
    }

    /// Low-precision base-10 logarithm.
    ///
    /// Computed as `log2_lowp(x) * log10(2)`.
    /// Handles edge cases correctly.
    #[inline(always)]
    pub fn log10_lowp(self) -> Self {
        const LOG10_2: f32 = core::f32::consts::LOG10_2;
        Self(f32x4_mul(self.log2_lowp().0, f32x4_splat(LOG10_2)))
    }

    /// Low-precision power function - unchecked variant.
    ///
    /// **Warning**: Does not handle edge cases. Only valid for positive self values.
    #[inline(always)]
    pub fn pow_lowp_unchecked(self, n: f32) -> Self {
        Self(f32x4_mul(self.log2_lowp_unchecked().0, f32x4_splat(n))).exp2_lowp_unchecked()
    }

    /// Low-precision power function (self^n).
    ///
    /// Computed as `exp2_lowp(n * log2_lowp(self))`.
    /// Note: Only valid for positive self values.
    #[inline(always)]
    pub fn pow_lowp(self, n: f32) -> Self {
        Self(f32x4_mul(self.log2_lowp().0, f32x4_splat(n))).exp2_lowp()
    }

    // ========== Mid-Precision Transcendental Operations ==========

    /// Mid-precision base-2 logarithm - unchecked variant.
    ///
    /// **Warning**: Does not handle edge cases (0, negative, inf, NaN, denormals).
    #[inline(always)]
    pub fn log2_midp_unchecked(self) -> Self {
        const SQRT2_OVER_2: u32 = 0x3f3504f3;
        const ONE: u32 = 0x3f800000;
        const MANTISSA_MASK: u32 = 0x007fffff;
        const EXPONENT_BIAS: i32 = 127;

        // Coefficients for odd polynomial
        const C0: f32 = 2.885_390_08;
        const C1: f32 = 0.961_800_76;
        const C2: f32 = 0.576_974_45;
        const C3: f32 = 0.434_411_97;

        let x = self.0;

        // Normalize mantissa to [sqrt(2)/2, sqrt(2)]
        let offset = u32x4_splat(ONE - SQRT2_OVER_2);
        let adjusted = i32x4_add(x, offset);

        // Extract exponent
        let exp_raw = i32x4_shr(adjusted, 23);
        let exp_biased = i32x4_sub(exp_raw, i32x4_splat(EXPONENT_BIAS));
        let n = f32x4_convert_i32x4(exp_biased);

        // Reconstruct normalized mantissa
        let mantissa_bits = v128_and(adjusted, u32x4_splat(MANTISSA_MASK));
        let a_bits = i32x4_add(mantissa_bits, u32x4_splat(SQRT2_OVER_2));

        // y = (a - 1) / (a + 1)
        let one = f32x4_splat(1.0);
        let a_minus_1 = f32x4_sub(a_bits, one);
        let a_plus_1 = f32x4_add(a_bits, one);
        let y = f32x4_div(a_minus_1, a_plus_1);

        let y2 = f32x4_mul(y, y);

        // Polynomial: c0 + y^2*(c1 + y^2*(c2 + y^2*c3))
        let poly = f32x4_add(f32x4_mul(f32x4_splat(C3), y2), f32x4_splat(C2));
        let poly = f32x4_add(f32x4_mul(poly, y2), f32x4_splat(C1));
        let poly = f32x4_add(f32x4_mul(poly, y2), f32x4_splat(C0));

        Self(f32x4_add(f32x4_mul(y, poly), n))
    }

    /// Mid-precision base-2 logarithm (~3 ULP max error).
    ///
    /// Handles edge cases: log2(0) = -inf, log2(negative) = NaN,
    /// log2(+inf) = +inf, log2(NaN) = NaN.
    ///
    /// Note: Does not handle denormals. Use `log2_midp_precise()` for full IEEE.
    #[inline(always)]
    pub fn log2_midp(self) -> Self {
        let result = self.log2_midp_unchecked();

        let zero = f32x4_splat(0.0);
        let is_zero = f32x4_eq(self.0, zero);
        let is_neg = f32x4_lt(self.0, zero);
        let is_inf = f32x4_eq(self.0, f32x4_splat(f32::INFINITY));
        let is_nan = v128_not(f32x4_eq(self.0, self.0));

        let neg_inf = f32x4_splat(f32::NEG_INFINITY);
        let pos_inf = f32x4_splat(f32::INFINITY);
        let nan = f32x4_splat(f32::NAN);

        let r = v128_bitselect(neg_inf, result.0, is_zero);
        let r = v128_bitselect(nan, r, is_neg);
        let r = v128_bitselect(pos_inf, r, is_inf);
        let r = v128_bitselect(nan, r, is_nan);
        Self(r)
    }

    /// Mid-precision base-2 logarithm with full IEEE compliance.
    ///
    /// Handles all edge cases including denormals.
    /// About 50% slower than `log2_midp()` due to denormal scaling.
    #[inline(always)]
    pub fn log2_midp_precise(self) -> Self {
        // Scale factor for denormals: 2^24
        const SCALE_UP: f32 = 16777216.0; // 2^24
        const SCALE_ADJUST: f32 = 24.0; // log2(2^24)
        const DENORM_LIMIT: f32 = 1.17549435e-38; // Smallest normal f32

        // Detect denormals (positive values smaller than smallest normal)
        let zero = f32x4_splat(0.0);
        let is_positive = f32x4_gt(self.0, zero);
        let is_small = f32x4_lt(self.0, f32x4_splat(DENORM_LIMIT));
        let is_denorm = v128_and(is_positive, is_small);

        // Scale up denormals
        let scaled_x = f32x4_mul(self.0, f32x4_splat(SCALE_UP));
        let x_for_log = v128_bitselect(scaled_x, self.0, is_denorm);

        // Compute log2 with edge case handling
        let result = Self(x_for_log).log2_midp();

        // Adjust result for denormals (subtract 24)
        let adjusted = f32x4_sub(result.0, f32x4_splat(SCALE_ADJUST));
        Self(v128_bitselect(adjusted, result.0, is_denorm))
    }

    /// Mid-precision base-2 exponential - unchecked variant.
    ///
    /// **Warning**: Clamps to finite range. Does not return inf for overflow.
    #[inline(always)]
    pub fn exp2_midp_unchecked(self) -> Self {
        const C0: f32 = 1.0;
        const C1: f32 = 0.693_147_180_559_945;
        const C2: f32 = 0.240_226_506_959_101;
        const C3: f32 = 0.055_504_108_664_822;
        const C4: f32 = 0.009_618_129_107_629;
        const C5: f32 = 0.001_333_355_814_497;
        const C6: f32 = 0.000_154_035_303_933;

        let x = f32x4_pmax(self.0, f32x4_splat(-126.0));
        let x = f32x4_pmin(x, f32x4_splat(126.0));

        let xi = f32x4_floor(x);
        let xf = f32x4_sub(x, xi);

        // Horner's method with 6 coefficients
        let poly = f32x4_add(f32x4_mul(f32x4_splat(C6), xf), f32x4_splat(C5));
        let poly = f32x4_add(f32x4_mul(poly, xf), f32x4_splat(C4));
        let poly = f32x4_add(f32x4_mul(poly, xf), f32x4_splat(C3));
        let poly = f32x4_add(f32x4_mul(poly, xf), f32x4_splat(C2));
        let poly = f32x4_add(f32x4_mul(poly, xf), f32x4_splat(C1));
        let poly = f32x4_add(f32x4_mul(poly, xf), f32x4_splat(C0));

        let xi_i32 = i32x4_trunc_sat_f32x4(xi);
        let bias = i32x4_splat(127);
        let scale_bits = i32x4_shl(i32x4_add(xi_i32, bias), 23);

        Self(f32x4_mul(poly, scale_bits))
    }

    /// Mid-precision base-2 exponential (~8e-6 max relative error).
    ///
    /// Handles edge cases: exp2(x > 128) = +inf, exp2(x < -150) = 0,
    /// exp2(NaN) = NaN.
    #[inline(always)]
    pub fn exp2_midp(self) -> Self {
        let result = self.exp2_midp_unchecked();

        let is_overflow = f32x4_ge(self.0, f32x4_splat(128.0));
        let is_underflow = f32x4_lt(self.0, f32x4_splat(-150.0));
        let is_nan = v128_not(f32x4_eq(self.0, self.0));

        let pos_inf = f32x4_splat(f32::INFINITY);
        let zero = f32x4_splat(0.0);
        let nan = f32x4_splat(f32::NAN);

        let r = v128_bitselect(pos_inf, result.0, is_overflow);
        let r = v128_bitselect(zero, r, is_underflow);
        let r = v128_bitselect(nan, r, is_nan);
        Self(r)
    }

    /// Mid-precision natural logarithm - unchecked variant.
    ///
    /// **Warning**: Does not handle edge cases (0, negative, inf, NaN, denormals).
    #[inline(always)]
    pub fn ln_midp_unchecked(self) -> Self {
        const LN2: f32 = core::f32::consts::LN_2;
        Self(f32x4_mul(self.log2_midp_unchecked().0, f32x4_splat(LN2)))
    }

    /// Mid-precision natural logarithm.
    ///
    /// Computed as `log2_midp(x) * ln(2)`.
    /// Handles edge cases correctly.
    ///
    /// Note: Does not handle denormals. Use `ln_midp_precise()` for full IEEE.
    #[inline(always)]
    pub fn ln_midp(self) -> Self {
        const LN2: f32 = core::f32::consts::LN_2;
        Self(f32x4_mul(self.log2_midp().0, f32x4_splat(LN2)))
    }

    /// Mid-precision natural logarithm with full IEEE compliance.
    ///
    /// Handles all edge cases including denormals.
    /// About 50% slower than `ln_midp()` due to denormal scaling.
    #[inline(always)]
    pub fn ln_midp_precise(self) -> Self {
        const LN2: f32 = core::f32::consts::LN_2;
        Self(f32x4_mul(self.log2_midp_precise().0, f32x4_splat(LN2)))
    }

    /// Mid-precision natural exponential - unchecked variant.
    ///
    /// **Warning**: Clamps to finite range. Does not return inf for overflow.
    #[inline(always)]
    pub fn exp_midp_unchecked(self) -> Self {
        const LOG2_E: f32 = core::f32::consts::LOG2_E;
        Self(f32x4_mul(self.0, f32x4_splat(LOG2_E))).exp2_midp_unchecked()
    }

    /// Mid-precision natural exponential (e^x).
    ///
    /// Computed as `exp2_midp(x * log2(e))`.
    /// Handles edge cases correctly.
    #[inline(always)]
    pub fn exp_midp(self) -> Self {
        const LOG2_E: f32 = core::f32::consts::LOG2_E;
        Self(f32x4_mul(self.0, f32x4_splat(LOG2_E))).exp2_midp()
    }

    /// Mid-precision base-10 logarithm - unchecked variant.
    ///
    /// **Warning**: Does not handle edge cases (0, negative, inf, NaN, denormals).
    #[inline(always)]
    pub fn log10_midp_unchecked(self) -> Self {
        const LOG10_2: f32 = core::f32::consts::LOG10_2;
        Self(f32x4_mul(
            self.log2_midp_unchecked().0,
            f32x4_splat(LOG10_2),
        ))
    }

    /// Mid-precision base-10 logarithm.
    ///
    /// Computed as `log2_midp(x) * log10(2)`.
    /// Handles edge cases correctly.
    ///
    /// Note: Does not handle denormals. Use `log10_midp_precise()` for full IEEE.
    #[inline(always)]
    pub fn log10_midp(self) -> Self {
        const LOG10_2: f32 = core::f32::consts::LOG10_2;
        Self(f32x4_mul(self.log2_midp().0, f32x4_splat(LOG10_2)))
    }

    /// Mid-precision base-10 logarithm with full IEEE compliance.
    ///
    /// Handles all edge cases including denormals.
    /// About 50% slower than `log10_midp()` due to denormal scaling.
    #[inline(always)]
    pub fn log10_midp_precise(self) -> Self {
        const LOG10_2: f32 = core::f32::consts::LOG10_2;
        Self(f32x4_mul(self.log2_midp_precise().0, f32x4_splat(LOG10_2)))
    }

    /// Mid-precision power function - unchecked variant.
    ///
    /// **Warning**: Does not handle edge cases. Only valid for positive self values.
    #[inline(always)]
    pub fn pow_midp_unchecked(self, n: f32) -> Self {
        Self(f32x4_mul(self.log2_midp_unchecked().0, f32x4_splat(n))).exp2_midp_unchecked()
    }

    /// Mid-precision power function (self^n).
    ///
    /// Computed as `exp2_midp(n * log2_midp(self))`.
    /// Achieves 100% exact round-trips for 8-bit, 10-bit, and 12-bit values.
    /// Note: Only valid for positive self values.
    #[inline(always)]
    pub fn pow_midp(self, n: f32) -> Self {
        Self(f32x4_mul(self.log2_midp().0, f32x4_splat(n))).exp2_midp()
    }

    /// Mid-precision power function with denormal handling.
    ///
    /// Uses `log2_midp_precise()` to handle denormal inputs correctly.
    /// Note: Only valid for positive self values.
    #[inline(always)]
    pub fn pow_midp_precise(self, n: f32) -> Self {
        Self(f32x4_mul(self.log2_midp_precise().0, f32x4_splat(n))).exp2_midp()
    }

    // ========== Bitcast (reinterpret bits, zero-cost) ==========

    /// Reinterpret bits as `i32x4` (zero-cost).
    #[inline(always)]
    pub fn bitcast_i32x4(self) -> i32x4 {
        i32x4(self.0)
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
        u32x4(self.0)
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

impl core::ops::Add for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(f32x4_add(self.0, rhs.0))
    }
}

impl core::ops::Sub for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(f32x4_sub(self.0, rhs.0))
    }
}

impl core::ops::Mul for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(f32x4_mul(self.0, rhs.0))
    }
}

impl core::ops::Div for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        Self(f32x4_div(self.0, rhs.0))
    }
}

impl core::ops::Neg for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(f32x4_neg(self.0))
    }
}

impl core::ops::BitAnd for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(v128_and(self.0, rhs.0))
    }
}

impl core::ops::BitOr for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self(v128_or(self.0, rhs.0))
    }
}

impl core::ops::BitXor for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Self(v128_xor(self.0, rhs.0))
    }
}

impl core::ops::AddAssign for f32x4 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl core::ops::SubAssign for f32x4 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl core::ops::MulAssign for f32x4 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl core::ops::DivAssign for f32x4 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl core::ops::BitAndAssign for f32x4 {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

impl core::ops::BitOrAssign for f32x4 {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl core::ops::BitXorAssign for f32x4 {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = *self ^ rhs;
    }
}

impl core::ops::Index<usize> for f32x4 {
    type Output = f32;
    #[inline(always)]
    fn index(&self, i: usize) -> &Self::Output {
        assert!(i < 4, "index out of bounds");
        unsafe { &*(self as *const Self as *const f32).add(i) }
    }
}

impl core::ops::IndexMut<usize> for f32x4 {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        assert!(i < 4, "index out of bounds");
        unsafe { &mut *(self as *mut Self as *mut f32).add(i) }
    }
}

impl From<[f32; 4]> for f32x4 {
    #[inline(always)]
    fn from(arr: [f32; 4]) -> Self {
        // SAFETY: [f32; 4] and v128 have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}

impl From<f32x4> for [f32; 4] {
    #[inline(always)]
    fn from(v: f32x4) -> Self {
        // SAFETY: v128 and [f32; 4] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}

// ============================================================================
// f64x2 - 2 x f64 (128-bit WASM SIMD)
// ============================================================================

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct f64x2(v128);

impl f64x2 {
    pub const LANES: usize = 2;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::Simd128Token, data: &[f64; 2]) -> Self {
        Self(unsafe { v128_load(data.as_ptr() as *const v128) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::Simd128Token, v: f64) -> Self {
        Self(f64x2_splat(v))
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::Simd128Token) -> Self {
        Self(f64x2_splat(0.0f64))
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::Simd128Token, arr: [f64; 2]) -> Self {
        // SAFETY: [f64; 2] and v128 have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [f64; 2]) {
        unsafe { v128_store(out.as_mut_ptr() as *mut v128, self.0) };
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
    pub fn raw(self) -> v128 {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports WASM SIMD128.
    #[inline(always)]
    pub unsafe fn from_raw(v: v128) -> Self {
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
    pub fn cast_slice(_: archmage::Simd128Token, slice: &[f64]) -> Option<&[Self]> {
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
    pub fn cast_slice_mut(_: archmage::Simd128Token, slice: &mut [f64]) -> Option<&mut [Self]> {
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
        // SAFETY: Self is repr(transparent) over v128 which is 16 bytes
        unsafe { &*(self as *const Self as *const [u8; 16]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 16] {
        // SAFETY: Self is repr(transparent) over v128 which is 16 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 16]) }
    }

    /// Create from a byte array reference (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::Simd128Token, bytes: &[u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Create from an owned byte array (token-gated, zero-cost).
    ///
    /// This is a zero-cost transmute from an owned byte array.
    #[inline(always)]
    pub fn from_bytes_owned(_: archmage::Simd128Token, bytes: [u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(bytes) })
    }

    /// Element-wise minimum
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(f64x2_min(self.0, other.0))
    }

    /// Element-wise maximum
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(f64x2_max(self.0, other.0))
    }

    /// Clamp values between lo and hi
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }

    /// Square root
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        Self(f64x2_sqrt(self.0))
    }

    /// Absolute value
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(f64x2_abs(self.0))
    }

    /// Floor
    #[inline(always)]
    pub fn floor(self) -> Self {
        Self(f64x2_floor(self.0))
    }

    /// Ceil
    #[inline(always)]
    pub fn ceil(self) -> Self {
        Self(f64x2_ceil(self.0))
    }

    /// Round to nearest
    #[inline(always)]
    pub fn round(self) -> Self {
        Self(f64x2_nearest(self.0))
    }

    /// Fused multiply-add: self * a + b
    ///
    /// Note: WASM doesn't have native FMA in stable SIMD,
    /// this is emulated with separate mul and add.
    #[inline(always)]
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        Self(f64x2_add(f64x2_mul(self.0, a.0), b.0))
    }

    /// Reduce: sum all lanes
    #[inline(always)]
    pub fn reduce_add(self) -> f64 {
        f64x2_extract_lane::<0>(self.0) + f64x2_extract_lane::<1>(self.0)
    }

    /// Reduce: max of all lanes
    #[inline(always)]
    pub fn reduce_max(self) -> f64 {
        let arr = self.to_array();
        arr.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    }

    /// Reduce: min of all lanes
    #[inline(always)]
    pub fn reduce_min(self) -> f64 {
        let arr = self.to_array();
        arr.iter().copied().fold(f64::INFINITY, f64::min)
    }

    /// Element-wise equality comparison (returns mask)
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> Self {
        Self(f64x2_eq(self.0, other.0))
    }

    /// Element-wise inequality comparison (returns mask)
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(v128_not(f64x2_eq(self.0, other.0)))
    }

    /// Element-wise less-than comparison (returns mask)
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(f64x2_lt(self.0, other.0))
    }

    /// Element-wise less-than-or-equal comparison (returns mask)
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(f64x2_le(self.0, other.0))
    }

    /// Element-wise greater-than comparison (returns mask)
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(f64x2_gt(self.0, other.0))
    }

    /// Element-wise greater-than-or-equal comparison (returns mask)
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(f64x2_ge(self.0, other.0))
    }

    /// Blend two vectors based on a mask
    ///
    /// For each lane, selects from `self` if the corresponding mask lane is all-ones,
    /// otherwise selects from `other`.
    ///
    /// The mask should come from a comparison operation like `simd_lt()`.
    ///
    /// # Example
    /// ```ignore
    /// let a = f64x2::splat(token, 1.0);
    /// let b = f64x2::splat(token, 2.0);
    /// let mask = a.simd_lt(b);  // all true
    /// let result = a.blend(b, mask);  // selects from a (all ones in mask)
    /// ```
    #[inline(always)]
    pub fn blend(self, other: Self, mask: Self) -> Self {
        Self(v128_bitselect(self.0, other.0, mask.0))
    }

    /// Bitwise NOT
    #[inline(always)]
    pub fn not(self) -> Self {
        Self(v128_not(self.0))
    }

    // ========== Transcendental Operations (Polynomial Approximations) ==========
    //
    // WASM SIMD has no native transcendental intrinsics.
    // Provides _unchecked (fast) and checked (handles edge cases) variants.

    /// Low-precision base-2 logarithm.
    ///
    /// Uses scalar fallback for simplicity (only 2 lanes).
    /// Handles edge cases correctly via std.
    #[inline(always)]
    pub fn log2_lowp(self) -> Self {
        let arr = self.to_array();
        let result = [arr[0].log2(), arr[1].log2()];
        Self::from(result)
    }

    /// Low-precision base-2 exponential (2^x).
    ///
    /// Handles edge cases correctly via std.
    #[inline(always)]
    pub fn exp2_lowp(self) -> Self {
        let arr = self.to_array();
        let result = [arr[0].exp2(), arr[1].exp2()];
        Self::from(result)
    }

    /// Low-precision natural logarithm.
    ///
    /// Handles edge cases correctly via std.
    #[inline(always)]
    pub fn ln_lowp(self) -> Self {
        let arr = self.to_array();
        let result = [arr[0].ln(), arr[1].ln()];
        Self::from(result)
    }

    /// Low-precision natural exponential (e^x).
    ///
    /// Handles edge cases correctly via std.
    #[inline(always)]
    pub fn exp_lowp(self) -> Self {
        let arr = self.to_array();
        let result = [arr[0].exp(), arr[1].exp()];
        Self::from(result)
    }

    /// Low-precision power function (self^n).
    ///
    /// Handles edge cases correctly via std.
    #[inline(always)]
    pub fn pow_lowp(self, n: f64) -> Self {
        let arr = self.to_array();
        let result = [arr[0].powf(n), arr[1].powf(n)];
        Self::from(result)
    }

    // ========== Bitcast (reinterpret bits, zero-cost) ==========

    /// Reinterpret bits as `i64x2` (zero-cost).
    #[inline(always)]
    pub fn bitcast_i64x2(self) -> i64x2 {
        i64x2(self.0)
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
        u64x2(self.0)
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

impl core::ops::Add for f64x2 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(f64x2_add(self.0, rhs.0))
    }
}

impl core::ops::Sub for f64x2 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(f64x2_sub(self.0, rhs.0))
    }
}

impl core::ops::Mul for f64x2 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(f64x2_mul(self.0, rhs.0))
    }
}

impl core::ops::Div for f64x2 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        Self(f64x2_div(self.0, rhs.0))
    }
}

impl core::ops::Neg for f64x2 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(f64x2_neg(self.0))
    }
}

impl core::ops::BitAnd for f64x2 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(v128_and(self.0, rhs.0))
    }
}

impl core::ops::BitOr for f64x2 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self(v128_or(self.0, rhs.0))
    }
}

impl core::ops::BitXor for f64x2 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Self(v128_xor(self.0, rhs.0))
    }
}

impl core::ops::AddAssign for f64x2 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl core::ops::SubAssign for f64x2 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl core::ops::MulAssign for f64x2 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl core::ops::DivAssign for f64x2 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl core::ops::BitAndAssign for f64x2 {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

impl core::ops::BitOrAssign for f64x2 {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl core::ops::BitXorAssign for f64x2 {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = *self ^ rhs;
    }
}

impl core::ops::Index<usize> for f64x2 {
    type Output = f64;
    #[inline(always)]
    fn index(&self, i: usize) -> &Self::Output {
        assert!(i < 2, "index out of bounds");
        unsafe { &*(self as *const Self as *const f64).add(i) }
    }
}

impl core::ops::IndexMut<usize> for f64x2 {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        assert!(i < 2, "index out of bounds");
        unsafe { &mut *(self as *mut Self as *mut f64).add(i) }
    }
}

impl From<[f64; 2]> for f64x2 {
    #[inline(always)]
    fn from(arr: [f64; 2]) -> Self {
        // SAFETY: [f64; 2] and v128 have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}

impl From<f64x2> for [f64; 2] {
    #[inline(always)]
    fn from(v: f64x2) -> Self {
        // SAFETY: v128 and [f64; 2] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}

// ============================================================================
// i8x16 - 16 x i8 (128-bit WASM SIMD)
// ============================================================================

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct i8x16(v128);

impl i8x16 {
    pub const LANES: usize = 16;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::Simd128Token, data: &[i8; 16]) -> Self {
        Self(unsafe { v128_load(data.as_ptr() as *const v128) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::Simd128Token, v: i8) -> Self {
        Self(i8x16_splat(v))
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::Simd128Token) -> Self {
        Self(i8x16_splat(0i8))
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::Simd128Token, arr: [i8; 16]) -> Self {
        // SAFETY: [i8; 16] and v128 have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [i8; 16]) {
        unsafe { v128_store(out.as_mut_ptr() as *mut v128, self.0) };
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
    pub fn raw(self) -> v128 {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports WASM SIMD128.
    #[inline(always)]
    pub unsafe fn from_raw(v: v128) -> Self {
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
    pub fn cast_slice(_: archmage::Simd128Token, slice: &[i8]) -> Option<&[Self]> {
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
    pub fn cast_slice_mut(_: archmage::Simd128Token, slice: &mut [i8]) -> Option<&mut [Self]> {
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
        // SAFETY: Self is repr(transparent) over v128 which is 16 bytes
        unsafe { &*(self as *const Self as *const [u8; 16]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 16] {
        // SAFETY: Self is repr(transparent) over v128 which is 16 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 16]) }
    }

    /// Create from a byte array reference (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::Simd128Token, bytes: &[u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Create from an owned byte array (token-gated, zero-cost).
    ///
    /// This is a zero-cost transmute from an owned byte array.
    #[inline(always)]
    pub fn from_bytes_owned(_: archmage::Simd128Token, bytes: [u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(bytes) })
    }

    /// Element-wise minimum
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(i8x16_min(self.0, other.0))
    }

    /// Element-wise maximum
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(i8x16_max(self.0, other.0))
    }

    /// Clamp values between lo and hi
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }

    /// Absolute value
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(i8x16_abs(self.0))
    }

    /// Reduce: sum all lanes
    #[inline(always)]
    pub fn reduce_add(self) -> i8 {
        let arr = self.to_array();
        arr.iter().copied().fold(0i8, |a, b| a.wrapping_add(b))
    }

    /// Element-wise equality comparison (returns mask)
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> Self {
        Self(i8x16_eq(self.0, other.0))
    }

    /// Element-wise inequality comparison (returns mask)
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(v128_not(i8x16_eq(self.0, other.0)))
    }

    /// Element-wise less-than comparison (returns mask)
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(i8x16_lt(self.0, other.0))
    }

    /// Element-wise less-than-or-equal comparison (returns mask)
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(i8x16_le(self.0, other.0))
    }

    /// Element-wise greater-than comparison (returns mask)
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(i8x16_gt(self.0, other.0))
    }

    /// Element-wise greater-than-or-equal comparison (returns mask)
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(i8x16_ge(self.0, other.0))
    }

    /// Blend two vectors based on a mask
    ///
    /// For each lane, selects from `self` if the corresponding mask lane is all-ones,
    /// otherwise selects from `other`.
    ///
    /// The mask should come from a comparison operation like `simd_lt()`.
    ///
    /// # Example
    /// ```ignore
    /// let a = i8x16::splat(token, 1.0);
    /// let b = i8x16::splat(token, 2.0);
    /// let mask = a.simd_lt(b);  // all true
    /// let result = a.blend(b, mask);  // selects from a (all ones in mask)
    /// ```
    #[inline(always)]
    pub fn blend(self, other: Self, mask: Self) -> Self {
        Self(v128_bitselect(self.0, other.0, mask.0))
    }

    /// Bitwise NOT
    #[inline(always)]
    pub fn not(self) -> Self {
        Self(v128_not(self.0))
    }

    /// Shift left by constant
    #[inline(always)]
    pub fn shl<const N: u32>(self) -> Self {
        Self(i8x16_shl(self.0, N))
    }

    /// Shift right by constant
    ///
    /// For signed types, this is an arithmetic shift (sign-extending).
    #[inline(always)]
    pub fn shr<const N: u32>(self) -> Self {
        Self(i8x16_shr(self.0, N))
    }

    /// Check if all lanes are non-zero (all true)
    #[inline(always)]
    pub fn all_true(self) -> bool {
        i8x16_all_true(self.0)
    }

    /// Check if any lane is non-zero (any true)
    #[inline(always)]
    pub fn any_true(self) -> bool {
        v128_any_true(self.0)
    }

    /// Extract the high bit of each lane as a bitmask
    #[inline(always)]
    pub fn bitmask(self) -> u32 {
        i8x16_bitmask(self.0) as u32
    }

    // ========== Bitcast (reinterpret bits, zero-cost) ==========

    /// Reinterpret bits as `u8x16` (zero-cost).
    #[inline(always)]
    pub fn bitcast_u8x16(self) -> u8x16 {
        u8x16(self.0)
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

impl core::ops::Add for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(i8x16_add(self.0, rhs.0))
    }
}

impl core::ops::Sub for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(i8x16_sub(self.0, rhs.0))
    }
}

impl core::ops::Neg for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(i8x16_neg(self.0))
    }
}

impl core::ops::BitAnd for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(v128_and(self.0, rhs.0))
    }
}

impl core::ops::BitOr for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self(v128_or(self.0, rhs.0))
    }
}

impl core::ops::BitXor for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Self(v128_xor(self.0, rhs.0))
    }
}

impl core::ops::AddAssign for i8x16 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl core::ops::SubAssign for i8x16 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl core::ops::BitAndAssign for i8x16 {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

impl core::ops::BitOrAssign for i8x16 {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl core::ops::BitXorAssign for i8x16 {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = *self ^ rhs;
    }
}

impl core::ops::Index<usize> for i8x16 {
    type Output = i8;
    #[inline(always)]
    fn index(&self, i: usize) -> &Self::Output {
        assert!(i < 16, "index out of bounds");
        unsafe { &*(self as *const Self as *const i8).add(i) }
    }
}

impl core::ops::IndexMut<usize> for i8x16 {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        assert!(i < 16, "index out of bounds");
        unsafe { &mut *(self as *mut Self as *mut i8).add(i) }
    }
}

impl From<[i8; 16]> for i8x16 {
    #[inline(always)]
    fn from(arr: [i8; 16]) -> Self {
        // SAFETY: [i8; 16] and v128 have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}

impl From<i8x16> for [i8; 16] {
    #[inline(always)]
    fn from(v: i8x16) -> Self {
        // SAFETY: v128 and [i8; 16] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}

// ============================================================================
// u8x16 - 16 x u8 (128-bit WASM SIMD)
// ============================================================================

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct u8x16(v128);

impl u8x16 {
    pub const LANES: usize = 16;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::Simd128Token, data: &[u8; 16]) -> Self {
        Self(unsafe { v128_load(data.as_ptr() as *const v128) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::Simd128Token, v: u8) -> Self {
        Self(u8x16_splat(v))
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::Simd128Token) -> Self {
        Self(u8x16_splat(0u8))
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::Simd128Token, arr: [u8; 16]) -> Self {
        // SAFETY: [u8; 16] and v128 have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [u8; 16]) {
        unsafe { v128_store(out.as_mut_ptr() as *mut v128, self.0) };
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
    pub fn raw(self) -> v128 {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports WASM SIMD128.
    #[inline(always)]
    pub unsafe fn from_raw(v: v128) -> Self {
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
    pub fn cast_slice(_: archmage::Simd128Token, slice: &[u8]) -> Option<&[Self]> {
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
    pub fn cast_slice_mut(_: archmage::Simd128Token, slice: &mut [u8]) -> Option<&mut [Self]> {
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
        // SAFETY: Self is repr(transparent) over v128 which is 16 bytes
        unsafe { &*(self as *const Self as *const [u8; 16]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 16] {
        // SAFETY: Self is repr(transparent) over v128 which is 16 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 16]) }
    }

    /// Create from a byte array reference (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::Simd128Token, bytes: &[u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Create from an owned byte array (token-gated, zero-cost).
    ///
    /// This is a zero-cost transmute from an owned byte array.
    #[inline(always)]
    pub fn from_bytes_owned(_: archmage::Simd128Token, bytes: [u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(bytes) })
    }

    /// Element-wise minimum
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(u8x16_min(self.0, other.0))
    }

    /// Element-wise maximum
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(u8x16_max(self.0, other.0))
    }

    /// Clamp values between lo and hi
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }

    /// Reduce: sum all lanes
    #[inline(always)]
    pub fn reduce_add(self) -> u8 {
        let arr = self.to_array();
        arr.iter().copied().fold(0u8, |a, b| a.wrapping_add(b))
    }

    /// Element-wise equality comparison (returns mask)
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> Self {
        Self(u8x16_eq(self.0, other.0))
    }

    /// Element-wise inequality comparison (returns mask)
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(v128_not(u8x16_eq(self.0, other.0)))
    }

    /// Element-wise less-than comparison (returns mask)
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(u8x16_lt(self.0, other.0))
    }

    /// Element-wise less-than-or-equal comparison (returns mask)
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(u8x16_le(self.0, other.0))
    }

    /// Element-wise greater-than comparison (returns mask)
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(u8x16_gt(self.0, other.0))
    }

    /// Element-wise greater-than-or-equal comparison (returns mask)
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(u8x16_ge(self.0, other.0))
    }

    /// Blend two vectors based on a mask
    ///
    /// For each lane, selects from `self` if the corresponding mask lane is all-ones,
    /// otherwise selects from `other`.
    ///
    /// The mask should come from a comparison operation like `simd_lt()`.
    ///
    /// # Example
    /// ```ignore
    /// let a = u8x16::splat(token, 1.0);
    /// let b = u8x16::splat(token, 2.0);
    /// let mask = a.simd_lt(b);  // all true
    /// let result = a.blend(b, mask);  // selects from a (all ones in mask)
    /// ```
    #[inline(always)]
    pub fn blend(self, other: Self, mask: Self) -> Self {
        Self(v128_bitselect(self.0, other.0, mask.0))
    }

    /// Bitwise NOT
    #[inline(always)]
    pub fn not(self) -> Self {
        Self(v128_not(self.0))
    }

    /// Shift left by constant
    #[inline(always)]
    pub fn shl<const N: u32>(self) -> Self {
        Self(u8x16_shl(self.0, N))
    }

    /// Shift right by constant
    #[inline(always)]
    pub fn shr<const N: u32>(self) -> Self {
        Self(u8x16_shr(self.0, N))
    }

    /// Check if all lanes are non-zero (all true)
    #[inline(always)]
    pub fn all_true(self) -> bool {
        u8x16_all_true(self.0)
    }

    /// Check if any lane is non-zero (any true)
    #[inline(always)]
    pub fn any_true(self) -> bool {
        v128_any_true(self.0)
    }

    /// Extract the high bit of each lane as a bitmask
    #[inline(always)]
    pub fn bitmask(self) -> u32 {
        u8x16_bitmask(self.0) as u32
    }

    // ========== Bitcast (reinterpret bits, zero-cost) ==========

    /// Reinterpret bits as `i8x16` (zero-cost).
    #[inline(always)]
    pub fn bitcast_i8x16(self) -> i8x16 {
        i8x16(self.0)
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

impl core::ops::Add for u8x16 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(u8x16_add(self.0, rhs.0))
    }
}

impl core::ops::Sub for u8x16 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(u8x16_sub(self.0, rhs.0))
    }
}

impl core::ops::BitAnd for u8x16 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(v128_and(self.0, rhs.0))
    }
}

impl core::ops::BitOr for u8x16 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self(v128_or(self.0, rhs.0))
    }
}

impl core::ops::BitXor for u8x16 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Self(v128_xor(self.0, rhs.0))
    }
}

impl core::ops::AddAssign for u8x16 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl core::ops::SubAssign for u8x16 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl core::ops::BitAndAssign for u8x16 {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

impl core::ops::BitOrAssign for u8x16 {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl core::ops::BitXorAssign for u8x16 {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = *self ^ rhs;
    }
}

impl core::ops::Index<usize> for u8x16 {
    type Output = u8;
    #[inline(always)]
    fn index(&self, i: usize) -> &Self::Output {
        assert!(i < 16, "index out of bounds");
        unsafe { &*(self as *const Self as *const u8).add(i) }
    }
}

impl core::ops::IndexMut<usize> for u8x16 {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        assert!(i < 16, "index out of bounds");
        unsafe { &mut *(self as *mut Self as *mut u8).add(i) }
    }
}

impl From<[u8; 16]> for u8x16 {
    #[inline(always)]
    fn from(arr: [u8; 16]) -> Self {
        // SAFETY: [u8; 16] and v128 have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}

impl From<u8x16> for [u8; 16] {
    #[inline(always)]
    fn from(v: u8x16) -> Self {
        // SAFETY: v128 and [u8; 16] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}

// ============================================================================
// i16x8 - 8 x i16 (128-bit WASM SIMD)
// ============================================================================

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct i16x8(v128);

impl i16x8 {
    pub const LANES: usize = 8;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::Simd128Token, data: &[i16; 8]) -> Self {
        Self(unsafe { v128_load(data.as_ptr() as *const v128) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::Simd128Token, v: i16) -> Self {
        Self(i16x8_splat(v))
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::Simd128Token) -> Self {
        Self(i16x8_splat(0i16))
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::Simd128Token, arr: [i16; 8]) -> Self {
        // SAFETY: [i16; 8] and v128 have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [i16; 8]) {
        unsafe { v128_store(out.as_mut_ptr() as *mut v128, self.0) };
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
    pub fn raw(self) -> v128 {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports WASM SIMD128.
    #[inline(always)]
    pub unsafe fn from_raw(v: v128) -> Self {
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
    pub fn cast_slice(_: archmage::Simd128Token, slice: &[i16]) -> Option<&[Self]> {
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
    pub fn cast_slice_mut(_: archmage::Simd128Token, slice: &mut [i16]) -> Option<&mut [Self]> {
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
        // SAFETY: Self is repr(transparent) over v128 which is 16 bytes
        unsafe { &*(self as *const Self as *const [u8; 16]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 16] {
        // SAFETY: Self is repr(transparent) over v128 which is 16 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 16]) }
    }

    /// Create from a byte array reference (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::Simd128Token, bytes: &[u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Create from an owned byte array (token-gated, zero-cost).
    ///
    /// This is a zero-cost transmute from an owned byte array.
    #[inline(always)]
    pub fn from_bytes_owned(_: archmage::Simd128Token, bytes: [u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(bytes) })
    }

    /// Element-wise minimum
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(i16x8_min(self.0, other.0))
    }

    /// Element-wise maximum
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(i16x8_max(self.0, other.0))
    }

    /// Clamp values between lo and hi
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }

    /// Absolute value
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(i16x8_abs(self.0))
    }

    /// Reduce: sum all lanes
    #[inline(always)]
    pub fn reduce_add(self) -> i16 {
        let arr = self.to_array();
        arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7]
    }

    /// Element-wise equality comparison (returns mask)
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> Self {
        Self(i16x8_eq(self.0, other.0))
    }

    /// Element-wise inequality comparison (returns mask)
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(v128_not(i16x8_eq(self.0, other.0)))
    }

    /// Element-wise less-than comparison (returns mask)
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(i16x8_lt(self.0, other.0))
    }

    /// Element-wise less-than-or-equal comparison (returns mask)
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(i16x8_le(self.0, other.0))
    }

    /// Element-wise greater-than comparison (returns mask)
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(i16x8_gt(self.0, other.0))
    }

    /// Element-wise greater-than-or-equal comparison (returns mask)
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(i16x8_ge(self.0, other.0))
    }

    /// Blend two vectors based on a mask
    ///
    /// For each lane, selects from `self` if the corresponding mask lane is all-ones,
    /// otherwise selects from `other`.
    ///
    /// The mask should come from a comparison operation like `simd_lt()`.
    ///
    /// # Example
    /// ```ignore
    /// let a = i16x8::splat(token, 1.0);
    /// let b = i16x8::splat(token, 2.0);
    /// let mask = a.simd_lt(b);  // all true
    /// let result = a.blend(b, mask);  // selects from a (all ones in mask)
    /// ```
    #[inline(always)]
    pub fn blend(self, other: Self, mask: Self) -> Self {
        Self(v128_bitselect(self.0, other.0, mask.0))
    }

    /// Bitwise NOT
    #[inline(always)]
    pub fn not(self) -> Self {
        Self(v128_not(self.0))
    }

    /// Shift left by constant
    #[inline(always)]
    pub fn shl<const N: u32>(self) -> Self {
        Self(i16x8_shl(self.0, N))
    }

    /// Shift right by constant
    ///
    /// For signed types, this is an arithmetic shift (sign-extending).
    #[inline(always)]
    pub fn shr<const N: u32>(self) -> Self {
        Self(i16x8_shr(self.0, N))
    }

    /// Check if all lanes are non-zero (all true)
    #[inline(always)]
    pub fn all_true(self) -> bool {
        i16x8_all_true(self.0)
    }

    /// Check if any lane is non-zero (any true)
    #[inline(always)]
    pub fn any_true(self) -> bool {
        v128_any_true(self.0)
    }

    /// Extract the high bit of each lane as a bitmask
    #[inline(always)]
    pub fn bitmask(self) -> u32 {
        i16x8_bitmask(self.0) as u32
    }

    // ========== Bitcast (reinterpret bits, zero-cost) ==========

    /// Reinterpret bits as `u16x8` (zero-cost).
    #[inline(always)]
    pub fn bitcast_u16x8(self) -> u16x8 {
        u16x8(self.0)
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

impl core::ops::Add for i16x8 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(i16x8_add(self.0, rhs.0))
    }
}

impl core::ops::Sub for i16x8 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(i16x8_sub(self.0, rhs.0))
    }
}

impl core::ops::Mul for i16x8 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(i16x8_mul(self.0, rhs.0))
    }
}

impl core::ops::Neg for i16x8 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(i16x8_neg(self.0))
    }
}

impl core::ops::BitAnd for i16x8 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(v128_and(self.0, rhs.0))
    }
}

impl core::ops::BitOr for i16x8 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self(v128_or(self.0, rhs.0))
    }
}

impl core::ops::BitXor for i16x8 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Self(v128_xor(self.0, rhs.0))
    }
}

impl core::ops::AddAssign for i16x8 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl core::ops::SubAssign for i16x8 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl core::ops::MulAssign for i16x8 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl core::ops::BitAndAssign for i16x8 {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

impl core::ops::BitOrAssign for i16x8 {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl core::ops::BitXorAssign for i16x8 {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = *self ^ rhs;
    }
}

impl core::ops::Index<usize> for i16x8 {
    type Output = i16;
    #[inline(always)]
    fn index(&self, i: usize) -> &Self::Output {
        assert!(i < 8, "index out of bounds");
        unsafe { &*(self as *const Self as *const i16).add(i) }
    }
}

impl core::ops::IndexMut<usize> for i16x8 {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        assert!(i < 8, "index out of bounds");
        unsafe { &mut *(self as *mut Self as *mut i16).add(i) }
    }
}

impl From<[i16; 8]> for i16x8 {
    #[inline(always)]
    fn from(arr: [i16; 8]) -> Self {
        // SAFETY: [i16; 8] and v128 have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}

impl From<i16x8> for [i16; 8] {
    #[inline(always)]
    fn from(v: i16x8) -> Self {
        // SAFETY: v128 and [i16; 8] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}

// ============================================================================
// u16x8 - 8 x u16 (128-bit WASM SIMD)
// ============================================================================

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct u16x8(v128);

impl u16x8 {
    pub const LANES: usize = 8;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::Simd128Token, data: &[u16; 8]) -> Self {
        Self(unsafe { v128_load(data.as_ptr() as *const v128) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::Simd128Token, v: u16) -> Self {
        Self(u16x8_splat(v))
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::Simd128Token) -> Self {
        Self(u16x8_splat(0u16))
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::Simd128Token, arr: [u16; 8]) -> Self {
        // SAFETY: [u16; 8] and v128 have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [u16; 8]) {
        unsafe { v128_store(out.as_mut_ptr() as *mut v128, self.0) };
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
    pub fn raw(self) -> v128 {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports WASM SIMD128.
    #[inline(always)]
    pub unsafe fn from_raw(v: v128) -> Self {
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
    pub fn cast_slice(_: archmage::Simd128Token, slice: &[u16]) -> Option<&[Self]> {
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
    pub fn cast_slice_mut(_: archmage::Simd128Token, slice: &mut [u16]) -> Option<&mut [Self]> {
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
        // SAFETY: Self is repr(transparent) over v128 which is 16 bytes
        unsafe { &*(self as *const Self as *const [u8; 16]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 16] {
        // SAFETY: Self is repr(transparent) over v128 which is 16 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 16]) }
    }

    /// Create from a byte array reference (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::Simd128Token, bytes: &[u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Create from an owned byte array (token-gated, zero-cost).
    ///
    /// This is a zero-cost transmute from an owned byte array.
    #[inline(always)]
    pub fn from_bytes_owned(_: archmage::Simd128Token, bytes: [u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(bytes) })
    }

    /// Element-wise minimum
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(u16x8_min(self.0, other.0))
    }

    /// Element-wise maximum
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(u16x8_max(self.0, other.0))
    }

    /// Clamp values between lo and hi
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }

    /// Reduce: sum all lanes
    #[inline(always)]
    pub fn reduce_add(self) -> u16 {
        let arr = self.to_array();
        arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7]
    }

    /// Element-wise equality comparison (returns mask)
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> Self {
        Self(u16x8_eq(self.0, other.0))
    }

    /// Element-wise inequality comparison (returns mask)
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(v128_not(u16x8_eq(self.0, other.0)))
    }

    /// Element-wise less-than comparison (returns mask)
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(u16x8_lt(self.0, other.0))
    }

    /// Element-wise less-than-or-equal comparison (returns mask)
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(u16x8_le(self.0, other.0))
    }

    /// Element-wise greater-than comparison (returns mask)
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(u16x8_gt(self.0, other.0))
    }

    /// Element-wise greater-than-or-equal comparison (returns mask)
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(u16x8_ge(self.0, other.0))
    }

    /// Blend two vectors based on a mask
    ///
    /// For each lane, selects from `self` if the corresponding mask lane is all-ones,
    /// otherwise selects from `other`.
    ///
    /// The mask should come from a comparison operation like `simd_lt()`.
    ///
    /// # Example
    /// ```ignore
    /// let a = u16x8::splat(token, 1.0);
    /// let b = u16x8::splat(token, 2.0);
    /// let mask = a.simd_lt(b);  // all true
    /// let result = a.blend(b, mask);  // selects from a (all ones in mask)
    /// ```
    #[inline(always)]
    pub fn blend(self, other: Self, mask: Self) -> Self {
        Self(v128_bitselect(self.0, other.0, mask.0))
    }

    /// Bitwise NOT
    #[inline(always)]
    pub fn not(self) -> Self {
        Self(v128_not(self.0))
    }

    /// Shift left by constant
    #[inline(always)]
    pub fn shl<const N: u32>(self) -> Self {
        Self(u16x8_shl(self.0, N))
    }

    /// Shift right by constant
    #[inline(always)]
    pub fn shr<const N: u32>(self) -> Self {
        Self(u16x8_shr(self.0, N))
    }

    /// Check if all lanes are non-zero (all true)
    #[inline(always)]
    pub fn all_true(self) -> bool {
        u16x8_all_true(self.0)
    }

    /// Check if any lane is non-zero (any true)
    #[inline(always)]
    pub fn any_true(self) -> bool {
        v128_any_true(self.0)
    }

    /// Extract the high bit of each lane as a bitmask
    #[inline(always)]
    pub fn bitmask(self) -> u32 {
        u16x8_bitmask(self.0) as u32
    }

    // ========== Bitcast (reinterpret bits, zero-cost) ==========

    /// Reinterpret bits as `i16x8` (zero-cost).
    #[inline(always)]
    pub fn bitcast_i16x8(self) -> i16x8 {
        i16x8(self.0)
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

impl core::ops::Add for u16x8 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(u16x8_add(self.0, rhs.0))
    }
}

impl core::ops::Sub for u16x8 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(u16x8_sub(self.0, rhs.0))
    }
}

impl core::ops::Mul for u16x8 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(u16x8_mul(self.0, rhs.0))
    }
}

impl core::ops::BitAnd for u16x8 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(v128_and(self.0, rhs.0))
    }
}

impl core::ops::BitOr for u16x8 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self(v128_or(self.0, rhs.0))
    }
}

impl core::ops::BitXor for u16x8 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Self(v128_xor(self.0, rhs.0))
    }
}

impl core::ops::AddAssign for u16x8 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl core::ops::SubAssign for u16x8 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl core::ops::MulAssign for u16x8 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl core::ops::BitAndAssign for u16x8 {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

impl core::ops::BitOrAssign for u16x8 {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl core::ops::BitXorAssign for u16x8 {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = *self ^ rhs;
    }
}

impl core::ops::Index<usize> for u16x8 {
    type Output = u16;
    #[inline(always)]
    fn index(&self, i: usize) -> &Self::Output {
        assert!(i < 8, "index out of bounds");
        unsafe { &*(self as *const Self as *const u16).add(i) }
    }
}

impl core::ops::IndexMut<usize> for u16x8 {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        assert!(i < 8, "index out of bounds");
        unsafe { &mut *(self as *mut Self as *mut u16).add(i) }
    }
}

impl From<[u16; 8]> for u16x8 {
    #[inline(always)]
    fn from(arr: [u16; 8]) -> Self {
        // SAFETY: [u16; 8] and v128 have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}

impl From<u16x8> for [u16; 8] {
    #[inline(always)]
    fn from(v: u16x8) -> Self {
        // SAFETY: v128 and [u16; 8] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}

// ============================================================================
// i32x4 - 4 x i32 (128-bit WASM SIMD)
// ============================================================================

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct i32x4(v128);

impl i32x4 {
    pub const LANES: usize = 4;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::Simd128Token, data: &[i32; 4]) -> Self {
        Self(unsafe { v128_load(data.as_ptr() as *const v128) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::Simd128Token, v: i32) -> Self {
        Self(i32x4_splat(v))
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::Simd128Token) -> Self {
        Self(i32x4_splat(0i32))
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::Simd128Token, arr: [i32; 4]) -> Self {
        // SAFETY: [i32; 4] and v128 have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [i32; 4]) {
        unsafe { v128_store(out.as_mut_ptr() as *mut v128, self.0) };
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
    pub fn raw(self) -> v128 {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports WASM SIMD128.
    #[inline(always)]
    pub unsafe fn from_raw(v: v128) -> Self {
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
    pub fn cast_slice(_: archmage::Simd128Token, slice: &[i32]) -> Option<&[Self]> {
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
    pub fn cast_slice_mut(_: archmage::Simd128Token, slice: &mut [i32]) -> Option<&mut [Self]> {
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
        // SAFETY: Self is repr(transparent) over v128 which is 16 bytes
        unsafe { &*(self as *const Self as *const [u8; 16]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 16] {
        // SAFETY: Self is repr(transparent) over v128 which is 16 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 16]) }
    }

    /// Create from a byte array reference (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::Simd128Token, bytes: &[u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Create from an owned byte array (token-gated, zero-cost).
    ///
    /// This is a zero-cost transmute from an owned byte array.
    #[inline(always)]
    pub fn from_bytes_owned(_: archmage::Simd128Token, bytes: [u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(bytes) })
    }

    /// Element-wise minimum
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(i32x4_min(self.0, other.0))
    }

    /// Element-wise maximum
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(i32x4_max(self.0, other.0))
    }

    /// Clamp values between lo and hi
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }

    /// Absolute value
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(i32x4_abs(self.0))
    }

    /// Reduce: sum all lanes
    #[inline(always)]
    pub fn reduce_add(self) -> i32 {
        i32x4_extract_lane::<0>(self.0)
            + i32x4_extract_lane::<1>(self.0)
            + i32x4_extract_lane::<2>(self.0)
            + i32x4_extract_lane::<3>(self.0)
    }

    /// Element-wise equality comparison (returns mask)
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> Self {
        Self(i32x4_eq(self.0, other.0))
    }

    /// Element-wise inequality comparison (returns mask)
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(v128_not(i32x4_eq(self.0, other.0)))
    }

    /// Element-wise less-than comparison (returns mask)
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(i32x4_lt(self.0, other.0))
    }

    /// Element-wise less-than-or-equal comparison (returns mask)
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(i32x4_le(self.0, other.0))
    }

    /// Element-wise greater-than comparison (returns mask)
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(i32x4_gt(self.0, other.0))
    }

    /// Element-wise greater-than-or-equal comparison (returns mask)
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(i32x4_ge(self.0, other.0))
    }

    /// Blend two vectors based on a mask
    ///
    /// For each lane, selects from `self` if the corresponding mask lane is all-ones,
    /// otherwise selects from `other`.
    ///
    /// The mask should come from a comparison operation like `simd_lt()`.
    ///
    /// # Example
    /// ```ignore
    /// let a = i32x4::splat(token, 1.0);
    /// let b = i32x4::splat(token, 2.0);
    /// let mask = a.simd_lt(b);  // all true
    /// let result = a.blend(b, mask);  // selects from a (all ones in mask)
    /// ```
    #[inline(always)]
    pub fn blend(self, other: Self, mask: Self) -> Self {
        Self(v128_bitselect(self.0, other.0, mask.0))
    }

    /// Bitwise NOT
    #[inline(always)]
    pub fn not(self) -> Self {
        Self(v128_not(self.0))
    }

    /// Shift left by constant
    #[inline(always)]
    pub fn shl<const N: u32>(self) -> Self {
        Self(i32x4_shl(self.0, N))
    }

    /// Shift right by constant
    ///
    /// For signed types, this is an arithmetic shift (sign-extending).
    #[inline(always)]
    pub fn shr<const N: u32>(self) -> Self {
        Self(i32x4_shr(self.0, N))
    }

    /// Check if all lanes are non-zero (all true)
    #[inline(always)]
    pub fn all_true(self) -> bool {
        i32x4_all_true(self.0)
    }

    /// Check if any lane is non-zero (any true)
    #[inline(always)]
    pub fn any_true(self) -> bool {
        v128_any_true(self.0)
    }

    /// Extract the high bit of each lane as a bitmask
    #[inline(always)]
    pub fn bitmask(self) -> u32 {
        i32x4_bitmask(self.0) as u32
    }

    // ========== Bitcast (reinterpret bits, zero-cost) ==========

    /// Reinterpret bits as `f32x4` (zero-cost).
    #[inline(always)]
    pub fn bitcast_f32x4(self) -> f32x4 {
        f32x4(self.0)
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
        u32x4(self.0)
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

impl core::ops::Add for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(i32x4_add(self.0, rhs.0))
    }
}

impl core::ops::Sub for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(i32x4_sub(self.0, rhs.0))
    }
}

impl core::ops::Mul for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(i32x4_mul(self.0, rhs.0))
    }
}

impl core::ops::Neg for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(i32x4_neg(self.0))
    }
}

impl core::ops::BitAnd for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(v128_and(self.0, rhs.0))
    }
}

impl core::ops::BitOr for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self(v128_or(self.0, rhs.0))
    }
}

impl core::ops::BitXor for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Self(v128_xor(self.0, rhs.0))
    }
}

impl core::ops::AddAssign for i32x4 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl core::ops::SubAssign for i32x4 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl core::ops::MulAssign for i32x4 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl core::ops::BitAndAssign for i32x4 {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

impl core::ops::BitOrAssign for i32x4 {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl core::ops::BitXorAssign for i32x4 {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = *self ^ rhs;
    }
}

impl core::ops::Index<usize> for i32x4 {
    type Output = i32;
    #[inline(always)]
    fn index(&self, i: usize) -> &Self::Output {
        assert!(i < 4, "index out of bounds");
        unsafe { &*(self as *const Self as *const i32).add(i) }
    }
}

impl core::ops::IndexMut<usize> for i32x4 {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        assert!(i < 4, "index out of bounds");
        unsafe { &mut *(self as *mut Self as *mut i32).add(i) }
    }
}

impl From<[i32; 4]> for i32x4 {
    #[inline(always)]
    fn from(arr: [i32; 4]) -> Self {
        // SAFETY: [i32; 4] and v128 have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}

impl From<i32x4> for [i32; 4] {
    #[inline(always)]
    fn from(v: i32x4) -> Self {
        // SAFETY: v128 and [i32; 4] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}

// ============================================================================
// u32x4 - 4 x u32 (128-bit WASM SIMD)
// ============================================================================

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct u32x4(v128);

impl u32x4 {
    pub const LANES: usize = 4;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::Simd128Token, data: &[u32; 4]) -> Self {
        Self(unsafe { v128_load(data.as_ptr() as *const v128) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::Simd128Token, v: u32) -> Self {
        Self(u32x4_splat(v))
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::Simd128Token) -> Self {
        Self(u32x4_splat(0u32))
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::Simd128Token, arr: [u32; 4]) -> Self {
        // SAFETY: [u32; 4] and v128 have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [u32; 4]) {
        unsafe { v128_store(out.as_mut_ptr() as *mut v128, self.0) };
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
    pub fn raw(self) -> v128 {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports WASM SIMD128.
    #[inline(always)]
    pub unsafe fn from_raw(v: v128) -> Self {
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
    pub fn cast_slice(_: archmage::Simd128Token, slice: &[u32]) -> Option<&[Self]> {
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
    pub fn cast_slice_mut(_: archmage::Simd128Token, slice: &mut [u32]) -> Option<&mut [Self]> {
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
        // SAFETY: Self is repr(transparent) over v128 which is 16 bytes
        unsafe { &*(self as *const Self as *const [u8; 16]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 16] {
        // SAFETY: Self is repr(transparent) over v128 which is 16 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 16]) }
    }

    /// Create from a byte array reference (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::Simd128Token, bytes: &[u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Create from an owned byte array (token-gated, zero-cost).
    ///
    /// This is a zero-cost transmute from an owned byte array.
    #[inline(always)]
    pub fn from_bytes_owned(_: archmage::Simd128Token, bytes: [u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(bytes) })
    }

    /// Element-wise minimum
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(u32x4_min(self.0, other.0))
    }

    /// Element-wise maximum
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(u32x4_max(self.0, other.0))
    }

    /// Clamp values between lo and hi
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }

    /// Reduce: sum all lanes
    #[inline(always)]
    pub fn reduce_add(self) -> u32 {
        u32x4_extract_lane::<0>(self.0)
            + u32x4_extract_lane::<1>(self.0)
            + u32x4_extract_lane::<2>(self.0)
            + u32x4_extract_lane::<3>(self.0)
    }

    /// Element-wise equality comparison (returns mask)
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> Self {
        Self(u32x4_eq(self.0, other.0))
    }

    /// Element-wise inequality comparison (returns mask)
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(v128_not(u32x4_eq(self.0, other.0)))
    }

    /// Element-wise less-than comparison (returns mask)
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(u32x4_lt(self.0, other.0))
    }

    /// Element-wise less-than-or-equal comparison (returns mask)
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(u32x4_le(self.0, other.0))
    }

    /// Element-wise greater-than comparison (returns mask)
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(u32x4_gt(self.0, other.0))
    }

    /// Element-wise greater-than-or-equal comparison (returns mask)
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(u32x4_ge(self.0, other.0))
    }

    /// Blend two vectors based on a mask
    ///
    /// For each lane, selects from `self` if the corresponding mask lane is all-ones,
    /// otherwise selects from `other`.
    ///
    /// The mask should come from a comparison operation like `simd_lt()`.
    ///
    /// # Example
    /// ```ignore
    /// let a = u32x4::splat(token, 1.0);
    /// let b = u32x4::splat(token, 2.0);
    /// let mask = a.simd_lt(b);  // all true
    /// let result = a.blend(b, mask);  // selects from a (all ones in mask)
    /// ```
    #[inline(always)]
    pub fn blend(self, other: Self, mask: Self) -> Self {
        Self(v128_bitselect(self.0, other.0, mask.0))
    }

    /// Bitwise NOT
    #[inline(always)]
    pub fn not(self) -> Self {
        Self(v128_not(self.0))
    }

    /// Shift left by constant
    #[inline(always)]
    pub fn shl<const N: u32>(self) -> Self {
        Self(u32x4_shl(self.0, N))
    }

    /// Shift right by constant
    #[inline(always)]
    pub fn shr<const N: u32>(self) -> Self {
        Self(u32x4_shr(self.0, N))
    }

    /// Check if all lanes are non-zero (all true)
    #[inline(always)]
    pub fn all_true(self) -> bool {
        u32x4_all_true(self.0)
    }

    /// Check if any lane is non-zero (any true)
    #[inline(always)]
    pub fn any_true(self) -> bool {
        v128_any_true(self.0)
    }

    /// Extract the high bit of each lane as a bitmask
    #[inline(always)]
    pub fn bitmask(self) -> u32 {
        u32x4_bitmask(self.0) as u32
    }

    // ========== Bitcast (reinterpret bits, zero-cost) ==========

    /// Reinterpret bits as `f32x4` (zero-cost).
    #[inline(always)]
    pub fn bitcast_f32x4(self) -> f32x4 {
        f32x4(self.0)
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
        i32x4(self.0)
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

impl core::ops::Add for u32x4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(u32x4_add(self.0, rhs.0))
    }
}

impl core::ops::Sub for u32x4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(u32x4_sub(self.0, rhs.0))
    }
}

impl core::ops::Mul for u32x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(u32x4_mul(self.0, rhs.0))
    }
}

impl core::ops::BitAnd for u32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(v128_and(self.0, rhs.0))
    }
}

impl core::ops::BitOr for u32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self(v128_or(self.0, rhs.0))
    }
}

impl core::ops::BitXor for u32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Self(v128_xor(self.0, rhs.0))
    }
}

impl core::ops::AddAssign for u32x4 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl core::ops::SubAssign for u32x4 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl core::ops::MulAssign for u32x4 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl core::ops::BitAndAssign for u32x4 {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

impl core::ops::BitOrAssign for u32x4 {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl core::ops::BitXorAssign for u32x4 {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = *self ^ rhs;
    }
}

impl core::ops::Index<usize> for u32x4 {
    type Output = u32;
    #[inline(always)]
    fn index(&self, i: usize) -> &Self::Output {
        assert!(i < 4, "index out of bounds");
        unsafe { &*(self as *const Self as *const u32).add(i) }
    }
}

impl core::ops::IndexMut<usize> for u32x4 {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        assert!(i < 4, "index out of bounds");
        unsafe { &mut *(self as *mut Self as *mut u32).add(i) }
    }
}

impl From<[u32; 4]> for u32x4 {
    #[inline(always)]
    fn from(arr: [u32; 4]) -> Self {
        // SAFETY: [u32; 4] and v128 have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}

impl From<u32x4> for [u32; 4] {
    #[inline(always)]
    fn from(v: u32x4) -> Self {
        // SAFETY: v128 and [u32; 4] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}

// ============================================================================
// i64x2 - 2 x i64 (128-bit WASM SIMD)
// ============================================================================

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct i64x2(v128);

impl i64x2 {
    pub const LANES: usize = 2;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::Simd128Token, data: &[i64; 2]) -> Self {
        Self(unsafe { v128_load(data.as_ptr() as *const v128) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::Simd128Token, v: i64) -> Self {
        Self(i64x2_splat(v))
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::Simd128Token) -> Self {
        Self(i64x2_splat(0i64))
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::Simd128Token, arr: [i64; 2]) -> Self {
        // SAFETY: [i64; 2] and v128 have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [i64; 2]) {
        unsafe { v128_store(out.as_mut_ptr() as *mut v128, self.0) };
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
    pub fn raw(self) -> v128 {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports WASM SIMD128.
    #[inline(always)]
    pub unsafe fn from_raw(v: v128) -> Self {
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
    pub fn cast_slice(_: archmage::Simd128Token, slice: &[i64]) -> Option<&[Self]> {
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
    pub fn cast_slice_mut(_: archmage::Simd128Token, slice: &mut [i64]) -> Option<&mut [Self]> {
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
        // SAFETY: Self is repr(transparent) over v128 which is 16 bytes
        unsafe { &*(self as *const Self as *const [u8; 16]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 16] {
        // SAFETY: Self is repr(transparent) over v128 which is 16 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 16]) }
    }

    /// Create from a byte array reference (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::Simd128Token, bytes: &[u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Create from an owned byte array (token-gated, zero-cost).
    ///
    /// This is a zero-cost transmute from an owned byte array.
    #[inline(always)]
    pub fn from_bytes_owned(_: archmage::Simd128Token, bytes: [u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(bytes) })
    }

    /// Reduce: sum all lanes
    #[inline(always)]
    pub fn reduce_add(self) -> i64 {
        i64x2_extract_lane::<0>(self.0) + i64x2_extract_lane::<1>(self.0)
    }

    /// Element-wise equality comparison (returns mask)
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> Self {
        Self(i64x2_eq(self.0, other.0))
    }

    /// Element-wise inequality comparison (returns mask)
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(v128_not(i64x2_eq(self.0, other.0)))
    }

    /// Element-wise less-than comparison (returns mask)
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(i64x2_lt(self.0, other.0))
    }

    /// Element-wise less-than-or-equal comparison (returns mask)
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(i64x2_le(self.0, other.0))
    }

    /// Element-wise greater-than comparison (returns mask)
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(i64x2_gt(self.0, other.0))
    }

    /// Element-wise greater-than-or-equal comparison (returns mask)
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(i64x2_ge(self.0, other.0))
    }

    /// Blend two vectors based on a mask
    ///
    /// For each lane, selects from `self` if the corresponding mask lane is all-ones,
    /// otherwise selects from `other`.
    ///
    /// The mask should come from a comparison operation like `simd_lt()`.
    ///
    /// # Example
    /// ```ignore
    /// let a = i64x2::splat(token, 1.0);
    /// let b = i64x2::splat(token, 2.0);
    /// let mask = a.simd_lt(b);  // all true
    /// let result = a.blend(b, mask);  // selects from a (all ones in mask)
    /// ```
    #[inline(always)]
    pub fn blend(self, other: Self, mask: Self) -> Self {
        Self(v128_bitselect(self.0, other.0, mask.0))
    }

    /// Bitwise NOT
    #[inline(always)]
    pub fn not(self) -> Self {
        Self(v128_not(self.0))
    }

    /// Shift left by constant
    #[inline(always)]
    pub fn shl<const N: u32>(self) -> Self {
        Self(i64x2_shl(self.0, N))
    }

    /// Shift right by constant
    ///
    /// For signed types, this is an arithmetic shift (sign-extending).
    #[inline(always)]
    pub fn shr<const N: u32>(self) -> Self {
        Self(i64x2_shr(self.0, N))
    }

    /// Check if all lanes are non-zero (all true)
    #[inline(always)]
    pub fn all_true(self) -> bool {
        i64x2_all_true(self.0)
    }

    /// Check if any lane is non-zero (any true)
    #[inline(always)]
    pub fn any_true(self) -> bool {
        v128_any_true(self.0)
    }

    /// Extract the high bit of each lane as a bitmask
    #[inline(always)]
    pub fn bitmask(self) -> u32 {
        i64x2_bitmask(self.0) as u32
    }

    // ========== Bitcast (reinterpret bits, zero-cost) ==========

    /// Reinterpret bits as `f64x2` (zero-cost).
    #[inline(always)]
    pub fn bitcast_f64x2(self) -> f64x2 {
        f64x2(self.0)
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
        u64x2(self.0)
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

impl core::ops::Add for i64x2 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(i64x2_add(self.0, rhs.0))
    }
}

impl core::ops::Sub for i64x2 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(i64x2_sub(self.0, rhs.0))
    }
}

impl core::ops::Mul for i64x2 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(i64x2_mul(self.0, rhs.0))
    }
}

impl core::ops::Neg for i64x2 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(i64x2_neg(self.0))
    }
}

impl core::ops::BitAnd for i64x2 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(v128_and(self.0, rhs.0))
    }
}

impl core::ops::BitOr for i64x2 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self(v128_or(self.0, rhs.0))
    }
}

impl core::ops::BitXor for i64x2 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Self(v128_xor(self.0, rhs.0))
    }
}

impl core::ops::AddAssign for i64x2 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl core::ops::SubAssign for i64x2 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl core::ops::MulAssign for i64x2 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl core::ops::BitAndAssign for i64x2 {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

impl core::ops::BitOrAssign for i64x2 {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl core::ops::BitXorAssign for i64x2 {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = *self ^ rhs;
    }
}

impl core::ops::Index<usize> for i64x2 {
    type Output = i64;
    #[inline(always)]
    fn index(&self, i: usize) -> &Self::Output {
        assert!(i < 2, "index out of bounds");
        unsafe { &*(self as *const Self as *const i64).add(i) }
    }
}

impl core::ops::IndexMut<usize> for i64x2 {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        assert!(i < 2, "index out of bounds");
        unsafe { &mut *(self as *mut Self as *mut i64).add(i) }
    }
}

impl From<[i64; 2]> for i64x2 {
    #[inline(always)]
    fn from(arr: [i64; 2]) -> Self {
        // SAFETY: [i64; 2] and v128 have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}

impl From<i64x2> for [i64; 2] {
    #[inline(always)]
    fn from(v: i64x2) -> Self {
        // SAFETY: v128 and [i64; 2] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}

// ============================================================================
// u64x2 - 2 x u64 (128-bit WASM SIMD)
// ============================================================================

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct u64x2(v128);

impl u64x2 {
    pub const LANES: usize = 2;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::Simd128Token, data: &[u64; 2]) -> Self {
        Self(unsafe { v128_load(data.as_ptr() as *const v128) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::Simd128Token, v: u64) -> Self {
        Self(u64x2_splat(v))
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::Simd128Token) -> Self {
        Self(u64x2_splat(0u64))
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::Simd128Token, arr: [u64; 2]) -> Self {
        // SAFETY: [u64; 2] and v128 have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [u64; 2]) {
        unsafe { v128_store(out.as_mut_ptr() as *mut v128, self.0) };
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
    pub fn raw(self) -> v128 {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports WASM SIMD128.
    #[inline(always)]
    pub unsafe fn from_raw(v: v128) -> Self {
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
    pub fn cast_slice(_: archmage::Simd128Token, slice: &[u64]) -> Option<&[Self]> {
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
    pub fn cast_slice_mut(_: archmage::Simd128Token, slice: &mut [u64]) -> Option<&mut [Self]> {
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
        // SAFETY: Self is repr(transparent) over v128 which is 16 bytes
        unsafe { &*(self as *const Self as *const [u8; 16]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 16] {
        // SAFETY: Self is repr(transparent) over v128 which is 16 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 16]) }
    }

    /// Create from a byte array reference (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::Simd128Token, bytes: &[u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Create from an owned byte array (token-gated, zero-cost).
    ///
    /// This is a zero-cost transmute from an owned byte array.
    #[inline(always)]
    pub fn from_bytes_owned(_: archmage::Simd128Token, bytes: [u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(bytes) })
    }

    /// Reduce: sum all lanes
    #[inline(always)]
    pub fn reduce_add(self) -> u64 {
        u64x2_extract_lane::<0>(self.0) + u64x2_extract_lane::<1>(self.0)
    }

    /// Element-wise equality comparison (returns mask)
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> Self {
        Self(u64x2_eq(self.0, other.0))
    }

    /// Element-wise inequality comparison (returns mask)
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(v128_not(u64x2_eq(self.0, other.0)))
    }

    /// Blend two vectors based on a mask
    ///
    /// For each lane, selects from `self` if the corresponding mask lane is all-ones,
    /// otherwise selects from `other`.
    ///
    /// The mask should come from a comparison operation like `simd_lt()`.
    ///
    /// # Example
    /// ```ignore
    /// let a = u64x2::splat(token, 1.0);
    /// let b = u64x2::splat(token, 2.0);
    /// let mask = a.simd_lt(b);  // all true
    /// let result = a.blend(b, mask);  // selects from a (all ones in mask)
    /// ```
    #[inline(always)]
    pub fn blend(self, other: Self, mask: Self) -> Self {
        Self(v128_bitselect(self.0, other.0, mask.0))
    }

    /// Bitwise NOT
    #[inline(always)]
    pub fn not(self) -> Self {
        Self(v128_not(self.0))
    }

    /// Shift left by constant
    #[inline(always)]
    pub fn shl<const N: u32>(self) -> Self {
        Self(u64x2_shl(self.0, N))
    }

    /// Shift right by constant
    #[inline(always)]
    pub fn shr<const N: u32>(self) -> Self {
        Self(u64x2_shr(self.0, N))
    }

    /// Check if all lanes are non-zero (all true)
    #[inline(always)]
    pub fn all_true(self) -> bool {
        u64x2_all_true(self.0)
    }

    /// Check if any lane is non-zero (any true)
    #[inline(always)]
    pub fn any_true(self) -> bool {
        v128_any_true(self.0)
    }

    /// Extract the high bit of each lane as a bitmask
    #[inline(always)]
    pub fn bitmask(self) -> u32 {
        u64x2_bitmask(self.0) as u32
    }

    // ========== Bitcast (reinterpret bits, zero-cost) ==========

    /// Reinterpret bits as `f64x2` (zero-cost).
    #[inline(always)]
    pub fn bitcast_f64x2(self) -> f64x2 {
        f64x2(self.0)
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
        i64x2(self.0)
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

impl core::ops::Add for u64x2 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(u64x2_add(self.0, rhs.0))
    }
}

impl core::ops::Sub for u64x2 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(u64x2_sub(self.0, rhs.0))
    }
}

impl core::ops::Mul for u64x2 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(u64x2_mul(self.0, rhs.0))
    }
}

impl core::ops::BitAnd for u64x2 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(v128_and(self.0, rhs.0))
    }
}

impl core::ops::BitOr for u64x2 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self(v128_or(self.0, rhs.0))
    }
}

impl core::ops::BitXor for u64x2 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Self(v128_xor(self.0, rhs.0))
    }
}

impl core::ops::AddAssign for u64x2 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl core::ops::SubAssign for u64x2 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl core::ops::MulAssign for u64x2 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl core::ops::BitAndAssign for u64x2 {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

impl core::ops::BitOrAssign for u64x2 {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl core::ops::BitXorAssign for u64x2 {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = *self ^ rhs;
    }
}

impl core::ops::Index<usize> for u64x2 {
    type Output = u64;
    #[inline(always)]
    fn index(&self, i: usize) -> &Self::Output {
        assert!(i < 2, "index out of bounds");
        unsafe { &*(self as *const Self as *const u64).add(i) }
    }
}

impl core::ops::IndexMut<usize> for u64x2 {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        assert!(i < 2, "index out of bounds");
        unsafe { &mut *(self as *mut Self as *mut u64).add(i) }
    }
}

impl From<[u64; 2]> for u64x2 {
    #[inline(always)]
    fn from(arr: [u64; 2]) -> Self {
        // SAFETY: [u64; 2] and v128 have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}

impl From<u64x2> for [u64; 2] {
    #[inline(always)]
    fn from(v: u64x2) -> Self {
        // SAFETY: v128 and [u64; 2] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}

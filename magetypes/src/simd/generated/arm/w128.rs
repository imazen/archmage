//! 128-bit (NEON) SIMD types.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

use core::arch::aarch64::*;

// ============================================================================
// f32x4 - 4 x f32 (128-bit NEON)
// ============================================================================

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct f32x4(float32x4_t);

impl f32x4 {
    pub const LANES: usize = 4;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::NeonToken, data: &[f32; 4]) -> Self {
        Self(unsafe { vld1q_f32(data.as_ptr()) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::NeonToken, v: f32) -> Self {
        Self(unsafe { vdupq_n_f32(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::NeonToken) -> Self {
        Self(unsafe { vdupq_n_f32(0.0f32) })
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::NeonToken, arr: [f32; 4]) -> Self {
        // SAFETY: [f32; 4] and float32x4_t have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [f32; 4]) {
        unsafe { vst1q_f32(out.as_mut_ptr(), self.0) };
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
    pub fn raw(self) -> float32x4_t {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports the required SIMD features.
    #[inline(always)]
    pub unsafe fn from_raw(v: float32x4_t) -> Self {
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
    pub fn cast_slice(_: archmage::NeonToken, slice: &[f32]) -> Option<&[Self]> {
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
    pub fn cast_slice_mut(_: archmage::NeonToken, slice: &mut [f32]) -> Option<&mut [Self]> {
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
        // SAFETY: Self is repr(transparent) over float32x4_t which is 16 bytes
        unsafe { &*(self as *const Self as *const [u8; 16]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 16] {
        // SAFETY: Self is repr(transparent) over float32x4_t which is 16 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 16]) }
    }

    /// Create from a byte array reference (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::NeonToken, bytes: &[u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Create from an owned byte array (token-gated, zero-cost).
    ///
    /// This is a zero-cost transmute from an owned byte array.
    #[inline(always)]
    pub fn from_bytes_owned(_: archmage::NeonToken, bytes: [u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(bytes) })
    }

    // ========== Implementation identification ==========

    /// Returns a string identifying this type's implementation.
    ///
    /// This is useful for verifying that the correct implementation is being used
    /// at compile time (via `-Ctarget-cpu`) or at runtime (via `#[magetypes]` dispatch).
    ///
    /// Returns `"arm::neon::f32x4"`.
    #[inline(always)]
    pub const fn implementation_name() -> &'static str {
        "arm::neon::f32x4"
    }

    /// Element-wise minimum
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(unsafe { vminq_f32(self.0, other.0) })
    }

    /// Element-wise maximum
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(unsafe { vmaxq_f32(self.0, other.0) })
    }

    /// Clamp values between lo and hi
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }

    /// Square root
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        Self(unsafe { vsqrtq_f32(self.0) })
    }

    /// Absolute value
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(unsafe { vabsq_f32(self.0) })
    }

    /// Floor
    #[inline(always)]
    pub fn floor(self) -> Self {
        Self(unsafe { vrndmq_f32(self.0) })
    }

    /// Ceil
    #[inline(always)]
    pub fn ceil(self) -> Self {
        Self(unsafe { vrndpq_f32(self.0) })
    }

    /// Round to nearest
    #[inline(always)]
    pub fn round(self) -> Self {
        Self(unsafe { vrndnq_f32(self.0) })
    }

    /// Fused multiply-add: self * a + b
    #[inline(always)]
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        Self(unsafe { vfmaq_f32(b.0, self.0, a.0) })
    }

    /// Fused multiply-subtract: self * a - b
    #[inline(always)]
    pub fn mul_sub(self, a: Self, b: Self) -> Self {
        let neg_b = unsafe { vnegq_f32(b.0) };
        Self(unsafe { vfmaq_f32(neg_b, self.0, a.0) })
    }

    // ========== Approximation Operations ==========

    /// Fast reciprocal approximation (1/x) with ~8-12 bit precision.
    ///
    /// For full precision, use `recip()` which applies Newton-Raphson refinement.
    #[inline(always)]
    pub fn rcp_approx(self) -> Self {
        Self(unsafe { vrecpeq_f32(self.0) })
    }

    /// Precise reciprocal (1/x) using Newton-Raphson refinement.
    ///
    /// More accurate than `rcp_approx()` but slower. For maximum speed
    /// with acceptable precision loss, use `rcp_approx()`.
    #[inline(always)]
    pub fn recip(self) -> Self {
        // Newton-Raphson: x' = x * (2 - a*x)
        let approx = self.rcp_approx();
        let two = Self(unsafe { vdupq_n_f32(2.0) });
        // One iteration gives ~24-bit precision from ~12-bit
        approx * (two - self * approx)
    }

    /// Fast reciprocal square root approximation (1/sqrt(x)) with ~8-12 bit precision.
    ///
    /// For full precision, use `rsqrt()` which applies Newton-Raphson refinement.
    #[inline(always)]
    pub fn rsqrt_approx(self) -> Self {
        Self(unsafe { vrsqrteq_f32(self.0) })
    }

    /// Precise reciprocal square root (1/sqrt(x)) using Newton-Raphson refinement.
    #[inline(always)]
    pub fn rsqrt(self) -> Self {
        // Newton-Raphson for rsqrt: y' = 0.5 * y * (3 - x * y * y)
        let approx = self.rsqrt_approx();
        let half = Self(unsafe { vdupq_n_f32(0.5) });
        let three = Self(unsafe { vdupq_n_f32(3.0) });
        half * approx * (three - self * approx * approx)
    }

    /// Reduce: sum all lanes
    #[inline(always)]
    pub fn reduce_add(self) -> f32 {
        unsafe {
            let sum = vpaddq_f32(self.0, self.0);
            let sum = vpaddq_f32(sum, sum);
            vgetq_lane_f32::<0>(sum)
        }
    }

    /// Reduce: max of all lanes
    #[inline(always)]
    pub fn reduce_max(self) -> f32 {
        unsafe {
            let m = vpmaxq_f32(self.0, self.0);
            let m = vpmaxq_f32(m, m);
            vgetq_lane_f32::<0>(m)
        }
    }

    /// Reduce: min of all lanes
    #[inline(always)]
    pub fn reduce_min(self) -> f32 {
        unsafe {
            let m = vpminq_f32(self.0, self.0);
            let m = vpminq_f32(m, m);
            vgetq_lane_f32::<0>(m)
        }
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
        Self(unsafe { vreinterpretq_f32_u32(vceqq_f32(self.0, other.0)) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if not equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        self.simd_eq(other).not()
    }

    /// Lane-wise less-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if self < other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(unsafe { vreinterpretq_f32_u32(vcltq_f32(self.0, other.0)) })
    }

    /// Lane-wise less-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if self <= other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(unsafe { vreinterpretq_f32_u32(vcleq_f32(self.0, other.0)) })
    }

    /// Lane-wise greater-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if self > other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe { vreinterpretq_f32_u32(vcgtq_f32(self.0, other.0)) })
    }

    /// Lane-wise greater-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if self >= other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(unsafe { vreinterpretq_f32_u32(vcgeq_f32(self.0, other.0)) })
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
        Self(unsafe { vbslq_f32(vreinterpretq_u32_f32(mask.0), if_true.0, if_false.0) })
    }

    /// Bitwise NOT (complement)
    #[inline(always)]
    pub fn not(self) -> Self {
        Self(unsafe { vreinterpretq_f32_u32(vmvnq_u32(vreinterpretq_u32_f32(self.0))) })
    }

    // ========== Type Conversions ==========

    /// Convert to signed 32-bit integers, rounding toward zero (truncation).
    ///
    /// Values outside the representable range become `i32::MIN` (0x80000000).
    #[inline(always)]
    pub fn to_i32x4(self) -> i32x4 {
        i32x4(unsafe { vcvtq_s32_f32(self.0) })
    }

    /// Convert to signed 32-bit integers, rounding to nearest even.
    ///
    /// Values outside the representable range become `i32::MIN` (0x80000000).
    #[inline(always)]
    pub fn to_i32x4_round(self) -> i32x4 {
        i32x4(unsafe { vcvtnq_s32_f32(self.0) })
    }

    /// Create from signed 32-bit integers.
    #[inline(always)]
    pub fn from_i32x4(v: i32x4) -> Self {
        Self(unsafe { vcvtq_f32_s32(v.0) })
    }

    // ========== Bitcast ==========
    /// Reinterpret bits as `i32x4` (zero-cost).
    #[inline(always)]
    pub fn bitcast_i32x4(self) -> i32x4 {
        i32x4(unsafe { vreinterpretq_s32_f32(self.0) })
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
        u32x4(unsafe { vreinterpretq_u32_f32(self.0) })
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
    // ========== Transcendental Operations (Polynomial Approximations) ==========

    /// Low-precision base-2 logarithm - unchecked variant (~7.7e-5 max relative error).
    ///
    /// Uses rational polynomial approximation.
    /// **Warning**: Does not handle edge cases (0, negative, inf, NaN).
    /// Use `log2_lowp()` for correct IEEE behavior.
    #[inline(always)]
    pub fn log2_lowp_unchecked(self) -> Self {
        const P0: f32 = -1.850_383_34e-6;
        const P1: f32 = 1.428_716_05;
        const P2: f32 = 0.742_458_73;
        const Q0: f32 = 0.990_328_14;
        const Q1: f32 = 1.009_671_86;
        const Q2: f32 = 0.174_093_43;
        const OFFSET: u32 = 0x3f2aaaab;

        unsafe {
            let x = self.0;

            // Extract exponent via bit manipulation
            let x_bits = vreinterpretq_u32_f32(x);
            let offset = vdupq_n_u32(OFFSET);
            let exp_bits = vsubq_u32(x_bits, offset);
            let exp_shifted = vreinterpretq_s32_u32(vshrq_n_u32::<23>(exp_bits));

            // Reconstruct mantissa
            let shift_back = vreinterpretq_u32_s32(vshlq_n_s32::<23>(exp_shifted));
            let mantissa_bits = vsubq_u32(x_bits, shift_back);
            let mantissa = vreinterpretq_f32_u32(mantissa_bits);
            let exp_val = vcvtq_f32_s32(exp_shifted);

            let one = vdupq_n_f32(1.0);
            let m = vsubq_f32(mantissa, one);

            // Horner's for numerator: P2*m^2 + P1*m + P0
            // vfmaq_f32(a, b, c) = a + b*c
            let yp = vfmaq_f32(vdupq_n_f32(P1), vdupq_n_f32(P2), m);
            let yp = vfmaq_f32(vdupq_n_f32(P0), yp, m);

            // Horner's for denominator: Q2*m^2 + Q1*m + Q0
            let yq = vfmaq_f32(vdupq_n_f32(Q1), vdupq_n_f32(Q2), m);
            let yq = vfmaq_f32(vdupq_n_f32(Q0), yq, m);

            Self(vaddq_f32(vdivq_f32(yp, yq), exp_val))
        }
    }

    /// Low-precision base-2 logarithm (~7.7e-5 max relative error).
    ///
    /// Handles edge cases: log2(0) = -inf, log2(negative) = NaN,
    /// log2(+inf) = +inf, log2(NaN) = NaN.
    #[inline(always)]
    pub fn log2_lowp(self) -> Self {
        let result = self.log2_lowp_unchecked();

        unsafe {
            let zero = vdupq_n_f32(0.0);
            let is_zero = vceqq_f32(self.0, zero);
            let is_neg = vcltq_f32(self.0, zero);
            let is_inf = vceqq_f32(self.0, vdupq_n_f32(f32::INFINITY));
            // NaN: x != x
            let is_nan = vmvnq_u32(vceqq_f32(self.0, self.0));

            let neg_inf = vdupq_n_f32(f32::NEG_INFINITY);
            let pos_inf = vdupq_n_f32(f32::INFINITY);
            let nan = vdupq_n_f32(f32::NAN);

            // vbslq_f32(mask, true_val, false_val)
            let r = vbslq_f32(is_zero, neg_inf, result.0);
            let r = vbslq_f32(is_neg, nan, r);
            let r = vbslq_f32(is_inf, pos_inf, r);
            let r = vbslq_f32(is_nan, nan, r);
            Self(r)
        }
    }

    /// Low-precision base-2 exponential - unchecked variant (~5.5e-3 max relative error).
    ///
    /// **Warning**: Clamps to [-126, 126]. Does not return inf for overflow.
    #[inline(always)]
    pub fn exp2_lowp_unchecked(self) -> Self {
        const C0: f32 = 1.0;
        const C1: f32 = core::f32::consts::LN_2;
        const C2: f32 = 0.240_226_5;
        const C3: f32 = 0.055_504_11;

        unsafe {
            // Clamp to safe range
            let x = vmaxq_f32(self.0, vdupq_n_f32(-126.0));
            let x = vminq_f32(x, vdupq_n_f32(126.0));

            let xi = vrndmq_f32(x);
            let xf = vsubq_f32(x, xi);

            // Polynomial for 2^frac using FMA
            // vfmaq_f32(a, b, c) = a + b*c
            let poly = vfmaq_f32(vdupq_n_f32(C2), vdupq_n_f32(C3), xf);
            let poly = vfmaq_f32(vdupq_n_f32(C1), poly, xf);
            let poly = vfmaq_f32(vdupq_n_f32(C0), poly, xf);

            // Scale by 2^integer using bit manipulation
            let xi_i32 = vcvtq_s32_f32(xi);
            let bias = vdupq_n_s32(127);
            let scale_bits = vshlq_n_s32::<23>(vaddq_s32(xi_i32, bias));
            let scale = vreinterpretq_f32_s32(scale_bits);

            Self(vmulq_f32(poly, scale))
        }
    }

    /// Low-precision base-2 exponential (~5.5e-3 max relative error).
    ///
    /// Handles edge cases: exp2(x > 128) = +inf, exp2(x < -150) = 0,
    /// exp2(NaN) = NaN.
    #[inline(always)]
    pub fn exp2_lowp(self) -> Self {
        let result = self.exp2_lowp_unchecked();

        unsafe {
            let is_overflow = vcgeq_f32(self.0, vdupq_n_f32(128.0));
            let is_underflow = vcltq_f32(self.0, vdupq_n_f32(-150.0));
            let is_nan = vmvnq_u32(vceqq_f32(self.0, self.0));

            let pos_inf = vdupq_n_f32(f32::INFINITY);
            let zero = vdupq_n_f32(0.0);
            let nan = vdupq_n_f32(f32::NAN);

            let r = vbslq_f32(is_overflow, pos_inf, result.0);
            let r = vbslq_f32(is_underflow, zero, r);
            let r = vbslq_f32(is_nan, nan, r);
            Self(r)
        }
    }

    /// Low-precision natural logarithm - unchecked variant.
    ///
    /// **Warning**: Does not handle edge cases (0, negative, inf, NaN).
    #[inline(always)]
    pub fn ln_lowp_unchecked(self) -> Self {
        const LN2: f32 = core::f32::consts::LN_2;
        unsafe { Self(vmulq_f32(self.log2_lowp_unchecked().0, vdupq_n_f32(LN2))) }
    }

    /// Low-precision natural logarithm.
    ///
    /// Computed as `log2_lowp(x) * ln(2)`.
    /// Handles edge cases correctly.
    #[inline(always)]
    pub fn ln_lowp(self) -> Self {
        const LN2: f32 = core::f32::consts::LN_2;
        unsafe { Self(vmulq_f32(self.log2_lowp().0, vdupq_n_f32(LN2))) }
    }

    /// Low-precision natural exponential - unchecked variant.
    ///
    /// **Warning**: Clamps to finite range. Does not return inf for overflow.
    #[inline(always)]
    pub fn exp_lowp_unchecked(self) -> Self {
        const LOG2_E: f32 = core::f32::consts::LOG2_E;
        unsafe { Self(vmulq_f32(self.0, vdupq_n_f32(LOG2_E))).exp2_lowp_unchecked() }
    }

    /// Low-precision natural exponential (e^x).
    ///
    /// Computed as `exp2_lowp(x * log2(e))`.
    /// Handles edge cases correctly.
    #[inline(always)]
    pub fn exp_lowp(self) -> Self {
        const LOG2_E: f32 = core::f32::consts::LOG2_E;
        unsafe { Self(vmulq_f32(self.0, vdupq_n_f32(LOG2_E))).exp2_lowp() }
    }

    /// Low-precision base-10 logarithm - unchecked variant.
    ///
    /// **Warning**: Does not handle edge cases (0, negative, inf, NaN).
    #[inline(always)]
    pub fn log10_lowp_unchecked(self) -> Self {
        const LOG10_2: f32 = core::f32::consts::LOG10_2;
        unsafe {
            Self(vmulq_f32(
                self.log2_lowp_unchecked().0,
                vdupq_n_f32(LOG10_2),
            ))
        }
    }

    /// Low-precision base-10 logarithm.
    ///
    /// Computed as `log2_lowp(x) * log10(2)`.
    /// Handles edge cases correctly.
    #[inline(always)]
    pub fn log10_lowp(self) -> Self {
        const LOG10_2: f32 = core::f32::consts::LOG10_2;
        unsafe { Self(vmulq_f32(self.log2_lowp().0, vdupq_n_f32(LOG10_2))) }
    }

    /// Low-precision power function - unchecked variant.
    ///
    /// **Warning**: Does not handle edge cases. Only valid for positive self values.
    #[inline(always)]
    pub fn pow_lowp_unchecked(self, n: f32) -> Self {
        unsafe {
            Self(vmulq_f32(self.log2_lowp_unchecked().0, vdupq_n_f32(n))).exp2_lowp_unchecked()
        }
    }

    /// Low-precision power function (self^n).
    ///
    /// Computed as `exp2_lowp(n * log2_lowp(self))`.
    /// Note: Only valid for positive self values.
    #[inline(always)]
    pub fn pow_lowp(self, n: f32) -> Self {
        unsafe { Self(vmulq_f32(self.log2_lowp().0, vdupq_n_f32(n))).exp2_lowp() }
    }

    // ========== Mid-Precision Transcendental Operations ==========

    /// Mid-precision base-2 logarithm (~3 ULP max error) - unchecked variant.
    ///
    /// Uses (a-1)/(a+1) transform with degree-6 odd polynomial.
    /// **Warning**: Does not handle edge cases (0, negative, inf, NaN).
    #[inline(always)]
    pub fn log2_midp_unchecked(self) -> Self {
        const SQRT2_OVER_2: u32 = 0x3f3504f3;
        const ONE: u32 = 0x3f800000;
        const MANTISSA_MASK: u32 = 0x007fffff;
        const EXPONENT_BIAS: i32 = 127;

        const C0: f32 = 2.885_390_08;
        const C1: f32 = 0.961_800_76;
        const C2: f32 = 0.576_974_45;
        const C3: f32 = 0.434_411_97;

        unsafe {
            let x_bits = vreinterpretq_u32_f32(self.0);

            // Normalize mantissa to [sqrt(2)/2, sqrt(2)]
            let offset = vdupq_n_u32(ONE - SQRT2_OVER_2);
            let adjusted = vaddq_u32(x_bits, offset);

            // Extract exponent
            let exp_raw = vreinterpretq_s32_u32(vshrq_n_u32::<23>(adjusted));
            let exp_biased = vsubq_s32(exp_raw, vdupq_n_s32(EXPONENT_BIAS));
            let n = vcvtq_f32_s32(exp_biased);

            // Reconstruct normalized mantissa
            let mantissa_bits = vandq_u32(adjusted, vdupq_n_u32(MANTISSA_MASK));
            let a_bits = vaddq_u32(mantissa_bits, vdupq_n_u32(SQRT2_OVER_2));
            let a = vreinterpretq_f32_u32(a_bits);

            // y = (a - 1) / (a + 1)
            let one = vdupq_n_f32(1.0);
            let y = vdivq_f32(vsubq_f32(a, one), vaddq_f32(a, one));
            let y2 = vmulq_f32(y, y);

            // Polynomial: y*(C0 + y^2*(C1 + y^2*(C2 + C3*y^2)))
            // vfmaq_f32(a, b, c) = a + b*c
            let poly = vfmaq_f32(vdupq_n_f32(C2), vdupq_n_f32(C3), y2);
            let poly = vfmaq_f32(vdupq_n_f32(C1), poly, y2);
            let poly = vfmaq_f32(vdupq_n_f32(C0), poly, y2);

            // result = y * poly + n
            Self(vfmaq_f32(n, poly, y))
        }
    }

    /// Mid-precision base-2 logarithm (~3 ULP max error).
    ///
    /// Handles edge cases: log2(0) = -inf, log2(negative) = NaN,
    /// log2(+inf) = +inf, log2(NaN) = NaN.
    #[inline(always)]
    pub fn log2_midp(self) -> Self {
        let result = self.log2_midp_unchecked();

        unsafe {
            let zero = vdupq_n_f32(0.0);
            let is_zero = vceqq_f32(self.0, zero);
            let is_neg = vcltq_f32(self.0, zero);
            let is_inf = vceqq_f32(self.0, vdupq_n_f32(f32::INFINITY));
            let is_nan = vmvnq_u32(vceqq_f32(self.0, self.0));

            let neg_inf = vdupq_n_f32(f32::NEG_INFINITY);
            let pos_inf = vdupq_n_f32(f32::INFINITY);
            let nan = vdupq_n_f32(f32::NAN);

            let r = vbslq_f32(is_zero, neg_inf, result.0);
            let r = vbslq_f32(is_neg, nan, r);
            let r = vbslq_f32(is_inf, pos_inf, r);
            let r = vbslq_f32(is_nan, nan, r);
            Self(r)
        }
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

            let zero = vdupq_n_f32(0.0);
            let is_positive = vcgtq_f32(self.0, zero);
            let abs_x = vabsq_f32(self.0);
            let is_small = vcltq_f32(abs_x, vdupq_n_f32(DENORM_LIMIT));
            let is_denorm = vandq_u32(is_positive, is_small);

            let scaled_x = vmulq_f32(self.0, vdupq_n_f32(SCALE_UP));
            let x_for_log = vbslq_f32(is_denorm, scaled_x, self.0);

            let result = Self(x_for_log).log2_midp();

            let adjusted = vsubq_f32(result.0, vdupq_n_f32(SCALE_ADJUST));
            Self(vbslq_f32(is_denorm, adjusted, result.0))
        }
    }

    /// Mid-precision base-2 exponential (~2 ULP max error) - unchecked variant.
    ///
    /// Uses degree-6 polynomial approximation.
    /// **Warning**: Does not handle edge cases (underflow, overflow).
    #[inline(always)]
    pub fn exp2_midp_unchecked(self) -> Self {
        const C0: f32 = 1.0;
        const C1: f32 = 0.693_147_182;
        const C2: f32 = 0.240_226_463;
        const C3: f32 = 0.055_504_545;
        const C4: f32 = 0.009_618_055;
        const C5: f32 = 0.001_333_37;
        const C6: f32 = 0.000_154_47;

        unsafe {
            let x = self.0;

            let xi = vrndmq_f32(x);
            let xf = vsubq_f32(x, xi);

            // Horner's method with 6 coefficients
            let poly = vfmaq_f32(vdupq_n_f32(C5), vdupq_n_f32(C6), xf);
            let poly = vfmaq_f32(vdupq_n_f32(C4), poly, xf);
            let poly = vfmaq_f32(vdupq_n_f32(C3), poly, xf);
            let poly = vfmaq_f32(vdupq_n_f32(C2), poly, xf);
            let poly = vfmaq_f32(vdupq_n_f32(C1), poly, xf);
            let poly = vfmaq_f32(vdupq_n_f32(C0), poly, xf);

            // Scale by 2^integer
            let xi_i32 = vcvtq_s32_f32(xi);
            let bias = vdupq_n_s32(127);
            let scale_bits = vshlq_n_s32::<23>(vaddq_s32(xi_i32, bias));
            let scale = vreinterpretq_f32_s32(scale_bits);

            Self(vmulq_f32(poly, scale))
        }
    }

    /// Mid-precision base-2 exponential (~2 ULP max error).
    ///
    /// Handles edge cases: exp2(x > 128) = +inf, exp2(x < -150) = 0,
    /// exp2(NaN) = NaN.
    #[inline(always)]
    pub fn exp2_midp(self) -> Self {
        unsafe {
            let x = self.0;

            // Clamp to prevent overflow in intermediate calculations
            let x_clamped = vmaxq_f32(x, vdupq_n_f32(-150.0));
            let x_clamped = vminq_f32(x_clamped, vdupq_n_f32(128.0));

            let exp_result = Self(x_clamped).exp2_midp_unchecked().0;

            let is_underflow = vcltq_f32(x, vdupq_n_f32(-150.0));
            let is_overflow = vcgtq_f32(x, vdupq_n_f32(128.0));

            let zero = vdupq_n_f32(0.0);
            let inf = vdupq_n_f32(f32::INFINITY);

            let r = vbslq_f32(is_underflow, zero, exp_result);
            let r = vbslq_f32(is_overflow, inf, r);
            Self(r)
        }
    }

    /// Mid-precision natural logarithm - unchecked variant.
    ///
    /// Computed as `log2_midp_unchecked(x) * ln(2)`.
    #[inline(always)]
    pub fn ln_midp_unchecked(self) -> Self {
        const LN2: f32 = core::f32::consts::LN_2;
        unsafe { Self(vmulq_f32(self.log2_midp_unchecked().0, vdupq_n_f32(LN2))) }
    }

    /// Mid-precision natural logarithm with edge case handling.
    ///
    /// Returns -inf for 0, NaN for negative, correct results for inf/NaN.
    #[inline(always)]
    pub fn ln_midp(self) -> Self {
        const LN2: f32 = core::f32::consts::LN_2;
        unsafe { Self(vmulq_f32(self.log2_midp().0, vdupq_n_f32(LN2))) }
    }

    /// Mid-precision natural logarithm with full IEEE compliance.
    ///
    /// Handles all edge cases including denormals.
    #[inline(always)]
    pub fn ln_midp_precise(self) -> Self {
        const LN2: f32 = core::f32::consts::LN_2;
        unsafe { Self(vmulq_f32(self.log2_midp_precise().0, vdupq_n_f32(LN2))) }
    }

    /// Mid-precision natural exponential (e^x) - unchecked variant.
    ///
    /// Computed as `exp2_midp_unchecked(x * log2(e))`.
    #[inline(always)]
    pub fn exp_midp_unchecked(self) -> Self {
        const LOG2_E: f32 = core::f32::consts::LOG2_E;
        unsafe { Self(vmulq_f32(self.0, vdupq_n_f32(LOG2_E))).exp2_midp_unchecked() }
    }

    /// Mid-precision natural exponential (e^x) with edge case handling.
    ///
    /// Returns 0 for large negative, inf for large positive, correct results for inf/NaN.
    #[inline(always)]
    pub fn exp_midp(self) -> Self {
        const LOG2_E: f32 = core::f32::consts::LOG2_E;
        unsafe { Self(vmulq_f32(self.0, vdupq_n_f32(LOG2_E))).exp2_midp() }
    }

    /// Mid-precision base-10 logarithm - unchecked variant.
    ///
    /// Computed as `log2_midp_unchecked(x) * log10(2)`.
    #[inline(always)]
    pub fn log10_midp_unchecked(self) -> Self {
        const LOG10_2: f32 = core::f32::consts::LOG10_2;
        unsafe {
            Self(vmulq_f32(
                self.log2_midp_unchecked().0,
                vdupq_n_f32(LOG10_2),
            ))
        }
    }

    /// Mid-precision base-10 logarithm with edge case handling.
    ///
    /// Computed as `log2_midp(x) * log10(2)`.
    #[inline(always)]
    pub fn log10_midp(self) -> Self {
        const LOG10_2: f32 = core::f32::consts::LOG10_2;
        unsafe { Self(vmulq_f32(self.log2_midp().0, vdupq_n_f32(LOG10_2))) }
    }

    /// Mid-precision base-10 logarithm with full IEEE compliance.
    ///
    /// Handles all edge cases including denormals.
    #[inline(always)]
    pub fn log10_midp_precise(self) -> Self {
        const LOG10_2: f32 = core::f32::consts::LOG10_2;
        unsafe { Self(vmulq_f32(self.log2_midp_precise().0, vdupq_n_f32(LOG10_2))) }
    }

    /// Mid-precision power function (self^n) - unchecked variant.
    ///
    /// Computed as `exp2_midp_unchecked(n * log2_midp_unchecked(self))`.
    /// Note: Only valid for positive self values.
    #[inline(always)]
    pub fn pow_midp_unchecked(self, n: f32) -> Self {
        unsafe {
            Self(vmulq_f32(self.log2_midp_unchecked().0, vdupq_n_f32(n))).exp2_midp_unchecked()
        }
    }

    /// Mid-precision power function (self^n) with edge case handling.
    ///
    /// Handles 0, negative, inf, and NaN in base. Only valid for positive self values.
    #[inline(always)]
    pub fn pow_midp(self, n: f32) -> Self {
        unsafe { Self(vmulq_f32(self.log2_midp().0, vdupq_n_f32(n))).exp2_midp() }
    }

    /// Mid-precision power function with denormal handling.
    ///
    /// Uses `log2_midp_precise()` to handle denormal inputs correctly.
    /// Note: Only valid for positive self values.
    #[inline(always)]
    pub fn pow_midp_precise(self, n: f32) -> Self {
        unsafe { Self(vmulq_f32(self.log2_midp_precise().0, vdupq_n_f32(n))).exp2_midp() }
    }

    /// Mid-precision cube root (~1 ULP max error).
    ///
    /// Uses scalar extraction for initial guess + Newton-Raphson.
    /// Handles negative values correctly (returns -cbrt(|x|)).
    ///
    /// Does not handle denormals. Use `cbrt_midp_precise()` if denormal support is needed.
    #[inline(always)]
    pub fn cbrt_midp(self) -> Self {
        const TWO_THIRDS: f32 = 0.666_666_627;

        unsafe {
            let x = self.0;

            // Save sign and work with absolute value
            let abs_x = vabsq_f32(x);
            let sign_mask = vdupq_n_f32(-0.0);
            let sign = vreinterpretq_f32_u32(vandq_u32(
                vreinterpretq_u32_f32(x),
                vreinterpretq_u32_f32(sign_mask),
            ));

            // Extract to scalar for initial approximation
            let arr: [f32; 4] = core::mem::transmute(abs_x);
            let approx: [f32; 4] = [
                f32::from_bits((arr[0].to_bits() / 3) + 0x2a508c2d),
                f32::from_bits((arr[1].to_bits() / 3) + 0x2a508c2d),
                f32::from_bits((arr[2].to_bits() / 3) + 0x2a508c2d),
                f32::from_bits((arr[3].to_bits() / 3) + 0x2a508c2d),
            ];
            let mut y = vld1q_f32(approx.as_ptr());

            // Newton-Raphson iterations: y = y * (2/3 + x/(3*y^3))
            for _ in 0..3 {
                let y2 = vmulq_f32(y, y);
                let y3 = vmulq_f32(y2, y);
                let term = vdivq_f32(abs_x, vmulq_f32(vdupq_n_f32(3.0), y3));
                y = vmulq_f32(y, vaddq_f32(vdupq_n_f32(TWO_THIRDS), term));
            }

            // Restore sign
            Self(vreinterpretq_f32_u32(vorrq_u32(
                vreinterpretq_u32_f32(y),
                vreinterpretq_u32_f32(sign),
            )))
        }
    }

    /// Mid-precision cube root with denormal handling (~1 ULP max error).
    ///
    /// Handles negative values correctly (returns -cbrt(|x|)).
    /// Handles all edge cases including denormals. About 67% slower than `cbrt_midp()`.
    #[inline(always)]
    pub fn cbrt_midp_precise(self) -> Self {
        unsafe {
            const SCALE_UP: f32 = 16777216.0; // 2^24
            const SCALE_DOWN: f32 = 0.00390625; // 2^(-8) = cbrt(2^(-24))
            const DENORM_LIMIT: f32 = 1.17549435e-38;

            let abs_x = vabsq_f32(self.0);
            let is_denorm = vcltq_f32(abs_x, vdupq_n_f32(DENORM_LIMIT));
            // Exclude zeros from denormal handling
            let is_zero = vceqq_f32(self.0, vdupq_n_f32(0.0));
            let is_denorm = vandq_u32(is_denorm, vmvnq_u32(is_zero));

            let scaled_x = vmulq_f32(self.0, vdupq_n_f32(SCALE_UP));
            let x_for_cbrt = vbslq_f32(is_denorm, scaled_x, self.0);

            let result = Self(x_for_cbrt).cbrt_midp();

            let scaled_result = vmulq_f32(result.0, vdupq_n_f32(SCALE_DOWN));
            Self(vbslq_f32(is_denorm, scaled_result, result.0))
        }
    }

    // ========== Interleave Operations ==========

    /// Interleave low elements: [a0,a1,a2,a3] + [b0,b1,b2,b3] → [a0,b0,a1,b1]
    #[inline(always)]
    pub fn interleave_lo(self, other: Self) -> Self {
        unsafe { Self(vzip1q_f32(self.0, other.0)) }
    }

    /// Interleave high elements: [a0,a1,a2,a3] + [b0,b1,b2,b3] → [a2,b2,a3,b3]
    #[inline(always)]
    pub fn interleave_hi(self, other: Self) -> Self {
        unsafe { Self(vzip2q_f32(self.0, other.0)) }
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

    // ========== Matrix Transpose ==========

    /// Transpose a 4x4 matrix represented as 4 row vectors.
    ///
    /// After transpose, `rows[i][j]` becomes `rows[j][i]`.
    #[inline]
    pub fn transpose_4x4(rows: &mut [Self; 4]) {
        unsafe {
            // Step 1: zip pairs
            // t0 = [a0,c0,a1,c1], t1 = [a2,c2,a3,c3]
            let t0 = vzip1q_f32(rows[0].0, rows[2].0);
            let t1 = vzip2q_f32(rows[0].0, rows[2].0);
            // t2 = [b0,d0,b1,d1], t3 = [b2,d2,b3,d3]
            let t2 = vzip1q_f32(rows[1].0, rows[3].0);
            let t3 = vzip2q_f32(rows[1].0, rows[3].0);

            // Step 2: zip again to get final columns
            rows[0] = Self(vzip1q_f32(t0, t2)); // [a0,b0,c0,d0]
            rows[1] = Self(vzip2q_f32(t0, t2)); // [a1,b1,c1,d1]
            rows[2] = Self(vzip1q_f32(t1, t3)); // [a2,b2,c2,d2]
            rows[3] = Self(vzip2q_f32(t1, t3)); // [a3,b3,c3,d3]
        }
    }

    /// Transpose a 4x4 matrix, returning the transposed rows.
    #[inline]
    pub fn transpose_4x4_copy(rows: [Self; 4]) -> [Self; 4] {
        let mut result = rows;
        Self::transpose_4x4(&mut result);
        result
    }

    // ========== Load and Convert ==========

    /// Load 4 u8 values and convert to f32x4.
    ///
    /// Useful for image processing: load pixel values directly to float.
    #[inline(always)]
    pub fn from_u8(bytes: &[u8; 4]) -> Self {
        unsafe {
            // Load 4 bytes as a u32 into lane 0 of a u8x16 vector
            let val = u32::from_ne_bytes(*bytes);
            let v = vsetq_lane_u32::<0>(val, vdupq_n_u32(0));
            let v8 = vreinterpretq_u8_u32(v);
            // Extend u8 -> u16 -> u32 -> f32
            let v16 = vmovl_u8(vget_low_u8(v8));
            let v32 = vmovl_u16(vget_low_u16(v16));
            Self(vcvtq_f32_u32(v32))
        }
    }

    /// Convert to 4 u8 values with saturation.
    ///
    /// Values are clamped to [0, 255] and rounded.
    #[inline(always)]
    pub fn to_u8(self) -> [u8; 4] {
        unsafe {
            // Round to nearest i32
            let i32s = vcvtnq_s32_f32(self.0);
            // Narrow i32 -> i16 (signed saturation)
            let i16s = vqmovn_s32(i32s);
            // Narrow i16 -> u8 (unsigned saturation, clamps to [0, 255])
            let u8s = vqmovun_s16(vcombine_s16(i16s, vdup_n_s16(0)));
            let val = vget_lane_u32::<0>(vreinterpret_u32_u8(u8s));
            val.to_ne_bytes()
        }
    }

    /// Load 4 RGBA u8 pixels and deinterleave to 4 f32x4 channel vectors.
    ///
    /// Input: 16 bytes = 4 RGBA pixels in interleaved format.
    /// Output: (R, G, B, A) where each is f32x4 with values in [0.0, 255.0].
    #[inline]
    pub fn load_4_rgba_u8(rgba: &[u8; 16]) -> (Self, Self, Self, Self) {
        unsafe {
            // Load 16 bytes and reinterpret as 4 u32 pixels
            let v = vld1q_u8(rgba.as_ptr());
            let v32 = vreinterpretq_u32_u8(v);
            let mask = vdupq_n_u32(0xFF);

            // Extract channels via mask and shift
            let r_u32 = vandq_u32(v32, mask);
            let g_u32 = vandq_u32(vshrq_n_u32::<8>(v32), mask);
            let b_u32 = vandq_u32(vshrq_n_u32::<16>(v32), mask);
            let a_u32 = vshrq_n_u32::<24>(v32);

            (
                Self(vcvtq_f32_u32(r_u32)),
                Self(vcvtq_f32_u32(g_u32)),
                Self(vcvtq_f32_u32(b_u32)),
                Self(vcvtq_f32_u32(a_u32)),
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
            // Round to nearest i32 and clamp to [0, 255]
            let zero = vdupq_n_s32(0);
            let max_val = vdupq_n_s32(255);
            let ri = vminq_s32(vmaxq_s32(vcvtnq_s32_f32(r.0), zero), max_val);
            let gi = vminq_s32(vmaxq_s32(vcvtnq_s32_f32(g.0), zero), max_val);
            let bi = vminq_s32(vmaxq_s32(vcvtnq_s32_f32(b.0), zero), max_val);
            let ai = vminq_s32(vmaxq_s32(vcvtnq_s32_f32(a.0), zero), max_val);

            // Combine channels: R | (G << 8) | (B << 16) | (A << 24)
            let ri = vreinterpretq_u32_s32(ri);
            let gi = vreinterpretq_u32_s32(gi);
            let bi = vreinterpretq_u32_s32(bi);
            let ai = vreinterpretq_u32_s32(ai);

            let pixels = vorrq_u32(
                vorrq_u32(ri, vshlq_n_u32::<8>(gi)),
                vorrq_u32(vshlq_n_u32::<16>(bi), vshlq_n_u32::<24>(ai)),
            );

            let mut out = [0u8; 16];
            vst1q_u8(out.as_mut_ptr(), vreinterpretq_u8_u32(pixels));
            out
        }
    }
}

impl core::ops::Add for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(unsafe { vaddq_f32(self.0, rhs.0) })
    }
}

impl core::ops::Sub for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(unsafe { vsubq_f32(self.0, rhs.0) })
    }
}

impl core::ops::Mul for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(unsafe { vmulq_f32(self.0, rhs.0) })
    }
}

impl core::ops::Div for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        Self(unsafe { vdivq_f32(self.0, rhs.0) })
    }
}

impl core::ops::Neg for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(unsafe { vnegq_f32(self.0) })
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
        // SAFETY: [f32; 4] and float32x4_t have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}

impl From<f32x4> for [f32; 4] {
    #[inline(always)]
    fn from(v: f32x4) -> Self {
        // SAFETY: float32x4_t and [f32; 4] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}

// ============================================================================
// f64x2 - 2 x f64 (128-bit NEON)
// ============================================================================

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct f64x2(float64x2_t);

impl f64x2 {
    pub const LANES: usize = 2;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::NeonToken, data: &[f64; 2]) -> Self {
        Self(unsafe { vld1q_f64(data.as_ptr()) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::NeonToken, v: f64) -> Self {
        Self(unsafe { vdupq_n_f64(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::NeonToken) -> Self {
        Self(unsafe { vdupq_n_f64(0.0f64) })
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::NeonToken, arr: [f64; 2]) -> Self {
        // SAFETY: [f64; 2] and float64x2_t have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [f64; 2]) {
        unsafe { vst1q_f64(out.as_mut_ptr(), self.0) };
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
    pub fn raw(self) -> float64x2_t {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports the required SIMD features.
    #[inline(always)]
    pub unsafe fn from_raw(v: float64x2_t) -> Self {
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
    pub fn cast_slice(_: archmage::NeonToken, slice: &[f64]) -> Option<&[Self]> {
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
    pub fn cast_slice_mut(_: archmage::NeonToken, slice: &mut [f64]) -> Option<&mut [Self]> {
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
        // SAFETY: Self is repr(transparent) over float64x2_t which is 16 bytes
        unsafe { &*(self as *const Self as *const [u8; 16]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 16] {
        // SAFETY: Self is repr(transparent) over float64x2_t which is 16 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 16]) }
    }

    /// Create from a byte array reference (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::NeonToken, bytes: &[u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Create from an owned byte array (token-gated, zero-cost).
    ///
    /// This is a zero-cost transmute from an owned byte array.
    #[inline(always)]
    pub fn from_bytes_owned(_: archmage::NeonToken, bytes: [u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(bytes) })
    }

    // ========== Implementation identification ==========

    /// Returns a string identifying this type's implementation.
    ///
    /// This is useful for verifying that the correct implementation is being used
    /// at compile time (via `-Ctarget-cpu`) or at runtime (via `#[magetypes]` dispatch).
    ///
    /// Returns `"arm::neon::f64x2"`.
    #[inline(always)]
    pub const fn implementation_name() -> &'static str {
        "arm::neon::f64x2"
    }

    /// Element-wise minimum
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(unsafe { vminq_f64(self.0, other.0) })
    }

    /// Element-wise maximum
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(unsafe { vmaxq_f64(self.0, other.0) })
    }

    /// Clamp values between lo and hi
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }

    /// Square root
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        Self(unsafe { vsqrtq_f64(self.0) })
    }

    /// Absolute value
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(unsafe { vabsq_f64(self.0) })
    }

    /// Floor
    #[inline(always)]
    pub fn floor(self) -> Self {
        Self(unsafe { vrndmq_f64(self.0) })
    }

    /// Ceil
    #[inline(always)]
    pub fn ceil(self) -> Self {
        Self(unsafe { vrndpq_f64(self.0) })
    }

    /// Round to nearest
    #[inline(always)]
    pub fn round(self) -> Self {
        Self(unsafe { vrndnq_f64(self.0) })
    }

    /// Fused multiply-add: self * a + b
    #[inline(always)]
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        Self(unsafe { vfmaq_f64(b.0, self.0, a.0) })
    }

    /// Fused multiply-subtract: self * a - b
    #[inline(always)]
    pub fn mul_sub(self, a: Self, b: Self) -> Self {
        let neg_b = unsafe { vnegq_f64(b.0) };
        Self(unsafe { vfmaq_f64(neg_b, self.0, a.0) })
    }

    /// Reduce: sum all lanes
    #[inline(always)]
    pub fn reduce_add(self) -> f64 {
        unsafe {
            let sum = vpaddq_f64(self.0, self.0);
            vgetq_lane_f64::<0>(sum)
        }
    }

    /// Reduce: max of all lanes
    #[inline(always)]
    pub fn reduce_max(self) -> f64 {
        unsafe {
            let m = vpmaxq_f64(self.0, self.0);
            vgetq_lane_f64::<0>(m)
        }
    }

    /// Reduce: min of all lanes
    #[inline(always)]
    pub fn reduce_min(self) -> f64 {
        unsafe {
            let m = vpminq_f64(self.0, self.0);
            vgetq_lane_f64::<0>(m)
        }
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
        Self(unsafe { vreinterpretq_f64_u64(vceqq_f64(self.0, other.0)) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if not equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        self.simd_eq(other).not()
    }

    /// Lane-wise less-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if self < other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(unsafe { vreinterpretq_f64_u64(vcltq_f64(self.0, other.0)) })
    }

    /// Lane-wise less-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if self <= other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(unsafe { vreinterpretq_f64_u64(vcleq_f64(self.0, other.0)) })
    }

    /// Lane-wise greater-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if self > other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe { vreinterpretq_f64_u64(vcgtq_f64(self.0, other.0)) })
    }

    /// Lane-wise greater-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if self >= other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(unsafe { vreinterpretq_f64_u64(vcgeq_f64(self.0, other.0)) })
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
        Self(unsafe { vbslq_f64(vreinterpretq_u64_f64(mask.0), if_true.0, if_false.0) })
    }

    /// Bitwise NOT (complement)
    #[inline(always)]
    pub fn not(self) -> Self {
        // NEON lacks vmvnq_u64, use XOR with all-ones
        unsafe {
            let bits = vreinterpretq_u64_f64(self.0);
            let ones = vdupq_n_u64(u64::MAX);
            Self(vreinterpretq_f64_u64(veorq_u64(bits, ones)))
        }
    }

    // ========== Type Conversions ==========

    /// Convert to signed 32-bit integers (2 lanes), rounding toward zero.
    ///
    /// Returns an `i32x4` where only the lower 2 lanes are valid.
    #[inline(always)]
    pub fn to_i32x4_low(self) -> i32x4 {
        // NEON: f64->s64->s32 via vcvtq_s64_f64 + vmovn_s64
        let s64 = unsafe { vcvtq_s64_f64(self.0) };
        let s32_low = unsafe { vmovn_s64(s64) };
        i32x4(unsafe { vcombine_s32(s32_low, vdup_n_s32(0)) })
    }

    // ========== Bitcast ==========
    /// Reinterpret bits as `i64x2` (zero-cost).
    #[inline(always)]
    pub fn bitcast_i64x2(self) -> i64x2 {
        i64x2(unsafe { vreinterpretq_s64_f64(self.0) })
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
        u64x2(unsafe { vreinterpretq_u64_f64(self.0) })
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
    // ========== Transcendental Operations (Polynomial Approximations) ==========

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

    /// Low-precision base-10 logarithm.
    ///
    /// Handles edge cases correctly via std.
    #[inline(always)]
    pub fn log10_lowp(self) -> Self {
        let arr = self.to_array();
        let result = [arr[0].log10(), arr[1].log10()];
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
}

impl core::ops::Add for f64x2 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(unsafe { vaddq_f64(self.0, rhs.0) })
    }
}

impl core::ops::Sub for f64x2 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(unsafe { vsubq_f64(self.0, rhs.0) })
    }
}

impl core::ops::Mul for f64x2 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(unsafe { vmulq_f64(self.0, rhs.0) })
    }
}

impl core::ops::Div for f64x2 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        Self(unsafe { vdivq_f64(self.0, rhs.0) })
    }
}

impl core::ops::Neg for f64x2 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(unsafe { vnegq_f64(self.0) })
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
        // SAFETY: [f64; 2] and float64x2_t have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}

impl From<f64x2> for [f64; 2] {
    #[inline(always)]
    fn from(v: f64x2) -> Self {
        // SAFETY: float64x2_t and [f64; 2] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}

// ============================================================================
// i8x16 - 16 x i8 (128-bit NEON)
// ============================================================================

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct i8x16(int8x16_t);

impl i8x16 {
    pub const LANES: usize = 16;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::NeonToken, data: &[i8; 16]) -> Self {
        Self(unsafe { vld1q_s8(data.as_ptr()) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::NeonToken, v: i8) -> Self {
        Self(unsafe { vdupq_n_s8(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::NeonToken) -> Self {
        Self(unsafe { vdupq_n_s8(0i8) })
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::NeonToken, arr: [i8; 16]) -> Self {
        // SAFETY: [i8; 16] and int8x16_t have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [i8; 16]) {
        unsafe { vst1q_s8(out.as_mut_ptr(), self.0) };
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
    pub fn raw(self) -> int8x16_t {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports the required SIMD features.
    #[inline(always)]
    pub unsafe fn from_raw(v: int8x16_t) -> Self {
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
    pub fn cast_slice(_: archmage::NeonToken, slice: &[i8]) -> Option<&[Self]> {
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
    pub fn cast_slice_mut(_: archmage::NeonToken, slice: &mut [i8]) -> Option<&mut [Self]> {
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
        // SAFETY: Self is repr(transparent) over int8x16_t which is 16 bytes
        unsafe { &*(self as *const Self as *const [u8; 16]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 16] {
        // SAFETY: Self is repr(transparent) over int8x16_t which is 16 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 16]) }
    }

    /// Create from a byte array reference (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::NeonToken, bytes: &[u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Create from an owned byte array (token-gated, zero-cost).
    ///
    /// This is a zero-cost transmute from an owned byte array.
    #[inline(always)]
    pub fn from_bytes_owned(_: archmage::NeonToken, bytes: [u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(bytes) })
    }

    // ========== Implementation identification ==========

    /// Returns a string identifying this type's implementation.
    ///
    /// This is useful for verifying that the correct implementation is being used
    /// at compile time (via `-Ctarget-cpu`) or at runtime (via `#[magetypes]` dispatch).
    ///
    /// Returns `"arm::neon::i8x16"`.
    #[inline(always)]
    pub const fn implementation_name() -> &'static str {
        "arm::neon::i8x16"
    }

    /// Element-wise minimum
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(unsafe { vminq_s8(self.0, other.0) })
    }

    /// Element-wise maximum
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(unsafe { vmaxq_s8(self.0, other.0) })
    }

    /// Clamp values between lo and hi
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }

    /// Absolute value
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(unsafe { vabsq_s8(self.0) })
    }

    /// Reduce: sum all lanes
    #[inline(always)]
    pub fn reduce_add(self) -> i8 {
        unsafe {
            let sum = vpaddq_s8(self.0, self.0);
            let sum = vpaddq_s8(sum, sum);
            let sum = vpaddq_s8(sum, sum);
            let sum = vpaddq_s8(sum, sum);
            vgetq_lane_s8::<0>(sum)
        }
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
        Self(unsafe { vreinterpretq_s8_u8(vceqq_s8(self.0, other.0)) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if not equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        self.simd_eq(other).not()
    }

    /// Lane-wise less-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if self < other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(unsafe { vreinterpretq_s8_u8(vcltq_s8(self.0, other.0)) })
    }

    /// Lane-wise less-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if self <= other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(unsafe { vreinterpretq_s8_u8(vcleq_s8(self.0, other.0)) })
    }

    /// Lane-wise greater-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if self > other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe { vreinterpretq_s8_u8(vcgtq_s8(self.0, other.0)) })
    }

    /// Lane-wise greater-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if self >= other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(unsafe { vreinterpretq_s8_u8(vcgeq_s8(self.0, other.0)) })
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
        Self(unsafe { vbslq_s8(vreinterpretq_u8_s8(mask.0), if_true.0, if_false.0) })
    }

    /// Bitwise NOT (complement)
    #[inline(always)]
    pub fn not(self) -> Self {
        Self(unsafe { vmvnq_s8(self.0) })
    }

    /// Shift left by immediate (const generic)
    #[inline(always)]
    pub fn shl<const N: i32>(self) -> Self {
        Self(unsafe { vshlq_n_s8::<N>(self.0) })
    }

    /// Shift right by immediate (const generic)
    ///
    /// For signed types, this is an arithmetic shift (sign-extending).
    #[inline(always)]
    pub fn shr<const N: i32>(self) -> Self {
        Self(unsafe { vshrq_n_s8::<N>(self.0) })
    }

    /// Arithmetic shift right by `N` bits (sign-extending).
    ///
    /// The sign bit is replicated into the vacated positions.
    /// On ARM NEON, this is the same as `shr()` for signed types.
    #[inline(always)]
    pub fn shr_arithmetic<const N: i32>(self) -> Self {
        Self(unsafe { vshrq_n_s8::<N>(self.0) })
    }

    // ========== Boolean Reductions ==========

    /// Returns true if all lanes are non-zero (truthy).
    ///
    /// Typically used with comparison results where true lanes are all-1s.
    #[inline(always)]
    pub fn all_true(self) -> bool {
        unsafe { vminvq_u8(vreinterpretq_u8_s8(self.0)) != 0 }
    }

    /// Returns true if any lane is non-zero (truthy).
    #[inline(always)]
    pub fn any_true(self) -> bool {
        unsafe { vmaxvq_u8(vreinterpretq_u8_s8(self.0)) != 0 }
    }

    /// Extract the high bit of each lane as a bitmask.
    ///
    /// Returns a u32 where bit N corresponds to the sign bit of lane N.
    #[inline(always)]
    pub fn bitmask(self) -> u32 {
        unsafe {
            let signs = vshrq_n_u8::<7>(vreinterpretq_u8_s8(self.0));
            let arr: [u8; 16] = core::mem::transmute(signs);
            let mut r = 0u32;
            let mut i = 0;
            while i < 16 {
                r |= ((arr[i] & 1) as u32) << i;
                i += 1;
            }
            r
        }
    }

    // ========== Bitcast ==========
    /// Reinterpret bits as `u8x16` (zero-cost).
    #[inline(always)]
    pub fn bitcast_u8x16(self) -> u8x16 {
        u8x16(unsafe { vreinterpretq_u8_s8(self.0) })
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
        Self(unsafe { vaddq_s8(self.0, rhs.0) })
    }
}

impl core::ops::Sub for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(unsafe { vsubq_s8(self.0, rhs.0) })
    }
}

impl core::ops::Neg for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(unsafe { vnegq_s8(self.0) })
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
        // SAFETY: [i8; 16] and int8x16_t have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}

impl From<i8x16> for [i8; 16] {
    #[inline(always)]
    fn from(v: i8x16) -> Self {
        // SAFETY: int8x16_t and [i8; 16] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}

// ============================================================================
// u8x16 - 16 x u8 (128-bit NEON)
// ============================================================================

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct u8x16(uint8x16_t);

impl u8x16 {
    pub const LANES: usize = 16;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::NeonToken, data: &[u8; 16]) -> Self {
        Self(unsafe { vld1q_u8(data.as_ptr()) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::NeonToken, v: u8) -> Self {
        Self(unsafe { vdupq_n_u8(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::NeonToken) -> Self {
        Self(unsafe { vdupq_n_u8(0u8) })
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::NeonToken, arr: [u8; 16]) -> Self {
        // SAFETY: [u8; 16] and uint8x16_t have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [u8; 16]) {
        unsafe { vst1q_u8(out.as_mut_ptr(), self.0) };
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
    pub fn raw(self) -> uint8x16_t {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports the required SIMD features.
    #[inline(always)]
    pub unsafe fn from_raw(v: uint8x16_t) -> Self {
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
    pub fn cast_slice(_: archmage::NeonToken, slice: &[u8]) -> Option<&[Self]> {
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
    pub fn cast_slice_mut(_: archmage::NeonToken, slice: &mut [u8]) -> Option<&mut [Self]> {
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
        // SAFETY: Self is repr(transparent) over uint8x16_t which is 16 bytes
        unsafe { &*(self as *const Self as *const [u8; 16]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 16] {
        // SAFETY: Self is repr(transparent) over uint8x16_t which is 16 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 16]) }
    }

    /// Create from a byte array reference (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::NeonToken, bytes: &[u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Create from an owned byte array (token-gated, zero-cost).
    ///
    /// This is a zero-cost transmute from an owned byte array.
    #[inline(always)]
    pub fn from_bytes_owned(_: archmage::NeonToken, bytes: [u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(bytes) })
    }

    // ========== Implementation identification ==========

    /// Returns a string identifying this type's implementation.
    ///
    /// This is useful for verifying that the correct implementation is being used
    /// at compile time (via `-Ctarget-cpu`) or at runtime (via `#[magetypes]` dispatch).
    ///
    /// Returns `"arm::neon::u8x16"`.
    #[inline(always)]
    pub const fn implementation_name() -> &'static str {
        "arm::neon::u8x16"
    }

    /// Element-wise minimum
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(unsafe { vminq_u8(self.0, other.0) })
    }

    /// Element-wise maximum
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(unsafe { vmaxq_u8(self.0, other.0) })
    }

    /// Clamp values between lo and hi
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }

    /// Reduce: sum all lanes
    #[inline(always)]
    pub fn reduce_add(self) -> u8 {
        unsafe {
            let sum = vpaddq_u8(self.0, self.0);
            let sum = vpaddq_u8(sum, sum);
            let sum = vpaddq_u8(sum, sum);
            let sum = vpaddq_u8(sum, sum);
            vgetq_lane_u8::<0>(sum)
        }
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
        Self(unsafe { vceqq_u8(self.0, other.0) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if not equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        self.simd_eq(other).not()
    }

    /// Lane-wise less-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if self < other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(unsafe { vcltq_u8(self.0, other.0) })
    }

    /// Lane-wise less-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if self <= other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(unsafe { vcleq_u8(self.0, other.0) })
    }

    /// Lane-wise greater-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if self > other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe { vcgtq_u8(self.0, other.0) })
    }

    /// Lane-wise greater-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if self >= other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(unsafe { vcgeq_u8(self.0, other.0) })
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
        Self(unsafe { vbslq_u8(mask.0, if_true.0, if_false.0) })
    }

    /// Bitwise NOT (complement)
    #[inline(always)]
    pub fn not(self) -> Self {
        Self(unsafe { vmvnq_u8(self.0) })
    }

    /// Shift left by immediate (const generic)
    #[inline(always)]
    pub fn shl<const N: i32>(self) -> Self {
        Self(unsafe { vshlq_n_u8::<N>(self.0) })
    }

    /// Shift right by immediate (const generic)
    ///
    /// For unsigned types, this is a logical shift (zero-extending).
    #[inline(always)]
    pub fn shr<const N: i32>(self) -> Self {
        Self(unsafe { vshrq_n_u8::<N>(self.0) })
    }

    // ========== Boolean Reductions ==========

    /// Returns true if all lanes are non-zero (truthy).
    ///
    /// Typically used with comparison results where true lanes are all-1s.
    #[inline(always)]
    pub fn all_true(self) -> bool {
        unsafe { vminvq_u8(self.0) != 0 }
    }

    /// Returns true if any lane is non-zero (truthy).
    #[inline(always)]
    pub fn any_true(self) -> bool {
        unsafe { vmaxvq_u8(self.0) != 0 }
    }

    /// Extract the high bit of each lane as a bitmask.
    ///
    /// Returns a u32 where bit N corresponds to the sign bit of lane N.
    #[inline(always)]
    pub fn bitmask(self) -> u32 {
        unsafe {
            let signs = vshrq_n_u8::<7>(self.0);
            let arr: [u8; 16] = core::mem::transmute(signs);
            let mut r = 0u32;
            let mut i = 0;
            while i < 16 {
                r |= ((arr[i] & 1) as u32) << i;
                i += 1;
            }
            r
        }
    }

    // ========== Bitcast ==========
    /// Reinterpret bits as `i8x16` (zero-cost).
    #[inline(always)]
    pub fn bitcast_i8x16(self) -> i8x16 {
        i8x16(unsafe { vreinterpretq_s8_u8(self.0) })
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
    // ========== Extend/Widen Operations ==========

    /// Zero-extend low 8 u8 values to i16x8.
    ///
    /// Takes the lower 8 bytes and zero-extends each to 16 bits.
    #[inline(always)]
    pub fn extend_lo_i16(self) -> i16x8 {
        unsafe { i16x8(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(self.0)))) }
    }

    /// Zero-extend high 8 u8 values to i16x8.
    ///
    /// Takes the upper 8 bytes and zero-extends each to 16 bits.
    #[inline(always)]
    pub fn extend_hi_i16(self) -> i16x8 {
        unsafe { i16x8(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(self.0)))) }
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
        unsafe {
            let u16s = vmovl_u8(vget_low_u8(self.0));
            i32x4(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(u16s))))
        }
    }

    /// Zero-extend low 4 u8 values to f32x4.
    #[inline(always)]
    pub fn extend_lo_f32(self) -> f32x4 {
        unsafe {
            let u16s = vmovl_u8(vget_low_u8(self.0));
            let u32s = vmovl_u16(vget_low_u16(u16s));
            f32x4(vcvtq_f32_u32(u32s))
        }
    }
}

impl core::ops::Add for u8x16 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(unsafe { vaddq_u8(self.0, rhs.0) })
    }
}

impl core::ops::Sub for u8x16 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(unsafe { vsubq_u8(self.0, rhs.0) })
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
        // SAFETY: [u8; 16] and uint8x16_t have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}

impl From<u8x16> for [u8; 16] {
    #[inline(always)]
    fn from(v: u8x16) -> Self {
        // SAFETY: uint8x16_t and [u8; 16] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}

// ============================================================================
// i16x8 - 8 x i16 (128-bit NEON)
// ============================================================================

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct i16x8(int16x8_t);

impl i16x8 {
    pub const LANES: usize = 8;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::NeonToken, data: &[i16; 8]) -> Self {
        Self(unsafe { vld1q_s16(data.as_ptr()) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::NeonToken, v: i16) -> Self {
        Self(unsafe { vdupq_n_s16(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::NeonToken) -> Self {
        Self(unsafe { vdupq_n_s16(0i16) })
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::NeonToken, arr: [i16; 8]) -> Self {
        // SAFETY: [i16; 8] and int16x8_t have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [i16; 8]) {
        unsafe { vst1q_s16(out.as_mut_ptr(), self.0) };
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
    pub fn raw(self) -> int16x8_t {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports the required SIMD features.
    #[inline(always)]
    pub unsafe fn from_raw(v: int16x8_t) -> Self {
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
    pub fn cast_slice(_: archmage::NeonToken, slice: &[i16]) -> Option<&[Self]> {
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
    pub fn cast_slice_mut(_: archmage::NeonToken, slice: &mut [i16]) -> Option<&mut [Self]> {
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
        // SAFETY: Self is repr(transparent) over int16x8_t which is 16 bytes
        unsafe { &*(self as *const Self as *const [u8; 16]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 16] {
        // SAFETY: Self is repr(transparent) over int16x8_t which is 16 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 16]) }
    }

    /// Create from a byte array reference (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::NeonToken, bytes: &[u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Create from an owned byte array (token-gated, zero-cost).
    ///
    /// This is a zero-cost transmute from an owned byte array.
    #[inline(always)]
    pub fn from_bytes_owned(_: archmage::NeonToken, bytes: [u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(bytes) })
    }

    // ========== Implementation identification ==========

    /// Returns a string identifying this type's implementation.
    ///
    /// This is useful for verifying that the correct implementation is being used
    /// at compile time (via `-Ctarget-cpu`) or at runtime (via `#[magetypes]` dispatch).
    ///
    /// Returns `"arm::neon::i16x8"`.
    #[inline(always)]
    pub const fn implementation_name() -> &'static str {
        "arm::neon::i16x8"
    }

    /// Element-wise minimum
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(unsafe { vminq_s16(self.0, other.0) })
    }

    /// Element-wise maximum
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(unsafe { vmaxq_s16(self.0, other.0) })
    }

    /// Clamp values between lo and hi
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }

    /// Absolute value
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(unsafe { vabsq_s16(self.0) })
    }

    /// Reduce: sum all lanes
    #[inline(always)]
    pub fn reduce_add(self) -> i16 {
        unsafe {
            let sum = vpaddq_s16(self.0, self.0);
            let sum = vpaddq_s16(sum, sum);
            let sum = vpaddq_s16(sum, sum);
            vgetq_lane_s16::<0>(sum)
        }
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
        Self(unsafe { vreinterpretq_s16_u16(vceqq_s16(self.0, other.0)) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if not equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        self.simd_eq(other).not()
    }

    /// Lane-wise less-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if self < other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(unsafe { vreinterpretq_s16_u16(vcltq_s16(self.0, other.0)) })
    }

    /// Lane-wise less-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if self <= other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(unsafe { vreinterpretq_s16_u16(vcleq_s16(self.0, other.0)) })
    }

    /// Lane-wise greater-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if self > other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe { vreinterpretq_s16_u16(vcgtq_s16(self.0, other.0)) })
    }

    /// Lane-wise greater-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if self >= other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(unsafe { vreinterpretq_s16_u16(vcgeq_s16(self.0, other.0)) })
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
        Self(unsafe { vbslq_s16(vreinterpretq_u16_s16(mask.0), if_true.0, if_false.0) })
    }

    /// Bitwise NOT (complement)
    #[inline(always)]
    pub fn not(self) -> Self {
        Self(unsafe { vmvnq_s16(self.0) })
    }

    /// Shift left by immediate (const generic)
    #[inline(always)]
    pub fn shl<const N: i32>(self) -> Self {
        Self(unsafe { vshlq_n_s16::<N>(self.0) })
    }

    /// Shift right by immediate (const generic)
    ///
    /// For signed types, this is an arithmetic shift (sign-extending).
    #[inline(always)]
    pub fn shr<const N: i32>(self) -> Self {
        Self(unsafe { vshrq_n_s16::<N>(self.0) })
    }

    /// Arithmetic shift right by `N` bits (sign-extending).
    ///
    /// The sign bit is replicated into the vacated positions.
    /// On ARM NEON, this is the same as `shr()` for signed types.
    #[inline(always)]
    pub fn shr_arithmetic<const N: i32>(self) -> Self {
        Self(unsafe { vshrq_n_s16::<N>(self.0) })
    }

    // ========== Boolean Reductions ==========

    /// Returns true if all lanes are non-zero (truthy).
    ///
    /// Typically used with comparison results where true lanes are all-1s.
    #[inline(always)]
    pub fn all_true(self) -> bool {
        unsafe { vminvq_u16(vreinterpretq_u16_s16(self.0)) != 0 }
    }

    /// Returns true if any lane is non-zero (truthy).
    #[inline(always)]
    pub fn any_true(self) -> bool {
        unsafe { vmaxvq_u16(vreinterpretq_u16_s16(self.0)) != 0 }
    }

    /// Extract the high bit of each lane as a bitmask.
    ///
    /// Returns a u32 where bit N corresponds to the sign bit of lane N.
    #[inline(always)]
    pub fn bitmask(self) -> u32 {
        unsafe {
            let signs = vshrq_n_u16::<15>(vreinterpretq_u16_s16(self.0));
            let arr: [u16; 8] = core::mem::transmute(signs);
            let mut r = 0u32;
            let mut i = 0;
            while i < 8 {
                r |= ((arr[i] & 1) as u32) << i;
                i += 1;
            }
            r
        }
    }

    // ========== Bitcast ==========
    /// Reinterpret bits as `u16x8` (zero-cost).
    #[inline(always)]
    pub fn bitcast_u16x8(self) -> u16x8 {
        u16x8(unsafe { vreinterpretq_u16_s16(self.0) })
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
    // ========== Extend/Widen Operations ==========

    /// Sign-extend low 4 i16 values to i32x4.
    #[inline(always)]
    pub fn extend_lo_i32(self) -> i32x4 {
        unsafe { i32x4(vmovl_s16(vget_low_s16(self.0))) }
    }

    /// Sign-extend high 4 i16 values to i32x4.
    #[inline(always)]
    pub fn extend_hi_i32(self) -> i32x4 {
        unsafe { i32x4(vmovl_s16(vget_high_s16(self.0))) }
    }

    /// Sign-extend all 8 i16 values to two i32x4 vectors.
    ///
    /// Returns (low 4 as i32x4, high 4 as i32x4).
    #[inline(always)]
    pub fn extend_i32(self) -> (i32x4, i32x4) {
        (self.extend_lo_i32(), self.extend_hi_i32())
    }

    /// Sign-extend low 4 i16 values to f32x4.
    #[inline(always)]
    pub fn extend_lo_f32(self) -> f32x4 {
        unsafe {
            let i32s = vmovl_s16(vget_low_s16(self.0));
            f32x4(vcvtq_f32_s32(i32s))
        }
    }

    // ========== Pack/Narrow Operations ==========

    /// Pack two i16x8 to u8x16 with unsigned saturation.
    ///
    /// `self` provides the low 8 bytes, `other` provides the high 8 bytes.
    /// Values are clamped to [0, 255].
    #[inline(always)]
    pub fn pack_u8(self, other: Self) -> u8x16 {
        unsafe { u8x16(vcombine_u8(vqmovun_s16(self.0), vqmovun_s16(other.0))) }
    }

    /// Pack two i16x8 to i8x16 with signed saturation.
    ///
    /// `self` provides the low 8 bytes, `other` provides the high 8 bytes.
    /// Values are clamped to [-128, 127].
    #[inline(always)]
    pub fn pack_i8(self, other: Self) -> i8x16 {
        unsafe { i8x16(vcombine_s8(vqmovn_s16(self.0), vqmovn_s16(other.0))) }
    }
}

impl core::ops::Add for i16x8 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(unsafe { vaddq_s16(self.0, rhs.0) })
    }
}

impl core::ops::Sub for i16x8 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(unsafe { vsubq_s16(self.0, rhs.0) })
    }
}

impl core::ops::Mul for i16x8 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(unsafe { vmulq_s16(self.0, rhs.0) })
    }
}

impl core::ops::Neg for i16x8 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(unsafe { vnegq_s16(self.0) })
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
        // SAFETY: [i16; 8] and int16x8_t have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}

impl From<i16x8> for [i16; 8] {
    #[inline(always)]
    fn from(v: i16x8) -> Self {
        // SAFETY: int16x8_t and [i16; 8] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}

// ============================================================================
// u16x8 - 8 x u16 (128-bit NEON)
// ============================================================================

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct u16x8(uint16x8_t);

impl u16x8 {
    pub const LANES: usize = 8;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::NeonToken, data: &[u16; 8]) -> Self {
        Self(unsafe { vld1q_u16(data.as_ptr()) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::NeonToken, v: u16) -> Self {
        Self(unsafe { vdupq_n_u16(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::NeonToken) -> Self {
        Self(unsafe { vdupq_n_u16(0u16) })
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::NeonToken, arr: [u16; 8]) -> Self {
        // SAFETY: [u16; 8] and uint16x8_t have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [u16; 8]) {
        unsafe { vst1q_u16(out.as_mut_ptr(), self.0) };
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
    pub fn raw(self) -> uint16x8_t {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports the required SIMD features.
    #[inline(always)]
    pub unsafe fn from_raw(v: uint16x8_t) -> Self {
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
    pub fn cast_slice(_: archmage::NeonToken, slice: &[u16]) -> Option<&[Self]> {
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
    pub fn cast_slice_mut(_: archmage::NeonToken, slice: &mut [u16]) -> Option<&mut [Self]> {
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
        // SAFETY: Self is repr(transparent) over uint16x8_t which is 16 bytes
        unsafe { &*(self as *const Self as *const [u8; 16]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 16] {
        // SAFETY: Self is repr(transparent) over uint16x8_t which is 16 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 16]) }
    }

    /// Create from a byte array reference (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::NeonToken, bytes: &[u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Create from an owned byte array (token-gated, zero-cost).
    ///
    /// This is a zero-cost transmute from an owned byte array.
    #[inline(always)]
    pub fn from_bytes_owned(_: archmage::NeonToken, bytes: [u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(bytes) })
    }

    // ========== Implementation identification ==========

    /// Returns a string identifying this type's implementation.
    ///
    /// This is useful for verifying that the correct implementation is being used
    /// at compile time (via `-Ctarget-cpu`) or at runtime (via `#[magetypes]` dispatch).
    ///
    /// Returns `"arm::neon::u16x8"`.
    #[inline(always)]
    pub const fn implementation_name() -> &'static str {
        "arm::neon::u16x8"
    }

    /// Element-wise minimum
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(unsafe { vminq_u16(self.0, other.0) })
    }

    /// Element-wise maximum
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(unsafe { vmaxq_u16(self.0, other.0) })
    }

    /// Clamp values between lo and hi
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }

    /// Reduce: sum all lanes
    #[inline(always)]
    pub fn reduce_add(self) -> u16 {
        unsafe {
            let sum = vpaddq_u16(self.0, self.0);
            let sum = vpaddq_u16(sum, sum);
            let sum = vpaddq_u16(sum, sum);
            vgetq_lane_u16::<0>(sum)
        }
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
        Self(unsafe { vceqq_u16(self.0, other.0) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if not equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        self.simd_eq(other).not()
    }

    /// Lane-wise less-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if self < other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(unsafe { vcltq_u16(self.0, other.0) })
    }

    /// Lane-wise less-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if self <= other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(unsafe { vcleq_u16(self.0, other.0) })
    }

    /// Lane-wise greater-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if self > other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe { vcgtq_u16(self.0, other.0) })
    }

    /// Lane-wise greater-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if self >= other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(unsafe { vcgeq_u16(self.0, other.0) })
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
        Self(unsafe { vbslq_u16(mask.0, if_true.0, if_false.0) })
    }

    /// Bitwise NOT (complement)
    #[inline(always)]
    pub fn not(self) -> Self {
        Self(unsafe { vmvnq_u16(self.0) })
    }

    /// Shift left by immediate (const generic)
    #[inline(always)]
    pub fn shl<const N: i32>(self) -> Self {
        Self(unsafe { vshlq_n_u16::<N>(self.0) })
    }

    /// Shift right by immediate (const generic)
    ///
    /// For unsigned types, this is a logical shift (zero-extending).
    #[inline(always)]
    pub fn shr<const N: i32>(self) -> Self {
        Self(unsafe { vshrq_n_u16::<N>(self.0) })
    }

    // ========== Boolean Reductions ==========

    /// Returns true if all lanes are non-zero (truthy).
    ///
    /// Typically used with comparison results where true lanes are all-1s.
    #[inline(always)]
    pub fn all_true(self) -> bool {
        unsafe { vminvq_u16(self.0) != 0 }
    }

    /// Returns true if any lane is non-zero (truthy).
    #[inline(always)]
    pub fn any_true(self) -> bool {
        unsafe { vmaxvq_u16(self.0) != 0 }
    }

    /// Extract the high bit of each lane as a bitmask.
    ///
    /// Returns a u32 where bit N corresponds to the sign bit of lane N.
    #[inline(always)]
    pub fn bitmask(self) -> u32 {
        unsafe {
            let signs = vshrq_n_u16::<15>(self.0);
            let arr: [u16; 8] = core::mem::transmute(signs);
            let mut r = 0u32;
            let mut i = 0;
            while i < 8 {
                r |= ((arr[i] & 1) as u32) << i;
                i += 1;
            }
            r
        }
    }

    // ========== Bitcast ==========
    /// Reinterpret bits as `i16x8` (zero-cost).
    #[inline(always)]
    pub fn bitcast_i16x8(self) -> i16x8 {
        i16x8(unsafe { vreinterpretq_s16_u16(self.0) })
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
    // ========== Extend/Widen Operations ==========

    /// Zero-extend low 4 u16 values to i32x4.
    #[inline(always)]
    pub fn extend_lo_i32(self) -> i32x4 {
        unsafe { i32x4(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(self.0)))) }
    }

    /// Zero-extend high 4 u16 values to i32x4.
    #[inline(always)]
    pub fn extend_hi_i32(self) -> i32x4 {
        unsafe { i32x4(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(self.0)))) }
    }

    /// Zero-extend all 8 u16 values to two i32x4 vectors.
    ///
    /// Returns (low 4 as i32x4, high 4 as i32x4).
    #[inline(always)]
    pub fn extend_i32(self) -> (i32x4, i32x4) {
        (self.extend_lo_i32(), self.extend_hi_i32())
    }

    /// Zero-extend low 4 u16 values to f32x4.
    #[inline(always)]
    pub fn extend_lo_f32(self) -> f32x4 {
        unsafe {
            let u32s = vmovl_u16(vget_low_u16(self.0));
            f32x4(vcvtq_f32_u32(u32s))
        }
    }
}

impl core::ops::Add for u16x8 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(unsafe { vaddq_u16(self.0, rhs.0) })
    }
}

impl core::ops::Sub for u16x8 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(unsafe { vsubq_u16(self.0, rhs.0) })
    }
}

impl core::ops::Mul for u16x8 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(unsafe { vmulq_u16(self.0, rhs.0) })
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
        // SAFETY: [u16; 8] and uint16x8_t have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}

impl From<u16x8> for [u16; 8] {
    #[inline(always)]
    fn from(v: u16x8) -> Self {
        // SAFETY: uint16x8_t and [u16; 8] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}

// ============================================================================
// i32x4 - 4 x i32 (128-bit NEON)
// ============================================================================

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct i32x4(int32x4_t);

impl i32x4 {
    pub const LANES: usize = 4;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::NeonToken, data: &[i32; 4]) -> Self {
        Self(unsafe { vld1q_s32(data.as_ptr()) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::NeonToken, v: i32) -> Self {
        Self(unsafe { vdupq_n_s32(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::NeonToken) -> Self {
        Self(unsafe { vdupq_n_s32(0i32) })
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::NeonToken, arr: [i32; 4]) -> Self {
        // SAFETY: [i32; 4] and int32x4_t have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [i32; 4]) {
        unsafe { vst1q_s32(out.as_mut_ptr(), self.0) };
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
    pub fn raw(self) -> int32x4_t {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports the required SIMD features.
    #[inline(always)]
    pub unsafe fn from_raw(v: int32x4_t) -> Self {
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
    pub fn cast_slice(_: archmage::NeonToken, slice: &[i32]) -> Option<&[Self]> {
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
    pub fn cast_slice_mut(_: archmage::NeonToken, slice: &mut [i32]) -> Option<&mut [Self]> {
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
        // SAFETY: Self is repr(transparent) over int32x4_t which is 16 bytes
        unsafe { &*(self as *const Self as *const [u8; 16]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 16] {
        // SAFETY: Self is repr(transparent) over int32x4_t which is 16 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 16]) }
    }

    /// Create from a byte array reference (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::NeonToken, bytes: &[u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Create from an owned byte array (token-gated, zero-cost).
    ///
    /// This is a zero-cost transmute from an owned byte array.
    #[inline(always)]
    pub fn from_bytes_owned(_: archmage::NeonToken, bytes: [u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(bytes) })
    }

    // ========== Implementation identification ==========

    /// Returns a string identifying this type's implementation.
    ///
    /// This is useful for verifying that the correct implementation is being used
    /// at compile time (via `-Ctarget-cpu`) or at runtime (via `#[magetypes]` dispatch).
    ///
    /// Returns `"arm::neon::i32x4"`.
    #[inline(always)]
    pub const fn implementation_name() -> &'static str {
        "arm::neon::i32x4"
    }

    /// Element-wise minimum
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(unsafe { vminq_s32(self.0, other.0) })
    }

    /// Element-wise maximum
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(unsafe { vmaxq_s32(self.0, other.0) })
    }

    /// Clamp values between lo and hi
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }

    /// Absolute value
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(unsafe { vabsq_s32(self.0) })
    }

    /// Reduce: sum all lanes
    #[inline(always)]
    pub fn reduce_add(self) -> i32 {
        unsafe {
            let sum = vpaddq_s32(self.0, self.0);
            let sum = vpaddq_s32(sum, sum);
            vgetq_lane_s32::<0>(sum)
        }
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
        Self(unsafe { vreinterpretq_s32_u32(vceqq_s32(self.0, other.0)) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if not equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        self.simd_eq(other).not()
    }

    /// Lane-wise less-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if self < other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(unsafe { vreinterpretq_s32_u32(vcltq_s32(self.0, other.0)) })
    }

    /// Lane-wise less-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if self <= other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(unsafe { vreinterpretq_s32_u32(vcleq_s32(self.0, other.0)) })
    }

    /// Lane-wise greater-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if self > other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe { vreinterpretq_s32_u32(vcgtq_s32(self.0, other.0)) })
    }

    /// Lane-wise greater-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if self >= other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(unsafe { vreinterpretq_s32_u32(vcgeq_s32(self.0, other.0)) })
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
        Self(unsafe { vbslq_s32(vreinterpretq_u32_s32(mask.0), if_true.0, if_false.0) })
    }

    /// Bitwise NOT (complement)
    #[inline(always)]
    pub fn not(self) -> Self {
        Self(unsafe { vmvnq_s32(self.0) })
    }

    /// Shift left by immediate (const generic)
    #[inline(always)]
    pub fn shl<const N: i32>(self) -> Self {
        Self(unsafe { vshlq_n_s32::<N>(self.0) })
    }

    /// Shift right by immediate (const generic)
    ///
    /// For signed types, this is an arithmetic shift (sign-extending).
    #[inline(always)]
    pub fn shr<const N: i32>(self) -> Self {
        Self(unsafe { vshrq_n_s32::<N>(self.0) })
    }

    /// Arithmetic shift right by `N` bits (sign-extending).
    ///
    /// The sign bit is replicated into the vacated positions.
    /// On ARM NEON, this is the same as `shr()` for signed types.
    #[inline(always)]
    pub fn shr_arithmetic<const N: i32>(self) -> Self {
        Self(unsafe { vshrq_n_s32::<N>(self.0) })
    }

    // ========== Boolean Reductions ==========

    /// Returns true if all lanes are non-zero (truthy).
    ///
    /// Typically used with comparison results where true lanes are all-1s.
    #[inline(always)]
    pub fn all_true(self) -> bool {
        unsafe { vminvq_u32(vreinterpretq_u32_s32(self.0)) != 0 }
    }

    /// Returns true if any lane is non-zero (truthy).
    #[inline(always)]
    pub fn any_true(self) -> bool {
        unsafe { vmaxvq_u32(vreinterpretq_u32_s32(self.0)) != 0 }
    }

    /// Extract the high bit of each lane as a bitmask.
    ///
    /// Returns a u32 where bit N corresponds to the sign bit of lane N.
    #[inline(always)]
    pub fn bitmask(self) -> u32 {
        unsafe {
            let signs = vshrq_n_u32::<31>(vreinterpretq_u32_s32(self.0));
            let arr: [u32; 4] = core::mem::transmute(signs);
            (arr[0] & 1) | ((arr[1] & 1) << 1) | ((arr[2] & 1) << 2) | ((arr[3] & 1) << 3)
        }
    }

    // ========== Type Conversions ==========

    /// Convert to single-precision floats.
    #[inline(always)]
    pub fn to_f32x4(self) -> f32x4 {
        f32x4(unsafe { vcvtq_f32_s32(self.0) })
    }

    /// Convert to single-precision floats (alias for `to_f32x4`).
    #[inline(always)]
    pub fn to_f32(self) -> f32x4 {
        self.to_f32x4()
    }

    // ========== Bitcast ==========
    /// Reinterpret bits as `f32x4` (zero-cost).
    #[inline(always)]
    pub fn bitcast_f32x4(self) -> f32x4 {
        f32x4(unsafe { vreinterpretq_f32_s32(self.0) })
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
        u32x4(unsafe { vreinterpretq_u32_s32(self.0) })
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
    // ========== Pack/Narrow Operations ==========

    /// Pack two i32x4 to i16x8 with signed saturation.
    ///
    /// `self` provides the low 4 values, `other` provides the high 4 values.
    /// Values are clamped to [-32768, 32767].
    #[inline(always)]
    pub fn pack_i16(self, other: Self) -> i16x8 {
        unsafe { i16x8(vcombine_s16(vqmovn_s32(self.0), vqmovn_s32(other.0))) }
    }

    /// Pack two i32x4 to u16x8 with unsigned saturation.
    ///
    /// `self` provides the low 4 values, `other` provides the high 4 values.
    /// Values are clamped to [0, 65535].
    #[inline(always)]
    pub fn pack_u16(self, other: Self) -> u16x8 {
        unsafe { u16x8(vcombine_u16(vqmovun_s32(self.0), vqmovun_s32(other.0))) }
    }
}

impl core::ops::Add for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(unsafe { vaddq_s32(self.0, rhs.0) })
    }
}

impl core::ops::Sub for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(unsafe { vsubq_s32(self.0, rhs.0) })
    }
}

impl core::ops::Mul for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(unsafe { vmulq_s32(self.0, rhs.0) })
    }
}

impl core::ops::Neg for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(unsafe { vnegq_s32(self.0) })
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
        // SAFETY: [i32; 4] and int32x4_t have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}

impl From<i32x4> for [i32; 4] {
    #[inline(always)]
    fn from(v: i32x4) -> Self {
        // SAFETY: int32x4_t and [i32; 4] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}

// ============================================================================
// u32x4 - 4 x u32 (128-bit NEON)
// ============================================================================

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct u32x4(uint32x4_t);

impl u32x4 {
    pub const LANES: usize = 4;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::NeonToken, data: &[u32; 4]) -> Self {
        Self(unsafe { vld1q_u32(data.as_ptr()) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::NeonToken, v: u32) -> Self {
        Self(unsafe { vdupq_n_u32(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::NeonToken) -> Self {
        Self(unsafe { vdupq_n_u32(0u32) })
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::NeonToken, arr: [u32; 4]) -> Self {
        // SAFETY: [u32; 4] and uint32x4_t have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [u32; 4]) {
        unsafe { vst1q_u32(out.as_mut_ptr(), self.0) };
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
    pub fn raw(self) -> uint32x4_t {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports the required SIMD features.
    #[inline(always)]
    pub unsafe fn from_raw(v: uint32x4_t) -> Self {
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
    pub fn cast_slice(_: archmage::NeonToken, slice: &[u32]) -> Option<&[Self]> {
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
    pub fn cast_slice_mut(_: archmage::NeonToken, slice: &mut [u32]) -> Option<&mut [Self]> {
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
        // SAFETY: Self is repr(transparent) over uint32x4_t which is 16 bytes
        unsafe { &*(self as *const Self as *const [u8; 16]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 16] {
        // SAFETY: Self is repr(transparent) over uint32x4_t which is 16 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 16]) }
    }

    /// Create from a byte array reference (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::NeonToken, bytes: &[u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Create from an owned byte array (token-gated, zero-cost).
    ///
    /// This is a zero-cost transmute from an owned byte array.
    #[inline(always)]
    pub fn from_bytes_owned(_: archmage::NeonToken, bytes: [u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(bytes) })
    }

    // ========== Implementation identification ==========

    /// Returns a string identifying this type's implementation.
    ///
    /// This is useful for verifying that the correct implementation is being used
    /// at compile time (via `-Ctarget-cpu`) or at runtime (via `#[magetypes]` dispatch).
    ///
    /// Returns `"arm::neon::u32x4"`.
    #[inline(always)]
    pub const fn implementation_name() -> &'static str {
        "arm::neon::u32x4"
    }

    /// Element-wise minimum
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(unsafe { vminq_u32(self.0, other.0) })
    }

    /// Element-wise maximum
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(unsafe { vmaxq_u32(self.0, other.0) })
    }

    /// Clamp values between lo and hi
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }

    /// Reduce: sum all lanes
    #[inline(always)]
    pub fn reduce_add(self) -> u32 {
        unsafe {
            let sum = vpaddq_u32(self.0, self.0);
            let sum = vpaddq_u32(sum, sum);
            vgetq_lane_u32::<0>(sum)
        }
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
        Self(unsafe { vceqq_u32(self.0, other.0) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if not equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        self.simd_eq(other).not()
    }

    /// Lane-wise less-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if self < other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(unsafe { vcltq_u32(self.0, other.0) })
    }

    /// Lane-wise less-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if self <= other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(unsafe { vcleq_u32(self.0, other.0) })
    }

    /// Lane-wise greater-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if self > other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe { vcgtq_u32(self.0, other.0) })
    }

    /// Lane-wise greater-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if self >= other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(unsafe { vcgeq_u32(self.0, other.0) })
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
        Self(unsafe { vbslq_u32(mask.0, if_true.0, if_false.0) })
    }

    /// Bitwise NOT (complement)
    #[inline(always)]
    pub fn not(self) -> Self {
        Self(unsafe { vmvnq_u32(self.0) })
    }

    /// Shift left by immediate (const generic)
    #[inline(always)]
    pub fn shl<const N: i32>(self) -> Self {
        Self(unsafe { vshlq_n_u32::<N>(self.0) })
    }

    /// Shift right by immediate (const generic)
    ///
    /// For unsigned types, this is a logical shift (zero-extending).
    #[inline(always)]
    pub fn shr<const N: i32>(self) -> Self {
        Self(unsafe { vshrq_n_u32::<N>(self.0) })
    }

    // ========== Boolean Reductions ==========

    /// Returns true if all lanes are non-zero (truthy).
    ///
    /// Typically used with comparison results where true lanes are all-1s.
    #[inline(always)]
    pub fn all_true(self) -> bool {
        unsafe { vminvq_u32(self.0) != 0 }
    }

    /// Returns true if any lane is non-zero (truthy).
    #[inline(always)]
    pub fn any_true(self) -> bool {
        unsafe { vmaxvq_u32(self.0) != 0 }
    }

    /// Extract the high bit of each lane as a bitmask.
    ///
    /// Returns a u32 where bit N corresponds to the sign bit of lane N.
    #[inline(always)]
    pub fn bitmask(self) -> u32 {
        unsafe {
            let signs = vshrq_n_u32::<31>(self.0);
            let arr: [u32; 4] = core::mem::transmute(signs);
            (arr[0] & 1) | ((arr[1] & 1) << 1) | ((arr[2] & 1) << 2) | ((arr[3] & 1) << 3)
        }
    }

    // ========== Bitcast ==========
    /// Reinterpret bits as `f32x4` (zero-cost).
    #[inline(always)]
    pub fn bitcast_f32x4(self) -> f32x4 {
        f32x4(unsafe { vreinterpretq_f32_u32(self.0) })
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
        i32x4(unsafe { vreinterpretq_s32_u32(self.0) })
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
        Self(unsafe { vaddq_u32(self.0, rhs.0) })
    }
}

impl core::ops::Sub for u32x4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(unsafe { vsubq_u32(self.0, rhs.0) })
    }
}

impl core::ops::Mul for u32x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(unsafe { vmulq_u32(self.0, rhs.0) })
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
        // SAFETY: [u32; 4] and uint32x4_t have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}

impl From<u32x4> for [u32; 4] {
    #[inline(always)]
    fn from(v: u32x4) -> Self {
        // SAFETY: uint32x4_t and [u32; 4] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}

// ============================================================================
// i64x2 - 2 x i64 (128-bit NEON)
// ============================================================================

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct i64x2(int64x2_t);

impl i64x2 {
    pub const LANES: usize = 2;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::NeonToken, data: &[i64; 2]) -> Self {
        Self(unsafe { vld1q_s64(data.as_ptr()) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::NeonToken, v: i64) -> Self {
        Self(unsafe { vdupq_n_s64(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::NeonToken) -> Self {
        Self(unsafe { vdupq_n_s64(0i64) })
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::NeonToken, arr: [i64; 2]) -> Self {
        // SAFETY: [i64; 2] and int64x2_t have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [i64; 2]) {
        unsafe { vst1q_s64(out.as_mut_ptr(), self.0) };
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
    pub fn raw(self) -> int64x2_t {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports the required SIMD features.
    #[inline(always)]
    pub unsafe fn from_raw(v: int64x2_t) -> Self {
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
    pub fn cast_slice(_: archmage::NeonToken, slice: &[i64]) -> Option<&[Self]> {
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
    pub fn cast_slice_mut(_: archmage::NeonToken, slice: &mut [i64]) -> Option<&mut [Self]> {
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
        // SAFETY: Self is repr(transparent) over int64x2_t which is 16 bytes
        unsafe { &*(self as *const Self as *const [u8; 16]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 16] {
        // SAFETY: Self is repr(transparent) over int64x2_t which is 16 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 16]) }
    }

    /// Create from a byte array reference (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::NeonToken, bytes: &[u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Create from an owned byte array (token-gated, zero-cost).
    ///
    /// This is a zero-cost transmute from an owned byte array.
    #[inline(always)]
    pub fn from_bytes_owned(_: archmage::NeonToken, bytes: [u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(bytes) })
    }

    // ========== Implementation identification ==========

    /// Returns a string identifying this type's implementation.
    ///
    /// This is useful for verifying that the correct implementation is being used
    /// at compile time (via `-Ctarget-cpu`) or at runtime (via `#[magetypes]` dispatch).
    ///
    /// Returns `"arm::neon::i64x2"`.
    #[inline(always)]
    pub const fn implementation_name() -> &'static str {
        "arm::neon::i64x2"
    }

    /// Element-wise minimum
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        // NEON lacks native 64-bit min, use compare+select
        let mask = unsafe { vcltq_s64(self.0, other.0) };
        Self(unsafe { vbslq_s64(mask, self.0, other.0) })
    }

    /// Element-wise maximum
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        // NEON lacks native 64-bit max, use compare+select
        let mask = unsafe { vcgtq_s64(self.0, other.0) };
        Self(unsafe { vbslq_s64(mask, self.0, other.0) })
    }

    /// Clamp values between lo and hi
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }

    /// Absolute value
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(unsafe { vabsq_s64(self.0) })
    }

    /// Reduce: sum all lanes
    #[inline(always)]
    pub fn reduce_add(self) -> i64 {
        unsafe {
            let sum = vpaddq_s64(self.0, self.0);
            vgetq_lane_s64::<0>(sum)
        }
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
        Self(unsafe { vreinterpretq_s64_u64(vceqq_s64(self.0, other.0)) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if not equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        self.simd_eq(other).not()
    }

    /// Lane-wise less-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if self < other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(unsafe { vreinterpretq_s64_u64(vcltq_s64(self.0, other.0)) })
    }

    /// Lane-wise less-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if self <= other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(unsafe { vreinterpretq_s64_u64(vcleq_s64(self.0, other.0)) })
    }

    /// Lane-wise greater-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if self > other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe { vreinterpretq_s64_u64(vcgtq_s64(self.0, other.0)) })
    }

    /// Lane-wise greater-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if self >= other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(unsafe { vreinterpretq_s64_u64(vcgeq_s64(self.0, other.0)) })
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
        Self(unsafe { vbslq_s64(vreinterpretq_u64_s64(mask.0), if_true.0, if_false.0) })
    }

    /// Bitwise NOT (complement)
    #[inline(always)]
    pub fn not(self) -> Self {
        unsafe {
            let ones = vdupq_n_s64(-1i64);
            Self(veorq_s64(self.0, ones))
        }
    }

    /// Shift left by immediate (const generic)
    #[inline(always)]
    pub fn shl<const N: i32>(self) -> Self {
        Self(unsafe { vshlq_n_s64::<N>(self.0) })
    }

    /// Shift right by immediate (const generic)
    ///
    /// For signed types, this is an arithmetic shift (sign-extending).
    #[inline(always)]
    pub fn shr<const N: i32>(self) -> Self {
        Self(unsafe { vshrq_n_s64::<N>(self.0) })
    }

    // ========== Boolean Reductions ==========

    /// Returns true if all lanes are non-zero (truthy).
    ///
    /// Typically used with comparison results where true lanes are all-1s.
    #[inline(always)]
    pub fn all_true(self) -> bool {
        unsafe {
            let as_u64 = vreinterpretq_u64_s64(self.0);
            vgetq_lane_u64::<0>(as_u64) != 0 && vgetq_lane_u64::<1>(as_u64) != 0
        }
    }

    /// Returns true if any lane is non-zero (truthy).
    #[inline(always)]
    pub fn any_true(self) -> bool {
        unsafe {
            let as_u64 = vreinterpretq_u64_s64(self.0);
            (vgetq_lane_u64::<0>(as_u64) | vgetq_lane_u64::<1>(as_u64)) != 0
        }
    }

    /// Extract the high bit of each lane as a bitmask.
    ///
    /// Returns a u32 where bit N corresponds to the sign bit of lane N.
    #[inline(always)]
    pub fn bitmask(self) -> u32 {
        unsafe {
            let signs = vshrq_n_u64::<63>(vreinterpretq_u64_s64(self.0));
            ((vgetq_lane_u64::<0>(signs) & 1) | ((vgetq_lane_u64::<1>(signs) & 1) << 1)) as u32
        }
    }

    // ========== Bitcast ==========
    /// Reinterpret bits as `f64x2` (zero-cost).
    #[inline(always)]
    pub fn bitcast_f64x2(self) -> f64x2 {
        f64x2(unsafe { vreinterpretq_f64_s64(self.0) })
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
        u64x2(unsafe { vreinterpretq_u64_s64(self.0) })
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
        Self(unsafe { vaddq_s64(self.0, rhs.0) })
    }
}

impl core::ops::Sub for i64x2 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(unsafe { vsubq_s64(self.0, rhs.0) })
    }
}

impl core::ops::Neg for i64x2 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(unsafe { vnegq_s64(self.0) })
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
        // SAFETY: [i64; 2] and int64x2_t have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}

impl From<i64x2> for [i64; 2] {
    #[inline(always)]
    fn from(v: i64x2) -> Self {
        // SAFETY: int64x2_t and [i64; 2] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}

// ============================================================================
// u64x2 - 2 x u64 (128-bit NEON)
// ============================================================================

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct u64x2(uint64x2_t);

impl u64x2 {
    pub const LANES: usize = 2;

    /// Load from array (token-gated)
    #[inline(always)]
    pub fn load(_: archmage::NeonToken, data: &[u64; 2]) -> Self {
        Self(unsafe { vld1q_u64(data.as_ptr()) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: archmage::NeonToken, v: u64) -> Self {
        Self(unsafe { vdupq_n_u64(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: archmage::NeonToken) -> Self {
        Self(unsafe { vdupq_n_u64(0u64) })
    }

    /// Create from array (token-gated, zero-cost)
    ///
    /// This is a zero-cost transmute, not a memory load.
    #[inline(always)]
    pub fn from_array(_: archmage::NeonToken, arr: [u64; 2]) -> Self {
        // SAFETY: [u64; 2] and uint64x2_t have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }

    /// Store to array
    #[inline(always)]
    pub fn store(self, out: &mut [u64; 2]) {
        unsafe { vst1q_u64(out.as_mut_ptr(), self.0) };
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
    pub fn raw(self) -> uint64x2_t {
        self.0
    }

    /// Create from raw intrinsic (unsafe - no token check)
    ///
    /// # Safety
    /// Caller must ensure the CPU supports the required SIMD features.
    #[inline(always)]
    pub unsafe fn from_raw(v: uint64x2_t) -> Self {
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
    pub fn cast_slice(_: archmage::NeonToken, slice: &[u64]) -> Option<&[Self]> {
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
    pub fn cast_slice_mut(_: archmage::NeonToken, slice: &mut [u64]) -> Option<&mut [Self]> {
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
        // SAFETY: Self is repr(transparent) over uint64x2_t which is 16 bytes
        unsafe { &*(self as *const Self as *const [u8; 16]) }
    }

    /// View this vector as a mutable byte array.
    ///
    /// This is a safe replacement for `bytemuck::bytes_of_mut`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 16] {
        // SAFETY: Self is repr(transparent) over uint64x2_t which is 16 bytes
        unsafe { &mut *(self as *mut Self as *mut [u8; 16]) }
    }

    /// Create from a byte array reference (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::NeonToken, bytes: &[u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Create from an owned byte array (token-gated, zero-cost).
    ///
    /// This is a zero-cost transmute from an owned byte array.
    #[inline(always)]
    pub fn from_bytes_owned(_: archmage::NeonToken, bytes: [u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(bytes) })
    }

    // ========== Implementation identification ==========

    /// Returns a string identifying this type's implementation.
    ///
    /// This is useful for verifying that the correct implementation is being used
    /// at compile time (via `-Ctarget-cpu`) or at runtime (via `#[magetypes]` dispatch).
    ///
    /// Returns `"arm::neon::u64x2"`.
    #[inline(always)]
    pub const fn implementation_name() -> &'static str {
        "arm::neon::u64x2"
    }

    /// Element-wise minimum
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        // NEON lacks native 64-bit min, use compare+select
        let mask = unsafe { vcltq_u64(self.0, other.0) };
        Self(unsafe { vbslq_u64(mask, self.0, other.0) })
    }

    /// Element-wise maximum
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        // NEON lacks native 64-bit max, use compare+select
        let mask = unsafe { vcgtq_u64(self.0, other.0) };
        Self(unsafe { vbslq_u64(mask, self.0, other.0) })
    }

    /// Clamp values between lo and hi
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }

    /// Reduce: sum all lanes
    #[inline(always)]
    pub fn reduce_add(self) -> u64 {
        unsafe {
            let sum = vpaddq_u64(self.0, self.0);
            vgetq_lane_u64::<0>(sum)
        }
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
        Self(unsafe { vceqq_u64(self.0, other.0) })
    }

    /// Lane-wise inequality comparison.
    ///
    /// Returns a mask where each lane is all-1s if not equal, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        self.simd_eq(other).not()
    }

    /// Lane-wise less-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if self < other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(unsafe { vcltq_u64(self.0, other.0) })
    }

    /// Lane-wise less-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if self <= other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(unsafe { vcleq_u64(self.0, other.0) })
    }

    /// Lane-wise greater-than comparison.
    ///
    /// Returns a mask where each lane is all-1s if self > other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(unsafe { vcgtq_u64(self.0, other.0) })
    }

    /// Lane-wise greater-than-or-equal comparison.
    ///
    /// Returns a mask where each lane is all-1s if self >= other, all-0s otherwise.
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(unsafe { vcgeq_u64(self.0, other.0) })
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
        Self(unsafe { vbslq_u64(mask.0, if_true.0, if_false.0) })
    }

    /// Bitwise NOT (complement)
    #[inline(always)]
    pub fn not(self) -> Self {
        unsafe {
            let ones = vdupq_n_u64(u64::MAX);
            Self(veorq_u64(self.0, ones))
        }
    }

    /// Shift left by immediate (const generic)
    #[inline(always)]
    pub fn shl<const N: i32>(self) -> Self {
        Self(unsafe { vshlq_n_u64::<N>(self.0) })
    }

    /// Shift right by immediate (const generic)
    ///
    /// For unsigned types, this is a logical shift (zero-extending).
    #[inline(always)]
    pub fn shr<const N: i32>(self) -> Self {
        Self(unsafe { vshrq_n_u64::<N>(self.0) })
    }

    // ========== Boolean Reductions ==========

    /// Returns true if all lanes are non-zero (truthy).
    ///
    /// Typically used with comparison results where true lanes are all-1s.
    #[inline(always)]
    pub fn all_true(self) -> bool {
        unsafe { vgetq_lane_u64::<0>(self.0) != 0 && vgetq_lane_u64::<1>(self.0) != 0 }
    }

    /// Returns true if any lane is non-zero (truthy).
    #[inline(always)]
    pub fn any_true(self) -> bool {
        unsafe { (vgetq_lane_u64::<0>(self.0) | vgetq_lane_u64::<1>(self.0)) != 0 }
    }

    /// Extract the high bit of each lane as a bitmask.
    ///
    /// Returns a u32 where bit N corresponds to the sign bit of lane N.
    #[inline(always)]
    pub fn bitmask(self) -> u32 {
        unsafe {
            let signs = vshrq_n_u64::<63>(self.0);
            ((vgetq_lane_u64::<0>(signs) & 1) | ((vgetq_lane_u64::<1>(signs) & 1) << 1)) as u32
        }
    }

    // ========== Bitcast ==========
    /// Reinterpret bits as `f64x2` (zero-cost).
    #[inline(always)]
    pub fn bitcast_f64x2(self) -> f64x2 {
        f64x2(unsafe { vreinterpretq_f64_u64(self.0) })
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
        i64x2(unsafe { vreinterpretq_s64_u64(self.0) })
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
        Self(unsafe { vaddq_u64(self.0, rhs.0) })
    }
}

impl core::ops::Sub for u64x2 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(unsafe { vsubq_u64(self.0, rhs.0) })
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
        // SAFETY: [u64; 2] and uint64x2_t have identical size and layout
        Self(unsafe { core::mem::transmute(arr) })
    }
}

impl From<u64x2> for [u64; 2] {
    #[inline(always)]
    fn from(v: u64x2) -> Self {
        // SAFETY: uint64x2_t and [u64; 2] have identical size and layout
        unsafe { core::mem::transmute(v.0) }
    }
}

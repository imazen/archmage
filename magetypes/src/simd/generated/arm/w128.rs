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

    // ========== Bitcast (reinterpret bits, zero-cost) ==========

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

    // ========== Bitcast (reinterpret bits, zero-cost) ==========

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

    // ========== Bitcast (reinterpret bits, zero-cost) ==========

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

    // ========== Bitcast (reinterpret bits, zero-cost) ==========

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

    // ========== Bitcast (reinterpret bits, zero-cost) ==========

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

    // ========== Bitcast (reinterpret bits, zero-cost) ==========

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

    // ========== Bitcast (reinterpret bits, zero-cost) ==========

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

    // ========== Bitcast (reinterpret bits, zero-cost) ==========

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

    // ========== Bitcast (reinterpret bits, zero-cost) ==========

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

    // ========== Bitcast (reinterpret bits, zero-cost) ==========

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

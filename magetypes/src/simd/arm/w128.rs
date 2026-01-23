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

#[cfg(feature = "bytemuck")]
unsafe impl bytemuck::Zeroable for f32x4 {}
#[cfg(feature = "bytemuck")]
unsafe impl bytemuck::Pod for f32x4 {}

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

    /// Create from a byte array (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::NeonToken, bytes: &[u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
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

#[cfg(feature = "bytemuck")]
unsafe impl bytemuck::Zeroable for f64x2 {}
#[cfg(feature = "bytemuck")]
unsafe impl bytemuck::Pod for f64x2 {}

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

    /// Create from a byte array (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::NeonToken, bytes: &[u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
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

#[cfg(feature = "bytemuck")]
unsafe impl bytemuck::Zeroable for i8x16 {}
#[cfg(feature = "bytemuck")]
unsafe impl bytemuck::Pod for i8x16 {}

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

    /// Create from a byte array (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::NeonToken, bytes: &[u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
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

#[cfg(feature = "bytemuck")]
unsafe impl bytemuck::Zeroable for u8x16 {}
#[cfg(feature = "bytemuck")]
unsafe impl bytemuck::Pod for u8x16 {}

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

    /// Create from a byte array (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::NeonToken, bytes: &[u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
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

#[cfg(feature = "bytemuck")]
unsafe impl bytemuck::Zeroable for i16x8 {}
#[cfg(feature = "bytemuck")]
unsafe impl bytemuck::Pod for i16x8 {}

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

    /// Create from a byte array (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::NeonToken, bytes: &[u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
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

#[cfg(feature = "bytemuck")]
unsafe impl bytemuck::Zeroable for u16x8 {}
#[cfg(feature = "bytemuck")]
unsafe impl bytemuck::Pod for u16x8 {}

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

    /// Create from a byte array (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::NeonToken, bytes: &[u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
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

#[cfg(feature = "bytemuck")]
unsafe impl bytemuck::Zeroable for i32x4 {}
#[cfg(feature = "bytemuck")]
unsafe impl bytemuck::Pod for i32x4 {}

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

    /// Create from a byte array (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::NeonToken, bytes: &[u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
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

#[cfg(feature = "bytemuck")]
unsafe impl bytemuck::Zeroable for u32x4 {}
#[cfg(feature = "bytemuck")]
unsafe impl bytemuck::Pod for u32x4 {}

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

    /// Create from a byte array (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::NeonToken, bytes: &[u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
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

#[cfg(feature = "bytemuck")]
unsafe impl bytemuck::Zeroable for i64x2 {}
#[cfg(feature = "bytemuck")]
unsafe impl bytemuck::Pod for i64x2 {}

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

    /// Create from a byte array (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::NeonToken, bytes: &[u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Element-wise minimum
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(unsafe { vminq_s64(self.0, other.0) })
    }

    /// Element-wise maximum
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(unsafe { vmaxq_s64(self.0, other.0) })
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

#[cfg(feature = "bytemuck")]
unsafe impl bytemuck::Zeroable for u64x2 {}
#[cfg(feature = "bytemuck")]
unsafe impl bytemuck::Pod for u64x2 {}

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

    /// Create from a byte array (token-gated).
    ///
    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
    #[inline(always)]
    pub fn from_bytes(_: archmage::NeonToken, bytes: &[u8; 16]) -> Self {
        // SAFETY: [u8; 16] and Self have identical size
        Self(unsafe { core::mem::transmute(*bytes) })
    }

    /// Element-wise minimum
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(unsafe { vminq_u64(self.0, other.0) })
    }

    /// Element-wise maximum
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(unsafe { vmaxq_u64(self.0, other.0) })
    }

    /// Clamp values between lo and hi
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
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


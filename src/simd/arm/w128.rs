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
    pub fn load(_: crate::NeonToken, data: &[f32; 4]) -> Self {
        Self(unsafe { vld1q_f32(data.as_ptr()) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: crate::NeonToken, v: f32) -> Self {
        Self(unsafe { vdupq_n_f32(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: crate::NeonToken) -> Self {
        Self(unsafe { vdupq_n_f32(0.0f32) })
    }

    /// Create from array (token-gated)
    #[inline(always)]
    pub fn from_array(token: crate::NeonToken, arr: [f32; 4]) -> Self {
        Self::load(token, &arr)
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
    pub fn load(_: crate::NeonToken, data: &[f64; 2]) -> Self {
        Self(unsafe { vld1q_f64(data.as_ptr()) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: crate::NeonToken, v: f64) -> Self {
        Self(unsafe { vdupq_n_f64(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: crate::NeonToken) -> Self {
        Self(unsafe { vdupq_n_f64(0.0f64) })
    }

    /// Create from array (token-gated)
    #[inline(always)]
    pub fn from_array(token: crate::NeonToken, arr: [f64; 2]) -> Self {
        Self::load(token, &arr)
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
    pub fn load(_: crate::NeonToken, data: &[i8; 16]) -> Self {
        Self(unsafe { vld1q_s8(data.as_ptr()) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: crate::NeonToken, v: i8) -> Self {
        Self(unsafe { vdupq_n_s8(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: crate::NeonToken) -> Self {
        Self(unsafe { vdupq_n_s8(0i8) })
    }

    /// Create from array (token-gated)
    #[inline(always)]
    pub fn from_array(token: crate::NeonToken, arr: [i8; 16]) -> Self {
        Self::load(token, &arr)
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
    pub fn load(_: crate::NeonToken, data: &[u8; 16]) -> Self {
        Self(unsafe { vld1q_u8(data.as_ptr()) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: crate::NeonToken, v: u8) -> Self {
        Self(unsafe { vdupq_n_u8(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: crate::NeonToken) -> Self {
        Self(unsafe { vdupq_n_u8(0u8) })
    }

    /// Create from array (token-gated)
    #[inline(always)]
    pub fn from_array(token: crate::NeonToken, arr: [u8; 16]) -> Self {
        Self::load(token, &arr)
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
    pub fn load(_: crate::NeonToken, data: &[i16; 8]) -> Self {
        Self(unsafe { vld1q_s16(data.as_ptr()) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: crate::NeonToken, v: i16) -> Self {
        Self(unsafe { vdupq_n_s16(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: crate::NeonToken) -> Self {
        Self(unsafe { vdupq_n_s16(0i16) })
    }

    /// Create from array (token-gated)
    #[inline(always)]
    pub fn from_array(token: crate::NeonToken, arr: [i16; 8]) -> Self {
        Self::load(token, &arr)
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
    pub fn load(_: crate::NeonToken, data: &[u16; 8]) -> Self {
        Self(unsafe { vld1q_u16(data.as_ptr()) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: crate::NeonToken, v: u16) -> Self {
        Self(unsafe { vdupq_n_u16(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: crate::NeonToken) -> Self {
        Self(unsafe { vdupq_n_u16(0u16) })
    }

    /// Create from array (token-gated)
    #[inline(always)]
    pub fn from_array(token: crate::NeonToken, arr: [u16; 8]) -> Self {
        Self::load(token, &arr)
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
    pub fn load(_: crate::NeonToken, data: &[i32; 4]) -> Self {
        Self(unsafe { vld1q_s32(data.as_ptr()) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: crate::NeonToken, v: i32) -> Self {
        Self(unsafe { vdupq_n_s32(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: crate::NeonToken) -> Self {
        Self(unsafe { vdupq_n_s32(0i32) })
    }

    /// Create from array (token-gated)
    #[inline(always)]
    pub fn from_array(token: crate::NeonToken, arr: [i32; 4]) -> Self {
        Self::load(token, &arr)
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
    pub fn load(_: crate::NeonToken, data: &[u32; 4]) -> Self {
        Self(unsafe { vld1q_u32(data.as_ptr()) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: crate::NeonToken, v: u32) -> Self {
        Self(unsafe { vdupq_n_u32(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: crate::NeonToken) -> Self {
        Self(unsafe { vdupq_n_u32(0u32) })
    }

    /// Create from array (token-gated)
    #[inline(always)]
    pub fn from_array(token: crate::NeonToken, arr: [u32; 4]) -> Self {
        Self::load(token, &arr)
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
    pub fn load(_: crate::NeonToken, data: &[i64; 2]) -> Self {
        Self(unsafe { vld1q_s64(data.as_ptr()) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: crate::NeonToken, v: i64) -> Self {
        Self(unsafe { vdupq_n_s64(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: crate::NeonToken) -> Self {
        Self(unsafe { vdupq_n_s64(0i64) })
    }

    /// Create from array (token-gated)
    #[inline(always)]
    pub fn from_array(token: crate::NeonToken, arr: [i64; 2]) -> Self {
        Self::load(token, &arr)
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
    pub fn load(_: crate::NeonToken, data: &[u64; 2]) -> Self {
        Self(unsafe { vld1q_u64(data.as_ptr()) })
    }

    /// Broadcast scalar to all lanes (token-gated)
    #[inline(always)]
    pub fn splat(_: crate::NeonToken, v: u64) -> Self {
        Self(unsafe { vdupq_n_u64(v) })
    }

    /// Zero vector (token-gated)
    #[inline(always)]
    pub fn zero(_: crate::NeonToken) -> Self {
        Self(unsafe { vdupq_n_u64(0u64) })
    }

    /// Create from array (token-gated)
    #[inline(always)]
    pub fn from_array(token: crate::NeonToken, arr: [u64; 2]) -> Self {
        Self::load(token, &arr)
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


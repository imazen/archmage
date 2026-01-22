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

}

#[cfg(target_arch = "x86_64")]
impl_int_arithmetic_ops!(i8x16, _mm_add_epi8, _mm_sub_epi8);
#[cfg(target_arch = "x86_64")]
impl_assign_ops!(i8x16);
#[cfg(target_arch = "x86_64")]
impl_bitwise_ops!(i8x16, __m128i, _mm_and_si128, _mm_or_si128, _mm_xor_si128);
#[cfg(target_arch = "x86_64")]
impl_index!(i8x16, i8, 16);


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

}

#[cfg(target_arch = "x86_64")]
impl_int_arithmetic_ops!(u8x16, _mm_add_epi8, _mm_sub_epi8);
#[cfg(target_arch = "x86_64")]
impl_assign_ops!(u8x16);
#[cfg(target_arch = "x86_64")]
impl_bitwise_ops!(u8x16, __m128i, _mm_and_si128, _mm_or_si128, _mm_xor_si128);
#[cfg(target_arch = "x86_64")]
impl_index!(u8x16, u8, 16);


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

}

#[cfg(target_arch = "x86_64")]
impl_int_arithmetic_ops!(i64x2, _mm_add_epi64, _mm_sub_epi64);
#[cfg(target_arch = "x86_64")]
impl_assign_ops!(i64x2);
#[cfg(target_arch = "x86_64")]
impl_bitwise_ops!(i64x2, __m128i, _mm_and_si128, _mm_or_si128, _mm_xor_si128);
#[cfg(target_arch = "x86_64")]
impl_index!(i64x2, i64, 2);


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

}

#[cfg(target_arch = "x86_64")]
impl_int_arithmetic_ops!(u64x2, _mm_add_epi64, _mm_sub_epi64);
#[cfg(target_arch = "x86_64")]
impl_assign_ops!(u64x2);
#[cfg(target_arch = "x86_64")]
impl_bitwise_ops!(u64x2, __m128i, _mm_and_si128, _mm_or_si128, _mm_xor_si128);
#[cfg(target_arch = "x86_64")]
impl_index!(u64x2, u64, 2);


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

}

#[cfg(target_arch = "x86_64")]
impl_int_arithmetic_ops!(i8x32, _mm256_add_epi8, _mm256_sub_epi8);
#[cfg(target_arch = "x86_64")]
impl_assign_ops!(i8x32);
#[cfg(target_arch = "x86_64")]
impl_bitwise_ops!(i8x32, __m256i, _mm256_and_si256, _mm256_or_si256, _mm256_xor_si256);
#[cfg(target_arch = "x86_64")]
impl_index!(i8x32, i8, 32);


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

}

#[cfg(target_arch = "x86_64")]
impl_int_arithmetic_ops!(u8x32, _mm256_add_epi8, _mm256_sub_epi8);
#[cfg(target_arch = "x86_64")]
impl_assign_ops!(u8x32);
#[cfg(target_arch = "x86_64")]
impl_bitwise_ops!(u8x32, __m256i, _mm256_and_si256, _mm256_or_si256, _mm256_xor_si256);
#[cfg(target_arch = "x86_64")]
impl_index!(u8x32, u8, 32);


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

}

#[cfg(target_arch = "x86_64")]
impl_int_arithmetic_ops!(i64x4, _mm256_add_epi64, _mm256_sub_epi64);
#[cfg(target_arch = "x86_64")]
impl_assign_ops!(i64x4);
#[cfg(target_arch = "x86_64")]
impl_bitwise_ops!(i64x4, __m256i, _mm256_and_si256, _mm256_or_si256, _mm256_xor_si256);
#[cfg(target_arch = "x86_64")]
impl_index!(i64x4, i64, 4);


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

}

#[cfg(target_arch = "x86_64")]
impl_int_arithmetic_ops!(u64x4, _mm256_add_epi64, _mm256_sub_epi64);
#[cfg(target_arch = "x86_64")]
impl_assign_ops!(u64x4);
#[cfg(target_arch = "x86_64")]
impl_bitwise_ops!(u64x4, __m256i, _mm256_and_si256, _mm256_or_si256, _mm256_xor_si256);
#[cfg(target_arch = "x86_64")]
impl_index!(u64x4, u64, 4);


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

}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_int_arithmetic_ops!(i8x64, _mm512_add_epi8, _mm512_sub_epi8);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_assign_ops!(i8x64);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_bitwise_ops!(i8x64, __m512i, _mm512_and_si512, _mm512_or_si512, _mm512_xor_si512);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_index!(i8x64, i8, 64);


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

}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_int_arithmetic_ops!(u8x64, _mm512_add_epi8, _mm512_sub_epi8);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_assign_ops!(u8x64);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_bitwise_ops!(u8x64, __m512i, _mm512_and_si512, _mm512_or_si512, _mm512_xor_si512);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_index!(u8x64, u8, 64);


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

}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_int_arithmetic_ops!(i64x8, _mm512_add_epi64, _mm512_sub_epi64);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_assign_ops!(i64x8);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_bitwise_ops!(i64x8, __m512i, _mm512_and_si512, _mm512_or_si512, _mm512_xor_si512);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_index!(i64x8, i64, 8);


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

}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_int_arithmetic_ops!(u64x8, _mm512_add_epi64, _mm512_sub_epi64);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_assign_ops!(u64x8);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_bitwise_ops!(u64x8, __m512i, _mm512_and_si512, _mm512_or_si512, _mm512_xor_si512);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_index!(u64x8, u64, 8);


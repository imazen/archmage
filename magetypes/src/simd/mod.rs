//! Token-gated SIMD types with natural operators.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

#![allow(dead_code)]
#![allow(non_camel_case_types)]
#![allow(missing_docs)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::approx_constant)]
#![allow(clippy::missing_transmute_annotations)]


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

#[doc(hidden)]
#[macro_export]
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

#[doc(hidden)]
#[macro_export]
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

#[doc(hidden)]
#[macro_export]
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

#[doc(hidden)]
#[macro_export]
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

#[doc(hidden)]
#[macro_export]
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

#[doc(hidden)]
#[macro_export]
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

#[doc(hidden)]
#[macro_export]
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

#[doc(hidden)]
#[macro_export]
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
// Type modules
// ============================================================================

// x86-64 types (SSE, AVX, AVX-512)
#[cfg(target_arch = "x86_64")]
mod x86 {
    pub mod w128;
    pub mod w256;
    #[cfg(feature = "avx512")]
    pub mod w512;
}

// AArch64 types (NEON)
#[cfg(target_arch = "aarch64")]
mod arm {
    pub mod w128;
}

// Re-export all types
#[cfg(target_arch = "x86_64")]
pub use x86::w128::*;
#[cfg(target_arch = "x86_64")]
pub use x86::w256::*;
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub use x86::w512::*;

#[cfg(target_arch = "aarch64")]
pub use arm::w128::*;

// Polyfill module for emulating wider types on narrower hardware
pub mod polyfill;

// ============================================================================
// Width-aliased namespaces for multi-width dispatch
//
// Use these with the #[multiwidth] macro to write width-agnostic code:
//   use archmage::simd::avx2::*;  // f32xN = f32x8, Token = Avx2FmaToken
// ============================================================================

#[cfg(target_arch = "x86_64")]
pub mod sse {
    //! SSE/SSE4.1 width aliases (128-bit SIMD)
    //!
    //! - `f32xN` = `f32x4` (4 lanes)
    //! - `Token` = `Sse41Token`

    pub use super::x86::w128::{
        f32x4 as f32xN,
        f64x2 as f64xN,
        i8x16 as i8xN,
        u8x16 as u8xN,
        i16x8 as i16xN,
        u16x8 as u16xN,
        i32x4 as i32xN,
        u32x4 as u32xN,
        i64x2 as i64xN,
        u64x2 as u64xN,
    };

    pub use super::x86::w128::*;

    /// Token type for this width level
    pub type Token = archmage::Sse41Token;

    /// Number of f32 lanes
    pub const LANES_F32: usize = 4;
    /// Number of f64 lanes
    pub const LANES_F64: usize = 2;
    /// Number of i32/u32 lanes
    pub const LANES_32: usize = 4;
    /// Number of i16/u16 lanes
    pub const LANES_16: usize = 8;
    /// Number of i8/u8 lanes
    pub const LANES_8: usize = 16;
}

#[cfg(target_arch = "x86_64")]
pub mod avx2 {
    //! AVX2+FMA width aliases (256-bit SIMD)
    //!
    //! - `f32xN` = `f32x8` (8 lanes)
    //! - `Token` = `Avx2FmaToken`

    pub use super::x86::w256::{
        f32x8 as f32xN,
        f64x4 as f64xN,
        i8x32 as i8xN,
        u8x32 as u8xN,
        i16x16 as i16xN,
        u16x16 as u16xN,
        i32x8 as i32xN,
        u32x8 as u32xN,
        i64x4 as i64xN,
        u64x4 as u64xN,
    };

    pub use super::x86::w256::*;

    /// Token type for this width level
    pub type Token = archmage::Avx2FmaToken;

    pub const LANES_F32: usize = 8;
    pub const LANES_F64: usize = 4;
    pub const LANES_32: usize = 8;
    pub const LANES_16: usize = 16;
    pub const LANES_8: usize = 32;
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub mod avx512 {
    //! AVX-512 width aliases (512-bit SIMD)
    //!
    //! - `f32xN` = `f32x16` (16 lanes)
    //! - `Token` = `X64V4Token`

    pub use super::x86::w512::{
        f32x16 as f32xN,
        f64x8 as f64xN,
        i8x64 as i8xN,
        u8x64 as u8xN,
        i16x32 as i16xN,
        u16x32 as u16xN,
        i32x16 as i32xN,
        u32x16 as u32xN,
        i64x8 as i64xN,
        u64x8 as u64xN,
    };

    pub use super::x86::w512::*;

    /// Token type for this width level
    pub type Token = archmage::X64V4Token;

    pub const LANES_F32: usize = 16;
    pub const LANES_F64: usize = 8;
    pub const LANES_32: usize = 16;
    pub const LANES_16: usize = 32;
    pub const LANES_8: usize = 64;
}

#[cfg(target_arch = "aarch64")]
pub mod neon {
    //! NEON width aliases (128-bit SIMD)
    //!
    //! - `f32xN` = `f32x4` (4 lanes)
    //! - `Token` = `NeonToken`

    pub use super::arm::w128::{
        f32x4 as f32xN, f64x2 as f64xN, i8x16 as i8xN, i16x8 as i16xN,
        i32x4 as i32xN, i64x2 as i64xN, u8x16 as u8xN, u16x8 as u16xN,
        u32x4 as u32xN, u64x2 as u64xN,
    };

    pub use super::arm::w128::*;

    /// Token type for this width level
    pub type Token = archmage::NeonToken;

    /// Number of f32 lanes
    pub const LANES_F32: usize = 4;
    /// Number of f64 lanes
    pub const LANES_F64: usize = 2;
    /// Number of i32/u32 lanes
    pub const LANES_32: usize = 4;
    /// Number of i16/u16 lanes
    pub const LANES_16: usize = 8;
    /// Number of i8/u8 lanes
    pub const LANES_8: usize = 16;
}

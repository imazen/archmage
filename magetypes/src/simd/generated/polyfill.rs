//! Polyfill module for emulating wider SIMD types on narrower hardware.
//!
//! This module provides types like `f32x8` that work on SSE/NEON/WASM hardware by
//! internally using two W128 operations. This allows writing code
//! targeting AVX2 widths while still running (slower) on SSE-only systems.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

#[cfg(target_arch = "x86_64")]
pub mod sse {
    //! Polyfilled 256-bit types using SSE (128-bit) operations.

    use crate::simd::generated::x86::w128::{
        f32x4, f64x2, i8x16, i16x8, i32x4, i64x2, u8x16, u16x8, u32x4, u64x2,
    };
    use archmage::X64V3Token;
    use core::ops::{Add, Div, Mul, Neg, Sub};

    /// Emulated 8-wide f32 vector using two f32x4 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct f32x8 {
        pub(crate) lo: f32x4,
        pub(crate) hi: f32x4,
    }

    impl f32x8 {
        pub const LANES: usize = 8;

        /// Load from array (token-gated)
        #[inline(always)]
        pub fn load(token: X64V3Token, data: &[f32; 8]) -> Self {
            let lo_arr: &[f32; 4] = data[0..4].try_into().unwrap();
            let hi_arr: &[f32; 4] = data[4..8].try_into().unwrap();
            Self {
                lo: f32x4::load(token, lo_arr),
                hi: f32x4::load(token, hi_arr),
            }
        }

        /// Broadcast scalar to all lanes (token-gated)
        #[inline(always)]
        pub fn splat(token: X64V3Token, v: f32) -> Self {
            Self {
                lo: f32x4::splat(token, v),
                hi: f32x4::splat(token, v),
            }
        }

        /// Zero vector (token-gated)
        #[inline(always)]
        pub fn zero(token: X64V3Token) -> Self {
            Self {
                lo: f32x4::zero(token),
                hi: f32x4::zero(token),
            }
        }

        /// Create from array (token-gated)
        #[inline(always)]
        pub fn from_array(token: X64V3Token, arr: [f32; 8]) -> Self {
            Self::load(token, &arr)
        }

        /// Store to array
        #[inline(always)]
        pub fn store(self, out: &mut [f32; 8]) {
            let (lo, hi) = out.split_at_mut(4);
            let lo_arr: &mut [f32; 4] = lo.try_into().unwrap();
            let hi_arr: &mut [f32; 4] = hi.try_into().unwrap();
            self.lo.store(lo_arr);
            self.hi.store(hi_arr);
        }

        /// Convert to array
        #[inline(always)]
        pub fn to_array(self) -> [f32; 8] {
            let mut out = [0.0f32; 8];
            self.store(&mut out);
            out
        }

        /// Element-wise minimum
        #[inline(always)]
        pub fn min(self, other: Self) -> Self {
            Self {
                lo: self.lo.min(other.lo),
                hi: self.hi.min(other.hi),
            }
        }

        /// Element-wise maximum
        #[inline(always)]
        pub fn max(self, other: Self) -> Self {
            Self {
                lo: self.lo.max(other.lo),
                hi: self.hi.max(other.hi),
            }
        }

        /// Clamp values between lo and hi
        #[inline(always)]
        pub fn clamp(self, lo: Self, hi: Self) -> Self {
            self.max(lo).min(hi)
        }

        /// Square root
        #[inline(always)]
        pub fn sqrt(self) -> Self {
            Self {
                lo: self.lo.sqrt(),
                hi: self.hi.sqrt(),
            }
        }

        /// Floor
        #[inline(always)]
        pub fn floor(self) -> Self {
            Self {
                lo: self.lo.floor(),
                hi: self.hi.floor(),
            }
        }

        /// Ceil
        #[inline(always)]
        pub fn ceil(self) -> Self {
            Self {
                lo: self.lo.ceil(),
                hi: self.hi.ceil(),
            }
        }

        /// Round to nearest
        #[inline(always)]
        pub fn round(self) -> Self {
            Self {
                lo: self.lo.round(),
                hi: self.hi.round(),
            }
        }

        /// Fused multiply-add: self * a + b
        #[inline(always)]
        pub fn mul_add(self, a: Self, b: Self) -> Self {
            Self {
                lo: self.lo.mul_add(a.lo, b.lo),
                hi: self.hi.mul_add(a.hi, b.hi),
            }
        }

        /// Absolute value
        #[inline(always)]
        pub fn abs(self) -> Self {
            Self {
                lo: self.lo.abs(),
                hi: self.hi.abs(),
            }
        }

        /// Reduce: sum all lanes
        #[inline(always)]
        pub fn reduce_add(self) -> f32 {
            self.lo.reduce_add() + self.hi.reduce_add()
        }

        /// Reduce: max of all lanes
        #[inline(always)]
        pub fn reduce_max(self) -> f32 {
            self.lo.reduce_max().max(self.hi.reduce_max())
        }

        /// Reduce: min of all lanes
        #[inline(always)]
        pub fn reduce_min(self) -> f32 {
            self.lo.reduce_min().min(self.hi.reduce_min())
        }

        // ========== Bitcast ==========
        /// Reinterpret bits as `i32x8` (zero-cost).
        #[inline(always)]
        pub fn bitcast_i32x8(self) -> i32x8 {
            i32x8 {
                lo: self.lo.bitcast_i32x4(),
                hi: self.hi.bitcast_i32x4(),
            }
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
            u32x8 {
                lo: self.lo.bitcast_u32x4(),
                hi: self.hi.bitcast_u32x4(),
            }
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

    impl Add for f32x8 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                lo: self.lo + rhs.lo,
                hi: self.hi + rhs.hi,
            }
        }
    }

    impl Sub for f32x8 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                lo: self.lo - rhs.lo,
                hi: self.hi - rhs.hi,
            }
        }
    }

    impl Mul for f32x8 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            Self {
                lo: self.lo * rhs.lo,
                hi: self.hi * rhs.hi,
            }
        }
    }

    impl Div for f32x8 {
        type Output = Self;
        #[inline(always)]
        fn div(self, rhs: Self) -> Self {
            Self {
                lo: self.lo / rhs.lo,
                hi: self.hi / rhs.hi,
            }
        }
    }

    impl Neg for f32x8 {
        type Output = Self;
        #[inline(always)]
        fn neg(self) -> Self {
            Self {
                lo: -self.lo,
                hi: -self.hi,
            }
        }
    }

    impl core::ops::AddAssign for f32x8 {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl core::ops::SubAssign for f32x8 {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    impl core::ops::MulAssign for f32x8 {
        #[inline(always)]
        fn mul_assign(&mut self, rhs: Self) {
            *self = *self * rhs;
        }
    }

    impl core::ops::DivAssign for f32x8 {
        #[inline(always)]
        fn div_assign(&mut self, rhs: Self) {
            *self = *self / rhs;
        }
    }

    /// Emulated 4-wide f64 vector using two f64x2 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct f64x4 {
        pub(crate) lo: f64x2,
        pub(crate) hi: f64x2,
    }

    impl f64x4 {
        pub const LANES: usize = 4;

        /// Load from array (token-gated)
        #[inline(always)]
        pub fn load(token: X64V3Token, data: &[f64; 4]) -> Self {
            let lo_arr: &[f64; 2] = data[0..2].try_into().unwrap();
            let hi_arr: &[f64; 2] = data[2..4].try_into().unwrap();
            Self {
                lo: f64x2::load(token, lo_arr),
                hi: f64x2::load(token, hi_arr),
            }
        }

        /// Broadcast scalar to all lanes (token-gated)
        #[inline(always)]
        pub fn splat(token: X64V3Token, v: f64) -> Self {
            Self {
                lo: f64x2::splat(token, v),
                hi: f64x2::splat(token, v),
            }
        }

        /// Zero vector (token-gated)
        #[inline(always)]
        pub fn zero(token: X64V3Token) -> Self {
            Self {
                lo: f64x2::zero(token),
                hi: f64x2::zero(token),
            }
        }

        /// Create from array (token-gated)
        #[inline(always)]
        pub fn from_array(token: X64V3Token, arr: [f64; 4]) -> Self {
            Self::load(token, &arr)
        }

        /// Store to array
        #[inline(always)]
        pub fn store(self, out: &mut [f64; 4]) {
            let (lo, hi) = out.split_at_mut(2);
            let lo_arr: &mut [f64; 2] = lo.try_into().unwrap();
            let hi_arr: &mut [f64; 2] = hi.try_into().unwrap();
            self.lo.store(lo_arr);
            self.hi.store(hi_arr);
        }

        /// Convert to array
        #[inline(always)]
        pub fn to_array(self) -> [f64; 4] {
            let mut out = [0.0f64; 4];
            self.store(&mut out);
            out
        }

        /// Element-wise minimum
        #[inline(always)]
        pub fn min(self, other: Self) -> Self {
            Self {
                lo: self.lo.min(other.lo),
                hi: self.hi.min(other.hi),
            }
        }

        /// Element-wise maximum
        #[inline(always)]
        pub fn max(self, other: Self) -> Self {
            Self {
                lo: self.lo.max(other.lo),
                hi: self.hi.max(other.hi),
            }
        }

        /// Clamp values between lo and hi
        #[inline(always)]
        pub fn clamp(self, lo: Self, hi: Self) -> Self {
            self.max(lo).min(hi)
        }

        /// Square root
        #[inline(always)]
        pub fn sqrt(self) -> Self {
            Self {
                lo: self.lo.sqrt(),
                hi: self.hi.sqrt(),
            }
        }

        /// Floor
        #[inline(always)]
        pub fn floor(self) -> Self {
            Self {
                lo: self.lo.floor(),
                hi: self.hi.floor(),
            }
        }

        /// Ceil
        #[inline(always)]
        pub fn ceil(self) -> Self {
            Self {
                lo: self.lo.ceil(),
                hi: self.hi.ceil(),
            }
        }

        /// Round to nearest
        #[inline(always)]
        pub fn round(self) -> Self {
            Self {
                lo: self.lo.round(),
                hi: self.hi.round(),
            }
        }

        /// Fused multiply-add: self * a + b
        #[inline(always)]
        pub fn mul_add(self, a: Self, b: Self) -> Self {
            Self {
                lo: self.lo.mul_add(a.lo, b.lo),
                hi: self.hi.mul_add(a.hi, b.hi),
            }
        }

        /// Absolute value
        #[inline(always)]
        pub fn abs(self) -> Self {
            Self {
                lo: self.lo.abs(),
                hi: self.hi.abs(),
            }
        }

        /// Reduce: sum all lanes
        #[inline(always)]
        pub fn reduce_add(self) -> f64 {
            self.lo.reduce_add() + self.hi.reduce_add()
        }

        /// Reduce: max of all lanes
        #[inline(always)]
        pub fn reduce_max(self) -> f64 {
            self.lo.reduce_max().max(self.hi.reduce_max())
        }

        /// Reduce: min of all lanes
        #[inline(always)]
        pub fn reduce_min(self) -> f64 {
            self.lo.reduce_min().min(self.hi.reduce_min())
        }

        // ========== Bitcast ==========
        /// Reinterpret bits as `i64x4` (zero-cost).
        #[inline(always)]
        pub fn bitcast_i64x4(self) -> i64x4 {
            i64x4 {
                lo: self.lo.bitcast_i64x2(),
                hi: self.hi.bitcast_i64x2(),
            }
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
            u64x4 {
                lo: self.lo.bitcast_u64x2(),
                hi: self.hi.bitcast_u64x2(),
            }
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

    impl Add for f64x4 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                lo: self.lo + rhs.lo,
                hi: self.hi + rhs.hi,
            }
        }
    }

    impl Sub for f64x4 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                lo: self.lo - rhs.lo,
                hi: self.hi - rhs.hi,
            }
        }
    }

    impl Mul for f64x4 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            Self {
                lo: self.lo * rhs.lo,
                hi: self.hi * rhs.hi,
            }
        }
    }

    impl Div for f64x4 {
        type Output = Self;
        #[inline(always)]
        fn div(self, rhs: Self) -> Self {
            Self {
                lo: self.lo / rhs.lo,
                hi: self.hi / rhs.hi,
            }
        }
    }

    impl Neg for f64x4 {
        type Output = Self;
        #[inline(always)]
        fn neg(self) -> Self {
            Self {
                lo: -self.lo,
                hi: -self.hi,
            }
        }
    }

    impl core::ops::AddAssign for f64x4 {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl core::ops::SubAssign for f64x4 {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    impl core::ops::MulAssign for f64x4 {
        #[inline(always)]
        fn mul_assign(&mut self, rhs: Self) {
            *self = *self * rhs;
        }
    }

    impl core::ops::DivAssign for f64x4 {
        #[inline(always)]
        fn div_assign(&mut self, rhs: Self) {
            *self = *self / rhs;
        }
    }

    /// Emulated 32-wide i8 vector using two i8x16 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct i8x32 {
        pub(crate) lo: i8x16,
        pub(crate) hi: i8x16,
    }

    impl i8x32 {
        pub const LANES: usize = 32;

        /// Load from array (token-gated)
        #[inline(always)]
        pub fn load(token: X64V3Token, data: &[i8; 32]) -> Self {
            let lo_arr: &[i8; 16] = data[0..16].try_into().unwrap();
            let hi_arr: &[i8; 16] = data[16..32].try_into().unwrap();
            Self {
                lo: i8x16::load(token, lo_arr),
                hi: i8x16::load(token, hi_arr),
            }
        }

        /// Broadcast scalar to all lanes (token-gated)
        #[inline(always)]
        pub fn splat(token: X64V3Token, v: i8) -> Self {
            Self {
                lo: i8x16::splat(token, v),
                hi: i8x16::splat(token, v),
            }
        }

        /// Zero vector (token-gated)
        #[inline(always)]
        pub fn zero(token: X64V3Token) -> Self {
            Self {
                lo: i8x16::zero(token),
                hi: i8x16::zero(token),
            }
        }

        /// Create from array (token-gated)
        #[inline(always)]
        pub fn from_array(token: X64V3Token, arr: [i8; 32]) -> Self {
            Self::load(token, &arr)
        }

        /// Store to array
        #[inline(always)]
        pub fn store(self, out: &mut [i8; 32]) {
            let (lo, hi) = out.split_at_mut(16);
            let lo_arr: &mut [i8; 16] = lo.try_into().unwrap();
            let hi_arr: &mut [i8; 16] = hi.try_into().unwrap();
            self.lo.store(lo_arr);
            self.hi.store(hi_arr);
        }

        /// Convert to array
        #[inline(always)]
        pub fn to_array(self) -> [i8; 32] {
            let mut out = [0i8; 32];
            self.store(&mut out);
            out
        }

        /// Element-wise minimum
        #[inline(always)]
        pub fn min(self, other: Self) -> Self {
            Self {
                lo: self.lo.min(other.lo),
                hi: self.hi.min(other.hi),
            }
        }

        /// Element-wise maximum
        #[inline(always)]
        pub fn max(self, other: Self) -> Self {
            Self {
                lo: self.lo.max(other.lo),
                hi: self.hi.max(other.hi),
            }
        }

        /// Clamp values between lo and hi
        #[inline(always)]
        pub fn clamp(self, lo: Self, hi: Self) -> Self {
            self.max(lo).min(hi)
        }

        /// Absolute value
        #[inline(always)]
        pub fn abs(self) -> Self {
            Self {
                lo: self.lo.abs(),
                hi: self.hi.abs(),
            }
        }

        /// Reduce: sum all lanes
        #[inline(always)]
        pub fn reduce_add(self) -> i8 {
            self.lo.reduce_add().wrapping_add(self.hi.reduce_add())
        }

        // ========== Bitcast ==========
        /// Reinterpret bits as `u8x32` (zero-cost).
        #[inline(always)]
        pub fn bitcast_u8x32(self) -> u8x32 {
            u8x32 {
                lo: self.lo.bitcast_u8x16(),
                hi: self.hi.bitcast_u8x16(),
            }
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

    impl Add for i8x32 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                lo: self.lo + rhs.lo,
                hi: self.hi + rhs.hi,
            }
        }
    }

    impl Sub for i8x32 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                lo: self.lo - rhs.lo,
                hi: self.hi - rhs.hi,
            }
        }
    }

    impl core::ops::AddAssign for i8x32 {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl core::ops::SubAssign for i8x32 {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    /// Emulated 32-wide u8 vector using two u8x16 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct u8x32 {
        pub(crate) lo: u8x16,
        pub(crate) hi: u8x16,
    }

    impl u8x32 {
        pub const LANES: usize = 32;

        /// Load from array (token-gated)
        #[inline(always)]
        pub fn load(token: X64V3Token, data: &[u8; 32]) -> Self {
            let lo_arr: &[u8; 16] = data[0..16].try_into().unwrap();
            let hi_arr: &[u8; 16] = data[16..32].try_into().unwrap();
            Self {
                lo: u8x16::load(token, lo_arr),
                hi: u8x16::load(token, hi_arr),
            }
        }

        /// Broadcast scalar to all lanes (token-gated)
        #[inline(always)]
        pub fn splat(token: X64V3Token, v: u8) -> Self {
            Self {
                lo: u8x16::splat(token, v),
                hi: u8x16::splat(token, v),
            }
        }

        /// Zero vector (token-gated)
        #[inline(always)]
        pub fn zero(token: X64V3Token) -> Self {
            Self {
                lo: u8x16::zero(token),
                hi: u8x16::zero(token),
            }
        }

        /// Create from array (token-gated)
        #[inline(always)]
        pub fn from_array(token: X64V3Token, arr: [u8; 32]) -> Self {
            Self::load(token, &arr)
        }

        /// Store to array
        #[inline(always)]
        pub fn store(self, out: &mut [u8; 32]) {
            let (lo, hi) = out.split_at_mut(16);
            let lo_arr: &mut [u8; 16] = lo.try_into().unwrap();
            let hi_arr: &mut [u8; 16] = hi.try_into().unwrap();
            self.lo.store(lo_arr);
            self.hi.store(hi_arr);
        }

        /// Convert to array
        #[inline(always)]
        pub fn to_array(self) -> [u8; 32] {
            let mut out = [0u8; 32];
            self.store(&mut out);
            out
        }

        /// Element-wise minimum
        #[inline(always)]
        pub fn min(self, other: Self) -> Self {
            Self {
                lo: self.lo.min(other.lo),
                hi: self.hi.min(other.hi),
            }
        }

        /// Element-wise maximum
        #[inline(always)]
        pub fn max(self, other: Self) -> Self {
            Self {
                lo: self.lo.max(other.lo),
                hi: self.hi.max(other.hi),
            }
        }

        /// Clamp values between lo and hi
        #[inline(always)]
        pub fn clamp(self, lo: Self, hi: Self) -> Self {
            self.max(lo).min(hi)
        }

        /// Reduce: sum all lanes
        #[inline(always)]
        pub fn reduce_add(self) -> u8 {
            self.lo.reduce_add().wrapping_add(self.hi.reduce_add())
        }

        // ========== Bitcast ==========
        /// Reinterpret bits as `i8x32` (zero-cost).
        #[inline(always)]
        pub fn bitcast_i8x32(self) -> i8x32 {
            i8x32 {
                lo: self.lo.bitcast_i8x16(),
                hi: self.hi.bitcast_i8x16(),
            }
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

    impl Add for u8x32 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                lo: self.lo + rhs.lo,
                hi: self.hi + rhs.hi,
            }
        }
    }

    impl Sub for u8x32 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                lo: self.lo - rhs.lo,
                hi: self.hi - rhs.hi,
            }
        }
    }

    impl core::ops::AddAssign for u8x32 {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl core::ops::SubAssign for u8x32 {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    /// Emulated 16-wide i16 vector using two i16x8 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct i16x16 {
        pub(crate) lo: i16x8,
        pub(crate) hi: i16x8,
    }

    impl i16x16 {
        pub const LANES: usize = 16;

        /// Load from array (token-gated)
        #[inline(always)]
        pub fn load(token: X64V3Token, data: &[i16; 16]) -> Self {
            let lo_arr: &[i16; 8] = data[0..8].try_into().unwrap();
            let hi_arr: &[i16; 8] = data[8..16].try_into().unwrap();
            Self {
                lo: i16x8::load(token, lo_arr),
                hi: i16x8::load(token, hi_arr),
            }
        }

        /// Broadcast scalar to all lanes (token-gated)
        #[inline(always)]
        pub fn splat(token: X64V3Token, v: i16) -> Self {
            Self {
                lo: i16x8::splat(token, v),
                hi: i16x8::splat(token, v),
            }
        }

        /// Zero vector (token-gated)
        #[inline(always)]
        pub fn zero(token: X64V3Token) -> Self {
            Self {
                lo: i16x8::zero(token),
                hi: i16x8::zero(token),
            }
        }

        /// Create from array (token-gated)
        #[inline(always)]
        pub fn from_array(token: X64V3Token, arr: [i16; 16]) -> Self {
            Self::load(token, &arr)
        }

        /// Store to array
        #[inline(always)]
        pub fn store(self, out: &mut [i16; 16]) {
            let (lo, hi) = out.split_at_mut(8);
            let lo_arr: &mut [i16; 8] = lo.try_into().unwrap();
            let hi_arr: &mut [i16; 8] = hi.try_into().unwrap();
            self.lo.store(lo_arr);
            self.hi.store(hi_arr);
        }

        /// Convert to array
        #[inline(always)]
        pub fn to_array(self) -> [i16; 16] {
            let mut out = [0i16; 16];
            self.store(&mut out);
            out
        }

        /// Element-wise minimum
        #[inline(always)]
        pub fn min(self, other: Self) -> Self {
            Self {
                lo: self.lo.min(other.lo),
                hi: self.hi.min(other.hi),
            }
        }

        /// Element-wise maximum
        #[inline(always)]
        pub fn max(self, other: Self) -> Self {
            Self {
                lo: self.lo.max(other.lo),
                hi: self.hi.max(other.hi),
            }
        }

        /// Clamp values between lo and hi
        #[inline(always)]
        pub fn clamp(self, lo: Self, hi: Self) -> Self {
            self.max(lo).min(hi)
        }

        /// Absolute value
        #[inline(always)]
        pub fn abs(self) -> Self {
            Self {
                lo: self.lo.abs(),
                hi: self.hi.abs(),
            }
        }

        /// Reduce: sum all lanes
        #[inline(always)]
        pub fn reduce_add(self) -> i16 {
            self.lo.reduce_add().wrapping_add(self.hi.reduce_add())
        }

        // ========== Bitcast ==========
        /// Reinterpret bits as `u16x16` (zero-cost).
        #[inline(always)]
        pub fn bitcast_u16x16(self) -> u16x16 {
            u16x16 {
                lo: self.lo.bitcast_u16x8(),
                hi: self.hi.bitcast_u16x8(),
            }
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

    impl Add for i16x16 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                lo: self.lo + rhs.lo,
                hi: self.hi + rhs.hi,
            }
        }
    }

    impl Sub for i16x16 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                lo: self.lo - rhs.lo,
                hi: self.hi - rhs.hi,
            }
        }
    }

    impl Mul for i16x16 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            Self {
                lo: self.lo * rhs.lo,
                hi: self.hi * rhs.hi,
            }
        }
    }

    impl core::ops::AddAssign for i16x16 {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl core::ops::SubAssign for i16x16 {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    impl core::ops::MulAssign for i16x16 {
        #[inline(always)]
        fn mul_assign(&mut self, rhs: Self) {
            *self = *self * rhs;
        }
    }

    /// Emulated 16-wide u16 vector using two u16x8 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct u16x16 {
        pub(crate) lo: u16x8,
        pub(crate) hi: u16x8,
    }

    impl u16x16 {
        pub const LANES: usize = 16;

        /// Load from array (token-gated)
        #[inline(always)]
        pub fn load(token: X64V3Token, data: &[u16; 16]) -> Self {
            let lo_arr: &[u16; 8] = data[0..8].try_into().unwrap();
            let hi_arr: &[u16; 8] = data[8..16].try_into().unwrap();
            Self {
                lo: u16x8::load(token, lo_arr),
                hi: u16x8::load(token, hi_arr),
            }
        }

        /// Broadcast scalar to all lanes (token-gated)
        #[inline(always)]
        pub fn splat(token: X64V3Token, v: u16) -> Self {
            Self {
                lo: u16x8::splat(token, v),
                hi: u16x8::splat(token, v),
            }
        }

        /// Zero vector (token-gated)
        #[inline(always)]
        pub fn zero(token: X64V3Token) -> Self {
            Self {
                lo: u16x8::zero(token),
                hi: u16x8::zero(token),
            }
        }

        /// Create from array (token-gated)
        #[inline(always)]
        pub fn from_array(token: X64V3Token, arr: [u16; 16]) -> Self {
            Self::load(token, &arr)
        }

        /// Store to array
        #[inline(always)]
        pub fn store(self, out: &mut [u16; 16]) {
            let (lo, hi) = out.split_at_mut(8);
            let lo_arr: &mut [u16; 8] = lo.try_into().unwrap();
            let hi_arr: &mut [u16; 8] = hi.try_into().unwrap();
            self.lo.store(lo_arr);
            self.hi.store(hi_arr);
        }

        /// Convert to array
        #[inline(always)]
        pub fn to_array(self) -> [u16; 16] {
            let mut out = [0u16; 16];
            self.store(&mut out);
            out
        }

        /// Element-wise minimum
        #[inline(always)]
        pub fn min(self, other: Self) -> Self {
            Self {
                lo: self.lo.min(other.lo),
                hi: self.hi.min(other.hi),
            }
        }

        /// Element-wise maximum
        #[inline(always)]
        pub fn max(self, other: Self) -> Self {
            Self {
                lo: self.lo.max(other.lo),
                hi: self.hi.max(other.hi),
            }
        }

        /// Clamp values between lo and hi
        #[inline(always)]
        pub fn clamp(self, lo: Self, hi: Self) -> Self {
            self.max(lo).min(hi)
        }

        /// Reduce: sum all lanes
        #[inline(always)]
        pub fn reduce_add(self) -> u16 {
            self.lo.reduce_add().wrapping_add(self.hi.reduce_add())
        }

        // ========== Bitcast ==========
        /// Reinterpret bits as `i16x16` (zero-cost).
        #[inline(always)]
        pub fn bitcast_i16x16(self) -> i16x16 {
            i16x16 {
                lo: self.lo.bitcast_i16x8(),
                hi: self.hi.bitcast_i16x8(),
            }
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

    impl Add for u16x16 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                lo: self.lo + rhs.lo,
                hi: self.hi + rhs.hi,
            }
        }
    }

    impl Sub for u16x16 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                lo: self.lo - rhs.lo,
                hi: self.hi - rhs.hi,
            }
        }
    }

    impl Mul for u16x16 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            Self {
                lo: self.lo * rhs.lo,
                hi: self.hi * rhs.hi,
            }
        }
    }

    impl core::ops::AddAssign for u16x16 {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl core::ops::SubAssign for u16x16 {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    impl core::ops::MulAssign for u16x16 {
        #[inline(always)]
        fn mul_assign(&mut self, rhs: Self) {
            *self = *self * rhs;
        }
    }

    /// Emulated 8-wide i32 vector using two i32x4 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct i32x8 {
        pub(crate) lo: i32x4,
        pub(crate) hi: i32x4,
    }

    impl i32x8 {
        pub const LANES: usize = 8;

        /// Load from array (token-gated)
        #[inline(always)]
        pub fn load(token: X64V3Token, data: &[i32; 8]) -> Self {
            let lo_arr: &[i32; 4] = data[0..4].try_into().unwrap();
            let hi_arr: &[i32; 4] = data[4..8].try_into().unwrap();
            Self {
                lo: i32x4::load(token, lo_arr),
                hi: i32x4::load(token, hi_arr),
            }
        }

        /// Broadcast scalar to all lanes (token-gated)
        #[inline(always)]
        pub fn splat(token: X64V3Token, v: i32) -> Self {
            Self {
                lo: i32x4::splat(token, v),
                hi: i32x4::splat(token, v),
            }
        }

        /// Zero vector (token-gated)
        #[inline(always)]
        pub fn zero(token: X64V3Token) -> Self {
            Self {
                lo: i32x4::zero(token),
                hi: i32x4::zero(token),
            }
        }

        /// Create from array (token-gated)
        #[inline(always)]
        pub fn from_array(token: X64V3Token, arr: [i32; 8]) -> Self {
            Self::load(token, &arr)
        }

        /// Store to array
        #[inline(always)]
        pub fn store(self, out: &mut [i32; 8]) {
            let (lo, hi) = out.split_at_mut(4);
            let lo_arr: &mut [i32; 4] = lo.try_into().unwrap();
            let hi_arr: &mut [i32; 4] = hi.try_into().unwrap();
            self.lo.store(lo_arr);
            self.hi.store(hi_arr);
        }

        /// Convert to array
        #[inline(always)]
        pub fn to_array(self) -> [i32; 8] {
            let mut out = [0i32; 8];
            self.store(&mut out);
            out
        }

        /// Element-wise minimum
        #[inline(always)]
        pub fn min(self, other: Self) -> Self {
            Self {
                lo: self.lo.min(other.lo),
                hi: self.hi.min(other.hi),
            }
        }

        /// Element-wise maximum
        #[inline(always)]
        pub fn max(self, other: Self) -> Self {
            Self {
                lo: self.lo.max(other.lo),
                hi: self.hi.max(other.hi),
            }
        }

        /// Clamp values between lo and hi
        #[inline(always)]
        pub fn clamp(self, lo: Self, hi: Self) -> Self {
            self.max(lo).min(hi)
        }

        /// Absolute value
        #[inline(always)]
        pub fn abs(self) -> Self {
            Self {
                lo: self.lo.abs(),
                hi: self.hi.abs(),
            }
        }

        /// Reduce: sum all lanes
        #[inline(always)]
        pub fn reduce_add(self) -> i32 {
            self.lo.reduce_add().wrapping_add(self.hi.reduce_add())
        }

        // ========== Bitcast ==========
        /// Reinterpret bits as `f32x8` (zero-cost).
        #[inline(always)]
        pub fn bitcast_f32x8(self) -> f32x8 {
            f32x8 {
                lo: self.lo.bitcast_f32x4(),
                hi: self.hi.bitcast_f32x4(),
            }
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
            u32x8 {
                lo: self.lo.bitcast_u32x4(),
                hi: self.hi.bitcast_u32x4(),
            }
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

    impl Add for i32x8 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                lo: self.lo + rhs.lo,
                hi: self.hi + rhs.hi,
            }
        }
    }

    impl Sub for i32x8 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                lo: self.lo - rhs.lo,
                hi: self.hi - rhs.hi,
            }
        }
    }

    impl Mul for i32x8 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            Self {
                lo: self.lo * rhs.lo,
                hi: self.hi * rhs.hi,
            }
        }
    }

    impl core::ops::AddAssign for i32x8 {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl core::ops::SubAssign for i32x8 {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    impl core::ops::MulAssign for i32x8 {
        #[inline(always)]
        fn mul_assign(&mut self, rhs: Self) {
            *self = *self * rhs;
        }
    }

    /// Emulated 8-wide u32 vector using two u32x4 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct u32x8 {
        pub(crate) lo: u32x4,
        pub(crate) hi: u32x4,
    }

    impl u32x8 {
        pub const LANES: usize = 8;

        /// Load from array (token-gated)
        #[inline(always)]
        pub fn load(token: X64V3Token, data: &[u32; 8]) -> Self {
            let lo_arr: &[u32; 4] = data[0..4].try_into().unwrap();
            let hi_arr: &[u32; 4] = data[4..8].try_into().unwrap();
            Self {
                lo: u32x4::load(token, lo_arr),
                hi: u32x4::load(token, hi_arr),
            }
        }

        /// Broadcast scalar to all lanes (token-gated)
        #[inline(always)]
        pub fn splat(token: X64V3Token, v: u32) -> Self {
            Self {
                lo: u32x4::splat(token, v),
                hi: u32x4::splat(token, v),
            }
        }

        /// Zero vector (token-gated)
        #[inline(always)]
        pub fn zero(token: X64V3Token) -> Self {
            Self {
                lo: u32x4::zero(token),
                hi: u32x4::zero(token),
            }
        }

        /// Create from array (token-gated)
        #[inline(always)]
        pub fn from_array(token: X64V3Token, arr: [u32; 8]) -> Self {
            Self::load(token, &arr)
        }

        /// Store to array
        #[inline(always)]
        pub fn store(self, out: &mut [u32; 8]) {
            let (lo, hi) = out.split_at_mut(4);
            let lo_arr: &mut [u32; 4] = lo.try_into().unwrap();
            let hi_arr: &mut [u32; 4] = hi.try_into().unwrap();
            self.lo.store(lo_arr);
            self.hi.store(hi_arr);
        }

        /// Convert to array
        #[inline(always)]
        pub fn to_array(self) -> [u32; 8] {
            let mut out = [0u32; 8];
            self.store(&mut out);
            out
        }

        /// Element-wise minimum
        #[inline(always)]
        pub fn min(self, other: Self) -> Self {
            Self {
                lo: self.lo.min(other.lo),
                hi: self.hi.min(other.hi),
            }
        }

        /// Element-wise maximum
        #[inline(always)]
        pub fn max(self, other: Self) -> Self {
            Self {
                lo: self.lo.max(other.lo),
                hi: self.hi.max(other.hi),
            }
        }

        /// Clamp values between lo and hi
        #[inline(always)]
        pub fn clamp(self, lo: Self, hi: Self) -> Self {
            self.max(lo).min(hi)
        }

        /// Reduce: sum all lanes
        #[inline(always)]
        pub fn reduce_add(self) -> u32 {
            self.lo.reduce_add().wrapping_add(self.hi.reduce_add())
        }

        // ========== Bitcast ==========
        /// Reinterpret bits as `f32x8` (zero-cost).
        #[inline(always)]
        pub fn bitcast_f32x8(self) -> f32x8 {
            f32x8 {
                lo: self.lo.bitcast_f32x4(),
                hi: self.hi.bitcast_f32x4(),
            }
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
            i32x8 {
                lo: self.lo.bitcast_i32x4(),
                hi: self.hi.bitcast_i32x4(),
            }
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

    impl Add for u32x8 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                lo: self.lo + rhs.lo,
                hi: self.hi + rhs.hi,
            }
        }
    }

    impl Sub for u32x8 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                lo: self.lo - rhs.lo,
                hi: self.hi - rhs.hi,
            }
        }
    }

    impl Mul for u32x8 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            Self {
                lo: self.lo * rhs.lo,
                hi: self.hi * rhs.hi,
            }
        }
    }

    impl core::ops::AddAssign for u32x8 {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl core::ops::SubAssign for u32x8 {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    impl core::ops::MulAssign for u32x8 {
        #[inline(always)]
        fn mul_assign(&mut self, rhs: Self) {
            *self = *self * rhs;
        }
    }

    /// Emulated 4-wide i64 vector using two i64x2 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct i64x4 {
        pub(crate) lo: i64x2,
        pub(crate) hi: i64x2,
    }

    impl i64x4 {
        pub const LANES: usize = 4;

        /// Load from array (token-gated)
        #[inline(always)]
        pub fn load(token: X64V3Token, data: &[i64; 4]) -> Self {
            let lo_arr: &[i64; 2] = data[0..2].try_into().unwrap();
            let hi_arr: &[i64; 2] = data[2..4].try_into().unwrap();
            Self {
                lo: i64x2::load(token, lo_arr),
                hi: i64x2::load(token, hi_arr),
            }
        }

        /// Broadcast scalar to all lanes (token-gated)
        #[inline(always)]
        pub fn splat(token: X64V3Token, v: i64) -> Self {
            Self {
                lo: i64x2::splat(token, v),
                hi: i64x2::splat(token, v),
            }
        }

        /// Zero vector (token-gated)
        #[inline(always)]
        pub fn zero(token: X64V3Token) -> Self {
            Self {
                lo: i64x2::zero(token),
                hi: i64x2::zero(token),
            }
        }

        /// Create from array (token-gated)
        #[inline(always)]
        pub fn from_array(token: X64V3Token, arr: [i64; 4]) -> Self {
            Self::load(token, &arr)
        }

        /// Store to array
        #[inline(always)]
        pub fn store(self, out: &mut [i64; 4]) {
            let (lo, hi) = out.split_at_mut(2);
            let lo_arr: &mut [i64; 2] = lo.try_into().unwrap();
            let hi_arr: &mut [i64; 2] = hi.try_into().unwrap();
            self.lo.store(lo_arr);
            self.hi.store(hi_arr);
        }

        /// Convert to array
        #[inline(always)]
        pub fn to_array(self) -> [i64; 4] {
            let mut out = [0i64; 4];
            self.store(&mut out);
            out
        }

        /// Reduce: sum all lanes
        #[inline(always)]
        pub fn reduce_add(self) -> i64 {
            self.lo.reduce_add().wrapping_add(self.hi.reduce_add())
        }

        // ========== Bitcast ==========
        /// Reinterpret bits as `f64x4` (zero-cost).
        #[inline(always)]
        pub fn bitcast_f64x4(self) -> f64x4 {
            f64x4 {
                lo: self.lo.bitcast_f64x2(),
                hi: self.hi.bitcast_f64x2(),
            }
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
            u64x4 {
                lo: self.lo.bitcast_u64x2(),
                hi: self.hi.bitcast_u64x2(),
            }
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

    impl Add for i64x4 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                lo: self.lo + rhs.lo,
                hi: self.hi + rhs.hi,
            }
        }
    }

    impl Sub for i64x4 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                lo: self.lo - rhs.lo,
                hi: self.hi - rhs.hi,
            }
        }
    }

    impl core::ops::AddAssign for i64x4 {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl core::ops::SubAssign for i64x4 {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    /// Emulated 4-wide u64 vector using two u64x2 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct u64x4 {
        pub(crate) lo: u64x2,
        pub(crate) hi: u64x2,
    }

    impl u64x4 {
        pub const LANES: usize = 4;

        /// Load from array (token-gated)
        #[inline(always)]
        pub fn load(token: X64V3Token, data: &[u64; 4]) -> Self {
            let lo_arr: &[u64; 2] = data[0..2].try_into().unwrap();
            let hi_arr: &[u64; 2] = data[2..4].try_into().unwrap();
            Self {
                lo: u64x2::load(token, lo_arr),
                hi: u64x2::load(token, hi_arr),
            }
        }

        /// Broadcast scalar to all lanes (token-gated)
        #[inline(always)]
        pub fn splat(token: X64V3Token, v: u64) -> Self {
            Self {
                lo: u64x2::splat(token, v),
                hi: u64x2::splat(token, v),
            }
        }

        /// Zero vector (token-gated)
        #[inline(always)]
        pub fn zero(token: X64V3Token) -> Self {
            Self {
                lo: u64x2::zero(token),
                hi: u64x2::zero(token),
            }
        }

        /// Create from array (token-gated)
        #[inline(always)]
        pub fn from_array(token: X64V3Token, arr: [u64; 4]) -> Self {
            Self::load(token, &arr)
        }

        /// Store to array
        #[inline(always)]
        pub fn store(self, out: &mut [u64; 4]) {
            let (lo, hi) = out.split_at_mut(2);
            let lo_arr: &mut [u64; 2] = lo.try_into().unwrap();
            let hi_arr: &mut [u64; 2] = hi.try_into().unwrap();
            self.lo.store(lo_arr);
            self.hi.store(hi_arr);
        }

        /// Convert to array
        #[inline(always)]
        pub fn to_array(self) -> [u64; 4] {
            let mut out = [0u64; 4];
            self.store(&mut out);
            out
        }

        /// Reduce: sum all lanes
        #[inline(always)]
        pub fn reduce_add(self) -> u64 {
            self.lo.reduce_add().wrapping_add(self.hi.reduce_add())
        }

        // ========== Bitcast ==========
        /// Reinterpret bits as `f64x4` (zero-cost).
        #[inline(always)]
        pub fn bitcast_f64x4(self) -> f64x4 {
            f64x4 {
                lo: self.lo.bitcast_f64x2(),
                hi: self.hi.bitcast_f64x2(),
            }
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
            i64x4 {
                lo: self.lo.bitcast_i64x2(),
                hi: self.hi.bitcast_i64x2(),
            }
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

    impl Add for u64x4 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                lo: self.lo + rhs.lo,
                hi: self.hi + rhs.hi,
            }
        }
    }

    impl Sub for u64x4 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                lo: self.lo - rhs.lo,
                hi: self.hi - rhs.hi,
            }
        }
    }

    impl core::ops::AddAssign for u64x4 {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl core::ops::SubAssign for u64x4 {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    // Width-aliased type names
    pub type f32xN = f32x8;
    pub type f64xN = f64x4;
    pub type i8xN = i8x32;
    pub type u8xN = u8x32;
    pub type i16xN = i16x16;
    pub type u16xN = u16x16;
    pub type i32xN = i32x8;
    pub type u32xN = u32x8;
    pub type i64xN = i64x4;
    pub type u64xN = u64x4;

    /// Number of f32 lanes
    pub const LANES_F32: usize = 8;
    /// Number of f64 lanes
    pub const LANES_F64: usize = 4;
    /// Number of i32/u32 lanes
    pub const LANES_32: usize = 8;
    /// Number of i16/u16 lanes
    pub const LANES_16: usize = 16;
    /// Number of i8/u8 lanes
    pub const LANES_8: usize = 32;

    /// Token type for this polyfill level
    pub type Token = archmage::X64V3Token;
}

#[cfg(target_arch = "aarch64")]
pub mod neon {
    //! Polyfilled 256-bit types using NEON (128-bit) operations.

    use crate::simd::generated::arm::w128::{
        f32x4, f64x2, i8x16, i16x8, i32x4, i64x2, u8x16, u16x8, u32x4, u64x2,
    };
    use archmage::NeonToken;
    use core::ops::{Add, Div, Mul, Neg, Sub};

    /// Emulated 8-wide f32 vector using two f32x4 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct f32x8 {
        pub(crate) lo: f32x4,
        pub(crate) hi: f32x4,
    }

    impl f32x8 {
        pub const LANES: usize = 8;

        /// Load from array (token-gated)
        #[inline(always)]
        pub fn load(token: NeonToken, data: &[f32; 8]) -> Self {
            let lo_arr: &[f32; 4] = data[0..4].try_into().unwrap();
            let hi_arr: &[f32; 4] = data[4..8].try_into().unwrap();
            Self {
                lo: f32x4::load(token, lo_arr),
                hi: f32x4::load(token, hi_arr),
            }
        }

        /// Broadcast scalar to all lanes (token-gated)
        #[inline(always)]
        pub fn splat(token: NeonToken, v: f32) -> Self {
            Self {
                lo: f32x4::splat(token, v),
                hi: f32x4::splat(token, v),
            }
        }

        /// Zero vector (token-gated)
        #[inline(always)]
        pub fn zero(token: NeonToken) -> Self {
            Self {
                lo: f32x4::zero(token),
                hi: f32x4::zero(token),
            }
        }

        /// Create from array (token-gated)
        #[inline(always)]
        pub fn from_array(token: NeonToken, arr: [f32; 8]) -> Self {
            Self::load(token, &arr)
        }

        /// Store to array
        #[inline(always)]
        pub fn store(self, out: &mut [f32; 8]) {
            let (lo, hi) = out.split_at_mut(4);
            let lo_arr: &mut [f32; 4] = lo.try_into().unwrap();
            let hi_arr: &mut [f32; 4] = hi.try_into().unwrap();
            self.lo.store(lo_arr);
            self.hi.store(hi_arr);
        }

        /// Convert to array
        #[inline(always)]
        pub fn to_array(self) -> [f32; 8] {
            let mut out = [0.0f32; 8];
            self.store(&mut out);
            out
        }

        /// Element-wise minimum
        #[inline(always)]
        pub fn min(self, other: Self) -> Self {
            Self {
                lo: self.lo.min(other.lo),
                hi: self.hi.min(other.hi),
            }
        }

        /// Element-wise maximum
        #[inline(always)]
        pub fn max(self, other: Self) -> Self {
            Self {
                lo: self.lo.max(other.lo),
                hi: self.hi.max(other.hi),
            }
        }

        /// Clamp values between lo and hi
        #[inline(always)]
        pub fn clamp(self, lo: Self, hi: Self) -> Self {
            self.max(lo).min(hi)
        }

        /// Square root
        #[inline(always)]
        pub fn sqrt(self) -> Self {
            Self {
                lo: self.lo.sqrt(),
                hi: self.hi.sqrt(),
            }
        }

        /// Floor
        #[inline(always)]
        pub fn floor(self) -> Self {
            Self {
                lo: self.lo.floor(),
                hi: self.hi.floor(),
            }
        }

        /// Ceil
        #[inline(always)]
        pub fn ceil(self) -> Self {
            Self {
                lo: self.lo.ceil(),
                hi: self.hi.ceil(),
            }
        }

        /// Round to nearest
        #[inline(always)]
        pub fn round(self) -> Self {
            Self {
                lo: self.lo.round(),
                hi: self.hi.round(),
            }
        }

        /// Fused multiply-add: self * a + b
        #[inline(always)]
        pub fn mul_add(self, a: Self, b: Self) -> Self {
            Self {
                lo: self.lo.mul_add(a.lo, b.lo),
                hi: self.hi.mul_add(a.hi, b.hi),
            }
        }

        /// Absolute value
        #[inline(always)]
        pub fn abs(self) -> Self {
            Self {
                lo: self.lo.abs(),
                hi: self.hi.abs(),
            }
        }

        /// Reduce: sum all lanes
        #[inline(always)]
        pub fn reduce_add(self) -> f32 {
            self.lo.reduce_add() + self.hi.reduce_add()
        }

        /// Reduce: max of all lanes
        #[inline(always)]
        pub fn reduce_max(self) -> f32 {
            self.lo.reduce_max().max(self.hi.reduce_max())
        }

        /// Reduce: min of all lanes
        #[inline(always)]
        pub fn reduce_min(self) -> f32 {
            self.lo.reduce_min().min(self.hi.reduce_min())
        }

        // ========== Bitcast ==========
        /// Reinterpret bits as `i32x8` (zero-cost).
        #[inline(always)]
        pub fn bitcast_i32x8(self) -> i32x8 {
            i32x8 {
                lo: self.lo.bitcast_i32x4(),
                hi: self.hi.bitcast_i32x4(),
            }
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
            u32x8 {
                lo: self.lo.bitcast_u32x4(),
                hi: self.hi.bitcast_u32x4(),
            }
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

    impl Add for f32x8 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                lo: self.lo + rhs.lo,
                hi: self.hi + rhs.hi,
            }
        }
    }

    impl Sub for f32x8 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                lo: self.lo - rhs.lo,
                hi: self.hi - rhs.hi,
            }
        }
    }

    impl Mul for f32x8 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            Self {
                lo: self.lo * rhs.lo,
                hi: self.hi * rhs.hi,
            }
        }
    }

    impl Div for f32x8 {
        type Output = Self;
        #[inline(always)]
        fn div(self, rhs: Self) -> Self {
            Self {
                lo: self.lo / rhs.lo,
                hi: self.hi / rhs.hi,
            }
        }
    }

    impl Neg for f32x8 {
        type Output = Self;
        #[inline(always)]
        fn neg(self) -> Self {
            Self {
                lo: -self.lo,
                hi: -self.hi,
            }
        }
    }

    impl core::ops::AddAssign for f32x8 {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl core::ops::SubAssign for f32x8 {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    impl core::ops::MulAssign for f32x8 {
        #[inline(always)]
        fn mul_assign(&mut self, rhs: Self) {
            *self = *self * rhs;
        }
    }

    impl core::ops::DivAssign for f32x8 {
        #[inline(always)]
        fn div_assign(&mut self, rhs: Self) {
            *self = *self / rhs;
        }
    }

    /// Emulated 4-wide f64 vector using two f64x2 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct f64x4 {
        pub(crate) lo: f64x2,
        pub(crate) hi: f64x2,
    }

    impl f64x4 {
        pub const LANES: usize = 4;

        /// Load from array (token-gated)
        #[inline(always)]
        pub fn load(token: NeonToken, data: &[f64; 4]) -> Self {
            let lo_arr: &[f64; 2] = data[0..2].try_into().unwrap();
            let hi_arr: &[f64; 2] = data[2..4].try_into().unwrap();
            Self {
                lo: f64x2::load(token, lo_arr),
                hi: f64x2::load(token, hi_arr),
            }
        }

        /// Broadcast scalar to all lanes (token-gated)
        #[inline(always)]
        pub fn splat(token: NeonToken, v: f64) -> Self {
            Self {
                lo: f64x2::splat(token, v),
                hi: f64x2::splat(token, v),
            }
        }

        /// Zero vector (token-gated)
        #[inline(always)]
        pub fn zero(token: NeonToken) -> Self {
            Self {
                lo: f64x2::zero(token),
                hi: f64x2::zero(token),
            }
        }

        /// Create from array (token-gated)
        #[inline(always)]
        pub fn from_array(token: NeonToken, arr: [f64; 4]) -> Self {
            Self::load(token, &arr)
        }

        /// Store to array
        #[inline(always)]
        pub fn store(self, out: &mut [f64; 4]) {
            let (lo, hi) = out.split_at_mut(2);
            let lo_arr: &mut [f64; 2] = lo.try_into().unwrap();
            let hi_arr: &mut [f64; 2] = hi.try_into().unwrap();
            self.lo.store(lo_arr);
            self.hi.store(hi_arr);
        }

        /// Convert to array
        #[inline(always)]
        pub fn to_array(self) -> [f64; 4] {
            let mut out = [0.0f64; 4];
            self.store(&mut out);
            out
        }

        /// Element-wise minimum
        #[inline(always)]
        pub fn min(self, other: Self) -> Self {
            Self {
                lo: self.lo.min(other.lo),
                hi: self.hi.min(other.hi),
            }
        }

        /// Element-wise maximum
        #[inline(always)]
        pub fn max(self, other: Self) -> Self {
            Self {
                lo: self.lo.max(other.lo),
                hi: self.hi.max(other.hi),
            }
        }

        /// Clamp values between lo and hi
        #[inline(always)]
        pub fn clamp(self, lo: Self, hi: Self) -> Self {
            self.max(lo).min(hi)
        }

        /// Square root
        #[inline(always)]
        pub fn sqrt(self) -> Self {
            Self {
                lo: self.lo.sqrt(),
                hi: self.hi.sqrt(),
            }
        }

        /// Floor
        #[inline(always)]
        pub fn floor(self) -> Self {
            Self {
                lo: self.lo.floor(),
                hi: self.hi.floor(),
            }
        }

        /// Ceil
        #[inline(always)]
        pub fn ceil(self) -> Self {
            Self {
                lo: self.lo.ceil(),
                hi: self.hi.ceil(),
            }
        }

        /// Round to nearest
        #[inline(always)]
        pub fn round(self) -> Self {
            Self {
                lo: self.lo.round(),
                hi: self.hi.round(),
            }
        }

        /// Fused multiply-add: self * a + b
        #[inline(always)]
        pub fn mul_add(self, a: Self, b: Self) -> Self {
            Self {
                lo: self.lo.mul_add(a.lo, b.lo),
                hi: self.hi.mul_add(a.hi, b.hi),
            }
        }

        /// Absolute value
        #[inline(always)]
        pub fn abs(self) -> Self {
            Self {
                lo: self.lo.abs(),
                hi: self.hi.abs(),
            }
        }

        /// Reduce: sum all lanes
        #[inline(always)]
        pub fn reduce_add(self) -> f64 {
            self.lo.reduce_add() + self.hi.reduce_add()
        }

        /// Reduce: max of all lanes
        #[inline(always)]
        pub fn reduce_max(self) -> f64 {
            self.lo.reduce_max().max(self.hi.reduce_max())
        }

        /// Reduce: min of all lanes
        #[inline(always)]
        pub fn reduce_min(self) -> f64 {
            self.lo.reduce_min().min(self.hi.reduce_min())
        }

        // ========== Bitcast ==========
        /// Reinterpret bits as `i64x4` (zero-cost).
        #[inline(always)]
        pub fn bitcast_i64x4(self) -> i64x4 {
            i64x4 {
                lo: self.lo.bitcast_i64x2(),
                hi: self.hi.bitcast_i64x2(),
            }
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
            u64x4 {
                lo: self.lo.bitcast_u64x2(),
                hi: self.hi.bitcast_u64x2(),
            }
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

    impl Add for f64x4 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                lo: self.lo + rhs.lo,
                hi: self.hi + rhs.hi,
            }
        }
    }

    impl Sub for f64x4 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                lo: self.lo - rhs.lo,
                hi: self.hi - rhs.hi,
            }
        }
    }

    impl Mul for f64x4 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            Self {
                lo: self.lo * rhs.lo,
                hi: self.hi * rhs.hi,
            }
        }
    }

    impl Div for f64x4 {
        type Output = Self;
        #[inline(always)]
        fn div(self, rhs: Self) -> Self {
            Self {
                lo: self.lo / rhs.lo,
                hi: self.hi / rhs.hi,
            }
        }
    }

    impl Neg for f64x4 {
        type Output = Self;
        #[inline(always)]
        fn neg(self) -> Self {
            Self {
                lo: -self.lo,
                hi: -self.hi,
            }
        }
    }

    impl core::ops::AddAssign for f64x4 {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl core::ops::SubAssign for f64x4 {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    impl core::ops::MulAssign for f64x4 {
        #[inline(always)]
        fn mul_assign(&mut self, rhs: Self) {
            *self = *self * rhs;
        }
    }

    impl core::ops::DivAssign for f64x4 {
        #[inline(always)]
        fn div_assign(&mut self, rhs: Self) {
            *self = *self / rhs;
        }
    }

    /// Emulated 32-wide i8 vector using two i8x16 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct i8x32 {
        pub(crate) lo: i8x16,
        pub(crate) hi: i8x16,
    }

    impl i8x32 {
        pub const LANES: usize = 32;

        /// Load from array (token-gated)
        #[inline(always)]
        pub fn load(token: NeonToken, data: &[i8; 32]) -> Self {
            let lo_arr: &[i8; 16] = data[0..16].try_into().unwrap();
            let hi_arr: &[i8; 16] = data[16..32].try_into().unwrap();
            Self {
                lo: i8x16::load(token, lo_arr),
                hi: i8x16::load(token, hi_arr),
            }
        }

        /// Broadcast scalar to all lanes (token-gated)
        #[inline(always)]
        pub fn splat(token: NeonToken, v: i8) -> Self {
            Self {
                lo: i8x16::splat(token, v),
                hi: i8x16::splat(token, v),
            }
        }

        /// Zero vector (token-gated)
        #[inline(always)]
        pub fn zero(token: NeonToken) -> Self {
            Self {
                lo: i8x16::zero(token),
                hi: i8x16::zero(token),
            }
        }

        /// Create from array (token-gated)
        #[inline(always)]
        pub fn from_array(token: NeonToken, arr: [i8; 32]) -> Self {
            Self::load(token, &arr)
        }

        /// Store to array
        #[inline(always)]
        pub fn store(self, out: &mut [i8; 32]) {
            let (lo, hi) = out.split_at_mut(16);
            let lo_arr: &mut [i8; 16] = lo.try_into().unwrap();
            let hi_arr: &mut [i8; 16] = hi.try_into().unwrap();
            self.lo.store(lo_arr);
            self.hi.store(hi_arr);
        }

        /// Convert to array
        #[inline(always)]
        pub fn to_array(self) -> [i8; 32] {
            let mut out = [0i8; 32];
            self.store(&mut out);
            out
        }

        /// Element-wise minimum
        #[inline(always)]
        pub fn min(self, other: Self) -> Self {
            Self {
                lo: self.lo.min(other.lo),
                hi: self.hi.min(other.hi),
            }
        }

        /// Element-wise maximum
        #[inline(always)]
        pub fn max(self, other: Self) -> Self {
            Self {
                lo: self.lo.max(other.lo),
                hi: self.hi.max(other.hi),
            }
        }

        /// Clamp values between lo and hi
        #[inline(always)]
        pub fn clamp(self, lo: Self, hi: Self) -> Self {
            self.max(lo).min(hi)
        }

        /// Absolute value
        #[inline(always)]
        pub fn abs(self) -> Self {
            Self {
                lo: self.lo.abs(),
                hi: self.hi.abs(),
            }
        }

        /// Reduce: sum all lanes
        #[inline(always)]
        pub fn reduce_add(self) -> i8 {
            self.lo.reduce_add().wrapping_add(self.hi.reduce_add())
        }

        // ========== Bitcast ==========
        /// Reinterpret bits as `u8x32` (zero-cost).
        #[inline(always)]
        pub fn bitcast_u8x32(self) -> u8x32 {
            u8x32 {
                lo: self.lo.bitcast_u8x16(),
                hi: self.hi.bitcast_u8x16(),
            }
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

    impl Add for i8x32 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                lo: self.lo + rhs.lo,
                hi: self.hi + rhs.hi,
            }
        }
    }

    impl Sub for i8x32 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                lo: self.lo - rhs.lo,
                hi: self.hi - rhs.hi,
            }
        }
    }

    impl core::ops::AddAssign for i8x32 {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl core::ops::SubAssign for i8x32 {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    /// Emulated 32-wide u8 vector using two u8x16 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct u8x32 {
        pub(crate) lo: u8x16,
        pub(crate) hi: u8x16,
    }

    impl u8x32 {
        pub const LANES: usize = 32;

        /// Load from array (token-gated)
        #[inline(always)]
        pub fn load(token: NeonToken, data: &[u8; 32]) -> Self {
            let lo_arr: &[u8; 16] = data[0..16].try_into().unwrap();
            let hi_arr: &[u8; 16] = data[16..32].try_into().unwrap();
            Self {
                lo: u8x16::load(token, lo_arr),
                hi: u8x16::load(token, hi_arr),
            }
        }

        /// Broadcast scalar to all lanes (token-gated)
        #[inline(always)]
        pub fn splat(token: NeonToken, v: u8) -> Self {
            Self {
                lo: u8x16::splat(token, v),
                hi: u8x16::splat(token, v),
            }
        }

        /// Zero vector (token-gated)
        #[inline(always)]
        pub fn zero(token: NeonToken) -> Self {
            Self {
                lo: u8x16::zero(token),
                hi: u8x16::zero(token),
            }
        }

        /// Create from array (token-gated)
        #[inline(always)]
        pub fn from_array(token: NeonToken, arr: [u8; 32]) -> Self {
            Self::load(token, &arr)
        }

        /// Store to array
        #[inline(always)]
        pub fn store(self, out: &mut [u8; 32]) {
            let (lo, hi) = out.split_at_mut(16);
            let lo_arr: &mut [u8; 16] = lo.try_into().unwrap();
            let hi_arr: &mut [u8; 16] = hi.try_into().unwrap();
            self.lo.store(lo_arr);
            self.hi.store(hi_arr);
        }

        /// Convert to array
        #[inline(always)]
        pub fn to_array(self) -> [u8; 32] {
            let mut out = [0u8; 32];
            self.store(&mut out);
            out
        }

        /// Element-wise minimum
        #[inline(always)]
        pub fn min(self, other: Self) -> Self {
            Self {
                lo: self.lo.min(other.lo),
                hi: self.hi.min(other.hi),
            }
        }

        /// Element-wise maximum
        #[inline(always)]
        pub fn max(self, other: Self) -> Self {
            Self {
                lo: self.lo.max(other.lo),
                hi: self.hi.max(other.hi),
            }
        }

        /// Clamp values between lo and hi
        #[inline(always)]
        pub fn clamp(self, lo: Self, hi: Self) -> Self {
            self.max(lo).min(hi)
        }

        /// Reduce: sum all lanes
        #[inline(always)]
        pub fn reduce_add(self) -> u8 {
            self.lo.reduce_add().wrapping_add(self.hi.reduce_add())
        }

        // ========== Bitcast ==========
        /// Reinterpret bits as `i8x32` (zero-cost).
        #[inline(always)]
        pub fn bitcast_i8x32(self) -> i8x32 {
            i8x32 {
                lo: self.lo.bitcast_i8x16(),
                hi: self.hi.bitcast_i8x16(),
            }
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

    impl Add for u8x32 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                lo: self.lo + rhs.lo,
                hi: self.hi + rhs.hi,
            }
        }
    }

    impl Sub for u8x32 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                lo: self.lo - rhs.lo,
                hi: self.hi - rhs.hi,
            }
        }
    }

    impl core::ops::AddAssign for u8x32 {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl core::ops::SubAssign for u8x32 {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    /// Emulated 16-wide i16 vector using two i16x8 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct i16x16 {
        pub(crate) lo: i16x8,
        pub(crate) hi: i16x8,
    }

    impl i16x16 {
        pub const LANES: usize = 16;

        /// Load from array (token-gated)
        #[inline(always)]
        pub fn load(token: NeonToken, data: &[i16; 16]) -> Self {
            let lo_arr: &[i16; 8] = data[0..8].try_into().unwrap();
            let hi_arr: &[i16; 8] = data[8..16].try_into().unwrap();
            Self {
                lo: i16x8::load(token, lo_arr),
                hi: i16x8::load(token, hi_arr),
            }
        }

        /// Broadcast scalar to all lanes (token-gated)
        #[inline(always)]
        pub fn splat(token: NeonToken, v: i16) -> Self {
            Self {
                lo: i16x8::splat(token, v),
                hi: i16x8::splat(token, v),
            }
        }

        /// Zero vector (token-gated)
        #[inline(always)]
        pub fn zero(token: NeonToken) -> Self {
            Self {
                lo: i16x8::zero(token),
                hi: i16x8::zero(token),
            }
        }

        /// Create from array (token-gated)
        #[inline(always)]
        pub fn from_array(token: NeonToken, arr: [i16; 16]) -> Self {
            Self::load(token, &arr)
        }

        /// Store to array
        #[inline(always)]
        pub fn store(self, out: &mut [i16; 16]) {
            let (lo, hi) = out.split_at_mut(8);
            let lo_arr: &mut [i16; 8] = lo.try_into().unwrap();
            let hi_arr: &mut [i16; 8] = hi.try_into().unwrap();
            self.lo.store(lo_arr);
            self.hi.store(hi_arr);
        }

        /// Convert to array
        #[inline(always)]
        pub fn to_array(self) -> [i16; 16] {
            let mut out = [0i16; 16];
            self.store(&mut out);
            out
        }

        /// Element-wise minimum
        #[inline(always)]
        pub fn min(self, other: Self) -> Self {
            Self {
                lo: self.lo.min(other.lo),
                hi: self.hi.min(other.hi),
            }
        }

        /// Element-wise maximum
        #[inline(always)]
        pub fn max(self, other: Self) -> Self {
            Self {
                lo: self.lo.max(other.lo),
                hi: self.hi.max(other.hi),
            }
        }

        /// Clamp values between lo and hi
        #[inline(always)]
        pub fn clamp(self, lo: Self, hi: Self) -> Self {
            self.max(lo).min(hi)
        }

        /// Absolute value
        #[inline(always)]
        pub fn abs(self) -> Self {
            Self {
                lo: self.lo.abs(),
                hi: self.hi.abs(),
            }
        }

        /// Reduce: sum all lanes
        #[inline(always)]
        pub fn reduce_add(self) -> i16 {
            self.lo.reduce_add().wrapping_add(self.hi.reduce_add())
        }

        // ========== Bitcast ==========
        /// Reinterpret bits as `u16x16` (zero-cost).
        #[inline(always)]
        pub fn bitcast_u16x16(self) -> u16x16 {
            u16x16 {
                lo: self.lo.bitcast_u16x8(),
                hi: self.hi.bitcast_u16x8(),
            }
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

    impl Add for i16x16 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                lo: self.lo + rhs.lo,
                hi: self.hi + rhs.hi,
            }
        }
    }

    impl Sub for i16x16 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                lo: self.lo - rhs.lo,
                hi: self.hi - rhs.hi,
            }
        }
    }

    impl Mul for i16x16 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            Self {
                lo: self.lo * rhs.lo,
                hi: self.hi * rhs.hi,
            }
        }
    }

    impl core::ops::AddAssign for i16x16 {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl core::ops::SubAssign for i16x16 {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    impl core::ops::MulAssign for i16x16 {
        #[inline(always)]
        fn mul_assign(&mut self, rhs: Self) {
            *self = *self * rhs;
        }
    }

    /// Emulated 16-wide u16 vector using two u16x8 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct u16x16 {
        pub(crate) lo: u16x8,
        pub(crate) hi: u16x8,
    }

    impl u16x16 {
        pub const LANES: usize = 16;

        /// Load from array (token-gated)
        #[inline(always)]
        pub fn load(token: NeonToken, data: &[u16; 16]) -> Self {
            let lo_arr: &[u16; 8] = data[0..8].try_into().unwrap();
            let hi_arr: &[u16; 8] = data[8..16].try_into().unwrap();
            Self {
                lo: u16x8::load(token, lo_arr),
                hi: u16x8::load(token, hi_arr),
            }
        }

        /// Broadcast scalar to all lanes (token-gated)
        #[inline(always)]
        pub fn splat(token: NeonToken, v: u16) -> Self {
            Self {
                lo: u16x8::splat(token, v),
                hi: u16x8::splat(token, v),
            }
        }

        /// Zero vector (token-gated)
        #[inline(always)]
        pub fn zero(token: NeonToken) -> Self {
            Self {
                lo: u16x8::zero(token),
                hi: u16x8::zero(token),
            }
        }

        /// Create from array (token-gated)
        #[inline(always)]
        pub fn from_array(token: NeonToken, arr: [u16; 16]) -> Self {
            Self::load(token, &arr)
        }

        /// Store to array
        #[inline(always)]
        pub fn store(self, out: &mut [u16; 16]) {
            let (lo, hi) = out.split_at_mut(8);
            let lo_arr: &mut [u16; 8] = lo.try_into().unwrap();
            let hi_arr: &mut [u16; 8] = hi.try_into().unwrap();
            self.lo.store(lo_arr);
            self.hi.store(hi_arr);
        }

        /// Convert to array
        #[inline(always)]
        pub fn to_array(self) -> [u16; 16] {
            let mut out = [0u16; 16];
            self.store(&mut out);
            out
        }

        /// Element-wise minimum
        #[inline(always)]
        pub fn min(self, other: Self) -> Self {
            Self {
                lo: self.lo.min(other.lo),
                hi: self.hi.min(other.hi),
            }
        }

        /// Element-wise maximum
        #[inline(always)]
        pub fn max(self, other: Self) -> Self {
            Self {
                lo: self.lo.max(other.lo),
                hi: self.hi.max(other.hi),
            }
        }

        /// Clamp values between lo and hi
        #[inline(always)]
        pub fn clamp(self, lo: Self, hi: Self) -> Self {
            self.max(lo).min(hi)
        }

        /// Reduce: sum all lanes
        #[inline(always)]
        pub fn reduce_add(self) -> u16 {
            self.lo.reduce_add().wrapping_add(self.hi.reduce_add())
        }

        // ========== Bitcast ==========
        /// Reinterpret bits as `i16x16` (zero-cost).
        #[inline(always)]
        pub fn bitcast_i16x16(self) -> i16x16 {
            i16x16 {
                lo: self.lo.bitcast_i16x8(),
                hi: self.hi.bitcast_i16x8(),
            }
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

    impl Add for u16x16 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                lo: self.lo + rhs.lo,
                hi: self.hi + rhs.hi,
            }
        }
    }

    impl Sub for u16x16 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                lo: self.lo - rhs.lo,
                hi: self.hi - rhs.hi,
            }
        }
    }

    impl Mul for u16x16 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            Self {
                lo: self.lo * rhs.lo,
                hi: self.hi * rhs.hi,
            }
        }
    }

    impl core::ops::AddAssign for u16x16 {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl core::ops::SubAssign for u16x16 {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    impl core::ops::MulAssign for u16x16 {
        #[inline(always)]
        fn mul_assign(&mut self, rhs: Self) {
            *self = *self * rhs;
        }
    }

    /// Emulated 8-wide i32 vector using two i32x4 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct i32x8 {
        pub(crate) lo: i32x4,
        pub(crate) hi: i32x4,
    }

    impl i32x8 {
        pub const LANES: usize = 8;

        /// Load from array (token-gated)
        #[inline(always)]
        pub fn load(token: NeonToken, data: &[i32; 8]) -> Self {
            let lo_arr: &[i32; 4] = data[0..4].try_into().unwrap();
            let hi_arr: &[i32; 4] = data[4..8].try_into().unwrap();
            Self {
                lo: i32x4::load(token, lo_arr),
                hi: i32x4::load(token, hi_arr),
            }
        }

        /// Broadcast scalar to all lanes (token-gated)
        #[inline(always)]
        pub fn splat(token: NeonToken, v: i32) -> Self {
            Self {
                lo: i32x4::splat(token, v),
                hi: i32x4::splat(token, v),
            }
        }

        /// Zero vector (token-gated)
        #[inline(always)]
        pub fn zero(token: NeonToken) -> Self {
            Self {
                lo: i32x4::zero(token),
                hi: i32x4::zero(token),
            }
        }

        /// Create from array (token-gated)
        #[inline(always)]
        pub fn from_array(token: NeonToken, arr: [i32; 8]) -> Self {
            Self::load(token, &arr)
        }

        /// Store to array
        #[inline(always)]
        pub fn store(self, out: &mut [i32; 8]) {
            let (lo, hi) = out.split_at_mut(4);
            let lo_arr: &mut [i32; 4] = lo.try_into().unwrap();
            let hi_arr: &mut [i32; 4] = hi.try_into().unwrap();
            self.lo.store(lo_arr);
            self.hi.store(hi_arr);
        }

        /// Convert to array
        #[inline(always)]
        pub fn to_array(self) -> [i32; 8] {
            let mut out = [0i32; 8];
            self.store(&mut out);
            out
        }

        /// Element-wise minimum
        #[inline(always)]
        pub fn min(self, other: Self) -> Self {
            Self {
                lo: self.lo.min(other.lo),
                hi: self.hi.min(other.hi),
            }
        }

        /// Element-wise maximum
        #[inline(always)]
        pub fn max(self, other: Self) -> Self {
            Self {
                lo: self.lo.max(other.lo),
                hi: self.hi.max(other.hi),
            }
        }

        /// Clamp values between lo and hi
        #[inline(always)]
        pub fn clamp(self, lo: Self, hi: Self) -> Self {
            self.max(lo).min(hi)
        }

        /// Absolute value
        #[inline(always)]
        pub fn abs(self) -> Self {
            Self {
                lo: self.lo.abs(),
                hi: self.hi.abs(),
            }
        }

        /// Reduce: sum all lanes
        #[inline(always)]
        pub fn reduce_add(self) -> i32 {
            self.lo.reduce_add().wrapping_add(self.hi.reduce_add())
        }

        // ========== Bitcast ==========
        /// Reinterpret bits as `f32x8` (zero-cost).
        #[inline(always)]
        pub fn bitcast_f32x8(self) -> f32x8 {
            f32x8 {
                lo: self.lo.bitcast_f32x4(),
                hi: self.hi.bitcast_f32x4(),
            }
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
            u32x8 {
                lo: self.lo.bitcast_u32x4(),
                hi: self.hi.bitcast_u32x4(),
            }
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

    impl Add for i32x8 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                lo: self.lo + rhs.lo,
                hi: self.hi + rhs.hi,
            }
        }
    }

    impl Sub for i32x8 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                lo: self.lo - rhs.lo,
                hi: self.hi - rhs.hi,
            }
        }
    }

    impl Mul for i32x8 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            Self {
                lo: self.lo * rhs.lo,
                hi: self.hi * rhs.hi,
            }
        }
    }

    impl core::ops::AddAssign for i32x8 {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl core::ops::SubAssign for i32x8 {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    impl core::ops::MulAssign for i32x8 {
        #[inline(always)]
        fn mul_assign(&mut self, rhs: Self) {
            *self = *self * rhs;
        }
    }

    /// Emulated 8-wide u32 vector using two u32x4 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct u32x8 {
        pub(crate) lo: u32x4,
        pub(crate) hi: u32x4,
    }

    impl u32x8 {
        pub const LANES: usize = 8;

        /// Load from array (token-gated)
        #[inline(always)]
        pub fn load(token: NeonToken, data: &[u32; 8]) -> Self {
            let lo_arr: &[u32; 4] = data[0..4].try_into().unwrap();
            let hi_arr: &[u32; 4] = data[4..8].try_into().unwrap();
            Self {
                lo: u32x4::load(token, lo_arr),
                hi: u32x4::load(token, hi_arr),
            }
        }

        /// Broadcast scalar to all lanes (token-gated)
        #[inline(always)]
        pub fn splat(token: NeonToken, v: u32) -> Self {
            Self {
                lo: u32x4::splat(token, v),
                hi: u32x4::splat(token, v),
            }
        }

        /// Zero vector (token-gated)
        #[inline(always)]
        pub fn zero(token: NeonToken) -> Self {
            Self {
                lo: u32x4::zero(token),
                hi: u32x4::zero(token),
            }
        }

        /// Create from array (token-gated)
        #[inline(always)]
        pub fn from_array(token: NeonToken, arr: [u32; 8]) -> Self {
            Self::load(token, &arr)
        }

        /// Store to array
        #[inline(always)]
        pub fn store(self, out: &mut [u32; 8]) {
            let (lo, hi) = out.split_at_mut(4);
            let lo_arr: &mut [u32; 4] = lo.try_into().unwrap();
            let hi_arr: &mut [u32; 4] = hi.try_into().unwrap();
            self.lo.store(lo_arr);
            self.hi.store(hi_arr);
        }

        /// Convert to array
        #[inline(always)]
        pub fn to_array(self) -> [u32; 8] {
            let mut out = [0u32; 8];
            self.store(&mut out);
            out
        }

        /// Element-wise minimum
        #[inline(always)]
        pub fn min(self, other: Self) -> Self {
            Self {
                lo: self.lo.min(other.lo),
                hi: self.hi.min(other.hi),
            }
        }

        /// Element-wise maximum
        #[inline(always)]
        pub fn max(self, other: Self) -> Self {
            Self {
                lo: self.lo.max(other.lo),
                hi: self.hi.max(other.hi),
            }
        }

        /// Clamp values between lo and hi
        #[inline(always)]
        pub fn clamp(self, lo: Self, hi: Self) -> Self {
            self.max(lo).min(hi)
        }

        /// Reduce: sum all lanes
        #[inline(always)]
        pub fn reduce_add(self) -> u32 {
            self.lo.reduce_add().wrapping_add(self.hi.reduce_add())
        }

        // ========== Bitcast ==========
        /// Reinterpret bits as `f32x8` (zero-cost).
        #[inline(always)]
        pub fn bitcast_f32x8(self) -> f32x8 {
            f32x8 {
                lo: self.lo.bitcast_f32x4(),
                hi: self.hi.bitcast_f32x4(),
            }
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
            i32x8 {
                lo: self.lo.bitcast_i32x4(),
                hi: self.hi.bitcast_i32x4(),
            }
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

    impl Add for u32x8 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                lo: self.lo + rhs.lo,
                hi: self.hi + rhs.hi,
            }
        }
    }

    impl Sub for u32x8 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                lo: self.lo - rhs.lo,
                hi: self.hi - rhs.hi,
            }
        }
    }

    impl Mul for u32x8 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            Self {
                lo: self.lo * rhs.lo,
                hi: self.hi * rhs.hi,
            }
        }
    }

    impl core::ops::AddAssign for u32x8 {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl core::ops::SubAssign for u32x8 {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    impl core::ops::MulAssign for u32x8 {
        #[inline(always)]
        fn mul_assign(&mut self, rhs: Self) {
            *self = *self * rhs;
        }
    }

    /// Emulated 4-wide i64 vector using two i64x2 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct i64x4 {
        pub(crate) lo: i64x2,
        pub(crate) hi: i64x2,
    }

    impl i64x4 {
        pub const LANES: usize = 4;

        /// Load from array (token-gated)
        #[inline(always)]
        pub fn load(token: NeonToken, data: &[i64; 4]) -> Self {
            let lo_arr: &[i64; 2] = data[0..2].try_into().unwrap();
            let hi_arr: &[i64; 2] = data[2..4].try_into().unwrap();
            Self {
                lo: i64x2::load(token, lo_arr),
                hi: i64x2::load(token, hi_arr),
            }
        }

        /// Broadcast scalar to all lanes (token-gated)
        #[inline(always)]
        pub fn splat(token: NeonToken, v: i64) -> Self {
            Self {
                lo: i64x2::splat(token, v),
                hi: i64x2::splat(token, v),
            }
        }

        /// Zero vector (token-gated)
        #[inline(always)]
        pub fn zero(token: NeonToken) -> Self {
            Self {
                lo: i64x2::zero(token),
                hi: i64x2::zero(token),
            }
        }

        /// Create from array (token-gated)
        #[inline(always)]
        pub fn from_array(token: NeonToken, arr: [i64; 4]) -> Self {
            Self::load(token, &arr)
        }

        /// Store to array
        #[inline(always)]
        pub fn store(self, out: &mut [i64; 4]) {
            let (lo, hi) = out.split_at_mut(2);
            let lo_arr: &mut [i64; 2] = lo.try_into().unwrap();
            let hi_arr: &mut [i64; 2] = hi.try_into().unwrap();
            self.lo.store(lo_arr);
            self.hi.store(hi_arr);
        }

        /// Convert to array
        #[inline(always)]
        pub fn to_array(self) -> [i64; 4] {
            let mut out = [0i64; 4];
            self.store(&mut out);
            out
        }

        /// Reduce: sum all lanes
        #[inline(always)]
        pub fn reduce_add(self) -> i64 {
            self.lo.reduce_add().wrapping_add(self.hi.reduce_add())
        }

        // ========== Bitcast ==========
        /// Reinterpret bits as `f64x4` (zero-cost).
        #[inline(always)]
        pub fn bitcast_f64x4(self) -> f64x4 {
            f64x4 {
                lo: self.lo.bitcast_f64x2(),
                hi: self.hi.bitcast_f64x2(),
            }
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
            u64x4 {
                lo: self.lo.bitcast_u64x2(),
                hi: self.hi.bitcast_u64x2(),
            }
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

    impl Add for i64x4 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                lo: self.lo + rhs.lo,
                hi: self.hi + rhs.hi,
            }
        }
    }

    impl Sub for i64x4 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                lo: self.lo - rhs.lo,
                hi: self.hi - rhs.hi,
            }
        }
    }

    impl core::ops::AddAssign for i64x4 {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl core::ops::SubAssign for i64x4 {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    /// Emulated 4-wide u64 vector using two u64x2 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct u64x4 {
        pub(crate) lo: u64x2,
        pub(crate) hi: u64x2,
    }

    impl u64x4 {
        pub const LANES: usize = 4;

        /// Load from array (token-gated)
        #[inline(always)]
        pub fn load(token: NeonToken, data: &[u64; 4]) -> Self {
            let lo_arr: &[u64; 2] = data[0..2].try_into().unwrap();
            let hi_arr: &[u64; 2] = data[2..4].try_into().unwrap();
            Self {
                lo: u64x2::load(token, lo_arr),
                hi: u64x2::load(token, hi_arr),
            }
        }

        /// Broadcast scalar to all lanes (token-gated)
        #[inline(always)]
        pub fn splat(token: NeonToken, v: u64) -> Self {
            Self {
                lo: u64x2::splat(token, v),
                hi: u64x2::splat(token, v),
            }
        }

        /// Zero vector (token-gated)
        #[inline(always)]
        pub fn zero(token: NeonToken) -> Self {
            Self {
                lo: u64x2::zero(token),
                hi: u64x2::zero(token),
            }
        }

        /// Create from array (token-gated)
        #[inline(always)]
        pub fn from_array(token: NeonToken, arr: [u64; 4]) -> Self {
            Self::load(token, &arr)
        }

        /// Store to array
        #[inline(always)]
        pub fn store(self, out: &mut [u64; 4]) {
            let (lo, hi) = out.split_at_mut(2);
            let lo_arr: &mut [u64; 2] = lo.try_into().unwrap();
            let hi_arr: &mut [u64; 2] = hi.try_into().unwrap();
            self.lo.store(lo_arr);
            self.hi.store(hi_arr);
        }

        /// Convert to array
        #[inline(always)]
        pub fn to_array(self) -> [u64; 4] {
            let mut out = [0u64; 4];
            self.store(&mut out);
            out
        }

        /// Reduce: sum all lanes
        #[inline(always)]
        pub fn reduce_add(self) -> u64 {
            self.lo.reduce_add().wrapping_add(self.hi.reduce_add())
        }

        // ========== Bitcast ==========
        /// Reinterpret bits as `f64x4` (zero-cost).
        #[inline(always)]
        pub fn bitcast_f64x4(self) -> f64x4 {
            f64x4 {
                lo: self.lo.bitcast_f64x2(),
                hi: self.hi.bitcast_f64x2(),
            }
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
            i64x4 {
                lo: self.lo.bitcast_i64x2(),
                hi: self.hi.bitcast_i64x2(),
            }
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

    impl Add for u64x4 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                lo: self.lo + rhs.lo,
                hi: self.hi + rhs.hi,
            }
        }
    }

    impl Sub for u64x4 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                lo: self.lo - rhs.lo,
                hi: self.hi - rhs.hi,
            }
        }
    }

    impl core::ops::AddAssign for u64x4 {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl core::ops::SubAssign for u64x4 {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    // Width-aliased type names
    pub type f32xN = f32x8;
    pub type f64xN = f64x4;
    pub type i8xN = i8x32;
    pub type u8xN = u8x32;
    pub type i16xN = i16x16;
    pub type u16xN = u16x16;
    pub type i32xN = i32x8;
    pub type u32xN = u32x8;
    pub type i64xN = i64x4;
    pub type u64xN = u64x4;

    /// Number of f32 lanes
    pub const LANES_F32: usize = 8;
    /// Number of f64 lanes
    pub const LANES_F64: usize = 4;
    /// Number of i32/u32 lanes
    pub const LANES_32: usize = 8;
    /// Number of i16/u16 lanes
    pub const LANES_16: usize = 16;
    /// Number of i8/u8 lanes
    pub const LANES_8: usize = 32;

    /// Token type for this polyfill level
    pub type Token = archmage::NeonToken;
}

#[cfg(target_arch = "wasm32")]
pub mod wasm128 {
    //! Polyfilled 256-bit types using WASM SIMD128 (128-bit) operations.

    use crate::simd::generated::wasm::w128::{
        f32x4, f64x2, i8x16, i16x8, i32x4, i64x2, u8x16, u16x8, u32x4, u64x2,
    };
    use archmage::Wasm128Token;
    use core::ops::{Add, Div, Mul, Neg, Sub};

    /// Emulated 8-wide f32 vector using two f32x4 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct f32x8 {
        pub(crate) lo: f32x4,
        pub(crate) hi: f32x4,
    }

    impl f32x8 {
        pub const LANES: usize = 8;

        /// Load from array (token-gated)
        #[inline(always)]
        pub fn load(token: Wasm128Token, data: &[f32; 8]) -> Self {
            let lo_arr: &[f32; 4] = data[0..4].try_into().unwrap();
            let hi_arr: &[f32; 4] = data[4..8].try_into().unwrap();
            Self {
                lo: f32x4::load(token, lo_arr),
                hi: f32x4::load(token, hi_arr),
            }
        }

        /// Broadcast scalar to all lanes (token-gated)
        #[inline(always)]
        pub fn splat(token: Wasm128Token, v: f32) -> Self {
            Self {
                lo: f32x4::splat(token, v),
                hi: f32x4::splat(token, v),
            }
        }

        /// Zero vector (token-gated)
        #[inline(always)]
        pub fn zero(token: Wasm128Token) -> Self {
            Self {
                lo: f32x4::zero(token),
                hi: f32x4::zero(token),
            }
        }

        /// Create from array (token-gated)
        #[inline(always)]
        pub fn from_array(token: Wasm128Token, arr: [f32; 8]) -> Self {
            Self::load(token, &arr)
        }

        /// Store to array
        #[inline(always)]
        pub fn store(self, out: &mut [f32; 8]) {
            let (lo, hi) = out.split_at_mut(4);
            let lo_arr: &mut [f32; 4] = lo.try_into().unwrap();
            let hi_arr: &mut [f32; 4] = hi.try_into().unwrap();
            self.lo.store(lo_arr);
            self.hi.store(hi_arr);
        }

        /// Convert to array
        #[inline(always)]
        pub fn to_array(self) -> [f32; 8] {
            let mut out = [0.0f32; 8];
            self.store(&mut out);
            out
        }

        /// Element-wise minimum
        #[inline(always)]
        pub fn min(self, other: Self) -> Self {
            Self {
                lo: self.lo.min(other.lo),
                hi: self.hi.min(other.hi),
            }
        }

        /// Element-wise maximum
        #[inline(always)]
        pub fn max(self, other: Self) -> Self {
            Self {
                lo: self.lo.max(other.lo),
                hi: self.hi.max(other.hi),
            }
        }

        /// Clamp values between lo and hi
        #[inline(always)]
        pub fn clamp(self, lo: Self, hi: Self) -> Self {
            self.max(lo).min(hi)
        }

        /// Square root
        #[inline(always)]
        pub fn sqrt(self) -> Self {
            Self {
                lo: self.lo.sqrt(),
                hi: self.hi.sqrt(),
            }
        }

        /// Floor
        #[inline(always)]
        pub fn floor(self) -> Self {
            Self {
                lo: self.lo.floor(),
                hi: self.hi.floor(),
            }
        }

        /// Ceil
        #[inline(always)]
        pub fn ceil(self) -> Self {
            Self {
                lo: self.lo.ceil(),
                hi: self.hi.ceil(),
            }
        }

        /// Round to nearest
        #[inline(always)]
        pub fn round(self) -> Self {
            Self {
                lo: self.lo.round(),
                hi: self.hi.round(),
            }
        }

        /// Fused multiply-add: self * a + b
        #[inline(always)]
        pub fn mul_add(self, a: Self, b: Self) -> Self {
            Self {
                lo: self.lo.mul_add(a.lo, b.lo),
                hi: self.hi.mul_add(a.hi, b.hi),
            }
        }

        /// Absolute value
        #[inline(always)]
        pub fn abs(self) -> Self {
            Self {
                lo: self.lo.abs(),
                hi: self.hi.abs(),
            }
        }

        /// Reduce: sum all lanes
        #[inline(always)]
        pub fn reduce_add(self) -> f32 {
            self.lo.reduce_add() + self.hi.reduce_add()
        }

        /// Reduce: max of all lanes
        #[inline(always)]
        pub fn reduce_max(self) -> f32 {
            self.lo.reduce_max().max(self.hi.reduce_max())
        }

        /// Reduce: min of all lanes
        #[inline(always)]
        pub fn reduce_min(self) -> f32 {
            self.lo.reduce_min().min(self.hi.reduce_min())
        }

        // ========== Bitcast ==========
        /// Reinterpret bits as `i32x8` (zero-cost).
        #[inline(always)]
        pub fn bitcast_i32x8(self) -> i32x8 {
            i32x8 {
                lo: self.lo.bitcast_i32x4(),
                hi: self.hi.bitcast_i32x4(),
            }
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
            u32x8 {
                lo: self.lo.bitcast_u32x4(),
                hi: self.hi.bitcast_u32x4(),
            }
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

    impl Add for f32x8 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                lo: self.lo + rhs.lo,
                hi: self.hi + rhs.hi,
            }
        }
    }

    impl Sub for f32x8 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                lo: self.lo - rhs.lo,
                hi: self.hi - rhs.hi,
            }
        }
    }

    impl Mul for f32x8 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            Self {
                lo: self.lo * rhs.lo,
                hi: self.hi * rhs.hi,
            }
        }
    }

    impl Div for f32x8 {
        type Output = Self;
        #[inline(always)]
        fn div(self, rhs: Self) -> Self {
            Self {
                lo: self.lo / rhs.lo,
                hi: self.hi / rhs.hi,
            }
        }
    }

    impl Neg for f32x8 {
        type Output = Self;
        #[inline(always)]
        fn neg(self) -> Self {
            Self {
                lo: -self.lo,
                hi: -self.hi,
            }
        }
    }

    impl core::ops::AddAssign for f32x8 {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl core::ops::SubAssign for f32x8 {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    impl core::ops::MulAssign for f32x8 {
        #[inline(always)]
        fn mul_assign(&mut self, rhs: Self) {
            *self = *self * rhs;
        }
    }

    impl core::ops::DivAssign for f32x8 {
        #[inline(always)]
        fn div_assign(&mut self, rhs: Self) {
            *self = *self / rhs;
        }
    }

    /// Emulated 4-wide f64 vector using two f64x2 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct f64x4 {
        pub(crate) lo: f64x2,
        pub(crate) hi: f64x2,
    }

    impl f64x4 {
        pub const LANES: usize = 4;

        /// Load from array (token-gated)
        #[inline(always)]
        pub fn load(token: Wasm128Token, data: &[f64; 4]) -> Self {
            let lo_arr: &[f64; 2] = data[0..2].try_into().unwrap();
            let hi_arr: &[f64; 2] = data[2..4].try_into().unwrap();
            Self {
                lo: f64x2::load(token, lo_arr),
                hi: f64x2::load(token, hi_arr),
            }
        }

        /// Broadcast scalar to all lanes (token-gated)
        #[inline(always)]
        pub fn splat(token: Wasm128Token, v: f64) -> Self {
            Self {
                lo: f64x2::splat(token, v),
                hi: f64x2::splat(token, v),
            }
        }

        /// Zero vector (token-gated)
        #[inline(always)]
        pub fn zero(token: Wasm128Token) -> Self {
            Self {
                lo: f64x2::zero(token),
                hi: f64x2::zero(token),
            }
        }

        /// Create from array (token-gated)
        #[inline(always)]
        pub fn from_array(token: Wasm128Token, arr: [f64; 4]) -> Self {
            Self::load(token, &arr)
        }

        /// Store to array
        #[inline(always)]
        pub fn store(self, out: &mut [f64; 4]) {
            let (lo, hi) = out.split_at_mut(2);
            let lo_arr: &mut [f64; 2] = lo.try_into().unwrap();
            let hi_arr: &mut [f64; 2] = hi.try_into().unwrap();
            self.lo.store(lo_arr);
            self.hi.store(hi_arr);
        }

        /// Convert to array
        #[inline(always)]
        pub fn to_array(self) -> [f64; 4] {
            let mut out = [0.0f64; 4];
            self.store(&mut out);
            out
        }

        /// Element-wise minimum
        #[inline(always)]
        pub fn min(self, other: Self) -> Self {
            Self {
                lo: self.lo.min(other.lo),
                hi: self.hi.min(other.hi),
            }
        }

        /// Element-wise maximum
        #[inline(always)]
        pub fn max(self, other: Self) -> Self {
            Self {
                lo: self.lo.max(other.lo),
                hi: self.hi.max(other.hi),
            }
        }

        /// Clamp values between lo and hi
        #[inline(always)]
        pub fn clamp(self, lo: Self, hi: Self) -> Self {
            self.max(lo).min(hi)
        }

        /// Square root
        #[inline(always)]
        pub fn sqrt(self) -> Self {
            Self {
                lo: self.lo.sqrt(),
                hi: self.hi.sqrt(),
            }
        }

        /// Floor
        #[inline(always)]
        pub fn floor(self) -> Self {
            Self {
                lo: self.lo.floor(),
                hi: self.hi.floor(),
            }
        }

        /// Ceil
        #[inline(always)]
        pub fn ceil(self) -> Self {
            Self {
                lo: self.lo.ceil(),
                hi: self.hi.ceil(),
            }
        }

        /// Round to nearest
        #[inline(always)]
        pub fn round(self) -> Self {
            Self {
                lo: self.lo.round(),
                hi: self.hi.round(),
            }
        }

        /// Fused multiply-add: self * a + b
        #[inline(always)]
        pub fn mul_add(self, a: Self, b: Self) -> Self {
            Self {
                lo: self.lo.mul_add(a.lo, b.lo),
                hi: self.hi.mul_add(a.hi, b.hi),
            }
        }

        /// Absolute value
        #[inline(always)]
        pub fn abs(self) -> Self {
            Self {
                lo: self.lo.abs(),
                hi: self.hi.abs(),
            }
        }

        /// Reduce: sum all lanes
        #[inline(always)]
        pub fn reduce_add(self) -> f64 {
            self.lo.reduce_add() + self.hi.reduce_add()
        }

        /// Reduce: max of all lanes
        #[inline(always)]
        pub fn reduce_max(self) -> f64 {
            self.lo.reduce_max().max(self.hi.reduce_max())
        }

        /// Reduce: min of all lanes
        #[inline(always)]
        pub fn reduce_min(self) -> f64 {
            self.lo.reduce_min().min(self.hi.reduce_min())
        }

        // ========== Bitcast ==========
        /// Reinterpret bits as `i64x4` (zero-cost).
        #[inline(always)]
        pub fn bitcast_i64x4(self) -> i64x4 {
            i64x4 {
                lo: self.lo.bitcast_i64x2(),
                hi: self.hi.bitcast_i64x2(),
            }
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
            u64x4 {
                lo: self.lo.bitcast_u64x2(),
                hi: self.hi.bitcast_u64x2(),
            }
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

    impl Add for f64x4 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                lo: self.lo + rhs.lo,
                hi: self.hi + rhs.hi,
            }
        }
    }

    impl Sub for f64x4 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                lo: self.lo - rhs.lo,
                hi: self.hi - rhs.hi,
            }
        }
    }

    impl Mul for f64x4 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            Self {
                lo: self.lo * rhs.lo,
                hi: self.hi * rhs.hi,
            }
        }
    }

    impl Div for f64x4 {
        type Output = Self;
        #[inline(always)]
        fn div(self, rhs: Self) -> Self {
            Self {
                lo: self.lo / rhs.lo,
                hi: self.hi / rhs.hi,
            }
        }
    }

    impl Neg for f64x4 {
        type Output = Self;
        #[inline(always)]
        fn neg(self) -> Self {
            Self {
                lo: -self.lo,
                hi: -self.hi,
            }
        }
    }

    impl core::ops::AddAssign for f64x4 {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl core::ops::SubAssign for f64x4 {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    impl core::ops::MulAssign for f64x4 {
        #[inline(always)]
        fn mul_assign(&mut self, rhs: Self) {
            *self = *self * rhs;
        }
    }

    impl core::ops::DivAssign for f64x4 {
        #[inline(always)]
        fn div_assign(&mut self, rhs: Self) {
            *self = *self / rhs;
        }
    }

    /// Emulated 32-wide i8 vector using two i8x16 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct i8x32 {
        pub(crate) lo: i8x16,
        pub(crate) hi: i8x16,
    }

    impl i8x32 {
        pub const LANES: usize = 32;

        /// Load from array (token-gated)
        #[inline(always)]
        pub fn load(token: Wasm128Token, data: &[i8; 32]) -> Self {
            let lo_arr: &[i8; 16] = data[0..16].try_into().unwrap();
            let hi_arr: &[i8; 16] = data[16..32].try_into().unwrap();
            Self {
                lo: i8x16::load(token, lo_arr),
                hi: i8x16::load(token, hi_arr),
            }
        }

        /// Broadcast scalar to all lanes (token-gated)
        #[inline(always)]
        pub fn splat(token: Wasm128Token, v: i8) -> Self {
            Self {
                lo: i8x16::splat(token, v),
                hi: i8x16::splat(token, v),
            }
        }

        /// Zero vector (token-gated)
        #[inline(always)]
        pub fn zero(token: Wasm128Token) -> Self {
            Self {
                lo: i8x16::zero(token),
                hi: i8x16::zero(token),
            }
        }

        /// Create from array (token-gated)
        #[inline(always)]
        pub fn from_array(token: Wasm128Token, arr: [i8; 32]) -> Self {
            Self::load(token, &arr)
        }

        /// Store to array
        #[inline(always)]
        pub fn store(self, out: &mut [i8; 32]) {
            let (lo, hi) = out.split_at_mut(16);
            let lo_arr: &mut [i8; 16] = lo.try_into().unwrap();
            let hi_arr: &mut [i8; 16] = hi.try_into().unwrap();
            self.lo.store(lo_arr);
            self.hi.store(hi_arr);
        }

        /// Convert to array
        #[inline(always)]
        pub fn to_array(self) -> [i8; 32] {
            let mut out = [0i8; 32];
            self.store(&mut out);
            out
        }

        /// Element-wise minimum
        #[inline(always)]
        pub fn min(self, other: Self) -> Self {
            Self {
                lo: self.lo.min(other.lo),
                hi: self.hi.min(other.hi),
            }
        }

        /// Element-wise maximum
        #[inline(always)]
        pub fn max(self, other: Self) -> Self {
            Self {
                lo: self.lo.max(other.lo),
                hi: self.hi.max(other.hi),
            }
        }

        /// Clamp values between lo and hi
        #[inline(always)]
        pub fn clamp(self, lo: Self, hi: Self) -> Self {
            self.max(lo).min(hi)
        }

        /// Absolute value
        #[inline(always)]
        pub fn abs(self) -> Self {
            Self {
                lo: self.lo.abs(),
                hi: self.hi.abs(),
            }
        }

        /// Reduce: sum all lanes
        #[inline(always)]
        pub fn reduce_add(self) -> i8 {
            self.lo.reduce_add().wrapping_add(self.hi.reduce_add())
        }

        // ========== Bitcast ==========
        /// Reinterpret bits as `u8x32` (zero-cost).
        #[inline(always)]
        pub fn bitcast_u8x32(self) -> u8x32 {
            u8x32 {
                lo: self.lo.bitcast_u8x16(),
                hi: self.hi.bitcast_u8x16(),
            }
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

    impl Add for i8x32 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                lo: self.lo + rhs.lo,
                hi: self.hi + rhs.hi,
            }
        }
    }

    impl Sub for i8x32 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                lo: self.lo - rhs.lo,
                hi: self.hi - rhs.hi,
            }
        }
    }

    impl core::ops::AddAssign for i8x32 {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl core::ops::SubAssign for i8x32 {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    /// Emulated 32-wide u8 vector using two u8x16 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct u8x32 {
        pub(crate) lo: u8x16,
        pub(crate) hi: u8x16,
    }

    impl u8x32 {
        pub const LANES: usize = 32;

        /// Load from array (token-gated)
        #[inline(always)]
        pub fn load(token: Wasm128Token, data: &[u8; 32]) -> Self {
            let lo_arr: &[u8; 16] = data[0..16].try_into().unwrap();
            let hi_arr: &[u8; 16] = data[16..32].try_into().unwrap();
            Self {
                lo: u8x16::load(token, lo_arr),
                hi: u8x16::load(token, hi_arr),
            }
        }

        /// Broadcast scalar to all lanes (token-gated)
        #[inline(always)]
        pub fn splat(token: Wasm128Token, v: u8) -> Self {
            Self {
                lo: u8x16::splat(token, v),
                hi: u8x16::splat(token, v),
            }
        }

        /// Zero vector (token-gated)
        #[inline(always)]
        pub fn zero(token: Wasm128Token) -> Self {
            Self {
                lo: u8x16::zero(token),
                hi: u8x16::zero(token),
            }
        }

        /// Create from array (token-gated)
        #[inline(always)]
        pub fn from_array(token: Wasm128Token, arr: [u8; 32]) -> Self {
            Self::load(token, &arr)
        }

        /// Store to array
        #[inline(always)]
        pub fn store(self, out: &mut [u8; 32]) {
            let (lo, hi) = out.split_at_mut(16);
            let lo_arr: &mut [u8; 16] = lo.try_into().unwrap();
            let hi_arr: &mut [u8; 16] = hi.try_into().unwrap();
            self.lo.store(lo_arr);
            self.hi.store(hi_arr);
        }

        /// Convert to array
        #[inline(always)]
        pub fn to_array(self) -> [u8; 32] {
            let mut out = [0u8; 32];
            self.store(&mut out);
            out
        }

        /// Element-wise minimum
        #[inline(always)]
        pub fn min(self, other: Self) -> Self {
            Self {
                lo: self.lo.min(other.lo),
                hi: self.hi.min(other.hi),
            }
        }

        /// Element-wise maximum
        #[inline(always)]
        pub fn max(self, other: Self) -> Self {
            Self {
                lo: self.lo.max(other.lo),
                hi: self.hi.max(other.hi),
            }
        }

        /// Clamp values between lo and hi
        #[inline(always)]
        pub fn clamp(self, lo: Self, hi: Self) -> Self {
            self.max(lo).min(hi)
        }

        /// Reduce: sum all lanes
        #[inline(always)]
        pub fn reduce_add(self) -> u8 {
            self.lo.reduce_add().wrapping_add(self.hi.reduce_add())
        }

        // ========== Bitcast ==========
        /// Reinterpret bits as `i8x32` (zero-cost).
        #[inline(always)]
        pub fn bitcast_i8x32(self) -> i8x32 {
            i8x32 {
                lo: self.lo.bitcast_i8x16(),
                hi: self.hi.bitcast_i8x16(),
            }
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

    impl Add for u8x32 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                lo: self.lo + rhs.lo,
                hi: self.hi + rhs.hi,
            }
        }
    }

    impl Sub for u8x32 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                lo: self.lo - rhs.lo,
                hi: self.hi - rhs.hi,
            }
        }
    }

    impl core::ops::AddAssign for u8x32 {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl core::ops::SubAssign for u8x32 {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    /// Emulated 16-wide i16 vector using two i16x8 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct i16x16 {
        pub(crate) lo: i16x8,
        pub(crate) hi: i16x8,
    }

    impl i16x16 {
        pub const LANES: usize = 16;

        /// Load from array (token-gated)
        #[inline(always)]
        pub fn load(token: Wasm128Token, data: &[i16; 16]) -> Self {
            let lo_arr: &[i16; 8] = data[0..8].try_into().unwrap();
            let hi_arr: &[i16; 8] = data[8..16].try_into().unwrap();
            Self {
                lo: i16x8::load(token, lo_arr),
                hi: i16x8::load(token, hi_arr),
            }
        }

        /// Broadcast scalar to all lanes (token-gated)
        #[inline(always)]
        pub fn splat(token: Wasm128Token, v: i16) -> Self {
            Self {
                lo: i16x8::splat(token, v),
                hi: i16x8::splat(token, v),
            }
        }

        /// Zero vector (token-gated)
        #[inline(always)]
        pub fn zero(token: Wasm128Token) -> Self {
            Self {
                lo: i16x8::zero(token),
                hi: i16x8::zero(token),
            }
        }

        /// Create from array (token-gated)
        #[inline(always)]
        pub fn from_array(token: Wasm128Token, arr: [i16; 16]) -> Self {
            Self::load(token, &arr)
        }

        /// Store to array
        #[inline(always)]
        pub fn store(self, out: &mut [i16; 16]) {
            let (lo, hi) = out.split_at_mut(8);
            let lo_arr: &mut [i16; 8] = lo.try_into().unwrap();
            let hi_arr: &mut [i16; 8] = hi.try_into().unwrap();
            self.lo.store(lo_arr);
            self.hi.store(hi_arr);
        }

        /// Convert to array
        #[inline(always)]
        pub fn to_array(self) -> [i16; 16] {
            let mut out = [0i16; 16];
            self.store(&mut out);
            out
        }

        /// Element-wise minimum
        #[inline(always)]
        pub fn min(self, other: Self) -> Self {
            Self {
                lo: self.lo.min(other.lo),
                hi: self.hi.min(other.hi),
            }
        }

        /// Element-wise maximum
        #[inline(always)]
        pub fn max(self, other: Self) -> Self {
            Self {
                lo: self.lo.max(other.lo),
                hi: self.hi.max(other.hi),
            }
        }

        /// Clamp values between lo and hi
        #[inline(always)]
        pub fn clamp(self, lo: Self, hi: Self) -> Self {
            self.max(lo).min(hi)
        }

        /// Absolute value
        #[inline(always)]
        pub fn abs(self) -> Self {
            Self {
                lo: self.lo.abs(),
                hi: self.hi.abs(),
            }
        }

        /// Reduce: sum all lanes
        #[inline(always)]
        pub fn reduce_add(self) -> i16 {
            self.lo.reduce_add().wrapping_add(self.hi.reduce_add())
        }

        // ========== Bitcast ==========
        /// Reinterpret bits as `u16x16` (zero-cost).
        #[inline(always)]
        pub fn bitcast_u16x16(self) -> u16x16 {
            u16x16 {
                lo: self.lo.bitcast_u16x8(),
                hi: self.hi.bitcast_u16x8(),
            }
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

    impl Add for i16x16 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                lo: self.lo + rhs.lo,
                hi: self.hi + rhs.hi,
            }
        }
    }

    impl Sub for i16x16 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                lo: self.lo - rhs.lo,
                hi: self.hi - rhs.hi,
            }
        }
    }

    impl Mul for i16x16 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            Self {
                lo: self.lo * rhs.lo,
                hi: self.hi * rhs.hi,
            }
        }
    }

    impl core::ops::AddAssign for i16x16 {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl core::ops::SubAssign for i16x16 {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    impl core::ops::MulAssign for i16x16 {
        #[inline(always)]
        fn mul_assign(&mut self, rhs: Self) {
            *self = *self * rhs;
        }
    }

    /// Emulated 16-wide u16 vector using two u16x8 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct u16x16 {
        pub(crate) lo: u16x8,
        pub(crate) hi: u16x8,
    }

    impl u16x16 {
        pub const LANES: usize = 16;

        /// Load from array (token-gated)
        #[inline(always)]
        pub fn load(token: Wasm128Token, data: &[u16; 16]) -> Self {
            let lo_arr: &[u16; 8] = data[0..8].try_into().unwrap();
            let hi_arr: &[u16; 8] = data[8..16].try_into().unwrap();
            Self {
                lo: u16x8::load(token, lo_arr),
                hi: u16x8::load(token, hi_arr),
            }
        }

        /// Broadcast scalar to all lanes (token-gated)
        #[inline(always)]
        pub fn splat(token: Wasm128Token, v: u16) -> Self {
            Self {
                lo: u16x8::splat(token, v),
                hi: u16x8::splat(token, v),
            }
        }

        /// Zero vector (token-gated)
        #[inline(always)]
        pub fn zero(token: Wasm128Token) -> Self {
            Self {
                lo: u16x8::zero(token),
                hi: u16x8::zero(token),
            }
        }

        /// Create from array (token-gated)
        #[inline(always)]
        pub fn from_array(token: Wasm128Token, arr: [u16; 16]) -> Self {
            Self::load(token, &arr)
        }

        /// Store to array
        #[inline(always)]
        pub fn store(self, out: &mut [u16; 16]) {
            let (lo, hi) = out.split_at_mut(8);
            let lo_arr: &mut [u16; 8] = lo.try_into().unwrap();
            let hi_arr: &mut [u16; 8] = hi.try_into().unwrap();
            self.lo.store(lo_arr);
            self.hi.store(hi_arr);
        }

        /// Convert to array
        #[inline(always)]
        pub fn to_array(self) -> [u16; 16] {
            let mut out = [0u16; 16];
            self.store(&mut out);
            out
        }

        /// Element-wise minimum
        #[inline(always)]
        pub fn min(self, other: Self) -> Self {
            Self {
                lo: self.lo.min(other.lo),
                hi: self.hi.min(other.hi),
            }
        }

        /// Element-wise maximum
        #[inline(always)]
        pub fn max(self, other: Self) -> Self {
            Self {
                lo: self.lo.max(other.lo),
                hi: self.hi.max(other.hi),
            }
        }

        /// Clamp values between lo and hi
        #[inline(always)]
        pub fn clamp(self, lo: Self, hi: Self) -> Self {
            self.max(lo).min(hi)
        }

        /// Reduce: sum all lanes
        #[inline(always)]
        pub fn reduce_add(self) -> u16 {
            self.lo.reduce_add().wrapping_add(self.hi.reduce_add())
        }

        // ========== Bitcast ==========
        /// Reinterpret bits as `i16x16` (zero-cost).
        #[inline(always)]
        pub fn bitcast_i16x16(self) -> i16x16 {
            i16x16 {
                lo: self.lo.bitcast_i16x8(),
                hi: self.hi.bitcast_i16x8(),
            }
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

    impl Add for u16x16 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                lo: self.lo + rhs.lo,
                hi: self.hi + rhs.hi,
            }
        }
    }

    impl Sub for u16x16 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                lo: self.lo - rhs.lo,
                hi: self.hi - rhs.hi,
            }
        }
    }

    impl Mul for u16x16 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            Self {
                lo: self.lo * rhs.lo,
                hi: self.hi * rhs.hi,
            }
        }
    }

    impl core::ops::AddAssign for u16x16 {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl core::ops::SubAssign for u16x16 {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    impl core::ops::MulAssign for u16x16 {
        #[inline(always)]
        fn mul_assign(&mut self, rhs: Self) {
            *self = *self * rhs;
        }
    }

    /// Emulated 8-wide i32 vector using two i32x4 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct i32x8 {
        pub(crate) lo: i32x4,
        pub(crate) hi: i32x4,
    }

    impl i32x8 {
        pub const LANES: usize = 8;

        /// Load from array (token-gated)
        #[inline(always)]
        pub fn load(token: Wasm128Token, data: &[i32; 8]) -> Self {
            let lo_arr: &[i32; 4] = data[0..4].try_into().unwrap();
            let hi_arr: &[i32; 4] = data[4..8].try_into().unwrap();
            Self {
                lo: i32x4::load(token, lo_arr),
                hi: i32x4::load(token, hi_arr),
            }
        }

        /// Broadcast scalar to all lanes (token-gated)
        #[inline(always)]
        pub fn splat(token: Wasm128Token, v: i32) -> Self {
            Self {
                lo: i32x4::splat(token, v),
                hi: i32x4::splat(token, v),
            }
        }

        /// Zero vector (token-gated)
        #[inline(always)]
        pub fn zero(token: Wasm128Token) -> Self {
            Self {
                lo: i32x4::zero(token),
                hi: i32x4::zero(token),
            }
        }

        /// Create from array (token-gated)
        #[inline(always)]
        pub fn from_array(token: Wasm128Token, arr: [i32; 8]) -> Self {
            Self::load(token, &arr)
        }

        /// Store to array
        #[inline(always)]
        pub fn store(self, out: &mut [i32; 8]) {
            let (lo, hi) = out.split_at_mut(4);
            let lo_arr: &mut [i32; 4] = lo.try_into().unwrap();
            let hi_arr: &mut [i32; 4] = hi.try_into().unwrap();
            self.lo.store(lo_arr);
            self.hi.store(hi_arr);
        }

        /// Convert to array
        #[inline(always)]
        pub fn to_array(self) -> [i32; 8] {
            let mut out = [0i32; 8];
            self.store(&mut out);
            out
        }

        /// Element-wise minimum
        #[inline(always)]
        pub fn min(self, other: Self) -> Self {
            Self {
                lo: self.lo.min(other.lo),
                hi: self.hi.min(other.hi),
            }
        }

        /// Element-wise maximum
        #[inline(always)]
        pub fn max(self, other: Self) -> Self {
            Self {
                lo: self.lo.max(other.lo),
                hi: self.hi.max(other.hi),
            }
        }

        /// Clamp values between lo and hi
        #[inline(always)]
        pub fn clamp(self, lo: Self, hi: Self) -> Self {
            self.max(lo).min(hi)
        }

        /// Absolute value
        #[inline(always)]
        pub fn abs(self) -> Self {
            Self {
                lo: self.lo.abs(),
                hi: self.hi.abs(),
            }
        }

        /// Reduce: sum all lanes
        #[inline(always)]
        pub fn reduce_add(self) -> i32 {
            self.lo.reduce_add().wrapping_add(self.hi.reduce_add())
        }

        // ========== Bitcast ==========
        /// Reinterpret bits as `f32x8` (zero-cost).
        #[inline(always)]
        pub fn bitcast_f32x8(self) -> f32x8 {
            f32x8 {
                lo: self.lo.bitcast_f32x4(),
                hi: self.hi.bitcast_f32x4(),
            }
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
            u32x8 {
                lo: self.lo.bitcast_u32x4(),
                hi: self.hi.bitcast_u32x4(),
            }
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

    impl Add for i32x8 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                lo: self.lo + rhs.lo,
                hi: self.hi + rhs.hi,
            }
        }
    }

    impl Sub for i32x8 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                lo: self.lo - rhs.lo,
                hi: self.hi - rhs.hi,
            }
        }
    }

    impl Mul for i32x8 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            Self {
                lo: self.lo * rhs.lo,
                hi: self.hi * rhs.hi,
            }
        }
    }

    impl core::ops::AddAssign for i32x8 {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl core::ops::SubAssign for i32x8 {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    impl core::ops::MulAssign for i32x8 {
        #[inline(always)]
        fn mul_assign(&mut self, rhs: Self) {
            *self = *self * rhs;
        }
    }

    /// Emulated 8-wide u32 vector using two u32x4 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct u32x8 {
        pub(crate) lo: u32x4,
        pub(crate) hi: u32x4,
    }

    impl u32x8 {
        pub const LANES: usize = 8;

        /// Load from array (token-gated)
        #[inline(always)]
        pub fn load(token: Wasm128Token, data: &[u32; 8]) -> Self {
            let lo_arr: &[u32; 4] = data[0..4].try_into().unwrap();
            let hi_arr: &[u32; 4] = data[4..8].try_into().unwrap();
            Self {
                lo: u32x4::load(token, lo_arr),
                hi: u32x4::load(token, hi_arr),
            }
        }

        /// Broadcast scalar to all lanes (token-gated)
        #[inline(always)]
        pub fn splat(token: Wasm128Token, v: u32) -> Self {
            Self {
                lo: u32x4::splat(token, v),
                hi: u32x4::splat(token, v),
            }
        }

        /// Zero vector (token-gated)
        #[inline(always)]
        pub fn zero(token: Wasm128Token) -> Self {
            Self {
                lo: u32x4::zero(token),
                hi: u32x4::zero(token),
            }
        }

        /// Create from array (token-gated)
        #[inline(always)]
        pub fn from_array(token: Wasm128Token, arr: [u32; 8]) -> Self {
            Self::load(token, &arr)
        }

        /// Store to array
        #[inline(always)]
        pub fn store(self, out: &mut [u32; 8]) {
            let (lo, hi) = out.split_at_mut(4);
            let lo_arr: &mut [u32; 4] = lo.try_into().unwrap();
            let hi_arr: &mut [u32; 4] = hi.try_into().unwrap();
            self.lo.store(lo_arr);
            self.hi.store(hi_arr);
        }

        /// Convert to array
        #[inline(always)]
        pub fn to_array(self) -> [u32; 8] {
            let mut out = [0u32; 8];
            self.store(&mut out);
            out
        }

        /// Element-wise minimum
        #[inline(always)]
        pub fn min(self, other: Self) -> Self {
            Self {
                lo: self.lo.min(other.lo),
                hi: self.hi.min(other.hi),
            }
        }

        /// Element-wise maximum
        #[inline(always)]
        pub fn max(self, other: Self) -> Self {
            Self {
                lo: self.lo.max(other.lo),
                hi: self.hi.max(other.hi),
            }
        }

        /// Clamp values between lo and hi
        #[inline(always)]
        pub fn clamp(self, lo: Self, hi: Self) -> Self {
            self.max(lo).min(hi)
        }

        /// Reduce: sum all lanes
        #[inline(always)]
        pub fn reduce_add(self) -> u32 {
            self.lo.reduce_add().wrapping_add(self.hi.reduce_add())
        }

        // ========== Bitcast ==========
        /// Reinterpret bits as `f32x8` (zero-cost).
        #[inline(always)]
        pub fn bitcast_f32x8(self) -> f32x8 {
            f32x8 {
                lo: self.lo.bitcast_f32x4(),
                hi: self.hi.bitcast_f32x4(),
            }
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
            i32x8 {
                lo: self.lo.bitcast_i32x4(),
                hi: self.hi.bitcast_i32x4(),
            }
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

    impl Add for u32x8 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                lo: self.lo + rhs.lo,
                hi: self.hi + rhs.hi,
            }
        }
    }

    impl Sub for u32x8 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                lo: self.lo - rhs.lo,
                hi: self.hi - rhs.hi,
            }
        }
    }

    impl Mul for u32x8 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            Self {
                lo: self.lo * rhs.lo,
                hi: self.hi * rhs.hi,
            }
        }
    }

    impl core::ops::AddAssign for u32x8 {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl core::ops::SubAssign for u32x8 {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    impl core::ops::MulAssign for u32x8 {
        #[inline(always)]
        fn mul_assign(&mut self, rhs: Self) {
            *self = *self * rhs;
        }
    }

    /// Emulated 4-wide i64 vector using two i64x2 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct i64x4 {
        pub(crate) lo: i64x2,
        pub(crate) hi: i64x2,
    }

    impl i64x4 {
        pub const LANES: usize = 4;

        /// Load from array (token-gated)
        #[inline(always)]
        pub fn load(token: Wasm128Token, data: &[i64; 4]) -> Self {
            let lo_arr: &[i64; 2] = data[0..2].try_into().unwrap();
            let hi_arr: &[i64; 2] = data[2..4].try_into().unwrap();
            Self {
                lo: i64x2::load(token, lo_arr),
                hi: i64x2::load(token, hi_arr),
            }
        }

        /// Broadcast scalar to all lanes (token-gated)
        #[inline(always)]
        pub fn splat(token: Wasm128Token, v: i64) -> Self {
            Self {
                lo: i64x2::splat(token, v),
                hi: i64x2::splat(token, v),
            }
        }

        /// Zero vector (token-gated)
        #[inline(always)]
        pub fn zero(token: Wasm128Token) -> Self {
            Self {
                lo: i64x2::zero(token),
                hi: i64x2::zero(token),
            }
        }

        /// Create from array (token-gated)
        #[inline(always)]
        pub fn from_array(token: Wasm128Token, arr: [i64; 4]) -> Self {
            Self::load(token, &arr)
        }

        /// Store to array
        #[inline(always)]
        pub fn store(self, out: &mut [i64; 4]) {
            let (lo, hi) = out.split_at_mut(2);
            let lo_arr: &mut [i64; 2] = lo.try_into().unwrap();
            let hi_arr: &mut [i64; 2] = hi.try_into().unwrap();
            self.lo.store(lo_arr);
            self.hi.store(hi_arr);
        }

        /// Convert to array
        #[inline(always)]
        pub fn to_array(self) -> [i64; 4] {
            let mut out = [0i64; 4];
            self.store(&mut out);
            out
        }

        /// Reduce: sum all lanes
        #[inline(always)]
        pub fn reduce_add(self) -> i64 {
            self.lo.reduce_add().wrapping_add(self.hi.reduce_add())
        }

        // ========== Bitcast ==========
        /// Reinterpret bits as `f64x4` (zero-cost).
        #[inline(always)]
        pub fn bitcast_f64x4(self) -> f64x4 {
            f64x4 {
                lo: self.lo.bitcast_f64x2(),
                hi: self.hi.bitcast_f64x2(),
            }
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
            u64x4 {
                lo: self.lo.bitcast_u64x2(),
                hi: self.hi.bitcast_u64x2(),
            }
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

    impl Add for i64x4 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                lo: self.lo + rhs.lo,
                hi: self.hi + rhs.hi,
            }
        }
    }

    impl Sub for i64x4 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                lo: self.lo - rhs.lo,
                hi: self.hi - rhs.hi,
            }
        }
    }

    impl core::ops::AddAssign for i64x4 {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl core::ops::SubAssign for i64x4 {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    /// Emulated 4-wide u64 vector using two u64x2 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct u64x4 {
        pub(crate) lo: u64x2,
        pub(crate) hi: u64x2,
    }

    impl u64x4 {
        pub const LANES: usize = 4;

        /// Load from array (token-gated)
        #[inline(always)]
        pub fn load(token: Wasm128Token, data: &[u64; 4]) -> Self {
            let lo_arr: &[u64; 2] = data[0..2].try_into().unwrap();
            let hi_arr: &[u64; 2] = data[2..4].try_into().unwrap();
            Self {
                lo: u64x2::load(token, lo_arr),
                hi: u64x2::load(token, hi_arr),
            }
        }

        /// Broadcast scalar to all lanes (token-gated)
        #[inline(always)]
        pub fn splat(token: Wasm128Token, v: u64) -> Self {
            Self {
                lo: u64x2::splat(token, v),
                hi: u64x2::splat(token, v),
            }
        }

        /// Zero vector (token-gated)
        #[inline(always)]
        pub fn zero(token: Wasm128Token) -> Self {
            Self {
                lo: u64x2::zero(token),
                hi: u64x2::zero(token),
            }
        }

        /// Create from array (token-gated)
        #[inline(always)]
        pub fn from_array(token: Wasm128Token, arr: [u64; 4]) -> Self {
            Self::load(token, &arr)
        }

        /// Store to array
        #[inline(always)]
        pub fn store(self, out: &mut [u64; 4]) {
            let (lo, hi) = out.split_at_mut(2);
            let lo_arr: &mut [u64; 2] = lo.try_into().unwrap();
            let hi_arr: &mut [u64; 2] = hi.try_into().unwrap();
            self.lo.store(lo_arr);
            self.hi.store(hi_arr);
        }

        /// Convert to array
        #[inline(always)]
        pub fn to_array(self) -> [u64; 4] {
            let mut out = [0u64; 4];
            self.store(&mut out);
            out
        }

        /// Reduce: sum all lanes
        #[inline(always)]
        pub fn reduce_add(self) -> u64 {
            self.lo.reduce_add().wrapping_add(self.hi.reduce_add())
        }

        // ========== Bitcast ==========
        /// Reinterpret bits as `f64x4` (zero-cost).
        #[inline(always)]
        pub fn bitcast_f64x4(self) -> f64x4 {
            f64x4 {
                lo: self.lo.bitcast_f64x2(),
                hi: self.hi.bitcast_f64x2(),
            }
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
            i64x4 {
                lo: self.lo.bitcast_i64x2(),
                hi: self.hi.bitcast_i64x2(),
            }
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

    impl Add for u64x4 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                lo: self.lo + rhs.lo,
                hi: self.hi + rhs.hi,
            }
        }
    }

    impl Sub for u64x4 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                lo: self.lo - rhs.lo,
                hi: self.hi - rhs.hi,
            }
        }
    }

    impl core::ops::AddAssign for u64x4 {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl core::ops::SubAssign for u64x4 {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    // Width-aliased type names
    pub type f32xN = f32x8;
    pub type f64xN = f64x4;
    pub type i8xN = i8x32;
    pub type u8xN = u8x32;
    pub type i16xN = i16x16;
    pub type u16xN = u16x16;
    pub type i32xN = i32x8;
    pub type u32xN = u32x8;
    pub type i64xN = i64x4;
    pub type u64xN = u64x4;

    /// Number of f32 lanes
    pub const LANES_F32: usize = 8;
    /// Number of f64 lanes
    pub const LANES_F64: usize = 4;
    /// Number of i32/u32 lanes
    pub const LANES_32: usize = 8;
    /// Number of i16/u16 lanes
    pub const LANES_16: usize = 16;
    /// Number of i8/u8 lanes
    pub const LANES_8: usize = 32;

    /// Token type for this polyfill level
    pub type Token = archmage::Wasm128Token;
}

#[cfg(target_arch = "x86_64")]
pub mod avx2 {
    //! Polyfilled 512-bit types using AVX2 (256-bit) operations.

    use crate::simd::generated::x86::w256::{
        f32x8, f64x4, i8x32, i16x16, i32x8, i64x4, u8x32, u16x16, u32x8, u64x4,
    };
    use archmage::X64V3Token;
    use core::ops::{Add, Div, Mul, Neg, Sub};

    /// Emulated 16-wide f32 vector using two f32x8 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct f32x16 {
        pub(crate) lo: f32x8,
        pub(crate) hi: f32x8,
    }

    impl f32x16 {
        pub const LANES: usize = 16;

        /// Load from array (token-gated)
        #[inline(always)]
        pub fn load(token: X64V3Token, data: &[f32; 16]) -> Self {
            let lo_arr: &[f32; 8] = data[0..8].try_into().unwrap();
            let hi_arr: &[f32; 8] = data[8..16].try_into().unwrap();
            Self {
                lo: f32x8::load(token, lo_arr),
                hi: f32x8::load(token, hi_arr),
            }
        }

        /// Broadcast scalar to all lanes (token-gated)
        #[inline(always)]
        pub fn splat(token: X64V3Token, v: f32) -> Self {
            Self {
                lo: f32x8::splat(token, v),
                hi: f32x8::splat(token, v),
            }
        }

        /// Zero vector (token-gated)
        #[inline(always)]
        pub fn zero(token: X64V3Token) -> Self {
            Self {
                lo: f32x8::zero(token),
                hi: f32x8::zero(token),
            }
        }

        /// Store to array
        #[inline(always)]
        pub fn store(self, out: &mut [f32; 16]) {
            let (lo, hi) = out.split_at_mut(8);
            let lo_arr: &mut [f32; 8] = lo.try_into().unwrap();
            let hi_arr: &mut [f32; 8] = hi.try_into().unwrap();
            self.lo.store(lo_arr);
            self.hi.store(hi_arr);
        }

        /// Convert to array
        #[inline(always)]
        pub fn to_array(self) -> [f32; 16] {
            let mut out = [0.0f32; 16];
            self.store(&mut out);
            out
        }

        /// Element-wise minimum
        #[inline(always)]
        pub fn min(self, other: Self) -> Self {
            Self {
                lo: self.lo.min(other.lo),
                hi: self.hi.min(other.hi),
            }
        }

        /// Element-wise maximum
        #[inline(always)]
        pub fn max(self, other: Self) -> Self {
            Self {
                lo: self.lo.max(other.lo),
                hi: self.hi.max(other.hi),
            }
        }

        /// Clamp values between lo and hi
        #[inline(always)]
        pub fn clamp(self, lo: Self, hi: Self) -> Self {
            self.max(lo).min(hi)
        }

        /// Square root
        #[inline(always)]
        pub fn sqrt(self) -> Self {
            Self {
                lo: self.lo.sqrt(),
                hi: self.hi.sqrt(),
            }
        }

        /// Floor
        #[inline(always)]
        pub fn floor(self) -> Self {
            Self {
                lo: self.lo.floor(),
                hi: self.hi.floor(),
            }
        }

        /// Ceil
        #[inline(always)]
        pub fn ceil(self) -> Self {
            Self {
                lo: self.lo.ceil(),
                hi: self.hi.ceil(),
            }
        }

        /// Round to nearest
        #[inline(always)]
        pub fn round(self) -> Self {
            Self {
                lo: self.lo.round(),
                hi: self.hi.round(),
            }
        }

        /// Fused multiply-add: self * a + b
        #[inline(always)]
        pub fn mul_add(self, a: Self, b: Self) -> Self {
            Self {
                lo: self.lo.mul_add(a.lo, b.lo),
                hi: self.hi.mul_add(a.hi, b.hi),
            }
        }

        /// Absolute value
        #[inline(always)]
        pub fn abs(self) -> Self {
            Self {
                lo: self.lo.abs(),
                hi: self.hi.abs(),
            }
        }

        /// Reduce: sum all lanes
        #[inline(always)]
        pub fn reduce_add(self) -> f32 {
            self.lo.reduce_add() + self.hi.reduce_add()
        }

        /// Reduce: max of all lanes
        #[inline(always)]
        pub fn reduce_max(self) -> f32 {
            self.lo.reduce_max().max(self.hi.reduce_max())
        }

        /// Reduce: min of all lanes
        #[inline(always)]
        pub fn reduce_min(self) -> f32 {
            self.lo.reduce_min().min(self.hi.reduce_min())
        }

        // ========== Bitcast ==========
        /// Reinterpret bits as `i32x16` (zero-cost).
        #[inline(always)]
        pub fn bitcast_i32x16(self) -> i32x16 {
            i32x16 {
                lo: self.lo.bitcast_i32x8(),
                hi: self.hi.bitcast_i32x8(),
            }
        }

        /// Reinterpret bits as `&i32x16` (zero-cost pointer cast).
        #[inline(always)]
        pub fn bitcast_ref_i32x16(&self) -> &i32x16 {
            unsafe { &*(self as *const Self as *const i32x16) }
        }

        /// Reinterpret bits as `&mut i32x16` (zero-cost pointer cast).
        #[inline(always)]
        pub fn bitcast_mut_i32x16(&mut self) -> &mut i32x16 {
            unsafe { &mut *(self as *mut Self as *mut i32x16) }
        }
        /// Reinterpret bits as `u32x16` (zero-cost).
        #[inline(always)]
        pub fn bitcast_u32x16(self) -> u32x16 {
            u32x16 {
                lo: self.lo.bitcast_u32x8(),
                hi: self.hi.bitcast_u32x8(),
            }
        }

        /// Reinterpret bits as `&u32x16` (zero-cost pointer cast).
        #[inline(always)]
        pub fn bitcast_ref_u32x16(&self) -> &u32x16 {
            unsafe { &*(self as *const Self as *const u32x16) }
        }

        /// Reinterpret bits as `&mut u32x16` (zero-cost pointer cast).
        #[inline(always)]
        pub fn bitcast_mut_u32x16(&mut self) -> &mut u32x16 {
            unsafe { &mut *(self as *mut Self as *mut u32x16) }
        }
    }

    impl Add for f32x16 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                lo: self.lo + rhs.lo,
                hi: self.hi + rhs.hi,
            }
        }
    }

    impl Sub for f32x16 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                lo: self.lo - rhs.lo,
                hi: self.hi - rhs.hi,
            }
        }
    }

    impl Mul for f32x16 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            Self {
                lo: self.lo * rhs.lo,
                hi: self.hi * rhs.hi,
            }
        }
    }

    impl Div for f32x16 {
        type Output = Self;
        #[inline(always)]
        fn div(self, rhs: Self) -> Self {
            Self {
                lo: self.lo / rhs.lo,
                hi: self.hi / rhs.hi,
            }
        }
    }

    impl Neg for f32x16 {
        type Output = Self;
        #[inline(always)]
        fn neg(self) -> Self {
            Self {
                lo: -self.lo,
                hi: -self.hi,
            }
        }
    }

    impl core::ops::AddAssign for f32x16 {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl core::ops::SubAssign for f32x16 {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    impl core::ops::MulAssign for f32x16 {
        #[inline(always)]
        fn mul_assign(&mut self, rhs: Self) {
            *self = *self * rhs;
        }
    }

    impl core::ops::DivAssign for f32x16 {
        #[inline(always)]
        fn div_assign(&mut self, rhs: Self) {
            *self = *self / rhs;
        }
    }

    /// Emulated 8-wide f64 vector using two f64x4 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct f64x8 {
        pub(crate) lo: f64x4,
        pub(crate) hi: f64x4,
    }

    impl f64x8 {
        pub const LANES: usize = 8;

        /// Load from array (token-gated)
        #[inline(always)]
        pub fn load(token: X64V3Token, data: &[f64; 8]) -> Self {
            let lo_arr: &[f64; 4] = data[0..4].try_into().unwrap();
            let hi_arr: &[f64; 4] = data[4..8].try_into().unwrap();
            Self {
                lo: f64x4::load(token, lo_arr),
                hi: f64x4::load(token, hi_arr),
            }
        }

        /// Broadcast scalar to all lanes (token-gated)
        #[inline(always)]
        pub fn splat(token: X64V3Token, v: f64) -> Self {
            Self {
                lo: f64x4::splat(token, v),
                hi: f64x4::splat(token, v),
            }
        }

        /// Zero vector (token-gated)
        #[inline(always)]
        pub fn zero(token: X64V3Token) -> Self {
            Self {
                lo: f64x4::zero(token),
                hi: f64x4::zero(token),
            }
        }

        /// Store to array
        #[inline(always)]
        pub fn store(self, out: &mut [f64; 8]) {
            let (lo, hi) = out.split_at_mut(4);
            let lo_arr: &mut [f64; 4] = lo.try_into().unwrap();
            let hi_arr: &mut [f64; 4] = hi.try_into().unwrap();
            self.lo.store(lo_arr);
            self.hi.store(hi_arr);
        }

        /// Convert to array
        #[inline(always)]
        pub fn to_array(self) -> [f64; 8] {
            let mut out = [0.0f64; 8];
            self.store(&mut out);
            out
        }

        /// Element-wise minimum
        #[inline(always)]
        pub fn min(self, other: Self) -> Self {
            Self {
                lo: self.lo.min(other.lo),
                hi: self.hi.min(other.hi),
            }
        }

        /// Element-wise maximum
        #[inline(always)]
        pub fn max(self, other: Self) -> Self {
            Self {
                lo: self.lo.max(other.lo),
                hi: self.hi.max(other.hi),
            }
        }

        /// Clamp values between lo and hi
        #[inline(always)]
        pub fn clamp(self, lo: Self, hi: Self) -> Self {
            self.max(lo).min(hi)
        }

        /// Square root
        #[inline(always)]
        pub fn sqrt(self) -> Self {
            Self {
                lo: self.lo.sqrt(),
                hi: self.hi.sqrt(),
            }
        }

        /// Floor
        #[inline(always)]
        pub fn floor(self) -> Self {
            Self {
                lo: self.lo.floor(),
                hi: self.hi.floor(),
            }
        }

        /// Ceil
        #[inline(always)]
        pub fn ceil(self) -> Self {
            Self {
                lo: self.lo.ceil(),
                hi: self.hi.ceil(),
            }
        }

        /// Round to nearest
        #[inline(always)]
        pub fn round(self) -> Self {
            Self {
                lo: self.lo.round(),
                hi: self.hi.round(),
            }
        }

        /// Fused multiply-add: self * a + b
        #[inline(always)]
        pub fn mul_add(self, a: Self, b: Self) -> Self {
            Self {
                lo: self.lo.mul_add(a.lo, b.lo),
                hi: self.hi.mul_add(a.hi, b.hi),
            }
        }

        /// Absolute value
        #[inline(always)]
        pub fn abs(self) -> Self {
            Self {
                lo: self.lo.abs(),
                hi: self.hi.abs(),
            }
        }

        /// Reduce: sum all lanes
        #[inline(always)]
        pub fn reduce_add(self) -> f64 {
            self.lo.reduce_add() + self.hi.reduce_add()
        }

        /// Reduce: max of all lanes
        #[inline(always)]
        pub fn reduce_max(self) -> f64 {
            self.lo.reduce_max().max(self.hi.reduce_max())
        }

        /// Reduce: min of all lanes
        #[inline(always)]
        pub fn reduce_min(self) -> f64 {
            self.lo.reduce_min().min(self.hi.reduce_min())
        }

        // ========== Bitcast ==========
        /// Reinterpret bits as `i64x8` (zero-cost).
        #[inline(always)]
        pub fn bitcast_i64x8(self) -> i64x8 {
            i64x8 {
                lo: self.lo.bitcast_i64x4(),
                hi: self.hi.bitcast_i64x4(),
            }
        }

        /// Reinterpret bits as `&i64x8` (zero-cost pointer cast).
        #[inline(always)]
        pub fn bitcast_ref_i64x8(&self) -> &i64x8 {
            unsafe { &*(self as *const Self as *const i64x8) }
        }

        /// Reinterpret bits as `&mut i64x8` (zero-cost pointer cast).
        #[inline(always)]
        pub fn bitcast_mut_i64x8(&mut self) -> &mut i64x8 {
            unsafe { &mut *(self as *mut Self as *mut i64x8) }
        }
        /// Reinterpret bits as `u64x8` (zero-cost).
        #[inline(always)]
        pub fn bitcast_u64x8(self) -> u64x8 {
            u64x8 {
                lo: self.lo.bitcast_u64x4(),
                hi: self.hi.bitcast_u64x4(),
            }
        }

        /// Reinterpret bits as `&u64x8` (zero-cost pointer cast).
        #[inline(always)]
        pub fn bitcast_ref_u64x8(&self) -> &u64x8 {
            unsafe { &*(self as *const Self as *const u64x8) }
        }

        /// Reinterpret bits as `&mut u64x8` (zero-cost pointer cast).
        #[inline(always)]
        pub fn bitcast_mut_u64x8(&mut self) -> &mut u64x8 {
            unsafe { &mut *(self as *mut Self as *mut u64x8) }
        }
    }

    impl Add for f64x8 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                lo: self.lo + rhs.lo,
                hi: self.hi + rhs.hi,
            }
        }
    }

    impl Sub for f64x8 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                lo: self.lo - rhs.lo,
                hi: self.hi - rhs.hi,
            }
        }
    }

    impl Mul for f64x8 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            Self {
                lo: self.lo * rhs.lo,
                hi: self.hi * rhs.hi,
            }
        }
    }

    impl Div for f64x8 {
        type Output = Self;
        #[inline(always)]
        fn div(self, rhs: Self) -> Self {
            Self {
                lo: self.lo / rhs.lo,
                hi: self.hi / rhs.hi,
            }
        }
    }

    impl Neg for f64x8 {
        type Output = Self;
        #[inline(always)]
        fn neg(self) -> Self {
            Self {
                lo: -self.lo,
                hi: -self.hi,
            }
        }
    }

    impl core::ops::AddAssign for f64x8 {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl core::ops::SubAssign for f64x8 {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    impl core::ops::MulAssign for f64x8 {
        #[inline(always)]
        fn mul_assign(&mut self, rhs: Self) {
            *self = *self * rhs;
        }
    }

    impl core::ops::DivAssign for f64x8 {
        #[inline(always)]
        fn div_assign(&mut self, rhs: Self) {
            *self = *self / rhs;
        }
    }

    /// Emulated 64-wide i8 vector using two i8x32 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct i8x64 {
        pub(crate) lo: i8x32,
        pub(crate) hi: i8x32,
    }

    impl i8x64 {
        pub const LANES: usize = 64;

        /// Load from array (token-gated)
        #[inline(always)]
        pub fn load(token: X64V3Token, data: &[i8; 64]) -> Self {
            let lo_arr: &[i8; 32] = data[0..32].try_into().unwrap();
            let hi_arr: &[i8; 32] = data[32..64].try_into().unwrap();
            Self {
                lo: i8x32::load(token, lo_arr),
                hi: i8x32::load(token, hi_arr),
            }
        }

        /// Broadcast scalar to all lanes (token-gated)
        #[inline(always)]
        pub fn splat(token: X64V3Token, v: i8) -> Self {
            Self {
                lo: i8x32::splat(token, v),
                hi: i8x32::splat(token, v),
            }
        }

        /// Zero vector (token-gated)
        #[inline(always)]
        pub fn zero(token: X64V3Token) -> Self {
            Self {
                lo: i8x32::zero(token),
                hi: i8x32::zero(token),
            }
        }

        /// Store to array
        #[inline(always)]
        pub fn store(self, out: &mut [i8; 64]) {
            let (lo, hi) = out.split_at_mut(32);
            let lo_arr: &mut [i8; 32] = lo.try_into().unwrap();
            let hi_arr: &mut [i8; 32] = hi.try_into().unwrap();
            self.lo.store(lo_arr);
            self.hi.store(hi_arr);
        }

        /// Convert to array
        #[inline(always)]
        pub fn to_array(self) -> [i8; 64] {
            let mut out = [0i8; 64];
            self.store(&mut out);
            out
        }

        /// Element-wise minimum
        #[inline(always)]
        pub fn min(self, other: Self) -> Self {
            Self {
                lo: self.lo.min(other.lo),
                hi: self.hi.min(other.hi),
            }
        }

        /// Element-wise maximum
        #[inline(always)]
        pub fn max(self, other: Self) -> Self {
            Self {
                lo: self.lo.max(other.lo),
                hi: self.hi.max(other.hi),
            }
        }

        /// Clamp values between lo and hi
        #[inline(always)]
        pub fn clamp(self, lo: Self, hi: Self) -> Self {
            self.max(lo).min(hi)
        }

        /// Absolute value
        #[inline(always)]
        pub fn abs(self) -> Self {
            Self {
                lo: self.lo.abs(),
                hi: self.hi.abs(),
            }
        }

        /// Reduce: sum all lanes
        #[inline(always)]
        pub fn reduce_add(self) -> i8 {
            self.lo.reduce_add().wrapping_add(self.hi.reduce_add())
        }

        // ========== Bitcast ==========
        /// Reinterpret bits as `u8x64` (zero-cost).
        #[inline(always)]
        pub fn bitcast_u8x64(self) -> u8x64 {
            u8x64 {
                lo: self.lo.bitcast_u8x32(),
                hi: self.hi.bitcast_u8x32(),
            }
        }

        /// Reinterpret bits as `&u8x64` (zero-cost pointer cast).
        #[inline(always)]
        pub fn bitcast_ref_u8x64(&self) -> &u8x64 {
            unsafe { &*(self as *const Self as *const u8x64) }
        }

        /// Reinterpret bits as `&mut u8x64` (zero-cost pointer cast).
        #[inline(always)]
        pub fn bitcast_mut_u8x64(&mut self) -> &mut u8x64 {
            unsafe { &mut *(self as *mut Self as *mut u8x64) }
        }
    }

    impl Add for i8x64 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                lo: self.lo + rhs.lo,
                hi: self.hi + rhs.hi,
            }
        }
    }

    impl Sub for i8x64 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                lo: self.lo - rhs.lo,
                hi: self.hi - rhs.hi,
            }
        }
    }

    impl core::ops::AddAssign for i8x64 {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl core::ops::SubAssign for i8x64 {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    /// Emulated 64-wide u8 vector using two u8x32 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct u8x64 {
        pub(crate) lo: u8x32,
        pub(crate) hi: u8x32,
    }

    impl u8x64 {
        pub const LANES: usize = 64;

        /// Load from array (token-gated)
        #[inline(always)]
        pub fn load(token: X64V3Token, data: &[u8; 64]) -> Self {
            let lo_arr: &[u8; 32] = data[0..32].try_into().unwrap();
            let hi_arr: &[u8; 32] = data[32..64].try_into().unwrap();
            Self {
                lo: u8x32::load(token, lo_arr),
                hi: u8x32::load(token, hi_arr),
            }
        }

        /// Broadcast scalar to all lanes (token-gated)
        #[inline(always)]
        pub fn splat(token: X64V3Token, v: u8) -> Self {
            Self {
                lo: u8x32::splat(token, v),
                hi: u8x32::splat(token, v),
            }
        }

        /// Zero vector (token-gated)
        #[inline(always)]
        pub fn zero(token: X64V3Token) -> Self {
            Self {
                lo: u8x32::zero(token),
                hi: u8x32::zero(token),
            }
        }

        /// Store to array
        #[inline(always)]
        pub fn store(self, out: &mut [u8; 64]) {
            let (lo, hi) = out.split_at_mut(32);
            let lo_arr: &mut [u8; 32] = lo.try_into().unwrap();
            let hi_arr: &mut [u8; 32] = hi.try_into().unwrap();
            self.lo.store(lo_arr);
            self.hi.store(hi_arr);
        }

        /// Convert to array
        #[inline(always)]
        pub fn to_array(self) -> [u8; 64] {
            let mut out = [0u8; 64];
            self.store(&mut out);
            out
        }

        /// Element-wise minimum
        #[inline(always)]
        pub fn min(self, other: Self) -> Self {
            Self {
                lo: self.lo.min(other.lo),
                hi: self.hi.min(other.hi),
            }
        }

        /// Element-wise maximum
        #[inline(always)]
        pub fn max(self, other: Self) -> Self {
            Self {
                lo: self.lo.max(other.lo),
                hi: self.hi.max(other.hi),
            }
        }

        /// Clamp values between lo and hi
        #[inline(always)]
        pub fn clamp(self, lo: Self, hi: Self) -> Self {
            self.max(lo).min(hi)
        }

        /// Reduce: sum all lanes
        #[inline(always)]
        pub fn reduce_add(self) -> u8 {
            self.lo.reduce_add().wrapping_add(self.hi.reduce_add())
        }

        // ========== Bitcast ==========
        /// Reinterpret bits as `i8x64` (zero-cost).
        #[inline(always)]
        pub fn bitcast_i8x64(self) -> i8x64 {
            i8x64 {
                lo: self.lo.bitcast_i8x32(),
                hi: self.hi.bitcast_i8x32(),
            }
        }

        /// Reinterpret bits as `&i8x64` (zero-cost pointer cast).
        #[inline(always)]
        pub fn bitcast_ref_i8x64(&self) -> &i8x64 {
            unsafe { &*(self as *const Self as *const i8x64) }
        }

        /// Reinterpret bits as `&mut i8x64` (zero-cost pointer cast).
        #[inline(always)]
        pub fn bitcast_mut_i8x64(&mut self) -> &mut i8x64 {
            unsafe { &mut *(self as *mut Self as *mut i8x64) }
        }
    }

    impl Add for u8x64 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                lo: self.lo + rhs.lo,
                hi: self.hi + rhs.hi,
            }
        }
    }

    impl Sub for u8x64 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                lo: self.lo - rhs.lo,
                hi: self.hi - rhs.hi,
            }
        }
    }

    impl core::ops::AddAssign for u8x64 {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl core::ops::SubAssign for u8x64 {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    /// Emulated 32-wide i16 vector using two i16x16 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct i16x32 {
        pub(crate) lo: i16x16,
        pub(crate) hi: i16x16,
    }

    impl i16x32 {
        pub const LANES: usize = 32;

        /// Load from array (token-gated)
        #[inline(always)]
        pub fn load(token: X64V3Token, data: &[i16; 32]) -> Self {
            let lo_arr: &[i16; 16] = data[0..16].try_into().unwrap();
            let hi_arr: &[i16; 16] = data[16..32].try_into().unwrap();
            Self {
                lo: i16x16::load(token, lo_arr),
                hi: i16x16::load(token, hi_arr),
            }
        }

        /// Broadcast scalar to all lanes (token-gated)
        #[inline(always)]
        pub fn splat(token: X64V3Token, v: i16) -> Self {
            Self {
                lo: i16x16::splat(token, v),
                hi: i16x16::splat(token, v),
            }
        }

        /// Zero vector (token-gated)
        #[inline(always)]
        pub fn zero(token: X64V3Token) -> Self {
            Self {
                lo: i16x16::zero(token),
                hi: i16x16::zero(token),
            }
        }

        /// Store to array
        #[inline(always)]
        pub fn store(self, out: &mut [i16; 32]) {
            let (lo, hi) = out.split_at_mut(16);
            let lo_arr: &mut [i16; 16] = lo.try_into().unwrap();
            let hi_arr: &mut [i16; 16] = hi.try_into().unwrap();
            self.lo.store(lo_arr);
            self.hi.store(hi_arr);
        }

        /// Convert to array
        #[inline(always)]
        pub fn to_array(self) -> [i16; 32] {
            let mut out = [0i16; 32];
            self.store(&mut out);
            out
        }

        /// Element-wise minimum
        #[inline(always)]
        pub fn min(self, other: Self) -> Self {
            Self {
                lo: self.lo.min(other.lo),
                hi: self.hi.min(other.hi),
            }
        }

        /// Element-wise maximum
        #[inline(always)]
        pub fn max(self, other: Self) -> Self {
            Self {
                lo: self.lo.max(other.lo),
                hi: self.hi.max(other.hi),
            }
        }

        /// Clamp values between lo and hi
        #[inline(always)]
        pub fn clamp(self, lo: Self, hi: Self) -> Self {
            self.max(lo).min(hi)
        }

        /// Absolute value
        #[inline(always)]
        pub fn abs(self) -> Self {
            Self {
                lo: self.lo.abs(),
                hi: self.hi.abs(),
            }
        }

        /// Reduce: sum all lanes
        #[inline(always)]
        pub fn reduce_add(self) -> i16 {
            self.lo.reduce_add().wrapping_add(self.hi.reduce_add())
        }

        // ========== Bitcast ==========
        /// Reinterpret bits as `u16x32` (zero-cost).
        #[inline(always)]
        pub fn bitcast_u16x32(self) -> u16x32 {
            u16x32 {
                lo: self.lo.bitcast_u16x16(),
                hi: self.hi.bitcast_u16x16(),
            }
        }

        /// Reinterpret bits as `&u16x32` (zero-cost pointer cast).
        #[inline(always)]
        pub fn bitcast_ref_u16x32(&self) -> &u16x32 {
            unsafe { &*(self as *const Self as *const u16x32) }
        }

        /// Reinterpret bits as `&mut u16x32` (zero-cost pointer cast).
        #[inline(always)]
        pub fn bitcast_mut_u16x32(&mut self) -> &mut u16x32 {
            unsafe { &mut *(self as *mut Self as *mut u16x32) }
        }
    }

    impl Add for i16x32 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                lo: self.lo + rhs.lo,
                hi: self.hi + rhs.hi,
            }
        }
    }

    impl Sub for i16x32 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                lo: self.lo - rhs.lo,
                hi: self.hi - rhs.hi,
            }
        }
    }

    impl Mul for i16x32 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            Self {
                lo: self.lo * rhs.lo,
                hi: self.hi * rhs.hi,
            }
        }
    }

    impl core::ops::AddAssign for i16x32 {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl core::ops::SubAssign for i16x32 {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    impl core::ops::MulAssign for i16x32 {
        #[inline(always)]
        fn mul_assign(&mut self, rhs: Self) {
            *self = *self * rhs;
        }
    }

    /// Emulated 32-wide u16 vector using two u16x16 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct u16x32 {
        pub(crate) lo: u16x16,
        pub(crate) hi: u16x16,
    }

    impl u16x32 {
        pub const LANES: usize = 32;

        /// Load from array (token-gated)
        #[inline(always)]
        pub fn load(token: X64V3Token, data: &[u16; 32]) -> Self {
            let lo_arr: &[u16; 16] = data[0..16].try_into().unwrap();
            let hi_arr: &[u16; 16] = data[16..32].try_into().unwrap();
            Self {
                lo: u16x16::load(token, lo_arr),
                hi: u16x16::load(token, hi_arr),
            }
        }

        /// Broadcast scalar to all lanes (token-gated)
        #[inline(always)]
        pub fn splat(token: X64V3Token, v: u16) -> Self {
            Self {
                lo: u16x16::splat(token, v),
                hi: u16x16::splat(token, v),
            }
        }

        /// Zero vector (token-gated)
        #[inline(always)]
        pub fn zero(token: X64V3Token) -> Self {
            Self {
                lo: u16x16::zero(token),
                hi: u16x16::zero(token),
            }
        }

        /// Store to array
        #[inline(always)]
        pub fn store(self, out: &mut [u16; 32]) {
            let (lo, hi) = out.split_at_mut(16);
            let lo_arr: &mut [u16; 16] = lo.try_into().unwrap();
            let hi_arr: &mut [u16; 16] = hi.try_into().unwrap();
            self.lo.store(lo_arr);
            self.hi.store(hi_arr);
        }

        /// Convert to array
        #[inline(always)]
        pub fn to_array(self) -> [u16; 32] {
            let mut out = [0u16; 32];
            self.store(&mut out);
            out
        }

        /// Element-wise minimum
        #[inline(always)]
        pub fn min(self, other: Self) -> Self {
            Self {
                lo: self.lo.min(other.lo),
                hi: self.hi.min(other.hi),
            }
        }

        /// Element-wise maximum
        #[inline(always)]
        pub fn max(self, other: Self) -> Self {
            Self {
                lo: self.lo.max(other.lo),
                hi: self.hi.max(other.hi),
            }
        }

        /// Clamp values between lo and hi
        #[inline(always)]
        pub fn clamp(self, lo: Self, hi: Self) -> Self {
            self.max(lo).min(hi)
        }

        /// Reduce: sum all lanes
        #[inline(always)]
        pub fn reduce_add(self) -> u16 {
            self.lo.reduce_add().wrapping_add(self.hi.reduce_add())
        }

        // ========== Bitcast ==========
        /// Reinterpret bits as `i16x32` (zero-cost).
        #[inline(always)]
        pub fn bitcast_i16x32(self) -> i16x32 {
            i16x32 {
                lo: self.lo.bitcast_i16x16(),
                hi: self.hi.bitcast_i16x16(),
            }
        }

        /// Reinterpret bits as `&i16x32` (zero-cost pointer cast).
        #[inline(always)]
        pub fn bitcast_ref_i16x32(&self) -> &i16x32 {
            unsafe { &*(self as *const Self as *const i16x32) }
        }

        /// Reinterpret bits as `&mut i16x32` (zero-cost pointer cast).
        #[inline(always)]
        pub fn bitcast_mut_i16x32(&mut self) -> &mut i16x32 {
            unsafe { &mut *(self as *mut Self as *mut i16x32) }
        }
    }

    impl Add for u16x32 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                lo: self.lo + rhs.lo,
                hi: self.hi + rhs.hi,
            }
        }
    }

    impl Sub for u16x32 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                lo: self.lo - rhs.lo,
                hi: self.hi - rhs.hi,
            }
        }
    }

    impl Mul for u16x32 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            Self {
                lo: self.lo * rhs.lo,
                hi: self.hi * rhs.hi,
            }
        }
    }

    impl core::ops::AddAssign for u16x32 {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl core::ops::SubAssign for u16x32 {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    impl core::ops::MulAssign for u16x32 {
        #[inline(always)]
        fn mul_assign(&mut self, rhs: Self) {
            *self = *self * rhs;
        }
    }

    /// Emulated 16-wide i32 vector using two i32x8 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct i32x16 {
        pub(crate) lo: i32x8,
        pub(crate) hi: i32x8,
    }

    impl i32x16 {
        pub const LANES: usize = 16;

        /// Load from array (token-gated)
        #[inline(always)]
        pub fn load(token: X64V3Token, data: &[i32; 16]) -> Self {
            let lo_arr: &[i32; 8] = data[0..8].try_into().unwrap();
            let hi_arr: &[i32; 8] = data[8..16].try_into().unwrap();
            Self {
                lo: i32x8::load(token, lo_arr),
                hi: i32x8::load(token, hi_arr),
            }
        }

        /// Broadcast scalar to all lanes (token-gated)
        #[inline(always)]
        pub fn splat(token: X64V3Token, v: i32) -> Self {
            Self {
                lo: i32x8::splat(token, v),
                hi: i32x8::splat(token, v),
            }
        }

        /// Zero vector (token-gated)
        #[inline(always)]
        pub fn zero(token: X64V3Token) -> Self {
            Self {
                lo: i32x8::zero(token),
                hi: i32x8::zero(token),
            }
        }

        /// Store to array
        #[inline(always)]
        pub fn store(self, out: &mut [i32; 16]) {
            let (lo, hi) = out.split_at_mut(8);
            let lo_arr: &mut [i32; 8] = lo.try_into().unwrap();
            let hi_arr: &mut [i32; 8] = hi.try_into().unwrap();
            self.lo.store(lo_arr);
            self.hi.store(hi_arr);
        }

        /// Convert to array
        #[inline(always)]
        pub fn to_array(self) -> [i32; 16] {
            let mut out = [0i32; 16];
            self.store(&mut out);
            out
        }

        /// Element-wise minimum
        #[inline(always)]
        pub fn min(self, other: Self) -> Self {
            Self {
                lo: self.lo.min(other.lo),
                hi: self.hi.min(other.hi),
            }
        }

        /// Element-wise maximum
        #[inline(always)]
        pub fn max(self, other: Self) -> Self {
            Self {
                lo: self.lo.max(other.lo),
                hi: self.hi.max(other.hi),
            }
        }

        /// Clamp values between lo and hi
        #[inline(always)]
        pub fn clamp(self, lo: Self, hi: Self) -> Self {
            self.max(lo).min(hi)
        }

        /// Absolute value
        #[inline(always)]
        pub fn abs(self) -> Self {
            Self {
                lo: self.lo.abs(),
                hi: self.hi.abs(),
            }
        }

        /// Reduce: sum all lanes
        #[inline(always)]
        pub fn reduce_add(self) -> i32 {
            self.lo.reduce_add().wrapping_add(self.hi.reduce_add())
        }

        // ========== Bitcast ==========
        /// Reinterpret bits as `f32x16` (zero-cost).
        #[inline(always)]
        pub fn bitcast_f32x16(self) -> f32x16 {
            f32x16 {
                lo: self.lo.bitcast_f32x8(),
                hi: self.hi.bitcast_f32x8(),
            }
        }

        /// Reinterpret bits as `&f32x16` (zero-cost pointer cast).
        #[inline(always)]
        pub fn bitcast_ref_f32x16(&self) -> &f32x16 {
            unsafe { &*(self as *const Self as *const f32x16) }
        }

        /// Reinterpret bits as `&mut f32x16` (zero-cost pointer cast).
        #[inline(always)]
        pub fn bitcast_mut_f32x16(&mut self) -> &mut f32x16 {
            unsafe { &mut *(self as *mut Self as *mut f32x16) }
        }
        /// Reinterpret bits as `u32x16` (zero-cost).
        #[inline(always)]
        pub fn bitcast_u32x16(self) -> u32x16 {
            u32x16 {
                lo: self.lo.bitcast_u32x8(),
                hi: self.hi.bitcast_u32x8(),
            }
        }

        /// Reinterpret bits as `&u32x16` (zero-cost pointer cast).
        #[inline(always)]
        pub fn bitcast_ref_u32x16(&self) -> &u32x16 {
            unsafe { &*(self as *const Self as *const u32x16) }
        }

        /// Reinterpret bits as `&mut u32x16` (zero-cost pointer cast).
        #[inline(always)]
        pub fn bitcast_mut_u32x16(&mut self) -> &mut u32x16 {
            unsafe { &mut *(self as *mut Self as *mut u32x16) }
        }
    }

    impl Add for i32x16 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                lo: self.lo + rhs.lo,
                hi: self.hi + rhs.hi,
            }
        }
    }

    impl Sub for i32x16 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                lo: self.lo - rhs.lo,
                hi: self.hi - rhs.hi,
            }
        }
    }

    impl Mul for i32x16 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            Self {
                lo: self.lo * rhs.lo,
                hi: self.hi * rhs.hi,
            }
        }
    }

    impl core::ops::AddAssign for i32x16 {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl core::ops::SubAssign for i32x16 {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    impl core::ops::MulAssign for i32x16 {
        #[inline(always)]
        fn mul_assign(&mut self, rhs: Self) {
            *self = *self * rhs;
        }
    }

    /// Emulated 16-wide u32 vector using two u32x8 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct u32x16 {
        pub(crate) lo: u32x8,
        pub(crate) hi: u32x8,
    }

    impl u32x16 {
        pub const LANES: usize = 16;

        /// Load from array (token-gated)
        #[inline(always)]
        pub fn load(token: X64V3Token, data: &[u32; 16]) -> Self {
            let lo_arr: &[u32; 8] = data[0..8].try_into().unwrap();
            let hi_arr: &[u32; 8] = data[8..16].try_into().unwrap();
            Self {
                lo: u32x8::load(token, lo_arr),
                hi: u32x8::load(token, hi_arr),
            }
        }

        /// Broadcast scalar to all lanes (token-gated)
        #[inline(always)]
        pub fn splat(token: X64V3Token, v: u32) -> Self {
            Self {
                lo: u32x8::splat(token, v),
                hi: u32x8::splat(token, v),
            }
        }

        /// Zero vector (token-gated)
        #[inline(always)]
        pub fn zero(token: X64V3Token) -> Self {
            Self {
                lo: u32x8::zero(token),
                hi: u32x8::zero(token),
            }
        }

        /// Store to array
        #[inline(always)]
        pub fn store(self, out: &mut [u32; 16]) {
            let (lo, hi) = out.split_at_mut(8);
            let lo_arr: &mut [u32; 8] = lo.try_into().unwrap();
            let hi_arr: &mut [u32; 8] = hi.try_into().unwrap();
            self.lo.store(lo_arr);
            self.hi.store(hi_arr);
        }

        /// Convert to array
        #[inline(always)]
        pub fn to_array(self) -> [u32; 16] {
            let mut out = [0u32; 16];
            self.store(&mut out);
            out
        }

        /// Element-wise minimum
        #[inline(always)]
        pub fn min(self, other: Self) -> Self {
            Self {
                lo: self.lo.min(other.lo),
                hi: self.hi.min(other.hi),
            }
        }

        /// Element-wise maximum
        #[inline(always)]
        pub fn max(self, other: Self) -> Self {
            Self {
                lo: self.lo.max(other.lo),
                hi: self.hi.max(other.hi),
            }
        }

        /// Clamp values between lo and hi
        #[inline(always)]
        pub fn clamp(self, lo: Self, hi: Self) -> Self {
            self.max(lo).min(hi)
        }

        /// Reduce: sum all lanes
        #[inline(always)]
        pub fn reduce_add(self) -> u32 {
            self.lo.reduce_add().wrapping_add(self.hi.reduce_add())
        }

        // ========== Bitcast ==========
        /// Reinterpret bits as `f32x16` (zero-cost).
        #[inline(always)]
        pub fn bitcast_f32x16(self) -> f32x16 {
            f32x16 {
                lo: self.lo.bitcast_f32x8(),
                hi: self.hi.bitcast_f32x8(),
            }
        }

        /// Reinterpret bits as `&f32x16` (zero-cost pointer cast).
        #[inline(always)]
        pub fn bitcast_ref_f32x16(&self) -> &f32x16 {
            unsafe { &*(self as *const Self as *const f32x16) }
        }

        /// Reinterpret bits as `&mut f32x16` (zero-cost pointer cast).
        #[inline(always)]
        pub fn bitcast_mut_f32x16(&mut self) -> &mut f32x16 {
            unsafe { &mut *(self as *mut Self as *mut f32x16) }
        }
        /// Reinterpret bits as `i32x16` (zero-cost).
        #[inline(always)]
        pub fn bitcast_i32x16(self) -> i32x16 {
            i32x16 {
                lo: self.lo.bitcast_i32x8(),
                hi: self.hi.bitcast_i32x8(),
            }
        }

        /// Reinterpret bits as `&i32x16` (zero-cost pointer cast).
        #[inline(always)]
        pub fn bitcast_ref_i32x16(&self) -> &i32x16 {
            unsafe { &*(self as *const Self as *const i32x16) }
        }

        /// Reinterpret bits as `&mut i32x16` (zero-cost pointer cast).
        #[inline(always)]
        pub fn bitcast_mut_i32x16(&mut self) -> &mut i32x16 {
            unsafe { &mut *(self as *mut Self as *mut i32x16) }
        }
    }

    impl Add for u32x16 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                lo: self.lo + rhs.lo,
                hi: self.hi + rhs.hi,
            }
        }
    }

    impl Sub for u32x16 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                lo: self.lo - rhs.lo,
                hi: self.hi - rhs.hi,
            }
        }
    }

    impl Mul for u32x16 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            Self {
                lo: self.lo * rhs.lo,
                hi: self.hi * rhs.hi,
            }
        }
    }

    impl core::ops::AddAssign for u32x16 {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl core::ops::SubAssign for u32x16 {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    impl core::ops::MulAssign for u32x16 {
        #[inline(always)]
        fn mul_assign(&mut self, rhs: Self) {
            *self = *self * rhs;
        }
    }

    /// Emulated 8-wide i64 vector using two i64x4 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct i64x8 {
        pub(crate) lo: i64x4,
        pub(crate) hi: i64x4,
    }

    impl i64x8 {
        pub const LANES: usize = 8;

        /// Load from array (token-gated)
        #[inline(always)]
        pub fn load(token: X64V3Token, data: &[i64; 8]) -> Self {
            let lo_arr: &[i64; 4] = data[0..4].try_into().unwrap();
            let hi_arr: &[i64; 4] = data[4..8].try_into().unwrap();
            Self {
                lo: i64x4::load(token, lo_arr),
                hi: i64x4::load(token, hi_arr),
            }
        }

        /// Broadcast scalar to all lanes (token-gated)
        #[inline(always)]
        pub fn splat(token: X64V3Token, v: i64) -> Self {
            Self {
                lo: i64x4::splat(token, v),
                hi: i64x4::splat(token, v),
            }
        }

        /// Zero vector (token-gated)
        #[inline(always)]
        pub fn zero(token: X64V3Token) -> Self {
            Self {
                lo: i64x4::zero(token),
                hi: i64x4::zero(token),
            }
        }

        /// Store to array
        #[inline(always)]
        pub fn store(self, out: &mut [i64; 8]) {
            let (lo, hi) = out.split_at_mut(4);
            let lo_arr: &mut [i64; 4] = lo.try_into().unwrap();
            let hi_arr: &mut [i64; 4] = hi.try_into().unwrap();
            self.lo.store(lo_arr);
            self.hi.store(hi_arr);
        }

        /// Convert to array
        #[inline(always)]
        pub fn to_array(self) -> [i64; 8] {
            let mut out = [0i64; 8];
            self.store(&mut out);
            out
        }

        /// Reduce: sum all lanes
        #[inline(always)]
        pub fn reduce_add(self) -> i64 {
            self.lo.reduce_add().wrapping_add(self.hi.reduce_add())
        }

        // ========== Bitcast ==========
        /// Reinterpret bits as `f64x8` (zero-cost).
        #[inline(always)]
        pub fn bitcast_f64x8(self) -> f64x8 {
            f64x8 {
                lo: self.lo.bitcast_f64x4(),
                hi: self.hi.bitcast_f64x4(),
            }
        }

        /// Reinterpret bits as `&f64x8` (zero-cost pointer cast).
        #[inline(always)]
        pub fn bitcast_ref_f64x8(&self) -> &f64x8 {
            unsafe { &*(self as *const Self as *const f64x8) }
        }

        /// Reinterpret bits as `&mut f64x8` (zero-cost pointer cast).
        #[inline(always)]
        pub fn bitcast_mut_f64x8(&mut self) -> &mut f64x8 {
            unsafe { &mut *(self as *mut Self as *mut f64x8) }
        }
        /// Reinterpret bits as `u64x8` (zero-cost).
        #[inline(always)]
        pub fn bitcast_u64x8(self) -> u64x8 {
            u64x8 {
                lo: self.lo.bitcast_u64x4(),
                hi: self.hi.bitcast_u64x4(),
            }
        }

        /// Reinterpret bits as `&u64x8` (zero-cost pointer cast).
        #[inline(always)]
        pub fn bitcast_ref_u64x8(&self) -> &u64x8 {
            unsafe { &*(self as *const Self as *const u64x8) }
        }

        /// Reinterpret bits as `&mut u64x8` (zero-cost pointer cast).
        #[inline(always)]
        pub fn bitcast_mut_u64x8(&mut self) -> &mut u64x8 {
            unsafe { &mut *(self as *mut Self as *mut u64x8) }
        }
    }

    impl Add for i64x8 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                lo: self.lo + rhs.lo,
                hi: self.hi + rhs.hi,
            }
        }
    }

    impl Sub for i64x8 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                lo: self.lo - rhs.lo,
                hi: self.hi - rhs.hi,
            }
        }
    }

    impl core::ops::AddAssign for i64x8 {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl core::ops::SubAssign for i64x8 {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    /// Emulated 8-wide u64 vector using two u64x4 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct u64x8 {
        pub(crate) lo: u64x4,
        pub(crate) hi: u64x4,
    }

    impl u64x8 {
        pub const LANES: usize = 8;

        /// Load from array (token-gated)
        #[inline(always)]
        pub fn load(token: X64V3Token, data: &[u64; 8]) -> Self {
            let lo_arr: &[u64; 4] = data[0..4].try_into().unwrap();
            let hi_arr: &[u64; 4] = data[4..8].try_into().unwrap();
            Self {
                lo: u64x4::load(token, lo_arr),
                hi: u64x4::load(token, hi_arr),
            }
        }

        /// Broadcast scalar to all lanes (token-gated)
        #[inline(always)]
        pub fn splat(token: X64V3Token, v: u64) -> Self {
            Self {
                lo: u64x4::splat(token, v),
                hi: u64x4::splat(token, v),
            }
        }

        /// Zero vector (token-gated)
        #[inline(always)]
        pub fn zero(token: X64V3Token) -> Self {
            Self {
                lo: u64x4::zero(token),
                hi: u64x4::zero(token),
            }
        }

        /// Store to array
        #[inline(always)]
        pub fn store(self, out: &mut [u64; 8]) {
            let (lo, hi) = out.split_at_mut(4);
            let lo_arr: &mut [u64; 4] = lo.try_into().unwrap();
            let hi_arr: &mut [u64; 4] = hi.try_into().unwrap();
            self.lo.store(lo_arr);
            self.hi.store(hi_arr);
        }

        /// Convert to array
        #[inline(always)]
        pub fn to_array(self) -> [u64; 8] {
            let mut out = [0u64; 8];
            self.store(&mut out);
            out
        }

        /// Reduce: sum all lanes
        #[inline(always)]
        pub fn reduce_add(self) -> u64 {
            self.lo.reduce_add().wrapping_add(self.hi.reduce_add())
        }

        // ========== Bitcast ==========
        /// Reinterpret bits as `f64x8` (zero-cost).
        #[inline(always)]
        pub fn bitcast_f64x8(self) -> f64x8 {
            f64x8 {
                lo: self.lo.bitcast_f64x4(),
                hi: self.hi.bitcast_f64x4(),
            }
        }

        /// Reinterpret bits as `&f64x8` (zero-cost pointer cast).
        #[inline(always)]
        pub fn bitcast_ref_f64x8(&self) -> &f64x8 {
            unsafe { &*(self as *const Self as *const f64x8) }
        }

        /// Reinterpret bits as `&mut f64x8` (zero-cost pointer cast).
        #[inline(always)]
        pub fn bitcast_mut_f64x8(&mut self) -> &mut f64x8 {
            unsafe { &mut *(self as *mut Self as *mut f64x8) }
        }
        /// Reinterpret bits as `i64x8` (zero-cost).
        #[inline(always)]
        pub fn bitcast_i64x8(self) -> i64x8 {
            i64x8 {
                lo: self.lo.bitcast_i64x4(),
                hi: self.hi.bitcast_i64x4(),
            }
        }

        /// Reinterpret bits as `&i64x8` (zero-cost pointer cast).
        #[inline(always)]
        pub fn bitcast_ref_i64x8(&self) -> &i64x8 {
            unsafe { &*(self as *const Self as *const i64x8) }
        }

        /// Reinterpret bits as `&mut i64x8` (zero-cost pointer cast).
        #[inline(always)]
        pub fn bitcast_mut_i64x8(&mut self) -> &mut i64x8 {
            unsafe { &mut *(self as *mut Self as *mut i64x8) }
        }
    }

    impl Add for u64x8 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                lo: self.lo + rhs.lo,
                hi: self.hi + rhs.hi,
            }
        }
    }

    impl Sub for u64x8 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                lo: self.lo - rhs.lo,
                hi: self.hi - rhs.hi,
            }
        }
    }

    impl core::ops::AddAssign for u64x8 {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl core::ops::SubAssign for u64x8 {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    // Width-aliased type names
    pub type f32xN = f32x16;
    pub type f64xN = f64x8;
    pub type i8xN = i8x64;
    pub type u8xN = u8x64;
    pub type i16xN = i16x32;
    pub type u16xN = u16x32;
    pub type i32xN = i32x16;
    pub type u32xN = u32x16;
    pub type i64xN = i64x8;
    pub type u64xN = u64x8;

    /// Number of f32 lanes
    pub const LANES_F32: usize = 16;
    /// Number of f64 lanes
    pub const LANES_F64: usize = 8;
    /// Number of i32/u32 lanes
    pub const LANES_32: usize = 16;
    /// Number of i16/u16 lanes
    pub const LANES_16: usize = 32;
    /// Number of i8/u8 lanes
    pub const LANES_8: usize = 64;

    /// Token type for this polyfill level
    pub type Token = archmage::X64V3Token;
}

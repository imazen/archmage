//! Scalar polyfill types for `#[magetypes]` fallback.
//!
//! These types wrap single values but provide the same API as SIMD types,
//! allowing `#[magetypes]`-generated code to compile for the scalar fallback.

#![allow(non_camel_case_types)]

use archmage::ScalarToken;
use core::ops::{Add, Div, Mul, Neg, Sub};

/// Scalar f32 with SIMD-compatible API.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct f32x1(pub f32);

impl f32x1 {
    /// Number of lanes (always 1 for scalar).
    pub const LANES: usize = 1;

    /// Broadcast a scalar value.
    #[inline(always)]
    pub fn splat(_: ScalarToken, v: f32) -> Self {
        Self(v)
    }

    /// Zero vector.
    #[inline(always)]
    pub fn zero(_: ScalarToken) -> Self {
        Self(0.0)
    }

    /// Load from array.
    #[inline(always)]
    pub fn load(_: ScalarToken, data: &[f32; 1]) -> Self {
        Self(data[0])
    }

    /// Create from array.
    #[inline(always)]
    pub fn from_array(_: ScalarToken, arr: [f32; 1]) -> Self {
        Self(arr[0])
    }

    /// Convert to array.
    #[inline(always)]
    pub fn to_array(self) -> [f32; 1] {
        [self.0]
    }

    /// Store to array.
    #[inline(always)]
    pub fn store(self, out: &mut [f32; 1]) {
        out[0] = self.0;
    }

    /// Sum of all lanes.
    #[inline(always)]
    pub fn reduce_add(self) -> f32 {
        self.0
    }

    /// Minimum across all lanes.
    #[inline(always)]
    pub fn reduce_min(self) -> f32 {
        self.0
    }

    /// Maximum across all lanes.
    #[inline(always)]
    pub fn reduce_max(self) -> f32 {
        self.0
    }

    /// Element-wise minimum.
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(self.0.min(other.0))
    }

    /// Element-wise maximum.
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(self.0.max(other.0))
    }

    /// Element-wise clamp.
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        Self(self.0.clamp(lo.0, hi.0))
    }

    /// Element-wise square root.
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        Self(self.0.sqrt())
    }

    /// Element-wise absolute value.
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(self.0.abs())
    }

    /// Element-wise floor.
    #[inline(always)]
    pub fn floor(self) -> Self {
        Self(self.0.floor())
    }

    /// Element-wise ceiling.
    #[inline(always)]
    pub fn ceil(self) -> Self {
        Self(self.0.ceil())
    }

    /// Element-wise round to nearest.
    #[inline(always)]
    pub fn round(self) -> Self {
        Self(self.0.round())
    }

    /// Fused multiply-add: `self * b + c`.
    #[inline(always)]
    pub fn mul_add(self, b: Self, c: Self) -> Self {
        Self(self.0.mul_add(b.0, c.0))
    }

    /// Fused multiply-subtract: `self * b - c`.
    #[inline(always)]
    pub fn mul_sub(self, b: Self, c: Self) -> Self {
        Self(self.0.mul_add(b.0, -c.0))
    }
}

impl Add for f32x1 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Sub for f32x1 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl Mul for f32x1 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(self.0 * rhs.0)
    }
}

impl Div for f32x1 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        Self(self.0 / rhs.0)
    }
}

impl Neg for f32x1 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

/// Scalar f64 with SIMD-compatible API.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct f64x1(pub f64);

impl f64x1 {
    /// Number of lanes (always 1 for scalar).
    pub const LANES: usize = 1;

    /// Broadcast a scalar value.
    #[inline(always)]
    pub fn splat(_: ScalarToken, v: f64) -> Self {
        Self(v)
    }

    /// Zero vector.
    #[inline(always)]
    pub fn zero(_: ScalarToken) -> Self {
        Self(0.0)
    }

    /// Load from array.
    #[inline(always)]
    pub fn load(_: ScalarToken, data: &[f64; 1]) -> Self {
        Self(data[0])
    }

    /// Create from array.
    #[inline(always)]
    pub fn from_array(_: ScalarToken, arr: [f64; 1]) -> Self {
        Self(arr[0])
    }

    /// Convert to array.
    #[inline(always)]
    pub fn to_array(self) -> [f64; 1] {
        [self.0]
    }

    /// Store to array.
    #[inline(always)]
    pub fn store(self, out: &mut [f64; 1]) {
        out[0] = self.0;
    }

    /// Sum of all lanes.
    #[inline(always)]
    pub fn reduce_add(self) -> f64 {
        self.0
    }

    /// Element-wise square root.
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        Self(self.0.sqrt())
    }

    /// Element-wise absolute value.
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(self.0.abs())
    }

    /// Fused multiply-add: `self * b + c`.
    #[inline(always)]
    pub fn mul_add(self, b: Self, c: Self) -> Self {
        Self(self.0.mul_add(b.0, c.0))
    }
}

impl Add for f64x1 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Sub for f64x1 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl Mul for f64x1 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(self.0 * rhs.0)
    }
}

impl Div for f64x1 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        Self(self.0 / rhs.0)
    }
}

impl Neg for f64x1 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

/// Scalar i32 with SIMD-compatible API.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct i32x1(pub i32);

impl i32x1 {
    /// Number of lanes (always 1 for scalar).
    pub const LANES: usize = 1;

    /// Broadcast a scalar value.
    #[inline(always)]
    pub fn splat(_: ScalarToken, v: i32) -> Self {
        Self(v)
    }

    /// Zero vector.
    #[inline(always)]
    pub fn zero(_: ScalarToken) -> Self {
        Self(0)
    }

    /// Create from array.
    #[inline(always)]
    pub fn from_array(_: ScalarToken, arr: [i32; 1]) -> Self {
        Self(arr[0])
    }

    /// Convert to array.
    #[inline(always)]
    pub fn to_array(self) -> [i32; 1] {
        [self.0]
    }

    /// Sum of all lanes.
    #[inline(always)]
    pub fn reduce_add(self) -> i32 {
        self.0
    }
}

impl Add for i32x1 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(self.0.wrapping_add(rhs.0))
    }
}

impl Sub for i32x1 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0.wrapping_sub(rhs.0))
    }
}

impl Mul for i32x1 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(self.0.wrapping_mul(rhs.0))
    }
}

/// Scalar u32 with SIMD-compatible API.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct u32x1(pub u32);

impl u32x1 {
    /// Number of lanes (always 1 for scalar).
    pub const LANES: usize = 1;

    /// Broadcast a scalar value.
    #[inline(always)]
    pub fn splat(_: ScalarToken, v: u32) -> Self {
        Self(v)
    }

    /// Zero vector.
    #[inline(always)]
    pub fn zero(_: ScalarToken) -> Self {
        Self(0)
    }

    /// Create from array.
    #[inline(always)]
    pub fn from_array(_: ScalarToken, arr: [u32; 1]) -> Self {
        Self(arr[0])
    }

    /// Convert to array.
    #[inline(always)]
    pub fn to_array(self) -> [u32; 1] {
        [self.0]
    }

    /// Sum of all lanes.
    #[inline(always)]
    pub fn reduce_add(self) -> u32 {
        self.0
    }
}

impl Add for u32x1 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(self.0.wrapping_add(rhs.0))
    }
}

impl Sub for u32x1 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0.wrapping_sub(rhs.0))
    }
}

impl Mul for u32x1 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(self.0.wrapping_mul(rhs.0))
    }
}

/// Scalar i8 with SIMD-compatible API.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct i8x1(pub i8);

impl i8x1 {
    /// Number of lanes (always 1 for scalar).
    pub const LANES: usize = 1;

    /// Broadcast a scalar value.
    #[inline(always)]
    pub fn splat(_: ScalarToken, v: i8) -> Self {
        Self(v)
    }

    /// Zero vector.
    #[inline(always)]
    pub fn zero(_: ScalarToken) -> Self {
        Self(0)
    }

    /// Create from array.
    #[inline(always)]
    pub fn from_array(_: ScalarToken, arr: [i8; 1]) -> Self {
        Self(arr[0])
    }

    /// Convert to array.
    #[inline(always)]
    pub fn to_array(self) -> [i8; 1] {
        [self.0]
    }
}

impl Add for i8x1 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(self.0.wrapping_add(rhs.0))
    }
}

impl Sub for i8x1 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0.wrapping_sub(rhs.0))
    }
}

/// Scalar u8 with SIMD-compatible API.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct u8x1(pub u8);

impl u8x1 {
    /// Number of lanes (always 1 for scalar).
    pub const LANES: usize = 1;

    /// Broadcast a scalar value.
    #[inline(always)]
    pub fn splat(_: ScalarToken, v: u8) -> Self {
        Self(v)
    }

    /// Zero vector.
    #[inline(always)]
    pub fn zero(_: ScalarToken) -> Self {
        Self(0)
    }

    /// Create from array.
    #[inline(always)]
    pub fn from_array(_: ScalarToken, arr: [u8; 1]) -> Self {
        Self(arr[0])
    }

    /// Convert to array.
    #[inline(always)]
    pub fn to_array(self) -> [u8; 1] {
        [self.0]
    }
}

impl Add for u8x1 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(self.0.wrapping_add(rhs.0))
    }
}

impl Sub for u8x1 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0.wrapping_sub(rhs.0))
    }
}

/// Scalar i16 with SIMD-compatible API.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct i16x1(pub i16);

impl i16x1 {
    /// Number of lanes (always 1 for scalar).
    pub const LANES: usize = 1;

    /// Broadcast a scalar value.
    #[inline(always)]
    pub fn splat(_: ScalarToken, v: i16) -> Self {
        Self(v)
    }

    /// Zero vector.
    #[inline(always)]
    pub fn zero(_: ScalarToken) -> Self {
        Self(0)
    }

    /// Create from array.
    #[inline(always)]
    pub fn from_array(_: ScalarToken, arr: [i16; 1]) -> Self {
        Self(arr[0])
    }

    /// Convert to array.
    #[inline(always)]
    pub fn to_array(self) -> [i16; 1] {
        [self.0]
    }
}

impl Add for i16x1 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(self.0.wrapping_add(rhs.0))
    }
}

impl Sub for i16x1 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0.wrapping_sub(rhs.0))
    }
}

/// Scalar u16 with SIMD-compatible API.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct u16x1(pub u16);

impl u16x1 {
    /// Number of lanes (always 1 for scalar).
    pub const LANES: usize = 1;

    /// Broadcast a scalar value.
    #[inline(always)]
    pub fn splat(_: ScalarToken, v: u16) -> Self {
        Self(v)
    }

    /// Zero vector.
    #[inline(always)]
    pub fn zero(_: ScalarToken) -> Self {
        Self(0)
    }

    /// Create from array.
    #[inline(always)]
    pub fn from_array(_: ScalarToken, arr: [u16; 1]) -> Self {
        Self(arr[0])
    }

    /// Convert to array.
    #[inline(always)]
    pub fn to_array(self) -> [u16; 1] {
        [self.0]
    }
}

impl Add for u16x1 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(self.0.wrapping_add(rhs.0))
    }
}

impl Sub for u16x1 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0.wrapping_sub(rhs.0))
    }
}

/// Scalar i64 with SIMD-compatible API.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct i64x1(pub i64);

impl i64x1 {
    /// Number of lanes (always 1 for scalar).
    pub const LANES: usize = 1;

    /// Broadcast a scalar value.
    #[inline(always)]
    pub fn splat(_: ScalarToken, v: i64) -> Self {
        Self(v)
    }

    /// Zero vector.
    #[inline(always)]
    pub fn zero(_: ScalarToken) -> Self {
        Self(0)
    }

    /// Create from array.
    #[inline(always)]
    pub fn from_array(_: ScalarToken, arr: [i64; 1]) -> Self {
        Self(arr[0])
    }

    /// Convert to array.
    #[inline(always)]
    pub fn to_array(self) -> [i64; 1] {
        [self.0]
    }
}

impl Add for i64x1 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(self.0.wrapping_add(rhs.0))
    }
}

impl Sub for i64x1 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0.wrapping_sub(rhs.0))
    }
}

/// Scalar u64 with SIMD-compatible API.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct u64x1(pub u64);

impl u64x1 {
    /// Number of lanes (always 1 for scalar).
    pub const LANES: usize = 1;

    /// Broadcast a scalar value.
    #[inline(always)]
    pub fn splat(_: ScalarToken, v: u64) -> Self {
        Self(v)
    }

    /// Zero vector.
    #[inline(always)]
    pub fn zero(_: ScalarToken) -> Self {
        Self(0)
    }

    /// Create from array.
    #[inline(always)]
    pub fn from_array(_: ScalarToken, arr: [u64; 1]) -> Self {
        Self(arr[0])
    }

    /// Convert to array.
    #[inline(always)]
    pub fn to_array(self) -> [u64; 1] {
        [self.0]
    }
}

impl Add for u64x1 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(self.0.wrapping_add(rhs.0))
    }
}

impl Sub for u64x1 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0.wrapping_sub(rhs.0))
    }
}

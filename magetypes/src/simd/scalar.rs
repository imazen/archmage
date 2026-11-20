//! **Superseded (queued for deprecation):** Use generic vectors bound to
//! [`ScalarToken`] (for example [`f32x8<ScalarToken>`](crate::simd::generic::f32x8))
//! for the scalar tier, and plain Rust arithmetic or [`crate::scalar`] (or
//! `magemath` `*_with_token`) for scalar remainder loops.
//!
//! Single-lane scalar wrapper types retained for backwards compatibility.

#![allow(non_camel_case_types)]

use archmage::ScalarToken;
use core::ops::{Add, Div, Mul, Neg, Sub};

/// **Superseded (queued for deprecation):** Use [`f32x8<ScalarToken>`](crate::simd::generic::f32x8) for the scalar tier, or plain `f32` / [`crate::scalar`] (and `magemath` `*_with_token`) for scalar tails.
///
/// Scalar `f32` with SIMD-compatible API.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct f32x1(pub f32);

impl f32x1 {
    /// Number of lanes (always 1 for scalar).
    pub const LANES: usize = 1;

    /// Broadcast a scalar value.
    #[inline(always)]
    pub fn splat_t(_: ScalarToken, v: f32) -> Self {
        Self(v)
    }

    /// Zero vector.
    #[inline(always)]
    pub fn zero_t(_: ScalarToken) -> Self {
        Self(0.0)
    }

    /// Load from array.
    #[inline(always)]
    pub fn load_t(_: ScalarToken, data: &[f32; 1]) -> Self {
        Self(data[0])
    }

    /// Create from array.
    #[inline(always)]
    pub fn from_array_t(_: ScalarToken, arr: [f32; 1]) -> Self {
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

    /// Element-wise minimum (matches [`f32::min`]).
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(self.0.min(other.0))
    }

    /// Element-wise maximum (matches [`f32::max`]).
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(self.0.max(other.0))
    }

    /// IEEE 754-2019 lane-wise minimum (propagates `NaN` and orders `-0.0 < +0.0`).
    #[inline(always)]
    pub fn minimum(self, other: Self) -> Self {
        Self(crate::nostd_math::minimumf(self.0, other.0))
    }

    /// IEEE 754-2019 lane-wise maximum (propagates `NaN` and orders `-0.0 < +0.0`).
    #[inline(always)]
    pub fn maximum(self, other: Self) -> Self {
        Self(crate::nostd_math::maximumf(self.0, other.0))
    }

    /// Fast lane-wise minimum ignoring `NaN` and signed-zero ordering.
    #[inline(always)]
    pub fn min_fast(self, other: Self) -> Self {
        Self(if self.0 < other.0 { self.0 } else { other.0 })
    }

    /// Fast lane-wise maximum ignoring `NaN` and signed-zero ordering.
    #[inline(always)]
    pub fn max_fast(self, other: Self) -> Self {
        Self(if self.0 > other.0 { self.0 } else { other.0 })
    }

    /// Fast truncation toward zero to `i32x1`.
    #[inline(always)]
    pub fn to_i32_fast(self) -> i32x1 {
        i32x1(self.0 as i32)
    }

    /// Element-wise clamp (matches [`f32::clamp`]).
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        Self(self.0.clamp(lo.0, hi.0))
    }

    /// Element-wise square root.
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        Self(crate::nostd_math::sqrtf(self.0))
    }

    /// Element-wise absolute value.
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(f32::from_bits(self.0.to_bits() & 0x7FFF_FFFF))
    }

    /// Element-wise floor.
    #[inline(always)]
    pub fn floor(self) -> Self {
        Self(crate::nostd_math::floorf(self.0))
    }

    /// Element-wise ceiling.
    #[inline(always)]
    pub fn ceil(self) -> Self {
        Self(crate::nostd_math::ceilf(self.0))
    }

    /// Element-wise round to nearest (ties to even, matching hardware SIMD).
    #[inline(always)]
    pub fn round(self) -> Self {
        Self(crate::nostd_math::roundevenf(self.0))
    }

    /// Multiply-add: `self * b + c`, computed as a multiply then an add (two
    /// roundings), like the scalar backend's vector `mul_add_relaxed`.
    ///
    /// For one rounding, use [`mul_add_fused`](Self::mul_add_fused).
    #[inline(always)]
    pub fn mul_add_relaxed(self, b: Self, c: Self) -> Self {
        Self(self.0 * b.0 + c.0)
    }

    /// Multiply-subtract: `self * b - c`. Same contract as
    /// [`mul_add_relaxed`](Self::mul_add_relaxed).
    #[inline(always)]
    pub fn mul_sub_relaxed(self, b: Self, c: Self) -> Self {
        Self(self.0 * b.0 - c.0)
    }

    /// Multiply-add with one rounding: `self * b + c`, the FMA instruction
    /// where the build guarantees one and correctly rounded software
    /// otherwise. NaN payload/sign are unspecified.
    #[doc(alias = "fma", alias = "fmadd", alias = "fmla")]
    #[inline(always)]
    pub fn mul_add_fused(self, b: Self, c: Self) -> Self {
        Self(crate::nostd_math::fmaf(self.0, b.0, c.0))
    }

    /// Multiply-subtract with one rounding: `self * b - c`. Same contract as
    /// [`mul_add_fused`](Self::mul_add_fused).
    #[doc(alias = "fms", alias = "fmsub", alias = "fmls")]
    #[inline(always)]
    pub fn mul_sub_fused(self, b: Self, c: Self) -> Self {
        Self(crate::nostd_math::fmaf(self.0, b.0, -c.0))
    }

    /// The same operation as [`mul_add_fused`](Self::mul_add_fused).
    #[inline(always)]
    pub fn mul_add_portable(self, b: Self, c: Self) -> Self {
        self.mul_add_fused(b, c)
    }

    /// The same operation as [`mul_sub_fused`](Self::mul_sub_fused).
    #[inline(always)]
    pub fn mul_sub_portable(self, b: Self, c: Self) -> Self {
        self.mul_sub_fused(b, c)
    }

    /// Multiply-add as a multiply then an add. Deprecated so that the
    /// rounding is an explicit choice: [`mul_add_relaxed`](Self::mul_add_relaxed)
    /// keeps this behavior, [`mul_add_fused`](Self::mul_add_fused) rounds once.
    #[deprecated(
        since = "0.9.31",
        note = "use mul_add_relaxed (today's behavior: multiply then add here) or mul_add_fused (one rounding); mul_add becomes the fused operation in magetypes 0.11"
    )]
    #[inline(always)]
    pub fn mul_add(self, b: Self, c: Self) -> Self {
        self.mul_add_relaxed(b, c)
    }

    /// Multiply-subtract as a multiply then a subtract. Deprecated like
    /// [`mul_add`](Self::mul_add).
    #[deprecated(
        since = "0.9.31",
        note = "use mul_sub_relaxed (today's behavior: multiply then subtract here) or mul_sub_fused (one rounding); mul_sub becomes the fused operation in magetypes 0.11"
    )]
    #[inline(always)]
    pub fn mul_sub(self, b: Self, c: Self) -> Self {
        self.mul_sub_relaxed(b, c)
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

/// **Superseded (queued for deprecation):** Use [`f64x4<ScalarToken>`](crate::simd::generic::f64x4) for the scalar tier, or plain `f64` / [`crate::scalar`] for scalar tails.
///
/// Scalar `f64` with SIMD-compatible API.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct f64x1(pub f64);

impl f64x1 {
    /// Number of lanes (always 1 for scalar).
    pub const LANES: usize = 1;

    /// Broadcast a scalar value.
    #[inline(always)]
    pub fn splat_t(_: ScalarToken, v: f64) -> Self {
        Self(v)
    }

    /// Zero vector.
    #[inline(always)]
    pub fn zero_t(_: ScalarToken) -> Self {
        Self(0.0)
    }

    /// Load from array.
    #[inline(always)]
    pub fn load_t(_: ScalarToken, data: &[f64; 1]) -> Self {
        Self(data[0])
    }

    /// Create from array.
    #[inline(always)]
    pub fn from_array_t(_: ScalarToken, arr: [f64; 1]) -> Self {
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

    /// Minimum across all lanes.
    #[inline(always)]
    pub fn reduce_min(self) -> f64 {
        self.0
    }

    /// Maximum across all lanes.
    #[inline(always)]
    pub fn reduce_max(self) -> f64 {
        self.0
    }

    /// Element-wise minimum (matches [`f64::min`]).
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(self.0.min(other.0))
    }

    /// Element-wise maximum (matches [`f64::max`]).
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(self.0.max(other.0))
    }

    /// IEEE 754-2019 lane-wise minimum (propagates `NaN` and orders `-0.0 < +0.0`).
    #[inline(always)]
    pub fn minimum(self, other: Self) -> Self {
        Self(crate::nostd_math::minimum(self.0, other.0))
    }

    /// IEEE 754-2019 lane-wise maximum (propagates `NaN` and orders `-0.0 < +0.0`).
    #[inline(always)]
    pub fn maximum(self, other: Self) -> Self {
        Self(crate::nostd_math::maximum(self.0, other.0))
    }

    /// Fast lane-wise minimum ignoring `NaN` and signed-zero ordering.
    #[inline(always)]
    pub fn min_fast(self, other: Self) -> Self {
        Self(if self.0 < other.0 { self.0 } else { other.0 })
    }

    /// Fast lane-wise maximum ignoring `NaN` and signed-zero ordering.
    #[inline(always)]
    pub fn max_fast(self, other: Self) -> Self {
        Self(if self.0 > other.0 { self.0 } else { other.0 })
    }

    /// Element-wise clamp matching `f64x2::<ScalarToken>::clamp`: a quiet
    /// `NaN` input becomes `lo`, and it never panics (unlike [`f64::clamp`]).
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max_fast(lo).min_fast(hi)
    }

    /// Element-wise square root.
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        Self(crate::nostd_math::sqrt(self.0))
    }

    /// Element-wise absolute value.
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(f64::from_bits(self.0.to_bits() & 0x7FFF_FFFF_FFFF_FFFF))
    }

    /// Multiply-add: `self * b + c`, computed as a multiply then an add (two
    /// roundings), like the scalar backend's vector `mul_add_relaxed`.
    ///
    /// For one rounding, use [`mul_add_fused`](Self::mul_add_fused).
    #[inline(always)]
    pub fn mul_add_relaxed(self, b: Self, c: Self) -> Self {
        Self(self.0 * b.0 + c.0)
    }

    /// Multiply-add with one rounding: `self * b + c`, the FMA instruction
    /// where the build guarantees one or (x86_64 with std) the CPU has one,
    /// correctly rounded software otherwise. NaN payload/sign are unspecified.
    #[doc(alias = "fma", alias = "fmadd", alias = "fmla")]
    #[inline(always)]
    pub fn mul_add_fused(self, b: Self, c: Self) -> Self {
        Self(crate::nostd_math::fma(self.0, b.0, c.0))
    }

    /// The same operation as [`mul_add_fused`](Self::mul_add_fused).
    #[inline(always)]
    pub fn mul_add_portable(self, b: Self, c: Self) -> Self {
        self.mul_add_fused(b, c)
    }

    /// Multiply-add as a multiply then an add. Deprecated so that the
    /// rounding is an explicit choice: [`mul_add_relaxed`](Self::mul_add_relaxed)
    /// keeps this behavior, [`mul_add_fused`](Self::mul_add_fused) rounds once.
    #[deprecated(
        since = "0.9.31",
        note = "use mul_add_relaxed (today's behavior: multiply then add here) or mul_add_fused (one rounding); mul_add becomes the fused operation in magetypes 0.11"
    )]
    #[inline(always)]
    pub fn mul_add(self, b: Self, c: Self) -> Self {
        self.mul_add_relaxed(b, c)
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

/// **Superseded (queued for deprecation):** Use [`i32x8<ScalarToken>`](crate::simd::generic::i32x8) for the scalar tier, or plain `i32` / [`crate::scalar`] for scalar tails.
///
/// Scalar `i32` with SIMD-compatible API.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct i32x1(pub i32);

impl i32x1 {
    /// Number of lanes (always 1 for scalar).
    pub const LANES: usize = 1;

    /// Broadcast a scalar value.
    #[inline(always)]
    pub fn splat_t(_: ScalarToken, v: i32) -> Self {
        Self(v)
    }

    /// Zero vector.
    #[inline(always)]
    pub fn zero_t(_: ScalarToken) -> Self {
        Self(0)
    }

    /// Create from array.
    #[inline(always)]
    pub fn from_array_t(_: ScalarToken, arr: [i32; 1]) -> Self {
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

/// **Superseded (queued for deprecation):** Use [`u32x8<ScalarToken>`](crate::simd::generic::u32x8) for the scalar tier, or plain `u32` for scalar tails.
///
/// Scalar `u32` with SIMD-compatible API.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct u32x1(pub u32);

impl u32x1 {
    /// Number of lanes (always 1 for scalar).
    pub const LANES: usize = 1;

    /// Broadcast a scalar value.
    #[inline(always)]
    pub fn splat_t(_: ScalarToken, v: u32) -> Self {
        Self(v)
    }

    /// Zero vector.
    #[inline(always)]
    pub fn zero_t(_: ScalarToken) -> Self {
        Self(0)
    }

    /// Create from array.
    #[inline(always)]
    pub fn from_array_t(_: ScalarToken, arr: [u32; 1]) -> Self {
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

/// **Superseded (queued for deprecation):** Use [`i8x16<ScalarToken>`](crate::simd::generic::i8x16) for the scalar tier, or plain `i8` for scalar tails.
///
/// Scalar `i8` with SIMD-compatible API.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct i8x1(pub i8);

impl i8x1 {
    /// Number of lanes (always 1 for scalar).
    pub const LANES: usize = 1;

    /// Broadcast a scalar value.
    #[inline(always)]
    pub fn splat_t(_: ScalarToken, v: i8) -> Self {
        Self(v)
    }

    /// Zero vector.
    #[inline(always)]
    pub fn zero_t(_: ScalarToken) -> Self {
        Self(0)
    }

    /// Create from array.
    #[inline(always)]
    pub fn from_array_t(_: ScalarToken, arr: [i8; 1]) -> Self {
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

/// **Superseded (queued for deprecation):** Use [`u8x16<ScalarToken>`](crate::simd::generic::u8x16) for the scalar tier, or plain `u8` for scalar tails.
///
/// Scalar `u8` with SIMD-compatible API.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct u8x1(pub u8);

impl u8x1 {
    /// Number of lanes (always 1 for scalar).
    pub const LANES: usize = 1;

    /// Broadcast a scalar value.
    #[inline(always)]
    pub fn splat_t(_: ScalarToken, v: u8) -> Self {
        Self(v)
    }

    /// Zero vector.
    #[inline(always)]
    pub fn zero_t(_: ScalarToken) -> Self {
        Self(0)
    }

    /// Create from array.
    #[inline(always)]
    pub fn from_array_t(_: ScalarToken, arr: [u8; 1]) -> Self {
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

/// **Superseded (queued for deprecation):** Use [`i16x8<ScalarToken>`](crate::simd::generic::i16x8) for the scalar tier, or plain `i16` for scalar tails.
///
/// Scalar `i16` with SIMD-compatible API.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct i16x1(pub i16);

impl i16x1 {
    /// Number of lanes (always 1 for scalar).
    pub const LANES: usize = 1;

    /// Broadcast a scalar value.
    #[inline(always)]
    pub fn splat_t(_: ScalarToken, v: i16) -> Self {
        Self(v)
    }

    /// Zero vector.
    #[inline(always)]
    pub fn zero_t(_: ScalarToken) -> Self {
        Self(0)
    }

    /// Create from array.
    #[inline(always)]
    pub fn from_array_t(_: ScalarToken, arr: [i16; 1]) -> Self {
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

/// **Superseded (queued for deprecation):** Use [`u16x8<ScalarToken>`](crate::simd::generic::u16x8) for the scalar tier, or plain `u16` for scalar tails.
///
/// Scalar `u16` with SIMD-compatible API.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct u16x1(pub u16);

impl u16x1 {
    /// Number of lanes (always 1 for scalar).
    pub const LANES: usize = 1;

    /// Broadcast a scalar value.
    #[inline(always)]
    pub fn splat_t(_: ScalarToken, v: u16) -> Self {
        Self(v)
    }

    /// Zero vector.
    #[inline(always)]
    pub fn zero_t(_: ScalarToken) -> Self {
        Self(0)
    }

    /// Create from array.
    #[inline(always)]
    pub fn from_array_t(_: ScalarToken, arr: [u16; 1]) -> Self {
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

/// **Superseded (queued for deprecation):** Use [`i64x4<ScalarToken>`](crate::simd::generic::i64x4) for the scalar tier, or plain `i64` for scalar tails.
///
/// Scalar `i64` with SIMD-compatible API.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct i64x1(pub i64);

impl i64x1 {
    /// Number of lanes (always 1 for scalar).
    pub const LANES: usize = 1;

    /// Broadcast a scalar value.
    #[inline(always)]
    pub fn splat_t(_: ScalarToken, v: i64) -> Self {
        Self(v)
    }

    /// Zero vector.
    #[inline(always)]
    pub fn zero_t(_: ScalarToken) -> Self {
        Self(0)
    }

    /// Create from array.
    #[inline(always)]
    pub fn from_array_t(_: ScalarToken, arr: [i64; 1]) -> Self {
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

/// **Superseded (queued for deprecation):** Use [`u64x4<ScalarToken>`](crate::simd::generic::u64x4) for the scalar tier, or plain `u64` for scalar tails.
///
/// Scalar `u64` with SIMD-compatible API.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct u64x1(pub u64);

impl u64x1 {
    /// Number of lanes (always 1 for scalar).
    pub const LANES: usize = 1;

    /// Broadcast a scalar value.
    #[inline(always)]
    pub fn splat_t(_: ScalarToken, v: u64) -> Self {
        Self(v)
    }

    /// Zero vector.
    #[inline(always)]
    pub fn zero_t(_: ScalarToken) -> Self {
        Self(0)
    }

    /// Create from array.
    #[inline(always)]
    pub fn from_array_t(_: ScalarToken, arr: [u64; 1]) -> Self {
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

include!("generic/generated/scalar_token_aliases.rs");

impl f32x1 {
    /// Truncate toward zero, retaining the sign of zero.
    #[inline(always)]
    pub fn trunc(self) -> Self {
        Self(crate::scalar::trunc_f32(self.0))
    }

    /// Compute `c - self * b` using separate rounding.
    #[inline(always)]
    pub fn mul_neg_add_relaxed(self, b: Self, c: Self) -> Self {
        Self((-self.0) * b.0 + c.0)
    }

    /// Compute `c - self * b` with one rounding.
    #[inline(always)]
    pub fn mul_neg_add_fused(self, b: Self, c: Self) -> Self {
        Self(crate::scalar::mul_neg_add_fused_f32(self.0, b.0, c.0))
    }
}

impl f64x1 {
    /// Truncate toward zero, retaining the sign of zero.
    #[inline(always)]
    pub fn trunc(self) -> Self {
        Self(crate::scalar::trunc_f64(self.0))
    }

    /// Compute `c - self * b` using separate rounding.
    #[inline(always)]
    pub fn mul_neg_add_relaxed(self, b: Self, c: Self) -> Self {
        Self((-self.0) * b.0 + c.0)
    }

    /// Compute `c - self * b` with one rounding.
    #[inline(always)]
    pub fn mul_neg_add_fused(self, b: Self, c: Self) -> Self {
        Self(crate::scalar::mul_neg_add_fused_f64(self.0, b.0, c.0))
    }
}

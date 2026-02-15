//! Generic `f32x8<T>` — 8-lane f32 SIMD vector parameterized by backend.
//!
//! `T` is a token type (e.g., `X64V3Token`, `NeonToken`, `ScalarToken`)
//! that determines the platform-native representation and intrinsics used.
//! The struct delegates all operations to the [`F32x8Backend`] trait.
//!
//! # Example
//!
//! ```ignore
//! use magetypes::simd::backends::{F32x8Backend, x64v3};
//! use magetypes::simd::generic::f32x8;
//!
//! fn sum<T: F32x8Backend>(token: T, data: &[f32]) -> f32 {
//!     let mut acc = f32x8::<T>::zero(token);
//!     for chunk in data.chunks_exact(8) {
//!         acc = acc + f32x8::<T>::load(token, chunk.try_into().unwrap());
//!     }
//!     acc.reduce_add()
//! }
//! ```

#![allow(clippy::should_implement_trait)]

use core::marker::PhantomData;
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use crate::simd::backends::F32x8Backend;

/// 8-lane f32 SIMD vector, generic over backend `T`.
///
/// `T` is a token type that proves CPU support for the required SIMD features.
/// The inner representation is `T::Repr` (e.g., `__m256` on AVX2, `[f32; 8]` on scalar).
///
/// Construction requires a token value to prove CPU support at runtime.
/// After construction, operations don't need the token — it's baked into the type.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct f32x8<T: F32x8Backend>(T::Repr, PhantomData<T>);

// PhantomData is ZST, so f32x8<T> has the same size as T::Repr.

impl<T: F32x8Backend> f32x8<T> {
    /// Number of f32 lanes.
    pub const LANES: usize = 8;

    // ====== Construction (token-gated) ======

    /// Broadcast scalar to all 8 lanes.
    #[inline(always)]
    pub fn splat(_: T, v: f32) -> Self {
        Self(T::splat(v), PhantomData)
    }

    /// All lanes zero.
    #[inline(always)]
    pub fn zero(_: T) -> Self {
        Self(T::zero(), PhantomData)
    }

    /// Load from a `[f32; 8]` array.
    #[inline(always)]
    pub fn load(_: T, data: &[f32; 8]) -> Self {
        Self(T::load(data), PhantomData)
    }

    /// Create from array (zero-cost where possible).
    #[inline(always)]
    pub fn from_array(_: T, arr: [f32; 8]) -> Self {
        Self(T::from_array(arr), PhantomData)
    }

    /// Create from slice. Panics if `slice.len() < 8`.
    #[inline(always)]
    pub fn from_slice(_: T, slice: &[f32]) -> Self {
        let arr: [f32; 8] = slice[..8].try_into().unwrap();
        Self(T::from_array(arr), PhantomData)
    }

    // ====== Accessors ======

    /// Store to array.
    #[inline(always)]
    pub fn store(self, out: &mut [f32; 8]) {
        T::store(self.0, out);
    }

    /// Convert to array.
    #[inline(always)]
    pub fn to_array(self) -> [f32; 8] {
        T::to_array(self.0)
    }

    /// Get the underlying platform representation.
    #[inline(always)]
    pub fn into_repr(self) -> T::Repr {
        self.0
    }

    /// Wrap a platform representation (token-gated).
    #[inline(always)]
    pub fn from_repr(_: T, repr: T::Repr) -> Self {
        Self(repr, PhantomData)
    }

    // ====== Math ======

    /// Lane-wise minimum.
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(T::min(self.0, other.0), PhantomData)
    }

    /// Lane-wise maximum.
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(T::max(self.0, other.0), PhantomData)
    }

    /// Clamp between lo and hi.
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        Self(T::clamp(self.0, lo.0, hi.0), PhantomData)
    }

    /// Square root.
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        Self(T::sqrt(self.0), PhantomData)
    }

    /// Absolute value.
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(T::abs(self.0), PhantomData)
    }

    /// Round toward negative infinity.
    #[inline(always)]
    pub fn floor(self) -> Self {
        Self(T::floor(self.0), PhantomData)
    }

    /// Round toward positive infinity.
    #[inline(always)]
    pub fn ceil(self) -> Self {
        Self(T::ceil(self.0), PhantomData)
    }

    /// Round to nearest integer.
    #[inline(always)]
    pub fn round(self) -> Self {
        Self(T::round(self.0), PhantomData)
    }

    /// Fused multiply-add: `self * a + b`.
    #[inline(always)]
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        Self(T::mul_add(self.0, a.0, b.0), PhantomData)
    }

    /// Fused multiply-sub: `self * a - b`.
    #[inline(always)]
    pub fn mul_sub(self, a: Self, b: Self) -> Self {
        Self(T::mul_sub(self.0, a.0, b.0), PhantomData)
    }

    // ====== Comparisons ======

    /// Lane-wise equality (returns mask).
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> Self {
        Self(T::simd_eq(self.0, other.0), PhantomData)
    }

    /// Lane-wise inequality (returns mask).
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(T::simd_ne(self.0, other.0), PhantomData)
    }

    /// Lane-wise less-than (returns mask).
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(T::simd_lt(self.0, other.0), PhantomData)
    }

    /// Lane-wise less-than-or-equal (returns mask).
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(T::simd_le(self.0, other.0), PhantomData)
    }

    /// Lane-wise greater-than (returns mask).
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(T::simd_gt(self.0, other.0), PhantomData)
    }

    /// Lane-wise greater-than-or-equal (returns mask).
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(T::simd_ge(self.0, other.0), PhantomData)
    }

    /// Select lanes: where mask is all-1s pick `if_true`, else `if_false`.
    #[inline(always)]
    pub fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self(T::blend(mask.0, if_true.0, if_false.0), PhantomData)
    }

    // ====== Reductions ======

    /// Sum all 8 lanes.
    #[inline(always)]
    pub fn reduce_add(self) -> f32 {
        T::reduce_add(self.0)
    }

    /// Minimum across all 8 lanes.
    #[inline(always)]
    pub fn reduce_min(self) -> f32 {
        T::reduce_min(self.0)
    }

    /// Maximum across all 8 lanes.
    #[inline(always)]
    pub fn reduce_max(self) -> f32 {
        T::reduce_max(self.0)
    }

    // ====== Approximations ======

    /// Fast reciprocal approximation (~12-bit precision).
    #[inline(always)]
    pub fn rcp_approx(self) -> Self {
        Self(T::rcp_approx(self.0), PhantomData)
    }

    /// Precise reciprocal (Newton-Raphson refined).
    #[inline(always)]
    pub fn recip(self) -> Self {
        Self(T::recip(self.0), PhantomData)
    }

    /// Fast reciprocal square root approximation (~12-bit precision).
    #[inline(always)]
    pub fn rsqrt_approx(self) -> Self {
        Self(T::rsqrt_approx(self.0), PhantomData)
    }

    /// Precise reciprocal square root (Newton-Raphson refined).
    #[inline(always)]
    pub fn rsqrt(self) -> Self {
        Self(T::rsqrt(self.0), PhantomData)
    }

    // ====== Bitwise ======

    /// Bitwise NOT.
    #[inline(always)]
    pub fn not(self) -> Self {
        Self(T::not(self.0), PhantomData)
    }
}

// ============================================================================
// Operator implementations
// ============================================================================

impl<T: F32x8Backend> Add for f32x8<T> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(T::add(self.0, rhs.0), PhantomData)
    }
}

impl<T: F32x8Backend> Sub for f32x8<T> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(T::sub(self.0, rhs.0), PhantomData)
    }
}

impl<T: F32x8Backend> Mul for f32x8<T> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(T::mul(self.0, rhs.0), PhantomData)
    }
}

impl<T: F32x8Backend> Div for f32x8<T> {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        Self(T::div(self.0, rhs.0), PhantomData)
    }
}

impl<T: F32x8Backend> Neg for f32x8<T> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(T::neg(self.0), PhantomData)
    }
}

impl<T: F32x8Backend> BitAnd for f32x8<T> {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(T::bitand(self.0, rhs.0), PhantomData)
    }
}

impl<T: F32x8Backend> BitOr for f32x8<T> {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self(T::bitor(self.0, rhs.0), PhantomData)
    }
}

impl<T: F32x8Backend> BitXor for f32x8<T> {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Self(T::bitxor(self.0, rhs.0), PhantomData)
    }
}

// ============================================================================
// Assign operators
// ============================================================================

impl<T: F32x8Backend> AddAssign for f32x8<T> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<T: F32x8Backend> SubAssign for f32x8<T> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<T: F32x8Backend> MulAssign for f32x8<T> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<T: F32x8Backend> DivAssign for f32x8<T> {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<T: F32x8Backend> BitAndAssign for f32x8<T> {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

impl<T: F32x8Backend> BitOrAssign for f32x8<T> {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl<T: F32x8Backend> BitXorAssign for f32x8<T> {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = *self ^ rhs;
    }
}

// ============================================================================
// Scalar broadcast operators (v + 2.0, v * 0.5, etc.)
// ============================================================================

impl<T: F32x8Backend> Add<f32> for f32x8<T> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: f32) -> Self {
        Self(T::add(self.0, T::splat(rhs)), PhantomData)
    }
}

impl<T: F32x8Backend> Sub<f32> for f32x8<T> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: f32) -> Self {
        Self(T::sub(self.0, T::splat(rhs)), PhantomData)
    }
}

impl<T: F32x8Backend> Mul<f32> for f32x8<T> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f32) -> Self {
        Self(T::mul(self.0, T::splat(rhs)), PhantomData)
    }
}

impl<T: F32x8Backend> Div<f32> for f32x8<T> {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: f32) -> Self {
        Self(T::div(self.0, T::splat(rhs)), PhantomData)
    }
}

// ============================================================================
// Index
// ============================================================================

impl<T: F32x8Backend> Index<usize> for f32x8<T> {
    type Output = f32;
    #[inline(always)]
    fn index(&self, i: usize) -> &f32 {
        assert!(i < 8, "f32x8 index out of bounds: {i}");
        // SAFETY: f32x8's repr is layout-compatible with [f32; 8], and i < 8.
        unsafe { &*(core::ptr::from_ref(self).cast::<f32>()).add(i) }
    }
}

impl<T: F32x8Backend> IndexMut<usize> for f32x8<T> {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut f32 {
        assert!(i < 8, "f32x8 index out of bounds: {i}");
        // SAFETY: f32x8's repr is layout-compatible with [f32; 8], and i < 8.
        unsafe { &mut *(core::ptr::from_mut(self).cast::<f32>()).add(i) }
    }
}

// ============================================================================
// Conversions
// ============================================================================

impl<T: F32x8Backend> From<f32x8<T>> for [f32; 8] {
    #[inline(always)]
    fn from(v: f32x8<T>) -> [f32; 8] {
        T::to_array(v.0)
    }
}

// ============================================================================
// Debug
// ============================================================================

impl<T: F32x8Backend> core::fmt::Debug for f32x8<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let arr = T::to_array(self.0);
        f.debug_tuple("f32x8").field(&arr).finish()
    }
}

// ============================================================================
// Platform-specific concrete impls
// ============================================================================

#[cfg(target_arch = "x86_64")]
impl f32x8<archmage::X64V3Token> {
    /// Get the raw `__m256` value.
    #[inline(always)]
    pub fn raw(self) -> core::arch::x86_64::__m256 {
        self.0
    }

    /// Create from a raw `__m256` (token-gated, zero-cost).
    #[inline(always)]
    pub fn from_m256(_: archmage::X64V3Token, v: core::arch::x86_64::__m256) -> Self {
        Self(v, PhantomData)
    }
}

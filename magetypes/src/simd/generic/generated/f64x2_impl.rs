//! Generic `f64x2<T>` — 2-lane f64 SIMD vector parameterized by backend.
//!
//! `T` is a token type (e.g., `X64V3Token`, `NeonToken`, `ScalarToken`)
//! that determines the platform-native representation and intrinsics used.
//! The struct delegates all operations to the [`F64x2Backend`] trait.

#![allow(clippy::should_implement_trait)]

use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use crate::simd::backends::F64x2Backend;

/// 2-lane f64 SIMD vector, generic over backend `T`.
///
/// `T` is a token type that proves CPU support for the required SIMD features.
/// The inner representation is `T::Repr` (e.g., `__m128d` on x86, `float64x2_t` on ARM).
///
/// **The token is stored** (as a zero-sized field) so methods receiving
/// `self: f64x2<T>` can re-supply it to backend operations that
/// require a token value (e.g. `T::splat(token, v)`). This carries the
/// token-as-feature-proof guarantee through every method call without
/// runtime overhead — `T` is ZST, so `sizeof(f64x2<T>) == sizeof(T::Repr)`
/// and `align_of(f64x2<T>) == align_of(T::Repr)` under `#[repr(C)]`.
///
/// # Layout
///
/// `#[repr(C)]` with a ZST trailing field: `T::Repr` lives at offset 0
/// and `T` is a 0-byte tail. Bitcasts between `f64x2<T>` values of
/// different element-types are sound when the Repr types share a layout
/// (e.g. `__m128` and `__m128i` are both 16-byte aligned 128-bit values).
/// `#[repr(transparent)]` cannot be used because Rust cannot prove at
/// the struct definition site that a generic `T` is a 1-ZST.
///
/// Construction requires a token value to prove CPU support at runtime.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct f64x2<T: F64x2Backend>(pub(crate) T::Repr, pub(crate) T);

impl<T: F64x2Backend> f64x2<T> {
    /// Number of f64 lanes.
    pub const LANES: usize = 2;

    // ====== Construction (token-gated) ======

    /// Broadcast scalar to all 2 lanes.
    #[inline(always)]
    pub fn splat(token: T, v: f64) -> Self {
        Self(T::splat(token, v), token)
    }

    /// All lanes zero.
    #[inline(always)]
    pub fn zero(token: T) -> Self {
        Self(T::zero(token), token)
    }

    /// Load from a `[f64; 2]` array.
    #[inline(always)]
    pub fn load(token: T, data: &[f64; 2]) -> Self {
        Self(T::load(token, data), token)
    }

    /// Create from array (zero-cost where possible).
    #[inline(always)]
    pub fn from_array(token: T, arr: [f64; 2]) -> Self {
        Self(T::from_array(token, arr), token)
    }

    /// Create from slice. Panics if `slice.len() < 2`.
    #[inline(always)]
    pub fn from_slice(token: T, slice: &[f64]) -> Self {
        let arr: [f64; 2] = slice[..2].try_into().unwrap();
        Self(T::from_array(token, arr), token)
    }

    /// Split a slice into SIMD-width chunks and a scalar remainder.
    ///
    /// Returns `(&[[f64; 2]], &[f64])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[inline(always)]
    pub fn partition_slice(_: T, data: &[f64]) -> (&[[f64; 2]], &[f64]) {
        let bulk = data.len() / 2;
        let (head, tail) = data.split_at(bulk * 2);
        // SAFETY: head.len() is bulk * 2, so it's exactly `bulk` chunks of [f64; 2].
        // The pointer cast is valid because [f64] and [[f64; 2]] have the same alignment.
        let chunks = unsafe { core::slice::from_raw_parts(head.as_ptr().cast::<[f64; 2]>(), bulk) };
        (chunks, tail)
    }

    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    ///
    /// Returns `(&mut [[f64; 2]], &mut [f64])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[inline(always)]
    pub fn partition_slice_mut(_: T, data: &mut [f64]) -> (&mut [[f64; 2]], &mut [f64]) {
        let bulk = data.len() / 2;
        let (head, tail) = data.split_at_mut(bulk * 2);
        // SAFETY: head.len() is bulk * 2, so it's exactly `bulk` chunks of [f64; 2].
        // The pointer cast is valid because [f64] and [[f64; 2]] have the same alignment.
        let chunks =
            unsafe { core::slice::from_raw_parts_mut(head.as_mut_ptr().cast::<[f64; 2]>(), bulk) };
        (chunks, tail)
    }

    // ====== Accessors ======

    /// Store to array.
    #[inline(always)]
    pub fn store(self, out: &mut [f64; 2]) {
        T::store(self.0, out);
    }

    /// Convert to array.
    #[inline(always)]
    pub fn to_array(self) -> [f64; 2] {
        T::to_array(self.0)
    }

    /// Get the underlying platform representation.
    #[inline(always)]
    pub fn into_repr(self) -> T::Repr {
        self.0
    }

    /// Wrap a platform representation (token-gated).
    #[inline(always)]
    pub fn from_repr(token: T, repr: T::Repr) -> Self {
        Self(repr, token)
    }

    /// Wrap a repr with a token. Used by cross-type/cross-width helpers
    /// in `simd::generic::*` where the token is already proven by the
    /// caller's wider input type.
    #[inline(always)]
    #[allow(dead_code)]
    pub(crate) fn from_repr_unchecked(token: T, repr: T::Repr) -> Self {
        Self(repr, token)
    }

    // ====== Math ======

    /// Lane-wise minimum.
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(T::min(self.0, other.0), self.1)
    }

    /// Lane-wise maximum.
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(T::max(self.0, other.0), self.1)
    }

    /// Clamp between lo and hi.
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        Self(T::clamp(self.0, lo.0, hi.0), self.1)
    }

    /// Square root.
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        Self(T::sqrt(self.0), self.1)
    }

    /// Absolute value.
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(T::abs(self.0), self.1)
    }

    /// Round toward negative infinity.
    #[inline(always)]
    pub fn floor(self) -> Self {
        Self(T::floor(self.0), self.1)
    }

    /// Round toward positive infinity.
    #[inline(always)]
    pub fn ceil(self) -> Self {
        Self(T::ceil(self.0), self.1)
    }

    /// Round to nearest integer.
    #[inline(always)]
    pub fn round(self) -> Self {
        Self(T::round(self.0), self.1)
    }

    /// Fused multiply-add: `self * a + b`.
    #[inline(always)]
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        Self(T::mul_add(self.0, a.0, b.0), self.1)
    }

    /// Fused multiply-sub: `self * a - b`.
    #[inline(always)]
    pub fn mul_sub(self, a: Self, b: Self) -> Self {
        Self(T::mul_sub(self.0, a.0, b.0), self.1)
    }

    // ====== Comparisons ======

    /// Lane-wise equality (returns mask).
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> Self {
        Self(T::simd_eq(self.0, other.0), self.1)
    }

    /// Lane-wise inequality (returns mask).
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(T::simd_ne(self.0, other.0), self.1)
    }

    /// Lane-wise less-than (returns mask).
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(T::simd_lt(self.0, other.0), self.1)
    }

    /// Lane-wise less-than-or-equal (returns mask).
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(T::simd_le(self.0, other.0), self.1)
    }

    /// Lane-wise greater-than (returns mask).
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(T::simd_gt(self.0, other.0), self.1)
    }

    /// Lane-wise greater-than-or-equal (returns mask).
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(T::simd_ge(self.0, other.0), self.1)
    }

    /// Select lanes: where mask is all-1s pick `if_true`, else `if_false`.
    #[inline(always)]
    pub fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self(T::blend(mask.0, if_true.0, if_false.0), mask.1)
    }

    // ====== Reductions ======

    /// Sum all 2 lanes.
    #[inline(always)]
    pub fn reduce_add(self) -> f64 {
        T::reduce_add(self.0)
    }

    /// Minimum across all 2 lanes.
    #[inline(always)]
    pub fn reduce_min(self) -> f64 {
        T::reduce_min(self.0)
    }

    /// Maximum across all 2 lanes.
    #[inline(always)]
    pub fn reduce_max(self) -> f64 {
        T::reduce_max(self.0)
    }

    // ====== Approximations ======

    /// Fast reciprocal approximation (~12-bit precision).
    #[inline(always)]
    pub fn rcp_approx(self) -> Self {
        Self(T::rcp_approx(self.1, self.0), self.1)
    }

    /// Precise reciprocal (Newton-Raphson refined).
    #[inline(always)]
    pub fn recip(self) -> Self {
        Self(T::recip(self.1, self.0), self.1)
    }

    /// Fast reciprocal square root approximation (~12-bit precision).
    #[inline(always)]
    pub fn rsqrt_approx(self) -> Self {
        Self(T::rsqrt_approx(self.1, self.0), self.1)
    }

    /// Precise reciprocal square root (Newton-Raphson refined).
    #[inline(always)]
    pub fn rsqrt(self) -> Self {
        Self(T::rsqrt(self.1, self.0), self.1)
    }

    // ====== Bitwise ======

    /// Bitwise NOT.
    #[inline(always)]
    pub fn not(self) -> Self {
        Self(T::not(self.0), self.1)
    }
}

// ============================================================================
// Operator implementations
// ============================================================================

impl<T: F64x2Backend> Add for f64x2<T> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(T::add(self.0, rhs.0), self.1)
    }
}

impl<T: F64x2Backend> Sub for f64x2<T> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(T::sub(self.0, rhs.0), self.1)
    }
}

impl<T: F64x2Backend> Mul for f64x2<T> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(T::mul(self.0, rhs.0), self.1)
    }
}

impl<T: F64x2Backend> Div for f64x2<T> {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        Self(T::div(self.0, rhs.0), self.1)
    }
}

impl<T: F64x2Backend> Neg for f64x2<T> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(T::neg(self.1, self.0), self.1)
    }
}

impl<T: F64x2Backend> BitAnd for f64x2<T> {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(T::bitand(self.0, rhs.0), self.1)
    }
}

impl<T: F64x2Backend> BitOr for f64x2<T> {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self(T::bitor(self.0, rhs.0), self.1)
    }
}

impl<T: F64x2Backend> BitXor for f64x2<T> {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Self(T::bitxor(self.0, rhs.0), self.1)
    }
}

// ============================================================================
// Assign operators
// ============================================================================

impl<T: F64x2Backend> AddAssign for f64x2<T> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<T: F64x2Backend> SubAssign for f64x2<T> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<T: F64x2Backend> MulAssign for f64x2<T> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<T: F64x2Backend> DivAssign for f64x2<T> {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<T: F64x2Backend> BitAndAssign for f64x2<T> {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

impl<T: F64x2Backend> BitOrAssign for f64x2<T> {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl<T: F64x2Backend> BitXorAssign for f64x2<T> {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = *self ^ rhs;
    }
}

// ============================================================================
// Scalar broadcast operators (v + 2.0, v * 0.5, etc.)
// ============================================================================

impl<T: F64x2Backend> Add<f64> for f64x2<T> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: f64) -> Self {
        Self(T::add(self.0, T::splat(self.1, rhs)), self.1)
    }
}

impl<T: F64x2Backend> Sub<f64> for f64x2<T> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: f64) -> Self {
        Self(T::sub(self.0, T::splat(self.1, rhs)), self.1)
    }
}

impl<T: F64x2Backend> Mul<f64> for f64x2<T> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f64) -> Self {
        Self(T::mul(self.0, T::splat(self.1, rhs)), self.1)
    }
}

impl<T: F64x2Backend> Div<f64> for f64x2<T> {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: f64) -> Self {
        Self(T::div(self.0, T::splat(self.1, rhs)), self.1)
    }
}

// ============================================================================
// Index
// ============================================================================

impl<T: F64x2Backend> Index<usize> for f64x2<T> {
    type Output = f64;
    #[inline(always)]
    fn index(&self, i: usize) -> &f64 {
        assert!(i < 2, "f64x2 index out of bounds: {i}");
        // SAFETY: f64x2's repr is layout-compatible with [f64; 2], and i < 2.
        unsafe { &*(core::ptr::from_ref(self).cast::<f64>()).add(i) }
    }
}

impl<T: F64x2Backend> IndexMut<usize> for f64x2<T> {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut f64 {
        assert!(i < 2, "f64x2 index out of bounds: {i}");
        // SAFETY: f64x2's repr is layout-compatible with [f64; 2], and i < 2.
        unsafe { &mut *(core::ptr::from_mut(self).cast::<f64>()).add(i) }
    }
}

// ============================================================================
// Conversions
// ============================================================================

impl<T: F64x2Backend> From<f64x2<T>> for [f64; 2] {
    #[inline(always)]
    fn from(v: f64x2<T>) -> [f64; 2] {
        T::to_array(v.0)
    }
}

// ============================================================================
// Debug
// ============================================================================

impl<T: F64x2Backend> core::fmt::Debug for f64x2<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let arr = T::to_array(self.0);
        f.debug_tuple("f64x2").field(&arr).finish()
    }
}

// ============================================================================
// Platform-specific concrete impls
// ============================================================================

#[cfg(target_arch = "x86_64")]
impl f64x2<archmage::X64V3Token> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "x86::v3::f64x2"
    }

    /// Get the raw `__m128d` value.
    #[inline(always)]
    pub fn raw(self) -> core::arch::x86_64::__m128d {
        self.0
    }

    /// Create from a raw `__m128d` (token-gated, zero-cost).
    #[inline(always)]
    pub fn from_m128d(token: archmage::X64V3Token, v: core::arch::x86_64::__m128d) -> Self {
        Self(v, token)
    }
}

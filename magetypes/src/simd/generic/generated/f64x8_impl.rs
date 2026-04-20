//! Generic `f64x8<T>` — 8-lane f64 SIMD vector parameterized by backend.
//!
//! `T` is a token type (e.g., `X64V4Token`, `ScalarToken`)
//! that determines the platform-native representation and intrinsics used.
//! The struct delegates all operations to the [`F64x8Backend`] trait.
//!
//! # Example
//!
//! ```ignore
//! use magetypes::simd::backends::{F64x8Backend, x64v4};
//! use magetypes::simd::generic::f64x8;
//!
//! fn sum<T: F64x8Backend>(token: T, data: &[f64]) -> f64 {
//!     let mut acc = f64x8::<T>::zero(token);
//!     for chunk in data.chunks_exact(8) {
//!         acc = acc + f64x8::<T>::load(token, chunk.try_into().unwrap());
//!     }
//!     acc.reduce_add()
//! }
//! ```

#![allow(clippy::should_implement_trait)]

use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use crate::simd::backends::F64x8Backend;

/// 8-lane f64 SIMD vector, generic over backend `T`.
///
/// `T` is a token type that proves CPU support for the required SIMD features.
/// The inner representation is `T::Repr` (e.g., `__m512d` on AVX-512, `[f64; 8]` on scalar).
///
/// **The token is stored** (as a zero-sized field) so methods receiving
/// `self: f64x8<T>` can re-supply it to backend operations that
/// require a token value (e.g. `T::splat(token, v)`). This carries the
/// token-as-feature-proof guarantee through every method call without
/// runtime overhead — `T` is ZST, so `sizeof(f64x8<T>) == sizeof(T::Repr)`
/// and `align_of(f64x8<T>) == align_of(T::Repr)` under `#[repr(C)]`.
///
/// # Layout
///
/// `#[repr(C)]` with a ZST trailing field: `T::Repr` lives at offset 0
/// and `T` is a 0-byte tail. Bitcasts between `f64x8<T>` values of
/// different element-types are sound when the Repr types share a layout
/// (e.g. `__m128` and `__m128i` are both 16-byte aligned 128-bit values).
/// `#[repr(transparent)]` cannot be used because Rust cannot prove at
/// the struct definition site that a generic `T` is a 1-ZST.
///
/// Construction requires a token value to prove CPU support at runtime.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct f64x8<T: F64x8Backend>(pub(crate) T::Repr, pub(crate) T);

// PhantomData is ZST, so f64x8<T> has the same size as T::Repr.

// Layout invariant: struct is `#[repr(C)]` with a trailing ZST `T`
// field, so `sizeof/alignof(f64x8<T>) == sizeof/alignof(T::Repr)`
// iff `T` is a 1-ZST. Every archmage token currently satisfies this;
// if a future refactor adds a non-ZST field to a token, this const
// assert fires at compile time.
const _: () = {
    assert!(
        core::mem::size_of::<f64x8<archmage::ScalarToken>>()
            == core::mem::size_of::<
                <archmage::ScalarToken as crate::simd::backends::F64x8Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<f64x8<archmage::ScalarToken>>()
            == core::mem::align_of::<
                <archmage::ScalarToken as crate::simd::backends::F64x8Backend>::Repr,
            >()
    );
};

#[cfg(target_arch = "x86_64")]
const _: () = {
    assert!(
        core::mem::size_of::<f64x8<archmage::X64V3Token>>()
            == core::mem::size_of::<
                <archmage::X64V3Token as crate::simd::backends::F64x8Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<f64x8<archmage::X64V3Token>>()
            == core::mem::align_of::<
                <archmage::X64V3Token as crate::simd::backends::F64x8Backend>::Repr,
            >()
    );
};

// Native AVX-512 (`__m512`/`__m512d`/`__m512i`) — gated on the
// `avx512` feature, which is how archmage exposes X64V4Token's
// 512-bit backend impls.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
const _: () = {
    assert!(
        core::mem::size_of::<f64x8<archmage::X64V4Token>>()
            == core::mem::size_of::<
                <archmage::X64V4Token as crate::simd::backends::F64x8Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<f64x8<archmage::X64V4Token>>()
            == core::mem::align_of::<
                <archmage::X64V4Token as crate::simd::backends::F64x8Backend>::Repr,
            >()
    );
};

#[cfg(target_arch = "aarch64")]
const _: () = {
    assert!(
        core::mem::size_of::<f64x8<archmage::NeonToken>>()
            == core::mem::size_of::<
                <archmage::NeonToken as crate::simd::backends::F64x8Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<f64x8<archmage::NeonToken>>()
            == core::mem::align_of::<
                <archmage::NeonToken as crate::simd::backends::F64x8Backend>::Repr,
            >()
    );
};

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
const _: () = {
    assert!(
        core::mem::size_of::<f64x8<archmage::Wasm128Token>>()
            == core::mem::size_of::<
                <archmage::Wasm128Token as crate::simd::backends::F64x8Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<f64x8<archmage::Wasm128Token>>()
            == core::mem::align_of::<
                <archmage::Wasm128Token as crate::simd::backends::F64x8Backend>::Repr,
            >()
    );
};

impl<T: F64x8Backend> f64x8<T> {
    /// Number of f64 lanes.
    pub const LANES: usize = 8;

    // ====== Construction (token-gated) ======

    /// Broadcast scalar to all 8 lanes.
    #[inline(always)]
    pub fn splat(token: T, v: f64) -> Self {
        Self(T::splat(token, v), token)
    }

    /// All lanes zero.
    #[inline(always)]
    pub fn zero(token: T) -> Self {
        Self(T::zero(token), token)
    }

    /// Load from a `[f64; 8]` array.
    #[inline(always)]
    pub fn load(token: T, data: &[f64; 8]) -> Self {
        Self(T::load(token, data), token)
    }

    /// Create from array (zero-cost where possible).
    #[inline(always)]
    pub fn from_array(token: T, arr: [f64; 8]) -> Self {
        Self(T::from_array(token, arr), token)
    }

    /// Create from slice. Panics if `slice.len() < 8`.
    #[inline(always)]
    pub fn from_slice(token: T, slice: &[f64]) -> Self {
        let arr: [f64; 8] = slice[..8].try_into().unwrap();
        Self(T::from_array(token, arr), token)
    }

    /// Split a slice into SIMD-width chunks and a scalar remainder.
    ///
    /// Returns `(&[[f64; 8]], &[f64])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[inline(always)]
    pub fn partition_slice(_: T, data: &[f64]) -> (&[[f64; 8]], &[f64]) {
        let bulk = data.len() / 8;
        let (head, tail) = data.split_at(bulk * 8);
        // SAFETY: head.len() is bulk * 8, so it's exactly `bulk` chunks of [f64; 8].
        // The pointer cast is valid because [f64] and [[f64; 8]] have the same alignment.
        let chunks = unsafe { core::slice::from_raw_parts(head.as_ptr().cast::<[f64; 8]>(), bulk) };
        (chunks, tail)
    }

    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    ///
    /// Returns `(&mut [[f64; 8]], &mut [f64])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[inline(always)]
    pub fn partition_slice_mut(_: T, data: &mut [f64]) -> (&mut [[f64; 8]], &mut [f64]) {
        let bulk = data.len() / 8;
        let (head, tail) = data.split_at_mut(bulk * 8);
        // SAFETY: head.len() is bulk * 8, so it's exactly `bulk` chunks of [f64; 8].
        // The pointer cast is valid because [f64] and [[f64; 8]] have the same alignment.
        let chunks =
            unsafe { core::slice::from_raw_parts_mut(head.as_mut_ptr().cast::<[f64; 8]>(), bulk) };
        (chunks, tail)
    }

    // ====== Accessors ======

    /// Store to array.
    #[inline(always)]
    pub fn store(self, out: &mut [f64; 8]) {
        T::store(self.1, self.0, out);
    }

    /// Convert to array.
    #[inline(always)]
    pub fn to_array(self) -> [f64; 8] {
        T::to_array(self.1, self.0)
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
        Self(T::min(self.1, self.0, other.0), self.1)
    }

    /// Lane-wise maximum.
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(T::max(self.1, self.0, other.0), self.1)
    }

    /// Clamp between lo and hi.
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        Self(T::clamp(self.1, self.0, lo.0, hi.0), self.1)
    }

    /// Square root.
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        Self(T::sqrt(self.1, self.0), self.1)
    }

    /// Absolute value.
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(T::abs(self.1, self.0), self.1)
    }

    /// Round toward negative infinity.
    #[inline(always)]
    pub fn floor(self) -> Self {
        Self(T::floor(self.1, self.0), self.1)
    }

    /// Round toward positive infinity.
    #[inline(always)]
    pub fn ceil(self) -> Self {
        Self(T::ceil(self.1, self.0), self.1)
    }

    /// Round to nearest integer.
    #[inline(always)]
    pub fn round(self) -> Self {
        Self(T::round(self.1, self.0), self.1)
    }

    /// Fused multiply-add: `self * a + b`.
    #[inline(always)]
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        Self(T::mul_add(self.1, self.0, a.0, b.0), self.1)
    }

    /// Fused multiply-sub: `self * a - b`.
    #[inline(always)]
    pub fn mul_sub(self, a: Self, b: Self) -> Self {
        Self(T::mul_sub(self.1, self.0, a.0, b.0), self.1)
    }

    // ====== Comparisons ======

    /// Lane-wise equality (returns mask).
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> Self {
        Self(T::simd_eq(self.1, self.0, other.0), self.1)
    }

    /// Lane-wise inequality (returns mask).
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self(T::simd_ne(self.1, self.0, other.0), self.1)
    }

    /// Lane-wise less-than (returns mask).
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(T::simd_lt(self.1, self.0, other.0), self.1)
    }

    /// Lane-wise less-than-or-equal (returns mask).
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(T::simd_le(self.1, self.0, other.0), self.1)
    }

    /// Lane-wise greater-than (returns mask).
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(T::simd_gt(self.1, self.0, other.0), self.1)
    }

    /// Lane-wise greater-than-or-equal (returns mask).
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self(T::simd_ge(self.1, self.0, other.0), self.1)
    }

    /// Select lanes: where mask is all-1s pick `if_true`, else `if_false`.
    #[inline(always)]
    pub fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self(T::blend(mask.1, mask.0, if_true.0, if_false.0), mask.1)
    }

    // ====== Reductions ======

    /// Sum all 8 lanes.
    #[inline(always)]
    pub fn reduce_add(self) -> f64 {
        T::reduce_add(self.1, self.0)
    }

    /// Minimum across all 8 lanes.
    #[inline(always)]
    pub fn reduce_min(self) -> f64 {
        T::reduce_min(self.1, self.0)
    }

    /// Maximum across all 8 lanes.
    #[inline(always)]
    pub fn reduce_max(self) -> f64 {
        T::reduce_max(self.1, self.0)
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
        Self(T::not(self.1, self.0), self.1)
    }
}

// ============================================================================
// Operator implementations
// ============================================================================

impl<T: F64x8Backend> Add for f64x8<T> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(T::add(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: F64x8Backend> Sub for f64x8<T> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(T::sub(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: F64x8Backend> Mul for f64x8<T> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(T::mul(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: F64x8Backend> Div for f64x8<T> {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        Self(T::div(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: F64x8Backend> Neg for f64x8<T> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(T::neg(self.1, self.0), self.1)
    }
}

impl<T: F64x8Backend> BitAnd for f64x8<T> {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(T::bitand(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: F64x8Backend> BitOr for f64x8<T> {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self(T::bitor(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: F64x8Backend> BitXor for f64x8<T> {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Self(T::bitxor(self.1, self.0, rhs.0), self.1)
    }
}

// ============================================================================
// Assign operators
// ============================================================================

impl<T: F64x8Backend> AddAssign for f64x8<T> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<T: F64x8Backend> SubAssign for f64x8<T> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<T: F64x8Backend> MulAssign for f64x8<T> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<T: F64x8Backend> DivAssign for f64x8<T> {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<T: F64x8Backend> BitAndAssign for f64x8<T> {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

impl<T: F64x8Backend> BitOrAssign for f64x8<T> {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl<T: F64x8Backend> BitXorAssign for f64x8<T> {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = *self ^ rhs;
    }
}

// ============================================================================
// Scalar broadcast operators (v + 2.0, v * 0.5, etc.)
// ============================================================================

impl<T: F64x8Backend> Add<f64> for f64x8<T> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: f64) -> Self {
        Self(T::add(self.1, self.0, T::splat(self.1, rhs)), self.1)
    }
}

impl<T: F64x8Backend> Sub<f64> for f64x8<T> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: f64) -> Self {
        Self(T::sub(self.1, self.0, T::splat(self.1, rhs)), self.1)
    }
}

impl<T: F64x8Backend> Mul<f64> for f64x8<T> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f64) -> Self {
        Self(T::mul(self.1, self.0, T::splat(self.1, rhs)), self.1)
    }
}

impl<T: F64x8Backend> Div<f64> for f64x8<T> {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: f64) -> Self {
        Self(T::div(self.1, self.0, T::splat(self.1, rhs)), self.1)
    }
}

// ============================================================================
// Index
// ============================================================================

impl<T: F64x8Backend> Index<usize> for f64x8<T> {
    type Output = f64;
    #[inline(always)]
    fn index(&self, i: usize) -> &f64 {
        assert!(i < 8, "f64x8 index out of bounds: {i}");
        // SAFETY: f64x8's repr is layout-compatible with [f64; 8], and i < 8.
        unsafe { &*(core::ptr::from_ref(self).cast::<f64>()).add(i) }
    }
}

impl<T: F64x8Backend> IndexMut<usize> for f64x8<T> {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut f64 {
        assert!(i < 8, "f64x8 index out of bounds: {i}");
        // SAFETY: f64x8's repr is layout-compatible with [f64; 8], and i < 8.
        unsafe { &mut *(core::ptr::from_mut(self).cast::<f64>()).add(i) }
    }
}

// ============================================================================
// Conversions
// ============================================================================

impl<T: F64x8Backend> From<f64x8<T>> for [f64; 8] {
    #[inline(always)]
    fn from(v: f64x8<T>) -> [f64; 8] {
        T::to_array(v.1, v.0)
    }
}

// ============================================================================
// Debug
// ============================================================================

impl<T: F64x8Backend> core::fmt::Debug for f64x8<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let arr = T::to_array(self.1, self.0);
        f.debug_tuple("f64x8").field(&arr).finish()
    }
}

// ============================================================================
// Platform-specific implementation info
// ============================================================================

#[cfg(target_arch = "x86_64")]
impl f64x8<archmage::X64V3Token> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "polyfill::v3_512::f64x8"
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl f64x8<archmage::X64V4Token> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "x86::v4::f64x8"
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl f64x8<archmage::X64V4xToken> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "x86::v4x::f64x8"
    }
}

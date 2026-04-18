//! Generic `i64x8<T>` — 8-lane i64 SIMD vector parameterized by backend.
//!
//! `T` is a token type (e.g., `X64V4Token`, `ScalarToken`)
//! that determines the platform-native representation and intrinsics used.
//! The struct delegates all operations to the [`I64x8Backend`] trait.

#![allow(clippy::should_implement_trait)]

use core::marker::PhantomData;
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Index,
    IndexMut, Neg, Sub, SubAssign,
};

use crate::simd::backends::I64x8Backend;

/// 8-lane i64 SIMD vector, generic over backend `T`.
///
/// `T` is a token type that proves CPU support for the required SIMD features.
/// The inner representation is `T::Repr` (e.g., `__m512i` on AVX-512, `[i64; 8]` on scalar).
///
/// **The token is stored** (as a zero-sized field) so methods receiving
/// `self: i64x8<T>` can re-supply it to backend operations that
/// require a token value (e.g. `T::splat(token, v)`). This carries the
/// token-as-feature-proof guarantee through every method call without
/// runtime overhead — `T` is ZST, so `sizeof(i64x8<T>) == sizeof(T::Repr)`,
/// and `#[repr(transparent)]` is preserved.
///
/// Construction requires a token value to prove CPU support at runtime.
///
/// # Note
///
/// 64-bit integer SIMD has limited native support: no hardware multiply on
/// AVX2/NEON/WASM, and arithmetic right shift requires AVX-512 on x86.
/// Operations like `min`, `max`, and `abs` are polyfilled where needed.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct i64x8<T: I64x8Backend>(T::Repr, T);

impl<T: I64x8Backend> i64x8<T> {
    /// Number of i64 lanes.
    pub const LANES: usize = 8;

    // ====== Construction (token-gated) ======

    /// Broadcast scalar to all 8 lanes.
    #[inline(always)]
    pub fn splat(token: T, v: i64) -> Self {
        Self(T::splat(token, v), token)
    }

    /// All lanes zero.
    #[inline(always)]
    pub fn zero(token: T) -> Self {
        Self(T::zero(token), token)
    }

    /// Load from a `[i64; 8]` array.
    #[inline(always)]
    pub fn load(token: T, data: &[i64; 8]) -> Self {
        Self(T::load(token, data), token)
    }

    /// Create from array (zero-cost where possible).
    #[inline(always)]
    pub fn from_array(token: T, arr: [i64; 8]) -> Self {
        Self(T::from_array(token, arr), token)
    }

    /// Create from slice. Panics if `slice.len() < 8`.
    #[inline(always)]
    pub fn from_slice(token: T, slice: &[i64]) -> Self {
        let arr: [i64; 8] = slice[..8].try_into().unwrap();
        Self(T::from_array(token, arr), token)
    }

    /// Split a slice into SIMD-width chunks and a scalar remainder.
    ///
    /// Returns `(&[[i64; 8]], &[i64])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[inline(always)]
    pub fn partition_slice(_: T, data: &[i64]) -> (&[[i64; 8]], &[i64]) {
        let bulk = data.len() / 8;
        let (head, tail) = data.split_at(bulk * 8);
        // SAFETY: head.len() is bulk * 8, so it's exactly `bulk` chunks of [i64; 8].
        // The pointer cast is valid because [i64] and [[i64; 8]] have the same alignment.
        let chunks = unsafe { core::slice::from_raw_parts(head.as_ptr().cast::<[i64; 8]>(), bulk) };
        (chunks, tail)
    }

    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    ///
    /// Returns `(&mut [[i64; 8]], &mut [i64])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[inline(always)]
    pub fn partition_slice_mut(_: T, data: &mut [i64]) -> (&mut [[i64; 8]], &mut [i64]) {
        let bulk = data.len() / 8;
        let (head, tail) = data.split_at_mut(bulk * 8);
        // SAFETY: head.len() is bulk * 8, so it's exactly `bulk` chunks of [i64; 8].
        // The pointer cast is valid because [i64] and [[i64; 8]] have the same alignment.
        let chunks =
            unsafe { core::slice::from_raw_parts_mut(head.as_mut_ptr().cast::<[i64; 8]>(), bulk) };
        (chunks, tail)
    }

    // ====== Accessors ======

    /// Store to array.
    #[inline(always)]
    pub fn store(self, out: &mut [i64; 8]) {
        T::store(self.0, out);
    }

    /// Convert to array.
    #[inline(always)]
    pub fn to_array(self) -> [i64; 8] {
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

    /// Lane-wise absolute value.
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(T::abs(self.0), self.1)
    }

    /// Clamp between lo and hi.
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        Self(T::clamp(self.0, lo.0, hi.0), self.1)
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
        Self(T::blend(mask.0, if_true.0, if_false.0), self.1)
    }

    // ====== Reductions ======

    /// Sum all 8 lanes (wrapping).
    #[inline(always)]
    pub fn reduce_add(self) -> i64 {
        T::reduce_add(self.0)
    }

    // ====== Shifts ======

    /// Shift left by constant.
    #[inline(always)]
    pub fn shl_const<const N: i32>(self) -> Self {
        Self(T::shl_const::<N>(self.0), self.1)
    }

    /// Arithmetic shift right by constant (sign-extending).
    #[inline(always)]
    pub fn shr_arithmetic_const<const N: i32>(self) -> Self {
        Self(T::shr_arithmetic_const::<N>(self.0), self.1)
    }

    /// Logical shift right by constant (zero-filling).
    #[inline(always)]
    pub fn shr_logical_const<const N: i32>(self) -> Self {
        Self(T::shr_logical_const::<N>(self.0), self.1)
    }

    /// Alias for [`shl_const`](Self::shl_const).
    #[inline(always)]
    pub fn shl<const N: i32>(self) -> Self {
        self.shl_const::<N>()
    }

    /// Alias for [`shr_arithmetic_const`](Self::shr_arithmetic_const).
    #[inline(always)]
    pub fn shr_arithmetic<const N: i32>(self) -> Self {
        self.shr_arithmetic_const::<N>()
    }

    /// Alias for [`shr_logical_const`](Self::shr_logical_const).
    #[inline(always)]
    pub fn shr_logical<const N: i32>(self) -> Self {
        self.shr_logical_const::<N>()
    }

    // ====== Bitwise ======

    /// Bitwise NOT.
    #[inline(always)]
    pub fn not(self) -> Self {
        Self(T::not(self.0), self.1)
    }

    // ====== Boolean ======

    /// True if all lanes have their sign bit set (all-1s mask).
    #[inline(always)]
    pub fn all_true(self) -> bool {
        T::all_true(self.0)
    }

    /// True if any lane has its sign bit set.
    #[inline(always)]
    pub fn any_true(self) -> bool {
        T::any_true(self.0)
    }

    /// Extract the high bit of each 64-bit lane as a bitmask.
    #[inline(always)]
    pub fn bitmask(self) -> u64 {
        T::bitmask(self.0)
    }
}

// ============================================================================
// Operator implementations
// ============================================================================

impl<T: I64x8Backend> Add for i64x8<T> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(T::add(self.0, rhs.0), self.1)
    }
}

impl<T: I64x8Backend> Sub for i64x8<T> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(T::sub(self.0, rhs.0), self.1)
    }
}

impl<T: I64x8Backend> Neg for i64x8<T> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(T::neg(self.0), self.1)
    }
}

impl<T: I64x8Backend> BitAnd for i64x8<T> {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(T::bitand(self.0, rhs.0), self.1)
    }
}

impl<T: I64x8Backend> BitOr for i64x8<T> {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self(T::bitor(self.0, rhs.0), self.1)
    }
}

impl<T: I64x8Backend> BitXor for i64x8<T> {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Self(T::bitxor(self.0, rhs.0), self.1)
    }
}

// ============================================================================
// Assign operators
// ============================================================================

impl<T: I64x8Backend> AddAssign for i64x8<T> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<T: I64x8Backend> SubAssign for i64x8<T> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<T: I64x8Backend> BitAndAssign for i64x8<T> {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

impl<T: I64x8Backend> BitOrAssign for i64x8<T> {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl<T: I64x8Backend> BitXorAssign for i64x8<T> {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = *self ^ rhs;
    }
}

// ============================================================================
// Scalar broadcast operators (v + 2, etc.)
// ============================================================================

impl<T: I64x8Backend> Add<i64> for i64x8<T> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: i64) -> Self {
        Self(T::add(self.0, T::splat(rhs)), self.1)
    }
}

impl<T: I64x8Backend> Sub<i64> for i64x8<T> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: i64) -> Self {
        Self(T::sub(self.0, T::splat(rhs)), self.1)
    }
}

// ============================================================================
// Index
// ============================================================================

impl<T: I64x8Backend> Index<usize> for i64x8<T> {
    type Output = i64;
    #[inline(always)]
    fn index(&self, i: usize) -> &i64 {
        assert!(i < 8, "i64x8 index out of bounds: {i}");
        // SAFETY: i64x8's repr is layout-compatible with [i64; 8], and i < 8.
        unsafe { &*(core::ptr::from_ref(self).cast::<i64>()).add(i) }
    }
}

impl<T: I64x8Backend> IndexMut<usize> for i64x8<T> {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut i64 {
        assert!(i < 8, "i64x8 index out of bounds: {i}");
        // SAFETY: i64x8's repr is layout-compatible with [i64; 8], and i < 8.
        unsafe { &mut *(core::ptr::from_mut(self).cast::<i64>()).add(i) }
    }
}

// ============================================================================
// Conversions
// ============================================================================

impl<T: I64x8Backend> From<i64x8<T>> for [i64; 8] {
    #[inline(always)]
    fn from(v: i64x8<T>) -> [i64; 8] {
        T::to_array(v.0)
    }
}

// ============================================================================
// Debug
// ============================================================================

impl<T: I64x8Backend> core::fmt::Debug for i64x8<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let arr = T::to_array(self.0);
        f.debug_tuple("i64x8").field(&arr).finish()
    }
}

// ============================================================================
// Platform-specific implementation info
// ============================================================================

#[cfg(target_arch = "x86_64")]
impl i64x8<archmage::X64V3Token> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "polyfill::v3_512::i64x8"
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl i64x8<archmage::X64V4Token> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "x86::v4::i64x8"
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl i64x8<archmage::X64V4xToken> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "x86::v4x::i64x8"
    }
}

// ============================================================================
// Extension: popcnt (requires Modern token)
// ============================================================================

#[cfg(feature = "avx512")]
impl<T: crate::simd::backends::i64x8PopcntBackend> i64x8<T> {
    /// Count set bits in each lane (popcnt).
    ///
    /// Returns a vector where each lane contains the number of 1-bits
    /// in the corresponding lane of `self`.
    ///
    /// Requires AVX-512 Modern token (VPOPCNTDQ or BITALG extension).
    #[inline(always)]
    pub fn popcnt(self) -> Self {
        Self(T::popcnt(self.0), core::marker::PhantomData)
    }
}

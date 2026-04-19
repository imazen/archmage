//! Generic `i32x8<T>` — 8-lane i32 SIMD vector parameterized by backend.
//!
//! `T` is a token type (e.g., `X64V3Token`, `NeonToken`, `ScalarToken`)
//! that determines the platform-native representation and intrinsics used.
//! The struct delegates all operations to the [`I32x8Backend`] trait.

#![allow(clippy::should_implement_trait)]

use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Index,
    IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use crate::simd::backends::I32x8Backend;

/// 8-lane i32 SIMD vector, generic over backend `T`.
///
/// `T` is a token type that proves CPU support for the required SIMD features.
/// The inner representation is `T::Repr` (e.g., `__m256i` on AVX2, `[i32; 8]` on scalar).
///
/// **The token is stored** (as a zero-sized field) so methods receiving
/// `self: i32x8<T>` can re-supply it to backend operations that
/// require a token value (e.g. `T::splat(token, v)`). This carries the
/// token-as-feature-proof guarantee through every method call without
/// runtime overhead — `T` is ZST, so `sizeof(i32x8<T>) == sizeof(T::Repr)`
/// and `align_of(i32x8<T>) == align_of(T::Repr)` under `#[repr(C)]`.
///
/// # Layout
///
/// `#[repr(C)]` with a ZST trailing field: `T::Repr` lives at offset 0
/// and `T` is a 0-byte tail. Bitcasts between `i32x8<T>` values of
/// different element-types are sound when the Repr types share a layout
/// (e.g. `__m128` and `__m128i` are both 16-byte aligned 128-bit values).
/// `#[repr(transparent)]` cannot be used because Rust cannot prove at
/// the struct definition site that a generic `T` is a 1-ZST.
///
/// Construction requires a token value to prove CPU support at runtime.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct i32x8<T: I32x8Backend>(pub(crate) T::Repr, pub(crate) T);

impl<T: I32x8Backend> i32x8<T> {
    /// Number of i32 lanes.
    pub const LANES: usize = 8;

    // ====== Construction (token-gated) ======

    /// Broadcast scalar to all 8 lanes.
    #[inline(always)]
    pub fn splat(token: T, v: i32) -> Self {
        Self(T::splat(token, v), token)
    }

    /// All lanes zero.
    #[inline(always)]
    pub fn zero(token: T) -> Self {
        Self(T::zero(token), token)
    }

    /// Load from a `[i32; 8]` array.
    #[inline(always)]
    pub fn load(token: T, data: &[i32; 8]) -> Self {
        Self(T::load(token, data), token)
    }

    /// Create from array (zero-cost where possible).
    #[inline(always)]
    pub fn from_array(token: T, arr: [i32; 8]) -> Self {
        Self(T::from_array(token, arr), token)
    }

    /// Create from slice. Panics if `slice.len() < 8`.
    #[inline(always)]
    pub fn from_slice(token: T, slice: &[i32]) -> Self {
        let arr: [i32; 8] = slice[..8].try_into().unwrap();
        Self(T::from_array(token, arr), token)
    }

    /// Split a slice into SIMD-width chunks and a scalar remainder.
    ///
    /// Returns `(&[[i32; 8]], &[i32])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[inline(always)]
    pub fn partition_slice(_: T, data: &[i32]) -> (&[[i32; 8]], &[i32]) {
        let bulk = data.len() / 8;
        let (head, tail) = data.split_at(bulk * 8);
        // SAFETY: head.len() is bulk * 8, so it's exactly `bulk` chunks of [i32; 8].
        // The pointer cast is valid because [i32] and [[i32; 8]] have the same alignment.
        let chunks = unsafe { core::slice::from_raw_parts(head.as_ptr().cast::<[i32; 8]>(), bulk) };
        (chunks, tail)
    }

    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    ///
    /// Returns `(&mut [[i32; 8]], &mut [i32])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[inline(always)]
    pub fn partition_slice_mut(_: T, data: &mut [i32]) -> (&mut [[i32; 8]], &mut [i32]) {
        let bulk = data.len() / 8;
        let (head, tail) = data.split_at_mut(bulk * 8);
        // SAFETY: head.len() is bulk * 8, so it's exactly `bulk` chunks of [i32; 8].
        // The pointer cast is valid because [i32] and [[i32; 8]] have the same alignment.
        let chunks =
            unsafe { core::slice::from_raw_parts_mut(head.as_mut_ptr().cast::<[i32; 8]>(), bulk) };
        (chunks, tail)
    }

    // ====== Accessors ======

    /// Store to array.
    #[inline(always)]
    pub fn store(self, out: &mut [i32; 8]) {
        T::store(self.1, self.0, out);
    }

    /// Convert to array.
    #[inline(always)]
    pub fn to_array(self) -> [i32; 8] {
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

    /// Lane-wise absolute value.
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(T::abs(self.1, self.0), self.1)
    }

    /// Clamp between lo and hi.
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        Self(T::clamp(self.1, self.0, lo.0, hi.0), self.1)
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

    /// Sum all 8 lanes (wrapping).
    #[inline(always)]
    pub fn reduce_add(self) -> i32 {
        T::reduce_add(self.1, self.0)
    }

    // ====== Shifts ======

    /// Shift left by constant.
    #[inline(always)]
    pub fn shl_const<const N: i32>(self) -> Self {
        Self(T::shl_const::<N>(self.1, self.0), self.1)
    }

    /// Arithmetic shift right by constant (sign-extending).
    #[inline(always)]
    pub fn shr_arithmetic_const<const N: i32>(self) -> Self {
        Self(T::shr_arithmetic_const::<N>(self.1, self.0), self.1)
    }

    /// Logical shift right by constant (zero-filling).
    #[inline(always)]
    pub fn shr_logical_const<const N: i32>(self) -> Self {
        Self(T::shr_logical_const::<N>(self.1, self.0), self.1)
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
        Self(T::not(self.1, self.0), self.1)
    }

    // ====== Boolean ======

    /// True if all lanes have their sign bit set (all-1s mask).
    #[inline(always)]
    pub fn all_true(self) -> bool {
        T::all_true(self.1, self.0)
    }

    /// True if any lane has its sign bit set.
    #[inline(always)]
    pub fn any_true(self) -> bool {
        T::any_true(self.1, self.0)
    }

    /// Extract the high bit of each 32-bit lane as a bitmask.
    #[inline(always)]
    pub fn bitmask(self) -> u32 {
        T::bitmask(self.1, self.0)
    }
}

// ============================================================================
// Operator implementations
// ============================================================================

impl<T: I32x8Backend> Add for i32x8<T> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(T::add(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: I32x8Backend> Sub for i32x8<T> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(T::sub(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: I32x8Backend> Mul for i32x8<T> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(T::mul(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: I32x8Backend> Neg for i32x8<T> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(T::neg(self.1, self.0), self.1)
    }
}

impl<T: I32x8Backend> BitAnd for i32x8<T> {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(T::bitand(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: I32x8Backend> BitOr for i32x8<T> {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self(T::bitor(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: I32x8Backend> BitXor for i32x8<T> {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Self(T::bitxor(self.1, self.0, rhs.0), self.1)
    }
}

// ============================================================================
// Assign operators
// ============================================================================

impl<T: I32x8Backend> AddAssign for i32x8<T> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<T: I32x8Backend> SubAssign for i32x8<T> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<T: I32x8Backend> MulAssign for i32x8<T> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<T: I32x8Backend> BitAndAssign for i32x8<T> {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

impl<T: I32x8Backend> BitOrAssign for i32x8<T> {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl<T: I32x8Backend> BitXorAssign for i32x8<T> {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = *self ^ rhs;
    }
}

// ============================================================================
// Scalar broadcast operators (v + 2, v * 3, etc.)
// ============================================================================

impl<T: I32x8Backend> Add<i32> for i32x8<T> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: i32) -> Self {
        Self(T::add(self.1, self.0, T::splat(self.1, rhs)), self.1)
    }
}

impl<T: I32x8Backend> Sub<i32> for i32x8<T> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: i32) -> Self {
        Self(T::sub(self.1, self.0, T::splat(self.1, rhs)), self.1)
    }
}

impl<T: I32x8Backend> Mul<i32> for i32x8<T> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: i32) -> Self {
        Self(T::mul(self.1, self.0, T::splat(self.1, rhs)), self.1)
    }
}

// ============================================================================
// Index
// ============================================================================

impl<T: I32x8Backend> Index<usize> for i32x8<T> {
    type Output = i32;
    #[inline(always)]
    fn index(&self, i: usize) -> &i32 {
        assert!(i < 8, "i32x8 index out of bounds: {i}");
        // SAFETY: i32x8's repr is layout-compatible with [i32; 8], and i < 8.
        unsafe { &*(core::ptr::from_ref(self).cast::<i32>()).add(i) }
    }
}

impl<T: I32x8Backend> IndexMut<usize> for i32x8<T> {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut i32 {
        assert!(i < 8, "i32x8 index out of bounds: {i}");
        // SAFETY: i32x8's repr is layout-compatible with [i32; 8], and i < 8.
        unsafe { &mut *(core::ptr::from_mut(self).cast::<i32>()).add(i) }
    }
}

// ============================================================================
// Conversions
// ============================================================================

impl<T: I32x8Backend> From<i32x8<T>> for [i32; 8] {
    #[inline(always)]
    fn from(v: i32x8<T>) -> [i32; 8] {
        T::to_array(v.1, v.0)
    }
}

// ============================================================================
// Debug
// ============================================================================

impl<T: I32x8Backend> core::fmt::Debug for i32x8<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let arr = T::to_array(self.1, self.0);
        f.debug_tuple("i32x8").field(&arr).finish()
    }
}

// ============================================================================
// Cross-type conversions (available when T implements conversion traits)
// ============================================================================

impl<T: crate::simd::backends::F32x8Convert> i32x8<T> {
    /// Bitcast to f32x8 (reinterpret bits, no conversion).
    #[inline(always)]
    pub fn bitcast_to_f32(self) -> super::f32x8<T> {
        super::f32x8::from_repr_unchecked(self.1, T::bitcast_i32_to_f32(self.1, self.0))
    }

    /// Convert to f32x8 (numeric conversion).
    #[inline(always)]
    pub fn to_f32(self) -> super::f32x8<T> {
        super::f32x8::from_repr_unchecked(self.1, T::convert_i32_to_f32(self.1, self.0))
    }

    // ====== Backward-compatible aliases (old generated API names) ======

    /// Alias for [`bitcast_to_f32`](Self::bitcast_to_f32).
    #[inline(always)]
    pub fn bitcast_f32x8(self) -> super::f32x8<T> {
        self.bitcast_to_f32()
    }

    /// Alias for [`to_f32`](Self::to_f32).
    #[inline(always)]
    pub fn to_f32x8(self) -> super::f32x8<T> {
        self.to_f32()
    }
}

// ============================================================================
// Platform-specific concrete impls
// ============================================================================

#[cfg(target_arch = "x86_64")]
impl i32x8<archmage::X64V3Token> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "x86::v3::i32x8"
    }

    /// Get the raw `__m256i` value.
    #[inline(always)]
    pub fn raw(self) -> core::arch::x86_64::__m256i {
        self.0
    }

    /// Create from a raw `__m256i` (token-gated, zero-cost).
    #[inline(always)]
    pub fn from_m256i(token: archmage::X64V3Token, v: core::arch::x86_64::__m256i) -> Self {
        Self(v, token)
    }
}

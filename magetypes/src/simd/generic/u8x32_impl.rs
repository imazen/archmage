//! Generic `u8x32<T>` — 32-lane u8 SIMD vector parameterized by backend.
//!
//! `T` is a token type (e.g., `X64V3Token`, `NeonToken`, `ScalarToken`)
//! that determines the platform-native representation and intrinsics used.
//! The struct delegates all operations to the [`U8x32Backend`] trait.

#![allow(clippy::should_implement_trait)]

use core::marker::PhantomData;
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Index,
    IndexMut, Sub, SubAssign,
};

use crate::simd::backends::U8x32Backend;

/// 32-lane u8 SIMD vector, generic over backend `T`.
///
/// `T` is a token type that proves CPU support for the required SIMD features.
/// The inner representation is `T::Repr` (e.g., `__m256i` on x86).
///
/// Construction requires a token value to prove CPU support at runtime.
/// After construction, operations don't need the token — it's baked into the type.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct u8x32<T: U8x32Backend>(T::Repr, PhantomData<T>);

impl<T: U8x32Backend> u8x32<T> {
    /// Number of u8 lanes.
    pub const LANES: usize = 32;

    // ====== Construction (token-gated) ======

    /// Broadcast scalar to all 32 lanes.
    #[inline(always)]
    pub fn splat(_: T, v: u8) -> Self {
        Self(T::splat(v), PhantomData)
    }

    /// All lanes zero.
    #[inline(always)]
    pub fn zero(_: T) -> Self {
        Self(T::zero(), PhantomData)
    }

    /// Load from a `[u8; 32]` array.
    #[inline(always)]
    pub fn load(_: T, data: &[u8; 32]) -> Self {
        Self(T::load(data), PhantomData)
    }

    /// Create from array (zero-cost where possible).
    #[inline(always)]
    pub fn from_array(_: T, arr: [u8; 32]) -> Self {
        Self(T::from_array(arr), PhantomData)
    }

    /// Create from slice. Panics if `slice.len() < 32`.
    #[inline(always)]
    pub fn from_slice(_: T, slice: &[u8]) -> Self {
        let arr: [u8; 32] = slice[..32].try_into().unwrap();
        Self(T::from_array(arr), PhantomData)
    }

    // ====== Accessors ======

    /// Store to array.
    #[inline(always)]
    pub fn store(self, out: &mut [u8; 32]) {
        T::store(self.0, out);
    }

    /// Convert to array.
    #[inline(always)]
    pub fn to_array(self) -> [u8; 32] {
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

    /// Wrap a repr without requiring a token value.
    /// Only usable within the `generic` module (for cross-type conversions).
    #[inline(always)]
    pub(super) fn from_repr_unchecked(repr: T::Repr) -> Self {
        Self(repr, PhantomData)
    }

    // ====== Math ======

    /// Lane-wise minimum (unsigned).
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(T::min(self.0, other.0), PhantomData)
    }

    /// Lane-wise maximum (unsigned).
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(T::max(self.0, other.0), PhantomData)
    }

    /// Clamp between lo and hi.
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        Self(T::clamp(self.0, lo.0, hi.0), PhantomData)
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

    /// Lane-wise less-than, unsigned (returns mask).
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(T::simd_lt(self.0, other.0), PhantomData)
    }

    /// Lane-wise less-than-or-equal, unsigned (returns mask).
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(T::simd_le(self.0, other.0), PhantomData)
    }

    /// Lane-wise greater-than, unsigned (returns mask).
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(T::simd_gt(self.0, other.0), PhantomData)
    }

    /// Lane-wise greater-than-or-equal, unsigned (returns mask).
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

    /// Sum all 32 lanes (wrapping).
    #[inline(always)]
    pub fn reduce_add(self) -> u8 {
        T::reduce_add(self.0)
    }

    // ====== Shifts ======

    /// Shift left by constant.
    #[inline(always)]
    pub fn shl_const<const N: i32>(self) -> Self {
        Self(T::shl_const::<N>(self.0), PhantomData)
    }

    /// Logical shift right by constant (zero-filling).
    #[inline(always)]
    pub fn shr_logical_const<const N: i32>(self) -> Self {
        Self(T::shr_logical_const::<N>(self.0), PhantomData)
    }

    /// Alias for [`shl_const`](Self::shl_const).
    #[inline(always)]
    pub fn shl<const N: i32>(self) -> Self {
        self.shl_const::<N>()
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
        Self(T::not(self.0), PhantomData)
    }

    // ====== Boolean ======

    /// True if all lanes have their high bit set (all-1s mask).
    #[inline(always)]
    pub fn all_true(self) -> bool {
        T::all_true(self.0)
    }

    /// True if any lane has its high bit set.
    #[inline(always)]
    pub fn any_true(self) -> bool {
        T::any_true(self.0)
    }

    /// Extract the high bit of each 8-bit lane as a bitmask.
    #[inline(always)]
    pub fn bitmask(self) -> u32 {
        T::bitmask(self.0)
    }
}

// ============================================================================
// Operator implementations
// ============================================================================

impl<T: U8x32Backend> Add for u8x32<T> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(T::add(self.0, rhs.0), PhantomData)
    }
}

impl<T: U8x32Backend> Sub for u8x32<T> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(T::sub(self.0, rhs.0), PhantomData)
    }
}

impl<T: U8x32Backend> BitAnd for u8x32<T> {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(T::bitand(self.0, rhs.0), PhantomData)
    }
}

impl<T: U8x32Backend> BitOr for u8x32<T> {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self(T::bitor(self.0, rhs.0), PhantomData)
    }
}

impl<T: U8x32Backend> BitXor for u8x32<T> {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Self(T::bitxor(self.0, rhs.0), PhantomData)
    }
}

// ============================================================================
// Assign operators
// ============================================================================

impl<T: U8x32Backend> AddAssign for u8x32<T> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<T: U8x32Backend> SubAssign for u8x32<T> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<T: U8x32Backend> BitAndAssign for u8x32<T> {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

impl<T: U8x32Backend> BitOrAssign for u8x32<T> {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl<T: U8x32Backend> BitXorAssign for u8x32<T> {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = *self ^ rhs;
    }
}

// ============================================================================
// Scalar broadcast operators (v + 2, etc.)
// ============================================================================

impl<T: U8x32Backend> Add<u8> for u8x32<T> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: u8) -> Self {
        Self(T::add(self.0, T::splat(rhs)), PhantomData)
    }
}

impl<T: U8x32Backend> Sub<u8> for u8x32<T> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: u8) -> Self {
        Self(T::sub(self.0, T::splat(rhs)), PhantomData)
    }
}

// ============================================================================
// Index
// ============================================================================

impl<T: U8x32Backend> Index<usize> for u8x32<T> {
    type Output = u8;
    #[inline(always)]
    fn index(&self, i: usize) -> &u8 {
        assert!(i < 32, "u8x32 index out of bounds: {i}");
        // SAFETY: u8x32's repr is layout-compatible with [u8; 32], and i < 32.
        unsafe { &*(core::ptr::from_ref(self).cast::<u8>()).add(i) }
    }
}

impl<T: U8x32Backend> IndexMut<usize> for u8x32<T> {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut u8 {
        assert!(i < 32, "u8x32 index out of bounds: {i}");
        // SAFETY: u8x32's repr is layout-compatible with [u8; 32], and i < 32.
        unsafe { &mut *(core::ptr::from_mut(self).cast::<u8>()).add(i) }
    }
}

// ============================================================================
// Conversions
// ============================================================================

impl<T: U8x32Backend> From<u8x32<T>> for [u8; 32] {
    #[inline(always)]
    fn from(v: u8x32<T>) -> [u8; 32] {
        T::to_array(v.0)
    }
}

// ============================================================================
// Debug
// ============================================================================

impl<T: U8x32Backend> core::fmt::Debug for u8x32<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let arr = T::to_array(self.0);
        f.debug_tuple("u8x32").field(&arr).finish()
    }
}

// ============================================================================
// Cross-type conversions (u8 ↔ i8 bitcast)
// ============================================================================

impl<T: crate::simd::backends::I8x32Bitcast> u8x32<T> {
    /// Bitcast to i8x32 (reinterpret bits, no conversion).
    #[inline(always)]
    pub fn bitcast_i8x32(self) -> super::i8x32<T> {
        super::i8x32::from_repr_unchecked(T::bitcast_u8_to_i8(self.0))
    }

    /// Bitcast to i8x32 by reference (zero-cost).
    #[inline(always)]
    pub fn bitcast_ref_i8x32(&self) -> &super::i8x32<T> {
        // SAFETY: u8x32 and i8x32 share the same repr (__m256i / [u8;32] / etc.)
        unsafe { &*(core::ptr::from_ref(self).cast()) }
    }

    /// Bitcast to i8x32 by mutable reference (zero-cost).
    #[inline(always)]
    pub fn bitcast_mut_i8x32(&mut self) -> &mut super::i8x32<T> {
        // SAFETY: u8x32 and i8x32 share the same repr
        unsafe { &mut *(core::ptr::from_mut(self).cast()) }
    }
}

// ============================================================================
// Platform-specific concrete impls
// ============================================================================

#[cfg(target_arch = "x86_64")]
impl u8x32<archmage::X64V3Token> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "x86::v3::u8x32"
    }

    /// Get the raw `__m256i` value.
    #[inline(always)]
    pub fn raw(self) -> core::arch::x86_64::__m256i {
        self.0
    }

    /// Create from a raw `__m256i` (token-gated, zero-cost).
    #[inline(always)]
    pub fn from_m256i(_: archmage::X64V3Token, v: core::arch::x86_64::__m256i) -> Self {
        Self(v, PhantomData)
    }
}

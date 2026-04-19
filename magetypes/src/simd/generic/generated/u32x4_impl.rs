//! Generic `u32x4<T>` — 4-lane u32 SIMD vector parameterized by backend.
//!
//! `T` is a token type (e.g., `X64V3Token`, `NeonToken`, `ScalarToken`)
//! that determines the platform-native representation and intrinsics used.
//! The struct delegates all operations to the [`U32x4Backend`] trait.

#![allow(clippy::should_implement_trait)]

use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Index,
    IndexMut, Mul, MulAssign, Sub, SubAssign,
};

use crate::simd::backends::U32x4Backend;

/// 4-lane u32 SIMD vector, generic over backend `T`.
///
/// `T` is a token type that proves CPU support for the required SIMD features.
/// The inner representation is `T::Repr` (e.g., `__m128i` on x86, `uint32x4_t` on ARM).
///
/// **The token is stored** (as a zero-sized field) so methods receiving
/// `self: u32x4<T>` can re-supply it to backend operations that
/// require a token value (e.g. `T::splat(token, v)`). This carries the
/// token-as-feature-proof guarantee through every method call without
/// runtime overhead — `T` is ZST, so `sizeof(u32x4<T>) == sizeof(T::Repr)`
/// and `align_of(u32x4<T>) == align_of(T::Repr)` under `#[repr(C)]`.
///
/// # Layout
///
/// `#[repr(C)]` with a ZST trailing field: `T::Repr` lives at offset 0
/// and `T` is a 0-byte tail. Bitcasts between `u32x4<T>` values of
/// different element-types are sound when the Repr types share a layout
/// (e.g. `__m128` and `__m128i` are both 16-byte aligned 128-bit values).
/// `#[repr(transparent)]` cannot be used because Rust cannot prove at
/// the struct definition site that a generic `T` is a 1-ZST.
///
/// Construction requires a token value to prove CPU support at runtime.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct u32x4<T: U32x4Backend>(pub(crate) T::Repr, pub(crate) T);

// Layout invariant: struct is `#[repr(C)]` with a trailing ZST `T`
// field, so `sizeof/alignof(u32x4<T>) == sizeof/alignof(T::Repr)`
// iff `T` is a 1-ZST. Every archmage token currently satisfies this;
// if a future refactor adds a non-ZST field to a token, this const
// assert fires at compile time.
const _: () = {
    assert!(
        core::mem::size_of::<u32x4<archmage::ScalarToken>>()
            == core::mem::size_of::<
                <archmage::ScalarToken as crate::simd::backends::U32x4Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<u32x4<archmage::ScalarToken>>()
            == core::mem::align_of::<
                <archmage::ScalarToken as crate::simd::backends::U32x4Backend>::Repr,
            >()
    );
};

#[cfg(target_arch = "x86_64")]
const _: () = {
    assert!(
        core::mem::size_of::<u32x4<archmage::X64V3Token>>()
            == core::mem::size_of::<
                <archmage::X64V3Token as crate::simd::backends::U32x4Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<u32x4<archmage::X64V3Token>>()
            == core::mem::align_of::<
                <archmage::X64V3Token as crate::simd::backends::U32x4Backend>::Repr,
            >()
    );
};

#[cfg(target_arch = "aarch64")]
const _: () = {
    assert!(
        core::mem::size_of::<u32x4<archmage::NeonToken>>()
            == core::mem::size_of::<
                <archmage::NeonToken as crate::simd::backends::U32x4Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<u32x4<archmage::NeonToken>>()
            == core::mem::align_of::<
                <archmage::NeonToken as crate::simd::backends::U32x4Backend>::Repr,
            >()
    );
};

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
const _: () = {
    assert!(
        core::mem::size_of::<u32x4<archmage::Wasm128Token>>()
            == core::mem::size_of::<
                <archmage::Wasm128Token as crate::simd::backends::U32x4Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<u32x4<archmage::Wasm128Token>>()
            == core::mem::align_of::<
                <archmage::Wasm128Token as crate::simd::backends::U32x4Backend>::Repr,
            >()
    );
};

impl<T: U32x4Backend> u32x4<T> {
    /// Number of u32 lanes.
    pub const LANES: usize = 4;

    // ====== Construction (token-gated) ======

    /// Broadcast scalar to all 4 lanes.
    #[inline(always)]
    pub fn splat(token: T, v: u32) -> Self {
        Self(T::splat(token, v), token)
    }

    /// All lanes zero.
    #[inline(always)]
    pub fn zero(token: T) -> Self {
        Self(T::zero(token), token)
    }

    /// Load from a `[u32; 4]` array.
    #[inline(always)]
    pub fn load(token: T, data: &[u32; 4]) -> Self {
        Self(T::load(token, data), token)
    }

    /// Create from array (zero-cost where possible).
    #[inline(always)]
    pub fn from_array(token: T, arr: [u32; 4]) -> Self {
        Self(T::from_array(token, arr), token)
    }

    /// Create from slice. Panics if `slice.len() < 4`.
    #[inline(always)]
    pub fn from_slice(token: T, slice: &[u32]) -> Self {
        let arr: [u32; 4] = slice[..4].try_into().unwrap();
        Self(T::from_array(token, arr), token)
    }

    /// Split a slice into SIMD-width chunks and a scalar remainder.
    ///
    /// Returns `(&[[u32; 4]], &[u32])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[inline(always)]
    pub fn partition_slice(_: T, data: &[u32]) -> (&[[u32; 4]], &[u32]) {
        let bulk = data.len() / 4;
        let (head, tail) = data.split_at(bulk * 4);
        // SAFETY: head.len() is bulk * 4, so it's exactly `bulk` chunks of [u32; 4].
        // The pointer cast is valid because [u32] and [[u32; 4]] have the same alignment.
        let chunks = unsafe { core::slice::from_raw_parts(head.as_ptr().cast::<[u32; 4]>(), bulk) };
        (chunks, tail)
    }

    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    ///
    /// Returns `(&mut [[u32; 4]], &mut [u32])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[inline(always)]
    pub fn partition_slice_mut(_: T, data: &mut [u32]) -> (&mut [[u32; 4]], &mut [u32]) {
        let bulk = data.len() / 4;
        let (head, tail) = data.split_at_mut(bulk * 4);
        // SAFETY: head.len() is bulk * 4, so it's exactly `bulk` chunks of [u32; 4].
        // The pointer cast is valid because [u32] and [[u32; 4]] have the same alignment.
        let chunks =
            unsafe { core::slice::from_raw_parts_mut(head.as_mut_ptr().cast::<[u32; 4]>(), bulk) };
        (chunks, tail)
    }

    // ====== Accessors ======

    /// Store to array.
    #[inline(always)]
    pub fn store(self, out: &mut [u32; 4]) {
        T::store(self.1, self.0, out);
    }

    /// Convert to array.
    #[inline(always)]
    pub fn to_array(self) -> [u32; 4] {
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

    /// Lane-wise minimum (unsigned).
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(T::min(self.1, self.0, other.0), self.1)
    }

    /// Lane-wise maximum (unsigned).
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(T::max(self.1, self.0, other.0), self.1)
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

    /// Lane-wise less-than, unsigned (returns mask).
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(T::simd_lt(self.1, self.0, other.0), self.1)
    }

    /// Lane-wise less-than-or-equal, unsigned (returns mask).
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(T::simd_le(self.1, self.0, other.0), self.1)
    }

    /// Lane-wise greater-than, unsigned (returns mask).
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(T::simd_gt(self.1, self.0, other.0), self.1)
    }

    /// Lane-wise greater-than-or-equal, unsigned (returns mask).
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

    /// Sum all 4 lanes (wrapping).
    #[inline(always)]
    pub fn reduce_add(self) -> u32 {
        T::reduce_add(self.1, self.0)
    }

    // ====== Shifts ======

    /// Shift left by constant.
    #[inline(always)]
    pub fn shl_const<const N: i32>(self) -> Self {
        Self(T::shl_const::<N>(self.1, self.0), self.1)
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

    /// True if all lanes have their high bit set (all-1s mask).
    #[inline(always)]
    pub fn all_true(self) -> bool {
        T::all_true(self.1, self.0)
    }

    /// True if any lane has its high bit set.
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

impl<T: U32x4Backend> Add for u32x4<T> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(T::add(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: U32x4Backend> Sub for u32x4<T> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(T::sub(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: U32x4Backend> Mul for u32x4<T> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(T::mul(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: U32x4Backend> BitAnd for u32x4<T> {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(T::bitand(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: U32x4Backend> BitOr for u32x4<T> {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self(T::bitor(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: U32x4Backend> BitXor for u32x4<T> {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Self(T::bitxor(self.1, self.0, rhs.0), self.1)
    }
}

// ============================================================================
// Assign operators
// ============================================================================

impl<T: U32x4Backend> AddAssign for u32x4<T> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<T: U32x4Backend> SubAssign for u32x4<T> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<T: U32x4Backend> MulAssign for u32x4<T> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<T: U32x4Backend> BitAndAssign for u32x4<T> {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

impl<T: U32x4Backend> BitOrAssign for u32x4<T> {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl<T: U32x4Backend> BitXorAssign for u32x4<T> {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = *self ^ rhs;
    }
}

// ============================================================================
// Scalar broadcast operators (v + 2, v * 3, etc.)
// ============================================================================

impl<T: U32x4Backend> Add<u32> for u32x4<T> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: u32) -> Self {
        Self(T::add(self.1, self.0, T::splat(self.1, rhs)), self.1)
    }
}

impl<T: U32x4Backend> Sub<u32> for u32x4<T> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: u32) -> Self {
        Self(T::sub(self.1, self.0, T::splat(self.1, rhs)), self.1)
    }
}

impl<T: U32x4Backend> Mul<u32> for u32x4<T> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: u32) -> Self {
        Self(T::mul(self.1, self.0, T::splat(self.1, rhs)), self.1)
    }
}

// ============================================================================
// Index
// ============================================================================

impl<T: U32x4Backend> Index<usize> for u32x4<T> {
    type Output = u32;
    #[inline(always)]
    fn index(&self, i: usize) -> &u32 {
        assert!(i < 4, "u32x4 index out of bounds: {i}");
        // SAFETY: u32x4's repr is layout-compatible with [u32; 4], and i < 4.
        unsafe { &*(core::ptr::from_ref(self).cast::<u32>()).add(i) }
    }
}

impl<T: U32x4Backend> IndexMut<usize> for u32x4<T> {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut u32 {
        assert!(i < 4, "u32x4 index out of bounds: {i}");
        // SAFETY: u32x4's repr is layout-compatible with [u32; 4], and i < 4.
        unsafe { &mut *(core::ptr::from_mut(self).cast::<u32>()).add(i) }
    }
}

// ============================================================================
// Conversions
// ============================================================================

impl<T: U32x4Backend> From<u32x4<T>> for [u32; 4] {
    #[inline(always)]
    fn from(v: u32x4<T>) -> [u32; 4] {
        T::to_array(v.1, v.0)
    }
}

// ============================================================================
// Debug
// ============================================================================

impl<T: U32x4Backend> core::fmt::Debug for u32x4<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let arr = T::to_array(self.1, self.0);
        f.debug_tuple("u32x4").field(&arr).finish()
    }
}

// ============================================================================
// Cross-type conversions (i32 ↔ u32 bitcast)
// ============================================================================

impl<T: crate::simd::backends::U32x4Bitcast> u32x4<T> {
    /// Bitcast to i32x4 (reinterpret bits, no conversion).
    #[inline(always)]
    pub fn bitcast_to_i32(self) -> super::i32x4<T> {
        super::i32x4::from_repr_unchecked(self.1, T::bitcast_u32_to_i32(self.1, self.0))
    }

    /// Bitcast to i32x4 by reference (zero-cost).
    #[inline(always)]
    pub fn bitcast_ref_i32x4(&self) -> &super::i32x4<T> {
        // SAFETY: u32x4 and i32x4 share the same repr (__m128i / [u32;4] / etc.)
        unsafe { &*(core::ptr::from_ref(self).cast()) }
    }

    /// Bitcast to i32x4 by mutable reference (zero-cost).
    #[inline(always)]
    pub fn bitcast_mut_i32x4(&mut self) -> &mut super::i32x4<T> {
        // SAFETY: u32x4 and i32x4 share the same repr
        unsafe { &mut *(core::ptr::from_mut(self).cast()) }
    }

    // ====== Backward-compatible aliases ======

    /// Alias for [`bitcast_to_i32`](Self::bitcast_to_i32).
    #[inline(always)]
    pub fn bitcast_i32x4(self) -> super::i32x4<T> {
        self.bitcast_to_i32()
    }
}

// ============================================================================
// Platform-specific concrete impls
// ============================================================================

#[cfg(target_arch = "x86_64")]
impl u32x4<archmage::X64V3Token> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "x86::v3::u32x4"
    }

    /// Get the raw `__m128i` value.
    #[inline(always)]
    pub fn raw(self) -> core::arch::x86_64::__m128i {
        self.0
    }

    /// Create from a raw `__m128i` (token-gated, zero-cost).
    #[inline(always)]
    pub fn from_m128i(token: archmage::X64V3Token, v: core::arch::x86_64::__m128i) -> Self {
        Self(v, token)
    }
}

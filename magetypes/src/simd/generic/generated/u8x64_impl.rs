//! Generic `u8x64<T>` — 64-lane u8 SIMD vector parameterized by backend.
//!
//! `T` is a token type (e.g., `X64V4Token`, `ScalarToken`)
//! that determines the platform-native representation and intrinsics used.
//! The struct delegates all operations to the [`U8x64Backend`] trait.

#![allow(clippy::should_implement_trait)]

use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Index,
    IndexMut, Sub, SubAssign,
};

use crate::simd::backends::U8x64Backend;

/// 64-lane u8 SIMD vector, generic over backend `T`.
///
/// `T` is a token type that proves CPU support for the required SIMD features.
/// The inner representation is `T::Repr` (e.g., `__m512i` on AVX-512, `[u8; 64]` on scalar).
///
/// **The token is stored** (as a zero-sized field) so methods receiving
/// `self: u8x64<T>` can re-supply it to backend operations that
/// require a token value (e.g. `T::splat(token, v)`). This carries the
/// token-as-feature-proof guarantee through every method call without
/// runtime overhead — `T` is ZST, so `sizeof(u8x64<T>) == sizeof(T::Repr)`
/// and `align_of(u8x64<T>) == align_of(T::Repr)` under `#[repr(C)]`.
///
/// # Layout
///
/// `#[repr(C)]` with a ZST trailing field: `T::Repr` lives at offset 0
/// and `T` is a 0-byte tail. Bitcasts between `u8x64<T>` values of
/// different element-types are sound when the Repr types share a layout
/// (e.g. `__m128` and `__m128i` are both 16-byte aligned 128-bit values).
/// `#[repr(transparent)]` cannot be used because Rust cannot prove at
/// the struct definition site that a generic `T` is a 1-ZST.
///
/// Construction requires a token value to prove CPU support at runtime.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct u8x64<T: U8x64Backend>(pub(crate) T::Repr, pub(crate) T);

// Layout invariant: struct is `#[repr(C)]` with a trailing ZST `T`
// field, so `sizeof/alignof(u8x64<T>) == sizeof/alignof(T::Repr)`
// iff `T` is a 1-ZST. Every archmage token currently satisfies this;
// if a future refactor adds a non-ZST field to a token, this const
// assert fires at compile time.
const _: () = {
    assert!(
        core::mem::size_of::<u8x64<archmage::ScalarToken>>()
            == core::mem::size_of::<
                <archmage::ScalarToken as crate::simd::backends::U8x64Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<u8x64<archmage::ScalarToken>>()
            == core::mem::align_of::<
                <archmage::ScalarToken as crate::simd::backends::U8x64Backend>::Repr,
            >()
    );
};

#[cfg(target_arch = "x86_64")]
const _: () = {
    assert!(
        core::mem::size_of::<u8x64<archmage::X64V3Token>>()
            == core::mem::size_of::<
                <archmage::X64V3Token as crate::simd::backends::U8x64Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<u8x64<archmage::X64V3Token>>()
            == core::mem::align_of::<
                <archmage::X64V3Token as crate::simd::backends::U8x64Backend>::Repr,
            >()
    );
};

// Native AVX-512 (`__m512`/`__m512d`/`__m512i`) — gated on the
// `avx512` feature, which is how archmage exposes X64V4Token's
// 512-bit backend impls.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
const _: () = {
    assert!(
        core::mem::size_of::<u8x64<archmage::X64V4Token>>()
            == core::mem::size_of::<
                <archmage::X64V4Token as crate::simd::backends::U8x64Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<u8x64<archmage::X64V4Token>>()
            == core::mem::align_of::<
                <archmage::X64V4Token as crate::simd::backends::U8x64Backend>::Repr,
            >()
    );
};

#[cfg(target_arch = "aarch64")]
const _: () = {
    assert!(
        core::mem::size_of::<u8x64<archmage::NeonToken>>()
            == core::mem::size_of::<
                <archmage::NeonToken as crate::simd::backends::U8x64Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<u8x64<archmage::NeonToken>>()
            == core::mem::align_of::<
                <archmage::NeonToken as crate::simd::backends::U8x64Backend>::Repr,
            >()
    );
};

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
const _: () = {
    assert!(
        core::mem::size_of::<u8x64<archmage::Wasm128Token>>()
            == core::mem::size_of::<
                <archmage::Wasm128Token as crate::simd::backends::U8x64Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<u8x64<archmage::Wasm128Token>>()
            == core::mem::align_of::<
                <archmage::Wasm128Token as crate::simd::backends::U8x64Backend>::Repr,
            >()
    );
};

impl<T: U8x64Backend> u8x64<T> {
    /// Number of u8 lanes.
    pub const LANES: usize = 64;

    // ====== Construction (token-gated) ======

    /// Broadcast scalar to all 64 lanes.
    #[inline(always)]
    pub fn splat(token: T, v: u8) -> Self {
        Self(T::splat(token, v), token)
    }

    /// All lanes zero.
    #[inline(always)]
    pub fn zero(token: T) -> Self {
        Self(T::zero(token), token)
    }

    /// Load from a `[u8; 64]` array.
    #[inline(always)]
    pub fn load(token: T, data: &[u8; 64]) -> Self {
        Self(T::load(token, data), token)
    }

    /// Create from array (zero-cost where possible).
    #[inline(always)]
    pub fn from_array(token: T, arr: [u8; 64]) -> Self {
        Self(T::from_array(token, arr), token)
    }

    /// Create from slice. Panics if `slice.len() < 64`.
    #[inline(always)]
    pub fn from_slice(token: T, slice: &[u8]) -> Self {
        let arr: [u8; 64] = slice[..64].try_into().unwrap();
        Self(T::from_array(token, arr), token)
    }

    /// Split a slice into SIMD-width chunks and a scalar remainder.
    ///
    /// Returns `(&[[u8; 64]], &[u8])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[inline(always)]
    pub fn partition_slice(_: T, data: &[u8]) -> (&[[u8; 64]], &[u8]) {
        let bulk = data.len() / 64;
        let (head, tail) = data.split_at(bulk * 64);
        // SAFETY: head.len() is bulk * 64, so it's exactly `bulk` chunks of [u8; 64].
        // The pointer cast is valid because [u8] and [[u8; 64]] have the same alignment.
        let chunks = unsafe { core::slice::from_raw_parts(head.as_ptr().cast::<[u8; 64]>(), bulk) };
        (chunks, tail)
    }

    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    ///
    /// Returns `(&mut [[u8; 64]], &mut [u8])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[inline(always)]
    pub fn partition_slice_mut(_: T, data: &mut [u8]) -> (&mut [[u8; 64]], &mut [u8]) {
        let bulk = data.len() / 64;
        let (head, tail) = data.split_at_mut(bulk * 64);
        // SAFETY: head.len() is bulk * 64, so it's exactly `bulk` chunks of [u8; 64].
        // The pointer cast is valid because [u8] and [[u8; 64]] have the same alignment.
        let chunks =
            unsafe { core::slice::from_raw_parts_mut(head.as_mut_ptr().cast::<[u8; 64]>(), bulk) };
        (chunks, tail)
    }

    // ====== Accessors ======

    /// Store to array.
    #[inline(always)]
    pub fn store(self, out: &mut [u8; 64]) {
        T::store(self.1, self.0, out);
    }

    /// Convert to array.
    #[inline(always)]
    pub fn to_array(self) -> [u8; 64] {
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

    /// Sum all 64 lanes (wrapping).
    #[inline(always)]
    pub fn reduce_add(self) -> u8 {
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

    /// Extract the high bit of each 8-bit lane as a bitmask.
    #[inline(always)]
    pub fn bitmask(self) -> u64 {
        T::bitmask(self.1, self.0)
    }
}

// ============================================================================
// Operator implementations
// ============================================================================

impl<T: U8x64Backend> Add for u8x64<T> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(T::add(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: U8x64Backend> Sub for u8x64<T> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(T::sub(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: U8x64Backend> BitAnd for u8x64<T> {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(T::bitand(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: U8x64Backend> BitOr for u8x64<T> {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self(T::bitor(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: U8x64Backend> BitXor for u8x64<T> {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Self(T::bitxor(self.1, self.0, rhs.0), self.1)
    }
}

// ============================================================================
// Assign operators
// ============================================================================

impl<T: U8x64Backend> AddAssign for u8x64<T> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<T: U8x64Backend> SubAssign for u8x64<T> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<T: U8x64Backend> BitAndAssign for u8x64<T> {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

impl<T: U8x64Backend> BitOrAssign for u8x64<T> {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl<T: U8x64Backend> BitXorAssign for u8x64<T> {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = *self ^ rhs;
    }
}

// ============================================================================
// Scalar broadcast operators (v + 2, etc.)
// ============================================================================

impl<T: U8x64Backend> Add<u8> for u8x64<T> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: u8) -> Self {
        Self(T::add(self.1, self.0, T::splat(self.1, rhs)), self.1)
    }
}

impl<T: U8x64Backend> Sub<u8> for u8x64<T> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: u8) -> Self {
        Self(T::sub(self.1, self.0, T::splat(self.1, rhs)), self.1)
    }
}

// ============================================================================
// Index
// ============================================================================

impl<T: U8x64Backend> Index<usize> for u8x64<T> {
    type Output = u8;
    #[inline(always)]
    fn index(&self, i: usize) -> &u8 {
        assert!(i < 64, "u8x64 index out of bounds: {i}");
        // SAFETY: u8x64's repr is layout-compatible with [u8; 64], and i < 64.
        unsafe { &*(core::ptr::from_ref(self).cast::<u8>()).add(i) }
    }
}

impl<T: U8x64Backend> IndexMut<usize> for u8x64<T> {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut u8 {
        assert!(i < 64, "u8x64 index out of bounds: {i}");
        // SAFETY: u8x64's repr is layout-compatible with [u8; 64], and i < 64.
        unsafe { &mut *(core::ptr::from_mut(self).cast::<u8>()).add(i) }
    }
}

// ============================================================================
// Conversions
// ============================================================================

impl<T: U8x64Backend> From<u8x64<T>> for [u8; 64] {
    #[inline(always)]
    fn from(v: u8x64<T>) -> [u8; 64] {
        T::to_array(v.1, v.0)
    }
}

// ============================================================================
// Debug
// ============================================================================

impl<T: U8x64Backend> core::fmt::Debug for u8x64<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let arr = T::to_array(self.1, self.0);
        f.debug_tuple("u8x64").field(&arr).finish()
    }
}

// ============================================================================
// Platform-specific implementation info
// ============================================================================

#[cfg(target_arch = "x86_64")]
impl u8x64<archmage::X64V3Token> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "polyfill::v3_512::u8x64"
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl u8x64<archmage::X64V4Token> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "x86::v4::u8x64"
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl u8x64<archmage::X64V4xToken> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "x86::v4x::u8x64"
    }
}

// ============================================================================
// Extension: popcnt (requires Modern token)
// ============================================================================

#[cfg(feature = "avx512")]
impl<T: crate::simd::backends::u8x64PopcntBackend> u8x64<T> {
    /// Count set bits in each lane (popcnt).
    ///
    /// Returns a vector where each lane contains the number of 1-bits
    /// in the corresponding lane of `self`.
    ///
    /// Requires AVX-512 Modern token (VPOPCNTDQ or BITALG extension).
    #[inline(always)]
    pub fn popcnt(self) -> Self {
        Self(T::popcnt(self.1, self.0), self.1)
    }
}

//! Generic `u64x8<T>` — 8-lane u64 SIMD vector parameterized by backend.
//!
//! `T` is a token type (e.g., `X64V4Token`, `ScalarToken`)
//! that determines the platform-native representation and intrinsics used.
//! The struct delegates all operations to the [`U64x8Backend`] trait.

#![allow(clippy::should_implement_trait)]

use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Index,
    IndexMut, Sub, SubAssign,
};

use crate::simd::backends::U64x8Backend;

/// 8-lane u64 SIMD vector, generic over backend `T`.
///
/// `T` is a token type that proves CPU support for the required SIMD features.
/// The inner representation is `T::Repr` (e.g., `__m512i` on AVX-512, `[u64; 8]` on scalar).
///
/// **The token is stored** (as a zero-sized field) so methods receiving
/// `self: u64x8<T>` can re-supply it to backend operations that
/// require a token value (e.g. `T::splat(token, v)`). This carries the
/// token-as-feature-proof guarantee through every method call without
/// runtime overhead — `T` is ZST, so `sizeof(u64x8<T>) == sizeof(T::Repr)`
/// and `align_of(u64x8<T>) == align_of(T::Repr)` under `#[repr(C)]`.
///
/// # Layout
///
/// `#[repr(C)]` with a ZST trailing field: `T::Repr` lives at offset 0
/// and `T` is a 0-byte tail. Bitcasts between `u64x8<T>` values of
/// different element-types are sound when the Repr types share a layout
/// (e.g. `__m128` and `__m128i` are both 16-byte aligned 128-bit values).
/// `#[repr(transparent)]` cannot be used because Rust cannot prove at
/// the struct definition site that a generic `T` is a 1-ZST.
///
/// Construction requires a token value to prove CPU support at runtime.
///
/// # Note
///
/// 64-bit integer SIMD has limited native support: no hardware multiply on
/// AVX2/NEON/WASM.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct u64x8<T: U64x8Backend>(pub(crate) T::Repr, pub(crate) T);
// The `unsafe impl` and the checks behind it live in `simd_storage`.
crate::simd_storage::impl_token_storage!(u64x8, U64x8Backend);

// Layout invariant: struct is `#[repr(C)]` with a trailing ZST `T`
// field, so `sizeof/alignof(u64x8<T>) == sizeof/alignof(T::Repr)`
// when `T` is a 1-ZST. Every archmage token currently satisfies this;
// if a future refactor adds a non-ZST field to a token, this const
// assert fires at compile time.
const _: () = {
    assert!(
        core::mem::size_of::<u64x8<archmage::ScalarToken>>()
            == core::mem::size_of::<
                <archmage::ScalarToken as crate::simd::backends::U64x8Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<u64x8<archmage::ScalarToken>>()
            == core::mem::align_of::<
                <archmage::ScalarToken as crate::simd::backends::U64x8Backend>::Repr,
            >()
    );
};

#[cfg(target_arch = "x86_64")]
const _: () = {
    assert!(
        core::mem::size_of::<u64x8<archmage::X64V3Token>>()
            == core::mem::size_of::<
                <archmage::X64V3Token as crate::simd::backends::U64x8Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<u64x8<archmage::X64V3Token>>()
            == core::mem::align_of::<
                <archmage::X64V3Token as crate::simd::backends::U64x8Backend>::Repr,
            >()
    );
};

// Native AVX-512 (`__m512`/`__m512d`/`__m512i`) — gated on the
// `avx512` feature, which is how archmage exposes X64V4Token's
// 512-bit backend impls.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
const _: () = {
    assert!(
        core::mem::size_of::<u64x8<archmage::X64V4Token>>()
            == core::mem::size_of::<
                <archmage::X64V4Token as crate::simd::backends::U64x8Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<u64x8<archmage::X64V4Token>>()
            == core::mem::align_of::<
                <archmage::X64V4Token as crate::simd::backends::U64x8Backend>::Repr,
            >()
    );
};

#[cfg(target_arch = "aarch64")]
const _: () = {
    assert!(
        core::mem::size_of::<u64x8<archmage::NeonToken>>()
            == core::mem::size_of::<
                <archmage::NeonToken as crate::simd::backends::U64x8Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<u64x8<archmage::NeonToken>>()
            == core::mem::align_of::<
                <archmage::NeonToken as crate::simd::backends::U64x8Backend>::Repr,
            >()
    );
};

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
const _: () = {
    assert!(
        core::mem::size_of::<u64x8<archmage::Wasm128Token>>()
            == core::mem::size_of::<
                <archmage::Wasm128Token as crate::simd::backends::U64x8Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<u64x8<archmage::Wasm128Token>>()
            == core::mem::align_of::<
                <archmage::Wasm128Token as crate::simd::backends::U64x8Backend>::Repr,
            >()
    );
};

impl<T: U64x8Backend> u64x8<T> {
    /// Number of u64 lanes.
    pub const LANES: usize = 8;

    // ====== Construction (token-gated) ======

    /// Broadcast scalar to all 8 lanes.
    #[inline(always)]
    pub fn splat_t(token: T, v: u64) -> Self {
        Self(T::splat(token, v), token)
    }

    /// All lanes zero.
    #[inline(always)]
    pub fn zero_t(token: T) -> Self {
        Self(T::zero(token), token)
    }

    /// Load from a `[u64; 8]` array.
    #[inline(always)]
    pub fn load_t(token: T, data: &[u64; 8]) -> Self {
        Self(T::load(token, data), token)
    }

    /// Create from array (zero-cost where possible).
    #[inline(always)]
    pub fn from_array_t(token: T, arr: [u64; 8]) -> Self {
        Self(T::from_array(token, arr), token)
    }

    /// Create from slice. Panics if `slice.len() < 8`.
    #[inline(always)]
    pub fn from_slice_t(token: T, slice: &[u64]) -> Self {
        let arr: [u64; 8] = slice[..8].try_into().unwrap();
        Self(T::from_array(token, arr), token)
    }

    /// Split a slice into SIMD-width chunks and a scalar remainder.
    ///
    /// Returns `(&[[u64; 8]], &[u64])` — fixed-size arrays suitable
    /// for [`load_t`](Self::load_t), plus any leftover elements.
    #[inline(always)]
    pub fn partition_slice_t(_: T, data: &[u64]) -> (&[[u64; 8]], &[u64]) {
        data.as_chunks::<8>()
    }

    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    ///
    /// Returns `(&mut [[u64; 8]], &mut [u64])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load_t`](Self::load_t), plus any leftover elements.
    #[inline(always)]
    pub fn partition_slice_mut_t(_: T, data: &mut [u64]) -> (&mut [[u64; 8]], &mut [u64]) {
        data.as_chunks_mut::<8>()
    }

    // ====== Accessors ======

    /// Store to array.
    #[inline(always)]
    pub fn store(self, out: &mut [u64; 8]) {
        T::store(self.1, self.0, out);
    }

    /// Convert to array.
    #[inline(always)]
    pub fn to_array(self) -> [u64; 8] {
        T::to_array(self.1, self.0)
    }

    /// Get the underlying platform representation.
    #[inline(always)]
    pub fn into_repr(self) -> T::Repr {
        self.0
    }

    /// Wrap a platform representation (token-gated).
    #[inline(always)]
    pub fn from_repr_t(token: T, repr: T::Repr) -> Self {
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

    /// Sum all 8 lanes (wrapping).
    #[inline(always)]
    pub fn reduce_add(self) -> u64 {
        T::reduce_add(self.1, self.0)
    }

    // ====== Shifts ======

    /// Shift left by constant.
    ///
    /// `N` must be in `0..=63`; out-of-range `N` fails to compile,
    /// identically on every backend.
    #[inline(always)]
    pub fn shl_const<const N: i32>(self) -> Self {
        const { assert!(N >= 0 && N <= 63, "shift amount out of range") };
        Self(T::shl_const::<N>(self.1, self.0), self.1)
    }

    /// Logical shift right by constant (zero-filling).
    ///
    /// `N` must be in `0..=63` (`N == 0` is the identity shift);
    /// out-of-range `N` fails to compile, identically on every backend.
    #[inline(always)]
    pub fn shr_logical_const<const N: i32>(self) -> Self {
        const { assert!(N >= 0 && N <= 63, "shift amount out of range") };
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

    /// Extract the high bit of each 64-bit lane as a bitmask.
    #[inline(always)]
    pub fn bitmask(self) -> u64 {
        T::bitmask(self.1, self.0)
    }
}

// ============================================================================
// Operator implementations
// ============================================================================

impl<T: U64x8Backend> Add for u64x8<T> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(T::add(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: U64x8Backend> Sub for u64x8<T> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(T::sub(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: U64x8Backend> BitAnd for u64x8<T> {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(T::bitand(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: U64x8Backend> BitOr for u64x8<T> {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self(T::bitor(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: U64x8Backend> BitXor for u64x8<T> {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Self(T::bitxor(self.1, self.0, rhs.0), self.1)
    }
}

// ============================================================================
// Assign operators
// ============================================================================

impl<T: U64x8Backend> AddAssign for u64x8<T> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<T: U64x8Backend> SubAssign for u64x8<T> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<T: U64x8Backend> BitAndAssign for u64x8<T> {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

impl<T: U64x8Backend> BitOrAssign for u64x8<T> {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl<T: U64x8Backend> BitXorAssign for u64x8<T> {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = *self ^ rhs;
    }
}

// ============================================================================
// Scalar broadcast operators (v + 2, etc.)
// ============================================================================

impl<T: U64x8Backend> Add<u64> for u64x8<T> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: u64) -> Self {
        Self(T::add(self.1, self.0, T::splat(self.1, rhs)), self.1)
    }
}

impl<T: U64x8Backend> Sub<u64> for u64x8<T> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: u64) -> Self {
        Self(T::sub(self.1, self.0, T::splat(self.1, rhs)), self.1)
    }
}

// ============================================================================
// Index
// ============================================================================

impl<T: U64x8Backend> Index<usize> for u64x8<T> {
    type Output = u64;
    #[inline(always)]
    fn index(&self, i: usize) -> &u64 {
        &crate::simd_storage::view::<_, [u64; 8]>(&self.0)[i]
    }
}

impl<T: U64x8Backend> IndexMut<usize> for u64x8<T> {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut u64 {
        &mut crate::simd_storage::view_mut::<_, [u64; 8]>(&mut self.0)[i]
    }
}

// ============================================================================
// Conversions
// ============================================================================

impl<T: U64x8Backend> From<u64x8<T>> for [u64; 8] {
    #[inline(always)]
    fn from(v: u64x8<T>) -> [u64; 8] {
        T::to_array(v.1, v.0)
    }
}

// ============================================================================
// Debug
// ============================================================================

impl<T: U64x8Backend> core::fmt::Debug for u64x8<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let arr = T::to_array(self.1, self.0);
        f.debug_tuple("u64x8").field(&arr).finish()
    }
}

// ============================================================================
// Platform-specific concrete impls
// ============================================================================

impl u64x8<archmage::ScalarToken> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "scalar::u64x8"
    }
}

#[cfg(target_arch = "x86_64")]
impl u64x8<archmage::X64V3Token> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "polyfill::v3_512::u64x8"
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl u64x8<archmage::X64V4Token> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "x86::v4::u64x8"
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl u64x8<archmage::X64V4xToken> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "x86::v4x::u64x8"
    }
}

#[cfg(target_arch = "aarch64")]
impl u64x8<archmage::NeonToken> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "polyfill::neon_512::u64x8"
    }
}

#[cfg(target_arch = "wasm32")]
impl u64x8<archmage::Wasm128Token> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "polyfill::wasm128_512::u64x8"
    }
}

// ============================================================================
// Extension: popcnt (requires Modern token)
// ============================================================================

#[cfg(feature = "avx512")]
impl<T: crate::simd::backends::u64x8PopcntBackend> u64x8<T> {
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
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl u64x8<archmage::X64V4Token> {
    /// Get the raw `__m512i` value.
    #[inline(always)]
    pub fn raw(self) -> core::arch::x86_64::__m512i {
        self.0
    }

    /// Wrap a raw `__m512i` using an existing CPU capability token.
    #[inline(always)]
    pub fn from_m512i_t(token: archmage::X64V4Token, value: core::arch::x86_64::__m512i) -> Self {
        Self(value, token)
    }

    /// Wrap a raw `__m512i` using an explicit CPU capability token.
    ///
    /// The caller does not need a target-feature annotation.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn from_raw_t(token: archmage::X64V4Token, value: core::arch::x86_64::__m512i) -> Self {
        Self(value, token)
    }

    /// Wrap a raw `__m512i` in a matching target-feature context.
    ///
    /// Rust requires the caller to enable the `v4` tier's features.
    /// Use an archmage `#[rite(v4)]` helper or `#[arcane]` entry point.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
    )]
    #[inline]
    pub fn from_raw(value: core::arch::x86_64::__m512i) -> Self {
        Self(value, archmage::X64V4Token::from_context())
    }
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl u64x8<archmage::X64V4xToken> {
    /// Get the raw `__m512i` value.
    #[inline(always)]
    pub fn raw(self) -> core::arch::x86_64::__m512i {
        self.0
    }

    /// Wrap a raw `__m512i` using an existing CPU capability token.
    #[inline(always)]
    pub fn from_m512i_t(token: archmage::X64V4xToken, value: core::arch::x86_64::__m512i) -> Self {
        Self(value, token)
    }

    /// Wrap a raw `__m512i` using an explicit CPU capability token.
    ///
    /// The caller does not need a target-feature annotation.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn from_raw_t(token: archmage::X64V4xToken, value: core::arch::x86_64::__m512i) -> Self {
        Self(value, token)
    }

    /// Wrap a raw `__m512i` in a matching target-feature context.
    ///
    /// Rust requires the caller to enable the `v4x` tier's features.
    /// Use an archmage `#[rite(v4x)]` helper or `#[arcane]` entry point.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512vpopcntdq,avx512ifma,avx512vbmi,avx512vbmi2,avx512bitalg,avx512vnni,vpclmulqdq,gfni,vaes"
    )]
    #[inline]
    pub fn from_raw(value: core::arch::x86_64::__m512i) -> Self {
        Self(value, archmage::X64V4xToken::from_context())
    }
}
// Generated deprecated token-constructor forwarders. Do not edit.
impl<T: U64x8Backend> u64x8<T> {
    #[inline(always)]
    #[doc = "Deprecated token-taking spelling of [`Self::splat_t`].\n\nUse `splat_t` to keep explicit-token construction when `splat` becomes tokenless in magetypes 0.10."]
    #[deprecated(note = "Use splat_t(token, v); splat becomes tokenless in magetypes 0.10.")]
    #[forbid(unsafe_code)]
    pub fn splat(token: T, v: u64) -> Self {
        Self::splat_t(token, v)
    }
    #[inline(always)]
    #[doc = "Deprecated token-taking spelling of [`Self::zero_t`].\n\nUse `zero_t` to keep explicit-token construction when `zero` becomes tokenless in magetypes 0.10."]
    #[deprecated(note = "Use zero_t(token); zero becomes tokenless in magetypes 0.10.")]
    #[forbid(unsafe_code)]
    pub fn zero(token: T) -> Self {
        Self::zero_t(token)
    }
    #[inline(always)]
    #[doc = "Deprecated token-taking spelling of [`Self::load_t`].\n\nUse `load_t` to keep explicit-token construction when `load` becomes tokenless in magetypes 0.10."]
    #[deprecated(note = "Use load_t(token, data); load becomes tokenless in magetypes 0.10.")]
    #[forbid(unsafe_code)]
    pub fn load(token: T, data: &[u64; 8]) -> Self {
        Self::load_t(token, data)
    }
    #[inline(always)]
    #[doc = "Deprecated token-taking spelling of [`Self::from_array_t`].\n\nUse `from_array_t` to keep explicit-token construction when `from_array` becomes tokenless in magetypes 0.10."]
    #[deprecated(
        note = "Use from_array_t(token, arr); from_array becomes tokenless in magetypes 0.10."
    )]
    #[forbid(unsafe_code)]
    pub fn from_array(token: T, arr: [u64; 8]) -> Self {
        Self::from_array_t(token, arr)
    }
    #[inline(always)]
    #[doc = "Deprecated token-taking spelling of [`Self::from_slice_t`].\n\nUse `from_slice_t` to keep explicit-token construction when `from_slice` becomes tokenless in magetypes 0.10."]
    #[deprecated(
        note = "Use from_slice_t(token, slice); from_slice becomes tokenless in magetypes 0.10."
    )]
    #[forbid(unsafe_code)]
    pub fn from_slice(token: T, slice: &[u64]) -> Self {
        Self::from_slice_t(token, slice)
    }
    #[inline(always)]
    #[doc = "Deprecated token-taking spelling of [`Self::partition_slice_t`].\n\nUse `partition_slice_t` to keep explicit-token construction when `partition_slice` becomes tokenless in magetypes 0.10."]
    #[deprecated(
        note = "Use partition_slice_t(token, data); partition_slice becomes tokenless in magetypes 0.10."
    )]
    #[forbid(unsafe_code)]
    pub fn partition_slice(token: T, data: &[u64]) -> (&[[u64; 8]], &[u64]) {
        Self::partition_slice_t(token, data)
    }
    #[inline(always)]
    #[doc = "Deprecated token-taking spelling of [`Self::partition_slice_mut_t`].\n\nUse `partition_slice_mut_t` to keep explicit-token construction when `partition_slice_mut` becomes tokenless in magetypes 0.10."]
    #[deprecated(
        note = "Use partition_slice_mut_t(token, data); partition_slice_mut becomes tokenless in magetypes 0.10."
    )]
    #[forbid(unsafe_code)]
    pub fn partition_slice_mut(token: T, data: &mut [u64]) -> (&mut [[u64; 8]], &mut [u64]) {
        Self::partition_slice_mut_t(token, data)
    }
    #[inline(always)]
    #[doc = "Deprecated token-taking spelling of [`Self::from_repr_t`].\n\nUse `from_repr_t` to keep explicit-token construction when `from_repr` becomes tokenless in magetypes 0.10."]
    #[deprecated(
        note = "Use from_repr_t(token, repr); from_repr becomes tokenless in magetypes 0.10."
    )]
    #[forbid(unsafe_code)]
    pub fn from_repr(token: T, repr: T::Repr) -> Self {
        Self::from_repr_t(token, repr)
    }
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl u64x8<archmage::X64V4Token> {
    #[inline(always)]
    #[doc = "Deprecated token-taking spelling of [`Self::from_m512i_t`].\n\nUse `from_m512i_t` to keep explicit-token construction when `from_m512i` becomes tokenless in magetypes 0.10."]
    #[deprecated(
        note = "Use from_m512i_t(token, value); from_m512i becomes tokenless in magetypes 0.10."
    )]
    #[forbid(unsafe_code)]
    pub fn from_m512i(token: archmage::X64V4Token, value: core::arch::x86_64::__m512i) -> Self {
        Self::from_m512i_t(token, value)
    }
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl u64x8<archmage::X64V4xToken> {
    #[inline(always)]
    #[doc = "Deprecated token-taking spelling of [`Self::from_m512i_t`].\n\nUse `from_m512i_t` to keep explicit-token construction when `from_m512i` becomes tokenless in magetypes 0.10."]
    #[deprecated(
        note = "Use from_m512i_t(token, value); from_m512i becomes tokenless in magetypes 0.10."
    )]
    #[forbid(unsafe_code)]
    pub fn from_m512i(token: archmage::X64V4xToken, value: core::arch::x86_64::__m512i) -> Self {
        Self::from_m512i_t(token, value)
    }
}

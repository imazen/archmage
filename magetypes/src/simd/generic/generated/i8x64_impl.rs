//! Generic `i8x64<T>` — 64-lane i8 SIMD vector parameterized by backend.
//!
//! `T` is a token type (e.g., `X64V4Token`, `ScalarToken`)
//! that determines the platform-native representation and intrinsics used.
//! The struct delegates all operations to the [`I8x64Backend`] trait.

#![allow(clippy::should_implement_trait)]

use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Index,
    IndexMut, Neg, Sub, SubAssign,
};

use crate::simd::backends::I8x64Backend;

/// 64-lane i8 SIMD vector, generic over backend `T`.
///
/// `T` is a token type that proves CPU support for the required SIMD features.
/// The inner representation is `T::Repr` (e.g., `__m512i` on AVX-512, `[i8; 64]` on scalar).
///
/// **The token is stored** (as a zero-sized field) so methods receiving
/// `self: i8x64<T>` can re-supply it to backend operations that
/// require a token value (e.g. `T::splat(token, v)`). This carries the
/// token-as-feature-proof guarantee through every method call without
/// runtime overhead — `T` is ZST, so `sizeof(i8x64<T>) == sizeof(T::Repr)`
/// and `align_of(i8x64<T>) == align_of(T::Repr)` under `#[repr(C)]`.
///
/// # Layout
///
/// `#[repr(C)]` with a ZST trailing field: `T::Repr` lives at offset 0
/// and `T` is a 0-byte tail. Bitcasts between `i8x64<T>` values of
/// different element-types are sound when the Repr types share a layout
/// (e.g. `__m128` and `__m128i` are both 16-byte aligned 128-bit values).
/// `#[repr(transparent)]` cannot be used because Rust cannot prove at
/// the struct definition site that a generic `T` is a 1-ZST.
///
/// Construction requires a token value to prove CPU support at runtime.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct i8x64<T: I8x64Backend>(pub(crate) T::Repr, pub(crate) T);
// The `unsafe impl` and the checks behind it live in `simd_storage`.
crate::simd_storage::impl_token_storage!(i8x64, I8x64Backend);

// Layout invariant: struct is `#[repr(C)]` with a trailing ZST `T`
// field, so `sizeof/alignof(i8x64<T>) == sizeof/alignof(T::Repr)`
// when `T` is a 1-ZST. Every archmage token currently satisfies this;
// if a future refactor adds a non-ZST field to a token, this const
// assert fires at compile time.
const _: () = {
    assert!(
        core::mem::size_of::<i8x64<archmage::ScalarToken>>()
            == core::mem::size_of::<
                <archmage::ScalarToken as crate::simd::backends::I8x64Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<i8x64<archmage::ScalarToken>>()
            == core::mem::align_of::<
                <archmage::ScalarToken as crate::simd::backends::I8x64Backend>::Repr,
            >()
    );
};

#[cfg(target_arch = "x86_64")]
const _: () = {
    assert!(
        core::mem::size_of::<i8x64<archmage::X64V3Token>>()
            == core::mem::size_of::<
                <archmage::X64V3Token as crate::simd::backends::I8x64Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<i8x64<archmage::X64V3Token>>()
            == core::mem::align_of::<
                <archmage::X64V3Token as crate::simd::backends::I8x64Backend>::Repr,
            >()
    );
};

// Native AVX-512 (`__m512`/`__m512d`/`__m512i`) — gated on the
// `avx512` feature, which is how archmage exposes X64V4Token's
// 512-bit backend impls.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
const _: () = {
    assert!(
        core::mem::size_of::<i8x64<archmage::X64V4Token>>()
            == core::mem::size_of::<
                <archmage::X64V4Token as crate::simd::backends::I8x64Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<i8x64<archmage::X64V4Token>>()
            == core::mem::align_of::<
                <archmage::X64V4Token as crate::simd::backends::I8x64Backend>::Repr,
            >()
    );
};

#[cfg(target_arch = "aarch64")]
const _: () = {
    assert!(
        core::mem::size_of::<i8x64<archmage::NeonToken>>()
            == core::mem::size_of::<
                <archmage::NeonToken as crate::simd::backends::I8x64Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<i8x64<archmage::NeonToken>>()
            == core::mem::align_of::<
                <archmage::NeonToken as crate::simd::backends::I8x64Backend>::Repr,
            >()
    );
};

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
const _: () = {
    assert!(
        core::mem::size_of::<i8x64<archmage::Wasm128Token>>()
            == core::mem::size_of::<
                <archmage::Wasm128Token as crate::simd::backends::I8x64Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<i8x64<archmage::Wasm128Token>>()
            == core::mem::align_of::<
                <archmage::Wasm128Token as crate::simd::backends::I8x64Backend>::Repr,
            >()
    );
};

impl<T: I8x64Backend> i8x64<T> {
    /// Number of i8 lanes.
    pub const LANES: usize = 64;

    // ====== Construction (token-gated) ======

    /// Broadcast scalar to all 64 lanes.
    #[inline(always)]
    pub fn splat_t(token: T, v: i8) -> Self {
        Self(T::splat(token, v), token)
    }

    /// All lanes zero.
    #[inline(always)]
    pub fn zero_t(token: T) -> Self {
        Self(T::zero(token), token)
    }

    /// Load from a `[i8; 64]` array.
    #[inline(always)]
    pub fn load_t(token: T, data: &[i8; 64]) -> Self {
        Self(T::load(token, data), token)
    }

    /// Create from array (zero-cost where possible).
    #[inline(always)]
    pub fn from_array_t(token: T, arr: [i8; 64]) -> Self {
        Self(T::from_array(token, arr), token)
    }

    /// Create from slice. Panics if `slice.len() < 64`.
    #[inline(always)]
    pub fn from_slice_t(token: T, slice: &[i8]) -> Self {
        let arr: [i8; 64] = slice[..64].try_into().unwrap();
        Self(T::from_array(token, arr), token)
    }

    /// Split a slice into SIMD-width chunks and a scalar remainder.
    ///
    /// Returns `(&[[i8; 64]], &[i8])` — fixed-size arrays suitable
    /// for [`load_t`](Self::load_t), plus any leftover elements.
    #[inline(always)]
    pub fn partition_slice_t(_: T, data: &[i8]) -> (&[[i8; 64]], &[i8]) {
        data.as_chunks::<64>()
    }

    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    ///
    /// Returns `(&mut [[i8; 64]], &mut [i8])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load_t`](Self::load_t), plus any leftover elements.
    #[inline(always)]
    pub fn partition_slice_mut_t(_: T, data: &mut [i8]) -> (&mut [[i8; 64]], &mut [i8]) {
        data.as_chunks_mut::<64>()
    }

    // ====== Accessors ======

    /// Store to array.
    #[inline(always)]
    pub fn store(self, out: &mut [i8; 64]) {
        T::store(self.1, self.0, out);
    }

    /// Convert to array.
    #[inline(always)]
    pub fn to_array(self) -> [i8; 64] {
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

    /// Sum all 64 lanes (wrapping).
    #[inline(always)]
    pub fn reduce_add(self) -> i8 {
        T::reduce_add(self.1, self.0)
    }

    // ====== Shifts ======

    /// Shift left by constant.
    ///
    /// `N` must be in `0..=7`; out-of-range `N` fails to compile,
    /// identically on every backend.
    #[inline(always)]
    pub fn shl_const<const N: i32>(self) -> Self {
        const { assert!(N >= 0 && N <= 7, "shift amount out of range") };
        Self(T::shl_const::<N>(self.1, self.0), self.1)
    }

    /// Arithmetic shift right by constant (sign-extending).
    ///
    /// `N` must be in `0..=7` (`N == 0` is the identity shift);
    /// out-of-range `N` fails to compile, identically on every backend.
    #[inline(always)]
    pub fn shr_arithmetic_const<const N: i32>(self) -> Self {
        const { assert!(N >= 0 && N <= 7, "shift amount out of range") };
        Self(T::shr_arithmetic_const::<N>(self.1, self.0), self.1)
    }

    /// Logical shift right by constant (zero-filling).
    ///
    /// `N` must be in `0..=7` (`N == 0` is the identity shift);
    /// out-of-range `N` fails to compile, identically on every backend.
    #[inline(always)]
    pub fn shr_logical_const<const N: i32>(self) -> Self {
        const { assert!(N >= 0 && N <= 7, "shift amount out of range") };
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

    // ====== Saturating arithmetic ======

    /// Lane-wise addition that clamps to the `i8` range instead of
    /// wrapping — `i8::saturating_add`, per lane.
    #[inline(always)]
    pub fn saturating_add(self, other: Self) -> Self {
        Self(T::saturating_add(self.1, self.0, other.0), self.1)
    }

    /// Lane-wise subtraction that clamps to the `i8` range instead of
    /// wrapping — `i8::saturating_sub`, per lane.
    #[inline(always)]
    pub fn saturating_sub(self, other: Self) -> Self {
        Self(T::saturating_sub(self.1, self.0, other.0), self.1)
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

    /// Extract the high bit of each 8-bit lane as a bitmask.
    #[inline(always)]
    pub fn bitmask(self) -> u64 {
        T::bitmask(self.1, self.0)
    }
}

// ============================================================================
// Operator implementations
// ============================================================================

impl<T: I8x64Backend> Add for i8x64<T> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(T::add(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: I8x64Backend> Sub for i8x64<T> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(T::sub(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: I8x64Backend> Neg for i8x64<T> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(T::neg(self.1, self.0), self.1)
    }
}

impl<T: I8x64Backend> BitAnd for i8x64<T> {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(T::bitand(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: I8x64Backend> BitOr for i8x64<T> {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self(T::bitor(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: I8x64Backend> BitXor for i8x64<T> {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Self(T::bitxor(self.1, self.0, rhs.0), self.1)
    }
}

// ============================================================================
// Assign operators
// ============================================================================

impl<T: I8x64Backend> AddAssign for i8x64<T> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<T: I8x64Backend> SubAssign for i8x64<T> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<T: I8x64Backend> BitAndAssign for i8x64<T> {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

impl<T: I8x64Backend> BitOrAssign for i8x64<T> {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl<T: I8x64Backend> BitXorAssign for i8x64<T> {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = *self ^ rhs;
    }
}

// ============================================================================
// Scalar broadcast operators (v + 2, etc.)
// ============================================================================

impl<T: I8x64Backend> Add<i8> for i8x64<T> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: i8) -> Self {
        Self(T::add(self.1, self.0, T::splat(self.1, rhs)), self.1)
    }
}

impl<T: I8x64Backend> Sub<i8> for i8x64<T> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: i8) -> Self {
        Self(T::sub(self.1, self.0, T::splat(self.1, rhs)), self.1)
    }
}

// ============================================================================
// Index
// ============================================================================

impl<T: I8x64Backend> Index<usize> for i8x64<T> {
    type Output = i8;
    #[inline(always)]
    fn index(&self, i: usize) -> &i8 {
        &crate::simd_storage::view::<_, [i8; 64]>(&self.0)[i]
    }
}

impl<T: I8x64Backend> IndexMut<usize> for i8x64<T> {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut i8 {
        &mut crate::simd_storage::view_mut::<_, [i8; 64]>(&mut self.0)[i]
    }
}

// ============================================================================
// Conversions
// ============================================================================

impl<T: I8x64Backend> From<i8x64<T>> for [i8; 64] {
    #[inline(always)]
    fn from(v: i8x64<T>) -> [i8; 64] {
        T::to_array(v.1, v.0)
    }
}

// ============================================================================
// Debug
// ============================================================================

impl<T: I8x64Backend> core::fmt::Debug for i8x64<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let arr = T::to_array(self.1, self.0);
        f.debug_tuple("i8x64").field(&arr).finish()
    }
}

// ============================================================================
// Widening (i8x64 -> i16x32)
// ============================================================================

impl<T: crate::simd::backends::I8x64Backend + crate::simd::backends::I16x32Backend> i8x64<T> {
    /// Sign-extend the low half of the lanes to `i16x32`.
    ///
    /// Result lane `i` is `self[i] as i16` for `i` in `0..32`.
    /// Natural lane order on every backend. Instruction count depends
    /// on the ISA, vector width, and surrounding loads.
    #[inline(always)]
    pub fn widen_low(self) -> super::i16x32<T> {
        super::i16x32::from_repr_unchecked(
            self.1,
            <T as crate::simd::backends::I8x64Backend>::widen_low_i8_to_i16(self.1, self.0),
        )
    }

    /// Sign-extend the high half of the lanes to `i16x32`.
    ///
    /// Result lane `i` is `self[i + 32] as i16`.
    #[inline(always)]
    pub fn widen_high(self) -> super::i16x32<T> {
        super::i16x32::from_repr_unchecked(
            self.1,
            <T as crate::simd::backends::I8x64Backend>::widen_high_i8_to_i16(self.1, self.0),
        )
    }
}

// ============================================================================
// Platform-specific concrete impls
// ============================================================================

impl i8x64<archmage::ScalarToken> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "scalar::i8x64"
    }
}

#[cfg(target_arch = "x86_64")]
impl i8x64<archmage::X64V3Token> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "polyfill::v3_512::i8x64"
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl i8x64<archmage::X64V4Token> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "x86::v4::i8x64"
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl i8x64<archmage::X64V4xToken> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "x86::v4x::i8x64"
    }
}

#[cfg(target_arch = "aarch64")]
impl i8x64<archmage::NeonToken> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "polyfill::neon_512::i8x64"
    }
}

#[cfg(target_arch = "wasm32")]
impl i8x64<archmage::Wasm128Token> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "polyfill::wasm128_512::i8x64"
    }
}

// ============================================================================
// Extension: popcnt (requires Modern token)
// ============================================================================

#[cfg(feature = "avx512")]
impl<T: crate::simd::backends::i8x64PopcntBackend> i8x64<T> {
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
impl i8x64<archmage::X64V4Token> {
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
impl i8x64<archmage::X64V4xToken> {
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
impl<T: I8x64Backend> i8x64<T> {
    #[inline(always)]
    #[doc = "Deprecated token-taking spelling of [`Self::splat_t`].\n\nUse `splat_t` to keep explicit-token construction when `splat` becomes tokenless in magetypes 0.10."]
    #[deprecated(note = "Use splat_t(token, v); splat becomes tokenless in magetypes 0.10.")]
    #[forbid(unsafe_code)]
    pub fn splat(token: T, v: i8) -> Self {
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
    pub fn load(token: T, data: &[i8; 64]) -> Self {
        Self::load_t(token, data)
    }
    #[inline(always)]
    #[doc = "Deprecated token-taking spelling of [`Self::from_array_t`].\n\nUse `from_array_t` to keep explicit-token construction when `from_array` becomes tokenless in magetypes 0.10."]
    #[deprecated(
        note = "Use from_array_t(token, arr); from_array becomes tokenless in magetypes 0.10."
    )]
    #[forbid(unsafe_code)]
    pub fn from_array(token: T, arr: [i8; 64]) -> Self {
        Self::from_array_t(token, arr)
    }
    #[inline(always)]
    #[doc = "Deprecated token-taking spelling of [`Self::from_slice_t`].\n\nUse `from_slice_t` to keep explicit-token construction when `from_slice` becomes tokenless in magetypes 0.10."]
    #[deprecated(
        note = "Use from_slice_t(token, slice); from_slice becomes tokenless in magetypes 0.10."
    )]
    #[forbid(unsafe_code)]
    pub fn from_slice(token: T, slice: &[i8]) -> Self {
        Self::from_slice_t(token, slice)
    }
    #[inline(always)]
    #[doc = "Deprecated token-taking spelling of [`Self::partition_slice_t`].\n\nUse `partition_slice_t` to keep explicit-token construction when `partition_slice` becomes tokenless in magetypes 0.10."]
    #[deprecated(
        note = "Use partition_slice_t(token, data); partition_slice becomes tokenless in magetypes 0.10."
    )]
    #[forbid(unsafe_code)]
    pub fn partition_slice(token: T, data: &[i8]) -> (&[[i8; 64]], &[i8]) {
        Self::partition_slice_t(token, data)
    }
    #[inline(always)]
    #[doc = "Deprecated token-taking spelling of [`Self::partition_slice_mut_t`].\n\nUse `partition_slice_mut_t` to keep explicit-token construction when `partition_slice_mut` becomes tokenless in magetypes 0.10."]
    #[deprecated(
        note = "Use partition_slice_mut_t(token, data); partition_slice_mut becomes tokenless in magetypes 0.10."
    )]
    #[forbid(unsafe_code)]
    pub fn partition_slice_mut(token: T, data: &mut [i8]) -> (&mut [[i8; 64]], &mut [i8]) {
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
impl i8x64<archmage::X64V4Token> {
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
impl i8x64<archmage::X64V4xToken> {
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

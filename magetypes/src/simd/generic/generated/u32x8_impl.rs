//! Generic `u32x8<T>` — 8-lane u32 SIMD vector parameterized by backend.
//!
//! `T` is a token type (e.g., `X64V3Token`, `NeonToken`, `ScalarToken`)
//! that determines the platform-native representation and intrinsics used.
//! The struct delegates all operations to the [`U32x8Backend`] trait.

#![allow(clippy::should_implement_trait)]

use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Index,
    IndexMut, Mul, MulAssign, Sub, SubAssign,
};

use crate::simd::backends::U32x8Backend;

/// 8-lane u32 SIMD vector, generic over backend `T`.
///
/// `T` is a token type that proves CPU support for the required SIMD features.
/// The inner representation is `T::Repr` (e.g., `__m256i` on AVX2, `[u32; 8]` on scalar).
///
/// **The token is stored** (as a zero-sized field) so methods receiving
/// `self: u32x8<T>` can re-supply it to backend operations that
/// require a token value (e.g. `T::splat(token, v)`). This carries the
/// token-as-feature-proof guarantee through every method call without
/// runtime overhead — `T` is ZST, so `sizeof(u32x8<T>) == sizeof(T::Repr)`
/// and `align_of(u32x8<T>) == align_of(T::Repr)` under `#[repr(C)]`.
///
/// # Layout
///
/// `#[repr(C)]` with a ZST trailing field: `T::Repr` lives at offset 0
/// and `T` plus the sealed policy marker are zero-sized tails. Bitcasts between `u32x8<T>` values of
/// different element-types are sound when the Repr types share a layout
/// (e.g. `__m128` and `__m128i` are both 16-byte aligned 128-bit values).
/// `#[repr(transparent)]` cannot be used because Rust cannot prove at
/// the struct definition site that a generic `T` is a 1-ZST.
///
/// Fixed-policy aliases select explicit-token or feature-context constructors.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct u32x8<
    T: U32x8Backend,
    M: crate::simd::generic::ConstructorMode = crate::simd::generic::Explicit,
>(
    pub(crate) T::Repr,
    pub(crate) T,
    pub(crate) core::marker::PhantomData<M>,
);
// SAFETY: repr(C) pair of Pod storage and a sealed 1-ZST token.
// A supplied T proves CPU support; the wrapper adds no bit invariants.
// Helpers additionally check token size/alignment at monomorphization.
unsafe impl<M: crate::simd::generic::ConstructorMode, T: U32x8Backend>
    crate::simd_storage::TokenStorage for u32x8<T, M>
{
    type Token = T;
}

// Layout invariant: struct is `#[repr(C)]` with a trailing ZST `T`
// field, so `sizeof/alignof(u32x8<T>) == sizeof/alignof(T::Repr)`
// when `T` is a 1-ZST. Every archmage token currently satisfies this;
// if a future refactor adds a non-ZST field to a token, this const
// assert fires at compile time.
const _: () = {
    assert!(
        core::mem::size_of::<u32x8<archmage::ScalarToken>>()
            == core::mem::size_of::<
                <archmage::ScalarToken as crate::simd::backends::U32x8Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<u32x8<archmage::ScalarToken>>()
            == core::mem::align_of::<
                <archmage::ScalarToken as crate::simd::backends::U32x8Backend>::Repr,
            >()
    );
};

#[cfg(target_arch = "x86_64")]
const _: () = {
    assert!(
        core::mem::size_of::<u32x8<archmage::X64V3Token>>()
            == core::mem::size_of::<
                <archmage::X64V3Token as crate::simd::backends::U32x8Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<u32x8<archmage::X64V3Token>>()
            == core::mem::align_of::<
                <archmage::X64V3Token as crate::simd::backends::U32x8Backend>::Repr,
            >()
    );
};

#[cfg(target_arch = "aarch64")]
const _: () = {
    assert!(
        core::mem::size_of::<u32x8<archmage::NeonToken>>()
            == core::mem::size_of::<
                <archmage::NeonToken as crate::simd::backends::U32x8Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<u32x8<archmage::NeonToken>>()
            == core::mem::align_of::<
                <archmage::NeonToken as crate::simd::backends::U32x8Backend>::Repr,
            >()
    );
};

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
const _: () = {
    assert!(
        core::mem::size_of::<u32x8<archmage::Wasm128Token>>()
            == core::mem::size_of::<
                <archmage::Wasm128Token as crate::simd::backends::U32x8Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<u32x8<archmage::Wasm128Token>>()
            == core::mem::align_of::<
                <archmage::Wasm128Token as crate::simd::backends::U32x8Backend>::Repr,
            >()
    );
};

impl<M: crate::simd::generic::ConstructorMode, T: U32x8Backend> u32x8<T, M> {
    #[inline(always)]
    pub(crate) fn new_repr(repr: T::Repr, token: T) -> Self {
        Self(repr, token, core::marker::PhantomData)
    }

    /// Number of u32 lanes.
    pub const LANES: usize = 8;

    // ====== Construction (token-gated) ======

    /// Broadcast scalar to all 8 lanes.
    #[inline(always)]
    pub(crate) fn splat_with_token(token: T, v: u32) -> Self {
        Self::new_repr(T::splat(token, v), token)
    }

    /// All lanes zero.
    #[inline(always)]
    pub(crate) fn zero_with_token(token: T) -> Self {
        Self::new_repr(T::zero(token), token)
    }

    /// Load from a `[u32; 8]` array.
    #[inline(always)]
    pub(crate) fn load_with_token(token: T, data: &[u32; 8]) -> Self {
        Self::new_repr(T::load(token, data), token)
    }

    /// Create from array (zero-cost where possible).
    #[inline(always)]
    pub(crate) fn from_array_with_token(token: T, arr: [u32; 8]) -> Self {
        Self::new_repr(T::from_array(token, arr), token)
    }

    /// Create from slice. Panics if `slice.len() < 8`.
    #[inline(always)]
    pub(crate) fn from_slice_with_token(token: T, slice: &[u32]) -> Self {
        let arr: [u32; 8] = slice[..8].try_into().unwrap();
        Self::new_repr(T::from_array(token, arr), token)
    }

    /// Split a slice into SIMD-width chunks and a scalar remainder.
    ///
    /// Returns `(&[[u32; 8]], &[u32])` — fixed-size arrays suitable
    /// for [`load`](Self::load), plus any leftover elements.
    #[inline(always)]
    pub(crate) fn partition_slice_with_token(_token: T, data: &[u32]) -> (&[[u32; 8]], &[u32]) {
        data.as_chunks::<8>()
    }

    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    ///
    /// Returns `(&mut [[u32; 8]], &mut [u32])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[inline(always)]
    pub(crate) fn partition_slice_mut_with_token(
        _token: T,
        data: &mut [u32],
    ) -> (&mut [[u32; 8]], &mut [u32]) {
        data.as_chunks_mut::<8>()
    }

    // ====== Accessors ======

    /// Store to array.
    #[inline(always)]
    pub fn store(self, out: &mut [u32; 8]) {
        T::store(self.1, self.0, out);
    }

    /// Convert to array.
    #[inline(always)]
    pub fn to_array(self) -> [u32; 8] {
        T::to_array(self.1, self.0)
    }

    /// Get the underlying platform representation.
    #[inline(always)]
    pub fn into_repr(self) -> T::Repr {
        self.0
    }

    /// Wrap a platform representation (token-gated).
    #[inline(always)]
    pub(crate) fn from_repr_with_token(token: T, repr: T::Repr) -> Self {
        Self::new_repr(repr, token)
    }

    /// Wrap a repr with a token. Used by cross-type/cross-width helpers
    /// in `simd::generic::*` where the token is already proven by the
    /// caller's wider input type.
    #[inline(always)]
    #[allow(dead_code)]
    pub(crate) fn from_repr_unchecked(token: T, repr: T::Repr) -> Self {
        Self::new_repr(repr, token)
    }

    // ====== Math ======

    /// Lane-wise minimum (unsigned).
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self::new_repr(T::min(self.1, self.0, other.0), self.1)
    }

    /// Lane-wise maximum (unsigned).
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self::new_repr(T::max(self.1, self.0, other.0), self.1)
    }

    /// Clamp between lo and hi.
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        Self::new_repr(T::clamp(self.1, self.0, lo.0, hi.0), self.1)
    }

    // ====== Comparisons ======

    /// Lane-wise equality (returns mask).
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> Self {
        Self::new_repr(T::simd_eq(self.1, self.0, other.0), self.1)
    }

    /// Lane-wise inequality (returns mask).
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> Self {
        Self::new_repr(T::simd_ne(self.1, self.0, other.0), self.1)
    }

    /// Lane-wise less-than, unsigned (returns mask).
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self::new_repr(T::simd_lt(self.1, self.0, other.0), self.1)
    }

    /// Lane-wise less-than-or-equal, unsigned (returns mask).
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self::new_repr(T::simd_le(self.1, self.0, other.0), self.1)
    }

    /// Lane-wise greater-than, unsigned (returns mask).
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self::new_repr(T::simd_gt(self.1, self.0, other.0), self.1)
    }

    /// Lane-wise greater-than-or-equal, unsigned (returns mask).
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> Self {
        Self::new_repr(T::simd_ge(self.1, self.0, other.0), self.1)
    }

    /// Select lanes: where mask is all-1s pick `if_true`, else `if_false`.
    #[inline(always)]
    pub fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self::new_repr(T::blend(mask.1, mask.0, if_true.0, if_false.0), mask.1)
    }

    // ====== Reductions ======

    /// Sum all 8 lanes (wrapping).
    #[inline(always)]
    pub fn reduce_add(self) -> u32 {
        T::reduce_add(self.1, self.0)
    }

    // ====== Shifts ======

    /// Shift left by constant.
    ///
    /// `N` must be in `0..=31`; out-of-range `N` fails to compile,
    /// identically on every backend.
    #[inline(always)]
    pub fn shl_const<const N: i32>(self) -> Self {
        const { assert!(N >= 0 && N <= 31, "shift amount out of range") };
        Self::new_repr(T::shl_const::<N>(self.1, self.0), self.1)
    }

    /// Logical shift right by constant (zero-filling).
    ///
    /// `N` must be in `0..=31` (`N == 0` is the identity shift);
    /// out-of-range `N` fails to compile, identically on every backend.
    #[inline(always)]
    pub fn shr_logical_const<const N: i32>(self) -> Self {
        const { assert!(N >= 0 && N <= 31, "shift amount out of range") };
        Self::new_repr(T::shr_logical_const::<N>(self.1, self.0), self.1)
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

    // ====== Uniform variable shifts ======

    /// Shift left by a runtime `count`, applied identically to every lane.
    ///
    /// Unlike [`shl_const`](Self::shl_const), `count` is a runtime value.
    /// `count >= 32` yields all-zero lanes — the same result on every
    /// backend, by contract (see `docs/CROSS-ISA-INT-PRIMITIVES.md`).
    ///
    /// The count is *uniform*: one value for the whole vector. A per-lane
    /// variable shift is deliberately not offered — at 16-bit it needs
    /// AVX-512BW+VL, and wasm128 has no per-lane variable shift at all.
    #[inline(always)]
    pub fn shl_uniform(self, count: u32) -> Self {
        Self::new_repr(T::shl_uniform(self.1, self.0, count), self.1)
    }

    /// Logical (zero-filling) shift right by a runtime `count`, applied
    /// identically to every lane.
    ///
    /// `count >= 32` yields all-zero lanes on every backend.
    #[inline(always)]
    pub fn shr_logical_uniform(self, count: u32) -> Self {
        Self::new_repr(T::shr_logical_uniform(self.1, self.0, count), self.1)
    }

    // ====== Bitwise ======

    /// Bitwise NOT.
    #[inline(always)]
    pub fn not(self) -> Self {
        Self::new_repr(T::not(self.1, self.0), self.1)
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

impl<M: crate::simd::generic::ConstructorMode, T: U32x8Backend> Add for u32x8<T, M> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::new_repr(T::add(self.1, self.0, rhs.0), self.1)
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: U32x8Backend> Sub for u32x8<T, M> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::new_repr(T::sub(self.1, self.0, rhs.0), self.1)
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: U32x8Backend> Mul for u32x8<T, M> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self::new_repr(T::mul(self.1, self.0, rhs.0), self.1)
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: U32x8Backend> BitAnd for u32x8<T, M> {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self::new_repr(T::bitand(self.1, self.0, rhs.0), self.1)
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: U32x8Backend> BitOr for u32x8<T, M> {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self::new_repr(T::bitor(self.1, self.0, rhs.0), self.1)
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: U32x8Backend> BitXor for u32x8<T, M> {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Self::new_repr(T::bitxor(self.1, self.0, rhs.0), self.1)
    }
}

// ============================================================================
// Assign operators
// ============================================================================

impl<M: crate::simd::generic::ConstructorMode, T: U32x8Backend> AddAssign for u32x8<T, M> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: U32x8Backend> SubAssign for u32x8<T, M> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: U32x8Backend> MulAssign for u32x8<T, M> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: U32x8Backend> BitAndAssign for u32x8<T, M> {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: U32x8Backend> BitOrAssign for u32x8<T, M> {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: U32x8Backend> BitXorAssign for u32x8<T, M> {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = *self ^ rhs;
    }
}

// ============================================================================
// Scalar broadcast operators (v + 2, v * 3, etc.)
// ============================================================================

impl<M: crate::simd::generic::ConstructorMode, T: U32x8Backend> Add<u32> for u32x8<T, M> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: u32) -> Self {
        Self::new_repr(T::add(self.1, self.0, T::splat(self.1, rhs)), self.1)
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: U32x8Backend> Sub<u32> for u32x8<T, M> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: u32) -> Self {
        Self::new_repr(T::sub(self.1, self.0, T::splat(self.1, rhs)), self.1)
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: U32x8Backend> Mul<u32> for u32x8<T, M> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: u32) -> Self {
        Self::new_repr(T::mul(self.1, self.0, T::splat(self.1, rhs)), self.1)
    }
}

// ============================================================================
// Index
// ============================================================================

impl<M: crate::simd::generic::ConstructorMode, T: U32x8Backend> Index<usize> for u32x8<T, M> {
    type Output = u32;
    #[inline(always)]
    fn index(&self, i: usize) -> &u32 {
        &crate::simd_storage::view::<_, [u32; 8]>(&self.0)[i]
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: U32x8Backend> IndexMut<usize> for u32x8<T, M> {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut u32 {
        &mut crate::simd_storage::view_mut::<_, [u32; 8]>(&mut self.0)[i]
    }
}

// ============================================================================
// Conversions
// ============================================================================

impl<M: crate::simd::generic::ConstructorMode, T: U32x8Backend> From<u32x8<T, M>> for [u32; 8] {
    #[inline(always)]
    fn from(v: u32x8<T, M>) -> [u32; 8] {
        T::to_array(v.1, v.0)
    }
}

// ============================================================================
// Debug
// ============================================================================

impl<M: crate::simd::generic::ConstructorMode, T: U32x8Backend> core::fmt::Debug for u32x8<T, M> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let arr = T::to_array(self.1, self.0);
        f.debug_tuple("u32x8").field(&arr).finish()
    }
}

// ============================================================================
// Cross-type conversions (i32 ↔ u32 bitcast)
// ============================================================================

impl<M: crate::simd::generic::ConstructorMode, T: crate::simd::backends::U32x8Bitcast> u32x8<T, M> {
    /// Bitcast to i32x8 (reinterpret bits, no conversion).
    #[inline(always)]
    pub fn bitcast_to_i32(self) -> super::i32x8<T, M> {
        super::i32x8::from_repr_unchecked(self.1, T::bitcast_u32_to_i32(self.1, self.0))
    }

    /// Bitcast to i32x8 by reference (zero-cost).
    #[inline(always)]
    pub fn bitcast_ref_i32x8(&self) -> &super::i32x8<T, M> {
        crate::simd_storage::vector_view(self.1, &self.0)
    }

    /// Bitcast to i32x8 by mutable reference (zero-cost).
    #[inline(always)]
    pub fn bitcast_mut_i32x8(&mut self) -> &mut super::i32x8<T, M> {
        crate::simd_storage::vector_view_mut(self.1, &mut self.0)
    }

    // ====== Backward-compatible aliases ======

    /// Alias for [`bitcast_to_i32`](Self::bitcast_to_i32).
    #[inline(always)]
    pub fn bitcast_i32x8(self) -> super::i32x8<T, M> {
        self.bitcast_to_i32()
    }
}

// ============================================================================
// Platform-specific concrete impls
// ============================================================================

impl<M: crate::simd::generic::ConstructorMode> u32x8<archmage::ScalarToken, M> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "scalar::u32x8"
    }
}

#[cfg(target_arch = "x86_64")]
impl<M: crate::simd::generic::ConstructorMode> u32x8<archmage::X64V3Token, M> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "x86::v3::u32x8"
    }

    /// Get the raw `__m256i` value.
    #[inline(always)]
    pub fn raw(self) -> core::arch::x86_64::__m256i {
        self.0
    }

    /// Create from a raw `__m256i` (token-gated, zero-cost).
    #[inline(always)]
    pub(crate) fn from_m256i_with_token(
        token: archmage::X64V3Token,
        v: core::arch::x86_64::__m256i,
    ) -> Self {
        Self::new_repr(v, token)
    }
}

#[cfg(target_arch = "aarch64")]
impl<M: crate::simd::generic::ConstructorMode> u32x8<archmage::NeonToken, M> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "polyfill::neon::u32x8"
    }
}

#[cfg(target_arch = "wasm32")]
impl<M: crate::simd::generic::ConstructorMode> u32x8<archmage::Wasm128Token, M> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "polyfill::wasm128::u32x8"
    }
}
#[cfg(target_arch = "x86_64")]
impl<M: crate::simd::generic::ConstructorMode> u32x8<archmage::X64V3Token, M> {
    /// Wrap a raw `__m256i` in a matching target-feature context.
    ///
    /// Rust requires the caller to enable the `v3` tier's features.
    /// Use an archmage `#[rite(v3)]` helper or `#[arcane]` entry point.
    #[forbid(unsafe_code)]
    #[archmage::rite(v3)]
    pub fn from_raw(value: core::arch::x86_64::__m256i) -> Self {
        Self::new_repr(value, archmage::X64V3Token::from_context())
    }
}
impl<T: U32x8Backend> From<u32x8<T, crate::simd::generic::Explicit>>
    for u32x8<T, crate::simd::generic::Context>
{
    #[inline(always)]
    fn from(value: u32x8<T, crate::simd::generic::Explicit>) -> Self {
        Self::new_repr(value.0, value.1)
    }
}
impl<T: U32x8Backend> From<u32x8<T, crate::simd::generic::Context>>
    for u32x8<T, crate::simd::generic::Explicit>
{
    #[inline(always)]
    fn from(value: u32x8<T, crate::simd::generic::Context>) -> Self {
        Self::new_repr(value.0, value.1)
    }
}

#[cfg(target_arch = "x86_64")]
#[cfg(target_arch = "x86_64")]
impl u32x8<archmage::X64V3Token, crate::simd::generic::Context> {
    /// Create from a raw `__m256i` (token-gated, zero-cost).
    #[forbid(unsafe_code)]
    #[archmage::rite(v3)]
    pub fn from_m256i(v: core::arch::x86_64::__m256i) -> Self {
        Self::from_m256i_with_token(archmage::X64V3Token::from_context(), v)
    }
}

#[cfg(target_arch = "x86_64")]
impl u32x8<archmage::X64V3Token, crate::simd::generic::Explicit> {
    /// Create from a raw `__m256i` (token-gated, zero-cost).
    #[inline(always)]
    pub fn from_m256i(token: archmage::X64V3Token, v: core::arch::x86_64::__m256i) -> Self {
        Self::from_m256i_with_token(token, v)
    }
}

#[cfg(target_arch = "aarch64")]
impl u32x8<archmage::NeonToken, crate::simd::generic::Context> {
    /// Broadcast scalar to all 8 lanes.
    #[forbid(unsafe_code)]
    #[archmage::rite(neon)]
    pub fn splat(v: u32) -> Self {
        Self::splat_with_token(archmage::NeonToken::from_context(), v)
    }
    /// All lanes zero.
    #[forbid(unsafe_code)]
    #[archmage::rite(neon)]
    pub fn zero() -> Self {
        Self::zero_with_token(archmage::NeonToken::from_context())
    }
    /// Load from a `[u32; 8]` array.
    #[forbid(unsafe_code)]
    #[archmage::rite(neon)]
    pub fn load(data: &[u32; 8]) -> Self {
        Self::load_with_token(archmage::NeonToken::from_context(), data)
    }
    /// Create from array (zero-cost where possible).
    #[forbid(unsafe_code)]
    #[archmage::rite(neon)]
    pub fn from_array(arr: [u32; 8]) -> Self {
        Self::from_array_with_token(archmage::NeonToken::from_context(), arr)
    }
    /// Create from slice. Panics if `slice.len() < 8`.
    #[forbid(unsafe_code)]
    #[archmage::rite(neon)]
    pub fn from_slice(slice: &[u32]) -> Self {
        Self::from_slice_with_token(archmage::NeonToken::from_context(), slice)
    }
    /// Split a slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&[[u32; 8]], &[u32])` — fixed-size arrays suitable
    /// for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    #[archmage::rite(neon)]
    pub fn partition_slice(data: &[u32]) -> (&[[u32; 8]], &[u32]) {
        Self::partition_slice_with_token(archmage::NeonToken::from_context(), data)
    }
    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&mut [[u32; 8]], &mut [u32])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    #[archmage::rite(neon)]
    pub fn partition_slice_mut(data: &mut [u32]) -> (&mut [[u32; 8]], &mut [u32]) {
        Self::partition_slice_mut_with_token(archmage::NeonToken::from_context(), data)
    }
    /// Wrap a platform representation (token-gated).
    #[forbid(unsafe_code)]
    #[archmage::rite(neon)]
    pub fn from_repr(
        repr: <archmage::NeonToken as crate::simd::backends::U32x8Backend>::Repr,
    ) -> Self {
        Self::from_repr_with_token(archmage::NeonToken::from_context(), repr)
    }
}

#[cfg(target_arch = "wasm32")]
impl u32x8<archmage::Wasm128Token, crate::simd::generic::Context> {
    /// Broadcast scalar to all 8 lanes.
    #[forbid(unsafe_code)]
    #[archmage::rite(wasm128)]
    pub fn splat(v: u32) -> Self {
        Self::splat_with_token(archmage::Wasm128Token::from_context(), v)
    }
    /// All lanes zero.
    #[forbid(unsafe_code)]
    #[archmage::rite(wasm128)]
    pub fn zero() -> Self {
        Self::zero_with_token(archmage::Wasm128Token::from_context())
    }
    /// Load from a `[u32; 8]` array.
    #[forbid(unsafe_code)]
    #[archmage::rite(wasm128)]
    pub fn load(data: &[u32; 8]) -> Self {
        Self::load_with_token(archmage::Wasm128Token::from_context(), data)
    }
    /// Create from array (zero-cost where possible).
    #[forbid(unsafe_code)]
    #[archmage::rite(wasm128)]
    pub fn from_array(arr: [u32; 8]) -> Self {
        Self::from_array_with_token(archmage::Wasm128Token::from_context(), arr)
    }
    /// Create from slice. Panics if `slice.len() < 8`.
    #[forbid(unsafe_code)]
    #[archmage::rite(wasm128)]
    pub fn from_slice(slice: &[u32]) -> Self {
        Self::from_slice_with_token(archmage::Wasm128Token::from_context(), slice)
    }
    /// Split a slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&[[u32; 8]], &[u32])` — fixed-size arrays suitable
    /// for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    #[archmage::rite(wasm128)]
    pub fn partition_slice(data: &[u32]) -> (&[[u32; 8]], &[u32]) {
        Self::partition_slice_with_token(archmage::Wasm128Token::from_context(), data)
    }
    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&mut [[u32; 8]], &mut [u32])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    #[archmage::rite(wasm128)]
    pub fn partition_slice_mut(data: &mut [u32]) -> (&mut [[u32; 8]], &mut [u32]) {
        Self::partition_slice_mut_with_token(archmage::Wasm128Token::from_context(), data)
    }
    /// Wrap a platform representation (token-gated).
    #[forbid(unsafe_code)]
    #[archmage::rite(wasm128)]
    pub fn from_repr(
        repr: <archmage::Wasm128Token as crate::simd::backends::U32x8Backend>::Repr,
    ) -> Self {
        Self::from_repr_with_token(archmage::Wasm128Token::from_context(), repr)
    }
}

#[cfg(target_arch = "x86_64")]
impl u32x8<archmage::X64V3Token, crate::simd::generic::Context> {
    /// Broadcast scalar to all 8 lanes.
    #[forbid(unsafe_code)]
    #[archmage::rite(v3)]
    pub fn splat(v: u32) -> Self {
        Self::splat_with_token(archmage::X64V3Token::from_context(), v)
    }
    /// All lanes zero.
    #[forbid(unsafe_code)]
    #[archmage::rite(v3)]
    pub fn zero() -> Self {
        Self::zero_with_token(archmage::X64V3Token::from_context())
    }
    /// Load from a `[u32; 8]` array.
    #[forbid(unsafe_code)]
    #[archmage::rite(v3)]
    pub fn load(data: &[u32; 8]) -> Self {
        Self::load_with_token(archmage::X64V3Token::from_context(), data)
    }
    /// Create from array (zero-cost where possible).
    #[forbid(unsafe_code)]
    #[archmage::rite(v3)]
    pub fn from_array(arr: [u32; 8]) -> Self {
        Self::from_array_with_token(archmage::X64V3Token::from_context(), arr)
    }
    /// Create from slice. Panics if `slice.len() < 8`.
    #[forbid(unsafe_code)]
    #[archmage::rite(v3)]
    pub fn from_slice(slice: &[u32]) -> Self {
        Self::from_slice_with_token(archmage::X64V3Token::from_context(), slice)
    }
    /// Split a slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&[[u32; 8]], &[u32])` — fixed-size arrays suitable
    /// for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    #[archmage::rite(v3)]
    pub fn partition_slice(data: &[u32]) -> (&[[u32; 8]], &[u32]) {
        Self::partition_slice_with_token(archmage::X64V3Token::from_context(), data)
    }
    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&mut [[u32; 8]], &mut [u32])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    #[archmage::rite(v3)]
    pub fn partition_slice_mut(data: &mut [u32]) -> (&mut [[u32; 8]], &mut [u32]) {
        Self::partition_slice_mut_with_token(archmage::X64V3Token::from_context(), data)
    }
    /// Wrap a platform representation (token-gated).
    #[forbid(unsafe_code)]
    #[archmage::rite(v3)]
    pub fn from_repr(
        repr: <archmage::X64V3Token as crate::simd::backends::U32x8Backend>::Repr,
    ) -> Self {
        Self::from_repr_with_token(archmage::X64V3Token::from_context(), repr)
    }
}

impl<T: U32x8Backend> u32x8<T, crate::simd::generic::Explicit> {
    /// Broadcast scalar to all 8 lanes.
    #[inline(always)]
    pub fn splat(token: T, v: u32) -> Self {
        Self::splat_with_token(token, v)
    }
    /// All lanes zero.
    #[inline(always)]
    pub fn zero(token: T) -> Self {
        Self::zero_with_token(token)
    }
    /// Load from a `[u32; 8]` array.
    #[inline(always)]
    pub fn load(token: T, data: &[u32; 8]) -> Self {
        Self::load_with_token(token, data)
    }
    /// Create from array (zero-cost where possible).
    #[inline(always)]
    pub fn from_array(token: T, arr: [u32; 8]) -> Self {
        Self::from_array_with_token(token, arr)
    }
    /// Create from slice. Panics if `slice.len() < 8`.
    #[inline(always)]
    pub fn from_slice(token: T, slice: &[u32]) -> Self {
        Self::from_slice_with_token(token, slice)
    }
    /// Split a slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&[[u32; 8]], &[u32])` — fixed-size arrays suitable
    /// for [`load`](Self::load), plus any leftover elements.
    #[inline(always)]
    pub fn partition_slice(_token: T, data: &[u32]) -> (&[[u32; 8]], &[u32]) {
        Self::partition_slice_with_token(_token, data)
    }
    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&mut [[u32; 8]], &mut [u32])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[inline(always)]
    pub fn partition_slice_mut(_token: T, data: &mut [u32]) -> (&mut [[u32; 8]], &mut [u32]) {
        Self::partition_slice_mut_with_token(_token, data)
    }
    /// Wrap a platform representation (token-gated).
    #[inline(always)]
    pub fn from_repr(token: T, repr: T::Repr) -> Self {
        Self::from_repr_with_token(token, repr)
    }
}

impl u32x8<archmage::ScalarToken, crate::simd::generic::Context> {
    /// Broadcast scalar to all 8 lanes.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn splat(v: u32) -> Self {
        Self::splat_with_token(archmage::ScalarToken, v)
    }
    /// All lanes zero.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn zero() -> Self {
        Self::zero_with_token(archmage::ScalarToken)
    }
    /// Load from a `[u32; 8]` array.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn load(data: &[u32; 8]) -> Self {
        Self::load_with_token(archmage::ScalarToken, data)
    }
    /// Create from array (zero-cost where possible).
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn from_array(arr: [u32; 8]) -> Self {
        Self::from_array_with_token(archmage::ScalarToken, arr)
    }
    /// Create from slice. Panics if `slice.len() < 8`.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn from_slice(slice: &[u32]) -> Self {
        Self::from_slice_with_token(archmage::ScalarToken, slice)
    }
    /// Split a slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&[[u32; 8]], &[u32])` — fixed-size arrays suitable
    /// for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn partition_slice(data: &[u32]) -> (&[[u32; 8]], &[u32]) {
        Self::partition_slice_with_token(archmage::ScalarToken, data)
    }
    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&mut [[u32; 8]], &mut [u32])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn partition_slice_mut(data: &mut [u32]) -> (&mut [[u32; 8]], &mut [u32]) {
        Self::partition_slice_mut_with_token(archmage::ScalarToken, data)
    }
    /// Wrap a platform representation (token-gated).
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn from_repr(
        repr: <archmage::ScalarToken as crate::simd::backends::U32x8Backend>::Repr,
    ) -> Self {
        Self::from_repr_with_token(archmage::ScalarToken, repr)
    }
}

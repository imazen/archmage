//! Generic `u8x16<T>` — 16-lane u8 SIMD vector parameterized by backend.
//!
//! `T` is a token type (e.g., `X64V3Token`, `NeonToken`, `ScalarToken`)
//! that determines the platform-native representation and intrinsics used.
//! The struct delegates all operations to the [`U8x16Backend`] trait.

#![allow(clippy::should_implement_trait)]

use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Index,
    IndexMut, Sub, SubAssign,
};

use crate::simd::backends::U8x16Backend;

/// 16-lane u8 SIMD vector, generic over backend `T`.
///
/// `T` is a token type that proves CPU support for the required SIMD features.
/// The inner representation is `T::Repr` (e.g., `__m128i` on x86, `uint8x16_t` on ARM).
///
/// **The token is stored** (as a zero-sized field) so methods receiving
/// `self: u8x16<T>` can re-supply it to backend operations that
/// require a token value (e.g. `T::splat(token, v)`). This carries the
/// token-as-feature-proof guarantee through every method call without
/// runtime overhead — `T` is ZST, so `sizeof(u8x16<T>) == sizeof(T::Repr)`
/// and `align_of(u8x16<T>) == align_of(T::Repr)` under `#[repr(C)]`.
///
/// # Layout
///
/// `#[repr(C)]` with a ZST trailing field: `T::Repr` lives at offset 0
/// and `T` plus the sealed policy marker are zero-sized tails. Bitcasts between `u8x16<T>` values of
/// different element-types are sound when the Repr types share a layout
/// (e.g. `__m128` and `__m128i` are both 16-byte aligned 128-bit values).
/// `#[repr(transparent)]` cannot be used because Rust cannot prove at
/// the struct definition site that a generic `T` is a 1-ZST.
///
/// Fixed-policy aliases select explicit-token or feature-context constructors.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct u8x16<
    T: U8x16Backend,
    M: crate::simd::generic::ConstructorMode = crate::simd::generic::Explicit,
>(
    pub(crate) T::Repr,
    pub(crate) T,
    pub(crate) core::marker::PhantomData<M>,
);
// SAFETY: repr(C) pair of Pod storage and a sealed 1-ZST token.
// A supplied T proves CPU support; the wrapper adds no bit invariants.
// Helpers additionally check token size/alignment at monomorphization.
unsafe impl<M: crate::simd::generic::ConstructorMode, T: U8x16Backend>
    crate::simd_storage::TokenStorage for u8x16<T, M>
{
    type Token = T;
}

// Layout invariant: struct is `#[repr(C)]` with a trailing ZST `T`
// field, so `sizeof/alignof(u8x16<T>) == sizeof/alignof(T::Repr)`
// when `T` is a 1-ZST. Every archmage token currently satisfies this;
// if a future refactor adds a non-ZST field to a token, this const
// assert fires at compile time.
const _: () = {
    assert!(
        core::mem::size_of::<u8x16<archmage::ScalarToken>>()
            == core::mem::size_of::<
                <archmage::ScalarToken as crate::simd::backends::U8x16Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<u8x16<archmage::ScalarToken>>()
            == core::mem::align_of::<
                <archmage::ScalarToken as crate::simd::backends::U8x16Backend>::Repr,
            >()
    );
};

#[cfg(target_arch = "x86_64")]
const _: () = {
    assert!(
        core::mem::size_of::<u8x16<archmage::X64V3Token>>()
            == core::mem::size_of::<
                <archmage::X64V3Token as crate::simd::backends::U8x16Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<u8x16<archmage::X64V3Token>>()
            == core::mem::align_of::<
                <archmage::X64V3Token as crate::simd::backends::U8x16Backend>::Repr,
            >()
    );
};

#[cfg(target_arch = "aarch64")]
const _: () = {
    assert!(
        core::mem::size_of::<u8x16<archmage::NeonToken>>()
            == core::mem::size_of::<
                <archmage::NeonToken as crate::simd::backends::U8x16Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<u8x16<archmage::NeonToken>>()
            == core::mem::align_of::<
                <archmage::NeonToken as crate::simd::backends::U8x16Backend>::Repr,
            >()
    );
};

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
const _: () = {
    assert!(
        core::mem::size_of::<u8x16<archmage::Wasm128Token>>()
            == core::mem::size_of::<
                <archmage::Wasm128Token as crate::simd::backends::U8x16Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<u8x16<archmage::Wasm128Token>>()
            == core::mem::align_of::<
                <archmage::Wasm128Token as crate::simd::backends::U8x16Backend>::Repr,
            >()
    );
};

impl<M: crate::simd::generic::ConstructorMode, T: U8x16Backend> u8x16<T, M> {
    #[inline(always)]
    pub(crate) fn new_repr(repr: T::Repr, token: T) -> Self {
        Self(repr, token, core::marker::PhantomData)
    }

    /// Number of u8 lanes.
    pub const LANES: usize = 16;

    // ====== Construction (token-gated) ======

    /// Broadcast scalar to all 16 lanes.
    #[inline(always)]
    pub(crate) fn splat_with_token(token: T, v: u8) -> Self {
        Self::new_repr(T::splat(token, v), token)
    }

    /// All lanes zero.
    #[inline(always)]
    pub(crate) fn zero_with_token(token: T) -> Self {
        Self::new_repr(T::zero(token), token)
    }

    /// Load from a `[u8; 16]` array.
    #[inline(always)]
    pub(crate) fn load_with_token(token: T, data: &[u8; 16]) -> Self {
        Self::new_repr(T::load(token, data), token)
    }

    /// Create from array (zero-cost where possible).
    #[inline(always)]
    pub(crate) fn from_array_with_token(token: T, arr: [u8; 16]) -> Self {
        Self::new_repr(T::from_array(token, arr), token)
    }

    /// Create from slice. Panics if `slice.len() < 16`.
    #[inline(always)]
    pub(crate) fn from_slice_with_token(token: T, slice: &[u8]) -> Self {
        let arr: [u8; 16] = slice[..16].try_into().unwrap();
        Self::new_repr(T::from_array(token, arr), token)
    }

    /// Split a slice into SIMD-width chunks and a scalar remainder.
    ///
    /// Returns `(&[[u8; 16]], &[u8])` — fixed-size arrays suitable
    /// for [`load`](Self::load), plus any leftover elements.
    #[inline(always)]
    pub(crate) fn partition_slice_with_token(_token: T, data: &[u8]) -> (&[[u8; 16]], &[u8]) {
        data.as_chunks::<16>()
    }

    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    ///
    /// Returns `(&mut [[u8; 16]], &mut [u8])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[inline(always)]
    pub(crate) fn partition_slice_mut_with_token(
        _token: T,
        data: &mut [u8],
    ) -> (&mut [[u8; 16]], &mut [u8]) {
        data.as_chunks_mut::<16>()
    }

    // ====== Accessors ======

    /// Store to array.
    #[inline(always)]
    pub fn store(self, out: &mut [u8; 16]) {
        T::store(self.1, self.0, out);
    }

    /// Convert to array.
    #[inline(always)]
    pub fn to_array(self) -> [u8; 16] {
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

    /// Sum all 16 lanes (wrapping).
    #[inline(always)]
    pub fn reduce_add(self) -> u8 {
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
        Self::new_repr(T::shl_const::<N>(self.1, self.0), self.1)
    }

    /// Logical shift right by constant (zero-filling).
    ///
    /// `N` must be in `0..=7` (`N == 0` is the identity shift);
    /// out-of-range `N` fails to compile, identically on every backend.
    #[inline(always)]
    pub fn shr_logical_const<const N: i32>(self) -> Self {
        const { assert!(N >= 0 && N <= 7, "shift amount out of range") };
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

    // ====== Saturating arithmetic ======

    /// Lane-wise addition that clamps to the `u8` range instead of
    /// wrapping — `u8::saturating_add`, per lane.
    #[inline(always)]
    pub fn saturating_add(self, other: Self) -> Self {
        Self::new_repr(T::saturating_add(self.1, self.0, other.0), self.1)
    }

    /// Lane-wise subtraction that clamps to the `u8` range instead of
    /// wrapping — `u8::saturating_sub`, per lane.
    #[inline(always)]
    pub fn saturating_sub(self, other: Self) -> Self {
        Self::new_repr(T::saturating_sub(self.1, self.0, other.0), self.1)
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

    /// Extract the high bit of each 8-bit lane as a bitmask.
    #[inline(always)]
    pub fn bitmask(self) -> u32 {
        T::bitmask(self.1, self.0)
    }
}

// ============================================================================
// Operator implementations
// ============================================================================

impl<M: crate::simd::generic::ConstructorMode, T: U8x16Backend> Add for u8x16<T, M> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::new_repr(T::add(self.1, self.0, rhs.0), self.1)
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: U8x16Backend> Sub for u8x16<T, M> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::new_repr(T::sub(self.1, self.0, rhs.0), self.1)
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: U8x16Backend> BitAnd for u8x16<T, M> {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self::new_repr(T::bitand(self.1, self.0, rhs.0), self.1)
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: U8x16Backend> BitOr for u8x16<T, M> {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self::new_repr(T::bitor(self.1, self.0, rhs.0), self.1)
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: U8x16Backend> BitXor for u8x16<T, M> {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Self::new_repr(T::bitxor(self.1, self.0, rhs.0), self.1)
    }
}

// ============================================================================
// Assign operators
// ============================================================================

impl<M: crate::simd::generic::ConstructorMode, T: U8x16Backend> AddAssign for u8x16<T, M> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: U8x16Backend> SubAssign for u8x16<T, M> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: U8x16Backend> BitAndAssign for u8x16<T, M> {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: U8x16Backend> BitOrAssign for u8x16<T, M> {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: U8x16Backend> BitXorAssign for u8x16<T, M> {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = *self ^ rhs;
    }
}

// ============================================================================
// Scalar broadcast operators (v + 2, etc.)
// ============================================================================

impl<M: crate::simd::generic::ConstructorMode, T: U8x16Backend> Add<u8> for u8x16<T, M> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: u8) -> Self {
        Self::new_repr(T::add(self.1, self.0, T::splat(self.1, rhs)), self.1)
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: U8x16Backend> Sub<u8> for u8x16<T, M> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: u8) -> Self {
        Self::new_repr(T::sub(self.1, self.0, T::splat(self.1, rhs)), self.1)
    }
}

// ============================================================================
// Index
// ============================================================================

impl<M: crate::simd::generic::ConstructorMode, T: U8x16Backend> Index<usize> for u8x16<T, M> {
    type Output = u8;
    #[inline(always)]
    fn index(&self, i: usize) -> &u8 {
        &crate::simd_storage::view::<_, [u8; 16]>(&self.0)[i]
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: U8x16Backend> IndexMut<usize> for u8x16<T, M> {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut u8 {
        &mut crate::simd_storage::view_mut::<_, [u8; 16]>(&mut self.0)[i]
    }
}

// ============================================================================
// Conversions
// ============================================================================

impl<M: crate::simd::generic::ConstructorMode, T: U8x16Backend> From<u8x16<T, M>> for [u8; 16] {
    #[inline(always)]
    fn from(v: u8x16<T, M>) -> [u8; 16] {
        T::to_array(v.1, v.0)
    }
}

// ============================================================================
// Debug
// ============================================================================

impl<M: crate::simd::generic::ConstructorMode, T: U8x16Backend> core::fmt::Debug for u8x16<T, M> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let arr = T::to_array(self.1, self.0);
        f.debug_tuple("u8x16").field(&arr).finish()
    }
}

// ============================================================================
// Cross-type conversions (u8 ↔ i8 bitcast)
// ============================================================================

impl<M: crate::simd::generic::ConstructorMode, T: crate::simd::backends::I8x16Bitcast> u8x16<T, M> {
    /// Bitcast to i8x16 (reinterpret bits, no conversion).
    #[inline(always)]
    pub fn bitcast_i8x16(self) -> super::i8x16<T, M> {
        super::i8x16::from_repr_unchecked(self.1, T::bitcast_u8_to_i8(self.1, self.0))
    }

    /// Bitcast to i8x16 by reference (zero-cost).
    #[inline(always)]
    pub fn bitcast_ref_i8x16(&self) -> &super::i8x16<T, M> {
        crate::simd_storage::vector_view(self.1, &self.0)
    }

    /// Bitcast to i8x16 by mutable reference (zero-cost).
    #[inline(always)]
    pub fn bitcast_mut_i8x16(&mut self) -> &mut super::i8x16<T, M> {
        crate::simd_storage::vector_view_mut(self.1, &mut self.0)
    }
}

// ============================================================================
// Widening (u8x16 -> u16x8)
// ============================================================================

impl<
    M: crate::simd::generic::ConstructorMode,
    T: crate::simd::backends::U8x16Backend + crate::simd::backends::U16x8Backend,
> u8x16<T, M>
{
    /// Zero-extend the low half of the lanes to `u16x8`.
    ///
    /// Result lane `i` is `self[i] as u16` for `i` in `0..8`.
    /// Natural lane order on every backend. Instruction count depends
    /// on the ISA, vector width, and surrounding loads.
    #[inline(always)]
    pub fn widen_low(self) -> super::u16x8<T, M> {
        super::u16x8::from_repr_unchecked(
            self.1,
            <T as crate::simd::backends::U8x16Backend>::widen_low_u8_to_u16(self.1, self.0),
        )
    }

    /// Zero-extend the high half of the lanes to `u16x8`.
    ///
    /// Result lane `i` is `self[i + 8] as u16`.
    #[inline(always)]
    pub fn widen_high(self) -> super::u16x8<T, M> {
        super::u16x8::from_repr_unchecked(
            self.1,
            <T as crate::simd::backends::U8x16Backend>::widen_high_u8_to_u16(self.1, self.0),
        )
    }
}

impl<T: crate::simd::backends::U8x16Backend> u8x16<T> {
    /// Exact lane-wise absolute difference, without saturation or wrapping.
    #[inline(always)]
    pub fn abs_diff(self, rhs: Self) -> super::u8x16<T> {
        super::u8x16::from_repr_unchecked(
            self.1,
            <T as crate::simd::backends::U8x16Backend>::abs_diff(self.1, self.0, rhs.0),
        )
    }
}
impl<T: crate::simd::backends::U8x16Backend> u8x16<T> {
    /// Sum all lanes exactly into u32 (unlike wrapping reduce_add).
    #[inline(always)]
    pub fn reduce_add_u32(self) -> u32 {
        <T as crate::simd::backends::U8x16Backend>::reduce_add_u32(self.1, self.0)
    }

    /// Exact sum of absolute byte differences (SAD).
    /// Terminal reduction of one vector pair; x86 can use psadbw.
    /// For long loops, accumulating vector partial sums and reducing once
    /// can be faster than returning a scalar sum on every iteration.
    #[inline(always)]
    pub fn sum_abs_diff(self, rhs: Self) -> u32 {
        <T as crate::simd::backends::U8x16Backend>::sum_abs_diff(self.1, self.0, rhs.0)
    }
}
impl<T: crate::simd::backends::U8x16Backend> u8x16<T> {
    /// Sum adjacent pairs into unsigned lanes twice as wide.
    ///
    /// Output lane `k` is `self[2*k] + self[2*k+1]`, with both
    /// inputs widened before addition. The result is exact for the
    /// full input range; no lane wraps or saturates. Pair ordering
    /// is unchanged across native and polyfilled widths.
    /// Subsequent accumulation uses the destination's normal wrapping addition.
    #[inline(always)]
    pub fn pairwise_widen_add(self) -> super::u16x8<T>
    where
        T: crate::simd::backends::U16x8Backend,
    {
        super::u16x8::from_repr_unchecked(
            self.1,
            <T as crate::simd::backends::U8x16Backend>::pairwise_widen_add(self.1, self.0),
        )
    }
}
// ============================================================================
// Platform-specific concrete impls
// ============================================================================

impl<M: crate::simd::generic::ConstructorMode> u8x16<archmage::ScalarToken, M> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "scalar::u8x16"
    }
}

#[cfg(target_arch = "x86_64")]
impl<M: crate::simd::generic::ConstructorMode> u8x16<archmage::X64V3Token, M> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "x86::v3::u8x16"
    }

    /// Get the raw `__m128i` value.
    #[inline(always)]
    pub fn raw(self) -> core::arch::x86_64::__m128i {
        self.0
    }

    /// Create from a raw `__m128i` (token-gated, zero-cost).
    #[inline(always)]
    pub(crate) fn from_m128i_with_token(
        token: archmage::X64V3Token,
        v: core::arch::x86_64::__m128i,
    ) -> Self {
        Self::new_repr(v, token)
    }
}

#[cfg(target_arch = "aarch64")]
impl<M: crate::simd::generic::ConstructorMode> u8x16<archmage::NeonToken, M> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "arm::neon::u8x16"
    }
}

#[cfg(target_arch = "wasm32")]
impl<M: crate::simd::generic::ConstructorMode> u8x16<archmage::Wasm128Token, M> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "wasm::wasm128::u8x16"
    }
}
#[cfg(target_arch = "x86_64")]
impl<M: crate::simd::generic::ConstructorMode> u8x16<archmage::X64V3Token, M> {
    /// Wrap a raw `__m128i` in a matching target-feature context.
    ///
    /// Rust requires the caller to enable the `v3` tier's features.
    /// Use an archmage `#[rite(v3)]` helper or `#[arcane]` entry point.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    pub fn from_raw(value: core::arch::x86_64::__m128i) -> Self {
        Self::new_repr(value, archmage::X64V3Token::from_context())
    }
}
#[cfg(target_arch = "aarch64")]
impl<M: crate::simd::generic::ConstructorMode> u8x16<archmage::NeonToken, M> {
    /// Get the raw `uint8x16_t` value.
    #[inline(always)]
    pub fn raw(self) -> core::arch::aarch64::uint8x16_t {
        self.0
    }

    /// Wrap a raw `uint8x16_t` using an existing CPU capability token.
    #[inline(always)]
    pub(crate) fn from_uint8x16_t_with_token(
        token: archmage::NeonToken,
        value: core::arch::aarch64::uint8x16_t,
    ) -> Self {
        Self::new_repr(value, token)
    }

    /// Wrap a raw `uint8x16_t` in a matching target-feature context.
    ///
    /// Rust requires the caller to enable the `neon` tier's features.
    /// Use an archmage `#[rite(neon)]` helper or `#[arcane]` entry point.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(enable = "neon")]
    #[inline]
    pub fn from_raw(value: core::arch::aarch64::uint8x16_t) -> Self {
        Self::new_repr(value, archmage::NeonToken::from_context())
    }
}
#[cfg(target_arch = "wasm32")]
impl<M: crate::simd::generic::ConstructorMode> u8x16<archmage::Wasm128Token, M> {
    /// Get the raw `v128` value.
    #[inline(always)]
    pub fn raw(self) -> core::arch::wasm32::v128 {
        self.0
    }

    /// Wrap a raw `v128` using an existing CPU capability token.
    #[inline(always)]
    pub(crate) fn from_v128_with_token(
        token: archmage::Wasm128Token,
        value: core::arch::wasm32::v128,
    ) -> Self {
        Self::new_repr(value, token)
    }

    /// Wrap a raw `v128` in a matching target-feature context.
    ///
    /// Rust requires the caller to enable the `wasm128` tier's features.
    /// Use an archmage `#[rite(wasm128)]` helper or `#[arcane]` entry point.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(enable = "simd128")]
    #[inline]
    pub fn from_raw(value: core::arch::wasm32::v128) -> Self {
        Self::new_repr(value, archmage::Wasm128Token::from_context())
    }
}
impl<T: U8x16Backend> From<u8x16<T, crate::simd::generic::Explicit>>
    for u8x16<T, crate::simd::generic::Context>
{
    #[inline(always)]
    fn from(value: u8x16<T, crate::simd::generic::Explicit>) -> Self {
        Self::new_repr(value.0, value.1)
    }
}
impl<T: U8x16Backend> From<u8x16<T, crate::simd::generic::Context>>
    for u8x16<T, crate::simd::generic::Explicit>
{
    #[inline(always)]
    fn from(value: u8x16<T, crate::simd::generic::Context>) -> Self {
        Self::new_repr(value.0, value.1)
    }
}

#[cfg(target_arch = "aarch64")]
#[cfg(target_arch = "aarch64")]
impl u8x16<archmage::NeonToken, crate::simd::generic::Context> {
    /// Wrap a raw `uint8x16_t` using an existing CPU capability token.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(enable = "neon")]
    #[inline]
    pub fn from_uint8x16_t(value: core::arch::aarch64::uint8x16_t) -> Self {
        let token = archmage::NeonToken::from_context();
        Self::new_repr(value, token)
    }
}

#[cfg(target_arch = "aarch64")]
impl u8x16<archmage::NeonToken, crate::simd::generic::Explicit> {
    /// Wrap a raw `uint8x16_t` using an existing CPU capability token.
    #[inline(always)]
    pub fn from_uint8x16_t(
        token: archmage::NeonToken,
        value: core::arch::aarch64::uint8x16_t,
    ) -> Self {
        Self::from_uint8x16_t_with_token(token, value)
    }
}

#[cfg(target_arch = "wasm32")]
#[cfg(target_arch = "wasm32")]
impl u8x16<archmage::Wasm128Token, crate::simd::generic::Context> {
    /// Wrap a raw `v128` using an existing CPU capability token.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(enable = "simd128")]
    #[inline]
    pub fn from_v128(value: core::arch::wasm32::v128) -> Self {
        let token = archmage::Wasm128Token::from_context();
        Self::new_repr(value, token)
    }
}

#[cfg(target_arch = "wasm32")]
impl u8x16<archmage::Wasm128Token, crate::simd::generic::Explicit> {
    /// Wrap a raw `v128` using an existing CPU capability token.
    #[inline(always)]
    pub fn from_v128(token: archmage::Wasm128Token, value: core::arch::wasm32::v128) -> Self {
        Self::from_v128_with_token(token, value)
    }
}

#[cfg(target_arch = "x86_64")]
#[cfg(target_arch = "x86_64")]
impl u8x16<archmage::X64V3Token, crate::simd::generic::Context> {
    /// Create from a raw `__m128i` (token-gated, zero-cost).
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    pub fn from_m128i(v: core::arch::x86_64::__m128i) -> Self {
        let token = archmage::X64V3Token::from_context();
        Self::new_repr(v, token)
    }
}

#[cfg(target_arch = "x86_64")]
impl u8x16<archmage::X64V3Token, crate::simd::generic::Explicit> {
    /// Create from a raw `__m128i` (token-gated, zero-cost).
    #[inline(always)]
    pub fn from_m128i(token: archmage::X64V3Token, v: core::arch::x86_64::__m128i) -> Self {
        Self::from_m128i_with_token(token, v)
    }
}

#[cfg(target_arch = "aarch64")]
impl u8x16<archmage::NeonToken, crate::simd::generic::Context> {
    /// Broadcast scalar to all 16 lanes.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(enable = "neon")]
    #[inline]
    pub fn splat(v: u8) -> Self {
        let token = archmage::NeonToken::from_context();
        Self::new_repr(
            <archmage::NeonToken as U8x16Backend>::splat(token, v),
            token,
        )
    }
    /// All lanes zero.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(enable = "neon")]
    #[inline]
    pub fn zero() -> Self {
        let token = archmage::NeonToken::from_context();
        Self::new_repr(<archmage::NeonToken as U8x16Backend>::zero(token), token)
    }
    /// Load from a `[u8; 16]` array.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(enable = "neon")]
    #[inline]
    pub fn load(data: &[u8; 16]) -> Self {
        let token = archmage::NeonToken::from_context();
        Self::new_repr(
            <archmage::NeonToken as U8x16Backend>::load(token, data),
            token,
        )
    }
    /// Create from array (zero-cost where possible).
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(enable = "neon")]
    #[inline]
    pub fn from_array(arr: [u8; 16]) -> Self {
        let token = archmage::NeonToken::from_context();
        Self::new_repr(
            <archmage::NeonToken as U8x16Backend>::from_array(token, arr),
            token,
        )
    }
    /// Create from slice. Panics if `slice.len() < 16`.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(enable = "neon")]
    #[inline]
    pub fn from_slice(slice: &[u8]) -> Self {
        Self::from_slice_with_token(archmage::NeonToken::from_context(), slice)
    }
    /// Split a slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&[[u8; 16]], &[u8])` — fixed-size arrays suitable
    /// for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(enable = "neon")]
    #[inline]
    pub fn partition_slice(data: &[u8]) -> (&[[u8; 16]], &[u8]) {
        Self::partition_slice_with_token(archmage::NeonToken::from_context(), data)
    }
    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&mut [[u8; 16]], &mut [u8])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(enable = "neon")]
    #[inline]
    pub fn partition_slice_mut(data: &mut [u8]) -> (&mut [[u8; 16]], &mut [u8]) {
        Self::partition_slice_mut_with_token(archmage::NeonToken::from_context(), data)
    }
    /// Wrap a platform representation (requires matching target features).
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(enable = "neon")]
    #[inline]
    pub fn from_repr(
        repr: <archmage::NeonToken as crate::simd::backends::U8x16Backend>::Repr,
    ) -> Self {
        let token = archmage::NeonToken::from_context();
        Self::new_repr(repr, token)
    }
}

#[cfg(target_arch = "wasm32")]
impl u8x16<archmage::Wasm128Token, crate::simd::generic::Context> {
    /// Broadcast scalar to all 16 lanes.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(enable = "simd128")]
    #[inline]
    pub fn splat(v: u8) -> Self {
        let token = archmage::Wasm128Token::from_context();
        Self::new_repr(
            <archmage::Wasm128Token as U8x16Backend>::splat(token, v),
            token,
        )
    }
    /// All lanes zero.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(enable = "simd128")]
    #[inline]
    pub fn zero() -> Self {
        let token = archmage::Wasm128Token::from_context();
        Self::new_repr(<archmage::Wasm128Token as U8x16Backend>::zero(token), token)
    }
    /// Load from a `[u8; 16]` array.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(enable = "simd128")]
    #[inline]
    pub fn load(data: &[u8; 16]) -> Self {
        let token = archmage::Wasm128Token::from_context();
        Self::new_repr(
            <archmage::Wasm128Token as U8x16Backend>::load(token, data),
            token,
        )
    }
    /// Create from array (zero-cost where possible).
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(enable = "simd128")]
    #[inline]
    pub fn from_array(arr: [u8; 16]) -> Self {
        let token = archmage::Wasm128Token::from_context();
        Self::new_repr(
            <archmage::Wasm128Token as U8x16Backend>::from_array(token, arr),
            token,
        )
    }
    /// Create from slice. Panics if `slice.len() < 16`.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(enable = "simd128")]
    #[inline]
    pub fn from_slice(slice: &[u8]) -> Self {
        Self::from_slice_with_token(archmage::Wasm128Token::from_context(), slice)
    }
    /// Split a slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&[[u8; 16]], &[u8])` — fixed-size arrays suitable
    /// for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(enable = "simd128")]
    #[inline]
    pub fn partition_slice(data: &[u8]) -> (&[[u8; 16]], &[u8]) {
        Self::partition_slice_with_token(archmage::Wasm128Token::from_context(), data)
    }
    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&mut [[u8; 16]], &mut [u8])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(enable = "simd128")]
    #[inline]
    pub fn partition_slice_mut(data: &mut [u8]) -> (&mut [[u8; 16]], &mut [u8]) {
        Self::partition_slice_mut_with_token(archmage::Wasm128Token::from_context(), data)
    }
    /// Wrap a platform representation (requires matching target features).
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(enable = "simd128")]
    #[inline]
    pub fn from_repr(
        repr: <archmage::Wasm128Token as crate::simd::backends::U8x16Backend>::Repr,
    ) -> Self {
        let token = archmage::Wasm128Token::from_context();
        Self::new_repr(repr, token)
    }
}

#[cfg(target_arch = "x86_64")]
impl u8x16<archmage::X64V3Token, crate::simd::generic::Context> {
    /// Broadcast scalar to all 16 lanes.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    pub fn splat(v: u8) -> Self {
        let token = archmage::X64V3Token::from_context();
        Self::new_repr(
            <archmage::X64V3Token as U8x16Backend>::splat(token, v),
            token,
        )
    }
    /// All lanes zero.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    pub fn zero() -> Self {
        let token = archmage::X64V3Token::from_context();
        Self::new_repr(<archmage::X64V3Token as U8x16Backend>::zero(token), token)
    }
    /// Load from a `[u8; 16]` array.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    pub fn load(data: &[u8; 16]) -> Self {
        let token = archmage::X64V3Token::from_context();
        Self::new_repr(
            <archmage::X64V3Token as U8x16Backend>::load(token, data),
            token,
        )
    }
    /// Create from array (zero-cost where possible).
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    pub fn from_array(arr: [u8; 16]) -> Self {
        let token = archmage::X64V3Token::from_context();
        Self::new_repr(
            <archmage::X64V3Token as U8x16Backend>::from_array(token, arr),
            token,
        )
    }
    /// Create from slice. Panics if `slice.len() < 16`.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    pub fn from_slice(slice: &[u8]) -> Self {
        Self::from_slice_with_token(archmage::X64V3Token::from_context(), slice)
    }
    /// Split a slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&[[u8; 16]], &[u8])` — fixed-size arrays suitable
    /// for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    pub fn partition_slice(data: &[u8]) -> (&[[u8; 16]], &[u8]) {
        Self::partition_slice_with_token(archmage::X64V3Token::from_context(), data)
    }
    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&mut [[u8; 16]], &mut [u8])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    pub fn partition_slice_mut(data: &mut [u8]) -> (&mut [[u8; 16]], &mut [u8]) {
        Self::partition_slice_mut_with_token(archmage::X64V3Token::from_context(), data)
    }
    /// Wrap a platform representation (requires matching target features).
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    pub fn from_repr(
        repr: <archmage::X64V3Token as crate::simd::backends::U8x16Backend>::Repr,
    ) -> Self {
        let token = archmage::X64V3Token::from_context();
        Self::new_repr(repr, token)
    }
}

impl<T: U8x16Backend> u8x16<T, crate::simd::generic::Explicit> {
    /// Broadcast scalar to all 16 lanes.
    #[inline(always)]
    pub fn splat(token: T, v: u8) -> Self {
        Self::splat_with_token(token, v)
    }
    /// All lanes zero.
    #[inline(always)]
    pub fn zero(token: T) -> Self {
        Self::zero_with_token(token)
    }
    /// Load from a `[u8; 16]` array.
    #[inline(always)]
    pub fn load(token: T, data: &[u8; 16]) -> Self {
        Self::load_with_token(token, data)
    }
    /// Create from array (zero-cost where possible).
    #[inline(always)]
    pub fn from_array(token: T, arr: [u8; 16]) -> Self {
        Self::from_array_with_token(token, arr)
    }
    /// Create from slice. Panics if `slice.len() < 16`.
    #[inline(always)]
    pub fn from_slice(token: T, slice: &[u8]) -> Self {
        Self::from_slice_with_token(token, slice)
    }
    /// Split a slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&[[u8; 16]], &[u8])` — fixed-size arrays suitable
    /// for [`load`](Self::load), plus any leftover elements.
    #[inline(always)]
    pub fn partition_slice(_token: T, data: &[u8]) -> (&[[u8; 16]], &[u8]) {
        Self::partition_slice_with_token(_token, data)
    }
    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&mut [[u8; 16]], &mut [u8])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[inline(always)]
    pub fn partition_slice_mut(_token: T, data: &mut [u8]) -> (&mut [[u8; 16]], &mut [u8]) {
        Self::partition_slice_mut_with_token(_token, data)
    }
    /// Wrap a platform representation (token-gated).
    #[inline(always)]
    pub fn from_repr(token: T, repr: T::Repr) -> Self {
        Self::from_repr_with_token(token, repr)
    }
}

impl u8x16<archmage::ScalarToken, crate::simd::generic::Context> {
    /// Broadcast scalar to all 16 lanes.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn splat(v: u8) -> Self {
        let token = archmage::ScalarToken;
        Self::new_repr(
            <archmage::ScalarToken as U8x16Backend>::splat(token, v),
            token,
        )
    }
    /// All lanes zero.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn zero() -> Self {
        let token = archmage::ScalarToken;
        Self::new_repr(<archmage::ScalarToken as U8x16Backend>::zero(token), token)
    }
    /// Load from a `[u8; 16]` array.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn load(data: &[u8; 16]) -> Self {
        let token = archmage::ScalarToken;
        Self::new_repr(
            <archmage::ScalarToken as U8x16Backend>::load(token, data),
            token,
        )
    }
    /// Create from array (zero-cost where possible).
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn from_array(arr: [u8; 16]) -> Self {
        let token = archmage::ScalarToken;
        Self::new_repr(
            <archmage::ScalarToken as U8x16Backend>::from_array(token, arr),
            token,
        )
    }
    /// Create from slice. Panics if `slice.len() < 16`.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn from_slice(slice: &[u8]) -> Self {
        Self::from_slice_with_token(archmage::ScalarToken, slice)
    }
    /// Split a slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&[[u8; 16]], &[u8])` — fixed-size arrays suitable
    /// for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn partition_slice(data: &[u8]) -> (&[[u8; 16]], &[u8]) {
        Self::partition_slice_with_token(archmage::ScalarToken, data)
    }
    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&mut [[u8; 16]], &mut [u8])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn partition_slice_mut(data: &mut [u8]) -> (&mut [[u8; 16]], &mut [u8]) {
        Self::partition_slice_mut_with_token(archmage::ScalarToken, data)
    }
    /// Wrap a platform representation (requires matching target features).
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn from_repr(
        repr: <archmage::ScalarToken as crate::simd::backends::U8x16Backend>::Repr,
    ) -> Self {
        let token = archmage::ScalarToken;
        Self::new_repr(repr, token)
    }
}

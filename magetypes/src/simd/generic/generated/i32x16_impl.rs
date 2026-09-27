//! Generic `i32x16<T>` — 16-lane i32 SIMD vector parameterized by backend.
//!
//! `T` is a token type (e.g., `X64V4Token`, `ScalarToken`)
//! that determines the platform-native representation and intrinsics used.
//! The struct delegates all operations to the [`I32x16Backend`] trait.

#![allow(clippy::should_implement_trait)]

use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Index,
    IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use crate::simd::backends::I32x16Backend;

/// 16-lane i32 SIMD vector, generic over backend `T`.
///
/// `T` is a token type that proves CPU support for the required SIMD features.
/// The inner representation is `T::Repr` (e.g., `__m512i` on AVX-512, `[i32; 16]` on scalar).
///
/// **The token is stored** (as a zero-sized field) so methods receiving
/// `self: i32x16<T>` can re-supply it to backend operations that
/// require a token value (e.g. `T::splat(token, v)`). This carries the
/// token-as-feature-proof guarantee through every method call without
/// runtime overhead — `T` is ZST, so `sizeof(i32x16<T>) == sizeof(T::Repr)`
/// and `align_of(i32x16<T>) == align_of(T::Repr)` under `#[repr(C)]`.
///
/// # Layout
///
/// `#[repr(C)]` with a ZST trailing field: `T::Repr` lives at offset 0
/// and `T` plus the sealed policy marker are zero-sized tails. Bitcasts between `i32x16<T>` values of
/// different element-types are sound when the Repr types share a layout
/// (e.g. `__m128` and `__m128i` are both 16-byte aligned 128-bit values).
/// `#[repr(transparent)]` cannot be used because Rust cannot prove at
/// the struct definition site that a generic `T` is a 1-ZST.
///
/// Fixed-policy aliases select explicit-token or feature-context constructors.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct i32x16<
    T: I32x16Backend,
    M: crate::simd::generic::ConstructorMode = crate::simd::generic::Explicit,
>(
    pub(crate) T::Repr,
    pub(crate) T,
    pub(crate) core::marker::PhantomData<M>,
);
// SAFETY: repr(C) pair of Pod storage and a sealed 1-ZST token.
// A supplied T proves CPU support; the wrapper adds no bit invariants.
// Helpers additionally check token size/alignment at monomorphization.
unsafe impl<M: crate::simd::generic::ConstructorMode, T: I32x16Backend>
    crate::simd_storage::TokenStorage for i32x16<T, M>
{
    type Token = T;
}

// Layout invariant: struct is `#[repr(C)]` with a trailing ZST `T`
// field, so `sizeof/alignof(i32x16<T>) == sizeof/alignof(T::Repr)`
// when `T` is a 1-ZST. Every archmage token currently satisfies this;
// if a future refactor adds a non-ZST field to a token, this const
// assert fires at compile time.
const _: () = {
    assert!(
        core::mem::size_of::<i32x16<archmage::ScalarToken>>()
            == core::mem::size_of::<
                <archmage::ScalarToken as crate::simd::backends::I32x16Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<i32x16<archmage::ScalarToken>>()
            == core::mem::align_of::<
                <archmage::ScalarToken as crate::simd::backends::I32x16Backend>::Repr,
            >()
    );
};

#[cfg(target_arch = "x86_64")]
const _: () = {
    assert!(
        core::mem::size_of::<i32x16<archmage::X64V3Token>>()
            == core::mem::size_of::<
                <archmage::X64V3Token as crate::simd::backends::I32x16Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<i32x16<archmage::X64V3Token>>()
            == core::mem::align_of::<
                <archmage::X64V3Token as crate::simd::backends::I32x16Backend>::Repr,
            >()
    );
};

// Native AVX-512 (`__m512`/`__m512d`/`__m512i`) — gated on the
// `avx512` feature, which is how archmage exposes X64V4Token's
// 512-bit backend impls.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
const _: () = {
    assert!(
        core::mem::size_of::<i32x16<archmage::X64V4Token>>()
            == core::mem::size_of::<
                <archmage::X64V4Token as crate::simd::backends::I32x16Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<i32x16<archmage::X64V4Token>>()
            == core::mem::align_of::<
                <archmage::X64V4Token as crate::simd::backends::I32x16Backend>::Repr,
            >()
    );
};

#[cfg(target_arch = "aarch64")]
const _: () = {
    assert!(
        core::mem::size_of::<i32x16<archmage::NeonToken>>()
            == core::mem::size_of::<
                <archmage::NeonToken as crate::simd::backends::I32x16Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<i32x16<archmage::NeonToken>>()
            == core::mem::align_of::<
                <archmage::NeonToken as crate::simd::backends::I32x16Backend>::Repr,
            >()
    );
};

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
const _: () = {
    assert!(
        core::mem::size_of::<i32x16<archmage::Wasm128Token>>()
            == core::mem::size_of::<
                <archmage::Wasm128Token as crate::simd::backends::I32x16Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<i32x16<archmage::Wasm128Token>>()
            == core::mem::align_of::<
                <archmage::Wasm128Token as crate::simd::backends::I32x16Backend>::Repr,
            >()
    );
};

impl<M: crate::simd::generic::ConstructorMode, T: I32x16Backend> i32x16<T, M> {
    #[inline(always)]
    pub(crate) fn new_repr(repr: T::Repr, token: T) -> Self {
        Self(repr, token, core::marker::PhantomData)
    }

    /// Number of i32 lanes.
    pub const LANES: usize = 16;

    // ====== Construction (token-gated) ======

    /// Broadcast scalar to all 16 lanes.
    #[inline(always)]
    pub(crate) fn splat_with_token(token: T, v: i32) -> Self {
        Self::new_repr(T::splat(token, v), token)
    }

    /// All lanes zero.
    #[inline(always)]
    pub(crate) fn zero_with_token(token: T) -> Self {
        Self::new_repr(T::zero(token), token)
    }

    /// Load from a `[i32; 16]` array.
    #[inline(always)]
    pub(crate) fn load_with_token(token: T, data: &[i32; 16]) -> Self {
        Self::new_repr(T::load(token, data), token)
    }

    /// Create from array (zero-cost where possible).
    #[inline(always)]
    pub(crate) fn from_array_with_token(token: T, arr: [i32; 16]) -> Self {
        Self::new_repr(T::from_array(token, arr), token)
    }

    /// Create from slice. Panics if `slice.len() < 16`.
    #[inline(always)]
    pub(crate) fn from_slice_with_token(token: T, slice: &[i32]) -> Self {
        let arr: [i32; 16] = slice[..16].try_into().unwrap();
        Self::new_repr(T::from_array(token, arr), token)
    }

    /// Split a slice into SIMD-width chunks and a scalar remainder.
    ///
    /// Returns `(&[[i32; 16]], &[i32])` — fixed-size arrays suitable
    /// for [`load`](Self::load), plus any leftover elements.
    #[inline(always)]
    pub(crate) fn partition_slice_with_token(_token: T, data: &[i32]) -> (&[[i32; 16]], &[i32]) {
        data.as_chunks::<16>()
    }

    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    ///
    /// Returns `(&mut [[i32; 16]], &mut [i32])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[inline(always)]
    pub(crate) fn partition_slice_mut_with_token(
        _token: T,
        data: &mut [i32],
    ) -> (&mut [[i32; 16]], &mut [i32]) {
        data.as_chunks_mut::<16>()
    }

    // ====== Accessors ======

    /// Store to array.
    #[inline(always)]
    pub fn store(self, out: &mut [i32; 16]) {
        T::store(self.1, self.0, out);
    }

    /// Convert to array.
    #[inline(always)]
    pub fn to_array(self) -> [i32; 16] {
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

    /// Lane-wise minimum.
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self::new_repr(T::min(self.1, self.0, other.0), self.1)
    }

    /// Lane-wise maximum.
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self::new_repr(T::max(self.1, self.0, other.0), self.1)
    }

    /// Lane-wise absolute value.
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self::new_repr(T::abs(self.1, self.0), self.1)
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

    /// Lane-wise less-than (returns mask).
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self::new_repr(T::simd_lt(self.1, self.0, other.0), self.1)
    }

    /// Lane-wise less-than-or-equal (returns mask).
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self::new_repr(T::simd_le(self.1, self.0, other.0), self.1)
    }

    /// Lane-wise greater-than (returns mask).
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self::new_repr(T::simd_gt(self.1, self.0, other.0), self.1)
    }

    /// Lane-wise greater-than-or-equal (returns mask).
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
    pub fn reduce_add(self) -> i32 {
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

    /// Arithmetic shift right by constant (sign-extending).
    ///
    /// `N` must be in `0..=31` (`N == 0` is the identity shift);
    /// out-of-range `N` fails to compile, identically on every backend.
    #[inline(always)]
    pub fn shr_arithmetic_const<const N: i32>(self) -> Self {
        const { assert!(N >= 0 && N <= 31, "shift amount out of range") };
        Self::new_repr(T::shr_arithmetic_const::<N>(self.1, self.0), self.1)
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

    /// Arithmetic (sign-filling) shift right by a runtime `count`,
    /// applied identically to every lane.
    ///
    /// `count >= 32` yields a sign fill (every lane becomes `0` or
    /// `-1`), equivalent to shifting by 31, on every backend.
    #[inline(always)]
    pub fn shr_arithmetic_uniform(self, count: u32) -> Self {
        Self::new_repr(T::shr_arithmetic_uniform(self.1, self.0, count), self.1)
    }

    // ====== Bitwise ======

    /// Bitwise NOT.
    #[inline(always)]
    pub fn not(self) -> Self {
        Self::new_repr(T::not(self.1, self.0), self.1)
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
    pub fn bitmask(self) -> u64 {
        T::bitmask(self.1, self.0)
    }
}

// ============================================================================
// Operator implementations
// ============================================================================

impl<M: crate::simd::generic::ConstructorMode, T: I32x16Backend> Add for i32x16<T, M> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::new_repr(T::add(self.1, self.0, rhs.0), self.1)
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: I32x16Backend> Sub for i32x16<T, M> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::new_repr(T::sub(self.1, self.0, rhs.0), self.1)
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: I32x16Backend> Mul for i32x16<T, M> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self::new_repr(T::mul(self.1, self.0, rhs.0), self.1)
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: I32x16Backend> Neg for i32x16<T, M> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self::new_repr(T::neg(self.1, self.0), self.1)
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: I32x16Backend> BitAnd for i32x16<T, M> {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self::new_repr(T::bitand(self.1, self.0, rhs.0), self.1)
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: I32x16Backend> BitOr for i32x16<T, M> {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self::new_repr(T::bitor(self.1, self.0, rhs.0), self.1)
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: I32x16Backend> BitXor for i32x16<T, M> {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Self::new_repr(T::bitxor(self.1, self.0, rhs.0), self.1)
    }
}

// ============================================================================
// Assign operators
// ============================================================================

impl<M: crate::simd::generic::ConstructorMode, T: I32x16Backend> AddAssign for i32x16<T, M> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: I32x16Backend> SubAssign for i32x16<T, M> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: I32x16Backend> MulAssign for i32x16<T, M> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: I32x16Backend> BitAndAssign for i32x16<T, M> {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: I32x16Backend> BitOrAssign for i32x16<T, M> {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: I32x16Backend> BitXorAssign for i32x16<T, M> {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = *self ^ rhs;
    }
}

// ============================================================================
// Scalar broadcast operators (v + 2, v * 3, etc.)
// ============================================================================

impl<M: crate::simd::generic::ConstructorMode, T: I32x16Backend> Add<i32> for i32x16<T, M> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: i32) -> Self {
        Self::new_repr(T::add(self.1, self.0, T::splat(self.1, rhs)), self.1)
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: I32x16Backend> Sub<i32> for i32x16<T, M> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: i32) -> Self {
        Self::new_repr(T::sub(self.1, self.0, T::splat(self.1, rhs)), self.1)
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: I32x16Backend> Mul<i32> for i32x16<T, M> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: i32) -> Self {
        Self::new_repr(T::mul(self.1, self.0, T::splat(self.1, rhs)), self.1)
    }
}

// ============================================================================
// Index
// ============================================================================

impl<M: crate::simd::generic::ConstructorMode, T: I32x16Backend> Index<usize> for i32x16<T, M> {
    type Output = i32;
    #[inline(always)]
    fn index(&self, i: usize) -> &i32 {
        &crate::simd_storage::view::<_, [i32; 16]>(&self.0)[i]
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: I32x16Backend> IndexMut<usize> for i32x16<T, M> {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut i32 {
        &mut crate::simd_storage::view_mut::<_, [i32; 16]>(&mut self.0)[i]
    }
}

// ============================================================================
// Conversions
// ============================================================================

impl<M: crate::simd::generic::ConstructorMode, T: I32x16Backend> From<i32x16<T, M>> for [i32; 16] {
    #[inline(always)]
    fn from(v: i32x16<T, M>) -> [i32; 16] {
        T::to_array(v.1, v.0)
    }
}

// ============================================================================
// Debug
// ============================================================================

impl<M: crate::simd::generic::ConstructorMode, T: I32x16Backend> core::fmt::Debug for i32x16<T, M> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let arr = T::to_array(self.1, self.0);
        f.debug_tuple("i32x16").field(&arr).finish()
    }
}

// ============================================================================
// Cross-type conversions (available when T implements conversion traits)
// ============================================================================

impl<M: crate::simd::generic::ConstructorMode, T: crate::simd::backends::F32x16Convert>
    i32x16<T, M>
{
    /// Bitcast to f32x16 (reinterpret bits, no conversion).
    #[inline(always)]
    pub fn bitcast_to_f32(self) -> super::f32x16<T, M> {
        super::f32x16::from_repr_unchecked(self.1, T::bitcast_i32_to_f32(self.1, self.0))
    }

    /// Convert to f32x16 (numeric conversion).
    #[inline(always)]
    pub fn to_f32(self) -> super::f32x16<T, M> {
        super::f32x16::from_repr_unchecked(self.1, T::convert_i32_to_f32(self.1, self.0))
    }

    // ====== Backward-compatible aliases (old generated API names) ======

    /// Alias for [`bitcast_to_f32`](Self::bitcast_to_f32).
    #[inline(always)]
    pub fn bitcast_f32x16(self) -> super::f32x16<T, M> {
        self.bitcast_to_f32()
    }

    /// Alias for [`to_f32`](Self::to_f32).
    #[inline(always)]
    pub fn to_f32x16(self) -> super::f32x16<T, M> {
        self.to_f32()
    }
}

// ============================================================================
// Saturating narrowing (i32x16 -> i16x32 / u16x32)
// ============================================================================

impl<M: crate::simd::generic::ConstructorMode, T: crate::simd::backends::I32x16Backend>
    i32x16<T, M>
{
    /// Narrow `self` and `high` to `i16x32`, clamping each lane to
    /// the `i16` range.
    ///
    /// Result lane `i` is `self[i]` clamped for `i < 16`, and
    /// `high[i - 16]` clamped for `i >= 16` — the same lane order
    /// on every backend (the AVX2 arm pays one
    /// `permute4x64` to get there).
    #[inline(always)]
    pub fn narrow_saturating_i16(self, high: Self) -> super::i16x32<T, M>
    where
        T: crate::simd::backends::I16x32Backend,
    {
        super::i16x32::from_repr_unchecked(
            self.1,
            <T as crate::simd::backends::I32x16Backend>::narrow_saturating_i32_to_i16(
                self.1, self.0, high.0,
            ),
        )
    }

    /// Narrow `self` and `high` to `u16x32`, clamping each lane to
    /// the `u16` range.
    ///
    /// The source stays `i32` to match native signed-source packs on
    /// x86 and WASM. An unsigned-source operation would require a
    /// different lowering to preserve its full input range.
    #[inline(always)]
    pub fn narrow_saturating_u16(self, high: Self) -> super::u16x32<T, M>
    where
        T: crate::simd::backends::U16x32Backend,
    {
        super::u16x32::from_repr_unchecked(
            self.1,
            <T as crate::simd::backends::I32x16Backend>::narrow_saturating_i32_to_u16(
                self.1, self.0, high.0,
            ),
        )
    }
}

// ============================================================================
// Platform-specific concrete impls
// ============================================================================

impl<M: crate::simd::generic::ConstructorMode> i32x16<archmage::ScalarToken, M> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "scalar::i32x16"
    }
}

#[cfg(target_arch = "x86_64")]
impl<M: crate::simd::generic::ConstructorMode> i32x16<archmage::X64V3Token, M> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "polyfill::v3_512::i32x16"
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl<M: crate::simd::generic::ConstructorMode> i32x16<archmage::X64V4Token, M> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "x86::v4::i32x16"
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl<M: crate::simd::generic::ConstructorMode> i32x16<archmage::X64V4xToken, M> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "x86::v4x::i32x16"
    }
}

#[cfg(target_arch = "aarch64")]
impl<M: crate::simd::generic::ConstructorMode> i32x16<archmage::NeonToken, M> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "polyfill::neon_512::i32x16"
    }
}

#[cfg(target_arch = "wasm32")]
impl<M: crate::simd::generic::ConstructorMode> i32x16<archmage::Wasm128Token, M> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "polyfill::wasm128_512::i32x16"
    }
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl<M: crate::simd::generic::ConstructorMode> i32x16<archmage::X64V4Token, M> {
    /// Get the raw `__m512i` value.
    #[inline(always)]
    pub fn raw(self) -> core::arch::x86_64::__m512i {
        self.0
    }

    /// Wrap a raw `__m512i` using an existing CPU capability token.
    #[inline(always)]
    pub(crate) fn from_m512i_with_token(
        token: archmage::X64V4Token,
        value: core::arch::x86_64::__m512i,
    ) -> Self {
        Self::new_repr(value, token)
    }

    /// Wrap a raw `__m512i` in a matching target-feature context.
    ///
    /// Rust requires the caller to enable the `v4` tier's features.
    /// Use an archmage `#[rite(v4)]` helper or `#[arcane]` entry point.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
    )]
    #[inline]
    pub fn from_raw(value: core::arch::x86_64::__m512i) -> Self {
        Self::new_repr(value, archmage::X64V4Token::from_context())
    }
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl<M: crate::simd::generic::ConstructorMode> i32x16<archmage::X64V4xToken, M> {
    /// Get the raw `__m512i` value.
    #[inline(always)]
    pub fn raw(self) -> core::arch::x86_64::__m512i {
        self.0
    }

    /// Wrap a raw `__m512i` using an existing CPU capability token.
    #[inline(always)]
    pub(crate) fn from_m512i_with_token(
        token: archmage::X64V4xToken,
        value: core::arch::x86_64::__m512i,
    ) -> Self {
        Self::new_repr(value, token)
    }

    /// Wrap a raw `__m512i` in a matching target-feature context.
    ///
    /// Rust requires the caller to enable the `v4x` tier's features.
    /// Use an archmage `#[rite(v4x)]` helper or `#[arcane]` entry point.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512vpopcntdq,avx512ifma,avx512vbmi,avx512vbmi2,avx512bitalg,avx512vnni,vpclmulqdq,gfni,vaes"
    )]
    #[inline]
    pub fn from_raw(value: core::arch::x86_64::__m512i) -> Self {
        Self::new_repr(value, archmage::X64V4xToken::from_context())
    }
}

// ============================================================================
// Extension: popcnt (requires Modern token)
// ============================================================================

#[cfg(feature = "avx512")]
impl<M: crate::simd::generic::ConstructorMode, T: crate::simd::backends::i32x16PopcntBackend>
    i32x16<T, M>
{
    /// Count set bits in each lane (popcnt).
    ///
    /// Returns a vector where each lane contains the number of 1-bits
    /// in the corresponding lane of `self`.
    ///
    /// Requires AVX-512 Modern token (VPOPCNTDQ or BITALG extension).
    #[inline(always)]
    pub fn popcnt(self) -> Self {
        Self::new_repr(T::popcnt(self.1, self.0), self.1)
    }
}
impl<T: I32x16Backend> From<i32x16<T, crate::simd::generic::Explicit>>
    for i32x16<T, crate::simd::generic::Context>
{
    #[inline(always)]
    fn from(value: i32x16<T, crate::simd::generic::Explicit>) -> Self {
        Self::new_repr(value.0, value.1)
    }
}
impl<T: I32x16Backend> From<i32x16<T, crate::simd::generic::Context>>
    for i32x16<T, crate::simd::generic::Explicit>
{
    #[inline(always)]
    fn from(value: i32x16<T, crate::simd::generic::Context>) -> Self {
        Self::new_repr(value.0, value.1)
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl i32x16<archmage::X64V4Token, crate::simd::generic::Context> {
    /// Wrap a raw `__m512i` using an existing CPU capability token.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
    )]
    #[inline]
    pub fn from_m512i(value: core::arch::x86_64::__m512i) -> Self {
        let token = archmage::X64V4Token::from_context();
        Self::new_repr(value, token)
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl i32x16<archmage::X64V4xToken, crate::simd::generic::Context> {
    /// Wrap a raw `__m512i` using an existing CPU capability token.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512vpopcntdq,avx512ifma,avx512vbmi,avx512vbmi2,avx512bitalg,avx512vnni,vpclmulqdq,gfni,vaes"
    )]
    #[inline]
    pub fn from_m512i(value: core::arch::x86_64::__m512i) -> Self {
        let token = archmage::X64V4xToken::from_context();
        Self::new_repr(value, token)
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl i32x16<archmage::X64V4Token, crate::simd::generic::Explicit> {
    /// Wrap a raw `__m512i` using an existing CPU capability token.
    #[inline(always)]
    pub fn from_m512i(token: archmage::X64V4Token, value: core::arch::x86_64::__m512i) -> Self {
        Self::from_m512i_with_token(token, value)
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl i32x16<archmage::X64V4xToken, crate::simd::generic::Explicit> {
    /// Wrap a raw `__m512i` using an existing CPU capability token.
    #[inline(always)]
    pub fn from_m512i(token: archmage::X64V4xToken, value: core::arch::x86_64::__m512i) -> Self {
        Self::from_m512i_with_token(token, value)
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl i32x16<archmage::X64V4Token, crate::simd::generic::Context> {
    /// Broadcast scalar to all 16 lanes.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
    )]
    #[inline]
    pub fn splat(v: i32) -> Self {
        let token = archmage::X64V4Token::from_context();
        Self::new_repr(
            <archmage::X64V4Token as I32x16Backend>::splat(token, v),
            token,
        )
    }
    /// All lanes zero.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
    )]
    #[inline]
    pub fn zero() -> Self {
        let token = archmage::X64V4Token::from_context();
        Self::new_repr(<archmage::X64V4Token as I32x16Backend>::zero(token), token)
    }
    /// Load from a `[i32; 16]` array.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
    )]
    #[inline]
    pub fn load(data: &[i32; 16]) -> Self {
        let token = archmage::X64V4Token::from_context();
        Self::new_repr(
            <archmage::X64V4Token as I32x16Backend>::load(token, data),
            token,
        )
    }
    /// Create from array (zero-cost where possible).
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
    )]
    #[inline]
    pub fn from_array(arr: [i32; 16]) -> Self {
        let token = archmage::X64V4Token::from_context();
        Self::new_repr(
            <archmage::X64V4Token as I32x16Backend>::from_array(token, arr),
            token,
        )
    }
    /// Create from slice. Panics if `slice.len() < 16`.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
    )]
    #[inline]
    pub fn from_slice(slice: &[i32]) -> Self {
        Self::from_slice_with_token(archmage::X64V4Token::from_context(), slice)
    }
    /// Split a slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&[[i32; 16]], &[i32])` — fixed-size arrays suitable
    /// for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
    )]
    #[inline]
    pub fn partition_slice(data: &[i32]) -> (&[[i32; 16]], &[i32]) {
        Self::partition_slice_with_token(archmage::X64V4Token::from_context(), data)
    }
    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&mut [[i32; 16]], &mut [i32])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
    )]
    #[inline]
    pub fn partition_slice_mut(data: &mut [i32]) -> (&mut [[i32; 16]], &mut [i32]) {
        Self::partition_slice_mut_with_token(archmage::X64V4Token::from_context(), data)
    }
    /// Wrap a platform representation (requires matching target features).
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
    )]
    #[inline]
    pub fn from_repr(
        repr: <archmage::X64V4Token as crate::simd::backends::I32x16Backend>::Repr,
    ) -> Self {
        let token = archmage::X64V4Token::from_context();
        Self::new_repr(repr, token)
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl i32x16<archmage::X64V4xToken, crate::simd::generic::Context> {
    /// Broadcast scalar to all 16 lanes.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512vpopcntdq,avx512ifma,avx512vbmi,avx512vbmi2,avx512bitalg,avx512vnni,vpclmulqdq,gfni,vaes"
    )]
    #[inline]
    pub fn splat(v: i32) -> Self {
        let token = archmage::X64V4xToken::from_context();
        Self::new_repr(
            <archmage::X64V4xToken as I32x16Backend>::splat(token, v),
            token,
        )
    }
    /// All lanes zero.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512vpopcntdq,avx512ifma,avx512vbmi,avx512vbmi2,avx512bitalg,avx512vnni,vpclmulqdq,gfni,vaes"
    )]
    #[inline]
    pub fn zero() -> Self {
        let token = archmage::X64V4xToken::from_context();
        Self::new_repr(<archmage::X64V4xToken as I32x16Backend>::zero(token), token)
    }
    /// Load from a `[i32; 16]` array.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512vpopcntdq,avx512ifma,avx512vbmi,avx512vbmi2,avx512bitalg,avx512vnni,vpclmulqdq,gfni,vaes"
    )]
    #[inline]
    pub fn load(data: &[i32; 16]) -> Self {
        let token = archmage::X64V4xToken::from_context();
        Self::new_repr(
            <archmage::X64V4xToken as I32x16Backend>::load(token, data),
            token,
        )
    }
    /// Create from array (zero-cost where possible).
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512vpopcntdq,avx512ifma,avx512vbmi,avx512vbmi2,avx512bitalg,avx512vnni,vpclmulqdq,gfni,vaes"
    )]
    #[inline]
    pub fn from_array(arr: [i32; 16]) -> Self {
        let token = archmage::X64V4xToken::from_context();
        Self::new_repr(
            <archmage::X64V4xToken as I32x16Backend>::from_array(token, arr),
            token,
        )
    }
    /// Create from slice. Panics if `slice.len() < 16`.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512vpopcntdq,avx512ifma,avx512vbmi,avx512vbmi2,avx512bitalg,avx512vnni,vpclmulqdq,gfni,vaes"
    )]
    #[inline]
    pub fn from_slice(slice: &[i32]) -> Self {
        Self::from_slice_with_token(archmage::X64V4xToken::from_context(), slice)
    }
    /// Split a slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&[[i32; 16]], &[i32])` — fixed-size arrays suitable
    /// for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512vpopcntdq,avx512ifma,avx512vbmi,avx512vbmi2,avx512bitalg,avx512vnni,vpclmulqdq,gfni,vaes"
    )]
    #[inline]
    pub fn partition_slice(data: &[i32]) -> (&[[i32; 16]], &[i32]) {
        Self::partition_slice_with_token(archmage::X64V4xToken::from_context(), data)
    }
    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&mut [[i32; 16]], &mut [i32])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512vpopcntdq,avx512ifma,avx512vbmi,avx512vbmi2,avx512bitalg,avx512vnni,vpclmulqdq,gfni,vaes"
    )]
    #[inline]
    pub fn partition_slice_mut(data: &mut [i32]) -> (&mut [[i32; 16]], &mut [i32]) {
        Self::partition_slice_mut_with_token(archmage::X64V4xToken::from_context(), data)
    }
    /// Wrap a platform representation (requires matching target features).
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512vpopcntdq,avx512ifma,avx512vbmi,avx512vbmi2,avx512bitalg,avx512vnni,vpclmulqdq,gfni,vaes"
    )]
    #[inline]
    pub fn from_repr(
        repr: <archmage::X64V4xToken as crate::simd::backends::I32x16Backend>::Repr,
    ) -> Self {
        let token = archmage::X64V4xToken::from_context();
        Self::new_repr(repr, token)
    }
}

#[cfg(target_arch = "aarch64")]
impl i32x16<archmage::NeonToken, crate::simd::generic::Context> {
    /// Broadcast scalar to all 16 lanes.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(enable = "neon")]
    #[inline]
    pub fn splat(v: i32) -> Self {
        let token = archmage::NeonToken::from_context();
        Self::new_repr(
            <archmage::NeonToken as I32x16Backend>::splat(token, v),
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
        Self::new_repr(<archmage::NeonToken as I32x16Backend>::zero(token), token)
    }
    /// Load from a `[i32; 16]` array.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(enable = "neon")]
    #[inline]
    pub fn load(data: &[i32; 16]) -> Self {
        let token = archmage::NeonToken::from_context();
        Self::new_repr(
            <archmage::NeonToken as I32x16Backend>::load(token, data),
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
    pub fn from_array(arr: [i32; 16]) -> Self {
        let token = archmage::NeonToken::from_context();
        Self::new_repr(
            <archmage::NeonToken as I32x16Backend>::from_array(token, arr),
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
    pub fn from_slice(slice: &[i32]) -> Self {
        Self::from_slice_with_token(archmage::NeonToken::from_context(), slice)
    }
    /// Split a slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&[[i32; 16]], &[i32])` — fixed-size arrays suitable
    /// for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(enable = "neon")]
    #[inline]
    pub fn partition_slice(data: &[i32]) -> (&[[i32; 16]], &[i32]) {
        Self::partition_slice_with_token(archmage::NeonToken::from_context(), data)
    }
    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&mut [[i32; 16]], &mut [i32])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(enable = "neon")]
    #[inline]
    pub fn partition_slice_mut(data: &mut [i32]) -> (&mut [[i32; 16]], &mut [i32]) {
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
        repr: <archmage::NeonToken as crate::simd::backends::I32x16Backend>::Repr,
    ) -> Self {
        let token = archmage::NeonToken::from_context();
        Self::new_repr(repr, token)
    }
}

#[cfg(target_arch = "wasm32")]
impl i32x16<archmage::Wasm128Token, crate::simd::generic::Context> {
    /// Broadcast scalar to all 16 lanes.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(enable = "simd128")]
    #[inline]
    pub fn splat(v: i32) -> Self {
        let token = archmage::Wasm128Token::from_context();
        Self::new_repr(
            <archmage::Wasm128Token as I32x16Backend>::splat(token, v),
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
        Self::new_repr(
            <archmage::Wasm128Token as I32x16Backend>::zero(token),
            token,
        )
    }
    /// Load from a `[i32; 16]` array.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(enable = "simd128")]
    #[inline]
    pub fn load(data: &[i32; 16]) -> Self {
        let token = archmage::Wasm128Token::from_context();
        Self::new_repr(
            <archmage::Wasm128Token as I32x16Backend>::load(token, data),
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
    pub fn from_array(arr: [i32; 16]) -> Self {
        let token = archmage::Wasm128Token::from_context();
        Self::new_repr(
            <archmage::Wasm128Token as I32x16Backend>::from_array(token, arr),
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
    pub fn from_slice(slice: &[i32]) -> Self {
        Self::from_slice_with_token(archmage::Wasm128Token::from_context(), slice)
    }
    /// Split a slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&[[i32; 16]], &[i32])` — fixed-size arrays suitable
    /// for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(enable = "simd128")]
    #[inline]
    pub fn partition_slice(data: &[i32]) -> (&[[i32; 16]], &[i32]) {
        Self::partition_slice_with_token(archmage::Wasm128Token::from_context(), data)
    }
    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&mut [[i32; 16]], &mut [i32])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(enable = "simd128")]
    #[inline]
    pub fn partition_slice_mut(data: &mut [i32]) -> (&mut [[i32; 16]], &mut [i32]) {
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
        repr: <archmage::Wasm128Token as crate::simd::backends::I32x16Backend>::Repr,
    ) -> Self {
        let token = archmage::Wasm128Token::from_context();
        Self::new_repr(repr, token)
    }
}

#[cfg(target_arch = "x86_64")]
impl i32x16<archmage::X64V3Token, crate::simd::generic::Context> {
    /// Broadcast scalar to all 16 lanes.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    pub fn splat(v: i32) -> Self {
        let token = archmage::X64V3Token::from_context();
        Self::new_repr(
            <archmage::X64V3Token as I32x16Backend>::splat(token, v),
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
        Self::new_repr(<archmage::X64V3Token as I32x16Backend>::zero(token), token)
    }
    /// Load from a `[i32; 16]` array.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    pub fn load(data: &[i32; 16]) -> Self {
        let token = archmage::X64V3Token::from_context();
        Self::new_repr(
            <archmage::X64V3Token as I32x16Backend>::load(token, data),
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
    pub fn from_array(arr: [i32; 16]) -> Self {
        let token = archmage::X64V3Token::from_context();
        Self::new_repr(
            <archmage::X64V3Token as I32x16Backend>::from_array(token, arr),
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
    pub fn from_slice(slice: &[i32]) -> Self {
        Self::from_slice_with_token(archmage::X64V3Token::from_context(), slice)
    }
    /// Split a slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&[[i32; 16]], &[i32])` — fixed-size arrays suitable
    /// for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    pub fn partition_slice(data: &[i32]) -> (&[[i32; 16]], &[i32]) {
        Self::partition_slice_with_token(archmage::X64V3Token::from_context(), data)
    }
    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&mut [[i32; 16]], &mut [i32])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    pub fn partition_slice_mut(data: &mut [i32]) -> (&mut [[i32; 16]], &mut [i32]) {
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
        repr: <archmage::X64V3Token as crate::simd::backends::I32x16Backend>::Repr,
    ) -> Self {
        let token = archmage::X64V3Token::from_context();
        Self::new_repr(repr, token)
    }
}

impl<T: I32x16Backend> i32x16<T, crate::simd::generic::Explicit> {
    /// Broadcast scalar to all 16 lanes.
    #[inline(always)]
    pub fn splat(token: T, v: i32) -> Self {
        Self::splat_with_token(token, v)
    }
    /// All lanes zero.
    #[inline(always)]
    pub fn zero(token: T) -> Self {
        Self::zero_with_token(token)
    }
    /// Load from a `[i32; 16]` array.
    #[inline(always)]
    pub fn load(token: T, data: &[i32; 16]) -> Self {
        Self::load_with_token(token, data)
    }
    /// Create from array (zero-cost where possible).
    #[inline(always)]
    pub fn from_array(token: T, arr: [i32; 16]) -> Self {
        Self::from_array_with_token(token, arr)
    }
    /// Create from slice. Panics if `slice.len() < 16`.
    #[inline(always)]
    pub fn from_slice(token: T, slice: &[i32]) -> Self {
        Self::from_slice_with_token(token, slice)
    }
    /// Split a slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&[[i32; 16]], &[i32])` — fixed-size arrays suitable
    /// for [`load`](Self::load), plus any leftover elements.
    #[inline(always)]
    pub fn partition_slice(_token: T, data: &[i32]) -> (&[[i32; 16]], &[i32]) {
        Self::partition_slice_with_token(_token, data)
    }
    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&mut [[i32; 16]], &mut [i32])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[inline(always)]
    pub fn partition_slice_mut(_token: T, data: &mut [i32]) -> (&mut [[i32; 16]], &mut [i32]) {
        Self::partition_slice_mut_with_token(_token, data)
    }
    /// Wrap a platform representation (token-gated).
    #[inline(always)]
    pub fn from_repr(token: T, repr: T::Repr) -> Self {
        Self::from_repr_with_token(token, repr)
    }
}

impl i32x16<archmage::ScalarToken, crate::simd::generic::Context> {
    /// Broadcast scalar to all 16 lanes.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn splat(v: i32) -> Self {
        let token = archmage::ScalarToken;
        Self::new_repr(
            <archmage::ScalarToken as I32x16Backend>::splat(token, v),
            token,
        )
    }
    /// All lanes zero.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn zero() -> Self {
        let token = archmage::ScalarToken;
        Self::new_repr(<archmage::ScalarToken as I32x16Backend>::zero(token), token)
    }
    /// Load from a `[i32; 16]` array.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn load(data: &[i32; 16]) -> Self {
        let token = archmage::ScalarToken;
        Self::new_repr(
            <archmage::ScalarToken as I32x16Backend>::load(token, data),
            token,
        )
    }
    /// Create from array (zero-cost where possible).
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn from_array(arr: [i32; 16]) -> Self {
        let token = archmage::ScalarToken;
        Self::new_repr(
            <archmage::ScalarToken as I32x16Backend>::from_array(token, arr),
            token,
        )
    }
    /// Create from slice. Panics if `slice.len() < 16`.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn from_slice(slice: &[i32]) -> Self {
        Self::from_slice_with_token(archmage::ScalarToken, slice)
    }
    /// Split a slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&[[i32; 16]], &[i32])` — fixed-size arrays suitable
    /// for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn partition_slice(data: &[i32]) -> (&[[i32; 16]], &[i32]) {
        Self::partition_slice_with_token(archmage::ScalarToken, data)
    }
    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&mut [[i32; 16]], &mut [i32])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn partition_slice_mut(data: &mut [i32]) -> (&mut [[i32; 16]], &mut [i32]) {
        Self::partition_slice_mut_with_token(archmage::ScalarToken, data)
    }
    /// Wrap a platform representation (requires matching target features).
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn from_repr(
        repr: <archmage::ScalarToken as crate::simd::backends::I32x16Backend>::Repr,
    ) -> Self {
        let token = archmage::ScalarToken;
        Self::new_repr(repr, token)
    }
}

//! Generic `i16x32<T>` — 32-lane i16 SIMD vector parameterized by backend.
//!
//! `T` is a token type (e.g., `X64V4Token`, `ScalarToken`)
//! that determines the platform-native representation and intrinsics used.
//! The struct delegates all operations to the [`I16x32Backend`] trait.

#![allow(clippy::should_implement_trait)]

use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Index,
    IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use crate::simd::backends::I16x32Backend;

/// 32-lane i16 SIMD vector, generic over backend `T`.
///
/// `T` is a token type that proves CPU support for the required SIMD features.
/// The inner representation is `T::Repr` (e.g., `__m512i` on AVX-512, `[i16; 32]` on scalar).
///
/// **The token is stored** (as a zero-sized field) so methods receiving
/// `self: i16x32<T>` can re-supply it to backend operations that
/// require a token value (e.g. `T::splat(token, v)`). This carries the
/// token-as-feature-proof guarantee through every method call without
/// runtime overhead — `T` is ZST, so `sizeof(i16x32<T>) == sizeof(T::Repr)`
/// and `align_of(i16x32<T>) == align_of(T::Repr)` under `#[repr(C)]`.
///
/// # Layout
///
/// `#[repr(C)]` with a ZST trailing field: `T::Repr` lives at offset 0
/// and `T` plus the sealed policy marker are zero-sized tails. Bitcasts between `i16x32<T>` values of
/// different element-types are sound when the Repr types share a layout
/// (e.g. `__m128` and `__m128i` are both 16-byte aligned 128-bit values).
/// `#[repr(transparent)]` cannot be used because Rust cannot prove at
/// the struct definition site that a generic `T` is a 1-ZST.
///
/// Fixed-policy aliases select explicit-token or feature-context constructors.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct i16x32<
    T: I16x32Backend,
    M: crate::simd::generic::ConstructorMode = crate::simd::generic::Explicit,
>(
    pub(crate) T::Repr,
    pub(crate) T,
    pub(crate) core::marker::PhantomData<M>,
);
// SAFETY: repr(C) pair of Pod storage and a sealed 1-ZST token.
// A supplied T proves CPU support; the wrapper adds no bit invariants.
// Helpers additionally check token size/alignment at monomorphization.
unsafe impl<M: crate::simd::generic::ConstructorMode, T: I16x32Backend>
    crate::simd_storage::TokenStorage for i16x32<T, M>
{
    type Token = T;
}

// Layout invariant: struct is `#[repr(C)]` with a trailing ZST `T`
// field, so `sizeof/alignof(i16x32<T>) == sizeof/alignof(T::Repr)`
// when `T` is a 1-ZST. Every archmage token currently satisfies this;
// if a future refactor adds a non-ZST field to a token, this const
// assert fires at compile time.
const _: () = {
    assert!(
        core::mem::size_of::<i16x32<archmage::ScalarToken>>()
            == core::mem::size_of::<
                <archmage::ScalarToken as crate::simd::backends::I16x32Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<i16x32<archmage::ScalarToken>>()
            == core::mem::align_of::<
                <archmage::ScalarToken as crate::simd::backends::I16x32Backend>::Repr,
            >()
    );
};

#[cfg(target_arch = "x86_64")]
const _: () = {
    assert!(
        core::mem::size_of::<i16x32<archmage::X64V3Token>>()
            == core::mem::size_of::<
                <archmage::X64V3Token as crate::simd::backends::I16x32Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<i16x32<archmage::X64V3Token>>()
            == core::mem::align_of::<
                <archmage::X64V3Token as crate::simd::backends::I16x32Backend>::Repr,
            >()
    );
};

// Native AVX-512 (`__m512`/`__m512d`/`__m512i`) — gated on the
// `avx512` feature, which is how archmage exposes X64V4Token's
// 512-bit backend impls.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
const _: () = {
    assert!(
        core::mem::size_of::<i16x32<archmage::X64V4Token>>()
            == core::mem::size_of::<
                <archmage::X64V4Token as crate::simd::backends::I16x32Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<i16x32<archmage::X64V4Token>>()
            == core::mem::align_of::<
                <archmage::X64V4Token as crate::simd::backends::I16x32Backend>::Repr,
            >()
    );
};

#[cfg(target_arch = "aarch64")]
const _: () = {
    assert!(
        core::mem::size_of::<i16x32<archmage::NeonToken>>()
            == core::mem::size_of::<
                <archmage::NeonToken as crate::simd::backends::I16x32Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<i16x32<archmage::NeonToken>>()
            == core::mem::align_of::<
                <archmage::NeonToken as crate::simd::backends::I16x32Backend>::Repr,
            >()
    );
};

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
const _: () = {
    assert!(
        core::mem::size_of::<i16x32<archmage::Wasm128Token>>()
            == core::mem::size_of::<
                <archmage::Wasm128Token as crate::simd::backends::I16x32Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<i16x32<archmage::Wasm128Token>>()
            == core::mem::align_of::<
                <archmage::Wasm128Token as crate::simd::backends::I16x32Backend>::Repr,
            >()
    );
};

impl<M: crate::simd::generic::ConstructorMode, T: I16x32Backend> i16x32<T, M> {
    #[inline(always)]
    pub(crate) fn new_repr(repr: T::Repr, token: T) -> Self {
        Self(repr, token, core::marker::PhantomData)
    }

    /// Number of i16 lanes.
    pub const LANES: usize = 32;

    // ====== Construction (token-gated) ======

    /// Broadcast scalar to all 32 lanes.
    #[inline(always)]
    pub(crate) fn splat_with_token(token: T, v: i16) -> Self {
        Self::new_repr(T::splat(token, v), token)
    }

    /// All lanes zero.
    #[inline(always)]
    pub(crate) fn zero_with_token(token: T) -> Self {
        Self::new_repr(T::zero(token), token)
    }

    /// Load from a `[i16; 32]` array.
    #[inline(always)]
    pub(crate) fn load_with_token(token: T, data: &[i16; 32]) -> Self {
        Self::new_repr(T::load(token, data), token)
    }

    /// Create from array (zero-cost where possible).
    #[inline(always)]
    pub(crate) fn from_array_with_token(token: T, arr: [i16; 32]) -> Self {
        Self::new_repr(T::from_array(token, arr), token)
    }

    /// Create from slice. Panics if `slice.len() < 32`.
    #[inline(always)]
    pub(crate) fn from_slice_with_token(token: T, slice: &[i16]) -> Self {
        let arr: [i16; 32] = slice[..32].try_into().unwrap();
        Self::new_repr(T::from_array(token, arr), token)
    }

    /// Split a slice into SIMD-width chunks and a scalar remainder.
    ///
    /// Returns `(&[[i16; 32]], &[i16])` — fixed-size arrays suitable
    /// for [`load`](Self::load), plus any leftover elements.
    #[inline(always)]
    pub(crate) fn partition_slice_with_token(_token: T, data: &[i16]) -> (&[[i16; 32]], &[i16]) {
        data.as_chunks::<32>()
    }

    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    ///
    /// Returns `(&mut [[i16; 32]], &mut [i16])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[inline(always)]
    pub(crate) fn partition_slice_mut_with_token(
        _token: T,
        data: &mut [i16],
    ) -> (&mut [[i16; 32]], &mut [i16]) {
        data.as_chunks_mut::<32>()
    }

    // ====== Accessors ======

    /// Store to array.
    #[inline(always)]
    pub fn store(self, out: &mut [i16; 32]) {
        T::store(self.1, self.0, out);
    }

    /// Convert to array.
    #[inline(always)]
    pub fn to_array(self) -> [i16; 32] {
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

    /// Sum all 32 lanes (wrapping).
    #[inline(always)]
    pub fn reduce_add(self) -> i16 {
        T::reduce_add(self.1, self.0)
    }

    // ====== Shifts ======

    /// Shift left by constant.
    ///
    /// `N` must be in `0..=15`; out-of-range `N` fails to compile,
    /// identically on every backend.
    #[inline(always)]
    pub fn shl_const<const N: i32>(self) -> Self {
        const { assert!(N >= 0 && N <= 15, "shift amount out of range") };
        Self::new_repr(T::shl_const::<N>(self.1, self.0), self.1)
    }

    /// Arithmetic shift right by constant (sign-extending).
    ///
    /// `N` must be in `0..=15` (`N == 0` is the identity shift);
    /// out-of-range `N` fails to compile, identically on every backend.
    #[inline(always)]
    pub fn shr_arithmetic_const<const N: i32>(self) -> Self {
        const { assert!(N >= 0 && N <= 15, "shift amount out of range") };
        Self::new_repr(T::shr_arithmetic_const::<N>(self.1, self.0), self.1)
    }

    /// Logical shift right by constant (zero-filling).
    ///
    /// `N` must be in `0..=15` (`N == 0` is the identity shift);
    /// out-of-range `N` fails to compile, identically on every backend.
    #[inline(always)]
    pub fn shr_logical_const<const N: i32>(self) -> Self {
        const { assert!(N >= 0 && N <= 15, "shift amount out of range") };
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
    /// `count >= 16` yields all-zero lanes — the same result on every
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
    /// `count >= 16` yields all-zero lanes on every backend.
    #[inline(always)]
    pub fn shr_logical_uniform(self, count: u32) -> Self {
        Self::new_repr(T::shr_logical_uniform(self.1, self.0, count), self.1)
    }

    /// Arithmetic (sign-filling) shift right by a runtime `count`,
    /// applied identically to every lane.
    ///
    /// `count >= 16` yields a sign fill (every lane becomes `0` or
    /// `-1`), equivalent to shifting by 15, on every backend.
    #[inline(always)]
    pub fn shr_arithmetic_uniform(self, count: u32) -> Self {
        Self::new_repr(T::shr_arithmetic_uniform(self.1, self.0, count), self.1)
    }

    // ====== Saturating arithmetic ======

    /// Lane-wise addition that clamps to the `i16` range instead of
    /// wrapping — `i16::saturating_add`, per lane.
    #[inline(always)]
    pub fn saturating_add(self, other: Self) -> Self {
        Self::new_repr(T::saturating_add(self.1, self.0, other.0), self.1)
    }

    /// Lane-wise subtraction that clamps to the `i16` range instead of
    /// wrapping — `i16::saturating_sub`, per lane.
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

    /// Extract the high bit of each 16-bit lane as a bitmask.
    #[inline(always)]
    pub fn bitmask(self) -> u64 {
        T::bitmask(self.1, self.0)
    }
}

// ============================================================================
// Operator implementations
// ============================================================================

impl<M: crate::simd::generic::ConstructorMode, T: I16x32Backend> Add for i16x32<T, M> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::new_repr(T::add(self.1, self.0, rhs.0), self.1)
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: I16x32Backend> Sub for i16x32<T, M> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::new_repr(T::sub(self.1, self.0, rhs.0), self.1)
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: I16x32Backend> Mul for i16x32<T, M> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self::new_repr(T::mul(self.1, self.0, rhs.0), self.1)
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: I16x32Backend> Neg for i16x32<T, M> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self::new_repr(T::neg(self.1, self.0), self.1)
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: I16x32Backend> BitAnd for i16x32<T, M> {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self::new_repr(T::bitand(self.1, self.0, rhs.0), self.1)
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: I16x32Backend> BitOr for i16x32<T, M> {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self::new_repr(T::bitor(self.1, self.0, rhs.0), self.1)
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: I16x32Backend> BitXor for i16x32<T, M> {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Self::new_repr(T::bitxor(self.1, self.0, rhs.0), self.1)
    }
}

// ============================================================================
// Assign operators
// ============================================================================

impl<M: crate::simd::generic::ConstructorMode, T: I16x32Backend> AddAssign for i16x32<T, M> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: I16x32Backend> SubAssign for i16x32<T, M> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: I16x32Backend> MulAssign for i16x32<T, M> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: I16x32Backend> BitAndAssign for i16x32<T, M> {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: I16x32Backend> BitOrAssign for i16x32<T, M> {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: I16x32Backend> BitXorAssign for i16x32<T, M> {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = *self ^ rhs;
    }
}

// ============================================================================
// Scalar broadcast operators (v + 2, v * 3, etc.)
// ============================================================================

impl<M: crate::simd::generic::ConstructorMode, T: I16x32Backend> Add<i16> for i16x32<T, M> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: i16) -> Self {
        Self::new_repr(T::add(self.1, self.0, T::splat(self.1, rhs)), self.1)
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: I16x32Backend> Sub<i16> for i16x32<T, M> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: i16) -> Self {
        Self::new_repr(T::sub(self.1, self.0, T::splat(self.1, rhs)), self.1)
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: I16x32Backend> Mul<i16> for i16x32<T, M> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: i16) -> Self {
        Self::new_repr(T::mul(self.1, self.0, T::splat(self.1, rhs)), self.1)
    }
}

// ============================================================================
// Index
// ============================================================================

impl<M: crate::simd::generic::ConstructorMode, T: I16x32Backend> Index<usize> for i16x32<T, M> {
    type Output = i16;
    #[inline(always)]
    fn index(&self, i: usize) -> &i16 {
        &crate::simd_storage::view::<_, [i16; 32]>(&self.0)[i]
    }
}

impl<M: crate::simd::generic::ConstructorMode, T: I16x32Backend> IndexMut<usize> for i16x32<T, M> {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut i16 {
        &mut crate::simd_storage::view_mut::<_, [i16; 32]>(&mut self.0)[i]
    }
}

// ============================================================================
// Conversions
// ============================================================================

impl<M: crate::simd::generic::ConstructorMode, T: I16x32Backend> From<i16x32<T, M>> for [i16; 32] {
    #[inline(always)]
    fn from(v: i16x32<T, M>) -> [i16; 32] {
        T::to_array(v.1, v.0)
    }
}

// ============================================================================
// Debug
// ============================================================================

impl<M: crate::simd::generic::ConstructorMode, T: I16x32Backend> core::fmt::Debug for i16x32<T, M> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let arr = T::to_array(self.1, self.0);
        f.debug_tuple("i16x32").field(&arr).finish()
    }
}

impl<
    M: crate::simd::generic::ConstructorMode,
    T: crate::simd::backends::I16x32Backend + crate::simd::backends::U16x32Backend,
> i16x32<T, M>
{
    /// Reinterpret all 32 lanes as u16x32, preserving every bit.
    #[inline(always)]
    pub fn bitcast_u16x32(self) -> super::u16x32<T, M> {
        super::u16x32::from_repr_unchecked(self.1, crate::simd_storage::cast(self.0))
    }
}

// ============================================================================
// Widening (i16x32 -> i32x16)
// ============================================================================

impl<
    M: crate::simd::generic::ConstructorMode,
    T: crate::simd::backends::I16x32Backend + crate::simd::backends::I32x16Backend,
> i16x32<T, M>
{
    /// Sign-extend the low half of the lanes to `i32x16`.
    ///
    /// Result lane `i` is `self[i] as i32` for `i` in `0..16`.
    /// Natural lane order on every backend. Instruction count depends
    /// on the ISA, vector width, and surrounding loads.
    #[inline(always)]
    pub fn widen_low(self) -> super::i32x16<T, M> {
        super::i32x16::from_repr_unchecked(
            self.1,
            <T as crate::simd::backends::I16x32Backend>::widen_low_i16_to_i32(self.1, self.0),
        )
    }

    /// Sign-extend the high half of the lanes to `i32x16`.
    ///
    /// Result lane `i` is `self[i + 16] as i32`.
    #[inline(always)]
    pub fn widen_high(self) -> super::i32x16<T, M> {
        super::i32x16::from_repr_unchecked(
            self.1,
            <T as crate::simd::backends::I16x32Backend>::widen_high_i16_to_i32(self.1, self.0),
        )
    }
}

// ============================================================================
// Saturating narrowing (i16x32 -> i8x64 / u8x64)
// ============================================================================

impl<M: crate::simd::generic::ConstructorMode, T: crate::simd::backends::I16x32Backend>
    i16x32<T, M>
{
    /// Narrow `self` and `high` to `i8x64`, clamping each lane to
    /// the `i8` range.
    ///
    /// Result lane `i` is `self[i]` clamped for `i < 32`, and
    /// `high[i - 32]` clamped for `i >= 32` — the same lane order
    /// on every backend (the AVX2 arm pays one
    /// `permute4x64` to get there).
    #[inline(always)]
    pub fn narrow_saturating_i8(self, high: Self) -> super::i8x64<T, M>
    where
        T: crate::simd::backends::I8x64Backend,
    {
        super::i8x64::from_repr_unchecked(
            self.1,
            <T as crate::simd::backends::I16x32Backend>::narrow_saturating_i16_to_i8(
                self.1, self.0, high.0,
            ),
        )
    }

    /// Narrow `self` and `high` to `u8x64`, clamping each lane to
    /// the `u8` range.
    ///
    /// The source stays `i16` to match native signed-source packs on
    /// x86 and WASM. An unsigned-source operation would require a
    /// different lowering to preserve its full input range.
    #[inline(always)]
    pub fn narrow_saturating_u8(self, high: Self) -> super::u8x64<T, M>
    where
        T: crate::simd::backends::U8x64Backend,
    {
        super::u8x64::from_repr_unchecked(
            self.1,
            <T as crate::simd::backends::I16x32Backend>::narrow_saturating_i16_to_u8(
                self.1, self.0, high.0,
            ),
        )
    }
}

impl<T: crate::simd::backends::I16x32Backend + crate::simd::backends::U16x32Backend> i16x32<T> {
    /// MIN.abs_diff(MAX) is u16::MAX.
    /// Exact lane-wise absolute difference, without saturation or wrapping.
    #[inline(always)]
    pub fn abs_diff(self, rhs: Self) -> super::u16x32<T> {
        super::u16x32::from_repr_unchecked(
            self.1,
            <T as crate::simd::backends::I16x32Backend>::abs_diff(self.1, self.0, rhs.0),
        )
    }
}
impl<T: crate::simd::backends::I16x32Backend + crate::simd::backends::I32x16Backend> i16x32<T> {
    /// Multiply signed lanes, then sum adjacent pairs into i32 lanes.
    /// Lane k uses exactly input lanes 2k and 2k+1, in that order.
    /// The sum wraps modulo 2^32: two MIN*MIN products yield i32::MIN.
    /// Neither this operation nor the WASM dot instruction saturates.
    #[inline(always)]
    pub fn madd_adjacent(self, rhs: Self) -> super::i32x16<T> {
        super::i32x16::from_repr_unchecked(
            self.1,
            <T as crate::simd::backends::I16x32Backend>::madd_adjacent(self.1, self.0, rhs.0),
        )
    }
}
// ============================================================================
// Platform-specific concrete impls
// ============================================================================

impl<M: crate::simd::generic::ConstructorMode> i16x32<archmage::ScalarToken, M> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "scalar::i16x32"
    }
}

#[cfg(target_arch = "x86_64")]
impl<M: crate::simd::generic::ConstructorMode> i16x32<archmage::X64V3Token, M> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "polyfill::v3_512::i16x32"
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl<M: crate::simd::generic::ConstructorMode> i16x32<archmage::X64V4Token, M> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "x86::v4::i16x32"
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl<M: crate::simd::generic::ConstructorMode> i16x32<archmage::X64V4xToken, M> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "x86::v4x::i16x32"
    }
}

#[cfg(target_arch = "aarch64")]
impl<M: crate::simd::generic::ConstructorMode> i16x32<archmage::NeonToken, M> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "polyfill::neon_512::i16x32"
    }
}

#[cfg(target_arch = "wasm32")]
impl<M: crate::simd::generic::ConstructorMode> i16x32<archmage::Wasm128Token, M> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "polyfill::wasm128_512::i16x32"
    }
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl<M: crate::simd::generic::ConstructorMode> i16x32<archmage::X64V4Token, M> {
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
    #[archmage::rite(v4)]
    pub fn from_raw(value: core::arch::x86_64::__m512i) -> Self {
        Self::new_repr(value, archmage::X64V4Token::from_context())
    }
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl<M: crate::simd::generic::ConstructorMode> i16x32<archmage::X64V4xToken, M> {
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
    #[archmage::rite(v4x)]
    pub fn from_raw(value: core::arch::x86_64::__m512i) -> Self {
        Self::new_repr(value, archmage::X64V4xToken::from_context())
    }
}

// ============================================================================
// Extension: popcnt (requires Modern token)
// ============================================================================

#[cfg(feature = "avx512")]
impl<M: crate::simd::generic::ConstructorMode, T: crate::simd::backends::i16x32PopcntBackend>
    i16x32<T, M>
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
impl<T: I16x32Backend> From<i16x32<T, crate::simd::generic::Explicit>>
    for i16x32<T, crate::simd::generic::Context>
{
    #[inline(always)]
    fn from(value: i16x32<T, crate::simd::generic::Explicit>) -> Self {
        Self::new_repr(value.0, value.1)
    }
}
impl<T: I16x32Backend> From<i16x32<T, crate::simd::generic::Context>>
    for i16x32<T, crate::simd::generic::Explicit>
{
    #[inline(always)]
    fn from(value: i16x32<T, crate::simd::generic::Context>) -> Self {
        Self::new_repr(value.0, value.1)
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl i16x32<archmage::X64V4Token, crate::simd::generic::Context> {
    /// Wrap a raw `__m512i` using an existing CPU capability token.
    #[forbid(unsafe_code)]
    #[archmage::rite(v4)]
    pub fn from_m512i(value: core::arch::x86_64::__m512i) -> Self {
        Self::from_m512i_with_token(archmage::X64V4Token::from_context(), value)
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl i16x32<archmage::X64V4xToken, crate::simd::generic::Context> {
    /// Wrap a raw `__m512i` using an existing CPU capability token.
    #[forbid(unsafe_code)]
    #[archmage::rite(v4x)]
    pub fn from_m512i(value: core::arch::x86_64::__m512i) -> Self {
        Self::from_m512i_with_token(archmage::X64V4xToken::from_context(), value)
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl i16x32<archmage::X64V4Token, crate::simd::generic::Explicit> {
    /// Wrap a raw `__m512i` using an existing CPU capability token.
    #[inline(always)]
    pub fn from_m512i(token: archmage::X64V4Token, value: core::arch::x86_64::__m512i) -> Self {
        Self::from_m512i_with_token(token, value)
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl i16x32<archmage::X64V4xToken, crate::simd::generic::Explicit> {
    /// Wrap a raw `__m512i` using an existing CPU capability token.
    #[inline(always)]
    pub fn from_m512i(token: archmage::X64V4xToken, value: core::arch::x86_64::__m512i) -> Self {
        Self::from_m512i_with_token(token, value)
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl i16x32<archmage::X64V4Token, crate::simd::generic::Context> {
    /// Broadcast scalar to all 32 lanes.
    #[forbid(unsafe_code)]
    #[archmage::rite(v4)]
    pub fn splat(v: i16) -> Self {
        Self::splat_with_token(archmage::X64V4Token::from_context(), v)
    }
    /// All lanes zero.
    #[forbid(unsafe_code)]
    #[archmage::rite(v4)]
    pub fn zero() -> Self {
        Self::zero_with_token(archmage::X64V4Token::from_context())
    }
    /// Load from a `[i16; 32]` array.
    #[forbid(unsafe_code)]
    #[archmage::rite(v4)]
    pub fn load(data: &[i16; 32]) -> Self {
        Self::load_with_token(archmage::X64V4Token::from_context(), data)
    }
    /// Create from array (zero-cost where possible).
    #[forbid(unsafe_code)]
    #[archmage::rite(v4)]
    pub fn from_array(arr: [i16; 32]) -> Self {
        Self::from_array_with_token(archmage::X64V4Token::from_context(), arr)
    }
    /// Create from slice. Panics if `slice.len() < 32`.
    #[forbid(unsafe_code)]
    #[archmage::rite(v4)]
    pub fn from_slice(slice: &[i16]) -> Self {
        Self::from_slice_with_token(archmage::X64V4Token::from_context(), slice)
    }
    /// Split a slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&[[i16; 32]], &[i16])` — fixed-size arrays suitable
    /// for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    #[archmage::rite(v4)]
    pub fn partition_slice(data: &[i16]) -> (&[[i16; 32]], &[i16]) {
        Self::partition_slice_with_token(archmage::X64V4Token::from_context(), data)
    }
    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&mut [[i16; 32]], &mut [i16])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    #[archmage::rite(v4)]
    pub fn partition_slice_mut(data: &mut [i16]) -> (&mut [[i16; 32]], &mut [i16]) {
        Self::partition_slice_mut_with_token(archmage::X64V4Token::from_context(), data)
    }
    /// Wrap a platform representation (token-gated).
    #[forbid(unsafe_code)]
    #[archmage::rite(v4)]
    pub fn from_repr(
        repr: <archmage::X64V4Token as crate::simd::backends::I16x32Backend>::Repr,
    ) -> Self {
        Self::from_repr_with_token(archmage::X64V4Token::from_context(), repr)
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl i16x32<archmage::X64V4xToken, crate::simd::generic::Context> {
    /// Broadcast scalar to all 32 lanes.
    #[forbid(unsafe_code)]
    #[archmage::rite(v4x)]
    pub fn splat(v: i16) -> Self {
        Self::splat_with_token(archmage::X64V4xToken::from_context(), v)
    }
    /// All lanes zero.
    #[forbid(unsafe_code)]
    #[archmage::rite(v4x)]
    pub fn zero() -> Self {
        Self::zero_with_token(archmage::X64V4xToken::from_context())
    }
    /// Load from a `[i16; 32]` array.
    #[forbid(unsafe_code)]
    #[archmage::rite(v4x)]
    pub fn load(data: &[i16; 32]) -> Self {
        Self::load_with_token(archmage::X64V4xToken::from_context(), data)
    }
    /// Create from array (zero-cost where possible).
    #[forbid(unsafe_code)]
    #[archmage::rite(v4x)]
    pub fn from_array(arr: [i16; 32]) -> Self {
        Self::from_array_with_token(archmage::X64V4xToken::from_context(), arr)
    }
    /// Create from slice. Panics if `slice.len() < 32`.
    #[forbid(unsafe_code)]
    #[archmage::rite(v4x)]
    pub fn from_slice(slice: &[i16]) -> Self {
        Self::from_slice_with_token(archmage::X64V4xToken::from_context(), slice)
    }
    /// Split a slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&[[i16; 32]], &[i16])` — fixed-size arrays suitable
    /// for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    #[archmage::rite(v4x)]
    pub fn partition_slice(data: &[i16]) -> (&[[i16; 32]], &[i16]) {
        Self::partition_slice_with_token(archmage::X64V4xToken::from_context(), data)
    }
    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&mut [[i16; 32]], &mut [i16])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    #[archmage::rite(v4x)]
    pub fn partition_slice_mut(data: &mut [i16]) -> (&mut [[i16; 32]], &mut [i16]) {
        Self::partition_slice_mut_with_token(archmage::X64V4xToken::from_context(), data)
    }
    /// Wrap a platform representation (token-gated).
    #[forbid(unsafe_code)]
    #[archmage::rite(v4x)]
    pub fn from_repr(
        repr: <archmage::X64V4xToken as crate::simd::backends::I16x32Backend>::Repr,
    ) -> Self {
        Self::from_repr_with_token(archmage::X64V4xToken::from_context(), repr)
    }
}

#[cfg(target_arch = "aarch64")]
impl i16x32<archmage::NeonToken, crate::simd::generic::Context> {
    /// Broadcast scalar to all 32 lanes.
    #[forbid(unsafe_code)]
    #[archmage::rite(neon)]
    pub fn splat(v: i16) -> Self {
        Self::splat_with_token(archmage::NeonToken::from_context(), v)
    }
    /// All lanes zero.
    #[forbid(unsafe_code)]
    #[archmage::rite(neon)]
    pub fn zero() -> Self {
        Self::zero_with_token(archmage::NeonToken::from_context())
    }
    /// Load from a `[i16; 32]` array.
    #[forbid(unsafe_code)]
    #[archmage::rite(neon)]
    pub fn load(data: &[i16; 32]) -> Self {
        Self::load_with_token(archmage::NeonToken::from_context(), data)
    }
    /// Create from array (zero-cost where possible).
    #[forbid(unsafe_code)]
    #[archmage::rite(neon)]
    pub fn from_array(arr: [i16; 32]) -> Self {
        Self::from_array_with_token(archmage::NeonToken::from_context(), arr)
    }
    /// Create from slice. Panics if `slice.len() < 32`.
    #[forbid(unsafe_code)]
    #[archmage::rite(neon)]
    pub fn from_slice(slice: &[i16]) -> Self {
        Self::from_slice_with_token(archmage::NeonToken::from_context(), slice)
    }
    /// Split a slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&[[i16; 32]], &[i16])` — fixed-size arrays suitable
    /// for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    #[archmage::rite(neon)]
    pub fn partition_slice(data: &[i16]) -> (&[[i16; 32]], &[i16]) {
        Self::partition_slice_with_token(archmage::NeonToken::from_context(), data)
    }
    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&mut [[i16; 32]], &mut [i16])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    #[archmage::rite(neon)]
    pub fn partition_slice_mut(data: &mut [i16]) -> (&mut [[i16; 32]], &mut [i16]) {
        Self::partition_slice_mut_with_token(archmage::NeonToken::from_context(), data)
    }
    /// Wrap a platform representation (token-gated).
    #[forbid(unsafe_code)]
    #[archmage::rite(neon)]
    pub fn from_repr(
        repr: <archmage::NeonToken as crate::simd::backends::I16x32Backend>::Repr,
    ) -> Self {
        Self::from_repr_with_token(archmage::NeonToken::from_context(), repr)
    }
}

#[cfg(target_arch = "wasm32")]
impl i16x32<archmage::Wasm128Token, crate::simd::generic::Context> {
    /// Broadcast scalar to all 32 lanes.
    #[forbid(unsafe_code)]
    #[archmage::rite(wasm128)]
    pub fn splat(v: i16) -> Self {
        Self::splat_with_token(archmage::Wasm128Token::from_context(), v)
    }
    /// All lanes zero.
    #[forbid(unsafe_code)]
    #[archmage::rite(wasm128)]
    pub fn zero() -> Self {
        Self::zero_with_token(archmage::Wasm128Token::from_context())
    }
    /// Load from a `[i16; 32]` array.
    #[forbid(unsafe_code)]
    #[archmage::rite(wasm128)]
    pub fn load(data: &[i16; 32]) -> Self {
        Self::load_with_token(archmage::Wasm128Token::from_context(), data)
    }
    /// Create from array (zero-cost where possible).
    #[forbid(unsafe_code)]
    #[archmage::rite(wasm128)]
    pub fn from_array(arr: [i16; 32]) -> Self {
        Self::from_array_with_token(archmage::Wasm128Token::from_context(), arr)
    }
    /// Create from slice. Panics if `slice.len() < 32`.
    #[forbid(unsafe_code)]
    #[archmage::rite(wasm128)]
    pub fn from_slice(slice: &[i16]) -> Self {
        Self::from_slice_with_token(archmage::Wasm128Token::from_context(), slice)
    }
    /// Split a slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&[[i16; 32]], &[i16])` — fixed-size arrays suitable
    /// for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    #[archmage::rite(wasm128)]
    pub fn partition_slice(data: &[i16]) -> (&[[i16; 32]], &[i16]) {
        Self::partition_slice_with_token(archmage::Wasm128Token::from_context(), data)
    }
    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&mut [[i16; 32]], &mut [i16])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    #[archmage::rite(wasm128)]
    pub fn partition_slice_mut(data: &mut [i16]) -> (&mut [[i16; 32]], &mut [i16]) {
        Self::partition_slice_mut_with_token(archmage::Wasm128Token::from_context(), data)
    }
    /// Wrap a platform representation (token-gated).
    #[forbid(unsafe_code)]
    #[archmage::rite(wasm128)]
    pub fn from_repr(
        repr: <archmage::Wasm128Token as crate::simd::backends::I16x32Backend>::Repr,
    ) -> Self {
        Self::from_repr_with_token(archmage::Wasm128Token::from_context(), repr)
    }
}

#[cfg(target_arch = "x86_64")]
impl i16x32<archmage::X64V3Token, crate::simd::generic::Context> {
    /// Broadcast scalar to all 32 lanes.
    #[forbid(unsafe_code)]
    #[archmage::rite(v3)]
    pub fn splat(v: i16) -> Self {
        Self::splat_with_token(archmage::X64V3Token::from_context(), v)
    }
    /// All lanes zero.
    #[forbid(unsafe_code)]
    #[archmage::rite(v3)]
    pub fn zero() -> Self {
        Self::zero_with_token(archmage::X64V3Token::from_context())
    }
    /// Load from a `[i16; 32]` array.
    #[forbid(unsafe_code)]
    #[archmage::rite(v3)]
    pub fn load(data: &[i16; 32]) -> Self {
        Self::load_with_token(archmage::X64V3Token::from_context(), data)
    }
    /// Create from array (zero-cost where possible).
    #[forbid(unsafe_code)]
    #[archmage::rite(v3)]
    pub fn from_array(arr: [i16; 32]) -> Self {
        Self::from_array_with_token(archmage::X64V3Token::from_context(), arr)
    }
    /// Create from slice. Panics if `slice.len() < 32`.
    #[forbid(unsafe_code)]
    #[archmage::rite(v3)]
    pub fn from_slice(slice: &[i16]) -> Self {
        Self::from_slice_with_token(archmage::X64V3Token::from_context(), slice)
    }
    /// Split a slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&[[i16; 32]], &[i16])` — fixed-size arrays suitable
    /// for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    #[archmage::rite(v3)]
    pub fn partition_slice(data: &[i16]) -> (&[[i16; 32]], &[i16]) {
        Self::partition_slice_with_token(archmage::X64V3Token::from_context(), data)
    }
    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&mut [[i16; 32]], &mut [i16])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    #[archmage::rite(v3)]
    pub fn partition_slice_mut(data: &mut [i16]) -> (&mut [[i16; 32]], &mut [i16]) {
        Self::partition_slice_mut_with_token(archmage::X64V3Token::from_context(), data)
    }
    /// Wrap a platform representation (token-gated).
    #[forbid(unsafe_code)]
    #[archmage::rite(v3)]
    pub fn from_repr(
        repr: <archmage::X64V3Token as crate::simd::backends::I16x32Backend>::Repr,
    ) -> Self {
        Self::from_repr_with_token(archmage::X64V3Token::from_context(), repr)
    }
}

impl<T: I16x32Backend> i16x32<T, crate::simd::generic::Explicit> {
    /// Broadcast scalar to all 32 lanes.
    #[inline(always)]
    pub fn splat(token: T, v: i16) -> Self {
        Self::splat_with_token(token, v)
    }
    /// All lanes zero.
    #[inline(always)]
    pub fn zero(token: T) -> Self {
        Self::zero_with_token(token)
    }
    /// Load from a `[i16; 32]` array.
    #[inline(always)]
    pub fn load(token: T, data: &[i16; 32]) -> Self {
        Self::load_with_token(token, data)
    }
    /// Create from array (zero-cost where possible).
    #[inline(always)]
    pub fn from_array(token: T, arr: [i16; 32]) -> Self {
        Self::from_array_with_token(token, arr)
    }
    /// Create from slice. Panics if `slice.len() < 32`.
    #[inline(always)]
    pub fn from_slice(token: T, slice: &[i16]) -> Self {
        Self::from_slice_with_token(token, slice)
    }
    /// Split a slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&[[i16; 32]], &[i16])` — fixed-size arrays suitable
    /// for [`load`](Self::load), plus any leftover elements.
    #[inline(always)]
    pub fn partition_slice(_token: T, data: &[i16]) -> (&[[i16; 32]], &[i16]) {
        Self::partition_slice_with_token(_token, data)
    }
    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&mut [[i16; 32]], &mut [i16])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[inline(always)]
    pub fn partition_slice_mut(_token: T, data: &mut [i16]) -> (&mut [[i16; 32]], &mut [i16]) {
        Self::partition_slice_mut_with_token(_token, data)
    }
    /// Wrap a platform representation (token-gated).
    #[inline(always)]
    pub fn from_repr(token: T, repr: T::Repr) -> Self {
        Self::from_repr_with_token(token, repr)
    }
}

impl i16x32<archmage::ScalarToken, crate::simd::generic::Context> {
    /// Broadcast scalar to all 32 lanes.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn splat(v: i16) -> Self {
        Self::splat_with_token(archmage::ScalarToken, v)
    }
    /// All lanes zero.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn zero() -> Self {
        Self::zero_with_token(archmage::ScalarToken)
    }
    /// Load from a `[i16; 32]` array.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn load(data: &[i16; 32]) -> Self {
        Self::load_with_token(archmage::ScalarToken, data)
    }
    /// Create from array (zero-cost where possible).
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn from_array(arr: [i16; 32]) -> Self {
        Self::from_array_with_token(archmage::ScalarToken, arr)
    }
    /// Create from slice. Panics if `slice.len() < 32`.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn from_slice(slice: &[i16]) -> Self {
        Self::from_slice_with_token(archmage::ScalarToken, slice)
    }
    /// Split a slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&[[i16; 32]], &[i16])` — fixed-size arrays suitable
    /// for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn partition_slice(data: &[i16]) -> (&[[i16; 32]], &[i16]) {
        Self::partition_slice_with_token(archmage::ScalarToken, data)
    }
    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    /// Returns `(&mut [[i16; 32]], &mut [i16])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn partition_slice_mut(data: &mut [i16]) -> (&mut [[i16; 32]], &mut [i16]) {
        Self::partition_slice_mut_with_token(archmage::ScalarToken, data)
    }
    /// Wrap a platform representation (token-gated).
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn from_repr(
        repr: <archmage::ScalarToken as crate::simd::backends::I16x32Backend>::Repr,
    ) -> Self {
        Self::from_repr_with_token(archmage::ScalarToken, repr)
    }
}

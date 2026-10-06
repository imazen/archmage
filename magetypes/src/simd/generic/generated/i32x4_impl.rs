//! Generic `i32x4<T>` — 4-lane i32 SIMD vector parameterized by backend.
//!
//! `T` is a token type (e.g., `X64V3Token`, `NeonToken`, `ScalarToken`)
//! that determines the platform-native representation and intrinsics used.
//! The struct delegates all operations to the [`I32x4Backend`] trait.

#![allow(clippy::should_implement_trait)]

use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Index,
    IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use crate::simd::backends::I32x4Backend;

/// 4-lane i32 SIMD vector, generic over backend `T`.
///
/// `T` is a token type that proves CPU support for the required SIMD features.
/// The inner representation is `T::Repr` (e.g., `__m128i` on x86, `int32x4_t` on ARM).
///
/// **The token is stored** (as a zero-sized field) so methods receiving
/// `self: i32x4<T>` can re-supply it to backend operations that
/// require a token value (e.g. `T::splat(token, v)`). This carries the
/// token-as-feature-proof guarantee through every method call without
/// runtime overhead — `T` is ZST, so `sizeof(i32x4<T>) == sizeof(T::Repr)`
/// and `align_of(i32x4<T>) == align_of(T::Repr)` under `#[repr(C)]`.
///
/// # Layout
///
/// `#[repr(C)]` with a ZST trailing field: `T::Repr` lives at offset 0
/// and `T` is a 0-byte tail. Bitcasts between `i32x4<T>` values of
/// different element-types are sound when the Repr types share a layout
/// (e.g. `__m128` and `__m128i` are both 16-byte aligned 128-bit values).
/// `#[repr(transparent)]` cannot be used because Rust cannot prove at
/// the struct definition site that a generic `T` is a 1-ZST.
///
/// Construction requires a token value to prove CPU support at runtime.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct i32x4<T: I32x4Backend>(pub(crate) T::Repr, pub(crate) T);
// The `unsafe impl` and the checks behind it live in `simd_storage`.
crate::simd_storage::impl_token_storage!(i32x4, I32x4Backend);

// Layout invariant: struct is `#[repr(C)]` with a trailing ZST `T`
// field, so `sizeof/alignof(i32x4<T>) == sizeof/alignof(T::Repr)`
// when `T` is a 1-ZST. Every archmage token currently satisfies this;
// if a future refactor adds a non-ZST field to a token, this const
// assert fires at compile time.
const _: () = {
    assert!(
        core::mem::size_of::<i32x4<archmage::ScalarToken>>()
            == core::mem::size_of::<
                <archmage::ScalarToken as crate::simd::backends::I32x4Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<i32x4<archmage::ScalarToken>>()
            == core::mem::align_of::<
                <archmage::ScalarToken as crate::simd::backends::I32x4Backend>::Repr,
            >()
    );
};

#[cfg(target_arch = "x86_64")]
const _: () = {
    assert!(
        core::mem::size_of::<i32x4<archmage::X64V3Token>>()
            == core::mem::size_of::<
                <archmage::X64V3Token as crate::simd::backends::I32x4Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<i32x4<archmage::X64V3Token>>()
            == core::mem::align_of::<
                <archmage::X64V3Token as crate::simd::backends::I32x4Backend>::Repr,
            >()
    );
};

#[cfg(target_arch = "aarch64")]
const _: () = {
    assert!(
        core::mem::size_of::<i32x4<archmage::NeonToken>>()
            == core::mem::size_of::<
                <archmage::NeonToken as crate::simd::backends::I32x4Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<i32x4<archmage::NeonToken>>()
            == core::mem::align_of::<
                <archmage::NeonToken as crate::simd::backends::I32x4Backend>::Repr,
            >()
    );
};

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
const _: () = {
    assert!(
        core::mem::size_of::<i32x4<archmage::Wasm128Token>>()
            == core::mem::size_of::<
                <archmage::Wasm128Token as crate::simd::backends::I32x4Backend>::Repr,
            >()
    );
    assert!(
        core::mem::align_of::<i32x4<archmage::Wasm128Token>>()
            == core::mem::align_of::<
                <archmage::Wasm128Token as crate::simd::backends::I32x4Backend>::Repr,
            >()
    );
};

impl<T: I32x4Backend> i32x4<T> {
    /// Number of i32 lanes.
    pub const LANES: usize = 4;

    // ====== Construction (token-gated) ======

    /// Broadcast scalar to all 4 lanes.
    #[inline(always)]
    pub fn splat_t(token: T, v: i32) -> Self {
        Self(T::splat(token, v), token)
    }

    /// All lanes zero.
    #[inline(always)]
    pub fn zero_t(token: T) -> Self {
        Self(T::zero(token), token)
    }

    /// Load from a `[i32; 4]` array.
    #[inline(always)]
    pub fn load_t(token: T, data: &[i32; 4]) -> Self {
        Self(T::load(token, data), token)
    }

    /// Create from array (zero-cost where possible).
    #[inline(always)]
    pub fn from_array_t(token: T, arr: [i32; 4]) -> Self {
        Self(T::from_array(token, arr), token)
    }

    /// Create from slice. Panics if `slice.len() < 4`.
    #[inline(always)]
    pub fn from_slice_t(token: T, slice: &[i32]) -> Self {
        let arr: [i32; 4] = slice[..4].try_into().unwrap();
        Self(T::from_array(token, arr), token)
    }

    /// Split a slice into SIMD-width chunks and a scalar remainder.
    ///
    /// Returns `(&[[i32; 4]], &[i32])` — fixed-size arrays suitable
    /// for [`load_t`](Self::load_t), plus any leftover elements.
    #[inline(always)]
    pub fn partition_slice_t(_: T, data: &[i32]) -> (&[[i32; 4]], &[i32]) {
        data.as_chunks::<4>()
    }

    /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
    ///
    /// Returns `(&mut [[i32; 4]], &mut [i32])` — the bulk portion reinterpreted
    /// as fixed-size arrays suitable for [`load_t`](Self::load_t), plus any leftover elements.
    #[inline(always)]
    pub fn partition_slice_mut_t(_: T, data: &mut [i32]) -> (&mut [[i32; 4]], &mut [i32]) {
        data.as_chunks_mut::<4>()
    }

    // ====== Accessors ======

    /// Store to array.
    #[inline(always)]
    pub fn store(self, out: &mut [i32; 4]) {
        T::store(self.1, self.0, out);
    }

    /// Convert to array.
    #[inline(always)]
    pub fn to_array(self) -> [i32; 4] {
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

    /// Sum all 4 lanes (wrapping).
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
        Self(T::shl_const::<N>(self.1, self.0), self.1)
    }

    /// Arithmetic shift right by constant (sign-extending).
    ///
    /// `N` must be in `0..=31` (`N == 0` is the identity shift);
    /// out-of-range `N` fails to compile, identically on every backend.
    #[inline(always)]
    pub fn shr_arithmetic_const<const N: i32>(self) -> Self {
        const { assert!(N >= 0 && N <= 31, "shift amount out of range") };
        Self(T::shr_arithmetic_const::<N>(self.1, self.0), self.1)
    }

    /// Logical shift right by constant (zero-filling).
    ///
    /// `N` must be in `0..=31` (`N == 0` is the identity shift);
    /// out-of-range `N` fails to compile, identically on every backend.
    #[inline(always)]
    pub fn shr_logical_const<const N: i32>(self) -> Self {
        const { assert!(N >= 0 && N <= 31, "shift amount out of range") };
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
        Self(T::shl_uniform(self.1, self.0, count), self.1)
    }

    /// Logical (zero-filling) shift right by a runtime `count`, applied
    /// identically to every lane.
    ///
    /// `count >= 32` yields all-zero lanes on every backend.
    #[inline(always)]
    pub fn shr_logical_uniform(self, count: u32) -> Self {
        Self(T::shr_logical_uniform(self.1, self.0, count), self.1)
    }

    /// Arithmetic (sign-filling) shift right by a runtime `count`,
    /// applied identically to every lane.
    ///
    /// `count >= 32` yields a sign fill (every lane becomes `0` or
    /// `-1`), equivalent to shifting by 31, on every backend.
    #[inline(always)]
    pub fn shr_arithmetic_uniform(self, count: u32) -> Self {
        Self(T::shr_arithmetic_uniform(self.1, self.0, count), self.1)
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

impl<T: I32x4Backend> Add for i32x4<T> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(T::add(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: I32x4Backend> Sub for i32x4<T> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(T::sub(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: I32x4Backend> Mul for i32x4<T> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(T::mul(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: I32x4Backend> Neg for i32x4<T> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(T::neg(self.1, self.0), self.1)
    }
}

impl<T: I32x4Backend> BitAnd for i32x4<T> {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(T::bitand(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: I32x4Backend> BitOr for i32x4<T> {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self(T::bitor(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: I32x4Backend> BitXor for i32x4<T> {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Self(T::bitxor(self.1, self.0, rhs.0), self.1)
    }
}

// ============================================================================
// Assign operators
// ============================================================================

impl<T: I32x4Backend> AddAssign for i32x4<T> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<T: I32x4Backend> SubAssign for i32x4<T> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<T: I32x4Backend> MulAssign for i32x4<T> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<T: I32x4Backend> BitAndAssign for i32x4<T> {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

impl<T: I32x4Backend> BitOrAssign for i32x4<T> {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl<T: I32x4Backend> BitXorAssign for i32x4<T> {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = *self ^ rhs;
    }
}

// ============================================================================
// Scalar broadcast operators (v + 2, v * 3, etc.)
// ============================================================================

impl<T: I32x4Backend> Add<i32> for i32x4<T> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: i32) -> Self {
        Self(T::add(self.1, self.0, T::splat(self.1, rhs)), self.1)
    }
}

impl<T: I32x4Backend> Sub<i32> for i32x4<T> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: i32) -> Self {
        Self(T::sub(self.1, self.0, T::splat(self.1, rhs)), self.1)
    }
}

impl<T: I32x4Backend> Mul<i32> for i32x4<T> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: i32) -> Self {
        Self(T::mul(self.1, self.0, T::splat(self.1, rhs)), self.1)
    }
}

// ============================================================================
// Index
// ============================================================================

impl<T: I32x4Backend> Index<usize> for i32x4<T> {
    type Output = i32;
    #[inline(always)]
    fn index(&self, i: usize) -> &i32 {
        &crate::simd_storage::view::<_, [i32; 4]>(&self.0)[i]
    }
}

impl<T: I32x4Backend> IndexMut<usize> for i32x4<T> {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut i32 {
        &mut crate::simd_storage::view_mut::<_, [i32; 4]>(&mut self.0)[i]
    }
}

// ============================================================================
// Conversions
// ============================================================================

impl<T: I32x4Backend> From<i32x4<T>> for [i32; 4] {
    #[inline(always)]
    fn from(v: i32x4<T>) -> [i32; 4] {
        T::to_array(v.1, v.0)
    }
}

// ============================================================================
// Debug
// ============================================================================

impl<T: I32x4Backend> core::fmt::Debug for i32x4<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let arr = T::to_array(self.1, self.0);
        f.debug_tuple("i32x4").field(&arr).finish()
    }
}

// ============================================================================
// Cross-type conversions (available when T implements conversion traits)
// ============================================================================

impl<T: crate::simd::backends::F32x4Convert> i32x4<T> {
    /// Bitcast to f32x4 (reinterpret bits, no conversion).
    #[inline(always)]
    pub fn bitcast_to_f32(self) -> super::f32x4<T> {
        super::f32x4::from_repr_unchecked(self.1, T::bitcast_i32_to_f32(self.1, self.0))
    }

    /// Convert to f32x4 (numeric conversion).
    #[inline(always)]
    pub fn to_f32(self) -> super::f32x4<T> {
        super::f32x4::from_repr_unchecked(self.1, T::convert_i32_to_f32(self.1, self.0))
    }

    // ====== Backward-compatible aliases (old generated API names) ======

    /// Alias for [`bitcast_to_f32`](Self::bitcast_to_f32).
    #[inline(always)]
    pub fn bitcast_f32x4(self) -> super::f32x4<T> {
        self.bitcast_to_f32()
    }

    /// Alias for [`to_f32`](Self::to_f32).
    #[inline(always)]
    pub fn to_f32x4(self) -> super::f32x4<T> {
        self.to_f32()
    }
}

// ============================================================================
// Saturating narrowing (i32x4 -> i16x8 / u16x8)
// ============================================================================

impl<T: crate::simd::backends::I32x4Backend> i32x4<T> {
    /// Narrow `self` and `high` to `i16x8`, clamping each lane to
    /// the `i16` range.
    ///
    /// Result lane `i` is `self[i]` clamped for `i < 4`, and
    /// `high[i - 4]` clamped for `i >= 4` — the same lane order
    /// on every backend (the AVX2 arm pays one
    /// `permute4x64` to get there).
    #[inline(always)]
    pub fn narrow_saturating_i16(self, high: Self) -> super::i16x8<T>
    where
        T: crate::simd::backends::I16x8Backend,
    {
        super::i16x8::from_repr_unchecked(
            self.1,
            <T as crate::simd::backends::I32x4Backend>::narrow_saturating_i32_to_i16(
                self.1, self.0, high.0,
            ),
        )
    }

    /// Narrow `self` and `high` to `u16x8`, clamping each lane to
    /// the `u16` range.
    ///
    /// The source stays `i32` to match native signed-source packs on
    /// x86 and WASM. An unsigned-source operation would require a
    /// different lowering to preserve its full input range.
    #[inline(always)]
    pub fn narrow_saturating_u16(self, high: Self) -> super::u16x8<T>
    where
        T: crate::simd::backends::U16x8Backend,
    {
        super::u16x8::from_repr_unchecked(
            self.1,
            <T as crate::simd::backends::I32x4Backend>::narrow_saturating_i32_to_u16(
                self.1, self.0, high.0,
            ),
        )
    }
}

// ============================================================================
// Platform-specific concrete impls
// ============================================================================

impl i32x4<archmage::ScalarToken> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "scalar::i32x4"
    }
}

#[cfg(target_arch = "x86_64")]
impl i32x4<archmage::X64V3Token> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "x86::v3::i32x4"
    }

    /// Get the raw `__m128i` value.
    #[inline(always)]
    pub fn raw(self) -> core::arch::x86_64::__m128i {
        self.0
    }

    /// Create from a raw `__m128i` (token-gated, zero-cost).
    #[inline(always)]
    pub fn from_m128i_t(token: archmage::X64V3Token, v: core::arch::x86_64::__m128i) -> Self {
        Self(v, token)
    }
}

#[cfg(target_arch = "aarch64")]
impl i32x4<archmage::NeonToken> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "arm::neon::i32x4"
    }
}

#[cfg(target_arch = "wasm32")]
impl i32x4<archmage::Wasm128Token> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "wasm::wasm128::i32x4"
    }
}
#[cfg(target_arch = "x86_64")]
impl i32x4<archmage::X64V3Token> {
    /// Wrap a raw `__m128i` using an explicit CPU capability token.
    ///
    /// The caller does not need a target-feature annotation.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn from_raw_t(token: archmage::X64V3Token, value: core::arch::x86_64::__m128i) -> Self {
        Self(value, token)
    }

    /// Wrap a raw `__m128i` in a matching target-feature context.
    ///
    /// Rust requires the caller to enable the `v3` tier's features.
    /// Use an archmage `#[rite(v3)]` helper or `#[arcane]` entry point.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    pub fn from_raw(value: core::arch::x86_64::__m128i) -> Self {
        Self(value, archmage::X64V3Token::from_context())
    }
}
#[cfg(target_arch = "aarch64")]
impl i32x4<archmage::NeonToken> {
    /// Get the raw `int32x4_t` value.
    #[inline(always)]
    pub fn raw(self) -> core::arch::aarch64::int32x4_t {
        self.0
    }

    /// Wrap a raw `int32x4_t` using an existing CPU capability token.
    #[inline(always)]
    pub fn from_int32x4_t(
        token: archmage::NeonToken,
        value: core::arch::aarch64::int32x4_t,
    ) -> Self {
        Self(value, token)
    }

    /// Wrap a raw `int32x4_t` using an explicit CPU capability token.
    ///
    /// The caller does not need a target-feature annotation.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn from_raw_t(token: archmage::NeonToken, value: core::arch::aarch64::int32x4_t) -> Self {
        Self(value, token)
    }

    /// Wrap a raw `int32x4_t` in a matching target-feature context.
    ///
    /// Rust requires the caller to enable the `neon` tier's features.
    /// Use an archmage `#[rite(neon)]` helper or `#[arcane]` entry point.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger feature context, which Rust checks.
    #[target_feature(enable = "neon")]
    #[inline]
    pub fn from_raw(value: core::arch::aarch64::int32x4_t) -> Self {
        Self(value, archmage::NeonToken::from_context())
    }
}
#[cfg(target_arch = "wasm32")]
impl i32x4<archmage::Wasm128Token> {
    /// Get the raw `v128` value.
    #[inline(always)]
    pub fn raw(self) -> core::arch::wasm32::v128 {
        self.0
    }

    /// Wrap a raw `v128` using an existing CPU capability token.
    #[inline(always)]
    pub fn from_v128_t(token: archmage::Wasm128Token, value: core::arch::wasm32::v128) -> Self {
        Self(value, token)
    }

    /// Wrap a raw `v128` using an explicit CPU capability token.
    ///
    /// The caller does not need a target-feature annotation.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn from_raw_t(token: archmage::Wasm128Token, value: core::arch::wasm32::v128) -> Self {
        Self(value, token)
    }

    /// Wrap a raw `v128` in a matching target-feature context.
    ///
    /// Rust requires the caller to enable the `wasm128` tier's features.
    /// Use an archmage `#[rite(wasm128)]` helper or `#[arcane]` entry point.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger feature context, which Rust checks.
    #[target_feature(enable = "simd128")]
    #[inline]
    pub fn from_raw(value: core::arch::wasm32::v128) -> Self {
        Self(value, archmage::Wasm128Token::from_context())
    }
}
// Generated deprecated token-constructor forwarders. Do not edit.
impl<T: I32x4Backend> i32x4<T> {
    #[inline(always)]
    #[doc = "Deprecated token-taking spelling of [`Self::splat_t`].\n\nUse `splat_t` to keep explicit-token construction when `splat` becomes tokenless in magetypes 0.10."]
    #[deprecated(note = "Use splat_t(token, v); splat becomes tokenless in magetypes 0.10.")]
    #[forbid(unsafe_code)]
    pub fn splat(token: T, v: i32) -> Self {
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
    pub fn load(token: T, data: &[i32; 4]) -> Self {
        Self::load_t(token, data)
    }
    #[inline(always)]
    #[doc = "Deprecated token-taking spelling of [`Self::from_array_t`].\n\nUse `from_array_t` to keep explicit-token construction when `from_array` becomes tokenless in magetypes 0.10."]
    #[deprecated(
        note = "Use from_array_t(token, arr); from_array becomes tokenless in magetypes 0.10."
    )]
    #[forbid(unsafe_code)]
    pub fn from_array(token: T, arr: [i32; 4]) -> Self {
        Self::from_array_t(token, arr)
    }
    #[inline(always)]
    #[doc = "Deprecated token-taking spelling of [`Self::from_slice_t`].\n\nUse `from_slice_t` to keep explicit-token construction when `from_slice` becomes tokenless in magetypes 0.10."]
    #[deprecated(
        note = "Use from_slice_t(token, slice); from_slice becomes tokenless in magetypes 0.10."
    )]
    #[forbid(unsafe_code)]
    pub fn from_slice(token: T, slice: &[i32]) -> Self {
        Self::from_slice_t(token, slice)
    }
    #[inline(always)]
    #[doc = "Deprecated token-taking spelling of [`Self::partition_slice_t`].\n\nUse `partition_slice_t` to keep explicit-token construction when `partition_slice` becomes tokenless in magetypes 0.10."]
    #[deprecated(
        note = "Use partition_slice_t(token, data); partition_slice becomes tokenless in magetypes 0.10."
    )]
    #[forbid(unsafe_code)]
    pub fn partition_slice(token: T, data: &[i32]) -> (&[[i32; 4]], &[i32]) {
        Self::partition_slice_t(token, data)
    }
    #[inline(always)]
    #[doc = "Deprecated token-taking spelling of [`Self::partition_slice_mut_t`].\n\nUse `partition_slice_mut_t` to keep explicit-token construction when `partition_slice_mut` becomes tokenless in magetypes 0.10."]
    #[deprecated(
        note = "Use partition_slice_mut_t(token, data); partition_slice_mut becomes tokenless in magetypes 0.10."
    )]
    #[forbid(unsafe_code)]
    pub fn partition_slice_mut(token: T, data: &mut [i32]) -> (&mut [[i32; 4]], &mut [i32]) {
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
#[cfg(target_arch = "x86_64")]
impl i32x4<archmage::X64V3Token> {
    #[inline(always)]
    #[doc = "Deprecated token-taking spelling of [`Self::from_m128i_t`].\n\nUse `from_m128i_t` to keep explicit-token construction when `from_m128i` becomes tokenless in magetypes 0.10."]
    #[deprecated(
        note = "Use from_m128i_t(token, v); from_m128i becomes tokenless in magetypes 0.10."
    )]
    #[forbid(unsafe_code)]
    pub fn from_m128i(token: archmage::X64V3Token, v: core::arch::x86_64::__m128i) -> Self {
        Self::from_m128i_t(token, v)
    }
}
#[cfg(target_arch = "wasm32")]
impl i32x4<archmage::Wasm128Token> {
    #[inline(always)]
    #[doc = "Deprecated token-taking spelling of [`Self::from_v128_t`].\n\nUse `from_v128_t` to keep explicit-token construction when `from_v128` becomes tokenless in magetypes 0.10."]
    #[deprecated(
        note = "Use from_v128_t(token, value); from_v128 becomes tokenless in magetypes 0.10."
    )]
    #[forbid(unsafe_code)]
    pub fn from_v128(token: archmage::Wasm128Token, value: core::arch::wasm32::v128) -> Self {
        Self::from_v128_t(token, value)
    }
}

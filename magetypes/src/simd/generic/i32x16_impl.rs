//! Generic `i32x16<T>` — 16-lane i32 SIMD vector parameterized by backend.
//!
//! `T` is a token type (e.g., `X64V4Token`, `ScalarToken`)
//! that determines the platform-native representation and intrinsics used.
//! The struct delegates all operations to the [`I32x16Backend`] trait.

#![allow(clippy::should_implement_trait)]

use core::marker::PhantomData;
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
/// Construction requires a token value to prove CPU support at runtime.
/// After construction, operations don't need the token — it's baked into the type.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct i32x16<T: I32x16Backend>(T::Repr, PhantomData<T>);

impl<T: I32x16Backend> i32x16<T> {
    /// Number of i32 lanes.
    pub const LANES: usize = 16;

    // ====== Construction (token-gated) ======

    /// Broadcast scalar to all 16 lanes.
    #[inline(always)]
    pub fn splat(_: T, v: i32) -> Self {
        Self(T::splat(v), PhantomData)
    }

    /// All lanes zero.
    #[inline(always)]
    pub fn zero(_: T) -> Self {
        Self(T::zero(), PhantomData)
    }

    /// Load from a `[i32; 16]` array.
    #[inline(always)]
    pub fn load(_: T, data: &[i32; 16]) -> Self {
        Self(T::load(data), PhantomData)
    }

    /// Create from array (zero-cost where possible).
    #[inline(always)]
    pub fn from_array(_: T, arr: [i32; 16]) -> Self {
        Self(T::from_array(arr), PhantomData)
    }

    /// Create from slice. Panics if `slice.len() < 16`.
    #[inline(always)]
    pub fn from_slice(_: T, slice: &[i32]) -> Self {
        let arr: [i32; 16] = slice[..16].try_into().unwrap();
        Self(T::from_array(arr), PhantomData)
    }

    // ====== Accessors ======

    /// Store to array.
    #[inline(always)]
    pub fn store(self, out: &mut [i32; 16]) {
        T::store(self.0, out);
    }

    /// Convert to array.
    #[inline(always)]
    pub fn to_array(self) -> [i32; 16] {
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

    /// Lane-wise minimum.
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(T::min(self.0, other.0), PhantomData)
    }

    /// Lane-wise maximum.
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(T::max(self.0, other.0), PhantomData)
    }

    /// Lane-wise absolute value.
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(T::abs(self.0), PhantomData)
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

    /// Lane-wise less-than (returns mask).
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> Self {
        Self(T::simd_lt(self.0, other.0), PhantomData)
    }

    /// Lane-wise less-than-or-equal (returns mask).
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> Self {
        Self(T::simd_le(self.0, other.0), PhantomData)
    }

    /// Lane-wise greater-than (returns mask).
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> Self {
        Self(T::simd_gt(self.0, other.0), PhantomData)
    }

    /// Lane-wise greater-than-or-equal (returns mask).
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

    /// Sum all 16 lanes.
    #[inline(always)]
    pub fn reduce_add(self) -> i32 {
        T::reduce_add(self.0)
    }

    // ====== Shifts ======

    /// Shift left by constant.
    #[inline(always)]
    pub fn shl_const<const N: i32>(self) -> Self {
        Self(T::shl_const::<N>(self.0), PhantomData)
    }

    /// Arithmetic shift right by constant (sign-extending).
    #[inline(always)]
    pub fn shr_arithmetic_const<const N: i32>(self) -> Self {
        Self(T::shr_arithmetic_const::<N>(self.0), PhantomData)
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

    // ====== Bitwise ======

    /// Bitwise NOT.
    #[inline(always)]
    pub fn not(self) -> Self {
        Self(T::not(self.0), PhantomData)
    }

    // ====== Boolean ======

    /// True if all lanes have their sign bit set (all-1s mask).
    #[inline(always)]
    pub fn all_true(self) -> bool {
        T::all_true(self.0)
    }

    /// True if any lane has its sign bit set.
    #[inline(always)]
    pub fn any_true(self) -> bool {
        T::any_true(self.0)
    }

    /// Extract the high bit of each 32-bit lane as a bitmask.
    #[inline(always)]
    pub fn bitmask(self) -> u64 {
        T::bitmask(self.0)
    }
}

// ============================================================================
// Operator implementations
// ============================================================================

impl<T: I32x16Backend> Add for i32x16<T> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(T::add(self.0, rhs.0), PhantomData)
    }
}

impl<T: I32x16Backend> Sub for i32x16<T> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(T::sub(self.0, rhs.0), PhantomData)
    }
}

impl<T: I32x16Backend> Mul for i32x16<T> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(T::mul(self.0, rhs.0), PhantomData)
    }
}

impl<T: I32x16Backend> Neg for i32x16<T> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(T::neg(self.0), PhantomData)
    }
}

impl<T: I32x16Backend> BitAnd for i32x16<T> {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(T::bitand(self.0, rhs.0), PhantomData)
    }
}

impl<T: I32x16Backend> BitOr for i32x16<T> {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self(T::bitor(self.0, rhs.0), PhantomData)
    }
}

impl<T: I32x16Backend> BitXor for i32x16<T> {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Self(T::bitxor(self.0, rhs.0), PhantomData)
    }
}

// ============================================================================
// Assign operators
// ============================================================================

impl<T: I32x16Backend> AddAssign for i32x16<T> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<T: I32x16Backend> SubAssign for i32x16<T> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<T: I32x16Backend> MulAssign for i32x16<T> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<T: I32x16Backend> BitAndAssign for i32x16<T> {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

impl<T: I32x16Backend> BitOrAssign for i32x16<T> {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl<T: I32x16Backend> BitXorAssign for i32x16<T> {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = *self ^ rhs;
    }
}

// ============================================================================
// Scalar broadcast operators (v + 2, v * 3, etc.)
// ============================================================================

impl<T: I32x16Backend> Add<i32> for i32x16<T> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: i32) -> Self {
        Self(T::add(self.0, T::splat(rhs)), PhantomData)
    }
}

impl<T: I32x16Backend> Sub<i32> for i32x16<T> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: i32) -> Self {
        Self(T::sub(self.0, T::splat(rhs)), PhantomData)
    }
}

impl<T: I32x16Backend> Mul<i32> for i32x16<T> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: i32) -> Self {
        Self(T::mul(self.0, T::splat(rhs)), PhantomData)
    }
}

// ============================================================================
// Index
// ============================================================================

impl<T: I32x16Backend> Index<usize> for i32x16<T> {
    type Output = i32;
    #[inline(always)]
    fn index(&self, i: usize) -> &i32 {
        assert!(i < 16, "i32x16 index out of bounds: {i}");
        // SAFETY: i32x16's repr is layout-compatible with [i32; 16], and i < 16.
        unsafe { &*(core::ptr::from_ref(self).cast::<i32>()).add(i) }
    }
}

impl<T: I32x16Backend> IndexMut<usize> for i32x16<T> {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut i32 {
        assert!(i < 16, "i32x16 index out of bounds: {i}");
        // SAFETY: i32x16's repr is layout-compatible with [i32; 16], and i < 16.
        unsafe { &mut *(core::ptr::from_mut(self).cast::<i32>()).add(i) }
    }
}

// ============================================================================
// Conversions
// ============================================================================

impl<T: I32x16Backend> From<i32x16<T>> for [i32; 16] {
    #[inline(always)]
    fn from(v: i32x16<T>) -> [i32; 16] {
        T::to_array(v.0)
    }
}

// ============================================================================
// Debug
// ============================================================================

impl<T: I32x16Backend> core::fmt::Debug for i32x16<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let arr = T::to_array(self.0);
        f.debug_tuple("i32x16").field(&arr).finish()
    }
}

// ============================================================================
// Platform-specific implementation info
// ============================================================================

#[cfg(target_arch = "x86_64")]
impl i32x16<archmage::X64V3Token> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "polyfill::v3_512::i32x16"
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl i32x16<archmage::X64V4Token> {
    /// Implementation identifier for this backend.
    pub const fn implementation_name() -> &'static str {
        "x86::v4::i32x16"
    }
}

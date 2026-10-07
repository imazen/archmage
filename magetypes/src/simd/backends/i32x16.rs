//! Backend trait for `i32x16<T>` — 16-lane i32 SIMD vector.
//!
//! Each token type implements this trait with its platform-native representation.
//! The generic wrapper `i32x16<T>` delegates all operations to these trait methods.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

use super::sealed::Sealed;
use archmage::SimdToken;

/// Backend implementation for 16-lane i32 SIMD vectors.
///
/// Trait methods take `self` (the token) as receiver — the token value
/// is the proof of CPU support, and requiring it as the receiver means
/// the methods cannot be invoked via UFCS without holding one. The
/// implementing type `Self` (a token type) determines which platform
/// intrinsics are used. All methods are `#[inline(always)]` in
/// implementations.
///
/// # Sealed
///
/// This trait is sealed — only archmage token types can implement it.
/// The token proves CPU support was verified via `summon()`.
pub trait I32x16Backend: SimdToken + Sealed + Copy + 'static {
    /// Platform-native SIMD representation.
    #[allow(private_bounds)]
    type Repr: Copy + Clone + Send + Sync + crate::simd_storage::Pod;

    // ====== Construction ======

    /// Broadcast scalar to all 16 lanes.
    fn splat(self, v: i32) -> Self::Repr;

    /// All lanes zero.
    fn zero(self) -> Self::Repr;

    /// Load from an aligned array.
    fn load(self, data: &[i32; 16]) -> Self::Repr;

    /// Create from array (zero-cost transmute where possible).
    fn from_array(self, arr: [i32; 16]) -> Self::Repr;

    /// Store to array.
    fn store(self, repr: Self::Repr, out: &mut [i32; 16]);

    /// Convert to array.
    fn to_array(self, repr: Self::Repr) -> [i32; 16];

    // ====== Arithmetic ======

    /// Lane-wise addition.
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Lane-wise subtraction.
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Lane-wise multiplication (low bits of product).
    fn mul(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Lane-wise negation.
    fn neg(self, a: Self::Repr) -> Self::Repr;

    // ====== Math ======

    /// Lane-wise minimum.
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Lane-wise maximum.
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Lane-wise absolute value.
    fn abs(self, a: Self::Repr) -> Self::Repr;

    // ====== Comparisons ======
    // Return masks where each lane is all-1s (true) or all-0s (false).

    /// Lane-wise equality.
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Lane-wise inequality.
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Lane-wise less-than.
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Lane-wise less-than-or-equal.
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Lane-wise greater-than.
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Lane-wise greater-than-or-equal.
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Select lanes: where mask is all-1s pick `if_true`, else `if_false`.
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr;

    // ====== Reductions ======

    /// Sum all 16 lanes.
    fn reduce_add(self, a: Self::Repr) -> i32;

    // ====== Bitwise ======

    /// Bitwise NOT.
    fn not(self, a: Self::Repr) -> Self::Repr;

    /// Bitwise AND.
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Bitwise OR.
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Bitwise XOR.
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

    // ====== Shifts ======

    /// Shift left by constant.
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;

    /// Arithmetic shift right by constant (sign-extending).
    /// `N` must be in `0..=lane_bits-1`.
    fn shr_arithmetic_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;

    /// Logical shift right by constant (zero-filling).
    /// `N` must be in `0..=lane_bits-1`.
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    // ====== Uniform variable shifts ======

    /// Shift left by a runtime `count` applied identically to every lane.
    ///
    /// `count >= 32` produces all-zero lanes on every backend.
    fn shl_uniform(self, a: Self::Repr, count: u32) -> Self::Repr;

    /// Logical (zero-filling) shift right by a runtime `count` applied
    /// identically to every lane.
    ///
    /// `count >= 32` produces all-zero lanes on every backend.
    fn shr_logical_uniform(self, a: Self::Repr, count: u32) -> Self::Repr;

    /// Arithmetic (sign-filling) shift right by a runtime `count`
    /// applied identically to every lane. Identical to
    /// `shr_logical_uniform` for unsigned element types.
    ///
    /// `count >= 32` produces a sign fill on every backend.
    fn shr_arithmetic_uniform(self, a: Self::Repr, count: u32) -> Self::Repr;

    // ====== Boolean ======

    /// True if **every** lane has its sign bit set.
    ///
    /// This is a sign-bit test, not a "lane is nonzero" test, on
    /// every backend and at every width. A lane holding `1` is
    /// false; a lane holding `-1` (or `0x80` in its top bit) is
    /// true. Comparison results are all-ones or all-zeros per lane,
    /// so for masks — the intended input — the two readings agree
    /// and this is the cheap native reduction. They diverge only on
    /// hand-built vectors, which is where the backends used to
    /// disagree with each other.
    fn all_true(self, a: Self::Repr) -> bool;

    /// True if **any** lane has its sign bit set.
    ///
    /// Sign-bit test, not "lane is nonzero" — see `all_true`.
    fn any_true(self, a: Self::Repr) -> bool;

    /// Extract the high bit of each lane as a bitmask.
    fn bitmask(self, a: Self::Repr) -> u64;

    // ====== Default implementations ======

    /// Clamp values between lo and hi.
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <Self as I32x16Backend>::min(self, <Self as I32x16Backend>::max(self, a, lo), hi)
    }
    /// Clamp to i16's range, then concatenate a's 16 lanes followed by b's.
    fn narrow_saturating_i32_to_i16(
        self,
        a: <Self as super::I32x16Backend>::Repr,
        b: <Self as super::I32x16Backend>::Repr,
    ) -> <Self as super::I16x32Backend>::Repr
    where
        Self: super::I16x32Backend;
    /// Clamp to u16's range, then concatenate a's 16 lanes followed by b's.
    fn narrow_saturating_i32_to_u16(
        self,
        a: <Self as super::I32x16Backend>::Repr,
        b: <Self as super::I32x16Backend>::Repr,
    ) -> <Self as super::U16x32Backend>::Repr
    where
        Self: super::U16x32Backend;
}

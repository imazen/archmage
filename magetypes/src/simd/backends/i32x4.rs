//! Backend trait for `i32x4<T>` — 4-lane i32 SIMD vector.
//!
//! Each token type implements this trait with its platform-native representation.
//! The generic wrapper `i32x4<T>` delegates all operations to these trait methods.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

use super::sealed::Sealed;
use archmage::SimdToken;

/// Backend implementation for 4-lane i32 SIMD vectors.
///
/// Trait methods are **associated functions** (no `self`/token parameter).
/// The implementing type `Self` (a token type) determines which platform
/// intrinsics are used. All methods are `#[inline(always)]` in implementations.
///
/// # Sealed
///
/// This trait is sealed — only archmage token types can implement it.
/// The token proves CPU support was verified via `summon()`.
pub trait I32x4Backend: SimdToken + Sealed + Copy + 'static {
    /// Platform-native SIMD representation.
    type Repr: Copy + Clone + Send + Sync;

    // ====== Construction ======

    /// Broadcast scalar to all 4 lanes.
    fn splat(self, v: i32) -> Self::Repr;

    /// All lanes zero.
    fn zero(self) -> Self::Repr;

    /// Load from an aligned array.
    fn load(self, data: &[i32; 4]) -> Self::Repr;

    /// Create from array (zero-cost transmute where possible).
    fn from_array(self, arr: [i32; 4]) -> Self::Repr;

    /// Store to array.
    fn store(self, repr: Self::Repr, out: &mut [i32; 4]);

    /// Convert to array.
    fn to_array(self, repr: Self::Repr) -> [i32; 4];

    // ====== Arithmetic ======

    /// Lane-wise addition.
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Lane-wise subtraction.
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Lane-wise multiplication (low 32 bits of each 32x32 product).
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

    /// Sum all 4 lanes.
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
    fn shr_arithmetic_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;

    /// Logical shift right by constant (zero-filling).
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;

    // ====== Boolean ======

    /// True if all lanes have their sign bit set (all-1s mask).
    fn all_true(self, a: Self::Repr) -> bool;

    /// True if any lane has its sign bit set (any all-1s mask lane).
    fn any_true(self, a: Self::Repr) -> bool;

    /// Extract the high bit of each 32-bit lane as a bitmask.
    fn bitmask(self, a: Self::Repr) -> u32;

    // ====== Default implementations ======

    /// Clamp values between lo and hi.
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <Self as I32x4Backend>::min(self, <Self as I32x4Backend>::max(self, a, lo), hi)
    }
}

//! Backend trait for `u16x32<T>` — 32-lane u16 SIMD vector.
//!
//! Each token type implements this trait with its platform-native representation.
//! The generic wrapper `u16x32<T>` delegates all operations to these trait methods.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

use super::sealed::Sealed;
use archmage::SimdToken;

/// Backend implementation for 32-lane u16 SIMD vectors.
///
/// Trait methods are **associated functions** (no `self`/token parameter).
/// The implementing type `Self` (a token type) determines which platform
/// intrinsics are used. All methods are `#[inline(always)]` in implementations.
///
/// # Sealed
///
/// This trait is sealed — only archmage token types can implement it.
/// The token proves CPU support was verified via `summon()`.
pub trait U16x32Backend: SimdToken + Sealed + Copy + 'static {
    /// Platform-native SIMD representation.
    type Repr: Copy + Clone + Send + Sync;

    // ====== Construction ======

    /// Broadcast scalar to all 32 lanes.
    fn splat(v: u16) -> Self::Repr;

    /// All lanes zero.
    fn zero() -> Self::Repr;

    /// Load from an aligned array.
    fn load(data: &[u16; 32]) -> Self::Repr;

    /// Create from array (zero-cost transmute where possible).
    fn from_array(arr: [u16; 32]) -> Self::Repr;

    /// Store to array.
    fn store(repr: Self::Repr, out: &mut [u16; 32]);

    /// Convert to array.
    fn to_array(repr: Self::Repr) -> [u16; 32];

    // ====== Arithmetic ======

    /// Lane-wise addition.
    fn add(a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Lane-wise subtraction.
    fn sub(a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Lane-wise multiplication (low bits of product).
    fn mul(a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Lane-wise negation.
    fn neg(a: Self::Repr) -> Self::Repr;

    // ====== Math ======

    /// Lane-wise minimum.
    fn min(a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Lane-wise maximum.
    fn max(a: Self::Repr, b: Self::Repr) -> Self::Repr;

    // ====== Comparisons ======
    // Return masks where each lane is all-1s (true) or all-0s (false).

    /// Lane-wise equality.
    fn simd_eq(a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Lane-wise inequality.
    fn simd_ne(a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Lane-wise less-than.
    fn simd_lt(a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Lane-wise less-than-or-equal.
    fn simd_le(a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Lane-wise greater-than.
    fn simd_gt(a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Lane-wise greater-than-or-equal.
    fn simd_ge(a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Select lanes: where mask is all-1s pick `if_true`, else `if_false`.
    fn blend(mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr;

    // ====== Reductions ======

    /// Sum all 32 lanes.
    fn reduce_add(a: Self::Repr) -> u16;

    // ====== Bitwise ======

    /// Bitwise NOT.
    fn not(a: Self::Repr) -> Self::Repr;

    /// Bitwise AND.
    fn bitand(a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Bitwise OR.
    fn bitor(a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Bitwise XOR.
    fn bitxor(a: Self::Repr, b: Self::Repr) -> Self::Repr;

    // ====== Shifts ======

    /// Shift left by constant.
    fn shl_const<const N: i32>(a: Self::Repr) -> Self::Repr;

    /// Arithmetic shift right by constant (sign-extending).
    fn shr_arithmetic_const<const N: i32>(a: Self::Repr) -> Self::Repr;

    /// Logical shift right by constant (zero-filling).
    fn shr_logical_const<const N: i32>(a: Self::Repr) -> Self::Repr;

    // ====== Boolean ======

    /// True if all lanes have their sign bit set (all-1s mask).
    fn all_true(a: Self::Repr) -> bool;

    /// True if any lane has its sign bit set (any all-1s mask lane).
    fn any_true(a: Self::Repr) -> bool;

    /// Extract the high bit of each lane as a bitmask.
    fn bitmask(a: Self::Repr) -> u64;

    // ====== Default implementations ======

    /// Clamp values between lo and hi.
    #[inline(always)]
    fn clamp(a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        Self::min(Self::max(a, lo), hi)
    }
}

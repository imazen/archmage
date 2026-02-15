//! Backend trait for `f64x2<T>` — 2-lane f64 SIMD vector.
//!
//! Each token type implements this trait with its platform-native representation.
//! The generic wrapper `f64x2<T>` delegates all operations to these trait methods.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

use super::sealed::Sealed;
use archmage::SimdToken;

/// Backend implementation for 2-lane f64 SIMD vectors.
///
/// Trait methods are **associated functions** (no `self`/token parameter).
/// The implementing type `Self` (a token type) determines which platform
/// intrinsics are used. All methods are `#[inline(always)]` in implementations.
///
/// # Sealed
///
/// This trait is sealed — only archmage token types can implement it.
/// The token proves CPU support was verified via `summon()`.
pub trait F64x2Backend: SimdToken + Sealed + Copy + 'static {
    /// Platform-native SIMD representation.
    type Repr: Copy + Clone + Send + Sync;

    // ====== Construction ======

    /// Broadcast scalar to all 2 lanes.
    fn splat(v: f64) -> Self::Repr;

    /// All lanes zero.
    fn zero() -> Self::Repr;

    /// Load from an aligned array.
    fn load(data: &[f64; 2]) -> Self::Repr;

    /// Create from array (zero-cost transmute where possible).
    fn from_array(arr: [f64; 2]) -> Self::Repr;

    /// Store to array.
    fn store(repr: Self::Repr, out: &mut [f64; 2]);

    /// Convert to array.
    fn to_array(repr: Self::Repr) -> [f64; 2];

    // ====== Arithmetic ======

    /// Lane-wise addition.
    fn add(a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Lane-wise subtraction.
    fn sub(a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Lane-wise multiplication.
    fn mul(a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Lane-wise division.
    fn div(a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Lane-wise negation.
    fn neg(a: Self::Repr) -> Self::Repr;

    // ====== Math ======

    /// Lane-wise minimum.
    fn min(a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Lane-wise maximum.
    fn max(a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Square root.
    fn sqrt(a: Self::Repr) -> Self::Repr;

    /// Absolute value.
    fn abs(a: Self::Repr) -> Self::Repr;

    /// Round toward negative infinity.
    fn floor(a: Self::Repr) -> Self::Repr;

    /// Round toward positive infinity.
    fn ceil(a: Self::Repr) -> Self::Repr;

    /// Round to nearest integer.
    fn round(a: Self::Repr) -> Self::Repr;

    /// Fused multiply-add: a * b + c.
    fn mul_add(a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr;

    /// Fused multiply-sub: a * b - c.
    fn mul_sub(a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr;

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

    /// Sum all 2 lanes.
    fn reduce_add(a: Self::Repr) -> f64;

    /// Minimum across all 2 lanes.
    fn reduce_min(a: Self::Repr) -> f64;

    /// Maximum across all 2 lanes.
    fn reduce_max(a: Self::Repr) -> f64;

    // ====== Approximations ======

    /// Fast reciprocal approximation (~12-bit precision where available).
    ///
    /// On platforms without native approximation, falls back to full division.
    fn rcp_approx(a: Self::Repr) -> Self::Repr {
        Self::div(Self::splat(1.0), a)
    }

    /// Fast reciprocal square root approximation (~12-bit precision where available).
    ///
    /// On platforms without native approximation, falls back to 1/sqrt.
    fn rsqrt_approx(a: Self::Repr) -> Self::Repr {
        Self::div(Self::splat(1.0), Self::sqrt(a))
    }

    // ====== Bitwise ======

    /// Bitwise NOT.
    fn not(a: Self::Repr) -> Self::Repr;

    /// Bitwise AND.
    fn bitand(a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Bitwise OR.
    fn bitor(a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Bitwise XOR.
    fn bitxor(a: Self::Repr, b: Self::Repr) -> Self::Repr;

    // ====== Default implementations ======

    /// Clamp values between lo and hi.
    #[inline(always)]
    fn clamp(a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        Self::min(Self::max(a, lo), hi)
    }

    /// Precise reciprocal (Newton-Raphson from rcp_approx).
    #[inline(always)]
    fn recip(a: Self::Repr) -> Self::Repr {
        let approx = Self::rcp_approx(a);
        let two = Self::splat(2.0);
        // x' = x * (2 - a*x)
        Self::mul(approx, Self::sub(two, Self::mul(a, approx)))
    }

    /// Precise reciprocal square root (Newton-Raphson from rsqrt_approx).
    #[inline(always)]
    fn rsqrt(a: Self::Repr) -> Self::Repr {
        let approx = Self::rsqrt_approx(a);
        let half = Self::splat(0.5);
        let three = Self::splat(3.0);
        // y' = 0.5 * y * (3 - x * y * y)
        Self::mul(
            Self::mul(half, approx),
            Self::sub(three, Self::mul(a, Self::mul(approx, approx))),
        )
    }
}

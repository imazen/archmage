//! Backend trait for `f32x4<T>` — 4-lane f32 SIMD vector.
//!
//! Each token type implements this trait with its platform-native representation.
//! The generic wrapper `f32x4<T>` delegates all operations to these trait methods.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

use super::sealed::Sealed;
use archmage::SimdToken;

/// Backend implementation for 4-lane f32 SIMD vectors.
///
/// Trait methods are **associated functions** (no `self`/token parameter).
/// The implementing type `Self` (a token type) determines which platform
/// intrinsics are used. All methods are `#[inline(always)]` in implementations.
///
/// # Sealed
///
/// This trait is sealed — only archmage token types can implement it.
/// The token proves CPU support was verified via `summon()`.
pub trait F32x4Backend: SimdToken + Sealed + Copy + 'static {
    /// Platform-native SIMD representation.
    type Repr: Copy + Clone + Send + Sync;

    // ====== Construction ======

    /// Broadcast scalar to all 4 lanes.
    fn splat(self, v: f32) -> Self::Repr;

    /// All lanes zero.
    fn zero(self) -> Self::Repr;

    /// Load from an aligned array.
    fn load(self, data: &[f32; 4]) -> Self::Repr;

    /// Create from array (zero-cost transmute where possible).
    fn from_array(self, arr: [f32; 4]) -> Self::Repr;

    /// Store to array.
    fn store(self, repr: Self::Repr, out: &mut [f32; 4]);

    /// Convert to array.
    fn to_array(self, repr: Self::Repr) -> [f32; 4];

    // ====== Arithmetic ======

    /// Lane-wise addition.
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Lane-wise subtraction.
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Lane-wise multiplication.
    fn mul(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Lane-wise division.
    fn div(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Lane-wise negation.
    fn neg(self, a: Self::Repr) -> Self::Repr;

    // ====== Math ======

    /// Lane-wise minimum.
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Lane-wise maximum.
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Square root.
    fn sqrt(self, a: Self::Repr) -> Self::Repr;

    /// Absolute value.
    fn abs(self, a: Self::Repr) -> Self::Repr;

    /// Round toward negative infinity.
    fn floor(self, a: Self::Repr) -> Self::Repr;

    /// Round toward positive infinity.
    fn ceil(self, a: Self::Repr) -> Self::Repr;

    /// Round to nearest integer.
    fn round(self, a: Self::Repr) -> Self::Repr;

    /// Fused multiply-add: a * b + c.
    fn mul_add(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr;

    /// Fused multiply-sub: a * b - c.
    fn mul_sub(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr;

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
    fn reduce_add(self, a: Self::Repr) -> f32;

    /// Minimum across all 4 lanes.
    fn reduce_min(self, a: Self::Repr) -> f32;

    /// Maximum across all 4 lanes.
    fn reduce_max(self, a: Self::Repr) -> f32;

    // ====== Approximations ======

    /// Fast reciprocal approximation (~12-bit precision where available).
    ///
    /// **Default body returns the input unchanged** — every shipped
    /// backend overrides this with a native intrinsic. The original
    /// default `Self::div(Self::splat(1.0), a)` would require `splat`
    /// to be tokenless; with the soundness fix on `splat`, the default
    /// can no longer construct a `1.0` constant. New backends MUST
    /// override.
    #[inline(always)]
    fn rcp_approx(self, a: Self::Repr) -> Self::Repr {
        a
    }

    /// Fast reciprocal square root approximation (~12-bit precision where available).
    ///
    /// See [`rcp_approx`] for default-body rationale.
    #[inline(always)]
    fn rsqrt_approx(self, a: Self::Repr) -> Self::Repr {
        a
    }

    // ====== Bitwise ======

    /// Bitwise NOT.
    fn not(self, a: Self::Repr) -> Self::Repr;

    /// Bitwise AND.
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Bitwise OR.
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Bitwise XOR.
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

    // ====== Default implementations ======

    /// Clamp values between lo and hi.
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <Self as F32x4Backend>::min(self, <Self as F32x4Backend>::max(self, a, lo), hi)
    }

    /// Precise reciprocal — defaults to delegating to [`rcp_approx`]
    /// (which itself defaults to identity). Backends override with
    /// Newton-Raphson refinement using a native splat for the constant.
    #[inline(always)]
    fn recip(self, a: Self::Repr) -> Self::Repr {
        Self::rcp_approx(self, a)
    }

    /// Precise reciprocal square root — see [`recip`] for rationale.
    #[inline(always)]
    fn rsqrt(self, a: Self::Repr) -> Self::Repr {
        Self::rsqrt_approx(self, a)
    }
}

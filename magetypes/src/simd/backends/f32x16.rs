//! Backend trait for `f32x16<T>` — 16-lane f32 SIMD vector.
//!
//! Each token type implements this trait with its platform-native representation.
//! The generic wrapper `f32x16<T>` delegates all operations to these trait methods.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

use super::sealed::Sealed;
use archmage::SimdToken;

/// Backend implementation for 16-lane f32 SIMD vectors.
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
pub trait F32x16Backend:
    SimdToken + Sealed + crate::scalar::F32ScalarBackend + Copy + 'static
{
    /// Platform-native SIMD representation.
    #[allow(private_bounds)]
    type Repr: Copy + Clone + Send + Sync + crate::simd_storage::Pod;

    // ====== Construction ======

    /// Broadcast scalar to all 16 lanes.
    fn splat(self, v: f32) -> Self::Repr;

    /// All lanes zero.
    fn zero(self) -> Self::Repr;

    /// Load from an aligned array.
    fn load(self, data: &[f32; 16]) -> Self::Repr;

    /// Create from array (zero-cost transmute where possible).
    fn from_array(self, arr: [f32; 16]) -> Self::Repr;

    /// Store to array.
    fn store(self, repr: Self::Repr, out: &mut [f32; 16]);

    /// Convert to array.
    fn to_array(self, repr: Self::Repr) -> [f32; 16];

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

    /// Lane-wise minimum matching `f32::min` (returns the numeric operand on a single quiet `NaN`; signalling `NaN` and `±0.0` unspecified).
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Lane-wise maximum matching `f32::max` (returns the numeric operand on a single quiet `NaN`; signalling `NaN` and `±0.0` unspecified).
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// IEEE 754-2019 minimum (`NaN`-propagating, `-0.0 < +0.0`).
    fn minimum(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// IEEE 754-2019 maximum (`NaN`-propagating, `-0.0 < +0.0`).
    fn maximum(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

    /// Fast lane-wise minimum (`NaN` and `±0.0` unspecified).
    #[inline(always)]
    fn min_fast(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        Self::min(self, a, b)
    }

    /// Fast lane-wise maximum (`NaN` and `±0.0` unspecified).
    #[inline(always)]
    fn max_fast(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        Self::max(self, a, b)
    }

    /// Square root.
    fn sqrt(self, a: Self::Repr) -> Self::Repr;

    /// Absolute value.
    fn abs(self, a: Self::Repr) -> Self::Repr;

    /// Truncate toward zero without changing the sign of zero.
    #[inline(always)]
    fn trunc(self, a: Self::Repr) -> Self::Repr {
        let lanes = <Self as F32x16Backend>::to_array(self, a);
        <Self as F32x16Backend>::from_array(self, lanes.map(crate::scalar::trunc_f32))
    }

    /// Multiply-negate-add with backend-dependent rounding.
    #[inline(always)]
    fn mul_neg_add_relaxed(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr {
        <Self as F32x16Backend>::mul_add(self, <Self as F32x16Backend>::neg(self, a), b, c)
    }

    /// Multiply-negate-add with one rounding.
    #[inline(always)]
    fn mul_neg_add_fused(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr {
        <Self as F32x16Backend>::mul_add_portable(self, <Self as F32x16Backend>::neg(self, a), b, c)
    }

    /// Round toward negative infinity.
    fn floor(self, a: Self::Repr) -> Self::Repr;

    /// Round toward positive infinity.
    fn ceil(self, a: Self::Repr) -> Self::Repr;

    /// Round to nearest integer.
    fn round(self, a: Self::Repr) -> Self::Repr;

    /// Multiply-add: `a * b + c`, fused where the hardware fuses.
    ///
    /// One rounding on x86 v3/v4 and NEON. Two roundings (multiply,
    /// then add) on the scalar backend and on WASM without
    /// `relaxed-simd`; relaxed WASM uses the engine's madd, which may
    /// round either way. NaN payload/sign are unspecified.
    fn mul_add(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr;

    /// Multiply-sub: `a * b - c`. Same rounding contract as
    /// [`mul_add`](Self::mul_add).
    fn mul_sub(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr;

    /// Multiply-add with one rounding on every backend: `a * b + c`.
    ///
    /// The same result everywhere with round-to-nearest and gradual
    /// underflow, NaN payload/sign aside. Software FMA where the
    /// selected backend has no guaranteed hardware FMA.
    fn mul_add_portable(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr;

    /// Multiply-sub with one rounding on every backend: `a * b - c`.
    fn mul_sub_portable(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr;

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
    fn reduce_add(self, a: Self::Repr) -> f32;

    /// Minimum across all 16 lanes.
    fn reduce_min(self, a: Self::Repr) -> f32;

    /// Maximum across all 16 lanes.
    fn reduce_max(self, a: Self::Repr) -> f32;

    // ====== Approximations ======

    /// Fast reciprocal approximation. Default returns the input
    /// unchanged; backends override with native intrinsics. The
    /// previous default `Self::div(Self::splat(1.0), a)` would
    /// require splat to be tokenless — incompatible with the
    /// soundness fix that gated splat on a token value.
    #[inline(always)]
    fn rcp_approx(self, a: Self::Repr) -> Self::Repr {
        a
    }

    /// Fast reciprocal square root approximation — see [`rcp_approx`](Self::rcp_approx).
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

    /// Clamp values between `lo` and `hi`: `min_fast(max_fast(a, lo), hi)`,
    /// two instructions on native backends.
    ///
    /// A quiet `NaN` lane in `a` becomes `lo` on x86, NEON, strict WASM, and
    /// `ScalarToken`; signalling `NaN`, WASM `relaxed-simd` `NaN` lanes, `NaN`
    /// bounds, and `lo > hi` are unspecified per backend, and `clamp` never panics.
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <Self as F32x16Backend>::min_fast(self, <Self as F32x16Backend>::max_fast(self, a, lo), hi)
    }

    /// Precise reciprocal — defaults to delegating to rcp_approx.
    /// Every shipped backend overrides with exact IEEE division.
    #[inline(always)]
    fn recip(self, a: Self::Repr) -> Self::Repr {
        Self::rcp_approx(self, a)
    }

    /// Precise reciprocal square root — see [`recip`](Self::recip).
    #[inline(always)]
    fn rsqrt(self, a: Self::Repr) -> Self::Repr {
        Self::rsqrt_approx(self, a)
    }

    /// Load unsigned lanes and convert them exactly, without normalization.
    #[doc(hidden)]
    #[inline(always)]
    fn from_u16_lanes(self, lanes: &[u16; 16]) -> Self::Repr {
        <Self as F32x16Backend>::from_array(self, lanes.map(f32::from))
    }

    /// Convert lanes to u16 using the encoded policy.
    #[doc(hidden)]
    #[inline(always)]
    fn to_u16_lanes<const CFG: u32>(self, value: Self::Repr) -> [u16; 16] {
        <Self as F32x16Backend>::to_array(self, value).map(crate::scalar::to_u16_with::<CFG>)
    }

    /// Convert lanes to u8 using the encoded policy.
    #[doc(hidden)]
    #[inline(always)]
    fn to_u8_lanes<const CFG: u32>(self, value: Self::Repr) -> [u8; 16] {
        <Self as F32x16Backend>::to_array(self, value).map(crate::scalar::to_u8_with::<CFG>)
    }

    /// Convert f32 lanes to unsigned u32 stored in the float representation.
    #[doc(hidden)]
    #[inline(always)]
    fn to_u32_lanes<const CFG: u32>(self, value: Self::Repr) -> Self::Repr {
        let lanes = <Self as F32x16Backend>::to_array(self, value);
        <Self as F32x16Backend>::from_array(
            self,
            lanes.map(|x| f32::from_bits(crate::scalar::to_u32_with::<CFG>(x))),
        )
    }

    /// Widen unsigned u32 bit patterns stored in the float representation.
    #[doc(hidden)]
    #[inline(always)]
    fn from_u32_lanes(self, value: Self::Repr) -> Self::Repr {
        let lanes = <Self as F32x16Backend>::to_array(self, value);
        <Self as F32x16Backend>::from_array(self, lanes.map(|x| x.to_bits() as f32))
    }

    /// Decode binary16 using the encoded policy.
    #[doc(hidden)]
    #[inline(always)]
    fn from_f16_lanes<const CFG: u32>(self, lanes: &[u16; 16]) -> Self::Repr {
        <Self as F32x16Backend>::from_array(self, lanes.map(crate::scalar::from_f16_with::<CFG>))
    }

    /// Encode binary16 using the encoded policy.
    #[doc(hidden)]
    #[inline(always)]
    fn to_f16_lanes<const CFG: u32>(self, value: Self::Repr) -> [u16; 16] {
        <Self as F32x16Backend>::to_array(self, value).map(crate::scalar::to_f16_bits_with::<CFG>)
    }

    /// Widen bfloat16 bit patterns.
    #[doc(hidden)]
    #[inline(always)]
    fn from_bf16_lanes(self, lanes: &[u16; 16]) -> Self::Repr {
        <Self as F32x16Backend>::from_array(self, lanes.map(crate::scalar::from_bf16))
    }

    /// Encode bfloat16 using the encoded policy.
    #[doc(hidden)]
    #[inline(always)]
    fn to_bf16_lanes<const CFG: u32>(self, value: Self::Repr) -> [u16; 16] {
        <Self as F32x16Backend>::to_array(self, value).map(crate::scalar::to_bf16_with::<CFG>)
    }

    /// Broadcast lane 3 in each consecutive group of four lanes.
    #[doc(hidden)]
    #[inline(always)]
    fn broadcast_rgba_alpha_repr(self, value: Self::Repr) -> Self::Repr {
        let value = <Self as F32x16Backend>::to_array(self, value);
        <Self as F32x16Backend>::from_array(self, core::array::from_fn(|i| value[i | 3]))
    }

    /// Replace lane 3 in each consecutive group of four lanes.
    #[doc(hidden)]
    #[inline(always)]
    fn with_rgba_alpha_repr(self, value: Self::Repr, src: Self::Repr) -> Self::Repr {
        let mut value = <Self as F32x16Backend>::to_array(self, value);
        let src = <Self as F32x16Backend>::to_array(self, src);
        for i in (3..16).step_by(4) {
            value[i] = src[i];
        }
        <Self as F32x16Backend>::from_array(self, value)
    }

    /// Widen sixteen consecutive bytes to float.
    #[doc(hidden)]
    #[inline(always)]
    fn from_u8_lanes_wide(self, lanes: &[u8; 16]) -> Self::Repr {
        <Self as F32x16Backend>::from_array(self, lanes.map(f32::from))
    }

    /// Deinterleave sixteen RGBA pixels into planar float storage.
    #[doc(hidden)]
    #[inline(always)]
    fn deinterleave_rgba_wide(self, rgba: [Self::Repr; 4]) -> [Self::Repr; 4] {
        let pixels = rgba.map(|r| <Self as F32x16Backend>::to_array(self, r));
        core::array::from_fn(|c| {
            <Self as F32x16Backend>::from_array(
                self,
                core::array::from_fn(|i| pixels[i / 4][4 * (i % 4) + c]),
            )
        })
    }

    /// Interleave sixteen planar float pixels into RGBA storage.
    #[doc(hidden)]
    #[inline(always)]
    fn interleave_rgba_wide(self, channels: [Self::Repr; 4]) -> [Self::Repr; 4] {
        let channels = channels.map(|r| <Self as F32x16Backend>::to_array(self, r));
        core::array::from_fn(|group| {
            <Self as F32x16Backend>::from_array(
                self,
                core::array::from_fn(|i| channels[i % 4][4 * group + i / 4]),
            )
        })
    }
}

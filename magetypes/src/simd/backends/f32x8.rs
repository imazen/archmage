//! Backend trait for `f32x8<T>` — 8-lane f32 SIMD vector.
//!
//! Each token type implements this trait with its platform-native representation.
//! The generic wrapper `f32x8<T>` delegates all operations to these trait methods.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

use super::sealed::Sealed;
use archmage::SimdToken;

/// Crate-private layout and conversion hooks for `8`-lane `f32` backends.
pub(crate) trait F32x8Internal: SimdToken + Sealed + Copy + 'static {
    type InternalRepr: Copy + Clone + Send + Sync;

    /// Load unsigned lanes and convert them exactly, without normalization.
    fn from_u16_lanes(self, lanes: &[u16; 8]) -> Self::InternalRepr;

    /// Convert lanes to u16 using the encoded policy.
    fn to_u16_lanes<const CFG: u32>(self, value: Self::InternalRepr) -> [u16; 8];

    /// Convert lanes to u8 using the encoded policy.
    fn to_u8_lanes<const CFG: u32>(self, value: Self::InternalRepr) -> [u8; 8];

    /// Convert f32 lanes to unsigned u32 stored in the float representation.
    fn to_u32_lanes<const CFG: u32>(self, value: Self::InternalRepr) -> Self::InternalRepr;

    /// Widen unsigned u32 bit patterns stored in the float representation.
    fn from_u32_lanes(self, value: Self::InternalRepr) -> Self::InternalRepr;

    /// Decode binary16 using the encoded policy.
    fn from_f16_lanes<const CFG: u32>(self, lanes: &[u16; 8]) -> Self::InternalRepr;

    /// Encode binary16 using the encoded policy.
    fn to_f16_lanes<const CFG: u32>(self, value: Self::InternalRepr) -> [u16; 8];

    /// Widen bfloat16 bit patterns.
    fn from_bf16_lanes(self, lanes: &[u16; 8]) -> Self::InternalRepr;

    /// Encode bfloat16 using the encoded policy.
    fn to_bf16_lanes<const CFG: u32>(self, value: Self::InternalRepr) -> [u16; 8];

    /// Broadcast lane 3 in each consecutive group of four lanes.
    fn broadcast_rgba_alpha_repr(self, value: Self::InternalRepr) -> Self::InternalRepr;

    /// Replace lane 3 in each consecutive group of four lanes.
    fn with_rgba_alpha_repr(
        self,
        value: Self::InternalRepr,
        src: Self::InternalRepr,
    ) -> Self::InternalRepr;

    /// Quantize and store planar channels into packed RGBA bytes.
    fn store_rgba_u8_into(
        self,
        r: Self::InternalRepr,
        g: Self::InternalRepr,
        b: Self::InternalRepr,
        a: Self::InternalRepr,
        out: &mut [u8; 32],
    );

    /// Quantize and pack planar RGBA with the encoded integer policy.
    #[inline(always)]
    fn store_rgba_u8_with_into<const CFG: u32>(
        self,
        r: Self::InternalRepr,
        g: Self::InternalRepr,
        b: Self::InternalRepr,
        a: Self::InternalRepr,
        out: &mut [u8; 32],
    ) {
        let channels = [r, g, b, a].map(|v| Self::to_u8_lanes::<CFG>(self, v));
        for (i, byte) in out.iter_mut().enumerate() {
            *byte = channels[i % 4][i / 4];
        }
    }

    /// Deinterleave 8 packed RGB pixels from AoS into planar `[R, G, B]`.
    fn deinterleave_3ch_repr(self, rgb: [Self::InternalRepr; 3]) -> [Self::InternalRepr; 3];

    /// Interleave 3 planar `[R, G, B]` vectors into 3 AoS RGB vectors.
    fn interleave_3ch_repr(self, channels: [Self::InternalRepr; 3]) -> [Self::InternalRepr; 3];

    /// Load 8 packed RGB `u8` pixels into planar `f32` `(R, G, B)`.
    fn load_rgb_bytes(
        self,
        rgb: &[u8; 24],
    ) -> (Self::InternalRepr, Self::InternalRepr, Self::InternalRepr);

    /// Round-to-nearest-even, clamp to `0..=255`, and interleave into 8 RGB pixels.
    fn store_rgb_u8_into(
        self,
        r: Self::InternalRepr,
        g: Self::InternalRepr,
        b: Self::InternalRepr,
        out: &mut [u8; 24],
    );
}

/// Backend implementation for 8-lane f32 SIMD vectors.
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
#[allow(private_bounds)]
pub trait F32x8Backend:
    SimdToken
    + Sealed
    + F32x8Internal<InternalRepr = <Self as F32x8Backend>::Repr>
    + crate::scalar::F32ScalarBackend
    + Copy
    + 'static
{
    /// Platform-native SIMD representation.
    #[allow(private_bounds)]
    type Repr: Copy + Clone + Send + Sync + crate::simd_storage::Pod;

    // ====== Construction ======

    /// Broadcast scalar to all 8 lanes.
    fn splat(self, v: f32) -> Self::Repr;

    /// All lanes zero.
    fn zero(self) -> Self::Repr;

    /// Load from an aligned array.
    fn load(self, data: &[f32; 8]) -> Self::Repr;

    /// Create from array (zero-cost transmute where possible).
    fn from_array(self, arr: [f32; 8]) -> Self::Repr;

    /// Store to array.
    fn store(self, repr: Self::Repr, out: &mut [f32; 8]);

    /// Convert to array.
    fn to_array(self, repr: Self::Repr) -> [f32; 8];

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
        let lanes = <Self as F32x8Backend>::to_array(self, a);
        <Self as F32x8Backend>::from_array(self, lanes.map(crate::scalar::trunc_f32))
    }

    /// Multiply-negate-add with backend-dependent rounding.
    #[inline(always)]
    fn mul_neg_add_relaxed(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr {
        <Self as F32x8Backend>::mul_add(self, <Self as F32x8Backend>::neg(self, a), b, c)
    }

    /// Multiply-negate-add with one rounding.
    #[inline(always)]
    fn mul_neg_add_fused(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr {
        <Self as F32x8Backend>::mul_add_portable(self, <Self as F32x8Backend>::neg(self, a), b, c)
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

    /// Sum all 8 lanes.
    fn reduce_add(self, a: Self::Repr) -> f32;

    /// Minimum across all 8 lanes.
    fn reduce_min(self, a: Self::Repr) -> f32;

    /// Maximum across all 8 lanes.
    fn reduce_max(self, a: Self::Repr) -> f32;

    // ====== Approximations ======

    /// Raw per-backend reciprocal seed (1/x). On x86/ARM this is the
    /// hardware estimate; the generic `rcp_approx` refines it to a ≥12-bit
    /// floor. WASM/scalar override `rcp_approx` on the generic type.
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

    /// Fast reciprocal square root approximation: backend-dependent
    /// (x86 ~12-bit, ARM ~8-bit, WASM full). [`rsqrt`](Self::rsqrt) is full f32 everywhere.
    ///
    /// See [`rcp_approx`](Self::rcp_approx) for default-body rationale.
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
        <Self as F32x8Backend>::min_fast(self, <Self as F32x8Backend>::max_fast(self, a, lo), hi)
    }

    /// Working-tier reciprocal — defaults to delegating to
    /// [`rcp_approx`](Self::rcp_approx) (which itself defaults to identity), so every
    /// backend MUST override. Contract: <= 4 ULP by the backend's
    /// fastest conforming path (estimate + Newton on x86/NEON f32,
    /// exact division elsewhere); rails are per-backend and may be
    /// NaN at `±0`/`±inf` where an estimate is refined. The 0 ULP +
    /// IEEE-rails tier is the generic `recip_portable` (issue #64).
    #[inline(always)]
    fn recip(self, a: Self::Repr) -> Self::Repr {
        Self::rcp_approx(self, a)
    }

    /// Working-tier reciprocal square root — see [`recip`](Self::recip) for the
    /// contract shape.
    #[inline(always)]
    fn rsqrt(self, a: Self::Repr) -> Self::Repr {
        Self::rsqrt_approx(self, a)
    }

    // ====== Pixel pack (f32 only) ======

    /// Round-to-nearest-even then saturate each lane to `u8` (0..=255).
    ///
    /// Default body is scalar; `x86`/`aarch64` backends override with a
    /// native `cvt`+saturating-`pack` sequence.
    #[inline(always)]
    fn to_u8_bytes(self, a: Self::Repr) -> [u8; 8] {
        let arr = <Self as F32x8Backend>::to_array(self, a);
        arr.map(|x| crate::nostd_math::round_to_small_uint(x, 255.0) as u8)
    }

    /// Round/clamp 4 planar channels and interleave into RGBA bytes
    /// (8 pixels = 32 bytes). Each channel converts via
    /// the native `to_u8_bytes` (so x86/aarch64 get cvt+pack, not scalar
    /// `roundevenf`); the byte interleave is left to LLVM, which recovers
    /// it to native shuffles. A backend may still override for a tighter
    /// fused sequence.
    #[deprecated(since = "0.9.32", note = "Use f32x8::store_rgba_u8")]
    #[inline(always)]
    fn store_rgba_bytes(
        self,
        r: Self::Repr,
        g: Self::Repr,
        b: Self::Repr,
        a: Self::Repr,
    ) -> [u8; 32] {
        let mut out = [0u8; 32];
        <Self as F32x8Internal>::store_rgba_u8_into(self, r, g, b, a, &mut out);
        out
    }

    // ====== 8x8 transpose (f32x8 only) ======

    /// Transpose 8 row vectors of an 8x8 f32 matrix.
    #[inline(always)]
    fn transpose_8x8_repr(self, rows: [Self::Repr; 8]) -> [Self::Repr; 8] {
        let r: [[f32; 8]; 8] =
            core::array::from_fn(|i| <Self as F32x8Backend>::to_array(self, rows[i]));
        core::array::from_fn(|i| {
            <Self as F32x8Backend>::from_array(self, core::array::from_fn(|j| r[j][i]))
        })
    }
}

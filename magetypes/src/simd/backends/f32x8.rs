//! Backend trait for `f32x8<T>` — 8-lane f32 SIMD vector.
//!
//! Each token type implements this trait with its platform-native representation.
//! The generic wrapper `f32x8<T>` delegates all operations to these trait methods.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

use super::sealed::Sealed;
use archmage::SimdToken;

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
pub trait F32x8Backend: SimdToken + Sealed + Copy + 'static {
    /// Platform-native SIMD representation.
    type Repr: Copy + Clone + Send + Sync;

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
    /// (x86 ~12-bit, ARM ~8-bit, WASM full). [`rsqrt`] is full f32 everywhere.
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
        <Self as F32x8Backend>::min(self, <Self as F32x8Backend>::max(self, a, lo), hi)
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

    // ====== Pixel pack (f32 only) ======

    /// Round-to-nearest-even then saturate each lane to `u8` (0..=255).
    ///
    /// Default body is scalar; `x86`/`aarch64` backends override with a
    /// native `cvt`+saturating-`pack` sequence.
    #[inline(always)]
    fn to_u8_bytes(self, a: Self::Repr) -> [u8; 8] {
        let arr = <Self as F32x8Backend>::to_array(self, a);
        core::array::from_fn(|i| crate::nostd_math::roundevenf(arr[i]).clamp(0.0, 255.0) as u8)
    }

    /// Round/clamp 4 planar channels and interleave into RGBA bytes
    /// (8 pixels = 32 bytes). Each channel converts via
    /// the native `to_u8_bytes` (so x86/aarch64 get cvt+pack, not scalar
    /// `roundevenf`); the byte interleave is left to LLVM, which recovers
    /// it to native shuffles. A backend may still override for a tighter
    /// fused sequence.
    #[inline(always)]
    fn store_rgba_bytes(
        self,
        r: Self::Repr,
        g: Self::Repr,
        b: Self::Repr,
        a: Self::Repr,
    ) -> [u8; 32] {
        let rb = <Self as F32x8Backend>::to_u8_bytes(self, r);
        let gb = <Self as F32x8Backend>::to_u8_bytes(self, g);
        let bb = <Self as F32x8Backend>::to_u8_bytes(self, b);
        let ab = <Self as F32x8Backend>::to_u8_bytes(self, a);
        let mut out = [0u8; 32];
        let mut i = 0;
        while i < 8 {
            out[i * 4] = rb[i];
            out[i * 4 + 1] = gb[i];
            out[i * 4 + 2] = bb[i];
            out[i * 4 + 3] = ab[i];
            i += 1;
        }
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

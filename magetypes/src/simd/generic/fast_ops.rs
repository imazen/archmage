//! Unchecked float-to-integer truncation (`to_i32_fast`) on generic `f32`
//! vector types.
//!
//! Scalar remainder equivalents live in [`crate::scalar`].
//!
//! [`f32x4::to_i32`] (`to_i32_saturating` on x86) guarantees uniform saturation
//! across ISAs by emitting extra compare+blend instructions around `vcvttps2dq`
//! on x86. When inputs are already clamped or known finite and in-bounds
//! (quantization after ReLU6 / `clamp`, LUT index generation after range
//! reduction, pixel packing), `to_i32_fast` emits the native truncation
//! instruction directly without overflow/`NaN` fixup masks (`vcvttps2dq` on x86,
//! `vcvtq_s32_f32` on NEON, `i32x4.trunc_sat_f32x4_s` on strict WASM or
//! `i32x4.relaxed_trunc_f32x4_s` when compiled with `relaxed-simd`).

#![forbid(unsafe_code)]

#[cfg(feature = "w512")]
use crate::simd::backends::F32x16Convert;
use crate::simd::backends::{F32x4Convert, F32x8Convert};
use crate::simd::generic::{f32x4, f32x8, i32x4, i32x8};
#[cfg(feature = "w512")]
use crate::simd::generic::{f32x16, i32x16};

impl<T: F32x4Convert> f32x4<T> {
    /// Fast single-instruction truncation toward zero to `i32x4<T>` without
    /// out-of-range/NaN saturation fixups.
    ///
    /// Exact for finite values in `-2_147_483_520.0..=2_147_483_520.0` (every `f32`
    /// whose truncated integer fits in `i32`, plus `-2_147_483_648.0`). Values
    /// outside `i32` range or `NaN` return an unspecified `i32` (`i32::MIN` on x86,
    /// saturated/zero on NEON/strict WASM/Scalar, engine-defined with `relaxed-simd`)
    /// — never UB.
    ///
    /// Scalar tail equivalent: [`to_i32_fast`](crate::scalar::to_i32_fast).
    #[inline(always)]
    pub fn to_i32_fast(self) -> i32x4<T> {
        i32x4::from_repr_unchecked(self.1, T::convert_f32_to_i32_fast(self.1, self.0))
    }
}

impl<T: F32x8Convert> f32x8<T> {
    /// Fast single-instruction truncation toward zero to `i32x8<T>`.
    /// See [`f32x4::to_i32_fast`].
    #[inline(always)]
    pub fn to_i32_fast(self) -> i32x8<T> {
        i32x8::from_repr_unchecked(self.1, T::convert_f32_to_i32_fast(self.1, self.0))
    }
}

#[cfg(feature = "w512")]
impl<T: F32x16Convert> f32x16<T> {
    /// Fast single-instruction truncation toward zero to `i32x16<T>`.
    /// See [`f32x4::to_i32_fast`].
    #[inline(always)]
    pub fn to_i32_fast(self) -> i32x16<T> {
        i32x16::from_repr_unchecked(self.1, T::convert_f32_to_i32_fast(self.1, self.0))
    }
}

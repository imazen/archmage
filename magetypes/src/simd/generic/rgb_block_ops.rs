//! 3-channel RGB AoS↔SoA interleave/deinterleave and packed `u8` pixel helpers
//! on [`f32x4`] and [`f32x8`].
//!
//! Complements the 4-channel RGBA helpers (`deinterleave_4ch`, `interleave_4ch`,
//! `load_{4,8}_rgba_u8_t`, `store_{4,8}_rgba_u8`) for vision preprocessing and
//! 3-channel NHWC neural network stems that consume or emit packed RGB without
//! an alpha channel.

#![forbid(unsafe_code)]

use crate::simd::backends::{F32x4Backend, F32x8Backend};
use crate::simd::generic::{f32x4, f32x8};

impl<T: F32x4Backend> f32x4<T> {
    /// Deinterleave 4 packed RGB pixels (`12` floats across 3 [`f32x4`] vectors)
    /// from AoS to SoA planar format.
    ///
    /// Input layout:
    /// - `rgb[0]` = `[R0, G0, B0, R1]`
    /// - `rgb[1]` = `[G1, B1, R2, G2]`
    /// - `rgb[2]` = `[B2, R3, G3, B3]`
    ///
    /// Output layout: `[R_all, G_all, B_all]` where `R_all = [R0, R1, R2, R3]`,
    /// `G_all = [G0, G1, G2, G3]`, `B_all = [B0, B1, B2, B3]`.
    #[inline(always)]
    pub fn deinterleave_3ch(rgb: [Self; 3]) -> [Self; 3] {
        let token = rgb[0].1;
        let v0 = rgb[0].to_array();
        let v1 = rgb[1].to_array();
        let v2 = rgb[2].to_array();
        [
            Self::from_array_t(token, [v0[0], v0[3], v1[2], v2[1]]),
            Self::from_array_t(token, [v0[1], v1[0], v1[3], v2[2]]),
            Self::from_array_t(token, [v0[2], v1[1], v2[0], v2[3]]),
        ]
    }

    /// Interleave 3 planar [`f32x4`] channels (`[R, G, B]`) from SoA into 3
    /// contiguous AoS RGB vectors (`12` floats = 4 packed RGB pixels).
    ///
    /// Inverse of [`Self::deinterleave_3ch`]:
    /// - `out[0]` = `[R0, G0, B0, R1]`
    /// - `out[1]` = `[G1, B1, R2, G2]`
    /// - `out[2]` = `[B2, R3, G3, B3]`
    #[inline(always)]
    pub fn interleave_3ch(channels: [Self; 3]) -> [Self; 3] {
        let token = channels[0].1;
        let r = channels[0].to_array();
        let g = channels[1].to_array();
        let b = channels[2].to_array();
        [
            Self::from_array_t(token, [r[0], g[0], b[0], r[1]]),
            Self::from_array_t(token, [g[1], b[1], r[2], g[2]]),
            Self::from_array_t(token, [b[2], r[3], g[3], b[3]]),
        ]
    }

    /// Load 4 packed RGB `u8` pixels (`12` bytes) and deinterleave into 3 planar
    /// [`f32x4`] channel vectors `(R, G, B)` with values in `[0.0, 255.0]`.
    #[inline(always)]
    pub fn load_4_rgb_u8_t(token: T, rgb: &[u8; 12]) -> (Self, Self, Self) {
        let r: [f32; 4] = core::array::from_fn(|i| rgb[i * 3] as f32);
        let g: [f32; 4] = core::array::from_fn(|i| rgb[i * 3 + 1] as f32);
        let b: [f32; 4] = core::array::from_fn(|i| rgb[i * 3 + 2] as f32);
        (
            Self::from_array_t(token, r),
            Self::from_array_t(token, g),
            Self::from_array_t(token, b),
        )
    }

    /// Round, clamp to `[0, 255]`, and interleave 3 planar [`f32x4`] channel
    /// vectors into 4 packed RGB `u8` pixels (`12` bytes).
    ///
    /// Uses the same nearest-even rounding and `[0, 255]` saturation contract as
    /// [`Self::store_4_rgba_u8`] and [`Self::to_u8`].
    #[inline(always)]
    pub fn store_4_rgb_u8(r: Self, g: Self, b: Self) -> [u8; 12] {
        let rgba = Self::store_4_rgba_u8(r, g, b, Self::zero_t(r.1));
        [
            rgba[0], rgba[1], rgba[2], rgba[4], rgba[5], rgba[6], rgba[8], rgba[9], rgba[10],
            rgba[12], rgba[13], rgba[14],
        ]
    }
}

impl<T: F32x8Backend> f32x8<T> {
    /// Deinterleave 8 packed RGB pixels (`24` floats across 3 [`f32x8`] vectors)
    /// from AoS to SoA planar format.
    ///
    /// Input layout:
    /// - `rgb[0]` = `[R0, G0, B0, R1, G1, B1, R2, G2]`
    /// - `rgb[1]` = `[B2, R3, G3, B3, R4, G4, B4, R5]`
    /// - `rgb[2]` = `[G5, B5, R6, G6, B6, R7, G7, B7]`
    ///
    /// Output layout: `[R_all, G_all, B_all]` — one [`f32x8`] per channel across
    /// pixels `0..8`.
    #[inline(always)]
    pub fn deinterleave_3ch(rgb: [Self; 3]) -> [Self; 3] {
        let token = rgb[0].1;
        let v0 = rgb[0].to_array();
        let v1 = rgb[1].to_array();
        let v2 = rgb[2].to_array();
        [
            Self::from_array_t(
                token,
                [v0[0], v0[3], v0[6], v1[1], v1[4], v1[7], v2[2], v2[5]],
            ),
            Self::from_array_t(
                token,
                [v0[1], v0[4], v0[7], v1[2], v1[5], v2[0], v2[3], v2[6]],
            ),
            Self::from_array_t(
                token,
                [v0[2], v0[5], v1[0], v1[3], v1[6], v2[1], v2[4], v2[7]],
            ),
        ]
    }

    /// Interleave 3 planar [`f32x8`] channels (`[R, G, B]`) from SoA into 3
    /// contiguous AoS RGB vectors (`24` floats = 8 packed RGB pixels).
    ///
    /// Inverse of [`Self::deinterleave_3ch`]:
    /// - `out[0]` = `[R0, G0, B0, R1, G1, B1, R2, G2]`
    /// - `out[1]` = `[B2, R3, G3, B3, R4, G4, B4, R5]`
    /// - `out[2]` = `[G5, B5, R6, G6, B6, R7, G7, B7]`
    #[inline(always)]
    pub fn interleave_3ch(channels: [Self; 3]) -> [Self; 3] {
        let token = channels[0].1;
        let r = channels[0].to_array();
        let g = channels[1].to_array();
        let b = channels[2].to_array();
        [
            Self::from_array_t(token, [r[0], g[0], b[0], r[1], g[1], b[1], r[2], g[2]]),
            Self::from_array_t(token, [b[2], r[3], g[3], b[3], r[4], g[4], b[4], r[5]]),
            Self::from_array_t(token, [g[5], b[5], r[6], g[6], b[6], r[7], g[7], b[7]]),
        ]
    }

    /// Load 8 packed RGB `u8` pixels (`24` bytes) and deinterleave into 3 planar
    /// [`f32x8`] channel vectors `(R, G, B)` with values in `[0.0, 255.0]`.
    #[inline(always)]
    pub fn load_8_rgb_u8_t(token: T, rgb: &[u8; 24]) -> (Self, Self, Self) {
        let r: [f32; 8] = core::array::from_fn(|i| rgb[i * 3] as f32);
        let g: [f32; 8] = core::array::from_fn(|i| rgb[i * 3 + 1] as f32);
        let b: [f32; 8] = core::array::from_fn(|i| rgb[i * 3 + 2] as f32);
        (
            Self::from_array_t(token, r),
            Self::from_array_t(token, g),
            Self::from_array_t(token, b),
        )
    }

    /// Round, clamp to `[0, 255]`, and interleave 3 planar [`f32x8`] channel
    /// vectors into 8 packed RGB `u8` pixels (`24` bytes).
    ///
    /// Uses the same nearest-even rounding and `[0, 255]` saturation contract as
    /// [`Self::store_8_rgba_u8`] and [`Self::to_u8`].
    #[inline(always)]
    pub fn store_8_rgb_u8(r: Self, g: Self, b: Self) -> [u8; 24] {
        let rgba = Self::store_8_rgba_u8(r, g, b, Self::zero_t(r.1));
        core::array::from_fn(|i| rgba[(i / 3) * 4 + (i % 3)])
    }
}

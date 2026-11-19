//! 3-channel packed RGB <-> planar deinterleave/interleave and u8 load/store helpers.
//!
//! Native SIMD backend implementations route through [`F32x4Backend`] and
//! [`F32x8Backend`] across x86_64 (AVX2/V4), AArch64 NEON, WebAssembly SIMD128,
//! and scalar backends without any `unsafe` code.

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
    /// Output layout: `[R_all, G_all, B_all]` — one [`f32x4`] per channel across
    /// pixels `0..4`.
    #[inline(always)]
    pub fn deinterleave_3ch(rgb: [Self; 3]) -> [Self; 3] {
        let token = rgb[0].1;
        let [r, g, b] = T::deinterleave_3ch_repr(token, [rgb[0].0, rgb[1].0, rgb[2].0]);
        [Self(r, token), Self(g, token), Self(b, token)]
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
        let [v0, v1, v2] =
            T::interleave_3ch_repr(token, [channels[0].0, channels[1].0, channels[2].0]);
        [Self(v0, token), Self(v1, token), Self(v2, token)]
    }

    /// Load 4 packed RGB `u8` pixels (`12` bytes) and deinterleave into 3 planar
    /// [`f32x4`] channel vectors `(R, G, B)` with values in `[0.0, 255.0]`.
    #[inline(always)]
    pub fn load_4_rgb_u8_t(token: T, rgb: &[u8; 12]) -> (Self, Self, Self) {
        let (r, g, b) = T::load_rgb_bytes(token, rgb);
        (Self(r, token), Self(g, token), Self(b, token))
    }

    /// Load 4 packed RGB16 pixels as exact, unnormalized R, G and B planes.
    ///
    /// Reads exactly the 12 words in the input block.
    #[inline(always)]
    pub fn load_4_rgb_u16_t(token: T, rgb: &[u16; 12]) -> (Self, Self, Self) {
        let (chunks, _) = rgb.as_chunks::<4>();
        let [r, g, b] = Self::deinterleave_3ch(core::array::from_fn(|i| {
            Self::from_u16_t(token, &chunks[i])
        }));
        (r, g, b)
    }

    /// Round, clamp to `[0, 255]`, and interleave 3 planar [`f32x4`] channel
    /// vectors into 4 packed RGB `u8` pixels (`12` bytes).
    ///
    /// Uses the same nearest-even rounding and `[0, 255]` saturation contract as
    /// [`Self::store_4_rgba_u8`] and [`Self::to_u8`].
    #[inline(always)]
    pub fn store_4_rgb_u8(r: Self, g: Self, b: Self) -> [u8; 12] {
        T::store_rgb_bytes(r.1, r.0, g.0, b.0)
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
        let [r, g, b] = T::deinterleave_3ch_repr(token, [rgb[0].0, rgb[1].0, rgb[2].0]);
        [Self(r, token), Self(g, token), Self(b, token)]
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
        let [v0, v1, v2] =
            T::interleave_3ch_repr(token, [channels[0].0, channels[1].0, channels[2].0]);
        [Self(v0, token), Self(v1, token), Self(v2, token)]
    }

    /// Load 8 packed RGB `u8` pixels (`24` bytes) and deinterleave into 3 planar
    /// [`f32x8`] channel vectors `(R, G, B)` with values in `[0.0, 255.0]`.
    #[inline(always)]
    pub fn load_8_rgb_u8_t(token: T, rgb: &[u8; 24]) -> (Self, Self, Self) {
        let (r, g, b) = T::load_rgb_bytes(token, rgb);
        (Self(r, token), Self(g, token), Self(b, token))
    }

    /// Load 8 packed RGB16 pixels as exact, unnormalized R, G and B planes.
    ///
    /// Reads exactly the 24 words in the input block.
    #[inline(always)]
    pub fn load_8_rgb_u16_t(token: T, rgb: &[u16; 24]) -> (Self, Self, Self) {
        let (chunks, _) = rgb.as_chunks::<8>();
        let [r, g, b] = Self::deinterleave_3ch(core::array::from_fn(|i| {
            Self::from_u16_t(token, &chunks[i])
        }));
        (r, g, b)
    }

    /// Round, clamp to `[0, 255]`, and interleave 3 planar [`f32x8`] channel
    /// vectors into 8 packed RGB `u8` pixels (`24` bytes).
    ///
    /// Uses the same nearest-even rounding and `[0, 255]` saturation contract as
    /// [`Self::store_8_rgba_u8`] and [`Self::to_u8`].
    #[inline(always)]
    pub fn store_8_rgb_u8(r: Self, g: Self, b: Self) -> [u8; 24] {
        T::store_rgb_bytes(r.1, r.0, g.0, b.0)
    }
}

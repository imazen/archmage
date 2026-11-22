//! 3-channel packed RGB <-> planar deinterleave/interleave and u8 load/store helpers.
//!
//! Supports x86-64 (AVX2 and AVX-512), AArch64 NEON, WebAssembly SIMD128,
//! and scalar backends.

#![forbid(unsafe_code)]

use crate::simd::backends::{F32x4Backend, F32x8Backend};
use crate::simd::generic::{f32x4, f32x8};
#[cfg(feature = "w512")]
use crate::simd::{backends::F32x16Backend, generic::f32x16};

/// ### Packed RGB layout
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
    pub fn load_rgb_u8_t(token: T, rgb: &[u8; 12]) -> (Self, Self, Self) {
        let (r, g, b) = T::load_rgb_bytes(token, rgb);
        (Self(r, token), Self(g, token), Self(b, token))
    }

    /// Load 4 packed RGB16 pixels as exact, unnormalized R, G and B planes.
    ///
    /// Reads exactly the 12 words in the input block.
    #[inline(always)]
    pub fn load_rgb_u16_t(token: T, rgb: &[u16; 12]) -> (Self, Self, Self) {
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
    /// [`Self::store_rgba_u8`] and [`Self::to_u8`].
    #[inline(always)]
    pub fn store_rgb_u8(r: Self, g: Self, b: Self, out: &mut [u8; 12]) {
        T::store_rgb_u8_into(r.1, r.0, g.0, b.0, out);
    }

    /// Store 4 packed RGB pixels with the configured rounding and bounds.
    ///
    /// Configure with [`IntConvert`](crate::numerics::IntConvert).
    #[inline(always)]
    pub fn store_rgb_u8_with<const CFG: u32>(r: Self, g: Self, b: Self, out: &mut [u8; 12]) {
        let [r, g, b] = [r, g, b].map(Self::to_u8_with::<CFG>);
        for (i, pixel) in out.as_chunks_mut::<3>().0.iter_mut().enumerate() {
            *pixel = [r[i], g[i], b[i]];
        }
    }

    /// Store 4 packed RGBA pixels with the configured rounding and bounds.
    ///
    /// Configure with [`IntConvert`](crate::numerics::IntConvert).
    #[inline(always)]
    pub fn store_rgba_u8_with<const CFG: u32>(
        r: Self,
        g: Self,
        b: Self,
        a: Self,
        out: &mut [u8; 16],
    ) {
        const {
            let _ = crate::numerics::IntConvert::decode(CFG);
        }
        T::store_rgba_u8_with_into::<CFG>(r.1, r.0, g.0, b.0, a.0, out);
    }
}

/// ### Packed RGB layout
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
    pub fn load_rgb_u8_t(token: T, rgb: &[u8; 24]) -> (Self, Self, Self) {
        let (r, g, b) = T::load_rgb_bytes(token, rgb);
        (Self(r, token), Self(g, token), Self(b, token))
    }

    /// Load 8 packed RGB16 pixels as exact, unnormalized R, G and B planes.
    ///
    /// Reads exactly the 24 words in the input block.
    #[inline(always)]
    pub fn load_rgb_u16_t(token: T, rgb: &[u16; 24]) -> (Self, Self, Self) {
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
    /// [`Self::store_rgba_u8`] and [`Self::to_u8`].
    #[inline(always)]
    pub fn store_rgb_u8(r: Self, g: Self, b: Self, out: &mut [u8; 24]) {
        T::store_rgb_u8_into(r.1, r.0, g.0, b.0, out);
    }

    /// Store 8 packed RGB pixels with the configured rounding and bounds.
    ///
    /// Configure with [`IntConvert`](crate::numerics::IntConvert).
    #[inline(always)]
    pub fn store_rgb_u8_with<const CFG: u32>(r: Self, g: Self, b: Self, out: &mut [u8; 24]) {
        let [r, g, b] = [r, g, b].map(Self::to_u8_with::<CFG>);
        for (i, pixel) in out.as_chunks_mut::<3>().0.iter_mut().enumerate() {
            *pixel = [r[i], g[i], b[i]];
        }
    }

    /// Store 8 packed RGBA pixels with the configured rounding and bounds.
    ///
    /// Configure with [`IntConvert`](crate::numerics::IntConvert).
    #[inline(always)]
    pub fn store_rgba_u8_with<const CFG: u32>(
        r: Self,
        g: Self,
        b: Self,
        a: Self,
        out: &mut [u8; 32],
    ) {
        const {
            let _ = crate::numerics::IntConvert::decode(CFG);
        }
        T::store_rgba_u8_with_into::<CFG>(r.1, r.0, g.0, b.0, a.0, out);
    }
}

/// ### Packed RGB layout
#[cfg(feature = "w512")]
impl<T: F32x16Backend> f32x16<T> {
    /// Deinterleave 16 packed RGB pixels (`48` floats across 3 [`f32x16`] vectors)
    /// from AoS to SoA planar format.
    ///
    /// Input layout: `rgb[0]` starts at `R0`, `rgb[1]` at `G5`, and
    /// `rgb[2]` at `B10`. Each vector holds 16 consecutive floats.
    ///
    /// Output layout: `[R_all, G_all, B_all]` — one [`f32x16`] per channel
    /// across pixels `0..16`. All lane bits are preserved.
    #[inline(always)]
    pub fn deinterleave_3ch(rgb: [Self; 3]) -> [Self; 3] {
        let token = rgb[0].1;
        T::deinterleave_3ch_repr(token, rgb.map(Self::into_repr))
            .map(|v| Self::from_repr_unchecked(token, v))
    }

    /// Interleave 3 planar [`f32x16`] channels (`[R, G, B]`) from SoA into 3
    /// consecutive AoS RGB vectors (`48` floats = 16 packed RGB pixels).
    ///
    /// Inverse of [`Self::deinterleave_3ch`]. All lane bits are preserved.
    #[inline(always)]
    pub fn interleave_3ch(channels: [Self; 3]) -> [Self; 3] {
        let token = channels[0].1;
        T::interleave_3ch_repr(token, channels.map(Self::into_repr))
            .map(|v| Self::from_repr_unchecked(token, v))
    }

    /// Load 16 packed RGB `u8` pixels (`48` bytes) and deinterleave into 3 planar
    /// [`f32x16`] channel vectors `(R, G, B)` with values in `[0.0, 255.0]`.
    #[inline(always)]
    pub fn load_rgb_u8_t(token: T, rgb: &[u8; 48]) -> (Self, Self, Self) {
        let (chunks, _) = rgb.as_chunks::<16>();
        let [r, g, b] =
            Self::deinterleave_3ch(core::array::from_fn(|i| Self::from_u8_t(token, &chunks[i])));
        (r, g, b)
    }

    /// Load 16 packed RGB16 pixels as exact, unnormalized R, G and B planes.
    ///
    /// Reads exactly the 48 words in the input block.
    #[inline(always)]
    pub fn load_rgb_u16_t(token: T, rgb: &[u16; 48]) -> (Self, Self, Self) {
        let (chunks, _) = rgb.as_chunks::<16>();
        let [r, g, b] = Self::deinterleave_3ch(core::array::from_fn(|i| {
            Self::from_u16_t(token, &chunks[i])
        }));
        (r, g, b)
    }

    /// Round, clamp to `[0, 255]`, and interleave 3 planar [`f32x16`] channel
    /// vectors into 16 packed RGB `u8` pixels (`48` bytes).
    ///
    /// Uses the same nearest-even rounding and `[0, 255]` saturation contract as
    /// [`Self::store_rgba_u8`] and [`Self::to_u8`].
    #[inline(always)]
    pub fn store_rgb_u8(r: Self, g: Self, b: Self, out: &mut [u8; 48]) {
        use crate::numerics::IntConvert;

        Self::store_rgb_u8_with::<{ IntConvert::new().saturate_all().bits() }>(r, g, b, out);
    }

    /// Store 16 packed RGB pixels with the configured rounding and bounds.
    ///
    /// Configure with [`IntConvert`](crate::numerics::IntConvert). Policies for
    /// other conversion families are rejected at compile time.
    ///
    /// ```compile_fail
    /// use archmage::ScalarToken;
    /// use magetypes::{numerics::HalfConvert, simd::generic::f32x16};
    ///
    /// let v = f32x16::zero_t(ScalarToken);
    /// f32x16::store_rgb_u8_with::<{ HalfConvert::new().bits() }>(v, v, v, &mut [0; 48]);
    /// ```
    #[inline(always)]
    pub fn store_rgb_u8_with<const CFG: u32>(r: Self, g: Self, b: Self, out: &mut [u8; 48]) {
        for (block, lanes) in out
            .as_chunks_mut::<16>()
            .0
            .iter_mut()
            .zip(Self::interleave_3ch([r, g, b]))
        {
            *block = lanes.to_u8_with::<CFG>();
        }
    }
}

#![forbid(unsafe_code)]

use super::{f32x4, f32x8};
use crate::simd::backends::{F32x4Backend, F32x8Backend};

macro_rules! float_layout {
    ($vector:ident, $backend:ident, $lanes:literal) => {
        impl<T: $backend> $vector<T> {
            /// Load unsigned integers in lane order and convert exactly to `f32`.
            ///
            /// The values are in `[0.0, 65535.0]`.
            #[inline(always)]
            pub fn from_u16_t(token: T, lanes: &[u16; $lanes]) -> Self {
                Self(T::from_u16_lanes(token, lanes), token)
            }

            /// Convert each lane using Rust's `as u16` semantics.
            ///
            /// Truncates toward zero and saturates to `0..=65535`. NaN becomes zero.
            #[inline(always)]
            pub fn to_u16_trunc_saturating(self) -> [u16; $lanes] {
                T::to_u16_lanes(self.1, self.0)
            }

            /// Copy the alpha of each packed RGBA pixel into all four of its lanes.
            ///
            /// Preserves the original alpha bits, including signed zero and NaN payloads.
            #[inline(always)]
            #[must_use]
            pub fn broadcast_rgba_alpha(self) -> Self {
                Self(T::broadcast_rgba_alpha_repr(self.1, self.0), self.1)
            }

            /// Replace each packed RGBA pixel's alpha with the corresponding alpha from `src`.
            ///
            /// Preserves the bits of all selected color and alpha lanes.
            #[inline(always)]
            #[must_use]
            pub fn with_rgba_alpha(self, src: Self) -> Self {
                Self(T::with_rgba_alpha_repr(self.1, self.0, src.0), self.1)
            }
        }
    };
}

float_layout!(f32x4, F32x4Backend, 4);
float_layout!(f32x8, F32x8Backend, 8);

#[cfg(feature = "w512")]
mod w512 {
    use crate::simd::{backends::F32x16Backend, generic::f32x16};

    float_layout!(f32x16, F32x16Backend, 16);

    impl<T: F32x16Backend> f32x16<T> {
        /// Load 16 unsigned bytes in order and convert exactly to `f32`.
        #[inline(always)]
        pub fn from_u8_t(token: T, bytes: &[u8; 16]) -> Self {
            Self::from_array_t(token, bytes.map(f32::from))
        }

        /// Round each lane to nearest-even and saturate to `0..=255`. NaN becomes zero.
        #[inline(always)]
        pub fn to_u8(self) -> [u8; 16] {
            self.round()
                .to_u16_trunc_saturating()
                .map(|x| x.min(255) as u8)
        }

        /// Deinterleave 16 packed RGBA pixels into four planes in R, G, B, A order.
        ///
        /// Each input contains four consecutive pixels. All lane bits are preserved.
        #[inline(always)]
        pub fn deinterleave_4ch(rgba: [Self; 4]) -> [Self; 4] {
            let token = rgba[0].1;
            let packed = rgba.map(Self::to_array);
            core::array::from_fn(|c| {
                Self::from_array_t(
                    token,
                    core::array::from_fn(|i| packed[i / 4][4 * (i % 4) + c]),
                )
            })
        }

        /// Interleave R, G, B, A planes into four vectors of four consecutive RGBA pixels.
        ///
        /// All lane bits are preserved.
        #[inline(always)]
        pub fn interleave_4ch(channels: [Self; 4]) -> [Self; 4] {
            let token = channels[0].1;
            let channels = channels.map(Self::to_array);
            core::array::from_fn(|group| {
                Self::from_array_t(
                    token,
                    core::array::from_fn(|i| channels[i % 4][4 * group + i / 4]),
                )
            })
        }

        /// Load 16 packed RGBA byte pixels as four planes with values in `[0.0, 255.0]`.
        #[inline(always)]
        pub fn load_16_rgba_u8_t(token: T, rgba: &[u8; 64]) -> (Self, Self, Self, Self) {
            let (blocks, _) = rgba.as_chunks::<16>();
            let [r, g, b, a] = Self::deinterleave_4ch(core::array::from_fn(|i| {
                Self::from_u8_t(token, &blocks[i])
            }));
            (r, g, b, a)
        }

        /// Store 16 packed RGBA byte pixels from four planes using nearest-even saturation.
        #[inline(always)]
        pub fn store_16_rgba_u8(r: Self, g: Self, b: Self, a: Self) -> [u8; 64] {
            let groups = Self::interleave_4ch([r, g, b, a]).map(Self::to_u8);
            core::array::from_fn(|i| groups[i / 16][i % 16])
        }
    }
}

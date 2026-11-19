#![forbid(unsafe_code)]

use super::{f32x4, f32x8};
use crate::simd::backends::{F32x4Backend, F32x8Backend};
use crate::{HalfConvert, IntConvert};

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

            /// Round finite lanes to nearest-even, saturating to `0..=65535`.
            /// Inputs are in `[-2^31, 2^31)`.
            #[inline(always)]
            pub fn to_u16(self) -> [u16; $lanes] {
                self.to_u16_with::<{ IntConvert::new().bits() }>()
            }

            /// Convert lanes to u16 with the encoded rounding and bounds.
            #[inline(always)]
            pub fn to_u16_with<const CFG: u32>(self) -> [u16; $lanes] {
                const {
                    let _ = IntConvert::decode(CFG);
                }
                T::to_u16_lanes::<CFG>(self.1, self.0)
            }

            /// Convert lanes to u8 with the encoded rounding and bounds.
            #[inline(always)]
            pub fn to_u8_with<const CFG: u32>(self) -> [u8; $lanes] {
                const {
                    let _ = IntConvert::decode(CFG);
                }
                T::to_u8_lanes::<CFG>(self.1, self.0)
            }

            /// Decode binary16 lanes, flushing half subnormals to signed zero.
            #[inline(always)]
            pub fn from_f16_t(token: T, bits: &[u16; $lanes]) -> Self {
                Self::from_f16_with_t::<{ HalfConvert::new().bits() }>(token, bits)
            }

            /// Decode binary16 lanes with the encoded domain and subnormal mode.
            #[inline(always)]
            pub fn from_f16_with_t<const CFG: u32>(token: T, bits: &[u16; $lanes]) -> Self {
                const {
                    let _ = HalfConvert::decode(CFG);
                }
                Self(T::from_f16_lanes::<CFG>(token, bits), token)
            }

            /// Encode to binary16, rounding ties to even and flushing half subnormals.
            #[inline(always)]
            pub fn to_f16_bits(self) -> [u16; $lanes] {
                self.to_f16_bits_with::<{ HalfConvert::new().bits() }>()
            }

            /// Encode to binary16 with the selected domain, subnormal and rounding policy.
            #[inline(always)]
            pub fn to_f16_bits_with<const CFG: u32>(self) -> [u16; $lanes] {
                const {
                    let _ = HalfConvert::decode(CFG);
                }
                T::to_f16_lanes::<CFG>(self.1, self.0)
            }

            /// Widen bfloat16 bit patterns, retaining every input bit.
            #[inline(always)]
            pub fn from_bf16_t(token: T, bits: &[u16; $lanes]) -> Self {
                Self(T::from_bf16_lanes(token, bits), token)
            }

            /// Encode finite bfloat16 lanes, rounding ties to even.
            #[inline(always)]
            pub fn to_bf16(self) -> [u16; $lanes] {
                self.to_bf16_with::<{ HalfConvert::new().normalize().bits() }>()
            }

            /// Encode bfloat16 with the selected domain, subnormal and rounding policy.
            #[inline(always)]
            pub fn to_bf16_with<const CFG: u32>(self) -> [u16; $lanes] {
                const {
                    let _ = HalfConvert::decode(CFG);
                }
                T::to_bf16_lanes::<CFG>(self.1, self.0)
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
    use crate::{HalfConvert, IntConvert};

    float_layout!(f32x16, F32x16Backend, 16);

    impl<T: F32x16Backend> f32x16<T> {
        /// Load 16 unsigned bytes in order and convert exactly to `f32`.
        #[inline(always)]
        pub fn from_u8_t(token: T, bytes: &[u8; 16]) -> Self {
            Self(T::from_u8_lanes_wide(token, bytes), token)
        }

        /// Round each lane to nearest-even and saturate to `0..=255`. NaN becomes zero.
        #[inline(always)]
        pub fn to_u8(self) -> [u8; 16] {
            self.to_u8_with::<{ IntConvert::new().saturate_all().bits() }>()
        }

        /// Deinterleave 16 packed RGBA pixels into four planes in R, G, B, A order.
        ///
        /// Each input contains four consecutive pixels. All lane bits are preserved.
        #[inline(always)]
        pub fn deinterleave_4ch(rgba: [Self; 4]) -> [Self; 4] {
            let token = rgba[0].1;
            T::deinterleave_rgba_wide(token, rgba.map(Self::into_repr)).map(|x| Self(x, token))
        }

        /// Interleave R, G, B, A planes into four vectors of four consecutive RGBA pixels.
        ///
        /// All lane bits are preserved.
        #[inline(always)]
        pub fn interleave_4ch(channels: [Self; 4]) -> [Self; 4] {
            let token = channels[0].1;
            T::interleave_rgba_wide(token, channels.map(Self::into_repr)).map(|x| Self(x, token))
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
            Self::store_16_rgba_u8_with::<{ IntConvert::new().saturate_all().bits() }>(r, g, b, a)
        }

        /// Store 16 RGBA pixels with the configured rounding and bounds.
        #[inline(always)]
        pub fn store_16_rgba_u8_with<const CFG: u32>(
            r: Self,
            g: Self,
            b: Self,
            a: Self,
        ) -> [u8; 64] {
            let groups = Self::interleave_4ch([r, g, b, a]).map(Self::to_u8_with::<CFG>);
            crate::simd_storage::cast(groups)
        }
    }
}

#[cfg(any(target_arch = "aarch64", target_arch = "wasm32"))]
pub(crate) mod half4 {
    use super::super::{f32x4, i32x4};
    use crate::HalfConvert;
    use crate::simd::backends::F32x4Convert;

    #[inline(always)]
    #[cfg_attr(target_arch = "aarch64", rustversion::before(1.94))]
    pub(crate) fn widen_f16<const CFG: u32, T: F32x4Convert>(bits: i32x4<T>) -> f32x4<T> {
        let opts = const { HalfConvert::decode(CFG) };
        if !opts.flushes() {
            return bits.f16_to_f32();
        }
        let splat = |x| i32x4::splat_t(bits.token(), x);
        let mag = bits & splat(0x7fff);
        let sign = (bits & splat(0x8000)).shl_const::<16>();
        let shifted = mag.shl_const::<13>();
        let finite = (shifted + splat(112 << 23)) & mag.simd_ge(splat(0x400));
        let body = if opts.is_finite() {
            finite
        } else {
            let special = (shifted & splat(0x007f_ffff)) | splat(0x7f80_0000);
            i32x4::blend(mag.simd_ge(splat(0x7c00)), special, finite)
        };
        (body | sign).bitcast_to_f32()
    }

    #[inline(always)]
    pub(crate) fn narrow_f16<const CFG: u32, T: F32x4Convert>(value: f32x4<T>) -> i32x4<T> {
        let opts = const { HalfConvert::decode(CFG) };
        let splat = |x| i32x4::splat_t(value.token(), x);
        if opts.flushes() && opts.is_finite() {
            let bits = value.bitcast_to_i32();
            let mag = bits & splat(0x7fff_ffff);
            let sign = bits.shr_logical_const::<16>() & splat(0x8000);
            let (low, cap, rounded) = if opts.rounds() {
                let odd = mag.shr_logical_const::<13>() & splat(1);
                (0x387f_e000, 0x7c00, mag + splat(0xfff) + odd)
            } else {
                (0x3880_0000, 0x7bff, mag)
            };
            let packed = (rounded.shr_logical_const::<13>() - splat(112 << 10)).min(splat(cap));
            return (packed & mag.simd_ge(splat(low))) | sign;
        }
        if opts.rounds() {
            let half = value.to_f16();
            if opts.flushes() {
                let normal = (half & splat(0x7c00)).simd_ne(splat(0));
                half & (normal | splat(0x8000))
            } else {
                half
            }
        } else {
            i32x4::from_array_t(
                value.token(),
                value
                    .to_array()
                    .map(|x| i32::from(crate::scalar::to_f16_bits_with::<CFG>(x))),
            )
        }
    }

    #[inline(always)]
    pub(crate) fn narrow_bf16<const CFG: u32, T: F32x4Convert>(value: f32x4<T>) -> i32x4<T> {
        let opts = const { HalfConvert::decode(CFG) };
        let splat = |x| i32x4::splat_t(value.token(), x);
        let bits = value.bitcast_to_i32();
        let rounded = if opts.rounds() {
            let odd = bits.shr_logical_const::<16>() & splat(1);
            bits + splat(0x7fff) + odd
        } else {
            bits
        };
        let rounded = if opts.is_finite() {
            rounded
        } else {
            let is_nan = (bits & splat(0x7fff_ffff)).simd_gt(splat(0x7f80_0000));
            i32x4::blend(is_nan, bits | splat(0x0040_0000), rounded)
        };
        let half = rounded.shr_logical_const::<16>();
        if opts.flushes() {
            let normal = (half & splat(0x7f80)).simd_ne(splat(0));
            half & (normal | splat(0x8000))
        } else {
            half
        }
    }
}

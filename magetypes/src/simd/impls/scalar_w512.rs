//! W512 backend implementations for ScalarToken (fallback).
//!
//! All operations are plain array math. Gated on `feature = "w512"` at
//! the module declaration in `impls/mod.rs`.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

#![allow(
    clippy::needless_range_loop,
    clippy::identity_op,
    clippy::collapsible_else_if,
    clippy::unnecessary_cast
)]

use crate::simd::backends::*;

impl F32x16Convert for archmage::ScalarToken {
    #[inline(always)]
    fn bitcast_f32_to_i32(self, a: [f32; 16]) -> [i32; 16] {
        a.map(|x| x.to_bits() as i32)
    }

    #[inline(always)]
    fn bitcast_i32_to_f32(self, a: [i32; 16]) -> [f32; 16] {
        a.map(|x| f32::from_bits(x as u32))
    }

    #[inline(always)]
    fn convert_f32_to_i32(self, a: [f32; 16]) -> [i32; 16] {
        a.map(|x| x as i32)
    }

    #[inline(always)]
    fn convert_f32_to_i32_round(self, a: [f32; 16]) -> [i32; 16] {
        a.map(|x| crate::scalar::round_f32(x) as i32)
    }

    #[inline(always)]
    fn convert_i32_to_f32(self, a: [i32; 16]) -> [f32; 16] {
        a.map(|x| x as f32)
    }
}

impl F32x16Backend for archmage::ScalarToken {
    type Repr = [f32; 16];

    #[inline(always)]
    fn splat(self, v: f32) -> [f32; 16] {
        [0.0f32; 16].map(|_| v)
    }

    #[inline(always)]
    fn zero(self) -> [f32; 16] {
        [0.0f32; 16]
    }

    #[inline(always)]
    fn load(self, data: &[f32; 16]) -> [f32; 16] {
        *data
    }

    #[inline(always)]
    fn from_array(self, arr: [f32; 16]) -> [f32; 16] {
        arr
    }

    #[inline(always)]
    fn store(self, repr: [f32; 16], out: &mut [f32; 16]) {
        *out = repr;
    }

    #[inline(always)]
    fn to_array(self, repr: [f32; 16]) -> [f32; 16] {
        repr
    }

    #[inline(always)]
    fn add(self, a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
        core::array::from_fn(|i| a[i] + b[i])
    }

    #[inline(always)]
    fn sub(self, a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
        core::array::from_fn(|i| a[i] - b[i])
    }

    #[inline(always)]
    fn mul(self, a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
        core::array::from_fn(|i| a[i] * b[i])
    }

    #[inline(always)]
    fn div(self, a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
        core::array::from_fn(|i| a[i] / b[i])
    }

    #[inline(always)]
    fn neg(self, a: [f32; 16]) -> [f32; 16] {
        core::array::from_fn(|i| -a[i])
    }

    #[inline(always)]
    fn min(self, a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
        core::array::from_fn(|i| a[i].min(b[i]))
    }

    #[inline(always)]
    fn max(self, a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
        core::array::from_fn(|i| a[i].max(b[i]))
    }

    #[inline(always)]
    fn minimum(self, a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
        core::array::from_fn(|i| crate::scalar_extrema::minimumf(a[i], b[i]))
    }

    #[inline(always)]
    fn maximum(self, a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
        core::array::from_fn(|i| crate::scalar_extrema::maximumf(a[i], b[i]))
    }

    #[inline(always)]
    fn min_fast(self, a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
        core::array::from_fn(|i| if a[i] < b[i] { a[i] } else { b[i] })
    }

    #[inline(always)]
    fn max_fast(self, a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
        core::array::from_fn(|i| if a[i] > b[i] { a[i] } else { b[i] })
    }

    #[inline(always)]
    fn sqrt(self, a: [f32; 16]) -> [f32; 16] {
        core::array::from_fn(|i| crate::scalar_sqrt::sqrtf(a[i]))
    }

    #[inline(always)]
    fn rcp_approx(self, a: [f32; 16]) -> [f32; 16] {
        core::array::from_fn(|i| 1.0 / a[i])
    }
    #[inline(always)]
    fn rsqrt_approx(self, a: [f32; 16]) -> [f32; 16] {
        core::array::from_fn(|i| {
            let x = a[i];
            let seed = f32::from_bits(0x5f37_59df_u32.wrapping_sub(x.to_bits() >> 1));
            let hx = 0.5 * x;
            let y = seed * (1.5 - hx * seed * seed);
            y * (1.5 - hx * y * y)
        })
    }
    #[inline(always)]
    fn recip(self, a: [f32; 16]) -> [f32; 16] {
        core::array::from_fn(|i| 1.0 / a[i])
    }
    #[inline(always)]
    fn rsqrt(self, a: [f32; 16]) -> [f32; 16] {
        core::array::from_fn(|i| 1.0 / crate::scalar_sqrt::sqrtf(a[i]))
    }

    #[inline(always)]
    fn abs(self, a: [f32; 16]) -> [f32; 16] {
        core::array::from_fn(|i| f32::from_bits(a[i].to_bits() & 0x7FFF_FFFF))
    }

    #[inline(always)]
    fn floor(self, a: [f32; 16]) -> [f32; 16] {
        a.map(crate::scalar::floor_f32)
    }

    #[inline(always)]
    fn ceil(self, a: [f32; 16]) -> [f32; 16] {
        a.map(crate::scalar::ceil_f32)
    }

    #[inline(always)]
    fn round(self, a: [f32; 16]) -> [f32; 16] {
        a.map(crate::scalar::round_f32)
    }

    #[inline(always)]
    fn mul_add(self, a: [f32; 16], b: [f32; 16], c: [f32; 16]) -> [f32; 16] {
        core::array::from_fn(|i| a[i] * b[i] + c[i])
    }

    #[inline(always)]
    fn mul_sub(self, a: [f32; 16], b: [f32; 16], c: [f32; 16]) -> [f32; 16] {
        core::array::from_fn(|i| a[i] * b[i] - c[i])
    }

    #[inline(always)]
    fn mul_add_portable(self, a: [f32; 16], b: [f32; 16], c: [f32; 16]) -> [f32; 16] {
        crate::soft_fma::fmaf_lanes(a, b, c)
    }

    #[inline(always)]
    fn mul_sub_portable(self, a: [f32; 16], b: [f32; 16], c: [f32; 16]) -> [f32; 16] {
        crate::soft_fma::fmaf_lanes(
            a,
            b,
            [
                -c[0], -c[1], -c[2], -c[3], -c[4], -c[5], -c[6], -c[7], -c[8], -c[9], -c[10],
                -c[11], -c[12], -c[13], -c[14], -c[15],
            ],
        )
    }

    #[inline(always)]
    fn simd_eq(self, a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
        core::array::from_fn(|i| {
            if a[i] == b[i] {
                f32::from_bits(!0u32)
            } else {
                0.0f32
            }
        })
    }

    #[inline(always)]
    fn simd_ne(self, a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
        core::array::from_fn(|i| {
            if a[i] != b[i] {
                f32::from_bits(!0u32)
            } else {
                0.0f32
            }
        })
    }

    #[inline(always)]
    fn simd_lt(self, a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
        core::array::from_fn(|i| {
            if a[i] < b[i] {
                f32::from_bits(!0u32)
            } else {
                0.0f32
            }
        })
    }

    #[inline(always)]
    fn simd_le(self, a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
        core::array::from_fn(|i| {
            if a[i] <= b[i] {
                f32::from_bits(!0u32)
            } else {
                0.0f32
            }
        })
    }

    #[inline(always)]
    fn simd_gt(self, a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
        core::array::from_fn(|i| {
            if a[i] > b[i] {
                f32::from_bits(!0u32)
            } else {
                0.0f32
            }
        })
    }

    #[inline(always)]
    fn simd_ge(self, a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
        core::array::from_fn(|i| {
            if a[i] >= b[i] {
                f32::from_bits(!0u32)
            } else {
                0.0f32
            }
        })
    }

    #[inline(always)]
    fn blend(self, mask: [f32; 16], if_true: [f32; 16], if_false: [f32; 16]) -> [f32; 16] {
        core::array::from_fn(|i| {
            if mask[i].to_bits() != 0 {
                if_true[i]
            } else {
                if_false[i]
            }
        })
    }

    #[inline(always)]
    fn reduce_add(self, a: [f32; 16]) -> f32 {
        a.iter().sum()
    }

    #[inline(always)]
    fn reduce_min(self, a: [f32; 16]) -> f32 {
        a.iter().copied().fold(f32::INFINITY, f32::min)
    }

    #[inline(always)]
    fn reduce_max(self, a: [f32; 16]) -> f32 {
        a.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    }

    #[inline(always)]
    fn not(self, a: [f32; 16]) -> [f32; 16] {
        core::array::from_fn(|i| f32::from_bits(!a[i].to_bits()))
    }

    #[inline(always)]
    fn bitand(self, a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
        core::array::from_fn(|i| f32::from_bits(a[i].to_bits() & b[i].to_bits()))
    }

    #[inline(always)]
    fn bitor(self, a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
        core::array::from_fn(|i| f32::from_bits(a[i].to_bits() | b[i].to_bits()))
    }

    #[inline(always)]
    fn bitxor(self, a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
        core::array::from_fn(|i| f32::from_bits(a[i].to_bits() ^ b[i].to_bits()))
    }
}
impl F32x16Internal for archmage::ScalarToken {
    type InternalRepr = [f32; 16];

    #[inline(always)]
    fn from_u16_lanes(self, lanes: &[u16; 16]) -> [f32; 16] {
        lanes.map(f32::from)
    }

    #[inline(always)]
    fn to_u16_lanes<const CFG: u32>(self, value: [f32; 16]) -> [u16; 16] {
        value.map(crate::scalar::to_u16_with::<CFG>)
    }

    #[inline(always)]
    fn to_u8_lanes<const CFG: u32>(self, value: [f32; 16]) -> [u8; 16] {
        value.map(crate::scalar::to_u8_with::<CFG>)
    }

    #[inline(always)]
    fn to_u32_lanes<const CFG: u32>(self, value: [f32; 16]) -> [f32; 16] {
        value.map(|x| f32::from_bits(crate::scalar::to_u32_with::<CFG>(x)))
    }

    #[inline(always)]
    fn from_u32_lanes(self, value: [f32; 16]) -> [f32; 16] {
        value.map(|x| x.to_bits() as f32)
    }

    #[inline(always)]
    fn from_f16_lanes<const CFG: u32>(self, lanes: &[u16; 16]) -> [f32; 16] {
        lanes.map(crate::scalar::from_f16_with::<CFG>)
    }

    #[inline(always)]
    fn to_f16_lanes<const CFG: u32>(self, value: [f32; 16]) -> [u16; 16] {
        value.map(crate::scalar::to_f16_bits_with::<CFG>)
    }

    #[inline(always)]
    fn from_bf16_lanes(self, lanes: &[u16; 16]) -> [f32; 16] {
        lanes.map(crate::scalar::from_bf16)
    }

    #[inline(always)]
    fn to_bf16_lanes<const CFG: u32>(self, value: [f32; 16]) -> [u16; 16] {
        value.map(crate::scalar::to_bf16_with::<CFG>)
    }

    #[inline(always)]
    fn broadcast_rgba_alpha_repr(self, value: [f32; 16]) -> [f32; 16] {
        core::array::from_fn(|i| value[i | 3])
    }

    #[inline(always)]
    fn with_rgba_alpha_repr(self, mut value: [f32; 16], src: [f32; 16]) -> [f32; 16] {
        for i in (3..16).step_by(4) {
            value[i] = src[i];
        }
        value
    }

    #[inline]
    fn transpose_16x16_repr(self, rows: [[f32; 16]; 16]) -> [[f32; 16]; 16] {
        let mut out = [[0.0; 16]; 16];
        for (i, row) in rows.iter().enumerate() {
            for (col, &lane) in out.iter_mut().zip(row) {
                col[i & 15] = lane;
            }
        }
        out
    }

    #[inline]
    fn deinterleave_3ch_repr(self, rgb: [[f32; 16]; 3]) -> [[f32; 16]; 3] {
        let [mut r, mut g, mut b] = [[0.0; 16]; 3];
        for (i, &[red, green, blue]) in rgb.as_flattened().as_chunks::<3>().0.iter().enumerate() {
            r[i] = red;
            g[i] = green;
            b[i] = blue;
        }
        [r, g, b]
    }

    #[inline]
    fn interleave_3ch_repr(self, [r, g, b]: [[f32; 16]; 3]) -> [[f32; 16]; 3] {
        let mut rgb = [[0.0; 16]; 3];
        for (i, pixel) in rgb
            .as_flattened_mut()
            .as_chunks_mut::<3>()
            .0
            .iter_mut()
            .enumerate()
        {
            *pixel = [r[i], g[i], b[i]];
        }
        rgb
    }

    #[inline(always)]
    fn from_u8_lanes_wide(self, lanes: &[u8; 16]) -> [f32; 16] {
        lanes.map(f32::from)
    }

    #[inline(always)]
    fn deinterleave_rgba_wide(self, rgba: [[f32; 16]; 4]) -> [[f32; 16]; 4] {
        core::array::from_fn(|c| core::array::from_fn(|i| rgba[i / 4][4 * (i % 4) + c]))
    }

    #[inline(always)]
    fn interleave_rgba_wide(self, channels: [[f32; 16]; 4]) -> [[f32; 16]; 4] {
        core::array::from_fn(|group| core::array::from_fn(|i| channels[i % 4][4 * group + i / 4]))
    }
}

impl F64x8Backend for archmage::ScalarToken {
    type Repr = [f64; 8];

    #[inline(always)]
    fn splat(self, v: f64) -> [f64; 8] {
        [0.0f64; 8].map(|_| v)
    }

    #[inline(always)]
    fn zero(self) -> [f64; 8] {
        [0.0f64; 8]
    }

    #[inline(always)]
    fn load(self, data: &[f64; 8]) -> [f64; 8] {
        *data
    }

    #[inline(always)]
    fn from_array(self, arr: [f64; 8]) -> [f64; 8] {
        arr
    }

    #[inline(always)]
    fn store(self, repr: [f64; 8], out: &mut [f64; 8]) {
        *out = repr;
    }

    #[inline(always)]
    fn to_array(self, repr: [f64; 8]) -> [f64; 8] {
        repr
    }

    #[inline(always)]
    fn add(self, a: [f64; 8], b: [f64; 8]) -> [f64; 8] {
        core::array::from_fn(|i| a[i] + b[i])
    }

    #[inline(always)]
    fn sub(self, a: [f64; 8], b: [f64; 8]) -> [f64; 8] {
        core::array::from_fn(|i| a[i] - b[i])
    }

    #[inline(always)]
    fn mul(self, a: [f64; 8], b: [f64; 8]) -> [f64; 8] {
        core::array::from_fn(|i| a[i] * b[i])
    }

    #[inline(always)]
    fn div(self, a: [f64; 8], b: [f64; 8]) -> [f64; 8] {
        core::array::from_fn(|i| a[i] / b[i])
    }

    #[inline(always)]
    fn neg(self, a: [f64; 8]) -> [f64; 8] {
        core::array::from_fn(|i| -a[i])
    }

    #[inline(always)]
    fn min(self, a: [f64; 8], b: [f64; 8]) -> [f64; 8] {
        core::array::from_fn(|i| a[i].min(b[i]))
    }

    #[inline(always)]
    fn max(self, a: [f64; 8], b: [f64; 8]) -> [f64; 8] {
        core::array::from_fn(|i| a[i].max(b[i]))
    }

    #[inline(always)]
    fn minimum(self, a: [f64; 8], b: [f64; 8]) -> [f64; 8] {
        core::array::from_fn(|i| crate::scalar_extrema::minimum(a[i], b[i]))
    }

    #[inline(always)]
    fn maximum(self, a: [f64; 8], b: [f64; 8]) -> [f64; 8] {
        core::array::from_fn(|i| crate::scalar_extrema::maximum(a[i], b[i]))
    }

    #[inline(always)]
    fn min_fast(self, a: [f64; 8], b: [f64; 8]) -> [f64; 8] {
        core::array::from_fn(|i| if a[i] < b[i] { a[i] } else { b[i] })
    }

    #[inline(always)]
    fn max_fast(self, a: [f64; 8], b: [f64; 8]) -> [f64; 8] {
        core::array::from_fn(|i| if a[i] > b[i] { a[i] } else { b[i] })
    }

    #[inline(always)]
    fn sqrt(self, a: [f64; 8]) -> [f64; 8] {
        core::array::from_fn(|i| crate::scalar_sqrt::sqrt(a[i]))
    }

    #[inline(always)]
    fn rcp_approx(self, a: [f64; 8]) -> [f64; 8] {
        core::array::from_fn(|i| 1.0 / a[i])
    }
    #[inline(always)]
    fn rsqrt_approx(self, a: [f64; 8]) -> [f64; 8] {
        core::array::from_fn(|i| 1.0 / crate::scalar_sqrt::sqrt(a[i]))
    }
    #[inline(always)]
    fn recip(self, a: [f64; 8]) -> [f64; 8] {
        core::array::from_fn(|i| 1.0 / a[i])
    }
    #[inline(always)]
    fn rsqrt(self, a: [f64; 8]) -> [f64; 8] {
        core::array::from_fn(|i| 1.0 / crate::scalar_sqrt::sqrt(a[i]))
    }

    #[inline(always)]
    fn abs(self, a: [f64; 8]) -> [f64; 8] {
        core::array::from_fn(|i| f64::from_bits(a[i].to_bits() & 0x7FFF_FFFF_FFFF_FFFF))
    }

    #[inline(always)]
    fn floor(self, a: [f64; 8]) -> [f64; 8] {
        a.map(crate::scalar::floor_f64)
    }

    #[inline(always)]
    fn ceil(self, a: [f64; 8]) -> [f64; 8] {
        a.map(crate::scalar::ceil_f64)
    }

    #[inline(always)]
    fn round(self, a: [f64; 8]) -> [f64; 8] {
        a.map(crate::scalar::round_f64)
    }

    #[inline(always)]
    fn mul_add(self, a: [f64; 8], b: [f64; 8], c: [f64; 8]) -> [f64; 8] {
        core::array::from_fn(|i| a[i] * b[i] + c[i])
    }

    #[inline(always)]
    fn mul_sub(self, a: [f64; 8], b: [f64; 8], c: [f64; 8]) -> [f64; 8] {
        core::array::from_fn(|i| a[i] * b[i] - c[i])
    }

    #[inline(always)]
    fn mul_add_portable(self, a: [f64; 8], b: [f64; 8], c: [f64; 8]) -> [f64; 8] {
        crate::soft_fma::fma_lanes(a, b, c)
    }

    #[inline(always)]
    fn mul_sub_portable(self, a: [f64; 8], b: [f64; 8], c: [f64; 8]) -> [f64; 8] {
        crate::soft_fma::fma_lanes(
            a,
            b,
            [-c[0], -c[1], -c[2], -c[3], -c[4], -c[5], -c[6], -c[7]],
        )
    }

    #[inline(always)]
    fn simd_eq(self, a: [f64; 8], b: [f64; 8]) -> [f64; 8] {
        core::array::from_fn(|i| {
            if a[i] == b[i] {
                f64::from_bits(!0u64)
            } else {
                0.0f64
            }
        })
    }

    #[inline(always)]
    fn simd_ne(self, a: [f64; 8], b: [f64; 8]) -> [f64; 8] {
        core::array::from_fn(|i| {
            if a[i] != b[i] {
                f64::from_bits(!0u64)
            } else {
                0.0f64
            }
        })
    }

    #[inline(always)]
    fn simd_lt(self, a: [f64; 8], b: [f64; 8]) -> [f64; 8] {
        core::array::from_fn(|i| {
            if a[i] < b[i] {
                f64::from_bits(!0u64)
            } else {
                0.0f64
            }
        })
    }

    #[inline(always)]
    fn simd_le(self, a: [f64; 8], b: [f64; 8]) -> [f64; 8] {
        core::array::from_fn(|i| {
            if a[i] <= b[i] {
                f64::from_bits(!0u64)
            } else {
                0.0f64
            }
        })
    }

    #[inline(always)]
    fn simd_gt(self, a: [f64; 8], b: [f64; 8]) -> [f64; 8] {
        core::array::from_fn(|i| {
            if a[i] > b[i] {
                f64::from_bits(!0u64)
            } else {
                0.0f64
            }
        })
    }

    #[inline(always)]
    fn simd_ge(self, a: [f64; 8], b: [f64; 8]) -> [f64; 8] {
        core::array::from_fn(|i| {
            if a[i] >= b[i] {
                f64::from_bits(!0u64)
            } else {
                0.0f64
            }
        })
    }

    #[inline(always)]
    fn blend(self, mask: [f64; 8], if_true: [f64; 8], if_false: [f64; 8]) -> [f64; 8] {
        core::array::from_fn(|i| {
            if mask[i].to_bits() != 0 {
                if_true[i]
            } else {
                if_false[i]
            }
        })
    }

    #[inline(always)]
    fn reduce_add(self, a: [f64; 8]) -> f64 {
        a.iter().sum()
    }

    #[inline(always)]
    fn reduce_min(self, a: [f64; 8]) -> f64 {
        a.iter().copied().fold(f64::INFINITY, f64::min)
    }

    #[inline(always)]
    fn reduce_max(self, a: [f64; 8]) -> f64 {
        a.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    }

    #[inline(always)]
    fn not(self, a: [f64; 8]) -> [f64; 8] {
        core::array::from_fn(|i| f64::from_bits(!a[i].to_bits()))
    }

    #[inline(always)]
    fn bitand(self, a: [f64; 8], b: [f64; 8]) -> [f64; 8] {
        core::array::from_fn(|i| f64::from_bits(a[i].to_bits() & b[i].to_bits()))
    }

    #[inline(always)]
    fn bitor(self, a: [f64; 8], b: [f64; 8]) -> [f64; 8] {
        core::array::from_fn(|i| f64::from_bits(a[i].to_bits() | b[i].to_bits()))
    }

    #[inline(always)]
    fn bitxor(self, a: [f64; 8], b: [f64; 8]) -> [f64; 8] {
        core::array::from_fn(|i| f64::from_bits(a[i].to_bits() ^ b[i].to_bits()))
    }
}

impl I8x64Backend for archmage::ScalarToken {
    type Repr = [i8; 64];

    #[inline(always)]
    fn splat(self, v: i8) -> [i8; 64] {
        [v; 64]
    }

    #[inline(always)]
    fn zero(self) -> [i8; 64] {
        [0; 64]
    }

    #[inline(always)]
    fn load(self, data: &[i8; 64]) -> [i8; 64] {
        *data
    }

    #[inline(always)]
    fn from_array(self, arr: [i8; 64]) -> [i8; 64] {
        arr
    }

    #[inline(always)]
    fn store(self, repr: [i8; 64], out: &mut [i8; 64]) {
        *out = repr;
    }

    #[inline(always)]
    fn to_array(self, repr: [i8; 64]) -> [i8; 64] {
        repr
    }

    #[inline(always)]
    fn add(self, a: [i8; 64], b: [i8; 64]) -> [i8; 64] {
        core::array::from_fn(|i| a[i].wrapping_add(b[i]))
    }

    #[inline(always)]
    fn sub(self, a: [i8; 64], b: [i8; 64]) -> [i8; 64] {
        core::array::from_fn(|i| a[i].wrapping_sub(b[i]))
    }

    #[inline(always)]
    fn neg(self, a: [i8; 64]) -> [i8; 64] {
        core::array::from_fn(|i| a[i].wrapping_neg())
    }

    #[inline(always)]
    fn min(self, a: [i8; 64], b: [i8; 64]) -> [i8; 64] {
        core::array::from_fn(|i| if a[i] < b[i] { a[i] } else { b[i] })
    }

    #[inline(always)]
    fn max(self, a: [i8; 64], b: [i8; 64]) -> [i8; 64] {
        core::array::from_fn(|i| if a[i] > b[i] { a[i] } else { b[i] })
    }

    #[inline(always)]
    fn abs(self, a: [i8; 64]) -> [i8; 64] {
        core::array::from_fn(|i| a[i].wrapping_abs())
    }

    #[inline(always)]
    fn simd_eq(self, a: [i8; 64], b: [i8; 64]) -> [i8; 64] {
        core::array::from_fn(|i| if a[i] == b[i] { !0i8 } else { 0 })
    }

    #[inline(always)]
    fn simd_ne(self, a: [i8; 64], b: [i8; 64]) -> [i8; 64] {
        core::array::from_fn(|i| if a[i] != b[i] { !0i8 } else { 0 })
    }

    #[inline(always)]
    fn simd_lt(self, a: [i8; 64], b: [i8; 64]) -> [i8; 64] {
        core::array::from_fn(|i| if a[i] < b[i] { !0i8 } else { 0 })
    }

    #[inline(always)]
    fn simd_le(self, a: [i8; 64], b: [i8; 64]) -> [i8; 64] {
        core::array::from_fn(|i| if a[i] <= b[i] { !0i8 } else { 0 })
    }

    #[inline(always)]
    fn simd_gt(self, a: [i8; 64], b: [i8; 64]) -> [i8; 64] {
        core::array::from_fn(|i| if a[i] > b[i] { !0i8 } else { 0 })
    }

    #[inline(always)]
    fn simd_ge(self, a: [i8; 64], b: [i8; 64]) -> [i8; 64] {
        core::array::from_fn(|i| if a[i] >= b[i] { !0i8 } else { 0 })
    }

    #[inline(always)]
    fn blend(self, mask: [i8; 64], if_true: [i8; 64], if_false: [i8; 64]) -> [i8; 64] {
        core::array::from_fn(|i| {
            if mask[i] != 0 {
                if_true[i]
            } else {
                if_false[i]
            }
        })
    }

    #[inline(always)]
    fn reduce_add(self, a: [i8; 64]) -> i8 {
        a.iter().copied().fold(0i8, i8::wrapping_add)
    }

    #[inline(always)]
    fn not(self, a: [i8; 64]) -> [i8; 64] {
        core::array::from_fn(|i| !a[i])
    }

    #[inline(always)]
    fn bitand(self, a: [i8; 64], b: [i8; 64]) -> [i8; 64] {
        core::array::from_fn(|i| a[i] & b[i])
    }

    #[inline(always)]
    fn bitor(self, a: [i8; 64], b: [i8; 64]) -> [i8; 64] {
        core::array::from_fn(|i| a[i] | b[i])
    }

    #[inline(always)]
    fn bitxor(self, a: [i8; 64], b: [i8; 64]) -> [i8; 64] {
        core::array::from_fn(|i| a[i] ^ b[i])
    }

    #[inline(always)]
    fn shl_const<const N: i32>(self, a: [i8; 64]) -> [i8; 64] {
        core::array::from_fn(|i| {
            if N < 8 {
                (a[i] as u8).wrapping_shl(N as u32) as i8
            } else {
                0
            }
        })
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(self, a: [i8; 64]) -> [i8; 64] {
        core::array::from_fn(|i| {
            if N < 8 {
                a[i].wrapping_shr(N as u32)
            } else if a[i] < 0 {
                -1
            } else {
                0
            }
        })
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: [i8; 64]) -> [i8; 64] {
        core::array::from_fn(|i| {
            if N < 8 {
                (a[i] as u8).wrapping_shr(N as u32) as i8
            } else {
                0
            }
        })
    }

    #[inline(always)]
    fn saturating_add(self, a: [i8; 64], b: [i8; 64]) -> [i8; 64] {
        core::array::from_fn(|i| a[i].saturating_add(b[i]))
    }

    #[inline(always)]
    fn saturating_sub(self, a: [i8; 64], b: [i8; 64]) -> [i8; 64] {
        core::array::from_fn(|i| a[i].saturating_sub(b[i]))
    }

    #[inline(always)]
    fn all_true(self, a: [i8; 64]) -> bool {
        a.iter().all(|&v| v < 0)
    }

    #[inline(always)]
    fn any_true(self, a: [i8; 64]) -> bool {
        a.iter().any(|&v| v < 0)
    }

    #[inline(always)]
    fn bitmask(self, a: [i8; 64]) -> u64 {
        let mut mask = 0u64;
        for i in 0..64 {
            if a[i] < 0 {
                mask |= 1u64 << i;
            }
        }
        mask
    }
    #[inline(always)]
    fn widen_low_i8_to_i16(self, a: [i8; 64]) -> [i16; 32] {
        // Widen all lanes, keep the low half: dodges SROA integer
        // promotion of a half-array read (#77); the high half is DCE'd.
        let f: [i16; 64] = core::array::from_fn(|i| a[i] as i16);
        core::array::from_fn(|i| f[i])
    }

    #[inline(always)]
    fn widen_high_i8_to_i16(self, a: [i8; 64]) -> [i16; 32] {
        // Widen all lanes, keep the high half: dodges SROA integer
        // promotion of a half-array read (#77); the low half is DCE'd.
        let f: [i16; 64] = core::array::from_fn(|i| a[i] as i16);
        core::array::from_fn(|i| f[i + 32])
    }
}

impl U8x64Backend for archmage::ScalarToken {
    type Repr = [u8; 64];

    #[inline(always)]
    fn splat(self, v: u8) -> [u8; 64] {
        [v; 64]
    }

    #[inline(always)]
    fn zero(self) -> [u8; 64] {
        [0; 64]
    }

    #[inline(always)]
    fn load(self, data: &[u8; 64]) -> [u8; 64] {
        *data
    }

    #[inline(always)]
    fn from_array(self, arr: [u8; 64]) -> [u8; 64] {
        arr
    }

    #[inline(always)]
    fn store(self, repr: [u8; 64], out: &mut [u8; 64]) {
        *out = repr;
    }

    #[inline(always)]
    fn to_array(self, repr: [u8; 64]) -> [u8; 64] {
        repr
    }

    #[inline(always)]
    fn add(self, a: [u8; 64], b: [u8; 64]) -> [u8; 64] {
        core::array::from_fn(|i| a[i].wrapping_add(b[i]))
    }

    #[inline(always)]
    fn sub(self, a: [u8; 64], b: [u8; 64]) -> [u8; 64] {
        core::array::from_fn(|i| a[i].wrapping_sub(b[i]))
    }

    #[inline(always)]
    fn neg(self, a: [u8; 64]) -> [u8; 64] {
        core::array::from_fn(|i| (0u8).wrapping_sub(a[i]))
    }

    #[inline(always)]
    fn min(self, a: [u8; 64], b: [u8; 64]) -> [u8; 64] {
        core::array::from_fn(|i| if a[i] < b[i] { a[i] } else { b[i] })
    }

    #[inline(always)]
    fn max(self, a: [u8; 64], b: [u8; 64]) -> [u8; 64] {
        core::array::from_fn(|i| if a[i] > b[i] { a[i] } else { b[i] })
    }

    #[inline(always)]
    fn simd_eq(self, a: [u8; 64], b: [u8; 64]) -> [u8; 64] {
        core::array::from_fn(|i| if a[i] == b[i] { !0u8 } else { 0 })
    }

    #[inline(always)]
    fn simd_ne(self, a: [u8; 64], b: [u8; 64]) -> [u8; 64] {
        core::array::from_fn(|i| if a[i] != b[i] { !0u8 } else { 0 })
    }

    #[inline(always)]
    fn simd_lt(self, a: [u8; 64], b: [u8; 64]) -> [u8; 64] {
        core::array::from_fn(|i| if a[i] < b[i] { !0u8 } else { 0 })
    }

    #[inline(always)]
    fn simd_le(self, a: [u8; 64], b: [u8; 64]) -> [u8; 64] {
        core::array::from_fn(|i| if a[i] <= b[i] { !0u8 } else { 0 })
    }

    #[inline(always)]
    fn simd_gt(self, a: [u8; 64], b: [u8; 64]) -> [u8; 64] {
        core::array::from_fn(|i| if a[i] > b[i] { !0u8 } else { 0 })
    }

    #[inline(always)]
    fn simd_ge(self, a: [u8; 64], b: [u8; 64]) -> [u8; 64] {
        core::array::from_fn(|i| if a[i] >= b[i] { !0u8 } else { 0 })
    }

    #[inline(always)]
    fn blend(self, mask: [u8; 64], if_true: [u8; 64], if_false: [u8; 64]) -> [u8; 64] {
        core::array::from_fn(|i| {
            if mask[i] != 0 {
                if_true[i]
            } else {
                if_false[i]
            }
        })
    }

    #[inline(always)]
    fn reduce_add(self, a: [u8; 64]) -> u8 {
        a.iter().copied().fold(0u8, u8::wrapping_add)
    }

    #[inline(always)]
    fn not(self, a: [u8; 64]) -> [u8; 64] {
        core::array::from_fn(|i| !a[i])
    }

    #[inline(always)]
    fn bitand(self, a: [u8; 64], b: [u8; 64]) -> [u8; 64] {
        core::array::from_fn(|i| a[i] & b[i])
    }

    #[inline(always)]
    fn bitor(self, a: [u8; 64], b: [u8; 64]) -> [u8; 64] {
        core::array::from_fn(|i| a[i] | b[i])
    }

    #[inline(always)]
    fn bitxor(self, a: [u8; 64], b: [u8; 64]) -> [u8; 64] {
        core::array::from_fn(|i| a[i] ^ b[i])
    }

    #[inline(always)]
    fn shl_const<const N: i32>(self, a: [u8; 64]) -> [u8; 64] {
        core::array::from_fn(|i| {
            if N < 8 {
                a[i].wrapping_shl(N as u32)
            } else {
                0
            }
        })
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(self, a: [u8; 64]) -> [u8; 64] {
        core::array::from_fn(|i| {
            if N < 8 {
                a[i].wrapping_shr(N as u32)
            } else {
                0
            }
        })
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: [u8; 64]) -> [u8; 64] {
        core::array::from_fn(|i| {
            if N < 8 {
                a[i].wrapping_shr(N as u32)
            } else {
                0
            }
        })
    }

    #[inline(always)]
    fn saturating_add(self, a: [u8; 64], b: [u8; 64]) -> [u8; 64] {
        core::array::from_fn(|i| a[i].saturating_add(b[i]))
    }

    #[inline(always)]
    fn saturating_sub(self, a: [u8; 64], b: [u8; 64]) -> [u8; 64] {
        core::array::from_fn(|i| a[i].saturating_sub(b[i]))
    }

    #[inline(always)]
    fn all_true(self, a: [u8; 64]) -> bool {
        a.iter().all(|&v| (v as i8) < 0)
    }

    #[inline(always)]
    fn any_true(self, a: [u8; 64]) -> bool {
        a.iter().any(|&v| (v as i8) < 0)
    }

    #[inline(always)]
    fn bitmask(self, a: [u8; 64]) -> u64 {
        let mut mask = 0u64;
        for i in 0..64 {
            if (a[i] as i8) < 0 {
                mask |= 1u64 << i;
            }
        }
        mask
    }
    #[inline(always)]
    fn widen_low_u8_to_u16(self, a: [u8; 64]) -> [u16; 32] {
        // Widen all lanes, keep the low half: dodges SROA integer
        // promotion of a half-array read (#77); the high half is DCE'd.
        let f: [u16; 64] = core::array::from_fn(|i| a[i] as u16);
        core::array::from_fn(|i| f[i])
    }

    #[inline(always)]
    fn widen_high_u8_to_u16(self, a: [u8; 64]) -> [u16; 32] {
        // Widen all lanes, keep the high half: dodges SROA integer
        // promotion of a half-array read (#77); the low half is DCE'd.
        let f: [u16; 64] = core::array::from_fn(|i| a[i] as u16);
        core::array::from_fn(|i| f[i + 32])
    }
    #[inline(always)]
    fn abs_diff(self, a: [u8; 64], b: [u8; 64]) -> [u8; 64] {
        core::array::from_fn(|k| a[k].abs_diff(b[k]))
    }

    #[inline(always)]
    fn reduce_add_u32(self, a: [u8; 64]) -> u32 {
        a.into_iter().map(u32::from).sum()
    }

    #[inline(always)]
    fn sum_abs_diff(self, a: [u8; 64], b: [u8; 64]) -> u32 {
        a.into_iter()
            .zip(b)
            .map(|(a, b)| u32::from(a.abs_diff(b)))
            .sum()
    }

    #[inline(always)]
    fn pairwise_widen_add(self, a: [u8; 64]) -> [u16; 32] {
        core::array::from_fn(|k| u16::from(a[2 * k]).wrapping_add(u16::from(a[2 * k + 1])))
    }
}

impl I16x32Backend for archmage::ScalarToken {
    type Repr = [i16; 32];

    #[inline(always)]
    fn splat(self, v: i16) -> [i16; 32] {
        [v; 32]
    }

    #[inline(always)]
    fn zero(self) -> [i16; 32] {
        [0; 32]
    }

    #[inline(always)]
    fn load(self, data: &[i16; 32]) -> [i16; 32] {
        *data
    }

    #[inline(always)]
    fn from_array(self, arr: [i16; 32]) -> [i16; 32] {
        arr
    }

    #[inline(always)]
    fn store(self, repr: [i16; 32], out: &mut [i16; 32]) {
        *out = repr;
    }

    #[inline(always)]
    fn to_array(self, repr: [i16; 32]) -> [i16; 32] {
        repr
    }

    #[inline(always)]
    fn add(self, a: [i16; 32], b: [i16; 32]) -> [i16; 32] {
        core::array::from_fn(|i| a[i].wrapping_add(b[i]))
    }

    #[inline(always)]
    fn sub(self, a: [i16; 32], b: [i16; 32]) -> [i16; 32] {
        core::array::from_fn(|i| a[i].wrapping_sub(b[i]))
    }

    #[inline(always)]
    fn mul(self, a: [i16; 32], b: [i16; 32]) -> [i16; 32] {
        core::array::from_fn(|i| a[i].wrapping_mul(b[i]))
    }

    #[inline(always)]
    fn neg(self, a: [i16; 32]) -> [i16; 32] {
        core::array::from_fn(|i| a[i].wrapping_neg())
    }

    #[inline(always)]
    fn min(self, a: [i16; 32], b: [i16; 32]) -> [i16; 32] {
        core::array::from_fn(|i| if a[i] < b[i] { a[i] } else { b[i] })
    }

    #[inline(always)]
    fn max(self, a: [i16; 32], b: [i16; 32]) -> [i16; 32] {
        core::array::from_fn(|i| if a[i] > b[i] { a[i] } else { b[i] })
    }

    #[inline(always)]
    fn abs(self, a: [i16; 32]) -> [i16; 32] {
        core::array::from_fn(|i| a[i].wrapping_abs())
    }

    #[inline(always)]
    fn simd_eq(self, a: [i16; 32], b: [i16; 32]) -> [i16; 32] {
        core::array::from_fn(|i| if a[i] == b[i] { !0i16 } else { 0 })
    }

    #[inline(always)]
    fn simd_ne(self, a: [i16; 32], b: [i16; 32]) -> [i16; 32] {
        core::array::from_fn(|i| if a[i] != b[i] { !0i16 } else { 0 })
    }

    #[inline(always)]
    fn simd_lt(self, a: [i16; 32], b: [i16; 32]) -> [i16; 32] {
        core::array::from_fn(|i| if a[i] < b[i] { !0i16 } else { 0 })
    }

    #[inline(always)]
    fn simd_le(self, a: [i16; 32], b: [i16; 32]) -> [i16; 32] {
        core::array::from_fn(|i| if a[i] <= b[i] { !0i16 } else { 0 })
    }

    #[inline(always)]
    fn simd_gt(self, a: [i16; 32], b: [i16; 32]) -> [i16; 32] {
        core::array::from_fn(|i| if a[i] > b[i] { !0i16 } else { 0 })
    }

    #[inline(always)]
    fn simd_ge(self, a: [i16; 32], b: [i16; 32]) -> [i16; 32] {
        core::array::from_fn(|i| if a[i] >= b[i] { !0i16 } else { 0 })
    }

    #[inline(always)]
    fn blend(self, mask: [i16; 32], if_true: [i16; 32], if_false: [i16; 32]) -> [i16; 32] {
        core::array::from_fn(|i| {
            if mask[i] != 0 {
                if_true[i]
            } else {
                if_false[i]
            }
        })
    }

    #[inline(always)]
    fn reduce_add(self, a: [i16; 32]) -> i16 {
        a.iter().copied().fold(0i16, i16::wrapping_add)
    }

    #[inline(always)]
    fn not(self, a: [i16; 32]) -> [i16; 32] {
        core::array::from_fn(|i| !a[i])
    }

    #[inline(always)]
    fn bitand(self, a: [i16; 32], b: [i16; 32]) -> [i16; 32] {
        core::array::from_fn(|i| a[i] & b[i])
    }

    #[inline(always)]
    fn bitor(self, a: [i16; 32], b: [i16; 32]) -> [i16; 32] {
        core::array::from_fn(|i| a[i] | b[i])
    }

    #[inline(always)]
    fn bitxor(self, a: [i16; 32], b: [i16; 32]) -> [i16; 32] {
        core::array::from_fn(|i| a[i] ^ b[i])
    }

    #[inline(always)]
    fn shl_const<const N: i32>(self, a: [i16; 32]) -> [i16; 32] {
        core::array::from_fn(|i| {
            if N < 16 {
                (a[i] as u16).wrapping_shl(N as u32) as i16
            } else {
                0
            }
        })
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(self, a: [i16; 32]) -> [i16; 32] {
        core::array::from_fn(|i| {
            if N < 16 {
                a[i].wrapping_shr(N as u32)
            } else if a[i] < 0 {
                -1
            } else {
                0
            }
        })
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: [i16; 32]) -> [i16; 32] {
        core::array::from_fn(|i| {
            if N < 16 {
                (a[i] as u16).wrapping_shr(N as u32) as i16
            } else {
                0
            }
        })
    }

    #[inline(always)]
    fn shl_uniform(self, a: [i16; 32], count: u32) -> [i16; 32] {
        core::array::from_fn(|i| {
            if count >= 16 {
                0
            } else {
                (a[i] as u16).wrapping_shl(count) as i16
            }
        })
    }

    #[inline(always)]
    fn shr_logical_uniform(self, a: [i16; 32], count: u32) -> [i16; 32] {
        core::array::from_fn(|i| {
            if count >= 16 {
                0
            } else {
                (a[i] as u16).wrapping_shr(count) as i16
            }
        })
    }

    #[inline(always)]
    fn shr_arithmetic_uniform(self, a: [i16; 32], count: u32) -> [i16; 32] {
        core::array::from_fn(|i| a[i].wrapping_shr(if count > 15 { 15 } else { count }))
    }

    #[inline(always)]
    fn saturating_add(self, a: [i16; 32], b: [i16; 32]) -> [i16; 32] {
        core::array::from_fn(|i| a[i].saturating_add(b[i]))
    }

    #[inline(always)]
    fn saturating_sub(self, a: [i16; 32], b: [i16; 32]) -> [i16; 32] {
        core::array::from_fn(|i| a[i].saturating_sub(b[i]))
    }

    #[inline(always)]
    fn all_true(self, a: [i16; 32]) -> bool {
        a.iter().all(|&v| v < 0)
    }

    #[inline(always)]
    fn any_true(self, a: [i16; 32]) -> bool {
        a.iter().any(|&v| v < 0)
    }

    #[inline(always)]
    fn bitmask(self, a: [i16; 32]) -> u64 {
        let mut mask = 0u64;
        for i in 0..32 {
            if a[i] < 0 {
                mask |= 1u64 << i;
            }
        }
        mask
    }
    #[inline(always)]
    fn widen_low_i16_to_i32(self, a: [i16; 32]) -> [i32; 16] {
        // Widen all lanes, keep the low half: dodges SROA integer
        // promotion of a half-array read (#77); the high half is DCE'd.
        let f: [i32; 32] = core::array::from_fn(|i| a[i] as i32);
        core::array::from_fn(|i| f[i])
    }

    #[inline(always)]
    fn widen_high_i16_to_i32(self, a: [i16; 32]) -> [i32; 16] {
        // Widen all lanes, keep the high half: dodges SROA integer
        // promotion of a half-array read (#77); the low half is DCE'd.
        let f: [i32; 32] = core::array::from_fn(|i| a[i] as i32);
        core::array::from_fn(|i| f[i + 16])
    }
    #[inline(always)]
    fn narrow_saturating_i16_to_i8(self, a: [i16; 32], b: [i16; 32]) -> [i8; 64] {
        core::array::from_fn(|i| {
            let v: i16 = if i < 32 { a[i] } else { b[i - 32] };
            v.clamp(i8::MIN as i16, i8::MAX as i16) as i8
        })
    }

    #[inline(always)]
    fn narrow_saturating_i16_to_u8(self, a: [i16; 32], b: [i16; 32]) -> [u8; 64] {
        core::array::from_fn(|i| {
            let v: i16 = if i < 32 { a[i] } else { b[i - 32] };
            v.clamp(0, u8::MAX as i16) as u8
        })
    }
    #[inline(always)]
    fn madd_adjacent(self, a: [i16; 32], b: [i16; 32]) -> [i32; 16] {
        core::array::from_fn(|k| {
            (a[2 * k] as i32 * b[2 * k] as i32)
                .wrapping_add(a[2 * k + 1] as i32 * b[2 * k + 1] as i32)
        })
    }

    #[inline(always)]
    fn abs_diff(self, a: [i16; 32], b: [i16; 32]) -> [u16; 32] {
        core::array::from_fn(|k| a[k].abs_diff(b[k]))
    }
}

impl U16x32Backend for archmage::ScalarToken {
    type Repr = [u16; 32];

    #[inline(always)]
    fn splat(self, v: u16) -> [u16; 32] {
        [v; 32]
    }

    #[inline(always)]
    fn zero(self) -> [u16; 32] {
        [0; 32]
    }

    #[inline(always)]
    fn load(self, data: &[u16; 32]) -> [u16; 32] {
        *data
    }

    #[inline(always)]
    fn from_array(self, arr: [u16; 32]) -> [u16; 32] {
        arr
    }

    #[inline(always)]
    fn store(self, repr: [u16; 32], out: &mut [u16; 32]) {
        *out = repr;
    }

    #[inline(always)]
    fn to_array(self, repr: [u16; 32]) -> [u16; 32] {
        repr
    }

    #[inline(always)]
    fn add(self, a: [u16; 32], b: [u16; 32]) -> [u16; 32] {
        core::array::from_fn(|i| a[i].wrapping_add(b[i]))
    }

    #[inline(always)]
    fn sub(self, a: [u16; 32], b: [u16; 32]) -> [u16; 32] {
        core::array::from_fn(|i| a[i].wrapping_sub(b[i]))
    }

    #[inline(always)]
    fn mul(self, a: [u16; 32], b: [u16; 32]) -> [u16; 32] {
        core::array::from_fn(|i| a[i].wrapping_mul(b[i]))
    }

    #[inline(always)]
    fn neg(self, a: [u16; 32]) -> [u16; 32] {
        core::array::from_fn(|i| (0u16).wrapping_sub(a[i]))
    }

    #[inline(always)]
    fn min(self, a: [u16; 32], b: [u16; 32]) -> [u16; 32] {
        core::array::from_fn(|i| if a[i] < b[i] { a[i] } else { b[i] })
    }

    #[inline(always)]
    fn max(self, a: [u16; 32], b: [u16; 32]) -> [u16; 32] {
        core::array::from_fn(|i| if a[i] > b[i] { a[i] } else { b[i] })
    }

    #[inline(always)]
    fn simd_eq(self, a: [u16; 32], b: [u16; 32]) -> [u16; 32] {
        core::array::from_fn(|i| if a[i] == b[i] { !0u16 } else { 0 })
    }

    #[inline(always)]
    fn simd_ne(self, a: [u16; 32], b: [u16; 32]) -> [u16; 32] {
        core::array::from_fn(|i| if a[i] != b[i] { !0u16 } else { 0 })
    }

    #[inline(always)]
    fn simd_lt(self, a: [u16; 32], b: [u16; 32]) -> [u16; 32] {
        core::array::from_fn(|i| if a[i] < b[i] { !0u16 } else { 0 })
    }

    #[inline(always)]
    fn simd_le(self, a: [u16; 32], b: [u16; 32]) -> [u16; 32] {
        core::array::from_fn(|i| if a[i] <= b[i] { !0u16 } else { 0 })
    }

    #[inline(always)]
    fn simd_gt(self, a: [u16; 32], b: [u16; 32]) -> [u16; 32] {
        core::array::from_fn(|i| if a[i] > b[i] { !0u16 } else { 0 })
    }

    #[inline(always)]
    fn simd_ge(self, a: [u16; 32], b: [u16; 32]) -> [u16; 32] {
        core::array::from_fn(|i| if a[i] >= b[i] { !0u16 } else { 0 })
    }

    #[inline(always)]
    fn blend(self, mask: [u16; 32], if_true: [u16; 32], if_false: [u16; 32]) -> [u16; 32] {
        core::array::from_fn(|i| {
            if mask[i] != 0 {
                if_true[i]
            } else {
                if_false[i]
            }
        })
    }

    #[inline(always)]
    fn reduce_add(self, a: [u16; 32]) -> u16 {
        a.iter().copied().fold(0u16, u16::wrapping_add)
    }

    #[inline(always)]
    fn not(self, a: [u16; 32]) -> [u16; 32] {
        core::array::from_fn(|i| !a[i])
    }

    #[inline(always)]
    fn bitand(self, a: [u16; 32], b: [u16; 32]) -> [u16; 32] {
        core::array::from_fn(|i| a[i] & b[i])
    }

    #[inline(always)]
    fn bitor(self, a: [u16; 32], b: [u16; 32]) -> [u16; 32] {
        core::array::from_fn(|i| a[i] | b[i])
    }

    #[inline(always)]
    fn bitxor(self, a: [u16; 32], b: [u16; 32]) -> [u16; 32] {
        core::array::from_fn(|i| a[i] ^ b[i])
    }

    #[inline(always)]
    fn shl_const<const N: i32>(self, a: [u16; 32]) -> [u16; 32] {
        core::array::from_fn(|i| {
            if N < 16 {
                a[i].wrapping_shl(N as u32)
            } else {
                0
            }
        })
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(self, a: [u16; 32]) -> [u16; 32] {
        core::array::from_fn(|i| {
            if N < 16 {
                a[i].wrapping_shr(N as u32)
            } else {
                0
            }
        })
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: [u16; 32]) -> [u16; 32] {
        core::array::from_fn(|i| {
            if N < 16 {
                a[i].wrapping_shr(N as u32)
            } else {
                0
            }
        })
    }

    #[inline(always)]
    fn shl_uniform(self, a: [u16; 32], count: u32) -> [u16; 32] {
        core::array::from_fn(|i| {
            if count >= 16 {
                0
            } else {
                a[i].wrapping_shl(count)
            }
        })
    }

    #[inline(always)]
    fn shr_logical_uniform(self, a: [u16; 32], count: u32) -> [u16; 32] {
        core::array::from_fn(|i| {
            if count >= 16 {
                0
            } else {
                a[i].wrapping_shr(count)
            }
        })
    }

    #[inline(always)]
    fn shr_arithmetic_uniform(self, a: [u16; 32], count: u32) -> [u16; 32] {
        core::array::from_fn(|i| {
            if count >= 16 {
                0
            } else {
                a[i].wrapping_shr(count)
            }
        })
    }

    #[inline(always)]
    fn saturating_add(self, a: [u16; 32], b: [u16; 32]) -> [u16; 32] {
        core::array::from_fn(|i| a[i].saturating_add(b[i]))
    }

    #[inline(always)]
    fn saturating_sub(self, a: [u16; 32], b: [u16; 32]) -> [u16; 32] {
        core::array::from_fn(|i| a[i].saturating_sub(b[i]))
    }

    #[inline(always)]
    fn all_true(self, a: [u16; 32]) -> bool {
        a.iter().all(|&v| (v as i16) < 0)
    }

    #[inline(always)]
    fn any_true(self, a: [u16; 32]) -> bool {
        a.iter().any(|&v| (v as i16) < 0)
    }

    #[inline(always)]
    fn bitmask(self, a: [u16; 32]) -> u64 {
        let mut mask = 0u64;
        for i in 0..32 {
            if (a[i] as i16) < 0 {
                mask |= 1u64 << i;
            }
        }
        mask
    }
    #[inline(always)]
    fn widen_low_u16_to_u32(self, a: [u16; 32]) -> [u32; 16] {
        // Widen all lanes, keep the low half: dodges SROA integer
        // promotion of a half-array read (#77); the high half is DCE'd.
        let f: [u32; 32] = core::array::from_fn(|i| a[i] as u32);
        core::array::from_fn(|i| f[i])
    }

    #[inline(always)]
    fn widen_high_u16_to_u32(self, a: [u16; 32]) -> [u32; 16] {
        // Widen all lanes, keep the high half: dodges SROA integer
        // promotion of a half-array read (#77); the low half is DCE'd.
        let f: [u32; 32] = core::array::from_fn(|i| a[i] as u32);
        core::array::from_fn(|i| f[i + 16])
    }
    #[inline(always)]
    fn pairwise_widen_add(self, a: [u16; 32]) -> [u32; 16] {
        core::array::from_fn(|k| u32::from(a[2 * k]).wrapping_add(u32::from(a[2 * k + 1])))
    }
}

impl I32x16Backend for archmage::ScalarToken {
    type Repr = [i32; 16];

    #[inline(always)]
    fn splat(self, v: i32) -> [i32; 16] {
        [v; 16]
    }

    #[inline(always)]
    fn zero(self) -> [i32; 16] {
        [0; 16]
    }

    #[inline(always)]
    fn load(self, data: &[i32; 16]) -> [i32; 16] {
        *data
    }

    #[inline(always)]
    fn from_array(self, arr: [i32; 16]) -> [i32; 16] {
        arr
    }

    #[inline(always)]
    fn store(self, repr: [i32; 16], out: &mut [i32; 16]) {
        *out = repr;
    }

    #[inline(always)]
    fn to_array(self, repr: [i32; 16]) -> [i32; 16] {
        repr
    }

    #[inline(always)]
    fn add(self, a: [i32; 16], b: [i32; 16]) -> [i32; 16] {
        core::array::from_fn(|i| a[i].wrapping_add(b[i]))
    }

    #[inline(always)]
    fn sub(self, a: [i32; 16], b: [i32; 16]) -> [i32; 16] {
        core::array::from_fn(|i| a[i].wrapping_sub(b[i]))
    }

    #[inline(always)]
    fn mul(self, a: [i32; 16], b: [i32; 16]) -> [i32; 16] {
        core::array::from_fn(|i| a[i].wrapping_mul(b[i]))
    }

    #[inline(always)]
    fn neg(self, a: [i32; 16]) -> [i32; 16] {
        core::array::from_fn(|i| a[i].wrapping_neg())
    }

    #[inline(always)]
    fn min(self, a: [i32; 16], b: [i32; 16]) -> [i32; 16] {
        core::array::from_fn(|i| if a[i] < b[i] { a[i] } else { b[i] })
    }

    #[inline(always)]
    fn max(self, a: [i32; 16], b: [i32; 16]) -> [i32; 16] {
        core::array::from_fn(|i| if a[i] > b[i] { a[i] } else { b[i] })
    }

    #[inline(always)]
    fn abs(self, a: [i32; 16]) -> [i32; 16] {
        core::array::from_fn(|i| a[i].wrapping_abs())
    }

    #[inline(always)]
    fn simd_eq(self, a: [i32; 16], b: [i32; 16]) -> [i32; 16] {
        core::array::from_fn(|i| if a[i] == b[i] { !0i32 } else { 0 })
    }

    #[inline(always)]
    fn simd_ne(self, a: [i32; 16], b: [i32; 16]) -> [i32; 16] {
        core::array::from_fn(|i| if a[i] != b[i] { !0i32 } else { 0 })
    }

    #[inline(always)]
    fn simd_lt(self, a: [i32; 16], b: [i32; 16]) -> [i32; 16] {
        core::array::from_fn(|i| if a[i] < b[i] { !0i32 } else { 0 })
    }

    #[inline(always)]
    fn simd_le(self, a: [i32; 16], b: [i32; 16]) -> [i32; 16] {
        core::array::from_fn(|i| if a[i] <= b[i] { !0i32 } else { 0 })
    }

    #[inline(always)]
    fn simd_gt(self, a: [i32; 16], b: [i32; 16]) -> [i32; 16] {
        core::array::from_fn(|i| if a[i] > b[i] { !0i32 } else { 0 })
    }

    #[inline(always)]
    fn simd_ge(self, a: [i32; 16], b: [i32; 16]) -> [i32; 16] {
        core::array::from_fn(|i| if a[i] >= b[i] { !0i32 } else { 0 })
    }

    #[inline(always)]
    fn blend(self, mask: [i32; 16], if_true: [i32; 16], if_false: [i32; 16]) -> [i32; 16] {
        core::array::from_fn(|i| {
            if mask[i] != 0 {
                if_true[i]
            } else {
                if_false[i]
            }
        })
    }

    #[inline(always)]
    fn reduce_add(self, a: [i32; 16]) -> i32 {
        a.iter().copied().fold(0i32, i32::wrapping_add)
    }

    #[inline(always)]
    fn not(self, a: [i32; 16]) -> [i32; 16] {
        core::array::from_fn(|i| !a[i])
    }

    #[inline(always)]
    fn bitand(self, a: [i32; 16], b: [i32; 16]) -> [i32; 16] {
        core::array::from_fn(|i| a[i] & b[i])
    }

    #[inline(always)]
    fn bitor(self, a: [i32; 16], b: [i32; 16]) -> [i32; 16] {
        core::array::from_fn(|i| a[i] | b[i])
    }

    #[inline(always)]
    fn bitxor(self, a: [i32; 16], b: [i32; 16]) -> [i32; 16] {
        core::array::from_fn(|i| a[i] ^ b[i])
    }

    #[inline(always)]
    fn shl_const<const N: i32>(self, a: [i32; 16]) -> [i32; 16] {
        core::array::from_fn(|i| {
            if N < 32 {
                (a[i] as u32).wrapping_shl(N as u32) as i32
            } else {
                0
            }
        })
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(self, a: [i32; 16]) -> [i32; 16] {
        core::array::from_fn(|i| {
            if N < 32 {
                a[i].wrapping_shr(N as u32)
            } else if a[i] < 0 {
                -1
            } else {
                0
            }
        })
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: [i32; 16]) -> [i32; 16] {
        core::array::from_fn(|i| {
            if N < 32 {
                (a[i] as u32).wrapping_shr(N as u32) as i32
            } else {
                0
            }
        })
    }

    #[inline(always)]
    fn shl_uniform(self, a: [i32; 16], count: u32) -> [i32; 16] {
        core::array::from_fn(|i| {
            if count >= 32 {
                0
            } else {
                (a[i] as u32).wrapping_shl(count) as i32
            }
        })
    }

    #[inline(always)]
    fn shr_logical_uniform(self, a: [i32; 16], count: u32) -> [i32; 16] {
        core::array::from_fn(|i| {
            if count >= 32 {
                0
            } else {
                (a[i] as u32).wrapping_shr(count) as i32
            }
        })
    }

    #[inline(always)]
    fn shr_arithmetic_uniform(self, a: [i32; 16], count: u32) -> [i32; 16] {
        core::array::from_fn(|i| a[i].wrapping_shr(if count > 31 { 31 } else { count }))
    }

    #[inline(always)]
    fn all_true(self, a: [i32; 16]) -> bool {
        a.iter().all(|&v| v < 0)
    }

    #[inline(always)]
    fn any_true(self, a: [i32; 16]) -> bool {
        a.iter().any(|&v| v < 0)
    }

    #[inline(always)]
    fn bitmask(self, a: [i32; 16]) -> u64 {
        let mut mask = 0u64;
        for i in 0..16 {
            if a[i] < 0 {
                mask |= 1u64 << i;
            }
        }
        mask
    }
    #[inline(always)]
    fn narrow_saturating_i32_to_i16(self, a: [i32; 16], b: [i32; 16]) -> [i16; 32] {
        core::array::from_fn(|i| {
            let v: i32 = if i < 16 { a[i] } else { b[i - 16] };
            v.clamp(i16::MIN as i32, i16::MAX as i32) as i16
        })
    }

    #[inline(always)]
    fn narrow_saturating_i32_to_u16(self, a: [i32; 16], b: [i32; 16]) -> [u16; 32] {
        core::array::from_fn(|i| {
            let v: i32 = if i < 16 { a[i] } else { b[i - 16] };
            v.clamp(0, u16::MAX as i32) as u16
        })
    }
}

impl U32x16Backend for archmage::ScalarToken {
    type Repr = [u32; 16];

    #[inline(always)]
    fn splat(self, v: u32) -> [u32; 16] {
        [v; 16]
    }

    #[inline(always)]
    fn zero(self) -> [u32; 16] {
        [0; 16]
    }

    #[inline(always)]
    fn load(self, data: &[u32; 16]) -> [u32; 16] {
        *data
    }

    #[inline(always)]
    fn from_array(self, arr: [u32; 16]) -> [u32; 16] {
        arr
    }

    #[inline(always)]
    fn store(self, repr: [u32; 16], out: &mut [u32; 16]) {
        *out = repr;
    }

    #[inline(always)]
    fn to_array(self, repr: [u32; 16]) -> [u32; 16] {
        repr
    }

    #[inline(always)]
    fn add(self, a: [u32; 16], b: [u32; 16]) -> [u32; 16] {
        core::array::from_fn(|i| a[i].wrapping_add(b[i]))
    }

    #[inline(always)]
    fn sub(self, a: [u32; 16], b: [u32; 16]) -> [u32; 16] {
        core::array::from_fn(|i| a[i].wrapping_sub(b[i]))
    }

    #[inline(always)]
    fn mul(self, a: [u32; 16], b: [u32; 16]) -> [u32; 16] {
        core::array::from_fn(|i| a[i].wrapping_mul(b[i]))
    }

    #[inline(always)]
    fn neg(self, a: [u32; 16]) -> [u32; 16] {
        core::array::from_fn(|i| (0u32).wrapping_sub(a[i]))
    }

    #[inline(always)]
    fn min(self, a: [u32; 16], b: [u32; 16]) -> [u32; 16] {
        core::array::from_fn(|i| if a[i] < b[i] { a[i] } else { b[i] })
    }

    #[inline(always)]
    fn max(self, a: [u32; 16], b: [u32; 16]) -> [u32; 16] {
        core::array::from_fn(|i| if a[i] > b[i] { a[i] } else { b[i] })
    }

    #[inline(always)]
    fn simd_eq(self, a: [u32; 16], b: [u32; 16]) -> [u32; 16] {
        core::array::from_fn(|i| if a[i] == b[i] { !0u32 } else { 0 })
    }

    #[inline(always)]
    fn simd_ne(self, a: [u32; 16], b: [u32; 16]) -> [u32; 16] {
        core::array::from_fn(|i| if a[i] != b[i] { !0u32 } else { 0 })
    }

    #[inline(always)]
    fn simd_lt(self, a: [u32; 16], b: [u32; 16]) -> [u32; 16] {
        core::array::from_fn(|i| if a[i] < b[i] { !0u32 } else { 0 })
    }

    #[inline(always)]
    fn simd_le(self, a: [u32; 16], b: [u32; 16]) -> [u32; 16] {
        core::array::from_fn(|i| if a[i] <= b[i] { !0u32 } else { 0 })
    }

    #[inline(always)]
    fn simd_gt(self, a: [u32; 16], b: [u32; 16]) -> [u32; 16] {
        core::array::from_fn(|i| if a[i] > b[i] { !0u32 } else { 0 })
    }

    #[inline(always)]
    fn simd_ge(self, a: [u32; 16], b: [u32; 16]) -> [u32; 16] {
        core::array::from_fn(|i| if a[i] >= b[i] { !0u32 } else { 0 })
    }

    #[inline(always)]
    fn blend(self, mask: [u32; 16], if_true: [u32; 16], if_false: [u32; 16]) -> [u32; 16] {
        core::array::from_fn(|i| {
            if mask[i] != 0 {
                if_true[i]
            } else {
                if_false[i]
            }
        })
    }

    #[inline(always)]
    fn reduce_add(self, a: [u32; 16]) -> u32 {
        a.iter().copied().fold(0u32, u32::wrapping_add)
    }

    #[inline(always)]
    fn not(self, a: [u32; 16]) -> [u32; 16] {
        core::array::from_fn(|i| !a[i])
    }

    #[inline(always)]
    fn bitand(self, a: [u32; 16], b: [u32; 16]) -> [u32; 16] {
        core::array::from_fn(|i| a[i] & b[i])
    }

    #[inline(always)]
    fn bitor(self, a: [u32; 16], b: [u32; 16]) -> [u32; 16] {
        core::array::from_fn(|i| a[i] | b[i])
    }

    #[inline(always)]
    fn bitxor(self, a: [u32; 16], b: [u32; 16]) -> [u32; 16] {
        core::array::from_fn(|i| a[i] ^ b[i])
    }

    #[inline(always)]
    fn shl_const<const N: i32>(self, a: [u32; 16]) -> [u32; 16] {
        core::array::from_fn(|i| {
            if N < 32 {
                a[i].wrapping_shl(N as u32)
            } else {
                0
            }
        })
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(self, a: [u32; 16]) -> [u32; 16] {
        core::array::from_fn(|i| {
            if N < 32 {
                a[i].wrapping_shr(N as u32)
            } else {
                0
            }
        })
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: [u32; 16]) -> [u32; 16] {
        core::array::from_fn(|i| {
            if N < 32 {
                a[i].wrapping_shr(N as u32)
            } else {
                0
            }
        })
    }

    #[inline(always)]
    fn shl_uniform(self, a: [u32; 16], count: u32) -> [u32; 16] {
        core::array::from_fn(|i| {
            if count >= 32 {
                0
            } else {
                a[i].wrapping_shl(count)
            }
        })
    }

    #[inline(always)]
    fn shr_logical_uniform(self, a: [u32; 16], count: u32) -> [u32; 16] {
        core::array::from_fn(|i| {
            if count >= 32 {
                0
            } else {
                a[i].wrapping_shr(count)
            }
        })
    }

    #[inline(always)]
    fn shr_arithmetic_uniform(self, a: [u32; 16], count: u32) -> [u32; 16] {
        core::array::from_fn(|i| {
            if count >= 32 {
                0
            } else {
                a[i].wrapping_shr(count)
            }
        })
    }

    #[inline(always)]
    fn all_true(self, a: [u32; 16]) -> bool {
        a.iter().all(|&v| (v as i32) < 0)
    }

    #[inline(always)]
    fn any_true(self, a: [u32; 16]) -> bool {
        a.iter().any(|&v| (v as i32) < 0)
    }

    #[inline(always)]
    fn bitmask(self, a: [u32; 16]) -> u64 {
        let mut mask = 0u64;
        for i in 0..16 {
            if (a[i] as i32) < 0 {
                mask |= 1u64 << i;
            }
        }
        mask
    }
}

impl I64x8Backend for archmage::ScalarToken {
    type Repr = [i64; 8];

    #[inline(always)]
    fn splat(self, v: i64) -> [i64; 8] {
        [v; 8]
    }

    #[inline(always)]
    fn zero(self) -> [i64; 8] {
        [0; 8]
    }

    #[inline(always)]
    fn load(self, data: &[i64; 8]) -> [i64; 8] {
        *data
    }

    #[inline(always)]
    fn from_array(self, arr: [i64; 8]) -> [i64; 8] {
        arr
    }

    #[inline(always)]
    fn store(self, repr: [i64; 8], out: &mut [i64; 8]) {
        *out = repr;
    }

    #[inline(always)]
    fn to_array(self, repr: [i64; 8]) -> [i64; 8] {
        repr
    }

    #[inline(always)]
    fn add(self, a: [i64; 8], b: [i64; 8]) -> [i64; 8] {
        core::array::from_fn(|i| a[i].wrapping_add(b[i]))
    }

    #[inline(always)]
    fn sub(self, a: [i64; 8], b: [i64; 8]) -> [i64; 8] {
        core::array::from_fn(|i| a[i].wrapping_sub(b[i]))
    }

    #[inline(always)]
    fn neg(self, a: [i64; 8]) -> [i64; 8] {
        core::array::from_fn(|i| a[i].wrapping_neg())
    }

    #[inline(always)]
    fn min(self, a: [i64; 8], b: [i64; 8]) -> [i64; 8] {
        core::array::from_fn(|i| if a[i] < b[i] { a[i] } else { b[i] })
    }

    #[inline(always)]
    fn max(self, a: [i64; 8], b: [i64; 8]) -> [i64; 8] {
        core::array::from_fn(|i| if a[i] > b[i] { a[i] } else { b[i] })
    }

    #[inline(always)]
    fn abs(self, a: [i64; 8]) -> [i64; 8] {
        core::array::from_fn(|i| a[i].wrapping_abs())
    }

    #[inline(always)]
    fn simd_eq(self, a: [i64; 8], b: [i64; 8]) -> [i64; 8] {
        core::array::from_fn(|i| if a[i] == b[i] { !0i64 } else { 0 })
    }

    #[inline(always)]
    fn simd_ne(self, a: [i64; 8], b: [i64; 8]) -> [i64; 8] {
        core::array::from_fn(|i| if a[i] != b[i] { !0i64 } else { 0 })
    }

    #[inline(always)]
    fn simd_lt(self, a: [i64; 8], b: [i64; 8]) -> [i64; 8] {
        core::array::from_fn(|i| if a[i] < b[i] { !0i64 } else { 0 })
    }

    #[inline(always)]
    fn simd_le(self, a: [i64; 8], b: [i64; 8]) -> [i64; 8] {
        core::array::from_fn(|i| if a[i] <= b[i] { !0i64 } else { 0 })
    }

    #[inline(always)]
    fn simd_gt(self, a: [i64; 8], b: [i64; 8]) -> [i64; 8] {
        core::array::from_fn(|i| if a[i] > b[i] { !0i64 } else { 0 })
    }

    #[inline(always)]
    fn simd_ge(self, a: [i64; 8], b: [i64; 8]) -> [i64; 8] {
        core::array::from_fn(|i| if a[i] >= b[i] { !0i64 } else { 0 })
    }

    #[inline(always)]
    fn blend(self, mask: [i64; 8], if_true: [i64; 8], if_false: [i64; 8]) -> [i64; 8] {
        core::array::from_fn(|i| {
            if mask[i] != 0 {
                if_true[i]
            } else {
                if_false[i]
            }
        })
    }

    #[inline(always)]
    fn reduce_add(self, a: [i64; 8]) -> i64 {
        a.iter().copied().fold(0i64, i64::wrapping_add)
    }

    #[inline(always)]
    fn not(self, a: [i64; 8]) -> [i64; 8] {
        core::array::from_fn(|i| !a[i])
    }

    #[inline(always)]
    fn bitand(self, a: [i64; 8], b: [i64; 8]) -> [i64; 8] {
        core::array::from_fn(|i| a[i] & b[i])
    }

    #[inline(always)]
    fn bitor(self, a: [i64; 8], b: [i64; 8]) -> [i64; 8] {
        core::array::from_fn(|i| a[i] | b[i])
    }

    #[inline(always)]
    fn bitxor(self, a: [i64; 8], b: [i64; 8]) -> [i64; 8] {
        core::array::from_fn(|i| a[i] ^ b[i])
    }

    #[inline(always)]
    fn shl_const<const N: i32>(self, a: [i64; 8]) -> [i64; 8] {
        core::array::from_fn(|i| {
            if N < 64 {
                (a[i] as u64).wrapping_shl(N as u32) as i64
            } else {
                0
            }
        })
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(self, a: [i64; 8]) -> [i64; 8] {
        core::array::from_fn(|i| {
            if N < 64 {
                a[i].wrapping_shr(N as u32)
            } else if a[i] < 0 {
                -1
            } else {
                0
            }
        })
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: [i64; 8]) -> [i64; 8] {
        core::array::from_fn(|i| {
            if N < 64 {
                (a[i] as u64).wrapping_shr(N as u32) as i64
            } else {
                0
            }
        })
    }

    #[inline(always)]
    fn all_true(self, a: [i64; 8]) -> bool {
        a.iter().all(|&v| v < 0)
    }

    #[inline(always)]
    fn any_true(self, a: [i64; 8]) -> bool {
        a.iter().any(|&v| v < 0)
    }

    #[inline(always)]
    fn bitmask(self, a: [i64; 8]) -> u64 {
        let mut mask = 0u64;
        for i in 0..8 {
            if a[i] < 0 {
                mask |= 1u64 << i;
            }
        }
        mask
    }
}

impl U64x8Backend for archmage::ScalarToken {
    type Repr = [u64; 8];

    #[inline(always)]
    fn splat(self, v: u64) -> [u64; 8] {
        [v; 8]
    }

    #[inline(always)]
    fn zero(self) -> [u64; 8] {
        [0; 8]
    }

    #[inline(always)]
    fn load(self, data: &[u64; 8]) -> [u64; 8] {
        *data
    }

    #[inline(always)]
    fn from_array(self, arr: [u64; 8]) -> [u64; 8] {
        arr
    }

    #[inline(always)]
    fn store(self, repr: [u64; 8], out: &mut [u64; 8]) {
        *out = repr;
    }

    #[inline(always)]
    fn to_array(self, repr: [u64; 8]) -> [u64; 8] {
        repr
    }

    #[inline(always)]
    fn add(self, a: [u64; 8], b: [u64; 8]) -> [u64; 8] {
        core::array::from_fn(|i| a[i].wrapping_add(b[i]))
    }

    #[inline(always)]
    fn sub(self, a: [u64; 8], b: [u64; 8]) -> [u64; 8] {
        core::array::from_fn(|i| a[i].wrapping_sub(b[i]))
    }

    #[inline(always)]
    fn neg(self, a: [u64; 8]) -> [u64; 8] {
        core::array::from_fn(|i| (0u64).wrapping_sub(a[i]))
    }

    #[inline(always)]
    fn min(self, a: [u64; 8], b: [u64; 8]) -> [u64; 8] {
        core::array::from_fn(|i| if a[i] < b[i] { a[i] } else { b[i] })
    }

    #[inline(always)]
    fn max(self, a: [u64; 8], b: [u64; 8]) -> [u64; 8] {
        core::array::from_fn(|i| if a[i] > b[i] { a[i] } else { b[i] })
    }

    #[inline(always)]
    fn simd_eq(self, a: [u64; 8], b: [u64; 8]) -> [u64; 8] {
        core::array::from_fn(|i| if a[i] == b[i] { !0u64 } else { 0 })
    }

    #[inline(always)]
    fn simd_ne(self, a: [u64; 8], b: [u64; 8]) -> [u64; 8] {
        core::array::from_fn(|i| if a[i] != b[i] { !0u64 } else { 0 })
    }

    #[inline(always)]
    fn simd_lt(self, a: [u64; 8], b: [u64; 8]) -> [u64; 8] {
        core::array::from_fn(|i| if a[i] < b[i] { !0u64 } else { 0 })
    }

    #[inline(always)]
    fn simd_le(self, a: [u64; 8], b: [u64; 8]) -> [u64; 8] {
        core::array::from_fn(|i| if a[i] <= b[i] { !0u64 } else { 0 })
    }

    #[inline(always)]
    fn simd_gt(self, a: [u64; 8], b: [u64; 8]) -> [u64; 8] {
        core::array::from_fn(|i| if a[i] > b[i] { !0u64 } else { 0 })
    }

    #[inline(always)]
    fn simd_ge(self, a: [u64; 8], b: [u64; 8]) -> [u64; 8] {
        core::array::from_fn(|i| if a[i] >= b[i] { !0u64 } else { 0 })
    }

    #[inline(always)]
    fn blend(self, mask: [u64; 8], if_true: [u64; 8], if_false: [u64; 8]) -> [u64; 8] {
        core::array::from_fn(|i| {
            if mask[i] != 0 {
                if_true[i]
            } else {
                if_false[i]
            }
        })
    }

    #[inline(always)]
    fn reduce_add(self, a: [u64; 8]) -> u64 {
        a.iter().copied().fold(0u64, u64::wrapping_add)
    }

    #[inline(always)]
    fn not(self, a: [u64; 8]) -> [u64; 8] {
        core::array::from_fn(|i| !a[i])
    }

    #[inline(always)]
    fn bitand(self, a: [u64; 8], b: [u64; 8]) -> [u64; 8] {
        core::array::from_fn(|i| a[i] & b[i])
    }

    #[inline(always)]
    fn bitor(self, a: [u64; 8], b: [u64; 8]) -> [u64; 8] {
        core::array::from_fn(|i| a[i] | b[i])
    }

    #[inline(always)]
    fn bitxor(self, a: [u64; 8], b: [u64; 8]) -> [u64; 8] {
        core::array::from_fn(|i| a[i] ^ b[i])
    }

    #[inline(always)]
    fn shl_const<const N: i32>(self, a: [u64; 8]) -> [u64; 8] {
        core::array::from_fn(|i| {
            if N < 64 {
                a[i].wrapping_shl(N as u32)
            } else {
                0
            }
        })
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(self, a: [u64; 8]) -> [u64; 8] {
        core::array::from_fn(|i| {
            if N < 64 {
                a[i].wrapping_shr(N as u32)
            } else {
                0
            }
        })
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: [u64; 8]) -> [u64; 8] {
        core::array::from_fn(|i| {
            if N < 64 {
                a[i].wrapping_shr(N as u32)
            } else {
                0
            }
        })
    }

    #[inline(always)]
    fn all_true(self, a: [u64; 8]) -> bool {
        a.iter().all(|&v| (v as i64) < 0)
    }

    #[inline(always)]
    fn any_true(self, a: [u64; 8]) -> bool {
        a.iter().any(|&v| (v as i64) < 0)
    }

    #[inline(always)]
    fn bitmask(self, a: [u64; 8]) -> u64 {
        let mut mask = 0u64;
        for i in 0..8 {
            if (a[i] as i64) < 0 {
                mask |= 1u64 << i;
            }
        }
        mask
    }
}

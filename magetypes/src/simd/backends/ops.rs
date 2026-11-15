//! Crate-private canonical backend operation traits.
//!
//! Implemented only by the five canonical base tokens (`ScalarToken`,
//! `X64V3Token`, `X64V4Token`, `NeonToken`, `Wasm128Token`). Public
//! `*Backend` / `*Convert` / `*Bitcast` traits delegate to these via
//! blanket impls over `SimdToken + Sealed`.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

#![allow(dead_code)]

pub trait NarrowOps:
    F32x4Ops
    + F32x8Ops
    + F64x2Ops
    + F64x4Ops
    + I32x4Ops
    + I32x8Ops
    + U32x4Ops
    + U32x8Ops
    + I64x2Ops
    + I64x4Ops
    + I8x16Ops
    + I8x32Ops
    + U8x16Ops
    + U8x32Ops
    + I16x8Ops
    + I16x16Ops
    + U16x8Ops
    + U16x16Ops
    + U64x2Ops
    + U64x4Ops
    + F32x4ConvertOps
    + F32x8ConvertOps
    + F64x2ConvertOps
    + F64x4ConvertOps
    + U32x4BitcastOps
    + U32x8BitcastOps
    + I64x2BitcastOps
    + I64x4BitcastOps
    + I8x16BitcastOps
    + I8x32BitcastOps
    + I16x8BitcastOps
    + I16x16BitcastOps
    + U64x2BitcastOps
    + U64x4BitcastOps
    + crate::simd::generic::cross_width::NarrowCrossWidthOps
    + crate::simd::generic::fast_ops::NarrowToU32Ops
    + crate::simd::generic::quant_ops::NarrowDotI8Ops
    + archmage::SimdToken
    + Copy
    + 'static
{
}
impl<T> NarrowOps for T where
    T: F32x4Ops
        + F32x8Ops
        + F64x2Ops
        + F64x4Ops
        + I32x4Ops
        + I32x8Ops
        + U32x4Ops
        + U32x8Ops
        + I64x2Ops
        + I64x4Ops
        + I8x16Ops
        + I8x32Ops
        + U8x16Ops
        + U8x32Ops
        + I16x8Ops
        + I16x16Ops
        + U16x8Ops
        + U16x16Ops
        + U64x2Ops
        + U64x4Ops
        + F32x4ConvertOps
        + F32x8ConvertOps
        + F64x2ConvertOps
        + F64x4ConvertOps
        + U32x4BitcastOps
        + U32x8BitcastOps
        + I64x2BitcastOps
        + I64x4BitcastOps
        + I8x16BitcastOps
        + I8x32BitcastOps
        + I16x8BitcastOps
        + I16x16BitcastOps
        + U64x2BitcastOps
        + U64x4BitcastOps
        + crate::simd::generic::cross_width::NarrowCrossWidthOps
        + crate::simd::generic::fast_ops::NarrowToU32Ops
        + crate::simd::generic::quant_ops::NarrowDotI8Ops
        + archmage::SimdToken
        + Copy
        + 'static
{
}

#[cfg(feature = "w512")]
pub trait W512Ops:
    F32x16Ops
    + F64x8Ops
    + I8x64Ops
    + U8x64Ops
    + I16x32Ops
    + U16x32Ops
    + I32x16Ops
    + U32x16Ops
    + I64x8Ops
    + U64x8Ops
    + F32x16ConvertOps
    + crate::simd::generic::fast_ops::W512ToU32Ops
    + crate::simd::generic::quant_ops::W512DotI8Ops
    + archmage::SimdToken
    + Copy
    + 'static
{
}
#[cfg(feature = "w512")]
impl<T> W512Ops for T where
    T: F32x16Ops
        + F64x8Ops
        + I8x64Ops
        + U8x64Ops
        + I16x32Ops
        + U16x32Ops
        + I32x16Ops
        + U32x16Ops
        + I64x8Ops
        + U64x8Ops
        + F32x16ConvertOps
        + crate::simd::generic::fast_ops::W512ToU32Ops
        + crate::simd::generic::quant_ops::W512DotI8Ops
        + archmage::SimdToken
        + Copy
        + 'static
{
}

pub trait F32x4Ops: 'static + archmage::SimdToken + Copy {
    #[allow(private_bounds)]
    type Repr: Copy + Clone + Send + Sync + crate::simd_storage::Pod;
    fn splat(self, v: f32) -> Self::Repr;
    fn zero(self) -> Self::Repr;
    fn load(self, data: &[f32; 4]) -> Self::Repr;
    fn from_array(self, arr: [f32; 4]) -> Self::Repr;
    fn store(self, repr: Self::Repr, out: &mut [f32; 4]);
    fn to_array(self, repr: Self::Repr) -> [f32; 4];
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn mul(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn div(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn neg(self, a: Self::Repr) -> Self::Repr;
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    #[inline(always)]
    fn min_fast(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <Self as F32x4Ops>::min(self, a, b)
    }
    #[inline(always)]
    fn max_fast(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <Self as F32x4Ops>::max(self, a, b)
    }
    fn sqrt(self, a: Self::Repr) -> Self::Repr;
    fn abs(self, a: Self::Repr) -> Self::Repr;
    fn floor(self, a: Self::Repr) -> Self::Repr;
    fn ceil(self, a: Self::Repr) -> Self::Repr;
    fn round(self, a: Self::Repr) -> Self::Repr;
    fn mul_add(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr;
    fn mul_sub(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr;
    fn mul_add_portable(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr;
    fn mul_sub_portable(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr;
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr;
    fn reduce_add(self, a: Self::Repr) -> f32;
    fn reduce_min(self, a: Self::Repr) -> f32;
    fn reduce_max(self, a: Self::Repr) -> f32;
    #[inline(always)]
    fn rcp_approx(self, a: Self::Repr) -> Self::Repr {
        a
    }
    #[inline(always)]
    fn rsqrt_approx(self, a: Self::Repr) -> Self::Repr {
        a
    }
    fn not(self, a: Self::Repr) -> Self::Repr;
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <Self as F32x4Ops>::min(self, <Self as F32x4Ops>::max(self, a, lo), hi)
    }
    #[inline(always)]
    fn recip(self, a: Self::Repr) -> Self::Repr {
        Self::rcp_approx(self, a)
    }
    #[inline(always)]
    fn rsqrt(self, a: Self::Repr) -> Self::Repr {
        Self::rsqrt_approx(self, a)
    }
    #[inline(always)]
    fn broadcast_lane<const N: usize>(self, a: Self::Repr) -> Self::Repr {
        const { assert!(N < 4, "lane index out of bounds") };
        let arr = <Self as F32x4Ops>::to_array(self, a);
        <Self as F32x4Ops>::splat(self, arr[N])
    }
    #[inline(always)]
    fn shift_left_lanes<const N: usize>(self, a: Self::Repr) -> Self::Repr {
        const { assert!(N <= 4, "shift count out of bounds") };
        let arr = <Self as F32x4Ops>::to_array(self, a);
        let mut out = [0.0f32; 4];
        let mut i = N;
        while i < 4 {
            out[i] = arr[i - N];
            i += 1;
        }
        <Self as F32x4Ops>::from_array(self, out)
    }
    #[inline(always)]
    fn shift_right_lanes<const N: usize>(self, a: Self::Repr) -> Self::Repr {
        const { assert!(N <= 4, "shift count out of bounds") };
        let arr = <Self as F32x4Ops>::to_array(self, a);
        let mut out = [0.0f32; 4];
        let mut i = 0;
        while i + N < 4 {
            out[i] = arr[i + N];
            i += 1;
        }
        <Self as F32x4Ops>::from_array(self, out)
    }
    #[inline(always)]
    fn mul_add_lane_relaxed<const LANE: i32>(
        self,
        a: Self::Repr,
        v_in: Self::Repr,
        acc: Self::Repr,
    ) -> Self::Repr {
        const { assert!(LANE >= 0 && (LANE as usize) < 4, "LANE out of bounds") };
        let arr = <Self as F32x4Ops>::to_array(self, v_in);
        let b = <Self as F32x4Ops>::splat(self, arr[LANE as usize]);
        <Self as F32x4Ops>::mul_add(self, a, b, acc)
    }
    #[inline(always)]
    fn to_u8_bytes(self, a: Self::Repr) -> [u8; 4] {
        let arr = <Self as F32x4Ops>::to_array(self, a);
        core::array::from_fn(|i| crate::nostd_math::roundevenf(arr[i]).clamp(0.0, 255.0) as u8)
    }
    #[inline(always)]
    fn store_rgba_bytes(
        self,
        r: Self::Repr,
        g: Self::Repr,
        b: Self::Repr,
        a: Self::Repr,
    ) -> [u8; 16] {
        let rb = <Self as F32x4Ops>::to_u8_bytes(self, r);
        let gb = <Self as F32x4Ops>::to_u8_bytes(self, g);
        let bb = <Self as F32x4Ops>::to_u8_bytes(self, b);
        let ab = <Self as F32x4Ops>::to_u8_bytes(self, a);
        let mut out = [0u8; 16];
        let mut i = 0;
        while i < 4 {
            out[i * 4] = rb[i];
            out[i * 4 + 1] = gb[i];
            out[i * 4 + 2] = bb[i];
            out[i * 4 + 3] = ab[i];
            i += 1;
        }
        out
    }
    #[inline(always)]
    fn deinterleave_3ch_repr(self, rgb: [Self::Repr; 3]) -> [Self::Repr; 3] {
        let v0 = <Self as F32x4Ops>::to_array(self, rgb[0]);
        let v1 = <Self as F32x4Ops>::to_array(self, rgb[1]);
        let v2 = <Self as F32x4Ops>::to_array(self, rgb[2]);
        [
            <Self as F32x4Ops>::from_array(self, [v0[0], v0[3], v1[2], v2[1]]),
            <Self as F32x4Ops>::from_array(self, [v0[1], v1[0], v1[3], v2[2]]),
            <Self as F32x4Ops>::from_array(self, [v0[2], v1[1], v2[0], v2[3]]),
        ]
    }
    #[inline(always)]
    fn interleave_3ch_repr(self, channels: [Self::Repr; 3]) -> [Self::Repr; 3] {
        let r = <Self as F32x4Ops>::to_array(self, channels[0]);
        let g = <Self as F32x4Ops>::to_array(self, channels[1]);
        let b = <Self as F32x4Ops>::to_array(self, channels[2]);
        [
            <Self as F32x4Ops>::from_array(self, [r[0], g[0], b[0], r[1]]),
            <Self as F32x4Ops>::from_array(self, [g[1], b[1], r[2], g[2]]),
            <Self as F32x4Ops>::from_array(self, [b[2], r[3], g[3], b[3]]),
        ]
    }
    #[inline(always)]
    fn load_rgb_bytes(self, rgb: &[u8; 12]) -> (Self::Repr, Self::Repr, Self::Repr) {
        let r: [f32; 4] = core::array::from_fn(|i| rgb[i * 3] as f32);
        let g: [f32; 4] = core::array::from_fn(|i| rgb[i * 3 + 1] as f32);
        let b: [f32; 4] = core::array::from_fn(|i| rgb[i * 3 + 2] as f32);
        (
            <Self as F32x4Ops>::from_array(self, r),
            <Self as F32x4Ops>::from_array(self, g),
            <Self as F32x4Ops>::from_array(self, b),
        )
    }
    #[inline(always)]
    fn store_rgb_bytes(self, r: Self::Repr, g: Self::Repr, b: Self::Repr) -> [u8; 12] {
        let rb = <Self as F32x4Ops>::to_u8_bytes(self, r);
        let gb = <Self as F32x4Ops>::to_u8_bytes(self, g);
        let bb = <Self as F32x4Ops>::to_u8_bytes(self, b);
        let mut out = [0u8; 12];
        let mut i = 0;
        while i < 4 {
            out[i * 3] = rb[i];
            out[i * 3 + 1] = gb[i];
            out[i * 3 + 2] = bb[i];
            i += 1;
        }
        out
    }
}

pub trait F32x8Ops: 'static + archmage::SimdToken + Copy {
    #[allow(private_bounds)]
    type Repr: Copy + Clone + Send + Sync + crate::simd_storage::Pod;
    fn splat(self, v: f32) -> Self::Repr;
    fn zero(self) -> Self::Repr;
    fn load(self, data: &[f32; 8]) -> Self::Repr;
    fn from_array(self, arr: [f32; 8]) -> Self::Repr;
    fn store(self, repr: Self::Repr, out: &mut [f32; 8]);
    fn to_array(self, repr: Self::Repr) -> [f32; 8];
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn mul(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn div(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn neg(self, a: Self::Repr) -> Self::Repr;
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    #[inline(always)]
    fn min_fast(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <Self as F32x8Ops>::min(self, a, b)
    }
    #[inline(always)]
    fn max_fast(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <Self as F32x8Ops>::max(self, a, b)
    }
    fn sqrt(self, a: Self::Repr) -> Self::Repr;
    fn abs(self, a: Self::Repr) -> Self::Repr;
    fn floor(self, a: Self::Repr) -> Self::Repr;
    fn ceil(self, a: Self::Repr) -> Self::Repr;
    fn round(self, a: Self::Repr) -> Self::Repr;
    fn mul_add(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr;
    fn mul_sub(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr;
    fn mul_add_portable(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr;
    fn mul_sub_portable(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr;
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr;
    fn reduce_add(self, a: Self::Repr) -> f32;
    fn reduce_min(self, a: Self::Repr) -> f32;
    fn reduce_max(self, a: Self::Repr) -> f32;
    #[inline(always)]
    fn rcp_approx(self, a: Self::Repr) -> Self::Repr {
        a
    }
    #[inline(always)]
    fn rsqrt_approx(self, a: Self::Repr) -> Self::Repr {
        a
    }
    fn not(self, a: Self::Repr) -> Self::Repr;
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <Self as F32x8Ops>::min(self, <Self as F32x8Ops>::max(self, a, lo), hi)
    }
    #[inline(always)]
    fn recip(self, a: Self::Repr) -> Self::Repr {
        Self::rcp_approx(self, a)
    }
    #[inline(always)]
    fn rsqrt(self, a: Self::Repr) -> Self::Repr {
        Self::rsqrt_approx(self, a)
    }
    #[inline(always)]
    fn broadcast_lane<const N: usize>(self, a: Self::Repr) -> Self::Repr {
        const { assert!(N < 8, "lane index out of bounds") };
        let arr = <Self as F32x8Ops>::to_array(self, a);
        <Self as F32x8Ops>::splat(self, arr[N])
    }
    #[inline(always)]
    fn shift_left_lanes<const N: usize>(self, a: Self::Repr) -> Self::Repr {
        const { assert!(N <= 8, "shift count out of bounds") };
        let arr = <Self as F32x8Ops>::to_array(self, a);
        let mut out = [0.0f32; 8];
        let mut i = N;
        while i < 8 {
            out[i] = arr[i - N];
            i += 1;
        }
        <Self as F32x8Ops>::from_array(self, out)
    }
    #[inline(always)]
    fn shift_right_lanes<const N: usize>(self, a: Self::Repr) -> Self::Repr {
        const { assert!(N <= 8, "shift count out of bounds") };
        let arr = <Self as F32x8Ops>::to_array(self, a);
        let mut out = [0.0f32; 8];
        let mut i = 0;
        while i + N < 8 {
            out[i] = arr[i + N];
            i += 1;
        }
        <Self as F32x8Ops>::from_array(self, out)
    }
    #[inline(always)]
    fn mul_add_lane_relaxed<const LANE: i32>(
        self,
        a: Self::Repr,
        v_in: Self::Repr,
        acc: Self::Repr,
    ) -> Self::Repr {
        const { assert!(LANE >= 0 && (LANE as usize) < 8, "LANE out of bounds") };
        let arr = <Self as F32x8Ops>::to_array(self, v_in);
        let b = <Self as F32x8Ops>::splat(self, arr[LANE as usize]);
        <Self as F32x8Ops>::mul_add(self, a, b, acc)
    }
    #[inline(always)]
    fn to_u8_bytes(self, a: Self::Repr) -> [u8; 8] {
        let arr = <Self as F32x8Ops>::to_array(self, a);
        core::array::from_fn(|i| crate::nostd_math::roundevenf(arr[i]).clamp(0.0, 255.0) as u8)
    }
    #[inline(always)]
    fn store_rgba_bytes(
        self,
        r: Self::Repr,
        g: Self::Repr,
        b: Self::Repr,
        a: Self::Repr,
    ) -> [u8; 32] {
        let rb = <Self as F32x8Ops>::to_u8_bytes(self, r);
        let gb = <Self as F32x8Ops>::to_u8_bytes(self, g);
        let bb = <Self as F32x8Ops>::to_u8_bytes(self, b);
        let ab = <Self as F32x8Ops>::to_u8_bytes(self, a);
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
    #[inline(always)]
    fn deinterleave_3ch_repr(self, rgb: [Self::Repr; 3]) -> [Self::Repr; 3] {
        let v0 = <Self as F32x8Ops>::to_array(self, rgb[0]);
        let v1 = <Self as F32x8Ops>::to_array(self, rgb[1]);
        let v2 = <Self as F32x8Ops>::to_array(self, rgb[2]);
        [
            <Self as F32x8Ops>::from_array(
                self,
                [v0[0], v0[3], v0[6], v1[1], v1[4], v1[7], v2[2], v2[5]],
            ),
            <Self as F32x8Ops>::from_array(
                self,
                [v0[1], v0[4], v0[7], v1[2], v1[5], v2[0], v2[3], v2[6]],
            ),
            <Self as F32x8Ops>::from_array(
                self,
                [v0[2], v0[5], v1[0], v1[3], v1[6], v2[1], v2[4], v2[7]],
            ),
        ]
    }
    #[inline(always)]
    fn interleave_3ch_repr(self, channels: [Self::Repr; 3]) -> [Self::Repr; 3] {
        let r = <Self as F32x8Ops>::to_array(self, channels[0]);
        let g = <Self as F32x8Ops>::to_array(self, channels[1]);
        let b = <Self as F32x8Ops>::to_array(self, channels[2]);
        [
            <Self as F32x8Ops>::from_array(self, [r[0], g[0], b[0], r[1], g[1], b[1], r[2], g[2]]),
            <Self as F32x8Ops>::from_array(self, [b[2], r[3], g[3], b[3], r[4], g[4], b[4], r[5]]),
            <Self as F32x8Ops>::from_array(self, [g[5], b[5], r[6], g[6], b[6], r[7], g[7], b[7]]),
        ]
    }
    #[inline(always)]
    fn load_rgb_bytes(self, rgb: &[u8; 24]) -> (Self::Repr, Self::Repr, Self::Repr) {
        let r: [f32; 8] = core::array::from_fn(|i| rgb[i * 3] as f32);
        let g: [f32; 8] = core::array::from_fn(|i| rgb[i * 3 + 1] as f32);
        let b: [f32; 8] = core::array::from_fn(|i| rgb[i * 3 + 2] as f32);
        (
            <Self as F32x8Ops>::from_array(self, r),
            <Self as F32x8Ops>::from_array(self, g),
            <Self as F32x8Ops>::from_array(self, b),
        )
    }
    #[inline(always)]
    fn store_rgb_bytes(self, r: Self::Repr, g: Self::Repr, b: Self::Repr) -> [u8; 24] {
        let rb = <Self as F32x8Ops>::to_u8_bytes(self, r);
        let gb = <Self as F32x8Ops>::to_u8_bytes(self, g);
        let bb = <Self as F32x8Ops>::to_u8_bytes(self, b);
        let mut out = [0u8; 24];
        let mut i = 0;
        while i < 8 {
            out[i * 3] = rb[i];
            out[i * 3 + 1] = gb[i];
            out[i * 3 + 2] = bb[i];
            i += 1;
        }
        out
    }
    #[inline(always)]
    fn transpose_8x8_repr(self, rows: [Self::Repr; 8]) -> [Self::Repr; 8] {
        let r: [[f32; 8]; 8] =
            core::array::from_fn(|i| <Self as F32x8Ops>::to_array(self, rows[i]));
        core::array::from_fn(|i| {
            <Self as F32x8Ops>::from_array(self, core::array::from_fn(|j| r[j][i]))
        })
    }
}

pub trait F64x2Ops: 'static + archmage::SimdToken + Copy {
    #[allow(private_bounds)]
    type Repr: Copy + Clone + Send + Sync + crate::simd_storage::Pod;
    fn splat(self, v: f64) -> Self::Repr;
    fn zero(self) -> Self::Repr;
    fn load(self, data: &[f64; 2]) -> Self::Repr;
    fn from_array(self, arr: [f64; 2]) -> Self::Repr;
    fn store(self, repr: Self::Repr, out: &mut [f64; 2]);
    fn to_array(self, repr: Self::Repr) -> [f64; 2];
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn mul(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn div(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn neg(self, a: Self::Repr) -> Self::Repr;
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    #[inline(always)]
    fn min_fast(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <Self as F64x2Ops>::min(self, a, b)
    }
    #[inline(always)]
    fn max_fast(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <Self as F64x2Ops>::max(self, a, b)
    }
    fn sqrt(self, a: Self::Repr) -> Self::Repr;
    fn abs(self, a: Self::Repr) -> Self::Repr;
    fn floor(self, a: Self::Repr) -> Self::Repr;
    fn ceil(self, a: Self::Repr) -> Self::Repr;
    fn round(self, a: Self::Repr) -> Self::Repr;
    fn mul_add(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr;
    fn mul_sub(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr;
    fn mul_add_portable(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr;
    fn mul_sub_portable(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr;
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr;
    fn reduce_add(self, a: Self::Repr) -> f64;
    fn reduce_min(self, a: Self::Repr) -> f64;
    fn reduce_max(self, a: Self::Repr) -> f64;
    #[inline(always)]
    fn rcp_approx(self, a: Self::Repr) -> Self::Repr {
        a
    }
    #[inline(always)]
    fn rsqrt_approx(self, a: Self::Repr) -> Self::Repr {
        a
    }
    fn not(self, a: Self::Repr) -> Self::Repr;
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <Self as F64x2Ops>::min(self, <Self as F64x2Ops>::max(self, a, lo), hi)
    }
    #[inline(always)]
    fn recip(self, a: Self::Repr) -> Self::Repr {
        Self::rcp_approx(self, a)
    }
    #[inline(always)]
    fn rsqrt(self, a: Self::Repr) -> Self::Repr {
        Self::rsqrt_approx(self, a)
    }
}

pub trait F64x4Ops: 'static + archmage::SimdToken + Copy {
    #[allow(private_bounds)]
    type Repr: Copy + Clone + Send + Sync + crate::simd_storage::Pod;
    fn splat(self, v: f64) -> Self::Repr;
    fn zero(self) -> Self::Repr;
    fn load(self, data: &[f64; 4]) -> Self::Repr;
    fn from_array(self, arr: [f64; 4]) -> Self::Repr;
    fn store(self, repr: Self::Repr, out: &mut [f64; 4]);
    fn to_array(self, repr: Self::Repr) -> [f64; 4];
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn mul(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn div(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn neg(self, a: Self::Repr) -> Self::Repr;
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    #[inline(always)]
    fn min_fast(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <Self as F64x4Ops>::min(self, a, b)
    }
    #[inline(always)]
    fn max_fast(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <Self as F64x4Ops>::max(self, a, b)
    }
    fn sqrt(self, a: Self::Repr) -> Self::Repr;
    fn abs(self, a: Self::Repr) -> Self::Repr;
    fn floor(self, a: Self::Repr) -> Self::Repr;
    fn ceil(self, a: Self::Repr) -> Self::Repr;
    fn round(self, a: Self::Repr) -> Self::Repr;
    fn mul_add(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr;
    fn mul_sub(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr;
    fn mul_add_portable(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr;
    fn mul_sub_portable(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr;
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr;
    fn reduce_add(self, a: Self::Repr) -> f64;
    fn reduce_min(self, a: Self::Repr) -> f64;
    fn reduce_max(self, a: Self::Repr) -> f64;
    #[inline(always)]
    fn rcp_approx(self, a: Self::Repr) -> Self::Repr {
        a
    }
    #[inline(always)]
    fn rsqrt_approx(self, a: Self::Repr) -> Self::Repr {
        a
    }
    fn not(self, a: Self::Repr) -> Self::Repr;
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <Self as F64x4Ops>::min(self, <Self as F64x4Ops>::max(self, a, lo), hi)
    }
    #[inline(always)]
    fn recip(self, a: Self::Repr) -> Self::Repr {
        Self::rcp_approx(self, a)
    }
    #[inline(always)]
    fn rsqrt(self, a: Self::Repr) -> Self::Repr {
        Self::rsqrt_approx(self, a)
    }
}

pub trait I32x4Ops: 'static + archmage::SimdToken + Copy {
    #[allow(private_bounds)]
    type Repr: Copy + Clone + Send + Sync + crate::simd_storage::Pod;
    fn splat(self, v: i32) -> Self::Repr;
    fn zero(self) -> Self::Repr;
    fn load(self, data: &[i32; 4]) -> Self::Repr;
    fn from_array(self, arr: [i32; 4]) -> Self::Repr;
    fn store(self, repr: Self::Repr, out: &mut [i32; 4]);
    fn to_array(self, repr: Self::Repr) -> [i32; 4];
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn mul(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn neg(self, a: Self::Repr) -> Self::Repr;
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn abs(self, a: Self::Repr) -> Self::Repr;
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr;
    fn reduce_add(self, a: Self::Repr) -> i32;
    fn not(self, a: Self::Repr) -> Self::Repr;
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shr_arithmetic_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shl_uniform(self, a: Self::Repr, count: u32) -> Self::Repr;
    fn shr_logical_uniform(self, a: Self::Repr, count: u32) -> Self::Repr;
    fn shr_arithmetic_uniform(self, a: Self::Repr, count: u32) -> Self::Repr;
    fn all_true(self, a: Self::Repr) -> bool;
    fn any_true(self, a: Self::Repr) -> bool;
    fn bitmask(self, a: Self::Repr) -> u32;
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <Self as I32x4Ops>::min(self, <Self as I32x4Ops>::max(self, a, lo), hi)
    }
    fn narrow_saturating_i32_to_i16(
        self,
        a: <Self as I32x4Ops>::Repr,
        b: <Self as I32x4Ops>::Repr,
    ) -> <Self as I16x8Ops>::Repr
    where
        Self: I16x8Ops;
    fn narrow_saturating_i32_to_u16(
        self,
        a: <Self as I32x4Ops>::Repr,
        b: <Self as I32x4Ops>::Repr,
    ) -> <Self as U16x8Ops>::Repr
    where
        Self: U16x8Ops;
}

pub trait I32x8Ops: 'static + archmage::SimdToken + Copy {
    #[allow(private_bounds)]
    type Repr: Copy + Clone + Send + Sync + crate::simd_storage::Pod;
    fn splat(self, v: i32) -> Self::Repr;
    fn zero(self) -> Self::Repr;
    fn load(self, data: &[i32; 8]) -> Self::Repr;
    fn from_array(self, arr: [i32; 8]) -> Self::Repr;
    fn store(self, repr: Self::Repr, out: &mut [i32; 8]);
    fn to_array(self, repr: Self::Repr) -> [i32; 8];
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn mul(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn neg(self, a: Self::Repr) -> Self::Repr;
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn abs(self, a: Self::Repr) -> Self::Repr;
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr;
    fn reduce_add(self, a: Self::Repr) -> i32;
    fn not(self, a: Self::Repr) -> Self::Repr;
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shr_arithmetic_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shl_uniform(self, a: Self::Repr, count: u32) -> Self::Repr;
    fn shr_logical_uniform(self, a: Self::Repr, count: u32) -> Self::Repr;
    fn shr_arithmetic_uniform(self, a: Self::Repr, count: u32) -> Self::Repr;
    fn all_true(self, a: Self::Repr) -> bool;
    fn any_true(self, a: Self::Repr) -> bool;
    fn bitmask(self, a: Self::Repr) -> u32;
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <Self as I32x8Ops>::min(self, <Self as I32x8Ops>::max(self, a, lo), hi)
    }
    fn narrow_saturating_i32_to_i16(
        self,
        a: <Self as I32x8Ops>::Repr,
        b: <Self as I32x8Ops>::Repr,
    ) -> <Self as I16x16Ops>::Repr
    where
        Self: I16x16Ops;
    fn narrow_saturating_i32_to_u16(
        self,
        a: <Self as I32x8Ops>::Repr,
        b: <Self as I32x8Ops>::Repr,
    ) -> <Self as U16x16Ops>::Repr
    where
        Self: U16x16Ops;
}

pub trait U32x4Ops: 'static + archmage::SimdToken + Copy {
    #[allow(private_bounds)]
    type Repr: Copy + Clone + Send + Sync + crate::simd_storage::Pod;
    fn splat(self, v: u32) -> Self::Repr;
    fn zero(self) -> Self::Repr;
    fn load(self, data: &[u32; 4]) -> Self::Repr;
    fn from_array(self, arr: [u32; 4]) -> Self::Repr;
    fn store(self, repr: Self::Repr, out: &mut [u32; 4]);
    fn to_array(self, repr: Self::Repr) -> [u32; 4];
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn mul(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr;
    fn reduce_add(self, a: Self::Repr) -> u32;
    fn not(self, a: Self::Repr) -> Self::Repr;
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shl_uniform(self, a: Self::Repr, count: u32) -> Self::Repr;
    fn shr_logical_uniform(self, a: Self::Repr, count: u32) -> Self::Repr;
    fn all_true(self, a: Self::Repr) -> bool;
    fn any_true(self, a: Self::Repr) -> bool;
    fn bitmask(self, a: Self::Repr) -> u32;
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <Self as U32x4Ops>::min(self, <Self as U32x4Ops>::max(self, a, lo), hi)
    }
}

pub trait U32x8Ops: 'static + archmage::SimdToken + Copy {
    #[allow(private_bounds)]
    type Repr: Copy + Clone + Send + Sync + crate::simd_storage::Pod;
    fn splat(self, v: u32) -> Self::Repr;
    fn zero(self) -> Self::Repr;
    fn load(self, data: &[u32; 8]) -> Self::Repr;
    fn from_array(self, arr: [u32; 8]) -> Self::Repr;
    fn store(self, repr: Self::Repr, out: &mut [u32; 8]);
    fn to_array(self, repr: Self::Repr) -> [u32; 8];
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn mul(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr;
    fn reduce_add(self, a: Self::Repr) -> u32;
    fn not(self, a: Self::Repr) -> Self::Repr;
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shl_uniform(self, a: Self::Repr, count: u32) -> Self::Repr;
    fn shr_logical_uniform(self, a: Self::Repr, count: u32) -> Self::Repr;
    fn all_true(self, a: Self::Repr) -> bool;
    fn any_true(self, a: Self::Repr) -> bool;
    fn bitmask(self, a: Self::Repr) -> u32;
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <Self as U32x8Ops>::min(self, <Self as U32x8Ops>::max(self, a, lo), hi)
    }
}

pub trait I64x2Ops: 'static + archmage::SimdToken + Copy {
    #[allow(private_bounds)]
    type Repr: Copy + Clone + Send + Sync + crate::simd_storage::Pod;
    fn splat(self, v: i64) -> Self::Repr;
    fn zero(self) -> Self::Repr;
    fn load(self, data: &[i64; 2]) -> Self::Repr;
    fn from_array(self, arr: [i64; 2]) -> Self::Repr;
    fn store(self, repr: Self::Repr, out: &mut [i64; 2]);
    fn to_array(self, repr: Self::Repr) -> [i64; 2];
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn neg(self, a: Self::Repr) -> Self::Repr;
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn abs(self, a: Self::Repr) -> Self::Repr;
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr;
    fn reduce_add(self, a: Self::Repr) -> i64;
    fn not(self, a: Self::Repr) -> Self::Repr;
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shr_arithmetic_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn all_true(self, a: Self::Repr) -> bool;
    fn any_true(self, a: Self::Repr) -> bool;
    fn bitmask(self, a: Self::Repr) -> u32;
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <Self as I64x2Ops>::min(self, <Self as I64x2Ops>::max(self, a, lo), hi)
    }
}

pub trait I64x4Ops: 'static + archmage::SimdToken + Copy {
    #[allow(private_bounds)]
    type Repr: Copy + Clone + Send + Sync + crate::simd_storage::Pod;
    fn splat(self, v: i64) -> Self::Repr;
    fn zero(self) -> Self::Repr;
    fn load(self, data: &[i64; 4]) -> Self::Repr;
    fn from_array(self, arr: [i64; 4]) -> Self::Repr;
    fn store(self, repr: Self::Repr, out: &mut [i64; 4]);
    fn to_array(self, repr: Self::Repr) -> [i64; 4];
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn neg(self, a: Self::Repr) -> Self::Repr;
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn abs(self, a: Self::Repr) -> Self::Repr;
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr;
    fn reduce_add(self, a: Self::Repr) -> i64;
    fn not(self, a: Self::Repr) -> Self::Repr;
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shr_arithmetic_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn all_true(self, a: Self::Repr) -> bool;
    fn any_true(self, a: Self::Repr) -> bool;
    fn bitmask(self, a: Self::Repr) -> u32;
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <Self as I64x4Ops>::min(self, <Self as I64x4Ops>::max(self, a, lo), hi)
    }
}

pub trait I8x16Ops: 'static + archmage::SimdToken + Copy {
    #[allow(private_bounds)]
    type Repr: Copy + Clone + Send + Sync + crate::simd_storage::Pod;
    fn splat(self, v: i8) -> Self::Repr;
    fn zero(self) -> Self::Repr;
    fn load(self, data: &[i8; 16]) -> Self::Repr;
    fn from_array(self, arr: [i8; 16]) -> Self::Repr;
    fn store(self, repr: Self::Repr, out: &mut [i8; 16]);
    fn to_array(self, repr: Self::Repr) -> [i8; 16];
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn neg(self, a: Self::Repr) -> Self::Repr;
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn abs(self, a: Self::Repr) -> Self::Repr;
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr;
    fn reduce_add(self, a: Self::Repr) -> i8;
    fn not(self, a: Self::Repr) -> Self::Repr;
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shr_arithmetic_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn saturating_add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn saturating_sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn all_true(self, a: Self::Repr) -> bool;
    fn any_true(self, a: Self::Repr) -> bool;
    fn bitmask(self, a: Self::Repr) -> u32;
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <Self as I8x16Ops>::min(self, <Self as I8x16Ops>::max(self, a, lo), hi)
    }
    fn widen_low_i8_to_i16(self, a: <Self as I8x16Ops>::Repr) -> <Self as I16x8Ops>::Repr
    where
        Self: I16x8Ops;
    fn widen_high_i8_to_i16(self, a: <Self as I8x16Ops>::Repr) -> <Self as I16x8Ops>::Repr
    where
        Self: I16x8Ops;
}

pub trait I8x32Ops: 'static + archmage::SimdToken + Copy {
    #[allow(private_bounds)]
    type Repr: Copy + Clone + Send + Sync + crate::simd_storage::Pod;
    fn splat(self, v: i8) -> Self::Repr;
    fn zero(self) -> Self::Repr;
    fn load(self, data: &[i8; 32]) -> Self::Repr;
    fn from_array(self, arr: [i8; 32]) -> Self::Repr;
    fn store(self, repr: Self::Repr, out: &mut [i8; 32]);
    fn to_array(self, repr: Self::Repr) -> [i8; 32];
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn neg(self, a: Self::Repr) -> Self::Repr;
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn abs(self, a: Self::Repr) -> Self::Repr;
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr;
    fn reduce_add(self, a: Self::Repr) -> i8;
    fn not(self, a: Self::Repr) -> Self::Repr;
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shr_arithmetic_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn saturating_add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn saturating_sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn all_true(self, a: Self::Repr) -> bool;
    fn any_true(self, a: Self::Repr) -> bool;
    fn bitmask(self, a: Self::Repr) -> u32;
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <Self as I8x32Ops>::min(self, <Self as I8x32Ops>::max(self, a, lo), hi)
    }
    fn widen_low_i8_to_i16(self, a: <Self as I8x32Ops>::Repr) -> <Self as I16x16Ops>::Repr
    where
        Self: I16x16Ops;
    fn widen_high_i8_to_i16(self, a: <Self as I8x32Ops>::Repr) -> <Self as I16x16Ops>::Repr
    where
        Self: I16x16Ops;
}

pub trait U8x16Ops: 'static + archmage::SimdToken + Copy {
    #[allow(private_bounds)]
    type Repr: Copy + Clone + Send + Sync + crate::simd_storage::Pod;
    fn splat(self, v: u8) -> Self::Repr;
    fn zero(self) -> Self::Repr;
    fn load(self, data: &[u8; 16]) -> Self::Repr;
    fn from_array(self, arr: [u8; 16]) -> Self::Repr;
    fn store(self, repr: Self::Repr, out: &mut [u8; 16]);
    fn to_array(self, repr: Self::Repr) -> [u8; 16];
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr;
    fn reduce_add(self, a: Self::Repr) -> u8;
    fn not(self, a: Self::Repr) -> Self::Repr;
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn saturating_add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn saturating_sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn all_true(self, a: Self::Repr) -> bool;
    fn any_true(self, a: Self::Repr) -> bool;
    fn bitmask(self, a: Self::Repr) -> u32;
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <Self as U8x16Ops>::min(self, <Self as U8x16Ops>::max(self, a, lo), hi)
    }
    fn widen_low_u8_to_u16(self, a: <Self as U8x16Ops>::Repr) -> <Self as U16x8Ops>::Repr
    where
        Self: U16x8Ops;
    fn widen_high_u8_to_u16(self, a: <Self as U8x16Ops>::Repr) -> <Self as U16x8Ops>::Repr
    where
        Self: U16x8Ops;
    fn abs_diff(
        self,
        a: <Self as U8x16Ops>::Repr,
        b: <Self as U8x16Ops>::Repr,
    ) -> <Self as U8x16Ops>::Repr;
    fn reduce_add_u32(self, a: <Self as U8x16Ops>::Repr) -> u32;
    fn sum_abs_diff(self, a: <Self as U8x16Ops>::Repr, b: <Self as U8x16Ops>::Repr) -> u32;
    fn pairwise_widen_add(self, a: <Self as U8x16Ops>::Repr) -> <Self as U16x8Ops>::Repr
    where
        Self: U16x8Ops;
}

pub trait U8x32Ops: 'static + archmage::SimdToken + Copy {
    #[allow(private_bounds)]
    type Repr: Copy + Clone + Send + Sync + crate::simd_storage::Pod;
    fn splat(self, v: u8) -> Self::Repr;
    fn zero(self) -> Self::Repr;
    fn load(self, data: &[u8; 32]) -> Self::Repr;
    fn from_array(self, arr: [u8; 32]) -> Self::Repr;
    fn store(self, repr: Self::Repr, out: &mut [u8; 32]);
    fn to_array(self, repr: Self::Repr) -> [u8; 32];
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr;
    fn reduce_add(self, a: Self::Repr) -> u8;
    fn not(self, a: Self::Repr) -> Self::Repr;
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn saturating_add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn saturating_sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn all_true(self, a: Self::Repr) -> bool;
    fn any_true(self, a: Self::Repr) -> bool;
    fn bitmask(self, a: Self::Repr) -> u32;
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <Self as U8x32Ops>::min(self, <Self as U8x32Ops>::max(self, a, lo), hi)
    }
    fn widen_low_u8_to_u16(self, a: <Self as U8x32Ops>::Repr) -> <Self as U16x16Ops>::Repr
    where
        Self: U16x16Ops;
    fn widen_high_u8_to_u16(self, a: <Self as U8x32Ops>::Repr) -> <Self as U16x16Ops>::Repr
    where
        Self: U16x16Ops;
    fn abs_diff(
        self,
        a: <Self as U8x32Ops>::Repr,
        b: <Self as U8x32Ops>::Repr,
    ) -> <Self as U8x32Ops>::Repr;
    fn reduce_add_u32(self, a: <Self as U8x32Ops>::Repr) -> u32;
    fn sum_abs_diff(self, a: <Self as U8x32Ops>::Repr, b: <Self as U8x32Ops>::Repr) -> u32;
    fn pairwise_widen_add(self, a: <Self as U8x32Ops>::Repr) -> <Self as U16x16Ops>::Repr
    where
        Self: U16x16Ops;
}

pub trait I16x8Ops: 'static + archmage::SimdToken + Copy {
    #[allow(private_bounds)]
    type Repr: Copy + Clone + Send + Sync + crate::simd_storage::Pod;
    fn splat(self, v: i16) -> Self::Repr;
    fn zero(self) -> Self::Repr;
    fn load(self, data: &[i16; 8]) -> Self::Repr;
    fn from_array(self, arr: [i16; 8]) -> Self::Repr;
    fn store(self, repr: Self::Repr, out: &mut [i16; 8]);
    fn to_array(self, repr: Self::Repr) -> [i16; 8];
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn mul(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    #[inline(always)]
    fn mul_hrs_q15(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        let aa = <Self as I16x8Ops>::to_array(self, a);
        let bb = <Self as I16x8Ops>::to_array(self, b);
        <Self as I16x8Ops>::from_array(
            self,
            core::array::from_fn(|i| ((aa[i] as i32 * bb[i] as i32 + 0x4000) >> 15) as i16),
        )
    }
    fn neg(self, a: Self::Repr) -> Self::Repr;
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn abs(self, a: Self::Repr) -> Self::Repr;
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr;
    fn reduce_add(self, a: Self::Repr) -> i16;
    fn not(self, a: Self::Repr) -> Self::Repr;
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shr_arithmetic_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shl_uniform(self, a: Self::Repr, count: u32) -> Self::Repr;
    fn shr_logical_uniform(self, a: Self::Repr, count: u32) -> Self::Repr;
    fn shr_arithmetic_uniform(self, a: Self::Repr, count: u32) -> Self::Repr;
    fn saturating_add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn saturating_sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn all_true(self, a: Self::Repr) -> bool;
    fn any_true(self, a: Self::Repr) -> bool;
    fn bitmask(self, a: Self::Repr) -> u32;
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <Self as I16x8Ops>::min(self, <Self as I16x8Ops>::max(self, a, lo), hi)
    }
    fn widen_low_i16_to_i32(self, a: <Self as I16x8Ops>::Repr) -> <Self as I32x4Ops>::Repr
    where
        Self: I32x4Ops;
    fn widen_high_i16_to_i32(self, a: <Self as I16x8Ops>::Repr) -> <Self as I32x4Ops>::Repr
    where
        Self: I32x4Ops;
    fn narrow_saturating_i16_to_i8(
        self,
        a: <Self as I16x8Ops>::Repr,
        b: <Self as I16x8Ops>::Repr,
    ) -> <Self as I8x16Ops>::Repr
    where
        Self: I8x16Ops;
    fn narrow_saturating_i16_to_u8(
        self,
        a: <Self as I16x8Ops>::Repr,
        b: <Self as I16x8Ops>::Repr,
    ) -> <Self as U8x16Ops>::Repr
    where
        Self: U8x16Ops;
    fn abs_diff(
        self,
        a: <Self as I16x8Ops>::Repr,
        b: <Self as I16x8Ops>::Repr,
    ) -> <Self as U16x8Ops>::Repr
    where
        Self: U16x8Ops;
    fn madd_adjacent(
        self,
        a: <Self as I16x8Ops>::Repr,
        b: <Self as I16x8Ops>::Repr,
    ) -> <Self as I32x4Ops>::Repr
    where
        Self: I32x4Ops;
}

pub trait I16x16Ops: 'static + archmage::SimdToken + Copy {
    #[allow(private_bounds)]
    type Repr: Copy + Clone + Send + Sync + crate::simd_storage::Pod;
    fn splat(self, v: i16) -> Self::Repr;
    fn zero(self) -> Self::Repr;
    fn load(self, data: &[i16; 16]) -> Self::Repr;
    fn from_array(self, arr: [i16; 16]) -> Self::Repr;
    fn store(self, repr: Self::Repr, out: &mut [i16; 16]);
    fn to_array(self, repr: Self::Repr) -> [i16; 16];
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn mul(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    #[inline(always)]
    fn mul_hrs_q15(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        let aa = <Self as I16x16Ops>::to_array(self, a);
        let bb = <Self as I16x16Ops>::to_array(self, b);
        <Self as I16x16Ops>::from_array(
            self,
            core::array::from_fn(|i| ((aa[i] as i32 * bb[i] as i32 + 0x4000) >> 15) as i16),
        )
    }
    fn neg(self, a: Self::Repr) -> Self::Repr;
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn abs(self, a: Self::Repr) -> Self::Repr;
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr;
    fn reduce_add(self, a: Self::Repr) -> i16;
    fn not(self, a: Self::Repr) -> Self::Repr;
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shr_arithmetic_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shl_uniform(self, a: Self::Repr, count: u32) -> Self::Repr;
    fn shr_logical_uniform(self, a: Self::Repr, count: u32) -> Self::Repr;
    fn shr_arithmetic_uniform(self, a: Self::Repr, count: u32) -> Self::Repr;
    fn saturating_add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn saturating_sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn all_true(self, a: Self::Repr) -> bool;
    fn any_true(self, a: Self::Repr) -> bool;
    fn bitmask(self, a: Self::Repr) -> u32;
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <Self as I16x16Ops>::min(self, <Self as I16x16Ops>::max(self, a, lo), hi)
    }
    fn widen_low_i16_to_i32(self, a: <Self as I16x16Ops>::Repr) -> <Self as I32x8Ops>::Repr
    where
        Self: I32x8Ops;
    fn widen_high_i16_to_i32(self, a: <Self as I16x16Ops>::Repr) -> <Self as I32x8Ops>::Repr
    where
        Self: I32x8Ops;
    fn narrow_saturating_i16_to_i8(
        self,
        a: <Self as I16x16Ops>::Repr,
        b: <Self as I16x16Ops>::Repr,
    ) -> <Self as I8x32Ops>::Repr
    where
        Self: I8x32Ops;
    fn narrow_saturating_i16_to_u8(
        self,
        a: <Self as I16x16Ops>::Repr,
        b: <Self as I16x16Ops>::Repr,
    ) -> <Self as U8x32Ops>::Repr
    where
        Self: U8x32Ops;
    fn abs_diff(
        self,
        a: <Self as I16x16Ops>::Repr,
        b: <Self as I16x16Ops>::Repr,
    ) -> <Self as U16x16Ops>::Repr
    where
        Self: U16x16Ops;
    fn madd_adjacent(
        self,
        a: <Self as I16x16Ops>::Repr,
        b: <Self as I16x16Ops>::Repr,
    ) -> <Self as I32x8Ops>::Repr
    where
        Self: I32x8Ops;
}

pub trait U16x8Ops: 'static + archmage::SimdToken + Copy {
    #[allow(private_bounds)]
    type Repr: Copy + Clone + Send + Sync + crate::simd_storage::Pod;
    fn splat(self, v: u16) -> Self::Repr;
    fn zero(self) -> Self::Repr;
    fn load(self, data: &[u16; 8]) -> Self::Repr;
    fn from_array(self, arr: [u16; 8]) -> Self::Repr;
    fn store(self, repr: Self::Repr, out: &mut [u16; 8]);
    fn to_array(self, repr: Self::Repr) -> [u16; 8];
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn mul(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr;
    fn reduce_add(self, a: Self::Repr) -> u16;
    fn not(self, a: Self::Repr) -> Self::Repr;
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shl_uniform(self, a: Self::Repr, count: u32) -> Self::Repr;
    fn shr_logical_uniform(self, a: Self::Repr, count: u32) -> Self::Repr;
    fn saturating_add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn saturating_sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn all_true(self, a: Self::Repr) -> bool;
    fn any_true(self, a: Self::Repr) -> bool;
    fn bitmask(self, a: Self::Repr) -> u32;
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <Self as U16x8Ops>::min(self, <Self as U16x8Ops>::max(self, a, lo), hi)
    }
    fn widen_low_u16_to_u32(self, a: <Self as U16x8Ops>::Repr) -> <Self as U32x4Ops>::Repr
    where
        Self: U32x4Ops;
    fn widen_high_u16_to_u32(self, a: <Self as U16x8Ops>::Repr) -> <Self as U32x4Ops>::Repr
    where
        Self: U32x4Ops;
    fn pairwise_widen_add(self, a: <Self as U16x8Ops>::Repr) -> <Self as U32x4Ops>::Repr
    where
        Self: U32x4Ops;
}

pub trait U16x16Ops: 'static + archmage::SimdToken + Copy {
    #[allow(private_bounds)]
    type Repr: Copy + Clone + Send + Sync + crate::simd_storage::Pod;
    fn splat(self, v: u16) -> Self::Repr;
    fn zero(self) -> Self::Repr;
    fn load(self, data: &[u16; 16]) -> Self::Repr;
    fn from_array(self, arr: [u16; 16]) -> Self::Repr;
    fn store(self, repr: Self::Repr, out: &mut [u16; 16]);
    fn to_array(self, repr: Self::Repr) -> [u16; 16];
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn mul(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr;
    fn reduce_add(self, a: Self::Repr) -> u16;
    fn not(self, a: Self::Repr) -> Self::Repr;
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shl_uniform(self, a: Self::Repr, count: u32) -> Self::Repr;
    fn shr_logical_uniform(self, a: Self::Repr, count: u32) -> Self::Repr;
    fn saturating_add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn saturating_sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn all_true(self, a: Self::Repr) -> bool;
    fn any_true(self, a: Self::Repr) -> bool;
    fn bitmask(self, a: Self::Repr) -> u32;
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <Self as U16x16Ops>::min(self, <Self as U16x16Ops>::max(self, a, lo), hi)
    }
    fn widen_low_u16_to_u32(self, a: <Self as U16x16Ops>::Repr) -> <Self as U32x8Ops>::Repr
    where
        Self: U32x8Ops;
    fn widen_high_u16_to_u32(self, a: <Self as U16x16Ops>::Repr) -> <Self as U32x8Ops>::Repr
    where
        Self: U32x8Ops;
    fn pairwise_widen_add(self, a: <Self as U16x16Ops>::Repr) -> <Self as U32x8Ops>::Repr
    where
        Self: U32x8Ops;
}

pub trait U64x2Ops: 'static + archmage::SimdToken + Copy {
    #[allow(private_bounds)]
    type Repr: Copy + Clone + Send + Sync + crate::simd_storage::Pod;
    fn splat(self, v: u64) -> Self::Repr;
    fn zero(self) -> Self::Repr;
    fn load(self, data: &[u64; 2]) -> Self::Repr;
    fn from_array(self, arr: [u64; 2]) -> Self::Repr;
    fn store(self, repr: Self::Repr, out: &mut [u64; 2]);
    fn to_array(self, repr: Self::Repr) -> [u64; 2];
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr;
    fn reduce_add(self, a: Self::Repr) -> u64;
    fn not(self, a: Self::Repr) -> Self::Repr;
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn all_true(self, a: Self::Repr) -> bool;
    fn any_true(self, a: Self::Repr) -> bool;
    fn bitmask(self, a: Self::Repr) -> u32;
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <Self as U64x2Ops>::min(self, <Self as U64x2Ops>::max(self, a, lo), hi)
    }
}

pub trait U64x4Ops: 'static + archmage::SimdToken + Copy {
    #[allow(private_bounds)]
    type Repr: Copy + Clone + Send + Sync + crate::simd_storage::Pod;
    fn splat(self, v: u64) -> Self::Repr;
    fn zero(self) -> Self::Repr;
    fn load(self, data: &[u64; 4]) -> Self::Repr;
    fn from_array(self, arr: [u64; 4]) -> Self::Repr;
    fn store(self, repr: Self::Repr, out: &mut [u64; 4]);
    fn to_array(self, repr: Self::Repr) -> [u64; 4];
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr;
    fn reduce_add(self, a: Self::Repr) -> u64;
    fn not(self, a: Self::Repr) -> Self::Repr;
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn all_true(self, a: Self::Repr) -> bool;
    fn any_true(self, a: Self::Repr) -> bool;
    fn bitmask(self, a: Self::Repr) -> u32;
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <Self as U64x4Ops>::min(self, <Self as U64x4Ops>::max(self, a, lo), hi)
    }
}

#[cfg(feature = "w512")]
pub trait F32x16Ops: 'static + archmage::SimdToken + Copy {
    #[allow(private_bounds)]
    type Repr: Copy + Clone + Send + Sync + crate::simd_storage::Pod;
    fn splat(self, v: f32) -> Self::Repr;
    fn zero(self) -> Self::Repr;
    fn load(self, data: &[f32; 16]) -> Self::Repr;
    fn from_array(self, arr: [f32; 16]) -> Self::Repr;
    fn store(self, repr: Self::Repr, out: &mut [f32; 16]);
    fn to_array(self, repr: Self::Repr) -> [f32; 16];
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn mul(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn div(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn neg(self, a: Self::Repr) -> Self::Repr;
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    #[inline(always)]
    fn min_fast(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <Self as F32x16Ops>::min(self, a, b)
    }
    #[inline(always)]
    fn max_fast(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <Self as F32x16Ops>::max(self, a, b)
    }
    fn sqrt(self, a: Self::Repr) -> Self::Repr;
    fn abs(self, a: Self::Repr) -> Self::Repr;
    fn floor(self, a: Self::Repr) -> Self::Repr;
    fn ceil(self, a: Self::Repr) -> Self::Repr;
    fn round(self, a: Self::Repr) -> Self::Repr;
    fn mul_add(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr;
    fn mul_sub(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr;
    fn mul_add_portable(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr;
    fn mul_sub_portable(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr;
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr;
    fn reduce_add(self, a: Self::Repr) -> f32;
    fn reduce_min(self, a: Self::Repr) -> f32;
    fn reduce_max(self, a: Self::Repr) -> f32;
    #[inline(always)]
    fn rcp_approx(self, a: Self::Repr) -> Self::Repr {
        a
    }
    #[inline(always)]
    fn rsqrt_approx(self, a: Self::Repr) -> Self::Repr {
        a
    }
    fn not(self, a: Self::Repr) -> Self::Repr;
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <Self as F32x16Ops>::min(self, <Self as F32x16Ops>::max(self, a, lo), hi)
    }
    #[inline(always)]
    fn recip(self, a: Self::Repr) -> Self::Repr {
        Self::rcp_approx(self, a)
    }
    #[inline(always)]
    fn rsqrt(self, a: Self::Repr) -> Self::Repr {
        Self::rsqrt_approx(self, a)
    }
    #[inline(always)]
    fn to_u8_bytes(self, a: Self::Repr) -> [u8; 16] {
        let arr = <Self as F32x16Ops>::to_array(self, a);
        core::array::from_fn(|i| crate::nostd_math::roundevenf(arr[i]).clamp(0.0, 255.0) as u8)
    }
    #[inline(always)]
    fn store_rgba_bytes(
        self,
        r: Self::Repr,
        g: Self::Repr,
        b: Self::Repr,
        a: Self::Repr,
    ) -> [u8; 64] {
        let rb = <Self as F32x16Ops>::to_u8_bytes(self, r);
        let gb = <Self as F32x16Ops>::to_u8_bytes(self, g);
        let bb = <Self as F32x16Ops>::to_u8_bytes(self, b);
        let ab = <Self as F32x16Ops>::to_u8_bytes(self, a);
        let mut out = [0u8; 64];
        let mut i = 0;
        while i < 16 {
            out[i * 4] = rb[i];
            out[i * 4 + 1] = gb[i];
            out[i * 4 + 2] = bb[i];
            out[i * 4 + 3] = ab[i];
            i += 1;
        }
        out
    }
    #[inline(always)]
    fn transpose_dual_8x8_repr(self, rows: [Self::Repr; 8]) -> [Self::Repr; 8] {
        let r: [[f32; 16]; 8] =
            core::array::from_fn(|i| <Self as F32x16Ops>::to_array(self, rows[i]));
        core::array::from_fn(|i| {
            let out: [f32; 16] =
                core::array::from_fn(|k| if k < 8 { r[k][i] } else { r[k - 8][i + 8] });
            <Self as F32x16Ops>::from_array(self, out)
        })
    }
    #[inline(always)]
    fn transpose_16x16_repr(self, rows: [Self::Repr; 16]) -> [Self::Repr; 16] {
        let r: [[f32; 16]; 16] =
            core::array::from_fn(|i| <Self as F32x16Ops>::to_array(self, rows[i]));
        core::array::from_fn(|i| {
            <Self as F32x16Ops>::from_array(self, core::array::from_fn(|j| r[j][i]))
        })
    }
}

#[cfg(feature = "w512")]
pub trait F64x8Ops: 'static + archmage::SimdToken + Copy {
    #[allow(private_bounds)]
    type Repr: Copy + Clone + Send + Sync + crate::simd_storage::Pod;
    fn splat(self, v: f64) -> Self::Repr;
    fn zero(self) -> Self::Repr;
    fn load(self, data: &[f64; 8]) -> Self::Repr;
    fn from_array(self, arr: [f64; 8]) -> Self::Repr;
    fn store(self, repr: Self::Repr, out: &mut [f64; 8]);
    fn to_array(self, repr: Self::Repr) -> [f64; 8];
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn mul(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn div(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn neg(self, a: Self::Repr) -> Self::Repr;
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    #[inline(always)]
    fn min_fast(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <Self as F64x8Ops>::min(self, a, b)
    }
    #[inline(always)]
    fn max_fast(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        <Self as F64x8Ops>::max(self, a, b)
    }
    fn sqrt(self, a: Self::Repr) -> Self::Repr;
    fn abs(self, a: Self::Repr) -> Self::Repr;
    fn floor(self, a: Self::Repr) -> Self::Repr;
    fn ceil(self, a: Self::Repr) -> Self::Repr;
    fn round(self, a: Self::Repr) -> Self::Repr;
    fn mul_add(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr;
    fn mul_sub(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr;
    fn mul_add_portable(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr;
    fn mul_sub_portable(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr;
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr;
    fn reduce_add(self, a: Self::Repr) -> f64;
    fn reduce_min(self, a: Self::Repr) -> f64;
    fn reduce_max(self, a: Self::Repr) -> f64;
    #[inline(always)]
    fn rcp_approx(self, a: Self::Repr) -> Self::Repr {
        a
    }
    #[inline(always)]
    fn rsqrt_approx(self, a: Self::Repr) -> Self::Repr {
        a
    }
    fn not(self, a: Self::Repr) -> Self::Repr;
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <Self as F64x8Ops>::min(self, <Self as F64x8Ops>::max(self, a, lo), hi)
    }
    #[inline(always)]
    fn recip(self, a: Self::Repr) -> Self::Repr {
        Self::rcp_approx(self, a)
    }
    #[inline(always)]
    fn rsqrt(self, a: Self::Repr) -> Self::Repr {
        Self::rsqrt_approx(self, a)
    }
}

#[cfg(feature = "w512")]
pub trait I8x64Ops: 'static + archmage::SimdToken + Copy {
    #[allow(private_bounds)]
    type Repr: Copy + Clone + Send + Sync + crate::simd_storage::Pod;
    fn splat(self, v: i8) -> Self::Repr;
    fn zero(self) -> Self::Repr;
    fn load(self, data: &[i8; 64]) -> Self::Repr;
    fn from_array(self, arr: [i8; 64]) -> Self::Repr;
    fn store(self, repr: Self::Repr, out: &mut [i8; 64]);
    fn to_array(self, repr: Self::Repr) -> [i8; 64];
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn neg(self, a: Self::Repr) -> Self::Repr;
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn abs(self, a: Self::Repr) -> Self::Repr;
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr;
    fn reduce_add(self, a: Self::Repr) -> i8;
    fn not(self, a: Self::Repr) -> Self::Repr;
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shr_arithmetic_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn saturating_add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn saturating_sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn all_true(self, a: Self::Repr) -> bool;
    fn any_true(self, a: Self::Repr) -> bool;
    fn bitmask(self, a: Self::Repr) -> u64;
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <Self as I8x64Ops>::min(self, <Self as I8x64Ops>::max(self, a, lo), hi)
    }
    fn widen_low_i8_to_i16(self, a: <Self as I8x64Ops>::Repr) -> <Self as I16x32Ops>::Repr
    where
        Self: I16x32Ops;
    fn widen_high_i8_to_i16(self, a: <Self as I8x64Ops>::Repr) -> <Self as I16x32Ops>::Repr
    where
        Self: I16x32Ops;
}

#[cfg(feature = "w512")]
pub trait U8x64Ops: 'static + archmage::SimdToken + Copy {
    #[allow(private_bounds)]
    type Repr: Copy + Clone + Send + Sync + crate::simd_storage::Pod;
    fn splat(self, v: u8) -> Self::Repr;
    fn zero(self) -> Self::Repr;
    fn load(self, data: &[u8; 64]) -> Self::Repr;
    fn from_array(self, arr: [u8; 64]) -> Self::Repr;
    fn store(self, repr: Self::Repr, out: &mut [u8; 64]);
    fn to_array(self, repr: Self::Repr) -> [u8; 64];
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn neg(self, a: Self::Repr) -> Self::Repr;
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr;
    fn reduce_add(self, a: Self::Repr) -> u8;
    fn not(self, a: Self::Repr) -> Self::Repr;
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shr_arithmetic_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn saturating_add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn saturating_sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn all_true(self, a: Self::Repr) -> bool;
    fn any_true(self, a: Self::Repr) -> bool;
    fn bitmask(self, a: Self::Repr) -> u64;
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <Self as U8x64Ops>::min(self, <Self as U8x64Ops>::max(self, a, lo), hi)
    }
    fn widen_low_u8_to_u16(self, a: <Self as U8x64Ops>::Repr) -> <Self as U16x32Ops>::Repr
    where
        Self: U16x32Ops;
    fn widen_high_u8_to_u16(self, a: <Self as U8x64Ops>::Repr) -> <Self as U16x32Ops>::Repr
    where
        Self: U16x32Ops;
    fn abs_diff(
        self,
        a: <Self as U8x64Ops>::Repr,
        b: <Self as U8x64Ops>::Repr,
    ) -> <Self as U8x64Ops>::Repr;
    fn reduce_add_u32(self, a: <Self as U8x64Ops>::Repr) -> u32;
    fn sum_abs_diff(self, a: <Self as U8x64Ops>::Repr, b: <Self as U8x64Ops>::Repr) -> u32;
    fn pairwise_widen_add(self, a: <Self as U8x64Ops>::Repr) -> <Self as U16x32Ops>::Repr
    where
        Self: U16x32Ops;
}

#[cfg(feature = "w512")]
pub trait I16x32Ops: 'static + archmage::SimdToken + Copy {
    #[allow(private_bounds)]
    type Repr: Copy + Clone + Send + Sync + crate::simd_storage::Pod;
    fn splat(self, v: i16) -> Self::Repr;
    fn zero(self) -> Self::Repr;
    fn load(self, data: &[i16; 32]) -> Self::Repr;
    fn from_array(self, arr: [i16; 32]) -> Self::Repr;
    fn store(self, repr: Self::Repr, out: &mut [i16; 32]);
    fn to_array(self, repr: Self::Repr) -> [i16; 32];
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn mul(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    #[inline(always)]
    fn mul_hrs_q15(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
        let aa = <Self as I16x32Ops>::to_array(self, a);
        let bb = <Self as I16x32Ops>::to_array(self, b);
        <Self as I16x32Ops>::from_array(
            self,
            core::array::from_fn(|i| ((aa[i] as i32 * bb[i] as i32 + 0x4000) >> 15) as i16),
        )
    }
    fn neg(self, a: Self::Repr) -> Self::Repr;
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn abs(self, a: Self::Repr) -> Self::Repr;
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr;
    fn reduce_add(self, a: Self::Repr) -> i16;
    fn not(self, a: Self::Repr) -> Self::Repr;
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shr_arithmetic_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shl_uniform(self, a: Self::Repr, count: u32) -> Self::Repr;
    fn shr_logical_uniform(self, a: Self::Repr, count: u32) -> Self::Repr;
    fn shr_arithmetic_uniform(self, a: Self::Repr, count: u32) -> Self::Repr;
    fn saturating_add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn saturating_sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn all_true(self, a: Self::Repr) -> bool;
    fn any_true(self, a: Self::Repr) -> bool;
    fn bitmask(self, a: Self::Repr) -> u64;
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <Self as I16x32Ops>::min(self, <Self as I16x32Ops>::max(self, a, lo), hi)
    }
    fn widen_low_i16_to_i32(self, a: <Self as I16x32Ops>::Repr) -> <Self as I32x16Ops>::Repr
    where
        Self: I32x16Ops;
    fn widen_high_i16_to_i32(self, a: <Self as I16x32Ops>::Repr) -> <Self as I32x16Ops>::Repr
    where
        Self: I32x16Ops;
    fn narrow_saturating_i16_to_i8(
        self,
        a: <Self as I16x32Ops>::Repr,
        b: <Self as I16x32Ops>::Repr,
    ) -> <Self as I8x64Ops>::Repr
    where
        Self: I8x64Ops;
    fn narrow_saturating_i16_to_u8(
        self,
        a: <Self as I16x32Ops>::Repr,
        b: <Self as I16x32Ops>::Repr,
    ) -> <Self as U8x64Ops>::Repr
    where
        Self: U8x64Ops;
    fn abs_diff(
        self,
        a: <Self as I16x32Ops>::Repr,
        b: <Self as I16x32Ops>::Repr,
    ) -> <Self as U16x32Ops>::Repr
    where
        Self: U16x32Ops;
    fn madd_adjacent(
        self,
        a: <Self as I16x32Ops>::Repr,
        b: <Self as I16x32Ops>::Repr,
    ) -> <Self as I32x16Ops>::Repr
    where
        Self: I32x16Ops;
}

#[cfg(feature = "w512")]
pub trait U16x32Ops: 'static + archmage::SimdToken + Copy {
    #[allow(private_bounds)]
    type Repr: Copy + Clone + Send + Sync + crate::simd_storage::Pod;
    fn splat(self, v: u16) -> Self::Repr;
    fn zero(self) -> Self::Repr;
    fn load(self, data: &[u16; 32]) -> Self::Repr;
    fn from_array(self, arr: [u16; 32]) -> Self::Repr;
    fn store(self, repr: Self::Repr, out: &mut [u16; 32]);
    fn to_array(self, repr: Self::Repr) -> [u16; 32];
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn mul(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn neg(self, a: Self::Repr) -> Self::Repr;
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr;
    fn reduce_add(self, a: Self::Repr) -> u16;
    fn not(self, a: Self::Repr) -> Self::Repr;
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shr_arithmetic_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shl_uniform(self, a: Self::Repr, count: u32) -> Self::Repr;
    fn shr_logical_uniform(self, a: Self::Repr, count: u32) -> Self::Repr;
    fn shr_arithmetic_uniform(self, a: Self::Repr, count: u32) -> Self::Repr;
    fn saturating_add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn saturating_sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn all_true(self, a: Self::Repr) -> bool;
    fn any_true(self, a: Self::Repr) -> bool;
    fn bitmask(self, a: Self::Repr) -> u64;
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <Self as U16x32Ops>::min(self, <Self as U16x32Ops>::max(self, a, lo), hi)
    }
    fn widen_low_u16_to_u32(self, a: <Self as U16x32Ops>::Repr) -> <Self as U32x16Ops>::Repr
    where
        Self: U32x16Ops;
    fn widen_high_u16_to_u32(self, a: <Self as U16x32Ops>::Repr) -> <Self as U32x16Ops>::Repr
    where
        Self: U32x16Ops;
    fn pairwise_widen_add(self, a: <Self as U16x32Ops>::Repr) -> <Self as U32x16Ops>::Repr
    where
        Self: U32x16Ops;
}

#[cfg(feature = "w512")]
pub trait I32x16Ops: 'static + archmage::SimdToken + Copy {
    #[allow(private_bounds)]
    type Repr: Copy + Clone + Send + Sync + crate::simd_storage::Pod;
    fn splat(self, v: i32) -> Self::Repr;
    fn zero(self) -> Self::Repr;
    fn load(self, data: &[i32; 16]) -> Self::Repr;
    fn from_array(self, arr: [i32; 16]) -> Self::Repr;
    fn store(self, repr: Self::Repr, out: &mut [i32; 16]);
    fn to_array(self, repr: Self::Repr) -> [i32; 16];
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn mul(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn neg(self, a: Self::Repr) -> Self::Repr;
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn abs(self, a: Self::Repr) -> Self::Repr;
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr;
    fn reduce_add(self, a: Self::Repr) -> i32;
    fn not(self, a: Self::Repr) -> Self::Repr;
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shr_arithmetic_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shl_uniform(self, a: Self::Repr, count: u32) -> Self::Repr;
    fn shr_logical_uniform(self, a: Self::Repr, count: u32) -> Self::Repr;
    fn shr_arithmetic_uniform(self, a: Self::Repr, count: u32) -> Self::Repr;
    fn all_true(self, a: Self::Repr) -> bool;
    fn any_true(self, a: Self::Repr) -> bool;
    fn bitmask(self, a: Self::Repr) -> u64;
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <Self as I32x16Ops>::min(self, <Self as I32x16Ops>::max(self, a, lo), hi)
    }
    fn narrow_saturating_i32_to_i16(
        self,
        a: <Self as I32x16Ops>::Repr,
        b: <Self as I32x16Ops>::Repr,
    ) -> <Self as I16x32Ops>::Repr
    where
        Self: I16x32Ops;
    fn narrow_saturating_i32_to_u16(
        self,
        a: <Self as I32x16Ops>::Repr,
        b: <Self as I32x16Ops>::Repr,
    ) -> <Self as U16x32Ops>::Repr
    where
        Self: U16x32Ops;
}

#[cfg(feature = "w512")]
pub trait U32x16Ops: 'static + archmage::SimdToken + Copy {
    #[allow(private_bounds)]
    type Repr: Copy + Clone + Send + Sync + crate::simd_storage::Pod;
    fn splat(self, v: u32) -> Self::Repr;
    fn zero(self) -> Self::Repr;
    fn load(self, data: &[u32; 16]) -> Self::Repr;
    fn from_array(self, arr: [u32; 16]) -> Self::Repr;
    fn store(self, repr: Self::Repr, out: &mut [u32; 16]);
    fn to_array(self, repr: Self::Repr) -> [u32; 16];
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn mul(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn neg(self, a: Self::Repr) -> Self::Repr;
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr;
    fn reduce_add(self, a: Self::Repr) -> u32;
    fn not(self, a: Self::Repr) -> Self::Repr;
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shr_arithmetic_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shl_uniform(self, a: Self::Repr, count: u32) -> Self::Repr;
    fn shr_logical_uniform(self, a: Self::Repr, count: u32) -> Self::Repr;
    fn shr_arithmetic_uniform(self, a: Self::Repr, count: u32) -> Self::Repr;
    fn all_true(self, a: Self::Repr) -> bool;
    fn any_true(self, a: Self::Repr) -> bool;
    fn bitmask(self, a: Self::Repr) -> u64;
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <Self as U32x16Ops>::min(self, <Self as U32x16Ops>::max(self, a, lo), hi)
    }
}

#[cfg(feature = "w512")]
pub trait I64x8Ops: 'static + archmage::SimdToken + Copy {
    #[allow(private_bounds)]
    type Repr: Copy + Clone + Send + Sync + crate::simd_storage::Pod;
    fn splat(self, v: i64) -> Self::Repr;
    fn zero(self) -> Self::Repr;
    fn load(self, data: &[i64; 8]) -> Self::Repr;
    fn from_array(self, arr: [i64; 8]) -> Self::Repr;
    fn store(self, repr: Self::Repr, out: &mut [i64; 8]);
    fn to_array(self, repr: Self::Repr) -> [i64; 8];
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn neg(self, a: Self::Repr) -> Self::Repr;
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn abs(self, a: Self::Repr) -> Self::Repr;
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr;
    fn reduce_add(self, a: Self::Repr) -> i64;
    fn not(self, a: Self::Repr) -> Self::Repr;
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shr_arithmetic_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn all_true(self, a: Self::Repr) -> bool;
    fn any_true(self, a: Self::Repr) -> bool;
    fn bitmask(self, a: Self::Repr) -> u64;
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <Self as I64x8Ops>::min(self, <Self as I64x8Ops>::max(self, a, lo), hi)
    }
}

#[cfg(feature = "w512")]
pub trait U64x8Ops: 'static + archmage::SimdToken + Copy {
    #[allow(private_bounds)]
    type Repr: Copy + Clone + Send + Sync + crate::simd_storage::Pod;
    fn splat(self, v: u64) -> Self::Repr;
    fn zero(self) -> Self::Repr;
    fn load(self, data: &[u64; 8]) -> Self::Repr;
    fn from_array(self, arr: [u64; 8]) -> Self::Repr;
    fn store(self, repr: Self::Repr, out: &mut [u64; 8]);
    fn to_array(self, repr: Self::Repr) -> [u64; 8];
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn neg(self, a: Self::Repr) -> Self::Repr;
    fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr;
    fn reduce_add(self, a: Self::Repr) -> u64;
    fn not(self, a: Self::Repr) -> Self::Repr;
    fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shr_arithmetic_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
    fn all_true(self, a: Self::Repr) -> bool;
    fn any_true(self, a: Self::Repr) -> bool;
    fn bitmask(self, a: Self::Repr) -> u64;
    #[inline(always)]
    fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {
        <Self as U64x8Ops>::min(self, <Self as U64x8Ops>::max(self, a, lo), hi)
    }
}

pub trait F32x4ConvertOps: F32x4Ops + I32x4Ops + 'static + archmage::SimdToken + Copy {
    fn bitcast_f32_to_i32(self, a: <Self as F32x4Ops>::Repr) -> <Self as I32x4Ops>::Repr;
    fn bitcast_i32_to_f32(self, a: <Self as I32x4Ops>::Repr) -> <Self as F32x4Ops>::Repr;
    fn convert_f32_to_i32(self, a: <Self as F32x4Ops>::Repr) -> <Self as I32x4Ops>::Repr;
    #[inline(always)]
    fn convert_f32_to_i32_saturating(
        self,
        a: <Self as F32x4Ops>::Repr,
    ) -> <Self as I32x4Ops>::Repr {
        <Self as F32x4ConvertOps>::convert_f32_to_i32(self, a)
    }
    #[inline(always)]
    fn convert_f32_to_i32_fast(self, a: <Self as F32x4Ops>::Repr) -> <Self as I32x4Ops>::Repr {
        <Self as F32x4ConvertOps>::convert_f32_to_i32(self, a)
    }
    fn convert_f32_to_i32_round(self, a: <Self as F32x4Ops>::Repr) -> <Self as I32x4Ops>::Repr;
    fn convert_i32_to_f32(self, a: <Self as I32x4Ops>::Repr) -> <Self as F32x4Ops>::Repr;
}

pub trait F32x8ConvertOps: F32x8Ops + I32x8Ops + 'static + archmage::SimdToken + Copy {
    fn bitcast_f32_to_i32(self, a: <Self as F32x8Ops>::Repr) -> <Self as I32x8Ops>::Repr;
    fn bitcast_i32_to_f32(self, a: <Self as I32x8Ops>::Repr) -> <Self as F32x8Ops>::Repr;
    fn convert_f32_to_i32(self, a: <Self as F32x8Ops>::Repr) -> <Self as I32x8Ops>::Repr;
    #[inline(always)]
    fn convert_f32_to_i32_saturating(
        self,
        a: <Self as F32x8Ops>::Repr,
    ) -> <Self as I32x8Ops>::Repr {
        <Self as F32x8ConvertOps>::convert_f32_to_i32(self, a)
    }
    #[inline(always)]
    fn convert_f32_to_i32_fast(self, a: <Self as F32x8Ops>::Repr) -> <Self as I32x8Ops>::Repr {
        <Self as F32x8ConvertOps>::convert_f32_to_i32(self, a)
    }
    fn convert_f32_to_i32_round(self, a: <Self as F32x8Ops>::Repr) -> <Self as I32x8Ops>::Repr;
    fn convert_i32_to_f32(self, a: <Self as I32x8Ops>::Repr) -> <Self as F32x8Ops>::Repr;
}

pub trait F64x2ConvertOps: F64x2Ops + I32x4Ops + 'static + archmage::SimdToken + Copy {
    fn convert_f64_to_i32_low(self, a: <Self as F64x2Ops>::Repr) -> <Self as I32x4Ops>::Repr;
    #[inline(always)]
    fn convert_f64_to_i32_low_saturating(
        self,
        a: <Self as F64x2Ops>::Repr,
    ) -> <Self as I32x4Ops>::Repr {
        <Self as F64x2ConvertOps>::convert_f64_to_i32_low(self, a)
    }
    fn convert_f64_to_i32_low_round(self, a: <Self as F64x2Ops>::Repr) -> <Self as I32x4Ops>::Repr;
    fn convert_i32_low_to_f64(self, a: <Self as I32x4Ops>::Repr) -> <Self as F64x2Ops>::Repr;
}

pub trait F64x4ConvertOps: F64x4Ops + I32x4Ops + 'static + archmage::SimdToken + Copy {
    fn convert_f64_to_i32(self, a: <Self as F64x4Ops>::Repr) -> <Self as I32x4Ops>::Repr;
    #[inline(always)]
    fn convert_f64_to_i32_saturating(
        self,
        a: <Self as F64x4Ops>::Repr,
    ) -> <Self as I32x4Ops>::Repr {
        <Self as F64x4ConvertOps>::convert_f64_to_i32(self, a)
    }
    fn convert_f64_to_i32_round(self, a: <Self as F64x4Ops>::Repr) -> <Self as I32x4Ops>::Repr;
    fn convert_i32_to_f64(self, a: <Self as I32x4Ops>::Repr) -> <Self as F64x4Ops>::Repr;
}

pub trait U32x4BitcastOps: U32x4Ops + I32x4Ops + 'static + archmage::SimdToken + Copy {
    fn bitcast_u32_to_i32(self, a: <Self as U32x4Ops>::Repr) -> <Self as I32x4Ops>::Repr;
    fn bitcast_i32_to_u32(self, a: <Self as I32x4Ops>::Repr) -> <Self as U32x4Ops>::Repr;
}

pub trait U32x8BitcastOps: U32x8Ops + I32x8Ops + 'static + archmage::SimdToken + Copy {
    fn bitcast_u32_to_i32(self, a: <Self as U32x8Ops>::Repr) -> <Self as I32x8Ops>::Repr;
    fn bitcast_i32_to_u32(self, a: <Self as I32x8Ops>::Repr) -> <Self as U32x8Ops>::Repr;
}

pub trait I64x2BitcastOps: I64x2Ops + F64x2Ops + 'static + archmage::SimdToken + Copy {
    fn bitcast_i64_to_f64(self, a: <Self as I64x2Ops>::Repr) -> <Self as F64x2Ops>::Repr;
    fn bitcast_f64_to_i64(self, a: <Self as F64x2Ops>::Repr) -> <Self as I64x2Ops>::Repr;
}

pub trait I64x4BitcastOps: I64x4Ops + F64x4Ops + 'static + archmage::SimdToken + Copy {
    fn bitcast_i64_to_f64(self, a: <Self as I64x4Ops>::Repr) -> <Self as F64x4Ops>::Repr;
    fn bitcast_f64_to_i64(self, a: <Self as F64x4Ops>::Repr) -> <Self as I64x4Ops>::Repr;
}

#[cfg(feature = "w512")]
pub trait F32x16ConvertOps: F32x16Ops + I32x16Ops + 'static + archmage::SimdToken + Copy {
    fn bitcast_f32_to_i32(self, a: <Self as F32x16Ops>::Repr) -> <Self as I32x16Ops>::Repr;
    fn bitcast_i32_to_f32(self, a: <Self as I32x16Ops>::Repr) -> <Self as F32x16Ops>::Repr;
    fn convert_f32_to_i32(self, a: <Self as F32x16Ops>::Repr) -> <Self as I32x16Ops>::Repr;
    #[inline(always)]
    fn convert_f32_to_i32_saturating(
        self,
        a: <Self as F32x16Ops>::Repr,
    ) -> <Self as I32x16Ops>::Repr {
        <Self as F32x16ConvertOps>::convert_f32_to_i32(self, a)
    }
    #[inline(always)]
    fn convert_f32_to_i32_fast(self, a: <Self as F32x16Ops>::Repr) -> <Self as I32x16Ops>::Repr {
        <Self as F32x16ConvertOps>::convert_f32_to_i32(self, a)
    }
    fn convert_f32_to_i32_round(self, a: <Self as F32x16Ops>::Repr) -> <Self as I32x16Ops>::Repr;
    fn convert_i32_to_f32(self, a: <Self as I32x16Ops>::Repr) -> <Self as F32x16Ops>::Repr;
}

pub trait I8x16BitcastOps: I8x16Ops + U8x16Ops + 'static + archmage::SimdToken + Copy {
    fn bitcast_i8_to_u8(self, a: <Self as I8x16Ops>::Repr) -> <Self as U8x16Ops>::Repr;
    fn bitcast_u8_to_i8(self, a: <Self as U8x16Ops>::Repr) -> <Self as I8x16Ops>::Repr;
}

pub trait I8x32BitcastOps: I8x32Ops + U8x32Ops + 'static + archmage::SimdToken + Copy {
    fn bitcast_i8_to_u8(self, a: <Self as I8x32Ops>::Repr) -> <Self as U8x32Ops>::Repr;
    fn bitcast_u8_to_i8(self, a: <Self as U8x32Ops>::Repr) -> <Self as I8x32Ops>::Repr;
}

pub trait I16x8BitcastOps: I16x8Ops + U16x8Ops + 'static + archmage::SimdToken + Copy {
    fn bitcast_i16_to_u16(self, a: <Self as I16x8Ops>::Repr) -> <Self as U16x8Ops>::Repr;
    fn bitcast_u16_to_i16(self, a: <Self as U16x8Ops>::Repr) -> <Self as I16x8Ops>::Repr;
}

pub trait I16x16BitcastOps: I16x16Ops + U16x16Ops + 'static + archmage::SimdToken + Copy {
    fn bitcast_i16_to_u16(self, a: <Self as I16x16Ops>::Repr) -> <Self as U16x16Ops>::Repr;
    fn bitcast_u16_to_i16(self, a: <Self as U16x16Ops>::Repr) -> <Self as I16x16Ops>::Repr;
}

pub trait U64x2BitcastOps: U64x2Ops + I64x2Ops + 'static + archmage::SimdToken + Copy {
    fn bitcast_u64_to_i64(self, a: <Self as U64x2Ops>::Repr) -> <Self as I64x2Ops>::Repr;
    fn bitcast_i64_to_u64(self, a: <Self as I64x2Ops>::Repr) -> <Self as U64x2Ops>::Repr;
}

pub trait U64x4BitcastOps: U64x4Ops + I64x4Ops + 'static + archmage::SimdToken + Copy {
    fn bitcast_u64_to_i64(self, a: <Self as U64x4Ops>::Repr) -> <Self as I64x4Ops>::Repr;
    fn bitcast_i64_to_u64(self, a: <Self as I64x4Ops>::Repr) -> <Self as U64x4Ops>::Repr;
}

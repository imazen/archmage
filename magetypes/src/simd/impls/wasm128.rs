//! Backend implementations for Wasm128Token (WebAssembly SIMD).
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;

use crate::simd::backends::*;

#[cfg(target_arch = "wasm32")]
impl F32x4Backend for archmage::Wasm128Token {
    type Repr = v128;

    #[inline(always)]
    fn splat(v: f32) -> v128 {
        f32x4_splat(v)
    }
    #[inline(always)]
    fn zero() -> v128 {
        f32x4_splat(0.0)
    }
    #[inline(always)]
    fn load(data: &[f32; 4]) -> v128 {
        unsafe { v128_load(data.as_ptr().cast()) }
    }
    #[inline(always)]
    fn from_array(arr: [f32; 4]) -> v128 {
        unsafe { v128_load(arr.as_ptr().cast()) }
    }
    #[inline(always)]
    fn store(repr: v128, out: &mut [f32; 4]) {
        unsafe { v128_store(out.as_mut_ptr().cast(), repr) };
    }
    #[inline(always)]
    fn to_array(repr: v128) -> [f32; 4] {
        let mut out = [0.0f32; 4];
        unsafe { v128_store(out.as_mut_ptr().cast(), repr) };
        out
    }

    #[inline(always)]
    fn add(a: v128, b: v128) -> v128 {
        f32x4_add(a, b)
    }
    #[inline(always)]
    fn sub(a: v128, b: v128) -> v128 {
        f32x4_sub(a, b)
    }
    #[inline(always)]
    fn mul(a: v128, b: v128) -> v128 {
        f32x4_mul(a, b)
    }
    #[inline(always)]
    fn div(a: v128, b: v128) -> v128 {
        f32x4_div(a, b)
    }
    #[inline(always)]
    fn neg(a: v128) -> v128 {
        f32x4_neg(a)
    }
    #[inline(always)]
    fn min(a: v128, b: v128) -> v128 {
        f32x4_min(a, b)
    }
    #[inline(always)]
    fn max(a: v128, b: v128) -> v128 {
        f32x4_max(a, b)
    }
    #[inline(always)]
    fn sqrt(a: v128) -> v128 {
        f32x4_sqrt(a)
    }
    #[inline(always)]
    fn abs(a: v128) -> v128 {
        f32x4_abs(a)
    }
    #[inline(always)]
    fn floor(a: v128) -> v128 {
        f32x4_floor(a)
    }
    #[inline(always)]
    fn ceil(a: v128) -> v128 {
        f32x4_ceil(a)
    }
    #[inline(always)]
    fn round(a: v128) -> v128 {
        f32x4_nearest(a)
    }
    #[inline(always)]
    fn mul_add(a: v128, b: v128, c: v128) -> v128 {
        f32x4_add(f32x4_mul(a, b), c)
    }
    #[inline(always)]
    fn mul_sub(a: v128, b: v128, c: v128) -> v128 {
        f32x4_sub(f32x4_mul(a, b), c)
    }
    #[inline(always)]
    fn simd_eq(a: v128, b: v128) -> v128 {
        f32x4_eq(a, b)
    }
    #[inline(always)]
    fn simd_ne(a: v128, b: v128) -> v128 {
        f32x4_ne(a, b)
    }
    #[inline(always)]
    fn simd_lt(a: v128, b: v128) -> v128 {
        f32x4_lt(a, b)
    }
    #[inline(always)]
    fn simd_le(a: v128, b: v128) -> v128 {
        f32x4_le(a, b)
    }
    #[inline(always)]
    fn simd_gt(a: v128, b: v128) -> v128 {
        f32x4_gt(a, b)
    }
    #[inline(always)]
    fn simd_ge(a: v128, b: v128) -> v128 {
        f32x4_ge(a, b)
    }
    #[inline(always)]
    fn blend(mask: v128, if_true: v128, if_false: v128) -> v128 {
        v128_bitselect(if_true, if_false, mask)
    }

    #[inline(always)]
    fn reduce_add(a: v128) -> f32 {
        f32x4_extract_lane::<0>(a)
            + f32x4_extract_lane::<1>(a)
            + f32x4_extract_lane::<2>(a)
            + f32x4_extract_lane::<3>(a)
    }
    #[inline(always)]
    fn reduce_min(a: v128) -> f32 {
        let v0 = f32x4_extract_lane::<0>(a);
        let v1 = f32x4_extract_lane::<1>(a);
        let v2 = f32x4_extract_lane::<2>(a);
        let v3 = f32x4_extract_lane::<3>(a);
        v0.min(v1).min(v2.min(v3))
    }
    #[inline(always)]
    fn reduce_max(a: v128) -> f32 {
        let v0 = f32x4_extract_lane::<0>(a);
        let v1 = f32x4_extract_lane::<1>(a);
        let v2 = f32x4_extract_lane::<2>(a);
        let v3 = f32x4_extract_lane::<3>(a);
        v0.max(v1).max(v2.max(v3))
    }

    #[inline(always)]
    fn rcp_approx(a: v128) -> v128 {
        f32x4_div(f32x4_splat(1.0), a)
    }
    #[inline(always)]
    fn rsqrt_approx(a: v128) -> v128 {
        f32x4_div(f32x4_splat(1.0), f32x4_sqrt(a))
    }
    #[inline(always)]
    fn recip(a: v128) -> v128 {
        <archmage::Wasm128Token as F32x4Backend>::rcp_approx(a)
    }
    #[inline(always)]
    fn rsqrt(a: v128) -> v128 {
        <archmage::Wasm128Token as F32x4Backend>::rsqrt_approx(a)
    }

    #[inline(always)]
    fn not(a: v128) -> v128 {
        v128_not(a)
    }
    #[inline(always)]
    fn bitand(a: v128, b: v128) -> v128 {
        v128_and(a, b)
    }
    #[inline(always)]
    fn bitor(a: v128, b: v128) -> v128 {
        v128_or(a, b)
    }
    #[inline(always)]
    fn bitxor(a: v128, b: v128) -> v128 {
        v128_xor(a, b)
    }
}

#[cfg(target_arch = "wasm32")]
impl F32x8Backend for archmage::Wasm128Token {
    type Repr = [v128; 2];

    #[inline(always)]
    fn splat(v: f32) -> [v128; 2] {
        let v4 = f32x4_splat(v);
        [v4, v4]
    }

    #[inline(always)]
    fn zero() -> [v128; 2] {
        let z = f32x4_splat(0.0);
        [z, z]
    }

    #[inline(always)]
    fn load(data: &[f32; 8]) -> [v128; 2] {
        unsafe {
            [
                v128_load(data.as_ptr().add(0).cast()),
                v128_load(data.as_ptr().add(4).cast()),
            ]
        }
    }

    #[inline(always)]
    fn from_array(arr: [f32; 8]) -> [v128; 2] {
        <Self as F32x8Backend>::load(&arr)
    }

    #[inline(always)]
    fn store(repr: [v128; 2], out: &mut [f32; 8]) {
        unsafe {
            v128_store(out.as_mut_ptr().add(0).cast(), repr[0]);
            v128_store(out.as_mut_ptr().add(4).cast(), repr[1]);
        }
    }

    #[inline(always)]
    fn to_array(repr: [v128; 2]) -> [f32; 8] {
        let mut out = [0.0f32; 8];
        <Self as F32x8Backend>::store(repr, &mut out);
        out
    }

    #[inline(always)]
    fn add(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [f32x4_add(a[0], b[0]), f32x4_add(a[1], b[1])]
    }
    #[inline(always)]
    fn sub(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [f32x4_sub(a[0], b[0]), f32x4_sub(a[1], b[1])]
    }
    #[inline(always)]
    fn mul(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [f32x4_mul(a[0], b[0]), f32x4_mul(a[1], b[1])]
    }
    #[inline(always)]
    fn div(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [f32x4_div(a[0], b[0]), f32x4_div(a[1], b[1])]
    }
    #[inline(always)]
    fn neg(a: [v128; 2]) -> [v128; 2] {
        [f32x4_neg(a[0]), f32x4_neg(a[1])]
    }
    #[inline(always)]
    fn min(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [f32x4_min(a[0], b[0]), f32x4_min(a[1], b[1])]
    }
    #[inline(always)]
    fn max(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [f32x4_max(a[0], b[0]), f32x4_max(a[1], b[1])]
    }
    #[inline(always)]
    fn sqrt(a: [v128; 2]) -> [v128; 2] {
        [f32x4_sqrt(a[0]), f32x4_sqrt(a[1])]
    }
    #[inline(always)]
    fn abs(a: [v128; 2]) -> [v128; 2] {
        [f32x4_abs(a[0]), f32x4_abs(a[1])]
    }
    #[inline(always)]
    fn floor(a: [v128; 2]) -> [v128; 2] {
        [f32x4_floor(a[0]), f32x4_floor(a[1])]
    }
    #[inline(always)]
    fn ceil(a: [v128; 2]) -> [v128; 2] {
        [f32x4_ceil(a[0]), f32x4_ceil(a[1])]
    }
    #[inline(always)]
    fn round(a: [v128; 2]) -> [v128; 2] {
        [f32x4_nearest(a[0]), f32x4_nearest(a[1])]
    }

    #[inline(always)]
    fn mul_add(a: [v128; 2], b: [v128; 2], c: [v128; 2]) -> [v128; 2] {
        // WASM has no native FMA
        [
            f32x4_add(f32x4_mul(a[0], b[0]), c[0]),
            f32x4_add(f32x4_mul(a[1], b[1]), c[1]),
        ]
    }

    #[inline(always)]
    fn mul_sub(a: [v128; 2], b: [v128; 2], c: [v128; 2]) -> [v128; 2] {
        [
            f32x4_sub(f32x4_mul(a[0], b[0]), c[0]),
            f32x4_sub(f32x4_mul(a[1], b[1]), c[1]),
        ]
    }

    #[inline(always)]
    fn simd_eq(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [f32x4_eq(a[0], b[0]), f32x4_eq(a[1], b[1])]
    }
    #[inline(always)]
    fn simd_ne(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [f32x4_ne(a[0], b[0]), f32x4_ne(a[1], b[1])]
    }
    #[inline(always)]
    fn simd_lt(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [f32x4_lt(a[0], b[0]), f32x4_lt(a[1], b[1])]
    }
    #[inline(always)]
    fn simd_le(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [f32x4_le(a[0], b[0]), f32x4_le(a[1], b[1])]
    }
    #[inline(always)]
    fn simd_gt(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [f32x4_gt(a[0], b[0]), f32x4_gt(a[1], b[1])]
    }
    #[inline(always)]
    fn simd_ge(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [f32x4_ge(a[0], b[0]), f32x4_ge(a[1], b[1])]
    }

    #[inline(always)]
    fn blend(mask: [v128; 2], if_true: [v128; 2], if_false: [v128; 2]) -> [v128; 2] {
        [
            v128_bitselect(if_true[0], if_false[0], mask[0]),
            v128_bitselect(if_true[1], if_false[1], mask[1]),
        ]
    }

    #[inline(always)]
    fn reduce_add(a: [v128; 2]) -> f32 {
        f32x4_extract_lane::<0>(a[0])
            + f32x4_extract_lane::<1>(a[0])
            + f32x4_extract_lane::<2>(a[0])
            + f32x4_extract_lane::<3>(a[0])
            + f32x4_extract_lane::<0>(a[1])
            + f32x4_extract_lane::<1>(a[1])
            + f32x4_extract_lane::<2>(a[1])
            + f32x4_extract_lane::<3>(a[1])
    }

    #[inline(always)]
    fn reduce_min(a: [v128; 2]) -> f32 {
        let m = f32x4_min(a[0], a[1]);
        let v0 = f32x4_extract_lane::<0>(m);
        let v1 = f32x4_extract_lane::<1>(m);
        let v2 = f32x4_extract_lane::<2>(m);
        let v3 = f32x4_extract_lane::<3>(m);
        v0.min(v1).min(v2.min(v3))
    }

    #[inline(always)]
    fn reduce_max(a: [v128; 2]) -> f32 {
        let m = f32x4_max(a[0], a[1]);
        let v0 = f32x4_extract_lane::<0>(m);
        let v1 = f32x4_extract_lane::<1>(m);
        let v2 = f32x4_extract_lane::<2>(m);
        let v3 = f32x4_extract_lane::<3>(m);
        v0.max(v1).max(v2.max(v3))
    }

    #[inline(always)]
    fn rcp_approx(a: [v128; 2]) -> [v128; 2] {
        let one = f32x4_splat(1.0);
        [f32x4_div(one, a[0]), f32x4_div(one, a[1])]
    }

    #[inline(always)]
    fn rsqrt_approx(a: [v128; 2]) -> [v128; 2] {
        let one = f32x4_splat(1.0);
        [
            f32x4_div(one, f32x4_sqrt(a[0])),
            f32x4_div(one, f32x4_sqrt(a[1])),
        ]
    }

    // Override defaults: WASM has no fast approximation, already full precision
    #[inline(always)]
    fn recip(a: [v128; 2]) -> [v128; 2] {
        <archmage::Wasm128Token as F32x8Backend>::rcp_approx(a)
    }

    #[inline(always)]
    fn rsqrt(a: [v128; 2]) -> [v128; 2] {
        <archmage::Wasm128Token as F32x8Backend>::rsqrt_approx(a)
    }

    #[inline(always)]
    fn not(a: [v128; 2]) -> [v128; 2] {
        [v128_not(a[0]), v128_not(a[1])]
    }
    #[inline(always)]
    fn bitand(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [v128_and(a[0], b[0]), v128_and(a[1], b[1])]
    }
    #[inline(always)]
    fn bitor(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [v128_or(a[0], b[0]), v128_or(a[1], b[1])]
    }
    #[inline(always)]
    fn bitxor(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [v128_xor(a[0], b[0]), v128_xor(a[1], b[1])]
    }
}

#[cfg(target_arch = "wasm32")]
impl F64x2Backend for archmage::Wasm128Token {
    type Repr = v128;

    #[inline(always)]
    fn splat(v: f64) -> v128 {
        f64x2_splat(v)
    }
    #[inline(always)]
    fn zero() -> v128 {
        f64x2_splat(0.0)
    }
    #[inline(always)]
    fn load(data: &[f64; 2]) -> v128 {
        unsafe { v128_load(data.as_ptr().cast()) }
    }
    #[inline(always)]
    fn from_array(arr: [f64; 2]) -> v128 {
        unsafe { v128_load(arr.as_ptr().cast()) }
    }
    #[inline(always)]
    fn store(repr: v128, out: &mut [f64; 2]) {
        unsafe { v128_store(out.as_mut_ptr().cast(), repr) };
    }
    #[inline(always)]
    fn to_array(repr: v128) -> [f64; 2] {
        let mut out = [0.0f64; 2];
        unsafe { v128_store(out.as_mut_ptr().cast(), repr) };
        out
    }

    #[inline(always)]
    fn add(a: v128, b: v128) -> v128 {
        f64x2_add(a, b)
    }
    #[inline(always)]
    fn sub(a: v128, b: v128) -> v128 {
        f64x2_sub(a, b)
    }
    #[inline(always)]
    fn mul(a: v128, b: v128) -> v128 {
        f64x2_mul(a, b)
    }
    #[inline(always)]
    fn div(a: v128, b: v128) -> v128 {
        f64x2_div(a, b)
    }
    #[inline(always)]
    fn neg(a: v128) -> v128 {
        f64x2_neg(a)
    }
    #[inline(always)]
    fn min(a: v128, b: v128) -> v128 {
        f64x2_min(a, b)
    }
    #[inline(always)]
    fn max(a: v128, b: v128) -> v128 {
        f64x2_max(a, b)
    }
    #[inline(always)]
    fn sqrt(a: v128) -> v128 {
        f64x2_sqrt(a)
    }
    #[inline(always)]
    fn abs(a: v128) -> v128 {
        f64x2_abs(a)
    }
    #[inline(always)]
    fn floor(a: v128) -> v128 {
        f64x2_floor(a)
    }
    #[inline(always)]
    fn ceil(a: v128) -> v128 {
        f64x2_ceil(a)
    }
    #[inline(always)]
    fn round(a: v128) -> v128 {
        f64x2_nearest(a)
    }
    #[inline(always)]
    fn mul_add(a: v128, b: v128, c: v128) -> v128 {
        f64x2_add(f64x2_mul(a, b), c)
    }
    #[inline(always)]
    fn mul_sub(a: v128, b: v128, c: v128) -> v128 {
        f64x2_sub(f64x2_mul(a, b), c)
    }
    #[inline(always)]
    fn simd_eq(a: v128, b: v128) -> v128 {
        f64x2_eq(a, b)
    }
    #[inline(always)]
    fn simd_ne(a: v128, b: v128) -> v128 {
        f64x2_ne(a, b)
    }
    #[inline(always)]
    fn simd_lt(a: v128, b: v128) -> v128 {
        f64x2_lt(a, b)
    }
    #[inline(always)]
    fn simd_le(a: v128, b: v128) -> v128 {
        f64x2_le(a, b)
    }
    #[inline(always)]
    fn simd_gt(a: v128, b: v128) -> v128 {
        f64x2_gt(a, b)
    }
    #[inline(always)]
    fn simd_ge(a: v128, b: v128) -> v128 {
        f64x2_ge(a, b)
    }
    #[inline(always)]
    fn blend(mask: v128, if_true: v128, if_false: v128) -> v128 {
        v128_bitselect(if_true, if_false, mask)
    }

    #[inline(always)]
    fn reduce_add(a: v128) -> f64 {
        f64x2_extract_lane::<0>(a) + f64x2_extract_lane::<1>(a)
    }
    #[inline(always)]
    fn reduce_min(a: v128) -> f64 {
        let v0 = f64x2_extract_lane::<0>(a);
        let v1 = f64x2_extract_lane::<1>(a);
        v0.min(v1)
    }
    #[inline(always)]
    fn reduce_max(a: v128) -> f64 {
        let v0 = f64x2_extract_lane::<0>(a);
        let v1 = f64x2_extract_lane::<1>(a);
        v0.max(v1)
    }

    #[inline(always)]
    fn rcp_approx(a: v128) -> v128 {
        f64x2_div(f64x2_splat(1.0), a)
    }
    #[inline(always)]
    fn rsqrt_approx(a: v128) -> v128 {
        f64x2_div(f64x2_splat(1.0), f64x2_sqrt(a))
    }
    #[inline(always)]
    fn recip(a: v128) -> v128 {
        <archmage::Wasm128Token as F64x2Backend>::rcp_approx(a)
    }
    #[inline(always)]
    fn rsqrt(a: v128) -> v128 {
        <archmage::Wasm128Token as F64x2Backend>::rsqrt_approx(a)
    }

    #[inline(always)]
    fn not(a: v128) -> v128 {
        v128_not(a)
    }
    #[inline(always)]
    fn bitand(a: v128, b: v128) -> v128 {
        v128_and(a, b)
    }
    #[inline(always)]
    fn bitor(a: v128, b: v128) -> v128 {
        v128_or(a, b)
    }
    #[inline(always)]
    fn bitxor(a: v128, b: v128) -> v128 {
        v128_xor(a, b)
    }
}

#[cfg(target_arch = "wasm32")]
impl F64x4Backend for archmage::Wasm128Token {
    type Repr = [v128; 2];

    #[inline(always)]
    fn splat(v: f64) -> [v128; 2] {
        let v4 = f64x2_splat(v);
        [v4, v4]
    }

    #[inline(always)]
    fn zero() -> [v128; 2] {
        let z = f64x2_splat(0.0);
        [z, z]
    }

    #[inline(always)]
    fn load(data: &[f64; 4]) -> [v128; 2] {
        unsafe {
            [
                v128_load(data.as_ptr().add(0).cast()),
                v128_load(data.as_ptr().add(2).cast()),
            ]
        }
    }

    #[inline(always)]
    fn from_array(arr: [f64; 4]) -> [v128; 2] {
        <Self as F64x4Backend>::load(&arr)
    }

    #[inline(always)]
    fn store(repr: [v128; 2], out: &mut [f64; 4]) {
        unsafe {
            v128_store(out.as_mut_ptr().add(0).cast(), repr[0]);
            v128_store(out.as_mut_ptr().add(2).cast(), repr[1]);
        }
    }

    #[inline(always)]
    fn to_array(repr: [v128; 2]) -> [f64; 4] {
        let mut out = [0.0f64; 4];
        <Self as F64x4Backend>::store(repr, &mut out);
        out
    }

    #[inline(always)]
    fn add(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [f64x2_add(a[0], b[0]), f64x2_add(a[1], b[1])]
    }
    #[inline(always)]
    fn sub(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [f64x2_sub(a[0], b[0]), f64x2_sub(a[1], b[1])]
    }
    #[inline(always)]
    fn mul(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [f64x2_mul(a[0], b[0]), f64x2_mul(a[1], b[1])]
    }
    #[inline(always)]
    fn div(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [f64x2_div(a[0], b[0]), f64x2_div(a[1], b[1])]
    }
    #[inline(always)]
    fn neg(a: [v128; 2]) -> [v128; 2] {
        [f64x2_neg(a[0]), f64x2_neg(a[1])]
    }
    #[inline(always)]
    fn min(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [f64x2_min(a[0], b[0]), f64x2_min(a[1], b[1])]
    }
    #[inline(always)]
    fn max(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [f64x2_max(a[0], b[0]), f64x2_max(a[1], b[1])]
    }
    #[inline(always)]
    fn sqrt(a: [v128; 2]) -> [v128; 2] {
        [f64x2_sqrt(a[0]), f64x2_sqrt(a[1])]
    }
    #[inline(always)]
    fn abs(a: [v128; 2]) -> [v128; 2] {
        [f64x2_abs(a[0]), f64x2_abs(a[1])]
    }
    #[inline(always)]
    fn floor(a: [v128; 2]) -> [v128; 2] {
        [f64x2_floor(a[0]), f64x2_floor(a[1])]
    }
    #[inline(always)]
    fn ceil(a: [v128; 2]) -> [v128; 2] {
        [f64x2_ceil(a[0]), f64x2_ceil(a[1])]
    }
    #[inline(always)]
    fn round(a: [v128; 2]) -> [v128; 2] {
        [f64x2_nearest(a[0]), f64x2_nearest(a[1])]
    }

    #[inline(always)]
    fn mul_add(a: [v128; 2], b: [v128; 2], c: [v128; 2]) -> [v128; 2] {
        // WASM has no native FMA
        [
            f64x2_add(f64x2_mul(a[0], b[0]), c[0]),
            f64x2_add(f64x2_mul(a[1], b[1]), c[1]),
        ]
    }

    #[inline(always)]
    fn mul_sub(a: [v128; 2], b: [v128; 2], c: [v128; 2]) -> [v128; 2] {
        [
            f64x2_sub(f64x2_mul(a[0], b[0]), c[0]),
            f64x2_sub(f64x2_mul(a[1], b[1]), c[1]),
        ]
    }

    #[inline(always)]
    fn simd_eq(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [f64x2_eq(a[0], b[0]), f64x2_eq(a[1], b[1])]
    }
    #[inline(always)]
    fn simd_ne(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [f64x2_ne(a[0], b[0]), f64x2_ne(a[1], b[1])]
    }
    #[inline(always)]
    fn simd_lt(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [f64x2_lt(a[0], b[0]), f64x2_lt(a[1], b[1])]
    }
    #[inline(always)]
    fn simd_le(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [f64x2_le(a[0], b[0]), f64x2_le(a[1], b[1])]
    }
    #[inline(always)]
    fn simd_gt(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [f64x2_gt(a[0], b[0]), f64x2_gt(a[1], b[1])]
    }
    #[inline(always)]
    fn simd_ge(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [f64x2_ge(a[0], b[0]), f64x2_ge(a[1], b[1])]
    }

    #[inline(always)]
    fn blend(mask: [v128; 2], if_true: [v128; 2], if_false: [v128; 2]) -> [v128; 2] {
        [
            v128_bitselect(if_true[0], if_false[0], mask[0]),
            v128_bitselect(if_true[1], if_false[1], mask[1]),
        ]
    }

    #[inline(always)]
    fn reduce_add(a: [v128; 2]) -> f64 {
        f64x2_extract_lane::<0>(a[0])
            + f64x2_extract_lane::<1>(a[0])
            + f64x2_extract_lane::<0>(a[1])
            + f64x2_extract_lane::<1>(a[1])
    }

    #[inline(always)]
    fn reduce_min(a: [v128; 2]) -> f64 {
        let m = f64x2_min(a[0], a[1]);
        let v0 = f64x2_extract_lane::<0>(m);
        let v1 = f64x2_extract_lane::<1>(m);
        v0.min(v1)
    }

    #[inline(always)]
    fn reduce_max(a: [v128; 2]) -> f64 {
        let m = f64x2_max(a[0], a[1]);
        let v0 = f64x2_extract_lane::<0>(m);
        let v1 = f64x2_extract_lane::<1>(m);
        v0.max(v1)
    }

    #[inline(always)]
    fn rcp_approx(a: [v128; 2]) -> [v128; 2] {
        let one = f64x2_splat(1.0);
        [f64x2_div(one, a[0]), f64x2_div(one, a[1])]
    }

    #[inline(always)]
    fn rsqrt_approx(a: [v128; 2]) -> [v128; 2] {
        let one = f64x2_splat(1.0);
        [
            f64x2_div(one, f64x2_sqrt(a[0])),
            f64x2_div(one, f64x2_sqrt(a[1])),
        ]
    }

    // Override defaults: WASM has no fast approximation, already full precision
    #[inline(always)]
    fn recip(a: [v128; 2]) -> [v128; 2] {
        <archmage::Wasm128Token as F64x4Backend>::rcp_approx(a)
    }

    #[inline(always)]
    fn rsqrt(a: [v128; 2]) -> [v128; 2] {
        <archmage::Wasm128Token as F64x4Backend>::rsqrt_approx(a)
    }

    #[inline(always)]
    fn not(a: [v128; 2]) -> [v128; 2] {
        [v128_not(a[0]), v128_not(a[1])]
    }
    #[inline(always)]
    fn bitand(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [v128_and(a[0], b[0]), v128_and(a[1], b[1])]
    }
    #[inline(always)]
    fn bitor(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [v128_or(a[0], b[0]), v128_or(a[1], b[1])]
    }
    #[inline(always)]
    fn bitxor(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [v128_xor(a[0], b[0]), v128_xor(a[1], b[1])]
    }
}

#[cfg(target_arch = "wasm32")]
impl I32x4Backend for archmage::Wasm128Token {
    type Repr = v128;

    #[inline(always)]
    fn splat(v: i32) -> v128 {
        i32x4_splat(v)
    }
    #[inline(always)]
    fn zero() -> v128 {
        i32x4_splat(0)
    }
    #[inline(always)]
    fn load(data: &[i32; 4]) -> v128 {
        unsafe { v128_load(data.as_ptr().cast()) }
    }
    #[inline(always)]
    fn from_array(arr: [i32; 4]) -> v128 {
        unsafe { v128_load(arr.as_ptr().cast()) }
    }
    #[inline(always)]
    fn store(repr: v128, out: &mut [i32; 4]) {
        unsafe { v128_store(out.as_mut_ptr().cast(), repr) };
    }
    #[inline(always)]
    fn to_array(repr: v128) -> [i32; 4] {
        let mut out = [0i32; 4];
        unsafe { v128_store(out.as_mut_ptr().cast(), repr) };
        out
    }

    #[inline(always)]
    fn add(a: v128, b: v128) -> v128 {
        i32x4_add(a, b)
    }
    #[inline(always)]
    fn sub(a: v128, b: v128) -> v128 {
        i32x4_sub(a, b)
    }
    #[inline(always)]
    fn mul(a: v128, b: v128) -> v128 {
        i32x4_mul(a, b)
    }
    #[inline(always)]
    fn neg(a: v128) -> v128 {
        i32x4_neg(a)
    }
    #[inline(always)]
    fn min(a: v128, b: v128) -> v128 {
        i32x4_min(a, b)
    }
    #[inline(always)]
    fn max(a: v128, b: v128) -> v128 {
        i32x4_max(a, b)
    }
    #[inline(always)]
    fn abs(a: v128) -> v128 {
        i32x4_abs(a)
    }

    #[inline(always)]
    fn simd_eq(a: v128, b: v128) -> v128 {
        i32x4_eq(a, b)
    }
    #[inline(always)]
    fn simd_ne(a: v128, b: v128) -> v128 {
        i32x4_ne(a, b)
    }
    #[inline(always)]
    fn simd_lt(a: v128, b: v128) -> v128 {
        i32x4_lt(a, b)
    }
    #[inline(always)]
    fn simd_le(a: v128, b: v128) -> v128 {
        i32x4_le(a, b)
    }
    #[inline(always)]
    fn simd_gt(a: v128, b: v128) -> v128 {
        i32x4_gt(a, b)
    }
    #[inline(always)]
    fn simd_ge(a: v128, b: v128) -> v128 {
        i32x4_ge(a, b)
    }
    #[inline(always)]
    fn blend(mask: v128, if_true: v128, if_false: v128) -> v128 {
        v128_bitselect(if_true, if_false, mask)
    }

    #[inline(always)]
    fn reduce_add(a: v128) -> i32 {
        i32x4_extract_lane::<0>(a)
            + i32x4_extract_lane::<1>(a)
            + i32x4_extract_lane::<2>(a)
            + i32x4_extract_lane::<3>(a)
    }

    #[inline(always)]
    fn not(a: v128) -> v128 {
        v128_not(a)
    }
    #[inline(always)]
    fn bitand(a: v128, b: v128) -> v128 {
        v128_and(a, b)
    }
    #[inline(always)]
    fn bitor(a: v128, b: v128) -> v128 {
        v128_or(a, b)
    }
    #[inline(always)]
    fn bitxor(a: v128, b: v128) -> v128 {
        v128_xor(a, b)
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: v128) -> v128 {
        i32x4_shl(a, N as u32)
    }
    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: v128) -> v128 {
        i32x4_shr(a, N as u32)
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: v128) -> v128 {
        u32x4_shr(a, N as u32)
    }

    #[inline(always)]
    fn all_true(a: v128) -> bool {
        i32x4_all_true(a)
    }
    #[inline(always)]
    fn any_true(a: v128) -> bool {
        v128_any_true(a)
    }
    #[inline(always)]
    fn bitmask(a: v128) -> u32 {
        i32x4_bitmask(a) as u32
    }
}

#[cfg(target_arch = "wasm32")]
impl I32x8Backend for archmage::Wasm128Token {
    type Repr = [v128; 2];

    #[inline(always)]
    fn splat(v: i32) -> [v128; 2] {
        let v4 = i32x4_splat(v);
        [v4, v4]
    }

    #[inline(always)]
    fn zero() -> [v128; 2] {
        let z = i32x4_splat(0);
        [z, z]
    }

    #[inline(always)]
    fn load(data: &[i32; 8]) -> [v128; 2] {
        unsafe {
            [
                v128_load(data.as_ptr().add(0).cast()),
                v128_load(data.as_ptr().add(4).cast()),
            ]
        }
    }

    #[inline(always)]
    fn from_array(arr: [i32; 8]) -> [v128; 2] {
        <Self as I32x8Backend>::load(&arr)
    }

    #[inline(always)]
    fn store(repr: [v128; 2], out: &mut [i32; 8]) {
        unsafe {
            v128_store(out.as_mut_ptr().add(0).cast(), repr[0]);
            v128_store(out.as_mut_ptr().add(4).cast(), repr[1]);
        }
    }

    #[inline(always)]
    fn to_array(repr: [v128; 2]) -> [i32; 8] {
        let mut out = [0i32; 8];
        <Self as I32x8Backend>::store(repr, &mut out);
        out
    }

    #[inline(always)]
    fn add(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i32x4_add(a[0], b[0]), i32x4_add(a[1], b[1])]
    }
    #[inline(always)]
    fn sub(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i32x4_sub(a[0], b[0]), i32x4_sub(a[1], b[1])]
    }
    #[inline(always)]
    fn mul(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i32x4_mul(a[0], b[0]), i32x4_mul(a[1], b[1])]
    }
    #[inline(always)]
    fn neg(a: [v128; 2]) -> [v128; 2] {
        [i32x4_neg(a[0]), i32x4_neg(a[1])]
    }
    #[inline(always)]
    fn min(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i32x4_min(a[0], b[0]), i32x4_min(a[1], b[1])]
    }
    #[inline(always)]
    fn max(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i32x4_max(a[0], b[0]), i32x4_max(a[1], b[1])]
    }
    #[inline(always)]
    fn abs(a: [v128; 2]) -> [v128; 2] {
        [i32x4_abs(a[0]), i32x4_abs(a[1])]
    }

    #[inline(always)]
    fn simd_eq(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i32x4_eq(a[0], b[0]), i32x4_eq(a[1], b[1])]
    }
    #[inline(always)]
    fn simd_ne(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i32x4_ne(a[0], b[0]), i32x4_ne(a[1], b[1])]
    }
    #[inline(always)]
    fn simd_lt(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i32x4_lt(a[0], b[0]), i32x4_lt(a[1], b[1])]
    }
    #[inline(always)]
    fn simd_le(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i32x4_le(a[0], b[0]), i32x4_le(a[1], b[1])]
    }
    #[inline(always)]
    fn simd_gt(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i32x4_gt(a[0], b[0]), i32x4_gt(a[1], b[1])]
    }
    #[inline(always)]
    fn simd_ge(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i32x4_ge(a[0], b[0]), i32x4_ge(a[1], b[1])]
    }
    #[inline(always)]
    fn blend(mask: [v128; 2], if_true: [v128; 2], if_false: [v128; 2]) -> [v128; 2] {
        [
            v128_bitselect(if_true[0], if_false[0], mask[0]),
            v128_bitselect(if_true[1], if_false[1], mask[1]),
        ]
    }

    #[inline(always)]
    fn reduce_add(a: [v128; 2]) -> i32 {
        i32x4_extract_lane::<0>(a[0])
            + i32x4_extract_lane::<1>(a[0])
            + i32x4_extract_lane::<2>(a[0])
            + i32x4_extract_lane::<3>(a[0])
            + i32x4_extract_lane::<0>(a[1])
            + i32x4_extract_lane::<1>(a[1])
            + i32x4_extract_lane::<2>(a[1])
            + i32x4_extract_lane::<3>(a[1])
    }

    #[inline(always)]
    fn not(a: [v128; 2]) -> [v128; 2] {
        [v128_not(a[0]), v128_not(a[1])]
    }
    #[inline(always)]
    fn bitand(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [v128_and(a[0], b[0]), v128_and(a[1], b[1])]
    }
    #[inline(always)]
    fn bitor(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [v128_or(a[0], b[0]), v128_or(a[1], b[1])]
    }
    #[inline(always)]
    fn bitxor(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [v128_xor(a[0], b[0]), v128_xor(a[1], b[1])]
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: [v128; 2]) -> [v128; 2] {
        [i32x4_shl(a[0], N as u32), i32x4_shl(a[1], N as u32)]
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: [v128; 2]) -> [v128; 2] {
        [i32x4_shr(a[0], N as u32), i32x4_shr(a[1], N as u32)]
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: [v128; 2]) -> [v128; 2] {
        [u32x4_shr(a[0], N as u32), u32x4_shr(a[1], N as u32)]
    }

    #[inline(always)]
    fn all_true(a: [v128; 2]) -> bool {
        i32x4_all_true(a[0]) && i32x4_all_true(a[1])
    }

    #[inline(always)]
    fn any_true(a: [v128; 2]) -> bool {
        v128_any_true(a[0]) || v128_any_true(a[1])
    }

    #[inline(always)]
    fn bitmask(a: [v128; 2]) -> u32 {
        ((i32x4_bitmask(a[0]) as u32) << 0) | ((i32x4_bitmask(a[1]) as u32) << 4)
    }
}

#[cfg(target_arch = "wasm32")]
impl U32x4Backend for archmage::Wasm128Token {
    type Repr = v128;

    #[inline(always)]
    fn splat(v: u32) -> v128 {
        u32x4_splat(v)
    }
    #[inline(always)]
    fn zero() -> v128 {
        u32x4_splat(0)
    }
    #[inline(always)]
    fn load(data: &[u32; 4]) -> v128 {
        unsafe { v128_load(data.as_ptr().cast()) }
    }
    #[inline(always)]
    fn from_array(arr: [u32; 4]) -> v128 {
        unsafe { v128_load(arr.as_ptr().cast()) }
    }
    #[inline(always)]
    fn store(repr: v128, out: &mut [u32; 4]) {
        unsafe { v128_store(out.as_mut_ptr().cast(), repr) };
    }
    #[inline(always)]
    fn to_array(repr: v128) -> [u32; 4] {
        let mut out = [0u32; 4];
        unsafe { v128_store(out.as_mut_ptr().cast(), repr) };
        out
    }

    #[inline(always)]
    fn add(a: v128, b: v128) -> v128 {
        i32x4_add(a, b)
    }
    #[inline(always)]
    fn sub(a: v128, b: v128) -> v128 {
        i32x4_sub(a, b)
    }
    #[inline(always)]
    fn mul(a: v128, b: v128) -> v128 {
        i32x4_mul(a, b)
    }
    #[inline(always)]
    fn min(a: v128, b: v128) -> v128 {
        u32x4_min(a, b)
    }
    #[inline(always)]
    fn max(a: v128, b: v128) -> v128 {
        u32x4_max(a, b)
    }

    #[inline(always)]
    fn simd_eq(a: v128, b: v128) -> v128 {
        i32x4_eq(a, b)
    }
    #[inline(always)]
    fn simd_ne(a: v128, b: v128) -> v128 {
        i32x4_ne(a, b)
    }
    #[inline(always)]
    fn simd_lt(a: v128, b: v128) -> v128 {
        u32x4_lt(a, b)
    }
    #[inline(always)]
    fn simd_le(a: v128, b: v128) -> v128 {
        u32x4_le(a, b)
    }
    #[inline(always)]
    fn simd_gt(a: v128, b: v128) -> v128 {
        u32x4_gt(a, b)
    }
    #[inline(always)]
    fn simd_ge(a: v128, b: v128) -> v128 {
        u32x4_ge(a, b)
    }
    #[inline(always)]
    fn blend(mask: v128, if_true: v128, if_false: v128) -> v128 {
        v128_bitselect(if_true, if_false, mask)
    }

    #[inline(always)]
    fn reduce_add(a: v128) -> u32 {
        (i32x4_extract_lane::<0>(a) as u32).wrapping_add(
            (i32x4_extract_lane::<1>(a) as u32).wrapping_add(
                (i32x4_extract_lane::<2>(a) as u32)
                    .wrapping_add((i32x4_extract_lane::<3>(a) as u32)),
            ),
        )
    }

    #[inline(always)]
    fn not(a: v128) -> v128 {
        v128_not(a)
    }
    #[inline(always)]
    fn bitand(a: v128, b: v128) -> v128 {
        v128_and(a, b)
    }
    #[inline(always)]
    fn bitor(a: v128, b: v128) -> v128 {
        v128_or(a, b)
    }
    #[inline(always)]
    fn bitxor(a: v128, b: v128) -> v128 {
        v128_xor(a, b)
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: v128) -> v128 {
        u32x4_shl(a, N as u32)
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: v128) -> v128 {
        u32x4_shr(a, N as u32)
    }

    #[inline(always)]
    fn all_true(a: v128) -> bool {
        i32x4_all_true(a)
    }
    #[inline(always)]
    fn any_true(a: v128) -> bool {
        v128_any_true(a)
    }
    #[inline(always)]
    fn bitmask(a: v128) -> u32 {
        i32x4_bitmask(a) as u32
    }
}

#[cfg(target_arch = "wasm32")]
impl U32x8Backend for archmage::Wasm128Token {
    type Repr = [v128; 2];

    #[inline(always)]
    fn splat(v: u32) -> [v128; 2] {
        let v4 = u32x4_splat(v);
        [v4, v4]
    }

    #[inline(always)]
    fn zero() -> [v128; 2] {
        let z = u32x4_splat(0);
        [z, z]
    }

    #[inline(always)]
    fn load(data: &[u32; 8]) -> [v128; 2] {
        unsafe {
            [
                v128_load(data.as_ptr().add(0).cast()),
                v128_load(data.as_ptr().add(4).cast()),
            ]
        }
    }

    #[inline(always)]
    fn from_array(arr: [u32; 8]) -> [v128; 2] {
        <Self as U32x8Backend>::load(&arr)
    }

    #[inline(always)]
    fn store(repr: [v128; 2], out: &mut [u32; 8]) {
        unsafe {
            v128_store(out.as_mut_ptr().add(0).cast(), repr[0]);
            v128_store(out.as_mut_ptr().add(4).cast(), repr[1]);
        }
    }

    #[inline(always)]
    fn to_array(repr: [v128; 2]) -> [u32; 8] {
        let mut out = [0u32; 8];
        <Self as U32x8Backend>::store(repr, &mut out);
        out
    }

    #[inline(always)]
    fn add(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i32x4_add(a[0], b[0]), i32x4_add(a[1], b[1])]
    }
    #[inline(always)]
    fn sub(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i32x4_sub(a[0], b[0]), i32x4_sub(a[1], b[1])]
    }
    #[inline(always)]
    fn mul(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i32x4_mul(a[0], b[0]), i32x4_mul(a[1], b[1])]
    }
    #[inline(always)]
    fn min(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [u32x4_min(a[0], b[0]), u32x4_min(a[1], b[1])]
    }
    #[inline(always)]
    fn max(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [u32x4_max(a[0], b[0]), u32x4_max(a[1], b[1])]
    }

    #[inline(always)]
    fn simd_eq(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i32x4_eq(a[0], b[0]), i32x4_eq(a[1], b[1])]
    }
    #[inline(always)]
    fn simd_ne(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i32x4_ne(a[0], b[0]), i32x4_ne(a[1], b[1])]
    }
    #[inline(always)]
    fn simd_lt(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [u32x4_lt(a[0], b[0]), u32x4_lt(a[1], b[1])]
    }
    #[inline(always)]
    fn simd_le(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [u32x4_le(a[0], b[0]), u32x4_le(a[1], b[1])]
    }
    #[inline(always)]
    fn simd_gt(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [u32x4_gt(a[0], b[0]), u32x4_gt(a[1], b[1])]
    }
    #[inline(always)]
    fn simd_ge(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [u32x4_ge(a[0], b[0]), u32x4_ge(a[1], b[1])]
    }
    #[inline(always)]
    fn blend(mask: [v128; 2], if_true: [v128; 2], if_false: [v128; 2]) -> [v128; 2] {
        [
            v128_bitselect(if_true[0], if_false[0], mask[0]),
            v128_bitselect(if_true[1], if_false[1], mask[1]),
        ]
    }

    #[inline(always)]
    fn reduce_add(a: [v128; 2]) -> u32 {
        (i32x4_extract_lane::<0>(a[0]) as u32).wrapping_add(
            (i32x4_extract_lane::<1>(a[0]) as u32).wrapping_add(
                (i32x4_extract_lane::<2>(a[0]) as u32).wrapping_add(
                    (i32x4_extract_lane::<3>(a[0]) as u32).wrapping_add(
                        (i32x4_extract_lane::<0>(a[1]) as u32).wrapping_add(
                            (i32x4_extract_lane::<1>(a[1]) as u32).wrapping_add(
                                (i32x4_extract_lane::<2>(a[1]) as u32)
                                    .wrapping_add((i32x4_extract_lane::<3>(a[1]) as u32)),
                            ),
                        ),
                    ),
                ),
            ),
        )
    }

    #[inline(always)]
    fn not(a: [v128; 2]) -> [v128; 2] {
        [v128_not(a[0]), v128_not(a[1])]
    }
    #[inline(always)]
    fn bitand(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [v128_and(a[0], b[0]), v128_and(a[1], b[1])]
    }
    #[inline(always)]
    fn bitor(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [v128_or(a[0], b[0]), v128_or(a[1], b[1])]
    }
    #[inline(always)]
    fn bitxor(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [v128_xor(a[0], b[0]), v128_xor(a[1], b[1])]
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: [v128; 2]) -> [v128; 2] {
        [u32x4_shl(a[0], N as u32), u32x4_shl(a[1], N as u32)]
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: [v128; 2]) -> [v128; 2] {
        [u32x4_shr(a[0], N as u32), u32x4_shr(a[1], N as u32)]
    }

    #[inline(always)]
    fn all_true(a: [v128; 2]) -> bool {
        i32x4_all_true(a[0]) && i32x4_all_true(a[1])
    }

    #[inline(always)]
    fn any_true(a: [v128; 2]) -> bool {
        v128_any_true(a[0]) || v128_any_true(a[1])
    }

    #[inline(always)]
    fn bitmask(a: [v128; 2]) -> u32 {
        ((i32x4_bitmask(a[0]) as u32) << 0) | ((i32x4_bitmask(a[1]) as u32) << 4)
    }
}

#[cfg(target_arch = "wasm32")]
impl I64x2Backend for archmage::Wasm128Token {
    type Repr = v128;

    #[inline(always)]
    fn splat(v: i64) -> v128 {
        i64x2_splat(v)
    }
    #[inline(always)]
    fn zero() -> v128 {
        i64x2_splat(0i64)
    }
    #[inline(always)]
    fn load(data: &[i64; 2]) -> v128 {
        unsafe { v128_load(data.as_ptr().cast()) }
    }
    #[inline(always)]
    fn from_array(arr: [i64; 2]) -> v128 {
        unsafe { v128_load(arr.as_ptr().cast()) }
    }
    #[inline(always)]
    fn store(repr: v128, out: &mut [i64; 2]) {
        unsafe { v128_store(out.as_mut_ptr().cast(), repr) };
    }
    #[inline(always)]
    fn to_array(repr: v128) -> [i64; 2] {
        let mut out = [0i64; 2];
        unsafe { v128_store(out.as_mut_ptr().cast(), repr) };
        out
    }

    #[inline(always)]
    fn add(a: v128, b: v128) -> v128 {
        i64x2_add(a, b)
    }
    #[inline(always)]
    fn sub(a: v128, b: v128) -> v128 {
        i64x2_sub(a, b)
    }
    #[inline(always)]
    fn neg(a: v128) -> v128 {
        i64x2_neg(a)
    }
    #[inline(always)]
    fn min(a: v128, b: v128) -> v128 {
        // WASM SIMD lacks native i64 min; polyfill via compare+select
        let mask = i64x2_gt(a, b);
        v128_bitselect(b, a, mask)
    }
    #[inline(always)]
    fn max(a: v128, b: v128) -> v128 {
        // WASM SIMD lacks native i64 max; polyfill via compare+select
        let mask = i64x2_gt(a, b);
        v128_bitselect(a, b, mask)
    }
    #[inline(always)]
    fn abs(a: v128) -> v128 {
        // Polyfill: negate negative values
        let negated = i64x2_neg(a);
        let zero = i64x2_splat(0i64);
        let mask = i64x2_lt(a, zero);
        v128_bitselect(negated, a, mask)
    }

    #[inline(always)]
    fn simd_eq(a: v128, b: v128) -> v128 {
        i64x2_eq(a, b)
    }
    #[inline(always)]
    fn simd_ne(a: v128, b: v128) -> v128 {
        i64x2_ne(a, b)
    }
    #[inline(always)]
    fn simd_lt(a: v128, b: v128) -> v128 {
        i64x2_lt(a, b)
    }
    #[inline(always)]
    fn simd_le(a: v128, b: v128) -> v128 {
        i64x2_le(a, b)
    }
    #[inline(always)]
    fn simd_gt(a: v128, b: v128) -> v128 {
        i64x2_gt(a, b)
    }
    #[inline(always)]
    fn simd_ge(a: v128, b: v128) -> v128 {
        i64x2_ge(a, b)
    }
    #[inline(always)]
    fn blend(mask: v128, if_true: v128, if_false: v128) -> v128 {
        v128_bitselect(if_true, if_false, mask)
    }

    #[inline(always)]
    fn reduce_add(a: v128) -> i64 {
        i64x2_extract_lane::<0>(a) + i64x2_extract_lane::<1>(a)
    }

    #[inline(always)]
    fn not(a: v128) -> v128 {
        v128_not(a)
    }
    #[inline(always)]
    fn bitand(a: v128, b: v128) -> v128 {
        v128_and(a, b)
    }
    #[inline(always)]
    fn bitor(a: v128, b: v128) -> v128 {
        v128_or(a, b)
    }
    #[inline(always)]
    fn bitxor(a: v128, b: v128) -> v128 {
        v128_xor(a, b)
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: v128) -> v128 {
        i64x2_shl(a, N as u32)
    }
    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: v128) -> v128 {
        i64x2_shr(a, N as u32)
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: v128) -> v128 {
        u64x2_shr(a, N as u32)
    }

    #[inline(always)]
    fn all_true(a: v128) -> bool {
        i64x2_all_true(a)
    }
    #[inline(always)]
    fn any_true(a: v128) -> bool {
        v128_any_true(a)
    }
    #[inline(always)]
    fn bitmask(a: v128) -> u32 {
        i64x2_bitmask(a) as u32
    }
}

#[cfg(target_arch = "wasm32")]
impl I64x4Backend for archmage::Wasm128Token {
    type Repr = [v128; 2];

    #[inline(always)]
    fn splat(v: i64) -> [v128; 2] {
        let v2 = i64x2_splat(v);
        [v2, v2]
    }

    #[inline(always)]
    fn zero() -> [v128; 2] {
        let z = i64x2_splat(0i64);
        [z, z]
    }

    #[inline(always)]
    fn load(data: &[i64; 4]) -> [v128; 2] {
        unsafe {
            [
                v128_load(data.as_ptr().add(0).cast()),
                v128_load(data.as_ptr().add(2).cast()),
            ]
        }
    }

    #[inline(always)]
    fn from_array(arr: [i64; 4]) -> [v128; 2] {
        <Self as I64x4Backend>::load(&arr)
    }

    #[inline(always)]
    fn store(repr: [v128; 2], out: &mut [i64; 4]) {
        unsafe {
            v128_store(out.as_mut_ptr().add(0).cast(), repr[0]);
            v128_store(out.as_mut_ptr().add(2).cast(), repr[1]);
        }
    }

    #[inline(always)]
    fn to_array(repr: [v128; 2]) -> [i64; 4] {
        let mut out = [0i64; 4];
        <Self as I64x4Backend>::store(repr, &mut out);
        out
    }

    #[inline(always)]
    fn add(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i64x2_add(a[0], b[0]), i64x2_add(a[1], b[1])]
    }
    #[inline(always)]
    fn sub(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i64x2_sub(a[0], b[0]), i64x2_sub(a[1], b[1])]
    }
    #[inline(always)]
    fn neg(a: [v128; 2]) -> [v128; 2] {
        [i64x2_neg(a[0]), i64x2_neg(a[1])]
    }
    #[inline(always)]
    fn min(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        // WASM SIMD lacks native i64 min; polyfill via compare+select per sub-vector
        [
            {
                let mask = i64x2_gt(a[0], b[0]);
                v128_bitselect(b[0], a[0], mask)
            },
            {
                let mask = i64x2_gt(a[1], b[1]);
                v128_bitselect(b[1], a[1], mask)
            },
        ]
    }
    #[inline(always)]
    fn max(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        // WASM SIMD lacks native i64 max; polyfill via compare+select per sub-vector
        [
            {
                let mask = i64x2_gt(a[0], b[0]);
                v128_bitselect(a[0], b[0], mask)
            },
            {
                let mask = i64x2_gt(a[1], b[1]);
                v128_bitselect(a[1], b[1], mask)
            },
        ]
    }
    #[inline(always)]
    fn abs(a: [v128; 2]) -> [v128; 2] {
        // Polyfill: negate negative values per sub-vector
        [
            {
                let neg = i64x2_neg(a[0]);
                let zero = i64x2_splat(0i64);
                let mask = i64x2_lt(a[0], zero);
                v128_bitselect(neg, a[0], mask)
            },
            {
                let neg = i64x2_neg(a[1]);
                let zero = i64x2_splat(0i64);
                let mask = i64x2_lt(a[1], zero);
                v128_bitselect(neg, a[1], mask)
            },
        ]
    }

    #[inline(always)]
    fn simd_eq(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i64x2_eq(a[0], b[0]), i64x2_eq(a[1], b[1])]
    }
    #[inline(always)]
    fn simd_ne(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i64x2_ne(a[0], b[0]), i64x2_ne(a[1], b[1])]
    }
    #[inline(always)]
    fn simd_lt(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i64x2_lt(a[0], b[0]), i64x2_lt(a[1], b[1])]
    }
    #[inline(always)]
    fn simd_le(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i64x2_le(a[0], b[0]), i64x2_le(a[1], b[1])]
    }
    #[inline(always)]
    fn simd_gt(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i64x2_gt(a[0], b[0]), i64x2_gt(a[1], b[1])]
    }
    #[inline(always)]
    fn simd_ge(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i64x2_ge(a[0], b[0]), i64x2_ge(a[1], b[1])]
    }
    #[inline(always)]
    fn blend(mask: [v128; 2], if_true: [v128; 2], if_false: [v128; 2]) -> [v128; 2] {
        [
            v128_bitselect(if_true[0], if_false[0], mask[0]),
            v128_bitselect(if_true[1], if_false[1], mask[1]),
        ]
    }

    #[inline(always)]
    fn reduce_add(a: [v128; 2]) -> i64 {
        i64x2_extract_lane::<0>(a[0])
            + i64x2_extract_lane::<1>(a[0])
            + i64x2_extract_lane::<0>(a[1])
            + i64x2_extract_lane::<1>(a[1])
    }

    #[inline(always)]
    fn not(a: [v128; 2]) -> [v128; 2] {
        [v128_not(a[0]), v128_not(a[1])]
    }
    #[inline(always)]
    fn bitand(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [v128_and(a[0], b[0]), v128_and(a[1], b[1])]
    }
    #[inline(always)]
    fn bitor(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [v128_or(a[0], b[0]), v128_or(a[1], b[1])]
    }
    #[inline(always)]
    fn bitxor(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [v128_xor(a[0], b[0]), v128_xor(a[1], b[1])]
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: [v128; 2]) -> [v128; 2] {
        [i64x2_shl(a[0], N as u32), i64x2_shl(a[1], N as u32)]
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: [v128; 2]) -> [v128; 2] {
        [i64x2_shr(a[0], N as u32), i64x2_shr(a[1], N as u32)]
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: [v128; 2]) -> [v128; 2] {
        [u64x2_shr(a[0], N as u32), u64x2_shr(a[1], N as u32)]
    }

    #[inline(always)]
    fn all_true(a: [v128; 2]) -> bool {
        i64x2_all_true(a[0]) && i64x2_all_true(a[1])
    }

    #[inline(always)]
    fn any_true(a: [v128; 2]) -> bool {
        v128_any_true(a[0]) || v128_any_true(a[1])
    }

    #[inline(always)]
    fn bitmask(a: [v128; 2]) -> u32 {
        ((i64x2_bitmask(a[0]) as u32) << 0) | ((i64x2_bitmask(a[1]) as u32) << 2)
    }
}

#[cfg(target_arch = "wasm32")]
impl I8x16Backend for archmage::Wasm128Token {
    type Repr = v128;

    #[inline(always)]
    fn splat(v: i8) -> v128 {
        i8x16_splat(v)
    }

    #[inline(always)]
    fn zero() -> v128 {
        i8x16_splat(0)
    }

    #[inline(always)]
    fn load(data: &[i8; 16]) -> v128 {
        unsafe { v128_load(data.as_ptr().cast()) }
    }

    #[inline(always)]
    fn from_array(arr: [i8; 16]) -> v128 {
        unsafe { v128_load(arr.as_ptr().cast()) }
    }

    #[inline(always)]
    fn store(repr: v128, out: &mut [i8; 16]) {
        unsafe { v128_store(out.as_mut_ptr().cast(), repr) };
    }

    #[inline(always)]
    fn to_array(repr: v128) -> [i8; 16] {
        let mut out = [0i8; 16];
        unsafe { v128_store(out.as_mut_ptr().cast(), repr) };
        out
    }

    #[inline(always)]
    fn add(a: v128, b: v128) -> v128 {
        i8x16_add(a, b)
    }
    #[inline(always)]
    fn sub(a: v128, b: v128) -> v128 {
        i8x16_sub(a, b)
    }
    #[inline(always)]
    fn neg(a: v128) -> v128 {
        i8x16_neg(a)
    }
    #[inline(always)]
    fn min(a: v128, b: v128) -> v128 {
        i8x16_min(a, b)
    }
    #[inline(always)]
    fn max(a: v128, b: v128) -> v128 {
        i8x16_max(a, b)
    }
    #[inline(always)]
    fn abs(a: v128) -> v128 {
        i8x16_abs(a)
    }

    #[inline(always)]
    fn simd_eq(a: v128, b: v128) -> v128 {
        i8x16_eq(a, b)
    }
    #[inline(always)]
    fn simd_ne(a: v128, b: v128) -> v128 {
        i8x16_ne(a, b)
    }
    #[inline(always)]
    fn simd_lt(a: v128, b: v128) -> v128 {
        i8x16_lt(a, b)
    }
    #[inline(always)]
    fn simd_le(a: v128, b: v128) -> v128 {
        i8x16_le(a, b)
    }
    #[inline(always)]
    fn simd_gt(a: v128, b: v128) -> v128 {
        i8x16_gt(a, b)
    }
    #[inline(always)]
    fn simd_ge(a: v128, b: v128) -> v128 {
        i8x16_ge(a, b)
    }

    #[inline(always)]
    fn blend(mask: v128, if_true: v128, if_false: v128) -> v128 {
        v128_bitselect(if_true, if_false, mask)
    }

    #[inline(always)]
    fn reduce_add(a: v128) -> i8 {
        let arr = <Self as I8x16Backend>::to_array(a);
        arr.iter().copied().fold(0i8, i8::wrapping_add)
    }

    #[inline(always)]
    fn not(a: v128) -> v128 {
        v128_not(a)
    }
    #[inline(always)]
    fn bitand(a: v128, b: v128) -> v128 {
        v128_and(a, b)
    }
    #[inline(always)]
    fn bitor(a: v128, b: v128) -> v128 {
        v128_or(a, b)
    }
    #[inline(always)]
    fn bitxor(a: v128, b: v128) -> v128 {
        v128_xor(a, b)
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: v128) -> v128 {
        i8x16_shl(a, N as u32)
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: v128) -> v128 {
        i8x16_shr(a, N as u32)
    }
    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: v128) -> v128 {
        i8x16_shr(a, N as u32)
    }

    #[inline(always)]
    fn all_true(a: v128) -> bool {
        i8x16_all_true(a)
    }
    #[inline(always)]
    fn any_true(a: v128) -> bool {
        v128_any_true(a)
    }
    #[inline(always)]
    fn bitmask(a: v128) -> u32 {
        i8x16_bitmask(a) as u32
    }
}

#[cfg(target_arch = "wasm32")]
impl I8x32Backend for archmage::Wasm128Token {
    type Repr = [v128; 2];

    #[inline(always)]
    fn splat(v: i8) -> [v128; 2] {
        let v4 = i8x16_splat(v);
        [v4, v4]
    }

    #[inline(always)]
    fn zero() -> [v128; 2] {
        let z = i8x16_splat(0);
        [z, z]
    }

    #[inline(always)]
    fn load(data: &[i8; 32]) -> [v128; 2] {
        unsafe {
            [
                v128_load(data.as_ptr().cast::<u8>().add(0).cast()),
                v128_load(data.as_ptr().cast::<u8>().add(16).cast()),
            ]
        }
    }

    #[inline(always)]
    fn from_array(arr: [i8; 32]) -> [v128; 2] {
        <Self as I8x32Backend>::load(&arr)
    }

    #[inline(always)]
    fn store(repr: [v128; 2], out: &mut [i8; 32]) {
        unsafe {
            v128_store(out.as_mut_ptr().cast::<u8>().add(0).cast(), repr[0]);
            v128_store(out.as_mut_ptr().cast::<u8>().add(16).cast(), repr[1]);
        }
    }

    #[inline(always)]
    fn to_array(repr: [v128; 2]) -> [i8; 32] {
        let mut out = [0i8; 32];
        <Self as I8x32Backend>::store(repr, &mut out);
        out
    }

    #[inline(always)]
    fn add(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i8x16_add(a[0], b[0]), i8x16_add(a[1], b[1])]
    }
    #[inline(always)]
    fn sub(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i8x16_sub(a[0], b[0]), i8x16_sub(a[1], b[1])]
    }
    #[inline(always)]
    fn neg(a: [v128; 2]) -> [v128; 2] {
        [i8x16_neg(a[0]), i8x16_neg(a[1])]
    }
    #[inline(always)]
    fn min(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i8x16_min(a[0], b[0]), i8x16_min(a[1], b[1])]
    }
    #[inline(always)]
    fn max(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i8x16_max(a[0], b[0]), i8x16_max(a[1], b[1])]
    }
    #[inline(always)]
    fn abs(a: [v128; 2]) -> [v128; 2] {
        [i8x16_abs(a[0]), i8x16_abs(a[1])]
    }

    #[inline(always)]
    fn simd_eq(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i8x16_eq(a[0], b[0]), i8x16_eq(a[1], b[1])]
    }
    #[inline(always)]
    fn simd_ne(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i8x16_ne(a[0], b[0]), i8x16_ne(a[1], b[1])]
    }
    #[inline(always)]
    fn simd_lt(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i8x16_lt(a[0], b[0]), i8x16_lt(a[1], b[1])]
    }
    #[inline(always)]
    fn simd_le(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i8x16_le(a[0], b[0]), i8x16_le(a[1], b[1])]
    }
    #[inline(always)]
    fn simd_gt(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i8x16_gt(a[0], b[0]), i8x16_gt(a[1], b[1])]
    }
    #[inline(always)]
    fn simd_ge(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i8x16_ge(a[0], b[0]), i8x16_ge(a[1], b[1])]
    }

    #[inline(always)]
    fn blend(mask: [v128; 2], if_true: [v128; 2], if_false: [v128; 2]) -> [v128; 2] {
        [
            v128_bitselect(if_true[0], if_false[0], mask[0]),
            v128_bitselect(if_true[1], if_false[1], mask[1]),
        ]
    }

    #[inline(always)]
    fn reduce_add(a: [v128; 2]) -> i8 {
        let arr = <Self as I8x32Backend>::to_array(a);
        arr.iter().copied().fold(0i8, i8::wrapping_add)
    }

    #[inline(always)]
    fn not(a: [v128; 2]) -> [v128; 2] {
        [v128_not(a[0]), v128_not(a[1])]
    }
    #[inline(always)]
    fn bitand(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [v128_and(a[0], b[0]), v128_and(a[1], b[1])]
    }
    #[inline(always)]
    fn bitor(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [v128_or(a[0], b[0]), v128_or(a[1], b[1])]
    }
    #[inline(always)]
    fn bitxor(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [v128_xor(a[0], b[0]), v128_xor(a[1], b[1])]
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: [v128; 2]) -> [v128; 2] {
        [i8x16_shl(a[0], N as u32), i8x16_shl(a[1], N as u32)]
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: [v128; 2]) -> [v128; 2] {
        [i8x16_shr(a[0], N as u32), i8x16_shr(a[1], N as u32)]
    }
    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: [v128; 2]) -> [v128; 2] {
        [i8x16_shr(a[0], N as u32), i8x16_shr(a[1], N as u32)]
    }

    #[inline(always)]
    fn all_true(a: [v128; 2]) -> bool {
        i8x16_all_true(a[0]) && i8x16_all_true(a[1])
    }

    #[inline(always)]
    fn any_true(a: [v128; 2]) -> bool {
        v128_any_true(a[0]) || v128_any_true(a[1])
    }

    #[inline(always)]
    fn bitmask(a: [v128; 2]) -> u32 {
        let mut result = 0u32;
        for i in 0..2 {
            result |= (i8x16_bitmask(a[i]) as u32) << (i * 16);
        }
        result
    }
}

#[cfg(target_arch = "wasm32")]
impl U8x16Backend for archmage::Wasm128Token {
    type Repr = v128;

    #[inline(always)]
    fn splat(v: u8) -> v128 {
        u8x16_splat(v)
    }

    #[inline(always)]
    fn zero() -> v128 {
        u8x16_splat(0)
    }

    #[inline(always)]
    fn load(data: &[u8; 16]) -> v128 {
        unsafe { v128_load(data.as_ptr().cast()) }
    }

    #[inline(always)]
    fn from_array(arr: [u8; 16]) -> v128 {
        unsafe { v128_load(arr.as_ptr().cast()) }
    }

    #[inline(always)]
    fn store(repr: v128, out: &mut [u8; 16]) {
        unsafe { v128_store(out.as_mut_ptr().cast(), repr) };
    }

    #[inline(always)]
    fn to_array(repr: v128) -> [u8; 16] {
        let mut out = [0u8; 16];
        unsafe { v128_store(out.as_mut_ptr().cast(), repr) };
        out
    }

    #[inline(always)]
    fn add(a: v128, b: v128) -> v128 {
        i8x16_add(a, b)
    }
    #[inline(always)]
    fn sub(a: v128, b: v128) -> v128 {
        i8x16_sub(a, b)
    }
    #[inline(always)]
    fn min(a: v128, b: v128) -> v128 {
        u8x16_min(a, b)
    }
    #[inline(always)]
    fn max(a: v128, b: v128) -> v128 {
        u8x16_max(a, b)
    }

    #[inline(always)]
    fn simd_eq(a: v128, b: v128) -> v128 {
        i8x16_eq(a, b)
    }
    #[inline(always)]
    fn simd_ne(a: v128, b: v128) -> v128 {
        i8x16_ne(a, b)
    }
    #[inline(always)]
    fn simd_lt(a: v128, b: v128) -> v128 {
        u8x16_lt(a, b)
    }
    #[inline(always)]
    fn simd_le(a: v128, b: v128) -> v128 {
        u8x16_le(a, b)
    }
    #[inline(always)]
    fn simd_gt(a: v128, b: v128) -> v128 {
        u8x16_gt(a, b)
    }
    #[inline(always)]
    fn simd_ge(a: v128, b: v128) -> v128 {
        u8x16_ge(a, b)
    }

    #[inline(always)]
    fn blend(mask: v128, if_true: v128, if_false: v128) -> v128 {
        v128_bitselect(if_true, if_false, mask)
    }

    #[inline(always)]
    fn reduce_add(a: v128) -> u8 {
        let arr = <Self as U8x16Backend>::to_array(a);
        arr.iter().copied().fold(0u8, u8::wrapping_add)
    }

    #[inline(always)]
    fn not(a: v128) -> v128 {
        v128_not(a)
    }
    #[inline(always)]
    fn bitand(a: v128, b: v128) -> v128 {
        v128_and(a, b)
    }
    #[inline(always)]
    fn bitor(a: v128, b: v128) -> v128 {
        v128_or(a, b)
    }
    #[inline(always)]
    fn bitxor(a: v128, b: v128) -> v128 {
        v128_xor(a, b)
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: v128) -> v128 {
        i8x16_shl(a, N as u32)
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: v128) -> v128 {
        u8x16_shr(a, N as u32)
    }

    #[inline(always)]
    fn all_true(a: v128) -> bool {
        i8x16_all_true(a)
    }
    #[inline(always)]
    fn any_true(a: v128) -> bool {
        v128_any_true(a)
    }
    #[inline(always)]
    fn bitmask(a: v128) -> u32 {
        i8x16_bitmask(a) as u32
    }
}

#[cfg(target_arch = "wasm32")]
impl U8x32Backend for archmage::Wasm128Token {
    type Repr = [v128; 2];

    #[inline(always)]
    fn splat(v: u8) -> [v128; 2] {
        let v4 = u8x16_splat(v);
        [v4, v4]
    }

    #[inline(always)]
    fn zero() -> [v128; 2] {
        let z = u8x16_splat(0);
        [z, z]
    }

    #[inline(always)]
    fn load(data: &[u8; 32]) -> [v128; 2] {
        unsafe {
            [
                v128_load(data.as_ptr().cast::<u8>().add(0).cast()),
                v128_load(data.as_ptr().cast::<u8>().add(16).cast()),
            ]
        }
    }

    #[inline(always)]
    fn from_array(arr: [u8; 32]) -> [v128; 2] {
        <Self as U8x32Backend>::load(&arr)
    }

    #[inline(always)]
    fn store(repr: [v128; 2], out: &mut [u8; 32]) {
        unsafe {
            v128_store(out.as_mut_ptr().cast::<u8>().add(0).cast(), repr[0]);
            v128_store(out.as_mut_ptr().cast::<u8>().add(16).cast(), repr[1]);
        }
    }

    #[inline(always)]
    fn to_array(repr: [v128; 2]) -> [u8; 32] {
        let mut out = [0u8; 32];
        <Self as U8x32Backend>::store(repr, &mut out);
        out
    }

    #[inline(always)]
    fn add(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i8x16_add(a[0], b[0]), i8x16_add(a[1], b[1])]
    }
    #[inline(always)]
    fn sub(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i8x16_sub(a[0], b[0]), i8x16_sub(a[1], b[1])]
    }
    #[inline(always)]
    fn min(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [u8x16_min(a[0], b[0]), u8x16_min(a[1], b[1])]
    }
    #[inline(always)]
    fn max(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [u8x16_max(a[0], b[0]), u8x16_max(a[1], b[1])]
    }

    #[inline(always)]
    fn simd_eq(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i8x16_eq(a[0], b[0]), i8x16_eq(a[1], b[1])]
    }
    #[inline(always)]
    fn simd_ne(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i8x16_ne(a[0], b[0]), i8x16_ne(a[1], b[1])]
    }
    #[inline(always)]
    fn simd_lt(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [u8x16_lt(a[0], b[0]), u8x16_lt(a[1], b[1])]
    }
    #[inline(always)]
    fn simd_le(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [u8x16_le(a[0], b[0]), u8x16_le(a[1], b[1])]
    }
    #[inline(always)]
    fn simd_gt(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [u8x16_gt(a[0], b[0]), u8x16_gt(a[1], b[1])]
    }
    #[inline(always)]
    fn simd_ge(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [u8x16_ge(a[0], b[0]), u8x16_ge(a[1], b[1])]
    }

    #[inline(always)]
    fn blend(mask: [v128; 2], if_true: [v128; 2], if_false: [v128; 2]) -> [v128; 2] {
        [
            v128_bitselect(if_true[0], if_false[0], mask[0]),
            v128_bitselect(if_true[1], if_false[1], mask[1]),
        ]
    }

    #[inline(always)]
    fn reduce_add(a: [v128; 2]) -> u8 {
        let arr = <Self as U8x32Backend>::to_array(a);
        arr.iter().copied().fold(0u8, u8::wrapping_add)
    }

    #[inline(always)]
    fn not(a: [v128; 2]) -> [v128; 2] {
        [v128_not(a[0]), v128_not(a[1])]
    }
    #[inline(always)]
    fn bitand(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [v128_and(a[0], b[0]), v128_and(a[1], b[1])]
    }
    #[inline(always)]
    fn bitor(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [v128_or(a[0], b[0]), v128_or(a[1], b[1])]
    }
    #[inline(always)]
    fn bitxor(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [v128_xor(a[0], b[0]), v128_xor(a[1], b[1])]
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: [v128; 2]) -> [v128; 2] {
        [i8x16_shl(a[0], N as u32), i8x16_shl(a[1], N as u32)]
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: [v128; 2]) -> [v128; 2] {
        [u8x16_shr(a[0], N as u32), u8x16_shr(a[1], N as u32)]
    }

    #[inline(always)]
    fn all_true(a: [v128; 2]) -> bool {
        i8x16_all_true(a[0]) && i8x16_all_true(a[1])
    }

    #[inline(always)]
    fn any_true(a: [v128; 2]) -> bool {
        v128_any_true(a[0]) || v128_any_true(a[1])
    }

    #[inline(always)]
    fn bitmask(a: [v128; 2]) -> u32 {
        let mut result = 0u32;
        for i in 0..2 {
            result |= (i8x16_bitmask(a[i]) as u32) << (i * 16);
        }
        result
    }
}

#[cfg(target_arch = "wasm32")]
impl I16x8Backend for archmage::Wasm128Token {
    type Repr = v128;

    #[inline(always)]
    fn splat(v: i16) -> v128 {
        i16x8_splat(v)
    }

    #[inline(always)]
    fn zero() -> v128 {
        i16x8_splat(0)
    }

    #[inline(always)]
    fn load(data: &[i16; 8]) -> v128 {
        unsafe { v128_load(data.as_ptr().cast()) }
    }

    #[inline(always)]
    fn from_array(arr: [i16; 8]) -> v128 {
        unsafe { v128_load(arr.as_ptr().cast()) }
    }

    #[inline(always)]
    fn store(repr: v128, out: &mut [i16; 8]) {
        unsafe { v128_store(out.as_mut_ptr().cast(), repr) };
    }

    #[inline(always)]
    fn to_array(repr: v128) -> [i16; 8] {
        let mut out = [0i16; 8];
        unsafe { v128_store(out.as_mut_ptr().cast(), repr) };
        out
    }

    #[inline(always)]
    fn add(a: v128, b: v128) -> v128 {
        i16x8_add(a, b)
    }
    #[inline(always)]
    fn sub(a: v128, b: v128) -> v128 {
        i16x8_sub(a, b)
    }
    #[inline(always)]
    fn mul(a: v128, b: v128) -> v128 {
        i16x8_mul(a, b)
    }
    #[inline(always)]
    fn neg(a: v128) -> v128 {
        i16x8_neg(a)
    }
    #[inline(always)]
    fn min(a: v128, b: v128) -> v128 {
        i16x8_min(a, b)
    }
    #[inline(always)]
    fn max(a: v128, b: v128) -> v128 {
        i16x8_max(a, b)
    }
    #[inline(always)]
    fn abs(a: v128) -> v128 {
        i16x8_abs(a)
    }

    #[inline(always)]
    fn simd_eq(a: v128, b: v128) -> v128 {
        i16x8_eq(a, b)
    }
    #[inline(always)]
    fn simd_ne(a: v128, b: v128) -> v128 {
        i16x8_ne(a, b)
    }
    #[inline(always)]
    fn simd_lt(a: v128, b: v128) -> v128 {
        i16x8_lt(a, b)
    }
    #[inline(always)]
    fn simd_le(a: v128, b: v128) -> v128 {
        i16x8_le(a, b)
    }
    #[inline(always)]
    fn simd_gt(a: v128, b: v128) -> v128 {
        i16x8_gt(a, b)
    }
    #[inline(always)]
    fn simd_ge(a: v128, b: v128) -> v128 {
        i16x8_ge(a, b)
    }

    #[inline(always)]
    fn blend(mask: v128, if_true: v128, if_false: v128) -> v128 {
        v128_bitselect(if_true, if_false, mask)
    }

    #[inline(always)]
    fn reduce_add(a: v128) -> i16 {
        let arr = <Self as I16x8Backend>::to_array(a);
        arr.iter().copied().fold(0i16, i16::wrapping_add)
    }

    #[inline(always)]
    fn not(a: v128) -> v128 {
        v128_not(a)
    }
    #[inline(always)]
    fn bitand(a: v128, b: v128) -> v128 {
        v128_and(a, b)
    }
    #[inline(always)]
    fn bitor(a: v128, b: v128) -> v128 {
        v128_or(a, b)
    }
    #[inline(always)]
    fn bitxor(a: v128, b: v128) -> v128 {
        v128_xor(a, b)
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: v128) -> v128 {
        i16x8_shl(a, N as u32)
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: v128) -> v128 {
        i16x8_shr(a, N as u32)
    }
    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: v128) -> v128 {
        i16x8_shr(a, N as u32)
    }

    #[inline(always)]
    fn all_true(a: v128) -> bool {
        i16x8_all_true(a)
    }
    #[inline(always)]
    fn any_true(a: v128) -> bool {
        v128_any_true(a)
    }
    #[inline(always)]
    fn bitmask(a: v128) -> u32 {
        i16x8_bitmask(a) as u32
    }
}

#[cfg(target_arch = "wasm32")]
impl I16x16Backend for archmage::Wasm128Token {
    type Repr = [v128; 2];

    #[inline(always)]
    fn splat(v: i16) -> [v128; 2] {
        let v4 = i16x8_splat(v);
        [v4, v4]
    }

    #[inline(always)]
    fn zero() -> [v128; 2] {
        let z = i16x8_splat(0);
        [z, z]
    }

    #[inline(always)]
    fn load(data: &[i16; 16]) -> [v128; 2] {
        unsafe {
            [
                v128_load(data.as_ptr().cast::<u8>().add(0).cast()),
                v128_load(data.as_ptr().cast::<u8>().add(16).cast()),
            ]
        }
    }

    #[inline(always)]
    fn from_array(arr: [i16; 16]) -> [v128; 2] {
        <Self as I16x16Backend>::load(&arr)
    }

    #[inline(always)]
    fn store(repr: [v128; 2], out: &mut [i16; 16]) {
        unsafe {
            v128_store(out.as_mut_ptr().cast::<u8>().add(0).cast(), repr[0]);
            v128_store(out.as_mut_ptr().cast::<u8>().add(16).cast(), repr[1]);
        }
    }

    #[inline(always)]
    fn to_array(repr: [v128; 2]) -> [i16; 16] {
        let mut out = [0i16; 16];
        <Self as I16x16Backend>::store(repr, &mut out);
        out
    }

    #[inline(always)]
    fn add(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i16x8_add(a[0], b[0]), i16x8_add(a[1], b[1])]
    }
    #[inline(always)]
    fn sub(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i16x8_sub(a[0], b[0]), i16x8_sub(a[1], b[1])]
    }
    #[inline(always)]
    fn mul(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i16x8_mul(a[0], b[0]), i16x8_mul(a[1], b[1])]
    }
    #[inline(always)]
    fn neg(a: [v128; 2]) -> [v128; 2] {
        [i16x8_neg(a[0]), i16x8_neg(a[1])]
    }
    #[inline(always)]
    fn min(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i16x8_min(a[0], b[0]), i16x8_min(a[1], b[1])]
    }
    #[inline(always)]
    fn max(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i16x8_max(a[0], b[0]), i16x8_max(a[1], b[1])]
    }
    #[inline(always)]
    fn abs(a: [v128; 2]) -> [v128; 2] {
        [i16x8_abs(a[0]), i16x8_abs(a[1])]
    }

    #[inline(always)]
    fn simd_eq(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i16x8_eq(a[0], b[0]), i16x8_eq(a[1], b[1])]
    }
    #[inline(always)]
    fn simd_ne(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i16x8_ne(a[0], b[0]), i16x8_ne(a[1], b[1])]
    }
    #[inline(always)]
    fn simd_lt(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i16x8_lt(a[0], b[0]), i16x8_lt(a[1], b[1])]
    }
    #[inline(always)]
    fn simd_le(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i16x8_le(a[0], b[0]), i16x8_le(a[1], b[1])]
    }
    #[inline(always)]
    fn simd_gt(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i16x8_gt(a[0], b[0]), i16x8_gt(a[1], b[1])]
    }
    #[inline(always)]
    fn simd_ge(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i16x8_ge(a[0], b[0]), i16x8_ge(a[1], b[1])]
    }

    #[inline(always)]
    fn blend(mask: [v128; 2], if_true: [v128; 2], if_false: [v128; 2]) -> [v128; 2] {
        [
            v128_bitselect(if_true[0], if_false[0], mask[0]),
            v128_bitselect(if_true[1], if_false[1], mask[1]),
        ]
    }

    #[inline(always)]
    fn reduce_add(a: [v128; 2]) -> i16 {
        let arr = <Self as I16x16Backend>::to_array(a);
        arr.iter().copied().fold(0i16, i16::wrapping_add)
    }

    #[inline(always)]
    fn not(a: [v128; 2]) -> [v128; 2] {
        [v128_not(a[0]), v128_not(a[1])]
    }
    #[inline(always)]
    fn bitand(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [v128_and(a[0], b[0]), v128_and(a[1], b[1])]
    }
    #[inline(always)]
    fn bitor(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [v128_or(a[0], b[0]), v128_or(a[1], b[1])]
    }
    #[inline(always)]
    fn bitxor(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [v128_xor(a[0], b[0]), v128_xor(a[1], b[1])]
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: [v128; 2]) -> [v128; 2] {
        [i16x8_shl(a[0], N as u32), i16x8_shl(a[1], N as u32)]
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: [v128; 2]) -> [v128; 2] {
        [i16x8_shr(a[0], N as u32), i16x8_shr(a[1], N as u32)]
    }
    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: [v128; 2]) -> [v128; 2] {
        [i16x8_shr(a[0], N as u32), i16x8_shr(a[1], N as u32)]
    }

    #[inline(always)]
    fn all_true(a: [v128; 2]) -> bool {
        i16x8_all_true(a[0]) && i16x8_all_true(a[1])
    }

    #[inline(always)]
    fn any_true(a: [v128; 2]) -> bool {
        v128_any_true(a[0]) || v128_any_true(a[1])
    }

    #[inline(always)]
    fn bitmask(a: [v128; 2]) -> u32 {
        let mut result = 0u32;
        for i in 0..2 {
            result |= (i16x8_bitmask(a[i]) as u32) << (i * 8);
        }
        result
    }
}

#[cfg(target_arch = "wasm32")]
impl U16x8Backend for archmage::Wasm128Token {
    type Repr = v128;

    #[inline(always)]
    fn splat(v: u16) -> v128 {
        u16x8_splat(v)
    }

    #[inline(always)]
    fn zero() -> v128 {
        u16x8_splat(0)
    }

    #[inline(always)]
    fn load(data: &[u16; 8]) -> v128 {
        unsafe { v128_load(data.as_ptr().cast()) }
    }

    #[inline(always)]
    fn from_array(arr: [u16; 8]) -> v128 {
        unsafe { v128_load(arr.as_ptr().cast()) }
    }

    #[inline(always)]
    fn store(repr: v128, out: &mut [u16; 8]) {
        unsafe { v128_store(out.as_mut_ptr().cast(), repr) };
    }

    #[inline(always)]
    fn to_array(repr: v128) -> [u16; 8] {
        let mut out = [0u16; 8];
        unsafe { v128_store(out.as_mut_ptr().cast(), repr) };
        out
    }

    #[inline(always)]
    fn add(a: v128, b: v128) -> v128 {
        i16x8_add(a, b)
    }
    #[inline(always)]
    fn sub(a: v128, b: v128) -> v128 {
        i16x8_sub(a, b)
    }
    #[inline(always)]
    fn mul(a: v128, b: v128) -> v128 {
        i16x8_mul(a, b)
    }
    #[inline(always)]
    fn min(a: v128, b: v128) -> v128 {
        u16x8_min(a, b)
    }
    #[inline(always)]
    fn max(a: v128, b: v128) -> v128 {
        u16x8_max(a, b)
    }

    #[inline(always)]
    fn simd_eq(a: v128, b: v128) -> v128 {
        i16x8_eq(a, b)
    }
    #[inline(always)]
    fn simd_ne(a: v128, b: v128) -> v128 {
        i16x8_ne(a, b)
    }
    #[inline(always)]
    fn simd_lt(a: v128, b: v128) -> v128 {
        u16x8_lt(a, b)
    }
    #[inline(always)]
    fn simd_le(a: v128, b: v128) -> v128 {
        u16x8_le(a, b)
    }
    #[inline(always)]
    fn simd_gt(a: v128, b: v128) -> v128 {
        u16x8_gt(a, b)
    }
    #[inline(always)]
    fn simd_ge(a: v128, b: v128) -> v128 {
        u16x8_ge(a, b)
    }

    #[inline(always)]
    fn blend(mask: v128, if_true: v128, if_false: v128) -> v128 {
        v128_bitselect(if_true, if_false, mask)
    }

    #[inline(always)]
    fn reduce_add(a: v128) -> u16 {
        let arr = <Self as U16x8Backend>::to_array(a);
        arr.iter().copied().fold(0u16, u16::wrapping_add)
    }

    #[inline(always)]
    fn not(a: v128) -> v128 {
        v128_not(a)
    }
    #[inline(always)]
    fn bitand(a: v128, b: v128) -> v128 {
        v128_and(a, b)
    }
    #[inline(always)]
    fn bitor(a: v128, b: v128) -> v128 {
        v128_or(a, b)
    }
    #[inline(always)]
    fn bitxor(a: v128, b: v128) -> v128 {
        v128_xor(a, b)
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: v128) -> v128 {
        i16x8_shl(a, N as u32)
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: v128) -> v128 {
        u16x8_shr(a, N as u32)
    }

    #[inline(always)]
    fn all_true(a: v128) -> bool {
        i16x8_all_true(a)
    }
    #[inline(always)]
    fn any_true(a: v128) -> bool {
        v128_any_true(a)
    }
    #[inline(always)]
    fn bitmask(a: v128) -> u32 {
        i16x8_bitmask(a) as u32
    }
}

#[cfg(target_arch = "wasm32")]
impl U16x16Backend for archmage::Wasm128Token {
    type Repr = [v128; 2];

    #[inline(always)]
    fn splat(v: u16) -> [v128; 2] {
        let v4 = u16x8_splat(v);
        [v4, v4]
    }

    #[inline(always)]
    fn zero() -> [v128; 2] {
        let z = u16x8_splat(0);
        [z, z]
    }

    #[inline(always)]
    fn load(data: &[u16; 16]) -> [v128; 2] {
        unsafe {
            [
                v128_load(data.as_ptr().cast::<u8>().add(0).cast()),
                v128_load(data.as_ptr().cast::<u8>().add(16).cast()),
            ]
        }
    }

    #[inline(always)]
    fn from_array(arr: [u16; 16]) -> [v128; 2] {
        <Self as U16x16Backend>::load(&arr)
    }

    #[inline(always)]
    fn store(repr: [v128; 2], out: &mut [u16; 16]) {
        unsafe {
            v128_store(out.as_mut_ptr().cast::<u8>().add(0).cast(), repr[0]);
            v128_store(out.as_mut_ptr().cast::<u8>().add(16).cast(), repr[1]);
        }
    }

    #[inline(always)]
    fn to_array(repr: [v128; 2]) -> [u16; 16] {
        let mut out = [0u16; 16];
        <Self as U16x16Backend>::store(repr, &mut out);
        out
    }

    #[inline(always)]
    fn add(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i16x8_add(a[0], b[0]), i16x8_add(a[1], b[1])]
    }
    #[inline(always)]
    fn sub(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i16x8_sub(a[0], b[0]), i16x8_sub(a[1], b[1])]
    }
    #[inline(always)]
    fn mul(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i16x8_mul(a[0], b[0]), i16x8_mul(a[1], b[1])]
    }
    #[inline(always)]
    fn min(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [u16x8_min(a[0], b[0]), u16x8_min(a[1], b[1])]
    }
    #[inline(always)]
    fn max(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [u16x8_max(a[0], b[0]), u16x8_max(a[1], b[1])]
    }

    #[inline(always)]
    fn simd_eq(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i16x8_eq(a[0], b[0]), i16x8_eq(a[1], b[1])]
    }
    #[inline(always)]
    fn simd_ne(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i16x8_ne(a[0], b[0]), i16x8_ne(a[1], b[1])]
    }
    #[inline(always)]
    fn simd_lt(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [u16x8_lt(a[0], b[0]), u16x8_lt(a[1], b[1])]
    }
    #[inline(always)]
    fn simd_le(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [u16x8_le(a[0], b[0]), u16x8_le(a[1], b[1])]
    }
    #[inline(always)]
    fn simd_gt(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [u16x8_gt(a[0], b[0]), u16x8_gt(a[1], b[1])]
    }
    #[inline(always)]
    fn simd_ge(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [u16x8_ge(a[0], b[0]), u16x8_ge(a[1], b[1])]
    }

    #[inline(always)]
    fn blend(mask: [v128; 2], if_true: [v128; 2], if_false: [v128; 2]) -> [v128; 2] {
        [
            v128_bitselect(if_true[0], if_false[0], mask[0]),
            v128_bitselect(if_true[1], if_false[1], mask[1]),
        ]
    }

    #[inline(always)]
    fn reduce_add(a: [v128; 2]) -> u16 {
        let arr = <Self as U16x16Backend>::to_array(a);
        arr.iter().copied().fold(0u16, u16::wrapping_add)
    }

    #[inline(always)]
    fn not(a: [v128; 2]) -> [v128; 2] {
        [v128_not(a[0]), v128_not(a[1])]
    }
    #[inline(always)]
    fn bitand(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [v128_and(a[0], b[0]), v128_and(a[1], b[1])]
    }
    #[inline(always)]
    fn bitor(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [v128_or(a[0], b[0]), v128_or(a[1], b[1])]
    }
    #[inline(always)]
    fn bitxor(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [v128_xor(a[0], b[0]), v128_xor(a[1], b[1])]
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: [v128; 2]) -> [v128; 2] {
        [i16x8_shl(a[0], N as u32), i16x8_shl(a[1], N as u32)]
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: [v128; 2]) -> [v128; 2] {
        [u16x8_shr(a[0], N as u32), u16x8_shr(a[1], N as u32)]
    }

    #[inline(always)]
    fn all_true(a: [v128; 2]) -> bool {
        i16x8_all_true(a[0]) && i16x8_all_true(a[1])
    }

    #[inline(always)]
    fn any_true(a: [v128; 2]) -> bool {
        v128_any_true(a[0]) || v128_any_true(a[1])
    }

    #[inline(always)]
    fn bitmask(a: [v128; 2]) -> u32 {
        let mut result = 0u32;
        for i in 0..2 {
            result |= (i16x8_bitmask(a[i]) as u32) << (i * 8);
        }
        result
    }
}

#[cfg(target_arch = "wasm32")]
impl U64x2Backend for archmage::Wasm128Token {
    type Repr = v128;

    #[inline(always)]
    fn splat(v: u64) -> v128 {
        u64x2_splat(v)
    }

    #[inline(always)]
    fn zero() -> v128 {
        u64x2_splat(0)
    }

    #[inline(always)]
    fn load(data: &[u64; 2]) -> v128 {
        unsafe { v128_load(data.as_ptr().cast()) }
    }

    #[inline(always)]
    fn from_array(arr: [u64; 2]) -> v128 {
        unsafe { v128_load(arr.as_ptr().cast()) }
    }

    #[inline(always)]
    fn store(repr: v128, out: &mut [u64; 2]) {
        unsafe { v128_store(out.as_mut_ptr().cast(), repr) };
    }

    #[inline(always)]
    fn to_array(repr: v128) -> [u64; 2] {
        let mut out = [0u64; 2];
        unsafe { v128_store(out.as_mut_ptr().cast(), repr) };
        out
    }

    #[inline(always)]
    fn add(a: v128, b: v128) -> v128 {
        i64x2_add(a, b)
    }
    #[inline(always)]
    fn sub(a: v128, b: v128) -> v128 {
        i64x2_sub(a, b)
    }
    #[inline(always)]
    fn min(a: v128, b: v128) -> v128 {
        // u64 min polyfill: compare then select
        let mask = <Self as U64x2Backend>::simd_lt(a, b);
        v128_bitselect(a, b, mask)
    }
    #[inline(always)]
    fn max(a: v128, b: v128) -> v128 {
        let mask = <Self as U64x2Backend>::simd_gt(a, b);
        v128_bitselect(a, b, mask)
    }

    #[inline(always)]
    fn simd_eq(a: v128, b: v128) -> v128 {
        i64x2_eq(a, b)
    }
    #[inline(always)]
    fn simd_ne(a: v128, b: v128) -> v128 {
        i64x2_ne(a, b)
    }
    #[inline(always)]
    fn simd_gt(a: v128, b: v128) -> v128 {
        // Unsigned comparison via bias trick
        let bias = i64x2_splat(i64::MIN);
        let sa = v128_xor(a, bias);
        let sb = v128_xor(b, bias);
        i64x2_gt(sa, sb)
    }
    #[inline(always)]
    fn simd_lt(a: v128, b: v128) -> v128 {
        <Self as U64x2Backend>::simd_gt(b, a)
    }
    #[inline(always)]
    fn simd_le(a: v128, b: v128) -> v128 {
        v128_not(<Self as U64x2Backend>::simd_gt(a, b))
    }
    #[inline(always)]
    fn simd_ge(a: v128, b: v128) -> v128 {
        v128_not(<Self as U64x2Backend>::simd_gt(b, a))
    }

    #[inline(always)]
    fn blend(mask: v128, if_true: v128, if_false: v128) -> v128 {
        v128_bitselect(if_true, if_false, mask)
    }

    #[inline(always)]
    fn reduce_add(a: v128) -> u64 {
        let arr = <Self as U64x2Backend>::to_array(a);
        arr.iter().copied().fold(0u64, u64::wrapping_add)
    }

    #[inline(always)]
    fn not(a: v128) -> v128 {
        v128_not(a)
    }
    #[inline(always)]
    fn bitand(a: v128, b: v128) -> v128 {
        v128_and(a, b)
    }
    #[inline(always)]
    fn bitor(a: v128, b: v128) -> v128 {
        v128_or(a, b)
    }
    #[inline(always)]
    fn bitxor(a: v128, b: v128) -> v128 {
        v128_xor(a, b)
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: v128) -> v128 {
        i64x2_shl(a, N as u32)
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: v128) -> v128 {
        u64x2_shr(a, N as u32)
    }

    #[inline(always)]
    fn all_true(a: v128) -> bool {
        i64x2_all_true(a)
    }
    #[inline(always)]
    fn any_true(a: v128) -> bool {
        v128_any_true(a)
    }
    #[inline(always)]
    fn bitmask(a: v128) -> u32 {
        i64x2_bitmask(a) as u32
    }
}

#[cfg(target_arch = "wasm32")]
impl U64x4Backend for archmage::Wasm128Token {
    type Repr = [v128; 2];

    #[inline(always)]
    fn splat(v: u64) -> [v128; 2] {
        let v4 = u64x2_splat(v);
        [v4, v4]
    }

    #[inline(always)]
    fn zero() -> [v128; 2] {
        let z = u64x2_splat(0);
        [z, z]
    }

    #[inline(always)]
    fn load(data: &[u64; 4]) -> [v128; 2] {
        unsafe {
            [
                v128_load(data.as_ptr().cast::<u8>().add(0).cast()),
                v128_load(data.as_ptr().cast::<u8>().add(16).cast()),
            ]
        }
    }

    #[inline(always)]
    fn from_array(arr: [u64; 4]) -> [v128; 2] {
        <Self as U64x4Backend>::load(&arr)
    }

    #[inline(always)]
    fn store(repr: [v128; 2], out: &mut [u64; 4]) {
        unsafe {
            v128_store(out.as_mut_ptr().cast::<u8>().add(0).cast(), repr[0]);
            v128_store(out.as_mut_ptr().cast::<u8>().add(16).cast(), repr[1]);
        }
    }

    #[inline(always)]
    fn to_array(repr: [v128; 2]) -> [u64; 4] {
        let mut out = [0u64; 4];
        <Self as U64x4Backend>::store(repr, &mut out);
        out
    }

    #[inline(always)]
    fn add(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i64x2_add(a[0], b[0]), i64x2_add(a[1], b[1])]
    }
    #[inline(always)]
    fn sub(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i64x2_sub(a[0], b[0]), i64x2_sub(a[1], b[1])]
    }
    #[inline(always)]
    fn min(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [
            <archmage::Wasm128Token as U64x2Backend>::min(a[0], b[0]),
            <archmage::Wasm128Token as U64x2Backend>::min(a[1], b[1]),
        ]
    }
    #[inline(always)]
    fn max(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [
            <archmage::Wasm128Token as U64x2Backend>::max(a[0], b[0]),
            <archmage::Wasm128Token as U64x2Backend>::max(a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn simd_eq(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i64x2_eq(a[0], b[0]), i64x2_eq(a[1], b[1])]
    }
    #[inline(always)]
    fn simd_ne(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [i64x2_ne(a[0], b[0]), i64x2_ne(a[1], b[1])]
    }
    #[inline(always)]
    fn simd_lt(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [
            <archmage::Wasm128Token as U64x2Backend>::simd_lt(a[0], b[0]),
            <archmage::Wasm128Token as U64x2Backend>::simd_lt(a[1], b[1]),
        ]
    }
    #[inline(always)]
    fn simd_le(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [
            <archmage::Wasm128Token as U64x2Backend>::simd_le(a[0], b[0]),
            <archmage::Wasm128Token as U64x2Backend>::simd_le(a[1], b[1]),
        ]
    }
    #[inline(always)]
    fn simd_gt(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [
            <archmage::Wasm128Token as U64x2Backend>::simd_gt(a[0], b[0]),
            <archmage::Wasm128Token as U64x2Backend>::simd_gt(a[1], b[1]),
        ]
    }
    #[inline(always)]
    fn simd_ge(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [
            <archmage::Wasm128Token as U64x2Backend>::simd_ge(a[0], b[0]),
            <archmage::Wasm128Token as U64x2Backend>::simd_ge(a[1], b[1]),
        ]
    }

    #[inline(always)]
    fn blend(mask: [v128; 2], if_true: [v128; 2], if_false: [v128; 2]) -> [v128; 2] {
        [
            v128_bitselect(if_true[0], if_false[0], mask[0]),
            v128_bitselect(if_true[1], if_false[1], mask[1]),
        ]
    }

    #[inline(always)]
    fn reduce_add(a: [v128; 2]) -> u64 {
        let arr = <Self as U64x4Backend>::to_array(a);
        arr.iter().copied().fold(0u64, u64::wrapping_add)
    }

    #[inline(always)]
    fn not(a: [v128; 2]) -> [v128; 2] {
        [v128_not(a[0]), v128_not(a[1])]
    }
    #[inline(always)]
    fn bitand(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [v128_and(a[0], b[0]), v128_and(a[1], b[1])]
    }
    #[inline(always)]
    fn bitor(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [v128_or(a[0], b[0]), v128_or(a[1], b[1])]
    }
    #[inline(always)]
    fn bitxor(a: [v128; 2], b: [v128; 2]) -> [v128; 2] {
        [v128_xor(a[0], b[0]), v128_xor(a[1], b[1])]
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: [v128; 2]) -> [v128; 2] {
        [i64x2_shl(a[0], N as u32), i64x2_shl(a[1], N as u32)]
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: [v128; 2]) -> [v128; 2] {
        [u64x2_shr(a[0], N as u32), u64x2_shr(a[1], N as u32)]
    }

    #[inline(always)]
    fn all_true(a: [v128; 2]) -> bool {
        i64x2_all_true(a[0]) && i64x2_all_true(a[1])
    }

    #[inline(always)]
    fn any_true(a: [v128; 2]) -> bool {
        v128_any_true(a[0]) || v128_any_true(a[1])
    }

    #[inline(always)]
    fn bitmask(a: [v128; 2]) -> u32 {
        let mut result = 0u32;
        for i in 0..2 {
            result |= (i64x2_bitmask(a[i]) as u32) << (i * 2);
        }
        result
    }
}

#[cfg(target_arch = "wasm32")]
impl F32x4Convert for archmage::Wasm128Token {
    #[inline(always)]
    fn bitcast_f32_to_i32(a: v128) -> v128 {
        a
    }

    #[inline(always)]
    fn bitcast_i32_to_f32(a: v128) -> v128 {
        a
    }

    #[inline(always)]
    fn convert_f32_to_i32(a: v128) -> v128 {
        i32x4_trunc_sat_f32x4(a)
    }

    #[inline(always)]
    fn convert_f32_to_i32_round(a: v128) -> v128 {
        i32x4_trunc_sat_f32x4(f32x4_nearest(a))
    }

    #[inline(always)]
    fn convert_i32_to_f32(a: v128) -> v128 {
        f32x4_convert_i32x4(a)
    }
}

#[cfg(target_arch = "wasm32")]
impl F32x8Convert for archmage::Wasm128Token {
    #[inline(always)]
    fn bitcast_f32_to_i32(a: [v128; 2]) -> [v128; 2] {
        a
    }

    #[inline(always)]
    fn bitcast_i32_to_f32(a: [v128; 2]) -> [v128; 2] {
        a
    }

    #[inline(always)]
    fn convert_f32_to_i32(a: [v128; 2]) -> [v128; 2] {
        [i32x4_trunc_sat_f32x4(a[0]), i32x4_trunc_sat_f32x4(a[1])]
    }

    #[inline(always)]
    fn convert_f32_to_i32_round(a: [v128; 2]) -> [v128; 2] {
        [
            i32x4_trunc_sat_f32x4(f32x4_nearest(a[0])),
            i32x4_trunc_sat_f32x4(f32x4_nearest(a[1])),
        ]
    }

    #[inline(always)]
    fn convert_i32_to_f32(a: [v128; 2]) -> [v128; 2] {
        [f32x4_convert_i32x4(a[0]), f32x4_convert_i32x4(a[1])]
    }
}

#[cfg(target_arch = "wasm32")]
impl U32x4Bitcast for archmage::Wasm128Token {
    #[inline(always)]
    fn bitcast_u32_to_i32(a: v128) -> v128 {
        a
    }

    #[inline(always)]
    fn bitcast_i32_to_u32(a: v128) -> v128 {
        a
    }
}

#[cfg(target_arch = "wasm32")]
impl U32x8Bitcast for archmage::Wasm128Token {
    #[inline(always)]
    fn bitcast_u32_to_i32(a: [v128; 2]) -> [v128; 2] {
        a
    }

    #[inline(always)]
    fn bitcast_i32_to_u32(a: [v128; 2]) -> [v128; 2] {
        a
    }
}

#[cfg(target_arch = "wasm32")]
impl I64x2Bitcast for archmage::Wasm128Token {
    #[inline(always)]
    fn bitcast_i64_to_f64(a: v128) -> v128 {
        a
    }

    #[inline(always)]
    fn bitcast_f64_to_i64(a: v128) -> v128 {
        a
    }
}

#[cfg(target_arch = "wasm32")]
impl I64x4Bitcast for archmage::Wasm128Token {
    #[inline(always)]
    fn bitcast_i64_to_f64(a: [v128; 2]) -> [v128; 2] {
        a
    }

    #[inline(always)]
    fn bitcast_f64_to_i64(a: [v128; 2]) -> [v128; 2] {
        a
    }
}

#[cfg(target_arch = "wasm32")]
impl I8x16Bitcast for archmage::Wasm128Token {
    #[inline(always)]
    fn bitcast_i8_to_u8(a: v128) -> v128 {
        a
    }
    #[inline(always)]
    fn bitcast_u8_to_i8(a: v128) -> v128 {
        a
    }
}

#[cfg(target_arch = "wasm32")]
impl I8x32Bitcast for archmage::Wasm128Token {
    #[inline(always)]
    fn bitcast_i8_to_u8(a: [v128; 2]) -> [v128; 2] {
        a
    }
    #[inline(always)]
    fn bitcast_u8_to_i8(a: [v128; 2]) -> [v128; 2] {
        a
    }
}

#[cfg(target_arch = "wasm32")]
impl I16x8Bitcast for archmage::Wasm128Token {
    #[inline(always)]
    fn bitcast_i16_to_u16(a: v128) -> v128 {
        a
    }
    #[inline(always)]
    fn bitcast_u16_to_i16(a: v128) -> v128 {
        a
    }
}

#[cfg(target_arch = "wasm32")]
impl I16x16Bitcast for archmage::Wasm128Token {
    #[inline(always)]
    fn bitcast_i16_to_u16(a: [v128; 2]) -> [v128; 2] {
        a
    }
    #[inline(always)]
    fn bitcast_u16_to_i16(a: [v128; 2]) -> [v128; 2] {
        a
    }
}

#[cfg(target_arch = "wasm32")]
impl U64x2Bitcast for archmage::Wasm128Token {
    #[inline(always)]
    fn bitcast_u64_to_i64(a: v128) -> v128 {
        a
    }
    #[inline(always)]
    fn bitcast_i64_to_u64(a: v128) -> v128 {
        a
    }
}

#[cfg(target_arch = "wasm32")]
impl U64x4Bitcast for archmage::Wasm128Token {
    #[inline(always)]
    fn bitcast_u64_to_i64(a: [v128; 2]) -> [v128; 2] {
        a
    }
    #[inline(always)]
    fn bitcast_i64_to_u64(a: [v128; 2]) -> [v128; 2] {
        a
    }
}
#[cfg(target_arch = "wasm32")]
impl F32x16Backend for archmage::Wasm128Token {
    type Repr = [v128; 4];

    #[inline(always)]
    fn splat(v: f32) -> [v128; 4] {
        let q = <archmage::Wasm128Token as F32x4Backend>::splat(v);
        [q, q, q, q]
    }

    #[inline(always)]
    fn zero() -> [v128; 4] {
        let q = <archmage::Wasm128Token as F32x4Backend>::zero();
        [q, q, q, q]
    }

    #[inline(always)]
    fn load(data: &[f32; 16]) -> [v128; 4] {
        [
            <archmage::Wasm128Token as F32x4Backend>::load(data[0..4].try_into().unwrap()),
            <archmage::Wasm128Token as F32x4Backend>::load(data[4..8].try_into().unwrap()),
            <archmage::Wasm128Token as F32x4Backend>::load(data[8..12].try_into().unwrap()),
            <archmage::Wasm128Token as F32x4Backend>::load(data[12..16].try_into().unwrap()),
        ]
    }

    #[inline(always)]
    fn from_array(arr: [f32; 16]) -> [v128; 4] {
        let mut q0 = [0.0f32; 4];
        let mut q1 = [0.0f32; 4];
        let mut q2 = [0.0f32; 4];
        let mut q3 = [0.0f32; 4];
        q0.copy_from_slice(&arr[0..4]);
        q1.copy_from_slice(&arr[4..8]);
        q2.copy_from_slice(&arr[8..12]);
        q3.copy_from_slice(&arr[12..16]);
        [
            <archmage::Wasm128Token as F32x4Backend>::from_array(q0),
            <archmage::Wasm128Token as F32x4Backend>::from_array(q1),
            <archmage::Wasm128Token as F32x4Backend>::from_array(q2),
            <archmage::Wasm128Token as F32x4Backend>::from_array(q3),
        ]
    }

    #[inline(always)]
    fn store(repr: [v128; 4], out: &mut [f32; 16]) {
        let (o01, o23) = out.split_at_mut(8);
        let (o0, o1) = o01.split_at_mut(4);
        let (o2, o3) = o23.split_at_mut(4);
        <archmage::Wasm128Token as F32x4Backend>::store(repr[0], o0.try_into().unwrap());
        <archmage::Wasm128Token as F32x4Backend>::store(repr[1], o1.try_into().unwrap());
        <archmage::Wasm128Token as F32x4Backend>::store(repr[2], o2.try_into().unwrap());
        <archmage::Wasm128Token as F32x4Backend>::store(repr[3], o3.try_into().unwrap());
    }

    #[inline(always)]
    fn to_array(repr: [v128; 4]) -> [f32; 16] {
        let a0 = <archmage::Wasm128Token as F32x4Backend>::to_array(repr[0]);
        let a1 = <archmage::Wasm128Token as F32x4Backend>::to_array(repr[1]);
        let a2 = <archmage::Wasm128Token as F32x4Backend>::to_array(repr[2]);
        let a3 = <archmage::Wasm128Token as F32x4Backend>::to_array(repr[3]);
        let mut out = [0.0f32; 16];
        out[0..4].copy_from_slice(&a0);
        out[4..8].copy_from_slice(&a1);
        out[8..12].copy_from_slice(&a2);
        out[12..16].copy_from_slice(&a3);
        out
    }

    #[inline(always)]
    fn add(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as F32x4Backend>::add(a[i], b[i]))
    }

    #[inline(always)]
    fn sub(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as F32x4Backend>::sub(a[i], b[i]))
    }

    #[inline(always)]
    fn mul(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as F32x4Backend>::mul(a[i], b[i]))
    }

    #[inline(always)]
    fn div(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as F32x4Backend>::div(a[i], b[i]))
    }

    #[inline(always)]
    fn neg(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as F32x4Backend>::neg(a[i]))
    }

    #[inline(always)]
    fn min(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as F32x4Backend>::min(a[i], b[i]))
    }

    #[inline(always)]
    fn max(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as F32x4Backend>::max(a[i], b[i]))
    }

    #[inline(always)]
    fn sqrt(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as F32x4Backend>::sqrt(a[i]))
    }

    #[inline(always)]
    fn abs(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as F32x4Backend>::abs(a[i]))
    }

    #[inline(always)]
    fn floor(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as F32x4Backend>::floor(a[i]))
    }

    #[inline(always)]
    fn ceil(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as F32x4Backend>::ceil(a[i]))
    }

    #[inline(always)]
    fn round(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as F32x4Backend>::round(a[i]))
    }

    #[inline(always)]
    fn mul_add(a: [v128; 4], b: [v128; 4], c: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| {
            <archmage::Wasm128Token as F32x4Backend>::mul_add(a[i], b[i], c[i])
        })
    }

    #[inline(always)]
    fn mul_sub(a: [v128; 4], b: [v128; 4], c: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| {
            <archmage::Wasm128Token as F32x4Backend>::mul_sub(a[i], b[i], c[i])
        })
    }

    #[inline(always)]
    fn reduce_add(a: [v128; 4]) -> f32 {
        <archmage::Wasm128Token as F32x4Backend>::reduce_add(a[0])
            + <archmage::Wasm128Token as F32x4Backend>::reduce_add(a[1])
            + <archmage::Wasm128Token as F32x4Backend>::reduce_add(a[2])
            + <archmage::Wasm128Token as F32x4Backend>::reduce_add(a[3])
    }

    #[inline(always)]
    fn reduce_min(a: [v128; 4]) -> f32 {
        let m01 = {
            let l = <archmage::Wasm128Token as F32x4Backend>::reduce_min(a[0]);
            let r = <archmage::Wasm128Token as F32x4Backend>::reduce_min(a[1]);
            if l < r { l } else { r }
        };
        let m23 = {
            let l = <archmage::Wasm128Token as F32x4Backend>::reduce_min(a[2]);
            let r = <archmage::Wasm128Token as F32x4Backend>::reduce_min(a[3]);
            if l < r { l } else { r }
        };
        if m01 < m23 { m01 } else { m23 }
    }

    #[inline(always)]
    fn reduce_max(a: [v128; 4]) -> f32 {
        let m01 = {
            let l = <archmage::Wasm128Token as F32x4Backend>::reduce_max(a[0]);
            let r = <archmage::Wasm128Token as F32x4Backend>::reduce_max(a[1]);
            if l > r { l } else { r }
        };
        let m23 = {
            let l = <archmage::Wasm128Token as F32x4Backend>::reduce_max(a[2]);
            let r = <archmage::Wasm128Token as F32x4Backend>::reduce_max(a[3]);
            if l > r { l } else { r }
        };
        if m01 > m23 { m01 } else { m23 }
    }

    #[inline(always)]
    fn simd_eq(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as F32x4Backend>::simd_eq(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ne(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as F32x4Backend>::simd_ne(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_lt(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as F32x4Backend>::simd_lt(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_le(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as F32x4Backend>::simd_le(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_gt(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as F32x4Backend>::simd_gt(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ge(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as F32x4Backend>::simd_ge(a[i], b[i]))
    }

    #[inline(always)]
    fn blend(mask: [v128; 4], if_true: [v128; 4], if_false: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| {
            <archmage::Wasm128Token as F32x4Backend>::blend(mask[i], if_true[i], if_false[i])
        })
    }

    #[inline(always)]
    fn not(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as F32x4Backend>::not(a[i]))
    }

    #[inline(always)]
    fn bitand(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as F32x4Backend>::bitand(a[i], b[i]))
    }

    #[inline(always)]
    fn bitor(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as F32x4Backend>::bitor(a[i], b[i]))
    }

    #[inline(always)]
    fn bitxor(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as F32x4Backend>::bitxor(a[i], b[i]))
    }
}

#[cfg(target_arch = "wasm32")]
impl F64x8Backend for archmage::Wasm128Token {
    type Repr = [v128; 4];

    #[inline(always)]
    fn splat(v: f64) -> [v128; 4] {
        let q = <archmage::Wasm128Token as F64x2Backend>::splat(v);
        [q, q, q, q]
    }

    #[inline(always)]
    fn zero() -> [v128; 4] {
        let q = <archmage::Wasm128Token as F64x2Backend>::zero();
        [q, q, q, q]
    }

    #[inline(always)]
    fn load(data: &[f64; 8]) -> [v128; 4] {
        [
            <archmage::Wasm128Token as F64x2Backend>::load(data[0..2].try_into().unwrap()),
            <archmage::Wasm128Token as F64x2Backend>::load(data[2..4].try_into().unwrap()),
            <archmage::Wasm128Token as F64x2Backend>::load(data[4..6].try_into().unwrap()),
            <archmage::Wasm128Token as F64x2Backend>::load(data[6..8].try_into().unwrap()),
        ]
    }

    #[inline(always)]
    fn from_array(arr: [f64; 8]) -> [v128; 4] {
        let mut q0 = [0.0f64; 2];
        let mut q1 = [0.0f64; 2];
        let mut q2 = [0.0f64; 2];
        let mut q3 = [0.0f64; 2];
        q0.copy_from_slice(&arr[0..2]);
        q1.copy_from_slice(&arr[2..4]);
        q2.copy_from_slice(&arr[4..6]);
        q3.copy_from_slice(&arr[6..8]);
        [
            <archmage::Wasm128Token as F64x2Backend>::from_array(q0),
            <archmage::Wasm128Token as F64x2Backend>::from_array(q1),
            <archmage::Wasm128Token as F64x2Backend>::from_array(q2),
            <archmage::Wasm128Token as F64x2Backend>::from_array(q3),
        ]
    }

    #[inline(always)]
    fn store(repr: [v128; 4], out: &mut [f64; 8]) {
        let (o01, o23) = out.split_at_mut(4);
        let (o0, o1) = o01.split_at_mut(2);
        let (o2, o3) = o23.split_at_mut(2);
        <archmage::Wasm128Token as F64x2Backend>::store(repr[0], o0.try_into().unwrap());
        <archmage::Wasm128Token as F64x2Backend>::store(repr[1], o1.try_into().unwrap());
        <archmage::Wasm128Token as F64x2Backend>::store(repr[2], o2.try_into().unwrap());
        <archmage::Wasm128Token as F64x2Backend>::store(repr[3], o3.try_into().unwrap());
    }

    #[inline(always)]
    fn to_array(repr: [v128; 4]) -> [f64; 8] {
        let a0 = <archmage::Wasm128Token as F64x2Backend>::to_array(repr[0]);
        let a1 = <archmage::Wasm128Token as F64x2Backend>::to_array(repr[1]);
        let a2 = <archmage::Wasm128Token as F64x2Backend>::to_array(repr[2]);
        let a3 = <archmage::Wasm128Token as F64x2Backend>::to_array(repr[3]);
        let mut out = [0.0f64; 8];
        out[0..2].copy_from_slice(&a0);
        out[2..4].copy_from_slice(&a1);
        out[4..6].copy_from_slice(&a2);
        out[6..8].copy_from_slice(&a3);
        out
    }

    #[inline(always)]
    fn add(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as F64x2Backend>::add(a[i], b[i]))
    }

    #[inline(always)]
    fn sub(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as F64x2Backend>::sub(a[i], b[i]))
    }

    #[inline(always)]
    fn mul(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as F64x2Backend>::mul(a[i], b[i]))
    }

    #[inline(always)]
    fn div(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as F64x2Backend>::div(a[i], b[i]))
    }

    #[inline(always)]
    fn neg(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as F64x2Backend>::neg(a[i]))
    }

    #[inline(always)]
    fn min(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as F64x2Backend>::min(a[i], b[i]))
    }

    #[inline(always)]
    fn max(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as F64x2Backend>::max(a[i], b[i]))
    }

    #[inline(always)]
    fn sqrt(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as F64x2Backend>::sqrt(a[i]))
    }

    #[inline(always)]
    fn abs(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as F64x2Backend>::abs(a[i]))
    }

    #[inline(always)]
    fn floor(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as F64x2Backend>::floor(a[i]))
    }

    #[inline(always)]
    fn ceil(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as F64x2Backend>::ceil(a[i]))
    }

    #[inline(always)]
    fn round(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as F64x2Backend>::round(a[i]))
    }

    #[inline(always)]
    fn mul_add(a: [v128; 4], b: [v128; 4], c: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| {
            <archmage::Wasm128Token as F64x2Backend>::mul_add(a[i], b[i], c[i])
        })
    }

    #[inline(always)]
    fn mul_sub(a: [v128; 4], b: [v128; 4], c: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| {
            <archmage::Wasm128Token as F64x2Backend>::mul_sub(a[i], b[i], c[i])
        })
    }

    #[inline(always)]
    fn reduce_add(a: [v128; 4]) -> f64 {
        <archmage::Wasm128Token as F64x2Backend>::reduce_add(a[0])
            + <archmage::Wasm128Token as F64x2Backend>::reduce_add(a[1])
            + <archmage::Wasm128Token as F64x2Backend>::reduce_add(a[2])
            + <archmage::Wasm128Token as F64x2Backend>::reduce_add(a[3])
    }

    #[inline(always)]
    fn reduce_min(a: [v128; 4]) -> f64 {
        let m01 = {
            let l = <archmage::Wasm128Token as F64x2Backend>::reduce_min(a[0]);
            let r = <archmage::Wasm128Token as F64x2Backend>::reduce_min(a[1]);
            if l < r { l } else { r }
        };
        let m23 = {
            let l = <archmage::Wasm128Token as F64x2Backend>::reduce_min(a[2]);
            let r = <archmage::Wasm128Token as F64x2Backend>::reduce_min(a[3]);
            if l < r { l } else { r }
        };
        if m01 < m23 { m01 } else { m23 }
    }

    #[inline(always)]
    fn reduce_max(a: [v128; 4]) -> f64 {
        let m01 = {
            let l = <archmage::Wasm128Token as F64x2Backend>::reduce_max(a[0]);
            let r = <archmage::Wasm128Token as F64x2Backend>::reduce_max(a[1]);
            if l > r { l } else { r }
        };
        let m23 = {
            let l = <archmage::Wasm128Token as F64x2Backend>::reduce_max(a[2]);
            let r = <archmage::Wasm128Token as F64x2Backend>::reduce_max(a[3]);
            if l > r { l } else { r }
        };
        if m01 > m23 { m01 } else { m23 }
    }

    #[inline(always)]
    fn simd_eq(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as F64x2Backend>::simd_eq(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ne(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as F64x2Backend>::simd_ne(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_lt(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as F64x2Backend>::simd_lt(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_le(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as F64x2Backend>::simd_le(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_gt(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as F64x2Backend>::simd_gt(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ge(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as F64x2Backend>::simd_ge(a[i], b[i]))
    }

    #[inline(always)]
    fn blend(mask: [v128; 4], if_true: [v128; 4], if_false: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| {
            <archmage::Wasm128Token as F64x2Backend>::blend(mask[i], if_true[i], if_false[i])
        })
    }

    #[inline(always)]
    fn not(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as F64x2Backend>::not(a[i]))
    }

    #[inline(always)]
    fn bitand(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as F64x2Backend>::bitand(a[i], b[i]))
    }

    #[inline(always)]
    fn bitor(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as F64x2Backend>::bitor(a[i], b[i]))
    }

    #[inline(always)]
    fn bitxor(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as F64x2Backend>::bitxor(a[i], b[i]))
    }
}

#[cfg(target_arch = "wasm32")]
impl I8x64Backend for archmage::Wasm128Token {
    type Repr = [v128; 4];

    #[inline(always)]
    fn splat(v: i8) -> [v128; 4] {
        let q = <archmage::Wasm128Token as I8x16Backend>::splat(v);
        [q, q, q, q]
    }

    #[inline(always)]
    fn zero() -> [v128; 4] {
        let q = <archmage::Wasm128Token as I8x16Backend>::zero();
        [q, q, q, q]
    }

    #[inline(always)]
    fn load(data: &[i8; 64]) -> [v128; 4] {
        [
            <archmage::Wasm128Token as I8x16Backend>::load(data[0..16].try_into().unwrap()),
            <archmage::Wasm128Token as I8x16Backend>::load(data[16..32].try_into().unwrap()),
            <archmage::Wasm128Token as I8x16Backend>::load(data[32..48].try_into().unwrap()),
            <archmage::Wasm128Token as I8x16Backend>::load(data[48..64].try_into().unwrap()),
        ]
    }

    #[inline(always)]
    fn from_array(arr: [i8; 64]) -> [v128; 4] {
        let mut q0 = [0; 16];
        let mut q1 = [0; 16];
        let mut q2 = [0; 16];
        let mut q3 = [0; 16];
        q0.copy_from_slice(&arr[0..16]);
        q1.copy_from_slice(&arr[16..32]);
        q2.copy_from_slice(&arr[32..48]);
        q3.copy_from_slice(&arr[48..64]);
        [
            <archmage::Wasm128Token as I8x16Backend>::from_array(q0),
            <archmage::Wasm128Token as I8x16Backend>::from_array(q1),
            <archmage::Wasm128Token as I8x16Backend>::from_array(q2),
            <archmage::Wasm128Token as I8x16Backend>::from_array(q3),
        ]
    }

    #[inline(always)]
    fn store(repr: [v128; 4], out: &mut [i8; 64]) {
        let (o01, o23) = out.split_at_mut(32);
        let (o0, o1) = o01.split_at_mut(16);
        let (o2, o3) = o23.split_at_mut(16);
        <archmage::Wasm128Token as I8x16Backend>::store(repr[0], o0.try_into().unwrap());
        <archmage::Wasm128Token as I8x16Backend>::store(repr[1], o1.try_into().unwrap());
        <archmage::Wasm128Token as I8x16Backend>::store(repr[2], o2.try_into().unwrap());
        <archmage::Wasm128Token as I8x16Backend>::store(repr[3], o3.try_into().unwrap());
    }

    #[inline(always)]
    fn to_array(repr: [v128; 4]) -> [i8; 64] {
        let a0 = <archmage::Wasm128Token as I8x16Backend>::to_array(repr[0]);
        let a1 = <archmage::Wasm128Token as I8x16Backend>::to_array(repr[1]);
        let a2 = <archmage::Wasm128Token as I8x16Backend>::to_array(repr[2]);
        let a3 = <archmage::Wasm128Token as I8x16Backend>::to_array(repr[3]);
        let mut out = [0; 64];
        out[0..16].copy_from_slice(&a0);
        out[16..32].copy_from_slice(&a1);
        out[32..48].copy_from_slice(&a2);
        out[48..64].copy_from_slice(&a3);
        out
    }

    #[inline(always)]
    fn add(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I8x16Backend>::add(a[i], b[i]))
    }

    #[inline(always)]
    fn sub(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I8x16Backend>::sub(a[i], b[i]))
    }

    #[inline(always)]
    fn neg(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I8x16Backend>::neg(a[i]))
    }

    #[inline(always)]
    fn min(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I8x16Backend>::min(a[i], b[i]))
    }

    #[inline(always)]
    fn max(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I8x16Backend>::max(a[i], b[i]))
    }

    #[inline(always)]
    fn abs(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I8x16Backend>::abs(a[i]))
    }

    #[inline(always)]
    fn reduce_add(a: [v128; 4]) -> i8 {
        <archmage::Wasm128Token as I8x16Backend>::reduce_add(a[0])
            .wrapping_add(<archmage::Wasm128Token as I8x16Backend>::reduce_add(a[1]))
            .wrapping_add(<archmage::Wasm128Token as I8x16Backend>::reduce_add(a[2]))
            .wrapping_add(<archmage::Wasm128Token as I8x16Backend>::reduce_add(a[3]))
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I8x16Backend>::shl_const::<N>(a[i]))
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| {
            <archmage::Wasm128Token as I8x16Backend>::shr_arithmetic_const::<N>(a[i])
        })
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| {
            <archmage::Wasm128Token as I8x16Backend>::shr_logical_const::<N>(a[i])
        })
    }

    #[inline(always)]
    fn all_true(a: [v128; 4]) -> bool {
        <archmage::Wasm128Token as I8x16Backend>::all_true(a[0])
            && <archmage::Wasm128Token as I8x16Backend>::all_true(a[1])
            && <archmage::Wasm128Token as I8x16Backend>::all_true(a[2])
            && <archmage::Wasm128Token as I8x16Backend>::all_true(a[3])
    }

    #[inline(always)]
    fn any_true(a: [v128; 4]) -> bool {
        <archmage::Wasm128Token as I8x16Backend>::any_true(a[0])
            || <archmage::Wasm128Token as I8x16Backend>::any_true(a[1])
            || <archmage::Wasm128Token as I8x16Backend>::any_true(a[2])
            || <archmage::Wasm128Token as I8x16Backend>::any_true(a[3])
    }

    #[inline(always)]
    fn bitmask(a: [v128; 4]) -> u64 {
        let q0 = <archmage::Wasm128Token as I8x16Backend>::bitmask(a[0]) as u64;
        let q1 = <archmage::Wasm128Token as I8x16Backend>::bitmask(a[1]) as u64;
        let q2 = <archmage::Wasm128Token as I8x16Backend>::bitmask(a[2]) as u64;
        let q3 = <archmage::Wasm128Token as I8x16Backend>::bitmask(a[3]) as u64;
        q0 | (q1 << 16) | (q2 << 32) | (q3 << 48)
    }

    #[inline(always)]
    fn simd_eq(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I8x16Backend>::simd_eq(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ne(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I8x16Backend>::simd_ne(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_lt(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I8x16Backend>::simd_lt(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_le(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I8x16Backend>::simd_le(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_gt(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I8x16Backend>::simd_gt(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ge(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I8x16Backend>::simd_ge(a[i], b[i]))
    }

    #[inline(always)]
    fn blend(mask: [v128; 4], if_true: [v128; 4], if_false: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| {
            <archmage::Wasm128Token as I8x16Backend>::blend(mask[i], if_true[i], if_false[i])
        })
    }

    #[inline(always)]
    fn not(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I8x16Backend>::not(a[i]))
    }

    #[inline(always)]
    fn bitand(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I8x16Backend>::bitand(a[i], b[i]))
    }

    #[inline(always)]
    fn bitor(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I8x16Backend>::bitor(a[i], b[i]))
    }

    #[inline(always)]
    fn bitxor(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I8x16Backend>::bitxor(a[i], b[i]))
    }
}

#[cfg(target_arch = "wasm32")]
impl U8x64Backend for archmage::Wasm128Token {
    type Repr = [v128; 4];

    #[inline(always)]
    fn splat(v: u8) -> [v128; 4] {
        let q = <archmage::Wasm128Token as U8x16Backend>::splat(v);
        [q, q, q, q]
    }

    #[inline(always)]
    fn zero() -> [v128; 4] {
        let q = <archmage::Wasm128Token as U8x16Backend>::zero();
        [q, q, q, q]
    }

    #[inline(always)]
    fn load(data: &[u8; 64]) -> [v128; 4] {
        [
            <archmage::Wasm128Token as U8x16Backend>::load(data[0..16].try_into().unwrap()),
            <archmage::Wasm128Token as U8x16Backend>::load(data[16..32].try_into().unwrap()),
            <archmage::Wasm128Token as U8x16Backend>::load(data[32..48].try_into().unwrap()),
            <archmage::Wasm128Token as U8x16Backend>::load(data[48..64].try_into().unwrap()),
        ]
    }

    #[inline(always)]
    fn from_array(arr: [u8; 64]) -> [v128; 4] {
        let mut q0 = [0; 16];
        let mut q1 = [0; 16];
        let mut q2 = [0; 16];
        let mut q3 = [0; 16];
        q0.copy_from_slice(&arr[0..16]);
        q1.copy_from_slice(&arr[16..32]);
        q2.copy_from_slice(&arr[32..48]);
        q3.copy_from_slice(&arr[48..64]);
        [
            <archmage::Wasm128Token as U8x16Backend>::from_array(q0),
            <archmage::Wasm128Token as U8x16Backend>::from_array(q1),
            <archmage::Wasm128Token as U8x16Backend>::from_array(q2),
            <archmage::Wasm128Token as U8x16Backend>::from_array(q3),
        ]
    }

    #[inline(always)]
    fn store(repr: [v128; 4], out: &mut [u8; 64]) {
        let (o01, o23) = out.split_at_mut(32);
        let (o0, o1) = o01.split_at_mut(16);
        let (o2, o3) = o23.split_at_mut(16);
        <archmage::Wasm128Token as U8x16Backend>::store(repr[0], o0.try_into().unwrap());
        <archmage::Wasm128Token as U8x16Backend>::store(repr[1], o1.try_into().unwrap());
        <archmage::Wasm128Token as U8x16Backend>::store(repr[2], o2.try_into().unwrap());
        <archmage::Wasm128Token as U8x16Backend>::store(repr[3], o3.try_into().unwrap());
    }

    #[inline(always)]
    fn to_array(repr: [v128; 4]) -> [u8; 64] {
        let a0 = <archmage::Wasm128Token as U8x16Backend>::to_array(repr[0]);
        let a1 = <archmage::Wasm128Token as U8x16Backend>::to_array(repr[1]);
        let a2 = <archmage::Wasm128Token as U8x16Backend>::to_array(repr[2]);
        let a3 = <archmage::Wasm128Token as U8x16Backend>::to_array(repr[3]);
        let mut out = [0; 64];
        out[0..16].copy_from_slice(&a0);
        out[16..32].copy_from_slice(&a1);
        out[32..48].copy_from_slice(&a2);
        out[48..64].copy_from_slice(&a3);
        out
    }

    #[inline(always)]
    fn add(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U8x16Backend>::add(a[i], b[i]))
    }

    #[inline(always)]
    fn sub(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U8x16Backend>::sub(a[i], b[i]))
    }

    #[inline(always)]
    fn neg(a: [v128; 4]) -> [v128; 4] {
        let z = <archmage::Wasm128Token as U8x16Backend>::zero();
        core::array::from_fn(|i| <archmage::Wasm128Token as U8x16Backend>::sub(z, a[i]))
    }

    #[inline(always)]
    fn min(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U8x16Backend>::min(a[i], b[i]))
    }

    #[inline(always)]
    fn max(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U8x16Backend>::max(a[i], b[i]))
    }

    #[inline(always)]
    fn reduce_add(a: [v128; 4]) -> u8 {
        <archmage::Wasm128Token as U8x16Backend>::reduce_add(a[0])
            .wrapping_add(<archmage::Wasm128Token as U8x16Backend>::reduce_add(a[1]))
            .wrapping_add(<archmage::Wasm128Token as U8x16Backend>::reduce_add(a[2]))
            .wrapping_add(<archmage::Wasm128Token as U8x16Backend>::reduce_add(a[3]))
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U8x16Backend>::shl_const::<N>(a[i]))
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| {
            <archmage::Wasm128Token as U8x16Backend>::shr_logical_const::<N>(a[i])
        })
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| {
            <archmage::Wasm128Token as U8x16Backend>::shr_logical_const::<N>(a[i])
        })
    }

    #[inline(always)]
    fn all_true(a: [v128; 4]) -> bool {
        <archmage::Wasm128Token as U8x16Backend>::all_true(a[0])
            && <archmage::Wasm128Token as U8x16Backend>::all_true(a[1])
            && <archmage::Wasm128Token as U8x16Backend>::all_true(a[2])
            && <archmage::Wasm128Token as U8x16Backend>::all_true(a[3])
    }

    #[inline(always)]
    fn any_true(a: [v128; 4]) -> bool {
        <archmage::Wasm128Token as U8x16Backend>::any_true(a[0])
            || <archmage::Wasm128Token as U8x16Backend>::any_true(a[1])
            || <archmage::Wasm128Token as U8x16Backend>::any_true(a[2])
            || <archmage::Wasm128Token as U8x16Backend>::any_true(a[3])
    }

    #[inline(always)]
    fn bitmask(a: [v128; 4]) -> u64 {
        let q0 = <archmage::Wasm128Token as U8x16Backend>::bitmask(a[0]) as u64;
        let q1 = <archmage::Wasm128Token as U8x16Backend>::bitmask(a[1]) as u64;
        let q2 = <archmage::Wasm128Token as U8x16Backend>::bitmask(a[2]) as u64;
        let q3 = <archmage::Wasm128Token as U8x16Backend>::bitmask(a[3]) as u64;
        q0 | (q1 << 16) | (q2 << 32) | (q3 << 48)
    }

    #[inline(always)]
    fn simd_eq(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U8x16Backend>::simd_eq(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ne(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U8x16Backend>::simd_ne(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_lt(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U8x16Backend>::simd_lt(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_le(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U8x16Backend>::simd_le(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_gt(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U8x16Backend>::simd_gt(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ge(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U8x16Backend>::simd_ge(a[i], b[i]))
    }

    #[inline(always)]
    fn blend(mask: [v128; 4], if_true: [v128; 4], if_false: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| {
            <archmage::Wasm128Token as U8x16Backend>::blend(mask[i], if_true[i], if_false[i])
        })
    }

    #[inline(always)]
    fn not(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U8x16Backend>::not(a[i]))
    }

    #[inline(always)]
    fn bitand(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U8x16Backend>::bitand(a[i], b[i]))
    }

    #[inline(always)]
    fn bitor(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U8x16Backend>::bitor(a[i], b[i]))
    }

    #[inline(always)]
    fn bitxor(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U8x16Backend>::bitxor(a[i], b[i]))
    }
}

#[cfg(target_arch = "wasm32")]
impl I16x32Backend for archmage::Wasm128Token {
    type Repr = [v128; 4];

    #[inline(always)]
    fn splat(v: i16) -> [v128; 4] {
        let q = <archmage::Wasm128Token as I16x8Backend>::splat(v);
        [q, q, q, q]
    }

    #[inline(always)]
    fn zero() -> [v128; 4] {
        let q = <archmage::Wasm128Token as I16x8Backend>::zero();
        [q, q, q, q]
    }

    #[inline(always)]
    fn load(data: &[i16; 32]) -> [v128; 4] {
        [
            <archmage::Wasm128Token as I16x8Backend>::load(data[0..8].try_into().unwrap()),
            <archmage::Wasm128Token as I16x8Backend>::load(data[8..16].try_into().unwrap()),
            <archmage::Wasm128Token as I16x8Backend>::load(data[16..24].try_into().unwrap()),
            <archmage::Wasm128Token as I16x8Backend>::load(data[24..32].try_into().unwrap()),
        ]
    }

    #[inline(always)]
    fn from_array(arr: [i16; 32]) -> [v128; 4] {
        let mut q0 = [0; 8];
        let mut q1 = [0; 8];
        let mut q2 = [0; 8];
        let mut q3 = [0; 8];
        q0.copy_from_slice(&arr[0..8]);
        q1.copy_from_slice(&arr[8..16]);
        q2.copy_from_slice(&arr[16..24]);
        q3.copy_from_slice(&arr[24..32]);
        [
            <archmage::Wasm128Token as I16x8Backend>::from_array(q0),
            <archmage::Wasm128Token as I16x8Backend>::from_array(q1),
            <archmage::Wasm128Token as I16x8Backend>::from_array(q2),
            <archmage::Wasm128Token as I16x8Backend>::from_array(q3),
        ]
    }

    #[inline(always)]
    fn store(repr: [v128; 4], out: &mut [i16; 32]) {
        let (o01, o23) = out.split_at_mut(16);
        let (o0, o1) = o01.split_at_mut(8);
        let (o2, o3) = o23.split_at_mut(8);
        <archmage::Wasm128Token as I16x8Backend>::store(repr[0], o0.try_into().unwrap());
        <archmage::Wasm128Token as I16x8Backend>::store(repr[1], o1.try_into().unwrap());
        <archmage::Wasm128Token as I16x8Backend>::store(repr[2], o2.try_into().unwrap());
        <archmage::Wasm128Token as I16x8Backend>::store(repr[3], o3.try_into().unwrap());
    }

    #[inline(always)]
    fn to_array(repr: [v128; 4]) -> [i16; 32] {
        let a0 = <archmage::Wasm128Token as I16x8Backend>::to_array(repr[0]);
        let a1 = <archmage::Wasm128Token as I16x8Backend>::to_array(repr[1]);
        let a2 = <archmage::Wasm128Token as I16x8Backend>::to_array(repr[2]);
        let a3 = <archmage::Wasm128Token as I16x8Backend>::to_array(repr[3]);
        let mut out = [0; 32];
        out[0..8].copy_from_slice(&a0);
        out[8..16].copy_from_slice(&a1);
        out[16..24].copy_from_slice(&a2);
        out[24..32].copy_from_slice(&a3);
        out
    }

    #[inline(always)]
    fn add(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I16x8Backend>::add(a[i], b[i]))
    }

    #[inline(always)]
    fn sub(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I16x8Backend>::sub(a[i], b[i]))
    }

    #[inline(always)]
    fn mul(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I16x8Backend>::mul(a[i], b[i]))
    }

    #[inline(always)]
    fn neg(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I16x8Backend>::neg(a[i]))
    }

    #[inline(always)]
    fn min(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I16x8Backend>::min(a[i], b[i]))
    }

    #[inline(always)]
    fn max(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I16x8Backend>::max(a[i], b[i]))
    }

    #[inline(always)]
    fn abs(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I16x8Backend>::abs(a[i]))
    }

    #[inline(always)]
    fn reduce_add(a: [v128; 4]) -> i16 {
        <archmage::Wasm128Token as I16x8Backend>::reduce_add(a[0])
            .wrapping_add(<archmage::Wasm128Token as I16x8Backend>::reduce_add(a[1]))
            .wrapping_add(<archmage::Wasm128Token as I16x8Backend>::reduce_add(a[2]))
            .wrapping_add(<archmage::Wasm128Token as I16x8Backend>::reduce_add(a[3]))
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I16x8Backend>::shl_const::<N>(a[i]))
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| {
            <archmage::Wasm128Token as I16x8Backend>::shr_arithmetic_const::<N>(a[i])
        })
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| {
            <archmage::Wasm128Token as I16x8Backend>::shr_logical_const::<N>(a[i])
        })
    }

    #[inline(always)]
    fn all_true(a: [v128; 4]) -> bool {
        <archmage::Wasm128Token as I16x8Backend>::all_true(a[0])
            && <archmage::Wasm128Token as I16x8Backend>::all_true(a[1])
            && <archmage::Wasm128Token as I16x8Backend>::all_true(a[2])
            && <archmage::Wasm128Token as I16x8Backend>::all_true(a[3])
    }

    #[inline(always)]
    fn any_true(a: [v128; 4]) -> bool {
        <archmage::Wasm128Token as I16x8Backend>::any_true(a[0])
            || <archmage::Wasm128Token as I16x8Backend>::any_true(a[1])
            || <archmage::Wasm128Token as I16x8Backend>::any_true(a[2])
            || <archmage::Wasm128Token as I16x8Backend>::any_true(a[3])
    }

    #[inline(always)]
    fn bitmask(a: [v128; 4]) -> u64 {
        let q0 = <archmage::Wasm128Token as I16x8Backend>::bitmask(a[0]) as u64;
        let q1 = <archmage::Wasm128Token as I16x8Backend>::bitmask(a[1]) as u64;
        let q2 = <archmage::Wasm128Token as I16x8Backend>::bitmask(a[2]) as u64;
        let q3 = <archmage::Wasm128Token as I16x8Backend>::bitmask(a[3]) as u64;
        q0 | (q1 << 8) | (q2 << 16) | (q3 << 24)
    }

    #[inline(always)]
    fn simd_eq(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I16x8Backend>::simd_eq(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ne(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I16x8Backend>::simd_ne(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_lt(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I16x8Backend>::simd_lt(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_le(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I16x8Backend>::simd_le(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_gt(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I16x8Backend>::simd_gt(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ge(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I16x8Backend>::simd_ge(a[i], b[i]))
    }

    #[inline(always)]
    fn blend(mask: [v128; 4], if_true: [v128; 4], if_false: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| {
            <archmage::Wasm128Token as I16x8Backend>::blend(mask[i], if_true[i], if_false[i])
        })
    }

    #[inline(always)]
    fn not(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I16x8Backend>::not(a[i]))
    }

    #[inline(always)]
    fn bitand(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I16x8Backend>::bitand(a[i], b[i]))
    }

    #[inline(always)]
    fn bitor(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I16x8Backend>::bitor(a[i], b[i]))
    }

    #[inline(always)]
    fn bitxor(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I16x8Backend>::bitxor(a[i], b[i]))
    }
}

#[cfg(target_arch = "wasm32")]
impl U16x32Backend for archmage::Wasm128Token {
    type Repr = [v128; 4];

    #[inline(always)]
    fn splat(v: u16) -> [v128; 4] {
        let q = <archmage::Wasm128Token as U16x8Backend>::splat(v);
        [q, q, q, q]
    }

    #[inline(always)]
    fn zero() -> [v128; 4] {
        let q = <archmage::Wasm128Token as U16x8Backend>::zero();
        [q, q, q, q]
    }

    #[inline(always)]
    fn load(data: &[u16; 32]) -> [v128; 4] {
        [
            <archmage::Wasm128Token as U16x8Backend>::load(data[0..8].try_into().unwrap()),
            <archmage::Wasm128Token as U16x8Backend>::load(data[8..16].try_into().unwrap()),
            <archmage::Wasm128Token as U16x8Backend>::load(data[16..24].try_into().unwrap()),
            <archmage::Wasm128Token as U16x8Backend>::load(data[24..32].try_into().unwrap()),
        ]
    }

    #[inline(always)]
    fn from_array(arr: [u16; 32]) -> [v128; 4] {
        let mut q0 = [0; 8];
        let mut q1 = [0; 8];
        let mut q2 = [0; 8];
        let mut q3 = [0; 8];
        q0.copy_from_slice(&arr[0..8]);
        q1.copy_from_slice(&arr[8..16]);
        q2.copy_from_slice(&arr[16..24]);
        q3.copy_from_slice(&arr[24..32]);
        [
            <archmage::Wasm128Token as U16x8Backend>::from_array(q0),
            <archmage::Wasm128Token as U16x8Backend>::from_array(q1),
            <archmage::Wasm128Token as U16x8Backend>::from_array(q2),
            <archmage::Wasm128Token as U16x8Backend>::from_array(q3),
        ]
    }

    #[inline(always)]
    fn store(repr: [v128; 4], out: &mut [u16; 32]) {
        let (o01, o23) = out.split_at_mut(16);
        let (o0, o1) = o01.split_at_mut(8);
        let (o2, o3) = o23.split_at_mut(8);
        <archmage::Wasm128Token as U16x8Backend>::store(repr[0], o0.try_into().unwrap());
        <archmage::Wasm128Token as U16x8Backend>::store(repr[1], o1.try_into().unwrap());
        <archmage::Wasm128Token as U16x8Backend>::store(repr[2], o2.try_into().unwrap());
        <archmage::Wasm128Token as U16x8Backend>::store(repr[3], o3.try_into().unwrap());
    }

    #[inline(always)]
    fn to_array(repr: [v128; 4]) -> [u16; 32] {
        let a0 = <archmage::Wasm128Token as U16x8Backend>::to_array(repr[0]);
        let a1 = <archmage::Wasm128Token as U16x8Backend>::to_array(repr[1]);
        let a2 = <archmage::Wasm128Token as U16x8Backend>::to_array(repr[2]);
        let a3 = <archmage::Wasm128Token as U16x8Backend>::to_array(repr[3]);
        let mut out = [0; 32];
        out[0..8].copy_from_slice(&a0);
        out[8..16].copy_from_slice(&a1);
        out[16..24].copy_from_slice(&a2);
        out[24..32].copy_from_slice(&a3);
        out
    }

    #[inline(always)]
    fn add(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U16x8Backend>::add(a[i], b[i]))
    }

    #[inline(always)]
    fn sub(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U16x8Backend>::sub(a[i], b[i]))
    }

    #[inline(always)]
    fn mul(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U16x8Backend>::mul(a[i], b[i]))
    }

    #[inline(always)]
    fn neg(a: [v128; 4]) -> [v128; 4] {
        let z = <archmage::Wasm128Token as U16x8Backend>::zero();
        core::array::from_fn(|i| <archmage::Wasm128Token as U16x8Backend>::sub(z, a[i]))
    }

    #[inline(always)]
    fn min(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U16x8Backend>::min(a[i], b[i]))
    }

    #[inline(always)]
    fn max(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U16x8Backend>::max(a[i], b[i]))
    }

    #[inline(always)]
    fn reduce_add(a: [v128; 4]) -> u16 {
        <archmage::Wasm128Token as U16x8Backend>::reduce_add(a[0])
            .wrapping_add(<archmage::Wasm128Token as U16x8Backend>::reduce_add(a[1]))
            .wrapping_add(<archmage::Wasm128Token as U16x8Backend>::reduce_add(a[2]))
            .wrapping_add(<archmage::Wasm128Token as U16x8Backend>::reduce_add(a[3]))
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U16x8Backend>::shl_const::<N>(a[i]))
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| {
            <archmage::Wasm128Token as U16x8Backend>::shr_logical_const::<N>(a[i])
        })
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| {
            <archmage::Wasm128Token as U16x8Backend>::shr_logical_const::<N>(a[i])
        })
    }

    #[inline(always)]
    fn all_true(a: [v128; 4]) -> bool {
        <archmage::Wasm128Token as U16x8Backend>::all_true(a[0])
            && <archmage::Wasm128Token as U16x8Backend>::all_true(a[1])
            && <archmage::Wasm128Token as U16x8Backend>::all_true(a[2])
            && <archmage::Wasm128Token as U16x8Backend>::all_true(a[3])
    }

    #[inline(always)]
    fn any_true(a: [v128; 4]) -> bool {
        <archmage::Wasm128Token as U16x8Backend>::any_true(a[0])
            || <archmage::Wasm128Token as U16x8Backend>::any_true(a[1])
            || <archmage::Wasm128Token as U16x8Backend>::any_true(a[2])
            || <archmage::Wasm128Token as U16x8Backend>::any_true(a[3])
    }

    #[inline(always)]
    fn bitmask(a: [v128; 4]) -> u64 {
        let q0 = <archmage::Wasm128Token as U16x8Backend>::bitmask(a[0]) as u64;
        let q1 = <archmage::Wasm128Token as U16x8Backend>::bitmask(a[1]) as u64;
        let q2 = <archmage::Wasm128Token as U16x8Backend>::bitmask(a[2]) as u64;
        let q3 = <archmage::Wasm128Token as U16x8Backend>::bitmask(a[3]) as u64;
        q0 | (q1 << 8) | (q2 << 16) | (q3 << 24)
    }

    #[inline(always)]
    fn simd_eq(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U16x8Backend>::simd_eq(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ne(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U16x8Backend>::simd_ne(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_lt(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U16x8Backend>::simd_lt(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_le(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U16x8Backend>::simd_le(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_gt(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U16x8Backend>::simd_gt(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ge(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U16x8Backend>::simd_ge(a[i], b[i]))
    }

    #[inline(always)]
    fn blend(mask: [v128; 4], if_true: [v128; 4], if_false: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| {
            <archmage::Wasm128Token as U16x8Backend>::blend(mask[i], if_true[i], if_false[i])
        })
    }

    #[inline(always)]
    fn not(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U16x8Backend>::not(a[i]))
    }

    #[inline(always)]
    fn bitand(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U16x8Backend>::bitand(a[i], b[i]))
    }

    #[inline(always)]
    fn bitor(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U16x8Backend>::bitor(a[i], b[i]))
    }

    #[inline(always)]
    fn bitxor(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U16x8Backend>::bitxor(a[i], b[i]))
    }
}

#[cfg(target_arch = "wasm32")]
impl I32x16Backend for archmage::Wasm128Token {
    type Repr = [v128; 4];

    #[inline(always)]
    fn splat(v: i32) -> [v128; 4] {
        let q = <archmage::Wasm128Token as I32x4Backend>::splat(v);
        [q, q, q, q]
    }

    #[inline(always)]
    fn zero() -> [v128; 4] {
        let q = <archmage::Wasm128Token as I32x4Backend>::zero();
        [q, q, q, q]
    }

    #[inline(always)]
    fn load(data: &[i32; 16]) -> [v128; 4] {
        [
            <archmage::Wasm128Token as I32x4Backend>::load(data[0..4].try_into().unwrap()),
            <archmage::Wasm128Token as I32x4Backend>::load(data[4..8].try_into().unwrap()),
            <archmage::Wasm128Token as I32x4Backend>::load(data[8..12].try_into().unwrap()),
            <archmage::Wasm128Token as I32x4Backend>::load(data[12..16].try_into().unwrap()),
        ]
    }

    #[inline(always)]
    fn from_array(arr: [i32; 16]) -> [v128; 4] {
        let mut q0 = [0; 4];
        let mut q1 = [0; 4];
        let mut q2 = [0; 4];
        let mut q3 = [0; 4];
        q0.copy_from_slice(&arr[0..4]);
        q1.copy_from_slice(&arr[4..8]);
        q2.copy_from_slice(&arr[8..12]);
        q3.copy_from_slice(&arr[12..16]);
        [
            <archmage::Wasm128Token as I32x4Backend>::from_array(q0),
            <archmage::Wasm128Token as I32x4Backend>::from_array(q1),
            <archmage::Wasm128Token as I32x4Backend>::from_array(q2),
            <archmage::Wasm128Token as I32x4Backend>::from_array(q3),
        ]
    }

    #[inline(always)]
    fn store(repr: [v128; 4], out: &mut [i32; 16]) {
        let (o01, o23) = out.split_at_mut(8);
        let (o0, o1) = o01.split_at_mut(4);
        let (o2, o3) = o23.split_at_mut(4);
        <archmage::Wasm128Token as I32x4Backend>::store(repr[0], o0.try_into().unwrap());
        <archmage::Wasm128Token as I32x4Backend>::store(repr[1], o1.try_into().unwrap());
        <archmage::Wasm128Token as I32x4Backend>::store(repr[2], o2.try_into().unwrap());
        <archmage::Wasm128Token as I32x4Backend>::store(repr[3], o3.try_into().unwrap());
    }

    #[inline(always)]
    fn to_array(repr: [v128; 4]) -> [i32; 16] {
        let a0 = <archmage::Wasm128Token as I32x4Backend>::to_array(repr[0]);
        let a1 = <archmage::Wasm128Token as I32x4Backend>::to_array(repr[1]);
        let a2 = <archmage::Wasm128Token as I32x4Backend>::to_array(repr[2]);
        let a3 = <archmage::Wasm128Token as I32x4Backend>::to_array(repr[3]);
        let mut out = [0; 16];
        out[0..4].copy_from_slice(&a0);
        out[4..8].copy_from_slice(&a1);
        out[8..12].copy_from_slice(&a2);
        out[12..16].copy_from_slice(&a3);
        out
    }

    #[inline(always)]
    fn add(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I32x4Backend>::add(a[i], b[i]))
    }

    #[inline(always)]
    fn sub(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I32x4Backend>::sub(a[i], b[i]))
    }

    #[inline(always)]
    fn mul(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I32x4Backend>::mul(a[i], b[i]))
    }

    #[inline(always)]
    fn neg(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I32x4Backend>::neg(a[i]))
    }

    #[inline(always)]
    fn min(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I32x4Backend>::min(a[i], b[i]))
    }

    #[inline(always)]
    fn max(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I32x4Backend>::max(a[i], b[i]))
    }

    #[inline(always)]
    fn abs(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I32x4Backend>::abs(a[i]))
    }

    #[inline(always)]
    fn reduce_add(a: [v128; 4]) -> i32 {
        <archmage::Wasm128Token as I32x4Backend>::reduce_add(a[0])
            .wrapping_add(<archmage::Wasm128Token as I32x4Backend>::reduce_add(a[1]))
            .wrapping_add(<archmage::Wasm128Token as I32x4Backend>::reduce_add(a[2]))
            .wrapping_add(<archmage::Wasm128Token as I32x4Backend>::reduce_add(a[3]))
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I32x4Backend>::shl_const::<N>(a[i]))
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| {
            <archmage::Wasm128Token as I32x4Backend>::shr_arithmetic_const::<N>(a[i])
        })
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| {
            <archmage::Wasm128Token as I32x4Backend>::shr_logical_const::<N>(a[i])
        })
    }

    #[inline(always)]
    fn all_true(a: [v128; 4]) -> bool {
        <archmage::Wasm128Token as I32x4Backend>::all_true(a[0])
            && <archmage::Wasm128Token as I32x4Backend>::all_true(a[1])
            && <archmage::Wasm128Token as I32x4Backend>::all_true(a[2])
            && <archmage::Wasm128Token as I32x4Backend>::all_true(a[3])
    }

    #[inline(always)]
    fn any_true(a: [v128; 4]) -> bool {
        <archmage::Wasm128Token as I32x4Backend>::any_true(a[0])
            || <archmage::Wasm128Token as I32x4Backend>::any_true(a[1])
            || <archmage::Wasm128Token as I32x4Backend>::any_true(a[2])
            || <archmage::Wasm128Token as I32x4Backend>::any_true(a[3])
    }

    #[inline(always)]
    fn bitmask(a: [v128; 4]) -> u64 {
        let q0 = <archmage::Wasm128Token as I32x4Backend>::bitmask(a[0]) as u64;
        let q1 = <archmage::Wasm128Token as I32x4Backend>::bitmask(a[1]) as u64;
        let q2 = <archmage::Wasm128Token as I32x4Backend>::bitmask(a[2]) as u64;
        let q3 = <archmage::Wasm128Token as I32x4Backend>::bitmask(a[3]) as u64;
        q0 | (q1 << 4) | (q2 << 8) | (q3 << 12)
    }

    #[inline(always)]
    fn simd_eq(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I32x4Backend>::simd_eq(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ne(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I32x4Backend>::simd_ne(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_lt(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I32x4Backend>::simd_lt(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_le(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I32x4Backend>::simd_le(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_gt(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I32x4Backend>::simd_gt(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ge(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I32x4Backend>::simd_ge(a[i], b[i]))
    }

    #[inline(always)]
    fn blend(mask: [v128; 4], if_true: [v128; 4], if_false: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| {
            <archmage::Wasm128Token as I32x4Backend>::blend(mask[i], if_true[i], if_false[i])
        })
    }

    #[inline(always)]
    fn not(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I32x4Backend>::not(a[i]))
    }

    #[inline(always)]
    fn bitand(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I32x4Backend>::bitand(a[i], b[i]))
    }

    #[inline(always)]
    fn bitor(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I32x4Backend>::bitor(a[i], b[i]))
    }

    #[inline(always)]
    fn bitxor(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I32x4Backend>::bitxor(a[i], b[i]))
    }
}

#[cfg(target_arch = "wasm32")]
impl U32x16Backend for archmage::Wasm128Token {
    type Repr = [v128; 4];

    #[inline(always)]
    fn splat(v: u32) -> [v128; 4] {
        let q = <archmage::Wasm128Token as U32x4Backend>::splat(v);
        [q, q, q, q]
    }

    #[inline(always)]
    fn zero() -> [v128; 4] {
        let q = <archmage::Wasm128Token as U32x4Backend>::zero();
        [q, q, q, q]
    }

    #[inline(always)]
    fn load(data: &[u32; 16]) -> [v128; 4] {
        [
            <archmage::Wasm128Token as U32x4Backend>::load(data[0..4].try_into().unwrap()),
            <archmage::Wasm128Token as U32x4Backend>::load(data[4..8].try_into().unwrap()),
            <archmage::Wasm128Token as U32x4Backend>::load(data[8..12].try_into().unwrap()),
            <archmage::Wasm128Token as U32x4Backend>::load(data[12..16].try_into().unwrap()),
        ]
    }

    #[inline(always)]
    fn from_array(arr: [u32; 16]) -> [v128; 4] {
        let mut q0 = [0; 4];
        let mut q1 = [0; 4];
        let mut q2 = [0; 4];
        let mut q3 = [0; 4];
        q0.copy_from_slice(&arr[0..4]);
        q1.copy_from_slice(&arr[4..8]);
        q2.copy_from_slice(&arr[8..12]);
        q3.copy_from_slice(&arr[12..16]);
        [
            <archmage::Wasm128Token as U32x4Backend>::from_array(q0),
            <archmage::Wasm128Token as U32x4Backend>::from_array(q1),
            <archmage::Wasm128Token as U32x4Backend>::from_array(q2),
            <archmage::Wasm128Token as U32x4Backend>::from_array(q3),
        ]
    }

    #[inline(always)]
    fn store(repr: [v128; 4], out: &mut [u32; 16]) {
        let (o01, o23) = out.split_at_mut(8);
        let (o0, o1) = o01.split_at_mut(4);
        let (o2, o3) = o23.split_at_mut(4);
        <archmage::Wasm128Token as U32x4Backend>::store(repr[0], o0.try_into().unwrap());
        <archmage::Wasm128Token as U32x4Backend>::store(repr[1], o1.try_into().unwrap());
        <archmage::Wasm128Token as U32x4Backend>::store(repr[2], o2.try_into().unwrap());
        <archmage::Wasm128Token as U32x4Backend>::store(repr[3], o3.try_into().unwrap());
    }

    #[inline(always)]
    fn to_array(repr: [v128; 4]) -> [u32; 16] {
        let a0 = <archmage::Wasm128Token as U32x4Backend>::to_array(repr[0]);
        let a1 = <archmage::Wasm128Token as U32x4Backend>::to_array(repr[1]);
        let a2 = <archmage::Wasm128Token as U32x4Backend>::to_array(repr[2]);
        let a3 = <archmage::Wasm128Token as U32x4Backend>::to_array(repr[3]);
        let mut out = [0; 16];
        out[0..4].copy_from_slice(&a0);
        out[4..8].copy_from_slice(&a1);
        out[8..12].copy_from_slice(&a2);
        out[12..16].copy_from_slice(&a3);
        out
    }

    #[inline(always)]
    fn add(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U32x4Backend>::add(a[i], b[i]))
    }

    #[inline(always)]
    fn sub(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U32x4Backend>::sub(a[i], b[i]))
    }

    #[inline(always)]
    fn mul(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U32x4Backend>::mul(a[i], b[i]))
    }

    #[inline(always)]
    fn neg(a: [v128; 4]) -> [v128; 4] {
        let z = <archmage::Wasm128Token as U32x4Backend>::zero();
        core::array::from_fn(|i| <archmage::Wasm128Token as U32x4Backend>::sub(z, a[i]))
    }

    #[inline(always)]
    fn min(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U32x4Backend>::min(a[i], b[i]))
    }

    #[inline(always)]
    fn max(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U32x4Backend>::max(a[i], b[i]))
    }

    #[inline(always)]
    fn reduce_add(a: [v128; 4]) -> u32 {
        <archmage::Wasm128Token as U32x4Backend>::reduce_add(a[0])
            .wrapping_add(<archmage::Wasm128Token as U32x4Backend>::reduce_add(a[1]))
            .wrapping_add(<archmage::Wasm128Token as U32x4Backend>::reduce_add(a[2]))
            .wrapping_add(<archmage::Wasm128Token as U32x4Backend>::reduce_add(a[3]))
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U32x4Backend>::shl_const::<N>(a[i]))
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| {
            <archmage::Wasm128Token as U32x4Backend>::shr_logical_const::<N>(a[i])
        })
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| {
            <archmage::Wasm128Token as U32x4Backend>::shr_logical_const::<N>(a[i])
        })
    }

    #[inline(always)]
    fn all_true(a: [v128; 4]) -> bool {
        <archmage::Wasm128Token as U32x4Backend>::all_true(a[0])
            && <archmage::Wasm128Token as U32x4Backend>::all_true(a[1])
            && <archmage::Wasm128Token as U32x4Backend>::all_true(a[2])
            && <archmage::Wasm128Token as U32x4Backend>::all_true(a[3])
    }

    #[inline(always)]
    fn any_true(a: [v128; 4]) -> bool {
        <archmage::Wasm128Token as U32x4Backend>::any_true(a[0])
            || <archmage::Wasm128Token as U32x4Backend>::any_true(a[1])
            || <archmage::Wasm128Token as U32x4Backend>::any_true(a[2])
            || <archmage::Wasm128Token as U32x4Backend>::any_true(a[3])
    }

    #[inline(always)]
    fn bitmask(a: [v128; 4]) -> u64 {
        let q0 = <archmage::Wasm128Token as U32x4Backend>::bitmask(a[0]) as u64;
        let q1 = <archmage::Wasm128Token as U32x4Backend>::bitmask(a[1]) as u64;
        let q2 = <archmage::Wasm128Token as U32x4Backend>::bitmask(a[2]) as u64;
        let q3 = <archmage::Wasm128Token as U32x4Backend>::bitmask(a[3]) as u64;
        q0 | (q1 << 4) | (q2 << 8) | (q3 << 12)
    }

    #[inline(always)]
    fn simd_eq(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U32x4Backend>::simd_eq(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ne(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U32x4Backend>::simd_ne(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_lt(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U32x4Backend>::simd_lt(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_le(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U32x4Backend>::simd_le(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_gt(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U32x4Backend>::simd_gt(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ge(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U32x4Backend>::simd_ge(a[i], b[i]))
    }

    #[inline(always)]
    fn blend(mask: [v128; 4], if_true: [v128; 4], if_false: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| {
            <archmage::Wasm128Token as U32x4Backend>::blend(mask[i], if_true[i], if_false[i])
        })
    }

    #[inline(always)]
    fn not(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U32x4Backend>::not(a[i]))
    }

    #[inline(always)]
    fn bitand(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U32x4Backend>::bitand(a[i], b[i]))
    }

    #[inline(always)]
    fn bitor(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U32x4Backend>::bitor(a[i], b[i]))
    }

    #[inline(always)]
    fn bitxor(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U32x4Backend>::bitxor(a[i], b[i]))
    }
}

#[cfg(target_arch = "wasm32")]
impl I64x8Backend for archmage::Wasm128Token {
    type Repr = [v128; 4];

    #[inline(always)]
    fn splat(v: i64) -> [v128; 4] {
        let q = <archmage::Wasm128Token as I64x2Backend>::splat(v);
        [q, q, q, q]
    }

    #[inline(always)]
    fn zero() -> [v128; 4] {
        let q = <archmage::Wasm128Token as I64x2Backend>::zero();
        [q, q, q, q]
    }

    #[inline(always)]
    fn load(data: &[i64; 8]) -> [v128; 4] {
        [
            <archmage::Wasm128Token as I64x2Backend>::load(data[0..2].try_into().unwrap()),
            <archmage::Wasm128Token as I64x2Backend>::load(data[2..4].try_into().unwrap()),
            <archmage::Wasm128Token as I64x2Backend>::load(data[4..6].try_into().unwrap()),
            <archmage::Wasm128Token as I64x2Backend>::load(data[6..8].try_into().unwrap()),
        ]
    }

    #[inline(always)]
    fn from_array(arr: [i64; 8]) -> [v128; 4] {
        let mut q0 = [0; 2];
        let mut q1 = [0; 2];
        let mut q2 = [0; 2];
        let mut q3 = [0; 2];
        q0.copy_from_slice(&arr[0..2]);
        q1.copy_from_slice(&arr[2..4]);
        q2.copy_from_slice(&arr[4..6]);
        q3.copy_from_slice(&arr[6..8]);
        [
            <archmage::Wasm128Token as I64x2Backend>::from_array(q0),
            <archmage::Wasm128Token as I64x2Backend>::from_array(q1),
            <archmage::Wasm128Token as I64x2Backend>::from_array(q2),
            <archmage::Wasm128Token as I64x2Backend>::from_array(q3),
        ]
    }

    #[inline(always)]
    fn store(repr: [v128; 4], out: &mut [i64; 8]) {
        let (o01, o23) = out.split_at_mut(4);
        let (o0, o1) = o01.split_at_mut(2);
        let (o2, o3) = o23.split_at_mut(2);
        <archmage::Wasm128Token as I64x2Backend>::store(repr[0], o0.try_into().unwrap());
        <archmage::Wasm128Token as I64x2Backend>::store(repr[1], o1.try_into().unwrap());
        <archmage::Wasm128Token as I64x2Backend>::store(repr[2], o2.try_into().unwrap());
        <archmage::Wasm128Token as I64x2Backend>::store(repr[3], o3.try_into().unwrap());
    }

    #[inline(always)]
    fn to_array(repr: [v128; 4]) -> [i64; 8] {
        let a0 = <archmage::Wasm128Token as I64x2Backend>::to_array(repr[0]);
        let a1 = <archmage::Wasm128Token as I64x2Backend>::to_array(repr[1]);
        let a2 = <archmage::Wasm128Token as I64x2Backend>::to_array(repr[2]);
        let a3 = <archmage::Wasm128Token as I64x2Backend>::to_array(repr[3]);
        let mut out = [0; 8];
        out[0..2].copy_from_slice(&a0);
        out[2..4].copy_from_slice(&a1);
        out[4..6].copy_from_slice(&a2);
        out[6..8].copy_from_slice(&a3);
        out
    }

    #[inline(always)]
    fn add(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I64x2Backend>::add(a[i], b[i]))
    }

    #[inline(always)]
    fn sub(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I64x2Backend>::sub(a[i], b[i]))
    }

    #[inline(always)]
    fn neg(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I64x2Backend>::neg(a[i]))
    }

    #[inline(always)]
    fn min(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I64x2Backend>::min(a[i], b[i]))
    }

    #[inline(always)]
    fn max(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I64x2Backend>::max(a[i], b[i]))
    }

    #[inline(always)]
    fn abs(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I64x2Backend>::abs(a[i]))
    }

    #[inline(always)]
    fn reduce_add(a: [v128; 4]) -> i64 {
        <archmage::Wasm128Token as I64x2Backend>::reduce_add(a[0])
            .wrapping_add(<archmage::Wasm128Token as I64x2Backend>::reduce_add(a[1]))
            .wrapping_add(<archmage::Wasm128Token as I64x2Backend>::reduce_add(a[2]))
            .wrapping_add(<archmage::Wasm128Token as I64x2Backend>::reduce_add(a[3]))
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I64x2Backend>::shl_const::<N>(a[i]))
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| {
            <archmage::Wasm128Token as I64x2Backend>::shr_arithmetic_const::<N>(a[i])
        })
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| {
            <archmage::Wasm128Token as I64x2Backend>::shr_logical_const::<N>(a[i])
        })
    }

    #[inline(always)]
    fn all_true(a: [v128; 4]) -> bool {
        <archmage::Wasm128Token as I64x2Backend>::all_true(a[0])
            && <archmage::Wasm128Token as I64x2Backend>::all_true(a[1])
            && <archmage::Wasm128Token as I64x2Backend>::all_true(a[2])
            && <archmage::Wasm128Token as I64x2Backend>::all_true(a[3])
    }

    #[inline(always)]
    fn any_true(a: [v128; 4]) -> bool {
        <archmage::Wasm128Token as I64x2Backend>::any_true(a[0])
            || <archmage::Wasm128Token as I64x2Backend>::any_true(a[1])
            || <archmage::Wasm128Token as I64x2Backend>::any_true(a[2])
            || <archmage::Wasm128Token as I64x2Backend>::any_true(a[3])
    }

    #[inline(always)]
    fn bitmask(a: [v128; 4]) -> u64 {
        let q0 = <archmage::Wasm128Token as I64x2Backend>::bitmask(a[0]) as u64;
        let q1 = <archmage::Wasm128Token as I64x2Backend>::bitmask(a[1]) as u64;
        let q2 = <archmage::Wasm128Token as I64x2Backend>::bitmask(a[2]) as u64;
        let q3 = <archmage::Wasm128Token as I64x2Backend>::bitmask(a[3]) as u64;
        q0 | (q1 << 2) | (q2 << 4) | (q3 << 6)
    }

    #[inline(always)]
    fn simd_eq(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I64x2Backend>::simd_eq(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ne(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I64x2Backend>::simd_ne(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_lt(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I64x2Backend>::simd_lt(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_le(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I64x2Backend>::simd_le(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_gt(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I64x2Backend>::simd_gt(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ge(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I64x2Backend>::simd_ge(a[i], b[i]))
    }

    #[inline(always)]
    fn blend(mask: [v128; 4], if_true: [v128; 4], if_false: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| {
            <archmage::Wasm128Token as I64x2Backend>::blend(mask[i], if_true[i], if_false[i])
        })
    }

    #[inline(always)]
    fn not(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I64x2Backend>::not(a[i]))
    }

    #[inline(always)]
    fn bitand(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I64x2Backend>::bitand(a[i], b[i]))
    }

    #[inline(always)]
    fn bitor(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I64x2Backend>::bitor(a[i], b[i]))
    }

    #[inline(always)]
    fn bitxor(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as I64x2Backend>::bitxor(a[i], b[i]))
    }
}

#[cfg(target_arch = "wasm32")]
impl U64x8Backend for archmage::Wasm128Token {
    type Repr = [v128; 4];

    #[inline(always)]
    fn splat(v: u64) -> [v128; 4] {
        let q = <archmage::Wasm128Token as U64x2Backend>::splat(v);
        [q, q, q, q]
    }

    #[inline(always)]
    fn zero() -> [v128; 4] {
        let q = <archmage::Wasm128Token as U64x2Backend>::zero();
        [q, q, q, q]
    }

    #[inline(always)]
    fn load(data: &[u64; 8]) -> [v128; 4] {
        [
            <archmage::Wasm128Token as U64x2Backend>::load(data[0..2].try_into().unwrap()),
            <archmage::Wasm128Token as U64x2Backend>::load(data[2..4].try_into().unwrap()),
            <archmage::Wasm128Token as U64x2Backend>::load(data[4..6].try_into().unwrap()),
            <archmage::Wasm128Token as U64x2Backend>::load(data[6..8].try_into().unwrap()),
        ]
    }

    #[inline(always)]
    fn from_array(arr: [u64; 8]) -> [v128; 4] {
        let mut q0 = [0; 2];
        let mut q1 = [0; 2];
        let mut q2 = [0; 2];
        let mut q3 = [0; 2];
        q0.copy_from_slice(&arr[0..2]);
        q1.copy_from_slice(&arr[2..4]);
        q2.copy_from_slice(&arr[4..6]);
        q3.copy_from_slice(&arr[6..8]);
        [
            <archmage::Wasm128Token as U64x2Backend>::from_array(q0),
            <archmage::Wasm128Token as U64x2Backend>::from_array(q1),
            <archmage::Wasm128Token as U64x2Backend>::from_array(q2),
            <archmage::Wasm128Token as U64x2Backend>::from_array(q3),
        ]
    }

    #[inline(always)]
    fn store(repr: [v128; 4], out: &mut [u64; 8]) {
        let (o01, o23) = out.split_at_mut(4);
        let (o0, o1) = o01.split_at_mut(2);
        let (o2, o3) = o23.split_at_mut(2);
        <archmage::Wasm128Token as U64x2Backend>::store(repr[0], o0.try_into().unwrap());
        <archmage::Wasm128Token as U64x2Backend>::store(repr[1], o1.try_into().unwrap());
        <archmage::Wasm128Token as U64x2Backend>::store(repr[2], o2.try_into().unwrap());
        <archmage::Wasm128Token as U64x2Backend>::store(repr[3], o3.try_into().unwrap());
    }

    #[inline(always)]
    fn to_array(repr: [v128; 4]) -> [u64; 8] {
        let a0 = <archmage::Wasm128Token as U64x2Backend>::to_array(repr[0]);
        let a1 = <archmage::Wasm128Token as U64x2Backend>::to_array(repr[1]);
        let a2 = <archmage::Wasm128Token as U64x2Backend>::to_array(repr[2]);
        let a3 = <archmage::Wasm128Token as U64x2Backend>::to_array(repr[3]);
        let mut out = [0; 8];
        out[0..2].copy_from_slice(&a0);
        out[2..4].copy_from_slice(&a1);
        out[4..6].copy_from_slice(&a2);
        out[6..8].copy_from_slice(&a3);
        out
    }

    #[inline(always)]
    fn add(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U64x2Backend>::add(a[i], b[i]))
    }

    #[inline(always)]
    fn sub(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U64x2Backend>::sub(a[i], b[i]))
    }

    #[inline(always)]
    fn neg(a: [v128; 4]) -> [v128; 4] {
        let z = <archmage::Wasm128Token as U64x2Backend>::zero();
        core::array::from_fn(|i| <archmage::Wasm128Token as U64x2Backend>::sub(z, a[i]))
    }

    #[inline(always)]
    fn min(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U64x2Backend>::min(a[i], b[i]))
    }

    #[inline(always)]
    fn max(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U64x2Backend>::max(a[i], b[i]))
    }

    #[inline(always)]
    fn reduce_add(a: [v128; 4]) -> u64 {
        <archmage::Wasm128Token as U64x2Backend>::reduce_add(a[0])
            .wrapping_add(<archmage::Wasm128Token as U64x2Backend>::reduce_add(a[1]))
            .wrapping_add(<archmage::Wasm128Token as U64x2Backend>::reduce_add(a[2]))
            .wrapping_add(<archmage::Wasm128Token as U64x2Backend>::reduce_add(a[3]))
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U64x2Backend>::shl_const::<N>(a[i]))
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| {
            <archmage::Wasm128Token as U64x2Backend>::shr_logical_const::<N>(a[i])
        })
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| {
            <archmage::Wasm128Token as U64x2Backend>::shr_logical_const::<N>(a[i])
        })
    }

    #[inline(always)]
    fn all_true(a: [v128; 4]) -> bool {
        <archmage::Wasm128Token as U64x2Backend>::all_true(a[0])
            && <archmage::Wasm128Token as U64x2Backend>::all_true(a[1])
            && <archmage::Wasm128Token as U64x2Backend>::all_true(a[2])
            && <archmage::Wasm128Token as U64x2Backend>::all_true(a[3])
    }

    #[inline(always)]
    fn any_true(a: [v128; 4]) -> bool {
        <archmage::Wasm128Token as U64x2Backend>::any_true(a[0])
            || <archmage::Wasm128Token as U64x2Backend>::any_true(a[1])
            || <archmage::Wasm128Token as U64x2Backend>::any_true(a[2])
            || <archmage::Wasm128Token as U64x2Backend>::any_true(a[3])
    }

    #[inline(always)]
    fn bitmask(a: [v128; 4]) -> u64 {
        let q0 = <archmage::Wasm128Token as U64x2Backend>::bitmask(a[0]) as u64;
        let q1 = <archmage::Wasm128Token as U64x2Backend>::bitmask(a[1]) as u64;
        let q2 = <archmage::Wasm128Token as U64x2Backend>::bitmask(a[2]) as u64;
        let q3 = <archmage::Wasm128Token as U64x2Backend>::bitmask(a[3]) as u64;
        q0 | (q1 << 2) | (q2 << 4) | (q3 << 6)
    }

    #[inline(always)]
    fn simd_eq(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U64x2Backend>::simd_eq(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ne(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U64x2Backend>::simd_ne(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_lt(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U64x2Backend>::simd_lt(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_le(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U64x2Backend>::simd_le(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_gt(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U64x2Backend>::simd_gt(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ge(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U64x2Backend>::simd_ge(a[i], b[i]))
    }

    #[inline(always)]
    fn blend(mask: [v128; 4], if_true: [v128; 4], if_false: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| {
            <archmage::Wasm128Token as U64x2Backend>::blend(mask[i], if_true[i], if_false[i])
        })
    }

    #[inline(always)]
    fn not(a: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U64x2Backend>::not(a[i]))
    }

    #[inline(always)]
    fn bitand(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U64x2Backend>::bitand(a[i], b[i]))
    }

    #[inline(always)]
    fn bitor(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U64x2Backend>::bitor(a[i], b[i]))
    }

    #[inline(always)]
    fn bitxor(a: [v128; 4], b: [v128; 4]) -> [v128; 4] {
        core::array::from_fn(|i| <archmage::Wasm128Token as U64x2Backend>::bitxor(a[i], b[i]))
    }
}

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
        Self::load(&arr)
    }
    #[inline(always)]
    fn store(repr: v128, out: &mut [f32; 4]) {
        unsafe { v128_store(out.as_mut_ptr().cast(), repr) };
    }
    #[inline(always)]
    fn to_array(repr: v128) -> [f32; 4] {
        let mut out = [0.0f32; 4];
        Self::store(repr, &mut out);
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
        Self::load(&arr)
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
        Self::store(repr, &mut out);
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
        Self::load(&arr)
    }
    #[inline(always)]
    fn store(repr: v128, out: &mut [f64; 2]) {
        unsafe { v128_store(out.as_mut_ptr().cast(), repr) };
    }
    #[inline(always)]
    fn to_array(repr: v128) -> [f64; 2] {
        let mut out = [0.0f64; 2];
        Self::store(repr, &mut out);
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
        Self::load(&arr)
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
        Self::store(repr, &mut out);
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

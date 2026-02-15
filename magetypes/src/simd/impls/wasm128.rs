//! F32x8Backend implementation for Wasm128Token (WebAssembly SIMD).
//!
//! Repr = `[v128; 2]`. Each operation applies to both halves.
//! This is a 2x128-bit polyfill — WASM SIMD maxes out at 128 bits, so f32x8
//! is emulated with two f32x4 WASM SIMD operations.

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;

use crate::simd::backends::F32x8Backend;

#[cfg(target_arch = "wasm32")]
impl F32x8Backend for archmage::Wasm128Token {
    type Repr = [v128; 2];

    // ====== Construction ======

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
            let lo = v128_load(data.as_ptr().cast());
            let hi = v128_load(data.as_ptr().add(4).cast());
            [lo, hi]
        }
    }

    #[inline(always)]
    fn from_array(arr: [f32; 8]) -> [v128; 2] {
        Self::load(&arr)
    }

    #[inline(always)]
    fn store(repr: [v128; 2], out: &mut [f32; 8]) {
        unsafe {
            v128_store(out.as_mut_ptr().cast(), repr[0]);
            v128_store(out.as_mut_ptr().add(4).cast(), repr[1]);
        }
    }

    #[inline(always)]
    fn to_array(repr: [v128; 2]) -> [f32; 8] {
        let mut out = [0.0f32; 8];
        Self::store(repr, &mut out);
        out
    }

    // ====== Arithmetic ======

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

    // ====== Math ======

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
        // WASM has no native FMA — emulate with mul + add
        // mul_add(a, b, c) = a*b + c
        [
            f32x4_add(f32x4_mul(a[0], b[0]), c[0]),
            f32x4_add(f32x4_mul(a[1], b[1]), c[1]),
        ]
    }

    #[inline(always)]
    fn mul_sub(a: [v128; 2], b: [v128; 2], c: [v128; 2]) -> [v128; 2] {
        // mul_sub(a, b, c) = a*b - c
        [
            f32x4_sub(f32x4_mul(a[0], b[0]), c[0]),
            f32x4_sub(f32x4_mul(a[1], b[1]), c[1]),
        ]
    }

    // ====== Comparisons ======
    // WASM comparisons return v128 masks (all-1s or all-0s per lane).

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
        // v128_bitselect(v1, v2, mask): picks v1 where mask=1, v2 where mask=0
        [
            v128_bitselect(if_true[0], if_false[0], mask[0]),
            v128_bitselect(if_true[1], if_false[1], mask[1]),
        ]
    }

    // ====== Reductions ======
    // WASM has no pairwise reduction — extract and fold.

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

    // ====== Approximations ======
    // WASM has no native rcp/rsqrt — use full division.

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
        Self::rcp_approx(a)
    }

    #[inline(always)]
    fn rsqrt(a: [v128; 2]) -> [v128; 2] {
        Self::rsqrt_approx(a)
    }

    // ====== Bitwise ======
    // WASM v128 bitwise ops work on any lane interpretation.

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

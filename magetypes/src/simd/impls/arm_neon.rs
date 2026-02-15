//! F32x8Backend implementation for NeonToken (AArch64 NEON).
//!
//! Repr = `[float32x4_t; 2]`. Each operation applies to both halves.
//! This is a 2x128-bit polyfill — NEON maxes out at 128 bits, so f32x8
//! is emulated with two f32x4 NEON operations.

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

use crate::simd::backends::F32x8Backend;

#[cfg(target_arch = "aarch64")]
impl F32x8Backend for archmage::NeonToken {
    type Repr = [float32x4_t; 2];

    // ====== Construction ======

    #[inline(always)]
    fn splat(v: f32) -> [float32x4_t; 2] {
        unsafe {
            let v4 = vdupq_n_f32(v);
            [v4, v4]
        }
    }

    #[inline(always)]
    fn zero() -> [float32x4_t; 2] {
        unsafe {
            let z = vdupq_n_f32(0.0);
            [z, z]
        }
    }

    #[inline(always)]
    fn load(data: &[f32; 8]) -> [float32x4_t; 2] {
        unsafe {
            let lo = vld1q_f32(data.as_ptr());
            let hi = vld1q_f32(data.as_ptr().add(4));
            [lo, hi]
        }
    }

    #[inline(always)]
    fn from_array(arr: [f32; 8]) -> [float32x4_t; 2] {
        Self::load(&arr)
    }

    #[inline(always)]
    fn store(repr: [float32x4_t; 2], out: &mut [f32; 8]) {
        unsafe {
            vst1q_f32(out.as_mut_ptr(), repr[0]);
            vst1q_f32(out.as_mut_ptr().add(4), repr[1]);
        }
    }

    #[inline(always)]
    fn to_array(repr: [float32x4_t; 2]) -> [f32; 8] {
        let mut out = [0.0f32; 8];
        Self::store(repr, &mut out);
        out
    }

    // ====== Arithmetic ======

    #[inline(always)]
    fn add(a: [float32x4_t; 2], b: [float32x4_t; 2]) -> [float32x4_t; 2] {
        unsafe { [vaddq_f32(a[0], b[0]), vaddq_f32(a[1], b[1])] }
    }

    #[inline(always)]
    fn sub(a: [float32x4_t; 2], b: [float32x4_t; 2]) -> [float32x4_t; 2] {
        unsafe { [vsubq_f32(a[0], b[0]), vsubq_f32(a[1], b[1])] }
    }

    #[inline(always)]
    fn mul(a: [float32x4_t; 2], b: [float32x4_t; 2]) -> [float32x4_t; 2] {
        unsafe { [vmulq_f32(a[0], b[0]), vmulq_f32(a[1], b[1])] }
    }

    #[inline(always)]
    fn div(a: [float32x4_t; 2], b: [float32x4_t; 2]) -> [float32x4_t; 2] {
        unsafe { [vdivq_f32(a[0], b[0]), vdivq_f32(a[1], b[1])] }
    }

    #[inline(always)]
    fn neg(a: [float32x4_t; 2]) -> [float32x4_t; 2] {
        unsafe { [vnegq_f32(a[0]), vnegq_f32(a[1])] }
    }

    // ====== Math ======

    #[inline(always)]
    fn min(a: [float32x4_t; 2], b: [float32x4_t; 2]) -> [float32x4_t; 2] {
        unsafe { [vminq_f32(a[0], b[0]), vminq_f32(a[1], b[1])] }
    }

    #[inline(always)]
    fn max(a: [float32x4_t; 2], b: [float32x4_t; 2]) -> [float32x4_t; 2] {
        unsafe { [vmaxq_f32(a[0], b[0]), vmaxq_f32(a[1], b[1])] }
    }

    #[inline(always)]
    fn sqrt(a: [float32x4_t; 2]) -> [float32x4_t; 2] {
        unsafe { [vsqrtq_f32(a[0]), vsqrtq_f32(a[1])] }
    }

    #[inline(always)]
    fn abs(a: [float32x4_t; 2]) -> [float32x4_t; 2] {
        unsafe { [vabsq_f32(a[0]), vabsq_f32(a[1])] }
    }

    #[inline(always)]
    fn floor(a: [float32x4_t; 2]) -> [float32x4_t; 2] {
        unsafe { [vrndmq_f32(a[0]), vrndmq_f32(a[1])] }
    }

    #[inline(always)]
    fn ceil(a: [float32x4_t; 2]) -> [float32x4_t; 2] {
        unsafe { [vrndpq_f32(a[0]), vrndpq_f32(a[1])] }
    }

    #[inline(always)]
    fn round(a: [float32x4_t; 2]) -> [float32x4_t; 2] {
        unsafe { [vrndnq_f32(a[0]), vrndnq_f32(a[1])] }
    }

    #[inline(always)]
    fn mul_add(a: [float32x4_t; 2], b: [float32x4_t; 2], c: [float32x4_t; 2]) -> [float32x4_t; 2] {
        // vfmaq_f32(acc, x, y) = acc + x*y
        // mul_add(a, b, c) = a*b + c => vfmaq_f32(c, a, b)
        unsafe { [vfmaq_f32(c[0], a[0], b[0]), vfmaq_f32(c[1], a[1], b[1])] }
    }

    #[inline(always)]
    fn mul_sub(a: [float32x4_t; 2], b: [float32x4_t; 2], c: [float32x4_t; 2]) -> [float32x4_t; 2] {
        // mul_sub(a, b, c) = a*b - c => vfmaq_f32(-c, a, b) = -c + a*b
        unsafe {
            [
                vfmaq_f32(vnegq_f32(c[0]), a[0], b[0]),
                vfmaq_f32(vnegq_f32(c[1]), a[1], b[1]),
            ]
        }
    }

    // ====== Comparisons ======
    // NEON comparisons return u32 masks; reinterpret to f32 for our trait convention.

    #[inline(always)]
    fn simd_eq(a: [float32x4_t; 2], b: [float32x4_t; 2]) -> [float32x4_t; 2] {
        unsafe {
            [
                vreinterpretq_f32_u32(vceqq_f32(a[0], b[0])),
                vreinterpretq_f32_u32(vceqq_f32(a[1], b[1])),
            ]
        }
    }

    #[inline(always)]
    fn simd_ne(a: [float32x4_t; 2], b: [float32x4_t; 2]) -> [float32x4_t; 2] {
        unsafe {
            [
                vreinterpretq_f32_u32(vmvnq_u32(vceqq_f32(a[0], b[0]))),
                vreinterpretq_f32_u32(vmvnq_u32(vceqq_f32(a[1], b[1]))),
            ]
        }
    }

    #[inline(always)]
    fn simd_lt(a: [float32x4_t; 2], b: [float32x4_t; 2]) -> [float32x4_t; 2] {
        unsafe {
            [
                vreinterpretq_f32_u32(vcltq_f32(a[0], b[0])),
                vreinterpretq_f32_u32(vcltq_f32(a[1], b[1])),
            ]
        }
    }

    #[inline(always)]
    fn simd_le(a: [float32x4_t; 2], b: [float32x4_t; 2]) -> [float32x4_t; 2] {
        unsafe {
            [
                vreinterpretq_f32_u32(vcleq_f32(a[0], b[0])),
                vreinterpretq_f32_u32(vcleq_f32(a[1], b[1])),
            ]
        }
    }

    #[inline(always)]
    fn simd_gt(a: [float32x4_t; 2], b: [float32x4_t; 2]) -> [float32x4_t; 2] {
        unsafe {
            [
                vreinterpretq_f32_u32(vcgtq_f32(a[0], b[0])),
                vreinterpretq_f32_u32(vcgtq_f32(a[1], b[1])),
            ]
        }
    }

    #[inline(always)]
    fn simd_ge(a: [float32x4_t; 2], b: [float32x4_t; 2]) -> [float32x4_t; 2] {
        unsafe {
            [
                vreinterpretq_f32_u32(vcgeq_f32(a[0], b[0])),
                vreinterpretq_f32_u32(vcgeq_f32(a[1], b[1])),
            ]
        }
    }

    #[inline(always)]
    fn blend(
        mask: [float32x4_t; 2],
        if_true: [float32x4_t; 2],
        if_false: [float32x4_t; 2],
    ) -> [float32x4_t; 2] {
        unsafe {
            [
                vbslq_f32(vreinterpretq_u32_f32(mask[0]), if_true[0], if_false[0]),
                vbslq_f32(vreinterpretq_u32_f32(mask[1]), if_true[1], if_false[1]),
            ]
        }
    }

    // ====== Reductions ======

    #[inline(always)]
    fn reduce_add(a: [float32x4_t; 2]) -> f32 {
        unsafe {
            // Combine halves first, then pairwise-reduce the f32x4
            let sum = vaddq_f32(a[0], a[1]);
            let pair = vpaddq_f32(sum, sum);
            let pair = vpaddq_f32(pair, pair);
            vgetq_lane_f32::<0>(pair)
        }
    }

    #[inline(always)]
    fn reduce_min(a: [float32x4_t; 2]) -> f32 {
        unsafe {
            let m = vminq_f32(a[0], a[1]);
            let pair = vpminq_f32(m, m);
            let pair = vpminq_f32(pair, pair);
            vgetq_lane_f32::<0>(pair)
        }
    }

    #[inline(always)]
    fn reduce_max(a: [float32x4_t; 2]) -> f32 {
        unsafe {
            let m = vmaxq_f32(a[0], a[1]);
            let pair = vpmaxq_f32(m, m);
            let pair = vpmaxq_f32(pair, pair);
            vgetq_lane_f32::<0>(pair)
        }
    }

    // ====== Approximations ======

    #[inline(always)]
    fn rcp_approx(a: [float32x4_t; 2]) -> [float32x4_t; 2] {
        unsafe { [vrecpeq_f32(a[0]), vrecpeq_f32(a[1])] }
    }

    #[inline(always)]
    fn rsqrt_approx(a: [float32x4_t; 2]) -> [float32x4_t; 2] {
        unsafe { [vrsqrteq_f32(a[0]), vrsqrteq_f32(a[1])] }
    }

    // ====== Bitwise ======
    // NEON has no native float bitwise ops; reinterpret through u32.

    #[inline(always)]
    fn not(a: [float32x4_t; 2]) -> [float32x4_t; 2] {
        unsafe {
            [
                vreinterpretq_f32_u32(vmvnq_u32(vreinterpretq_u32_f32(a[0]))),
                vreinterpretq_f32_u32(vmvnq_u32(vreinterpretq_u32_f32(a[1]))),
            ]
        }
    }

    #[inline(always)]
    fn bitand(a: [float32x4_t; 2], b: [float32x4_t; 2]) -> [float32x4_t; 2] {
        unsafe {
            [
                vreinterpretq_f32_u32(vandq_u32(
                    vreinterpretq_u32_f32(a[0]),
                    vreinterpretq_u32_f32(b[0]),
                )),
                vreinterpretq_f32_u32(vandq_u32(
                    vreinterpretq_u32_f32(a[1]),
                    vreinterpretq_u32_f32(b[1]),
                )),
            ]
        }
    }

    #[inline(always)]
    fn bitor(a: [float32x4_t; 2], b: [float32x4_t; 2]) -> [float32x4_t; 2] {
        unsafe {
            [
                vreinterpretq_f32_u32(vorrq_u32(
                    vreinterpretq_u32_f32(a[0]),
                    vreinterpretq_u32_f32(b[0]),
                )),
                vreinterpretq_f32_u32(vorrq_u32(
                    vreinterpretq_u32_f32(a[1]),
                    vreinterpretq_u32_f32(b[1]),
                )),
            ]
        }
    }

    #[inline(always)]
    fn bitxor(a: [float32x4_t; 2], b: [float32x4_t; 2]) -> [float32x4_t; 2] {
        unsafe {
            [
                vreinterpretq_f32_u32(veorq_u32(
                    vreinterpretq_u32_f32(a[0]),
                    vreinterpretq_u32_f32(b[0]),
                )),
                vreinterpretq_f32_u32(veorq_u32(
                    vreinterpretq_u32_f32(a[1]),
                    vreinterpretq_u32_f32(b[1]),
                )),
            ]
        }
    }
}

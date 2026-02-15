//! Backend implementations for NeonToken (AArch64 NEON).
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

use crate::simd::backends::*;

#[cfg(target_arch = "aarch64")]
impl F32x4Backend for archmage::NeonToken {
    type Repr = float32x4_t;

    #[inline(always)]
    fn splat(v: f32) -> float32x4_t {
        unsafe { vdupq_n_f32(v) }
    }

    #[inline(always)]
    fn zero() -> float32x4_t {
        unsafe { vdupq_n_f32(0.0) }
    }

    #[inline(always)]
    fn load(data: &[f32; 4]) -> float32x4_t {
        unsafe { vld1q_f32(data.as_ptr()) }
    }

    #[inline(always)]
    fn from_array(arr: [f32; 4]) -> float32x4_t {
        Self::load(&arr)
    }

    #[inline(always)]
    fn store(repr: float32x4_t, out: &mut [f32; 4]) {
        unsafe { vst1q_f32(out.as_mut_ptr(), repr) };
    }

    #[inline(always)]
    fn to_array(repr: float32x4_t) -> [f32; 4] {
        let mut out = [0.0f32; 4];
        Self::store(repr, &mut out);
        out
    }

    #[inline(always)]
    fn add(a: float32x4_t, b: float32x4_t) -> float32x4_t {
        unsafe { vaddq_f32(a, b) }
    }
    #[inline(always)]
    fn sub(a: float32x4_t, b: float32x4_t) -> float32x4_t {
        unsafe { vsubq_f32(a, b) }
    }
    #[inline(always)]
    fn mul(a: float32x4_t, b: float32x4_t) -> float32x4_t {
        unsafe { vmulq_f32(a, b) }
    }
    #[inline(always)]
    fn div(a: float32x4_t, b: float32x4_t) -> float32x4_t {
        unsafe { vdivq_f32(a, b) }
    }
    #[inline(always)]
    fn neg(a: float32x4_t) -> float32x4_t {
        unsafe { vnegq_f32(a) }
    }
    #[inline(always)]
    fn min(a: float32x4_t, b: float32x4_t) -> float32x4_t {
        unsafe { vminq_f32(a, b) }
    }
    #[inline(always)]
    fn max(a: float32x4_t, b: float32x4_t) -> float32x4_t {
        unsafe { vmaxq_f32(a, b) }
    }
    #[inline(always)]
    fn sqrt(a: float32x4_t) -> float32x4_t {
        unsafe { vsqrtq_f32(a) }
    }
    #[inline(always)]
    fn abs(a: float32x4_t) -> float32x4_t {
        unsafe { vabsq_f32(a) }
    }
    #[inline(always)]
    fn floor(a: float32x4_t) -> float32x4_t {
        unsafe { vrndmq_f32(a) }
    }
    #[inline(always)]
    fn ceil(a: float32x4_t) -> float32x4_t {
        unsafe { vrndpq_f32(a) }
    }
    #[inline(always)]
    fn round(a: float32x4_t) -> float32x4_t {
        unsafe { vrndnq_f32(a) }
    }

    #[inline(always)]
    fn mul_add(a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
        unsafe { vfmaq_f32(c, a, b) }
    }

    #[inline(always)]
    fn mul_sub(a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
        unsafe { vfmaq_f32(vnegq_f32(c), a, b) }
    }

    #[inline(always)]
    fn simd_eq(a: float32x4_t, b: float32x4_t) -> float32x4_t {
        unsafe { vreinterpretq_f32_u32(vceqq_f32(a, b)) }
    }
    #[inline(always)]
    fn simd_ne(a: float32x4_t, b: float32x4_t) -> float32x4_t {
        unsafe { vreinterpretq_f32_u32(vmvnq_u32(vceqq_f32(a, b))) }
    }
    #[inline(always)]
    fn simd_lt(a: float32x4_t, b: float32x4_t) -> float32x4_t {
        unsafe { vreinterpretq_f32_u32(vcltq_f32(a, b)) }
    }
    #[inline(always)]
    fn simd_le(a: float32x4_t, b: float32x4_t) -> float32x4_t {
        unsafe { vreinterpretq_f32_u32(vcleq_f32(a, b)) }
    }
    #[inline(always)]
    fn simd_gt(a: float32x4_t, b: float32x4_t) -> float32x4_t {
        unsafe { vreinterpretq_f32_u32(vcgtq_f32(a, b)) }
    }
    #[inline(always)]
    fn simd_ge(a: float32x4_t, b: float32x4_t) -> float32x4_t {
        unsafe { vreinterpretq_f32_u32(vcgeq_f32(a, b)) }
    }

    #[inline(always)]
    fn blend(mask: float32x4_t, if_true: float32x4_t, if_false: float32x4_t) -> float32x4_t {
        unsafe { vbslq_f32(vreinterpretq_u32_f32(mask), if_true, if_false) }
    }

    #[inline(always)]
    fn reduce_add(a: float32x4_t) -> f32 {
        unsafe {
            let pair = vpaddq_f32(a, a);
            let pair = vpaddq_f32(pair, pair);
            vgetq_lane_f32::<0>(pair)
        }
    }

    #[inline(always)]
    fn reduce_min(a: float32x4_t) -> f32 {
        unsafe {
            let pair = vpminq_f32(a, a);
            let pair = vpminq_f32(pair, pair);
            vgetq_lane_f32::<0>(pair)
        }
    }

    #[inline(always)]
    fn reduce_max(a: float32x4_t) -> f32 {
        unsafe {
            let pair = vpmaxq_f32(a, a);
            let pair = vpmaxq_f32(pair, pair);
            vgetq_lane_f32::<0>(pair)
        }
    }

    #[inline(always)]
    fn rcp_approx(a: float32x4_t) -> float32x4_t {
        unsafe { vrecpeq_f32(a) }
    }
    #[inline(always)]
    fn rsqrt_approx(a: float32x4_t) -> float32x4_t {
        unsafe { vrsqrteq_f32(a) }
    }

    #[inline(always)]
    fn not(a: float32x4_t) -> float32x4_t {
        unsafe { vreinterpretq_f32_u32(vmvnq_u32(vreinterpretq_u32_f32(a))) }
    }
    #[inline(always)]
    fn bitand(a: float32x4_t, b: float32x4_t) -> float32x4_t {
        unsafe {
            vreinterpretq_f32_u32(vandq_u32(
                vreinterpretq_u32_f32(a),
                vreinterpretq_u32_f32(b),
            ))
        }
    }
    #[inline(always)]
    fn bitor(a: float32x4_t, b: float32x4_t) -> float32x4_t {
        unsafe {
            vreinterpretq_f32_u32(vorrq_u32(
                vreinterpretq_u32_f32(a),
                vreinterpretq_u32_f32(b),
            ))
        }
    }
    #[inline(always)]
    fn bitxor(a: float32x4_t, b: float32x4_t) -> float32x4_t {
        unsafe {
            vreinterpretq_f32_u32(veorq_u32(
                vreinterpretq_u32_f32(a),
                vreinterpretq_u32_f32(b),
            ))
        }
    }
}

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
            [
                vld1q_f32(data.as_ptr().add(0)),
                vld1q_f32(data.as_ptr().add(4)),
            ]
        }
    }

    #[inline(always)]
    fn from_array(arr: [f32; 8]) -> [float32x4_t; 2] {
        Self::load(&arr)
    }

    #[inline(always)]
    fn store(repr: [float32x4_t; 2], out: &mut [f32; 8]) {
        unsafe {
            vst1q_f32(out.as_mut_ptr().add(0), repr[0]);
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
        // vfmaq = acc + x*y, so mul_add(a, b, c) = a*b + c => vfmaq(c, a, b)
        unsafe { [vfmaq_f32(c[0], a[0], b[0]), vfmaq_f32(c[1], a[1], b[1])] }
    }

    #[inline(always)]
    fn mul_sub(a: [float32x4_t; 2], b: [float32x4_t; 2], c: [float32x4_t; 2]) -> [float32x4_t; 2] {
        // a*b - c => vfmaq(-c, a, b) = -c + a*b
        unsafe {
            [
                vfmaq_f32(vnegq_f32(c[0]), a[0], b[0]),
                vfmaq_f32(vnegq_f32(c[1]), a[1], b[1]),
            ]
        }
    }

    // ====== Comparisons ======

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
            let m = vaddq_f32(a[0], a[1]);
            let pair = vpaddq_f32(m, m);
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

#[cfg(target_arch = "aarch64")]
impl F64x2Backend for archmage::NeonToken {
    type Repr = float64x2_t;

    #[inline(always)]
    fn splat(v: f64) -> float64x2_t {
        unsafe { vdupq_n_f64(v) }
    }

    #[inline(always)]
    fn zero() -> float64x2_t {
        unsafe { vdupq_n_f64(0.0) }
    }

    #[inline(always)]
    fn load(data: &[f64; 2]) -> float64x2_t {
        unsafe { vld1q_f64(data.as_ptr()) }
    }

    #[inline(always)]
    fn from_array(arr: [f64; 2]) -> float64x2_t {
        Self::load(&arr)
    }

    #[inline(always)]
    fn store(repr: float64x2_t, out: &mut [f64; 2]) {
        unsafe { vst1q_f64(out.as_mut_ptr(), repr) };
    }

    #[inline(always)]
    fn to_array(repr: float64x2_t) -> [f64; 2] {
        let mut out = [0.0f64; 2];
        Self::store(repr, &mut out);
        out
    }

    #[inline(always)]
    fn add(a: float64x2_t, b: float64x2_t) -> float64x2_t {
        unsafe { vaddq_f64(a, b) }
    }
    #[inline(always)]
    fn sub(a: float64x2_t, b: float64x2_t) -> float64x2_t {
        unsafe { vsubq_f64(a, b) }
    }
    #[inline(always)]
    fn mul(a: float64x2_t, b: float64x2_t) -> float64x2_t {
        unsafe { vmulq_f64(a, b) }
    }
    #[inline(always)]
    fn div(a: float64x2_t, b: float64x2_t) -> float64x2_t {
        unsafe { vdivq_f64(a, b) }
    }
    #[inline(always)]
    fn neg(a: float64x2_t) -> float64x2_t {
        unsafe { vnegq_f64(a) }
    }
    #[inline(always)]
    fn min(a: float64x2_t, b: float64x2_t) -> float64x2_t {
        unsafe { vminq_f64(a, b) }
    }
    #[inline(always)]
    fn max(a: float64x2_t, b: float64x2_t) -> float64x2_t {
        unsafe { vmaxq_f64(a, b) }
    }
    #[inline(always)]
    fn sqrt(a: float64x2_t) -> float64x2_t {
        unsafe { vsqrtq_f64(a) }
    }
    #[inline(always)]
    fn abs(a: float64x2_t) -> float64x2_t {
        unsafe { vabsq_f64(a) }
    }
    #[inline(always)]
    fn floor(a: float64x2_t) -> float64x2_t {
        unsafe { vrndmq_f64(a) }
    }
    #[inline(always)]
    fn ceil(a: float64x2_t) -> float64x2_t {
        unsafe { vrndpq_f64(a) }
    }
    #[inline(always)]
    fn round(a: float64x2_t) -> float64x2_t {
        unsafe { vrndnq_f64(a) }
    }

    #[inline(always)]
    fn mul_add(a: float64x2_t, b: float64x2_t, c: float64x2_t) -> float64x2_t {
        unsafe { vfmaq_f64(c, a, b) }
    }

    #[inline(always)]
    fn mul_sub(a: float64x2_t, b: float64x2_t, c: float64x2_t) -> float64x2_t {
        unsafe { vfmaq_f64(vnegq_f64(c), a, b) }
    }

    #[inline(always)]
    fn simd_eq(a: float64x2_t, b: float64x2_t) -> float64x2_t {
        unsafe { vreinterpretq_f64_u64(vceqq_f64(a, b)) }
    }
    #[inline(always)]
    fn simd_ne(a: float64x2_t, b: float64x2_t) -> float64x2_t {
        unsafe { vreinterpretq_f64_u64(vmvnq_u64(vceqq_f64(a, b))) }
    }
    #[inline(always)]
    fn simd_lt(a: float64x2_t, b: float64x2_t) -> float64x2_t {
        unsafe { vreinterpretq_f64_u64(vcltq_f64(a, b)) }
    }
    #[inline(always)]
    fn simd_le(a: float64x2_t, b: float64x2_t) -> float64x2_t {
        unsafe { vreinterpretq_f64_u64(vcleq_f64(a, b)) }
    }
    #[inline(always)]
    fn simd_gt(a: float64x2_t, b: float64x2_t) -> float64x2_t {
        unsafe { vreinterpretq_f64_u64(vcgtq_f64(a, b)) }
    }
    #[inline(always)]
    fn simd_ge(a: float64x2_t, b: float64x2_t) -> float64x2_t {
        unsafe { vreinterpretq_f64_u64(vcgeq_f64(a, b)) }
    }

    #[inline(always)]
    fn blend(mask: float64x2_t, if_true: float64x2_t, if_false: float64x2_t) -> float64x2_t {
        unsafe { vbslq_f64(vreinterpretq_u64_f64(mask), if_true, if_false) }
    }

    #[inline(always)]
    fn reduce_add(a: float64x2_t) -> f64 {
        unsafe {
            let pair = vpaddq_f64(a, a);
            vgetq_lane_f64::<0>(pair)
        }
    }

    #[inline(always)]
    fn reduce_min(a: float64x2_t) -> f64 {
        unsafe {
            let pair = vpminq_f64(a, a);
            vgetq_lane_f64::<0>(pair)
        }
    }

    #[inline(always)]
    fn reduce_max(a: float64x2_t) -> f64 {
        unsafe {
            let pair = vpmaxq_f64(a, a);
            vgetq_lane_f64::<0>(pair)
        }
    }

    #[inline(always)]
    fn rcp_approx(a: float64x2_t) -> float64x2_t {
        unsafe { vrecpeq_f64(a) }
    }
    #[inline(always)]
    fn rsqrt_approx(a: float64x2_t) -> float64x2_t {
        unsafe { vrsqrteq_f64(a) }
    }

    #[inline(always)]
    fn not(a: float64x2_t) -> float64x2_t {
        unsafe { vreinterpretq_f64_u64(vmvnq_u64(vreinterpretq_u64_f64(a))) }
    }
    #[inline(always)]
    fn bitand(a: float64x2_t, b: float64x2_t) -> float64x2_t {
        unsafe {
            vreinterpretq_f64_u64(vandq_u64(
                vreinterpretq_u64_f64(a),
                vreinterpretq_u64_f64(b),
            ))
        }
    }
    #[inline(always)]
    fn bitor(a: float64x2_t, b: float64x2_t) -> float64x2_t {
        unsafe {
            vreinterpretq_f64_u64(vorrq_u64(
                vreinterpretq_u64_f64(a),
                vreinterpretq_u64_f64(b),
            ))
        }
    }
    #[inline(always)]
    fn bitxor(a: float64x2_t, b: float64x2_t) -> float64x2_t {
        unsafe {
            vreinterpretq_f64_u64(veorq_u64(
                vreinterpretq_u64_f64(a),
                vreinterpretq_u64_f64(b),
            ))
        }
    }
}

#[cfg(target_arch = "aarch64")]
impl F64x4Backend for archmage::NeonToken {
    type Repr = [float64x2_t; 2];

    // ====== Construction ======

    #[inline(always)]
    fn splat(v: f64) -> [float64x2_t; 2] {
        unsafe {
            let v4 = vdupq_n_f64(v);
            [v4, v4]
        }
    }

    #[inline(always)]
    fn zero() -> [float64x2_t; 2] {
        unsafe {
            let z = vdupq_n_f64(0.0);
            [z, z]
        }
    }

    #[inline(always)]
    fn load(data: &[f64; 4]) -> [float64x2_t; 2] {
        unsafe {
            [
                vld1q_f64(data.as_ptr().add(0)),
                vld1q_f64(data.as_ptr().add(2)),
            ]
        }
    }

    #[inline(always)]
    fn from_array(arr: [f64; 4]) -> [float64x2_t; 2] {
        Self::load(&arr)
    }

    #[inline(always)]
    fn store(repr: [float64x2_t; 2], out: &mut [f64; 4]) {
        unsafe {
            vst1q_f64(out.as_mut_ptr().add(0), repr[0]);
            vst1q_f64(out.as_mut_ptr().add(2), repr[1]);
        }
    }

    #[inline(always)]
    fn to_array(repr: [float64x2_t; 2]) -> [f64; 4] {
        let mut out = [0.0f64; 4];
        Self::store(repr, &mut out);
        out
    }

    // ====== Arithmetic ======

    #[inline(always)]
    fn add(a: [float64x2_t; 2], b: [float64x2_t; 2]) -> [float64x2_t; 2] {
        unsafe { [vaddq_f64(a[0], b[0]), vaddq_f64(a[1], b[1])] }
    }

    #[inline(always)]
    fn sub(a: [float64x2_t; 2], b: [float64x2_t; 2]) -> [float64x2_t; 2] {
        unsafe { [vsubq_f64(a[0], b[0]), vsubq_f64(a[1], b[1])] }
    }

    #[inline(always)]
    fn mul(a: [float64x2_t; 2], b: [float64x2_t; 2]) -> [float64x2_t; 2] {
        unsafe { [vmulq_f64(a[0], b[0]), vmulq_f64(a[1], b[1])] }
    }

    #[inline(always)]
    fn div(a: [float64x2_t; 2], b: [float64x2_t; 2]) -> [float64x2_t; 2] {
        unsafe { [vdivq_f64(a[0], b[0]), vdivq_f64(a[1], b[1])] }
    }

    #[inline(always)]
    fn neg(a: [float64x2_t; 2]) -> [float64x2_t; 2] {
        unsafe { [vnegq_f64(a[0]), vnegq_f64(a[1])] }
    }

    // ====== Math ======

    #[inline(always)]
    fn min(a: [float64x2_t; 2], b: [float64x2_t; 2]) -> [float64x2_t; 2] {
        unsafe { [vminq_f64(a[0], b[0]), vminq_f64(a[1], b[1])] }
    }

    #[inline(always)]
    fn max(a: [float64x2_t; 2], b: [float64x2_t; 2]) -> [float64x2_t; 2] {
        unsafe { [vmaxq_f64(a[0], b[0]), vmaxq_f64(a[1], b[1])] }
    }

    #[inline(always)]
    fn sqrt(a: [float64x2_t; 2]) -> [float64x2_t; 2] {
        unsafe { [vsqrtq_f64(a[0]), vsqrtq_f64(a[1])] }
    }

    #[inline(always)]
    fn abs(a: [float64x2_t; 2]) -> [float64x2_t; 2] {
        unsafe { [vabsq_f64(a[0]), vabsq_f64(a[1])] }
    }

    #[inline(always)]
    fn floor(a: [float64x2_t; 2]) -> [float64x2_t; 2] {
        unsafe { [vrndmq_f64(a[0]), vrndmq_f64(a[1])] }
    }

    #[inline(always)]
    fn ceil(a: [float64x2_t; 2]) -> [float64x2_t; 2] {
        unsafe { [vrndpq_f64(a[0]), vrndpq_f64(a[1])] }
    }

    #[inline(always)]
    fn round(a: [float64x2_t; 2]) -> [float64x2_t; 2] {
        unsafe { [vrndnq_f64(a[0]), vrndnq_f64(a[1])] }
    }

    #[inline(always)]
    fn mul_add(a: [float64x2_t; 2], b: [float64x2_t; 2], c: [float64x2_t; 2]) -> [float64x2_t; 2] {
        // vfmaq = acc + x*y, so mul_add(a, b, c) = a*b + c => vfmaq(c, a, b)
        unsafe { [vfmaq_f64(c[0], a[0], b[0]), vfmaq_f64(c[1], a[1], b[1])] }
    }

    #[inline(always)]
    fn mul_sub(a: [float64x2_t; 2], b: [float64x2_t; 2], c: [float64x2_t; 2]) -> [float64x2_t; 2] {
        // a*b - c => vfmaq(-c, a, b) = -c + a*b
        unsafe {
            [
                vfmaq_f64(vnegq_f64(c[0]), a[0], b[0]),
                vfmaq_f64(vnegq_f64(c[1]), a[1], b[1]),
            ]
        }
    }

    // ====== Comparisons ======

    #[inline(always)]
    fn simd_eq(a: [float64x2_t; 2], b: [float64x2_t; 2]) -> [float64x2_t; 2] {
        unsafe {
            [
                vreinterpretq_f64_u64(vceqq_f64(a[0], b[0])),
                vreinterpretq_f64_u64(vceqq_f64(a[1], b[1])),
            ]
        }
    }

    #[inline(always)]
    fn simd_ne(a: [float64x2_t; 2], b: [float64x2_t; 2]) -> [float64x2_t; 2] {
        unsafe {
            [
                vreinterpretq_f64_u64(vmvnq_u64(vceqq_f64(a[0], b[0]))),
                vreinterpretq_f64_u64(vmvnq_u64(vceqq_f64(a[1], b[1]))),
            ]
        }
    }

    #[inline(always)]
    fn simd_lt(a: [float64x2_t; 2], b: [float64x2_t; 2]) -> [float64x2_t; 2] {
        unsafe {
            [
                vreinterpretq_f64_u64(vcltq_f64(a[0], b[0])),
                vreinterpretq_f64_u64(vcltq_f64(a[1], b[1])),
            ]
        }
    }

    #[inline(always)]
    fn simd_le(a: [float64x2_t; 2], b: [float64x2_t; 2]) -> [float64x2_t; 2] {
        unsafe {
            [
                vreinterpretq_f64_u64(vcleq_f64(a[0], b[0])),
                vreinterpretq_f64_u64(vcleq_f64(a[1], b[1])),
            ]
        }
    }

    #[inline(always)]
    fn simd_gt(a: [float64x2_t; 2], b: [float64x2_t; 2]) -> [float64x2_t; 2] {
        unsafe {
            [
                vreinterpretq_f64_u64(vcgtq_f64(a[0], b[0])),
                vreinterpretq_f64_u64(vcgtq_f64(a[1], b[1])),
            ]
        }
    }

    #[inline(always)]
    fn simd_ge(a: [float64x2_t; 2], b: [float64x2_t; 2]) -> [float64x2_t; 2] {
        unsafe {
            [
                vreinterpretq_f64_u64(vcgeq_f64(a[0], b[0])),
                vreinterpretq_f64_u64(vcgeq_f64(a[1], b[1])),
            ]
        }
    }

    #[inline(always)]
    fn blend(
        mask: [float64x2_t; 2],
        if_true: [float64x2_t; 2],
        if_false: [float64x2_t; 2],
    ) -> [float64x2_t; 2] {
        unsafe {
            [
                vbslq_f64(vreinterpretq_u64_f64(mask[0]), if_true[0], if_false[0]),
                vbslq_f64(vreinterpretq_u64_f64(mask[1]), if_true[1], if_false[1]),
            ]
        }
    }

    // ====== Reductions ======

    #[inline(always)]
    fn reduce_add(a: [float64x2_t; 2]) -> f64 {
        unsafe {
            let m = vaddq_f64(a[0], a[1]);
            let pair = vpaddq_f64(m, m);
            vgetq_lane_f64::<0>(pair)
        }
    }

    #[inline(always)]
    fn reduce_min(a: [float64x2_t; 2]) -> f64 {
        unsafe {
            let m = vminq_f64(a[0], a[1]);
            let pair = vpminq_f64(m, m);
            vgetq_lane_f64::<0>(pair)
        }
    }

    #[inline(always)]
    fn reduce_max(a: [float64x2_t; 2]) -> f64 {
        unsafe {
            let m = vmaxq_f64(a[0], a[1]);
            let pair = vpmaxq_f64(m, m);
            vgetq_lane_f64::<0>(pair)
        }
    }

    // ====== Approximations ======

    #[inline(always)]
    fn rcp_approx(a: [float64x2_t; 2]) -> [float64x2_t; 2] {
        unsafe { [vrecpeq_f64(a[0]), vrecpeq_f64(a[1])] }
    }

    #[inline(always)]
    fn rsqrt_approx(a: [float64x2_t; 2]) -> [float64x2_t; 2] {
        unsafe { [vrsqrteq_f64(a[0]), vrsqrteq_f64(a[1])] }
    }

    // ====== Bitwise ======

    #[inline(always)]
    fn not(a: [float64x2_t; 2]) -> [float64x2_t; 2] {
        unsafe {
            [
                vreinterpretq_f64_u64(vmvnq_u64(vreinterpretq_u64_f64(a[0]))),
                vreinterpretq_f64_u64(vmvnq_u64(vreinterpretq_u64_f64(a[1]))),
            ]
        }
    }

    #[inline(always)]
    fn bitand(a: [float64x2_t; 2], b: [float64x2_t; 2]) -> [float64x2_t; 2] {
        unsafe {
            [
                vreinterpretq_f64_u64(vandq_u32(
                    vreinterpretq_u64_f64(a[0]),
                    vreinterpretq_u64_f64(b[0]),
                )),
                vreinterpretq_f64_u64(vandq_u32(
                    vreinterpretq_u64_f64(a[1]),
                    vreinterpretq_u64_f64(b[1]),
                )),
            ]
        }
    }

    #[inline(always)]
    fn bitor(a: [float64x2_t; 2], b: [float64x2_t; 2]) -> [float64x2_t; 2] {
        unsafe {
            [
                vreinterpretq_f64_u64(vorrq_u32(
                    vreinterpretq_u64_f64(a[0]),
                    vreinterpretq_u64_f64(b[0]),
                )),
                vreinterpretq_f64_u64(vorrq_u32(
                    vreinterpretq_u64_f64(a[1]),
                    vreinterpretq_u64_f64(b[1]),
                )),
            ]
        }
    }

    #[inline(always)]
    fn bitxor(a: [float64x2_t; 2], b: [float64x2_t; 2]) -> [float64x2_t; 2] {
        unsafe {
            [
                vreinterpretq_f64_u64(veorq_u32(
                    vreinterpretq_u64_f64(a[0]),
                    vreinterpretq_u64_f64(b[0]),
                )),
                vreinterpretq_f64_u64(veorq_u32(
                    vreinterpretq_u64_f64(a[1]),
                    vreinterpretq_u64_f64(b[1]),
                )),
            ]
        }
    }
}

#[cfg(target_arch = "aarch64")]
impl I32x4Backend for archmage::NeonToken {
    type Repr = int32x4_t;

    #[inline(always)]
    fn splat(v: i32) -> int32x4_t {
        unsafe { vdupq_n_s32(v) }
    }

    #[inline(always)]
    fn zero() -> int32x4_t {
        unsafe { vdupq_n_s32(0) }
    }

    #[inline(always)]
    fn load(data: &[i32; 4]) -> int32x4_t {
        unsafe { vld1q_s32(data.as_ptr()) }
    }

    #[inline(always)]
    fn from_array(arr: [i32; 4]) -> int32x4_t {
        unsafe { vld1q_s32(arr.as_ptr()) }
    }

    #[inline(always)]
    fn store(repr: int32x4_t, out: &mut [i32; 4]) {
        unsafe { vst1q_s32(out.as_mut_ptr(), repr) };
    }

    #[inline(always)]
    fn to_array(repr: int32x4_t) -> [i32; 4] {
        let mut out = [0i32; 4];
        unsafe { vst1q_s32(out.as_mut_ptr(), repr) };
        out
    }

    #[inline(always)]
    fn add(a: int32x4_t, b: int32x4_t) -> int32x4_t {
        unsafe { vaddq_s32(a, b) }
    }
    #[inline(always)]
    fn sub(a: int32x4_t, b: int32x4_t) -> int32x4_t {
        unsafe { vsubq_s32(a, b) }
    }
    #[inline(always)]
    fn mul(a: int32x4_t, b: int32x4_t) -> int32x4_t {
        unsafe { vmulq_s32(a, b) }
    }
    #[inline(always)]
    fn neg(a: int32x4_t) -> int32x4_t {
        unsafe { vnegq_s32(a) }
    }
    #[inline(always)]
    fn min(a: int32x4_t, b: int32x4_t) -> int32x4_t {
        unsafe { vminq_s32(a, b) }
    }
    #[inline(always)]
    fn max(a: int32x4_t, b: int32x4_t) -> int32x4_t {
        unsafe { vmaxq_s32(a, b) }
    }
    #[inline(always)]
    fn abs(a: int32x4_t) -> int32x4_t {
        unsafe { vabsq_s32(a) }
    }

    #[inline(always)]
    fn simd_eq(a: int32x4_t, b: int32x4_t) -> int32x4_t {
        unsafe { vreinterpretq_s32_u32(vceqq_s32(a, b)) }
    }
    #[inline(always)]
    fn simd_ne(a: int32x4_t, b: int32x4_t) -> int32x4_t {
        unsafe { vreinterpretq_s32_u32(vmvnq_u32(vceqq_s32(a, b))) }
    }
    #[inline(always)]
    fn simd_lt(a: int32x4_t, b: int32x4_t) -> int32x4_t {
        unsafe { vreinterpretq_s32_u32(vcltq_s32(a, b)) }
    }
    #[inline(always)]
    fn simd_le(a: int32x4_t, b: int32x4_t) -> int32x4_t {
        unsafe { vreinterpretq_s32_u32(vcleq_s32(a, b)) }
    }
    #[inline(always)]
    fn simd_gt(a: int32x4_t, b: int32x4_t) -> int32x4_t {
        unsafe { vreinterpretq_s32_u32(vcgtq_s32(a, b)) }
    }
    #[inline(always)]
    fn simd_ge(a: int32x4_t, b: int32x4_t) -> int32x4_t {
        unsafe { vreinterpretq_s32_u32(vcgeq_s32(a, b)) }
    }

    #[inline(always)]
    fn blend(mask: int32x4_t, if_true: int32x4_t, if_false: int32x4_t) -> int32x4_t {
        unsafe { vbslq_s32(vreinterpretq_u32_s32(mask), if_true, if_false) }
    }

    #[inline(always)]
    fn reduce_add(a: int32x4_t) -> i32 {
        unsafe { vaddvq_s32(a) }
    }

    #[inline(always)]
    fn not(a: int32x4_t) -> int32x4_t {
        unsafe { vmvnq_s32(a) }
    }
    #[inline(always)]
    fn bitand(a: int32x4_t, b: int32x4_t) -> int32x4_t {
        unsafe { vandq_s32(a, b) }
    }
    #[inline(always)]
    fn bitor(a: int32x4_t, b: int32x4_t) -> int32x4_t {
        unsafe { vorrq_s32(a, b) }
    }
    #[inline(always)]
    fn bitxor(a: int32x4_t, b: int32x4_t) -> int32x4_t {
        unsafe { veorq_s32(a, b) }
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: int32x4_t) -> int32x4_t {
        unsafe { vshlq_n_s32::<N>(a) }
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: int32x4_t) -> int32x4_t {
        unsafe { vshrq_n_s32::<N>(a) }
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: int32x4_t) -> int32x4_t {
        unsafe { vreinterpretq_s32_u32(vshrq_n_u32::<N>(vreinterpretq_u32_s32(a))) }
    }

    #[inline(always)]
    fn all_true(a: int32x4_t) -> bool {
        unsafe { vminvq_u32(vreinterpretq_u32_s32(a)) != 0 }
    }

    #[inline(always)]
    fn any_true(a: int32x4_t) -> bool {
        unsafe { vmaxvq_u32(vreinterpretq_u32_s32(a)) != 0 }
    }

    #[inline(always)]
    fn bitmask(a: int32x4_t) -> u32 {
        unsafe {
            // Extract sign bit of each 32-bit lane
            let shift = vreinterpretq_u32_s32(vshrq_n_s32::<31>(a));
            // Pack: lane0 | (lane1<<1) | (lane2<<2) | (lane3<<3)
            let lane0 = vgetq_lane_u32::<0>(shift);
            let lane1 = vgetq_lane_u32::<1>(shift);
            let lane2 = vgetq_lane_u32::<2>(shift);
            let lane3 = vgetq_lane_u32::<3>(shift);
            lane0 | (lane1 << 1) | (lane2 << 2) | (lane3 << 3)
        }
    }
}

#[cfg(target_arch = "aarch64")]
impl I32x8Backend for archmage::NeonToken {
    type Repr = [int32x4_t; 2];

    #[inline(always)]
    fn splat(v: i32) -> [int32x4_t; 2] {
        unsafe {
            let v4 = vdupq_n_s32(v);
            [v4, v4]
        }
    }

    #[inline(always)]
    fn zero() -> [int32x4_t; 2] {
        unsafe {
            let z = vdupq_n_s32(0);
            [z, z]
        }
    }

    #[inline(always)]
    fn load(data: &[i32; 8]) -> [int32x4_t; 2] {
        unsafe {
            [
                vld1q_s32(data.as_ptr().add(0)),
                vld1q_s32(data.as_ptr().add(4)),
            ]
        }
    }

    #[inline(always)]
    fn from_array(arr: [i32; 8]) -> [int32x4_t; 2] {
        Self::load(&arr)
    }

    #[inline(always)]
    fn store(repr: [int32x4_t; 2], out: &mut [i32; 8]) {
        unsafe {
            vst1q_s32(out.as_mut_ptr().add(0), repr[0]);
            vst1q_s32(out.as_mut_ptr().add(4), repr[1]);
        }
    }

    #[inline(always)]
    fn to_array(repr: [int32x4_t; 2]) -> [i32; 8] {
        let mut out = [0i32; 8];
        Self::store(repr, &mut out);
        out
    }

    #[inline(always)]
    fn add(a: [int32x4_t; 2], b: [int32x4_t; 2]) -> [int32x4_t; 2] {
        unsafe { [vaddq_s32(a[0], b[0]), vaddq_s32(a[1], b[1])] }
    }
    #[inline(always)]
    fn sub(a: [int32x4_t; 2], b: [int32x4_t; 2]) -> [int32x4_t; 2] {
        unsafe { [vsubq_s32(a[0], b[0]), vsubq_s32(a[1], b[1])] }
    }
    #[inline(always)]
    fn mul(a: [int32x4_t; 2], b: [int32x4_t; 2]) -> [int32x4_t; 2] {
        unsafe { [vmulq_s32(a[0], b[0]), vmulq_s32(a[1], b[1])] }
    }
    #[inline(always)]
    fn neg(a: [int32x4_t; 2]) -> [int32x4_t; 2] {
        unsafe { [vnegq_s32(a[0]), vnegq_s32(a[1])] }
    }
    #[inline(always)]
    fn min(a: [int32x4_t; 2], b: [int32x4_t; 2]) -> [int32x4_t; 2] {
        unsafe { [vminq_s32(a[0], b[0]), vminq_s32(a[1], b[1])] }
    }
    #[inline(always)]
    fn max(a: [int32x4_t; 2], b: [int32x4_t; 2]) -> [int32x4_t; 2] {
        unsafe { [vmaxq_s32(a[0], b[0]), vmaxq_s32(a[1], b[1])] }
    }
    #[inline(always)]
    fn abs(a: [int32x4_t; 2]) -> [int32x4_t; 2] {
        unsafe { [vabsq_s32(a[0]), vabsq_s32(a[1])] }
    }

    #[inline(always)]
    fn simd_eq(a: [int32x4_t; 2], b: [int32x4_t; 2]) -> [int32x4_t; 2] {
        unsafe {
            [
                vreinterpretq_s32_u32(vceqq_s32(a[0], b[0])),
                vreinterpretq_s32_u32(vceqq_s32(a[1], b[1])),
            ]
        }
    }
    #[inline(always)]
    fn simd_ne(a: [int32x4_t; 2], b: [int32x4_t; 2]) -> [int32x4_t; 2] {
        unsafe {
            [
                vreinterpretq_s32_u32(vmvnq_u32(vceqq_s32(a[0], b[0]))),
                vreinterpretq_s32_u32(vmvnq_u32(vceqq_s32(a[1], b[1]))),
            ]
        }
    }
    #[inline(always)]
    fn simd_lt(a: [int32x4_t; 2], b: [int32x4_t; 2]) -> [int32x4_t; 2] {
        unsafe {
            [
                vreinterpretq_s32_u32(vcltq_s32(a[0], b[0])),
                vreinterpretq_s32_u32(vcltq_s32(a[1], b[1])),
            ]
        }
    }
    #[inline(always)]
    fn simd_le(a: [int32x4_t; 2], b: [int32x4_t; 2]) -> [int32x4_t; 2] {
        unsafe {
            [
                vreinterpretq_s32_u32(vcleq_s32(a[0], b[0])),
                vreinterpretq_s32_u32(vcleq_s32(a[1], b[1])),
            ]
        }
    }
    #[inline(always)]
    fn simd_gt(a: [int32x4_t; 2], b: [int32x4_t; 2]) -> [int32x4_t; 2] {
        unsafe {
            [
                vreinterpretq_s32_u32(vcgtq_s32(a[0], b[0])),
                vreinterpretq_s32_u32(vcgtq_s32(a[1], b[1])),
            ]
        }
    }
    #[inline(always)]
    fn simd_ge(a: [int32x4_t; 2], b: [int32x4_t; 2]) -> [int32x4_t; 2] {
        unsafe {
            [
                vreinterpretq_s32_u32(vcgeq_s32(a[0], b[0])),
                vreinterpretq_s32_u32(vcgeq_s32(a[1], b[1])),
            ]
        }
    }

    #[inline(always)]
    fn blend(
        mask: [int32x4_t; 2],
        if_true: [int32x4_t; 2],
        if_false: [int32x4_t; 2],
    ) -> [int32x4_t; 2] {
        unsafe {
            [
                vbslq_s32(vreinterpretq_u32_s32(mask[0]), if_true[0], if_false[0]),
                vbslq_s32(vreinterpretq_u32_s32(mask[1]), if_true[1], if_false[1]),
            ]
        }
    }

    #[inline(always)]
    fn reduce_add(a: [int32x4_t; 2]) -> i32 {
        unsafe {
            let m = vaddq_s32(a[0], a[1]);
            vaddvq_s32(m)
        }
    }

    #[inline(always)]
    fn not(a: [int32x4_t; 2]) -> [int32x4_t; 2] {
        unsafe { [vmvnq_s32(a[0]), vmvnq_s32(a[1])] }
    }
    #[inline(always)]
    fn bitand(a: [int32x4_t; 2], b: [int32x4_t; 2]) -> [int32x4_t; 2] {
        unsafe { [vandq_s32(a[0], b[0]), vandq_s32(a[1], b[1])] }
    }
    #[inline(always)]
    fn bitor(a: [int32x4_t; 2], b: [int32x4_t; 2]) -> [int32x4_t; 2] {
        unsafe { [vorrq_s32(a[0], b[0]), vorrq_s32(a[1], b[1])] }
    }
    #[inline(always)]
    fn bitxor(a: [int32x4_t; 2], b: [int32x4_t; 2]) -> [int32x4_t; 2] {
        unsafe { [veorq_s32(a[0], b[0]), veorq_s32(a[1], b[1])] }
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: [int32x4_t; 2]) -> [int32x4_t; 2] {
        unsafe { [vshlq_n_s32::<N>(a[0]), vshlq_n_s32::<N>(a[1])] }
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: [int32x4_t; 2]) -> [int32x4_t; 2] {
        unsafe { [vshrq_n_s32::<N>(a[0]), vshrq_n_s32::<N>(a[1])] }
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: [int32x4_t; 2]) -> [int32x4_t; 2] {
        unsafe {
            [
                vreinterpretq_s32_u32(vshrq_n_u32::<N>(vreinterpretq_u32_s32(a[0]))),
                vreinterpretq_s32_u32(vshrq_n_u32::<N>(vreinterpretq_u32_s32(a[1]))),
            ]
        }
    }

    #[inline(always)]
    fn all_true(a: [int32x4_t; 2]) -> bool {
        unsafe {
            vminvq_u32(vreinterpretq_u32_s32(a[0])) != 0
                && vminvq_u32(vreinterpretq_u32_s32(a[1])) != 0
        }
    }

    #[inline(always)]
    fn any_true(a: [int32x4_t; 2]) -> bool {
        unsafe {
            vmaxvq_u32(vreinterpretq_u32_s32(a[0])) != 0
                || vmaxvq_u32(vreinterpretq_u32_s32(a[1])) != 0
        }
    }

    #[inline(always)]
    fn bitmask(a: [int32x4_t; 2]) -> u32 {
        unsafe {
            let mut bits = 0u32;
            let s0 = vreinterpretq_u32_s32(vshrq_n_s32::<31>(a[0]));
            bits |= vgetq_lane_u32::<0>(s0) << 0;
            bits |= (vgetq_lane_u32::<1>(s0)) << 1;
            bits |= (vgetq_lane_u32::<2>(s0)) << 2;
            bits |= (vgetq_lane_u32::<3>(s0)) << 3;
            let s1 = vreinterpretq_u32_s32(vshrq_n_s32::<31>(a[1]));
            bits |= vgetq_lane_u32::<0>(s1) << 4;
            bits |= (vgetq_lane_u32::<1>(s1)) << 5;
            bits |= (vgetq_lane_u32::<2>(s1)) << 6;
            bits |= (vgetq_lane_u32::<3>(s1)) << 7;
            bits
        }
    }
}

#[cfg(target_arch = "aarch64")]
impl F32x4Convert for archmage::NeonToken {
    #[inline(always)]
    fn bitcast_f32_to_i32(a: float32x4_t) -> int32x4_t {
        unsafe { vreinterpretq_s32_f32(a) }
    }

    #[inline(always)]
    fn bitcast_i32_to_f32(a: int32x4_t) -> float32x4_t {
        unsafe { vreinterpretq_f32_s32(a) }
    }

    #[inline(always)]
    fn convert_f32_to_i32(a: float32x4_t) -> int32x4_t {
        unsafe { vcvtq_s32_f32(a) }
    }

    #[inline(always)]
    fn convert_f32_to_i32_round(a: float32x4_t) -> int32x4_t {
        unsafe { vcvtnq_s32_f32(a) }
    }

    #[inline(always)]
    fn convert_i32_to_f32(a: int32x4_t) -> float32x4_t {
        unsafe { vcvtq_f32_s32(a) }
    }
}

#[cfg(target_arch = "aarch64")]
impl F32x8Convert for archmage::NeonToken {
    #[inline(always)]
    fn bitcast_f32_to_i32(a: [float32x4_t; 2]) -> [int32x4_t; 2] {
        unsafe { [vreinterpretq_s32_f32(a[0]), vreinterpretq_s32_f32(a[1])] }
    }

    #[inline(always)]
    fn bitcast_i32_to_f32(a: [int32x4_t; 2]) -> [float32x4_t; 2] {
        unsafe { [vreinterpretq_f32_s32(a[0]), vreinterpretq_f32_s32(a[1])] }
    }

    #[inline(always)]
    fn convert_f32_to_i32(a: [float32x4_t; 2]) -> [int32x4_t; 2] {
        unsafe { [vcvtq_s32_f32(a[0]), vcvtq_s32_f32(a[1])] }
    }

    #[inline(always)]
    fn convert_f32_to_i32_round(a: [float32x4_t; 2]) -> [int32x4_t; 2] {
        unsafe { [vcvtnq_s32_f32(a[0]), vcvtnq_s32_f32(a[1])] }
    }

    #[inline(always)]
    fn convert_i32_to_f32(a: [int32x4_t; 2]) -> [float32x4_t; 2] {
        unsafe { [vcvtq_f32_s32(a[0]), vcvtq_f32_s32(a[1])] }
    }
}

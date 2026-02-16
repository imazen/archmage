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
impl U32x4Backend for archmage::NeonToken {
    type Repr = uint32x4_t;

    #[inline(always)]
    fn splat(v: u32) -> uint32x4_t {
        unsafe { vdupq_n_u32(v) }
    }

    #[inline(always)]
    fn zero() -> uint32x4_t {
        unsafe { vdupq_n_u32(0) }
    }

    #[inline(always)]
    fn load(data: &[u32; 4]) -> uint32x4_t {
        unsafe { vld1q_u32(data.as_ptr()) }
    }

    #[inline(always)]
    fn from_array(arr: [u32; 4]) -> uint32x4_t {
        unsafe { vld1q_u32(arr.as_ptr()) }
    }

    #[inline(always)]
    fn store(repr: uint32x4_t, out: &mut [u32; 4]) {
        unsafe { vst1q_u32(out.as_mut_ptr(), repr) };
    }

    #[inline(always)]
    fn to_array(repr: uint32x4_t) -> [u32; 4] {
        let mut out = [0u32; 4];
        unsafe { vst1q_u32(out.as_mut_ptr(), repr) };
        out
    }

    #[inline(always)]
    fn add(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        unsafe { vaddq_u32(a, b) }
    }
    #[inline(always)]
    fn sub(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        unsafe { vsubq_u32(a, b) }
    }
    #[inline(always)]
    fn mul(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        unsafe { vmulq_u32(a, b) }
    }
    #[inline(always)]
    fn min(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        unsafe { vminq_u32(a, b) }
    }
    #[inline(always)]
    fn max(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        unsafe { vmaxq_u32(a, b) }
    }

    #[inline(always)]
    fn simd_eq(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        unsafe { vceqq_u32(a, b) }
    }
    #[inline(always)]
    fn simd_ne(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        unsafe { vmvnq_u32(vceqq_u32(a, b)) }
    }
    #[inline(always)]
    fn simd_lt(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        unsafe { vcltq_u32(a, b) }
    }
    #[inline(always)]
    fn simd_le(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        unsafe { vcleq_u32(a, b) }
    }
    #[inline(always)]
    fn simd_gt(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        unsafe { vcgtq_u32(a, b) }
    }
    #[inline(always)]
    fn simd_ge(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        unsafe { vcgeq_u32(a, b) }
    }

    #[inline(always)]
    fn blend(mask: uint32x4_t, if_true: uint32x4_t, if_false: uint32x4_t) -> uint32x4_t {
        unsafe { vbslq_u32(mask, if_true, if_false) }
    }

    #[inline(always)]
    fn reduce_add(a: uint32x4_t) -> u32 {
        unsafe { vaddvq_u32(a) }
    }

    #[inline(always)]
    fn not(a: uint32x4_t) -> uint32x4_t {
        unsafe { vmvnq_u32(a) }
    }
    #[inline(always)]
    fn bitand(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        unsafe { vandq_u32(a, b) }
    }
    #[inline(always)]
    fn bitor(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        unsafe { vorrq_u32(a, b) }
    }
    #[inline(always)]
    fn bitxor(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        unsafe { veorq_u32(a, b) }
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: uint32x4_t) -> uint32x4_t {
        unsafe { vshlq_n_u32::<N>(a) }
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: uint32x4_t) -> uint32x4_t {
        unsafe { vshrq_n_u32::<N>(a) }
    }

    #[inline(always)]
    fn all_true(a: uint32x4_t) -> bool {
        unsafe { vminvq_u32(a) == u32::MAX }
    }

    #[inline(always)]
    fn any_true(a: uint32x4_t) -> bool {
        unsafe { vmaxvq_u32(a) != 0 }
    }

    #[inline(always)]
    fn bitmask(a: uint32x4_t) -> u32 {
        unsafe {
            // Extract sign bit of each 32-bit lane
            let shift = vshrq_n_u32::<31>(a);
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
impl U32x8Backend for archmage::NeonToken {
    type Repr = [uint32x4_t; 2];

    #[inline(always)]
    fn splat(v: u32) -> [uint32x4_t; 2] {
        unsafe {
            let v4 = vdupq_n_u32(v);
            [v4, v4]
        }
    }

    #[inline(always)]
    fn zero() -> [uint32x4_t; 2] {
        unsafe {
            let z = vdupq_n_u32(0);
            [z, z]
        }
    }

    #[inline(always)]
    fn load(data: &[u32; 8]) -> [uint32x4_t; 2] {
        unsafe {
            [
                vld1q_u32(data.as_ptr().add(0)),
                vld1q_u32(data.as_ptr().add(4)),
            ]
        }
    }

    #[inline(always)]
    fn from_array(arr: [u32; 8]) -> [uint32x4_t; 2] {
        Self::load(&arr)
    }

    #[inline(always)]
    fn store(repr: [uint32x4_t; 2], out: &mut [u32; 8]) {
        unsafe {
            vst1q_u32(out.as_mut_ptr().add(0), repr[0]);
            vst1q_u32(out.as_mut_ptr().add(4), repr[1]);
        }
    }

    #[inline(always)]
    fn to_array(repr: [uint32x4_t; 2]) -> [u32; 8] {
        let mut out = [0u32; 8];
        Self::store(repr, &mut out);
        out
    }

    #[inline(always)]
    fn add(a: [uint32x4_t; 2], b: [uint32x4_t; 2]) -> [uint32x4_t; 2] {
        unsafe { [vaddq_u32(a[0], b[0]), vaddq_u32(a[1], b[1])] }
    }
    #[inline(always)]
    fn sub(a: [uint32x4_t; 2], b: [uint32x4_t; 2]) -> [uint32x4_t; 2] {
        unsafe { [vsubq_u32(a[0], b[0]), vsubq_u32(a[1], b[1])] }
    }
    #[inline(always)]
    fn mul(a: [uint32x4_t; 2], b: [uint32x4_t; 2]) -> [uint32x4_t; 2] {
        unsafe { [vmulq_u32(a[0], b[0]), vmulq_u32(a[1], b[1])] }
    }
    #[inline(always)]
    fn min(a: [uint32x4_t; 2], b: [uint32x4_t; 2]) -> [uint32x4_t; 2] {
        unsafe { [vminq_u32(a[0], b[0]), vminq_u32(a[1], b[1])] }
    }
    #[inline(always)]
    fn max(a: [uint32x4_t; 2], b: [uint32x4_t; 2]) -> [uint32x4_t; 2] {
        unsafe { [vmaxq_u32(a[0], b[0]), vmaxq_u32(a[1], b[1])] }
    }

    #[inline(always)]
    fn simd_eq(a: [uint32x4_t; 2], b: [uint32x4_t; 2]) -> [uint32x4_t; 2] {
        unsafe { [vceqq_u32(a[0], b[0]), vceqq_u32(a[1], b[1])] }
    }
    #[inline(always)]
    fn simd_ne(a: [uint32x4_t; 2], b: [uint32x4_t; 2]) -> [uint32x4_t; 2] {
        unsafe {
            [
                vmvnq_u32(vceqq_u32(a[0], b[0])),
                vmvnq_u32(vceqq_u32(a[1], b[1])),
            ]
        }
    }
    #[inline(always)]
    fn simd_lt(a: [uint32x4_t; 2], b: [uint32x4_t; 2]) -> [uint32x4_t; 2] {
        unsafe { [vcltq_u32(a[0], b[0]), vcltq_u32(a[1], b[1])] }
    }
    #[inline(always)]
    fn simd_le(a: [uint32x4_t; 2], b: [uint32x4_t; 2]) -> [uint32x4_t; 2] {
        unsafe { [vcleq_u32(a[0], b[0]), vcleq_u32(a[1], b[1])] }
    }
    #[inline(always)]
    fn simd_gt(a: [uint32x4_t; 2], b: [uint32x4_t; 2]) -> [uint32x4_t; 2] {
        unsafe { [vcgtq_u32(a[0], b[0]), vcgtq_u32(a[1], b[1])] }
    }
    #[inline(always)]
    fn simd_ge(a: [uint32x4_t; 2], b: [uint32x4_t; 2]) -> [uint32x4_t; 2] {
        unsafe { [vcgeq_u32(a[0], b[0]), vcgeq_u32(a[1], b[1])] }
    }

    #[inline(always)]
    fn blend(
        mask: [uint32x4_t; 2],
        if_true: [uint32x4_t; 2],
        if_false: [uint32x4_t; 2],
    ) -> [uint32x4_t; 2] {
        unsafe {
            [
                vbslq_u32(mask[0], if_true[0], if_false[0]),
                vbslq_u32(mask[1], if_true[1], if_false[1]),
            ]
        }
    }

    #[inline(always)]
    fn reduce_add(a: [uint32x4_t; 2]) -> u32 {
        unsafe {
            let m = vaddq_u32(a[0], a[1]);
            vaddvq_u32(m)
        }
    }

    #[inline(always)]
    fn not(a: [uint32x4_t; 2]) -> [uint32x4_t; 2] {
        unsafe { [vmvnq_u32(a[0]), vmvnq_u32(a[1])] }
    }
    #[inline(always)]
    fn bitand(a: [uint32x4_t; 2], b: [uint32x4_t; 2]) -> [uint32x4_t; 2] {
        unsafe { [vandq_u32(a[0], b[0]), vandq_u32(a[1], b[1])] }
    }
    #[inline(always)]
    fn bitor(a: [uint32x4_t; 2], b: [uint32x4_t; 2]) -> [uint32x4_t; 2] {
        unsafe { [vorrq_u32(a[0], b[0]), vorrq_u32(a[1], b[1])] }
    }
    #[inline(always)]
    fn bitxor(a: [uint32x4_t; 2], b: [uint32x4_t; 2]) -> [uint32x4_t; 2] {
        unsafe { [veorq_u32(a[0], b[0]), veorq_u32(a[1], b[1])] }
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: [uint32x4_t; 2]) -> [uint32x4_t; 2] {
        unsafe { [vshlq_n_u32::<N>(a[0]), vshlq_n_u32::<N>(a[1])] }
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: [uint32x4_t; 2]) -> [uint32x4_t; 2] {
        unsafe { [vshrq_n_u32::<N>(a[0]), vshrq_n_u32::<N>(a[1])] }
    }

    #[inline(always)]
    fn all_true(a: [uint32x4_t; 2]) -> bool {
        unsafe { vminvq_u32(a[0]) == u32::MAX && vminvq_u32(a[1]) == u32::MAX }
    }

    #[inline(always)]
    fn any_true(a: [uint32x4_t; 2]) -> bool {
        unsafe { vmaxvq_u32(a[0]) != 0 || vmaxvq_u32(a[1]) != 0 }
    }

    #[inline(always)]
    fn bitmask(a: [uint32x4_t; 2]) -> u32 {
        unsafe {
            let mut bits = 0u32;
            let s0 = vshrq_n_u32::<31>(a[0]);
            bits |= vgetq_lane_u32::<0>(s0) << 0;
            bits |= (vgetq_lane_u32::<1>(s0)) << 1;
            bits |= (vgetq_lane_u32::<2>(s0)) << 2;
            bits |= (vgetq_lane_u32::<3>(s0)) << 3;
            let s1 = vshrq_n_u32::<31>(a[1]);
            bits |= vgetq_lane_u32::<0>(s1) << 4;
            bits |= (vgetq_lane_u32::<1>(s1)) << 5;
            bits |= (vgetq_lane_u32::<2>(s1)) << 6;
            bits |= (vgetq_lane_u32::<3>(s1)) << 7;
            bits
        }
    }
}

#[cfg(target_arch = "aarch64")]
impl I64x2Backend for archmage::NeonToken {
    type Repr = int64x2_t;

    #[inline(always)]
    fn splat(v: i64) -> int64x2_t {
        unsafe { vdupq_n_s64(v) }
    }

    #[inline(always)]
    fn zero() -> int64x2_t {
        unsafe { vdupq_n_s64(0i64) }
    }

    #[inline(always)]
    fn load(data: &[i64; 2]) -> int64x2_t {
        unsafe { vld1q_s64(data.as_ptr()) }
    }

    #[inline(always)]
    fn from_array(arr: [i64; 2]) -> int64x2_t {
        unsafe { vld1q_s64(arr.as_ptr()) }
    }

    #[inline(always)]
    fn store(repr: int64x2_t, out: &mut [i64; 2]) {
        unsafe { vst1q_s64(out.as_mut_ptr(), repr) };
    }

    #[inline(always)]
    fn to_array(repr: int64x2_t) -> [i64; 2] {
        let mut out = [0i64; 2];
        unsafe { vst1q_s64(out.as_mut_ptr(), repr) };
        out
    }

    #[inline(always)]
    fn add(a: int64x2_t, b: int64x2_t) -> int64x2_t {
        unsafe { vaddq_s64(a, b) }
    }
    #[inline(always)]
    fn sub(a: int64x2_t, b: int64x2_t) -> int64x2_t {
        unsafe { vsubq_s64(a, b) }
    }
    #[inline(always)]
    fn neg(a: int64x2_t) -> int64x2_t {
        unsafe { vnegq_s64(a) }
    }
    #[inline(always)]
    fn min(a: int64x2_t, b: int64x2_t) -> int64x2_t {
        // NEON lacks native i64 min; polyfill via compare+select
        unsafe {
            let mask = vcltq_s64(a, b);
            vbslq_s64(mask, a, b)
        }
    }
    #[inline(always)]
    fn max(a: int64x2_t, b: int64x2_t) -> int64x2_t {
        // NEON lacks native i64 max; polyfill via compare+select
        unsafe {
            let mask = vcgtq_s64(a, b);
            vbslq_s64(mask, a, b)
        }
    }
    #[inline(always)]
    fn abs(a: int64x2_t) -> int64x2_t {
        unsafe { vabsq_s64(a) }
    }

    #[inline(always)]
    fn simd_eq(a: int64x2_t, b: int64x2_t) -> int64x2_t {
        unsafe { vreinterpretq_s64_u64(vceqq_s64(a, b)) }
    }
    #[inline(always)]
    fn simd_ne(a: int64x2_t, b: int64x2_t) -> int64x2_t {
        unsafe {
            let eq = vceqq_s64(a, b);
            // NOT via XOR with all-ones
            vreinterpretq_s64_u64(veorq_u64(eq, vdupq_n_u64(u64::MAX)))
        }
    }
    #[inline(always)]
    fn simd_lt(a: int64x2_t, b: int64x2_t) -> int64x2_t {
        unsafe { vreinterpretq_s64_u64(vcltq_s64(a, b)) }
    }
    #[inline(always)]
    fn simd_le(a: int64x2_t, b: int64x2_t) -> int64x2_t {
        unsafe { vreinterpretq_s64_u64(vcleq_s64(a, b)) }
    }
    #[inline(always)]
    fn simd_gt(a: int64x2_t, b: int64x2_t) -> int64x2_t {
        unsafe { vreinterpretq_s64_u64(vcgtq_s64(a, b)) }
    }
    #[inline(always)]
    fn simd_ge(a: int64x2_t, b: int64x2_t) -> int64x2_t {
        unsafe { vreinterpretq_s64_u64(vcgeq_s64(a, b)) }
    }

    #[inline(always)]
    fn blend(mask: int64x2_t, if_true: int64x2_t, if_false: int64x2_t) -> int64x2_t {
        unsafe { vbslq_s64(vreinterpretq_u64_s64(mask), if_true, if_false) }
    }

    #[inline(always)]
    fn reduce_add(a: int64x2_t) -> i64 {
        unsafe {
            let sum = vpaddq_s64(a, a);
            vgetq_lane_s64::<0>(sum)
        }
    }

    #[inline(always)]
    fn not(a: int64x2_t) -> int64x2_t {
        unsafe {
            let ones = vdupq_n_s64(-1i64);
            veorq_s64(a, ones)
        }
    }
    #[inline(always)]
    fn bitand(a: int64x2_t, b: int64x2_t) -> int64x2_t {
        unsafe { vandq_s64(a, b) }
    }
    #[inline(always)]
    fn bitor(a: int64x2_t, b: int64x2_t) -> int64x2_t {
        unsafe { vorrq_s64(a, b) }
    }
    #[inline(always)]
    fn bitxor(a: int64x2_t, b: int64x2_t) -> int64x2_t {
        unsafe { veorq_s64(a, b) }
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: int64x2_t) -> int64x2_t {
        unsafe { vshlq_n_s64::<N>(a) }
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: int64x2_t) -> int64x2_t {
        unsafe { vshrq_n_s64::<N>(a) }
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: int64x2_t) -> int64x2_t {
        unsafe { vreinterpretq_s64_u64(vshrq_n_u64::<N>(vreinterpretq_u64_s64(a))) }
    }

    #[inline(always)]
    fn all_true(a: int64x2_t) -> bool {
        unsafe {
            let as_u64 = vreinterpretq_u64_s64(a);
            vgetq_lane_u64::<0>(as_u64) != 0 && vgetq_lane_u64::<1>(as_u64) != 0
        }
    }

    #[inline(always)]
    fn any_true(a: int64x2_t) -> bool {
        unsafe {
            let as_u64 = vreinterpretq_u64_s64(a);
            (vgetq_lane_u64::<0>(as_u64) | vgetq_lane_u64::<1>(as_u64)) != 0
        }
    }

    #[inline(always)]
    fn bitmask(a: int64x2_t) -> u32 {
        unsafe {
            let signs = vshrq_n_u64::<63>(vreinterpretq_u64_s64(a));
            ((vgetq_lane_u64::<0>(signs) & 1) | ((vgetq_lane_u64::<1>(signs) & 1) << 1)) as u32
        }
    }
}

#[cfg(target_arch = "aarch64")]
impl I64x4Backend for archmage::NeonToken {
    type Repr = [int64x2_t; 2];

    #[inline(always)]
    fn splat(v: i64) -> [int64x2_t; 2] {
        unsafe {
            let v2 = vdupq_n_s64(v);
            [v2, v2]
        }
    }

    #[inline(always)]
    fn zero() -> [int64x2_t; 2] {
        unsafe {
            let z = vdupq_n_s64(0i64);
            [z, z]
        }
    }

    #[inline(always)]
    fn load(data: &[i64; 4]) -> [int64x2_t; 2] {
        unsafe {
            [
                vld1q_s64(data.as_ptr().add(0)),
                vld1q_s64(data.as_ptr().add(2)),
            ]
        }
    }

    #[inline(always)]
    fn from_array(arr: [i64; 4]) -> [int64x2_t; 2] {
        Self::load(&arr)
    }

    #[inline(always)]
    fn store(repr: [int64x2_t; 2], out: &mut [i64; 4]) {
        unsafe {
            vst1q_s64(out.as_mut_ptr().add(0), repr[0]);
            vst1q_s64(out.as_mut_ptr().add(2), repr[1]);
        }
    }

    #[inline(always)]
    fn to_array(repr: [int64x2_t; 2]) -> [i64; 4] {
        let mut out = [0i64; 4];
        Self::store(repr, &mut out);
        out
    }

    #[inline(always)]
    fn add(a: [int64x2_t; 2], b: [int64x2_t; 2]) -> [int64x2_t; 2] {
        unsafe { [vaddq_s64(a[0], b[0]), vaddq_s64(a[1], b[1])] }
    }
    #[inline(always)]
    fn sub(a: [int64x2_t; 2], b: [int64x2_t; 2]) -> [int64x2_t; 2] {
        unsafe { [vsubq_s64(a[0], b[0]), vsubq_s64(a[1], b[1])] }
    }
    #[inline(always)]
    fn neg(a: [int64x2_t; 2]) -> [int64x2_t; 2] {
        unsafe { [vnegq_s64(a[0]), vnegq_s64(a[1])] }
    }
    #[inline(always)]
    fn min(a: [int64x2_t; 2], b: [int64x2_t; 2]) -> [int64x2_t; 2] {
        // NEON lacks native i64 min; polyfill via compare+select per sub-vector
        unsafe {
            [
                vbslq_s64(vcltq_s64(a[0], b[0]), a[0], b[0]),
                vbslq_s64(vcltq_s64(a[1], b[1]), a[1], b[1]),
            ]
        }
    }
    #[inline(always)]
    fn max(a: [int64x2_t; 2], b: [int64x2_t; 2]) -> [int64x2_t; 2] {
        // NEON lacks native i64 max; polyfill via compare+select per sub-vector
        unsafe {
            [
                vbslq_s64(vcgtq_s64(a[0], b[0]), a[0], b[0]),
                vbslq_s64(vcgtq_s64(a[1], b[1]), a[1], b[1]),
            ]
        }
    }
    #[inline(always)]
    fn abs(a: [int64x2_t; 2]) -> [int64x2_t; 2] {
        unsafe { [vabsq_s64(a[0]), vabsq_s64(a[1])] }
    }

    #[inline(always)]
    fn simd_eq(a: [int64x2_t; 2], b: [int64x2_t; 2]) -> [int64x2_t; 2] {
        unsafe {
            [
                vreinterpretq_s64_u64(vceqq_s64(a[0], b[0])),
                vreinterpretq_s64_u64(vceqq_s64(a[1], b[1])),
            ]
        }
    }
    #[inline(always)]
    fn simd_ne(a: [int64x2_t; 2], b: [int64x2_t; 2]) -> [int64x2_t; 2] {
        unsafe {
            [
                vreinterpretq_s64_u64(veorq_u64(vceqq_s64(a[0], b[0]), vdupq_n_u64(u64::MAX))),
                vreinterpretq_s64_u64(veorq_u64(vceqq_s64(a[1], b[1]), vdupq_n_u64(u64::MAX))),
            ]
        }
    }
    #[inline(always)]
    fn simd_lt(a: [int64x2_t; 2], b: [int64x2_t; 2]) -> [int64x2_t; 2] {
        unsafe {
            [
                vreinterpretq_s64_u64(vcltq_s64(a[0], b[0])),
                vreinterpretq_s64_u64(vcltq_s64(a[1], b[1])),
            ]
        }
    }
    #[inline(always)]
    fn simd_le(a: [int64x2_t; 2], b: [int64x2_t; 2]) -> [int64x2_t; 2] {
        unsafe {
            [
                vreinterpretq_s64_u64(vcleq_s64(a[0], b[0])),
                vreinterpretq_s64_u64(vcleq_s64(a[1], b[1])),
            ]
        }
    }
    #[inline(always)]
    fn simd_gt(a: [int64x2_t; 2], b: [int64x2_t; 2]) -> [int64x2_t; 2] {
        unsafe {
            [
                vreinterpretq_s64_u64(vcgtq_s64(a[0], b[0])),
                vreinterpretq_s64_u64(vcgtq_s64(a[1], b[1])),
            ]
        }
    }
    #[inline(always)]
    fn simd_ge(a: [int64x2_t; 2], b: [int64x2_t; 2]) -> [int64x2_t; 2] {
        unsafe {
            [
                vreinterpretq_s64_u64(vcgeq_s64(a[0], b[0])),
                vreinterpretq_s64_u64(vcgeq_s64(a[1], b[1])),
            ]
        }
    }

    #[inline(always)]
    fn blend(
        mask: [int64x2_t; 2],
        if_true: [int64x2_t; 2],
        if_false: [int64x2_t; 2],
    ) -> [int64x2_t; 2] {
        unsafe {
            [
                vbslq_s64(vreinterpretq_u64_s64(mask[0]), if_true[0], if_false[0]),
                vbslq_s64(vreinterpretq_u64_s64(mask[1]), if_true[1], if_false[1]),
            ]
        }
    }

    #[inline(always)]
    fn reduce_add(a: [int64x2_t; 2]) -> i64 {
        unsafe {
            let m = vaddq_s64(a[0], a[1]);
            let sum = vpaddq_s64(m, m);
            vgetq_lane_s64::<0>(sum)
        }
    }

    #[inline(always)]
    fn not(a: [int64x2_t; 2]) -> [int64x2_t; 2] {
        unsafe {
            [
                veorq_s64(a[0], vdupq_n_s64(-1i64)),
                veorq_s64(a[1], vdupq_n_s64(-1i64)),
            ]
        }
    }
    #[inline(always)]
    fn bitand(a: [int64x2_t; 2], b: [int64x2_t; 2]) -> [int64x2_t; 2] {
        unsafe { [vandq_s64(a[0], b[0]), vandq_s64(a[1], b[1])] }
    }
    #[inline(always)]
    fn bitor(a: [int64x2_t; 2], b: [int64x2_t; 2]) -> [int64x2_t; 2] {
        unsafe { [vorrq_s64(a[0], b[0]), vorrq_s64(a[1], b[1])] }
    }
    #[inline(always)]
    fn bitxor(a: [int64x2_t; 2], b: [int64x2_t; 2]) -> [int64x2_t; 2] {
        unsafe { [veorq_s64(a[0], b[0]), veorq_s64(a[1], b[1])] }
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: [int64x2_t; 2]) -> [int64x2_t; 2] {
        unsafe { [vshlq_n_s64::<N>(a[0]), vshlq_n_s64::<N>(a[1])] }
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: [int64x2_t; 2]) -> [int64x2_t; 2] {
        unsafe { [vshrq_n_s64::<N>(a[0]), vshrq_n_s64::<N>(a[1])] }
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: [int64x2_t; 2]) -> [int64x2_t; 2] {
        unsafe {
            [
                vreinterpretq_s64_u64(vshrq_n_u64::<N>(vreinterpretq_u64_s64(a[0]))),
                vreinterpretq_s64_u64(vshrq_n_u64::<N>(vreinterpretq_u64_s64(a[1]))),
            ]
        }
    }

    #[inline(always)]
    fn all_true(a: [int64x2_t; 2]) -> bool {
        unsafe {
            (vgetq_lane_u64::<0>(vreinterpretq_u64_s64(a[0])) != 0
                && vgetq_lane_u64::<1>(vreinterpretq_u64_s64(a[0])) != 0)
                && (vgetq_lane_u64::<0>(vreinterpretq_u64_s64(a[1])) != 0
                    && vgetq_lane_u64::<1>(vreinterpretq_u64_s64(a[1])) != 0)
        }
    }

    #[inline(always)]
    fn any_true(a: [int64x2_t; 2]) -> bool {
        unsafe {
            ((vgetq_lane_u64::<0>(vreinterpretq_u64_s64(a[0]))
                | vgetq_lane_u64::<1>(vreinterpretq_u64_s64(a[0])))
                != 0)
                || ((vgetq_lane_u64::<0>(vreinterpretq_u64_s64(a[1]))
                    | vgetq_lane_u64::<1>(vreinterpretq_u64_s64(a[1])))
                    != 0)
        }
    }

    #[inline(always)]
    fn bitmask(a: [int64x2_t; 2]) -> u32 {
        unsafe {
            let mut bits = 0u32;
            let s0 = vshrq_n_u64::<63>(vreinterpretq_u64_s64(a[0]));
            bits |= (vgetq_lane_u64::<0>(s0) as u32) << 0;
            bits |= (vgetq_lane_u64::<1>(s0) as u32) << 1;
            let s1 = vshrq_n_u64::<63>(vreinterpretq_u64_s64(a[1]));
            bits |= (vgetq_lane_u64::<0>(s1) as u32) << 2;
            bits |= (vgetq_lane_u64::<1>(s1) as u32) << 3;
            bits
        }
    }
}

#[cfg(target_arch = "aarch64")]
impl I8x16Backend for archmage::NeonToken {
    type Repr = int8x16_t;

    #[inline(always)]
    fn splat(v: i8) -> int8x16_t {
        unsafe { vdupq_n_s8(v) }
    }

    #[inline(always)]
    fn zero() -> int8x16_t {
        unsafe { vdupq_n_s8(0) }
    }

    #[inline(always)]
    fn load(data: &[i8; 16]) -> int8x16_t {
        unsafe { vld1q_s8(data.as_ptr()) }
    }

    #[inline(always)]
    fn from_array(arr: [i8; 16]) -> int8x16_t {
        unsafe { vld1q_s8(arr.as_ptr()) }
    }

    #[inline(always)]
    fn store(repr: int8x16_t, out: &mut [i8; 16]) {
        unsafe { vst1q_s8(out.as_mut_ptr(), repr) };
    }

    #[inline(always)]
    fn to_array(repr: int8x16_t) -> [i8; 16] {
        let mut out = [0i8; 16];
        unsafe { vst1q_s8(out.as_mut_ptr(), repr) };
        out
    }

    #[inline(always)]
    fn add(a: int8x16_t, b: int8x16_t) -> int8x16_t {
        unsafe { vaddq_s8(a, b) }
    }
    #[inline(always)]
    fn sub(a: int8x16_t, b: int8x16_t) -> int8x16_t {
        unsafe { vsubq_s8(a, b) }
    }
    #[inline(always)]
    fn neg(a: int8x16_t) -> int8x16_t {
        unsafe { vnegq_s8(a) }
    }
    #[inline(always)]
    fn min(a: int8x16_t, b: int8x16_t) -> int8x16_t {
        unsafe { vminq_s8(a, b) }
    }
    #[inline(always)]
    fn max(a: int8x16_t, b: int8x16_t) -> int8x16_t {
        unsafe { vmaxq_s8(a, b) }
    }
    #[inline(always)]
    fn abs(a: int8x16_t) -> int8x16_t {
        unsafe { vabsq_s8(a) }
    }

    #[inline(always)]
    fn simd_eq(a: int8x16_t, b: int8x16_t) -> int8x16_t {
        unsafe { vreinterpretq_s8_8(vceqq_s8(a, b)) }
    }
    #[inline(always)]
    fn simd_ne(a: int8x16_t, b: int8x16_t) -> int8x16_t {
        unsafe { vreinterpretq_s8_8(vmvnq_8(vceqq_s8(a, b))) }
    }
    #[inline(always)]
    fn simd_lt(a: int8x16_t, b: int8x16_t) -> int8x16_t {
        unsafe { vreinterpretq_s8_8(vcltq_s8(a, b)) }
    }
    #[inline(always)]
    fn simd_le(a: int8x16_t, b: int8x16_t) -> int8x16_t {
        unsafe { vreinterpretq_s8_8(vcleq_s8(a, b)) }
    }
    #[inline(always)]
    fn simd_gt(a: int8x16_t, b: int8x16_t) -> int8x16_t {
        unsafe { vreinterpretq_s8_8(vcgtq_s8(a, b)) }
    }
    #[inline(always)]
    fn simd_ge(a: int8x16_t, b: int8x16_t) -> int8x16_t {
        unsafe { vreinterpretq_s8_8(vcgeq_s8(a, b)) }
    }

    #[inline(always)]
    fn blend(mask: int8x16_t, if_true: int8x16_t, if_false: int8x16_t) -> int8x16_t {
        unsafe { vbslq_s8(vreinterpretq_8_s8(mask), if_true, if_false) }
    }
    #[inline(always)]
    fn reduce_add(a: int8x16_t) -> i8 {
        unsafe { vaddvq_s8(a) }
    }
    #[inline(always)]
    fn not(a: int8x16_t) -> int8x16_t {
        unsafe { vmvnq_s8(a) }
    }
    #[inline(always)]
    fn bitand(a: int8x16_t, b: int8x16_t) -> int8x16_t {
        unsafe { vandq_s8(a, b) }
    }
    #[inline(always)]
    fn bitor(a: int8x16_t, b: int8x16_t) -> int8x16_t {
        unsafe { vorrq_s8(a, b) }
    }
    #[inline(always)]
    fn bitxor(a: int8x16_t, b: int8x16_t) -> int8x16_t {
        unsafe { veorq_s8(a, b) }
    }
    #[inline(always)]
    fn shl_const<const N: i32>(a: int8x16_t) -> int8x16_t {
        unsafe { vshlq_n_s8::<N>(a) }
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: int8x16_t) -> int8x16_t {
        unsafe { vreinterpretq_s8_u8(vshrq_n_u8::<N>(vreinterpretq_u8_s8(a))) }
    }
    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: int8x16_t) -> int8x16_t {
        unsafe { vshrq_n_s8::<N>(a) }
    }

    #[inline(always)]
    fn all_true(a: int8x16_t) -> bool {
        unsafe { vminvq_8(vreinterpretq_8_s8(a)) != 0 }
    }

    #[inline(always)]
    fn any_true(a: int8x16_t) -> bool {
        unsafe { vmaxvq_8(vreinterpretq_8_s8(a)) != 0 }
    }

    #[inline(always)]
    fn bitmask(a: int8x16_t) -> u32 {
        unsafe {
            // Shift each byte right by 7 to isolate sign bit
            let bits = vshrq_n_s8::<7>(a);
            // Use polynomial evaluation to pack bits
            // Each byte is now 0 or 1, multiply by position powers of 2
            let powers: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
            let pow_vec = vld1q_u8(powers.as_ptr());
            let weighted = vmulq_u8(vreinterpretq_u8_s8(bits), pow_vec);
            // Sum pairs: add adjacent bytes
            let pair_sum = vpaddlq_u8(weighted);
            let quad_sum = vpaddlq_u16(pair_sum);
            let oct_sum = vpaddlq_u32(quad_sum);
            // Extract low and high byte
            let lo = vgetq_lane_u64::<0>(oct_sum) as u32;
            let hi = vgetq_lane_u64::<1>(oct_sum) as u32;
            lo | (hi << 8)
        }
    }
}

#[cfg(target_arch = "aarch64")]
impl I8x32Backend for archmage::NeonToken {
    type Repr = [int8x16_t; 2];

    #[inline(always)]
    fn splat(v: i8) -> [int8x16_t; 2] {
        unsafe {
            let v4 = vdupq_n_s8(v);
            [v4, v4]
        }
    }

    #[inline(always)]
    fn zero() -> [int8x16_t; 2] {
        unsafe {
            let z = vdupq_n_s8(0);
            [z, z]
        }
    }

    #[inline(always)]
    fn load(data: &[i8; 32]) -> [int8x16_t; 2] {
        unsafe {
            [
                vld1q_s8(data.as_ptr().add(0)),
                vld1q_s8(data.as_ptr().add(16)),
            ]
        }
    }

    #[inline(always)]
    fn from_array(arr: [i8; 32]) -> [int8x16_t; 2] {
        Self::load(&arr)
    }

    #[inline(always)]
    fn store(repr: [int8x16_t; 2], out: &mut [i8; 32]) {
        unsafe {
            vst1q_s8(out.as_mut_ptr().add(0), repr[0]);
            vst1q_s8(out.as_mut_ptr().add(16), repr[1]);
        }
    }

    #[inline(always)]
    fn to_array(repr: [int8x16_t; 2]) -> [i8; 32] {
        let mut out = [0i8; 32];
        Self::store(repr, &mut out);
        out
    }

    #[inline(always)]
    fn add(a: [int8x16_t; 2], b: [int8x16_t; 2]) -> [int8x16_t; 2] {
        unsafe { [vaddq_s8(a[0], b[0]), vaddq_s8(a[1], b[1])] }
    }
    #[inline(always)]
    fn sub(a: [int8x16_t; 2], b: [int8x16_t; 2]) -> [int8x16_t; 2] {
        unsafe { [vsubq_s8(a[0], b[0]), vsubq_s8(a[1], b[1])] }
    }
    #[inline(always)]
    fn neg(a: [int8x16_t; 2]) -> [int8x16_t; 2] {
        unsafe { [vnegq_s8(a[0]), vnegq_s8(a[1])] }
    }
    #[inline(always)]
    fn min(a: [int8x16_t; 2], b: [int8x16_t; 2]) -> [int8x16_t; 2] {
        unsafe { [vminq_s8(a[0], b[0]), vminq_s8(a[1], b[1])] }
    }
    #[inline(always)]
    fn max(a: [int8x16_t; 2], b: [int8x16_t; 2]) -> [int8x16_t; 2] {
        unsafe { [vmaxq_s8(a[0], b[0]), vmaxq_s8(a[1], b[1])] }
    }
    #[inline(always)]
    fn abs(a: [int8x16_t; 2]) -> [int8x16_t; 2] {
        unsafe { [vabsq_s8(a[0]), vabsq_s8(a[1])] }
    }

    #[inline(always)]
    fn simd_eq(a: [int8x16_t; 2], b: [int8x16_t; 2]) -> [int8x16_t; 2] {
        unsafe {
            [
                vreinterpretq_s8_u8(vceqq_s8(a[0], b[0])),
                vreinterpretq_s8_u8(vceqq_s8(a[1], b[1])),
            ]
        }
    }
    #[inline(always)]
    fn simd_ne(a: [int8x16_t; 2], b: [int8x16_t; 2]) -> [int8x16_t; 2] {
        unsafe {
            [
                vreinterpretq_s8_u8(vmvnq_u8(vceqq_s8(a[0], b[0]))),
                vreinterpretq_s8_u8(vmvnq_u8(vceqq_s8(a[1], b[1]))),
            ]
        }
    }
    #[inline(always)]
    fn simd_lt(a: [int8x16_t; 2], b: [int8x16_t; 2]) -> [int8x16_t; 2] {
        unsafe {
            [
                vreinterpretq_s8_u8(vcltq_s8(a[0], b[0])),
                vreinterpretq_s8_u8(vcltq_s8(a[1], b[1])),
            ]
        }
    }
    #[inline(always)]
    fn simd_le(a: [int8x16_t; 2], b: [int8x16_t; 2]) -> [int8x16_t; 2] {
        unsafe {
            [
                vreinterpretq_s8_u8(vcleq_s8(a[0], b[0])),
                vreinterpretq_s8_u8(vcleq_s8(a[1], b[1])),
            ]
        }
    }
    #[inline(always)]
    fn simd_gt(a: [int8x16_t; 2], b: [int8x16_t; 2]) -> [int8x16_t; 2] {
        unsafe {
            [
                vreinterpretq_s8_u8(vcgtq_s8(a[0], b[0])),
                vreinterpretq_s8_u8(vcgtq_s8(a[1], b[1])),
            ]
        }
    }
    #[inline(always)]
    fn simd_ge(a: [int8x16_t; 2], b: [int8x16_t; 2]) -> [int8x16_t; 2] {
        unsafe {
            [
                vreinterpretq_s8_u8(vcgeq_s8(a[0], b[0])),
                vreinterpretq_s8_u8(vcgeq_s8(a[1], b[1])),
            ]
        }
    }

    #[inline(always)]
    fn blend(
        mask: [int8x16_t; 2],
        if_true: [int8x16_t; 2],
        if_false: [int8x16_t; 2],
    ) -> [int8x16_t; 2] {
        unsafe {
            [
                vbslq_s8(vreinterpretq_u8_s8(mask[0]), if_true[0], if_false[0]),
                vbslq_s8(vreinterpretq_u8_s8(mask[1]), if_true[1], if_false[1]),
            ]
        }
    }

    #[inline(always)]
    fn reduce_add(a: [int8x16_t; 2]) -> i8 {
        let mut sum = 0i8;
        for i in 0..2 {
            sum = sum.wrapping_add(unsafe { vaddvq_s8(a[i]) });
        }
        sum
    }

    #[inline(always)]
    fn not(a: [int8x16_t; 2]) -> [int8x16_t; 2] {
        unsafe { [vmvnq_s8(a[0]), vmvnq_s8(a[1])] }
    }
    #[inline(always)]
    fn bitand(a: [int8x16_t; 2], b: [int8x16_t; 2]) -> [int8x16_t; 2] {
        unsafe { [vandq_s8(a[0], b[0]), vandq_s8(a[1], b[1])] }
    }
    #[inline(always)]
    fn bitor(a: [int8x16_t; 2], b: [int8x16_t; 2]) -> [int8x16_t; 2] {
        unsafe { [vorrq_s8(a[0], b[0]), vorrq_s8(a[1], b[1])] }
    }
    #[inline(always)]
    fn bitxor(a: [int8x16_t; 2], b: [int8x16_t; 2]) -> [int8x16_t; 2] {
        unsafe { [veorq_s8(a[0], b[0]), veorq_s8(a[1], b[1])] }
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: [int8x16_t; 2]) -> [int8x16_t; 2] {
        unsafe { [vshlq_n_s8::<N>(a[0]), vshlq_n_s8::<N>(a[1])] }
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: [int8x16_t; 2]) -> [int8x16_t; 2] {
        unsafe {
            [
                vreinterpretq_s8_u8(vshrq_n_u8::<N>(vreinterpretq_u8_s8(a[0]))),
                vreinterpretq_s8_u8(vshrq_n_u8::<N>(vreinterpretq_u8_s8(a[1]))),
            ]
        }
    }
    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: [int8x16_t; 2]) -> [int8x16_t; 2] {
        unsafe { [vshrq_n_s8::<N>(a[0]), vshrq_n_s8::<N>(a[1])] }
    }

    #[inline(always)]
    fn all_true(a: [int8x16_t; 2]) -> bool {
        unsafe {
            vminvq_u8(vreinterpretq_u8_s8(a[0])) != 0 && vminvq_u8(vreinterpretq_u8_s8(a[1])) != 0
        }
    }

    #[inline(always)]
    fn any_true(a: [int8x16_t; 2]) -> bool {
        unsafe {
            vmaxvq_u8(vreinterpretq_u8_s8(a[0])) != 0 || vmaxvq_u8(vreinterpretq_u8_s8(a[1])) != 0
        }
    }

    #[inline(always)]
    fn bitmask(a: [int8x16_t; 2]) -> u32 {
        // Delegate to NeonToken native bitmask per sub-vector, combine
        let mut result = 0u32;
        for i in 0..2 {
            result |= (<archmage::NeonToken as I8x16Backend>::bitmask(a[i])) << (i * 16);
        }
        result
    }
}

#[cfg(target_arch = "aarch64")]
impl U8x16Backend for archmage::NeonToken {
    type Repr = uint8x16_t;

    #[inline(always)]
    fn splat(v: u8) -> uint8x16_t {
        unsafe { vdupq_n_u8(v) }
    }

    #[inline(always)]
    fn zero() -> uint8x16_t {
        unsafe { vdupq_n_u8(0) }
    }

    #[inline(always)]
    fn load(data: &[u8; 16]) -> uint8x16_t {
        unsafe { vld1q_u8(data.as_ptr()) }
    }

    #[inline(always)]
    fn from_array(arr: [u8; 16]) -> uint8x16_t {
        unsafe { vld1q_u8(arr.as_ptr()) }
    }

    #[inline(always)]
    fn store(repr: uint8x16_t, out: &mut [u8; 16]) {
        unsafe { vst1q_u8(out.as_mut_ptr(), repr) };
    }

    #[inline(always)]
    fn to_array(repr: uint8x16_t) -> [u8; 16] {
        let mut out = [0u8; 16];
        unsafe { vst1q_u8(out.as_mut_ptr(), repr) };
        out
    }

    #[inline(always)]
    fn add(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        unsafe { vaddq_u8(a, b) }
    }
    #[inline(always)]
    fn sub(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        unsafe { vsubq_u8(a, b) }
    }
    #[inline(always)]
    fn min(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        unsafe { vminq_u8(a, b) }
    }
    #[inline(always)]
    fn max(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        unsafe { vmaxq_u8(a, b) }
    }

    #[inline(always)]
    fn simd_eq(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        unsafe { vceqq_u8(a, b) }
    }
    #[inline(always)]
    fn simd_ne(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        unsafe { vmvnq_u8(vceqq_u8(a, b)) }
    }
    #[inline(always)]
    fn simd_lt(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        unsafe { vcltq_u8(a, b) }
    }
    #[inline(always)]
    fn simd_le(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        unsafe { vcleq_u8(a, b) }
    }
    #[inline(always)]
    fn simd_gt(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        unsafe { vcgtq_u8(a, b) }
    }
    #[inline(always)]
    fn simd_ge(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        unsafe { vcgeq_u8(a, b) }
    }

    #[inline(always)]
    fn blend(mask: uint8x16_t, if_true: uint8x16_t, if_false: uint8x16_t) -> uint8x16_t {
        unsafe { vbslq_u8(mask, if_true, if_false) }
    }
    #[inline(always)]
    fn reduce_add(a: uint8x16_t) -> u8 {
        unsafe { vaddvq_u8(a) }
    }
    #[inline(always)]
    fn not(a: uint8x16_t) -> uint8x16_t {
        unsafe { vmvnq_u8(a) }
    }
    #[inline(always)]
    fn bitand(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        unsafe { vandq_u8(a, b) }
    }
    #[inline(always)]
    fn bitor(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        unsafe { vorrq_u8(a, b) }
    }
    #[inline(always)]
    fn bitxor(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        unsafe { veorq_u8(a, b) }
    }
    #[inline(always)]
    fn shl_const<const N: i32>(a: uint8x16_t) -> uint8x16_t {
        unsafe { vshlq_n_u8::<N>(a) }
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: uint8x16_t) -> uint8x16_t {
        unsafe { vshrq_n_u8::<N>(a) }
    }

    #[inline(always)]
    fn all_true(a: uint8x16_t) -> bool {
        unsafe { vminvq_u8(a) != 0 }
    }

    #[inline(always)]
    fn any_true(a: uint8x16_t) -> bool {
        unsafe { vmaxvq_u8(a) != 0 }
    }

    #[inline(always)]
    fn bitmask(a: uint8x16_t) -> u32 {
        unsafe {
            // Shift each byte right by 7 to isolate sign bit
            let bits = vshrq_n_u8::<7>(a);
            // Use polynomial evaluation to pack bits
            // Each byte is now 0 or 1, multiply by position powers of 2
            let powers: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
            let pow_vec = vld1q_u8(powers.as_ptr());
            let weighted = vmulq_u8(bits, pow_vec);
            // Sum pairs: add adjacent bytes
            let pair_sum = vpaddlq_u8(weighted);
            let quad_sum = vpaddlq_u16(pair_sum);
            let oct_sum = vpaddlq_u32(quad_sum);
            // Extract low and high byte
            let lo = vgetq_lane_u64::<0>(oct_sum) as u32;
            let hi = vgetq_lane_u64::<1>(oct_sum) as u32;
            lo | (hi << 8)
        }
    }
}

#[cfg(target_arch = "aarch64")]
impl U8x32Backend for archmage::NeonToken {
    type Repr = [uint8x16_t; 2];

    #[inline(always)]
    fn splat(v: u8) -> [uint8x16_t; 2] {
        unsafe {
            let v4 = vdupq_n_u8(v);
            [v4, v4]
        }
    }

    #[inline(always)]
    fn zero() -> [uint8x16_t; 2] {
        unsafe {
            let z = vdupq_n_u8(0);
            [z, z]
        }
    }

    #[inline(always)]
    fn load(data: &[u8; 32]) -> [uint8x16_t; 2] {
        unsafe {
            [
                vld1q_u8(data.as_ptr().add(0)),
                vld1q_u8(data.as_ptr().add(16)),
            ]
        }
    }

    #[inline(always)]
    fn from_array(arr: [u8; 32]) -> [uint8x16_t; 2] {
        Self::load(&arr)
    }

    #[inline(always)]
    fn store(repr: [uint8x16_t; 2], out: &mut [u8; 32]) {
        unsafe {
            vst1q_u8(out.as_mut_ptr().add(0), repr[0]);
            vst1q_u8(out.as_mut_ptr().add(16), repr[1]);
        }
    }

    #[inline(always)]
    fn to_array(repr: [uint8x16_t; 2]) -> [u8; 32] {
        let mut out = [0u8; 32];
        Self::store(repr, &mut out);
        out
    }

    #[inline(always)]
    fn add(a: [uint8x16_t; 2], b: [uint8x16_t; 2]) -> [uint8x16_t; 2] {
        unsafe { [vaddq_u8(a[0], b[0]), vaddq_u8(a[1], b[1])] }
    }
    #[inline(always)]
    fn sub(a: [uint8x16_t; 2], b: [uint8x16_t; 2]) -> [uint8x16_t; 2] {
        unsafe { [vsubq_u8(a[0], b[0]), vsubq_u8(a[1], b[1])] }
    }
    #[inline(always)]
    fn min(a: [uint8x16_t; 2], b: [uint8x16_t; 2]) -> [uint8x16_t; 2] {
        unsafe { [vminq_u8(a[0], b[0]), vminq_u8(a[1], b[1])] }
    }
    #[inline(always)]
    fn max(a: [uint8x16_t; 2], b: [uint8x16_t; 2]) -> [uint8x16_t; 2] {
        unsafe { [vmaxq_u8(a[0], b[0]), vmaxq_u8(a[1], b[1])] }
    }

    #[inline(always)]
    fn simd_eq(a: [uint8x16_t; 2], b: [uint8x16_t; 2]) -> [uint8x16_t; 2] {
        unsafe { [vceqq_u8(a[0], b[0]), vceqq_u8(a[1], b[1])] }
    }
    #[inline(always)]
    fn simd_ne(a: [uint8x16_t; 2], b: [uint8x16_t; 2]) -> [uint8x16_t; 2] {
        unsafe {
            [
                vmvnq_u8(vceqq_u8(a[0], b[0])),
                vmvnq_u8(vceqq_u8(a[1], b[1])),
            ]
        }
    }
    #[inline(always)]
    fn simd_lt(a: [uint8x16_t; 2], b: [uint8x16_t; 2]) -> [uint8x16_t; 2] {
        unsafe { [vcltq_u8(a[0], b[0]), vcltq_u8(a[1], b[1])] }
    }
    #[inline(always)]
    fn simd_le(a: [uint8x16_t; 2], b: [uint8x16_t; 2]) -> [uint8x16_t; 2] {
        unsafe { [vcleq_u8(a[0], b[0]), vcleq_u8(a[1], b[1])] }
    }
    #[inline(always)]
    fn simd_gt(a: [uint8x16_t; 2], b: [uint8x16_t; 2]) -> [uint8x16_t; 2] {
        unsafe { [vcgtq_u8(a[0], b[0]), vcgtq_u8(a[1], b[1])] }
    }
    #[inline(always)]
    fn simd_ge(a: [uint8x16_t; 2], b: [uint8x16_t; 2]) -> [uint8x16_t; 2] {
        unsafe { [vcgeq_u8(a[0], b[0]), vcgeq_u8(a[1], b[1])] }
    }

    #[inline(always)]
    fn blend(
        mask: [uint8x16_t; 2],
        if_true: [uint8x16_t; 2],
        if_false: [uint8x16_t; 2],
    ) -> [uint8x16_t; 2] {
        unsafe {
            [
                vbslq_u8(mask[0], if_true[0], if_false[0]),
                vbslq_u8(mask[1], if_true[1], if_false[1]),
            ]
        }
    }

    #[inline(always)]
    fn reduce_add(a: [uint8x16_t; 2]) -> u8 {
        let mut sum = 0u8;
        for i in 0..2 {
            sum = sum.wrapping_add(unsafe { vaddvq_u8(a[i]) });
        }
        sum
    }

    #[inline(always)]
    fn not(a: [uint8x16_t; 2]) -> [uint8x16_t; 2] {
        unsafe { [vmvnq_u8(a[0]), vmvnq_u8(a[1])] }
    }
    #[inline(always)]
    fn bitand(a: [uint8x16_t; 2], b: [uint8x16_t; 2]) -> [uint8x16_t; 2] {
        unsafe { [vandq_u8(a[0], b[0]), vandq_u8(a[1], b[1])] }
    }
    #[inline(always)]
    fn bitor(a: [uint8x16_t; 2], b: [uint8x16_t; 2]) -> [uint8x16_t; 2] {
        unsafe { [vorrq_u8(a[0], b[0]), vorrq_u8(a[1], b[1])] }
    }
    #[inline(always)]
    fn bitxor(a: [uint8x16_t; 2], b: [uint8x16_t; 2]) -> [uint8x16_t; 2] {
        unsafe { [veorq_u8(a[0], b[0]), veorq_u8(a[1], b[1])] }
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: [uint8x16_t; 2]) -> [uint8x16_t; 2] {
        unsafe { [vshlq_n_u8::<N>(a[0]), vshlq_n_u8::<N>(a[1])] }
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: [uint8x16_t; 2]) -> [uint8x16_t; 2] {
        unsafe { [vshrq_n_u8::<N>(a[0]), vshrq_n_u8::<N>(a[1])] }
    }

    #[inline(always)]
    fn all_true(a: [uint8x16_t; 2]) -> bool {
        unsafe { vminvq_u8(a[0]) != 0 && vminvq_u8(a[1]) != 0 }
    }

    #[inline(always)]
    fn any_true(a: [uint8x16_t; 2]) -> bool {
        unsafe { vmaxvq_u8(a[0]) != 0 || vmaxvq_u8(a[1]) != 0 }
    }

    #[inline(always)]
    fn bitmask(a: [uint8x16_t; 2]) -> u32 {
        // Delegate to NeonToken native bitmask per sub-vector, combine
        let mut result = 0u32;
        for i in 0..2 {
            result |= (<archmage::NeonToken as U8x16Backend>::bitmask(a[i])) << (i * 16);
        }
        result
    }
}

#[cfg(target_arch = "aarch64")]
impl I16x8Backend for archmage::NeonToken {
    type Repr = int16x8_t;

    #[inline(always)]
    fn splat(v: i16) -> int16x8_t {
        unsafe { vdupq_n_s16(v) }
    }

    #[inline(always)]
    fn zero() -> int16x8_t {
        unsafe { vdupq_n_s16(0) }
    }

    #[inline(always)]
    fn load(data: &[i16; 8]) -> int16x8_t {
        unsafe { vld1q_s16(data.as_ptr()) }
    }

    #[inline(always)]
    fn from_array(arr: [i16; 8]) -> int16x8_t {
        unsafe { vld1q_s16(arr.as_ptr()) }
    }

    #[inline(always)]
    fn store(repr: int16x8_t, out: &mut [i16; 8]) {
        unsafe { vst1q_s16(out.as_mut_ptr(), repr) };
    }

    #[inline(always)]
    fn to_array(repr: int16x8_t) -> [i16; 8] {
        let mut out = [0i16; 8];
        unsafe { vst1q_s16(out.as_mut_ptr(), repr) };
        out
    }

    #[inline(always)]
    fn add(a: int16x8_t, b: int16x8_t) -> int16x8_t {
        unsafe { vaddq_s16(a, b) }
    }
    #[inline(always)]
    fn sub(a: int16x8_t, b: int16x8_t) -> int16x8_t {
        unsafe { vsubq_s16(a, b) }
    }
    #[inline(always)]
    fn mul(a: int16x8_t, b: int16x8_t) -> int16x8_t {
        unsafe { vmulq_s16(a, b) }
    }
    #[inline(always)]
    fn neg(a: int16x8_t) -> int16x8_t {
        unsafe { vnegq_s16(a) }
    }
    #[inline(always)]
    fn min(a: int16x8_t, b: int16x8_t) -> int16x8_t {
        unsafe { vminq_s16(a, b) }
    }
    #[inline(always)]
    fn max(a: int16x8_t, b: int16x8_t) -> int16x8_t {
        unsafe { vmaxq_s16(a, b) }
    }
    #[inline(always)]
    fn abs(a: int16x8_t) -> int16x8_t {
        unsafe { vabsq_s16(a) }
    }

    #[inline(always)]
    fn simd_eq(a: int16x8_t, b: int16x8_t) -> int16x8_t {
        unsafe { vreinterpretq_s16_16(vceqq_s16(a, b)) }
    }
    #[inline(always)]
    fn simd_ne(a: int16x8_t, b: int16x8_t) -> int16x8_t {
        unsafe { vreinterpretq_s16_16(vmvnq_16(vceqq_s16(a, b))) }
    }
    #[inline(always)]
    fn simd_lt(a: int16x8_t, b: int16x8_t) -> int16x8_t {
        unsafe { vreinterpretq_s16_16(vcltq_s16(a, b)) }
    }
    #[inline(always)]
    fn simd_le(a: int16x8_t, b: int16x8_t) -> int16x8_t {
        unsafe { vreinterpretq_s16_16(vcleq_s16(a, b)) }
    }
    #[inline(always)]
    fn simd_gt(a: int16x8_t, b: int16x8_t) -> int16x8_t {
        unsafe { vreinterpretq_s16_16(vcgtq_s16(a, b)) }
    }
    #[inline(always)]
    fn simd_ge(a: int16x8_t, b: int16x8_t) -> int16x8_t {
        unsafe { vreinterpretq_s16_16(vcgeq_s16(a, b)) }
    }

    #[inline(always)]
    fn blend(mask: int16x8_t, if_true: int16x8_t, if_false: int16x8_t) -> int16x8_t {
        unsafe { vbslq_s16(vreinterpretq_16_s16(mask), if_true, if_false) }
    }
    #[inline(always)]
    fn reduce_add(a: int16x8_t) -> i16 {
        unsafe { vaddvq_s16(a) }
    }
    #[inline(always)]
    fn not(a: int16x8_t) -> int16x8_t {
        unsafe { vmvnq_s16(a) }
    }
    #[inline(always)]
    fn bitand(a: int16x8_t, b: int16x8_t) -> int16x8_t {
        unsafe { vandq_s16(a, b) }
    }
    #[inline(always)]
    fn bitor(a: int16x8_t, b: int16x8_t) -> int16x8_t {
        unsafe { vorrq_s16(a, b) }
    }
    #[inline(always)]
    fn bitxor(a: int16x8_t, b: int16x8_t) -> int16x8_t {
        unsafe { veorq_s16(a, b) }
    }
    #[inline(always)]
    fn shl_const<const N: i32>(a: int16x8_t) -> int16x8_t {
        unsafe { vshlq_n_s16::<N>(a) }
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: int16x8_t) -> int16x8_t {
        unsafe { vreinterpretq_s16_u16(vshrq_n_u16::<N>(vreinterpretq_u16_s16(a))) }
    }
    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: int16x8_t) -> int16x8_t {
        unsafe { vshrq_n_s16::<N>(a) }
    }

    #[inline(always)]
    fn all_true(a: int16x8_t) -> bool {
        unsafe { vminvq_16(vreinterpretq_16_s16(a)) != 0 }
    }

    #[inline(always)]
    fn any_true(a: int16x8_t) -> bool {
        unsafe { vmaxvq_16(vreinterpretq_16_s16(a)) != 0 }
    }

    #[inline(always)]
    fn bitmask(a: int16x8_t) -> u32 {
        unsafe {
            (vgetq_lane_u16::<0>(vreinterpretq_u16_s16(vshrq_n_s16::<15>(a))) as u32 & 1) << 0
                | (vgetq_lane_u16::<1>(vreinterpretq_u16_s16(vshrq_n_s16::<15>(a))) as u32 & 1) << 1
                | (vgetq_lane_u16::<2>(vreinterpretq_u16_s16(vshrq_n_s16::<15>(a))) as u32 & 1) << 2
                | (vgetq_lane_u16::<3>(vreinterpretq_u16_s16(vshrq_n_s16::<15>(a))) as u32 & 1) << 3
                | (vgetq_lane_u16::<4>(vreinterpretq_u16_s16(vshrq_n_s16::<15>(a))) as u32 & 1) << 4
                | (vgetq_lane_u16::<5>(vreinterpretq_u16_s16(vshrq_n_s16::<15>(a))) as u32 & 1) << 5
                | (vgetq_lane_u16::<6>(vreinterpretq_u16_s16(vshrq_n_s16::<15>(a))) as u32 & 1) << 6
                | (vgetq_lane_u16::<7>(vreinterpretq_u16_s16(vshrq_n_s16::<15>(a))) as u32 & 1) << 7
        }
    }
}

#[cfg(target_arch = "aarch64")]
impl I16x16Backend for archmage::NeonToken {
    type Repr = [int16x8_t; 2];

    #[inline(always)]
    fn splat(v: i16) -> [int16x8_t; 2] {
        unsafe {
            let v4 = vdupq_n_s16(v);
            [v4, v4]
        }
    }

    #[inline(always)]
    fn zero() -> [int16x8_t; 2] {
        unsafe {
            let z = vdupq_n_s16(0);
            [z, z]
        }
    }

    #[inline(always)]
    fn load(data: &[i16; 16]) -> [int16x8_t; 2] {
        unsafe {
            [
                vld1q_s16(data.as_ptr().add(0)),
                vld1q_s16(data.as_ptr().add(8)),
            ]
        }
    }

    #[inline(always)]
    fn from_array(arr: [i16; 16]) -> [int16x8_t; 2] {
        Self::load(&arr)
    }

    #[inline(always)]
    fn store(repr: [int16x8_t; 2], out: &mut [i16; 16]) {
        unsafe {
            vst1q_s16(out.as_mut_ptr().add(0), repr[0]);
            vst1q_s16(out.as_mut_ptr().add(8), repr[1]);
        }
    }

    #[inline(always)]
    fn to_array(repr: [int16x8_t; 2]) -> [i16; 16] {
        let mut out = [0i16; 16];
        Self::store(repr, &mut out);
        out
    }

    #[inline(always)]
    fn add(a: [int16x8_t; 2], b: [int16x8_t; 2]) -> [int16x8_t; 2] {
        unsafe { [vaddq_s16(a[0], b[0]), vaddq_s16(a[1], b[1])] }
    }
    #[inline(always)]
    fn sub(a: [int16x8_t; 2], b: [int16x8_t; 2]) -> [int16x8_t; 2] {
        unsafe { [vsubq_s16(a[0], b[0]), vsubq_s16(a[1], b[1])] }
    }
    #[inline(always)]
    fn mul(a: [int16x8_t; 2], b: [int16x8_t; 2]) -> [int16x8_t; 2] {
        unsafe { [vmulq_s16(a[0], b[0]), vmulq_s16(a[1], b[1])] }
    }
    #[inline(always)]
    fn neg(a: [int16x8_t; 2]) -> [int16x8_t; 2] {
        unsafe { [vnegq_s16(a[0]), vnegq_s16(a[1])] }
    }
    #[inline(always)]
    fn min(a: [int16x8_t; 2], b: [int16x8_t; 2]) -> [int16x8_t; 2] {
        unsafe { [vminq_s16(a[0], b[0]), vminq_s16(a[1], b[1])] }
    }
    #[inline(always)]
    fn max(a: [int16x8_t; 2], b: [int16x8_t; 2]) -> [int16x8_t; 2] {
        unsafe { [vmaxq_s16(a[0], b[0]), vmaxq_s16(a[1], b[1])] }
    }
    #[inline(always)]
    fn abs(a: [int16x8_t; 2]) -> [int16x8_t; 2] {
        unsafe { [vabsq_s16(a[0]), vabsq_s16(a[1])] }
    }

    #[inline(always)]
    fn simd_eq(a: [int16x8_t; 2], b: [int16x8_t; 2]) -> [int16x8_t; 2] {
        unsafe {
            [
                vreinterpretq_s16_u16(vceqq_s16(a[0], b[0])),
                vreinterpretq_s16_u16(vceqq_s16(a[1], b[1])),
            ]
        }
    }
    #[inline(always)]
    fn simd_ne(a: [int16x8_t; 2], b: [int16x8_t; 2]) -> [int16x8_t; 2] {
        unsafe {
            [
                vreinterpretq_s16_u16(vmvnq_u16(vceqq_s16(a[0], b[0]))),
                vreinterpretq_s16_u16(vmvnq_u16(vceqq_s16(a[1], b[1]))),
            ]
        }
    }
    #[inline(always)]
    fn simd_lt(a: [int16x8_t; 2], b: [int16x8_t; 2]) -> [int16x8_t; 2] {
        unsafe {
            [
                vreinterpretq_s16_u16(vcltq_s16(a[0], b[0])),
                vreinterpretq_s16_u16(vcltq_s16(a[1], b[1])),
            ]
        }
    }
    #[inline(always)]
    fn simd_le(a: [int16x8_t; 2], b: [int16x8_t; 2]) -> [int16x8_t; 2] {
        unsafe {
            [
                vreinterpretq_s16_u16(vcleq_s16(a[0], b[0])),
                vreinterpretq_s16_u16(vcleq_s16(a[1], b[1])),
            ]
        }
    }
    #[inline(always)]
    fn simd_gt(a: [int16x8_t; 2], b: [int16x8_t; 2]) -> [int16x8_t; 2] {
        unsafe {
            [
                vreinterpretq_s16_u16(vcgtq_s16(a[0], b[0])),
                vreinterpretq_s16_u16(vcgtq_s16(a[1], b[1])),
            ]
        }
    }
    #[inline(always)]
    fn simd_ge(a: [int16x8_t; 2], b: [int16x8_t; 2]) -> [int16x8_t; 2] {
        unsafe {
            [
                vreinterpretq_s16_u16(vcgeq_s16(a[0], b[0])),
                vreinterpretq_s16_u16(vcgeq_s16(a[1], b[1])),
            ]
        }
    }

    #[inline(always)]
    fn blend(
        mask: [int16x8_t; 2],
        if_true: [int16x8_t; 2],
        if_false: [int16x8_t; 2],
    ) -> [int16x8_t; 2] {
        unsafe {
            [
                vbslq_s16(vreinterpretq_u16_s16(mask[0]), if_true[0], if_false[0]),
                vbslq_s16(vreinterpretq_u16_s16(mask[1]), if_true[1], if_false[1]),
            ]
        }
    }

    #[inline(always)]
    fn reduce_add(a: [int16x8_t; 2]) -> i16 {
        let mut sum = 0i16;
        for i in 0..2 {
            sum = sum.wrapping_add(unsafe { vaddvq_s16(a[i]) });
        }
        sum
    }

    #[inline(always)]
    fn not(a: [int16x8_t; 2]) -> [int16x8_t; 2] {
        unsafe { [vmvnq_s16(a[0]), vmvnq_s16(a[1])] }
    }
    #[inline(always)]
    fn bitand(a: [int16x8_t; 2], b: [int16x8_t; 2]) -> [int16x8_t; 2] {
        unsafe { [vandq_s16(a[0], b[0]), vandq_s16(a[1], b[1])] }
    }
    #[inline(always)]
    fn bitor(a: [int16x8_t; 2], b: [int16x8_t; 2]) -> [int16x8_t; 2] {
        unsafe { [vorrq_s16(a[0], b[0]), vorrq_s16(a[1], b[1])] }
    }
    #[inline(always)]
    fn bitxor(a: [int16x8_t; 2], b: [int16x8_t; 2]) -> [int16x8_t; 2] {
        unsafe { [veorq_s16(a[0], b[0]), veorq_s16(a[1], b[1])] }
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: [int16x8_t; 2]) -> [int16x8_t; 2] {
        unsafe { [vshlq_n_s16::<N>(a[0]), vshlq_n_s16::<N>(a[1])] }
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: [int16x8_t; 2]) -> [int16x8_t; 2] {
        unsafe {
            [
                vreinterpretq_s16_u16(vshrq_n_u16::<N>(vreinterpretq_u16_s16(a[0]))),
                vreinterpretq_s16_u16(vshrq_n_u16::<N>(vreinterpretq_u16_s16(a[1]))),
            ]
        }
    }
    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: [int16x8_t; 2]) -> [int16x8_t; 2] {
        unsafe { [vshrq_n_s16::<N>(a[0]), vshrq_n_s16::<N>(a[1])] }
    }

    #[inline(always)]
    fn all_true(a: [int16x8_t; 2]) -> bool {
        unsafe {
            vminvq_u16(vreinterpretq_u16_s16(a[0])) != 0
                && vminvq_u16(vreinterpretq_u16_s16(a[1])) != 0
        }
    }

    #[inline(always)]
    fn any_true(a: [int16x8_t; 2]) -> bool {
        unsafe {
            vmaxvq_u16(vreinterpretq_u16_s16(a[0])) != 0
                || vmaxvq_u16(vreinterpretq_u16_s16(a[1])) != 0
        }
    }

    #[inline(always)]
    fn bitmask(a: [int16x8_t; 2]) -> u32 {
        // Delegate to NeonToken native bitmask per sub-vector, combine
        let mut result = 0u32;
        for i in 0..2 {
            result |= (<archmage::NeonToken as I16x8Backend>::bitmask(a[i])) << (i * 8);
        }
        result
    }
}

#[cfg(target_arch = "aarch64")]
impl U16x8Backend for archmage::NeonToken {
    type Repr = uint16x8_t;

    #[inline(always)]
    fn splat(v: u16) -> uint16x8_t {
        unsafe { vdupq_n_u16(v) }
    }

    #[inline(always)]
    fn zero() -> uint16x8_t {
        unsafe { vdupq_n_u16(0) }
    }

    #[inline(always)]
    fn load(data: &[u16; 8]) -> uint16x8_t {
        unsafe { vld1q_u16(data.as_ptr()) }
    }

    #[inline(always)]
    fn from_array(arr: [u16; 8]) -> uint16x8_t {
        unsafe { vld1q_u16(arr.as_ptr()) }
    }

    #[inline(always)]
    fn store(repr: uint16x8_t, out: &mut [u16; 8]) {
        unsafe { vst1q_u16(out.as_mut_ptr(), repr) };
    }

    #[inline(always)]
    fn to_array(repr: uint16x8_t) -> [u16; 8] {
        let mut out = [0u16; 8];
        unsafe { vst1q_u16(out.as_mut_ptr(), repr) };
        out
    }

    #[inline(always)]
    fn add(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        unsafe { vaddq_u16(a, b) }
    }
    #[inline(always)]
    fn sub(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        unsafe { vsubq_u16(a, b) }
    }
    #[inline(always)]
    fn mul(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        unsafe { vmulq_u16(a, b) }
    }
    #[inline(always)]
    fn min(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        unsafe { vminq_u16(a, b) }
    }
    #[inline(always)]
    fn max(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        unsafe { vmaxq_u16(a, b) }
    }

    #[inline(always)]
    fn simd_eq(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        unsafe { vceqq_u16(a, b) }
    }
    #[inline(always)]
    fn simd_ne(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        unsafe { vmvnq_u16(vceqq_u16(a, b)) }
    }
    #[inline(always)]
    fn simd_lt(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        unsafe { vcltq_u16(a, b) }
    }
    #[inline(always)]
    fn simd_le(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        unsafe { vcleq_u16(a, b) }
    }
    #[inline(always)]
    fn simd_gt(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        unsafe { vcgtq_u16(a, b) }
    }
    #[inline(always)]
    fn simd_ge(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        unsafe { vcgeq_u16(a, b) }
    }

    #[inline(always)]
    fn blend(mask: uint16x8_t, if_true: uint16x8_t, if_false: uint16x8_t) -> uint16x8_t {
        unsafe { vbslq_u16(mask, if_true, if_false) }
    }
    #[inline(always)]
    fn reduce_add(a: uint16x8_t) -> u16 {
        unsafe { vaddvq_u16(a) }
    }
    #[inline(always)]
    fn not(a: uint16x8_t) -> uint16x8_t {
        unsafe { vmvnq_u16(a) }
    }
    #[inline(always)]
    fn bitand(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        unsafe { vandq_u16(a, b) }
    }
    #[inline(always)]
    fn bitor(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        unsafe { vorrq_u16(a, b) }
    }
    #[inline(always)]
    fn bitxor(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        unsafe { veorq_u16(a, b) }
    }
    #[inline(always)]
    fn shl_const<const N: i32>(a: uint16x8_t) -> uint16x8_t {
        unsafe { vshlq_n_u16::<N>(a) }
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: uint16x8_t) -> uint16x8_t {
        unsafe { vshrq_n_u16::<N>(a) }
    }

    #[inline(always)]
    fn all_true(a: uint16x8_t) -> bool {
        unsafe { vminvq_u16(a) != 0 }
    }

    #[inline(always)]
    fn any_true(a: uint16x8_t) -> bool {
        unsafe { vmaxvq_u16(a) != 0 }
    }

    #[inline(always)]
    fn bitmask(a: uint16x8_t) -> u32 {
        unsafe {
            (vgetq_lane_u16::<0>(vshrq_n_u16::<15>(a)) as u32 & 1) << 0
                | (vgetq_lane_u16::<1>(vshrq_n_u16::<15>(a)) as u32 & 1) << 1
                | (vgetq_lane_u16::<2>(vshrq_n_u16::<15>(a)) as u32 & 1) << 2
                | (vgetq_lane_u16::<3>(vshrq_n_u16::<15>(a)) as u32 & 1) << 3
                | (vgetq_lane_u16::<4>(vshrq_n_u16::<15>(a)) as u32 & 1) << 4
                | (vgetq_lane_u16::<5>(vshrq_n_u16::<15>(a)) as u32 & 1) << 5
                | (vgetq_lane_u16::<6>(vshrq_n_u16::<15>(a)) as u32 & 1) << 6
                | (vgetq_lane_u16::<7>(vshrq_n_u16::<15>(a)) as u32 & 1) << 7
        }
    }
}

#[cfg(target_arch = "aarch64")]
impl U16x16Backend for archmage::NeonToken {
    type Repr = [uint16x8_t; 2];

    #[inline(always)]
    fn splat(v: u16) -> [uint16x8_t; 2] {
        unsafe {
            let v4 = vdupq_n_u16(v);
            [v4, v4]
        }
    }

    #[inline(always)]
    fn zero() -> [uint16x8_t; 2] {
        unsafe {
            let z = vdupq_n_u16(0);
            [z, z]
        }
    }

    #[inline(always)]
    fn load(data: &[u16; 16]) -> [uint16x8_t; 2] {
        unsafe {
            [
                vld1q_u16(data.as_ptr().add(0)),
                vld1q_u16(data.as_ptr().add(8)),
            ]
        }
    }

    #[inline(always)]
    fn from_array(arr: [u16; 16]) -> [uint16x8_t; 2] {
        Self::load(&arr)
    }

    #[inline(always)]
    fn store(repr: [uint16x8_t; 2], out: &mut [u16; 16]) {
        unsafe {
            vst1q_u16(out.as_mut_ptr().add(0), repr[0]);
            vst1q_u16(out.as_mut_ptr().add(8), repr[1]);
        }
    }

    #[inline(always)]
    fn to_array(repr: [uint16x8_t; 2]) -> [u16; 16] {
        let mut out = [0u16; 16];
        Self::store(repr, &mut out);
        out
    }

    #[inline(always)]
    fn add(a: [uint16x8_t; 2], b: [uint16x8_t; 2]) -> [uint16x8_t; 2] {
        unsafe { [vaddq_u16(a[0], b[0]), vaddq_u16(a[1], b[1])] }
    }
    #[inline(always)]
    fn sub(a: [uint16x8_t; 2], b: [uint16x8_t; 2]) -> [uint16x8_t; 2] {
        unsafe { [vsubq_u16(a[0], b[0]), vsubq_u16(a[1], b[1])] }
    }
    #[inline(always)]
    fn mul(a: [uint16x8_t; 2], b: [uint16x8_t; 2]) -> [uint16x8_t; 2] {
        unsafe { [vmulq_u16(a[0], b[0]), vmulq_u16(a[1], b[1])] }
    }
    #[inline(always)]
    fn min(a: [uint16x8_t; 2], b: [uint16x8_t; 2]) -> [uint16x8_t; 2] {
        unsafe { [vminq_u16(a[0], b[0]), vminq_u16(a[1], b[1])] }
    }
    #[inline(always)]
    fn max(a: [uint16x8_t; 2], b: [uint16x8_t; 2]) -> [uint16x8_t; 2] {
        unsafe { [vmaxq_u16(a[0], b[0]), vmaxq_u16(a[1], b[1])] }
    }

    #[inline(always)]
    fn simd_eq(a: [uint16x8_t; 2], b: [uint16x8_t; 2]) -> [uint16x8_t; 2] {
        unsafe { [vceqq_u16(a[0], b[0]), vceqq_u16(a[1], b[1])] }
    }
    #[inline(always)]
    fn simd_ne(a: [uint16x8_t; 2], b: [uint16x8_t; 2]) -> [uint16x8_t; 2] {
        unsafe {
            [
                vmvnq_u16(vceqq_u16(a[0], b[0])),
                vmvnq_u16(vceqq_u16(a[1], b[1])),
            ]
        }
    }
    #[inline(always)]
    fn simd_lt(a: [uint16x8_t; 2], b: [uint16x8_t; 2]) -> [uint16x8_t; 2] {
        unsafe { [vcltq_u16(a[0], b[0]), vcltq_u16(a[1], b[1])] }
    }
    #[inline(always)]
    fn simd_le(a: [uint16x8_t; 2], b: [uint16x8_t; 2]) -> [uint16x8_t; 2] {
        unsafe { [vcleq_u16(a[0], b[0]), vcleq_u16(a[1], b[1])] }
    }
    #[inline(always)]
    fn simd_gt(a: [uint16x8_t; 2], b: [uint16x8_t; 2]) -> [uint16x8_t; 2] {
        unsafe { [vcgtq_u16(a[0], b[0]), vcgtq_u16(a[1], b[1])] }
    }
    #[inline(always)]
    fn simd_ge(a: [uint16x8_t; 2], b: [uint16x8_t; 2]) -> [uint16x8_t; 2] {
        unsafe { [vcgeq_u16(a[0], b[0]), vcgeq_u16(a[1], b[1])] }
    }

    #[inline(always)]
    fn blend(
        mask: [uint16x8_t; 2],
        if_true: [uint16x8_t; 2],
        if_false: [uint16x8_t; 2],
    ) -> [uint16x8_t; 2] {
        unsafe {
            [
                vbslq_u16(mask[0], if_true[0], if_false[0]),
                vbslq_u16(mask[1], if_true[1], if_false[1]),
            ]
        }
    }

    #[inline(always)]
    fn reduce_add(a: [uint16x8_t; 2]) -> u16 {
        let mut sum = 0u16;
        for i in 0..2 {
            sum = sum.wrapping_add(unsafe { vaddvq_u16(a[i]) });
        }
        sum
    }

    #[inline(always)]
    fn not(a: [uint16x8_t; 2]) -> [uint16x8_t; 2] {
        unsafe { [vmvnq_u16(a[0]), vmvnq_u16(a[1])] }
    }
    #[inline(always)]
    fn bitand(a: [uint16x8_t; 2], b: [uint16x8_t; 2]) -> [uint16x8_t; 2] {
        unsafe { [vandq_u16(a[0], b[0]), vandq_u16(a[1], b[1])] }
    }
    #[inline(always)]
    fn bitor(a: [uint16x8_t; 2], b: [uint16x8_t; 2]) -> [uint16x8_t; 2] {
        unsafe { [vorrq_u16(a[0], b[0]), vorrq_u16(a[1], b[1])] }
    }
    #[inline(always)]
    fn bitxor(a: [uint16x8_t; 2], b: [uint16x8_t; 2]) -> [uint16x8_t; 2] {
        unsafe { [veorq_u16(a[0], b[0]), veorq_u16(a[1], b[1])] }
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: [uint16x8_t; 2]) -> [uint16x8_t; 2] {
        unsafe { [vshlq_n_u16::<N>(a[0]), vshlq_n_u16::<N>(a[1])] }
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: [uint16x8_t; 2]) -> [uint16x8_t; 2] {
        unsafe { [vshrq_n_u16::<N>(a[0]), vshrq_n_u16::<N>(a[1])] }
    }

    #[inline(always)]
    fn all_true(a: [uint16x8_t; 2]) -> bool {
        unsafe { vminvq_u16(a[0]) != 0 && vminvq_u16(a[1]) != 0 }
    }

    #[inline(always)]
    fn any_true(a: [uint16x8_t; 2]) -> bool {
        unsafe { vmaxvq_u16(a[0]) != 0 || vmaxvq_u16(a[1]) != 0 }
    }

    #[inline(always)]
    fn bitmask(a: [uint16x8_t; 2]) -> u32 {
        // Delegate to NeonToken native bitmask per sub-vector, combine
        let mut result = 0u32;
        for i in 0..2 {
            result |= (<archmage::NeonToken as U16x8Backend>::bitmask(a[i])) << (i * 8);
        }
        result
    }
}

#[cfg(target_arch = "aarch64")]
impl U64x2Backend for archmage::NeonToken {
    type Repr = uint64x2_t;

    #[inline(always)]
    fn splat(v: u64) -> uint64x2_t {
        unsafe { vdupq_n_u64(v) }
    }

    #[inline(always)]
    fn zero() -> uint64x2_t {
        unsafe { vdupq_n_u64(0) }
    }

    #[inline(always)]
    fn load(data: &[u64; 2]) -> uint64x2_t {
        unsafe { vld1q_u64(data.as_ptr()) }
    }

    #[inline(always)]
    fn from_array(arr: [u64; 2]) -> uint64x2_t {
        unsafe { vld1q_u64(arr.as_ptr()) }
    }

    #[inline(always)]
    fn store(repr: uint64x2_t, out: &mut [u64; 2]) {
        unsafe { vst1q_u64(out.as_mut_ptr(), repr) };
    }

    #[inline(always)]
    fn to_array(repr: uint64x2_t) -> [u64; 2] {
        let mut out = [0u64; 2];
        unsafe { vst1q_u64(out.as_mut_ptr(), repr) };
        out
    }

    #[inline(always)]
    fn add(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        unsafe { vaddq_u64(a, b) }
    }
    #[inline(always)]
    fn sub(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        unsafe { vsubq_u64(a, b) }
    }
    #[inline(always)]
    fn min(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        unsafe { vminq_u64(a, b) }
    }
    #[inline(always)]
    fn max(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        unsafe { vmaxq_u64(a, b) }
    }

    #[inline(always)]
    fn simd_eq(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        unsafe { vceqq_u64(a, b) }
    }
    #[inline(always)]
    fn simd_ne(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        unsafe { vmvnq_u64(vceqq_u64(a, b)) }
    }
    #[inline(always)]
    fn simd_lt(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        unsafe { vcltq_u64(a, b) }
    }
    #[inline(always)]
    fn simd_le(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        unsafe { vcleq_u64(a, b) }
    }
    #[inline(always)]
    fn simd_gt(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        unsafe { vcgtq_u64(a, b) }
    }
    #[inline(always)]
    fn simd_ge(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        unsafe { vcgeq_u64(a, b) }
    }

    #[inline(always)]
    fn blend(mask: uint64x2_t, if_true: uint64x2_t, if_false: uint64x2_t) -> uint64x2_t {
        unsafe { vbslq_u64(mask, if_true, if_false) }
    }
    #[inline(always)]
    fn reduce_add(a: uint64x2_t) -> u64 {
        unsafe { vaddvq_u64(a) }
    }
    #[inline(always)]
    fn not(a: uint64x2_t) -> uint64x2_t {
        unsafe { veorq_u64(a, vdupq_n_u64(u64::MAX)) }
    }
    #[inline(always)]
    fn bitand(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        unsafe { vandq_u64(a, b) }
    }
    #[inline(always)]
    fn bitor(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        unsafe { vorrq_u64(a, b) }
    }
    #[inline(always)]
    fn bitxor(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        unsafe { veorq_u64(a, b) }
    }
    #[inline(always)]
    fn shl_const<const N: i32>(a: uint64x2_t) -> uint64x2_t {
        unsafe { vshlq_n_u64::<N>(a) }
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: uint64x2_t) -> uint64x2_t {
        unsafe { vshrq_n_u64::<N>(a) }
    }

    #[inline(always)]
    fn all_true(a: uint64x2_t) -> bool {
        unsafe { vminvq_u64(a) != 0 }
    }

    #[inline(always)]
    fn any_true(a: uint64x2_t) -> bool {
        unsafe { vmaxvq_u64(a) != 0 }
    }

    #[inline(always)]
    fn bitmask(a: uint64x2_t) -> u32 {
        unsafe {
            let shift = vshrq_n_u64::<63>(a);
            let lane0 = vgetq_lane_u64::<0>(shift) as u32;
            let lane1 = vgetq_lane_u64::<1>(shift) as u32;
            lane0 | (lane1 << 1)
        }
    }
}

#[cfg(target_arch = "aarch64")]
impl U64x4Backend for archmage::NeonToken {
    type Repr = [uint64x2_t; 2];

    #[inline(always)]
    fn splat(v: u64) -> [uint64x2_t; 2] {
        unsafe {
            let v4 = vdupq_n_u64(v);
            [v4, v4]
        }
    }

    #[inline(always)]
    fn zero() -> [uint64x2_t; 2] {
        unsafe {
            let z = vdupq_n_u64(0);
            [z, z]
        }
    }

    #[inline(always)]
    fn load(data: &[u64; 4]) -> [uint64x2_t; 2] {
        unsafe {
            [
                vld1q_u64(data.as_ptr().add(0)),
                vld1q_u64(data.as_ptr().add(2)),
            ]
        }
    }

    #[inline(always)]
    fn from_array(arr: [u64; 4]) -> [uint64x2_t; 2] {
        Self::load(&arr)
    }

    #[inline(always)]
    fn store(repr: [uint64x2_t; 2], out: &mut [u64; 4]) {
        unsafe {
            vst1q_u64(out.as_mut_ptr().add(0), repr[0]);
            vst1q_u64(out.as_mut_ptr().add(2), repr[1]);
        }
    }

    #[inline(always)]
    fn to_array(repr: [uint64x2_t; 2]) -> [u64; 4] {
        let mut out = [0u64; 4];
        Self::store(repr, &mut out);
        out
    }

    #[inline(always)]
    fn add(a: [uint64x2_t; 2], b: [uint64x2_t; 2]) -> [uint64x2_t; 2] {
        unsafe { [vaddq_u64(a[0], b[0]), vaddq_u64(a[1], b[1])] }
    }
    #[inline(always)]
    fn sub(a: [uint64x2_t; 2], b: [uint64x2_t; 2]) -> [uint64x2_t; 2] {
        unsafe { [vsubq_u64(a[0], b[0]), vsubq_u64(a[1], b[1])] }
    }
    #[inline(always)]
    fn min(a: [uint64x2_t; 2], b: [uint64x2_t; 2]) -> [uint64x2_t; 2] {
        unsafe { [vminq_u64(a[0], b[0]), vminq_u64(a[1], b[1])] }
    }
    #[inline(always)]
    fn max(a: [uint64x2_t; 2], b: [uint64x2_t; 2]) -> [uint64x2_t; 2] {
        unsafe { [vmaxq_u64(a[0], b[0]), vmaxq_u64(a[1], b[1])] }
    }

    #[inline(always)]
    fn simd_eq(a: [uint64x2_t; 2], b: [uint64x2_t; 2]) -> [uint64x2_t; 2] {
        unsafe { [vceqq_u64(a[0], b[0]), vceqq_u64(a[1], b[1])] }
    }
    #[inline(always)]
    fn simd_ne(a: [uint64x2_t; 2], b: [uint64x2_t; 2]) -> [uint64x2_t; 2] {
        unsafe {
            [
                vmvnq_u64(vceqq_u64(a[0], b[0])),
                vmvnq_u64(vceqq_u64(a[1], b[1])),
            ]
        }
    }
    #[inline(always)]
    fn simd_lt(a: [uint64x2_t; 2], b: [uint64x2_t; 2]) -> [uint64x2_t; 2] {
        unsafe { [vcltq_u64(a[0], b[0]), vcltq_u64(a[1], b[1])] }
    }
    #[inline(always)]
    fn simd_le(a: [uint64x2_t; 2], b: [uint64x2_t; 2]) -> [uint64x2_t; 2] {
        unsafe { [vcleq_u64(a[0], b[0]), vcleq_u64(a[1], b[1])] }
    }
    #[inline(always)]
    fn simd_gt(a: [uint64x2_t; 2], b: [uint64x2_t; 2]) -> [uint64x2_t; 2] {
        unsafe { [vcgtq_u64(a[0], b[0]), vcgtq_u64(a[1], b[1])] }
    }
    #[inline(always)]
    fn simd_ge(a: [uint64x2_t; 2], b: [uint64x2_t; 2]) -> [uint64x2_t; 2] {
        unsafe { [vcgeq_u64(a[0], b[0]), vcgeq_u64(a[1], b[1])] }
    }

    #[inline(always)]
    fn blend(
        mask: [uint64x2_t; 2],
        if_true: [uint64x2_t; 2],
        if_false: [uint64x2_t; 2],
    ) -> [uint64x2_t; 2] {
        unsafe {
            [
                vbslq_u64(mask[0], if_true[0], if_false[0]),
                vbslq_u64(mask[1], if_true[1], if_false[1]),
            ]
        }
    }

    #[inline(always)]
    fn reduce_add(a: [uint64x2_t; 2]) -> u64 {
        let mut sum = 0u64;
        for i in 0..2 {
            sum = sum.wrapping_add(unsafe { vaddvq_u64(a[i]) });
        }
        sum
    }

    #[inline(always)]
    fn not(a: [uint64x2_t; 2]) -> [uint64x2_t; 2] {
        unsafe {
            [
                veorq_u64(a[0], vdupq_n_u64(u64::MAX)),
                veorq_u64(a[1], vdupq_n_u64(u64::MAX)),
            ]
        }
    }
    #[inline(always)]
    fn bitand(a: [uint64x2_t; 2], b: [uint64x2_t; 2]) -> [uint64x2_t; 2] {
        unsafe { [vandq_u64(a[0], b[0]), vandq_u64(a[1], b[1])] }
    }
    #[inline(always)]
    fn bitor(a: [uint64x2_t; 2], b: [uint64x2_t; 2]) -> [uint64x2_t; 2] {
        unsafe { [vorrq_u64(a[0], b[0]), vorrq_u64(a[1], b[1])] }
    }
    #[inline(always)]
    fn bitxor(a: [uint64x2_t; 2], b: [uint64x2_t; 2]) -> [uint64x2_t; 2] {
        unsafe { [veorq_u64(a[0], b[0]), veorq_u64(a[1], b[1])] }
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: [uint64x2_t; 2]) -> [uint64x2_t; 2] {
        unsafe { [vshlq_n_u64::<N>(a[0]), vshlq_n_u64::<N>(a[1])] }
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: [uint64x2_t; 2]) -> [uint64x2_t; 2] {
        unsafe { [vshrq_n_u64::<N>(a[0]), vshrq_n_u64::<N>(a[1])] }
    }

    #[inline(always)]
    fn all_true(a: [uint64x2_t; 2]) -> bool {
        unsafe { vminvq_u64(a[0]) != 0 && vminvq_u64(a[1]) != 0 }
    }

    #[inline(always)]
    fn any_true(a: [uint64x2_t; 2]) -> bool {
        unsafe { vmaxvq_u64(a[0]) != 0 || vmaxvq_u64(a[1]) != 0 }
    }

    #[inline(always)]
    fn bitmask(a: [uint64x2_t; 2]) -> u32 {
        // Delegate to NeonToken native bitmask per sub-vector, combine
        let mut result = 0u32;
        for i in 0..2 {
            result |= (<archmage::NeonToken as U64x2Backend>::bitmask(a[i])) << (i * 2);
        }
        result
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

#[cfg(target_arch = "aarch64")]
impl U32x4Bitcast for archmage::NeonToken {
    #[inline(always)]
    fn bitcast_u32_to_i32(a: uint32x4_t) -> int32x4_t {
        unsafe { vreinterpretq_s32_u32(a) }
    }

    #[inline(always)]
    fn bitcast_i32_to_u32(a: int32x4_t) -> uint32x4_t {
        unsafe { vreinterpretq_u32_s32(a) }
    }
}

#[cfg(target_arch = "aarch64")]
impl U32x8Bitcast for archmage::NeonToken {
    #[inline(always)]
    fn bitcast_u32_to_i32(a: [uint32x4_t; 2]) -> [int32x4_t; 2] {
        unsafe { [vreinterpretq_s32_u32(a[0]), vreinterpretq_s32_u32(a[1])] }
    }

    #[inline(always)]
    fn bitcast_i32_to_u32(a: [int32x4_t; 2]) -> [uint32x4_t; 2] {
        unsafe { [vreinterpretq_u32_s32(a[0]), vreinterpretq_u32_s32(a[1])] }
    }
}

#[cfg(target_arch = "aarch64")]
impl I64x2Bitcast for archmage::NeonToken {
    #[inline(always)]
    fn bitcast_i64_to_f64(a: int64x2_t) -> float64x2_t {
        unsafe { vreinterpretq_f64_s64(a) }
    }

    #[inline(always)]
    fn bitcast_f64_to_i64(a: float64x2_t) -> int64x2_t {
        unsafe { vreinterpretq_s64_f64(a) }
    }
}

#[cfg(target_arch = "aarch64")]
impl I64x4Bitcast for archmage::NeonToken {
    #[inline(always)]
    fn bitcast_i64_to_f64(a: [int64x2_t; 2]) -> [float64x2_t; 2] {
        unsafe { [vreinterpretq_f64_s64(a[0]), vreinterpretq_f64_s64(a[1])] }
    }

    #[inline(always)]
    fn bitcast_f64_to_i64(a: [float64x2_t; 2]) -> [int64x2_t; 2] {
        unsafe { [vreinterpretq_s64_f64(a[0]), vreinterpretq_s64_f64(a[1])] }
    }
}

#[cfg(target_arch = "aarch64")]
impl I8x16Bitcast for archmage::NeonToken {
    #[inline(always)]
    fn bitcast_i8_to_u8(a: int8x16_t) -> uint8x16_t {
        unsafe { vreinterpretq_u8_s8(a) }
    }
    #[inline(always)]
    fn bitcast_u8_to_i8(a: uint8x16_t) -> int8x16_t {
        unsafe { vreinterpretq_s8_u8(a) }
    }
}

#[cfg(target_arch = "aarch64")]
impl I8x32Bitcast for archmage::NeonToken {
    #[inline(always)]
    fn bitcast_i8_to_u8(a: [int8x16_t; 2]) -> [uint8x16_t; 2] {
        unsafe { [vreinterpretq_u8_s8(a[0]), vreinterpretq_u8_s8(a[1])] }
    }
    #[inline(always)]
    fn bitcast_u8_to_i8(a: [uint8x16_t; 2]) -> [int8x16_t; 2] {
        unsafe { [vreinterpretq_s8_u8(a[0]), vreinterpretq_s8_u8(a[1])] }
    }
}

#[cfg(target_arch = "aarch64")]
impl I16x8Bitcast for archmage::NeonToken {
    #[inline(always)]
    fn bitcast_i16_to_u16(a: int16x8_t) -> uint16x8_t {
        unsafe { vreinterpretq_u16_s16(a) }
    }
    #[inline(always)]
    fn bitcast_u16_to_i16(a: uint16x8_t) -> int16x8_t {
        unsafe { vreinterpretq_s16_u16(a) }
    }
}

#[cfg(target_arch = "aarch64")]
impl I16x16Bitcast for archmage::NeonToken {
    #[inline(always)]
    fn bitcast_i16_to_u16(a: [int16x8_t; 2]) -> [uint16x8_t; 2] {
        unsafe { [vreinterpretq_u16_s16(a[0]), vreinterpretq_u16_s16(a[1])] }
    }
    #[inline(always)]
    fn bitcast_u16_to_i16(a: [uint16x8_t; 2]) -> [int16x8_t; 2] {
        unsafe { [vreinterpretq_s16_u16(a[0]), vreinterpretq_s16_u16(a[1])] }
    }
}

#[cfg(target_arch = "aarch64")]
impl U64x2Bitcast for archmage::NeonToken {
    #[inline(always)]
    fn bitcast_u64_to_i64(a: uint64x2_t) -> int64x2_t {
        unsafe { vreinterpretq_s64_u64(a) }
    }
    #[inline(always)]
    fn bitcast_i64_to_u64(a: int64x2_t) -> uint64x2_t {
        unsafe { vreinterpretq_u64_s64(a) }
    }
}

#[cfg(target_arch = "aarch64")]
impl U64x4Bitcast for archmage::NeonToken {
    #[inline(always)]
    fn bitcast_u64_to_i64(a: [uint64x2_t; 2]) -> [int64x2_t; 2] {
        unsafe { [vreinterpretq_s64_u64(a[0]), vreinterpretq_s64_u64(a[1])] }
    }
    #[inline(always)]
    fn bitcast_i64_to_u64(a: [int64x2_t; 2]) -> [uint64x2_t; 2] {
        unsafe { [vreinterpretq_u64_s64(a[0]), vreinterpretq_u64_s64(a[1])] }
    }
}
#[cfg(target_arch = "aarch64")]
impl F32x16Backend for archmage::NeonToken {
    type Repr = [float32x4_t; 4];

    #[inline(always)]
    fn splat(v: f32) -> [float32x4_t; 4] {
        let q = <archmage::NeonToken as F32x4Backend>::splat(v);
        [q, q, q, q]
    }

    #[inline(always)]
    fn zero() -> [float32x4_t; 4] {
        let q = <archmage::NeonToken as F32x4Backend>::zero();
        [q, q, q, q]
    }

    #[inline(always)]
    fn load(data: &[f32; 16]) -> [float32x4_t; 4] {
        [
            <archmage::NeonToken as F32x4Backend>::load(data[0..4].try_into().unwrap()),
            <archmage::NeonToken as F32x4Backend>::load(data[4..8].try_into().unwrap()),
            <archmage::NeonToken as F32x4Backend>::load(data[8..12].try_into().unwrap()),
            <archmage::NeonToken as F32x4Backend>::load(data[12..16].try_into().unwrap()),
        ]
    }

    #[inline(always)]
    fn from_array(arr: [f32; 16]) -> [float32x4_t; 4] {
        let mut q0 = [0.0f32; 4];
        let mut q1 = [0.0f32; 4];
        let mut q2 = [0.0f32; 4];
        let mut q3 = [0.0f32; 4];
        q0.copy_from_slice(&arr[0..4]);
        q1.copy_from_slice(&arr[4..8]);
        q2.copy_from_slice(&arr[8..12]);
        q3.copy_from_slice(&arr[12..16]);
        [
            <archmage::NeonToken as F32x4Backend>::from_array(q0),
            <archmage::NeonToken as F32x4Backend>::from_array(q1),
            <archmage::NeonToken as F32x4Backend>::from_array(q2),
            <archmage::NeonToken as F32x4Backend>::from_array(q3),
        ]
    }

    #[inline(always)]
    fn store(repr: [float32x4_t; 4], out: &mut [f32; 16]) {
        let (o01, o23) = out.split_at_mut(8);
        let (o0, o1) = o01.split_at_mut(4);
        let (o2, o3) = o23.split_at_mut(4);
        <archmage::NeonToken as F32x4Backend>::store(repr[0], o0.try_into().unwrap());
        <archmage::NeonToken as F32x4Backend>::store(repr[1], o1.try_into().unwrap());
        <archmage::NeonToken as F32x4Backend>::store(repr[2], o2.try_into().unwrap());
        <archmage::NeonToken as F32x4Backend>::store(repr[3], o3.try_into().unwrap());
    }

    #[inline(always)]
    fn to_array(repr: [float32x4_t; 4]) -> [f32; 16] {
        let a0 = <archmage::NeonToken as F32x4Backend>::to_array(repr[0]);
        let a1 = <archmage::NeonToken as F32x4Backend>::to_array(repr[1]);
        let a2 = <archmage::NeonToken as F32x4Backend>::to_array(repr[2]);
        let a3 = <archmage::NeonToken as F32x4Backend>::to_array(repr[3]);
        let mut out = [0.0f32; 16];
        out[0..4].copy_from_slice(&a0);
        out[4..8].copy_from_slice(&a1);
        out[8..12].copy_from_slice(&a2);
        out[12..16].copy_from_slice(&a3);
        out
    }

    #[inline(always)]
    fn add(a: [float32x4_t; 4], b: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::add(a[i], b[i]))
    }

    #[inline(always)]
    fn sub(a: [float32x4_t; 4], b: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::sub(a[i], b[i]))
    }

    #[inline(always)]
    fn mul(a: [float32x4_t; 4], b: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::mul(a[i], b[i]))
    }

    #[inline(always)]
    fn div(a: [float32x4_t; 4], b: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::div(a[i], b[i]))
    }

    #[inline(always)]
    fn neg(a: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::neg(a[i]))
    }

    #[inline(always)]
    fn min(a: [float32x4_t; 4], b: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::min(a[i], b[i]))
    }

    #[inline(always)]
    fn max(a: [float32x4_t; 4], b: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::max(a[i], b[i]))
    }

    #[inline(always)]
    fn sqrt(a: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::sqrt(a[i]))
    }

    #[inline(always)]
    fn abs(a: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::abs(a[i]))
    }

    #[inline(always)]
    fn floor(a: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::floor(a[i]))
    }

    #[inline(always)]
    fn ceil(a: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::ceil(a[i]))
    }

    #[inline(always)]
    fn round(a: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::round(a[i]))
    }

    #[inline(always)]
    fn mul_add(a: [float32x4_t; 4], b: [float32x4_t; 4], c: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::mul_add(a[i], b[i], c[i]))
    }

    #[inline(always)]
    fn mul_sub(a: [float32x4_t; 4], b: [float32x4_t; 4], c: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::mul_sub(a[i], b[i], c[i]))
    }

    #[inline(always)]
    fn reduce_add(a: [float32x4_t; 4]) -> f32 {
        <archmage::NeonToken as F32x4Backend>::reduce_add(a[0])
            + <archmage::NeonToken as F32x4Backend>::reduce_add(a[1])
            + <archmage::NeonToken as F32x4Backend>::reduce_add(a[2])
            + <archmage::NeonToken as F32x4Backend>::reduce_add(a[3])
    }

    #[inline(always)]
    fn reduce_min(a: [float32x4_t; 4]) -> f32 {
        let m01 = {
            let l = <archmage::NeonToken as F32x4Backend>::reduce_min(a[0]);
            let r = <archmage::NeonToken as F32x4Backend>::reduce_min(a[1]);
            if l < r { l } else { r }
        };
        let m23 = {
            let l = <archmage::NeonToken as F32x4Backend>::reduce_min(a[2]);
            let r = <archmage::NeonToken as F32x4Backend>::reduce_min(a[3]);
            if l < r { l } else { r }
        };
        if m01 < m23 { m01 } else { m23 }
    }

    #[inline(always)]
    fn reduce_max(a: [float32x4_t; 4]) -> f32 {
        let m01 = {
            let l = <archmage::NeonToken as F32x4Backend>::reduce_max(a[0]);
            let r = <archmage::NeonToken as F32x4Backend>::reduce_max(a[1]);
            if l > r { l } else { r }
        };
        let m23 = {
            let l = <archmage::NeonToken as F32x4Backend>::reduce_max(a[2]);
            let r = <archmage::NeonToken as F32x4Backend>::reduce_max(a[3]);
            if l > r { l } else { r }
        };
        if m01 > m23 { m01 } else { m23 }
    }

    #[inline(always)]
    fn simd_eq(a: [float32x4_t; 4], b: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::simd_eq(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ne(a: [float32x4_t; 4], b: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::simd_ne(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_lt(a: [float32x4_t; 4], b: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::simd_lt(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_le(a: [float32x4_t; 4], b: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::simd_le(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_gt(a: [float32x4_t; 4], b: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::simd_gt(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ge(a: [float32x4_t; 4], b: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::simd_ge(a[i], b[i]))
    }

    #[inline(always)]
    fn blend(
        mask: [float32x4_t; 4],
        if_true: [float32x4_t; 4],
        if_false: [float32x4_t; 4],
    ) -> [float32x4_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as F32x4Backend>::blend(mask[i], if_true[i], if_false[i])
        })
    }

    #[inline(always)]
    fn not(a: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::not(a[i]))
    }

    #[inline(always)]
    fn bitand(a: [float32x4_t; 4], b: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::bitand(a[i], b[i]))
    }

    #[inline(always)]
    fn bitor(a: [float32x4_t; 4], b: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::bitor(a[i], b[i]))
    }

    #[inline(always)]
    fn bitxor(a: [float32x4_t; 4], b: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::bitxor(a[i], b[i]))
    }
}

#[cfg(target_arch = "aarch64")]
impl F64x8Backend for archmage::NeonToken {
    type Repr = [float64x2_t; 4];

    #[inline(always)]
    fn splat(v: f64) -> [float64x2_t; 4] {
        let q = <archmage::NeonToken as F64x2Backend>::splat(v);
        [q, q, q, q]
    }

    #[inline(always)]
    fn zero() -> [float64x2_t; 4] {
        let q = <archmage::NeonToken as F64x2Backend>::zero();
        [q, q, q, q]
    }

    #[inline(always)]
    fn load(data: &[f64; 8]) -> [float64x2_t; 4] {
        [
            <archmage::NeonToken as F64x2Backend>::load(data[0..2].try_into().unwrap()),
            <archmage::NeonToken as F64x2Backend>::load(data[2..4].try_into().unwrap()),
            <archmage::NeonToken as F64x2Backend>::load(data[4..6].try_into().unwrap()),
            <archmage::NeonToken as F64x2Backend>::load(data[6..8].try_into().unwrap()),
        ]
    }

    #[inline(always)]
    fn from_array(arr: [f64; 8]) -> [float64x2_t; 4] {
        let mut q0 = [0.0f64; 2];
        let mut q1 = [0.0f64; 2];
        let mut q2 = [0.0f64; 2];
        let mut q3 = [0.0f64; 2];
        q0.copy_from_slice(&arr[0..2]);
        q1.copy_from_slice(&arr[2..4]);
        q2.copy_from_slice(&arr[4..6]);
        q3.copy_from_slice(&arr[6..8]);
        [
            <archmage::NeonToken as F64x2Backend>::from_array(q0),
            <archmage::NeonToken as F64x2Backend>::from_array(q1),
            <archmage::NeonToken as F64x2Backend>::from_array(q2),
            <archmage::NeonToken as F64x2Backend>::from_array(q3),
        ]
    }

    #[inline(always)]
    fn store(repr: [float64x2_t; 4], out: &mut [f64; 8]) {
        let (o01, o23) = out.split_at_mut(4);
        let (o0, o1) = o01.split_at_mut(2);
        let (o2, o3) = o23.split_at_mut(2);
        <archmage::NeonToken as F64x2Backend>::store(repr[0], o0.try_into().unwrap());
        <archmage::NeonToken as F64x2Backend>::store(repr[1], o1.try_into().unwrap());
        <archmage::NeonToken as F64x2Backend>::store(repr[2], o2.try_into().unwrap());
        <archmage::NeonToken as F64x2Backend>::store(repr[3], o3.try_into().unwrap());
    }

    #[inline(always)]
    fn to_array(repr: [float64x2_t; 4]) -> [f64; 8] {
        let a0 = <archmage::NeonToken as F64x2Backend>::to_array(repr[0]);
        let a1 = <archmage::NeonToken as F64x2Backend>::to_array(repr[1]);
        let a2 = <archmage::NeonToken as F64x2Backend>::to_array(repr[2]);
        let a3 = <archmage::NeonToken as F64x2Backend>::to_array(repr[3]);
        let mut out = [0.0f64; 8];
        out[0..2].copy_from_slice(&a0);
        out[2..4].copy_from_slice(&a1);
        out[4..6].copy_from_slice(&a2);
        out[6..8].copy_from_slice(&a3);
        out
    }

    #[inline(always)]
    fn add(a: [float64x2_t; 4], b: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::add(a[i], b[i]))
    }

    #[inline(always)]
    fn sub(a: [float64x2_t; 4], b: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::sub(a[i], b[i]))
    }

    #[inline(always)]
    fn mul(a: [float64x2_t; 4], b: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::mul(a[i], b[i]))
    }

    #[inline(always)]
    fn div(a: [float64x2_t; 4], b: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::div(a[i], b[i]))
    }

    #[inline(always)]
    fn neg(a: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::neg(a[i]))
    }

    #[inline(always)]
    fn min(a: [float64x2_t; 4], b: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::min(a[i], b[i]))
    }

    #[inline(always)]
    fn max(a: [float64x2_t; 4], b: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::max(a[i], b[i]))
    }

    #[inline(always)]
    fn sqrt(a: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::sqrt(a[i]))
    }

    #[inline(always)]
    fn abs(a: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::abs(a[i]))
    }

    #[inline(always)]
    fn floor(a: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::floor(a[i]))
    }

    #[inline(always)]
    fn ceil(a: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::ceil(a[i]))
    }

    #[inline(always)]
    fn round(a: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::round(a[i]))
    }

    #[inline(always)]
    fn mul_add(a: [float64x2_t; 4], b: [float64x2_t; 4], c: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::mul_add(a[i], b[i], c[i]))
    }

    #[inline(always)]
    fn mul_sub(a: [float64x2_t; 4], b: [float64x2_t; 4], c: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::mul_sub(a[i], b[i], c[i]))
    }

    #[inline(always)]
    fn reduce_add(a: [float64x2_t; 4]) -> f64 {
        <archmage::NeonToken as F64x2Backend>::reduce_add(a[0])
            + <archmage::NeonToken as F64x2Backend>::reduce_add(a[1])
            + <archmage::NeonToken as F64x2Backend>::reduce_add(a[2])
            + <archmage::NeonToken as F64x2Backend>::reduce_add(a[3])
    }

    #[inline(always)]
    fn reduce_min(a: [float64x2_t; 4]) -> f64 {
        let m01 = {
            let l = <archmage::NeonToken as F64x2Backend>::reduce_min(a[0]);
            let r = <archmage::NeonToken as F64x2Backend>::reduce_min(a[1]);
            if l < r { l } else { r }
        };
        let m23 = {
            let l = <archmage::NeonToken as F64x2Backend>::reduce_min(a[2]);
            let r = <archmage::NeonToken as F64x2Backend>::reduce_min(a[3]);
            if l < r { l } else { r }
        };
        if m01 < m23 { m01 } else { m23 }
    }

    #[inline(always)]
    fn reduce_max(a: [float64x2_t; 4]) -> f64 {
        let m01 = {
            let l = <archmage::NeonToken as F64x2Backend>::reduce_max(a[0]);
            let r = <archmage::NeonToken as F64x2Backend>::reduce_max(a[1]);
            if l > r { l } else { r }
        };
        let m23 = {
            let l = <archmage::NeonToken as F64x2Backend>::reduce_max(a[2]);
            let r = <archmage::NeonToken as F64x2Backend>::reduce_max(a[3]);
            if l > r { l } else { r }
        };
        if m01 > m23 { m01 } else { m23 }
    }

    #[inline(always)]
    fn simd_eq(a: [float64x2_t; 4], b: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::simd_eq(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ne(a: [float64x2_t; 4], b: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::simd_ne(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_lt(a: [float64x2_t; 4], b: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::simd_lt(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_le(a: [float64x2_t; 4], b: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::simd_le(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_gt(a: [float64x2_t; 4], b: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::simd_gt(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ge(a: [float64x2_t; 4], b: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::simd_ge(a[i], b[i]))
    }

    #[inline(always)]
    fn blend(
        mask: [float64x2_t; 4],
        if_true: [float64x2_t; 4],
        if_false: [float64x2_t; 4],
    ) -> [float64x2_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as F64x2Backend>::blend(mask[i], if_true[i], if_false[i])
        })
    }

    #[inline(always)]
    fn not(a: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::not(a[i]))
    }

    #[inline(always)]
    fn bitand(a: [float64x2_t; 4], b: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::bitand(a[i], b[i]))
    }

    #[inline(always)]
    fn bitor(a: [float64x2_t; 4], b: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::bitor(a[i], b[i]))
    }

    #[inline(always)]
    fn bitxor(a: [float64x2_t; 4], b: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::bitxor(a[i], b[i]))
    }
}

#[cfg(target_arch = "aarch64")]
impl I8x64Backend for archmage::NeonToken {
    type Repr = [int8x16_t; 4];

    #[inline(always)]
    fn splat(v: i8) -> [int8x16_t; 4] {
        let q = <archmage::NeonToken as I8x16Backend>::splat(v);
        [q, q, q, q]
    }

    #[inline(always)]
    fn zero() -> [int8x16_t; 4] {
        let q = <archmage::NeonToken as I8x16Backend>::zero();
        [q, q, q, q]
    }

    #[inline(always)]
    fn load(data: &[i8; 64]) -> [int8x16_t; 4] {
        [
            <archmage::NeonToken as I8x16Backend>::load(data[0..16].try_into().unwrap()),
            <archmage::NeonToken as I8x16Backend>::load(data[16..32].try_into().unwrap()),
            <archmage::NeonToken as I8x16Backend>::load(data[32..48].try_into().unwrap()),
            <archmage::NeonToken as I8x16Backend>::load(data[48..64].try_into().unwrap()),
        ]
    }

    #[inline(always)]
    fn from_array(arr: [i8; 64]) -> [int8x16_t; 4] {
        let mut q0 = [0; 16];
        let mut q1 = [0; 16];
        let mut q2 = [0; 16];
        let mut q3 = [0; 16];
        q0.copy_from_slice(&arr[0..16]);
        q1.copy_from_slice(&arr[16..32]);
        q2.copy_from_slice(&arr[32..48]);
        q3.copy_from_slice(&arr[48..64]);
        [
            <archmage::NeonToken as I8x16Backend>::from_array(q0),
            <archmage::NeonToken as I8x16Backend>::from_array(q1),
            <archmage::NeonToken as I8x16Backend>::from_array(q2),
            <archmage::NeonToken as I8x16Backend>::from_array(q3),
        ]
    }

    #[inline(always)]
    fn store(repr: [int8x16_t; 4], out: &mut [i8; 64]) {
        let (o01, o23) = out.split_at_mut(32);
        let (o0, o1) = o01.split_at_mut(16);
        let (o2, o3) = o23.split_at_mut(16);
        <archmage::NeonToken as I8x16Backend>::store(repr[0], o0.try_into().unwrap());
        <archmage::NeonToken as I8x16Backend>::store(repr[1], o1.try_into().unwrap());
        <archmage::NeonToken as I8x16Backend>::store(repr[2], o2.try_into().unwrap());
        <archmage::NeonToken as I8x16Backend>::store(repr[3], o3.try_into().unwrap());
    }

    #[inline(always)]
    fn to_array(repr: [int8x16_t; 4]) -> [i8; 64] {
        let a0 = <archmage::NeonToken as I8x16Backend>::to_array(repr[0]);
        let a1 = <archmage::NeonToken as I8x16Backend>::to_array(repr[1]);
        let a2 = <archmage::NeonToken as I8x16Backend>::to_array(repr[2]);
        let a3 = <archmage::NeonToken as I8x16Backend>::to_array(repr[3]);
        let mut out = [0; 64];
        out[0..16].copy_from_slice(&a0);
        out[16..32].copy_from_slice(&a1);
        out[32..48].copy_from_slice(&a2);
        out[48..64].copy_from_slice(&a3);
        out
    }

    #[inline(always)]
    fn add(a: [int8x16_t; 4], b: [int8x16_t; 4]) -> [int8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I8x16Backend>::add(a[i], b[i]))
    }

    #[inline(always)]
    fn sub(a: [int8x16_t; 4], b: [int8x16_t; 4]) -> [int8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I8x16Backend>::sub(a[i], b[i]))
    }

    #[inline(always)]
    fn neg(a: [int8x16_t; 4]) -> [int8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I8x16Backend>::neg(a[i]))
    }

    #[inline(always)]
    fn min(a: [int8x16_t; 4], b: [int8x16_t; 4]) -> [int8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I8x16Backend>::min(a[i], b[i]))
    }

    #[inline(always)]
    fn max(a: [int8x16_t; 4], b: [int8x16_t; 4]) -> [int8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I8x16Backend>::max(a[i], b[i]))
    }

    #[inline(always)]
    fn abs(a: [int8x16_t; 4]) -> [int8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I8x16Backend>::abs(a[i]))
    }

    #[inline(always)]
    fn reduce_add(a: [int8x16_t; 4]) -> i8 {
        <archmage::NeonToken as I8x16Backend>::reduce_add(a[0])
            .wrapping_add(<archmage::NeonToken as I8x16Backend>::reduce_add(a[1]))
            .wrapping_add(<archmage::NeonToken as I8x16Backend>::reduce_add(a[2]))
            .wrapping_add(<archmage::NeonToken as I8x16Backend>::reduce_add(a[3]))
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: [int8x16_t; 4]) -> [int8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I8x16Backend>::shl_const::<N>(a[i]))
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: [int8x16_t; 4]) -> [int8x16_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as I8x16Backend>::shr_arithmetic_const::<N>(a[i])
        })
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: [int8x16_t; 4]) -> [int8x16_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as I8x16Backend>::shr_logical_const::<N>(a[i])
        })
    }

    #[inline(always)]
    fn all_true(a: [int8x16_t; 4]) -> bool {
        <archmage::NeonToken as I8x16Backend>::all_true(a[0])
            && <archmage::NeonToken as I8x16Backend>::all_true(a[1])
            && <archmage::NeonToken as I8x16Backend>::all_true(a[2])
            && <archmage::NeonToken as I8x16Backend>::all_true(a[3])
    }

    #[inline(always)]
    fn any_true(a: [int8x16_t; 4]) -> bool {
        <archmage::NeonToken as I8x16Backend>::any_true(a[0])
            || <archmage::NeonToken as I8x16Backend>::any_true(a[1])
            || <archmage::NeonToken as I8x16Backend>::any_true(a[2])
            || <archmage::NeonToken as I8x16Backend>::any_true(a[3])
    }

    #[inline(always)]
    fn bitmask(a: [int8x16_t; 4]) -> u64 {
        let q0 = <archmage::NeonToken as I8x16Backend>::bitmask(a[0]) as u64;
        let q1 = <archmage::NeonToken as I8x16Backend>::bitmask(a[1]) as u64;
        let q2 = <archmage::NeonToken as I8x16Backend>::bitmask(a[2]) as u64;
        let q3 = <archmage::NeonToken as I8x16Backend>::bitmask(a[3]) as u64;
        q0 | (q1 << 16) | (q2 << 32) | (q3 << 48)
    }

    #[inline(always)]
    fn simd_eq(a: [int8x16_t; 4], b: [int8x16_t; 4]) -> [int8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I8x16Backend>::simd_eq(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ne(a: [int8x16_t; 4], b: [int8x16_t; 4]) -> [int8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I8x16Backend>::simd_ne(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_lt(a: [int8x16_t; 4], b: [int8x16_t; 4]) -> [int8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I8x16Backend>::simd_lt(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_le(a: [int8x16_t; 4], b: [int8x16_t; 4]) -> [int8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I8x16Backend>::simd_le(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_gt(a: [int8x16_t; 4], b: [int8x16_t; 4]) -> [int8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I8x16Backend>::simd_gt(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ge(a: [int8x16_t; 4], b: [int8x16_t; 4]) -> [int8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I8x16Backend>::simd_ge(a[i], b[i]))
    }

    #[inline(always)]
    fn blend(
        mask: [int8x16_t; 4],
        if_true: [int8x16_t; 4],
        if_false: [int8x16_t; 4],
    ) -> [int8x16_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as I8x16Backend>::blend(mask[i], if_true[i], if_false[i])
        })
    }

    #[inline(always)]
    fn not(a: [int8x16_t; 4]) -> [int8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I8x16Backend>::not(a[i]))
    }

    #[inline(always)]
    fn bitand(a: [int8x16_t; 4], b: [int8x16_t; 4]) -> [int8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I8x16Backend>::bitand(a[i], b[i]))
    }

    #[inline(always)]
    fn bitor(a: [int8x16_t; 4], b: [int8x16_t; 4]) -> [int8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I8x16Backend>::bitor(a[i], b[i]))
    }

    #[inline(always)]
    fn bitxor(a: [int8x16_t; 4], b: [int8x16_t; 4]) -> [int8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I8x16Backend>::bitxor(a[i], b[i]))
    }
}

#[cfg(target_arch = "aarch64")]
impl U8x64Backend for archmage::NeonToken {
    type Repr = [uint8x16_t; 4];

    #[inline(always)]
    fn splat(v: u8) -> [uint8x16_t; 4] {
        let q = <archmage::NeonToken as U8x16Backend>::splat(v);
        [q, q, q, q]
    }

    #[inline(always)]
    fn zero() -> [uint8x16_t; 4] {
        let q = <archmage::NeonToken as U8x16Backend>::zero();
        [q, q, q, q]
    }

    #[inline(always)]
    fn load(data: &[u8; 64]) -> [uint8x16_t; 4] {
        [
            <archmage::NeonToken as U8x16Backend>::load(data[0..16].try_into().unwrap()),
            <archmage::NeonToken as U8x16Backend>::load(data[16..32].try_into().unwrap()),
            <archmage::NeonToken as U8x16Backend>::load(data[32..48].try_into().unwrap()),
            <archmage::NeonToken as U8x16Backend>::load(data[48..64].try_into().unwrap()),
        ]
    }

    #[inline(always)]
    fn from_array(arr: [u8; 64]) -> [uint8x16_t; 4] {
        let mut q0 = [0; 16];
        let mut q1 = [0; 16];
        let mut q2 = [0; 16];
        let mut q3 = [0; 16];
        q0.copy_from_slice(&arr[0..16]);
        q1.copy_from_slice(&arr[16..32]);
        q2.copy_from_slice(&arr[32..48]);
        q3.copy_from_slice(&arr[48..64]);
        [
            <archmage::NeonToken as U8x16Backend>::from_array(q0),
            <archmage::NeonToken as U8x16Backend>::from_array(q1),
            <archmage::NeonToken as U8x16Backend>::from_array(q2),
            <archmage::NeonToken as U8x16Backend>::from_array(q3),
        ]
    }

    #[inline(always)]
    fn store(repr: [uint8x16_t; 4], out: &mut [u8; 64]) {
        let (o01, o23) = out.split_at_mut(32);
        let (o0, o1) = o01.split_at_mut(16);
        let (o2, o3) = o23.split_at_mut(16);
        <archmage::NeonToken as U8x16Backend>::store(repr[0], o0.try_into().unwrap());
        <archmage::NeonToken as U8x16Backend>::store(repr[1], o1.try_into().unwrap());
        <archmage::NeonToken as U8x16Backend>::store(repr[2], o2.try_into().unwrap());
        <archmage::NeonToken as U8x16Backend>::store(repr[3], o3.try_into().unwrap());
    }

    #[inline(always)]
    fn to_array(repr: [uint8x16_t; 4]) -> [u8; 64] {
        let a0 = <archmage::NeonToken as U8x16Backend>::to_array(repr[0]);
        let a1 = <archmage::NeonToken as U8x16Backend>::to_array(repr[1]);
        let a2 = <archmage::NeonToken as U8x16Backend>::to_array(repr[2]);
        let a3 = <archmage::NeonToken as U8x16Backend>::to_array(repr[3]);
        let mut out = [0; 64];
        out[0..16].copy_from_slice(&a0);
        out[16..32].copy_from_slice(&a1);
        out[32..48].copy_from_slice(&a2);
        out[48..64].copy_from_slice(&a3);
        out
    }

    #[inline(always)]
    fn add(a: [uint8x16_t; 4], b: [uint8x16_t; 4]) -> [uint8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U8x16Backend>::add(a[i], b[i]))
    }

    #[inline(always)]
    fn sub(a: [uint8x16_t; 4], b: [uint8x16_t; 4]) -> [uint8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U8x16Backend>::sub(a[i], b[i]))
    }

    #[inline(always)]
    fn neg(a: [uint8x16_t; 4]) -> [uint8x16_t; 4] {
        let z = <archmage::NeonToken as U8x16Backend>::zero();
        core::array::from_fn(|i| <archmage::NeonToken as U8x16Backend>::sub(z, a[i]))
    }

    #[inline(always)]
    fn min(a: [uint8x16_t; 4], b: [uint8x16_t; 4]) -> [uint8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U8x16Backend>::min(a[i], b[i]))
    }

    #[inline(always)]
    fn max(a: [uint8x16_t; 4], b: [uint8x16_t; 4]) -> [uint8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U8x16Backend>::max(a[i], b[i]))
    }

    #[inline(always)]
    fn reduce_add(a: [uint8x16_t; 4]) -> u8 {
        <archmage::NeonToken as U8x16Backend>::reduce_add(a[0])
            .wrapping_add(<archmage::NeonToken as U8x16Backend>::reduce_add(a[1]))
            .wrapping_add(<archmage::NeonToken as U8x16Backend>::reduce_add(a[2]))
            .wrapping_add(<archmage::NeonToken as U8x16Backend>::reduce_add(a[3]))
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: [uint8x16_t; 4]) -> [uint8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U8x16Backend>::shl_const::<N>(a[i]))
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: [uint8x16_t; 4]) -> [uint8x16_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as U8x16Backend>::shr_logical_const::<N>(a[i])
        })
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: [uint8x16_t; 4]) -> [uint8x16_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as U8x16Backend>::shr_logical_const::<N>(a[i])
        })
    }

    #[inline(always)]
    fn all_true(a: [uint8x16_t; 4]) -> bool {
        <archmage::NeonToken as U8x16Backend>::all_true(a[0])
            && <archmage::NeonToken as U8x16Backend>::all_true(a[1])
            && <archmage::NeonToken as U8x16Backend>::all_true(a[2])
            && <archmage::NeonToken as U8x16Backend>::all_true(a[3])
    }

    #[inline(always)]
    fn any_true(a: [uint8x16_t; 4]) -> bool {
        <archmage::NeonToken as U8x16Backend>::any_true(a[0])
            || <archmage::NeonToken as U8x16Backend>::any_true(a[1])
            || <archmage::NeonToken as U8x16Backend>::any_true(a[2])
            || <archmage::NeonToken as U8x16Backend>::any_true(a[3])
    }

    #[inline(always)]
    fn bitmask(a: [uint8x16_t; 4]) -> u64 {
        let q0 = <archmage::NeonToken as U8x16Backend>::bitmask(a[0]) as u64;
        let q1 = <archmage::NeonToken as U8x16Backend>::bitmask(a[1]) as u64;
        let q2 = <archmage::NeonToken as U8x16Backend>::bitmask(a[2]) as u64;
        let q3 = <archmage::NeonToken as U8x16Backend>::bitmask(a[3]) as u64;
        q0 | (q1 << 16) | (q2 << 32) | (q3 << 48)
    }

    #[inline(always)]
    fn simd_eq(a: [uint8x16_t; 4], b: [uint8x16_t; 4]) -> [uint8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U8x16Backend>::simd_eq(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ne(a: [uint8x16_t; 4], b: [uint8x16_t; 4]) -> [uint8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U8x16Backend>::simd_ne(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_lt(a: [uint8x16_t; 4], b: [uint8x16_t; 4]) -> [uint8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U8x16Backend>::simd_lt(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_le(a: [uint8x16_t; 4], b: [uint8x16_t; 4]) -> [uint8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U8x16Backend>::simd_le(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_gt(a: [uint8x16_t; 4], b: [uint8x16_t; 4]) -> [uint8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U8x16Backend>::simd_gt(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ge(a: [uint8x16_t; 4], b: [uint8x16_t; 4]) -> [uint8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U8x16Backend>::simd_ge(a[i], b[i]))
    }

    #[inline(always)]
    fn blend(
        mask: [uint8x16_t; 4],
        if_true: [uint8x16_t; 4],
        if_false: [uint8x16_t; 4],
    ) -> [uint8x16_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as U8x16Backend>::blend(mask[i], if_true[i], if_false[i])
        })
    }

    #[inline(always)]
    fn not(a: [uint8x16_t; 4]) -> [uint8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U8x16Backend>::not(a[i]))
    }

    #[inline(always)]
    fn bitand(a: [uint8x16_t; 4], b: [uint8x16_t; 4]) -> [uint8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U8x16Backend>::bitand(a[i], b[i]))
    }

    #[inline(always)]
    fn bitor(a: [uint8x16_t; 4], b: [uint8x16_t; 4]) -> [uint8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U8x16Backend>::bitor(a[i], b[i]))
    }

    #[inline(always)]
    fn bitxor(a: [uint8x16_t; 4], b: [uint8x16_t; 4]) -> [uint8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U8x16Backend>::bitxor(a[i], b[i]))
    }
}

#[cfg(target_arch = "aarch64")]
impl I16x32Backend for archmage::NeonToken {
    type Repr = [int16x8_t; 4];

    #[inline(always)]
    fn splat(v: i16) -> [int16x8_t; 4] {
        let q = <archmage::NeonToken as I16x8Backend>::splat(v);
        [q, q, q, q]
    }

    #[inline(always)]
    fn zero() -> [int16x8_t; 4] {
        let q = <archmage::NeonToken as I16x8Backend>::zero();
        [q, q, q, q]
    }

    #[inline(always)]
    fn load(data: &[i16; 32]) -> [int16x8_t; 4] {
        [
            <archmage::NeonToken as I16x8Backend>::load(data[0..8].try_into().unwrap()),
            <archmage::NeonToken as I16x8Backend>::load(data[8..16].try_into().unwrap()),
            <archmage::NeonToken as I16x8Backend>::load(data[16..24].try_into().unwrap()),
            <archmage::NeonToken as I16x8Backend>::load(data[24..32].try_into().unwrap()),
        ]
    }

    #[inline(always)]
    fn from_array(arr: [i16; 32]) -> [int16x8_t; 4] {
        let mut q0 = [0; 8];
        let mut q1 = [0; 8];
        let mut q2 = [0; 8];
        let mut q3 = [0; 8];
        q0.copy_from_slice(&arr[0..8]);
        q1.copy_from_slice(&arr[8..16]);
        q2.copy_from_slice(&arr[16..24]);
        q3.copy_from_slice(&arr[24..32]);
        [
            <archmage::NeonToken as I16x8Backend>::from_array(q0),
            <archmage::NeonToken as I16x8Backend>::from_array(q1),
            <archmage::NeonToken as I16x8Backend>::from_array(q2),
            <archmage::NeonToken as I16x8Backend>::from_array(q3),
        ]
    }

    #[inline(always)]
    fn store(repr: [int16x8_t; 4], out: &mut [i16; 32]) {
        let (o01, o23) = out.split_at_mut(16);
        let (o0, o1) = o01.split_at_mut(8);
        let (o2, o3) = o23.split_at_mut(8);
        <archmage::NeonToken as I16x8Backend>::store(repr[0], o0.try_into().unwrap());
        <archmage::NeonToken as I16x8Backend>::store(repr[1], o1.try_into().unwrap());
        <archmage::NeonToken as I16x8Backend>::store(repr[2], o2.try_into().unwrap());
        <archmage::NeonToken as I16x8Backend>::store(repr[3], o3.try_into().unwrap());
    }

    #[inline(always)]
    fn to_array(repr: [int16x8_t; 4]) -> [i16; 32] {
        let a0 = <archmage::NeonToken as I16x8Backend>::to_array(repr[0]);
        let a1 = <archmage::NeonToken as I16x8Backend>::to_array(repr[1]);
        let a2 = <archmage::NeonToken as I16x8Backend>::to_array(repr[2]);
        let a3 = <archmage::NeonToken as I16x8Backend>::to_array(repr[3]);
        let mut out = [0; 32];
        out[0..8].copy_from_slice(&a0);
        out[8..16].copy_from_slice(&a1);
        out[16..24].copy_from_slice(&a2);
        out[24..32].copy_from_slice(&a3);
        out
    }

    #[inline(always)]
    fn add(a: [int16x8_t; 4], b: [int16x8_t; 4]) -> [int16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I16x8Backend>::add(a[i], b[i]))
    }

    #[inline(always)]
    fn sub(a: [int16x8_t; 4], b: [int16x8_t; 4]) -> [int16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I16x8Backend>::sub(a[i], b[i]))
    }

    #[inline(always)]
    fn mul(a: [int16x8_t; 4], b: [int16x8_t; 4]) -> [int16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I16x8Backend>::mul(a[i], b[i]))
    }

    #[inline(always)]
    fn neg(a: [int16x8_t; 4]) -> [int16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I16x8Backend>::neg(a[i]))
    }

    #[inline(always)]
    fn min(a: [int16x8_t; 4], b: [int16x8_t; 4]) -> [int16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I16x8Backend>::min(a[i], b[i]))
    }

    #[inline(always)]
    fn max(a: [int16x8_t; 4], b: [int16x8_t; 4]) -> [int16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I16x8Backend>::max(a[i], b[i]))
    }

    #[inline(always)]
    fn abs(a: [int16x8_t; 4]) -> [int16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I16x8Backend>::abs(a[i]))
    }

    #[inline(always)]
    fn reduce_add(a: [int16x8_t; 4]) -> i16 {
        <archmage::NeonToken as I16x8Backend>::reduce_add(a[0])
            .wrapping_add(<archmage::NeonToken as I16x8Backend>::reduce_add(a[1]))
            .wrapping_add(<archmage::NeonToken as I16x8Backend>::reduce_add(a[2]))
            .wrapping_add(<archmage::NeonToken as I16x8Backend>::reduce_add(a[3]))
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: [int16x8_t; 4]) -> [int16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I16x8Backend>::shl_const::<N>(a[i]))
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: [int16x8_t; 4]) -> [int16x8_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as I16x8Backend>::shr_arithmetic_const::<N>(a[i])
        })
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: [int16x8_t; 4]) -> [int16x8_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as I16x8Backend>::shr_logical_const::<N>(a[i])
        })
    }

    #[inline(always)]
    fn all_true(a: [int16x8_t; 4]) -> bool {
        <archmage::NeonToken as I16x8Backend>::all_true(a[0])
            && <archmage::NeonToken as I16x8Backend>::all_true(a[1])
            && <archmage::NeonToken as I16x8Backend>::all_true(a[2])
            && <archmage::NeonToken as I16x8Backend>::all_true(a[3])
    }

    #[inline(always)]
    fn any_true(a: [int16x8_t; 4]) -> bool {
        <archmage::NeonToken as I16x8Backend>::any_true(a[0])
            || <archmage::NeonToken as I16x8Backend>::any_true(a[1])
            || <archmage::NeonToken as I16x8Backend>::any_true(a[2])
            || <archmage::NeonToken as I16x8Backend>::any_true(a[3])
    }

    #[inline(always)]
    fn bitmask(a: [int16x8_t; 4]) -> u64 {
        let q0 = <archmage::NeonToken as I16x8Backend>::bitmask(a[0]) as u64;
        let q1 = <archmage::NeonToken as I16x8Backend>::bitmask(a[1]) as u64;
        let q2 = <archmage::NeonToken as I16x8Backend>::bitmask(a[2]) as u64;
        let q3 = <archmage::NeonToken as I16x8Backend>::bitmask(a[3]) as u64;
        q0 | (q1 << 8) | (q2 << 16) | (q3 << 24)
    }

    #[inline(always)]
    fn simd_eq(a: [int16x8_t; 4], b: [int16x8_t; 4]) -> [int16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I16x8Backend>::simd_eq(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ne(a: [int16x8_t; 4], b: [int16x8_t; 4]) -> [int16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I16x8Backend>::simd_ne(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_lt(a: [int16x8_t; 4], b: [int16x8_t; 4]) -> [int16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I16x8Backend>::simd_lt(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_le(a: [int16x8_t; 4], b: [int16x8_t; 4]) -> [int16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I16x8Backend>::simd_le(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_gt(a: [int16x8_t; 4], b: [int16x8_t; 4]) -> [int16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I16x8Backend>::simd_gt(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ge(a: [int16x8_t; 4], b: [int16x8_t; 4]) -> [int16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I16x8Backend>::simd_ge(a[i], b[i]))
    }

    #[inline(always)]
    fn blend(
        mask: [int16x8_t; 4],
        if_true: [int16x8_t; 4],
        if_false: [int16x8_t; 4],
    ) -> [int16x8_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as I16x8Backend>::blend(mask[i], if_true[i], if_false[i])
        })
    }

    #[inline(always)]
    fn not(a: [int16x8_t; 4]) -> [int16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I16x8Backend>::not(a[i]))
    }

    #[inline(always)]
    fn bitand(a: [int16x8_t; 4], b: [int16x8_t; 4]) -> [int16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I16x8Backend>::bitand(a[i], b[i]))
    }

    #[inline(always)]
    fn bitor(a: [int16x8_t; 4], b: [int16x8_t; 4]) -> [int16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I16x8Backend>::bitor(a[i], b[i]))
    }

    #[inline(always)]
    fn bitxor(a: [int16x8_t; 4], b: [int16x8_t; 4]) -> [int16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I16x8Backend>::bitxor(a[i], b[i]))
    }
}

#[cfg(target_arch = "aarch64")]
impl U16x32Backend for archmage::NeonToken {
    type Repr = [uint16x8_t; 4];

    #[inline(always)]
    fn splat(v: u16) -> [uint16x8_t; 4] {
        let q = <archmage::NeonToken as U16x8Backend>::splat(v);
        [q, q, q, q]
    }

    #[inline(always)]
    fn zero() -> [uint16x8_t; 4] {
        let q = <archmage::NeonToken as U16x8Backend>::zero();
        [q, q, q, q]
    }

    #[inline(always)]
    fn load(data: &[u16; 32]) -> [uint16x8_t; 4] {
        [
            <archmage::NeonToken as U16x8Backend>::load(data[0..8].try_into().unwrap()),
            <archmage::NeonToken as U16x8Backend>::load(data[8..16].try_into().unwrap()),
            <archmage::NeonToken as U16x8Backend>::load(data[16..24].try_into().unwrap()),
            <archmage::NeonToken as U16x8Backend>::load(data[24..32].try_into().unwrap()),
        ]
    }

    #[inline(always)]
    fn from_array(arr: [u16; 32]) -> [uint16x8_t; 4] {
        let mut q0 = [0; 8];
        let mut q1 = [0; 8];
        let mut q2 = [0; 8];
        let mut q3 = [0; 8];
        q0.copy_from_slice(&arr[0..8]);
        q1.copy_from_slice(&arr[8..16]);
        q2.copy_from_slice(&arr[16..24]);
        q3.copy_from_slice(&arr[24..32]);
        [
            <archmage::NeonToken as U16x8Backend>::from_array(q0),
            <archmage::NeonToken as U16x8Backend>::from_array(q1),
            <archmage::NeonToken as U16x8Backend>::from_array(q2),
            <archmage::NeonToken as U16x8Backend>::from_array(q3),
        ]
    }

    #[inline(always)]
    fn store(repr: [uint16x8_t; 4], out: &mut [u16; 32]) {
        let (o01, o23) = out.split_at_mut(16);
        let (o0, o1) = o01.split_at_mut(8);
        let (o2, o3) = o23.split_at_mut(8);
        <archmage::NeonToken as U16x8Backend>::store(repr[0], o0.try_into().unwrap());
        <archmage::NeonToken as U16x8Backend>::store(repr[1], o1.try_into().unwrap());
        <archmage::NeonToken as U16x8Backend>::store(repr[2], o2.try_into().unwrap());
        <archmage::NeonToken as U16x8Backend>::store(repr[3], o3.try_into().unwrap());
    }

    #[inline(always)]
    fn to_array(repr: [uint16x8_t; 4]) -> [u16; 32] {
        let a0 = <archmage::NeonToken as U16x8Backend>::to_array(repr[0]);
        let a1 = <archmage::NeonToken as U16x8Backend>::to_array(repr[1]);
        let a2 = <archmage::NeonToken as U16x8Backend>::to_array(repr[2]);
        let a3 = <archmage::NeonToken as U16x8Backend>::to_array(repr[3]);
        let mut out = [0; 32];
        out[0..8].copy_from_slice(&a0);
        out[8..16].copy_from_slice(&a1);
        out[16..24].copy_from_slice(&a2);
        out[24..32].copy_from_slice(&a3);
        out
    }

    #[inline(always)]
    fn add(a: [uint16x8_t; 4], b: [uint16x8_t; 4]) -> [uint16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U16x8Backend>::add(a[i], b[i]))
    }

    #[inline(always)]
    fn sub(a: [uint16x8_t; 4], b: [uint16x8_t; 4]) -> [uint16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U16x8Backend>::sub(a[i], b[i]))
    }

    #[inline(always)]
    fn mul(a: [uint16x8_t; 4], b: [uint16x8_t; 4]) -> [uint16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U16x8Backend>::mul(a[i], b[i]))
    }

    #[inline(always)]
    fn neg(a: [uint16x8_t; 4]) -> [uint16x8_t; 4] {
        let z = <archmage::NeonToken as U16x8Backend>::zero();
        core::array::from_fn(|i| <archmage::NeonToken as U16x8Backend>::sub(z, a[i]))
    }

    #[inline(always)]
    fn min(a: [uint16x8_t; 4], b: [uint16x8_t; 4]) -> [uint16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U16x8Backend>::min(a[i], b[i]))
    }

    #[inline(always)]
    fn max(a: [uint16x8_t; 4], b: [uint16x8_t; 4]) -> [uint16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U16x8Backend>::max(a[i], b[i]))
    }

    #[inline(always)]
    fn reduce_add(a: [uint16x8_t; 4]) -> u16 {
        <archmage::NeonToken as U16x8Backend>::reduce_add(a[0])
            .wrapping_add(<archmage::NeonToken as U16x8Backend>::reduce_add(a[1]))
            .wrapping_add(<archmage::NeonToken as U16x8Backend>::reduce_add(a[2]))
            .wrapping_add(<archmage::NeonToken as U16x8Backend>::reduce_add(a[3]))
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: [uint16x8_t; 4]) -> [uint16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U16x8Backend>::shl_const::<N>(a[i]))
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: [uint16x8_t; 4]) -> [uint16x8_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as U16x8Backend>::shr_logical_const::<N>(a[i])
        })
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: [uint16x8_t; 4]) -> [uint16x8_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as U16x8Backend>::shr_logical_const::<N>(a[i])
        })
    }

    #[inline(always)]
    fn all_true(a: [uint16x8_t; 4]) -> bool {
        <archmage::NeonToken as U16x8Backend>::all_true(a[0])
            && <archmage::NeonToken as U16x8Backend>::all_true(a[1])
            && <archmage::NeonToken as U16x8Backend>::all_true(a[2])
            && <archmage::NeonToken as U16x8Backend>::all_true(a[3])
    }

    #[inline(always)]
    fn any_true(a: [uint16x8_t; 4]) -> bool {
        <archmage::NeonToken as U16x8Backend>::any_true(a[0])
            || <archmage::NeonToken as U16x8Backend>::any_true(a[1])
            || <archmage::NeonToken as U16x8Backend>::any_true(a[2])
            || <archmage::NeonToken as U16x8Backend>::any_true(a[3])
    }

    #[inline(always)]
    fn bitmask(a: [uint16x8_t; 4]) -> u64 {
        let q0 = <archmage::NeonToken as U16x8Backend>::bitmask(a[0]) as u64;
        let q1 = <archmage::NeonToken as U16x8Backend>::bitmask(a[1]) as u64;
        let q2 = <archmage::NeonToken as U16x8Backend>::bitmask(a[2]) as u64;
        let q3 = <archmage::NeonToken as U16x8Backend>::bitmask(a[3]) as u64;
        q0 | (q1 << 8) | (q2 << 16) | (q3 << 24)
    }

    #[inline(always)]
    fn simd_eq(a: [uint16x8_t; 4], b: [uint16x8_t; 4]) -> [uint16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U16x8Backend>::simd_eq(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ne(a: [uint16x8_t; 4], b: [uint16x8_t; 4]) -> [uint16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U16x8Backend>::simd_ne(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_lt(a: [uint16x8_t; 4], b: [uint16x8_t; 4]) -> [uint16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U16x8Backend>::simd_lt(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_le(a: [uint16x8_t; 4], b: [uint16x8_t; 4]) -> [uint16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U16x8Backend>::simd_le(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_gt(a: [uint16x8_t; 4], b: [uint16x8_t; 4]) -> [uint16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U16x8Backend>::simd_gt(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ge(a: [uint16x8_t; 4], b: [uint16x8_t; 4]) -> [uint16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U16x8Backend>::simd_ge(a[i], b[i]))
    }

    #[inline(always)]
    fn blend(
        mask: [uint16x8_t; 4],
        if_true: [uint16x8_t; 4],
        if_false: [uint16x8_t; 4],
    ) -> [uint16x8_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as U16x8Backend>::blend(mask[i], if_true[i], if_false[i])
        })
    }

    #[inline(always)]
    fn not(a: [uint16x8_t; 4]) -> [uint16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U16x8Backend>::not(a[i]))
    }

    #[inline(always)]
    fn bitand(a: [uint16x8_t; 4], b: [uint16x8_t; 4]) -> [uint16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U16x8Backend>::bitand(a[i], b[i]))
    }

    #[inline(always)]
    fn bitor(a: [uint16x8_t; 4], b: [uint16x8_t; 4]) -> [uint16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U16x8Backend>::bitor(a[i], b[i]))
    }

    #[inline(always)]
    fn bitxor(a: [uint16x8_t; 4], b: [uint16x8_t; 4]) -> [uint16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U16x8Backend>::bitxor(a[i], b[i]))
    }
}

#[cfg(target_arch = "aarch64")]
impl I32x16Backend for archmage::NeonToken {
    type Repr = [int32x4_t; 4];

    #[inline(always)]
    fn splat(v: i32) -> [int32x4_t; 4] {
        let q = <archmage::NeonToken as I32x4Backend>::splat(v);
        [q, q, q, q]
    }

    #[inline(always)]
    fn zero() -> [int32x4_t; 4] {
        let q = <archmage::NeonToken as I32x4Backend>::zero();
        [q, q, q, q]
    }

    #[inline(always)]
    fn load(data: &[i32; 16]) -> [int32x4_t; 4] {
        [
            <archmage::NeonToken as I32x4Backend>::load(data[0..4].try_into().unwrap()),
            <archmage::NeonToken as I32x4Backend>::load(data[4..8].try_into().unwrap()),
            <archmage::NeonToken as I32x4Backend>::load(data[8..12].try_into().unwrap()),
            <archmage::NeonToken as I32x4Backend>::load(data[12..16].try_into().unwrap()),
        ]
    }

    #[inline(always)]
    fn from_array(arr: [i32; 16]) -> [int32x4_t; 4] {
        let mut q0 = [0; 4];
        let mut q1 = [0; 4];
        let mut q2 = [0; 4];
        let mut q3 = [0; 4];
        q0.copy_from_slice(&arr[0..4]);
        q1.copy_from_slice(&arr[4..8]);
        q2.copy_from_slice(&arr[8..12]);
        q3.copy_from_slice(&arr[12..16]);
        [
            <archmage::NeonToken as I32x4Backend>::from_array(q0),
            <archmage::NeonToken as I32x4Backend>::from_array(q1),
            <archmage::NeonToken as I32x4Backend>::from_array(q2),
            <archmage::NeonToken as I32x4Backend>::from_array(q3),
        ]
    }

    #[inline(always)]
    fn store(repr: [int32x4_t; 4], out: &mut [i32; 16]) {
        let (o01, o23) = out.split_at_mut(8);
        let (o0, o1) = o01.split_at_mut(4);
        let (o2, o3) = o23.split_at_mut(4);
        <archmage::NeonToken as I32x4Backend>::store(repr[0], o0.try_into().unwrap());
        <archmage::NeonToken as I32x4Backend>::store(repr[1], o1.try_into().unwrap());
        <archmage::NeonToken as I32x4Backend>::store(repr[2], o2.try_into().unwrap());
        <archmage::NeonToken as I32x4Backend>::store(repr[3], o3.try_into().unwrap());
    }

    #[inline(always)]
    fn to_array(repr: [int32x4_t; 4]) -> [i32; 16] {
        let a0 = <archmage::NeonToken as I32x4Backend>::to_array(repr[0]);
        let a1 = <archmage::NeonToken as I32x4Backend>::to_array(repr[1]);
        let a2 = <archmage::NeonToken as I32x4Backend>::to_array(repr[2]);
        let a3 = <archmage::NeonToken as I32x4Backend>::to_array(repr[3]);
        let mut out = [0; 16];
        out[0..4].copy_from_slice(&a0);
        out[4..8].copy_from_slice(&a1);
        out[8..12].copy_from_slice(&a2);
        out[12..16].copy_from_slice(&a3);
        out
    }

    #[inline(always)]
    fn add(a: [int32x4_t; 4], b: [int32x4_t; 4]) -> [int32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I32x4Backend>::add(a[i], b[i]))
    }

    #[inline(always)]
    fn sub(a: [int32x4_t; 4], b: [int32x4_t; 4]) -> [int32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I32x4Backend>::sub(a[i], b[i]))
    }

    #[inline(always)]
    fn mul(a: [int32x4_t; 4], b: [int32x4_t; 4]) -> [int32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I32x4Backend>::mul(a[i], b[i]))
    }

    #[inline(always)]
    fn neg(a: [int32x4_t; 4]) -> [int32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I32x4Backend>::neg(a[i]))
    }

    #[inline(always)]
    fn min(a: [int32x4_t; 4], b: [int32x4_t; 4]) -> [int32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I32x4Backend>::min(a[i], b[i]))
    }

    #[inline(always)]
    fn max(a: [int32x4_t; 4], b: [int32x4_t; 4]) -> [int32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I32x4Backend>::max(a[i], b[i]))
    }

    #[inline(always)]
    fn abs(a: [int32x4_t; 4]) -> [int32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I32x4Backend>::abs(a[i]))
    }

    #[inline(always)]
    fn reduce_add(a: [int32x4_t; 4]) -> i32 {
        <archmage::NeonToken as I32x4Backend>::reduce_add(a[0])
            .wrapping_add(<archmage::NeonToken as I32x4Backend>::reduce_add(a[1]))
            .wrapping_add(<archmage::NeonToken as I32x4Backend>::reduce_add(a[2]))
            .wrapping_add(<archmage::NeonToken as I32x4Backend>::reduce_add(a[3]))
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: [int32x4_t; 4]) -> [int32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I32x4Backend>::shl_const::<N>(a[i]))
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: [int32x4_t; 4]) -> [int32x4_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as I32x4Backend>::shr_arithmetic_const::<N>(a[i])
        })
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: [int32x4_t; 4]) -> [int32x4_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as I32x4Backend>::shr_logical_const::<N>(a[i])
        })
    }

    #[inline(always)]
    fn all_true(a: [int32x4_t; 4]) -> bool {
        <archmage::NeonToken as I32x4Backend>::all_true(a[0])
            && <archmage::NeonToken as I32x4Backend>::all_true(a[1])
            && <archmage::NeonToken as I32x4Backend>::all_true(a[2])
            && <archmage::NeonToken as I32x4Backend>::all_true(a[3])
    }

    #[inline(always)]
    fn any_true(a: [int32x4_t; 4]) -> bool {
        <archmage::NeonToken as I32x4Backend>::any_true(a[0])
            || <archmage::NeonToken as I32x4Backend>::any_true(a[1])
            || <archmage::NeonToken as I32x4Backend>::any_true(a[2])
            || <archmage::NeonToken as I32x4Backend>::any_true(a[3])
    }

    #[inline(always)]
    fn bitmask(a: [int32x4_t; 4]) -> u64 {
        let q0 = <archmage::NeonToken as I32x4Backend>::bitmask(a[0]) as u64;
        let q1 = <archmage::NeonToken as I32x4Backend>::bitmask(a[1]) as u64;
        let q2 = <archmage::NeonToken as I32x4Backend>::bitmask(a[2]) as u64;
        let q3 = <archmage::NeonToken as I32x4Backend>::bitmask(a[3]) as u64;
        q0 | (q1 << 4) | (q2 << 8) | (q3 << 12)
    }

    #[inline(always)]
    fn simd_eq(a: [int32x4_t; 4], b: [int32x4_t; 4]) -> [int32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I32x4Backend>::simd_eq(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ne(a: [int32x4_t; 4], b: [int32x4_t; 4]) -> [int32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I32x4Backend>::simd_ne(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_lt(a: [int32x4_t; 4], b: [int32x4_t; 4]) -> [int32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I32x4Backend>::simd_lt(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_le(a: [int32x4_t; 4], b: [int32x4_t; 4]) -> [int32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I32x4Backend>::simd_le(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_gt(a: [int32x4_t; 4], b: [int32x4_t; 4]) -> [int32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I32x4Backend>::simd_gt(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ge(a: [int32x4_t; 4], b: [int32x4_t; 4]) -> [int32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I32x4Backend>::simd_ge(a[i], b[i]))
    }

    #[inline(always)]
    fn blend(
        mask: [int32x4_t; 4],
        if_true: [int32x4_t; 4],
        if_false: [int32x4_t; 4],
    ) -> [int32x4_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as I32x4Backend>::blend(mask[i], if_true[i], if_false[i])
        })
    }

    #[inline(always)]
    fn not(a: [int32x4_t; 4]) -> [int32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I32x4Backend>::not(a[i]))
    }

    #[inline(always)]
    fn bitand(a: [int32x4_t; 4], b: [int32x4_t; 4]) -> [int32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I32x4Backend>::bitand(a[i], b[i]))
    }

    #[inline(always)]
    fn bitor(a: [int32x4_t; 4], b: [int32x4_t; 4]) -> [int32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I32x4Backend>::bitor(a[i], b[i]))
    }

    #[inline(always)]
    fn bitxor(a: [int32x4_t; 4], b: [int32x4_t; 4]) -> [int32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I32x4Backend>::bitxor(a[i], b[i]))
    }
}

#[cfg(target_arch = "aarch64")]
impl U32x16Backend for archmage::NeonToken {
    type Repr = [uint32x4_t; 4];

    #[inline(always)]
    fn splat(v: u32) -> [uint32x4_t; 4] {
        let q = <archmage::NeonToken as U32x4Backend>::splat(v);
        [q, q, q, q]
    }

    #[inline(always)]
    fn zero() -> [uint32x4_t; 4] {
        let q = <archmage::NeonToken as U32x4Backend>::zero();
        [q, q, q, q]
    }

    #[inline(always)]
    fn load(data: &[u32; 16]) -> [uint32x4_t; 4] {
        [
            <archmage::NeonToken as U32x4Backend>::load(data[0..4].try_into().unwrap()),
            <archmage::NeonToken as U32x4Backend>::load(data[4..8].try_into().unwrap()),
            <archmage::NeonToken as U32x4Backend>::load(data[8..12].try_into().unwrap()),
            <archmage::NeonToken as U32x4Backend>::load(data[12..16].try_into().unwrap()),
        ]
    }

    #[inline(always)]
    fn from_array(arr: [u32; 16]) -> [uint32x4_t; 4] {
        let mut q0 = [0; 4];
        let mut q1 = [0; 4];
        let mut q2 = [0; 4];
        let mut q3 = [0; 4];
        q0.copy_from_slice(&arr[0..4]);
        q1.copy_from_slice(&arr[4..8]);
        q2.copy_from_slice(&arr[8..12]);
        q3.copy_from_slice(&arr[12..16]);
        [
            <archmage::NeonToken as U32x4Backend>::from_array(q0),
            <archmage::NeonToken as U32x4Backend>::from_array(q1),
            <archmage::NeonToken as U32x4Backend>::from_array(q2),
            <archmage::NeonToken as U32x4Backend>::from_array(q3),
        ]
    }

    #[inline(always)]
    fn store(repr: [uint32x4_t; 4], out: &mut [u32; 16]) {
        let (o01, o23) = out.split_at_mut(8);
        let (o0, o1) = o01.split_at_mut(4);
        let (o2, o3) = o23.split_at_mut(4);
        <archmage::NeonToken as U32x4Backend>::store(repr[0], o0.try_into().unwrap());
        <archmage::NeonToken as U32x4Backend>::store(repr[1], o1.try_into().unwrap());
        <archmage::NeonToken as U32x4Backend>::store(repr[2], o2.try_into().unwrap());
        <archmage::NeonToken as U32x4Backend>::store(repr[3], o3.try_into().unwrap());
    }

    #[inline(always)]
    fn to_array(repr: [uint32x4_t; 4]) -> [u32; 16] {
        let a0 = <archmage::NeonToken as U32x4Backend>::to_array(repr[0]);
        let a1 = <archmage::NeonToken as U32x4Backend>::to_array(repr[1]);
        let a2 = <archmage::NeonToken as U32x4Backend>::to_array(repr[2]);
        let a3 = <archmage::NeonToken as U32x4Backend>::to_array(repr[3]);
        let mut out = [0; 16];
        out[0..4].copy_from_slice(&a0);
        out[4..8].copy_from_slice(&a1);
        out[8..12].copy_from_slice(&a2);
        out[12..16].copy_from_slice(&a3);
        out
    }

    #[inline(always)]
    fn add(a: [uint32x4_t; 4], b: [uint32x4_t; 4]) -> [uint32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U32x4Backend>::add(a[i], b[i]))
    }

    #[inline(always)]
    fn sub(a: [uint32x4_t; 4], b: [uint32x4_t; 4]) -> [uint32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U32x4Backend>::sub(a[i], b[i]))
    }

    #[inline(always)]
    fn mul(a: [uint32x4_t; 4], b: [uint32x4_t; 4]) -> [uint32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U32x4Backend>::mul(a[i], b[i]))
    }

    #[inline(always)]
    fn neg(a: [uint32x4_t; 4]) -> [uint32x4_t; 4] {
        let z = <archmage::NeonToken as U32x4Backend>::zero();
        core::array::from_fn(|i| <archmage::NeonToken as U32x4Backend>::sub(z, a[i]))
    }

    #[inline(always)]
    fn min(a: [uint32x4_t; 4], b: [uint32x4_t; 4]) -> [uint32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U32x4Backend>::min(a[i], b[i]))
    }

    #[inline(always)]
    fn max(a: [uint32x4_t; 4], b: [uint32x4_t; 4]) -> [uint32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U32x4Backend>::max(a[i], b[i]))
    }

    #[inline(always)]
    fn reduce_add(a: [uint32x4_t; 4]) -> u32 {
        <archmage::NeonToken as U32x4Backend>::reduce_add(a[0])
            .wrapping_add(<archmage::NeonToken as U32x4Backend>::reduce_add(a[1]))
            .wrapping_add(<archmage::NeonToken as U32x4Backend>::reduce_add(a[2]))
            .wrapping_add(<archmage::NeonToken as U32x4Backend>::reduce_add(a[3]))
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: [uint32x4_t; 4]) -> [uint32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U32x4Backend>::shl_const::<N>(a[i]))
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: [uint32x4_t; 4]) -> [uint32x4_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as U32x4Backend>::shr_logical_const::<N>(a[i])
        })
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: [uint32x4_t; 4]) -> [uint32x4_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as U32x4Backend>::shr_logical_const::<N>(a[i])
        })
    }

    #[inline(always)]
    fn all_true(a: [uint32x4_t; 4]) -> bool {
        <archmage::NeonToken as U32x4Backend>::all_true(a[0])
            && <archmage::NeonToken as U32x4Backend>::all_true(a[1])
            && <archmage::NeonToken as U32x4Backend>::all_true(a[2])
            && <archmage::NeonToken as U32x4Backend>::all_true(a[3])
    }

    #[inline(always)]
    fn any_true(a: [uint32x4_t; 4]) -> bool {
        <archmage::NeonToken as U32x4Backend>::any_true(a[0])
            || <archmage::NeonToken as U32x4Backend>::any_true(a[1])
            || <archmage::NeonToken as U32x4Backend>::any_true(a[2])
            || <archmage::NeonToken as U32x4Backend>::any_true(a[3])
    }

    #[inline(always)]
    fn bitmask(a: [uint32x4_t; 4]) -> u64 {
        let q0 = <archmage::NeonToken as U32x4Backend>::bitmask(a[0]) as u64;
        let q1 = <archmage::NeonToken as U32x4Backend>::bitmask(a[1]) as u64;
        let q2 = <archmage::NeonToken as U32x4Backend>::bitmask(a[2]) as u64;
        let q3 = <archmage::NeonToken as U32x4Backend>::bitmask(a[3]) as u64;
        q0 | (q1 << 4) | (q2 << 8) | (q3 << 12)
    }

    #[inline(always)]
    fn simd_eq(a: [uint32x4_t; 4], b: [uint32x4_t; 4]) -> [uint32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U32x4Backend>::simd_eq(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ne(a: [uint32x4_t; 4], b: [uint32x4_t; 4]) -> [uint32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U32x4Backend>::simd_ne(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_lt(a: [uint32x4_t; 4], b: [uint32x4_t; 4]) -> [uint32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U32x4Backend>::simd_lt(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_le(a: [uint32x4_t; 4], b: [uint32x4_t; 4]) -> [uint32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U32x4Backend>::simd_le(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_gt(a: [uint32x4_t; 4], b: [uint32x4_t; 4]) -> [uint32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U32x4Backend>::simd_gt(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ge(a: [uint32x4_t; 4], b: [uint32x4_t; 4]) -> [uint32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U32x4Backend>::simd_ge(a[i], b[i]))
    }

    #[inline(always)]
    fn blend(
        mask: [uint32x4_t; 4],
        if_true: [uint32x4_t; 4],
        if_false: [uint32x4_t; 4],
    ) -> [uint32x4_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as U32x4Backend>::blend(mask[i], if_true[i], if_false[i])
        })
    }

    #[inline(always)]
    fn not(a: [uint32x4_t; 4]) -> [uint32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U32x4Backend>::not(a[i]))
    }

    #[inline(always)]
    fn bitand(a: [uint32x4_t; 4], b: [uint32x4_t; 4]) -> [uint32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U32x4Backend>::bitand(a[i], b[i]))
    }

    #[inline(always)]
    fn bitor(a: [uint32x4_t; 4], b: [uint32x4_t; 4]) -> [uint32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U32x4Backend>::bitor(a[i], b[i]))
    }

    #[inline(always)]
    fn bitxor(a: [uint32x4_t; 4], b: [uint32x4_t; 4]) -> [uint32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U32x4Backend>::bitxor(a[i], b[i]))
    }
}

#[cfg(target_arch = "aarch64")]
impl I64x8Backend for archmage::NeonToken {
    type Repr = [int64x2_t; 4];

    #[inline(always)]
    fn splat(v: i64) -> [int64x2_t; 4] {
        let q = <archmage::NeonToken as I64x2Backend>::splat(v);
        [q, q, q, q]
    }

    #[inline(always)]
    fn zero() -> [int64x2_t; 4] {
        let q = <archmage::NeonToken as I64x2Backend>::zero();
        [q, q, q, q]
    }

    #[inline(always)]
    fn load(data: &[i64; 8]) -> [int64x2_t; 4] {
        [
            <archmage::NeonToken as I64x2Backend>::load(data[0..2].try_into().unwrap()),
            <archmage::NeonToken as I64x2Backend>::load(data[2..4].try_into().unwrap()),
            <archmage::NeonToken as I64x2Backend>::load(data[4..6].try_into().unwrap()),
            <archmage::NeonToken as I64x2Backend>::load(data[6..8].try_into().unwrap()),
        ]
    }

    #[inline(always)]
    fn from_array(arr: [i64; 8]) -> [int64x2_t; 4] {
        let mut q0 = [0; 2];
        let mut q1 = [0; 2];
        let mut q2 = [0; 2];
        let mut q3 = [0; 2];
        q0.copy_from_slice(&arr[0..2]);
        q1.copy_from_slice(&arr[2..4]);
        q2.copy_from_slice(&arr[4..6]);
        q3.copy_from_slice(&arr[6..8]);
        [
            <archmage::NeonToken as I64x2Backend>::from_array(q0),
            <archmage::NeonToken as I64x2Backend>::from_array(q1),
            <archmage::NeonToken as I64x2Backend>::from_array(q2),
            <archmage::NeonToken as I64x2Backend>::from_array(q3),
        ]
    }

    #[inline(always)]
    fn store(repr: [int64x2_t; 4], out: &mut [i64; 8]) {
        let (o01, o23) = out.split_at_mut(4);
        let (o0, o1) = o01.split_at_mut(2);
        let (o2, o3) = o23.split_at_mut(2);
        <archmage::NeonToken as I64x2Backend>::store(repr[0], o0.try_into().unwrap());
        <archmage::NeonToken as I64x2Backend>::store(repr[1], o1.try_into().unwrap());
        <archmage::NeonToken as I64x2Backend>::store(repr[2], o2.try_into().unwrap());
        <archmage::NeonToken as I64x2Backend>::store(repr[3], o3.try_into().unwrap());
    }

    #[inline(always)]
    fn to_array(repr: [int64x2_t; 4]) -> [i64; 8] {
        let a0 = <archmage::NeonToken as I64x2Backend>::to_array(repr[0]);
        let a1 = <archmage::NeonToken as I64x2Backend>::to_array(repr[1]);
        let a2 = <archmage::NeonToken as I64x2Backend>::to_array(repr[2]);
        let a3 = <archmage::NeonToken as I64x2Backend>::to_array(repr[3]);
        let mut out = [0; 8];
        out[0..2].copy_from_slice(&a0);
        out[2..4].copy_from_slice(&a1);
        out[4..6].copy_from_slice(&a2);
        out[6..8].copy_from_slice(&a3);
        out
    }

    #[inline(always)]
    fn add(a: [int64x2_t; 4], b: [int64x2_t; 4]) -> [int64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I64x2Backend>::add(a[i], b[i]))
    }

    #[inline(always)]
    fn sub(a: [int64x2_t; 4], b: [int64x2_t; 4]) -> [int64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I64x2Backend>::sub(a[i], b[i]))
    }

    #[inline(always)]
    fn neg(a: [int64x2_t; 4]) -> [int64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I64x2Backend>::neg(a[i]))
    }

    #[inline(always)]
    fn min(a: [int64x2_t; 4], b: [int64x2_t; 4]) -> [int64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I64x2Backend>::min(a[i], b[i]))
    }

    #[inline(always)]
    fn max(a: [int64x2_t; 4], b: [int64x2_t; 4]) -> [int64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I64x2Backend>::max(a[i], b[i]))
    }

    #[inline(always)]
    fn abs(a: [int64x2_t; 4]) -> [int64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I64x2Backend>::abs(a[i]))
    }

    #[inline(always)]
    fn reduce_add(a: [int64x2_t; 4]) -> i64 {
        <archmage::NeonToken as I64x2Backend>::reduce_add(a[0])
            .wrapping_add(<archmage::NeonToken as I64x2Backend>::reduce_add(a[1]))
            .wrapping_add(<archmage::NeonToken as I64x2Backend>::reduce_add(a[2]))
            .wrapping_add(<archmage::NeonToken as I64x2Backend>::reduce_add(a[3]))
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: [int64x2_t; 4]) -> [int64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I64x2Backend>::shl_const::<N>(a[i]))
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: [int64x2_t; 4]) -> [int64x2_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as I64x2Backend>::shr_arithmetic_const::<N>(a[i])
        })
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: [int64x2_t; 4]) -> [int64x2_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as I64x2Backend>::shr_logical_const::<N>(a[i])
        })
    }

    #[inline(always)]
    fn all_true(a: [int64x2_t; 4]) -> bool {
        <archmage::NeonToken as I64x2Backend>::all_true(a[0])
            && <archmage::NeonToken as I64x2Backend>::all_true(a[1])
            && <archmage::NeonToken as I64x2Backend>::all_true(a[2])
            && <archmage::NeonToken as I64x2Backend>::all_true(a[3])
    }

    #[inline(always)]
    fn any_true(a: [int64x2_t; 4]) -> bool {
        <archmage::NeonToken as I64x2Backend>::any_true(a[0])
            || <archmage::NeonToken as I64x2Backend>::any_true(a[1])
            || <archmage::NeonToken as I64x2Backend>::any_true(a[2])
            || <archmage::NeonToken as I64x2Backend>::any_true(a[3])
    }

    #[inline(always)]
    fn bitmask(a: [int64x2_t; 4]) -> u64 {
        let q0 = <archmage::NeonToken as I64x2Backend>::bitmask(a[0]) as u64;
        let q1 = <archmage::NeonToken as I64x2Backend>::bitmask(a[1]) as u64;
        let q2 = <archmage::NeonToken as I64x2Backend>::bitmask(a[2]) as u64;
        let q3 = <archmage::NeonToken as I64x2Backend>::bitmask(a[3]) as u64;
        q0 | (q1 << 2) | (q2 << 4) | (q3 << 6)
    }

    #[inline(always)]
    fn simd_eq(a: [int64x2_t; 4], b: [int64x2_t; 4]) -> [int64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I64x2Backend>::simd_eq(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ne(a: [int64x2_t; 4], b: [int64x2_t; 4]) -> [int64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I64x2Backend>::simd_ne(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_lt(a: [int64x2_t; 4], b: [int64x2_t; 4]) -> [int64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I64x2Backend>::simd_lt(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_le(a: [int64x2_t; 4], b: [int64x2_t; 4]) -> [int64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I64x2Backend>::simd_le(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_gt(a: [int64x2_t; 4], b: [int64x2_t; 4]) -> [int64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I64x2Backend>::simd_gt(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ge(a: [int64x2_t; 4], b: [int64x2_t; 4]) -> [int64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I64x2Backend>::simd_ge(a[i], b[i]))
    }

    #[inline(always)]
    fn blend(
        mask: [int64x2_t; 4],
        if_true: [int64x2_t; 4],
        if_false: [int64x2_t; 4],
    ) -> [int64x2_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as I64x2Backend>::blend(mask[i], if_true[i], if_false[i])
        })
    }

    #[inline(always)]
    fn not(a: [int64x2_t; 4]) -> [int64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I64x2Backend>::not(a[i]))
    }

    #[inline(always)]
    fn bitand(a: [int64x2_t; 4], b: [int64x2_t; 4]) -> [int64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I64x2Backend>::bitand(a[i], b[i]))
    }

    #[inline(always)]
    fn bitor(a: [int64x2_t; 4], b: [int64x2_t; 4]) -> [int64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I64x2Backend>::bitor(a[i], b[i]))
    }

    #[inline(always)]
    fn bitxor(a: [int64x2_t; 4], b: [int64x2_t; 4]) -> [int64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I64x2Backend>::bitxor(a[i], b[i]))
    }
}

#[cfg(target_arch = "aarch64")]
impl U64x8Backend for archmage::NeonToken {
    type Repr = [uint64x2_t; 4];

    #[inline(always)]
    fn splat(v: u64) -> [uint64x2_t; 4] {
        let q = <archmage::NeonToken as U64x2Backend>::splat(v);
        [q, q, q, q]
    }

    #[inline(always)]
    fn zero() -> [uint64x2_t; 4] {
        let q = <archmage::NeonToken as U64x2Backend>::zero();
        [q, q, q, q]
    }

    #[inline(always)]
    fn load(data: &[u64; 8]) -> [uint64x2_t; 4] {
        [
            <archmage::NeonToken as U64x2Backend>::load(data[0..2].try_into().unwrap()),
            <archmage::NeonToken as U64x2Backend>::load(data[2..4].try_into().unwrap()),
            <archmage::NeonToken as U64x2Backend>::load(data[4..6].try_into().unwrap()),
            <archmage::NeonToken as U64x2Backend>::load(data[6..8].try_into().unwrap()),
        ]
    }

    #[inline(always)]
    fn from_array(arr: [u64; 8]) -> [uint64x2_t; 4] {
        let mut q0 = [0; 2];
        let mut q1 = [0; 2];
        let mut q2 = [0; 2];
        let mut q3 = [0; 2];
        q0.copy_from_slice(&arr[0..2]);
        q1.copy_from_slice(&arr[2..4]);
        q2.copy_from_slice(&arr[4..6]);
        q3.copy_from_slice(&arr[6..8]);
        [
            <archmage::NeonToken as U64x2Backend>::from_array(q0),
            <archmage::NeonToken as U64x2Backend>::from_array(q1),
            <archmage::NeonToken as U64x2Backend>::from_array(q2),
            <archmage::NeonToken as U64x2Backend>::from_array(q3),
        ]
    }

    #[inline(always)]
    fn store(repr: [uint64x2_t; 4], out: &mut [u64; 8]) {
        let (o01, o23) = out.split_at_mut(4);
        let (o0, o1) = o01.split_at_mut(2);
        let (o2, o3) = o23.split_at_mut(2);
        <archmage::NeonToken as U64x2Backend>::store(repr[0], o0.try_into().unwrap());
        <archmage::NeonToken as U64x2Backend>::store(repr[1], o1.try_into().unwrap());
        <archmage::NeonToken as U64x2Backend>::store(repr[2], o2.try_into().unwrap());
        <archmage::NeonToken as U64x2Backend>::store(repr[3], o3.try_into().unwrap());
    }

    #[inline(always)]
    fn to_array(repr: [uint64x2_t; 4]) -> [u64; 8] {
        let a0 = <archmage::NeonToken as U64x2Backend>::to_array(repr[0]);
        let a1 = <archmage::NeonToken as U64x2Backend>::to_array(repr[1]);
        let a2 = <archmage::NeonToken as U64x2Backend>::to_array(repr[2]);
        let a3 = <archmage::NeonToken as U64x2Backend>::to_array(repr[3]);
        let mut out = [0; 8];
        out[0..2].copy_from_slice(&a0);
        out[2..4].copy_from_slice(&a1);
        out[4..6].copy_from_slice(&a2);
        out[6..8].copy_from_slice(&a3);
        out
    }

    #[inline(always)]
    fn add(a: [uint64x2_t; 4], b: [uint64x2_t; 4]) -> [uint64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U64x2Backend>::add(a[i], b[i]))
    }

    #[inline(always)]
    fn sub(a: [uint64x2_t; 4], b: [uint64x2_t; 4]) -> [uint64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U64x2Backend>::sub(a[i], b[i]))
    }

    #[inline(always)]
    fn neg(a: [uint64x2_t; 4]) -> [uint64x2_t; 4] {
        let z = <archmage::NeonToken as U64x2Backend>::zero();
        core::array::from_fn(|i| <archmage::NeonToken as U64x2Backend>::sub(z, a[i]))
    }

    #[inline(always)]
    fn min(a: [uint64x2_t; 4], b: [uint64x2_t; 4]) -> [uint64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U64x2Backend>::min(a[i], b[i]))
    }

    #[inline(always)]
    fn max(a: [uint64x2_t; 4], b: [uint64x2_t; 4]) -> [uint64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U64x2Backend>::max(a[i], b[i]))
    }

    #[inline(always)]
    fn reduce_add(a: [uint64x2_t; 4]) -> u64 {
        <archmage::NeonToken as U64x2Backend>::reduce_add(a[0])
            .wrapping_add(<archmage::NeonToken as U64x2Backend>::reduce_add(a[1]))
            .wrapping_add(<archmage::NeonToken as U64x2Backend>::reduce_add(a[2]))
            .wrapping_add(<archmage::NeonToken as U64x2Backend>::reduce_add(a[3]))
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: [uint64x2_t; 4]) -> [uint64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U64x2Backend>::shl_const::<N>(a[i]))
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: [uint64x2_t; 4]) -> [uint64x2_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as U64x2Backend>::shr_logical_const::<N>(a[i])
        })
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: [uint64x2_t; 4]) -> [uint64x2_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as U64x2Backend>::shr_logical_const::<N>(a[i])
        })
    }

    #[inline(always)]
    fn all_true(a: [uint64x2_t; 4]) -> bool {
        <archmage::NeonToken as U64x2Backend>::all_true(a[0])
            && <archmage::NeonToken as U64x2Backend>::all_true(a[1])
            && <archmage::NeonToken as U64x2Backend>::all_true(a[2])
            && <archmage::NeonToken as U64x2Backend>::all_true(a[3])
    }

    #[inline(always)]
    fn any_true(a: [uint64x2_t; 4]) -> bool {
        <archmage::NeonToken as U64x2Backend>::any_true(a[0])
            || <archmage::NeonToken as U64x2Backend>::any_true(a[1])
            || <archmage::NeonToken as U64x2Backend>::any_true(a[2])
            || <archmage::NeonToken as U64x2Backend>::any_true(a[3])
    }

    #[inline(always)]
    fn bitmask(a: [uint64x2_t; 4]) -> u64 {
        let q0 = <archmage::NeonToken as U64x2Backend>::bitmask(a[0]) as u64;
        let q1 = <archmage::NeonToken as U64x2Backend>::bitmask(a[1]) as u64;
        let q2 = <archmage::NeonToken as U64x2Backend>::bitmask(a[2]) as u64;
        let q3 = <archmage::NeonToken as U64x2Backend>::bitmask(a[3]) as u64;
        q0 | (q1 << 2) | (q2 << 4) | (q3 << 6)
    }

    #[inline(always)]
    fn simd_eq(a: [uint64x2_t; 4], b: [uint64x2_t; 4]) -> [uint64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U64x2Backend>::simd_eq(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ne(a: [uint64x2_t; 4], b: [uint64x2_t; 4]) -> [uint64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U64x2Backend>::simd_ne(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_lt(a: [uint64x2_t; 4], b: [uint64x2_t; 4]) -> [uint64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U64x2Backend>::simd_lt(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_le(a: [uint64x2_t; 4], b: [uint64x2_t; 4]) -> [uint64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U64x2Backend>::simd_le(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_gt(a: [uint64x2_t; 4], b: [uint64x2_t; 4]) -> [uint64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U64x2Backend>::simd_gt(a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ge(a: [uint64x2_t; 4], b: [uint64x2_t; 4]) -> [uint64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U64x2Backend>::simd_ge(a[i], b[i]))
    }

    #[inline(always)]
    fn blend(
        mask: [uint64x2_t; 4],
        if_true: [uint64x2_t; 4],
        if_false: [uint64x2_t; 4],
    ) -> [uint64x2_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as U64x2Backend>::blend(mask[i], if_true[i], if_false[i])
        })
    }

    #[inline(always)]
    fn not(a: [uint64x2_t; 4]) -> [uint64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U64x2Backend>::not(a[i]))
    }

    #[inline(always)]
    fn bitand(a: [uint64x2_t; 4], b: [uint64x2_t; 4]) -> [uint64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U64x2Backend>::bitand(a[i], b[i]))
    }

    #[inline(always)]
    fn bitor(a: [uint64x2_t; 4], b: [uint64x2_t; 4]) -> [uint64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U64x2Backend>::bitor(a[i], b[i]))
    }

    #[inline(always)]
    fn bitxor(a: [uint64x2_t; 4], b: [uint64x2_t; 4]) -> [uint64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U64x2Backend>::bitxor(a[i], b[i]))
    }
}

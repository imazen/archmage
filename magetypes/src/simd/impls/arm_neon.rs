//! Backend implementations for NeonToken (AArch64 NEON).
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.
//!
//! # Safety (audit contract — checked backend boundaries)
//!
//! `#[arcane]` checks value intrinsics against the receiver token's features.
//! SSE2-only arithmetic uses the checked x86-64 baseline boundary.
//! Whole-array loads/stores and bit casts use `crate::simd_storage` helpers,
//! which require POD storage and enforce equal sizes at compile time.
//! Token-bearing wrappers never implement the storage POD trait.

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
use archmage::{NeonToken, arcane};

use crate::simd::backends::*;

#[cfg(target_arch = "aarch64")]
impl F32x4Backend for archmage::NeonToken {
    type Repr = float32x4_t;

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn splat(self, v: f32) -> float32x4_t {
        vdupq_n_f32(v)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn zero(self) -> float32x4_t {
        vdupq_n_f32(0.0)
    }

    #[inline(always)]
    fn load(self, data: &[f32; 4]) -> float32x4_t {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [f32; 4]) -> float32x4_t {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: float32x4_t, out: &mut [f32; 4]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: float32x4_t) -> [f32; 4] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn add(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vaddq_f32(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn sub(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vsubq_f32(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn mul(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vmulq_f32(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn div(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vdivq_f32(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn neg(self, a: float32x4_t) -> float32x4_t {
        vnegq_f32(a)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn min(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vminq_f32(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn max(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vmaxq_f32(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn sqrt(self, a: float32x4_t) -> float32x4_t {
        vsqrtq_f32(a)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn abs(self, a: float32x4_t) -> float32x4_t {
        vabsq_f32(a)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn floor(self, a: float32x4_t) -> float32x4_t {
        vrndmq_f32(a)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn ceil(self, a: float32x4_t) -> float32x4_t {
        vrndpq_f32(a)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn round(self, a: float32x4_t) -> float32x4_t {
        vrndnq_f32(a)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn mul_add(self, a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
        vfmaq_f32(c, a, b)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn mul_sub(self, a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
        vfmaq_f32(vnegq_f32(c), a, b)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_eq(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vreinterpretq_f32_u32(vceqq_f32(a, b))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_ne(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vreinterpretq_f32_u32(vmvnq_u32(vceqq_f32(a, b)))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_lt(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vreinterpretq_f32_u32(vcltq_f32(a, b))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_le(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vreinterpretq_f32_u32(vcleq_f32(a, b))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_gt(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vreinterpretq_f32_u32(vcgtq_f32(a, b))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_ge(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vreinterpretq_f32_u32(vcgeq_f32(a, b))
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn blend(self, mask: float32x4_t, if_true: float32x4_t, if_false: float32x4_t) -> float32x4_t {
        vbslq_f32(vreinterpretq_u32_f32(mask), if_true, if_false)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn reduce_add(self, a: float32x4_t) -> f32 {
        {
            let pair = vpaddq_f32(a, a);
            let pair = vpaddq_f32(pair, pair);
            vgetq_lane_f32::<0>(pair)
        }
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn reduce_min(self, a: float32x4_t) -> f32 {
        {
            let pair = vpminq_f32(a, a);
            let pair = vpminq_f32(pair, pair);
            vgetq_lane_f32::<0>(pair)
        }
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn reduce_max(self, a: float32x4_t) -> f32 {
        {
            let pair = vpmaxq_f32(a, a);
            let pair = vpmaxq_f32(pair, pair);
            vgetq_lane_f32::<0>(pair)
        }
    }

    // ====== Reciprocals: estimate tier vs working tier ======
    //
    // `_approx` = raw vrecpe/vrsqrte (~8-bit) + one fused FRECPS/FRSQRTS
    // step (~16-bit): the documented >=12-bit fast path. FRECPS
    // (`vrecpsq`) computes `2 - a*y` and FRSQRTS (`vrsqrtsq`) computes
    // `(3 - a*y*y)/2` as one fused step each — no intermediate rounding,
    // no 2.0/3.0/0.5 splats, measurably faster than hand-rolled mul/sub
    // on real silicon (see `benchmarks/rsqrt_arm_neoverse-n1`). Each step
    // roughly doubles the correct bits (~8 -> ~16 -> ~24). FRECPS/FRSQRTS
    // are baseline NEON.
    //
    // `recip`/`rsqrt` carry the working-tier contract (<= 4 ULP AND
    // exact IEEE rails at ±0/±inf/NaN), filled by the fastest
    // conforming lowering per element type. f32 adds one more fused
    // step to `_approx` (~2-3 ULP; faster than FDIV on Neoverse-class
    // cores, the published 0.9.28 behavior). Rails come free from the
    // hardware: FRECPS/FRSQRTS special-case inf*0 to 2.0/1.5 BY
    // DESIGN so Newton iterations pass rails through — PROVIDED the
    // dangerous product is formed INSIDE the fused op. recip does
    // this naturally (FRECPS(a, y)); rsqrt is therefore written
    // y*FRSQRTS(a, y*y), NOT y*FRSQRTS(a*y, y): a plain a*y mul is
    // NaN at the rails and was exactly how the pre-fix NaN leaked
    // (probe-verified under qemu, 2026-09-04). Cost: identical
    // instruction count. Tradeoff: y*y overflows f32 for a below
    // ~3e-39, so deep-subnormal inputs are unspecified at this tier.
    // f64 divides (no worthwhile f64 estimate exists, and FDIV is
    // both exact and measured faster on Apple).
    //
    // Cost of exactness is per-core and does NOT generalize — these
    // numbers are for f32 specifically:
    //   Apple Silicon (M-series): exact is FASTER
    //     recip 1.35x, rsqrt 1.18x  (commits defbbc2 / 1b36fc7)
    //   Neoverse-N1 (Ampere Altra): exact is SLOWER
    //     rcp 1.39x, rsqrt 2.15x
    //     (benchmarks/rsqrt_arm_neoverse-n1_2026-06-21.md)
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn rcp_approx(self, a: float32x4_t) -> float32x4_t {
        let y = vrecpeq_f32(a);
        vmulq_f32(vrecpsq_f32(a, y), y)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn rsqrt_approx(self, a: float32x4_t) -> float32x4_t {
        let y = vrsqrteq_f32(a);
        vmulq_f32(y, vrsqrtsq_f32(a, vmulq_f32(y, y)))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn recip(self, a: float32x4_t) -> float32x4_t {
        let y = <Self as F32x4Backend>::rcp_approx(_self, a);
        vmulq_f32(vrecpsq_f32(a, y), y)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn rsqrt(self, a: float32x4_t) -> float32x4_t {
        let y = <Self as F32x4Backend>::rsqrt_approx(_self, a);
        vmulq_f32(y, vrsqrtsq_f32(a, vmulq_f32(y, y)))
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn not(self, a: float32x4_t) -> float32x4_t {
        vreinterpretq_f32_u32(vmvnq_u32(vreinterpretq_u32_f32(a)))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitand(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vreinterpretq_f32_u32(vandq_u32(
            vreinterpretq_u32_f32(a),
            vreinterpretq_u32_f32(b),
        ))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitor(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vreinterpretq_f32_u32(vorrq_u32(
            vreinterpretq_u32_f32(a),
            vreinterpretq_u32_f32(b),
        ))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitxor(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vreinterpretq_f32_u32(veorq_u32(
            vreinterpretq_u32_f32(a),
            vreinterpretq_u32_f32(b),
        ))
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn to_u8_bytes(self, a: float32x4_t) -> [u8; 4] {
        let i16s = vqmovn_s32(vcvtnq_s32_f32(a));
        let u8s = vqmovun_s16(vcombine_s16(i16s, i16s));
        let bytes: [u8; 8] = crate::simd_storage::cast(u8s);
        [bytes[0], bytes[1], bytes[2], bytes[3]]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn store_rgba_bytes(
        self,
        r: float32x4_t,
        g: float32x4_t,
        b: float32x4_t,
        a: float32x4_t,
    ) -> [u8; 16] {
        let lo = vdupq_n_s32(0);
        let hi = vdupq_n_s32(255);
        let ri = vreinterpretq_u32_s32(vminq_s32(vmaxq_s32(vcvtnq_s32_f32(r), lo), hi));
        let gi = vreinterpretq_u32_s32(vminq_s32(vmaxq_s32(vcvtnq_s32_f32(g), lo), hi));
        let bi = vreinterpretq_u32_s32(vminq_s32(vmaxq_s32(vcvtnq_s32_f32(b), lo), hi));
        let ai = vreinterpretq_u32_s32(vminq_s32(vmaxq_s32(vcvtnq_s32_f32(a), lo), hi));
        let pixels = vorrq_u32(
            vorrq_u32(ri, vshlq_n_u32::<8>(gi)),
            vorrq_u32(vshlq_n_u32::<16>(bi), vshlq_n_u32::<24>(ai)),
        );
        crate::simd_storage::cast(vreinterpretq_u8_u32(pixels))
    }
}

#[cfg(target_arch = "aarch64")]
impl F32x8Backend for archmage::NeonToken {
    type Repr = [float32x4_t; 2];

    // ====== Construction ======

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn splat(self, v: f32) -> [float32x4_t; 2] {
        let v4 = vdupq_n_f32(v);
        [v4, v4]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn zero(self) -> [float32x4_t; 2] {
        let z = vdupq_n_f32(0.0);
        [z, z]
    }

    #[inline(always)]
    fn load(self, data: &[f32; 8]) -> [float32x4_t; 2] {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [f32; 8]) -> [float32x4_t; 2] {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: [float32x4_t; 2], out: &mut [f32; 8]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: [float32x4_t; 2]) -> [f32; 8] {
        crate::simd_storage::cast(repr)
    }

    // ====== Arithmetic ======

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn add(self, a: [float32x4_t; 2], b: [float32x4_t; 2]) -> [float32x4_t; 2] {
        [vaddq_f32(a[0], b[0]), vaddq_f32(a[1], b[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn sub(self, a: [float32x4_t; 2], b: [float32x4_t; 2]) -> [float32x4_t; 2] {
        [vsubq_f32(a[0], b[0]), vsubq_f32(a[1], b[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn mul(self, a: [float32x4_t; 2], b: [float32x4_t; 2]) -> [float32x4_t; 2] {
        [vmulq_f32(a[0], b[0]), vmulq_f32(a[1], b[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn div(self, a: [float32x4_t; 2], b: [float32x4_t; 2]) -> [float32x4_t; 2] {
        [vdivq_f32(a[0], b[0]), vdivq_f32(a[1], b[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn neg(self, a: [float32x4_t; 2]) -> [float32x4_t; 2] {
        [vnegq_f32(a[0]), vnegq_f32(a[1])]
    }

    // ====== Math ======

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn min(self, a: [float32x4_t; 2], b: [float32x4_t; 2]) -> [float32x4_t; 2] {
        [vminq_f32(a[0], b[0]), vminq_f32(a[1], b[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn max(self, a: [float32x4_t; 2], b: [float32x4_t; 2]) -> [float32x4_t; 2] {
        [vmaxq_f32(a[0], b[0]), vmaxq_f32(a[1], b[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn sqrt(self, a: [float32x4_t; 2]) -> [float32x4_t; 2] {
        [vsqrtq_f32(a[0]), vsqrtq_f32(a[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn abs(self, a: [float32x4_t; 2]) -> [float32x4_t; 2] {
        [vabsq_f32(a[0]), vabsq_f32(a[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn floor(self, a: [float32x4_t; 2]) -> [float32x4_t; 2] {
        [vrndmq_f32(a[0]), vrndmq_f32(a[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn ceil(self, a: [float32x4_t; 2]) -> [float32x4_t; 2] {
        [vrndpq_f32(a[0]), vrndpq_f32(a[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn round(self, a: [float32x4_t; 2]) -> [float32x4_t; 2] {
        [vrndnq_f32(a[0]), vrndnq_f32(a[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn mul_add(
        self,
        a: [float32x4_t; 2],
        b: [float32x4_t; 2],
        c: [float32x4_t; 2],
    ) -> [float32x4_t; 2] {
        // vfmaq = acc + x*y, so mul_add(a, b, c) = a*b + c => vfmaq(c, a, b)
        [vfmaq_f32(c[0], a[0], b[0]), vfmaq_f32(c[1], a[1], b[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn mul_sub(
        self,
        a: [float32x4_t; 2],
        b: [float32x4_t; 2],
        c: [float32x4_t; 2],
    ) -> [float32x4_t; 2] {
        // a*b - c => vfmaq(-c, a, b) = -c + a*b
        [
            vfmaq_f32(vnegq_f32(c[0]), a[0], b[0]),
            vfmaq_f32(vnegq_f32(c[1]), a[1], b[1]),
        ]
    }

    // ====== Comparisons ======

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_eq(self, a: [float32x4_t; 2], b: [float32x4_t; 2]) -> [float32x4_t; 2] {
        [
            vreinterpretq_f32_u32(vceqq_f32(a[0], b[0])),
            vreinterpretq_f32_u32(vceqq_f32(a[1], b[1])),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_ne(self, a: [float32x4_t; 2], b: [float32x4_t; 2]) -> [float32x4_t; 2] {
        [
            vreinterpretq_f32_u32(vmvnq_u32(vceqq_f32(a[0], b[0]))),
            vreinterpretq_f32_u32(vmvnq_u32(vceqq_f32(a[1], b[1]))),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_lt(self, a: [float32x4_t; 2], b: [float32x4_t; 2]) -> [float32x4_t; 2] {
        [
            vreinterpretq_f32_u32(vcltq_f32(a[0], b[0])),
            vreinterpretq_f32_u32(vcltq_f32(a[1], b[1])),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_le(self, a: [float32x4_t; 2], b: [float32x4_t; 2]) -> [float32x4_t; 2] {
        [
            vreinterpretq_f32_u32(vcleq_f32(a[0], b[0])),
            vreinterpretq_f32_u32(vcleq_f32(a[1], b[1])),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_gt(self, a: [float32x4_t; 2], b: [float32x4_t; 2]) -> [float32x4_t; 2] {
        [
            vreinterpretq_f32_u32(vcgtq_f32(a[0], b[0])),
            vreinterpretq_f32_u32(vcgtq_f32(a[1], b[1])),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_ge(self, a: [float32x4_t; 2], b: [float32x4_t; 2]) -> [float32x4_t; 2] {
        [
            vreinterpretq_f32_u32(vcgeq_f32(a[0], b[0])),
            vreinterpretq_f32_u32(vcgeq_f32(a[1], b[1])),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn blend(
        self,
        mask: [float32x4_t; 2],
        if_true: [float32x4_t; 2],
        if_false: [float32x4_t; 2],
    ) -> [float32x4_t; 2] {
        [
            vbslq_f32(vreinterpretq_u32_f32(mask[0]), if_true[0], if_false[0]),
            vbslq_f32(vreinterpretq_u32_f32(mask[1]), if_true[1], if_false[1]),
        ]
    }

    // ====== Reductions ======

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn reduce_add(self, a: [float32x4_t; 2]) -> f32 {
        {
            let m = vaddq_f32(a[0], a[1]);
            let pair = vpaddq_f32(m, m);
            let pair = vpaddq_f32(pair, pair);
            vgetq_lane_f32::<0>(pair)
        }
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn reduce_min(self, a: [float32x4_t; 2]) -> f32 {
        {
            let m = vminq_f32(a[0], a[1]);
            let pair = vpminq_f32(m, m);
            let pair = vpminq_f32(pair, pair);
            vgetq_lane_f32::<0>(pair)
        }
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn reduce_max(self, a: [float32x4_t; 2]) -> f32 {
        {
            let m = vmaxq_f32(a[0], a[1]);
            let pair = vpmaxq_f32(m, m);
            let pair = vpmaxq_f32(pair, pair);
            vgetq_lane_f32::<0>(pair)
        }
    }

    // ====== Approximations ======
    //
    // Delegate to the native f32x4/f64x2 backend so the polyfill inherits
    // its >=12-bit fused `_approx` (raw vrecpe + 1 FRECPS) and exact full
    // methods — one source of truth for the estimate.
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn rcp_approx(self, a: [float32x4_t; 2]) -> [float32x4_t; 2] {
        core::array::from_fn(|i| <Self as F32x4Backend>::rcp_approx(_self, a[i]))
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn rsqrt_approx(self, a: [float32x4_t; 2]) -> [float32x4_t; 2] {
        core::array::from_fn(|i| <Self as F32x4Backend>::rsqrt_approx(_self, a[i]))
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn recip(self, a: [float32x4_t; 2]) -> [float32x4_t; 2] {
        core::array::from_fn(|i| <Self as F32x4Backend>::recip(_self, a[i]))
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn rsqrt(self, a: [float32x4_t; 2]) -> [float32x4_t; 2] {
        core::array::from_fn(|i| <Self as F32x4Backend>::rsqrt(_self, a[i]))
    }

    // ====== Bitwise ======

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn not(self, a: [float32x4_t; 2]) -> [float32x4_t; 2] {
        [
            vreinterpretq_f32_u32(vmvnq_u32(vreinterpretq_u32_f32(a[0]))),
            vreinterpretq_f32_u32(vmvnq_u32(vreinterpretq_u32_f32(a[1]))),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitand(self, a: [float32x4_t; 2], b: [float32x4_t; 2]) -> [float32x4_t; 2] {
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

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitor(self, a: [float32x4_t; 2], b: [float32x4_t; 2]) -> [float32x4_t; 2] {
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

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitxor(self, a: [float32x4_t; 2], b: [float32x4_t; 2]) -> [float32x4_t; 2] {
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

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn to_u8_bytes(self, a: [float32x4_t; 2]) -> [u8; 8] {
        let i0 = vqmovn_s32(vcvtnq_s32_f32(a[0]));
        let i1 = vqmovn_s32(vcvtnq_s32_f32(a[1]));
        let u8s = vqmovun_s16(vcombine_s16(i0, i1));
        crate::simd_storage::cast(u8s)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn store_rgba_bytes(
        self,
        r: [float32x4_t; 2],
        g: [float32x4_t; 2],
        b: [float32x4_t; 2],
        a: [float32x4_t; 2],
    ) -> [u8; 32] {
        let lo = vdupq_n_s32(0);
        let hi = vdupq_n_s32(255);
        let r0 = vreinterpretq_u32_s32(vminq_s32(vmaxq_s32(vcvtnq_s32_f32(r[0]), lo), hi));
        let g0 = vreinterpretq_u32_s32(vminq_s32(vmaxq_s32(vcvtnq_s32_f32(g[0]), lo), hi));
        let b0 = vreinterpretq_u32_s32(vminq_s32(vmaxq_s32(vcvtnq_s32_f32(b[0]), lo), hi));
        let a0 = vreinterpretq_u32_s32(vminq_s32(vmaxq_s32(vcvtnq_s32_f32(a[0]), lo), hi));
        let p0 = vorrq_u32(
            vorrq_u32(r0, vshlq_n_u32::<8>(g0)),
            vorrq_u32(vshlq_n_u32::<16>(b0), vshlq_n_u32::<24>(a0)),
        );
        let r1 = vreinterpretq_u32_s32(vminq_s32(vmaxq_s32(vcvtnq_s32_f32(r[1]), lo), hi));
        let g1 = vreinterpretq_u32_s32(vminq_s32(vmaxq_s32(vcvtnq_s32_f32(g[1]), lo), hi));
        let b1 = vreinterpretq_u32_s32(vminq_s32(vmaxq_s32(vcvtnq_s32_f32(b[1]), lo), hi));
        let a1 = vreinterpretq_u32_s32(vminq_s32(vmaxq_s32(vcvtnq_s32_f32(a[1]), lo), hi));
        let p1 = vorrq_u32(
            vorrq_u32(r1, vshlq_n_u32::<8>(g1)),
            vorrq_u32(vshlq_n_u32::<16>(b1), vshlq_n_u32::<24>(a1)),
        );
        crate::simd_storage::cast([vreinterpretq_u8_u32(p0), vreinterpretq_u8_u32(p1)])
    }
}

#[cfg(target_arch = "aarch64")]
impl F64x2Backend for archmage::NeonToken {
    type Repr = float64x2_t;

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn splat(self, v: f64) -> float64x2_t {
        vdupq_n_f64(v)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn zero(self) -> float64x2_t {
        vdupq_n_f64(0.0)
    }

    #[inline(always)]
    fn load(self, data: &[f64; 2]) -> float64x2_t {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [f64; 2]) -> float64x2_t {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: float64x2_t, out: &mut [f64; 2]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: float64x2_t) -> [f64; 2] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn add(self, a: float64x2_t, b: float64x2_t) -> float64x2_t {
        vaddq_f64(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn sub(self, a: float64x2_t, b: float64x2_t) -> float64x2_t {
        vsubq_f64(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn mul(self, a: float64x2_t, b: float64x2_t) -> float64x2_t {
        vmulq_f64(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn div(self, a: float64x2_t, b: float64x2_t) -> float64x2_t {
        vdivq_f64(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn neg(self, a: float64x2_t) -> float64x2_t {
        vnegq_f64(a)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn min(self, a: float64x2_t, b: float64x2_t) -> float64x2_t {
        vminq_f64(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn max(self, a: float64x2_t, b: float64x2_t) -> float64x2_t {
        vmaxq_f64(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn sqrt(self, a: float64x2_t) -> float64x2_t {
        vsqrtq_f64(a)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn abs(self, a: float64x2_t) -> float64x2_t {
        vabsq_f64(a)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn floor(self, a: float64x2_t) -> float64x2_t {
        vrndmq_f64(a)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn ceil(self, a: float64x2_t) -> float64x2_t {
        vrndpq_f64(a)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn round(self, a: float64x2_t) -> float64x2_t {
        vrndnq_f64(a)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn mul_add(self, a: float64x2_t, b: float64x2_t, c: float64x2_t) -> float64x2_t {
        vfmaq_f64(c, a, b)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn mul_sub(self, a: float64x2_t, b: float64x2_t, c: float64x2_t) -> float64x2_t {
        vfmaq_f64(vnegq_f64(c), a, b)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_eq(self, a: float64x2_t, b: float64x2_t) -> float64x2_t {
        vreinterpretq_f64_u64(vceqq_f64(a, b))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_ne(self, a: float64x2_t, b: float64x2_t) -> float64x2_t {
        vreinterpretq_f64_u64(veorq_u64(vceqq_f64(a, b), vdupq_n_u64(u64::MAX)))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_lt(self, a: float64x2_t, b: float64x2_t) -> float64x2_t {
        vreinterpretq_f64_u64(vcltq_f64(a, b))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_le(self, a: float64x2_t, b: float64x2_t) -> float64x2_t {
        vreinterpretq_f64_u64(vcleq_f64(a, b))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_gt(self, a: float64x2_t, b: float64x2_t) -> float64x2_t {
        vreinterpretq_f64_u64(vcgtq_f64(a, b))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_ge(self, a: float64x2_t, b: float64x2_t) -> float64x2_t {
        vreinterpretq_f64_u64(vcgeq_f64(a, b))
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn blend(self, mask: float64x2_t, if_true: float64x2_t, if_false: float64x2_t) -> float64x2_t {
        vbslq_f64(vreinterpretq_u64_f64(mask), if_true, if_false)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn reduce_add(self, a: float64x2_t) -> f64 {
        {
            let pair = vpaddq_f64(a, a);
            vgetq_lane_f64::<0>(pair)
        }
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn reduce_min(self, a: float64x2_t) -> f64 {
        {
            let pair = vpminq_f64(a, a);
            vgetq_lane_f64::<0>(pair)
        }
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn reduce_max(self, a: float64x2_t) -> f64 {
        {
            let pair = vpmaxq_f64(a, a);
            vgetq_lane_f64::<0>(pair)
        }
    }

    // ====== Reciprocals: estimate tier vs working tier ======
    //
    // `_approx` = raw vrecpe/vrsqrte (~8-bit) + one fused FRECPS/FRSQRTS
    // step (~16-bit): the documented >=12-bit fast path. FRECPS
    // (`vrecpsq`) computes `2 - a*y` and FRSQRTS (`vrsqrtsq`) computes
    // `(3 - a*y*y)/2` as one fused step each — no intermediate rounding,
    // no 2.0/3.0/0.5 splats, measurably faster than hand-rolled mul/sub
    // on real silicon (see `benchmarks/rsqrt_arm_neoverse-n1`). Each step
    // roughly doubles the correct bits (~8 -> ~16 -> ~24). FRECPS/FRSQRTS
    // are baseline NEON.
    //
    // `recip`/`rsqrt` carry the working-tier contract (<= 4 ULP AND
    // exact IEEE rails at ±0/±inf/NaN), filled by the fastest
    // conforming lowering per element type. f32 adds one more fused
    // step to `_approx` (~2-3 ULP; faster than FDIV on Neoverse-class
    // cores, the published 0.9.28 behavior). Rails come free from the
    // hardware: FRECPS/FRSQRTS special-case inf*0 to 2.0/1.5 BY
    // DESIGN so Newton iterations pass rails through — PROVIDED the
    // dangerous product is formed INSIDE the fused op. recip does
    // this naturally (FRECPS(a, y)); rsqrt is therefore written
    // y*FRSQRTS(a, y*y), NOT y*FRSQRTS(a*y, y): a plain a*y mul is
    // NaN at the rails and was exactly how the pre-fix NaN leaked
    // (probe-verified under qemu, 2026-09-04). Cost: identical
    // instruction count. Tradeoff: y*y overflows f32 for a below
    // ~3e-39, so deep-subnormal inputs are unspecified at this tier.
    // f64 divides (no worthwhile f64 estimate exists, and FDIV is
    // both exact and measured faster on Apple).
    //
    // Cost of exactness is per-core and does NOT generalize — these
    // numbers are for f64 specifically:
    //   Apple M4 Pro: exact is FASTER
    //     rcp 3.2x, rsqrt 1.23x
    //     (benchmarks/rsqrt_f64_arm_apple-m4-pro_2026-08-02.md)
    //   Neoverse / server ARM: NOT MEASURED for f64. The f32 form is
    //     slower there, so treat an f64 regression as plausible until
    //     someone runs `bench_rsqrt_f64` on that class of core.
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn rcp_approx(self, a: float64x2_t) -> float64x2_t {
        let y = vrecpeq_f64(a);
        vmulq_f64(vrecpsq_f64(a, y), y)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn rsqrt_approx(self, a: float64x2_t) -> float64x2_t {
        let y = vrsqrteq_f64(a);
        vmulq_f64(y, vrsqrtsq_f64(a, vmulq_f64(y, y)))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn recip(self, a: float64x2_t) -> float64x2_t {
        vdivq_f64(vdupq_n_f64(1.0), a)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn rsqrt(self, a: float64x2_t) -> float64x2_t {
        vdivq_f64(vdupq_n_f64(1.0), vsqrtq_f64(a))
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn not(self, a: float64x2_t) -> float64x2_t {
        vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(a), vdupq_n_u64(u64::MAX)))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitand(self, a: float64x2_t, b: float64x2_t) -> float64x2_t {
        vreinterpretq_f64_u64(vandq_u64(
            vreinterpretq_u64_f64(a),
            vreinterpretq_u64_f64(b),
        ))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitor(self, a: float64x2_t, b: float64x2_t) -> float64x2_t {
        vreinterpretq_f64_u64(vorrq_u64(
            vreinterpretq_u64_f64(a),
            vreinterpretq_u64_f64(b),
        ))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitxor(self, a: float64x2_t, b: float64x2_t) -> float64x2_t {
        vreinterpretq_f64_u64(veorq_u64(
            vreinterpretq_u64_f64(a),
            vreinterpretq_u64_f64(b),
        ))
    }
}

#[cfg(target_arch = "aarch64")]
impl F64x4Backend for archmage::NeonToken {
    type Repr = [float64x2_t; 2];

    // ====== Construction ======

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn splat(self, v: f64) -> [float64x2_t; 2] {
        let v4 = vdupq_n_f64(v);
        [v4, v4]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn zero(self) -> [float64x2_t; 2] {
        let z = vdupq_n_f64(0.0);
        [z, z]
    }

    #[inline(always)]
    fn load(self, data: &[f64; 4]) -> [float64x2_t; 2] {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [f64; 4]) -> [float64x2_t; 2] {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: [float64x2_t; 2], out: &mut [f64; 4]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: [float64x2_t; 2]) -> [f64; 4] {
        crate::simd_storage::cast(repr)
    }

    // ====== Arithmetic ======

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn add(self, a: [float64x2_t; 2], b: [float64x2_t; 2]) -> [float64x2_t; 2] {
        [vaddq_f64(a[0], b[0]), vaddq_f64(a[1], b[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn sub(self, a: [float64x2_t; 2], b: [float64x2_t; 2]) -> [float64x2_t; 2] {
        [vsubq_f64(a[0], b[0]), vsubq_f64(a[1], b[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn mul(self, a: [float64x2_t; 2], b: [float64x2_t; 2]) -> [float64x2_t; 2] {
        [vmulq_f64(a[0], b[0]), vmulq_f64(a[1], b[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn div(self, a: [float64x2_t; 2], b: [float64x2_t; 2]) -> [float64x2_t; 2] {
        [vdivq_f64(a[0], b[0]), vdivq_f64(a[1], b[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn neg(self, a: [float64x2_t; 2]) -> [float64x2_t; 2] {
        [vnegq_f64(a[0]), vnegq_f64(a[1])]
    }

    // ====== Math ======

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn min(self, a: [float64x2_t; 2], b: [float64x2_t; 2]) -> [float64x2_t; 2] {
        [vminq_f64(a[0], b[0]), vminq_f64(a[1], b[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn max(self, a: [float64x2_t; 2], b: [float64x2_t; 2]) -> [float64x2_t; 2] {
        [vmaxq_f64(a[0], b[0]), vmaxq_f64(a[1], b[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn sqrt(self, a: [float64x2_t; 2]) -> [float64x2_t; 2] {
        [vsqrtq_f64(a[0]), vsqrtq_f64(a[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn abs(self, a: [float64x2_t; 2]) -> [float64x2_t; 2] {
        [vabsq_f64(a[0]), vabsq_f64(a[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn floor(self, a: [float64x2_t; 2]) -> [float64x2_t; 2] {
        [vrndmq_f64(a[0]), vrndmq_f64(a[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn ceil(self, a: [float64x2_t; 2]) -> [float64x2_t; 2] {
        [vrndpq_f64(a[0]), vrndpq_f64(a[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn round(self, a: [float64x2_t; 2]) -> [float64x2_t; 2] {
        [vrndnq_f64(a[0]), vrndnq_f64(a[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn mul_add(
        self,
        a: [float64x2_t; 2],
        b: [float64x2_t; 2],
        c: [float64x2_t; 2],
    ) -> [float64x2_t; 2] {
        // vfmaq = acc + x*y, so mul_add(a, b, c) = a*b + c => vfmaq(c, a, b)
        [vfmaq_f64(c[0], a[0], b[0]), vfmaq_f64(c[1], a[1], b[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn mul_sub(
        self,
        a: [float64x2_t; 2],
        b: [float64x2_t; 2],
        c: [float64x2_t; 2],
    ) -> [float64x2_t; 2] {
        // a*b - c => vfmaq(-c, a, b) = -c + a*b
        [
            vfmaq_f64(vnegq_f64(c[0]), a[0], b[0]),
            vfmaq_f64(vnegq_f64(c[1]), a[1], b[1]),
        ]
    }

    // ====== Comparisons ======

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_eq(self, a: [float64x2_t; 2], b: [float64x2_t; 2]) -> [float64x2_t; 2] {
        [
            vreinterpretq_f64_u64(vceqq_f64(a[0], b[0])),
            vreinterpretq_f64_u64(vceqq_f64(a[1], b[1])),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_ne(self, a: [float64x2_t; 2], b: [float64x2_t; 2]) -> [float64x2_t; 2] {
        [
            vreinterpretq_f64_u64(veorq_u64(vceqq_f64(a[0], b[0]), vdupq_n_u64(u64::MAX))),
            vreinterpretq_f64_u64(veorq_u64(vceqq_f64(a[1], b[1]), vdupq_n_u64(u64::MAX))),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_lt(self, a: [float64x2_t; 2], b: [float64x2_t; 2]) -> [float64x2_t; 2] {
        [
            vreinterpretq_f64_u64(vcltq_f64(a[0], b[0])),
            vreinterpretq_f64_u64(vcltq_f64(a[1], b[1])),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_le(self, a: [float64x2_t; 2], b: [float64x2_t; 2]) -> [float64x2_t; 2] {
        [
            vreinterpretq_f64_u64(vcleq_f64(a[0], b[0])),
            vreinterpretq_f64_u64(vcleq_f64(a[1], b[1])),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_gt(self, a: [float64x2_t; 2], b: [float64x2_t; 2]) -> [float64x2_t; 2] {
        [
            vreinterpretq_f64_u64(vcgtq_f64(a[0], b[0])),
            vreinterpretq_f64_u64(vcgtq_f64(a[1], b[1])),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_ge(self, a: [float64x2_t; 2], b: [float64x2_t; 2]) -> [float64x2_t; 2] {
        [
            vreinterpretq_f64_u64(vcgeq_f64(a[0], b[0])),
            vreinterpretq_f64_u64(vcgeq_f64(a[1], b[1])),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn blend(
        self,
        mask: [float64x2_t; 2],
        if_true: [float64x2_t; 2],
        if_false: [float64x2_t; 2],
    ) -> [float64x2_t; 2] {
        [
            vbslq_f64(vreinterpretq_u64_f64(mask[0]), if_true[0], if_false[0]),
            vbslq_f64(vreinterpretq_u64_f64(mask[1]), if_true[1], if_false[1]),
        ]
    }

    // ====== Reductions ======

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn reduce_add(self, a: [float64x2_t; 2]) -> f64 {
        {
            let m = vaddq_f64(a[0], a[1]);
            let pair = vpaddq_f64(m, m);
            vgetq_lane_f64::<0>(pair)
        }
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn reduce_min(self, a: [float64x2_t; 2]) -> f64 {
        {
            let m = vminq_f64(a[0], a[1]);
            let pair = vpminq_f64(m, m);
            vgetq_lane_f64::<0>(pair)
        }
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn reduce_max(self, a: [float64x2_t; 2]) -> f64 {
        {
            let m = vmaxq_f64(a[0], a[1]);
            let pair = vpmaxq_f64(m, m);
            vgetq_lane_f64::<0>(pair)
        }
    }

    // ====== Approximations ======
    //
    // Delegate to the native f32x4/f64x2 backend so the polyfill inherits
    // its >=12-bit fused `_approx` (raw vrecpe + 1 FRECPS) and exact full
    // methods — one source of truth for the estimate.
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn rcp_approx(self, a: [float64x2_t; 2]) -> [float64x2_t; 2] {
        core::array::from_fn(|i| <Self as F64x2Backend>::rcp_approx(_self, a[i]))
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn rsqrt_approx(self, a: [float64x2_t; 2]) -> [float64x2_t; 2] {
        core::array::from_fn(|i| <Self as F64x2Backend>::rsqrt_approx(_self, a[i]))
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn recip(self, a: [float64x2_t; 2]) -> [float64x2_t; 2] {
        core::array::from_fn(|i| <Self as F64x2Backend>::recip(_self, a[i]))
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn rsqrt(self, a: [float64x2_t; 2]) -> [float64x2_t; 2] {
        core::array::from_fn(|i| <Self as F64x2Backend>::rsqrt(_self, a[i]))
    }

    // ====== Bitwise ======

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn not(self, a: [float64x2_t; 2]) -> [float64x2_t; 2] {
        [
            vreinterpretq_f64_u64(veorq_u64(
                vreinterpretq_u64_f64(a[0]),
                vdupq_n_u64(u64::MAX),
            )),
            vreinterpretq_f64_u64(veorq_u64(
                vreinterpretq_u64_f64(a[1]),
                vdupq_n_u64(u64::MAX),
            )),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitand(self, a: [float64x2_t; 2], b: [float64x2_t; 2]) -> [float64x2_t; 2] {
        [
            vreinterpretq_f64_u64(vandq_u64(
                vreinterpretq_u64_f64(a[0]),
                vreinterpretq_u64_f64(b[0]),
            )),
            vreinterpretq_f64_u64(vandq_u64(
                vreinterpretq_u64_f64(a[1]),
                vreinterpretq_u64_f64(b[1]),
            )),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitor(self, a: [float64x2_t; 2], b: [float64x2_t; 2]) -> [float64x2_t; 2] {
        [
            vreinterpretq_f64_u64(vorrq_u64(
                vreinterpretq_u64_f64(a[0]),
                vreinterpretq_u64_f64(b[0]),
            )),
            vreinterpretq_f64_u64(vorrq_u64(
                vreinterpretq_u64_f64(a[1]),
                vreinterpretq_u64_f64(b[1]),
            )),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitxor(self, a: [float64x2_t; 2], b: [float64x2_t; 2]) -> [float64x2_t; 2] {
        [
            vreinterpretq_f64_u64(veorq_u64(
                vreinterpretq_u64_f64(a[0]),
                vreinterpretq_u64_f64(b[0]),
            )),
            vreinterpretq_f64_u64(veorq_u64(
                vreinterpretq_u64_f64(a[1]),
                vreinterpretq_u64_f64(b[1]),
            )),
        ]
    }
}

#[cfg(target_arch = "aarch64")]
impl I32x4Backend for archmage::NeonToken {
    type Repr = int32x4_t;

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn splat(self, v: i32) -> int32x4_t {
        vdupq_n_s32(v)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn zero(self) -> int32x4_t {
        vdupq_n_s32(0)
    }

    #[inline(always)]
    fn load(self, data: &[i32; 4]) -> int32x4_t {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [i32; 4]) -> int32x4_t {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: int32x4_t, out: &mut [i32; 4]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: int32x4_t) -> [i32; 4] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn add(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {
        vaddq_s32(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn sub(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {
        vsubq_s32(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn mul(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {
        vmulq_s32(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn neg(self, a: int32x4_t) -> int32x4_t {
        vnegq_s32(a)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn min(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {
        vminq_s32(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn max(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {
        vmaxq_s32(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn abs(self, a: int32x4_t) -> int32x4_t {
        vabsq_s32(a)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_eq(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {
        vreinterpretq_s32_u32(vceqq_s32(a, b))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_ne(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {
        vreinterpretq_s32_u32(vmvnq_u32(vceqq_s32(a, b)))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_lt(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {
        vreinterpretq_s32_u32(vcltq_s32(a, b))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_le(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {
        vreinterpretq_s32_u32(vcleq_s32(a, b))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_gt(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {
        vreinterpretq_s32_u32(vcgtq_s32(a, b))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_ge(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {
        vreinterpretq_s32_u32(vcgeq_s32(a, b))
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn blend(self, mask: int32x4_t, if_true: int32x4_t, if_false: int32x4_t) -> int32x4_t {
        vbslq_s32(vreinterpretq_u32_s32(mask), if_true, if_false)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn reduce_add(self, a: int32x4_t) -> i32 {
        vaddvq_s32(a)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn not(self, a: int32x4_t) -> int32x4_t {
        vmvnq_s32(a)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitand(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {
        vandq_s32(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitor(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {
        vorrq_s32(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitxor(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {
        veorq_s32(a, b)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shl_const<const N: i32>(self, a: int32x4_t) -> int32x4_t {
        vshlq_n_s32::<N>(a)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shr_arithmetic_const<const N: i32>(self, a: int32x4_t) -> int32x4_t {
        const { assert!(N >= 0 && N <= 31) };
        vshlq_s32(a, vdupq_n_s32(-N))
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shr_logical_const<const N: i32>(self, a: int32x4_t) -> int32x4_t {
        const { assert!(N >= 0 && N <= 31) };
        vreinterpretq_s32_u32(vshlq_u32(vreinterpretq_u32_s32(a), vdupq_n_s32(-N)))
    }

    // ====== Uniform variable shifts ======
    // Same vshlq lowering as the const forms; the count is clamped
    // because USHL/SSHL read only the low byte of each amount lane as
    // a signed value, so an unclamped 256 would wrap to a no-op
    // instead of the contracted zero.

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shl_uniform(self, a: int32x4_t, count: u32) -> int32x4_t {
        vshlq_s32(a, vdupq_n_s32(count.min(32) as i32))
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shr_logical_uniform(self, a: int32x4_t, count: u32) -> int32x4_t {
        vreinterpretq_s32_u32(vshlq_u32(
            vreinterpretq_u32_s32(a),
            vdupq_n_s32(-(count.min(32) as i32)),
        ))
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shr_arithmetic_uniform(self, a: int32x4_t, count: u32) -> int32x4_t {
        // Clamping to 31 gives the contracted sign fill.
        vshlq_s32(a, vdupq_n_s32(-(count.min(31) as i32)))
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn all_true(self, a: int32x4_t) -> bool {
        vminvq_u32(vreinterpretq_u32_s32(a)) != 0
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn any_true(self, a: int32x4_t) -> bool {
        vmaxvq_u32(vreinterpretq_u32_s32(a)) != 0
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitmask(self, a: int32x4_t) -> u32 {
        // Extract sign bit of each 32-bit lane as 0/1 (LOGICAL shift on
        // the u32 view — an arithmetic s32 shift would sign-extend to
        // 0xFFFF_FFFF and corrupt the packed mask).
        let shift = vshrq_n_u32::<31>(vreinterpretq_u32_s32(a));
        // Pack: lane0 | (lane1<<1) | (lane2<<2) | (lane3<<3)
        let lane0 = vgetq_lane_u32::<0>(shift);
        let lane1 = vgetq_lane_u32::<1>(shift);
        let lane2 = vgetq_lane_u32::<2>(shift);
        let lane3 = vgetq_lane_u32::<3>(shift);
        lane0 | (lane1 << 1) | (lane2 << 2) | (lane3 << 3)
    }
}

#[cfg(target_arch = "aarch64")]
impl I32x8Backend for archmage::NeonToken {
    type Repr = [int32x4_t; 2];

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn splat(self, v: i32) -> [int32x4_t; 2] {
        let v4 = vdupq_n_s32(v);
        [v4, v4]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn zero(self) -> [int32x4_t; 2] {
        let z = vdupq_n_s32(0);
        [z, z]
    }

    #[inline(always)]
    fn load(self, data: &[i32; 8]) -> [int32x4_t; 2] {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [i32; 8]) -> [int32x4_t; 2] {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: [int32x4_t; 2], out: &mut [i32; 8]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: [int32x4_t; 2]) -> [i32; 8] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn add(self, a: [int32x4_t; 2], b: [int32x4_t; 2]) -> [int32x4_t; 2] {
        [vaddq_s32(a[0], b[0]), vaddq_s32(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn sub(self, a: [int32x4_t; 2], b: [int32x4_t; 2]) -> [int32x4_t; 2] {
        [vsubq_s32(a[0], b[0]), vsubq_s32(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn mul(self, a: [int32x4_t; 2], b: [int32x4_t; 2]) -> [int32x4_t; 2] {
        [vmulq_s32(a[0], b[0]), vmulq_s32(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn neg(self, a: [int32x4_t; 2]) -> [int32x4_t; 2] {
        [vnegq_s32(a[0]), vnegq_s32(a[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn min(self, a: [int32x4_t; 2], b: [int32x4_t; 2]) -> [int32x4_t; 2] {
        [vminq_s32(a[0], b[0]), vminq_s32(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn max(self, a: [int32x4_t; 2], b: [int32x4_t; 2]) -> [int32x4_t; 2] {
        [vmaxq_s32(a[0], b[0]), vmaxq_s32(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn abs(self, a: [int32x4_t; 2]) -> [int32x4_t; 2] {
        [vabsq_s32(a[0]), vabsq_s32(a[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_eq(self, a: [int32x4_t; 2], b: [int32x4_t; 2]) -> [int32x4_t; 2] {
        [
            vreinterpretq_s32_u32(vceqq_s32(a[0], b[0])),
            vreinterpretq_s32_u32(vceqq_s32(a[1], b[1])),
        ]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_ne(self, a: [int32x4_t; 2], b: [int32x4_t; 2]) -> [int32x4_t; 2] {
        [
            vreinterpretq_s32_u32(vmvnq_u32(vceqq_s32(a[0], b[0]))),
            vreinterpretq_s32_u32(vmvnq_u32(vceqq_s32(a[1], b[1]))),
        ]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_lt(self, a: [int32x4_t; 2], b: [int32x4_t; 2]) -> [int32x4_t; 2] {
        [
            vreinterpretq_s32_u32(vcltq_s32(a[0], b[0])),
            vreinterpretq_s32_u32(vcltq_s32(a[1], b[1])),
        ]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_le(self, a: [int32x4_t; 2], b: [int32x4_t; 2]) -> [int32x4_t; 2] {
        [
            vreinterpretq_s32_u32(vcleq_s32(a[0], b[0])),
            vreinterpretq_s32_u32(vcleq_s32(a[1], b[1])),
        ]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_gt(self, a: [int32x4_t; 2], b: [int32x4_t; 2]) -> [int32x4_t; 2] {
        [
            vreinterpretq_s32_u32(vcgtq_s32(a[0], b[0])),
            vreinterpretq_s32_u32(vcgtq_s32(a[1], b[1])),
        ]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_ge(self, a: [int32x4_t; 2], b: [int32x4_t; 2]) -> [int32x4_t; 2] {
        [
            vreinterpretq_s32_u32(vcgeq_s32(a[0], b[0])),
            vreinterpretq_s32_u32(vcgeq_s32(a[1], b[1])),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn blend(
        self,
        mask: [int32x4_t; 2],
        if_true: [int32x4_t; 2],
        if_false: [int32x4_t; 2],
    ) -> [int32x4_t; 2] {
        [
            vbslq_s32(vreinterpretq_u32_s32(mask[0]), if_true[0], if_false[0]),
            vbslq_s32(vreinterpretq_u32_s32(mask[1]), if_true[1], if_false[1]),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn reduce_add(self, a: [int32x4_t; 2]) -> i32 {
        {
            let m = vaddq_s32(a[0], a[1]);
            vaddvq_s32(m)
        }
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn not(self, a: [int32x4_t; 2]) -> [int32x4_t; 2] {
        [vmvnq_s32(a[0]), vmvnq_s32(a[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitand(self, a: [int32x4_t; 2], b: [int32x4_t; 2]) -> [int32x4_t; 2] {
        [vandq_s32(a[0], b[0]), vandq_s32(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitor(self, a: [int32x4_t; 2], b: [int32x4_t; 2]) -> [int32x4_t; 2] {
        [vorrq_s32(a[0], b[0]), vorrq_s32(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitxor(self, a: [int32x4_t; 2], b: [int32x4_t; 2]) -> [int32x4_t; 2] {
        [veorq_s32(a[0], b[0]), veorq_s32(a[1], b[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shl_const<const N: i32>(self, a: [int32x4_t; 2]) -> [int32x4_t; 2] {
        [vshlq_n_s32::<N>(a[0]), vshlq_n_s32::<N>(a[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shr_arithmetic_const<const N: i32>(self, a: [int32x4_t; 2]) -> [int32x4_t; 2] {
        const { assert!(N >= 0 && N <= 31) };
        [
            vshlq_s32(a[0], vdupq_n_s32(-N)),
            vshlq_s32(a[1], vdupq_n_s32(-N)),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shr_logical_const<const N: i32>(self, a: [int32x4_t; 2]) -> [int32x4_t; 2] {
        const { assert!(N >= 0 && N <= 31) };
        [
            vreinterpretq_s32_u32(vshlq_u32(vreinterpretq_u32_s32(a[0]), vdupq_n_s32(-N))),
            vreinterpretq_s32_u32(vshlq_u32(vreinterpretq_u32_s32(a[1]), vdupq_n_s32(-N))),
        ]
    }

    // ====== Uniform variable shifts ======
    // Splat the clamped count once, apply to both halves — same
    // clamp rationale as the 128-bit impl (USHL/SSHL read the low
    // byte of each amount lane).

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shl_uniform(self, a: [int32x4_t; 2], count: u32) -> [int32x4_t; 2] {
        let c = vdupq_n_s32(count.min(32) as i32);
        [vshlq_s32(a[0], c), vshlq_s32(a[1], c)]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shr_logical_uniform(self, a: [int32x4_t; 2], count: u32) -> [int32x4_t; 2] {
        let c = vdupq_n_s32(-(count.min(32) as i32));
        [
            vreinterpretq_s32_u32(vshlq_u32(vreinterpretq_u32_s32(a[0]), c)),
            vreinterpretq_s32_u32(vshlq_u32(vreinterpretq_u32_s32(a[1]), c)),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shr_arithmetic_uniform(self, a: [int32x4_t; 2], count: u32) -> [int32x4_t; 2] {
        let c = vdupq_n_s32(-(count.min(31) as i32));
        [vshlq_s32(a[0], c), vshlq_s32(a[1], c)]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn all_true(self, a: [int32x4_t; 2]) -> bool {
        vminvq_u32(vreinterpretq_u32_s32(a[0])) != 0 && vminvq_u32(vreinterpretq_u32_s32(a[1])) != 0
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn any_true(self, a: [int32x4_t; 2]) -> bool {
        vmaxvq_u32(vreinterpretq_u32_s32(a[0])) != 0 || vmaxvq_u32(vreinterpretq_u32_s32(a[1])) != 0
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitmask(self, a: [int32x4_t; 2]) -> u32 {
        {
            let mut bits = 0u32;
            let s0 = vshrq_n_u32::<31>(vreinterpretq_u32_s32(a[0]));
            bits |= vgetq_lane_u32::<0>(s0);
            bits |= vgetq_lane_u32::<1>(s0) << 1;
            bits |= vgetq_lane_u32::<2>(s0) << 2;
            bits |= vgetq_lane_u32::<3>(s0) << 3;
            let s1 = vshrq_n_u32::<31>(vreinterpretq_u32_s32(a[1]));
            bits |= vgetq_lane_u32::<0>(s1) << 4;
            bits |= vgetq_lane_u32::<1>(s1) << 5;
            bits |= vgetq_lane_u32::<2>(s1) << 6;
            bits |= vgetq_lane_u32::<3>(s1) << 7;
            bits
        }
    }
}

#[cfg(target_arch = "aarch64")]
impl U32x4Backend for archmage::NeonToken {
    type Repr = uint32x4_t;

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn splat(self, v: u32) -> uint32x4_t {
        vdupq_n_u32(v)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn zero(self) -> uint32x4_t {
        vdupq_n_u32(0)
    }

    #[inline(always)]
    fn load(self, data: &[u32; 4]) -> uint32x4_t {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [u32; 4]) -> uint32x4_t {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: uint32x4_t, out: &mut [u32; 4]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: uint32x4_t) -> [u32; 4] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn add(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        vaddq_u32(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn sub(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        vsubq_u32(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn mul(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        vmulq_u32(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn min(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        vminq_u32(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn max(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        vmaxq_u32(a, b)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_eq(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        vceqq_u32(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_ne(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        vmvnq_u32(vceqq_u32(a, b))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_lt(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        vcltq_u32(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_le(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        vcleq_u32(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_gt(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        vcgtq_u32(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_ge(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        vcgeq_u32(a, b)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn blend(self, mask: uint32x4_t, if_true: uint32x4_t, if_false: uint32x4_t) -> uint32x4_t {
        vbslq_u32(mask, if_true, if_false)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn reduce_add(self, a: uint32x4_t) -> u32 {
        vaddvq_u32(a)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn not(self, a: uint32x4_t) -> uint32x4_t {
        vmvnq_u32(a)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitand(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        vandq_u32(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitor(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        vorrq_u32(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitxor(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        veorq_u32(a, b)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shl_const<const N: i32>(self, a: uint32x4_t) -> uint32x4_t {
        vshlq_n_u32::<N>(a)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shr_logical_const<const N: i32>(self, a: uint32x4_t) -> uint32x4_t {
        const { assert!(N >= 0 && N <= 31) };
        vshlq_u32(a, vdupq_n_s32(-N))
    }

    // ====== Uniform variable shifts ======
    // Same vshlq lowering as the const forms; the count is clamped
    // because USHL reads only the low byte of each amount lane as a
    // signed value, so an unclamped 256 would wrap to a no-op instead
    // of the contracted zero.

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shl_uniform(self, a: uint32x4_t, count: u32) -> uint32x4_t {
        vshlq_u32(a, vdupq_n_s32(count.min(32) as i32))
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shr_logical_uniform(self, a: uint32x4_t, count: u32) -> uint32x4_t {
        vshlq_u32(a, vdupq_n_s32(-(count.min(32) as i32)))
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn all_true(self, a: uint32x4_t) -> bool {
        vminvq_u32(a) == u32::MAX
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn any_true(self, a: uint32x4_t) -> bool {
        vmaxvq_u32(a) != 0
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitmask(self, a: uint32x4_t) -> u32 {
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

#[cfg(target_arch = "aarch64")]
impl U32x8Backend for archmage::NeonToken {
    type Repr = [uint32x4_t; 2];

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn splat(self, v: u32) -> [uint32x4_t; 2] {
        let v4 = vdupq_n_u32(v);
        [v4, v4]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn zero(self) -> [uint32x4_t; 2] {
        let z = vdupq_n_u32(0);
        [z, z]
    }

    #[inline(always)]
    fn load(self, data: &[u32; 8]) -> [uint32x4_t; 2] {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [u32; 8]) -> [uint32x4_t; 2] {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: [uint32x4_t; 2], out: &mut [u32; 8]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: [uint32x4_t; 2]) -> [u32; 8] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn add(self, a: [uint32x4_t; 2], b: [uint32x4_t; 2]) -> [uint32x4_t; 2] {
        [vaddq_u32(a[0], b[0]), vaddq_u32(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn sub(self, a: [uint32x4_t; 2], b: [uint32x4_t; 2]) -> [uint32x4_t; 2] {
        [vsubq_u32(a[0], b[0]), vsubq_u32(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn mul(self, a: [uint32x4_t; 2], b: [uint32x4_t; 2]) -> [uint32x4_t; 2] {
        [vmulq_u32(a[0], b[0]), vmulq_u32(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn min(self, a: [uint32x4_t; 2], b: [uint32x4_t; 2]) -> [uint32x4_t; 2] {
        [vminq_u32(a[0], b[0]), vminq_u32(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn max(self, a: [uint32x4_t; 2], b: [uint32x4_t; 2]) -> [uint32x4_t; 2] {
        [vmaxq_u32(a[0], b[0]), vmaxq_u32(a[1], b[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_eq(self, a: [uint32x4_t; 2], b: [uint32x4_t; 2]) -> [uint32x4_t; 2] {
        [vceqq_u32(a[0], b[0]), vceqq_u32(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_ne(self, a: [uint32x4_t; 2], b: [uint32x4_t; 2]) -> [uint32x4_t; 2] {
        [
            vmvnq_u32(vceqq_u32(a[0], b[0])),
            vmvnq_u32(vceqq_u32(a[1], b[1])),
        ]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_lt(self, a: [uint32x4_t; 2], b: [uint32x4_t; 2]) -> [uint32x4_t; 2] {
        [vcltq_u32(a[0], b[0]), vcltq_u32(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_le(self, a: [uint32x4_t; 2], b: [uint32x4_t; 2]) -> [uint32x4_t; 2] {
        [vcleq_u32(a[0], b[0]), vcleq_u32(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_gt(self, a: [uint32x4_t; 2], b: [uint32x4_t; 2]) -> [uint32x4_t; 2] {
        [vcgtq_u32(a[0], b[0]), vcgtq_u32(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_ge(self, a: [uint32x4_t; 2], b: [uint32x4_t; 2]) -> [uint32x4_t; 2] {
        [vcgeq_u32(a[0], b[0]), vcgeq_u32(a[1], b[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn blend(
        self,
        mask: [uint32x4_t; 2],
        if_true: [uint32x4_t; 2],
        if_false: [uint32x4_t; 2],
    ) -> [uint32x4_t; 2] {
        [
            vbslq_u32(mask[0], if_true[0], if_false[0]),
            vbslq_u32(mask[1], if_true[1], if_false[1]),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn reduce_add(self, a: [uint32x4_t; 2]) -> u32 {
        {
            let m = vaddq_u32(a[0], a[1]);
            vaddvq_u32(m)
        }
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn not(self, a: [uint32x4_t; 2]) -> [uint32x4_t; 2] {
        [vmvnq_u32(a[0]), vmvnq_u32(a[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitand(self, a: [uint32x4_t; 2], b: [uint32x4_t; 2]) -> [uint32x4_t; 2] {
        [vandq_u32(a[0], b[0]), vandq_u32(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitor(self, a: [uint32x4_t; 2], b: [uint32x4_t; 2]) -> [uint32x4_t; 2] {
        [vorrq_u32(a[0], b[0]), vorrq_u32(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitxor(self, a: [uint32x4_t; 2], b: [uint32x4_t; 2]) -> [uint32x4_t; 2] {
        [veorq_u32(a[0], b[0]), veorq_u32(a[1], b[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shl_const<const N: i32>(self, a: [uint32x4_t; 2]) -> [uint32x4_t; 2] {
        [vshlq_n_u32::<N>(a[0]), vshlq_n_u32::<N>(a[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shr_logical_const<const N: i32>(self, a: [uint32x4_t; 2]) -> [uint32x4_t; 2] {
        const { assert!(N >= 0 && N <= 31) };
        [
            vshlq_u32(a[0], vdupq_n_s32(-N)),
            vshlq_u32(a[1], vdupq_n_s32(-N)),
        ]
    }

    // ====== Uniform variable shifts ======
    // Splat the clamped count once, apply to both halves — same
    // clamp rationale as the 128-bit impl (USHL reads the low byte
    // of each amount lane).

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shl_uniform(self, a: [uint32x4_t; 2], count: u32) -> [uint32x4_t; 2] {
        let c = vdupq_n_s32(count.min(32) as i32);
        [vshlq_u32(a[0], c), vshlq_u32(a[1], c)]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shr_logical_uniform(self, a: [uint32x4_t; 2], count: u32) -> [uint32x4_t; 2] {
        let c = vdupq_n_s32(-(count.min(32) as i32));
        [vshlq_u32(a[0], c), vshlq_u32(a[1], c)]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn all_true(self, a: [uint32x4_t; 2]) -> bool {
        vminvq_u32(a[0]) == u32::MAX && vminvq_u32(a[1]) == u32::MAX
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn any_true(self, a: [uint32x4_t; 2]) -> bool {
        vmaxvq_u32(a[0]) != 0 || vmaxvq_u32(a[1]) != 0
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitmask(self, a: [uint32x4_t; 2]) -> u32 {
        {
            let mut bits = 0u32;
            let s0 = vshrq_n_u32::<31>(a[0]);
            bits |= vgetq_lane_u32::<0>(s0);
            bits |= vgetq_lane_u32::<1>(s0) << 1;
            bits |= vgetq_lane_u32::<2>(s0) << 2;
            bits |= vgetq_lane_u32::<3>(s0) << 3;
            let s1 = vshrq_n_u32::<31>(a[1]);
            bits |= vgetq_lane_u32::<0>(s1) << 4;
            bits |= vgetq_lane_u32::<1>(s1) << 5;
            bits |= vgetq_lane_u32::<2>(s1) << 6;
            bits |= vgetq_lane_u32::<3>(s1) << 7;
            bits
        }
    }
}

#[cfg(target_arch = "aarch64")]
impl I64x2Backend for archmage::NeonToken {
    type Repr = int64x2_t;

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn splat(self, v: i64) -> int64x2_t {
        vdupq_n_s64(v)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn zero(self) -> int64x2_t {
        vdupq_n_s64(0i64)
    }

    #[inline(always)]
    fn load(self, data: &[i64; 2]) -> int64x2_t {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [i64; 2]) -> int64x2_t {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: int64x2_t, out: &mut [i64; 2]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: int64x2_t) -> [i64; 2] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn add(self, a: int64x2_t, b: int64x2_t) -> int64x2_t {
        vaddq_s64(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn sub(self, a: int64x2_t, b: int64x2_t) -> int64x2_t {
        vsubq_s64(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn neg(self, a: int64x2_t) -> int64x2_t {
        vnegq_s64(a)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn min(self, a: int64x2_t, b: int64x2_t) -> int64x2_t {
        // NEON lacks native i64 min; polyfill via compare+select
        {
            let mask = vcltq_s64(a, b);
            vbslq_s64(mask, a, b)
        }
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn max(self, a: int64x2_t, b: int64x2_t) -> int64x2_t {
        // NEON lacks native i64 max; polyfill via compare+select
        {
            let mask = vcgtq_s64(a, b);
            vbslq_s64(mask, a, b)
        }
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn abs(self, a: int64x2_t) -> int64x2_t {
        vabsq_s64(a)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_eq(self, a: int64x2_t, b: int64x2_t) -> int64x2_t {
        vreinterpretq_s64_u64(vceqq_s64(a, b))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_ne(self, a: int64x2_t, b: int64x2_t) -> int64x2_t {
        let eq = vceqq_s64(a, b);
        // NOT via XOR with all-ones
        vreinterpretq_s64_u64(veorq_u64(eq, vdupq_n_u64(u64::MAX)))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_lt(self, a: int64x2_t, b: int64x2_t) -> int64x2_t {
        vreinterpretq_s64_u64(vcltq_s64(a, b))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_le(self, a: int64x2_t, b: int64x2_t) -> int64x2_t {
        vreinterpretq_s64_u64(vcleq_s64(a, b))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_gt(self, a: int64x2_t, b: int64x2_t) -> int64x2_t {
        vreinterpretq_s64_u64(vcgtq_s64(a, b))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_ge(self, a: int64x2_t, b: int64x2_t) -> int64x2_t {
        vreinterpretq_s64_u64(vcgeq_s64(a, b))
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn blend(self, mask: int64x2_t, if_true: int64x2_t, if_false: int64x2_t) -> int64x2_t {
        vbslq_s64(vreinterpretq_u64_s64(mask), if_true, if_false)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn reduce_add(self, a: int64x2_t) -> i64 {
        let sum = vpaddq_s64(a, a);
        vgetq_lane_s64::<0>(sum)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn not(self, a: int64x2_t) -> int64x2_t {
        let ones = vdupq_n_s64(-1i64);
        veorq_s64(a, ones)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitand(self, a: int64x2_t, b: int64x2_t) -> int64x2_t {
        vandq_s64(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitor(self, a: int64x2_t, b: int64x2_t) -> int64x2_t {
        vorrq_s64(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitxor(self, a: int64x2_t, b: int64x2_t) -> int64x2_t {
        veorq_s64(a, b)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shl_const<const N: i32>(self, a: int64x2_t) -> int64x2_t {
        vshlq_n_s64::<N>(a)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shr_arithmetic_const<const N: i32>(self, a: int64x2_t) -> int64x2_t {
        const { assert!(N >= 0 && N <= 63) };
        vshlq_s64(a, vdupq_n_s64((-N) as i64))
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shr_logical_const<const N: i32>(self, a: int64x2_t) -> int64x2_t {
        const { assert!(N >= 0 && N <= 63) };
        vreinterpretq_s64_u64(vshlq_u64(
            vreinterpretq_u64_s64(a),
            vdupq_n_s64((-N) as i64),
        ))
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn all_true(self, a: int64x2_t) -> bool {
        let as_u64 = vreinterpretq_u64_s64(a);
        vgetq_lane_u64::<0>(as_u64) != 0 && vgetq_lane_u64::<1>(as_u64) != 0
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn any_true(self, a: int64x2_t) -> bool {
        let as_u64 = vreinterpretq_u64_s64(a);
        (vgetq_lane_u64::<0>(as_u64) | vgetq_lane_u64::<1>(as_u64)) != 0
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitmask(self, a: int64x2_t) -> u32 {
        let signs = vshrq_n_u64::<63>(vreinterpretq_u64_s64(a));
        ((vgetq_lane_u64::<0>(signs) & 1) | ((vgetq_lane_u64::<1>(signs) & 1) << 1)) as u32
    }
}

#[cfg(target_arch = "aarch64")]
impl I64x4Backend for archmage::NeonToken {
    type Repr = [int64x2_t; 2];

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn splat(self, v: i64) -> [int64x2_t; 2] {
        let v2 = vdupq_n_s64(v);
        [v2, v2]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn zero(self) -> [int64x2_t; 2] {
        let z = vdupq_n_s64(0i64);
        [z, z]
    }

    #[inline(always)]
    fn load(self, data: &[i64; 4]) -> [int64x2_t; 2] {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [i64; 4]) -> [int64x2_t; 2] {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: [int64x2_t; 2], out: &mut [i64; 4]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: [int64x2_t; 2]) -> [i64; 4] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn add(self, a: [int64x2_t; 2], b: [int64x2_t; 2]) -> [int64x2_t; 2] {
        [vaddq_s64(a[0], b[0]), vaddq_s64(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn sub(self, a: [int64x2_t; 2], b: [int64x2_t; 2]) -> [int64x2_t; 2] {
        [vsubq_s64(a[0], b[0]), vsubq_s64(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn neg(self, a: [int64x2_t; 2]) -> [int64x2_t; 2] {
        [vnegq_s64(a[0]), vnegq_s64(a[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn min(self, a: [int64x2_t; 2], b: [int64x2_t; 2]) -> [int64x2_t; 2] {
        // NEON lacks native i64 min; polyfill via compare+select per sub-vector
        [
            vbslq_s64(vcltq_s64(a[0], b[0]), a[0], b[0]),
            vbslq_s64(vcltq_s64(a[1], b[1]), a[1], b[1]),
        ]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn max(self, a: [int64x2_t; 2], b: [int64x2_t; 2]) -> [int64x2_t; 2] {
        // NEON lacks native i64 max; polyfill via compare+select per sub-vector
        [
            vbslq_s64(vcgtq_s64(a[0], b[0]), a[0], b[0]),
            vbslq_s64(vcgtq_s64(a[1], b[1]), a[1], b[1]),
        ]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn abs(self, a: [int64x2_t; 2]) -> [int64x2_t; 2] {
        [vabsq_s64(a[0]), vabsq_s64(a[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_eq(self, a: [int64x2_t; 2], b: [int64x2_t; 2]) -> [int64x2_t; 2] {
        [
            vreinterpretq_s64_u64(vceqq_s64(a[0], b[0])),
            vreinterpretq_s64_u64(vceqq_s64(a[1], b[1])),
        ]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_ne(self, a: [int64x2_t; 2], b: [int64x2_t; 2]) -> [int64x2_t; 2] {
        [
            vreinterpretq_s64_u64(veorq_u64(vceqq_s64(a[0], b[0]), vdupq_n_u64(u64::MAX))),
            vreinterpretq_s64_u64(veorq_u64(vceqq_s64(a[1], b[1]), vdupq_n_u64(u64::MAX))),
        ]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_lt(self, a: [int64x2_t; 2], b: [int64x2_t; 2]) -> [int64x2_t; 2] {
        [
            vreinterpretq_s64_u64(vcltq_s64(a[0], b[0])),
            vreinterpretq_s64_u64(vcltq_s64(a[1], b[1])),
        ]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_le(self, a: [int64x2_t; 2], b: [int64x2_t; 2]) -> [int64x2_t; 2] {
        [
            vreinterpretq_s64_u64(vcleq_s64(a[0], b[0])),
            vreinterpretq_s64_u64(vcleq_s64(a[1], b[1])),
        ]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_gt(self, a: [int64x2_t; 2], b: [int64x2_t; 2]) -> [int64x2_t; 2] {
        [
            vreinterpretq_s64_u64(vcgtq_s64(a[0], b[0])),
            vreinterpretq_s64_u64(vcgtq_s64(a[1], b[1])),
        ]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_ge(self, a: [int64x2_t; 2], b: [int64x2_t; 2]) -> [int64x2_t; 2] {
        [
            vreinterpretq_s64_u64(vcgeq_s64(a[0], b[0])),
            vreinterpretq_s64_u64(vcgeq_s64(a[1], b[1])),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn blend(
        self,
        mask: [int64x2_t; 2],
        if_true: [int64x2_t; 2],
        if_false: [int64x2_t; 2],
    ) -> [int64x2_t; 2] {
        [
            vbslq_s64(vreinterpretq_u64_s64(mask[0]), if_true[0], if_false[0]),
            vbslq_s64(vreinterpretq_u64_s64(mask[1]), if_true[1], if_false[1]),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn reduce_add(self, a: [int64x2_t; 2]) -> i64 {
        {
            let m = vaddq_s64(a[0], a[1]);
            let sum = vpaddq_s64(m, m);
            vgetq_lane_s64::<0>(sum)
        }
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn not(self, a: [int64x2_t; 2]) -> [int64x2_t; 2] {
        [
            veorq_s64(a[0], vdupq_n_s64(-1i64)),
            veorq_s64(a[1], vdupq_n_s64(-1i64)),
        ]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitand(self, a: [int64x2_t; 2], b: [int64x2_t; 2]) -> [int64x2_t; 2] {
        [vandq_s64(a[0], b[0]), vandq_s64(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitor(self, a: [int64x2_t; 2], b: [int64x2_t; 2]) -> [int64x2_t; 2] {
        [vorrq_s64(a[0], b[0]), vorrq_s64(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitxor(self, a: [int64x2_t; 2], b: [int64x2_t; 2]) -> [int64x2_t; 2] {
        [veorq_s64(a[0], b[0]), veorq_s64(a[1], b[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shl_const<const N: i32>(self, a: [int64x2_t; 2]) -> [int64x2_t; 2] {
        [vshlq_n_s64::<N>(a[0]), vshlq_n_s64::<N>(a[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shr_arithmetic_const<const N: i32>(self, a: [int64x2_t; 2]) -> [int64x2_t; 2] {
        const { assert!(N >= 0 && N <= 63) };
        [
            vshlq_s64(a[0], vdupq_n_s64((-N) as i64)),
            vshlq_s64(a[1], vdupq_n_s64((-N) as i64)),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shr_logical_const<const N: i32>(self, a: [int64x2_t; 2]) -> [int64x2_t; 2] {
        const { assert!(N >= 0 && N <= 63) };
        [
            vreinterpretq_s64_u64(vshlq_u64(
                vreinterpretq_u64_s64(a[0]),
                vdupq_n_s64((-N) as i64),
            )),
            vreinterpretq_s64_u64(vshlq_u64(
                vreinterpretq_u64_s64(a[1]),
                vdupq_n_s64((-N) as i64),
            )),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn all_true(self, a: [int64x2_t; 2]) -> bool {
        (vgetq_lane_u64::<0>(vreinterpretq_u64_s64(a[0])) != 0
            && vgetq_lane_u64::<1>(vreinterpretq_u64_s64(a[0])) != 0)
            && (vgetq_lane_u64::<0>(vreinterpretq_u64_s64(a[1])) != 0
                && vgetq_lane_u64::<1>(vreinterpretq_u64_s64(a[1])) != 0)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn any_true(self, a: [int64x2_t; 2]) -> bool {
        ((vgetq_lane_u64::<0>(vreinterpretq_u64_s64(a[0]))
            | vgetq_lane_u64::<1>(vreinterpretq_u64_s64(a[0])))
            != 0)
            || ((vgetq_lane_u64::<0>(vreinterpretq_u64_s64(a[1]))
                | vgetq_lane_u64::<1>(vreinterpretq_u64_s64(a[1])))
                != 0)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitmask(self, a: [int64x2_t; 2]) -> u32 {
        {
            let mut bits = 0u32;
            let s0 = vshrq_n_u64::<63>(vreinterpretq_u64_s64(a[0]));
            bits |= vgetq_lane_u64::<0>(s0) as u32;
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

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn splat(self, v: i8) -> int8x16_t {
        vdupq_n_s8(v)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn zero(self) -> int8x16_t {
        vdupq_n_s8(0)
    }

    #[inline(always)]
    fn load(self, data: &[i8; 16]) -> int8x16_t {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [i8; 16]) -> int8x16_t {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: int8x16_t, out: &mut [i8; 16]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: int8x16_t) -> [i8; 16] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn add(self, a: int8x16_t, b: int8x16_t) -> int8x16_t {
        vaddq_s8(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn sub(self, a: int8x16_t, b: int8x16_t) -> int8x16_t {
        vsubq_s8(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn neg(self, a: int8x16_t) -> int8x16_t {
        vnegq_s8(a)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn min(self, a: int8x16_t, b: int8x16_t) -> int8x16_t {
        vminq_s8(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn max(self, a: int8x16_t, b: int8x16_t) -> int8x16_t {
        vmaxq_s8(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn abs(self, a: int8x16_t) -> int8x16_t {
        vabsq_s8(a)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_eq(self, a: int8x16_t, b: int8x16_t) -> int8x16_t {
        vreinterpretq_s8_u8(vceqq_s8(a, b))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_ne(self, a: int8x16_t, b: int8x16_t) -> int8x16_t {
        vreinterpretq_s8_u8(vmvnq_u8(vceqq_s8(a, b)))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_lt(self, a: int8x16_t, b: int8x16_t) -> int8x16_t {
        vreinterpretq_s8_u8(vcltq_s8(a, b))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_le(self, a: int8x16_t, b: int8x16_t) -> int8x16_t {
        vreinterpretq_s8_u8(vcleq_s8(a, b))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_gt(self, a: int8x16_t, b: int8x16_t) -> int8x16_t {
        vreinterpretq_s8_u8(vcgtq_s8(a, b))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_ge(self, a: int8x16_t, b: int8x16_t) -> int8x16_t {
        vreinterpretq_s8_u8(vcgeq_s8(a, b))
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn blend(self, mask: int8x16_t, if_true: int8x16_t, if_false: int8x16_t) -> int8x16_t {
        vbslq_s8(vreinterpretq_u8_s8(mask), if_true, if_false)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn reduce_add(self, a: int8x16_t) -> i8 {
        vaddvq_s8(a)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn not(self, a: int8x16_t) -> int8x16_t {
        vmvnq_s8(a)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitand(self, a: int8x16_t, b: int8x16_t) -> int8x16_t {
        vandq_s8(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitor(self, a: int8x16_t, b: int8x16_t) -> int8x16_t {
        vorrq_s8(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitxor(self, a: int8x16_t, b: int8x16_t) -> int8x16_t {
        veorq_s8(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shl_const<const N: i32>(self, a: int8x16_t) -> int8x16_t {
        vshlq_n_s8::<N>(a)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shr_logical_const<const N: i32>(self, a: int8x16_t) -> int8x16_t {
        const { assert!(N >= 0 && N <= 7) };
        vreinterpretq_s8_u8(vshlq_u8(vreinterpretq_u8_s8(a), vdupq_n_s8((-N) as i8)))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shr_arithmetic_const<const N: i32>(self, a: int8x16_t) -> int8x16_t {
        const { assert!(N >= 0 && N <= 7) };
        vshlq_s8(a, vdupq_n_s8((-N) as i8))
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shl_uniform(self, a: int8x16_t, count: u32) -> int8x16_t {
        vshlq_s8(a, vdupq_n_s8(count.min(8) as i8))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shr_logical_uniform(self, a: int8x16_t, count: u32) -> int8x16_t {
        vreinterpretq_s8_u8(vshlq_u8(
            vreinterpretq_u8_s8(a),
            vdupq_n_s8(-(count.min(8) as i8)),
        ))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shr_arithmetic_uniform(self, a: int8x16_t, count: u32) -> int8x16_t {
        // Clamping to lane_bits - 1 gives the contracted sign fill.
        vshlq_s8(a, vdupq_n_s8(-(count.min(7) as i8)))
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn saturating_add(self, a: int8x16_t, b: int8x16_t) -> int8x16_t {
        vqaddq_s8(a, b)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn saturating_sub(self, a: int8x16_t, b: int8x16_t) -> int8x16_t {
        vqsubq_s8(a, b)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn all_true(self, a: int8x16_t) -> bool {
        vminvq_u8(vreinterpretq_u8_s8(a)) != 0
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn any_true(self, a: int8x16_t) -> bool {
        vmaxvq_u8(vreinterpretq_u8_s8(a)) != 0
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitmask(self, a: int8x16_t) -> u32 {
        {
            // Shift each byte right by 7 to isolate sign bit
            let bits = vshrq_n_s8::<7>(a);
            // Use polynomial evaluation to pack bits
            // Each byte is now 0 or 1, multiply by position powers of 2
            let powers: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
            let pow_vec = crate::simd_storage::copy(&powers);
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

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn splat(self, v: i8) -> [int8x16_t; 2] {
        let v4 = vdupq_n_s8(v);
        [v4, v4]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn zero(self) -> [int8x16_t; 2] {
        let z = vdupq_n_s8(0);
        [z, z]
    }

    #[inline(always)]
    fn load(self, data: &[i8; 32]) -> [int8x16_t; 2] {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [i8; 32]) -> [int8x16_t; 2] {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: [int8x16_t; 2], out: &mut [i8; 32]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: [int8x16_t; 2]) -> [i8; 32] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn add(self, a: [int8x16_t; 2], b: [int8x16_t; 2]) -> [int8x16_t; 2] {
        [vaddq_s8(a[0], b[0]), vaddq_s8(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn sub(self, a: [int8x16_t; 2], b: [int8x16_t; 2]) -> [int8x16_t; 2] {
        [vsubq_s8(a[0], b[0]), vsubq_s8(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn neg(self, a: [int8x16_t; 2]) -> [int8x16_t; 2] {
        [vnegq_s8(a[0]), vnegq_s8(a[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn min(self, a: [int8x16_t; 2], b: [int8x16_t; 2]) -> [int8x16_t; 2] {
        [vminq_s8(a[0], b[0]), vminq_s8(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn max(self, a: [int8x16_t; 2], b: [int8x16_t; 2]) -> [int8x16_t; 2] {
        [vmaxq_s8(a[0], b[0]), vmaxq_s8(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn abs(self, a: [int8x16_t; 2]) -> [int8x16_t; 2] {
        [vabsq_s8(a[0]), vabsq_s8(a[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_eq(self, a: [int8x16_t; 2], b: [int8x16_t; 2]) -> [int8x16_t; 2] {
        [
            vreinterpretq_s8_u8(vceqq_s8(a[0], b[0])),
            vreinterpretq_s8_u8(vceqq_s8(a[1], b[1])),
        ]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_ne(self, a: [int8x16_t; 2], b: [int8x16_t; 2]) -> [int8x16_t; 2] {
        [
            vreinterpretq_s8_u8(vmvnq_u8(vceqq_s8(a[0], b[0]))),
            vreinterpretq_s8_u8(vmvnq_u8(vceqq_s8(a[1], b[1]))),
        ]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_lt(self, a: [int8x16_t; 2], b: [int8x16_t; 2]) -> [int8x16_t; 2] {
        [
            vreinterpretq_s8_u8(vcltq_s8(a[0], b[0])),
            vreinterpretq_s8_u8(vcltq_s8(a[1], b[1])),
        ]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_le(self, a: [int8x16_t; 2], b: [int8x16_t; 2]) -> [int8x16_t; 2] {
        [
            vreinterpretq_s8_u8(vcleq_s8(a[0], b[0])),
            vreinterpretq_s8_u8(vcleq_s8(a[1], b[1])),
        ]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_gt(self, a: [int8x16_t; 2], b: [int8x16_t; 2]) -> [int8x16_t; 2] {
        [
            vreinterpretq_s8_u8(vcgtq_s8(a[0], b[0])),
            vreinterpretq_s8_u8(vcgtq_s8(a[1], b[1])),
        ]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_ge(self, a: [int8x16_t; 2], b: [int8x16_t; 2]) -> [int8x16_t; 2] {
        [
            vreinterpretq_s8_u8(vcgeq_s8(a[0], b[0])),
            vreinterpretq_s8_u8(vcgeq_s8(a[1], b[1])),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn blend(
        self,
        mask: [int8x16_t; 2],
        if_true: [int8x16_t; 2],
        if_false: [int8x16_t; 2],
    ) -> [int8x16_t; 2] {
        [
            vbslq_s8(vreinterpretq_u8_s8(mask[0]), if_true[0], if_false[0]),
            vbslq_s8(vreinterpretq_u8_s8(mask[1]), if_true[1], if_false[1]),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn reduce_add(self, a: [int8x16_t; 2]) -> i8 {
        let mut sum = 0i8;
        // Iterate the array rather than indexing by range
        // (clippy::needless_range_loop).
        for v in a {
            sum = sum.wrapping_add(vaddvq_s8(v));
        }
        sum
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn not(self, a: [int8x16_t; 2]) -> [int8x16_t; 2] {
        [vmvnq_s8(a[0]), vmvnq_s8(a[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitand(self, a: [int8x16_t; 2], b: [int8x16_t; 2]) -> [int8x16_t; 2] {
        [vandq_s8(a[0], b[0]), vandq_s8(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitor(self, a: [int8x16_t; 2], b: [int8x16_t; 2]) -> [int8x16_t; 2] {
        [vorrq_s8(a[0], b[0]), vorrq_s8(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitxor(self, a: [int8x16_t; 2], b: [int8x16_t; 2]) -> [int8x16_t; 2] {
        [veorq_s8(a[0], b[0]), veorq_s8(a[1], b[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shl_const<const N: i32>(self, a: [int8x16_t; 2]) -> [int8x16_t; 2] {
        [vshlq_n_s8::<N>(a[0]), vshlq_n_s8::<N>(a[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shr_logical_const<const N: i32>(self, a: [int8x16_t; 2]) -> [int8x16_t; 2] {
        const { assert!(N >= 0 && N <= 7) };
        [
            vreinterpretq_s8_u8(vshlq_u8(vreinterpretq_u8_s8(a[0]), vdupq_n_s8((-N) as i8))),
            vreinterpretq_s8_u8(vshlq_u8(vreinterpretq_u8_s8(a[1]), vdupq_n_s8((-N) as i8))),
        ]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shr_arithmetic_const<const N: i32>(self, a: [int8x16_t; 2]) -> [int8x16_t; 2] {
        const { assert!(N >= 0 && N <= 7) };
        [
            vshlq_s8(a[0], vdupq_n_s8((-N) as i8)),
            vshlq_s8(a[1], vdupq_n_s8((-N) as i8)),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shl_uniform(self, a: [int8x16_t; 2], count: u32) -> [int8x16_t; 2] {
        [
            vshlq_s8(a[0], vdupq_n_s8(count.min(8) as i8)),
            vshlq_s8(a[1], vdupq_n_s8(count.min(8) as i8)),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shr_logical_uniform(self, a: [int8x16_t; 2], count: u32) -> [int8x16_t; 2] {
        [
            vreinterpretq_s8_u8(vshlq_u8(
                vreinterpretq_u8_s8(a[0]),
                vdupq_n_s8(-(count.min(8) as i8)),
            )),
            vreinterpretq_s8_u8(vshlq_u8(
                vreinterpretq_u8_s8(a[1]),
                vdupq_n_s8(-(count.min(8) as i8)),
            )),
        ]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shr_arithmetic_uniform(self, a: [int8x16_t; 2], count: u32) -> [int8x16_t; 2] {
        [
            vshlq_s8(a[0], vdupq_n_s8(-(count.min(7) as i8))),
            vshlq_s8(a[1], vdupq_n_s8(-(count.min(7) as i8))),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn saturating_add(self, a: [int8x16_t; 2], b: [int8x16_t; 2]) -> [int8x16_t; 2] {
        [vqaddq_s8(a[0], b[0]), vqaddq_s8(a[1], b[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn saturating_sub(self, a: [int8x16_t; 2], b: [int8x16_t; 2]) -> [int8x16_t; 2] {
        [vqsubq_s8(a[0], b[0]), vqsubq_s8(a[1], b[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn all_true(self, a: [int8x16_t; 2]) -> bool {
        vminvq_u8(vreinterpretq_u8_s8(a[0])) != 0 && vminvq_u8(vreinterpretq_u8_s8(a[1])) != 0
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn any_true(self, a: [int8x16_t; 2]) -> bool {
        vmaxvq_u8(vreinterpretq_u8_s8(a[0])) != 0 || vmaxvq_u8(vreinterpretq_u8_s8(a[1])) != 0
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitmask(self, a: [int8x16_t; 2]) -> u32 {
        // Delegate to NeonToken native bitmask per sub-vector, combine.
        // Enumerate the array rather than indexing by range
        // (clippy::needless_range_loop).
        let mut result = 0u32;
        for (i, v) in a.into_iter().enumerate() {
            result |= <archmage::NeonToken as I8x16Backend>::bitmask(_self, v) << (i * 16);
        }
        result
    }
}

#[cfg(target_arch = "aarch64")]
impl U8x16Backend for archmage::NeonToken {
    type Repr = uint8x16_t;

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn splat(self, v: u8) -> uint8x16_t {
        vdupq_n_u8(v)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn zero(self) -> uint8x16_t {
        vdupq_n_u8(0)
    }

    #[inline(always)]
    fn load(self, data: &[u8; 16]) -> uint8x16_t {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [u8; 16]) -> uint8x16_t {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: uint8x16_t, out: &mut [u8; 16]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: uint8x16_t) -> [u8; 16] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn add(self, a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        vaddq_u8(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn sub(self, a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        vsubq_u8(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn min(self, a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        vminq_u8(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn max(self, a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        vmaxq_u8(a, b)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_eq(self, a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        vceqq_u8(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_ne(self, a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        vmvnq_u8(vceqq_u8(a, b))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_lt(self, a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        vcltq_u8(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_le(self, a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        vcleq_u8(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_gt(self, a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        vcgtq_u8(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_ge(self, a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        vcgeq_u8(a, b)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn blend(self, mask: uint8x16_t, if_true: uint8x16_t, if_false: uint8x16_t) -> uint8x16_t {
        vbslq_u8(mask, if_true, if_false)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn reduce_add(self, a: uint8x16_t) -> u8 {
        vaddvq_u8(a)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn not(self, a: uint8x16_t) -> uint8x16_t {
        vmvnq_u8(a)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitand(self, a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        vandq_u8(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitor(self, a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        vorrq_u8(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitxor(self, a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        veorq_u8(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shl_const<const N: i32>(self, a: uint8x16_t) -> uint8x16_t {
        vshlq_n_u8::<N>(a)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shr_logical_const<const N: i32>(self, a: uint8x16_t) -> uint8x16_t {
        const { assert!(N >= 0 && N <= 7) };
        vshlq_u8(a, vdupq_n_s8((-N) as i8))
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shl_uniform(self, a: uint8x16_t, count: u32) -> uint8x16_t {
        vshlq_u8(a, vdupq_n_s8(count.min(8) as i8))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shr_logical_uniform(self, a: uint8x16_t, count: u32) -> uint8x16_t {
        vshlq_u8(a, vdupq_n_s8(-(count.min(8) as i8)))
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn saturating_add(self, a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        vqaddq_u8(a, b)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn saturating_sub(self, a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        vqsubq_u8(a, b)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn all_true(self, a: uint8x16_t) -> bool {
        vminvq_u8(a) != 0
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn any_true(self, a: uint8x16_t) -> bool {
        vmaxvq_u8(a) != 0
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitmask(self, a: uint8x16_t) -> u32 {
        {
            // Shift each byte right by 7 to isolate sign bit
            let bits = vshrq_n_u8::<7>(a);
            // Use polynomial evaluation to pack bits
            // Each byte is now 0 or 1, multiply by position powers of 2
            let powers: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
            let pow_vec = crate::simd_storage::copy(&powers);
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

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn splat(self, v: u8) -> [uint8x16_t; 2] {
        let v4 = vdupq_n_u8(v);
        [v4, v4]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn zero(self) -> [uint8x16_t; 2] {
        let z = vdupq_n_u8(0);
        [z, z]
    }

    #[inline(always)]
    fn load(self, data: &[u8; 32]) -> [uint8x16_t; 2] {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [u8; 32]) -> [uint8x16_t; 2] {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: [uint8x16_t; 2], out: &mut [u8; 32]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: [uint8x16_t; 2]) -> [u8; 32] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn add(self, a: [uint8x16_t; 2], b: [uint8x16_t; 2]) -> [uint8x16_t; 2] {
        [vaddq_u8(a[0], b[0]), vaddq_u8(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn sub(self, a: [uint8x16_t; 2], b: [uint8x16_t; 2]) -> [uint8x16_t; 2] {
        [vsubq_u8(a[0], b[0]), vsubq_u8(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn min(self, a: [uint8x16_t; 2], b: [uint8x16_t; 2]) -> [uint8x16_t; 2] {
        [vminq_u8(a[0], b[0]), vminq_u8(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn max(self, a: [uint8x16_t; 2], b: [uint8x16_t; 2]) -> [uint8x16_t; 2] {
        [vmaxq_u8(a[0], b[0]), vmaxq_u8(a[1], b[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_eq(self, a: [uint8x16_t; 2], b: [uint8x16_t; 2]) -> [uint8x16_t; 2] {
        [vceqq_u8(a[0], b[0]), vceqq_u8(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_ne(self, a: [uint8x16_t; 2], b: [uint8x16_t; 2]) -> [uint8x16_t; 2] {
        [
            vmvnq_u8(vceqq_u8(a[0], b[0])),
            vmvnq_u8(vceqq_u8(a[1], b[1])),
        ]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_lt(self, a: [uint8x16_t; 2], b: [uint8x16_t; 2]) -> [uint8x16_t; 2] {
        [vcltq_u8(a[0], b[0]), vcltq_u8(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_le(self, a: [uint8x16_t; 2], b: [uint8x16_t; 2]) -> [uint8x16_t; 2] {
        [vcleq_u8(a[0], b[0]), vcleq_u8(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_gt(self, a: [uint8x16_t; 2], b: [uint8x16_t; 2]) -> [uint8x16_t; 2] {
        [vcgtq_u8(a[0], b[0]), vcgtq_u8(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_ge(self, a: [uint8x16_t; 2], b: [uint8x16_t; 2]) -> [uint8x16_t; 2] {
        [vcgeq_u8(a[0], b[0]), vcgeq_u8(a[1], b[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn blend(
        self,
        mask: [uint8x16_t; 2],
        if_true: [uint8x16_t; 2],
        if_false: [uint8x16_t; 2],
    ) -> [uint8x16_t; 2] {
        [
            vbslq_u8(mask[0], if_true[0], if_false[0]),
            vbslq_u8(mask[1], if_true[1], if_false[1]),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn reduce_add(self, a: [uint8x16_t; 2]) -> u8 {
        let mut sum = 0u8;
        // Iterate the array rather than indexing by range
        // (clippy::needless_range_loop).
        for v in a {
            sum = sum.wrapping_add(vaddvq_u8(v));
        }
        sum
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn not(self, a: [uint8x16_t; 2]) -> [uint8x16_t; 2] {
        [vmvnq_u8(a[0]), vmvnq_u8(a[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitand(self, a: [uint8x16_t; 2], b: [uint8x16_t; 2]) -> [uint8x16_t; 2] {
        [vandq_u8(a[0], b[0]), vandq_u8(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitor(self, a: [uint8x16_t; 2], b: [uint8x16_t; 2]) -> [uint8x16_t; 2] {
        [vorrq_u8(a[0], b[0]), vorrq_u8(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitxor(self, a: [uint8x16_t; 2], b: [uint8x16_t; 2]) -> [uint8x16_t; 2] {
        [veorq_u8(a[0], b[0]), veorq_u8(a[1], b[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shl_const<const N: i32>(self, a: [uint8x16_t; 2]) -> [uint8x16_t; 2] {
        [vshlq_n_u8::<N>(a[0]), vshlq_n_u8::<N>(a[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shr_logical_const<const N: i32>(self, a: [uint8x16_t; 2]) -> [uint8x16_t; 2] {
        const { assert!(N >= 0 && N <= 7) };
        [
            vshlq_u8(a[0], vdupq_n_s8((-N) as i8)),
            vshlq_u8(a[1], vdupq_n_s8((-N) as i8)),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shl_uniform(self, a: [uint8x16_t; 2], count: u32) -> [uint8x16_t; 2] {
        [
            vshlq_u8(a[0], vdupq_n_s8(count.min(8) as i8)),
            vshlq_u8(a[1], vdupq_n_s8(count.min(8) as i8)),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shr_logical_uniform(self, a: [uint8x16_t; 2], count: u32) -> [uint8x16_t; 2] {
        [
            vshlq_u8(a[0], vdupq_n_s8(-(count.min(8) as i8))),
            vshlq_u8(a[1], vdupq_n_s8(-(count.min(8) as i8))),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn saturating_add(self, a: [uint8x16_t; 2], b: [uint8x16_t; 2]) -> [uint8x16_t; 2] {
        [vqaddq_u8(a[0], b[0]), vqaddq_u8(a[1], b[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn saturating_sub(self, a: [uint8x16_t; 2], b: [uint8x16_t; 2]) -> [uint8x16_t; 2] {
        [vqsubq_u8(a[0], b[0]), vqsubq_u8(a[1], b[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn all_true(self, a: [uint8x16_t; 2]) -> bool {
        vminvq_u8(a[0]) != 0 && vminvq_u8(a[1]) != 0
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn any_true(self, a: [uint8x16_t; 2]) -> bool {
        vmaxvq_u8(a[0]) != 0 || vmaxvq_u8(a[1]) != 0
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitmask(self, a: [uint8x16_t; 2]) -> u32 {
        // Delegate to NeonToken native bitmask per sub-vector, combine.
        // Enumerate the array rather than indexing by range
        // (clippy::needless_range_loop).
        let mut result = 0u32;
        for (i, v) in a.into_iter().enumerate() {
            result |= <archmage::NeonToken as U8x16Backend>::bitmask(_self, v) << (i * 16);
        }
        result
    }
}

#[cfg(target_arch = "aarch64")]
impl I16x8Backend for archmage::NeonToken {
    type Repr = int16x8_t;

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn splat(self, v: i16) -> int16x8_t {
        vdupq_n_s16(v)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn zero(self) -> int16x8_t {
        vdupq_n_s16(0)
    }

    #[inline(always)]
    fn load(self, data: &[i16; 8]) -> int16x8_t {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [i16; 8]) -> int16x8_t {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: int16x8_t, out: &mut [i16; 8]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: int16x8_t) -> [i16; 8] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn add(self, a: int16x8_t, b: int16x8_t) -> int16x8_t {
        vaddq_s16(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn sub(self, a: int16x8_t, b: int16x8_t) -> int16x8_t {
        vsubq_s16(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn mul(self, a: int16x8_t, b: int16x8_t) -> int16x8_t {
        vmulq_s16(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn neg(self, a: int16x8_t) -> int16x8_t {
        vnegq_s16(a)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn min(self, a: int16x8_t, b: int16x8_t) -> int16x8_t {
        vminq_s16(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn max(self, a: int16x8_t, b: int16x8_t) -> int16x8_t {
        vmaxq_s16(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn abs(self, a: int16x8_t) -> int16x8_t {
        vabsq_s16(a)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_eq(self, a: int16x8_t, b: int16x8_t) -> int16x8_t {
        vreinterpretq_s16_u16(vceqq_s16(a, b))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_ne(self, a: int16x8_t, b: int16x8_t) -> int16x8_t {
        vreinterpretq_s16_u16(vmvnq_u16(vceqq_s16(a, b)))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_lt(self, a: int16x8_t, b: int16x8_t) -> int16x8_t {
        vreinterpretq_s16_u16(vcltq_s16(a, b))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_le(self, a: int16x8_t, b: int16x8_t) -> int16x8_t {
        vreinterpretq_s16_u16(vcleq_s16(a, b))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_gt(self, a: int16x8_t, b: int16x8_t) -> int16x8_t {
        vreinterpretq_s16_u16(vcgtq_s16(a, b))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_ge(self, a: int16x8_t, b: int16x8_t) -> int16x8_t {
        vreinterpretq_s16_u16(vcgeq_s16(a, b))
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn blend(self, mask: int16x8_t, if_true: int16x8_t, if_false: int16x8_t) -> int16x8_t {
        vbslq_s16(vreinterpretq_u16_s16(mask), if_true, if_false)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn reduce_add(self, a: int16x8_t) -> i16 {
        vaddvq_s16(a)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn not(self, a: int16x8_t) -> int16x8_t {
        vmvnq_s16(a)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitand(self, a: int16x8_t, b: int16x8_t) -> int16x8_t {
        vandq_s16(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitor(self, a: int16x8_t, b: int16x8_t) -> int16x8_t {
        vorrq_s16(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitxor(self, a: int16x8_t, b: int16x8_t) -> int16x8_t {
        veorq_s16(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shl_const<const N: i32>(self, a: int16x8_t) -> int16x8_t {
        vshlq_n_s16::<N>(a)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shr_logical_const<const N: i32>(self, a: int16x8_t) -> int16x8_t {
        const { assert!(N >= 0 && N <= 15) };
        vreinterpretq_s16_u16(vshlq_u16(
            vreinterpretq_u16_s16(a),
            vdupq_n_s16((-N) as i16),
        ))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shr_arithmetic_const<const N: i32>(self, a: int16x8_t) -> int16x8_t {
        const { assert!(N >= 0 && N <= 15) };
        vshlq_s16(a, vdupq_n_s16((-N) as i16))
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shl_uniform(self, a: int16x8_t, count: u32) -> int16x8_t {
        vshlq_s16(a, vdupq_n_s16(count.min(16) as i16))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shr_logical_uniform(self, a: int16x8_t, count: u32) -> int16x8_t {
        vreinterpretq_s16_u16(vshlq_u16(
            vreinterpretq_u16_s16(a),
            vdupq_n_s16(-(count.min(16) as i16)),
        ))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shr_arithmetic_uniform(self, a: int16x8_t, count: u32) -> int16x8_t {
        // Clamping to lane_bits - 1 gives the contracted sign fill.
        vshlq_s16(a, vdupq_n_s16(-(count.min(15) as i16)))
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn saturating_add(self, a: int16x8_t, b: int16x8_t) -> int16x8_t {
        vqaddq_s16(a, b)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn saturating_sub(self, a: int16x8_t, b: int16x8_t) -> int16x8_t {
        vqsubq_s16(a, b)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn all_true(self, a: int16x8_t) -> bool {
        vminvq_u16(vreinterpretq_u16_s16(a)) != 0
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn any_true(self, a: int16x8_t) -> bool {
        vmaxvq_u16(vreinterpretq_u16_s16(a)) != 0
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitmask(self, a: int16x8_t) -> u32 {
        (vgetq_lane_u16::<0>(vreinterpretq_u16_s16(vshrq_n_s16::<15>(a))) as u32 & 1)
            | (vgetq_lane_u16::<1>(vreinterpretq_u16_s16(vshrq_n_s16::<15>(a))) as u32 & 1) << 1
            | (vgetq_lane_u16::<2>(vreinterpretq_u16_s16(vshrq_n_s16::<15>(a))) as u32 & 1) << 2
            | (vgetq_lane_u16::<3>(vreinterpretq_u16_s16(vshrq_n_s16::<15>(a))) as u32 & 1) << 3
            | (vgetq_lane_u16::<4>(vreinterpretq_u16_s16(vshrq_n_s16::<15>(a))) as u32 & 1) << 4
            | (vgetq_lane_u16::<5>(vreinterpretq_u16_s16(vshrq_n_s16::<15>(a))) as u32 & 1) << 5
            | (vgetq_lane_u16::<6>(vreinterpretq_u16_s16(vshrq_n_s16::<15>(a))) as u32 & 1) << 6
            | (vgetq_lane_u16::<7>(vreinterpretq_u16_s16(vshrq_n_s16::<15>(a))) as u32 & 1) << 7
    }
}

#[cfg(target_arch = "aarch64")]
impl I16x16Backend for archmage::NeonToken {
    type Repr = [int16x8_t; 2];

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn splat(self, v: i16) -> [int16x8_t; 2] {
        let v4 = vdupq_n_s16(v);
        [v4, v4]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn zero(self) -> [int16x8_t; 2] {
        let z = vdupq_n_s16(0);
        [z, z]
    }

    #[inline(always)]
    fn load(self, data: &[i16; 16]) -> [int16x8_t; 2] {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [i16; 16]) -> [int16x8_t; 2] {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: [int16x8_t; 2], out: &mut [i16; 16]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: [int16x8_t; 2]) -> [i16; 16] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn add(self, a: [int16x8_t; 2], b: [int16x8_t; 2]) -> [int16x8_t; 2] {
        [vaddq_s16(a[0], b[0]), vaddq_s16(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn sub(self, a: [int16x8_t; 2], b: [int16x8_t; 2]) -> [int16x8_t; 2] {
        [vsubq_s16(a[0], b[0]), vsubq_s16(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn mul(self, a: [int16x8_t; 2], b: [int16x8_t; 2]) -> [int16x8_t; 2] {
        [vmulq_s16(a[0], b[0]), vmulq_s16(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn neg(self, a: [int16x8_t; 2]) -> [int16x8_t; 2] {
        [vnegq_s16(a[0]), vnegq_s16(a[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn min(self, a: [int16x8_t; 2], b: [int16x8_t; 2]) -> [int16x8_t; 2] {
        [vminq_s16(a[0], b[0]), vminq_s16(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn max(self, a: [int16x8_t; 2], b: [int16x8_t; 2]) -> [int16x8_t; 2] {
        [vmaxq_s16(a[0], b[0]), vmaxq_s16(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn abs(self, a: [int16x8_t; 2]) -> [int16x8_t; 2] {
        [vabsq_s16(a[0]), vabsq_s16(a[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_eq(self, a: [int16x8_t; 2], b: [int16x8_t; 2]) -> [int16x8_t; 2] {
        [
            vreinterpretq_s16_u16(vceqq_s16(a[0], b[0])),
            vreinterpretq_s16_u16(vceqq_s16(a[1], b[1])),
        ]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_ne(self, a: [int16x8_t; 2], b: [int16x8_t; 2]) -> [int16x8_t; 2] {
        [
            vreinterpretq_s16_u16(vmvnq_u16(vceqq_s16(a[0], b[0]))),
            vreinterpretq_s16_u16(vmvnq_u16(vceqq_s16(a[1], b[1]))),
        ]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_lt(self, a: [int16x8_t; 2], b: [int16x8_t; 2]) -> [int16x8_t; 2] {
        [
            vreinterpretq_s16_u16(vcltq_s16(a[0], b[0])),
            vreinterpretq_s16_u16(vcltq_s16(a[1], b[1])),
        ]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_le(self, a: [int16x8_t; 2], b: [int16x8_t; 2]) -> [int16x8_t; 2] {
        [
            vreinterpretq_s16_u16(vcleq_s16(a[0], b[0])),
            vreinterpretq_s16_u16(vcleq_s16(a[1], b[1])),
        ]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_gt(self, a: [int16x8_t; 2], b: [int16x8_t; 2]) -> [int16x8_t; 2] {
        [
            vreinterpretq_s16_u16(vcgtq_s16(a[0], b[0])),
            vreinterpretq_s16_u16(vcgtq_s16(a[1], b[1])),
        ]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_ge(self, a: [int16x8_t; 2], b: [int16x8_t; 2]) -> [int16x8_t; 2] {
        [
            vreinterpretq_s16_u16(vcgeq_s16(a[0], b[0])),
            vreinterpretq_s16_u16(vcgeq_s16(a[1], b[1])),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn blend(
        self,
        mask: [int16x8_t; 2],
        if_true: [int16x8_t; 2],
        if_false: [int16x8_t; 2],
    ) -> [int16x8_t; 2] {
        [
            vbslq_s16(vreinterpretq_u16_s16(mask[0]), if_true[0], if_false[0]),
            vbslq_s16(vreinterpretq_u16_s16(mask[1]), if_true[1], if_false[1]),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn reduce_add(self, a: [int16x8_t; 2]) -> i16 {
        let mut sum = 0i16;
        // Iterate the array rather than indexing by range
        // (clippy::needless_range_loop).
        for v in a {
            sum = sum.wrapping_add(vaddvq_s16(v));
        }
        sum
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn not(self, a: [int16x8_t; 2]) -> [int16x8_t; 2] {
        [vmvnq_s16(a[0]), vmvnq_s16(a[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitand(self, a: [int16x8_t; 2], b: [int16x8_t; 2]) -> [int16x8_t; 2] {
        [vandq_s16(a[0], b[0]), vandq_s16(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitor(self, a: [int16x8_t; 2], b: [int16x8_t; 2]) -> [int16x8_t; 2] {
        [vorrq_s16(a[0], b[0]), vorrq_s16(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitxor(self, a: [int16x8_t; 2], b: [int16x8_t; 2]) -> [int16x8_t; 2] {
        [veorq_s16(a[0], b[0]), veorq_s16(a[1], b[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shl_const<const N: i32>(self, a: [int16x8_t; 2]) -> [int16x8_t; 2] {
        [vshlq_n_s16::<N>(a[0]), vshlq_n_s16::<N>(a[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shr_logical_const<const N: i32>(self, a: [int16x8_t; 2]) -> [int16x8_t; 2] {
        const { assert!(N >= 0 && N <= 15) };
        [
            vreinterpretq_s16_u16(vshlq_u16(
                vreinterpretq_u16_s16(a[0]),
                vdupq_n_s16((-N) as i16),
            )),
            vreinterpretq_s16_u16(vshlq_u16(
                vreinterpretq_u16_s16(a[1]),
                vdupq_n_s16((-N) as i16),
            )),
        ]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shr_arithmetic_const<const N: i32>(self, a: [int16x8_t; 2]) -> [int16x8_t; 2] {
        const { assert!(N >= 0 && N <= 15) };
        [
            vshlq_s16(a[0], vdupq_n_s16((-N) as i16)),
            vshlq_s16(a[1], vdupq_n_s16((-N) as i16)),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shl_uniform(self, a: [int16x8_t; 2], count: u32) -> [int16x8_t; 2] {
        [
            vshlq_s16(a[0], vdupq_n_s16(count.min(16) as i16)),
            vshlq_s16(a[1], vdupq_n_s16(count.min(16) as i16)),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shr_logical_uniform(self, a: [int16x8_t; 2], count: u32) -> [int16x8_t; 2] {
        [
            vreinterpretq_s16_u16(vshlq_u16(
                vreinterpretq_u16_s16(a[0]),
                vdupq_n_s16(-(count.min(16) as i16)),
            )),
            vreinterpretq_s16_u16(vshlq_u16(
                vreinterpretq_u16_s16(a[1]),
                vdupq_n_s16(-(count.min(16) as i16)),
            )),
        ]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shr_arithmetic_uniform(self, a: [int16x8_t; 2], count: u32) -> [int16x8_t; 2] {
        [
            vshlq_s16(a[0], vdupq_n_s16(-(count.min(15) as i16))),
            vshlq_s16(a[1], vdupq_n_s16(-(count.min(15) as i16))),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn saturating_add(self, a: [int16x8_t; 2], b: [int16x8_t; 2]) -> [int16x8_t; 2] {
        [vqaddq_s16(a[0], b[0]), vqaddq_s16(a[1], b[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn saturating_sub(self, a: [int16x8_t; 2], b: [int16x8_t; 2]) -> [int16x8_t; 2] {
        [vqsubq_s16(a[0], b[0]), vqsubq_s16(a[1], b[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn all_true(self, a: [int16x8_t; 2]) -> bool {
        vminvq_u16(vreinterpretq_u16_s16(a[0])) != 0 && vminvq_u16(vreinterpretq_u16_s16(a[1])) != 0
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn any_true(self, a: [int16x8_t; 2]) -> bool {
        vmaxvq_u16(vreinterpretq_u16_s16(a[0])) != 0 || vmaxvq_u16(vreinterpretq_u16_s16(a[1])) != 0
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitmask(self, a: [int16x8_t; 2]) -> u32 {
        // Delegate to NeonToken native bitmask per sub-vector, combine.
        // Enumerate the array rather than indexing by range
        // (clippy::needless_range_loop).
        let mut result = 0u32;
        for (i, v) in a.into_iter().enumerate() {
            result |= <archmage::NeonToken as I16x8Backend>::bitmask(_self, v) << (i * 8);
        }
        result
    }
}

#[cfg(target_arch = "aarch64")]
impl U16x8Backend for archmage::NeonToken {
    type Repr = uint16x8_t;

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn splat(self, v: u16) -> uint16x8_t {
        vdupq_n_u16(v)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn zero(self) -> uint16x8_t {
        vdupq_n_u16(0)
    }

    #[inline(always)]
    fn load(self, data: &[u16; 8]) -> uint16x8_t {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [u16; 8]) -> uint16x8_t {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: uint16x8_t, out: &mut [u16; 8]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: uint16x8_t) -> [u16; 8] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn add(self, a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        vaddq_u16(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn sub(self, a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        vsubq_u16(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn mul(self, a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        vmulq_u16(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn min(self, a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        vminq_u16(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn max(self, a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        vmaxq_u16(a, b)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_eq(self, a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        vceqq_u16(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_ne(self, a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        vmvnq_u16(vceqq_u16(a, b))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_lt(self, a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        vcltq_u16(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_le(self, a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        vcleq_u16(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_gt(self, a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        vcgtq_u16(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_ge(self, a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        vcgeq_u16(a, b)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn blend(self, mask: uint16x8_t, if_true: uint16x8_t, if_false: uint16x8_t) -> uint16x8_t {
        vbslq_u16(mask, if_true, if_false)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn reduce_add(self, a: uint16x8_t) -> u16 {
        vaddvq_u16(a)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn not(self, a: uint16x8_t) -> uint16x8_t {
        vmvnq_u16(a)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitand(self, a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        vandq_u16(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitor(self, a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        vorrq_u16(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitxor(self, a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        veorq_u16(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shl_const<const N: i32>(self, a: uint16x8_t) -> uint16x8_t {
        vshlq_n_u16::<N>(a)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shr_logical_const<const N: i32>(self, a: uint16x8_t) -> uint16x8_t {
        const { assert!(N >= 0 && N <= 15) };
        vshlq_u16(a, vdupq_n_s16((-N) as i16))
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shl_uniform(self, a: uint16x8_t, count: u32) -> uint16x8_t {
        vshlq_u16(a, vdupq_n_s16(count.min(16) as i16))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shr_logical_uniform(self, a: uint16x8_t, count: u32) -> uint16x8_t {
        vshlq_u16(a, vdupq_n_s16(-(count.min(16) as i16)))
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn saturating_add(self, a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        vqaddq_u16(a, b)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn saturating_sub(self, a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        vqsubq_u16(a, b)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn all_true(self, a: uint16x8_t) -> bool {
        vminvq_u16(a) != 0
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn any_true(self, a: uint16x8_t) -> bool {
        vmaxvq_u16(a) != 0
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitmask(self, a: uint16x8_t) -> u32 {
        (vgetq_lane_u16::<0>(vshrq_n_u16::<15>(a)) as u32 & 1)
            | (vgetq_lane_u16::<1>(vshrq_n_u16::<15>(a)) as u32 & 1) << 1
            | (vgetq_lane_u16::<2>(vshrq_n_u16::<15>(a)) as u32 & 1) << 2
            | (vgetq_lane_u16::<3>(vshrq_n_u16::<15>(a)) as u32 & 1) << 3
            | (vgetq_lane_u16::<4>(vshrq_n_u16::<15>(a)) as u32 & 1) << 4
            | (vgetq_lane_u16::<5>(vshrq_n_u16::<15>(a)) as u32 & 1) << 5
            | (vgetq_lane_u16::<6>(vshrq_n_u16::<15>(a)) as u32 & 1) << 6
            | (vgetq_lane_u16::<7>(vshrq_n_u16::<15>(a)) as u32 & 1) << 7
    }
}

#[cfg(target_arch = "aarch64")]
impl U16x16Backend for archmage::NeonToken {
    type Repr = [uint16x8_t; 2];

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn splat(self, v: u16) -> [uint16x8_t; 2] {
        let v4 = vdupq_n_u16(v);
        [v4, v4]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn zero(self) -> [uint16x8_t; 2] {
        let z = vdupq_n_u16(0);
        [z, z]
    }

    #[inline(always)]
    fn load(self, data: &[u16; 16]) -> [uint16x8_t; 2] {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [u16; 16]) -> [uint16x8_t; 2] {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: [uint16x8_t; 2], out: &mut [u16; 16]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: [uint16x8_t; 2]) -> [u16; 16] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn add(self, a: [uint16x8_t; 2], b: [uint16x8_t; 2]) -> [uint16x8_t; 2] {
        [vaddq_u16(a[0], b[0]), vaddq_u16(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn sub(self, a: [uint16x8_t; 2], b: [uint16x8_t; 2]) -> [uint16x8_t; 2] {
        [vsubq_u16(a[0], b[0]), vsubq_u16(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn mul(self, a: [uint16x8_t; 2], b: [uint16x8_t; 2]) -> [uint16x8_t; 2] {
        [vmulq_u16(a[0], b[0]), vmulq_u16(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn min(self, a: [uint16x8_t; 2], b: [uint16x8_t; 2]) -> [uint16x8_t; 2] {
        [vminq_u16(a[0], b[0]), vminq_u16(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn max(self, a: [uint16x8_t; 2], b: [uint16x8_t; 2]) -> [uint16x8_t; 2] {
        [vmaxq_u16(a[0], b[0]), vmaxq_u16(a[1], b[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_eq(self, a: [uint16x8_t; 2], b: [uint16x8_t; 2]) -> [uint16x8_t; 2] {
        [vceqq_u16(a[0], b[0]), vceqq_u16(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_ne(self, a: [uint16x8_t; 2], b: [uint16x8_t; 2]) -> [uint16x8_t; 2] {
        [
            vmvnq_u16(vceqq_u16(a[0], b[0])),
            vmvnq_u16(vceqq_u16(a[1], b[1])),
        ]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_lt(self, a: [uint16x8_t; 2], b: [uint16x8_t; 2]) -> [uint16x8_t; 2] {
        [vcltq_u16(a[0], b[0]), vcltq_u16(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_le(self, a: [uint16x8_t; 2], b: [uint16x8_t; 2]) -> [uint16x8_t; 2] {
        [vcleq_u16(a[0], b[0]), vcleq_u16(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_gt(self, a: [uint16x8_t; 2], b: [uint16x8_t; 2]) -> [uint16x8_t; 2] {
        [vcgtq_u16(a[0], b[0]), vcgtq_u16(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_ge(self, a: [uint16x8_t; 2], b: [uint16x8_t; 2]) -> [uint16x8_t; 2] {
        [vcgeq_u16(a[0], b[0]), vcgeq_u16(a[1], b[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn blend(
        self,
        mask: [uint16x8_t; 2],
        if_true: [uint16x8_t; 2],
        if_false: [uint16x8_t; 2],
    ) -> [uint16x8_t; 2] {
        [
            vbslq_u16(mask[0], if_true[0], if_false[0]),
            vbslq_u16(mask[1], if_true[1], if_false[1]),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn reduce_add(self, a: [uint16x8_t; 2]) -> u16 {
        let mut sum = 0u16;
        // Iterate the array rather than indexing by range
        // (clippy::needless_range_loop).
        for v in a {
            sum = sum.wrapping_add(vaddvq_u16(v));
        }
        sum
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn not(self, a: [uint16x8_t; 2]) -> [uint16x8_t; 2] {
        [vmvnq_u16(a[0]), vmvnq_u16(a[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitand(self, a: [uint16x8_t; 2], b: [uint16x8_t; 2]) -> [uint16x8_t; 2] {
        [vandq_u16(a[0], b[0]), vandq_u16(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitor(self, a: [uint16x8_t; 2], b: [uint16x8_t; 2]) -> [uint16x8_t; 2] {
        [vorrq_u16(a[0], b[0]), vorrq_u16(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitxor(self, a: [uint16x8_t; 2], b: [uint16x8_t; 2]) -> [uint16x8_t; 2] {
        [veorq_u16(a[0], b[0]), veorq_u16(a[1], b[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shl_const<const N: i32>(self, a: [uint16x8_t; 2]) -> [uint16x8_t; 2] {
        [vshlq_n_u16::<N>(a[0]), vshlq_n_u16::<N>(a[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shr_logical_const<const N: i32>(self, a: [uint16x8_t; 2]) -> [uint16x8_t; 2] {
        const { assert!(N >= 0 && N <= 15) };
        [
            vshlq_u16(a[0], vdupq_n_s16((-N) as i16)),
            vshlq_u16(a[1], vdupq_n_s16((-N) as i16)),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shl_uniform(self, a: [uint16x8_t; 2], count: u32) -> [uint16x8_t; 2] {
        [
            vshlq_u16(a[0], vdupq_n_s16(count.min(16) as i16)),
            vshlq_u16(a[1], vdupq_n_s16(count.min(16) as i16)),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shr_logical_uniform(self, a: [uint16x8_t; 2], count: u32) -> [uint16x8_t; 2] {
        [
            vshlq_u16(a[0], vdupq_n_s16(-(count.min(16) as i16))),
            vshlq_u16(a[1], vdupq_n_s16(-(count.min(16) as i16))),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn saturating_add(self, a: [uint16x8_t; 2], b: [uint16x8_t; 2]) -> [uint16x8_t; 2] {
        [vqaddq_u16(a[0], b[0]), vqaddq_u16(a[1], b[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn saturating_sub(self, a: [uint16x8_t; 2], b: [uint16x8_t; 2]) -> [uint16x8_t; 2] {
        [vqsubq_u16(a[0], b[0]), vqsubq_u16(a[1], b[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn all_true(self, a: [uint16x8_t; 2]) -> bool {
        vminvq_u16(a[0]) != 0 && vminvq_u16(a[1]) != 0
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn any_true(self, a: [uint16x8_t; 2]) -> bool {
        vmaxvq_u16(a[0]) != 0 || vmaxvq_u16(a[1]) != 0
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitmask(self, a: [uint16x8_t; 2]) -> u32 {
        // Delegate to NeonToken native bitmask per sub-vector, combine.
        // Enumerate the array rather than indexing by range
        // (clippy::needless_range_loop).
        let mut result = 0u32;
        for (i, v) in a.into_iter().enumerate() {
            result |= <archmage::NeonToken as U16x8Backend>::bitmask(_self, v) << (i * 8);
        }
        result
    }
}

#[cfg(target_arch = "aarch64")]
impl U64x2Backend for archmage::NeonToken {
    type Repr = uint64x2_t;

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn splat(self, v: u64) -> uint64x2_t {
        vdupq_n_u64(v)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn zero(self) -> uint64x2_t {
        vdupq_n_u64(0)
    }

    #[inline(always)]
    fn load(self, data: &[u64; 2]) -> uint64x2_t {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [u64; 2]) -> uint64x2_t {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: uint64x2_t, out: &mut [u64; 2]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: uint64x2_t) -> [u64; 2] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn add(self, a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        vaddq_u64(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn sub(self, a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        vsubq_u64(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn min(self, a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        vbslq_u64(vcltq_u64(a, b), a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn max(self, a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        vbslq_u64(vcgtq_u64(a, b), a, b)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_eq(self, a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        vceqq_u64(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_ne(self, a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        veorq_u64(vceqq_u64(a, b), vdupq_n_u64(u64::MAX))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_lt(self, a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        vcltq_u64(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_le(self, a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        vcleq_u64(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_gt(self, a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        vcgtq_u64(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_ge(self, a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        vcgeq_u64(a, b)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn blend(self, mask: uint64x2_t, if_true: uint64x2_t, if_false: uint64x2_t) -> uint64x2_t {
        vbslq_u64(mask, if_true, if_false)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn reduce_add(self, a: uint64x2_t) -> u64 {
        vaddvq_u64(a)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn not(self, a: uint64x2_t) -> uint64x2_t {
        veorq_u64(a, vdupq_n_u64(u64::MAX))
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitand(self, a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        vandq_u64(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitor(self, a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        vorrq_u64(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitxor(self, a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        veorq_u64(a, b)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shl_const<const N: i32>(self, a: uint64x2_t) -> uint64x2_t {
        vshlq_n_u64::<N>(a)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shr_logical_const<const N: i32>(self, a: uint64x2_t) -> uint64x2_t {
        const { assert!(N >= 0 && N <= 63) };
        vshlq_u64(a, vdupq_n_s64((-N) as i64))
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn all_true(self, a: uint64x2_t) -> bool {
        vgetq_lane_u64::<0>(a) != 0 && vgetq_lane_u64::<1>(a) != 0
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn any_true(self, a: uint64x2_t) -> bool {
        vgetq_lane_u64::<0>(a) != 0 || vgetq_lane_u64::<1>(a) != 0
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitmask(self, a: uint64x2_t) -> u32 {
        let shift = vshrq_n_u64::<63>(a);
        let lane0 = vgetq_lane_u64::<0>(shift) as u32;
        let lane1 = vgetq_lane_u64::<1>(shift) as u32;
        lane0 | (lane1 << 1)
    }
}

#[cfg(target_arch = "aarch64")]
impl U64x4Backend for archmage::NeonToken {
    type Repr = [uint64x2_t; 2];

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn splat(self, v: u64) -> [uint64x2_t; 2] {
        let v4 = vdupq_n_u64(v);
        [v4, v4]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn zero(self) -> [uint64x2_t; 2] {
        let z = vdupq_n_u64(0);
        [z, z]
    }

    #[inline(always)]
    fn load(self, data: &[u64; 4]) -> [uint64x2_t; 2] {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [u64; 4]) -> [uint64x2_t; 2] {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: [uint64x2_t; 2], out: &mut [u64; 4]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: [uint64x2_t; 2]) -> [u64; 4] {
        crate::simd_storage::cast(repr)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn add(self, a: [uint64x2_t; 2], b: [uint64x2_t; 2]) -> [uint64x2_t; 2] {
        [vaddq_u64(a[0], b[0]), vaddq_u64(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn sub(self, a: [uint64x2_t; 2], b: [uint64x2_t; 2]) -> [uint64x2_t; 2] {
        [vsubq_u64(a[0], b[0]), vsubq_u64(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn min(self, a: [uint64x2_t; 2], b: [uint64x2_t; 2]) -> [uint64x2_t; 2] {
        [
            vbslq_u64(vcltq_u64(a[0], b[0]), a[0], b[0]),
            vbslq_u64(vcltq_u64(a[1], b[1]), a[1], b[1]),
        ]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn max(self, a: [uint64x2_t; 2], b: [uint64x2_t; 2]) -> [uint64x2_t; 2] {
        [
            vbslq_u64(vcgtq_u64(a[0], b[0]), a[0], b[0]),
            vbslq_u64(vcgtq_u64(a[1], b[1]), a[1], b[1]),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_eq(self, a: [uint64x2_t; 2], b: [uint64x2_t; 2]) -> [uint64x2_t; 2] {
        [vceqq_u64(a[0], b[0]), vceqq_u64(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_ne(self, a: [uint64x2_t; 2], b: [uint64x2_t; 2]) -> [uint64x2_t; 2] {
        [
            veorq_u64(vceqq_u64(a[0], b[0]), vdupq_n_u64(u64::MAX)),
            veorq_u64(vceqq_u64(a[1], b[1]), vdupq_n_u64(u64::MAX)),
        ]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_lt(self, a: [uint64x2_t; 2], b: [uint64x2_t; 2]) -> [uint64x2_t; 2] {
        [vcltq_u64(a[0], b[0]), vcltq_u64(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_le(self, a: [uint64x2_t; 2], b: [uint64x2_t; 2]) -> [uint64x2_t; 2] {
        [vcleq_u64(a[0], b[0]), vcleq_u64(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_gt(self, a: [uint64x2_t; 2], b: [uint64x2_t; 2]) -> [uint64x2_t; 2] {
        [vcgtq_u64(a[0], b[0]), vcgtq_u64(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn simd_ge(self, a: [uint64x2_t; 2], b: [uint64x2_t; 2]) -> [uint64x2_t; 2] {
        [vcgeq_u64(a[0], b[0]), vcgeq_u64(a[1], b[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn blend(
        self,
        mask: [uint64x2_t; 2],
        if_true: [uint64x2_t; 2],
        if_false: [uint64x2_t; 2],
    ) -> [uint64x2_t; 2] {
        [
            vbslq_u64(mask[0], if_true[0], if_false[0]),
            vbslq_u64(mask[1], if_true[1], if_false[1]),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn reduce_add(self, a: [uint64x2_t; 2]) -> u64 {
        let mut sum = 0u64;
        // Iterate the array rather than indexing by range
        // (clippy::needless_range_loop).
        for v in a {
            sum = sum.wrapping_add(vaddvq_u64(v));
        }
        sum
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn not(self, a: [uint64x2_t; 2]) -> [uint64x2_t; 2] {
        [
            veorq_u64(a[0], vdupq_n_u64(u64::MAX)),
            veorq_u64(a[1], vdupq_n_u64(u64::MAX)),
        ]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitand(self, a: [uint64x2_t; 2], b: [uint64x2_t; 2]) -> [uint64x2_t; 2] {
        [vandq_u64(a[0], b[0]), vandq_u64(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitor(self, a: [uint64x2_t; 2], b: [uint64x2_t; 2]) -> [uint64x2_t; 2] {
        [vorrq_u64(a[0], b[0]), vorrq_u64(a[1], b[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitxor(self, a: [uint64x2_t; 2], b: [uint64x2_t; 2]) -> [uint64x2_t; 2] {
        [veorq_u64(a[0], b[0]), veorq_u64(a[1], b[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shl_const<const N: i32>(self, a: [uint64x2_t; 2]) -> [uint64x2_t; 2] {
        [vshlq_n_u64::<N>(a[0]), vshlq_n_u64::<N>(a[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn shr_logical_const<const N: i32>(self, a: [uint64x2_t; 2]) -> [uint64x2_t; 2] {
        const { assert!(N >= 0 && N <= 63) };
        [
            vshlq_u64(a[0], vdupq_n_s64((-N) as i64)),
            vshlq_u64(a[1], vdupq_n_s64((-N) as i64)),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn all_true(self, a: [uint64x2_t; 2]) -> bool {
        vgetq_lane_u64::<0>(a[0]) != 0
            && vgetq_lane_u64::<1>(a[0]) != 0
            && vgetq_lane_u64::<0>(a[1]) != 0
            && vgetq_lane_u64::<1>(a[1]) != 0
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn any_true(self, a: [uint64x2_t; 2]) -> bool {
        vgetq_lane_u64::<0>(a[0]) != 0
            || vgetq_lane_u64::<1>(a[0]) != 0
            || vgetq_lane_u64::<0>(a[1]) != 0
            || vgetq_lane_u64::<1>(a[1]) != 0
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitmask(self, a: [uint64x2_t; 2]) -> u32 {
        // Delegate to NeonToken native bitmask per sub-vector, combine.
        // Enumerate the array rather than indexing by range
        // (clippy::needless_range_loop).
        let mut result = 0u32;
        for (i, v) in a.into_iter().enumerate() {
            result |= <archmage::NeonToken as U64x2Backend>::bitmask(_self, v) << (i * 2);
        }
        result
    }
}

#[cfg(target_arch = "aarch64")]
impl F32x4Convert for archmage::NeonToken {
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitcast_f32_to_i32(self, a: float32x4_t) -> int32x4_t {
        vreinterpretq_s32_f32(a)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitcast_i32_to_f32(self, a: int32x4_t) -> float32x4_t {
        vreinterpretq_f32_s32(a)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn convert_f32_to_i32(self, a: float32x4_t) -> int32x4_t {
        vcvtq_s32_f32(a)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn convert_f32_to_i32_round(self, a: float32x4_t) -> int32x4_t {
        vcvtnq_s32_f32(a)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn convert_i32_to_f32(self, a: int32x4_t) -> float32x4_t {
        vcvtq_f32_s32(a)
    }
}

#[cfg(target_arch = "aarch64")]
impl F32x8Convert for archmage::NeonToken {
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitcast_f32_to_i32(self, a: [float32x4_t; 2]) -> [int32x4_t; 2] {
        [vreinterpretq_s32_f32(a[0]), vreinterpretq_s32_f32(a[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitcast_i32_to_f32(self, a: [int32x4_t; 2]) -> [float32x4_t; 2] {
        [vreinterpretq_f32_s32(a[0]), vreinterpretq_f32_s32(a[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn convert_f32_to_i32(self, a: [float32x4_t; 2]) -> [int32x4_t; 2] {
        [vcvtq_s32_f32(a[0]), vcvtq_s32_f32(a[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn convert_f32_to_i32_round(self, a: [float32x4_t; 2]) -> [int32x4_t; 2] {
        [vcvtnq_s32_f32(a[0]), vcvtnq_s32_f32(a[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn convert_i32_to_f32(self, a: [int32x4_t; 2]) -> [float32x4_t; 2] {
        [vcvtq_f32_s32(a[0]), vcvtq_f32_s32(a[1])]
    }
}

#[cfg(all(target_arch = "aarch64", feature = "w512"))]
impl F32x16Convert for archmage::NeonToken {
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitcast_f32_to_i32(self, a: [float32x4_t; 4]) -> [int32x4_t; 4] {
        [
            vreinterpretq_s32_f32(a[0]),
            vreinterpretq_s32_f32(a[1]),
            vreinterpretq_s32_f32(a[2]),
            vreinterpretq_s32_f32(a[3]),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitcast_i32_to_f32(self, a: [int32x4_t; 4]) -> [float32x4_t; 4] {
        [
            vreinterpretq_f32_s32(a[0]),
            vreinterpretq_f32_s32(a[1]),
            vreinterpretq_f32_s32(a[2]),
            vreinterpretq_f32_s32(a[3]),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn convert_f32_to_i32(self, a: [float32x4_t; 4]) -> [int32x4_t; 4] {
        [
            vcvtq_s32_f32(a[0]),
            vcvtq_s32_f32(a[1]),
            vcvtq_s32_f32(a[2]),
            vcvtq_s32_f32(a[3]),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn convert_f32_to_i32_round(self, a: [float32x4_t; 4]) -> [int32x4_t; 4] {
        [
            vcvtnq_s32_f32(a[0]),
            vcvtnq_s32_f32(a[1]),
            vcvtnq_s32_f32(a[2]),
            vcvtnq_s32_f32(a[3]),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn convert_i32_to_f32(self, a: [int32x4_t; 4]) -> [float32x4_t; 4] {
        [
            vcvtq_f32_s32(a[0]),
            vcvtq_f32_s32(a[1]),
            vcvtq_f32_s32(a[2]),
            vcvtq_f32_s32(a[3]),
        ]
    }
}

#[cfg(target_arch = "aarch64")]
impl U32x4Bitcast for archmage::NeonToken {
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitcast_u32_to_i32(self, a: uint32x4_t) -> int32x4_t {
        vreinterpretq_s32_u32(a)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitcast_i32_to_u32(self, a: int32x4_t) -> uint32x4_t {
        vreinterpretq_u32_s32(a)
    }
}

#[cfg(target_arch = "aarch64")]
impl U32x8Bitcast for archmage::NeonToken {
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitcast_u32_to_i32(self, a: [uint32x4_t; 2]) -> [int32x4_t; 2] {
        [vreinterpretq_s32_u32(a[0]), vreinterpretq_s32_u32(a[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitcast_i32_to_u32(self, a: [int32x4_t; 2]) -> [uint32x4_t; 2] {
        [vreinterpretq_u32_s32(a[0]), vreinterpretq_u32_s32(a[1])]
    }
}

#[cfg(target_arch = "aarch64")]
impl I64x2Bitcast for archmage::NeonToken {
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitcast_i64_to_f64(self, a: int64x2_t) -> float64x2_t {
        vreinterpretq_f64_s64(a)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitcast_f64_to_i64(self, a: float64x2_t) -> int64x2_t {
        vreinterpretq_s64_f64(a)
    }
}

#[cfg(target_arch = "aarch64")]
impl I64x4Bitcast for archmage::NeonToken {
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitcast_i64_to_f64(self, a: [int64x2_t; 2]) -> [float64x2_t; 2] {
        [vreinterpretq_f64_s64(a[0]), vreinterpretq_f64_s64(a[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitcast_f64_to_i64(self, a: [float64x2_t; 2]) -> [int64x2_t; 2] {
        [vreinterpretq_s64_f64(a[0]), vreinterpretq_s64_f64(a[1])]
    }
}

#[cfg(target_arch = "aarch64")]
impl I8x16Bitcast for archmage::NeonToken {
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitcast_i8_to_u8(self, a: int8x16_t) -> uint8x16_t {
        vreinterpretq_u8_s8(a)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitcast_u8_to_i8(self, a: uint8x16_t) -> int8x16_t {
        vreinterpretq_s8_u8(a)
    }
}

#[cfg(target_arch = "aarch64")]
impl I8x32Bitcast for archmage::NeonToken {
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitcast_i8_to_u8(self, a: [int8x16_t; 2]) -> [uint8x16_t; 2] {
        [vreinterpretq_u8_s8(a[0]), vreinterpretq_u8_s8(a[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitcast_u8_to_i8(self, a: [uint8x16_t; 2]) -> [int8x16_t; 2] {
        [vreinterpretq_s8_u8(a[0]), vreinterpretq_s8_u8(a[1])]
    }
}

#[cfg(target_arch = "aarch64")]
impl I16x8Bitcast for archmage::NeonToken {
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitcast_i16_to_u16(self, a: int16x8_t) -> uint16x8_t {
        vreinterpretq_u16_s16(a)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitcast_u16_to_i16(self, a: uint16x8_t) -> int16x8_t {
        vreinterpretq_s16_u16(a)
    }
}

#[cfg(target_arch = "aarch64")]
impl I16x16Bitcast for archmage::NeonToken {
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitcast_i16_to_u16(self, a: [int16x8_t; 2]) -> [uint16x8_t; 2] {
        [vreinterpretq_u16_s16(a[0]), vreinterpretq_u16_s16(a[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitcast_u16_to_i16(self, a: [uint16x8_t; 2]) -> [int16x8_t; 2] {
        [vreinterpretq_s16_u16(a[0]), vreinterpretq_s16_u16(a[1])]
    }
}

#[cfg(target_arch = "aarch64")]
impl U64x2Bitcast for archmage::NeonToken {
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitcast_u64_to_i64(self, a: uint64x2_t) -> int64x2_t {
        vreinterpretq_s64_u64(a)
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitcast_i64_to_u64(self, a: int64x2_t) -> uint64x2_t {
        vreinterpretq_u64_s64(a)
    }
}

#[cfg(target_arch = "aarch64")]
impl U64x4Bitcast for archmage::NeonToken {
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitcast_u64_to_i64(self, a: [uint64x2_t; 2]) -> [int64x2_t; 2] {
        [vreinterpretq_s64_u64(a[0]), vreinterpretq_s64_u64(a[1])]
    }
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn bitcast_i64_to_u64(self, a: [int64x2_t; 2]) -> [uint64x2_t; 2] {
        [vreinterpretq_u64_s64(a[0]), vreinterpretq_u64_s64(a[1])]
    }
}

#[cfg(target_arch = "aarch64")]
impl U8x16Widen for archmage::NeonToken {
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn widen_low_u8_to_u16(self, a: uint8x16_t) -> uint16x8_t {
        vmovl_u8(vget_low_u8(a))
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn widen_high_u8_to_u16(self, a: uint8x16_t) -> uint16x8_t {
        vmovl_high_u8(a)
    }
}

#[cfg(target_arch = "aarch64")]
impl U16x8Widen for archmage::NeonToken {
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn widen_low_u16_to_u32(self, a: uint16x8_t) -> uint32x4_t {
        vmovl_u16(vget_low_u16(a))
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn widen_high_u16_to_u32(self, a: uint16x8_t) -> uint32x4_t {
        vmovl_high_u16(a)
    }
}

#[cfg(target_arch = "aarch64")]
impl I8x16Widen for archmage::NeonToken {
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn widen_low_i8_to_i16(self, a: int8x16_t) -> int16x8_t {
        vmovl_s8(vget_low_s8(a))
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn widen_high_i8_to_i16(self, a: int8x16_t) -> int16x8_t {
        vmovl_high_s8(a)
    }
}

#[cfg(target_arch = "aarch64")]
impl I16x8Widen for archmage::NeonToken {
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn widen_low_i16_to_i32(self, a: int16x8_t) -> int32x4_t {
        vmovl_s16(vget_low_s16(a))
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn widen_high_i16_to_i32(self, a: int16x8_t) -> int32x4_t {
        vmovl_high_s16(a)
    }
}

#[cfg(target_arch = "aarch64")]
impl U8x32Widen for archmage::NeonToken {
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn widen_low_u8_to_u16(self, a: [uint8x16_t; 2]) -> [uint16x8_t; 2] {
        [vmovl_u8(vget_low_u8(a[0])), vmovl_high_u8(a[0])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn widen_high_u8_to_u16(self, a: [uint8x16_t; 2]) -> [uint16x8_t; 2] {
        [vmovl_u8(vget_low_u8(a[1])), vmovl_high_u8(a[1])]
    }
}

#[cfg(target_arch = "aarch64")]
impl U16x16Widen for archmage::NeonToken {
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn widen_low_u16_to_u32(self, a: [uint16x8_t; 2]) -> [uint32x4_t; 2] {
        [vmovl_u16(vget_low_u16(a[0])), vmovl_high_u16(a[0])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn widen_high_u16_to_u32(self, a: [uint16x8_t; 2]) -> [uint32x4_t; 2] {
        [vmovl_u16(vget_low_u16(a[1])), vmovl_high_u16(a[1])]
    }
}

#[cfg(target_arch = "aarch64")]
impl I8x32Widen for archmage::NeonToken {
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn widen_low_i8_to_i16(self, a: [int8x16_t; 2]) -> [int16x8_t; 2] {
        [vmovl_s8(vget_low_s8(a[0])), vmovl_high_s8(a[0])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn widen_high_i8_to_i16(self, a: [int8x16_t; 2]) -> [int16x8_t; 2] {
        [vmovl_s8(vget_low_s8(a[1])), vmovl_high_s8(a[1])]
    }
}

#[cfg(target_arch = "aarch64")]
impl I16x16Widen for archmage::NeonToken {
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn widen_low_i16_to_i32(self, a: [int16x8_t; 2]) -> [int32x4_t; 2] {
        [vmovl_s16(vget_low_s16(a[0])), vmovl_high_s16(a[0])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn widen_high_i16_to_i32(self, a: [int16x8_t; 2]) -> [int32x4_t; 2] {
        [vmovl_s16(vget_low_s16(a[1])), vmovl_high_s16(a[1])]
    }
}

#[cfg(target_arch = "aarch64")]
impl I16x8Narrow for archmage::NeonToken {
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn narrow_saturating_i16_to_i8(self, a: int16x8_t, b: int16x8_t) -> int8x16_t {
        vcombine_s8(vqmovn_s16(a), vqmovn_s16(b))
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn narrow_saturating_i16_to_u8(self, a: int16x8_t, b: int16x8_t) -> uint8x16_t {
        vcombine_u8(vqmovun_s16(a), vqmovun_s16(b))
    }
}

#[cfg(target_arch = "aarch64")]
impl I32x4Narrow for archmage::NeonToken {
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn narrow_saturating_i32_to_i16(self, a: int32x4_t, b: int32x4_t) -> int16x8_t {
        vcombine_s16(vqmovn_s32(a), vqmovn_s32(b))
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn narrow_saturating_i32_to_u16(self, a: int32x4_t, b: int32x4_t) -> uint16x8_t {
        vcombine_u16(vqmovun_s32(a), vqmovun_s32(b))
    }
}

#[cfg(target_arch = "aarch64")]
impl I16x16Narrow for archmage::NeonToken {
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn narrow_saturating_i16_to_i8(self, a: [int16x8_t; 2], b: [int16x8_t; 2]) -> [int8x16_t; 2] {
        [
            vcombine_s8(vqmovn_s16(a[0]), vqmovn_s16(a[1])),
            vcombine_s8(vqmovn_s16(b[0]), vqmovn_s16(b[1])),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn narrow_saturating_i16_to_u8(self, a: [int16x8_t; 2], b: [int16x8_t; 2]) -> [uint8x16_t; 2] {
        [
            vcombine_u8(vqmovun_s16(a[0]), vqmovun_s16(a[1])),
            vcombine_u8(vqmovun_s16(b[0]), vqmovun_s16(b[1])),
        ]
    }
}

#[cfg(target_arch = "aarch64")]
impl I32x8Narrow for archmage::NeonToken {
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn narrow_saturating_i32_to_i16(self, a: [int32x4_t; 2], b: [int32x4_t; 2]) -> [int16x8_t; 2] {
        [
            vcombine_s16(vqmovn_s32(a[0]), vqmovn_s32(a[1])),
            vcombine_s16(vqmovn_s32(b[0]), vqmovn_s32(b[1])),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn narrow_saturating_i32_to_u16(self, a: [int32x4_t; 2], b: [int32x4_t; 2]) -> [uint16x8_t; 2] {
        [
            vcombine_u16(vqmovun_s32(a[0]), vqmovun_s32(a[1])),
            vcombine_u16(vqmovun_s32(b[0]), vqmovun_s32(b[1])),
        ]
    }
}
impl I16x8Pairwise for archmage::NeonToken {
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn madd_adjacent(self, a: int16x8_t, b: int16x8_t) -> int32x4_t {
        vpaddq_s32(
            vmull_s16(vget_low_s16(a), vget_low_s16(b)),
            vmull_high_s16(a, b),
        )
    }
}
impl I16x8AbsDiff for archmage::NeonToken {
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn abs_diff(self, a: int16x8_t, b: int16x8_t) -> uint16x8_t {
        vreinterpretq_u16_s16(vabdq_s16(a, b))
    }
}
impl U8x16AbsDiff for archmage::NeonToken {
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn abs_diff(self, a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        vabdq_u8(a, b)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn reduce_add_u32(self, a: uint8x16_t) -> u32 {
        vaddlvq_u8(a) as u32
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn sum_abs_diff(self, a: uint8x16_t, b: uint8x16_t) -> u32 {
        vaddlvq_u8(vabdq_u8(a, b)) as u32
    }
}
impl I16x16Pairwise for archmage::NeonToken {
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn madd_adjacent(self, a: [int16x8_t; 2], b: [int16x8_t; 2]) -> [int32x4_t; 2] {
        [
            vpaddq_s32(
                vmull_s16(vget_low_s16(a[0]), vget_low_s16(b[0])),
                vmull_high_s16(a[0], b[0]),
            ),
            vpaddq_s32(
                vmull_s16(vget_low_s16(a[1]), vget_low_s16(b[1])),
                vmull_high_s16(a[1], b[1]),
            ),
        ]
    }
}
impl I16x16AbsDiff for archmage::NeonToken {
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn abs_diff(self, a: [int16x8_t; 2], b: [int16x8_t; 2]) -> [uint16x8_t; 2] {
        [
            vreinterpretq_u16_s16(vabdq_s16(a[0], b[0])),
            vreinterpretq_u16_s16(vabdq_s16(a[1], b[1])),
        ]
    }
}
impl U8x32AbsDiff for archmage::NeonToken {
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn abs_diff(self, a: [uint8x16_t; 2], b: [uint8x16_t; 2]) -> [uint8x16_t; 2] {
        [vabdq_u8(a[0], b[0]), vabdq_u8(a[1], b[1])]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn reduce_add_u32(self, a: [uint8x16_t; 2]) -> u32 {
        (vaddlvq_u8(a[0]) as u32) + (vaddlvq_u8(a[1]) as u32)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn sum_abs_diff(self, a: [uint8x16_t; 2], b: [uint8x16_t; 2]) -> u32 {
        (vaddlvq_u8(vabdq_u8(a[0], b[0])) as u32) + (vaddlvq_u8(vabdq_u8(a[1], b[1])) as u32)
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "aarch64")]
impl U8x64Widen for archmage::NeonToken {
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn widen_low_u8_to_u16(self, a: [uint8x16_t; 4]) -> [uint16x8_t; 4] {
        [
            vmovl_u8(vget_low_u8(a[0])),
            vmovl_high_u8(a[0]),
            vmovl_u8(vget_low_u8(a[1])),
            vmovl_high_u8(a[1]),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn widen_high_u8_to_u16(self, a: [uint8x16_t; 4]) -> [uint16x8_t; 4] {
        [
            vmovl_u8(vget_low_u8(a[2])),
            vmovl_high_u8(a[2]),
            vmovl_u8(vget_low_u8(a[3])),
            vmovl_high_u8(a[3]),
        ]
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "aarch64")]
impl U16x32Widen for archmage::NeonToken {
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn widen_low_u16_to_u32(self, a: [uint16x8_t; 4]) -> [uint32x4_t; 4] {
        [
            vmovl_u16(vget_low_u16(a[0])),
            vmovl_high_u16(a[0]),
            vmovl_u16(vget_low_u16(a[1])),
            vmovl_high_u16(a[1]),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn widen_high_u16_to_u32(self, a: [uint16x8_t; 4]) -> [uint32x4_t; 4] {
        [
            vmovl_u16(vget_low_u16(a[2])),
            vmovl_high_u16(a[2]),
            vmovl_u16(vget_low_u16(a[3])),
            vmovl_high_u16(a[3]),
        ]
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "aarch64")]
impl I8x64Widen for archmage::NeonToken {
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn widen_low_i8_to_i16(self, a: [int8x16_t; 4]) -> [int16x8_t; 4] {
        [
            vmovl_s8(vget_low_s8(a[0])),
            vmovl_high_s8(a[0]),
            vmovl_s8(vget_low_s8(a[1])),
            vmovl_high_s8(a[1]),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn widen_high_i8_to_i16(self, a: [int8x16_t; 4]) -> [int16x8_t; 4] {
        [
            vmovl_s8(vget_low_s8(a[2])),
            vmovl_high_s8(a[2]),
            vmovl_s8(vget_low_s8(a[3])),
            vmovl_high_s8(a[3]),
        ]
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "aarch64")]
impl I16x32Widen for archmage::NeonToken {
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn widen_low_i16_to_i32(self, a: [int16x8_t; 4]) -> [int32x4_t; 4] {
        [
            vmovl_s16(vget_low_s16(a[0])),
            vmovl_high_s16(a[0]),
            vmovl_s16(vget_low_s16(a[1])),
            vmovl_high_s16(a[1]),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn widen_high_i16_to_i32(self, a: [int16x8_t; 4]) -> [int32x4_t; 4] {
        [
            vmovl_s16(vget_low_s16(a[2])),
            vmovl_high_s16(a[2]),
            vmovl_s16(vget_low_s16(a[3])),
            vmovl_high_s16(a[3]),
        ]
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "aarch64")]
impl I16x32Narrow for archmage::NeonToken {
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn narrow_saturating_i16_to_i8(self, a: [int16x8_t; 4], b: [int16x8_t; 4]) -> [int8x16_t; 4] {
        [
            vcombine_s8(vqmovn_s16(a[0]), vqmovn_s16(a[1])),
            vcombine_s8(vqmovn_s16(a[2]), vqmovn_s16(a[3])),
            vcombine_s8(vqmovn_s16(b[0]), vqmovn_s16(b[1])),
            vcombine_s8(vqmovn_s16(b[2]), vqmovn_s16(b[3])),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn narrow_saturating_i16_to_u8(self, a: [int16x8_t; 4], b: [int16x8_t; 4]) -> [uint8x16_t; 4] {
        [
            vcombine_u8(vqmovun_s16(a[0]), vqmovun_s16(a[1])),
            vcombine_u8(vqmovun_s16(a[2]), vqmovun_s16(a[3])),
            vcombine_u8(vqmovun_s16(b[0]), vqmovun_s16(b[1])),
            vcombine_u8(vqmovun_s16(b[2]), vqmovun_s16(b[3])),
        ]
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "aarch64")]
impl I32x16Narrow for archmage::NeonToken {
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn narrow_saturating_i32_to_i16(self, a: [int32x4_t; 4], b: [int32x4_t; 4]) -> [int16x8_t; 4] {
        [
            vcombine_s16(vqmovn_s32(a[0]), vqmovn_s32(a[1])),
            vcombine_s16(vqmovn_s32(a[2]), vqmovn_s32(a[3])),
            vcombine_s16(vqmovn_s32(b[0]), vqmovn_s32(b[1])),
            vcombine_s16(vqmovn_s32(b[2]), vqmovn_s32(b[3])),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn narrow_saturating_i32_to_u16(self, a: [int32x4_t; 4], b: [int32x4_t; 4]) -> [uint16x8_t; 4] {
        [
            vcombine_u16(vqmovun_s32(a[0]), vqmovun_s32(a[1])),
            vcombine_u16(vqmovun_s32(a[2]), vqmovun_s32(a[3])),
            vcombine_u16(vqmovun_s32(b[0]), vqmovun_s32(b[1])),
            vcombine_u16(vqmovun_s32(b[2]), vqmovun_s32(b[3])),
        ]
    }
}
#[cfg(feature = "w512")]
impl I16x32Pairwise for archmage::NeonToken {
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn madd_adjacent(self, a: [int16x8_t; 4], b: [int16x8_t; 4]) -> [int32x4_t; 4] {
        [
            vpaddq_s32(
                vmull_s16(vget_low_s16(a[0]), vget_low_s16(b[0])),
                vmull_high_s16(a[0], b[0]),
            ),
            vpaddq_s32(
                vmull_s16(vget_low_s16(a[1]), vget_low_s16(b[1])),
                vmull_high_s16(a[1], b[1]),
            ),
            vpaddq_s32(
                vmull_s16(vget_low_s16(a[2]), vget_low_s16(b[2])),
                vmull_high_s16(a[2], b[2]),
            ),
            vpaddq_s32(
                vmull_s16(vget_low_s16(a[3]), vget_low_s16(b[3])),
                vmull_high_s16(a[3], b[3]),
            ),
        ]
    }
}
#[cfg(feature = "w512")]
impl I16x32AbsDiff for archmage::NeonToken {
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn abs_diff(self, a: [int16x8_t; 4], b: [int16x8_t; 4]) -> [uint16x8_t; 4] {
        [
            vreinterpretq_u16_s16(vabdq_s16(a[0], b[0])),
            vreinterpretq_u16_s16(vabdq_s16(a[1], b[1])),
            vreinterpretq_u16_s16(vabdq_s16(a[2], b[2])),
            vreinterpretq_u16_s16(vabdq_s16(a[3], b[3])),
        ]
    }
}
#[cfg(feature = "w512")]
impl U8x64AbsDiff for archmage::NeonToken {
    #[arcane(suppress_const_test, _self = NeonToken)]
    fn abs_diff(self, a: [uint8x16_t; 4], b: [uint8x16_t; 4]) -> [uint8x16_t; 4] {
        [
            vabdq_u8(a[0], b[0]),
            vabdq_u8(a[1], b[1]),
            vabdq_u8(a[2], b[2]),
            vabdq_u8(a[3], b[3]),
        ]
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn reduce_add_u32(self, a: [uint8x16_t; 4]) -> u32 {
        (vaddlvq_u8(a[0]) as u32)
            + (vaddlvq_u8(a[1]) as u32)
            + (vaddlvq_u8(a[2]) as u32)
            + (vaddlvq_u8(a[3]) as u32)
    }

    #[arcane(suppress_const_test, _self = NeonToken)]
    fn sum_abs_diff(self, a: [uint8x16_t; 4], b: [uint8x16_t; 4]) -> u32 {
        (vaddlvq_u8(vabdq_u8(a[0], b[0])) as u32)
            + (vaddlvq_u8(vabdq_u8(a[1], b[1])) as u32)
            + (vaddlvq_u8(vabdq_u8(a[2], b[2])) as u32)
            + (vaddlvq_u8(vabdq_u8(a[3], b[3])) as u32)
    }
}
#[cfg(feature = "w512")]
#[cfg(target_arch = "aarch64")]
impl F32x16Backend for archmage::NeonToken {
    type Repr = [float32x4_t; 4];

    #[inline(always)]
    fn splat(self, v: f32) -> [float32x4_t; 4] {
        let q = <archmage::NeonToken as F32x4Backend>::splat(self, v);
        [q, q, q, q]
    }

    #[inline(always)]
    fn zero(self) -> [float32x4_t; 4] {
        let q = <archmage::NeonToken as F32x4Backend>::zero(self);
        [q, q, q, q]
    }

    #[inline(always)]
    fn load(self, data: &[f32; 16]) -> [float32x4_t; 4] {
        [
            <archmage::NeonToken as F32x4Backend>::load(self, data[0..4].try_into().unwrap()),
            <archmage::NeonToken as F32x4Backend>::load(self, data[4..8].try_into().unwrap()),
            <archmage::NeonToken as F32x4Backend>::load(self, data[8..12].try_into().unwrap()),
            <archmage::NeonToken as F32x4Backend>::load(self, data[12..16].try_into().unwrap()),
        ]
    }

    #[inline(always)]
    fn from_array(self, arr: [f32; 16]) -> [float32x4_t; 4] {
        let mut q0 = [0.0f32; 4];
        let mut q1 = [0.0f32; 4];
        let mut q2 = [0.0f32; 4];
        let mut q3 = [0.0f32; 4];
        q0.copy_from_slice(&arr[0..4]);
        q1.copy_from_slice(&arr[4..8]);
        q2.copy_from_slice(&arr[8..12]);
        q3.copy_from_slice(&arr[12..16]);
        [
            <archmage::NeonToken as F32x4Backend>::from_array(self, q0),
            <archmage::NeonToken as F32x4Backend>::from_array(self, q1),
            <archmage::NeonToken as F32x4Backend>::from_array(self, q2),
            <archmage::NeonToken as F32x4Backend>::from_array(self, q3),
        ]
    }

    #[inline(always)]
    fn store(self, repr: [float32x4_t; 4], out: &mut [f32; 16]) {
        let (o01, o23) = out.split_at_mut(8);
        let (o0, o1) = o01.split_at_mut(4);
        let (o2, o3) = o23.split_at_mut(4);
        <archmage::NeonToken as F32x4Backend>::store(self, repr[0], o0.try_into().unwrap());
        <archmage::NeonToken as F32x4Backend>::store(self, repr[1], o1.try_into().unwrap());
        <archmage::NeonToken as F32x4Backend>::store(self, repr[2], o2.try_into().unwrap());
        <archmage::NeonToken as F32x4Backend>::store(self, repr[3], o3.try_into().unwrap());
    }

    #[inline(always)]
    fn to_array(self, repr: [float32x4_t; 4]) -> [f32; 16] {
        let a0 = <archmage::NeonToken as F32x4Backend>::to_array(self, repr[0]);
        let a1 = <archmage::NeonToken as F32x4Backend>::to_array(self, repr[1]);
        let a2 = <archmage::NeonToken as F32x4Backend>::to_array(self, repr[2]);
        let a3 = <archmage::NeonToken as F32x4Backend>::to_array(self, repr[3]);
        let mut out = [0.0f32; 16];
        out[0..4].copy_from_slice(&a0);
        out[4..8].copy_from_slice(&a1);
        out[8..12].copy_from_slice(&a2);
        out[12..16].copy_from_slice(&a3);
        out
    }

    #[inline(always)]
    fn add(self, a: [float32x4_t; 4], b: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::add(self, a[i], b[i]))
    }

    #[inline(always)]
    fn sub(self, a: [float32x4_t; 4], b: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::sub(self, a[i], b[i]))
    }

    #[inline(always)]
    fn mul(self, a: [float32x4_t; 4], b: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::mul(self, a[i], b[i]))
    }

    #[inline(always)]
    fn div(self, a: [float32x4_t; 4], b: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::div(self, a[i], b[i]))
    }

    #[inline(always)]
    fn neg(self, a: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::neg(self, a[i]))
    }

    #[inline(always)]
    fn min(self, a: [float32x4_t; 4], b: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::min(self, a[i], b[i]))
    }

    #[inline(always)]
    fn max(self, a: [float32x4_t; 4], b: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::max(self, a[i], b[i]))
    }

    #[inline(always)]
    fn sqrt(self, a: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::sqrt(self, a[i]))
    }

    // Reciprocals delegate to the quarter backend so the full-precision
    // and per-platform-approx behaviour matches the narrower widths.
    // (Without these the trait defaults make recip/rcp_approx the identity.)
    #[inline(always)]
    fn rcp_approx(self, a: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::rcp_approx(self, a[i]))
    }

    #[inline(always)]
    fn rsqrt_approx(self, a: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::rsqrt_approx(self, a[i]))
    }

    #[inline(always)]
    fn recip(self, a: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::recip(self, a[i]))
    }

    #[inline(always)]
    fn rsqrt(self, a: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::rsqrt(self, a[i]))
    }

    #[inline(always)]
    fn abs(self, a: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::abs(self, a[i]))
    }

    #[inline(always)]
    fn floor(self, a: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::floor(self, a[i]))
    }

    #[inline(always)]
    fn ceil(self, a: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::ceil(self, a[i]))
    }

    #[inline(always)]
    fn round(self, a: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::round(self, a[i]))
    }

    #[inline(always)]
    fn mul_add(
        self,
        a: [float32x4_t; 4],
        b: [float32x4_t; 4],
        c: [float32x4_t; 4],
    ) -> [float32x4_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as F32x4Backend>::mul_add(self, a[i], b[i], c[i])
        })
    }

    #[inline(always)]
    fn mul_sub(
        self,
        a: [float32x4_t; 4],
        b: [float32x4_t; 4],
        c: [float32x4_t; 4],
    ) -> [float32x4_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as F32x4Backend>::mul_sub(self, a[i], b[i], c[i])
        })
    }

    #[inline(always)]
    fn reduce_add(self, a: [float32x4_t; 4]) -> f32 {
        <archmage::NeonToken as F32x4Backend>::reduce_add(self, a[0])
            + <archmage::NeonToken as F32x4Backend>::reduce_add(self, a[1])
            + <archmage::NeonToken as F32x4Backend>::reduce_add(self, a[2])
            + <archmage::NeonToken as F32x4Backend>::reduce_add(self, a[3])
    }

    #[inline(always)]
    fn reduce_min(self, a: [float32x4_t; 4]) -> f32 {
        let m01 = {
            let l = <archmage::NeonToken as F32x4Backend>::reduce_min(self, a[0]);
            let r = <archmage::NeonToken as F32x4Backend>::reduce_min(self, a[1]);
            if l < r { l } else { r }
        };
        let m23 = {
            let l = <archmage::NeonToken as F32x4Backend>::reduce_min(self, a[2]);
            let r = <archmage::NeonToken as F32x4Backend>::reduce_min(self, a[3]);
            if l < r { l } else { r }
        };
        if m01 < m23 { m01 } else { m23 }
    }

    #[inline(always)]
    fn reduce_max(self, a: [float32x4_t; 4]) -> f32 {
        let m01 = {
            let l = <archmage::NeonToken as F32x4Backend>::reduce_max(self, a[0]);
            let r = <archmage::NeonToken as F32x4Backend>::reduce_max(self, a[1]);
            if l > r { l } else { r }
        };
        let m23 = {
            let l = <archmage::NeonToken as F32x4Backend>::reduce_max(self, a[2]);
            let r = <archmage::NeonToken as F32x4Backend>::reduce_max(self, a[3]);
            if l > r { l } else { r }
        };
        if m01 > m23 { m01 } else { m23 }
    }

    #[inline(always)]
    fn simd_eq(self, a: [float32x4_t; 4], b: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::simd_eq(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ne(self, a: [float32x4_t; 4], b: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::simd_ne(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_lt(self, a: [float32x4_t; 4], b: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::simd_lt(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_le(self, a: [float32x4_t; 4], b: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::simd_le(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_gt(self, a: [float32x4_t; 4], b: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::simd_gt(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ge(self, a: [float32x4_t; 4], b: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::simd_ge(self, a[i], b[i]))
    }

    #[inline(always)]
    fn blend(
        self,
        mask: [float32x4_t; 4],
        if_true: [float32x4_t; 4],
        if_false: [float32x4_t; 4],
    ) -> [float32x4_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as F32x4Backend>::blend(self, mask[i], if_true[i], if_false[i])
        })
    }

    #[inline(always)]
    fn not(self, a: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::not(self, a[i]))
    }

    #[inline(always)]
    fn bitand(self, a: [float32x4_t; 4], b: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::bitand(self, a[i], b[i]))
    }

    #[inline(always)]
    fn bitor(self, a: [float32x4_t; 4], b: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::bitor(self, a[i], b[i]))
    }

    #[inline(always)]
    fn bitxor(self, a: [float32x4_t; 4], b: [float32x4_t; 4]) -> [float32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F32x4Backend>::bitxor(self, a[i], b[i]))
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "aarch64")]
impl F64x8Backend for archmage::NeonToken {
    type Repr = [float64x2_t; 4];

    #[inline(always)]
    fn splat(self, v: f64) -> [float64x2_t; 4] {
        let q = <archmage::NeonToken as F64x2Backend>::splat(self, v);
        [q, q, q, q]
    }

    #[inline(always)]
    fn zero(self) -> [float64x2_t; 4] {
        let q = <archmage::NeonToken as F64x2Backend>::zero(self);
        [q, q, q, q]
    }

    #[inline(always)]
    fn load(self, data: &[f64; 8]) -> [float64x2_t; 4] {
        [
            <archmage::NeonToken as F64x2Backend>::load(self, data[0..2].try_into().unwrap()),
            <archmage::NeonToken as F64x2Backend>::load(self, data[2..4].try_into().unwrap()),
            <archmage::NeonToken as F64x2Backend>::load(self, data[4..6].try_into().unwrap()),
            <archmage::NeonToken as F64x2Backend>::load(self, data[6..8].try_into().unwrap()),
        ]
    }

    #[inline(always)]
    fn from_array(self, arr: [f64; 8]) -> [float64x2_t; 4] {
        let mut q0 = [0.0f64; 2];
        let mut q1 = [0.0f64; 2];
        let mut q2 = [0.0f64; 2];
        let mut q3 = [0.0f64; 2];
        q0.copy_from_slice(&arr[0..2]);
        q1.copy_from_slice(&arr[2..4]);
        q2.copy_from_slice(&arr[4..6]);
        q3.copy_from_slice(&arr[6..8]);
        [
            <archmage::NeonToken as F64x2Backend>::from_array(self, q0),
            <archmage::NeonToken as F64x2Backend>::from_array(self, q1),
            <archmage::NeonToken as F64x2Backend>::from_array(self, q2),
            <archmage::NeonToken as F64x2Backend>::from_array(self, q3),
        ]
    }

    #[inline(always)]
    fn store(self, repr: [float64x2_t; 4], out: &mut [f64; 8]) {
        let (o01, o23) = out.split_at_mut(4);
        let (o0, o1) = o01.split_at_mut(2);
        let (o2, o3) = o23.split_at_mut(2);
        <archmage::NeonToken as F64x2Backend>::store(self, repr[0], o0.try_into().unwrap());
        <archmage::NeonToken as F64x2Backend>::store(self, repr[1], o1.try_into().unwrap());
        <archmage::NeonToken as F64x2Backend>::store(self, repr[2], o2.try_into().unwrap());
        <archmage::NeonToken as F64x2Backend>::store(self, repr[3], o3.try_into().unwrap());
    }

    #[inline(always)]
    fn to_array(self, repr: [float64x2_t; 4]) -> [f64; 8] {
        let a0 = <archmage::NeonToken as F64x2Backend>::to_array(self, repr[0]);
        let a1 = <archmage::NeonToken as F64x2Backend>::to_array(self, repr[1]);
        let a2 = <archmage::NeonToken as F64x2Backend>::to_array(self, repr[2]);
        let a3 = <archmage::NeonToken as F64x2Backend>::to_array(self, repr[3]);
        let mut out = [0.0f64; 8];
        out[0..2].copy_from_slice(&a0);
        out[2..4].copy_from_slice(&a1);
        out[4..6].copy_from_slice(&a2);
        out[6..8].copy_from_slice(&a3);
        out
    }

    #[inline(always)]
    fn add(self, a: [float64x2_t; 4], b: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::add(self, a[i], b[i]))
    }

    #[inline(always)]
    fn sub(self, a: [float64x2_t; 4], b: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::sub(self, a[i], b[i]))
    }

    #[inline(always)]
    fn mul(self, a: [float64x2_t; 4], b: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::mul(self, a[i], b[i]))
    }

    #[inline(always)]
    fn div(self, a: [float64x2_t; 4], b: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::div(self, a[i], b[i]))
    }

    #[inline(always)]
    fn neg(self, a: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::neg(self, a[i]))
    }

    #[inline(always)]
    fn min(self, a: [float64x2_t; 4], b: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::min(self, a[i], b[i]))
    }

    #[inline(always)]
    fn max(self, a: [float64x2_t; 4], b: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::max(self, a[i], b[i]))
    }

    #[inline(always)]
    fn sqrt(self, a: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::sqrt(self, a[i]))
    }

    // Reciprocals delegate to the quarter backend so the full-precision
    // and per-platform-approx behaviour matches the narrower widths.
    // (Without these the trait defaults make recip/rcp_approx the identity.)
    #[inline(always)]
    fn rcp_approx(self, a: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::rcp_approx(self, a[i]))
    }

    #[inline(always)]
    fn rsqrt_approx(self, a: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::rsqrt_approx(self, a[i]))
    }

    #[inline(always)]
    fn recip(self, a: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::recip(self, a[i]))
    }

    #[inline(always)]
    fn rsqrt(self, a: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::rsqrt(self, a[i]))
    }

    #[inline(always)]
    fn abs(self, a: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::abs(self, a[i]))
    }

    #[inline(always)]
    fn floor(self, a: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::floor(self, a[i]))
    }

    #[inline(always)]
    fn ceil(self, a: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::ceil(self, a[i]))
    }

    #[inline(always)]
    fn round(self, a: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::round(self, a[i]))
    }

    #[inline(always)]
    fn mul_add(
        self,
        a: [float64x2_t; 4],
        b: [float64x2_t; 4],
        c: [float64x2_t; 4],
    ) -> [float64x2_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as F64x2Backend>::mul_add(self, a[i], b[i], c[i])
        })
    }

    #[inline(always)]
    fn mul_sub(
        self,
        a: [float64x2_t; 4],
        b: [float64x2_t; 4],
        c: [float64x2_t; 4],
    ) -> [float64x2_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as F64x2Backend>::mul_sub(self, a[i], b[i], c[i])
        })
    }

    #[inline(always)]
    fn reduce_add(self, a: [float64x2_t; 4]) -> f64 {
        <archmage::NeonToken as F64x2Backend>::reduce_add(self, a[0])
            + <archmage::NeonToken as F64x2Backend>::reduce_add(self, a[1])
            + <archmage::NeonToken as F64x2Backend>::reduce_add(self, a[2])
            + <archmage::NeonToken as F64x2Backend>::reduce_add(self, a[3])
    }

    #[inline(always)]
    fn reduce_min(self, a: [float64x2_t; 4]) -> f64 {
        let m01 = {
            let l = <archmage::NeonToken as F64x2Backend>::reduce_min(self, a[0]);
            let r = <archmage::NeonToken as F64x2Backend>::reduce_min(self, a[1]);
            if l < r { l } else { r }
        };
        let m23 = {
            let l = <archmage::NeonToken as F64x2Backend>::reduce_min(self, a[2]);
            let r = <archmage::NeonToken as F64x2Backend>::reduce_min(self, a[3]);
            if l < r { l } else { r }
        };
        if m01 < m23 { m01 } else { m23 }
    }

    #[inline(always)]
    fn reduce_max(self, a: [float64x2_t; 4]) -> f64 {
        let m01 = {
            let l = <archmage::NeonToken as F64x2Backend>::reduce_max(self, a[0]);
            let r = <archmage::NeonToken as F64x2Backend>::reduce_max(self, a[1]);
            if l > r { l } else { r }
        };
        let m23 = {
            let l = <archmage::NeonToken as F64x2Backend>::reduce_max(self, a[2]);
            let r = <archmage::NeonToken as F64x2Backend>::reduce_max(self, a[3]);
            if l > r { l } else { r }
        };
        if m01 > m23 { m01 } else { m23 }
    }

    #[inline(always)]
    fn simd_eq(self, a: [float64x2_t; 4], b: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::simd_eq(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ne(self, a: [float64x2_t; 4], b: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::simd_ne(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_lt(self, a: [float64x2_t; 4], b: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::simd_lt(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_le(self, a: [float64x2_t; 4], b: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::simd_le(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_gt(self, a: [float64x2_t; 4], b: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::simd_gt(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ge(self, a: [float64x2_t; 4], b: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::simd_ge(self, a[i], b[i]))
    }

    #[inline(always)]
    fn blend(
        self,
        mask: [float64x2_t; 4],
        if_true: [float64x2_t; 4],
        if_false: [float64x2_t; 4],
    ) -> [float64x2_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as F64x2Backend>::blend(self, mask[i], if_true[i], if_false[i])
        })
    }

    #[inline(always)]
    fn not(self, a: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::not(self, a[i]))
    }

    #[inline(always)]
    fn bitand(self, a: [float64x2_t; 4], b: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::bitand(self, a[i], b[i]))
    }

    #[inline(always)]
    fn bitor(self, a: [float64x2_t; 4], b: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::bitor(self, a[i], b[i]))
    }

    #[inline(always)]
    fn bitxor(self, a: [float64x2_t; 4], b: [float64x2_t; 4]) -> [float64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as F64x2Backend>::bitxor(self, a[i], b[i]))
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "aarch64")]
impl I8x64Backend for archmage::NeonToken {
    type Repr = [int8x16_t; 4];

    #[inline(always)]
    fn splat(self, v: i8) -> [int8x16_t; 4] {
        let q = <archmage::NeonToken as I8x16Backend>::splat(self, v);
        [q, q, q, q]
    }

    #[inline(always)]
    fn zero(self) -> [int8x16_t; 4] {
        let q = <archmage::NeonToken as I8x16Backend>::zero(self);
        [q, q, q, q]
    }

    #[inline(always)]
    fn load(self, data: &[i8; 64]) -> [int8x16_t; 4] {
        [
            <archmage::NeonToken as I8x16Backend>::load(self, data[0..16].try_into().unwrap()),
            <archmage::NeonToken as I8x16Backend>::load(self, data[16..32].try_into().unwrap()),
            <archmage::NeonToken as I8x16Backend>::load(self, data[32..48].try_into().unwrap()),
            <archmage::NeonToken as I8x16Backend>::load(self, data[48..64].try_into().unwrap()),
        ]
    }

    #[inline(always)]
    fn from_array(self, arr: [i8; 64]) -> [int8x16_t; 4] {
        let mut q0 = [0; 16];
        let mut q1 = [0; 16];
        let mut q2 = [0; 16];
        let mut q3 = [0; 16];
        q0.copy_from_slice(&arr[0..16]);
        q1.copy_from_slice(&arr[16..32]);
        q2.copy_from_slice(&arr[32..48]);
        q3.copy_from_slice(&arr[48..64]);
        [
            <archmage::NeonToken as I8x16Backend>::from_array(self, q0),
            <archmage::NeonToken as I8x16Backend>::from_array(self, q1),
            <archmage::NeonToken as I8x16Backend>::from_array(self, q2),
            <archmage::NeonToken as I8x16Backend>::from_array(self, q3),
        ]
    }

    #[inline(always)]
    fn store(self, repr: [int8x16_t; 4], out: &mut [i8; 64]) {
        let (o01, o23) = out.split_at_mut(32);
        let (o0, o1) = o01.split_at_mut(16);
        let (o2, o3) = o23.split_at_mut(16);
        <archmage::NeonToken as I8x16Backend>::store(self, repr[0], o0.try_into().unwrap());
        <archmage::NeonToken as I8x16Backend>::store(self, repr[1], o1.try_into().unwrap());
        <archmage::NeonToken as I8x16Backend>::store(self, repr[2], o2.try_into().unwrap());
        <archmage::NeonToken as I8x16Backend>::store(self, repr[3], o3.try_into().unwrap());
    }

    #[inline(always)]
    fn to_array(self, repr: [int8x16_t; 4]) -> [i8; 64] {
        let a0 = <archmage::NeonToken as I8x16Backend>::to_array(self, repr[0]);
        let a1 = <archmage::NeonToken as I8x16Backend>::to_array(self, repr[1]);
        let a2 = <archmage::NeonToken as I8x16Backend>::to_array(self, repr[2]);
        let a3 = <archmage::NeonToken as I8x16Backend>::to_array(self, repr[3]);
        let mut out = [0; 64];
        out[0..16].copy_from_slice(&a0);
        out[16..32].copy_from_slice(&a1);
        out[32..48].copy_from_slice(&a2);
        out[48..64].copy_from_slice(&a3);
        out
    }

    #[inline(always)]
    fn add(self, a: [int8x16_t; 4], b: [int8x16_t; 4]) -> [int8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I8x16Backend>::add(self, a[i], b[i]))
    }

    #[inline(always)]
    fn sub(self, a: [int8x16_t; 4], b: [int8x16_t; 4]) -> [int8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I8x16Backend>::sub(self, a[i], b[i]))
    }

    #[inline(always)]
    fn neg(self, a: [int8x16_t; 4]) -> [int8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I8x16Backend>::neg(self, a[i]))
    }

    #[inline(always)]
    fn min(self, a: [int8x16_t; 4], b: [int8x16_t; 4]) -> [int8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I8x16Backend>::min(self, a[i], b[i]))
    }

    #[inline(always)]
    fn max(self, a: [int8x16_t; 4], b: [int8x16_t; 4]) -> [int8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I8x16Backend>::max(self, a[i], b[i]))
    }

    #[inline(always)]
    fn abs(self, a: [int8x16_t; 4]) -> [int8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I8x16Backend>::abs(self, a[i]))
    }

    #[inline(always)]
    fn reduce_add(self, a: [int8x16_t; 4]) -> i8 {
        <archmage::NeonToken as I8x16Backend>::reduce_add(self, a[0])
            .wrapping_add(<archmage::NeonToken as I8x16Backend>::reduce_add(
                self, a[1],
            ))
            .wrapping_add(<archmage::NeonToken as I8x16Backend>::reduce_add(
                self, a[2],
            ))
            .wrapping_add(<archmage::NeonToken as I8x16Backend>::reduce_add(
                self, a[3],
            ))
    }

    #[inline(always)]
    fn shl_const<const N: i32>(self, a: [int8x16_t; 4]) -> [int8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I8x16Backend>::shl_const::<N>(self, a[i]))
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(self, a: [int8x16_t; 4]) -> [int8x16_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as I8x16Backend>::shr_arithmetic_const::<N>(self, a[i])
        })
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: [int8x16_t; 4]) -> [int8x16_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as I8x16Backend>::shr_logical_const::<N>(self, a[i])
        })
    }

    #[inline(always)]
    fn shl_uniform(self, a: [int8x16_t; 4], count: u32) -> [int8x16_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as I8x16Backend>::shl_uniform(self, a[i], count)
        })
    }

    #[inline(always)]
    fn shr_logical_uniform(self, a: [int8x16_t; 4], count: u32) -> [int8x16_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as I8x16Backend>::shr_logical_uniform(self, a[i], count)
        })
    }

    #[inline(always)]
    fn shr_arithmetic_uniform(self, a: [int8x16_t; 4], count: u32) -> [int8x16_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as I8x16Backend>::shr_arithmetic_uniform(self, a[i], count)
        })
    }

    #[inline(always)]
    fn saturating_add(self, a: [int8x16_t; 4], b: [int8x16_t; 4]) -> [int8x16_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as I8x16Backend>::saturating_add(self, a[i], b[i])
        })
    }

    #[inline(always)]
    fn saturating_sub(self, a: [int8x16_t; 4], b: [int8x16_t; 4]) -> [int8x16_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as I8x16Backend>::saturating_sub(self, a[i], b[i])
        })
    }

    // Reduce the quarters with vector AND/OR first, then ONE lane
    // reduction (fearless_simd's polyfill lesson; branchless).
    #[inline(always)]
    fn all_true(self, a: [int8x16_t; 4]) -> bool {
        let c01 = <archmage::NeonToken as I8x16Backend>::bitand(self, a[0], a[1]);
        let c23 = <archmage::NeonToken as I8x16Backend>::bitand(self, a[2], a[3]);
        let c = <archmage::NeonToken as I8x16Backend>::bitand(self, c01, c23);
        <archmage::NeonToken as I8x16Backend>::all_true(self, c)
    }

    #[inline(always)]
    fn any_true(self, a: [int8x16_t; 4]) -> bool {
        let c01 = <archmage::NeonToken as I8x16Backend>::bitor(self, a[0], a[1]);
        let c23 = <archmage::NeonToken as I8x16Backend>::bitor(self, a[2], a[3]);
        let c = <archmage::NeonToken as I8x16Backend>::bitor(self, c01, c23);
        <archmage::NeonToken as I8x16Backend>::any_true(self, c)
    }

    #[inline(always)]
    fn bitmask(self, a: [int8x16_t; 4]) -> u64 {
        let q0 = <archmage::NeonToken as I8x16Backend>::bitmask(self, a[0]) as u64;
        let q1 = <archmage::NeonToken as I8x16Backend>::bitmask(self, a[1]) as u64;
        let q2 = <archmage::NeonToken as I8x16Backend>::bitmask(self, a[2]) as u64;
        let q3 = <archmage::NeonToken as I8x16Backend>::bitmask(self, a[3]) as u64;
        q0 | (q1 << 16) | (q2 << 32) | (q3 << 48)
    }

    #[inline(always)]
    fn simd_eq(self, a: [int8x16_t; 4], b: [int8x16_t; 4]) -> [int8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I8x16Backend>::simd_eq(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ne(self, a: [int8x16_t; 4], b: [int8x16_t; 4]) -> [int8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I8x16Backend>::simd_ne(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_lt(self, a: [int8x16_t; 4], b: [int8x16_t; 4]) -> [int8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I8x16Backend>::simd_lt(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_le(self, a: [int8x16_t; 4], b: [int8x16_t; 4]) -> [int8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I8x16Backend>::simd_le(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_gt(self, a: [int8x16_t; 4], b: [int8x16_t; 4]) -> [int8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I8x16Backend>::simd_gt(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ge(self, a: [int8x16_t; 4], b: [int8x16_t; 4]) -> [int8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I8x16Backend>::simd_ge(self, a[i], b[i]))
    }

    #[inline(always)]
    fn blend(
        self,
        mask: [int8x16_t; 4],
        if_true: [int8x16_t; 4],
        if_false: [int8x16_t; 4],
    ) -> [int8x16_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as I8x16Backend>::blend(self, mask[i], if_true[i], if_false[i])
        })
    }

    #[inline(always)]
    fn not(self, a: [int8x16_t; 4]) -> [int8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I8x16Backend>::not(self, a[i]))
    }

    #[inline(always)]
    fn bitand(self, a: [int8x16_t; 4], b: [int8x16_t; 4]) -> [int8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I8x16Backend>::bitand(self, a[i], b[i]))
    }

    #[inline(always)]
    fn bitor(self, a: [int8x16_t; 4], b: [int8x16_t; 4]) -> [int8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I8x16Backend>::bitor(self, a[i], b[i]))
    }

    #[inline(always)]
    fn bitxor(self, a: [int8x16_t; 4], b: [int8x16_t; 4]) -> [int8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I8x16Backend>::bitxor(self, a[i], b[i]))
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "aarch64")]
impl U8x64Backend for archmage::NeonToken {
    type Repr = [uint8x16_t; 4];

    #[inline(always)]
    fn splat(self, v: u8) -> [uint8x16_t; 4] {
        let q = <archmage::NeonToken as U8x16Backend>::splat(self, v);
        [q, q, q, q]
    }

    #[inline(always)]
    fn zero(self) -> [uint8x16_t; 4] {
        let q = <archmage::NeonToken as U8x16Backend>::zero(self);
        [q, q, q, q]
    }

    #[inline(always)]
    fn load(self, data: &[u8; 64]) -> [uint8x16_t; 4] {
        [
            <archmage::NeonToken as U8x16Backend>::load(self, data[0..16].try_into().unwrap()),
            <archmage::NeonToken as U8x16Backend>::load(self, data[16..32].try_into().unwrap()),
            <archmage::NeonToken as U8x16Backend>::load(self, data[32..48].try_into().unwrap()),
            <archmage::NeonToken as U8x16Backend>::load(self, data[48..64].try_into().unwrap()),
        ]
    }

    #[inline(always)]
    fn from_array(self, arr: [u8; 64]) -> [uint8x16_t; 4] {
        let mut q0 = [0; 16];
        let mut q1 = [0; 16];
        let mut q2 = [0; 16];
        let mut q3 = [0; 16];
        q0.copy_from_slice(&arr[0..16]);
        q1.copy_from_slice(&arr[16..32]);
        q2.copy_from_slice(&arr[32..48]);
        q3.copy_from_slice(&arr[48..64]);
        [
            <archmage::NeonToken as U8x16Backend>::from_array(self, q0),
            <archmage::NeonToken as U8x16Backend>::from_array(self, q1),
            <archmage::NeonToken as U8x16Backend>::from_array(self, q2),
            <archmage::NeonToken as U8x16Backend>::from_array(self, q3),
        ]
    }

    #[inline(always)]
    fn store(self, repr: [uint8x16_t; 4], out: &mut [u8; 64]) {
        let (o01, o23) = out.split_at_mut(32);
        let (o0, o1) = o01.split_at_mut(16);
        let (o2, o3) = o23.split_at_mut(16);
        <archmage::NeonToken as U8x16Backend>::store(self, repr[0], o0.try_into().unwrap());
        <archmage::NeonToken as U8x16Backend>::store(self, repr[1], o1.try_into().unwrap());
        <archmage::NeonToken as U8x16Backend>::store(self, repr[2], o2.try_into().unwrap());
        <archmage::NeonToken as U8x16Backend>::store(self, repr[3], o3.try_into().unwrap());
    }

    #[inline(always)]
    fn to_array(self, repr: [uint8x16_t; 4]) -> [u8; 64] {
        let a0 = <archmage::NeonToken as U8x16Backend>::to_array(self, repr[0]);
        let a1 = <archmage::NeonToken as U8x16Backend>::to_array(self, repr[1]);
        let a2 = <archmage::NeonToken as U8x16Backend>::to_array(self, repr[2]);
        let a3 = <archmage::NeonToken as U8x16Backend>::to_array(self, repr[3]);
        let mut out = [0; 64];
        out[0..16].copy_from_slice(&a0);
        out[16..32].copy_from_slice(&a1);
        out[32..48].copy_from_slice(&a2);
        out[48..64].copy_from_slice(&a3);
        out
    }

    #[inline(always)]
    fn add(self, a: [uint8x16_t; 4], b: [uint8x16_t; 4]) -> [uint8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U8x16Backend>::add(self, a[i], b[i]))
    }

    #[inline(always)]
    fn sub(self, a: [uint8x16_t; 4], b: [uint8x16_t; 4]) -> [uint8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U8x16Backend>::sub(self, a[i], b[i]))
    }

    #[inline(always)]
    fn neg(self, a: [uint8x16_t; 4]) -> [uint8x16_t; 4] {
        let z = <archmage::NeonToken as U8x16Backend>::zero(self);
        core::array::from_fn(|i| <archmage::NeonToken as U8x16Backend>::sub(self, z, a[i]))
    }

    #[inline(always)]
    fn min(self, a: [uint8x16_t; 4], b: [uint8x16_t; 4]) -> [uint8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U8x16Backend>::min(self, a[i], b[i]))
    }

    #[inline(always)]
    fn max(self, a: [uint8x16_t; 4], b: [uint8x16_t; 4]) -> [uint8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U8x16Backend>::max(self, a[i], b[i]))
    }

    #[inline(always)]
    fn reduce_add(self, a: [uint8x16_t; 4]) -> u8 {
        <archmage::NeonToken as U8x16Backend>::reduce_add(self, a[0])
            .wrapping_add(<archmage::NeonToken as U8x16Backend>::reduce_add(
                self, a[1],
            ))
            .wrapping_add(<archmage::NeonToken as U8x16Backend>::reduce_add(
                self, a[2],
            ))
            .wrapping_add(<archmage::NeonToken as U8x16Backend>::reduce_add(
                self, a[3],
            ))
    }

    #[inline(always)]
    fn shl_const<const N: i32>(self, a: [uint8x16_t; 4]) -> [uint8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U8x16Backend>::shl_const::<N>(self, a[i]))
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(self, a: [uint8x16_t; 4]) -> [uint8x16_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as U8x16Backend>::shr_logical_const::<N>(self, a[i])
        })
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: [uint8x16_t; 4]) -> [uint8x16_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as U8x16Backend>::shr_logical_const::<N>(self, a[i])
        })
    }

    #[inline(always)]
    fn shl_uniform(self, a: [uint8x16_t; 4], count: u32) -> [uint8x16_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as U8x16Backend>::shl_uniform(self, a[i], count)
        })
    }

    #[inline(always)]
    fn shr_logical_uniform(self, a: [uint8x16_t; 4], count: u32) -> [uint8x16_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as U8x16Backend>::shr_logical_uniform(self, a[i], count)
        })
    }

    #[inline(always)]
    fn shr_arithmetic_uniform(self, a: [uint8x16_t; 4], count: u32) -> [uint8x16_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as U8x16Backend>::shr_logical_uniform(self, a[i], count)
        })
    }

    #[inline(always)]
    fn saturating_add(self, a: [uint8x16_t; 4], b: [uint8x16_t; 4]) -> [uint8x16_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as U8x16Backend>::saturating_add(self, a[i], b[i])
        })
    }

    #[inline(always)]
    fn saturating_sub(self, a: [uint8x16_t; 4], b: [uint8x16_t; 4]) -> [uint8x16_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as U8x16Backend>::saturating_sub(self, a[i], b[i])
        })
    }

    // Reduce the quarters with vector AND/OR first, then ONE lane
    // reduction (fearless_simd's polyfill lesson; branchless).
    #[inline(always)]
    fn all_true(self, a: [uint8x16_t; 4]) -> bool {
        let c01 = <archmage::NeonToken as U8x16Backend>::bitand(self, a[0], a[1]);
        let c23 = <archmage::NeonToken as U8x16Backend>::bitand(self, a[2], a[3]);
        let c = <archmage::NeonToken as U8x16Backend>::bitand(self, c01, c23);
        <archmage::NeonToken as U8x16Backend>::all_true(self, c)
    }

    #[inline(always)]
    fn any_true(self, a: [uint8x16_t; 4]) -> bool {
        let c01 = <archmage::NeonToken as U8x16Backend>::bitor(self, a[0], a[1]);
        let c23 = <archmage::NeonToken as U8x16Backend>::bitor(self, a[2], a[3]);
        let c = <archmage::NeonToken as U8x16Backend>::bitor(self, c01, c23);
        <archmage::NeonToken as U8x16Backend>::any_true(self, c)
    }

    #[inline(always)]
    fn bitmask(self, a: [uint8x16_t; 4]) -> u64 {
        let q0 = <archmage::NeonToken as U8x16Backend>::bitmask(self, a[0]) as u64;
        let q1 = <archmage::NeonToken as U8x16Backend>::bitmask(self, a[1]) as u64;
        let q2 = <archmage::NeonToken as U8x16Backend>::bitmask(self, a[2]) as u64;
        let q3 = <archmage::NeonToken as U8x16Backend>::bitmask(self, a[3]) as u64;
        q0 | (q1 << 16) | (q2 << 32) | (q3 << 48)
    }

    #[inline(always)]
    fn simd_eq(self, a: [uint8x16_t; 4], b: [uint8x16_t; 4]) -> [uint8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U8x16Backend>::simd_eq(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ne(self, a: [uint8x16_t; 4], b: [uint8x16_t; 4]) -> [uint8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U8x16Backend>::simd_ne(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_lt(self, a: [uint8x16_t; 4], b: [uint8x16_t; 4]) -> [uint8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U8x16Backend>::simd_lt(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_le(self, a: [uint8x16_t; 4], b: [uint8x16_t; 4]) -> [uint8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U8x16Backend>::simd_le(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_gt(self, a: [uint8x16_t; 4], b: [uint8x16_t; 4]) -> [uint8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U8x16Backend>::simd_gt(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ge(self, a: [uint8x16_t; 4], b: [uint8x16_t; 4]) -> [uint8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U8x16Backend>::simd_ge(self, a[i], b[i]))
    }

    #[inline(always)]
    fn blend(
        self,
        mask: [uint8x16_t; 4],
        if_true: [uint8x16_t; 4],
        if_false: [uint8x16_t; 4],
    ) -> [uint8x16_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as U8x16Backend>::blend(self, mask[i], if_true[i], if_false[i])
        })
    }

    #[inline(always)]
    fn not(self, a: [uint8x16_t; 4]) -> [uint8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U8x16Backend>::not(self, a[i]))
    }

    #[inline(always)]
    fn bitand(self, a: [uint8x16_t; 4], b: [uint8x16_t; 4]) -> [uint8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U8x16Backend>::bitand(self, a[i], b[i]))
    }

    #[inline(always)]
    fn bitor(self, a: [uint8x16_t; 4], b: [uint8x16_t; 4]) -> [uint8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U8x16Backend>::bitor(self, a[i], b[i]))
    }

    #[inline(always)]
    fn bitxor(self, a: [uint8x16_t; 4], b: [uint8x16_t; 4]) -> [uint8x16_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U8x16Backend>::bitxor(self, a[i], b[i]))
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "aarch64")]
impl I16x32Backend for archmage::NeonToken {
    type Repr = [int16x8_t; 4];

    #[inline(always)]
    fn splat(self, v: i16) -> [int16x8_t; 4] {
        let q = <archmage::NeonToken as I16x8Backend>::splat(self, v);
        [q, q, q, q]
    }

    #[inline(always)]
    fn zero(self) -> [int16x8_t; 4] {
        let q = <archmage::NeonToken as I16x8Backend>::zero(self);
        [q, q, q, q]
    }

    #[inline(always)]
    fn load(self, data: &[i16; 32]) -> [int16x8_t; 4] {
        [
            <archmage::NeonToken as I16x8Backend>::load(self, data[0..8].try_into().unwrap()),
            <archmage::NeonToken as I16x8Backend>::load(self, data[8..16].try_into().unwrap()),
            <archmage::NeonToken as I16x8Backend>::load(self, data[16..24].try_into().unwrap()),
            <archmage::NeonToken as I16x8Backend>::load(self, data[24..32].try_into().unwrap()),
        ]
    }

    #[inline(always)]
    fn from_array(self, arr: [i16; 32]) -> [int16x8_t; 4] {
        let mut q0 = [0; 8];
        let mut q1 = [0; 8];
        let mut q2 = [0; 8];
        let mut q3 = [0; 8];
        q0.copy_from_slice(&arr[0..8]);
        q1.copy_from_slice(&arr[8..16]);
        q2.copy_from_slice(&arr[16..24]);
        q3.copy_from_slice(&arr[24..32]);
        [
            <archmage::NeonToken as I16x8Backend>::from_array(self, q0),
            <archmage::NeonToken as I16x8Backend>::from_array(self, q1),
            <archmage::NeonToken as I16x8Backend>::from_array(self, q2),
            <archmage::NeonToken as I16x8Backend>::from_array(self, q3),
        ]
    }

    #[inline(always)]
    fn store(self, repr: [int16x8_t; 4], out: &mut [i16; 32]) {
        let (o01, o23) = out.split_at_mut(16);
        let (o0, o1) = o01.split_at_mut(8);
        let (o2, o3) = o23.split_at_mut(8);
        <archmage::NeonToken as I16x8Backend>::store(self, repr[0], o0.try_into().unwrap());
        <archmage::NeonToken as I16x8Backend>::store(self, repr[1], o1.try_into().unwrap());
        <archmage::NeonToken as I16x8Backend>::store(self, repr[2], o2.try_into().unwrap());
        <archmage::NeonToken as I16x8Backend>::store(self, repr[3], o3.try_into().unwrap());
    }

    #[inline(always)]
    fn to_array(self, repr: [int16x8_t; 4]) -> [i16; 32] {
        let a0 = <archmage::NeonToken as I16x8Backend>::to_array(self, repr[0]);
        let a1 = <archmage::NeonToken as I16x8Backend>::to_array(self, repr[1]);
        let a2 = <archmage::NeonToken as I16x8Backend>::to_array(self, repr[2]);
        let a3 = <archmage::NeonToken as I16x8Backend>::to_array(self, repr[3]);
        let mut out = [0; 32];
        out[0..8].copy_from_slice(&a0);
        out[8..16].copy_from_slice(&a1);
        out[16..24].copy_from_slice(&a2);
        out[24..32].copy_from_slice(&a3);
        out
    }

    #[inline(always)]
    fn add(self, a: [int16x8_t; 4], b: [int16x8_t; 4]) -> [int16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I16x8Backend>::add(self, a[i], b[i]))
    }

    #[inline(always)]
    fn sub(self, a: [int16x8_t; 4], b: [int16x8_t; 4]) -> [int16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I16x8Backend>::sub(self, a[i], b[i]))
    }

    #[inline(always)]
    fn mul(self, a: [int16x8_t; 4], b: [int16x8_t; 4]) -> [int16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I16x8Backend>::mul(self, a[i], b[i]))
    }

    #[inline(always)]
    fn neg(self, a: [int16x8_t; 4]) -> [int16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I16x8Backend>::neg(self, a[i]))
    }

    #[inline(always)]
    fn min(self, a: [int16x8_t; 4], b: [int16x8_t; 4]) -> [int16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I16x8Backend>::min(self, a[i], b[i]))
    }

    #[inline(always)]
    fn max(self, a: [int16x8_t; 4], b: [int16x8_t; 4]) -> [int16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I16x8Backend>::max(self, a[i], b[i]))
    }

    #[inline(always)]
    fn abs(self, a: [int16x8_t; 4]) -> [int16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I16x8Backend>::abs(self, a[i]))
    }

    #[inline(always)]
    fn reduce_add(self, a: [int16x8_t; 4]) -> i16 {
        <archmage::NeonToken as I16x8Backend>::reduce_add(self, a[0])
            .wrapping_add(<archmage::NeonToken as I16x8Backend>::reduce_add(
                self, a[1],
            ))
            .wrapping_add(<archmage::NeonToken as I16x8Backend>::reduce_add(
                self, a[2],
            ))
            .wrapping_add(<archmage::NeonToken as I16x8Backend>::reduce_add(
                self, a[3],
            ))
    }

    #[inline(always)]
    fn shl_const<const N: i32>(self, a: [int16x8_t; 4]) -> [int16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I16x8Backend>::shl_const::<N>(self, a[i]))
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(self, a: [int16x8_t; 4]) -> [int16x8_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as I16x8Backend>::shr_arithmetic_const::<N>(self, a[i])
        })
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: [int16x8_t; 4]) -> [int16x8_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as I16x8Backend>::shr_logical_const::<N>(self, a[i])
        })
    }

    #[inline(always)]
    fn shl_uniform(self, a: [int16x8_t; 4], count: u32) -> [int16x8_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as I16x8Backend>::shl_uniform(self, a[i], count)
        })
    }

    #[inline(always)]
    fn shr_logical_uniform(self, a: [int16x8_t; 4], count: u32) -> [int16x8_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as I16x8Backend>::shr_logical_uniform(self, a[i], count)
        })
    }

    #[inline(always)]
    fn shr_arithmetic_uniform(self, a: [int16x8_t; 4], count: u32) -> [int16x8_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as I16x8Backend>::shr_arithmetic_uniform(self, a[i], count)
        })
    }

    #[inline(always)]
    fn saturating_add(self, a: [int16x8_t; 4], b: [int16x8_t; 4]) -> [int16x8_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as I16x8Backend>::saturating_add(self, a[i], b[i])
        })
    }

    #[inline(always)]
    fn saturating_sub(self, a: [int16x8_t; 4], b: [int16x8_t; 4]) -> [int16x8_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as I16x8Backend>::saturating_sub(self, a[i], b[i])
        })
    }

    // Reduce the quarters with vector AND/OR first, then ONE lane
    // reduction (fearless_simd's polyfill lesson; branchless).
    #[inline(always)]
    fn all_true(self, a: [int16x8_t; 4]) -> bool {
        let c01 = <archmage::NeonToken as I16x8Backend>::bitand(self, a[0], a[1]);
        let c23 = <archmage::NeonToken as I16x8Backend>::bitand(self, a[2], a[3]);
        let c = <archmage::NeonToken as I16x8Backend>::bitand(self, c01, c23);
        <archmage::NeonToken as I16x8Backend>::all_true(self, c)
    }

    #[inline(always)]
    fn any_true(self, a: [int16x8_t; 4]) -> bool {
        let c01 = <archmage::NeonToken as I16x8Backend>::bitor(self, a[0], a[1]);
        let c23 = <archmage::NeonToken as I16x8Backend>::bitor(self, a[2], a[3]);
        let c = <archmage::NeonToken as I16x8Backend>::bitor(self, c01, c23);
        <archmage::NeonToken as I16x8Backend>::any_true(self, c)
    }

    #[inline(always)]
    fn bitmask(self, a: [int16x8_t; 4]) -> u64 {
        let q0 = <archmage::NeonToken as I16x8Backend>::bitmask(self, a[0]) as u64;
        let q1 = <archmage::NeonToken as I16x8Backend>::bitmask(self, a[1]) as u64;
        let q2 = <archmage::NeonToken as I16x8Backend>::bitmask(self, a[2]) as u64;
        let q3 = <archmage::NeonToken as I16x8Backend>::bitmask(self, a[3]) as u64;
        q0 | (q1 << 8) | (q2 << 16) | (q3 << 24)
    }

    #[inline(always)]
    fn simd_eq(self, a: [int16x8_t; 4], b: [int16x8_t; 4]) -> [int16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I16x8Backend>::simd_eq(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ne(self, a: [int16x8_t; 4], b: [int16x8_t; 4]) -> [int16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I16x8Backend>::simd_ne(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_lt(self, a: [int16x8_t; 4], b: [int16x8_t; 4]) -> [int16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I16x8Backend>::simd_lt(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_le(self, a: [int16x8_t; 4], b: [int16x8_t; 4]) -> [int16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I16x8Backend>::simd_le(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_gt(self, a: [int16x8_t; 4], b: [int16x8_t; 4]) -> [int16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I16x8Backend>::simd_gt(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ge(self, a: [int16x8_t; 4], b: [int16x8_t; 4]) -> [int16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I16x8Backend>::simd_ge(self, a[i], b[i]))
    }

    #[inline(always)]
    fn blend(
        self,
        mask: [int16x8_t; 4],
        if_true: [int16x8_t; 4],
        if_false: [int16x8_t; 4],
    ) -> [int16x8_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as I16x8Backend>::blend(self, mask[i], if_true[i], if_false[i])
        })
    }

    #[inline(always)]
    fn not(self, a: [int16x8_t; 4]) -> [int16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I16x8Backend>::not(self, a[i]))
    }

    #[inline(always)]
    fn bitand(self, a: [int16x8_t; 4], b: [int16x8_t; 4]) -> [int16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I16x8Backend>::bitand(self, a[i], b[i]))
    }

    #[inline(always)]
    fn bitor(self, a: [int16x8_t; 4], b: [int16x8_t; 4]) -> [int16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I16x8Backend>::bitor(self, a[i], b[i]))
    }

    #[inline(always)]
    fn bitxor(self, a: [int16x8_t; 4], b: [int16x8_t; 4]) -> [int16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I16x8Backend>::bitxor(self, a[i], b[i]))
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "aarch64")]
impl U16x32Backend for archmage::NeonToken {
    type Repr = [uint16x8_t; 4];

    #[inline(always)]
    fn splat(self, v: u16) -> [uint16x8_t; 4] {
        let q = <archmage::NeonToken as U16x8Backend>::splat(self, v);
        [q, q, q, q]
    }

    #[inline(always)]
    fn zero(self) -> [uint16x8_t; 4] {
        let q = <archmage::NeonToken as U16x8Backend>::zero(self);
        [q, q, q, q]
    }

    #[inline(always)]
    fn load(self, data: &[u16; 32]) -> [uint16x8_t; 4] {
        [
            <archmage::NeonToken as U16x8Backend>::load(self, data[0..8].try_into().unwrap()),
            <archmage::NeonToken as U16x8Backend>::load(self, data[8..16].try_into().unwrap()),
            <archmage::NeonToken as U16x8Backend>::load(self, data[16..24].try_into().unwrap()),
            <archmage::NeonToken as U16x8Backend>::load(self, data[24..32].try_into().unwrap()),
        ]
    }

    #[inline(always)]
    fn from_array(self, arr: [u16; 32]) -> [uint16x8_t; 4] {
        let mut q0 = [0; 8];
        let mut q1 = [0; 8];
        let mut q2 = [0; 8];
        let mut q3 = [0; 8];
        q0.copy_from_slice(&arr[0..8]);
        q1.copy_from_slice(&arr[8..16]);
        q2.copy_from_slice(&arr[16..24]);
        q3.copy_from_slice(&arr[24..32]);
        [
            <archmage::NeonToken as U16x8Backend>::from_array(self, q0),
            <archmage::NeonToken as U16x8Backend>::from_array(self, q1),
            <archmage::NeonToken as U16x8Backend>::from_array(self, q2),
            <archmage::NeonToken as U16x8Backend>::from_array(self, q3),
        ]
    }

    #[inline(always)]
    fn store(self, repr: [uint16x8_t; 4], out: &mut [u16; 32]) {
        let (o01, o23) = out.split_at_mut(16);
        let (o0, o1) = o01.split_at_mut(8);
        let (o2, o3) = o23.split_at_mut(8);
        <archmage::NeonToken as U16x8Backend>::store(self, repr[0], o0.try_into().unwrap());
        <archmage::NeonToken as U16x8Backend>::store(self, repr[1], o1.try_into().unwrap());
        <archmage::NeonToken as U16x8Backend>::store(self, repr[2], o2.try_into().unwrap());
        <archmage::NeonToken as U16x8Backend>::store(self, repr[3], o3.try_into().unwrap());
    }

    #[inline(always)]
    fn to_array(self, repr: [uint16x8_t; 4]) -> [u16; 32] {
        let a0 = <archmage::NeonToken as U16x8Backend>::to_array(self, repr[0]);
        let a1 = <archmage::NeonToken as U16x8Backend>::to_array(self, repr[1]);
        let a2 = <archmage::NeonToken as U16x8Backend>::to_array(self, repr[2]);
        let a3 = <archmage::NeonToken as U16x8Backend>::to_array(self, repr[3]);
        let mut out = [0; 32];
        out[0..8].copy_from_slice(&a0);
        out[8..16].copy_from_slice(&a1);
        out[16..24].copy_from_slice(&a2);
        out[24..32].copy_from_slice(&a3);
        out
    }

    #[inline(always)]
    fn add(self, a: [uint16x8_t; 4], b: [uint16x8_t; 4]) -> [uint16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U16x8Backend>::add(self, a[i], b[i]))
    }

    #[inline(always)]
    fn sub(self, a: [uint16x8_t; 4], b: [uint16x8_t; 4]) -> [uint16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U16x8Backend>::sub(self, a[i], b[i]))
    }

    #[inline(always)]
    fn mul(self, a: [uint16x8_t; 4], b: [uint16x8_t; 4]) -> [uint16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U16x8Backend>::mul(self, a[i], b[i]))
    }

    #[inline(always)]
    fn neg(self, a: [uint16x8_t; 4]) -> [uint16x8_t; 4] {
        let z = <archmage::NeonToken as U16x8Backend>::zero(self);
        core::array::from_fn(|i| <archmage::NeonToken as U16x8Backend>::sub(self, z, a[i]))
    }

    #[inline(always)]
    fn min(self, a: [uint16x8_t; 4], b: [uint16x8_t; 4]) -> [uint16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U16x8Backend>::min(self, a[i], b[i]))
    }

    #[inline(always)]
    fn max(self, a: [uint16x8_t; 4], b: [uint16x8_t; 4]) -> [uint16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U16x8Backend>::max(self, a[i], b[i]))
    }

    #[inline(always)]
    fn reduce_add(self, a: [uint16x8_t; 4]) -> u16 {
        <archmage::NeonToken as U16x8Backend>::reduce_add(self, a[0])
            .wrapping_add(<archmage::NeonToken as U16x8Backend>::reduce_add(
                self, a[1],
            ))
            .wrapping_add(<archmage::NeonToken as U16x8Backend>::reduce_add(
                self, a[2],
            ))
            .wrapping_add(<archmage::NeonToken as U16x8Backend>::reduce_add(
                self, a[3],
            ))
    }

    #[inline(always)]
    fn shl_const<const N: i32>(self, a: [uint16x8_t; 4]) -> [uint16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U16x8Backend>::shl_const::<N>(self, a[i]))
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(self, a: [uint16x8_t; 4]) -> [uint16x8_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as U16x8Backend>::shr_logical_const::<N>(self, a[i])
        })
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: [uint16x8_t; 4]) -> [uint16x8_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as U16x8Backend>::shr_logical_const::<N>(self, a[i])
        })
    }

    #[inline(always)]
    fn shl_uniform(self, a: [uint16x8_t; 4], count: u32) -> [uint16x8_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as U16x8Backend>::shl_uniform(self, a[i], count)
        })
    }

    #[inline(always)]
    fn shr_logical_uniform(self, a: [uint16x8_t; 4], count: u32) -> [uint16x8_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as U16x8Backend>::shr_logical_uniform(self, a[i], count)
        })
    }

    #[inline(always)]
    fn shr_arithmetic_uniform(self, a: [uint16x8_t; 4], count: u32) -> [uint16x8_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as U16x8Backend>::shr_logical_uniform(self, a[i], count)
        })
    }

    #[inline(always)]
    fn saturating_add(self, a: [uint16x8_t; 4], b: [uint16x8_t; 4]) -> [uint16x8_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as U16x8Backend>::saturating_add(self, a[i], b[i])
        })
    }

    #[inline(always)]
    fn saturating_sub(self, a: [uint16x8_t; 4], b: [uint16x8_t; 4]) -> [uint16x8_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as U16x8Backend>::saturating_sub(self, a[i], b[i])
        })
    }

    // Reduce the quarters with vector AND/OR first, then ONE lane
    // reduction (fearless_simd's polyfill lesson; branchless).
    #[inline(always)]
    fn all_true(self, a: [uint16x8_t; 4]) -> bool {
        let c01 = <archmage::NeonToken as U16x8Backend>::bitand(self, a[0], a[1]);
        let c23 = <archmage::NeonToken as U16x8Backend>::bitand(self, a[2], a[3]);
        let c = <archmage::NeonToken as U16x8Backend>::bitand(self, c01, c23);
        <archmage::NeonToken as U16x8Backend>::all_true(self, c)
    }

    #[inline(always)]
    fn any_true(self, a: [uint16x8_t; 4]) -> bool {
        let c01 = <archmage::NeonToken as U16x8Backend>::bitor(self, a[0], a[1]);
        let c23 = <archmage::NeonToken as U16x8Backend>::bitor(self, a[2], a[3]);
        let c = <archmage::NeonToken as U16x8Backend>::bitor(self, c01, c23);
        <archmage::NeonToken as U16x8Backend>::any_true(self, c)
    }

    #[inline(always)]
    fn bitmask(self, a: [uint16x8_t; 4]) -> u64 {
        let q0 = <archmage::NeonToken as U16x8Backend>::bitmask(self, a[0]) as u64;
        let q1 = <archmage::NeonToken as U16x8Backend>::bitmask(self, a[1]) as u64;
        let q2 = <archmage::NeonToken as U16x8Backend>::bitmask(self, a[2]) as u64;
        let q3 = <archmage::NeonToken as U16x8Backend>::bitmask(self, a[3]) as u64;
        q0 | (q1 << 8) | (q2 << 16) | (q3 << 24)
    }

    #[inline(always)]
    fn simd_eq(self, a: [uint16x8_t; 4], b: [uint16x8_t; 4]) -> [uint16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U16x8Backend>::simd_eq(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ne(self, a: [uint16x8_t; 4], b: [uint16x8_t; 4]) -> [uint16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U16x8Backend>::simd_ne(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_lt(self, a: [uint16x8_t; 4], b: [uint16x8_t; 4]) -> [uint16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U16x8Backend>::simd_lt(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_le(self, a: [uint16x8_t; 4], b: [uint16x8_t; 4]) -> [uint16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U16x8Backend>::simd_le(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_gt(self, a: [uint16x8_t; 4], b: [uint16x8_t; 4]) -> [uint16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U16x8Backend>::simd_gt(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ge(self, a: [uint16x8_t; 4], b: [uint16x8_t; 4]) -> [uint16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U16x8Backend>::simd_ge(self, a[i], b[i]))
    }

    #[inline(always)]
    fn blend(
        self,
        mask: [uint16x8_t; 4],
        if_true: [uint16x8_t; 4],
        if_false: [uint16x8_t; 4],
    ) -> [uint16x8_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as U16x8Backend>::blend(self, mask[i], if_true[i], if_false[i])
        })
    }

    #[inline(always)]
    fn not(self, a: [uint16x8_t; 4]) -> [uint16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U16x8Backend>::not(self, a[i]))
    }

    #[inline(always)]
    fn bitand(self, a: [uint16x8_t; 4], b: [uint16x8_t; 4]) -> [uint16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U16x8Backend>::bitand(self, a[i], b[i]))
    }

    #[inline(always)]
    fn bitor(self, a: [uint16x8_t; 4], b: [uint16x8_t; 4]) -> [uint16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U16x8Backend>::bitor(self, a[i], b[i]))
    }

    #[inline(always)]
    fn bitxor(self, a: [uint16x8_t; 4], b: [uint16x8_t; 4]) -> [uint16x8_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U16x8Backend>::bitxor(self, a[i], b[i]))
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "aarch64")]
impl I32x16Backend for archmage::NeonToken {
    type Repr = [int32x4_t; 4];

    #[inline(always)]
    fn splat(self, v: i32) -> [int32x4_t; 4] {
        let q = <archmage::NeonToken as I32x4Backend>::splat(self, v);
        [q, q, q, q]
    }

    #[inline(always)]
    fn zero(self) -> [int32x4_t; 4] {
        let q = <archmage::NeonToken as I32x4Backend>::zero(self);
        [q, q, q, q]
    }

    #[inline(always)]
    fn load(self, data: &[i32; 16]) -> [int32x4_t; 4] {
        [
            <archmage::NeonToken as I32x4Backend>::load(self, data[0..4].try_into().unwrap()),
            <archmage::NeonToken as I32x4Backend>::load(self, data[4..8].try_into().unwrap()),
            <archmage::NeonToken as I32x4Backend>::load(self, data[8..12].try_into().unwrap()),
            <archmage::NeonToken as I32x4Backend>::load(self, data[12..16].try_into().unwrap()),
        ]
    }

    #[inline(always)]
    fn from_array(self, arr: [i32; 16]) -> [int32x4_t; 4] {
        let mut q0 = [0; 4];
        let mut q1 = [0; 4];
        let mut q2 = [0; 4];
        let mut q3 = [0; 4];
        q0.copy_from_slice(&arr[0..4]);
        q1.copy_from_slice(&arr[4..8]);
        q2.copy_from_slice(&arr[8..12]);
        q3.copy_from_slice(&arr[12..16]);
        [
            <archmage::NeonToken as I32x4Backend>::from_array(self, q0),
            <archmage::NeonToken as I32x4Backend>::from_array(self, q1),
            <archmage::NeonToken as I32x4Backend>::from_array(self, q2),
            <archmage::NeonToken as I32x4Backend>::from_array(self, q3),
        ]
    }

    #[inline(always)]
    fn store(self, repr: [int32x4_t; 4], out: &mut [i32; 16]) {
        let (o01, o23) = out.split_at_mut(8);
        let (o0, o1) = o01.split_at_mut(4);
        let (o2, o3) = o23.split_at_mut(4);
        <archmage::NeonToken as I32x4Backend>::store(self, repr[0], o0.try_into().unwrap());
        <archmage::NeonToken as I32x4Backend>::store(self, repr[1], o1.try_into().unwrap());
        <archmage::NeonToken as I32x4Backend>::store(self, repr[2], o2.try_into().unwrap());
        <archmage::NeonToken as I32x4Backend>::store(self, repr[3], o3.try_into().unwrap());
    }

    #[inline(always)]
    fn to_array(self, repr: [int32x4_t; 4]) -> [i32; 16] {
        let a0 = <archmage::NeonToken as I32x4Backend>::to_array(self, repr[0]);
        let a1 = <archmage::NeonToken as I32x4Backend>::to_array(self, repr[1]);
        let a2 = <archmage::NeonToken as I32x4Backend>::to_array(self, repr[2]);
        let a3 = <archmage::NeonToken as I32x4Backend>::to_array(self, repr[3]);
        let mut out = [0; 16];
        out[0..4].copy_from_slice(&a0);
        out[4..8].copy_from_slice(&a1);
        out[8..12].copy_from_slice(&a2);
        out[12..16].copy_from_slice(&a3);
        out
    }

    #[inline(always)]
    fn add(self, a: [int32x4_t; 4], b: [int32x4_t; 4]) -> [int32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I32x4Backend>::add(self, a[i], b[i]))
    }

    #[inline(always)]
    fn sub(self, a: [int32x4_t; 4], b: [int32x4_t; 4]) -> [int32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I32x4Backend>::sub(self, a[i], b[i]))
    }

    #[inline(always)]
    fn mul(self, a: [int32x4_t; 4], b: [int32x4_t; 4]) -> [int32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I32x4Backend>::mul(self, a[i], b[i]))
    }

    #[inline(always)]
    fn neg(self, a: [int32x4_t; 4]) -> [int32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I32x4Backend>::neg(self, a[i]))
    }

    #[inline(always)]
    fn min(self, a: [int32x4_t; 4], b: [int32x4_t; 4]) -> [int32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I32x4Backend>::min(self, a[i], b[i]))
    }

    #[inline(always)]
    fn max(self, a: [int32x4_t; 4], b: [int32x4_t; 4]) -> [int32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I32x4Backend>::max(self, a[i], b[i]))
    }

    #[inline(always)]
    fn abs(self, a: [int32x4_t; 4]) -> [int32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I32x4Backend>::abs(self, a[i]))
    }

    #[inline(always)]
    fn reduce_add(self, a: [int32x4_t; 4]) -> i32 {
        <archmage::NeonToken as I32x4Backend>::reduce_add(self, a[0])
            .wrapping_add(<archmage::NeonToken as I32x4Backend>::reduce_add(
                self, a[1],
            ))
            .wrapping_add(<archmage::NeonToken as I32x4Backend>::reduce_add(
                self, a[2],
            ))
            .wrapping_add(<archmage::NeonToken as I32x4Backend>::reduce_add(
                self, a[3],
            ))
    }

    #[inline(always)]
    fn shl_const<const N: i32>(self, a: [int32x4_t; 4]) -> [int32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I32x4Backend>::shl_const::<N>(self, a[i]))
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(self, a: [int32x4_t; 4]) -> [int32x4_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as I32x4Backend>::shr_arithmetic_const::<N>(self, a[i])
        })
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: [int32x4_t; 4]) -> [int32x4_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as I32x4Backend>::shr_logical_const::<N>(self, a[i])
        })
    }

    #[inline(always)]
    fn shl_uniform(self, a: [int32x4_t; 4], count: u32) -> [int32x4_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as I32x4Backend>::shl_uniform(self, a[i], count)
        })
    }

    #[inline(always)]
    fn shr_logical_uniform(self, a: [int32x4_t; 4], count: u32) -> [int32x4_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as I32x4Backend>::shr_logical_uniform(self, a[i], count)
        })
    }

    #[inline(always)]
    fn shr_arithmetic_uniform(self, a: [int32x4_t; 4], count: u32) -> [int32x4_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as I32x4Backend>::shr_arithmetic_uniform(self, a[i], count)
        })
    }

    // Reduce the quarters with vector AND/OR first, then ONE lane
    // reduction (fearless_simd's polyfill lesson; branchless).
    #[inline(always)]
    fn all_true(self, a: [int32x4_t; 4]) -> bool {
        let c01 = <archmage::NeonToken as I32x4Backend>::bitand(self, a[0], a[1]);
        let c23 = <archmage::NeonToken as I32x4Backend>::bitand(self, a[2], a[3]);
        let c = <archmage::NeonToken as I32x4Backend>::bitand(self, c01, c23);
        <archmage::NeonToken as I32x4Backend>::all_true(self, c)
    }

    #[inline(always)]
    fn any_true(self, a: [int32x4_t; 4]) -> bool {
        let c01 = <archmage::NeonToken as I32x4Backend>::bitor(self, a[0], a[1]);
        let c23 = <archmage::NeonToken as I32x4Backend>::bitor(self, a[2], a[3]);
        let c = <archmage::NeonToken as I32x4Backend>::bitor(self, c01, c23);
        <archmage::NeonToken as I32x4Backend>::any_true(self, c)
    }

    #[inline(always)]
    fn bitmask(self, a: [int32x4_t; 4]) -> u64 {
        let q0 = <archmage::NeonToken as I32x4Backend>::bitmask(self, a[0]) as u64;
        let q1 = <archmage::NeonToken as I32x4Backend>::bitmask(self, a[1]) as u64;
        let q2 = <archmage::NeonToken as I32x4Backend>::bitmask(self, a[2]) as u64;
        let q3 = <archmage::NeonToken as I32x4Backend>::bitmask(self, a[3]) as u64;
        q0 | (q1 << 4) | (q2 << 8) | (q3 << 12)
    }

    #[inline(always)]
    fn simd_eq(self, a: [int32x4_t; 4], b: [int32x4_t; 4]) -> [int32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I32x4Backend>::simd_eq(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ne(self, a: [int32x4_t; 4], b: [int32x4_t; 4]) -> [int32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I32x4Backend>::simd_ne(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_lt(self, a: [int32x4_t; 4], b: [int32x4_t; 4]) -> [int32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I32x4Backend>::simd_lt(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_le(self, a: [int32x4_t; 4], b: [int32x4_t; 4]) -> [int32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I32x4Backend>::simd_le(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_gt(self, a: [int32x4_t; 4], b: [int32x4_t; 4]) -> [int32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I32x4Backend>::simd_gt(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ge(self, a: [int32x4_t; 4], b: [int32x4_t; 4]) -> [int32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I32x4Backend>::simd_ge(self, a[i], b[i]))
    }

    #[inline(always)]
    fn blend(
        self,
        mask: [int32x4_t; 4],
        if_true: [int32x4_t; 4],
        if_false: [int32x4_t; 4],
    ) -> [int32x4_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as I32x4Backend>::blend(self, mask[i], if_true[i], if_false[i])
        })
    }

    #[inline(always)]
    fn not(self, a: [int32x4_t; 4]) -> [int32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I32x4Backend>::not(self, a[i]))
    }

    #[inline(always)]
    fn bitand(self, a: [int32x4_t; 4], b: [int32x4_t; 4]) -> [int32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I32x4Backend>::bitand(self, a[i], b[i]))
    }

    #[inline(always)]
    fn bitor(self, a: [int32x4_t; 4], b: [int32x4_t; 4]) -> [int32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I32x4Backend>::bitor(self, a[i], b[i]))
    }

    #[inline(always)]
    fn bitxor(self, a: [int32x4_t; 4], b: [int32x4_t; 4]) -> [int32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I32x4Backend>::bitxor(self, a[i], b[i]))
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "aarch64")]
impl U32x16Backend for archmage::NeonToken {
    type Repr = [uint32x4_t; 4];

    #[inline(always)]
    fn splat(self, v: u32) -> [uint32x4_t; 4] {
        let q = <archmage::NeonToken as U32x4Backend>::splat(self, v);
        [q, q, q, q]
    }

    #[inline(always)]
    fn zero(self) -> [uint32x4_t; 4] {
        let q = <archmage::NeonToken as U32x4Backend>::zero(self);
        [q, q, q, q]
    }

    #[inline(always)]
    fn load(self, data: &[u32; 16]) -> [uint32x4_t; 4] {
        [
            <archmage::NeonToken as U32x4Backend>::load(self, data[0..4].try_into().unwrap()),
            <archmage::NeonToken as U32x4Backend>::load(self, data[4..8].try_into().unwrap()),
            <archmage::NeonToken as U32x4Backend>::load(self, data[8..12].try_into().unwrap()),
            <archmage::NeonToken as U32x4Backend>::load(self, data[12..16].try_into().unwrap()),
        ]
    }

    #[inline(always)]
    fn from_array(self, arr: [u32; 16]) -> [uint32x4_t; 4] {
        let mut q0 = [0; 4];
        let mut q1 = [0; 4];
        let mut q2 = [0; 4];
        let mut q3 = [0; 4];
        q0.copy_from_slice(&arr[0..4]);
        q1.copy_from_slice(&arr[4..8]);
        q2.copy_from_slice(&arr[8..12]);
        q3.copy_from_slice(&arr[12..16]);
        [
            <archmage::NeonToken as U32x4Backend>::from_array(self, q0),
            <archmage::NeonToken as U32x4Backend>::from_array(self, q1),
            <archmage::NeonToken as U32x4Backend>::from_array(self, q2),
            <archmage::NeonToken as U32x4Backend>::from_array(self, q3),
        ]
    }

    #[inline(always)]
    fn store(self, repr: [uint32x4_t; 4], out: &mut [u32; 16]) {
        let (o01, o23) = out.split_at_mut(8);
        let (o0, o1) = o01.split_at_mut(4);
        let (o2, o3) = o23.split_at_mut(4);
        <archmage::NeonToken as U32x4Backend>::store(self, repr[0], o0.try_into().unwrap());
        <archmage::NeonToken as U32x4Backend>::store(self, repr[1], o1.try_into().unwrap());
        <archmage::NeonToken as U32x4Backend>::store(self, repr[2], o2.try_into().unwrap());
        <archmage::NeonToken as U32x4Backend>::store(self, repr[3], o3.try_into().unwrap());
    }

    #[inline(always)]
    fn to_array(self, repr: [uint32x4_t; 4]) -> [u32; 16] {
        let a0 = <archmage::NeonToken as U32x4Backend>::to_array(self, repr[0]);
        let a1 = <archmage::NeonToken as U32x4Backend>::to_array(self, repr[1]);
        let a2 = <archmage::NeonToken as U32x4Backend>::to_array(self, repr[2]);
        let a3 = <archmage::NeonToken as U32x4Backend>::to_array(self, repr[3]);
        let mut out = [0; 16];
        out[0..4].copy_from_slice(&a0);
        out[4..8].copy_from_slice(&a1);
        out[8..12].copy_from_slice(&a2);
        out[12..16].copy_from_slice(&a3);
        out
    }

    #[inline(always)]
    fn add(self, a: [uint32x4_t; 4], b: [uint32x4_t; 4]) -> [uint32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U32x4Backend>::add(self, a[i], b[i]))
    }

    #[inline(always)]
    fn sub(self, a: [uint32x4_t; 4], b: [uint32x4_t; 4]) -> [uint32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U32x4Backend>::sub(self, a[i], b[i]))
    }

    #[inline(always)]
    fn mul(self, a: [uint32x4_t; 4], b: [uint32x4_t; 4]) -> [uint32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U32x4Backend>::mul(self, a[i], b[i]))
    }

    #[inline(always)]
    fn neg(self, a: [uint32x4_t; 4]) -> [uint32x4_t; 4] {
        let z = <archmage::NeonToken as U32x4Backend>::zero(self);
        core::array::from_fn(|i| <archmage::NeonToken as U32x4Backend>::sub(self, z, a[i]))
    }

    #[inline(always)]
    fn min(self, a: [uint32x4_t; 4], b: [uint32x4_t; 4]) -> [uint32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U32x4Backend>::min(self, a[i], b[i]))
    }

    #[inline(always)]
    fn max(self, a: [uint32x4_t; 4], b: [uint32x4_t; 4]) -> [uint32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U32x4Backend>::max(self, a[i], b[i]))
    }

    #[inline(always)]
    fn reduce_add(self, a: [uint32x4_t; 4]) -> u32 {
        <archmage::NeonToken as U32x4Backend>::reduce_add(self, a[0])
            .wrapping_add(<archmage::NeonToken as U32x4Backend>::reduce_add(
                self, a[1],
            ))
            .wrapping_add(<archmage::NeonToken as U32x4Backend>::reduce_add(
                self, a[2],
            ))
            .wrapping_add(<archmage::NeonToken as U32x4Backend>::reduce_add(
                self, a[3],
            ))
    }

    #[inline(always)]
    fn shl_const<const N: i32>(self, a: [uint32x4_t; 4]) -> [uint32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U32x4Backend>::shl_const::<N>(self, a[i]))
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(self, a: [uint32x4_t; 4]) -> [uint32x4_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as U32x4Backend>::shr_logical_const::<N>(self, a[i])
        })
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: [uint32x4_t; 4]) -> [uint32x4_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as U32x4Backend>::shr_logical_const::<N>(self, a[i])
        })
    }

    #[inline(always)]
    fn shl_uniform(self, a: [uint32x4_t; 4], count: u32) -> [uint32x4_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as U32x4Backend>::shl_uniform(self, a[i], count)
        })
    }

    #[inline(always)]
    fn shr_logical_uniform(self, a: [uint32x4_t; 4], count: u32) -> [uint32x4_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as U32x4Backend>::shr_logical_uniform(self, a[i], count)
        })
    }

    #[inline(always)]
    fn shr_arithmetic_uniform(self, a: [uint32x4_t; 4], count: u32) -> [uint32x4_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as U32x4Backend>::shr_logical_uniform(self, a[i], count)
        })
    }

    // Reduce the quarters with vector AND/OR first, then ONE lane
    // reduction (fearless_simd's polyfill lesson; branchless).
    #[inline(always)]
    fn all_true(self, a: [uint32x4_t; 4]) -> bool {
        let c01 = <archmage::NeonToken as U32x4Backend>::bitand(self, a[0], a[1]);
        let c23 = <archmage::NeonToken as U32x4Backend>::bitand(self, a[2], a[3]);
        let c = <archmage::NeonToken as U32x4Backend>::bitand(self, c01, c23);
        <archmage::NeonToken as U32x4Backend>::all_true(self, c)
    }

    #[inline(always)]
    fn any_true(self, a: [uint32x4_t; 4]) -> bool {
        let c01 = <archmage::NeonToken as U32x4Backend>::bitor(self, a[0], a[1]);
        let c23 = <archmage::NeonToken as U32x4Backend>::bitor(self, a[2], a[3]);
        let c = <archmage::NeonToken as U32x4Backend>::bitor(self, c01, c23);
        <archmage::NeonToken as U32x4Backend>::any_true(self, c)
    }

    #[inline(always)]
    fn bitmask(self, a: [uint32x4_t; 4]) -> u64 {
        let q0 = <archmage::NeonToken as U32x4Backend>::bitmask(self, a[0]) as u64;
        let q1 = <archmage::NeonToken as U32x4Backend>::bitmask(self, a[1]) as u64;
        let q2 = <archmage::NeonToken as U32x4Backend>::bitmask(self, a[2]) as u64;
        let q3 = <archmage::NeonToken as U32x4Backend>::bitmask(self, a[3]) as u64;
        q0 | (q1 << 4) | (q2 << 8) | (q3 << 12)
    }

    #[inline(always)]
    fn simd_eq(self, a: [uint32x4_t; 4], b: [uint32x4_t; 4]) -> [uint32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U32x4Backend>::simd_eq(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ne(self, a: [uint32x4_t; 4], b: [uint32x4_t; 4]) -> [uint32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U32x4Backend>::simd_ne(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_lt(self, a: [uint32x4_t; 4], b: [uint32x4_t; 4]) -> [uint32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U32x4Backend>::simd_lt(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_le(self, a: [uint32x4_t; 4], b: [uint32x4_t; 4]) -> [uint32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U32x4Backend>::simd_le(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_gt(self, a: [uint32x4_t; 4], b: [uint32x4_t; 4]) -> [uint32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U32x4Backend>::simd_gt(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ge(self, a: [uint32x4_t; 4], b: [uint32x4_t; 4]) -> [uint32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U32x4Backend>::simd_ge(self, a[i], b[i]))
    }

    #[inline(always)]
    fn blend(
        self,
        mask: [uint32x4_t; 4],
        if_true: [uint32x4_t; 4],
        if_false: [uint32x4_t; 4],
    ) -> [uint32x4_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as U32x4Backend>::blend(self, mask[i], if_true[i], if_false[i])
        })
    }

    #[inline(always)]
    fn not(self, a: [uint32x4_t; 4]) -> [uint32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U32x4Backend>::not(self, a[i]))
    }

    #[inline(always)]
    fn bitand(self, a: [uint32x4_t; 4], b: [uint32x4_t; 4]) -> [uint32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U32x4Backend>::bitand(self, a[i], b[i]))
    }

    #[inline(always)]
    fn bitor(self, a: [uint32x4_t; 4], b: [uint32x4_t; 4]) -> [uint32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U32x4Backend>::bitor(self, a[i], b[i]))
    }

    #[inline(always)]
    fn bitxor(self, a: [uint32x4_t; 4], b: [uint32x4_t; 4]) -> [uint32x4_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U32x4Backend>::bitxor(self, a[i], b[i]))
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "aarch64")]
impl I64x8Backend for archmage::NeonToken {
    type Repr = [int64x2_t; 4];

    #[inline(always)]
    fn splat(self, v: i64) -> [int64x2_t; 4] {
        let q = <archmage::NeonToken as I64x2Backend>::splat(self, v);
        [q, q, q, q]
    }

    #[inline(always)]
    fn zero(self) -> [int64x2_t; 4] {
        let q = <archmage::NeonToken as I64x2Backend>::zero(self);
        [q, q, q, q]
    }

    #[inline(always)]
    fn load(self, data: &[i64; 8]) -> [int64x2_t; 4] {
        [
            <archmage::NeonToken as I64x2Backend>::load(self, data[0..2].try_into().unwrap()),
            <archmage::NeonToken as I64x2Backend>::load(self, data[2..4].try_into().unwrap()),
            <archmage::NeonToken as I64x2Backend>::load(self, data[4..6].try_into().unwrap()),
            <archmage::NeonToken as I64x2Backend>::load(self, data[6..8].try_into().unwrap()),
        ]
    }

    #[inline(always)]
    fn from_array(self, arr: [i64; 8]) -> [int64x2_t; 4] {
        let mut q0 = [0; 2];
        let mut q1 = [0; 2];
        let mut q2 = [0; 2];
        let mut q3 = [0; 2];
        q0.copy_from_slice(&arr[0..2]);
        q1.copy_from_slice(&arr[2..4]);
        q2.copy_from_slice(&arr[4..6]);
        q3.copy_from_slice(&arr[6..8]);
        [
            <archmage::NeonToken as I64x2Backend>::from_array(self, q0),
            <archmage::NeonToken as I64x2Backend>::from_array(self, q1),
            <archmage::NeonToken as I64x2Backend>::from_array(self, q2),
            <archmage::NeonToken as I64x2Backend>::from_array(self, q3),
        ]
    }

    #[inline(always)]
    fn store(self, repr: [int64x2_t; 4], out: &mut [i64; 8]) {
        let (o01, o23) = out.split_at_mut(4);
        let (o0, o1) = o01.split_at_mut(2);
        let (o2, o3) = o23.split_at_mut(2);
        <archmage::NeonToken as I64x2Backend>::store(self, repr[0], o0.try_into().unwrap());
        <archmage::NeonToken as I64x2Backend>::store(self, repr[1], o1.try_into().unwrap());
        <archmage::NeonToken as I64x2Backend>::store(self, repr[2], o2.try_into().unwrap());
        <archmage::NeonToken as I64x2Backend>::store(self, repr[3], o3.try_into().unwrap());
    }

    #[inline(always)]
    fn to_array(self, repr: [int64x2_t; 4]) -> [i64; 8] {
        let a0 = <archmage::NeonToken as I64x2Backend>::to_array(self, repr[0]);
        let a1 = <archmage::NeonToken as I64x2Backend>::to_array(self, repr[1]);
        let a2 = <archmage::NeonToken as I64x2Backend>::to_array(self, repr[2]);
        let a3 = <archmage::NeonToken as I64x2Backend>::to_array(self, repr[3]);
        let mut out = [0; 8];
        out[0..2].copy_from_slice(&a0);
        out[2..4].copy_from_slice(&a1);
        out[4..6].copy_from_slice(&a2);
        out[6..8].copy_from_slice(&a3);
        out
    }

    #[inline(always)]
    fn add(self, a: [int64x2_t; 4], b: [int64x2_t; 4]) -> [int64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I64x2Backend>::add(self, a[i], b[i]))
    }

    #[inline(always)]
    fn sub(self, a: [int64x2_t; 4], b: [int64x2_t; 4]) -> [int64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I64x2Backend>::sub(self, a[i], b[i]))
    }

    #[inline(always)]
    fn neg(self, a: [int64x2_t; 4]) -> [int64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I64x2Backend>::neg(self, a[i]))
    }

    #[inline(always)]
    fn min(self, a: [int64x2_t; 4], b: [int64x2_t; 4]) -> [int64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I64x2Backend>::min(self, a[i], b[i]))
    }

    #[inline(always)]
    fn max(self, a: [int64x2_t; 4], b: [int64x2_t; 4]) -> [int64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I64x2Backend>::max(self, a[i], b[i]))
    }

    #[inline(always)]
    fn abs(self, a: [int64x2_t; 4]) -> [int64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I64x2Backend>::abs(self, a[i]))
    }

    #[inline(always)]
    fn reduce_add(self, a: [int64x2_t; 4]) -> i64 {
        <archmage::NeonToken as I64x2Backend>::reduce_add(self, a[0])
            .wrapping_add(<archmage::NeonToken as I64x2Backend>::reduce_add(
                self, a[1],
            ))
            .wrapping_add(<archmage::NeonToken as I64x2Backend>::reduce_add(
                self, a[2],
            ))
            .wrapping_add(<archmage::NeonToken as I64x2Backend>::reduce_add(
                self, a[3],
            ))
    }

    #[inline(always)]
    fn shl_const<const N: i32>(self, a: [int64x2_t; 4]) -> [int64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I64x2Backend>::shl_const::<N>(self, a[i]))
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(self, a: [int64x2_t; 4]) -> [int64x2_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as I64x2Backend>::shr_arithmetic_const::<N>(self, a[i])
        })
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: [int64x2_t; 4]) -> [int64x2_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as I64x2Backend>::shr_logical_const::<N>(self, a[i])
        })
    }

    // Reduce the quarters with vector AND/OR first, then ONE lane
    // reduction (fearless_simd's polyfill lesson; branchless).
    #[inline(always)]
    fn all_true(self, a: [int64x2_t; 4]) -> bool {
        let c01 = <archmage::NeonToken as I64x2Backend>::bitand(self, a[0], a[1]);
        let c23 = <archmage::NeonToken as I64x2Backend>::bitand(self, a[2], a[3]);
        let c = <archmage::NeonToken as I64x2Backend>::bitand(self, c01, c23);
        <archmage::NeonToken as I64x2Backend>::all_true(self, c)
    }

    #[inline(always)]
    fn any_true(self, a: [int64x2_t; 4]) -> bool {
        let c01 = <archmage::NeonToken as I64x2Backend>::bitor(self, a[0], a[1]);
        let c23 = <archmage::NeonToken as I64x2Backend>::bitor(self, a[2], a[3]);
        let c = <archmage::NeonToken as I64x2Backend>::bitor(self, c01, c23);
        <archmage::NeonToken as I64x2Backend>::any_true(self, c)
    }

    #[inline(always)]
    fn bitmask(self, a: [int64x2_t; 4]) -> u64 {
        let q0 = <archmage::NeonToken as I64x2Backend>::bitmask(self, a[0]) as u64;
        let q1 = <archmage::NeonToken as I64x2Backend>::bitmask(self, a[1]) as u64;
        let q2 = <archmage::NeonToken as I64x2Backend>::bitmask(self, a[2]) as u64;
        let q3 = <archmage::NeonToken as I64x2Backend>::bitmask(self, a[3]) as u64;
        q0 | (q1 << 2) | (q2 << 4) | (q3 << 6)
    }

    #[inline(always)]
    fn simd_eq(self, a: [int64x2_t; 4], b: [int64x2_t; 4]) -> [int64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I64x2Backend>::simd_eq(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ne(self, a: [int64x2_t; 4], b: [int64x2_t; 4]) -> [int64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I64x2Backend>::simd_ne(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_lt(self, a: [int64x2_t; 4], b: [int64x2_t; 4]) -> [int64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I64x2Backend>::simd_lt(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_le(self, a: [int64x2_t; 4], b: [int64x2_t; 4]) -> [int64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I64x2Backend>::simd_le(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_gt(self, a: [int64x2_t; 4], b: [int64x2_t; 4]) -> [int64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I64x2Backend>::simd_gt(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ge(self, a: [int64x2_t; 4], b: [int64x2_t; 4]) -> [int64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I64x2Backend>::simd_ge(self, a[i], b[i]))
    }

    #[inline(always)]
    fn blend(
        self,
        mask: [int64x2_t; 4],
        if_true: [int64x2_t; 4],
        if_false: [int64x2_t; 4],
    ) -> [int64x2_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as I64x2Backend>::blend(self, mask[i], if_true[i], if_false[i])
        })
    }

    #[inline(always)]
    fn not(self, a: [int64x2_t; 4]) -> [int64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I64x2Backend>::not(self, a[i]))
    }

    #[inline(always)]
    fn bitand(self, a: [int64x2_t; 4], b: [int64x2_t; 4]) -> [int64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I64x2Backend>::bitand(self, a[i], b[i]))
    }

    #[inline(always)]
    fn bitor(self, a: [int64x2_t; 4], b: [int64x2_t; 4]) -> [int64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I64x2Backend>::bitor(self, a[i], b[i]))
    }

    #[inline(always)]
    fn bitxor(self, a: [int64x2_t; 4], b: [int64x2_t; 4]) -> [int64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as I64x2Backend>::bitxor(self, a[i], b[i]))
    }
}

#[cfg(feature = "w512")]
#[cfg(target_arch = "aarch64")]
impl U64x8Backend for archmage::NeonToken {
    type Repr = [uint64x2_t; 4];

    #[inline(always)]
    fn splat(self, v: u64) -> [uint64x2_t; 4] {
        let q = <archmage::NeonToken as U64x2Backend>::splat(self, v);
        [q, q, q, q]
    }

    #[inline(always)]
    fn zero(self) -> [uint64x2_t; 4] {
        let q = <archmage::NeonToken as U64x2Backend>::zero(self);
        [q, q, q, q]
    }

    #[inline(always)]
    fn load(self, data: &[u64; 8]) -> [uint64x2_t; 4] {
        [
            <archmage::NeonToken as U64x2Backend>::load(self, data[0..2].try_into().unwrap()),
            <archmage::NeonToken as U64x2Backend>::load(self, data[2..4].try_into().unwrap()),
            <archmage::NeonToken as U64x2Backend>::load(self, data[4..6].try_into().unwrap()),
            <archmage::NeonToken as U64x2Backend>::load(self, data[6..8].try_into().unwrap()),
        ]
    }

    #[inline(always)]
    fn from_array(self, arr: [u64; 8]) -> [uint64x2_t; 4] {
        let mut q0 = [0; 2];
        let mut q1 = [0; 2];
        let mut q2 = [0; 2];
        let mut q3 = [0; 2];
        q0.copy_from_slice(&arr[0..2]);
        q1.copy_from_slice(&arr[2..4]);
        q2.copy_from_slice(&arr[4..6]);
        q3.copy_from_slice(&arr[6..8]);
        [
            <archmage::NeonToken as U64x2Backend>::from_array(self, q0),
            <archmage::NeonToken as U64x2Backend>::from_array(self, q1),
            <archmage::NeonToken as U64x2Backend>::from_array(self, q2),
            <archmage::NeonToken as U64x2Backend>::from_array(self, q3),
        ]
    }

    #[inline(always)]
    fn store(self, repr: [uint64x2_t; 4], out: &mut [u64; 8]) {
        let (o01, o23) = out.split_at_mut(4);
        let (o0, o1) = o01.split_at_mut(2);
        let (o2, o3) = o23.split_at_mut(2);
        <archmage::NeonToken as U64x2Backend>::store(self, repr[0], o0.try_into().unwrap());
        <archmage::NeonToken as U64x2Backend>::store(self, repr[1], o1.try_into().unwrap());
        <archmage::NeonToken as U64x2Backend>::store(self, repr[2], o2.try_into().unwrap());
        <archmage::NeonToken as U64x2Backend>::store(self, repr[3], o3.try_into().unwrap());
    }

    #[inline(always)]
    fn to_array(self, repr: [uint64x2_t; 4]) -> [u64; 8] {
        let a0 = <archmage::NeonToken as U64x2Backend>::to_array(self, repr[0]);
        let a1 = <archmage::NeonToken as U64x2Backend>::to_array(self, repr[1]);
        let a2 = <archmage::NeonToken as U64x2Backend>::to_array(self, repr[2]);
        let a3 = <archmage::NeonToken as U64x2Backend>::to_array(self, repr[3]);
        let mut out = [0; 8];
        out[0..2].copy_from_slice(&a0);
        out[2..4].copy_from_slice(&a1);
        out[4..6].copy_from_slice(&a2);
        out[6..8].copy_from_slice(&a3);
        out
    }

    #[inline(always)]
    fn add(self, a: [uint64x2_t; 4], b: [uint64x2_t; 4]) -> [uint64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U64x2Backend>::add(self, a[i], b[i]))
    }

    #[inline(always)]
    fn sub(self, a: [uint64x2_t; 4], b: [uint64x2_t; 4]) -> [uint64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U64x2Backend>::sub(self, a[i], b[i]))
    }

    #[inline(always)]
    fn neg(self, a: [uint64x2_t; 4]) -> [uint64x2_t; 4] {
        let z = <archmage::NeonToken as U64x2Backend>::zero(self);
        core::array::from_fn(|i| <archmage::NeonToken as U64x2Backend>::sub(self, z, a[i]))
    }

    #[inline(always)]
    fn min(self, a: [uint64x2_t; 4], b: [uint64x2_t; 4]) -> [uint64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U64x2Backend>::min(self, a[i], b[i]))
    }

    #[inline(always)]
    fn max(self, a: [uint64x2_t; 4], b: [uint64x2_t; 4]) -> [uint64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U64x2Backend>::max(self, a[i], b[i]))
    }

    #[inline(always)]
    fn reduce_add(self, a: [uint64x2_t; 4]) -> u64 {
        <archmage::NeonToken as U64x2Backend>::reduce_add(self, a[0])
            .wrapping_add(<archmage::NeonToken as U64x2Backend>::reduce_add(
                self, a[1],
            ))
            .wrapping_add(<archmage::NeonToken as U64x2Backend>::reduce_add(
                self, a[2],
            ))
            .wrapping_add(<archmage::NeonToken as U64x2Backend>::reduce_add(
                self, a[3],
            ))
    }

    #[inline(always)]
    fn shl_const<const N: i32>(self, a: [uint64x2_t; 4]) -> [uint64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U64x2Backend>::shl_const::<N>(self, a[i]))
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(self, a: [uint64x2_t; 4]) -> [uint64x2_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as U64x2Backend>::shr_logical_const::<N>(self, a[i])
        })
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: [uint64x2_t; 4]) -> [uint64x2_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as U64x2Backend>::shr_logical_const::<N>(self, a[i])
        })
    }

    // Reduce the quarters with vector AND/OR first, then ONE lane
    // reduction (fearless_simd's polyfill lesson; branchless).
    #[inline(always)]
    fn all_true(self, a: [uint64x2_t; 4]) -> bool {
        let c01 = <archmage::NeonToken as U64x2Backend>::bitand(self, a[0], a[1]);
        let c23 = <archmage::NeonToken as U64x2Backend>::bitand(self, a[2], a[3]);
        let c = <archmage::NeonToken as U64x2Backend>::bitand(self, c01, c23);
        <archmage::NeonToken as U64x2Backend>::all_true(self, c)
    }

    #[inline(always)]
    fn any_true(self, a: [uint64x2_t; 4]) -> bool {
        let c01 = <archmage::NeonToken as U64x2Backend>::bitor(self, a[0], a[1]);
        let c23 = <archmage::NeonToken as U64x2Backend>::bitor(self, a[2], a[3]);
        let c = <archmage::NeonToken as U64x2Backend>::bitor(self, c01, c23);
        <archmage::NeonToken as U64x2Backend>::any_true(self, c)
    }

    #[inline(always)]
    fn bitmask(self, a: [uint64x2_t; 4]) -> u64 {
        let q0 = <archmage::NeonToken as U64x2Backend>::bitmask(self, a[0]) as u64;
        let q1 = <archmage::NeonToken as U64x2Backend>::bitmask(self, a[1]) as u64;
        let q2 = <archmage::NeonToken as U64x2Backend>::bitmask(self, a[2]) as u64;
        let q3 = <archmage::NeonToken as U64x2Backend>::bitmask(self, a[3]) as u64;
        q0 | (q1 << 2) | (q2 << 4) | (q3 << 6)
    }

    #[inline(always)]
    fn simd_eq(self, a: [uint64x2_t; 4], b: [uint64x2_t; 4]) -> [uint64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U64x2Backend>::simd_eq(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ne(self, a: [uint64x2_t; 4], b: [uint64x2_t; 4]) -> [uint64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U64x2Backend>::simd_ne(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_lt(self, a: [uint64x2_t; 4], b: [uint64x2_t; 4]) -> [uint64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U64x2Backend>::simd_lt(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_le(self, a: [uint64x2_t; 4], b: [uint64x2_t; 4]) -> [uint64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U64x2Backend>::simd_le(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_gt(self, a: [uint64x2_t; 4], b: [uint64x2_t; 4]) -> [uint64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U64x2Backend>::simd_gt(self, a[i], b[i]))
    }

    #[inline(always)]
    fn simd_ge(self, a: [uint64x2_t; 4], b: [uint64x2_t; 4]) -> [uint64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U64x2Backend>::simd_ge(self, a[i], b[i]))
    }

    #[inline(always)]
    fn blend(
        self,
        mask: [uint64x2_t; 4],
        if_true: [uint64x2_t; 4],
        if_false: [uint64x2_t; 4],
    ) -> [uint64x2_t; 4] {
        core::array::from_fn(|i| {
            <archmage::NeonToken as U64x2Backend>::blend(self, mask[i], if_true[i], if_false[i])
        })
    }

    #[inline(always)]
    fn not(self, a: [uint64x2_t; 4]) -> [uint64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U64x2Backend>::not(self, a[i]))
    }

    #[inline(always)]
    fn bitand(self, a: [uint64x2_t; 4], b: [uint64x2_t; 4]) -> [uint64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U64x2Backend>::bitand(self, a[i], b[i]))
    }

    #[inline(always)]
    fn bitor(self, a: [uint64x2_t; 4], b: [uint64x2_t; 4]) -> [uint64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U64x2Backend>::bitor(self, a[i], b[i]))
    }

    #[inline(always)]
    fn bitxor(self, a: [uint64x2_t; 4], b: [uint64x2_t; 4]) -> [uint64x2_t; 4] {
        core::array::from_fn(|i| <archmage::NeonToken as U64x2Backend>::bitxor(self, a[i], b[i]))
    }
}

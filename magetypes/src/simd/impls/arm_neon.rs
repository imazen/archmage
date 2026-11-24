//! Backend implementations for NeonToken (AArch64 NEON).
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.
//!
//! # Safety (audit contract — checked backend boundaries)
//!
//! `#[attune(wrap)]` checks value intrinsics against the receiver token's features.
//! Whole-array loads/stores and bit casts use `crate::simd_storage` helpers,
//! which require POD storage and enforce equal sizes at compile time.
//! Token-bearing wrappers never implement the storage POD trait.

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
use archmage::{NeonToken, attune};

use crate::simd::backends::*;

#[cfg(target_arch = "aarch64")]
impl F32x4Backend for NeonToken {
    type Repr = float32x4_t;

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn splat(self, v: f32) -> float32x4_t {
        vdupq_n_f32(v)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
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

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn add(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vaddq_f32(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn sub(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vsubq_f32(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn mul(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vmulq_f32(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn div(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vdivq_f32(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn neg(self, a: float32x4_t) -> float32x4_t {
        vnegq_f32(a)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn min(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vminnmq_f32(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn max(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vmaxnmq_f32(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn minimum(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vminq_f32(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn maximum(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vmaxq_f32(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn min_fast(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vminnmq_f32(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn max_fast(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vmaxnmq_f32(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn sqrt(self, a: float32x4_t) -> float32x4_t {
        vsqrtq_f32(a)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn abs(self, a: float32x4_t) -> float32x4_t {
        vabsq_f32(a)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn trunc(self, a: float32x4_t) -> float32x4_t {
        vrndq_f32(a)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn mul_neg_add_relaxed(self, a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
        vfmsq_f32(c, a, b)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn mul_neg_add_fused(self, a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
        vfmsq_f32(c, a, b)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn floor(self, a: float32x4_t) -> float32x4_t {
        vrndmq_f32(a)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn ceil(self, a: float32x4_t) -> float32x4_t {
        vrndpq_f32(a)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn round(self, a: float32x4_t) -> float32x4_t {
        vrndnq_f32(a)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn mul_add(self, a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
        vfmaq_f32(c, a, b)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn mul_sub(self, a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
        vfmaq_f32(vnegq_f32(c), a, b)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn mul_add_portable(self, a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
        vfmaq_f32(c, a, b)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn mul_sub_portable(self, a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
        vfmaq_f32(vnegq_f32(c), a, b)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_eq(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vreinterpretq_f32_u32(vceqq_f32(a, b))
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_ne(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vreinterpretq_f32_u32(vmvnq_u32(vceqq_f32(a, b)))
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_lt(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vreinterpretq_f32_u32(vcltq_f32(a, b))
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_le(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vreinterpretq_f32_u32(vcleq_f32(a, b))
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_gt(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vreinterpretq_f32_u32(vcgtq_f32(a, b))
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_ge(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vreinterpretq_f32_u32(vcgeq_f32(a, b))
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn blend(self, mask: float32x4_t, if_true: float32x4_t, if_false: float32x4_t) -> float32x4_t {
        vbslq_f32(vreinterpretq_u32_f32(mask), if_true, if_false)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn reduce_add(self, a: float32x4_t) -> f32 {
        {
            let pair = vpaddq_f32(a, a);
            let pair = vpaddq_f32(pair, pair);
            vgetq_lane_f32::<0>(pair)
        }
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn reduce_min(self, a: float32x4_t) -> f32 {
        vminnmvq_f32(a)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn reduce_max(self, a: float32x4_t) -> f32 {
        vmaxnmvq_f32(a)
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
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn rcp_approx(self, a: float32x4_t) -> float32x4_t {
        let y = vrecpeq_f32(a);
        vmulq_f32(vrecpsq_f32(a, y), y)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn rsqrt_approx(self, a: float32x4_t) -> float32x4_t {
        let y = vrsqrteq_f32(a);
        vmulq_f32(y, vrsqrtsq_f32(a, vmulq_f32(y, y)))
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn recip(self, a: float32x4_t) -> float32x4_t {
        let y = <Self as F32x4Backend>::rcp_approx(_self, a);
        vmulq_f32(vrecpsq_f32(a, y), y)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn rsqrt(self, a: float32x4_t) -> float32x4_t {
        let y = <Self as F32x4Backend>::rsqrt_approx(_self, a);
        vmulq_f32(y, vrsqrtsq_f32(a, vmulq_f32(y, y)))
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn not(self, a: float32x4_t) -> float32x4_t {
        vreinterpretq_f32_u32(vmvnq_u32(vreinterpretq_u32_f32(a)))
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitand(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vreinterpretq_f32_u32(vandq_u32(
            vreinterpretq_u32_f32(a),
            vreinterpretq_u32_f32(b),
        ))
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitor(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vreinterpretq_f32_u32(vorrq_u32(
            vreinterpretq_u32_f32(a),
            vreinterpretq_u32_f32(b),
        ))
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitxor(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vreinterpretq_f32_u32(veorq_u32(
            vreinterpretq_u32_f32(a),
            vreinterpretq_u32_f32(b),
        ))
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn to_u8_bytes(self, a: float32x4_t) -> [u8; 4] {
        let i16s = vqmovn_s32(vcvtnq_s32_f32(a));
        let u8s = vqmovun_s16(vcombine_s16(i16s, i16s));
        let bytes: [u8; 8] = crate::simd_storage::cast(u8s);
        [bytes[0], bytes[1], bytes[2], bytes[3]]
    }
}
#[cfg(target_arch = "aarch64")]
impl F32x4Internal for NeonToken {
    type InternalRepr = float32x4_t;

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn store_rgba_u8_into(
        self,
        r: float32x4_t,
        g: float32x4_t,
        b: float32x4_t,
        a: float32x4_t,
        out: &mut [u8; 16],
    ) {
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
        let bytes = vreinterpretq_u8_u32(pixels);
        #[cfg(target_endian = "big")]
        let bytes = vrev32q_u8(bytes);
        crate::simd_storage::store(bytes, out);
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn from_u16_lanes(self, lanes: &[u16; 4]) -> float32x4_t {
        vcvtq_f32_u32(vmovl_u16(crate::simd_storage::cast(*lanes)))
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn to_u16_lanes<const CFG: u32>(self, value: float32x4_t) -> [u16; 4] {
        let i = if const { crate::numerics::IntConvert::decode(CFG).rounds() } {
            vcvtnq_u32_f32(value)
        } else {
            vcvtq_u32_f32(value)
        };
        crate::simd_storage::cast(vqmovn_u32(i))
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn to_u8_lanes<const CFG: u32>(self, value: float32x4_t) -> [u8; 4] {
        let i = if const { crate::numerics::IntConvert::decode(CFG).rounds() } {
            vcvtnq_u32_f32(value)
        } else {
            vcvtq_u32_f32(value)
        };
        let words = vqmovn_u32(i);
        crate::simd_storage::cast(vget_lane_u32::<0>(vreinterpret_u32_u8(vqmovn_u16(
            vcombine_u16(words, words),
        ))))
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn store_rgba_u8_with_into<const CFG: u32>(
        self,
        r: float32x4_t,
        g: float32x4_t,
        b: float32x4_t,
        a: float32x4_t,
        out: &mut [u8; 16],
    ) {
        let opts = const { crate::numerics::IntConvert::decode(CFG) };
        if cfg!(target_endian = "little") && opts.is_in_range() {
            let convert = |x| {
                if opts.rounds() {
                    vcvtnq_u32_f32(x)
                } else {
                    vcvtq_u32_f32(x)
                }
            };
            let rg = vsliq_n_u32::<8>(convert(r), convert(g));
            let rgb = vsliq_n_u32::<16>(rg, convert(b));
            crate::simd_storage::store(vsliq_n_u32::<24>(rgb, convert(a)), out);
            return;
        }
        let channels = [r, g, b, a].map(|v| <Self as F32x4Internal>::to_u8_lanes::<CFG>(self, v));
        for (i, byte) in out.iter_mut().enumerate() {
            *byte = channels[i % 4][i / 4];
        }
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn to_u32_lanes<const CFG: u32>(self, value: float32x4_t) -> float32x4_t {
        vreinterpretq_f32_u32(
            if const { crate::numerics::IntConvert::decode(CFG).rounds() } {
                vcvtnq_u32_f32(value)
            } else {
                vcvtq_u32_f32(value)
            },
        )
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn from_u32_lanes(self, value: float32x4_t) -> float32x4_t {
        vcvtq_f32_u32(vreinterpretq_u32_f32(value))
    }

    #[rustversion::since(1.94)]
    #[allow(clippy::incompatible_msrv)]
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn from_f16_lanes<const CFG: u32>(self, lanes: &[u16; 4]) -> float32x4_t {
        let words: uint16x4_t = crate::simd_storage::cast(*lanes);
        let words = if const { crate::numerics::HalfConvert::decode(CFG).flushes() } {
            let mag = vand_u16(words, vdup_n_u16(0x7fff));
            vbsl_u16(
                vclt_u16(mag, vdup_n_u16(0x400)),
                vand_u16(words, vdup_n_u16(0x8000)),
                words,
            )
        } else {
            words
        };
        vcvt_f32_f16(vreinterpret_f16_u16(words))
    }

    #[rustversion::before(1.94)]
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn from_f16_lanes<const CFG: u32>(self, lanes: &[u16; 4]) -> float32x4_t {
        let words = vreinterpretq_s32_u32(vmovl_u16(crate::simd_storage::cast(*lanes)));
        crate::simd::generic::half4::widen_f16::<CFG, _>(
            crate::simd::generic::i32x4::from_repr_unchecked(self, words),
        )
        .into_repr()
    }

    #[rustversion::since(1.94)]
    #[allow(clippy::incompatible_msrv)]
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn to_f16_lanes<const CFG: u32>(self, value: float32x4_t) -> [u16; 4] {
        let opts = const { crate::numerics::HalfConvert::decode(CFG) };
        if opts.rounds() {
            let h = vreinterpret_u16_f16(vcvt_f16_f32(value));
            let words = if opts.flushes() {
                let mag = vand_u16(h, vdup_n_u16(0x7fff));
                vbsl_u16(
                    vclt_u16(mag, vdup_n_u16(0x400)),
                    vand_u16(h, vdup_n_u16(0x8000)),
                    h,
                )
            } else {
                h
            };
            crate::simd_storage::cast(words)
        } else {
            let v = crate::simd::generic::f32x4::from_repr_unchecked(self, value);
            let i = crate::simd::generic::half4::narrow_f16::<CFG, _>(v).into_repr();
            crate::simd_storage::cast(vmovn_u32(vreinterpretq_u32_s32(i)))
        }
    }

    #[rustversion::before(1.94)]
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn to_f16_lanes<const CFG: u32>(self, value: float32x4_t) -> [u16; 4] {
        let v = crate::simd::generic::f32x4::from_repr_unchecked(self, value);
        let i = crate::simd::generic::half4::narrow_f16::<CFG, _>(v).into_repr();
        crate::simd_storage::cast(vmovn_u32(vreinterpretq_u32_s32(i)))
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn to_bf16_lanes<const CFG: u32>(self, value: float32x4_t) -> [u16; 4] {
        let v = crate::simd::generic::f32x4::from_repr_unchecked(self, value);
        let i = crate::simd::generic::half4::narrow_bf16::<CFG, _>(v).into_repr();
        crate::simd_storage::cast(vmovn_u32(vreinterpretq_u32_s32(i)))
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn from_bf16_lanes(self, lanes: &[u16; 4]) -> float32x4_t {
        vreinterpretq_f32_u32(vshll_n_u16::<16>(crate::simd_storage::cast(*lanes)))
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn broadcast_rgba_alpha_repr(self, value: float32x4_t) -> float32x4_t {
        vdupq_laneq_f32::<3>(value)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn with_rgba_alpha_repr(self, value: float32x4_t, src: float32x4_t) -> float32x4_t {
        vcopyq_laneq_f32::<3, 3>(value, src)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn transpose_4x4_repr(self, rows: [float32x4_t; 4]) -> [float32x4_t; 4] {
        let [a, b, c, d] = rows;
        let ab_lo = vzip1q_f32(a, b);
        let ab_hi = vzip2q_f32(a, b);
        let cd_lo = vzip1q_f32(c, d);
        let cd_hi = vzip2q_f32(c, d);
        [
            vcombine_f32(vget_low_f32(ab_lo), vget_low_f32(cd_lo)),
            vcombine_f32(vget_high_f32(ab_lo), vget_high_f32(cd_lo)),
            vcombine_f32(vget_low_f32(ab_hi), vget_low_f32(cd_hi)),
            vcombine_f32(vget_high_f32(ab_hi), vget_high_f32(cd_hi)),
        ]
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn deinterleave_3ch_repr(self, rgb: [float32x4_t; 3]) -> [float32x4_t; 3] {
        let idx_r: uint8x16_t = crate::simd_storage::cast([
            0u8, 1, 2, 3, 12, 13, 14, 15, 24, 25, 26, 27, 36, 37, 38, 39,
        ]);
        let idx_g: uint8x16_t = crate::simd_storage::cast([
            4u8, 5, 6, 7, 16, 17, 18, 19, 28, 29, 30, 31, 40, 41, 42, 43,
        ]);
        let idx_b: uint8x16_t = crate::simd_storage::cast([
            8u8, 9, 10, 11, 20, 21, 22, 23, 32, 33, 34, 35, 44, 45, 46, 47,
        ]);
        let tbl = uint8x16x3_t(
            vreinterpretq_u8_f32(rgb[0]),
            vreinterpretq_u8_f32(rgb[1]),
            vreinterpretq_u8_f32(rgb[2]),
        );
        [
            vreinterpretq_f32_u8(vqtbl3q_u8(tbl, idx_r)),
            vreinterpretq_f32_u8(vqtbl3q_u8(tbl, idx_g)),
            vreinterpretq_f32_u8(vqtbl3q_u8(tbl, idx_b)),
        ]
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn interleave_3ch_repr(self, channels: [float32x4_t; 3]) -> [float32x4_t; 3] {
        let idx_0: uint8x16_t =
            crate::simd_storage::cast([0u8, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 4, 5, 6, 7]);
        let idx_1: uint8x16_t = crate::simd_storage::cast([
            20u8, 21, 22, 23, 36, 37, 38, 39, 8, 9, 10, 11, 24, 25, 26, 27,
        ]);
        let idx_2: uint8x16_t = crate::simd_storage::cast([
            40u8, 41, 42, 43, 12, 13, 14, 15, 28, 29, 30, 31, 44, 45, 46, 47,
        ]);
        let tbl = uint8x16x3_t(
            vreinterpretq_u8_f32(channels[0]),
            vreinterpretq_u8_f32(channels[1]),
            vreinterpretq_u8_f32(channels[2]),
        );
        [
            vreinterpretq_f32_u8(vqtbl3q_u8(tbl, idx_0)),
            vreinterpretq_f32_u8(vqtbl3q_u8(tbl, idx_1)),
            vreinterpretq_f32_u8(vqtbl3q_u8(tbl, idx_2)),
        ]
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn load_rgb_bytes(self, rgb: &[u8; 12]) -> (float32x4_t, float32x4_t, float32x4_t) {
        let lo8: uint8x8_t = crate::simd_storage::copy(rgb.first_chunk::<8>().unwrap());
        let hi4: u32 = crate::simd_storage::copy(rgb.last_chunk::<4>().unwrap());
        let raw = vreinterpretq_u8_u32(vsetq_lane_u32::<2>(
            hi4,
            vreinterpretq_u32_u8(vcombine_u8(lo8, vdup_n_u8(0))),
        ));
        // All 12 channel bytes fit in one 16B tbl: [R0..3, G0..3, B0..3, 0..0].
        let idx: uint8x16_t =
            crate::simd_storage::cast([0u8, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11, 255, 255, 255, 255]);
        let rgb8 = vqtbl1q_u8(raw, idx);
        let rg16 = vmovl_u8(vget_low_u8(rgb8));
        let b16 = vmovl_high_u8(rgb8);
        (
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(rg16))),
            vcvtq_f32_u32(vmovl_high_u16(rg16)),
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(b16))),
        )
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn store_rgb_u8_into(self, r: float32x4_t, g: float32x4_t, b: float32x4_t, out: &mut [u8; 12]) {
        let ri = vqmovn_s32(vcvtnq_s32_f32(r));
        let gi = vqmovn_s32(vcvtnq_s32_f32(g));
        let bi = vqmovn_s32(vcvtnq_s32_f32(b));
        let rg8 = vqmovun_s16(vcombine_s16(ri, gi));
        let b8 = vqmovun_s16(vcombine_s16(bi, bi));
        let planes = vcombine_u8(rg8, b8);
        let idx: uint8x16_t =
            crate::simd_storage::cast([0u8, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11, 255, 255, 255, 255]);
        let packed = vqtbl1q_u8(planes, idx);
        crate::simd_storage::store(vget_low_u8(packed), out.first_chunk_mut::<8>().unwrap());
        crate::simd_storage::store(
            vgetq_lane_u32::<2>(vreinterpretq_u32_u8(packed)),
            out.last_chunk_mut::<4>().unwrap(),
        );
    }
}

#[cfg(target_arch = "aarch64")]
#[attune(wrap, suppress_const_test)]
pub(super) fn f32x8_to_u8_bytes(_token: NeonToken, a: [float32x4_t; 2]) -> [u8; 8] {
    let i0 = vqmovn_s32(vcvtnq_s32_f32(a[0]));
    let i1 = vqmovn_s32(vcvtnq_s32_f32(a[1]));
    let u8s = vqmovun_s16(vcombine_s16(i0, i1));
    crate::simd_storage::cast(u8s)
}

#[cfg(target_arch = "aarch64")]
impl F32x8Internal for NeonToken {
    type InternalRepr = [float32x4_t; 2];

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn store_rgba_u8_into(
        self,
        r: [float32x4_t; 2],
        g: [float32x4_t; 2],
        b: [float32x4_t; 2],
        a: [float32x4_t; 2],
        out: &mut [u8; 32],
    ) {
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
        let b0 = vreinterpretq_u8_u32(p0);
        let b1 = vreinterpretq_u8_u32(p1);
        #[cfg(target_endian = "big")]
        let (b0, b1) = (vrev32q_u8(b0), vrev32q_u8(b1));
        crate::simd_storage::store([b0, b1], out);
    }

    #[inline(always)]
    fn from_u16_lanes(self, lanes: &[u16; 8]) -> [float32x4_t; 2] {
        let (chunks, _) = lanes.as_chunks::<4>();
        [
            <Self as F32x4Internal>::from_u16_lanes(self, &chunks[0]),
            <Self as F32x4Internal>::from_u16_lanes(self, &chunks[1]),
        ]
    }

    #[inline(always)]
    fn to_u16_lanes<const CFG: u32>(self, value: [float32x4_t; 2]) -> [u16; 8] {
        crate::simd_storage::cast([
            <Self as F32x4Internal>::to_u16_lanes::<CFG>(self, value[0]),
            <Self as F32x4Internal>::to_u16_lanes::<CFG>(self, value[1]),
        ])
    }

    #[inline(always)]
    fn to_u8_lanes<const CFG: u32>(self, value: [float32x4_t; 2]) -> [u8; 8] {
        crate::simd_storage::cast([
            <Self as F32x4Internal>::to_u8_lanes::<CFG>(self, value[0]),
            <Self as F32x4Internal>::to_u8_lanes::<CFG>(self, value[1]),
        ])
    }

    #[inline(always)]
    fn to_u32_lanes<const CFG: u32>(self, value: [float32x4_t; 2]) -> [float32x4_t; 2] {
        [
            <Self as F32x4Internal>::to_u32_lanes::<CFG>(self, value[0]),
            <Self as F32x4Internal>::to_u32_lanes::<CFG>(self, value[1]),
        ]
    }

    #[inline(always)]
    fn from_u32_lanes(self, value: [float32x4_t; 2]) -> [float32x4_t; 2] {
        [
            <Self as F32x4Internal>::from_u32_lanes(self, value[0]),
            <Self as F32x4Internal>::from_u32_lanes(self, value[1]),
        ]
    }

    #[inline(always)]
    fn from_f16_lanes<const CFG: u32>(self, lanes: &[u16; 8]) -> [float32x4_t; 2] {
        let (chunks, _) = lanes.as_chunks::<4>();
        [
            <Self as F32x4Internal>::from_f16_lanes::<CFG>(self, &chunks[0]),
            <Self as F32x4Internal>::from_f16_lanes::<CFG>(self, &chunks[1]),
        ]
    }

    #[inline(always)]
    fn to_f16_lanes<const CFG: u32>(self, value: [float32x4_t; 2]) -> [u16; 8] {
        crate::simd_storage::cast([
            <Self as F32x4Internal>::to_f16_lanes::<CFG>(self, value[0]),
            <Self as F32x4Internal>::to_f16_lanes::<CFG>(self, value[1]),
        ])
    }

    #[inline(always)]
    fn from_bf16_lanes(self, lanes: &[u16; 8]) -> [float32x4_t; 2] {
        let (chunks, _) = lanes.as_chunks::<4>();
        [
            <Self as F32x4Internal>::from_bf16_lanes(self, &chunks[0]),
            <Self as F32x4Internal>::from_bf16_lanes(self, &chunks[1]),
        ]
    }

    #[inline(always)]
    fn to_bf16_lanes<const CFG: u32>(self, value: [float32x4_t; 2]) -> [u16; 8] {
        crate::simd_storage::cast([
            <Self as F32x4Internal>::to_bf16_lanes::<CFG>(self, value[0]),
            <Self as F32x4Internal>::to_bf16_lanes::<CFG>(self, value[1]),
        ])
    }

    #[inline(always)]
    fn broadcast_rgba_alpha_repr(self, value: [float32x4_t; 2]) -> [float32x4_t; 2] {
        [
            <Self as F32x4Internal>::broadcast_rgba_alpha_repr(self, value[0]),
            <Self as F32x4Internal>::broadcast_rgba_alpha_repr(self, value[1]),
        ]
    }

    #[inline(always)]
    fn with_rgba_alpha_repr(
        self,
        value: [float32x4_t; 2],
        src: [float32x4_t; 2],
    ) -> [float32x4_t; 2] {
        [
            <Self as F32x4Internal>::with_rgba_alpha_repr(self, value[0], src[0]),
            <Self as F32x4Internal>::with_rgba_alpha_repr(self, value[1], src[1]),
        ]
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn deinterleave_3ch_repr(self, rgb: [[float32x4_t; 2]; 3]) -> [[float32x4_t; 2]; 3] {
        let idx_r: uint8x16_t = crate::simd_storage::cast([
            0u8, 1, 2, 3, 12, 13, 14, 15, 24, 25, 26, 27, 36, 37, 38, 39,
        ]);
        let idx_g: uint8x16_t = crate::simd_storage::cast([
            4u8, 5, 6, 7, 16, 17, 18, 19, 28, 29, 30, 31, 40, 41, 42, 43,
        ]);
        let idx_b: uint8x16_t = crate::simd_storage::cast([
            8u8, 9, 10, 11, 20, 21, 22, 23, 32, 33, 34, 35, 44, 45, 46, 47,
        ]);
        let tbl_lo = uint8x16x3_t(
            vreinterpretq_u8_f32(rgb[0][0]),
            vreinterpretq_u8_f32(rgb[0][1]),
            vreinterpretq_u8_f32(rgb[1][0]),
        );
        let tbl_hi = uint8x16x3_t(
            vreinterpretq_u8_f32(rgb[1][1]),
            vreinterpretq_u8_f32(rgb[2][0]),
            vreinterpretq_u8_f32(rgb[2][1]),
        );
        [
            [
                vreinterpretq_f32_u8(vqtbl3q_u8(tbl_lo, idx_r)),
                vreinterpretq_f32_u8(vqtbl3q_u8(tbl_hi, idx_r)),
            ],
            [
                vreinterpretq_f32_u8(vqtbl3q_u8(tbl_lo, idx_g)),
                vreinterpretq_f32_u8(vqtbl3q_u8(tbl_hi, idx_g)),
            ],
            [
                vreinterpretq_f32_u8(vqtbl3q_u8(tbl_lo, idx_b)),
                vreinterpretq_f32_u8(vqtbl3q_u8(tbl_hi, idx_b)),
            ],
        ]
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn interleave_3ch_repr(self, channels: [[float32x4_t; 2]; 3]) -> [[float32x4_t; 2]; 3] {
        let [r, g, b] = channels;
        let idx_0: uint8x16_t =
            crate::simd_storage::cast([0u8, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 4, 5, 6, 7]);
        let idx_1: uint8x16_t = crate::simd_storage::cast([
            20u8, 21, 22, 23, 36, 37, 38, 39, 8, 9, 10, 11, 24, 25, 26, 27,
        ]);
        let idx_2: uint8x16_t = crate::simd_storage::cast([
            40u8, 41, 42, 43, 12, 13, 14, 15, 28, 29, 30, 31, 44, 45, 46, 47,
        ]);
        let tbl_lo = uint8x16x3_t(
            vreinterpretq_u8_f32(r[0]),
            vreinterpretq_u8_f32(g[0]),
            vreinterpretq_u8_f32(b[0]),
        );
        let tbl_hi = uint8x16x3_t(
            vreinterpretq_u8_f32(r[1]),
            vreinterpretq_u8_f32(g[1]),
            vreinterpretq_u8_f32(b[1]),
        );
        [
            [
                vreinterpretq_f32_u8(vqtbl3q_u8(tbl_lo, idx_0)),
                vreinterpretq_f32_u8(vqtbl3q_u8(tbl_lo, idx_1)),
            ],
            [
                vreinterpretq_f32_u8(vqtbl3q_u8(tbl_lo, idx_2)),
                vreinterpretq_f32_u8(vqtbl3q_u8(tbl_hi, idx_0)),
            ],
            [
                vreinterpretq_f32_u8(vqtbl3q_u8(tbl_hi, idx_1)),
                vreinterpretq_f32_u8(vqtbl3q_u8(tbl_hi, idx_2)),
            ],
        ]
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn load_rgb_bytes(
        self,
        rgb: &[u8; 24],
    ) -> ([float32x4_t; 2], [float32x4_t; 2], [float32x4_t; 2]) {
        let lo16: uint8x16_t = crate::simd_storage::copy(rgb.first_chunk::<16>().unwrap());
        let hi8: uint8x8_t = crate::simd_storage::copy(rgb.last_chunk::<8>().unwrap());
        let tbl = uint8x16x2_t(lo16, vcombine_u8(hi8, vdup_n_u8(0)));
        let rg_idx: uint8x16_t =
            crate::simd_storage::cast([0u8, 3, 6, 9, 12, 15, 18, 21, 1, 4, 7, 10, 13, 16, 19, 22]);
        let b_idx: uint8x16_t = crate::simd_storage::cast([
            2u8, 5, 8, 11, 14, 17, 20, 23, 255, 255, 255, 255, 255, 255, 255, 255,
        ]);
        let rg8 = vqtbl2q_u8(tbl, rg_idx);
        let b8 = vqtbl2q_u8(tbl, b_idx);
        let r16 = vmovl_u8(vget_low_u8(rg8));
        let g16 = vmovl_high_u8(rg8);
        let b16 = vmovl_u8(vget_low_u8(b8));
        (
            [
                vcvtq_f32_u32(vmovl_u16(vget_low_u16(r16))),
                vcvtq_f32_u32(vmovl_high_u16(r16)),
            ],
            [
                vcvtq_f32_u32(vmovl_u16(vget_low_u16(g16))),
                vcvtq_f32_u32(vmovl_high_u16(g16)),
            ],
            [
                vcvtq_f32_u32(vmovl_u16(vget_low_u16(b16))),
                vcvtq_f32_u32(vmovl_high_u16(b16)),
            ],
        )
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn store_rgb_u8_into(
        self,
        r: [float32x4_t; 2],
        g: [float32x4_t; 2],
        b: [float32x4_t; 2],
        out: &mut [u8; 24],
    ) {
        let r8 = vqmovun_s16(vcombine_s16(
            vqmovn_s32(vcvtnq_s32_f32(r[0])),
            vqmovn_s32(vcvtnq_s32_f32(r[1])),
        ));
        let g8 = vqmovun_s16(vcombine_s16(
            vqmovn_s32(vcvtnq_s32_f32(g[0])),
            vqmovn_s32(vcvtnq_s32_f32(g[1])),
        ));
        let b8 = vqmovun_s16(vcombine_s16(
            vqmovn_s32(vcvtnq_s32_f32(b[0])),
            vqmovn_s32(vcvtnq_s32_f32(b[1])),
        ));
        let tbl = uint8x16x2_t(vcombine_u8(r8, g8), vcombine_u8(b8, b8));
        let idx_lo: uint8x16_t =
            crate::simd_storage::cast([0u8, 8, 16, 1, 9, 17, 2, 10, 18, 3, 11, 19, 4, 12, 20, 5]);
        let idx_hi: uint8x16_t = crate::simd_storage::cast([
            13u8, 21, 6, 14, 22, 7, 15, 23, 255, 255, 255, 255, 255, 255, 255, 255,
        ]);
        let lo16 = vqtbl2q_u8(tbl, idx_lo);
        let hi8 = vqtbl2q_u8(tbl, idx_hi);
        crate::simd_storage::store(lo16, out.first_chunk_mut::<16>().unwrap());
        crate::simd_storage::store(vget_low_u8(hi8), out.last_chunk_mut::<8>().unwrap());
    }
}

#[cfg(target_arch = "aarch64")]
impl F64x2Backend for NeonToken {
    type Repr = float64x2_t;

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn splat(self, v: f64) -> float64x2_t {
        vdupq_n_f64(v)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
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

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn add(self, a: float64x2_t, b: float64x2_t) -> float64x2_t {
        vaddq_f64(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn sub(self, a: float64x2_t, b: float64x2_t) -> float64x2_t {
        vsubq_f64(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn mul(self, a: float64x2_t, b: float64x2_t) -> float64x2_t {
        vmulq_f64(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn div(self, a: float64x2_t, b: float64x2_t) -> float64x2_t {
        vdivq_f64(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn neg(self, a: float64x2_t) -> float64x2_t {
        vnegq_f64(a)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn min(self, a: float64x2_t, b: float64x2_t) -> float64x2_t {
        vminnmq_f64(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn max(self, a: float64x2_t, b: float64x2_t) -> float64x2_t {
        vmaxnmq_f64(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn minimum(self, a: float64x2_t, b: float64x2_t) -> float64x2_t {
        vminq_f64(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn maximum(self, a: float64x2_t, b: float64x2_t) -> float64x2_t {
        vmaxq_f64(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn min_fast(self, a: float64x2_t, b: float64x2_t) -> float64x2_t {
        vminnmq_f64(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn max_fast(self, a: float64x2_t, b: float64x2_t) -> float64x2_t {
        vmaxnmq_f64(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn sqrt(self, a: float64x2_t) -> float64x2_t {
        vsqrtq_f64(a)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn abs(self, a: float64x2_t) -> float64x2_t {
        vabsq_f64(a)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn trunc(self, a: float64x2_t) -> float64x2_t {
        vrndq_f64(a)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn mul_neg_add_relaxed(self, a: float64x2_t, b: float64x2_t, c: float64x2_t) -> float64x2_t {
        vfmsq_f64(c, a, b)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn mul_neg_add_fused(self, a: float64x2_t, b: float64x2_t, c: float64x2_t) -> float64x2_t {
        vfmsq_f64(c, a, b)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn floor(self, a: float64x2_t) -> float64x2_t {
        vrndmq_f64(a)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn ceil(self, a: float64x2_t) -> float64x2_t {
        vrndpq_f64(a)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn round(self, a: float64x2_t) -> float64x2_t {
        vrndnq_f64(a)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn mul_add(self, a: float64x2_t, b: float64x2_t, c: float64x2_t) -> float64x2_t {
        vfmaq_f64(c, a, b)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn mul_sub(self, a: float64x2_t, b: float64x2_t, c: float64x2_t) -> float64x2_t {
        vfmaq_f64(vnegq_f64(c), a, b)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn mul_add_portable(self, a: float64x2_t, b: float64x2_t, c: float64x2_t) -> float64x2_t {
        vfmaq_f64(c, a, b)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn mul_sub_portable(self, a: float64x2_t, b: float64x2_t, c: float64x2_t) -> float64x2_t {
        vfmaq_f64(vnegq_f64(c), a, b)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_eq(self, a: float64x2_t, b: float64x2_t) -> float64x2_t {
        vreinterpretq_f64_u64(vceqq_f64(a, b))
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_ne(self, a: float64x2_t, b: float64x2_t) -> float64x2_t {
        vreinterpretq_f64_u64(veorq_u64(vceqq_f64(a, b), vdupq_n_u64(u64::MAX)))
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_lt(self, a: float64x2_t, b: float64x2_t) -> float64x2_t {
        vreinterpretq_f64_u64(vcltq_f64(a, b))
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_le(self, a: float64x2_t, b: float64x2_t) -> float64x2_t {
        vreinterpretq_f64_u64(vcleq_f64(a, b))
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_gt(self, a: float64x2_t, b: float64x2_t) -> float64x2_t {
        vreinterpretq_f64_u64(vcgtq_f64(a, b))
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_ge(self, a: float64x2_t, b: float64x2_t) -> float64x2_t {
        vreinterpretq_f64_u64(vcgeq_f64(a, b))
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn blend(self, mask: float64x2_t, if_true: float64x2_t, if_false: float64x2_t) -> float64x2_t {
        vbslq_f64(vreinterpretq_u64_f64(mask), if_true, if_false)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn reduce_add(self, a: float64x2_t) -> f64 {
        {
            let pair = vpaddq_f64(a, a);
            vgetq_lane_f64::<0>(pair)
        }
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn reduce_min(self, a: float64x2_t) -> f64 {
        {
            let pair = vpminnmq_f64(a, a);
            vgetq_lane_f64::<0>(pair)
        }
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn reduce_max(self, a: float64x2_t) -> f64 {
        {
            let pair = vpmaxnmq_f64(a, a);
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
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn rcp_approx(self, a: float64x2_t) -> float64x2_t {
        let y = vrecpeq_f64(a);
        vmulq_f64(vrecpsq_f64(a, y), y)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn rsqrt_approx(self, a: float64x2_t) -> float64x2_t {
        let y = vrsqrteq_f64(a);
        vmulq_f64(y, vrsqrtsq_f64(a, vmulq_f64(y, y)))
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn recip(self, a: float64x2_t) -> float64x2_t {
        vdivq_f64(vdupq_n_f64(1.0), a)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn rsqrt(self, a: float64x2_t) -> float64x2_t {
        vdivq_f64(vdupq_n_f64(1.0), vsqrtq_f64(a))
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn not(self, a: float64x2_t) -> float64x2_t {
        vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(a), vdupq_n_u64(u64::MAX)))
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitand(self, a: float64x2_t, b: float64x2_t) -> float64x2_t {
        vreinterpretq_f64_u64(vandq_u64(
            vreinterpretq_u64_f64(a),
            vreinterpretq_u64_f64(b),
        ))
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitor(self, a: float64x2_t, b: float64x2_t) -> float64x2_t {
        vreinterpretq_f64_u64(vorrq_u64(
            vreinterpretq_u64_f64(a),
            vreinterpretq_u64_f64(b),
        ))
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitxor(self, a: float64x2_t, b: float64x2_t) -> float64x2_t {
        vreinterpretq_f64_u64(veorq_u64(
            vreinterpretq_u64_f64(a),
            vreinterpretq_u64_f64(b),
        ))
    }
}

#[cfg(target_arch = "aarch64")]
impl I32x4Backend for NeonToken {
    type Repr = int32x4_t;

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn splat(self, v: i32) -> int32x4_t {
        vdupq_n_s32(v)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
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

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn add(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {
        vaddq_s32(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn sub(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {
        vsubq_s32(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn mul(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {
        vmulq_s32(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn neg(self, a: int32x4_t) -> int32x4_t {
        vnegq_s32(a)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn min(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {
        vminq_s32(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn max(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {
        vmaxq_s32(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn abs(self, a: int32x4_t) -> int32x4_t {
        vabsq_s32(a)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_eq(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {
        vreinterpretq_s32_u32(vceqq_s32(a, b))
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_ne(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {
        vreinterpretq_s32_u32(vmvnq_u32(vceqq_s32(a, b)))
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_lt(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {
        vreinterpretq_s32_u32(vcltq_s32(a, b))
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_le(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {
        vreinterpretq_s32_u32(vcleq_s32(a, b))
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_gt(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {
        vreinterpretq_s32_u32(vcgtq_s32(a, b))
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_ge(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {
        vreinterpretq_s32_u32(vcgeq_s32(a, b))
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn blend(self, mask: int32x4_t, if_true: int32x4_t, if_false: int32x4_t) -> int32x4_t {
        vbslq_s32(vreinterpretq_u32_s32(mask), if_true, if_false)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn reduce_add(self, a: int32x4_t) -> i32 {
        vaddvq_s32(a)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn not(self, a: int32x4_t) -> int32x4_t {
        vmvnq_s32(a)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitand(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {
        vandq_s32(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitor(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {
        vorrq_s32(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitxor(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {
        veorq_s32(a, b)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn shl_const<const N: i32>(self, a: int32x4_t) -> int32x4_t {
        vshlq_n_s32::<N>(a)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn shr_arithmetic_const<const N: i32>(self, a: int32x4_t) -> int32x4_t {
        const { assert!(N >= 0 && N <= 31) };
        vshlq_s32(a, vdupq_n_s32(-N))
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn shr_logical_const<const N: i32>(self, a: int32x4_t) -> int32x4_t {
        const { assert!(N >= 0 && N <= 31) };
        vreinterpretq_s32_u32(vshlq_u32(vreinterpretq_u32_s32(a), vdupq_n_s32(-N)))
    }

    // ====== Uniform variable shifts ======
    // Same vshlq lowering as the const forms; the count is clamped
    // because USHL/SSHL read only the low byte of each amount lane as
    // a signed value, so an unclamped 256 would wrap to a no-op
    // instead of the contracted zero.

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn shl_uniform(self, a: int32x4_t, count: u32) -> int32x4_t {
        vshlq_s32(a, vdupq_n_s32(count.min(32) as i32))
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn shr_logical_uniform(self, a: int32x4_t, count: u32) -> int32x4_t {
        vreinterpretq_s32_u32(vshlq_u32(
            vreinterpretq_u32_s32(a),
            vdupq_n_s32(-(count.min(32) as i32)),
        ))
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn shr_arithmetic_uniform(self, a: int32x4_t, count: u32) -> int32x4_t {
        // Clamping to 31 gives the contracted sign fill.
        vshlq_s32(a, vdupq_n_s32(-(count.min(31) as i32)))
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn all_true(self, a: int32x4_t) -> bool {
        // Every lane's sign bit set <=> every lane negative <=> the
        // signed maximum is negative. Measured 28% faster on M4 Pro
        // than the unsigned-min form this replaces.
        vmaxvq_s32(a) < 0
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn any_true(self, a: int32x4_t) -> bool {
        vminvq_s32(a) < 0
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitmask(self, a: int32x4_t) -> u32 {
        // One sign bit per word -> 4-bit mask. Shift each 0/1 lane
        // into position, then one horizontal add. The shift on the
        // u32 view must be LOGICAL — an arithmetic s32 shift
        // sign-extends to 0xFFFF_FFFF and corrupts the mask.
        let shifts: [i32; 4] = [0, 1, 2, 3];
        let shift_vec: int32x4_t = crate::simd_storage::copy(&shifts);
        vaddvq_u32(vshlq_u32(
            vshrq_n_u32::<31>(vreinterpretq_u32_s32(a)),
            shift_vec,
        ))
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn narrow_saturating_i32_to_i16(self, a: int32x4_t, b: int32x4_t) -> int16x8_t {
        vcombine_s16(vqmovn_s32(a), vqmovn_s32(b))
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn narrow_saturating_i32_to_u16(self, a: int32x4_t, b: int32x4_t) -> uint16x8_t {
        vcombine_u16(vqmovun_s32(a), vqmovun_s32(b))
    }
}

#[cfg(target_arch = "aarch64")]
impl U32x4Backend for NeonToken {
    type Repr = uint32x4_t;

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn splat(self, v: u32) -> uint32x4_t {
        vdupq_n_u32(v)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
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

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn add(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        vaddq_u32(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn sub(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        vsubq_u32(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn mul(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        vmulq_u32(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn min(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        vminq_u32(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn max(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        vmaxq_u32(a, b)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_eq(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        vceqq_u32(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_ne(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        vmvnq_u32(vceqq_u32(a, b))
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_lt(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        vcltq_u32(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_le(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        vcleq_u32(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_gt(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        vcgtq_u32(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_ge(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        vcgeq_u32(a, b)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn blend(self, mask: uint32x4_t, if_true: uint32x4_t, if_false: uint32x4_t) -> uint32x4_t {
        vbslq_u32(mask, if_true, if_false)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn reduce_add(self, a: uint32x4_t) -> u32 {
        vaddvq_u32(a)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn not(self, a: uint32x4_t) -> uint32x4_t {
        vmvnq_u32(a)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitand(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        vandq_u32(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitor(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        vorrq_u32(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitxor(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        veorq_u32(a, b)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn shl_const<const N: i32>(self, a: uint32x4_t) -> uint32x4_t {
        vshlq_n_u32::<N>(a)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn shr_logical_const<const N: i32>(self, a: uint32x4_t) -> uint32x4_t {
        const { assert!(N >= 0 && N <= 31) };
        vshlq_u32(a, vdupq_n_s32(-N))
    }

    // ====== Uniform variable shifts ======
    // Same vshlq lowering as the const forms; the count is clamped
    // because USHL reads only the low byte of each amount lane as a
    // signed value, so an unclamped 256 would wrap to a no-op instead
    // of the contracted zero.

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn shl_uniform(self, a: uint32x4_t, count: u32) -> uint32x4_t {
        vshlq_u32(a, vdupq_n_s32(count.min(32) as i32))
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn shr_logical_uniform(self, a: uint32x4_t, count: u32) -> uint32x4_t {
        vshlq_u32(a, vdupq_n_s32(-(count.min(32) as i32)))
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn all_true(self, a: uint32x4_t) -> bool {
        // Sign-bit contract on the signed reinterpretation. This lane
        // type previously used a third rule (all lanes == u32::MAX).
        vmaxvq_s32(vreinterpretq_s32_u32(a)) < 0
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn any_true(self, a: uint32x4_t) -> bool {
        vminvq_s32(vreinterpretq_s32_u32(a)) < 0
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitmask(self, a: uint32x4_t) -> u32 {
        // One sign bit per word -> 4-bit mask. Shift each 0/1 lane
        // into position, then one horizontal add. The shift on the
        // u32 view must be LOGICAL — an arithmetic s32 shift
        // sign-extends to 0xFFFF_FFFF and corrupts the mask.
        let shifts: [i32; 4] = [0, 1, 2, 3];
        let shift_vec: int32x4_t = crate::simd_storage::copy(&shifts);
        vaddvq_u32(vshlq_u32(vshrq_n_u32::<31>(a), shift_vec))
    }
}

#[cfg(target_arch = "aarch64")]
impl I64x2Backend for NeonToken {
    type Repr = int64x2_t;

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn splat(self, v: i64) -> int64x2_t {
        vdupq_n_s64(v)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
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

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn add(self, a: int64x2_t, b: int64x2_t) -> int64x2_t {
        vaddq_s64(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn sub(self, a: int64x2_t, b: int64x2_t) -> int64x2_t {
        vsubq_s64(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn neg(self, a: int64x2_t) -> int64x2_t {
        vnegq_s64(a)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn min(self, a: int64x2_t, b: int64x2_t) -> int64x2_t {
        // NEON lacks native i64 min; polyfill via compare+select
        {
            let mask = vcltq_s64(a, b);
            vbslq_s64(mask, a, b)
        }
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn max(self, a: int64x2_t, b: int64x2_t) -> int64x2_t {
        // NEON lacks native i64 max; polyfill via compare+select
        {
            let mask = vcgtq_s64(a, b);
            vbslq_s64(mask, a, b)
        }
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn abs(self, a: int64x2_t) -> int64x2_t {
        vabsq_s64(a)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_eq(self, a: int64x2_t, b: int64x2_t) -> int64x2_t {
        vreinterpretq_s64_u64(vceqq_s64(a, b))
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_ne(self, a: int64x2_t, b: int64x2_t) -> int64x2_t {
        let eq = vceqq_s64(a, b);
        // NOT via XOR with all-ones
        vreinterpretq_s64_u64(veorq_u64(eq, vdupq_n_u64(u64::MAX)))
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_lt(self, a: int64x2_t, b: int64x2_t) -> int64x2_t {
        vreinterpretq_s64_u64(vcltq_s64(a, b))
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_le(self, a: int64x2_t, b: int64x2_t) -> int64x2_t {
        vreinterpretq_s64_u64(vcleq_s64(a, b))
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_gt(self, a: int64x2_t, b: int64x2_t) -> int64x2_t {
        vreinterpretq_s64_u64(vcgtq_s64(a, b))
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_ge(self, a: int64x2_t, b: int64x2_t) -> int64x2_t {
        vreinterpretq_s64_u64(vcgeq_s64(a, b))
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn blend(self, mask: int64x2_t, if_true: int64x2_t, if_false: int64x2_t) -> int64x2_t {
        vbslq_s64(vreinterpretq_u64_s64(mask), if_true, if_false)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn reduce_add(self, a: int64x2_t) -> i64 {
        let sum = vpaddq_s64(a, a);
        vgetq_lane_s64::<0>(sum)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn not(self, a: int64x2_t) -> int64x2_t {
        let ones = vdupq_n_s64(-1i64);
        veorq_s64(a, ones)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitand(self, a: int64x2_t, b: int64x2_t) -> int64x2_t {
        vandq_s64(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitor(self, a: int64x2_t, b: int64x2_t) -> int64x2_t {
        vorrq_s64(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitxor(self, a: int64x2_t, b: int64x2_t) -> int64x2_t {
        veorq_s64(a, b)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn shl_const<const N: i32>(self, a: int64x2_t) -> int64x2_t {
        vshlq_n_s64::<N>(a)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn shr_arithmetic_const<const N: i32>(self, a: int64x2_t) -> int64x2_t {
        const { assert!(N >= 0 && N <= 63) };
        vshlq_s64(a, vdupq_n_s64((-N) as i64))
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn shr_logical_const<const N: i32>(self, a: int64x2_t) -> int64x2_t {
        const { assert!(N >= 0 && N <= 63) };
        vreinterpretq_s64_u64(vshlq_u64(
            vreinterpretq_u64_s64(a),
            vdupq_n_s64((-N) as i64),
        ))
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn all_true(self, a: int64x2_t) -> bool {
        // Sign-bit contract. NEON has no 64-bit horizontal reduction,
        // so combine the lanes and test one sign bit.
        let as_u64 = vreinterpretq_u64_s64(a);
        ((vgetq_lane_u64::<0>(as_u64) & vgetq_lane_u64::<1>(as_u64)) >> 63) != 0
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn any_true(self, a: int64x2_t) -> bool {
        let as_u64 = vreinterpretq_u64_s64(a);
        ((vgetq_lane_u64::<0>(as_u64) | vgetq_lane_u64::<1>(as_u64)) >> 63) != 0
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitmask(self, a: int64x2_t) -> u32 {
        let signs = vshrq_n_u64::<63>(vreinterpretq_u64_s64(a));
        ((vgetq_lane_u64::<0>(signs) & 1) | ((vgetq_lane_u64::<1>(signs) & 1) << 1)) as u32
    }
}

#[cfg(target_arch = "aarch64")]
impl I8x16Backend for NeonToken {
    type Repr = int8x16_t;

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn splat(self, v: i8) -> int8x16_t {
        vdupq_n_s8(v)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
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

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn add(self, a: int8x16_t, b: int8x16_t) -> int8x16_t {
        vaddq_s8(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn sub(self, a: int8x16_t, b: int8x16_t) -> int8x16_t {
        vsubq_s8(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn neg(self, a: int8x16_t) -> int8x16_t {
        vnegq_s8(a)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn min(self, a: int8x16_t, b: int8x16_t) -> int8x16_t {
        vminq_s8(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn max(self, a: int8x16_t, b: int8x16_t) -> int8x16_t {
        vmaxq_s8(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn abs(self, a: int8x16_t) -> int8x16_t {
        vabsq_s8(a)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_eq(self, a: int8x16_t, b: int8x16_t) -> int8x16_t {
        vreinterpretq_s8_u8(vceqq_s8(a, b))
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_ne(self, a: int8x16_t, b: int8x16_t) -> int8x16_t {
        vreinterpretq_s8_u8(vmvnq_u8(vceqq_s8(a, b)))
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_lt(self, a: int8x16_t, b: int8x16_t) -> int8x16_t {
        vreinterpretq_s8_u8(vcltq_s8(a, b))
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_le(self, a: int8x16_t, b: int8x16_t) -> int8x16_t {
        vreinterpretq_s8_u8(vcleq_s8(a, b))
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_gt(self, a: int8x16_t, b: int8x16_t) -> int8x16_t {
        vreinterpretq_s8_u8(vcgtq_s8(a, b))
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_ge(self, a: int8x16_t, b: int8x16_t) -> int8x16_t {
        vreinterpretq_s8_u8(vcgeq_s8(a, b))
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn blend(self, mask: int8x16_t, if_true: int8x16_t, if_false: int8x16_t) -> int8x16_t {
        vbslq_s8(vreinterpretq_u8_s8(mask), if_true, if_false)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn reduce_add(self, a: int8x16_t) -> i8 {
        vaddvq_s8(a)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn not(self, a: int8x16_t) -> int8x16_t {
        vmvnq_s8(a)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitand(self, a: int8x16_t, b: int8x16_t) -> int8x16_t {
        vandq_s8(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitor(self, a: int8x16_t, b: int8x16_t) -> int8x16_t {
        vorrq_s8(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitxor(self, a: int8x16_t, b: int8x16_t) -> int8x16_t {
        veorq_s8(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn shl_const<const N: i32>(self, a: int8x16_t) -> int8x16_t {
        vshlq_n_s8::<N>(a)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn shr_logical_const<const N: i32>(self, a: int8x16_t) -> int8x16_t {
        const { assert!(N >= 0 && N <= 7) };
        vreinterpretq_s8_u8(vshlq_u8(vreinterpretq_u8_s8(a), vdupq_n_s8((-N) as i8)))
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn shr_arithmetic_const<const N: i32>(self, a: int8x16_t) -> int8x16_t {
        const { assert!(N >= 0 && N <= 7) };
        vshlq_s8(a, vdupq_n_s8((-N) as i8))
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn saturating_add(self, a: int8x16_t, b: int8x16_t) -> int8x16_t {
        vqaddq_s8(a, b)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn saturating_sub(self, a: int8x16_t, b: int8x16_t) -> int8x16_t {
        vqsubq_s8(a, b)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn all_true(self, a: int8x16_t) -> bool {
        vmaxvq_s8(a) < 0
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn any_true(self, a: int8x16_t) -> bool {
        vminvq_s8(a) < 0
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitmask(self, a: int8x16_t) -> u32 {
        {
            let powers: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
            let pow_vec: uint8x16_t = crate::simd_storage::copy(&powers);
            let weighted = vandq_u8(vcltzq_s8(a), pow_vec);
            let s = vpadd_u8(vget_low_u8(weighted), vget_high_u8(weighted));
            let s = vpadd_u8(s, s);
            let s = vpadd_u8(s, s);
            u32::from(vget_lane_u16::<0>(vreinterpret_u16_u8(s)))
        }
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn widen_low_i8_to_i16(self, a: int8x16_t) -> int16x8_t {
        vmovl_s8(vget_low_s8(a))
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn widen_high_i8_to_i16(self, a: int8x16_t) -> int16x8_t {
        vmovl_high_s8(a)
    }
}

#[cfg(target_arch = "aarch64")]
impl U8x16Backend for NeonToken {
    type Repr = uint8x16_t;

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn splat(self, v: u8) -> uint8x16_t {
        vdupq_n_u8(v)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
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

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn add(self, a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        vaddq_u8(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn sub(self, a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        vsubq_u8(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn min(self, a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        vminq_u8(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn max(self, a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        vmaxq_u8(a, b)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_eq(self, a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        vceqq_u8(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_ne(self, a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        vmvnq_u8(vceqq_u8(a, b))
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_lt(self, a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        vcltq_u8(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_le(self, a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        vcleq_u8(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_gt(self, a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        vcgtq_u8(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_ge(self, a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        vcgeq_u8(a, b)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn blend(self, mask: uint8x16_t, if_true: uint8x16_t, if_false: uint8x16_t) -> uint8x16_t {
        vbslq_u8(mask, if_true, if_false)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn reduce_add(self, a: uint8x16_t) -> u8 {
        vaddvq_u8(a)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn not(self, a: uint8x16_t) -> uint8x16_t {
        vmvnq_u8(a)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitand(self, a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        vandq_u8(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitor(self, a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        vorrq_u8(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitxor(self, a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        veorq_u8(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn shl_const<const N: i32>(self, a: uint8x16_t) -> uint8x16_t {
        vshlq_n_u8::<N>(a)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn shr_logical_const<const N: i32>(self, a: uint8x16_t) -> uint8x16_t {
        const { assert!(N >= 0 && N <= 7) };
        vshlq_u8(a, vdupq_n_s8((-N) as i8))
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn saturating_add(self, a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        vqaddq_u8(a, b)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn saturating_sub(self, a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        vqsubq_u8(a, b)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn all_true(self, a: uint8x16_t) -> bool {
        vmaxvq_s8(vreinterpretq_s8_u8(a)) < 0
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn any_true(self, a: uint8x16_t) -> bool {
        vminvq_s8(vreinterpretq_s8_u8(a)) < 0
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitmask(self, a: uint8x16_t) -> u32 {
        {
            let powers: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
            let pow_vec: uint8x16_t = crate::simd_storage::copy(&powers);
            let weighted = vandq_u8(vcltzq_s8(vreinterpretq_s8_u8(a)), pow_vec);
            let s = vpadd_u8(vget_low_u8(weighted), vget_high_u8(weighted));
            let s = vpadd_u8(s, s);
            let s = vpadd_u8(s, s);
            u32::from(vget_lane_u16::<0>(vreinterpret_u16_u8(s)))
        }
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn widen_low_u8_to_u16(self, a: uint8x16_t) -> uint16x8_t {
        vmovl_u8(vget_low_u8(a))
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn widen_high_u8_to_u16(self, a: uint8x16_t) -> uint16x8_t {
        vmovl_high_u8(a)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn abs_diff(self, a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        vabdq_u8(a, b)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn reduce_add_u32(self, a: uint8x16_t) -> u32 {
        vaddlvq_u8(a) as u32
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn sum_abs_diff(self, a: uint8x16_t, b: uint8x16_t) -> u32 {
        vaddlvq_u8(vabdq_u8(a, b)) as u32
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn pairwise_widen_add(self, a: uint8x16_t) -> uint16x8_t {
        vpaddlq_u8(a)
    }
}

#[cfg(target_arch = "aarch64")]
impl I16x8Backend for NeonToken {
    type Repr = int16x8_t;

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn splat(self, v: i16) -> int16x8_t {
        vdupq_n_s16(v)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
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

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn add(self, a: int16x8_t, b: int16x8_t) -> int16x8_t {
        vaddq_s16(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn sub(self, a: int16x8_t, b: int16x8_t) -> int16x8_t {
        vsubq_s16(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn mul(self, a: int16x8_t, b: int16x8_t) -> int16x8_t {
        vmulq_s16(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn neg(self, a: int16x8_t) -> int16x8_t {
        vnegq_s16(a)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn min(self, a: int16x8_t, b: int16x8_t) -> int16x8_t {
        vminq_s16(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn max(self, a: int16x8_t, b: int16x8_t) -> int16x8_t {
        vmaxq_s16(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn abs(self, a: int16x8_t) -> int16x8_t {
        vabsq_s16(a)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_eq(self, a: int16x8_t, b: int16x8_t) -> int16x8_t {
        vreinterpretq_s16_u16(vceqq_s16(a, b))
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_ne(self, a: int16x8_t, b: int16x8_t) -> int16x8_t {
        vreinterpretq_s16_u16(vmvnq_u16(vceqq_s16(a, b)))
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_lt(self, a: int16x8_t, b: int16x8_t) -> int16x8_t {
        vreinterpretq_s16_u16(vcltq_s16(a, b))
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_le(self, a: int16x8_t, b: int16x8_t) -> int16x8_t {
        vreinterpretq_s16_u16(vcleq_s16(a, b))
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_gt(self, a: int16x8_t, b: int16x8_t) -> int16x8_t {
        vreinterpretq_s16_u16(vcgtq_s16(a, b))
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_ge(self, a: int16x8_t, b: int16x8_t) -> int16x8_t {
        vreinterpretq_s16_u16(vcgeq_s16(a, b))
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn blend(self, mask: int16x8_t, if_true: int16x8_t, if_false: int16x8_t) -> int16x8_t {
        vbslq_s16(vreinterpretq_u16_s16(mask), if_true, if_false)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn reduce_add(self, a: int16x8_t) -> i16 {
        vaddvq_s16(a)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn not(self, a: int16x8_t) -> int16x8_t {
        vmvnq_s16(a)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitand(self, a: int16x8_t, b: int16x8_t) -> int16x8_t {
        vandq_s16(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitor(self, a: int16x8_t, b: int16x8_t) -> int16x8_t {
        vorrq_s16(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitxor(self, a: int16x8_t, b: int16x8_t) -> int16x8_t {
        veorq_s16(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn shl_const<const N: i32>(self, a: int16x8_t) -> int16x8_t {
        vshlq_n_s16::<N>(a)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn shr_logical_const<const N: i32>(self, a: int16x8_t) -> int16x8_t {
        const { assert!(N >= 0 && N <= 15) };
        vreinterpretq_s16_u16(vshlq_u16(
            vreinterpretq_u16_s16(a),
            vdupq_n_s16((-N) as i16),
        ))
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn shr_arithmetic_const<const N: i32>(self, a: int16x8_t) -> int16x8_t {
        const { assert!(N >= 0 && N <= 15) };
        vshlq_s16(a, vdupq_n_s16((-N) as i16))
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn shl_uniform(self, a: int16x8_t, count: u32) -> int16x8_t {
        vshlq_s16(a, vdupq_n_s16(count.min(16) as i16))
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn shr_logical_uniform(self, a: int16x8_t, count: u32) -> int16x8_t {
        vreinterpretq_s16_u16(vshlq_u16(
            vreinterpretq_u16_s16(a),
            vdupq_n_s16(-(count.min(16) as i16)),
        ))
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn shr_arithmetic_uniform(self, a: int16x8_t, count: u32) -> int16x8_t {
        // Clamping to lane_bits - 1 gives the contracted sign fill.
        vshlq_s16(a, vdupq_n_s16(-(count.min(15) as i16)))
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn saturating_add(self, a: int16x8_t, b: int16x8_t) -> int16x8_t {
        vqaddq_s16(a, b)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn saturating_sub(self, a: int16x8_t, b: int16x8_t) -> int16x8_t {
        vqsubq_s16(a, b)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn all_true(self, a: int16x8_t) -> bool {
        vmaxvq_s16(a) < 0
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn any_true(self, a: int16x8_t) -> bool {
        vminvq_s16(a) < 0
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitmask(self, a: int16x8_t) -> u32 {
        {
            let powers: [u16; 8] = [1, 2, 4, 8, 16, 32, 64, 128];
            let pow_vec: uint16x8_t = crate::simd_storage::copy(&powers);
            u32::from(vaddvq_u16(vandq_u16(vcltzq_s16(a), pow_vec)))
        }
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn widen_low_i16_to_i32(self, a: int16x8_t) -> int32x4_t {
        vmovl_s16(vget_low_s16(a))
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn widen_high_i16_to_i32(self, a: int16x8_t) -> int32x4_t {
        vmovl_high_s16(a)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn narrow_saturating_i16_to_i8(self, a: int16x8_t, b: int16x8_t) -> int8x16_t {
        vcombine_s8(vqmovn_s16(a), vqmovn_s16(b))
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn narrow_saturating_i16_to_u8(self, a: int16x8_t, b: int16x8_t) -> uint8x16_t {
        vcombine_u8(vqmovun_s16(a), vqmovun_s16(b))
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn madd_adjacent(self, a: int16x8_t, b: int16x8_t) -> int32x4_t {
        vpaddq_s32(
            vmull_s16(vget_low_s16(a), vget_low_s16(b)),
            vmull_high_s16(a, b),
        )
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn abs_diff(self, a: int16x8_t, b: int16x8_t) -> uint16x8_t {
        vreinterpretq_u16_s16(vabdq_s16(a, b))
    }
}

#[cfg(target_arch = "aarch64")]
impl U16x8Backend for NeonToken {
    type Repr = uint16x8_t;

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn splat(self, v: u16) -> uint16x8_t {
        vdupq_n_u16(v)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
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

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn add(self, a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        vaddq_u16(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn sub(self, a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        vsubq_u16(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn mul(self, a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        vmulq_u16(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn min(self, a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        vminq_u16(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn max(self, a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        vmaxq_u16(a, b)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_eq(self, a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        vceqq_u16(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_ne(self, a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        vmvnq_u16(vceqq_u16(a, b))
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_lt(self, a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        vcltq_u16(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_le(self, a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        vcleq_u16(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_gt(self, a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        vcgtq_u16(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_ge(self, a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        vcgeq_u16(a, b)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn blend(self, mask: uint16x8_t, if_true: uint16x8_t, if_false: uint16x8_t) -> uint16x8_t {
        vbslq_u16(mask, if_true, if_false)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn reduce_add(self, a: uint16x8_t) -> u16 {
        vaddvq_u16(a)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn not(self, a: uint16x8_t) -> uint16x8_t {
        vmvnq_u16(a)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitand(self, a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        vandq_u16(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitor(self, a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        vorrq_u16(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitxor(self, a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        veorq_u16(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn shl_const<const N: i32>(self, a: uint16x8_t) -> uint16x8_t {
        vshlq_n_u16::<N>(a)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn shr_logical_const<const N: i32>(self, a: uint16x8_t) -> uint16x8_t {
        const { assert!(N >= 0 && N <= 15) };
        vshlq_u16(a, vdupq_n_s16((-N) as i16))
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn shl_uniform(self, a: uint16x8_t, count: u32) -> uint16x8_t {
        vshlq_u16(a, vdupq_n_s16(count.min(16) as i16))
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn shr_logical_uniform(self, a: uint16x8_t, count: u32) -> uint16x8_t {
        vshlq_u16(a, vdupq_n_s16(-(count.min(16) as i16)))
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn saturating_add(self, a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        vqaddq_u16(a, b)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn saturating_sub(self, a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        vqsubq_u16(a, b)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn all_true(self, a: uint16x8_t) -> bool {
        vmaxvq_s16(vreinterpretq_s16_u16(a)) < 0
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn any_true(self, a: uint16x8_t) -> bool {
        vminvq_s16(vreinterpretq_s16_u16(a)) < 0
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitmask(self, a: uint16x8_t) -> u32 {
        {
            let powers: [u16; 8] = [1, 2, 4, 8, 16, 32, 64, 128];
            let pow_vec: uint16x8_t = crate::simd_storage::copy(&powers);
            u32::from(vaddvq_u16(vandq_u16(
                vcltzq_s16(vreinterpretq_s16_u16(a)),
                pow_vec,
            )))
        }
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn widen_low_u16_to_u32(self, a: uint16x8_t) -> uint32x4_t {
        vmovl_u16(vget_low_u16(a))
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn widen_high_u16_to_u32(self, a: uint16x8_t) -> uint32x4_t {
        vmovl_high_u16(a)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn pairwise_widen_add(self, a: uint16x8_t) -> uint32x4_t {
        vpaddlq_u16(a)
    }
}

#[cfg(target_arch = "aarch64")]
impl U64x2Backend for NeonToken {
    type Repr = uint64x2_t;

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn splat(self, v: u64) -> uint64x2_t {
        vdupq_n_u64(v)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
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

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn add(self, a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        vaddq_u64(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn sub(self, a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        vsubq_u64(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn min(self, a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        vbslq_u64(vcltq_u64(a, b), a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn max(self, a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        vbslq_u64(vcgtq_u64(a, b), a, b)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_eq(self, a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        vceqq_u64(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_ne(self, a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        veorq_u64(vceqq_u64(a, b), vdupq_n_u64(u64::MAX))
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_lt(self, a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        vcltq_u64(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_le(self, a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        vcleq_u64(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_gt(self, a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        vcgtq_u64(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn simd_ge(self, a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        vcgeq_u64(a, b)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn blend(self, mask: uint64x2_t, if_true: uint64x2_t, if_false: uint64x2_t) -> uint64x2_t {
        vbslq_u64(mask, if_true, if_false)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn reduce_add(self, a: uint64x2_t) -> u64 {
        vaddvq_u64(a)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn not(self, a: uint64x2_t) -> uint64x2_t {
        veorq_u64(a, vdupq_n_u64(u64::MAX))
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitand(self, a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        vandq_u64(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitor(self, a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        vorrq_u64(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitxor(self, a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        veorq_u64(a, b)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn shl_const<const N: i32>(self, a: uint64x2_t) -> uint64x2_t {
        vshlq_n_u64::<N>(a)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn shr_logical_const<const N: i32>(self, a: uint64x2_t) -> uint64x2_t {
        const { assert!(N >= 0 && N <= 63) };
        vshlq_u64(a, vdupq_n_s64((-N) as i64))
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn all_true(self, a: uint64x2_t) -> bool {
        // NEON has no 64-bit horizontal reduction; AND the lanes and
        // test the sign bit once.
        ((vgetq_lane_u64::<0>(a) & vgetq_lane_u64::<1>(a)) >> 63) != 0
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn any_true(self, a: uint64x2_t) -> bool {
        ((vgetq_lane_u64::<0>(a) | vgetq_lane_u64::<1>(a)) >> 63) != 0
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitmask(self, a: uint64x2_t) -> u32 {
        let shift = vshrq_n_u64::<63>(a);
        let lane0 = vgetq_lane_u64::<0>(shift) as u32;
        let lane1 = vgetq_lane_u64::<1>(shift) as u32;
        lane0 | (lane1 << 1)
    }
}

#[cfg(target_arch = "aarch64")]
impl F32x4Convert for NeonToken {
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitcast_f32_to_i32(self, a: float32x4_t) -> int32x4_t {
        vreinterpretq_s32_f32(a)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitcast_i32_to_f32(self, a: int32x4_t) -> float32x4_t {
        vreinterpretq_f32_s32(a)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn convert_f32_to_i32(self, a: float32x4_t) -> int32x4_t {
        vcvtq_s32_f32(a)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn convert_f32_to_i32_round(self, a: float32x4_t) -> int32x4_t {
        vcvtnq_s32_f32(a)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn convert_i32_to_f32(self, a: int32x4_t) -> float32x4_t {
        vcvtq_f32_s32(a)
    }
}

#[cfg(target_arch = "aarch64")]
impl U32x4Bitcast for NeonToken {
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitcast_u32_to_i32(self, a: uint32x4_t) -> int32x4_t {
        vreinterpretq_s32_u32(a)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitcast_i32_to_u32(self, a: int32x4_t) -> uint32x4_t {
        vreinterpretq_u32_s32(a)
    }
}

#[cfg(target_arch = "aarch64")]
impl I64x2Bitcast for NeonToken {
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitcast_i64_to_f64(self, a: int64x2_t) -> float64x2_t {
        vreinterpretq_f64_s64(a)
    }

    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitcast_f64_to_i64(self, a: float64x2_t) -> int64x2_t {
        vreinterpretq_s64_f64(a)
    }
}

#[cfg(target_arch = "aarch64")]
impl I8x16Bitcast for NeonToken {
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitcast_i8_to_u8(self, a: int8x16_t) -> uint8x16_t {
        vreinterpretq_u8_s8(a)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitcast_u8_to_i8(self, a: uint8x16_t) -> int8x16_t {
        vreinterpretq_s8_u8(a)
    }
}

#[cfg(target_arch = "aarch64")]
impl I16x8Bitcast for NeonToken {
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitcast_i16_to_u16(self, a: int16x8_t) -> uint16x8_t {
        vreinterpretq_u16_s16(a)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitcast_u16_to_i16(self, a: uint16x8_t) -> int16x8_t {
        vreinterpretq_s16_u16(a)
    }
}

#[cfg(target_arch = "aarch64")]
impl U64x2Bitcast for NeonToken {
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitcast_u64_to_i64(self, a: uint64x2_t) -> int64x2_t {
        vreinterpretq_s64_u64(a)
    }
    #[attune(wrap, suppress_const_test, _self = NeonToken)]
    fn bitcast_i64_to_u64(self, a: int64x2_t) -> uint64x2_t {
        vreinterpretq_u64_s64(a)
    }
}

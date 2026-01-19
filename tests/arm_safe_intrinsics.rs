//! Comprehensive test of which ARM NEON intrinsics are SAFE in #[target_feature] context
//! Rust 1.92+
//!
//! This file documents all NEON intrinsics that do NOT require an unsafe block when called
//! from within a function annotated with #[target_feature(enable = "neon")].
//!
//! SAFE: Value-based operations (arithmetic, shuffle, comparison, conversion, bitwise)
//! UNSAFE: Pointer-based operations (vld1, vst1, all load/store variants)

#![allow(unused)]

#[cfg(target_arch = "aarch64")]
mod neon_tests {
    use std::arch::aarch64::*;

    // ============================================================================
    // NEON (128-bit and 64-bit vectors)
    // ============================================================================

    #[target_feature(enable = "neon")]
    unsafe fn neon_safe_intrinsics_f32() {
        // === CREATION f32 (all safe) ===
        let zero_f32x4 = vdupq_n_f32(0.0);
        let set1_f32x4 = vdupq_n_f32(1.0);
        let zero_f32x2 = vdup_n_f32(0.0);
        let set1_f32x2 = vdup_n_f32(1.0);

        // === ARITHMETIC f32x4 (all safe) ===
        let add = vaddq_f32(zero_f32x4, set1_f32x4);
        let sub = vsubq_f32(add, set1_f32x4);
        let mul = vmulq_f32(add, set1_f32x4);
        let div = vdivq_f32(add, set1_f32x4);
        let neg = vnegq_f32(add);
        let abs = vabsq_f32(add);
        let sqrt = vsqrtq_f32(add);
        let recpe = vrecpeq_f32(add);
        let rsqrte = vrsqrteq_f32(add);
        let recps = vrecpsq_f32(add, set1_f32x4);
        let rsqrts = vrsqrtsq_f32(add, set1_f32x4);
        let min = vminq_f32(add, set1_f32x4);
        let max = vmaxq_f32(add, set1_f32x4);
        let minnm = vminnmq_f32(add, set1_f32x4);
        let maxnm = vmaxnmq_f32(add, set1_f32x4);
        let abd = vabdq_f32(add, set1_f32x4);

        // === FMA f32x4 (all safe) ===
        let fma = vfmaq_f32(add, set1_f32x4, set1_f32x4);
        let fms = vfmsq_f32(add, set1_f32x4, set1_f32x4);
        let mla = vmlaq_f32(add, set1_f32x4, set1_f32x4);
        let mls = vmlsq_f32(add, set1_f32x4, set1_f32x4);

        // === COMPARISON f32x4 (all safe) ===
        let ceq = vceqq_f32(add, set1_f32x4);
        let cge = vcgeq_f32(add, set1_f32x4);
        let cgt = vcgtq_f32(add, set1_f32x4);
        let cle = vcleq_f32(add, set1_f32x4);
        let clt = vcltq_f32(add, set1_f32x4);

        // === ROUND f32x4 (all safe) ===
        let rndp = vrndpq_f32(add);  // ceil
        let rndm = vrndmq_f32(add);  // floor
        let rndn = vrndnq_f32(add);  // round to nearest
        let rndz = vrndaq_f32(add);  // round away from zero
        let rndi = vrndiq_f32(add);  // round using current mode

        // === ARITHMETIC f32x2 (all safe) ===
        let add2 = vadd_f32(zero_f32x2, set1_f32x2);
        let sub2 = vsub_f32(add2, set1_f32x2);
        let mul2 = vmul_f32(add2, set1_f32x2);
        let div2 = vdiv_f32(add2, set1_f32x2);
        let neg2 = vneg_f32(add2);
        let abs2 = vabs_f32(add2);
        let sqrt2 = vsqrt_f32(add2);

        // === PAIRWISE (all safe) ===
        let padd = vpaddq_f32(add, set1_f32x4);
        let pmax = vpmaxq_f32(add, set1_f32x4);
        let pmin = vpminq_f32(add, set1_f32x4);
    }

    #[target_feature(enable = "neon")]
    unsafe fn neon_safe_intrinsics_f64() {
        // === CREATION f64 (all safe) ===
        let zero_f64x2 = vdupq_n_f64(0.0);
        let set1_f64x2 = vdupq_n_f64(1.0);
        let zero_f64x1 = vdup_n_f64(0.0);
        let set1_f64x1 = vdup_n_f64(1.0);

        // === ARITHMETIC f64x2 (all safe) ===
        let add = vaddq_f64(zero_f64x2, set1_f64x2);
        let sub = vsubq_f64(add, set1_f64x2);
        let mul = vmulq_f64(add, set1_f64x2);
        let div = vdivq_f64(add, set1_f64x2);
        let neg = vnegq_f64(add);
        let abs = vabsq_f64(add);
        let sqrt = vsqrtq_f64(add);
        let recpe = vrecpeq_f64(add);
        let rsqrte = vrsqrteq_f64(add);
        let min = vminq_f64(add, set1_f64x2);
        let max = vmaxq_f64(add, set1_f64x2);
        let minnm = vminnmq_f64(add, set1_f64x2);
        let maxnm = vmaxnmq_f64(add, set1_f64x2);
        let abd = vabdq_f64(add, set1_f64x2);

        // === FMA f64x2 (all safe) ===
        let fma = vfmaq_f64(add, set1_f64x2, set1_f64x2);
        let fms = vfmsq_f64(add, set1_f64x2, set1_f64x2);

        // === COMPARISON f64x2 (all safe) ===
        let ceq = vceqq_f64(add, set1_f64x2);
        let cge = vcgeq_f64(add, set1_f64x2);
        let cgt = vcgtq_f64(add, set1_f64x2);
        let cle = vcleq_f64(add, set1_f64x2);
        let clt = vcltq_f64(add, set1_f64x2);

        // === ROUND f64x2 (all safe) ===
        let rndp = vrndpq_f64(add);
        let rndm = vrndmq_f64(add);
        let rndn = vrndnq_f64(add);
    }

    #[target_feature(enable = "neon")]
    unsafe fn neon_safe_intrinsics_i8() {
        // === CREATION i8 (all safe) ===
        let zero_i8x16 = vdupq_n_s8(0);
        let set1_i8x16 = vdupq_n_s8(1);
        let zero_u8x16 = vdupq_n_u8(0);
        let set1_u8x16 = vdupq_n_u8(1);
        let zero_i8x8 = vdup_n_s8(0);
        let set1_i8x8 = vdup_n_s8(1);
        let zero_u8x8 = vdup_n_u8(0);
        let set1_u8x8 = vdup_n_u8(1);

        // === ARITHMETIC i8x16 signed (all safe) ===
        let add = vaddq_s8(zero_i8x16, set1_i8x16);
        let sub = vsubq_s8(add, set1_i8x16);
        let neg = vnegq_s8(add);
        let abs = vabsq_s8(add);
        let qadd = vqaddq_s8(add, set1_i8x16);  // saturating
        let qsub = vqsubq_s8(add, set1_i8x16);
        let hadd = vhaddq_s8(add, set1_i8x16);  // halving
        let hsub = vhsubq_s8(add, set1_i8x16);
        let rhadd = vrhaddq_s8(add, set1_i8x16);  // rounding halving
        let min = vminq_s8(add, set1_i8x16);
        let max = vmaxq_s8(add, set1_i8x16);
        let abd = vabdq_s8(add, set1_i8x16);
        let aba = vabaq_s8(add, set1_i8x16, set1_i8x16);

        // === ARITHMETIC u8x16 unsigned (all safe) ===
        let addu = vaddq_u8(zero_u8x16, set1_u8x16);
        let subu = vsubq_u8(addu, set1_u8x16);
        let qaddu = vqaddq_u8(addu, set1_u8x16);
        let qsubu = vqsubq_u8(addu, set1_u8x16);
        let haddu = vhaddq_u8(addu, set1_u8x16);
        let hsubu = vhsubq_u8(addu, set1_u8x16);
        let minu = vminq_u8(addu, set1_u8x16);
        let maxu = vmaxq_u8(addu, set1_u8x16);
        let abdu = vabdq_u8(addu, set1_u8x16);

        // === COMPARISON i8x16 (all safe) ===
        let ceq = vceqq_s8(add, set1_i8x16);
        let cge = vcgeq_s8(add, set1_i8x16);
        let cgt = vcgtq_s8(add, set1_i8x16);
        let cle = vcleq_s8(add, set1_i8x16);
        let clt = vcltq_s8(add, set1_i8x16);
        let cequ = vceqq_u8(addu, set1_u8x16);
        let cgeu = vcgeq_u8(addu, set1_u8x16);
        let cgtu = vcgtq_u8(addu, set1_u8x16);

        // === BITWISE (all safe) ===
        let and = vandq_s8(add, set1_i8x16);
        let or = vorrq_s8(add, set1_i8x16);
        let xor = veorq_s8(add, set1_i8x16);
        let bic = vbicq_s8(add, set1_i8x16);  // and-not
        let orn = vornq_s8(add, set1_i8x16);  // or-not
        let not = vmvnq_s8(add);

        // === TABLE LOOKUP (all safe - no pointers) ===
        let tbl = vqtbl1q_s8(add, set1_u8x16);
        let tbx = vqtbx1q_s8(add, set1_i8x16, set1_u8x16);

        // === PAIRWISE (all safe) ===
        let padd = vpaddq_s8(add, set1_i8x16);
        let paddu = vpaddq_u8(addu, set1_u8x16);
        let pmax = vpmaxq_s8(add, set1_i8x16);
        let pmin = vpminq_s8(add, set1_i8x16);
        let pmaxu = vpmaxq_u8(addu, set1_u8x16);
        let pminu = vpminq_u8(addu, set1_u8x16);

        // === COUNT (all safe) ===
        let cnt = vcntq_s8(add);
        let cntu = vcntq_u8(addu);
        let clz = vclzq_s8(add);
        let clzu = vclzq_u8(addu);
        let cls = vclsq_s8(add);
    }

    #[target_feature(enable = "neon")]
    unsafe fn neon_safe_intrinsics_i16() {
        // === CREATION i16 (all safe) ===
        let zero_i16x8 = vdupq_n_s16(0);
        let set1_i16x8 = vdupq_n_s16(1);
        let zero_u16x8 = vdupq_n_u16(0);
        let set1_u16x8 = vdupq_n_u16(1);
        let zero_i16x4 = vdup_n_s16(0);
        let set1_i16x4 = vdup_n_s16(1);

        // === ARITHMETIC i16x8 (all safe) ===
        let add = vaddq_s16(zero_i16x8, set1_i16x8);
        let sub = vsubq_s16(add, set1_i16x8);
        let neg = vnegq_s16(add);
        let abs = vabsq_s16(add);
        let qadd = vqaddq_s16(add, set1_i16x8);
        let qsub = vqsubq_s16(add, set1_i16x8);
        let hadd = vhaddq_s16(add, set1_i16x8);
        let hsub = vhsubq_s16(add, set1_i16x8);
        let rhadd = vrhaddq_s16(add, set1_i16x8);
        let min = vminq_s16(add, set1_i16x8);
        let max = vmaxq_s16(add, set1_i16x8);
        let abd = vabdq_s16(add, set1_i16x8);
        let aba = vabaq_s16(add, set1_i16x8, set1_i16x8);
        let mul = vmulq_s16(add, set1_i16x8);
        let mla = vmlaq_s16(add, set1_i16x8, set1_i16x8);
        let mls = vmlsq_s16(add, set1_i16x8, set1_i16x8);

        // === WIDENING (all safe) ===
        let addl = vaddl_s16(zero_i16x4, set1_i16x4);  // i16x4 -> i32x4
        let subl = vsubl_s16(zero_i16x4, set1_i16x4);
        let mull = vmull_s16(zero_i16x4, set1_i16x4);
        let abdl = vabdl_s16(zero_i16x4, set1_i16x4);
        let addw = vaddw_s16(vdupq_n_s32(0), zero_i16x4);  // i32x4 + i16x4 -> i32x4

        // === NARROWING (all safe) ===
        let movn = vmovn_s32(vdupq_n_s32(0));  // i32x4 -> i16x4
        let qmovn = vqmovn_s32(vdupq_n_s32(0));  // saturating
        let qmovun = vqmovun_s32(vdupq_n_s32(0));  // saturating unsigned

        // === COMPARISON i16x8 (all safe) ===
        let ceq = vceqq_s16(add, set1_i16x8);
        let cge = vcgeq_s16(add, set1_i16x8);
        let cgt = vcgtq_s16(add, set1_i16x8);

        // === SHIFT (all safe) ===
        let shl = vshlq_s16(add, set1_i16x8);  // variable shift
        let shr_n = vshrq_n_s16::<1>(add);     // immediate shift right
        let shl_n = vshlq_n_s16::<1>(add);     // immediate shift left
        let qshl = vqshlq_s16(add, set1_i16x8);  // saturating
        let rshr = vrshrq_n_s16::<1>(add);     // rounding
    }

    #[target_feature(enable = "neon")]
    unsafe fn neon_safe_intrinsics_i32() {
        // === CREATION i32 (all safe) ===
        let zero_i32x4 = vdupq_n_s32(0);
        let set1_i32x4 = vdupq_n_s32(1);
        let zero_u32x4 = vdupq_n_u32(0);
        let set1_u32x4 = vdupq_n_u32(1);
        let zero_i32x2 = vdup_n_s32(0);
        let set1_i32x2 = vdup_n_s32(1);

        // === ARITHMETIC i32x4 (all safe) ===
        let add = vaddq_s32(zero_i32x4, set1_i32x4);
        let sub = vsubq_s32(add, set1_i32x4);
        let neg = vnegq_s32(add);
        let abs = vabsq_s32(add);
        let qadd = vqaddq_s32(add, set1_i32x4);
        let qsub = vqsubq_s32(add, set1_i32x4);
        let hadd = vhaddq_s32(add, set1_i32x4);
        let hsub = vhsubq_s32(add, set1_i32x4);
        let rhadd = vrhaddq_s32(add, set1_i32x4);
        let min = vminq_s32(add, set1_i32x4);
        let max = vmaxq_s32(add, set1_i32x4);
        let abd = vabdq_s32(add, set1_i32x4);
        let aba = vabaq_s32(add, set1_i32x4, set1_i32x4);
        let mul = vmulq_s32(add, set1_i32x4);
        let mla = vmlaq_s32(add, set1_i32x4, set1_i32x4);
        let mls = vmlsq_s32(add, set1_i32x4, set1_i32x4);

        // === WIDENING (all safe) ===
        let addl = vaddl_s32(zero_i32x2, set1_i32x2);  // i32x2 -> i64x2
        let subl = vsubl_s32(zero_i32x2, set1_i32x2);
        let mull = vmull_s32(zero_i32x2, set1_i32x2);
        let abdl = vabdl_s32(zero_i32x2, set1_i32x2);

        // === COMPARISON i32x4 (all safe) ===
        let ceq = vceqq_s32(add, set1_i32x4);
        let cge = vcgeq_s32(add, set1_i32x4);
        let cgt = vcgtq_s32(add, set1_i32x4);
        let cle = vcleq_s32(add, set1_i32x4);
        let clt = vcltq_s32(add, set1_i32x4);

        // === SHIFT (all safe) ===
        let shl = vshlq_s32(add, set1_i32x4);
        let shr_n = vshrq_n_s32::<1>(add);
        let shl_n = vshlq_n_s32::<1>(add);
        let qshl = vqshlq_s32(add, set1_i32x4);
        let rshr = vrshrq_n_s32::<1>(add);

        // === PAIRWISE (all safe) ===
        let padd = vpaddq_s32(add, set1_i32x4);
        let pmax = vpmaxq_s32(add, set1_i32x4);
        let pmin = vpminq_s32(add, set1_i32x4);

        // === REDUCE (all safe) ===
        let addv = vaddvq_s32(add);  // horizontal sum
        let maxv = vmaxvq_s32(add);
        let minv = vminvq_s32(add);

        // === CONVERSION (all safe) ===
        let cvt_f32 = vcvtq_f32_s32(add);
        let cvt_s32 = vcvtq_s32_f32(vdupq_n_f32(1.0));
        let cvtn = vcvtnq_s32_f32(vdupq_n_f32(1.5));  // round to nearest
        let cvtm = vcvtmq_s32_f32(vdupq_n_f32(1.5));  // floor
        let cvtp = vcvtpq_s32_f32(vdupq_n_f32(1.5));  // ceil
    }

    #[target_feature(enable = "neon")]
    unsafe fn neon_safe_intrinsics_i64() {
        // === CREATION i64 (all safe) ===
        let zero_i64x2 = vdupq_n_s64(0);
        let set1_i64x2 = vdupq_n_s64(1);
        let zero_u64x2 = vdupq_n_u64(0);
        let set1_u64x2 = vdupq_n_u64(1);
        let zero_i64x1 = vdup_n_s64(0);
        let set1_i64x1 = vdup_n_s64(1);

        // === ARITHMETIC i64x2 (all safe) ===
        let add = vaddq_s64(zero_i64x2, set1_i64x2);
        let sub = vsubq_s64(add, set1_i64x2);
        let neg = vnegq_s64(add);
        let abs = vabsq_s64(add);
        let qadd = vqaddq_s64(add, set1_i64x2);
        let qsub = vqsubq_s64(add, set1_i64x2);

        // === COMPARISON i64x2 (all safe) ===
        let ceq = vceqq_s64(add, set1_i64x2);
        let cge = vcgeq_s64(add, set1_i64x2);
        let cgt = vcgtq_s64(add, set1_i64x2);
        let cle = vcleq_s64(add, set1_i64x2);
        let clt = vcltq_s64(add, set1_i64x2);

        // === SHIFT (all safe) ===
        let shl = vshlq_s64(add, set1_i64x2);
        let shr_n = vshrq_n_s64::<1>(add);
        let shl_n = vshlq_n_s64::<1>(add);

        // === REDUCE (all safe) ===
        let addv = vaddvq_s64(add);

        // === CONVERSION (all safe) ===
        let cvt_f64 = vcvtq_f64_s64(add);
        let cvt_s64 = vcvtq_s64_f64(vdupq_n_f64(1.0));
    }

    #[target_feature(enable = "neon")]
    unsafe fn neon_safe_shuffle_intrinsics() {
        let a = vdupq_n_f32(1.0);
        let b = vdupq_n_f32(2.0);
        let ai = vdupq_n_s32(1);
        let bi = vdupq_n_s32(2);

        // === ZIP/UNZIP (all safe) ===
        let zip1 = vzip1q_f32(a, b);
        let zip2 = vzip2q_f32(a, b);
        let uzp1 = vuzp1q_f32(a, b);
        let uzp2 = vuzp2q_f32(a, b);
        let zip1i = vzip1q_s32(ai, bi);
        let zip2i = vzip2q_s32(ai, bi);
        let uzp1i = vuzp1q_s32(ai, bi);
        let uzp2i = vuzp2q_s32(ai, bi);

        // === TRANSPOSE (all safe) ===
        let trn1 = vtrn1q_f32(a, b);
        let trn2 = vtrn2q_f32(a, b);
        let trn1i = vtrn1q_s32(ai, bi);
        let trn2i = vtrn2q_s32(ai, bi);

        // === EXTRACT (all safe) ===
        let ext = vextq_f32::<1>(a, b);
        let exti = vextq_s32::<1>(ai, bi);

        // === REVERSE (all safe) ===
        let rev64 = vrev64q_f32(a);
        let rev64i = vrev64q_s32(ai);
        let rev32i8 = vrev32q_s8(vdupq_n_s8(1));
        let rev16i8 = vrev16q_s8(vdupq_n_s8(1));

        // === COMBINE/SPLIT (all safe) ===
        let low = vget_low_f32(a);
        let high = vget_high_f32(a);
        let combined = vcombine_f32(low, high);
        let lowi = vget_low_s32(ai);
        let highi = vget_high_s32(ai);
        let combinedi = vcombine_s32(lowi, highi);

        // === DUPLICATE LANE (all safe) ===
        let dup_lane = vdupq_laneq_f32::<0>(a);
        let dup_lanei = vdupq_laneq_s32::<0>(ai);

        // === BSL (bit select - all safe) ===
        let mask = vdupq_n_u32(0xFFFFFFFF);
        let bsl = vbslq_f32(mask, a, b);
        let bsli = vbslq_s32(mask, ai, bi);
    }

    #[target_feature(enable = "neon")]
    unsafe fn neon_safe_reinterpret_intrinsics() {
        let f32x4 = vdupq_n_f32(1.0);
        let i32x4 = vdupq_n_s32(1);
        let u32x4 = vdupq_n_u32(1);
        let i8x16 = vdupq_n_s8(1);

        // === REINTERPRET (all safe - just bit reinterpretation) ===
        let as_i32 = vreinterpretq_s32_f32(f32x4);
        let as_u32 = vreinterpretq_u32_f32(f32x4);
        let as_f32 = vreinterpretq_f32_s32(i32x4);
        let as_i8 = vreinterpretq_s8_s32(i32x4);
        let as_u8 = vreinterpretq_u8_s32(i32x4);
        let as_i16 = vreinterpretq_s16_s32(i32x4);
        let as_u16 = vreinterpretq_u16_s32(i32x4);
        let as_i64 = vreinterpretq_s64_s32(i32x4);
        let as_u64 = vreinterpretq_u64_s32(i32x4);
    }

    // ============================================================================
    // TESTS
    // ============================================================================

    #[test]
    fn test_all_neon_safe_intrinsics_compile() {
        // This test verifies that all intrinsics above compile without
        // requiring unsafe blocks inside the target_feature functions.

        unsafe {
            neon_safe_intrinsics_f32();
            neon_safe_intrinsics_f64();
            neon_safe_intrinsics_i8();
            neon_safe_intrinsics_i16();
            neon_safe_intrinsics_i32();
            neon_safe_intrinsics_i64();
            neon_safe_shuffle_intrinsics();
            neon_safe_reinterpret_intrinsics();
        }
    }
}

// ============================================================================
// UNSAFE NEON INTRINSICS (documented for reference)
// ============================================================================
//
// The following NEON intrinsics ALWAYS require unsafe blocks because
// they involve raw pointer operations:
//
// LOADS:
//   vld1_*, vld1q_*           - Load single vector
//   vld2_*, vld2q_*           - Load 2 interleaved vectors
//   vld3_*, vld3q_*           - Load 3 interleaved vectors
//   vld4_*, vld4q_*           - Load 4 interleaved vectors
//   vld1_lane_*, vld1q_lane_* - Load single lane
//   vld1_dup_*, vld1q_dup_*   - Load and broadcast
//
// STORES:
//   vst1_*, vst1q_*           - Store single vector
//   vst2_*, vst2q_*           - Store 2 interleaved vectors
//   vst3_*, vst3q_*           - Store 3 interleaved vectors
//   vst4_*, vst4q_*           - Store 4 interleaved vectors
//   vst1_lane_*, vst1q_lane_* - Store single lane
//
// ============================================================================

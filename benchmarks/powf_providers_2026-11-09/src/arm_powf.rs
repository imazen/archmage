//! Experimental scalar port of Arm v26.07 powf, canonical unscaled reduction.
//! Copyright (c) 2017-2026, Arm Limited. SPDX-License-Identifier: MIT
//! Source: ARM-software/optimized-routines, 4be260a5; see ARM-LICENSE.txt.
//! RNE, gradual underflow. Result bits only; no errno/FP-exception promise.
use crate::arm_data::{A, C, LOG, OVERFLOW, OVERFLOW_BOUND, SHIFT, TABLE};

#[inline(always)]
fn madd<const FUSED: bool>(x: f64, y: f64, z: f64) -> f64 {
    if FUSED { x.mul_add(y, z) } else { x * y + z }
}
macro_rules! cold {
    ($enabled:expr) => {
        if $enabled {
            core::hint::cold_path();
        }
    };
}
#[inline(always)]
fn zeroinfnan(x: u32) -> bool {
    x.wrapping_mul(2).wrapping_sub(1) >= 0xfeff_ffff
}
#[inline(always)]
fn signaling(x: u32) -> bool {
    x & 0x7fff_ffff > 0x7f80_0000 && x & 0x0040_0000 == 0
}
#[inline(always)]
fn checkint(x: u32) -> u32 {
    let e = (x >> 23) & 255;
    if e < 127 {
        return 0;
    }
    if e > 150 {
        return 2;
    }
    if x & ((1u32 << (150 - e)) - 1) != 0 {
        return 0;
    }
    if x & (1u32 << (150 - e)) != 0 { 1 } else { 2 }
}

// These const parameters propagate into the whole graph. ASSUME is a bounded
// Gamma diagnostic, with its same input preconditions, not unchecked memory.
#[inline(always)]
pub fn pow<const FUSED: bool, const COLD: bool, const ASSUME: bool>(x: f32, y: f32) -> f32 {
    let mut ix = x.to_bits();
    let iy = y.to_bits();
    let mut sign_bias = 0u32;
    if !ASSUME && (ix.wrapping_sub(0x0080_0000) >= 0x7f00_0000 || zeroinfnan(iy)) {
        cold!(COLD);
        if zeroinfnan(iy) {
            if iy.wrapping_mul(2) == 0 {
                return if signaling(ix) { x + y } else { 1. };
            }
            if ix == 0x3f80_0000 {
                return if signaling(iy) { x + y } else { 1. };
            }
            if ix.wrapping_mul(2) > 0xff00_0000 || iy.wrapping_mul(2) > 0xff00_0000 {
                return x + y;
            }
            if ix.wrapping_mul(2) == 0x7f00_0000 {
                return 1.;
            }
            if (ix.wrapping_mul(2) < 0x7f00_0000) == (iy & 0x8000_0000 == 0) {
                return 0.;
            }
            return y * y;
        }
        if zeroinfnan(ix) {
            let mut x2 = x * x;
            if ix & 0x8000_0000 != 0 && checkint(iy) == 1 {
                x2 = -x2;
            }
            return if iy & 0x8000_0000 != 0 { 1. / x2 } else { x2 };
        }
        if ix & 0x8000_0000 != 0 {
            let integer = checkint(iy);
            if integer == 0 {
                return f32::NAN;
            }
            if integer == 1 {
                sign_bias = 1 << 16;
            }
            ix &= 0x7fff_ffff;
        }
        if ix < 0x0080_0000 {
            ix = (x * 8_388_608.).to_bits() & 0x7fff_ffff;
            ix = ix.wrapping_sub(23 << 23);
        }
    }
    if ASSUME && (ix == 0 || iy & 0x7fff_ffff == 0) {
        return if iy & 0x7fff_ffff == 0 { 1. } else { 0. };
    }
    let tmp = ix.wrapping_sub(0x3f33_0000);
    let index = ((tmp >> 19) & 15) as usize;
    let top = tmp & 0xff80_0000;
    let z = f32::from_bits(ix.wrapping_sub(top)) as f64;
    let k = (top as i32) >> 23;
    let r = madd::<FUSED>(z, LOG[index].0, -1.);
    let y0 = LOG[index].1 + f64::from(k);
    let r2 = r * r;
    let h = madd::<FUSED>(A[0], r, A[1]);
    let p = madd::<FUSED>(A[2], r, A[3]);
    let r4 = r2 * r2;
    let q = madd::<FUSED>(A[4], r, y0);
    let q = madd::<FUSED>(p, r2, q);
    let logx = madd::<FUSED>(h, r4, q);
    let ylogx = f64::from(y) * logx;
    if !ASSUME && ((ylogx.to_bits() >> 47) & 0xffff) >= (126f64.to_bits() >> 47) {
        cold!(COLD);
        let sign = if sign_bias == 0 { 0 } else { 0x8000_0000 };
        if ylogx <= -150. {
            return f32::from_bits(sign);
        }
        if ylogx < -149. {
            return f32::from_bits(sign | 1);
        }
        if ylogx > OVERFLOW {
            return f32::from_bits(sign | 0x7f80_0000);
        }
        if ylogx > OVERFLOW_BOUND {
            return f32::from_bits(sign | 0x7f7f_ffff);
        }
    }
    let kd = ylogx + SHIFT;
    let ki = kd.to_bits();
    let kd = kd - SHIFT;
    let r = ylogx - kd;
    let scale = TABLE[(ki & 31) as usize].wrapping_add(ki.wrapping_add(u64::from(sign_bias)) << 47);
    let z = madd::<FUSED>(C[0], r, C[1]);
    let r2 = r * r;
    let tail = madd::<FUSED>(C[2], r, 1.);
    (madd::<FUSED>(z, r2, tail) * f64::from_bits(scale)) as f32
}

#[cfg(test)]
mod tests {
    use super::pow;

    #[test]
    fn signed_zero_subnormal_and_range_regressions() {
        // Exact power/IEEE cases, plus the independently MPFR-enclosed input
        // on which the measured Rust libm overflowed incorrectly.
        let cases = [
            (0x00000000, 0x00000000, 0x3f800000),
            (0x80000000, 0x40400000, 0x80000000),
            (0x80000000, 0xc0400000, 0xff800000),
            (0x00000001, 0x3f800000, 0x00000001),
            (0x80000001, 0x3f800000, 0x80000001),
            (0x80800000, 0x40400000, 0x80000000),
            (0x7f7fffff, 0x3f800000, 0x7f7fffff),
            (0xff7fffff, 0x3f800000, 0xff7fffff),
            (0xff800000, 0xc0400000, 0x80000000),
            (0xbf800000, 0x7f800000, 0x3f800000),
            (0x3f7ffff6, 0xcd000001, 0x792abd43),
        ];
        for (xb, nb, expected) in cases {
            let x = f32::from_bits(xb);
            let n = f32::from_bits(nb);
            assert_eq!(
                pow::<true, true, false>(x, n).to_bits(),
                expected,
                "fused {xb:08x}^{nb:08x}"
            );
            assert_eq!(
                pow::<false, true, false>(x, n).to_bits(),
                expected,
                "separate {xb:08x}^{nb:08x}"
            );
        }
    }

    #[test]
    fn signaling_nan_precedence_and_gamma_zero() {
        assert!(pow::<true, true, false>(f32::from_bits(0x7f800001), 0.).is_nan());
        assert!(pow::<true, true, false>(1., f32::from_bits(0x7f800001)).is_nan());
        assert_eq!(pow::<true, true, false>(f32::NAN, 0.), 1.);
        assert_eq!(pow::<true, true, true>(0., 0.), 1.);
        assert_eq!(pow::<true, true, true>(0., 0.5).to_bits(), 0);
    }
}

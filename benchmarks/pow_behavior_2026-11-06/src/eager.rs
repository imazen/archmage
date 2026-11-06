//! Frozen eager-parity packed power from b7d0ec5a.
//! Source: archmageb7d0ec5a, magemath/src/correct/mod.rs. Experimental only.
use crate::powf_tables;
use magetypes::simd::{backends::F64x4Backend, generic::f64x4};
#[inline(always)]
fn v<T: F64x4Backend>(t: T, a: [f64; 4]) -> f64x4<T> {
    f64x4::from_array_t(t, a)
}
#[inline(always)]
fn c<T: F64x4Backend>(t: T, a: f64) -> f64x4<T> {
    f64x4::splat_t(t, a)
}
#[inline(always)]
pub(crate) fn pow<T: F64x4Backend>(t: T, x: [f32; 4], y: [f32; 4]) -> [f32; 4] {
    use powf_tables::*;
    // f32 integers above 2^24 are even. Inspect exponent/fraction bits so
    // negative nonintegers keep the complete scalar domain/special path.
    let parity = y.map(integer_parity);
    let valid: [bool; 4] = core::array::from_fn(|i| {
        x[i].abs() > 0.0
            && x[i].is_finite()
            && x[i].abs() != 1.0
            && y[i].is_finite()
            && y[i] != 0.0
            && y[i] != 1.0
            && (!x[i].is_sign_negative() || parity[i].0)
    });
    let xx: [f64; 4] = core::array::from_fn(|i| if valid[i] { x[i].abs() as f64 } else { 1.0 });
    let yy = v(
        t,
        core::array::from_fn(|i| if valid[i] { y[i] as f64 } else { 0.0 }),
    );
    let bits = xx.map(f64::to_bits);
    let m = bits.map(|b| b & 0x000fffffffffffff);
    let j = m.map(|b| ((b + (1u64 << 46)) >> 47) as usize);
    let e = v(
        t,
        core::array::from_fn(|i| {
            (((bits[i] >> 52) & 2047) as i32 - 1023 + i32::from(j[i] > 13)) as f64
        }),
    );
    let xd = v(t, m.map(|b| f64::from_bits(b | (0x3ff_u64 << 52))));
    let z = xd.mul_add_fused(v(t, j.map(|i| FAST_IX[i])), c(t, -1.0));
    let z2 = z * z;
    let z4 = z2 * z2;
    let c6 = c(t, FAST_C[6]) + z * c(t, FAST_C[7]);
    let c4 = c(t, FAST_C[4]) + z * c(t, FAST_C[5]);
    let c2 = c(t, FAST_C[2]) + z * c(t, FAST_C[3]);
    let c0 = c(t, FAST_C[0]) + z * c(t, FAST_C[1]);
    let l = z * ((c0 + z2 * c2) + z4 * (c4 + z2 * c6)) - v(t, j.map(|i| FAST_LIX[i][1]));
    let yy = yy * c(t, 16.0);
    let zt = (e - v(t, j.map(|i| FAST_LIX[i][0]))) * yy;
    let z = l * yy + zt;
    let zz = z.to_array();
    let in_range: [bool; 4] = core::array::from_fn(|i| {
        valid[i]
            && (-2400.0..=2048.0).contains(&zz[i])
            && zz[i].abs() >= f64::from_bits(0x3e50000000000000)
    });
    let ia = v(
        t,
        core::array::from_fn(|i| if in_range[i] { zz[i] } else { 0.0 }),
    )
    .floor();
    let h = l.mul_add_fused(yy, zt - ia);
    let il = ia.to_array().map(|a| a as i64);
    let s = v(
        t,
        il.map(|a| {
            let j = (a & 15) as usize;
            let e = (a - j as i64) >> 4;
            FAST_TB[j] * f64::from_bits(((e + 1023) as u64) << 52)
        }),
    );
    let h2 = h * h;
    let c0 = c(t, FAST_CE[0]) + h * c(t, FAST_CE[1]);
    let c2 = c(t, FAST_CE[2]) + h * c(t, FAST_CE[3]);
    let c4 = c(t, FAST_CE[4]) + h * c(t, FAST_CE[5]);
    let rr = (s + (s * h) * (c0 + h2 * (c2 + h2 * c4))).to_array();
    core::array::from_fn(|i| {
        if in_range[i] && (rr[i].to_bits().wrapping_add(468) & 0xfffffff) > 936 {
            let magnitude = (rr[i] as f32).to_bits();
            let sign = (x[i].to_bits() & 0x80000000) & (0u32.wrapping_sub(u32::from(parity[i].1)));
            f32::from_bits(magnitude | sign)
        } else {
            zenmath::scalar::powf(x[i], y[i])
        }
    })
}

// (integer, odd integer), including signed zero; infinity and NaN are neither.
#[inline(always)]
fn integer_parity(y: f32) -> (bool, bool) {
    let bits = y.to_bits() & 0x7fffffff;
    let exponent = (bits >> 23) as i32 - 127;
    if exponent >= 24 {
        return (bits < 0x7f800000, false);
    }
    if exponent < 0 {
        return (bits == 0, false);
    }
    let shift = (23 - exponent) as u32;
    let integer = bits & ((1u32 << shift) - 1) == 0;
    (integer, integer && ((bits >> shift) & 1) != 0)
}

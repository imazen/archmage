//! CORE-MATH common paths evaluated four binary64 lanes at a time.
//! Preserve the scalar operation order and rounding tests. Exceptional lanes
//! and ambiguous results use the complete zenmath implementation, individually.
use magetypes::simd::{backends::F64x4Backend, generic::f64x4};
mod expf_tables;
mod log2f_tables;
mod logf_tables;
mod powf_tables;

#[inline(always)]
fn v<T: F64x4Backend>(t: T, a: [f64; 4]) -> f64x4<T> {
    f64x4::from_array_t(t, a)
}
#[inline(always)]
fn c<T: F64x4Backend>(t: T, a: f64) -> f64x4<T> {
    f64x4::splat_t(t, a)
}

#[inline(always)]
pub(crate) fn log<T: F64x4Backend, const TWO: bool>(t: T, x: [f32; 4]) -> [f32; 4] {
    let bits = x.map(f32::to_bits);
    // Keep non-normal lanes harmless until the scalar rescue.
    let u = bits.map(|b| {
        if (0x00800000..0x7f800000).contains(&b) {
            b
        } else {
            0x3f800000
        }
    });
    let m = u.map(|b| b & 0x7fffff);
    let j = m.map(|b| ((b + (1 << 16)) >> 17) as usize);
    let e = v(t, u.map(|b| ((b >> 23) as i32 - 127) as f64));
    let xd = v(
        t,
        m.map(|b| f64::from_bits(((b as u64) << 29) | (0x3ff_u64 << 52))),
    );
    let (b, r, err) = if TWO {
        let z = xd * v(t, j.map(|i| log2f_tables::IX[i])) - c(t, 1.0);
        let el = e - v(t, j.map(|i| log2f_tables::LIX[i]));
        (log2f_tables::B, (z, el), f64::from_bits(0x3df6610000000000))
    } else {
        let z = xd * v(t, j.map(|i| logf_tables::TR[i])) - c(t, 1.0);
        let el = e * c(t, f64::from_bits(0x3fe62e42fefa39ef)) + v(t, j.map(|i| logf_tables::TL[i]));
        (logf_tables::B, (z, el), f64::from_bits(0x3def060000000000))
    };
    let (z, el) = r;
    let f = (el + z * c(t, b[0])) + (z * z) * (c(t, b[1]) + z * c(t, b[2]));
    let lo = f.to_array().map(|a| a as f32);
    let hi = (f + c(t, err)).to_array().map(|a| a as f32);
    // Resolve IEEE rails without a scalar call. Only subnormals and ambiguous
    // finite rounding intervals need the accurate implementation.
    let handled = x.map(|a| a <= 0.0 || !a.is_finite());
    let powers = core::array::from_fn::<_, 4, _>(|i| {
        bits[i] == u[i] && m[i] == 0 && (TWO || bits[i] == 0x3f800000)
    });
    let out = core::array::from_fn(|i| {
        if x[i] == 0.0 {
            f32::NEG_INFINITY
        } else if x[i] < 0.0 || x[i].is_nan() {
            f32::NAN
        } else if x[i] == f32::INFINITY {
            f32::INFINITY
        } else if powers[i] {
            if TWO {
                ((u[i] >> 23) as i32 - 127) as f32
            } else {
                0.0
            }
        } else {
            lo[i]
        }
    });
    let needs = core::array::from_fn(|i| {
        !handled[i] && !powers[i] && (bits[i] != u[i] || lo[i].to_bits() != hi[i].to_bits())
    });
    finish::<1>(x, [if TWO { 2.0 } else { 0.0 }; 4], out, needs)
}

#[inline(always)]
pub(crate) fn exp<T: F64x4Backend>(t: T, x: [f32; 4]) -> [f32; 4] {
    use expf_tables::*;
    // The vector path covers the scalar fast-range predicate exactly.
    let valid = x.map(|a| (0x6f93813e..=0x8562e42e).contains(&a.to_bits().wrapping_shl(1)));
    let z = v(
        t,
        core::array::from_fn(|i| if valid[i] { x[i] as f64 } else { 0.0 }),
    );
    let a = z * c(t, f64::from_bits(0x3ff71547652b82fe));
    let big = c(t, f64::from_bits(0x42d8000000000000));
    let rounded = a + big;
    let u = rounded.to_array().map(f64::to_bits);
    let h = a + (big - rounded);
    let s = v(
        t,
        u.map(|b| f64::from_bits(TB[(b & 63) as usize].wrapping_add((b >> 6) << 52))),
    );
    let r = ((c(t, B[0]) + h * c(t, B[1])) + (h * h) * (c(t, B[2]) + h * c(t, B[3]))) * s;
    let hi = r.to_array().map(|a| a as f32);
    let lo = (r - r * c(t, 1.45e-10)).to_array().map(|a| a as f32);
    let needs = core::array::from_fn(|i| {
        !x[i].is_nan() && (!valid[i] || hi[i].to_bits() != lo[i].to_bits())
    });
    let out = core::array::from_fn(|i| if x[i].is_nan() { f32::NAN } else { hi[i] });
    finish::<0>(x, [0.0; 4], out, needs)
}

#[inline(always)]
pub(crate) fn pow<T: F64x4Backend>(t: T, x: [f32; 4], y: [f32; 4]) -> [f32; 4] {
    use powf_tables::*;
    let valid: [bool; 4] = core::array::from_fn(|i| {
        x[i] > 0.0
            && x[i].is_finite()
            && x[i] != 1.0
            && y[i].is_finite()
            && y[i] != 0.0
            && y[i] != 1.0
    });
    let xx: [f64; 4] = core::array::from_fn(|i| if valid[i] { x[i] as f64 } else { 1.0 });
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
        if (x[i].is_nan() && y[i] != 0.0) || (y[i].is_nan() && x[i] != 1.0) {
            f32::NAN
        } else if in_range[i] && (rr[i].to_bits().wrapping_add(468) & 0xfffffff) > 936 {
            rr[i] as f32
        } else {
            zenmath::scalar::powf(x[i], y[i])
        }
    })
}

// Keep scalar calls off the common path, with one combined lane test.
#[inline(always)]
fn finish<const OP: u8>(x: [f32; 4], y: [f32; 4], out: [f32; 4], needs: [bool; 4]) -> [f32; 4] {
    if needs.into_iter().any(|v| v) {
        rescue::<OP>(x, y, out, needs)
    } else {
        out
    }
}
#[cold]
#[inline(never)]
fn rescue<const OP: u8>(x: [f32; 4], y: [f32; 4], mut out: [f32; 4], needs: [bool; 4]) -> [f32; 4] {
    for i in 0..4 {
        if needs[i] {
            out[i] = match OP {
                0 => zenmath::scalar::expf(x[i]),
                1 => {
                    if y[i] == 2.0 {
                        zenmath::scalar::log2f(x[i])
                    } else {
                        zenmath::scalar::logf(x[i])
                    }
                }
                _ => unreachable!(),
            };
        }
    }
    out
}

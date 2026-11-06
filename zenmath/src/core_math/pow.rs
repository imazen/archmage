/* Correctly-rounded power function for two binary64 values.

Copyright (c) 2022-2025 CERN and Inria
Authors: Tom Hubrecht and Paul Zimmermann

This file is part of the CORE-MATH project
(https://core-math.gitlabpages.inria.fr/).

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
use super::{
    fma, pow_d_tables as d,
    pow_int::{D, Q},
    pow_q_tables as q,
    pow_tables::*,
};
fn two(a: f64, b: f64) -> (f64, f64) {
    let h = a + b;
    (h, b - (h - a))
}
fn sum(a: f64, b: f64, c: f64) -> (f64, f64) {
    let (h, l) = two(a, b);
    (h, l + c)
}
fn amul(a: f64, b: f64) -> (f64, f64) {
    let h = a * b;
    (h, fma(a, b, -h))
}
fn smul(a: f64, b: f64, c: f64) -> (f64, f64) {
    let (h, l) = amul(a, b);
    (h, fma(a, c, l))
}
fn dmul(a: f64, b: f64, c: f64, d: f64) -> (f64, f64) {
    let (h, l) = amul(a, c);
    (h, fma(a, d, fma(b, c, l)))
}
fn log1(x: f64) -> (f64, f64, bool) {
    let bits = x.to_bits();
    let mut m = bits & 0xfffffffffffff;
    let mut e = ((bits >> 52) & 2047) as i64;
    let mut t = if e != 0 {
        e -= 1023;
        m += 1 << 52;
        f64::from_bits((m & 0xfffffffffffff) | (1023 << 52))
    } else {
        let k = m.leading_zeros() - 11;
        e = -1022 - i64::from(k);
        m <<= k;
        f64::from_bits(m | (1023 << 52))
    };
    let c = m >= 0x16a09e667f3bcd;
    e += i64::from(c);
    let i = (m >> if c { 45 } else { 44 }) as usize - 181;
    if c {
        t *= 0.5;
    }
    let z = fma(_INVERSE[i], t, -1.0);
    let th = fma(e as f64, f64::from_bits(0x3fe62e42fefa3800), _LOG_INV[i][0]);
    let tl = fma(e as f64, f64::from_bits(0x3d2ef35793c76730), _LOG_INV[i][1]);
    let (h, l) = sum(th, z, tl);
    let (wh, wl) = amul(z, z);
    let t = fma(P_1[5], z, P_1[4]);
    let u = fma(P_1[3], z, P_1[2]);
    let v = fma(P_1[1], z, P_1[0]);
    let u = fma(t, wh, u);
    let v = fma(u, wh, v);
    let u = v * wh;
    let ph = -0.5 * wh;
    let pl = fma(u, z, -0.5 * wl);
    let (h, l) = sum(h, ph, l + pl);
    if e == 0 && l.abs() > h.abs() * f64::from_bits(0x3e70000000000000) {
        let (h, l) = two(h, l);
        (h, l, true)
    } else {
        (h, l, false)
    }
}
fn exp1(rh: f64, rl: f64, s: f64) -> (f64, f64) {
    if rh.is_nan() || rh > f64::from_bits(0x40862e42e709a95b) {
        return if rh > f64::from_bits(0x40862e4316ea5df9) {
            (f64::MAX * s, f64::MAX * s)
        } else {
            (f64::NAN, f64::NAN)
        };
    }
    if rh < -f64::from_bits(0x408483b8cca421af) {
        return if rh < -f64::from_bits(0x40874910ee4e8a27) {
            (0.0 * s, 0.0 * s)
        } else {
            (f64::NAN, f64::NAN)
        };
    }
    let k = crate::scalar::roundeven(rh * f64::from_bits(0x40b71547652b82fe));
    let zh = fma(f64::from_bits(0x3f262e42fefa39ef), -k, rh);
    let zl = fma(f64::from_bits(0x3bbabc9e3b39803f), -k, rl);
    let ki = k as i64;
    let m = (ki >> 12) + 1023;
    let i2 = ((ki >> 6) & 63) as usize;
    let i1 = (ki & 63) as usize;
    let (eh, el) = dmul(T2[i1][0], T2[i1][1], T1[i2][0], T1[i2][1]);
    let z = zh + zl;
    let qq = fma(fma(Q_1[4], z, Q_1[3]), z, Q_1[2]);
    let h0 = fma(qq, z, Q_1[1]);
    let (h1, l1) = amul(z, h0);
    let (qh, ql) = sum(Q_1[0], h1, l1);
    let (eh, el) = dmul(eh, el, qh, ql);
    let scale = f64::from_bits((m as u64) << 52) * s;
    (eh * scale, el * scale)
}
fn log2(mut x: D) -> D {
    let mut e = x.ex;
    let i = if x.hi > 0xb504f333f9de6484 {
        e += 1;
        (x.hi >> 57) as usize
    } else {
        (x.hi >> 56) as usize
    };
    x.ex -= e;
    let z = x.mul11(d::_INVERSE_2_1[i - 90]);
    let j = (z.hi >> (50 - z.ex)) as usize;
    let z = d::M_ONE.add(z.mul11(d::_INVERSE_2_2[j - 8128]));
    let mut p = z.mul11(d::P_2[0]);
    for k in 1..=3 {
        p = d::P_2[k].add11(p);
        p = z.mul11(p);
    }
    p = d::P_2[4].add(p);
    for k in 5..=8 {
        p = p.mul21(z);
        p = d::P_2[k].add(p);
    }
    p = p.mul21(z);
    p = d::_LOG_INV_2_2[j - 8128].add(p);
    p = d::_LOG_INV_2_1[i - 90].add(p);
    p.add(d::LOG2.muli(e))
}
fn exp2(x: D) -> D {
    if x.ex >= 10 {
        return D {
            ex: if x.sgn != 0 { -1076 } else { 1025 },
            sgn: 0,
            ..x
        };
    }
    let k = x.mul11(d::LOG2_INV).to_int();
    let mut kk = d::LOG2.muli(k);
    kk.ex -= 12;
    let y = x.add(kk.neg());
    let mut r = y.mul11(d::Q_2[0]);
    r = d::Q_2[1].add11(r);
    r = y.mul11(r);
    r = d::Q_2[2].add11(r);
    r = y.mul11(r);
    r = d::Q_2[3].add(r);
    for i in 4..8 {
        r = y.mul(r);
        r = d::Q_2[i].add(r);
    }
    r = d::T1_2[((k >> 6) & 63) as usize].mul(r);
    r = d::T2_2[(k & 63) as usize].mul(r);
    r.ex += k >> 12;
    r
}
fn log3(mut x: Q) -> Q {
    let mut e = x.ex;
    let i = if x.hh > 0xb504f333f9de6484 {
        e += 1;
        (x.hh >> 57) as usize
    } else {
        (x.hh >> 56) as usize
    };
    x.ex -= e;
    let z = x.mul(q::_INVERSE_3_1[i - 90], 4, 4);
    let j = (z.hh >> (50 - z.ex)) as usize;
    let z = q::M_ONE_Q.add(z.mul(q::_INVERSE_3_2[j - 8128], 4, 4));
    let mut p = q::P_3[0].mul(z, 1, 1);
    p = q::P_3[1].add22(p);
    for k in 2..18 {
        p = p.mul(
            z,
            if k < 4 {
                1
            } else if k < 8 {
                2
            } else if k < 14 {
                3
            } else {
                4
            },
            1,
        );
        p = if k < 8 {
            q::P_3[k].add22(p)
        } else {
            q::P_3[k].add(p)
        };
    }
    p = p.mul(z, 4, 1);
    p = q::_LOG_INV_3_2[j - 8128].add(p);
    p = q::_LOG_INV_3_1[i - 90].add(p);
    p.add(q::LOG2_Q.muli(e))
}
fn exp3(x: Q) -> Q {
    let k = x.mul(q::LOG2_INV_Q, 1, 1).to_int();
    let mut kk = q::LOG2_Q.muli(k);
    kk.ex -= 12;
    let y = x.add(kk.neg());
    let mut r = y.mul(q::Q_3[0], 1, 1);
    r = q::Q_3[1].add22(r);
    for k in 2..15 {
        r = y.mul(
            r,
            if k < 7 {
                2
            } else if k < 12 {
                3
            } else {
                4
            },
            if k < 7 {
                2
            } else if k < 12 {
                3
            } else {
                4
            },
        );
        r = if k < 7 {
            q::Q_3[k].add22(r)
        } else {
            q::Q_3[k].add(r)
        };
    }
    r = q::T1_3[((k >> 6) & 63) as usize].mul(r, 4, 4);
    r = q::T2_3[(k & 63) as usize].mul(r, 4, 4);
    r.ex += k >> 12;
    r
}
fn extract(x: f64) -> (i64, u64) {
    let b = x.to_bits();
    let e = (b >> 52) & 2047;
    let m = (b & 0xfffffffffffff) + if e != 0 { 1 << 52 } else { 0 };
    let t = m.trailing_zeros();
    (e as i64 + i64::from(t) - 1075 + i64::from(e == 0), m >> t)
}
// Determine exact rational powers by integer arithmetic, without treating a
// nearby higher-precision approximation as proof of an exact result.
fn exact(x: f64, y: f64, sign: u64) -> Option<f64> {
    let (mut e, mut m) = extract(x);
    if m == 1 {
        let g = e as f64 * y;
        if g == crate::scalar::roundeven(g) {
            if g > 1023.0 {
                return Some(f64::INFINITY.copysign(if sign == 0 { 1.0 } else { -1.0 }));
            }
            if g < -1074.0 {
                return Some(f64::from_bits(sign << 63));
            }
            return Some(f64::from_bits(
                (sign << 63)
                    | if g >= -1022.0 {
                        ((g as i64 + 1023) as u64) << 52
                    } else {
                        1u64 << ((g as i64) + 1074)
                    },
            ));
        }
        return None;
    }
    if !(0.0..=34.0).contains(&y) {
        return None;
    }
    let (f, n) = extract(y);
    if n > 34 || f < -5 {
        return None;
    }
    if f < 0 {
        for _ in 0..-f {
            if e & 1 != 0 {
                return None;
            }
            let root = m.isqrt();
            if root * root != m {
                return None;
            }
            m = root;
            e /= 2;
        }
    }
    let mut t = if f >= 0 { n << f } else { n };
    let mut k = 1u64;
    let total = t;
    while t != 0 {
        if t & 1 != 0 {
            k = k.checked_mul(m)?;
        }
        t >>= 1;
        if t != 0 {
            m = m.checked_mul(m)?;
        }
    }
    if k >> 54 != 0 {
        return None;
    }
    let z = k.leading_zeros();
    Some(
        D {
            hi: k << z,
            lo: 0,
            ex: e * total as i64 + 63 - i64::from(z),
            sgn: sign,
        }
        .to_double(),
    )
}
/// Correctly rounded power, nearest-even with gradual underflow.
pub fn pow(mut x: f64, y: f64) -> f64 {
    if x.is_nan() {
        return if y == 0.0 && x.to_bits() & (1 << 51) != 0 {
            1.0
        } else {
            x + x
        };
    }
    if y.is_nan() {
        return if x == 1.0 && y.to_bits() & (1 << 51) != 0 {
            1.0
        } else {
            y + y
        };
    }
    if y == 0.0 {
        return 1.0;
    }
    let yi = y == crate::scalar::roundeven(y);
    let odd = yi && y.abs() < f64::from_bits(0x4340000000000000) && (y as i64) & 1 != 0;
    if y.is_infinite() {
        return if x.abs() == 1.0 {
            1.0
        } else if (x.abs() > 1.0) == (y > 0.0) {
            f64::INFINITY
        } else {
            0.0
        };
    }
    if x == 0.0 || x.is_infinite() {
        let sign = if x.is_sign_negative() && odd {
            1u64 << 63
        } else {
            0
        };
        let infinity = (x == 0.0) == (y < 0.0);
        return f64::from_bits(sign | if infinity { 0x7ff0000000000000 } else { 0 });
    }
    let sign = u64::from(x < 0.0 && odd);
    if x < 0.0 {
        if !yi {
            return f64::NAN;
        }
        x = -x;
    }
    let s = if sign == 0 { 1.0 } else { -1.0 };
    let (mut lh, mut ll, cancel) = log1(x);
    let ey = (y.to_bits() >> 52) & 2047;
    if !(0x36..0x7f5).contains(&ey) {
        lh = f64::NAN;
        ll = f64::NAN;
    }
    let (rh, rl) = smul(y, lh, ll);
    let (h, l) = exp1(rh, rl, s);
    let err = if cancel {
        f64::from_bits(0x3c55700000000000)
    } else {
        f64::from_bits(0x3bf2700000000000)
    };
    let lo = h + fma(err, -h, l);
    let hi = h + fma(err, h, l);
    if lo == hi {
        return hi;
    }
    if y == 1.0 {
        return s * x;
    }
    if y == 2.0 {
        return x * x;
    }
    if y == 0.5 {
        return crate::scalar::sqrt(x);
    }
    if x == 1.0 {
        return s;
    }
    let mut r = exp2(log2(D::from_double(x)).mul21(D::from_double(y)));
    r.sgn = sign;
    if r.ex < -1075 {
        return f64::from_bits(sign << 63);
    }
    let round = if r.ex < -1022 {
        let ex = -(1022 + r.ex);
        let m = (r.lo >> (10 + ex)) | (r.hi << (54 - ex));
        m.wrapping_add(14) > 28
    } else {
        let lo = (r.lo >> 10) | (r.hi << 54);
        lo.wrapping_add(28) > 56
    };
    if round {
        return r.to_double();
    }
    if let Some(result) = exact(x, y, sign) {
        return result;
    }
    let qr = log3(Q::from_double(x)).mul(Q::from_double(y), 4, 1);
    let mut z = exp3(qr);
    let r1 = (z.hh << 54) | (z.hl >> 10);
    let r2 = (z.hl << 54) | (z.lh >> 10);
    let r3 = (z.lh << 54) | (z.ll >> 10);
    let round = !((r1 == 0 && r2 == 0 && r3 <= 60)
        || (r1 == u64::MAX && r2 == u64::MAX && r3.wrapping_add(120) <= 60));
    if round {
        z.sgn = sign;
        z.ll &= u64::MAX << 10;
        return z.to_double();
    }
    if qr.ex < -56 {
        return 1.0;
    }
    panic!("CORE-MATH pow unresolved rounding");
}

#[cfg(test)]
#[path = "pow_tests.rs"]
mod tests;

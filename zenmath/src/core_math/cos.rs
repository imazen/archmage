/* Correctly-rounded cosine function for binary64 value.

Copyright (c) 2026 Paul Zimmermann

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
    fma,
    sin::{C2U, PC, PS, S1U, S2U, U1, U2, fixed_to_float, mh, mul, reduce, sum},
};
#[inline(never)]
fn accurate(x: f64) -> f64 {
    let (r, k, neg) = reduce(x, true);
    let r = r << 3;
    let u2 = mh(r, r);
    let u4 = mh(u2, u2);
    let u2h = u2 >> 64;
    let mut sign = (k >> 12) & 1;
    let i1 = ((k >> 6) & 63) as usize;
    let i2 = (k & 63) as usize;
    let s1 = S1U[if i1 <= 32 { i1 } else { 64 - i1 }];
    let c1 = S1U[if i1 <= 32 { 32 - i1 } else { i1 - 32 }];
    let mut s = mh(s1, C2U[i2]);
    let t = mh(c1, S2U[i2]);
    s = if i1 < 32 { s + t } else { s - t };
    let mut c = mh(c1, C2U[i2]);
    let t = mh(s1, S2U[i2]);
    c = if i1 < 32 { c - t } else { c + t };
    let sh = mh(PS[2] - PS[3] * u2h, u4);
    let sr = mh(PS[0] - mh(PS[1], u2) + sh, r);
    let sh = PC[3] - u2h * PC[4];
    let sh = PC[2] - mh(sh, u2);
    let cr = PC[0] - mh(PC[1], u2) + mh(sh, u4);
    c = mh(c, cr);
    s = mh(s, sr);
    if (i1 >= 32) ^ neg {
        c += s;
    } else if c <= s {
        c = s - c;
        sign ^= 1;
    } else {
        c -= s;
    }
    let ret = fixed_to_float(c, sign ^ u64::from(i1 >= 32));
    if (c as u64).wrapping_add(5) <= 3 {
        const CASES: [(f64, f64, f64); 4] = [
            (
                f64::from_bits(0x3e88000000000009),
                f64::from_bits(0x3fefffffffffff70),
                f64::from_bits(0x370b56666666666c),
            ),
            (
                f64::from_bits(0x3e98000000000024),
                f64::from_bits(0x3feffffffffffdc0),
                f64::from_bits(0x376b56666666667e),
            ),
            (
                f64::from_bits(0x3ea8000000000090),
                f64::from_bits(0x3feffffffffff700),
                f64::from_bits(0x37cb5666666666c4),
            ),
            (
                f64::from_bits(0x3eb20000000000f3),
                f64::from_bits(0x3fefffffffffebc0),
                f64::from_bits(0x38037642666666fd),
            ),
        ];
        if let Some(c) = CASES.iter().find(|c| c.0 == x) {
            return c.1 + c.2;
        }
    }
    ret
}
fn evaluate(x: f64, r: f64, j: u64, large: bool) -> f64 {
    let sign = ((j >> 14) & 1) as usize;
    let i1 = ((j >> 7) & 127) as usize;
    let i2 = (j & 127) as usize;
    let (c1h, c1l) = mul(U1[i1][2], U1[i1][3], U2[i2][2], U2[i2][3]);
    let (c2h, c2l) = mul(U2[i2][0], U2[i2][1], U1[i1][0], U1[i1][1]);
    let (ch, cl) = sum(c1h, c1l, -c2h, -c2l);
    let sh = U1[i1][0] * U2[i2][2] + U1[i1][2] * U2[i2][0];
    let r2 = r * r;
    let (sr, cr, eps) = if large {
        (
            r * (f64::from_bits(0x401921fb54442d18) - f64::from_bits(0x4044abbcdb6b26d1) * r2),
            r2 * (-f64::from_bits(0x4033bd3cc9be45de) + f64::from_bits(0x40503c1eee483083) * r2),
            f64::from_bits(0x3c00100000000000),
        )
    } else {
        (
            r * (1.0 - f64::from_bits(0x3fc55555553068f0) * r2),
            r2 * (-0.5 + f64::from_bits(0x3fa55555553bfd30) * r2),
            f64::from_bits(0x3bfde00000000000),
        )
    };
    let sg = [1.0, -1.0][sign];
    let fh = sg * ch;
    let fl = sg * (cl + ch * cr - sh * sr);
    let lb = fh + (fl - eps);
    let ub = fh + (fl + eps);
    if lb == ub { lb } else { accurate(x) }
}
/// Cosine, rounded to nearest with ties to even.
pub fn cos(x: f64) -> f64 {
    let z = x.abs();
    let e = z.to_bits() >> 52;
    if e < 1023 - 27 {
        return fma(x, -x, 1.0);
    }
    if e < 1023 + 31 {
        let k = crate::scalar::roundeven(f64::from_bits(0x40b45f306dc9c883) * z);
        let rh = fma(k, -f64::from_bits(0x3f2921fb54442d18), z);
        let rl = k * (-f64::from_bits(0x3bc1a62633145c07));
        evaluate(z, rh + rl, k as u64, false)
    } else if e == 2047 {
        f64::from_bits(x.to_bits() | (1 << 51))
    } else {
        let (u, j, _) = reduce(z, false);
        let r = ((u << 15) >> 75) as u64 as f64 * f64::from_bits(0x3bb0000000000000)
            - f64::from_bits(0x3ef0000000000000);
        evaluate(z, r, j, true)
    }
}

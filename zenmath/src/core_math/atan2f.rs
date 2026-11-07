/* Correctly-rounded arctangent function of two binary32 values.

Copyright (c) 2022-2025 Alexei Sibidanov and Paul Zimmermann.

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
use super::fma;
const CN: [f64; 7] = [
    f64::from_bits(0x3ff0000000000000),
    f64::from_bits(0x40040e0698f94c35),
    f64::from_bits(0x400248c5da347f0d),
    f64::from_bits(0x3fed873386572976),
    f64::from_bits(0x3fc46fa40b20f1d0),
    f64::from_bits(0x3f833f5e041eed0f),
    f64::from_bits(0x3f1546bbf28667c5),
];
const CD: [f64; 7] = [
    f64::from_bits(0x3ff0000000000000),
    f64::from_bits(0x4006b8b143a3f6da),
    f64::from_bits(0x4008421201d18ed5),
    f64::from_bits(0x3ff8221d086914eb),
    f64::from_bits(0x3fd670657e3a07ba),
    f64::from_bits(0x3fa0f4951fd1e72d),
    f64::from_bits(0x3f4b3874b8798286),
];
const C: [[f64; 2]; 32] = [
    [
        f64::from_bits(0x3ff0000000000000),
        f64::from_bits(0xba88c1dac5492248),
    ],
    [
        f64::from_bits(0xbfd5555555555555),
        f64::from_bits(0xbc755553bf3a2abe),
    ],
    [
        f64::from_bits(0x3fc999999999999a),
        f64::from_bits(0xbc699deed1ec9071),
    ],
    [
        f64::from_bits(0xbfc2492492492492),
        f64::from_bits(0xbc5fd99c8d18269a),
    ],
    [
        f64::from_bits(0x3fbc71c71c71c717),
        f64::from_bits(0xbc2651eee4c4d9d0),
    ],
    [
        f64::from_bits(0xbfb745d1745d1649),
        f64::from_bits(0xbc5632683d6c44a6),
    ],
    [
        f64::from_bits(0x3fb3b13b13b11c63),
        f64::from_bits(0x3c5bf69c1f8af41d),
    ],
    [
        f64::from_bits(0xbfb11111110e6338),
        f64::from_bits(0x3c23c3e431e8bb68),
    ],
    [
        f64::from_bits(0x3fae1e1e1dc45c4a),
        f64::from_bits(0xbc4be2db05c77bbf),
    ],
    [
        f64::from_bits(0xbfaaf286b8164b4f),
        f64::from_bits(0x3c2a4673491f0942),
    ],
    [
        f64::from_bits(0x3fa86185e9ad4846),
        f64::from_bits(0x3c4e12e32d79fcee),
    ],
    [
        f64::from_bits(0xbfa642c6d5161fae),
        f64::from_bits(0x3c43ce76c1ca03f0),
    ],
    [
        f64::from_bits(0x3fa47ad6f277e5bf),
        f64::from_bits(0xbc3abd8d85bdb714),
    ],
    [
        f64::from_bits(0xbfa2f64a2ee8896d),
        f64::from_bits(0x3c2ef87d4b615323),
    ],
    [
        f64::from_bits(0x3fa1a6a2b31741b5),
        f64::from_bits(0x3c1a5d9d973547ee),
    ],
    [
        f64::from_bits(0xbfa07fbdad65e0a6),
        f64::from_bits(0xbc265ac07f5d35f4),
    ],
    [
        f64::from_bits(0x3f9ee9932a9a5f8b),
        f64::from_bits(0x3c2f8b9623f6f55a),
    ],
    [
        f64::from_bits(0xbf9ce8b5b9584dc6),
        f64::from_bits(0x3c2fe5af96e8ea2d),
    ],
    [
        f64::from_bits(0x3f9ac9cb288087b7),
        f64::from_bits(0xbc3450cdfceaf5ca),
    ],
    [
        f64::from_bits(0xbf984b025351f3e6),
        f64::from_bits(0x3c2579561b0d73da),
    ],
    [
        f64::from_bits(0x3f952f5b8ecdd52b),
        f64::from_bits(0x3c3036bd2c6fba47),
    ],
    [
        f64::from_bits(0xbf9163a8c44909dc),
        f64::from_bits(0x3c318f735ffb9f16),
    ],
    [
        f64::from_bits(0x3f8a400dce3eea6f),
        f64::from_bits(0xbc2c90569c0c1b5c),
    ],
    [
        f64::from_bits(0xbf81caa78ae6db3a),
        f64::from_bits(0xbc24c60f8161ea09),
    ],
    [
        f64::from_bits(0x3f752672453c0731),
        f64::from_bits(0x3c1834efb598c338),
    ],
    [
        f64::from_bits(0xbf65850c5be137cf),
        f64::from_bits(0xbc0445fc150ca7f5),
    ],
    [
        f64::from_bits(0x3f523eb98d22e1ca),
        f64::from_bits(0xbbf388fbaf1d7830),
    ],
    [
        f64::from_bits(0xbf38f4e974a40741),
        f64::from_bits(0x3bd271198a97da34),
    ],
    [
        f64::from_bits(0x3f1a5cf2e9cf76e5),
        f64::from_bits(0xbbb887eb4a63b665),
    ],
    [
        f64::from_bits(0xbef420c270719e32),
        f64::from_bits(0x3b8efd595b27888b),
    ],
    [
        f64::from_bits(0x3ec3ba2d69b51677),
        f64::from_bits(0xbb64fb06829cdfc7),
    ],
    [
        f64::from_bits(0xbe829b7e6f676385),
        f64::from_bits(0xbb2a783b6de718fb),
    ],
];
fn muldd(xh: f64, xl: f64, ch: f64, cl: f64) -> (f64, f64) {
    let ahlh = ch * xl;
    let alhh = cl * xh;
    let ahhh = ch * xh;
    let ahhl = fma(ch, xh, -ahhh) + (alhh + ahlh);
    let ch = ahhh + ahhl;
    (ch, (ahhh - ch) + ahhl)
}
fn polydd(xh: f64, xl: f64, c: &[[f64; 2]]) -> (f64, f64) {
    let [mut ch, mut cl] = c[c.len() - 1];
    for &[hi, lo] in c[..c.len() - 1].iter().rev() {
        (ch, cl) = muldd(xh, xl, ch, cl);
        let th = ch + hi;
        let tl = (hi - th) + ch;
        ch = th;
        cl += tl + lo;
    }
    (ch, cl)
}
fn tiny(y: f32, x: f32) -> f32 {
    let z = y as f64 / x as f64;
    let e = fma(-z, x as f64, y as f64) / (x as f64)
        + (-f64::from_bits(0x3fd5555555555555) * z) * (z * z);
    let mut t = z.to_bits();
    if t & 0xfffffff == 0 {
        t = if z * e > 0.0 {
            t.wrapping_add(1)
        } else {
            t.wrapping_sub(1)
        };
    }
    f64::from_bits(t) as f32
}
/// Two-argument arctangent, rounded to nearest with ties to even.
pub fn atan2f(y: f32, x: f32) -> f32 {
    const PI: f64 = f64::from_bits(0x400921fb54442d18);
    const P2: f64 = f64::from_bits(0x3ff921fb54442d18);
    const PL: f64 = f64::from_bits(0x3c91a62633145c07);
    const OFF: [f64; 8] = [0.0, P2, PI, P2, -0.0, -P2, -PI, -P2];
    const LO: [f64; 8] = [0.0, PL, 2.0 * PL, PL, -0.0, -PL, -2.0 * PL, -PL];
    let ux = x.to_bits();
    let uy = y.to_bits();
    let ax = ux & 0x7fffffff;
    let ay = uy & 0x7fffffff;
    let sign = if uy >> 31 == 0 { 1.0 } else { -1.0 };
    if ay >= 0x7f800000 || ax >= 0x7f800000 {
        if x.is_nan() || y.is_nan() {
            return x + y;
        }
        if x.is_infinite() && y.is_infinite() {
            return (if ux >> 31 != 0 {
                f64::from_bits(0x4002d97c7f3321d2)
            } else {
                P2 * 0.5
            } * sign) as f32;
        }
        if x.is_infinite() {
            return (if ux >> 31 != 0 { PI } else { 0.0 } * sign) as f32;
        }
        return (P2 * sign) as f32;
    }
    if ay == 0 {
        if ax == 0 {
            let i = ((uy >> 31) * 4 + (ux >> 31) * 2) as usize;
            return (if ux >> 31 != 0 {
                OFF[i] + LO[i]
            } else {
                OFF[i]
            }) as f32;
        }
        if ux >> 31 == 0 {
            return (0.0 * sign) as f32;
        }
    }
    let gt = usize::from(ay > ax);
    let i = ((uy >> 31) * 4 + (ux >> 31) * 2) as usize + gt;
    let zx = x as f64;
    let zy = y as f64;
    let m = [0.0, 1.0];
    let z = (m[gt] * zx + m[1 - gt] * zy) / (m[gt] * zy + m[1 - gt] * zx);
    let d = ax as i32 - ay as i32;
    let ratio = if d < 27 << 23 && d > -(27 << 23) {
        let z2 = z * z;
        let z4 = z2 * z2;
        let z8 = z4 * z4;
        let poly = |c: [f64; 7]| {
            ((c[0] + z2 * c[1]) + z4 * (c[2] + z2 * c[3])) + z8 * ((c[4] + z2 * c[5]) + z4 * c[6])
        };
        poly(CN) / poly(CD)
    } else {
        1.0
    };
    let sg = [1.0, -1.0][gt];
    let mut r = (z * sg) * ratio + OFF[i];
    if r.to_bits().wrapping_add(8) & 0xfffffff <= 16 {
        if ay < ax && (ax - ay) >> 23 >= 25 {
            return tiny(y, x);
        }
        let (num, den) = if gt == 0 { (zy, zx) } else { (zx, zy) };
        let zh = num / den;
        let zl = fma(zh, -den, num) / den;
        let (z2h, z2l) = muldd(zh, zl, zh, zl);
        let (ph, pl) = polydd(z2h, z2l, &C);
        let (ph, pl) = muldd(zh * sg, zl * sg, ph, pl);
        let sh = ph + OFF[i];
        let sl = ((OFF[i] - sh) + ph) + pl + LO[i];
        let th = (sh as f32) as f64;
        let mut tm = (sh - th) + sl;
        let eps = f64::from_bits(0x3c30000000000000);
        if th + th * eps == th - th * eps {
            let threshold = f64::from_bits((th.to_bits() & (0x7ff << 52)).wrapping_sub(24 << 52));
            tm *= if tm.abs() > threshold { 1.25 } else { 0.75 };
        }
        r = th + tm;
    }
    r as f32
}

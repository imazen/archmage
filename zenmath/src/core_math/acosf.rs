/* Correctly-rounded arc-cosine function for binary32 value.

Copyright (c) 2023 Alexei Sibidanov.

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
const B: [f64; 16] = [
    f64::from_bits(0x3fefffffffd9ccb8),
    f64::from_bits(0x3fc5555c94838007),
    f64::from_bits(0x3fb32ded4b7c20fa),
    f64::from_bits(0x3fa8566df703309e),
    f64::from_bits(0xbf9980c959bec9a3),
    f64::from_bits(0x3fe56fbb04998344),
    f64::from_bits(0xc01403d8e4c49f52),
    f64::from_bits(0x403b06c3e9f311ea),
    f64::from_bits(0xc059ea97c4e2c21f),
    f64::from_bits(0x407200b8261cc61b),
    f64::from_bits(0xc082274c2799a5c7),
    f64::from_bits(0x408a558a59cc19d3),
    f64::from_bits(0xc08aca4b6a529ff0),
    f64::from_bits(0x408228744703f813),
    f64::from_bits(0xc06d7dbb0b322228),
    f64::from_bits(0x4045c2018c0c0105),
];
const C: [f64; 12] = [
    f64::from_bits(0x3fc555555555529c),
    f64::from_bits(0x3fb333333337e0dd),
    f64::from_bits(0x3fa6db6db3b4465e),
    f64::from_bits(0x3f9f1c72e13ac306),
    f64::from_bits(0x3f96e89cebe06bc4),
    f64::from_bits(0x3f91c6dcf5289094),
    f64::from_bits(0x3f8c6dbbcc7c6315),
    f64::from_bits(0x3f88f8dc2615e996),
    f64::from_bits(0x3f7a5833b7bf15e8),
    f64::from_bits(0x3f943f44ace1665c),
    f64::from_bits(0xbf90fb17df881c73),
    f64::from_bits(0x3fa07520c026b2d6),
];
const D: [f64; 12] = [
    f64::from_bits(0x3ff6a09e667f3bcb),
    f64::from_bits(0x3fbe2b7dddff2db9),
    f64::from_bits(0x3f9b27247ab42dbc),
    f64::from_bits(0x3f802995cc4e0744),
    f64::from_bits(0x3f65ffb0276ec8ea),
    f64::from_bits(0x3f5033885a928dec),
    f64::from_bits(0x3f3911f2be23f8c7),
    f64::from_bits(0x3f24c3c55d2437fd),
    f64::from_bits(0x3f0af477e1d7b461),
    f64::from_bits(0x3f0abd6bdff67dcb),
    f64::from_bits(0xbef1717e86d0fa28),
    f64::from_bits(0x3ef6ff526de46023),
];

fn poly(z: f64, c: &[f64; 12]) -> f64 {
    let z2 = z * z;
    let z4 = z2 * z2;
    let mut c0 = c[0] + z * c[1];
    let c2 = c[2] + z * c[3];
    let mut c4 = c[4] + z * c[5];
    let c6 = c[6] + z * c[7];
    let mut c8 = c[8] + z * c[9];
    let c10 = c[10] + z * c[11];
    c0 += c2 * z2;
    c4 += c6 * z2;
    c8 += z2 * c10;
    c0 += z4 * (c4 + z4 * c8);
    c0
}
/// Correctly rounded arccosine, nearest-even.
pub fn acosf(x: f32) -> f32 {
    let bits = x.to_bits();
    let ax = bits << 1;
    if ax >= 0x7f000000 {
        return if x == 1.0 {
            0.0
        } else if x == -1.0 {
            core::f32::consts::PI
        } else if x.is_nan() {
            x + x
        } else {
            f32::NAN
        };
    }
    let xs = f64::from(x);
    if ax < 0x7ec2a1dc {
        if ax < 0x40000000 {
            return core::f32::consts::FRAC_PI_2;
        }
        let z = xs;
        let z2 = z * z;
        let z4 = z2 * z2;
        let z8 = z4 * z4;
        let z16 = z8 * z8;
        let b = B;
        let r = z
            * ((((b[0] + z2 * b[1]) + z4 * (b[2] + z2 * b[3]))
                + z8 * ((b[4] + z2 * b[5]) + z4 * (b[6] + z2 * b[7])))
                + z16
                    * (((b[8] + z2 * b[9]) + z4 * (b[10] + z2 * b[11]))
                        + z8 * ((b[12] + z2 * b[13]) + z4 * (b[14] + z2 * b[15]))));
        let ub = (f64::from_bits(0x3ff921fb54574191) - r) as f32;
        let lb = (f64::from_bits(0x3ff921fb543118a0) - r) as f32;
        if ub == lb {
            return ub;
        }
    }
    let r = if ax < 0x7e000000 {
        if bits == 0x328885a3 {
            return f32::from_bits(0x3fc90fdb);
        }
        if bits == 0x39826222 {
            return f32::from_bits(0x3fc907b5);
        }
        let x2 = xs * xs;
        (core::f64::consts::FRAC_PI_2 - xs) - (xs * x2) * poly(x2, &C)
    } else {
        let z = 1.0 - xs.abs();
        let s = crate::scalar::sqrt(z).copysign(xs);
        let o = if bits >> 31 == 0 {
            0.0
        } else {
            core::f64::consts::PI
        };
        o + s * poly(z, &D)
    };
    r as f32
}

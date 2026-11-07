// Port of CORE-MATH 660067dfd3e751c8f07acc31c899ff410aa7206e.
/* Correctly-rounded hyperbolic tangent function for binary32 value.

Copyright (c) 2022-2025 Alexei Sibidanov.

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

const CN: [f64; 8] = [
    f64::from_bits(0x3ff0000000000000),
    f64::from_bits(0x3fc30877b8b72d33),
    f64::from_bits(0x3f7694aa09ae9e5e),
    f64::from_bits(0x3f14101377abb729),
    f64::from_bits(0x3e9e0392b1db0018),
    f64::from_bits(0x3e12533756e546f7),
    f64::from_bits(0x3d6d62e5abe6ae8a),
    f64::from_bits(0x3c9b06be534182de),
];
const CD: [f64; 8] = [
    f64::from_bits(0x3ff0000000000000),
    f64::from_bits(0x3fded99131b0ebea),
    f64::from_bits(0x3fa0d27ed6c95a69),
    f64::from_bits(0x3f47cbdaca0e9fcc),
    f64::from_bits(0x3edb4e60b892578e),
    f64::from_bits(0x3e5a6f707c5c71ab),
    f64::from_bits(0x3dc35a8b6e2cd94c),
    f64::from_bits(0x3d0ca8230677aa01),
];
/// Correctly rounded hyperbolic tangent (nearest-even).
pub fn tanhf(x: f32) -> f32 {
    let ux = x.to_bits();
    let e = (ux >> 23) & 255;
    if e == 255 {
        return if ux << 9 != 0 {
            x + x
        } else {
            1.0_f32.copysign(x)
        };
    }
    if e < 115 {
        if e < 102 {
            return if ux << 1 == 0 {
                x
            } else {
                crate::reference::fmaf(-x, x.abs(), x)
            };
        }
        let x2 = x * x;
        return crate::reference::fmaf(x, -f32::from_bits(0x3eaaaaab) * x2, x);
    }
    if ux << 1 > 0x41102cb3_u32 << 1 {
        return 1.0_f32.copysign(x) - f32::from_bits(0x33000000).copysign(x);
    }
    let z = x as f64;
    let z2 = z * z;
    let z4 = z2 * z2;
    let z8 = z4 * z4;
    let n0 = CN[0] + z2 * CN[1];
    let n2 = CN[2] + z2 * CN[3];
    let n4 = CN[4] + z2 * CN[5];
    let n6 = CN[6] + z2 * CN[7];
    let d0 = CD[0] + z2 * CD[1];
    let d2 = CD[2] + z2 * CD[3];
    let d4 = CD[4] + z2 * CD[5];
    let d6 = CD[6] + z2 * CD[7];
    (z * ((n0 + z4 * n2) + z8 * (n4 + z4 * n6)) / ((d0 + z4 * d2) + z8 * (d4 + z4 * d6))) as f32
}

// Port of CORE-MATH 660067dfd3e751c8f07acc31c899ff410aa7206e.
/* Correctly-rounded arc-tangent of binary32 value.

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

const CN: [f64; 7] = [
    f64::from_bits(0x3fd51eccde075d67),
    f64::from_bits(0x3fea76bb5637f2f2),
    f64::from_bits(0x3fe81e0eed20de88),
    f64::from_bits(0x3fd376c8ca67d11d),
    f64::from_bits(0x3faaec7b69202ac6),
    f64::from_bits(0x3f69561899acc73e),
    f64::from_bits(0x3efbf9fa5b67e600),
];
const CD: [f64; 7] = [
    f64::from_bits(0x3fd51eccde075d66),
    f64::from_bits(0x3fedfbdd7b392d28),
    f64::from_bits(0x3ff0000000000000),
    f64::from_bits(0x3fdfd22bf0e89b54),
    f64::from_bits(0x3fbd91ff8b576282),
    f64::from_bits(0x3f8653ea99fc9bb0),
    f64::from_bits(0x3f31e7fcc202340a),
];
/// Correctly rounded inverse tangent (nearest-even).
pub fn atanf(x: f32) -> f32 {
    let t = x.to_bits();
    let e = (t >> 23) & 255;
    let gt = e >= 127;
    let ta = t & 0x7fffffff;
    if ta >= 0x4c700518 {
        return if ta > 0x7f800000 {
            x + x
        } else {
            f64::from_bits(0x3ff921fb54442d18).copysign(x as f64) as f32
        };
    }
    if e < 114 {
        if e < 102 {
            return if t << 1 == 0 {
                x
            } else {
                crate::reference::fmaf(-x, x.abs(), x)
            };
        }
        return crate::reference::fmaf(-f32::from_bits(0x3eaaaaab) * x, x * x, x);
    }
    let z = if gt { 1.0 / (x as f64) } else { x as f64 };
    let z2 = z * z;
    let z4 = z2 * z2;
    let z8 = z4 * z4;
    let cn0 = CN[0] + z2 * CN[1];
    let cn2 = CN[2] + z2 * CN[3];
    let cn4 = CN[4] + z2 * CN[5];
    let cd0 = CD[0] + z2 * CD[1];
    let cd2 = CD[2] + z2 * CD[3];
    let cd4 = CD[4] + z2 * CD[5];
    let r = ((cn0 + z4 * cn2) + z8 * (cn4 + z4 * CN[6])) * z
        / ((cd0 + z4 * cd2) + z8 * (cd4 + z4 * CD[6]));
    if !gt {
        return r as f32;
    }
    ((f64::from_bits(0x3f80fdaa22168c23).copysign(z) - r) + 1.5625_f64.copysign(z)) as f32
}

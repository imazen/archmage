/* Correctly-rounded tangent of binary32 value.

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
const CN: [f64; 4] = [
    f64::from_bits(0x3ff921fb54442d18),
    f64::from_bits(0xbfdfd226e573289f),
    f64::from_bits(0x3f9b7a60c8dac9f6),
    f64::from_bits(0xbf2725beb40f33e5),
];
const CD: [f64; 4] = [
    f64::from_bits(0x3ff0000000000000),
    f64::from_bits(0xbff2395347fb829d),
    f64::from_bits(0x3fc2313660f29c36),
    f64::from_bits(0xbf69a707ab98d1c1),
];
const HARD: [(f32, f32, f32); 8] = [
    (
        f32::from_bits(0x3f8a1f62),
        f32::from_bits(0x3feefcfb),
        f32::from_bits(0xa5c48e92),
    ),
    (
        f32::from_bits(0x4d56d355),
        f32::from_bits(0x3e740182),
        f32::from_bits(0x22a0cfa3),
    ),
    (
        f32::from_bits(0x57d7b0ed),
        f32::from_bits(0x3eb068e4),
        f32::from_bits(0xa416b61d),
    ),
    (
        f32::from_bits(0x5980445e),
        f32::from_bits(0x3fe50f68),
        f32::from_bits(0x257b0298),
    ),
    (
        f32::from_bits(0x63fc86fe),
        f32::from_bits(0x3f2cbfce),
        f32::from_bits(0x25492cbc),
    ),
    (
        f32::from_bits(0x6a662711),
        f32::from_bits(0xc0799ac2),
        f32::from_bits(0x266b92a5),
    ),
    (
        f32::from_bits(0x6ad36709),
        f32::from_bits(0xbf62b097),
        f32::from_bits(0xa513619f),
    ),
    (
        f32::from_bits(0x72b505bb),
        f32::from_bits(0xbff2150f),
        f32::from_bits(0xa58ee483),
    ),
];
/// Tangent, rounded to nearest with ties to even.
pub fn tanf(x: f32) -> f32 {
    let u = x.to_bits();
    let e = (u >> 23) & 255;
    let (z, i) = if e < 155 {
        if e < 115 {
            if e < 102 {
                return crate::reference::fmaf(x, x.abs(), x);
            }
            let x2 = x * x;
            return crate::reference::fmaf(x, f32::from_bits(0x3eaaaaab) * x2, x);
        }
        let hi = f64::from_bits(0x3fe45f306e000000) * x as f64;
        let lo = -f64::from_bits(0x3dfb1bbead603d8b) * x as f64;
        let id = crate::reference::roundeven(hi);
        ((hi - id) + lo, id as i64 as i32)
    } else if e < 255 {
        super::trig_reduce::big_scaled(u, 150)
    } else {
        return if x.is_nan() { x + x } else { f32::NAN };
    };
    let z2 = z * z;
    let z4 = z2 * z2;
    let n = ((CN[0] + z2 * CN[1]) + z4 * (CN[2] + z2 * CN[3])) * z;
    let d = (CD[0] + z2 * CD[1]) + z4 * (CD[2] + z2 * CD[3]);
    let s0 = (i & 1) as f64;
    let s1 = (1 - (i & 1)) as f64;
    let r = (n * s1 - d * s0) / (n * s0 + d * s1);
    if r.to_bits().wrapping_add(7) & ((1 << 29) - 1) <= 14 {
        for (arg, hi, lo) in HARD {
            if u & 0x7fffffff == arg.to_bits() {
                return if u >> 31 != 0 { -hi - lo } else { hi + lo };
            }
        }
    }
    r as f32
}

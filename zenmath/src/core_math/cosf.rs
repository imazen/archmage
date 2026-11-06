/* Correctly-rounded cosine of binary32 value.

Copyright (c) 2022-2023 Alexei Sibidanov.

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

const B: [f64; 4] = [
    f64::from_bits(0x3f93bd3cc9be45dc),
    f64::from_bits(0xbf103c1f081b0833),
    f64::from_bits(0x3e755d3c6fc9ac1f),
    f64::from_bits(0xbdce1d3ff281b40d),
];
const A: [f64; 4] = [
    f64::from_bits(0x3fc921fb54442d17),
    f64::from_bits(0xbf54abbce6256a39),
    f64::from_bits(0x3ec466bc5a518c16),
    f64::from_bits(0xbe232bdc61074ff6),
];
const TB: [f64; 32] = [
    f64::from_bits(0x3ff0000000000000),
    f64::from_bits(0x3fef6297cff75cb0),
    f64::from_bits(0x3fed906bcf328d46),
    f64::from_bits(0x3fea9b66290ea1a3),
    f64::from_bits(0x3fe6a09e667f3bcd),
    f64::from_bits(0x3fe1c73b39ae68c8),
    f64::from_bits(0x3fd87de2a6aea963),
    f64::from_bits(0x3fc8f8b83c69a60b),
    f64::from_bits(0x0000000000000000),
    f64::from_bits(0xbfc8f8b83c69a60b),
    f64::from_bits(0xbfd87de2a6aea963),
    f64::from_bits(0xbfe1c73b39ae68c8),
    f64::from_bits(0xbfe6a09e667f3bcd),
    f64::from_bits(0xbfea9b66290ea1a3),
    f64::from_bits(0xbfed906bcf328d46),
    f64::from_bits(0xbfef6297cff75cb0),
    f64::from_bits(0xbff0000000000000),
    f64::from_bits(0xbfef6297cff75cb0),
    f64::from_bits(0xbfed906bcf328d46),
    f64::from_bits(0xbfea9b66290ea1a3),
    f64::from_bits(0xbfe6a09e667f3bcd),
    f64::from_bits(0xbfe1c73b39ae68c8),
    f64::from_bits(0xbfd87de2a6aea963),
    f64::from_bits(0xbfc8f8b83c69a60b),
    f64::from_bits(0x0000000000000000),
    f64::from_bits(0x3fc8f8b83c69a60b),
    f64::from_bits(0x3fd87de2a6aea963),
    f64::from_bits(0x3fe1c73b39ae68c8),
    f64::from_bits(0x3fe6a09e667f3bcd),
    f64::from_bits(0x3fea9b66290ea1a3),
    f64::from_bits(0x3fed906bcf328d46),
    f64::from_bits(0x3fef6297cff75cb0),
];
const HARD: [[f32; 3]; 5] = [
    [
        f32::from_bits(0x4096cbe4),
        f32::from_bits(0x324cde2e),
        f32::from_bits(0xa5800000),
    ],
    [
        f32::from_bits(0x5922aa80),
        f32::from_bits(0x3f08aebf),
        f32::from_bits(0xb2800000),
    ],
    [
        f32::from_bits(0x5aa4542c),
        f32::from_bits(0x3efa40a4),
        f32::from_bits(0x32000000),
    ],
    [
        f32::from_bits(0x5f18b878),
        f32::from_bits(0x3f7f14bb),
        f32::from_bits(0x32800000),
    ],
    [
        f32::from_bits(0x6115cb11),
        f32::from_bits(0x3f78142f),
        f32::from_bits(0xb2800000),
    ],
];
fn hard(x: f32, r: f64) -> f32 {
    for &[arg, hi, lo] in &HARD {
        if x.abs().to_bits() == arg.to_bits() {
            return hi + lo;
        }
    }
    r as f32
}
/// Correctly rounded cosine (nearest-even), with full-range integer reduction.
pub fn cosf(x: f32) -> f32 {
    let ax = x.to_bits() << 1;
    if ax < 0x74000000 {
        return 0.5 * crate::scalar::fmaf(x, -x, 2.0);
    }
    if ax >= 0xff000000 {
        return if x.is_nan() { x + x } else { f32::NAN };
    }
    let big = ax > 0x99000000;
    let (z, ia) = if big {
        super::trig_reduce::big(x.to_bits())
    } else {
        if ax == 0x812d97c8 {
            return hard(x, 0.0);
        }
        super::trig_reduce::small(x as f64, ax < 0x82a41896)
    };
    let z2 = z * z;
    let z4 = z2 * z2;
    let aa = (A[0] + z2 * A[1]) + z4 * (A[2] + z2 * A[3]);
    let bb = (B[0] + z2 * B[1]) + z4 * (B[2] + z2 * B[3]);
    let c0 = TB[(ia & 31) as usize];
    let s0 = TB[(ia.wrapping_add(8) & 31) as usize];
    if big {
        let r = c0 + z * (aa * s0 - bb * (z * c0));
        if (r.to_bits().wrapping_add(6) & 0xfffffff) <= 12 {
            return hard(x, r);
        }
        r as f32
    } else {
        (c0 + aa * (z * s0) - bb * (z2 * c0)) as f32
    }
}

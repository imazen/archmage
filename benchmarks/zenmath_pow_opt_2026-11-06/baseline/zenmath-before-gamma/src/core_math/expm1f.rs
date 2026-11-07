/* Correctly-rounded exponential function biased by 1 for binary32 value.

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

const C: [f64; 4] = [
    f64::from_bits(0x3ff0000000000000),
    f64::from_bits(0x3f962e42fef4c4e7),
    f64::from_bits(0x3f2ebfd1b232f475),
    f64::from_bits(0x3ebc6b19384ecd93),
];
const CH: [f64; 6] = [
    f64::from_bits(0x3f962e42fefa39ef),
    f64::from_bits(0x3f2ebfbdff82c58f),
    f64::from_bits(0x3ebc6b08d702e0ed),
    f64::from_bits(0x3e43b2ab6fb92e5e),
    f64::from_bits(0x3dc5d886e6d54203),
    f64::from_bits(0x3d4430976b8ce6ef),
];
const TD: [f64; 32] = [
    f64::from_bits(0x3ff0000000000000),
    f64::from_bits(0x3ff059b0d3158574),
    f64::from_bits(0x3ff0b5586cf9890f),
    f64::from_bits(0x3ff11301d0125b51),
    f64::from_bits(0x3ff172b83c7d517b),
    f64::from_bits(0x3ff1d4873168b9aa),
    f64::from_bits(0x3ff2387a6e756238),
    f64::from_bits(0x3ff29e9df51fdee1),
    f64::from_bits(0x3ff306fe0a31b715),
    f64::from_bits(0x3ff371a7373aa9cb),
    f64::from_bits(0x3ff3dea64c123422),
    f64::from_bits(0x3ff44e086061892d),
    f64::from_bits(0x3ff4bfdad5362a27),
    f64::from_bits(0x3ff5342b569d4f82),
    f64::from_bits(0x3ff5ab07dd485429),
    f64::from_bits(0x3ff6247eb03a5585),
    f64::from_bits(0x3ff6a09e667f3bcd),
    f64::from_bits(0x3ff71f75e8ec5f74),
    f64::from_bits(0x3ff7a11473eb0187),
    f64::from_bits(0x3ff82589994cce13),
    f64::from_bits(0x3ff8ace5422aa0db),
    f64::from_bits(0x3ff93737b0cdc5e5),
    f64::from_bits(0x3ff9c49182a3f090),
    f64::from_bits(0x3ffa5503b23e255d),
    f64::from_bits(0x3ffae89f995ad3ad),
    f64::from_bits(0x3ffb7f76f2fb5e47),
    f64::from_bits(0x3ffc199bdd85529c),
    f64::from_bits(0x3ffcb720dcef9069),
    f64::from_bits(0x3ffd5818dcfba487),
    f64::from_bits(0x3ffdfc97337b9b5f),
    f64::from_bits(0x3ffea4afa2a490da),
    f64::from_bits(0x3fff50765b6e4540),
];
const B: [f64; 8] = [
    f64::from_bits(0x3fdfffffffffffc2),
    f64::from_bits(0x3fc55555555555fe),
    f64::from_bits(0x3fa555555559767f),
    f64::from_bits(0x3f81111111098dc1),
    f64::from_bits(0x3f56c16bca988aa9),
    f64::from_bits(0x3f2a01a07658483f),
    f64::from_bits(0x3efa05b04d2c3503),
    f64::from_bits(0x3ec71de3a960b5e3),
];
/// Correctly rounded exp(x)-1, avoiding cancellation near zero (nearest-even).
pub fn expm1f(x: f32) -> f32 {
    let ux = x.to_bits();
    let ax = ux << 1;
    let z = x as f64;
    if ax < 0x7c400000 {
        if ax < 0x676a09e8 {
            return if ax == 0 {
                x
            } else {
                crate::reference::fmaf(x.abs(), f32::from_bits(0x33000000), x)
            };
        }
        let z2 = z * z;
        let z4 = z2 * z2;
        return (z + z2
            * ((B[0] + z * B[1])
                + z2 * (B[2] + z * B[3])
                + z4 * ((B[4] + z * B[5]) + z2 * (B[6] + z * B[7])))) as f32;
    }
    if ax >= 0x8562e430 {
        return if x.is_nan() {
            x + x
        } else if x.is_sign_negative() {
            -1.0
        } else {
            f32::INFINITY
        };
    }
    let a = f64::from_bits(0x40471547652b82fe) * z;
    let ia = crate::reference::roundeven(a);
    let mut h = a - ia;
    let h2 = h * h;
    let u = (ia + f64::from_bits(0x4338000000000000)).to_bits();
    let c2 = C[2] + h * C[3];
    let c0 = C[0] + h * C[1];
    let s = f64::from_bits(TD[(u & 31) as usize].to_bits().wrapping_add((u >> 5) << 52));
    let r = (c0 + h2 * c2) * s - 1.0;
    let ub = r as f32;
    let lb = (r - s * f64::from_bits(0x3de3b30000000000)) as f32;
    if ub == lb {
        return ub;
    }
    if ux > 0xc18aa123 {
        return -1.0;
    }
    h = (f64::from_bits(0x4047154765000000) * z - ia) + f64::from_bits(0x3e55c17f0bbbe880) * z;
    let h2 = h * h;
    let w = s * h;
    ((s - 1.0) + w * ((CH[0] + h * CH[1]) + h2 * ((CH[2] + h * CH[3]) + h2 * (CH[4] + h * CH[5]))))
        as f32
}

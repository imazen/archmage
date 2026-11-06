/* Correctly-rounded biased argument natural logarithm function for binary32 value.

Copyright (c) 2023-2026 Alexei Sibidanov <sibid@uvic.ca>.

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

const X0: [f64; 32] = [
    f64::from_bits(0x3fef81f800000000),
    f64::from_bits(0x3fee9131c0000000),
    f64::from_bits(0x3fedae6080000000),
    f64::from_bits(0x3fecd85680000000),
    f64::from_bits(0x3fec0e0700000000),
    f64::from_bits(0x3feb4e81c0000000),
    f64::from_bits(0x3fea98ef80000000),
    f64::from_bits(0x3fe9ec8e80000000),
    f64::from_bits(0x3fe948b100000000),
    f64::from_bits(0x3fe8acb900000000),
    f64::from_bits(0x3fe8181800000000),
    f64::from_bits(0x3fe78a4c80000000),
    f64::from_bits(0x3fe702e040000000),
    f64::from_bits(0x3fe6816800000000),
    f64::from_bits(0x3fe6058180000000),
    f64::from_bits(0x3fe58ed240000000),
    f64::from_bits(0x3fe51d0800000000),
    f64::from_bits(0x3fe4afd6c0000000),
    f64::from_bits(0x3fe446f880000000),
    f64::from_bits(0x3fe3e22cc0000000),
    f64::from_bits(0x3fe3813800000000),
    f64::from_bits(0x3fe323e340000000),
    f64::from_bits(0x3fe2c9fb40000000),
    f64::from_bits(0x3fe27350c0000000),
    f64::from_bits(0x3fe21fb780000000),
    f64::from_bits(0x3fe1cf06c0000000),
    f64::from_bits(0x3fe1811800000000),
    f64::from_bits(0x3fe135c800000000),
    f64::from_bits(0x3fe0ecf580000000),
    f64::from_bits(0x3fe0a68100000000),
    f64::from_bits(0x3fe0624dc0000000),
    f64::from_bits(0x3fe0204080000000),
];
const LIX: [f64; 32] = [
    f64::from_bits(0x3f8fc0b0b1599ce4),
    f64::from_bits(0x3fa77457a64a42ab),
    f64::from_bits(0x3fb341d74627847d),
    f64::from_bits(0x3fba926d8a568810),
    f64::from_bits(0x3fc0d77e8cd667aa),
    f64::from_bits(0x3fc44d2b38d15679),
    f64::from_bits(0x3fc7ab886a16b2b3),
    f64::from_bits(0x3fcaf3c9b686996d),
    f64::from_bits(0x3fce27075e30cc37),
    f64::from_bits(0x3fd0a3250a767d98),
    f64::from_bits(0x3fd229423bd2662e),
    f64::from_bits(0x3fd3a64c596c3292),
    f64::from_bits(0x3fd51aadd530e505),
    f64::from_bits(0x3fd686c85e9e0177),
    f64::from_bits(0x3fd7eaf7df859cae),
    f64::from_bits(0x3fd94793ee2403b3),
    f64::from_bits(0x3fda9cec5a9cf512),
    f64::from_bits(0x3fdbeb4d3baa086f),
    f64::from_bits(0x3fdd32fe2a03d8b5),
    f64::from_bits(0x3fde744257d97431),
    f64::from_bits(0x3fdfaf58cf7bdfe7),
    f64::from_bits(0x3fe0723e6d1e5599),
    f64::from_bits(0x3fe109f3b52ec2f3),
    f64::from_bits(0x3fe19ee6a7693fc5),
    f64::from_bits(0x3fe23130d9c03597),
    f64::from_bits(0x3fe2c0e9cc4604f1),
    f64::from_bits(0x3fe34e28bd9e5837),
    f64::from_bits(0x3fe3d9028a72cd5f),
    f64::from_bits(0x3fe4618b9c1dd52d),
    f64::from_bits(0x3fe4e7d825b8d20b),
    f64::from_bits(0x3fe56bf9fab56a02),
    f64::from_bits(0x3fe5ee02ab258cca),
];
const B: [f64; 8] = [
    f64::from_bits(0x3ff0000000000000),
    f64::from_bits(0xbfe0000000000000),
    f64::from_bits(0x3fd5555555556f6b),
    f64::from_bits(0xbfd00000000029b9),
    f64::from_bits(0x3fc9999988d176e4),
    f64::from_bits(0xbfc55555418889a7),
    f64::from_bits(0x3fc24adeca50e2bc),
    f64::from_bits(0xbfc001ba33bf57cf),
];
const C: [f64; 4] = [
    f64::from_bits(0x3feffffffe1eac82),
    f64::from_bits(0xbfdffffff7da1724),
    f64::from_bits(0x3fd5564d8fa59d0c),
    f64::from_bits(0xbfd001219d3dba2a),
];
/// Correctly rounded log(1+x), avoiding cancellation near zero (nearest-even).
pub fn log1pf(x: f32) -> f32 {
    if x == -1.0 {
        return f32::NEG_INFINITY;
    }
    if x < -1.0 {
        return f32::NAN;
    }
    if !x.is_finite() {
        return x + x;
    }
    let ax = x.to_bits() & 0x7fffffff;
    let mut z = x as f64;
    if ax < 0x3c880000 {
        if ax < 0x33000000 {
            return if ax == 0 {
                x
            } else {
                crate::reference::fmaf(x, -x, x)
            };
        }
        let z2 = z * z;
        let z4 = z2 * z2;
        let f =
            z2 * ((B[1] + z * B[2]) + z2 * (B[3] + z * B[4]) + z4 * (B[5] + z * (B[6] + z * B[7])));
        let mut r = z + f;
        if r.to_bits() & 0xfffffff == 0 {
            r += 16384.0 * (f + (z - r));
        }
        return r as f32;
    }
    let tp = (z + 1.0).to_bits();
    let e = ((tp >> 52) as i32 - 1023) as f64;
    let m = tp & 0xfffffffffffff;
    let j = ((tp >> 47) & 31) as usize;
    z = f64::from_bits(m | (0x3ff_u64 << 52)) * X0[j] - 1.0;
    let z2 = z * z;
    let r = (f64::from_bits(0x3fe62e42fefa39ef) * e + LIX[j])
        + z * ((C[0] + z * C[1]) + z2 * (C[2] + z * C[3]));
    let ub = r as f32;
    let lb = (r - 2.1555e-11) as f32;
    if ub == lb {
        return ub;
    }
    let z4 = z2 * z2;
    let f = z2 * ((B[1] + z * B[2]) + z2 * (B[3] + z * B[4]) + z4 * (B[5] + z * (B[6] + z * B[7])));
    let lj = LIX[j] - f64::from_bits(0x3da7654000000000);
    let lh = f64::from_bits(0x3fe62e4000000000) * e;
    let ll = f64::from_bits(0x3eb7f7d1cf79abca) * e + z;
    let rh = lh + lj;
    let rl = ((lh - rh) + lj) + (ll + f);
    let fh = (rh + rl) as f32;
    let low = (rh - fh as f64) + rl;
    let mut fl = low as f32;
    let tfl = fl * 2.0;
    if (fh + tfl) - fh == tfl {
        fl += 0.5_f32.copysign((low - fl as f64) as f32) * fl.abs();
    }
    fh + fl
}

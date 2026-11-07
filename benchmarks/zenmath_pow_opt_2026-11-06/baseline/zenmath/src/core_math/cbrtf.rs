// Port of CORE-MATH 660067dfd3e751c8f07acc31c899ff410aa7206e.
/* Correctly-rounded cubic root of binary32 value.

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

const ESCALE: [f64; 3] = [
    1.0,
    f64::from_bits(0x3ff428a2f98d728b),
    f64::from_bits(0x3ff965fea53d6e3d),
];
const C: [f64; 8] = [
    f64::from_bits(0x3fe2319d352ea5d5),
    f64::from_bits(0x3fe67ad8ee258d1a),
    f64::from_bits(0xbfd9342edf9cbad9),
    f64::from_bits(0x3fcb6388fc510a75),
    f64::from_bits(0xbfb6002455599e2f),
    f64::from_bits(0x3f97b096936192c4),
    f64::from_bits(0xbf6e5577187e8bf8),
    f64::from_bits(0x3f3169ef81d6c34e),
];

/// Correctly rounded cube root (nearest-even), preserving signed zero.
pub fn cbrtf(x: f32) -> f32 {
    let u = x.to_bits();
    let mut au = u.wrapping_shl(1);
    let sgn = u >> 31;
    let mut e = au >> 24;
    if !(1 << 24..0xff000000).contains(&au) {
        if au >= 0xff000000 {
            return x + x;
        }
        if au == 0 {
            return x;
        }
        let nz = au.leading_zeros() - 7;
        au <<= nz;
        e = e.wrapping_sub(nz - 1);
    }
    let mant = au & 0xffffff;
    let z = f64::from_bits(((mant as u64) << 28) | (0x3ff_u64 << 52));
    e = e.wrapping_add(899);
    let et = e / 3;
    let it = (e % 3) as usize;
    let scale = f64::from_bits(
        ESCALE[it]
            .to_bits()
            .wrapping_add((et.wrapping_sub(342) as u64) << 52)
            | ((sgn as u64) << 63),
    );
    let r0 = -f64::from_bits(0x3f99931c6c2d19d1) / z;
    let z2 = z * z;
    let z4 = z2 * z2;
    let mut f = ((C[0] + z * C[1]) + z2 * (C[2] + z * C[3]))
        + z4 * ((C[4] + z * C[5]) + z2 * (C[6] + z * C[7]))
        + r0;
    let r = f * scale;
    let ub = r as f32;
    let lb = (r - scale * 1.4182e-9) as f32;
    if ub == lb {
        return ub;
    }
    let h = f * f * f - z;
    f -= (f * r0 * (-f64::from_bits(0x402ab16ec65d138f))) * h;
    let r = f * scale;
    let m0 = (r.to_bits() << 19) as i64;
    let m1 = m0 >> 63;
    if (m0 ^ m1) < (1_i64 << 31) {
        return f64::from_bits(r.to_bits().wrapping_add(1 << 31) & 0xffffffff00000000) as f32;
    }
    r as f32
}

/* Correctly-rounded cubic root of binary64 value.

Copyright (c) 2021-2022 Alexei Sibidanov.

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
/// Cube root, rounded to nearest with ties to even, including subnormal inputs.
pub fn cbrt(x: f64) -> f64 {
    const ES: [f64; 3] = [
        1.0,
        f64::from_bits(0x3ff428a2f98d728b),
        f64::from_bits(0x3ff965fea53d6e3d),
    ];
    const C: [f64; 4] = [
        f64::from_bits(0x3fe1b0babccfef9c),
        f64::from_bits(0x3fe2c9a3e94d1da5),
        -f64::from_bits(0x3fc4dc30b1a1ddba),
        f64::from_bits(0x3f97a8d3e4ec9b07),
    ];
    let u0 = f64::from_bits(0x3fd5555555555555);
    let u1 = f64::from_bits(0x3fcc71c71c71c71c);
    let hx = x.to_bits();
    let mut mant = hx & 0xfffffffffffff;
    let sign = hx >> 63;
    let mut e = ((hx >> 52) & 0x7ff) as i32;
    if e == 0 || e == 0x7ff {
        let ix = hx & 0x7fffffffffffffff;
        if e == 0x7ff || ix == 0 {
            return x + x;
        }
        let nz = ix.leading_zeros() - 11;
        mant = (mant << nz) & 0xfffffffffffff;
        e -= nz as i32 - 1;
    }
    e += 3072;
    let et = e / 3;
    let it = (e % 3) as usize;
    let zb = mant | (0x3ff << 52);
    let z = f64::from_bits(zb);
    let zz = f64::from_bits((zb + ((it as u64) << 52)) | (sign << 63));
    let scale = f64::from_bits(ES[it].to_bits() | (sign << 63));
    let r = 1.0 / z;
    let rr = r * [1.0, -1.0, 0.5, -0.5, 0.25, -0.25][it * 2 + sign as usize];
    let z2 = z * z;
    let mut y = (C[0] + z * C[1]) + z2 * (C[2] + z * C[3]);
    let h = (y * y) * (y * r) - 1.0;
    y -= (h * y) * (u0 - u1 * h);
    y *= scale;
    let y2 = y * y;
    let y2l = fma(y, y, -y2);
    let y3 = y2 * y;
    let y3l = fma(y, y2, -y3) + y * y2l;
    let h = ((y3 - zz) + y3l) * rr;
    let mut dy = h * (y * u0);
    let mut y1 = y - dy;
    dy = (y - y1) - dy;
    let near = |dy: f64, eps: f64| {
        (dy.abs() - f64::from_bits(0x3ca0000000000000)).abs() < eps
            || (dy.abs()
                - (f64::from_bits(0x3cb0000000000000) + f64::from_bits(0x3ca0000000000000)))
            .abs()
                < eps
    };
    if near(dy, f64::from_bits(0x3b40000000000000)) {
        let y2 = y1 * y1;
        let y2l = fma(y1, y1, -y2);
        let y3 = y2 * y1;
        let y3l = fma(y1, y2, -y3) + y1 * y2l;
        let h = ((y3 - zz) + y3l) * rr;
        dy = h * (y1 * u0);
        y = y1 - dy;
        dy = (y1 - y) - dy;
        y1 = y;
        if near(dy, f64::from_bits(0x39d0000000000000)) {
            if zz.abs() == f64::from_bits(0x4009b78223aa307c) {
                y1 = f64::from_bits(f64::from_bits(0x3ff79d15d0e8d59c).to_bits() | (sign << 63));
            }
            if zz.abs() == f64::from_bits(0x401a202bfc89ddff) {
                y1 = f64::from_bits(f64::from_bits(0x3ffde87aa837820f).to_bits() | (sign << 63));
            }
        }
    }
    let mut result = y1.to_bits().wrapping_add(((et - 1365) as u64) << 52);
    let m0 = (result << 30) as i64;
    let m1 = m0 >> 63;
    if (m0 ^ m1) as u64 <= 1 << 30 {
        let rounded = f64::from_bits(y1.to_bits().wrapping_add(1 << 15) & 0xffffffffffff0000);
        if ((rounded - y1) - dy).abs() < f64::from_bits(0x3c30000000000000) || zz.abs() == 1.0 {
            result = result.wrapping_add(1 << 15) & 0xffffffffffff0000;
        }
    }
    f64::from_bits(result)
}

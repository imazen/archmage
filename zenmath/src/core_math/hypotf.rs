/* Correctly-rounded Euclidean distance function (hypot) for binary32 values.

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

use super::fma;
/// Correctly rounded hypotenuse, nearest-even with gradual underflow.
pub fn hypotf(x: f32, y: f32) -> f32 {
    let ax = x.abs();
    let ay = y.abs();
    let bx = ax.to_bits();
    let by = ay.to_bits();
    if bx >= 0x7f800000 || by >= 0x7f800000 {
        if (bx > 0x7f800000 && bx & (1 << 22) == 0) || (by > 0x7f800000 && by & (1 << 22) == 0) {
            return x + y;
        }
        if ax == f32::INFINITY || ay == f32::INFINITY {
            return f32::INFINITY;
        }
        return ax + ay;
    }
    let (at, ay) = if ax > ay { (ax, ay) } else { (ay, ax) };
    let xd = f64::from(at);
    let yd = f64::from(ay);
    let x2 = xd * xd;
    let y2 = yd * yd;
    let r2 = x2 + y2;
    if yd < xd * f64::from_bits(0x3f2fffffe0000000) {
        return crate::scalar::fmaf(1.0 / 8192.0, ay, at);
    }
    let r = crate::scalar::sqrt(r2);
    let bits = r.to_bits();
    let c = r as f32;
    if bits > 0x47efffffe0000000 || ((bits + 1) & 0xfffffff) > 2 {
        return c;
    }
    let cd = f64::from(c);
    if (cd * cd - x2) - y2 == 0.0 {
        return c;
    }
    let ir2 = 0.5 / r2;
    let dr2 = (x2 - r2) + y2;
    let rs = r * ir2;
    let dz = dr2 - fma(r, r, -r2);
    let dr = rs * dz;
    let rh = r + dr;
    let rl = dr + (r - rh);
    let mut bits = rh.to_bits();
    if bits & 0xfffffff == 0 {
        if rl > 0.0 {
            bits += 1;
        }
        if rl < 0.0 {
            bits -= 1;
        }
    }
    f64::from_bits(bits) as f32
}

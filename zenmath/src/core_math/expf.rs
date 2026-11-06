// Port of CORE-MATH 660067dfd3e751c8f07acc31c899ff410aa7206e.
/* Correctly-rounded natural exponential function for binary32 value.

Copyright (c) 2023-2025 Alexei Sibidanov.

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

const C: [f64; 6] = [
    f64::from_bits(0x3fe62e42fefa39ef),
    f64::from_bits(0x3fcebfbdff82c58f),
    f64::from_bits(0x3fac6b08d702e0ed),
    f64::from_bits(0x3f83b2ab6fb92e5e),
    f64::from_bits(0x3f55d886e6d54203),
    f64::from_bits(0x3f2430976b8ce6ef),
];
const B: [f64; 4] = [
    1.0,
    f64::from_bits(0x3fe62e42fef4c4e7),
    f64::from_bits(0x3fcebfd1b232f475),
    f64::from_bits(0x3fac6b19384ecd93),
];
const TB: [u64; 64] = [
    0x3ff0000000000000,
    0x3ff02c9a3e778061,
    0x3ff059b0d3158574,
    0x3ff0874518759bc8,
    0x3ff0b5586cf9890f,
    0x3ff0e3ec32d3d1a2,
    0x3ff11301d0125b51,
    0x3ff1429aaea92de0,
    0x3ff172b83c7d517b,
    0x3ff1a35beb6fcb75,
    0x3ff1d4873168b9aa,
    0x3ff2063b88628cd6,
    0x3ff2387a6e756238,
    0x3ff26b4565e27cdd,
    0x3ff29e9df51fdee1,
    0x3ff2d285a6e4030b,
    0x3ff306fe0a31b715,
    0x3ff33c08b26416ff,
    0x3ff371a7373aa9cb,
    0x3ff3a7db34e59ff7,
    0x3ff3dea64c123422,
    0x3ff4160a21f72e2a,
    0x3ff44e086061892d,
    0x3ff486a2b5c13cd0,
    0x3ff4bfdad5362a27,
    0x3ff4f9b2769d2ca7,
    0x3ff5342b569d4f82,
    0x3ff56f4736b527da,
    0x3ff5ab07dd485429,
    0x3ff5e76f15ad2148,
    0x3ff6247eb03a5585,
    0x3ff6623882552225,
    0x3ff6a09e667f3bcd,
    0x3ff6dfb23c651a2f,
    0x3ff71f75e8ec5f74,
    0x3ff75feb564267c9,
    0x3ff7a11473eb0187,
    0x3ff7e2f336cf4e62,
    0x3ff82589994cce13,
    0x3ff868d99b4492ed,
    0x3ff8ace5422aa0db,
    0x3ff8f1ae99157736,
    0x3ff93737b0cdc5e5,
    0x3ff97d829fde4e50,
    0x3ff9c49182a3f090,
    0x3ffa0c667b5de565,
    0x3ffa5503b23e255d,
    0x3ffa9e6b5579fdbf,
    0x3ffae89f995ad3ad,
    0x3ffb33a2b84f15fb,
    0x3ffb7f76f2fb5e47,
    0x3ffbcc1e904bc1d2,
    0x3ffc199bdd85529c,
    0x3ffc67f12e57d14b,
    0x3ffcb720dcef9069,
    0x3ffd072d4a07897c,
    0x3ffd5818dcfba487,
    0x3ffda9e603db3285,
    0x3ffdfc97337b9b5f,
    0x3ffe502ee78b3ff6,
    0x3ffea4afa2a490da,
    0x3ffefa1bee615a27,
    0x3fff50765b6e4540,
    0x3fffa7c1819e90d8,
];

/// Correctly rounded natural exponential (nearest-even).
pub fn expf(x: f32) -> f32 {
    let t = x.to_bits();
    let z = x as f64;
    let a = f64::from_bits(0x3ff71547652b82fe) * z;
    let big = f64::from_bits(0x42d8000000000000);
    let u = (a + big).to_bits();
    let ux = t.wrapping_shl(1);
    if !(0x6f93813e..=0x8562e42e).contains(&ux) {
        if ux < 0x6f93813e {
            return (1.0 + z * (1.0 + z * 0.5)) as f32;
        }
        if ux >= 0xff000000 {
            return if ux > 0xff000000 {
                x + x
            } else if t >> 31 == 0 {
                f32::INFINITY
            } else {
                0.0
            };
        }
        if t > 0xc2ce8ec0 {
            return (f64::from_bits(0x36a0000000000000)
                + (z + f64::from_bits(0x4059d1d9fccf4770)) * f64::from_bits(0x36971547652b82ed))
            .max(f64::from_bits(0x3680000000000000)) as f32;
        }
        if t >> 31 == 0 && t > 0x42b17217 {
            return f32::INFINITY;
        }
    }
    let ia = big - f64::from_bits(u);
    let mut h = a + ia;
    let s = f64::from_bits(TB[(u & 63) as usize].wrapping_add((u >> 6) << 52));
    let h2 = h * h;
    let r = ((B[0] + h * B[1]) + h2 * (B[2] + h * B[3])) * s;
    let ub = r as f32;
    let lb = (r - r * 1.45e-10) as f32;
    if ub == lb {
        return ub;
    }
    h = (f64::from_bits(0x3ff7154765000000) * z + ia) + f64::from_bits(0x3e05c17f0bbbe880) * z;
    let h2 = h * h;
    let w = s * h;
    (s + w * ((C[0] + h * C[1]) + h2 * ((C[2] + h * C[3]) + h2 * (C[4] + h * C[5])))) as f32
}

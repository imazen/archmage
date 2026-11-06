// Research translation: nearest-even results only; no errno/fenv promise.
// Source: CORE-MATH 660067dfd3e751c8f07acc31c899ff410aa7206e, exp2f.c.
/* Correctly-rounded 2^x function for binary32 value.

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

#[inline]
pub fn exp2f(x: f32) -> f32 {
    const TB: [f64; 64] = [
        f64::from_bits(0x3ff0000000000000),
        f64::from_bits(0x3ff02c9a3e778061),
        f64::from_bits(0x3ff059b0d3158574),
        f64::from_bits(0x3ff0874518759bc8),
        f64::from_bits(0x3ff0b5586cf9890f),
        f64::from_bits(0x3ff0e3ec32d3d1a2),
        f64::from_bits(0x3ff11301d0125b51),
        f64::from_bits(0x3ff1429aaea92de0),
        f64::from_bits(0x3ff172b83c7d517b),
        f64::from_bits(0x3ff1a35beb6fcb75),
        f64::from_bits(0x3ff1d4873168b9aa),
        f64::from_bits(0x3ff2063b88628cd6),
        f64::from_bits(0x3ff2387a6e756238),
        f64::from_bits(0x3ff26b4565e27cdd),
        f64::from_bits(0x3ff29e9df51fdee1),
        f64::from_bits(0x3ff2d285a6e4030b),
        f64::from_bits(0x3ff306fe0a31b715),
        f64::from_bits(0x3ff33c08b26416ff),
        f64::from_bits(0x3ff371a7373aa9cb),
        f64::from_bits(0x3ff3a7db34e59ff7),
        f64::from_bits(0x3ff3dea64c123422),
        f64::from_bits(0x3ff4160a21f72e2a),
        f64::from_bits(0x3ff44e086061892d),
        f64::from_bits(0x3ff486a2b5c13cd0),
        f64::from_bits(0x3ff4bfdad5362a27),
        f64::from_bits(0x3ff4f9b2769d2ca7),
        f64::from_bits(0x3ff5342b569d4f82),
        f64::from_bits(0x3ff56f4736b527da),
        f64::from_bits(0x3ff5ab07dd485429),
        f64::from_bits(0x3ff5e76f15ad2148),
        f64::from_bits(0x3ff6247eb03a5585),
        f64::from_bits(0x3ff6623882552225),
        f64::from_bits(0x3ff6a09e667f3bcd),
        f64::from_bits(0x3ff6dfb23c651a2f),
        f64::from_bits(0x3ff71f75e8ec5f74),
        f64::from_bits(0x3ff75feb564267c9),
        f64::from_bits(0x3ff7a11473eb0187),
        f64::from_bits(0x3ff7e2f336cf4e62),
        f64::from_bits(0x3ff82589994cce13),
        f64::from_bits(0x3ff868d99b4492ed),
        f64::from_bits(0x3ff8ace5422aa0db),
        f64::from_bits(0x3ff8f1ae99157736),
        f64::from_bits(0x3ff93737b0cdc5e5),
        f64::from_bits(0x3ff97d829fde4e50),
        f64::from_bits(0x3ff9c49182a3f090),
        f64::from_bits(0x3ffa0c667b5de565),
        f64::from_bits(0x3ffa5503b23e255d),
        f64::from_bits(0x3ffa9e6b5579fdbf),
        f64::from_bits(0x3ffae89f995ad3ad),
        f64::from_bits(0x3ffb33a2b84f15fb),
        f64::from_bits(0x3ffb7f76f2fb5e47),
        f64::from_bits(0x3ffbcc1e904bc1d2),
        f64::from_bits(0x3ffc199bdd85529c),
        f64::from_bits(0x3ffc67f12e57d14b),
        f64::from_bits(0x3ffcb720dcef9069),
        f64::from_bits(0x3ffd072d4a07897c),
        f64::from_bits(0x3ffd5818dcfba487),
        f64::from_bits(0x3ffda9e603db3285),
        f64::from_bits(0x3ffdfc97337b9b5f),
        f64::from_bits(0x3ffe502ee78b3ff6),
        f64::from_bits(0x3ffea4afa2a490da),
        f64::from_bits(0x3ffefa1bee615a27),
        f64::from_bits(0x3fff50765b6e4540),
        f64::from_bits(0x3fffa7c1819e90d8),
    ];
    let t = x.to_bits();
    if t & 0xffff == 0 {
        let k = ((t >> 23) & 255) as i32 - 127;
        if (0..9).contains(&k) && t.wrapping_shl((9 + k) as u32) == 0 {
            let sign = (t as i32) >> 31;
            let m = (((t & 0x7fffff) | (1 << 23)) >> (23 - k)) as i32;
            let m = (m ^ sign) - sign + 127;
            if m > 0 && m < 255 {
                return f32::from_bits((m as u32) << 23);
            }
            if m <= 0 && m > -23 {
                return f32::from_bits(1 << (22 + m));
            }
        }
    }
    let ux = t.wrapping_shl(1);
    if ux >= 0x86000000 || ux < 0x65000000 {
        if ux < 0x65000000 {
            return 1.0 + x;
        }
        if !(0xc3000000..0xc3150000).contains(&t) {
            if ux >= 0xff000000 {
                if ux > 0xff000000 {
                    return x + x;
                }
                return if t >> 31 == 0 { f32::INFINITY } else { 0.0 };
            }
            if t >= 0xc3150000 {
                return (f64::from_bits(0x36a0000000000000)
                    + (x as f64 + 149.0) * f64::from_bits(0x3690000000000000))
                .max(f64::from_bits(0x3680000000000000)) as f32;
            }
            return f32::INFINITY;
        }
    }
    let off = f64::from_bits(0x42d8000000000000);
    // This point admits only finite normal f32 inputs. The encoding conversion
    // is exact and avoids a scalar SSE conversion with a merge dependency.
    let xd = if cfg!(all(target_arch = "x86_64", not(target_feature = "avx"))) {
        f64::from_bits(
            ((t as u64 & 0x80000000) << 32)
                | (((((t >> 23) & 255) as u64) + 896) << 52)
                | (((t & 0x7fffff) as u64) << 29),
        )
    } else {
        x as f64
    };
    let h = xd - ((xd + off) - off);
    let h2 = h * h;
    let u = (x + 196608.0).to_bits();
    let sv = f64::from_bits(
        TB[(u & 63) as usize]
            .to_bits()
            .wrapping_add(((u >> 6) as u64) << 52),
    );
    let mut r = sv
        * ((1.0 + h * f64::from_bits(0x3fe62e42fef4c4e7))
            + h2 * (f64::from_bits(0x3fcebfd1b232f475) + h * f64::from_bits(0x3fac6b19384ecd93)));
    let mut ub = r as f32;
    let lb = (r - r * f64::from_bits(0x3de3d80000000000)) as f32;
    if ub != lb {
        if ux <= 0x79e7526e {
            if t == 0x3b429d37 {
                return f32::from_bits(0x3f804385) - f32::from_bits(0x33000000);
            }
            if t == 0xbcf3a937 {
                return f32::from_bits(0x3f7ac6b1) - f32::from_bits(0x32800000);
            }
            if t == 0xb8d3d026 {
                return f32::from_bits(0x3f7ffb69) + f32::from_bits(0x32800000);
            }
        }
        r = sv
            + (sv * h)
                * ((f64::from_bits(0x3fe62e42fefa39ef) + h * f64::from_bits(0x3fcebfbdff82c58f))
                    + h2 * ((f64::from_bits(0x3fac6b08d702e0ed)
                        + h * f64::from_bits(0x3f83b2ab6fb92e5e))
                        + h2 * (f64::from_bits(0x3f55d886e6d54203)
                            + h * f64::from_bits(0x3f2430976b8ce6ef))));
        ub = r as f32;
    }
    ub
}

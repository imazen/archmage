/* Correctly-rounded power function for binary32 values.

Copyright (c) 2022-2025 Alexei Sibidanov and Paul Zimmermann

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

use super::fma::fma;
const FAST_IX: [f64; 33] = [
    f64::from_bits(0x3ff0000000000000),
    f64::from_bits(0x3fef07c1f07c0000),
    f64::from_bits(0x3fee1e1e1e1e0000),
    f64::from_bits(0x3fed41d41d420000),
    f64::from_bits(0x3fec71c71c720000),
    f64::from_bits(0x3febacf914c20000),
    f64::from_bits(0x3feaf286bca20000),
    f64::from_bits(0x3fea41a41a420000),
    f64::from_bits(0x3fe99999999a0000),
    f64::from_bits(0x3fe8f9c18f9c0000),
    f64::from_bits(0x3fe8618618620000),
    f64::from_bits(0x3fe7d05f417d0000),
    f64::from_bits(0x3fe745d1745d0000),
    f64::from_bits(0x3fe6c16c16c10000),
    f64::from_bits(0x3fe642c8590b0000),
    f64::from_bits(0x3fe5c9882b930000),
    f64::from_bits(0x3fe5555555550000),
    f64::from_bits(0x3fe4e5e0a72f0000),
    f64::from_bits(0x3fe47ae147ae0000),
    f64::from_bits(0x3fe4141414140000),
    f64::from_bits(0x3fe3b13b13b10000),
    f64::from_bits(0x3fe3521cfb2b0000),
    f64::from_bits(0x3fe2f684bda10000),
    f64::from_bits(0x3fe29e4129e40000),
    f64::from_bits(0x3fe2492492490000),
    f64::from_bits(0x3fe1f7047dc10000),
    f64::from_bits(0x3fe1a7b9611a0000),
    f64::from_bits(0x3fe15b1e5f750000),
    f64::from_bits(0x3fe1111111110000),
    f64::from_bits(0x3fe0c9714fbd0000),
    f64::from_bits(0x3fe0842108420000),
    f64::from_bits(0x3fe0410410410000),
    f64::from_bits(0x3fe0000000000000),
];
const FAST_LIX: [[f64; 2]; 33] = [
    [
        f64::from_bits(0x0000000000000000),
        f64::from_bits(0x0000000000000000),
    ],
    [
        f64::from_bits(0xbfa6c00000000000),
        f64::from_bits(0x3f04b229b87f3f89),
    ],
    [
        f64::from_bits(0xbfb6600000000000),
        f64::from_bits(0xbf0fb7d654235799),
    ],
    [
        f64::from_bits(0xbfc0800000000000),
        f64::from_bits(0xbf38b119b2c9c87b),
    ],
    [
        f64::from_bits(0xbfc5c00000000000),
        f64::from_bits(0xbeca39fa6533294d),
    ],
    [
        f64::from_bits(0xbfcac00000000000),
        f64::from_bits(0xbf3ebc5b663dd4b8),
    ],
    [
        f64::from_bits(0xbfcfc00000000000),
        f64::from_bits(0x3f1f4a37fe0fa46f),
    ],
    [
        f64::from_bits(0xbfd2400000000000),
        f64::from_bits(0xbf301eac33103e6b),
    ],
    [
        f64::from_bits(0xbfd4a00000000000),
        f64::from_bits(0x3f361ed0d15725e0),
    ],
    [
        f64::from_bits(0xbfd6e00000000000),
        f64::from_bits(0xbf210e6ceb499ba9),
    ],
    [
        f64::from_bits(0xbfd9200000000000),
        f64::from_bits(0x3f3115db8ada837d),
    ],
    [
        f64::from_bits(0xbfdb400000000000),
        f64::from_bits(0xbf3fafdce266d7ae),
    ],
    [
        f64::from_bits(0xbfdd600000000000),
        f64::from_bits(0xbf3d4f80cd19906f),
    ],
    [
        f64::from_bits(0xbfdf800000000000),
        f64::from_bits(0x3f35ea5ccd0a7396),
    ],
    [
        f64::from_bits(0x3fde800000000000),
        f64::from_bits(0xbf20500d67fe62eb),
    ],
    [
        f64::from_bits(0x3fdc800000000000),
        f64::from_bits(0x3f19dc2d41aa4626),
    ],
    [
        f64::from_bits(0x3fda800000000000),
        f64::from_bits(0x3f4ff2e2ff321344),
    ],
    [
        f64::from_bits(0x3fd8a00000000000),
        f64::from_bits(0x3f4130157f4c3a3e),
    ],
    [
        f64::from_bits(0x3fd6c00000000000),
        f64::from_bits(0x3f461ed0cad929cc),
    ],
    [
        f64::from_bits(0x3fd5000000000000),
        f64::from_bits(0xbf42089a632d7949),
    ],
    [
        f64::from_bits(0x3fd3200000000000),
        f64::from_bits(0x3f47fdc6dfb2d21a),
    ],
    [
        f64::from_bits(0x3fd1600000000000),
        f64::from_bits(0x3f4380a6c36088f3),
    ],
    [
        f64::from_bits(0x3fcf600000000000),
        f64::from_bits(0xbed3ab7dc7ba81ac),
    ],
    [
        f64::from_bits(0x3fcc000000000000),
        f64::from_bits(0xbf1cc2c0061ef1a2),
    ],
    [
        f64::from_bits(0x3fc8a00000000000),
        f64::from_bits(0x3f3130157c97bbe0),
    ],
    [
        f64::from_bits(0x3fc5600000000000),
        f64::from_bits(0x3f1ee14ff34c4128),
    ],
    [
        f64::from_bits(0x3fc2200000000000),
        f64::from_bits(0x3f3b5b854c4fde69),
    ],
    [
        f64::from_bits(0x3fbe000000000000),
        f64::from_bits(0x3f2635d1df7cb0b5),
    ],
    [
        f64::from_bits(0x3fb7e00000000000),
        f64::from_bits(0xbf23f6d2636c101e),
    ],
    [
        f64::from_bits(0x3fb1c00000000000),
        f64::from_bits(0xbf133567f1b193a4),
    ],
    [
        f64::from_bits(0x3fa7800000000000),
        f64::from_bits(0xbf18d66c5313a71d),
    ],
    [
        f64::from_bits(0x3f97400000000000),
        f64::from_bits(0x3eef7430ee200e00),
    ],
    [
        f64::from_bits(0x0000000000000000),
        f64::from_bits(0x0000000000000000),
    ],
];
const FAST_C: [f64; 8] = [
    f64::from_bits(0x3ff71547652b82fe),
    f64::from_bits(0xbfe71547652b82fe),
    f64::from_bits(0x3fdec709dc3a2d0b),
    f64::from_bits(0xbfd71547652bc4a9),
    f64::from_bits(0x3fd2776c441b72e0),
    f64::from_bits(0xbfcec709bdf453ec),
    f64::from_bits(0x3fca6406efd4b877),
    f64::from_bits(0xbfc717d824a520f7),
];
const FAST_CE: [f64; 6] = [
    f64::from_bits(0x3fa62e42fefa398b),
    f64::from_bits(0x3f4ebfbdff84555a),
    f64::from_bits(0x3eec6b08d4ad86d3),
    f64::from_bits(0x3e83b2ad1b1716a2),
    f64::from_bits(0x3e15d7472718ce9d),
    f64::from_bits(0x3da4a1d7f457ac56),
];
const FAST_TB: [f64; 16] = [
    f64::from_bits(0x3ff0000000000000),
    f64::from_bits(0x3ff0b5586cf9890f),
    f64::from_bits(0x3ff172b83c7d517b),
    f64::from_bits(0x3ff2387a6e756238),
    f64::from_bits(0x3ff306fe0a31b715),
    f64::from_bits(0x3ff3dea64c123422),
    f64::from_bits(0x3ff4bfdad5362a27),
    f64::from_bits(0x3ff5ab07dd485429),
    f64::from_bits(0x3ff6a09e667f3bcd),
    f64::from_bits(0x3ff7a11473eb0187),
    f64::from_bits(0x3ff8ace5422aa0db),
    f64::from_bits(0x3ff9c49182a3f090),
    f64::from_bits(0x3ffae89f995ad3ad),
    f64::from_bits(0x3ffc199bdd85529c),
    f64::from_bits(0x3ffd5818dcfba487),
    f64::from_bits(0x3ffea4afa2a490da),
];
const ACCURATE_O: [f64; 2] = [
    f64::from_bits(0x3ff0000000000000),
    f64::from_bits(0x4000000000000000),
];
const ACCURATE_CH: [[f64; 2]; 13] = [
    [
        f64::from_bits(0x40071547652b82fe),
        f64::from_bits(0x3c8777d0ffda2b89),
    ],
    [
        f64::from_bits(0x3feec709dc3a03fd),
        f64::from_bits(0x3c8d27f04ff73b3a),
    ],
    [
        f64::from_bits(0x3fe2776c50ef9bfe),
        f64::from_bits(0x3c8e4b514251d0ec),
    ],
    [
        f64::from_bits(0x3fda61762a7aded9),
        f64::from_bits(0x3c6de632dc7f6998),
    ],
    [
        f64::from_bits(0x3fd484b13d7c02ae),
        f64::from_bits(0x3c7a320ec342ddb3),
    ],
    [
        f64::from_bits(0x3fd0c9a84993fd48),
        f64::from_bits(0xbc6e6425ce9a74a4),
    ],
    [
        f64::from_bits(0x3fcc68f568d8beaf),
        f64::from_bits(0xbc603a175487feab),
    ],
    [
        f64::from_bits(0x3fc89f3b14657dfb),
        f64::from_bits(0x3c6f04a3acf0bcf7),
    ],
    [
        f64::from_bits(0x3fc5b9ad2f2d12a0),
        f64::from_bits(0xbc568fdff6815a6f),
    ],
    [
        f64::from_bits(0x3fc3702165b88acb),
        f64::from_bits(0x3c345b052ace6c8e),
    ],
    [
        f64::from_bits(0x3fc1998f60f2f005),
        f64::from_bits(0xbc679a94f62fb524),
    ],
    [
        f64::from_bits(0x3fbf9bc428e30809),
        f64::from_bits(0xbc451f063387e470),
    ],
    [
        f64::from_bits(0x3fc1ac0ab871296a),
        f64::from_bits(0x3c62ba6a2e1a625b),
    ],
];
const ACCURATE_CE: [[f64; 2]; 18] = [
    [
        f64::from_bits(0x3ff0000000000000),
        f64::from_bits(0x39df7d70599926c4),
    ],
    [
        f64::from_bits(0x3fe62e42fefa39ef),
        f64::from_bits(0x3c7abc9e3b39856b),
    ],
    [
        f64::from_bits(0x3fcebfbdff82c58f),
        f64::from_bits(0xbc65e43a540c283d),
    ],
    [
        f64::from_bits(0x3fac6b08d704a0c0),
        f64::from_bits(0xbc4d3316277451e6),
    ],
    [
        f64::from_bits(0x3f83b2ab6fba4e77),
        f64::from_bits(0x3c14e66003ba7f85),
    ],
    [
        f64::from_bits(0x3f55d87fe78a6731),
        f64::from_bits(0x3bd07183d46a9697),
    ],
    [
        f64::from_bits(0x3f2430912f86c787),
        f64::from_bits(0x3bcbc81afca4c930),
    ],
    [
        f64::from_bits(0x3eeffcbfc588b0c7),
        f64::from_bits(0xbb8e63f6f0116f4c),
    ],
    [
        f64::from_bits(0x3eb62c0223a5c826),
        f64::from_bits(0xbb530542d98ea4a5),
    ],
    [
        f64::from_bits(0x3e7b5253d395e7c6),
        f64::from_bits(0xbaf9285a132ce05e),
    ],
    [
        f64::from_bits(0x3e3e4cf5158b7b01),
        f64::from_bits(0xbac9ac1facae1b88),
    ],
    [
        f64::from_bits(0x3dfe8cac7351a7a8),
        f64::from_bits(0xba44fb82adebd76b),
    ],
    [
        f64::from_bits(0x3dbc3bd65182746d),
        f64::from_bits(0x3a484ad0689d30e0),
    ],
    [
        f64::from_bits(0x3d78161931d765c3),
        f64::from_bits(0xba0254c6535279ce),
    ],
    [
        f64::from_bits(0x3d3314943a26c9e2),
        f64::from_bits(0xb9df4f2fdc14fb82),
    ],
    [
        f64::from_bits(0x3cec36e53b459602),
        f64::from_bits(0x398f0d06a5a63c41),
    ],
    [
        f64::from_bits(0x3ca397637b3876a4),
        f64::from_bits(0xb945632c551ae458),
    ],
    [
        f64::from_bits(0x3c598fbfefdddb51),
        f64::from_bits(0xb8cfd134923d52b4),
    ],
];
#[inline]
fn isint(x: f32) -> bool {
    let b = x.to_bits();
    let e = ((b >> 23) & 255) as i32 - 127;
    if e >= 23 {
        true
    } else if e >= 0 {
        b.wrapping_shl((e + 9) as u32) == 0
    } else {
        b << 1 == 0
    }
}
#[inline]
fn isodd(x: f32) -> bool {
    let b = x.to_bits();
    let e = ((b >> 23) & 255) as i32 - 127;
    (0..=23).contains(&e) && isint(x) && ((b >> (23 - e)) & 1) != 0
}
fn signaling(x: f32) -> bool {
    ((x.to_bits() ^ 0x00400000) & 0x7fffffff) > 0x7fc00000
}
fn muldd(xh: f64, xl: f64, ch: f64, cl: f64) -> (f64, f64) {
    let ahlh = ch * xl;
    let alhh = cl * xh;
    let ahhh = ch * xh;
    let ahhl = fma(ch, xh, -ahhh) + (alhh + ahlh);
    let ch = ahhh + ahhl;
    (ch, (ahhh - ch) + ahhl)
}
fn mulddd(xh: f64, xl: f64, ch: f64) -> (f64, f64) {
    let ahlh = ch * xl;
    let ahhh = ch * xh;
    let ahhl = fma(ch, xh, -ahhh) + ahlh;
    let ch = ahhh + ahhl;
    (ch, (ahhh - ch) + ahhl)
}
#[inline(never)]
fn polydd(xh: f64, xl: f64, c: &[[f64; 2]]) -> (f64, f64) {
    let [mut ch, mut cl] = c[c.len() - 1];
    for &[hi, lo] in c[..c.len() - 1].iter().rev() {
        (ch, cl) = muldd(xh, xl, ch, cl);
        let th = ch + hi;
        let tl = (hi - th) + ch;
        ch = th;
        cl += tl + lo;
    }
    (ch, cl)
}
fn odd_parts(x: f32) -> (u32, i32) {
    let b = x.to_bits();
    let mut m = b & 0x7fffff;
    let mut e = ((b << 1) >> 24) as i32 - 150;
    if e >= -149 {
        m |= 0x800000;
    } else {
        e += 1;
    }
    let t = m.trailing_zeros();
    (m >> t, e + t as i32)
}
fn is_exact(x: f32, y: f32) -> bool {
    let v = x.to_bits();
    let w = y.to_bits();
    if v << 1 != 0x7f000000 && w << 16 != 0 {
        return false;
    }
    if v << 1 == 0x7f000000 {
        return true;
    }
    const XMAX: [u32; 16] = [
        0, 0xffffff, 4095, 255, 63, 27, 15, 9, 7, 5, 5, 3, 3, 3, 3, 3,
    ];
    let (mut m, mut e) = odd_parts(x);
    if y >= 0.0 && isint(y) {
        if y == 0.0 || y == 1.0 {
            return true;
        }
        if m == 1 {
            return y * (e as f32) >= -149.0 && y * (e as f32) < 128.0;
        }
        if y > 15.0 {
            return false;
        }
        let yi = y as i32;
        if m > XMAX[yi as usize] {
            return false;
        }
        let t = 32 - m.leading_zeros() as i32;
        let ez = e * yi + t;
        return ez > -149 && ez <= 128 && e * yi >= -149;
    }
    let (n, mut f) = odd_parts(y);
    if y < 0.0 {
        if m != 1 {
            return false;
        }
        let ez = if f >= 0 {
            -(e as i64)
                .checked_shl(f as u32)
                .unwrap_or(i64::MAX)
                .saturating_mul(n as i64)
        } else {
            if -f > (e as u32).trailing_zeros() as i32 {
                return false;
            }
            ((-e as i64) >> (-f)) * n as i64
        };
        return (-149..128).contains(&ez);
    }
    while f < 0 {
        f += 1;
        if e & 1 != 0 {
            return false;
        }
        e /= 2;
        let dm = m as f32;
        let s = crate::scalar::roundf(crate::scalar::sqrtf(dm));
        if s * s != dm {
            return false;
        }
        m = s as u32;
    }
    if m > 1 && (n > 15 || m > XMAX[n as usize]) {
        return false;
    }
    let my = if m == 1 { 1 } else { m.pow(n) };
    let t = 32 - my.leading_zeros() as i32;
    let en = (e as i64) * (n as i64);
    en >= -149 && en + (t as i64) <= 128
}
/// Correctly rounded power (nearest-even), including negative-base and IEEE rails.
pub fn powf(x0: f32, y0: f32) -> f32 {
    let x = x0 as f64;
    let mut y = y0 as f64;
    let tx = x.to_bits();
    if x.abs() == 1.0 {
        if x < 0.0 {
            if y0.is_nan() {
                return y0 + y0;
            }
            return if isint(y0) {
                if isodd(y0) { x0 } else { -x0 }
            } else {
                f32::NAN
            };
        }
        return if signaling(y0) { x0 + y0 } else { x0 };
    }
    if y == 0.0 {
        return if signaling(x0) { x0 + y0 } else { 1.0 };
    }
    if y == 1.0 {
        return if signaling(x0) { x0 + y0 } else { x0 };
    }
    if !y.is_finite() {
        if x.is_nan() || y.is_nan() {
            return x0 + y0;
        }
        return if (x.abs() < 1.0) ^ y.is_sign_negative() {
            0.0
        } else {
            f32::INFINITY
        };
    }
    if x.is_infinite() {
        let v = if isodd(y0) { x0 } else { x0.abs() };
        return if y < 0.0 { 1.0 / v } else { v };
    }
    if x.is_nan() {
        return x0 + x0;
    }
    if x < 0.0 && !isint(y0) {
        return f32::NAN;
    }
    if x == 0.0 {
        let sign = if isodd(y0) { x0 } else { 0.0 };
        return if y < 0.0 {
            f32::INFINITY.copysign(sign)
        } else {
            sign
        };
    }
    let negative = x.is_sign_negative() && isodd(y0);
    let m = tx & 0x000fffffffffffff;
    let mut e = ((tx >> 52) & 2047) as i32 - 1023;
    let j = ((m + (1_u64 << 46)) >> 47) as usize;
    e += i32::from(j > 13);
    let xd = f64::from_bits(m | (0x3ff_u64 << 52));
    let mut z = fma(xd, FAST_IX[j], -1.0);
    let z2 = z * z;
    let z4 = z2 * z2;
    let c6 = FAST_C[6] + z * FAST_C[7];
    let c4 = FAST_C[4] + z * FAST_C[5];
    let c2 = FAST_C[2] + z * FAST_C[3];
    let c0 = FAST_C[0] + z * FAST_C[1];
    let c0 = (c0 + z2 * c2) + z4 * (c4 + z2 * c6);
    let l = z * c0 - FAST_LIX[j][1];
    y *= 16.0;
    let zt = (e as f64 - FAST_LIX[j][0]) * y;
    z = l * y + zt;
    if z > 2048.0 {
        return if negative {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };
    }
    if z < -2400.0 {
        return if negative { -0.0 } else { 0.0 };
    }
    if z.abs() < f64::from_bits(0x3e50000000000000) {
        return (1.0 + z) as f32;
    }
    let ia = crate::scalar::floor(z);
    let h = fma(l, y, zt - ia);
    let il = ia as i64;
    let jl = (il & 15) as usize;
    let el = (il - jl as i64) >> 4;
    let s = FAST_TB[jl] * f64::from_bits(((el + 1023) as u64) << 52);
    let h2 = h * h;
    let c0 = FAST_CE[0] + h * FAST_CE[1];
    let c2 = FAST_CE[2] + h * FAST_CE[3];
    let c4 = FAST_CE[4] + h * FAST_CE[5];
    let rr = s + (s * h) * (c0 + h2 * (c2 + h2 * c4));
    if (rr.to_bits().wrapping_add(468) & 0xfffffff) <= 936 {
        return accurate(x0, y0, is_exact(x0, y0));
    }
    (if negative { -rr } else { rr }) as f32
}
#[inline(never)]
fn accurate(x0: f32, y0: f32, exact: bool) -> f32 {
    let t = (x0 as f64).to_bits();
    let mut e = ((t >> 52) & 2047) as i32 - 1023;
    let mant = t & 0x000fffffffffffff;
    let k = usize::from(mant > 0x6a09e667f3bcd);
    e += k as i32;
    let x = f64::from_bits(mant | (0x3ff_u64 << 52));
    let y = y0 as f64;
    let xm = x - ACCURATE_O[k];
    let xp = x + ACCURATE_O[k];
    let zh = xm / xp;
    let zl = fma(zh, -xp, xm) / xp;
    let (z2h, z2l) = muldd(zh, zl, zh, zl);
    let (z2h, z2l) = polydd(z2h, z2l, &ACCURATE_CH);
    let (zh, zl) = muldd(zh, zl, z2h, z2l);
    let (zh, zl) = mulddd(zh, zl, y);
    let ey = e as f64 * y;
    let eh = ey + zh;
    let el = ((ey - eh) + zh) + zl;
    let ee = crate::scalar::roundeven(eh);
    let (mut eh, el) = polydd(eh - ee, el, &ACCURATE_CE);
    let mut lh = eh.to_bits();
    if ((!exact && lh & 0xfffffff == 0) || (exact && (lh.wrapping_add(1) & 0xfffffff) <= 2))
        && el.abs() > f64::from_bits(0x3a40000000000000)
    {
        lh = if el < 0.0 {
            lh.wrapping_sub(1)
        } else {
            lh.wrapping_add(1)
        };
        eh = f64::from_bits(lh);
    }
    eh *= f64::from_bits(((1023 + (ee as i64)) as u64) << 52);
    if x0.is_sign_negative() && isodd(y0) {
        eh = -eh;
    }
    eh as f32
}

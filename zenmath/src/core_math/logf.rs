// Port of CORE-MATH 660067dfd3e751c8f07acc31c899ff410aa7206e.
/* Correctly-rounded logarithm function for binary32 value.

Copyright (c) 2023-2026 Alexei Sibidanov and Paul Zimmermann.

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

const TR: [f64; 65] = [
    f64::from_bits(0x3ff0000000000000),
    f64::from_bits(0x3fef81f820000000),
    f64::from_bits(0x3fef07c1f0000000),
    f64::from_bits(0x3fee9131ac000000),
    f64::from_bits(0x3fee1e1e1e000000),
    f64::from_bits(0x3fedae6077000000),
    f64::from_bits(0x3fed41d41d000000),
    f64::from_bits(0x3fecd85689000000),
    f64::from_bits(0x3fec71c71c000000),
    f64::from_bits(0x3fec0e0704000000),
    f64::from_bits(0x3febacf915000000),
    f64::from_bits(0x3feb4e81b5000000),
    f64::from_bits(0x3feaf286bd000000),
    f64::from_bits(0x3fea98ef60000000),
    f64::from_bits(0x3fea41a41a000000),
    f64::from_bits(0x3fe9ec8e95000000),
    f64::from_bits(0x3fe999999a000000),
    f64::from_bits(0x3fe948b0fd000000),
    f64::from_bits(0x3fe8f9c190000000),
    f64::from_bits(0x3fe8acb90f000000),
    f64::from_bits(0x3fe8618618000000),
    f64::from_bits(0x3fe8181818000000),
    f64::from_bits(0x3fe7d05f41000000),
    f64::from_bits(0x3fe78a4c81000000),
    f64::from_bits(0x3fe745d174000000),
    f64::from_bits(0x3fe702e05c000000),
    f64::from_bits(0x3fe6c16c17000000),
    f64::from_bits(0x3fe6816817000000),
    f64::from_bits(0x3fe642c859000000),
    f64::from_bits(0x3fe6058160000000),
    f64::from_bits(0x3fe5c9882c000000),
    f64::from_bits(0x3fe58ed231000000),
    f64::from_bits(0x3fe5555555000000),
    f64::from_bits(0x3fe51d07eb000000),
    f64::from_bits(0x3fe4e5e0a7000000),
    f64::from_bits(0x3fe4afd6a0000000),
    f64::from_bits(0x3fe47ae148000000),
    f64::from_bits(0x3fe446f865000000),
    f64::from_bits(0x3fe4141414000000),
    f64::from_bits(0x3fe3e22cbd000000),
    f64::from_bits(0x3fe3b13b14000000),
    f64::from_bits(0x3fe3813814000000),
    f64::from_bits(0x3fe3521cfb000000),
    f64::from_bits(0x3fe323e34a000000),
    f64::from_bits(0x3fe2f684be000000),
    f64::from_bits(0x3fe2c9fb4e000000),
    f64::from_bits(0x3fe29e412a000000),
    f64::from_bits(0x3fe27350b9000000),
    f64::from_bits(0x3fe2492492000000),
    f64::from_bits(0x3fe21fb781000000),
    f64::from_bits(0x3fe1f7047e000000),
    f64::from_bits(0x3fe1cf06ae000000),
    f64::from_bits(0x3fe1a7b961000000),
    f64::from_bits(0x3fe1811812000000),
    f64::from_bits(0x3fe15b1e5f000000),
    f64::from_bits(0x3fe135c811000000),
    f64::from_bits(0x3fe1111111000000),
    f64::from_bits(0x3fe0ecf56c000000),
    f64::from_bits(0x3fe0c97150000000),
    f64::from_bits(0x3fe0a6810a000000),
    f64::from_bits(0x3fe0842108000000),
    f64::from_bits(0x3fe0624dd3000000),
    f64::from_bits(0x3fe0410410000000),
    f64::from_bits(0x3fe0204081000000),
    0.5,
];
const TL: [f64; 65] = [
    f64::from_bits(0xbd23b40815cd0628),
    f64::from_bits(0x3f8fc0a890fbb514),
    f64::from_bits(0x3f9f829b1e780b98),
    f64::from_bits(0x3fa77458f532c948),
    f64::from_bits(0x3faf0a30c2114ef2),
    f64::from_bits(0x3fb341d793bbc7f7),
    f64::from_bits(0x3fb6f0d28d256172),
    f64::from_bits(0x3fba926d3a6acb89),
    f64::from_bits(0x3fbe2707722ae90c),
    f64::from_bits(0x3fc0d77e7a90896c),
    f64::from_bits(0x3fc29552f6fff036),
    f64::from_bits(0x3fc44d2b6c5b7831),
    f64::from_bits(0x3fc5ff306ee78ee7),
    f64::from_bits(0x3fc7ab890410d41c),
    f64::from_bits(0x3fc9525a9e3451c7),
    f64::from_bits(0x3fcaf3c94ed0bb06),
    f64::from_bits(0x3fcc8ff7c59a9535),
    f64::from_bits(0x3fce27076d5aedf9),
    f64::from_bits(0x3fcfb9186b5e393e),
    f64::from_bits(0x3fd0a324e38b8e6d),
    f64::from_bits(0x3fd1675cacaba398),
    f64::from_bits(0x3fd22941fc0f76ef),
    f64::from_bits(0x3fd2e8e2bc311aba),
    f64::from_bits(0x3fd3a64c56b14373),
    f64::from_bits(0x3fd4618bc31c5c4c),
    f64::from_bits(0x3fd51aad874df5b7),
    f64::from_bits(0x3fd5d1bdbea80754),
    f64::from_bits(0x3fd686c81d331238),
    f64::from_bits(0x3fd739d7f6dbcd90),
    f64::from_bits(0x3fd7eaf83c82ad4d),
    f64::from_bits(0x3fd89a3385813fe4),
    f64::from_bits(0x3fd947941aa91484),
    f64::from_bits(0x3fd9f323edbf95d5),
    f64::from_bits(0x3fda9cec9a4205d3),
    f64::from_bits(0x3fdb44f77c5c8cec),
    f64::from_bits(0x3fdbeb4d9ea71905),
    f64::from_bits(0x3fdc8ff7c69a97ab),
    f64::from_bits(0x3fdd32fe7f38e95f),
    f64::from_bits(0x3fddd46a0501c22a),
    f64::from_bits(0x3fde7442617e8511),
    f64::from_bits(0x3fdf128f5eaf0476),
    f64::from_bits(0x3fdfaf588dd8f0a8),
    f64::from_bits(0x3fe02552a5edcfc4),
    f64::from_bits(0x3fe0723e5c64de05),
    f64::from_bits(0x3fe0be72e3852947),
    f64::from_bits(0x3fe109f39d554b5c),
    f64::from_bits(0x3fe154c3d2c4d4ae),
    f64::from_bits(0x3fe19ee6b38bc834),
    f64::from_bits(0x3fe1e85f5ef03f95),
    f64::from_bits(0x3fe23130d7fabe07),
    f64::from_bits(0x3fe2795e1219afe0),
    f64::from_bits(0x3fe2c0e9ec9c8d50),
    f64::from_bits(0x3fe307d7337f0f83),
    f64::from_bits(0x3fe34e289cb4e098),
    f64::from_bits(0x3fe393e0d42e28de),
    f64::from_bits(0x3fe3d9026ad555bf),
    f64::from_bits(0x3fe41d8fe8667173),
    f64::from_bits(0x3fe4618bc1ec5d87),
    f64::from_bits(0x3fe4a4f85d303d80),
    f64::from_bits(0x3fe4e7d8127f5a75),
    f64::from_bits(0x3fe52a2d26dbc470),
    f64::from_bits(0x3fe56bf9d597f25e),
    f64::from_bits(0x3fe5ad404cb59df2),
    f64::from_bits(0x3fe5ee02a928153a),
    f64::from_bits(0x3fe62e42fefa38b4),
];
const B: [f64; 3] = [
    f64::from_bits(0x3ff00000006342ea),
    f64::from_bits(0xbfe0001f7fdc3977),
    f64::from_bits(0x3fd554a4e5cae9cf),
];
const C: [f64; 7] = [
    f64::from_bits(0xbfe0000000000000),
    f64::from_bits(0x3fd55555555571ca),
    f64::from_bits(0xbfd0000000002d85),
    f64::from_bits(0x3fc9999987d0c963),
    f64::from_bits(0xbfc555554059a8bb),
    f64::from_bits(0x3fc24aebcf71a38f),
    f64::from_bits(0xbfc001c73915d758),
];

/// Correctly rounded natural logarithm (nearest-even).
pub fn logf(x: f32) -> f32 {
    let mut ux = x.to_bits();
    if ux == 0 || ux >= 0x7f800000 {
        return super::log_special(x);
    }
    if ux < 1 << 23 {
        let n = ux.leading_zeros() - 8;
        ux = (ux << n).wrapping_sub(n << 23);
    }
    if ux == 127 << 23 {
        return 0.0;
    }
    let m = ux & 0x7fffff;
    let j = ((m + (1 << 16)) >> 17) as usize;
    let e = (((ux as i32) >> 23) - 127) as f64;
    let z = f64::from_bits(((m as u64) << 29) | (0x3ff_u64 << 52)) * TR[j] - 1.0;
    let z2 = z * z;
    let mut r =
        ((e * f64::from_bits(0x3fe62e42fefa39ef) + TL[j]) + z * B[0]) + z2 * (B[1] + z * B[2]);
    let mut ub = r as f32;
    let lb = (r + f64::from_bits(0x3def060000000000)) as f32;
    if ub != lb {
        let mut f = z2
            * ((C[0] + z * C[1]) + z2 * ((C[2] + z * C[3]) + z2 * (C[4] + z * C[5] + z2 * C[6])));
        if (x - 1.0).abs() < f32::from_bits(0x3a800000) {
            return (z + f) as f32;
        }
        f -= f64::from_bits(0x3ce0ca86c3898d00) * e;
        f += z;
        f += TL[j] - TL[0];
        let el = e * f64::from_bits(0x3fe62e42fefa3a00);
        r = el + f;
        ub = r as f32;
        if r.to_bits() & ((1 << 28) - 1) == 0 {
            let dr = (el - r) + f;
            r += dr * 64.0;
            ub = r as f32;
        }
    }
    ub
}

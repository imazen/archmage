// Port of CORE-MATH 660067dfd3e751c8f07acc31c899ff410aa7206e.
/* Correctly-rounded radix-10 logarithm function for binary32 value.

Copyright (c) 2022-2026 Alexei Sibidanov <sibid@uvic.ca>.

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
    f64::from_bits(0xbd12000000000000),
    f64::from_bits(0x3f7b947689310dfa),
    f64::from_bits(0x3f8b5e909c96d11b),
    f64::from_bits(0x3f945f4f59ed1e08),
    f64::from_bits(0x3f9af5f92cbd8bc1),
    f64::from_bits(0x3fa0ba01a606dcde),
    f64::from_bits(0x3fa3ed119b9a29cd),
    f64::from_bits(0x3fa714834298ed14),
    f64::from_bits(0x3faa30a9d983564c),
    f64::from_bits(0x3fad41d512670665),
    f64::from_bits(0x3fb02428c0f65442),
    f64::from_bits(0x3fb1a23444eecb67),
    f64::from_bits(0x3fb31b30543f4bdd),
    f64::from_bits(0x3fb48f3ed39bfc2d),
    f64::from_bits(0x3fb5fe8049a0e34b),
    f64::from_bits(0x3fb769140a6a9f30),
    f64::from_bits(0x3fb8cf1836c98bdb),
    f64::from_bits(0x3fba30a9d55540ca),
    f64::from_bits(0x3fbb8de4d1ee8167),
    f64::from_bits(0x3fbce6e4202ca20f),
    f64::from_bits(0x3fbe3bc1accacd30),
    f64::from_bits(0x3fbf8c9683b5aafd),
    f64::from_bits(0x3fc06cbd68ca9a03),
    f64::from_bits(0x3fc11142f19df6c5),
    f64::from_bits(0x3fc1b3e71fa7a913),
    f64::from_bits(0x3fc254b4d37a4677),
    f64::from_bits(0x3fc2f3b6912cbe9c),
    f64::from_bits(0x3fc390f68311581a),
    f64::from_bits(0x3fc42c7e7fffc53d),
    f64::from_bits(0x3fc4c65808c78cd1),
    f64::from_bits(0x3fc55e8c50751bea),
    f64::from_bits(0x3fc5f52445dec36c),
    f64::from_bits(0x3fc68a288c3f1195),
    f64::from_bits(0x3fc71da17bdf0cad),
    f64::from_bits(0x3fc7af973608af6e),
    f64::from_bits(0x3fc84011952a250e),
    f64::from_bits(0x3fc8cf1837a7e9f4),
    f64::from_bits(0x3fc95cb2891e436a),
    f64::from_bits(0x3fc9e8e7b0f86974),
    f64::from_bits(0x3fca73beaa5db121),
    f64::from_bits(0x3fcafd3e39455867),
    f64::from_bits(0x3fcb856cf060d985),
    f64::from_bits(0x3fcc0c5134de1f90),
    f64::from_bits(0x3fcc91f1371bc934),
    f64::from_bits(0x3fcd1652ffcd3ee8),
    f64::from_bits(0x3fcd997c6f635e09),
    f64::from_bits(0x3fce1b733ab90ed0),
    f64::from_bits(0x3fce9c3ceadac7eb),
    f64::from_bits(0x3fcf1bdeec43a29a),
    f64::from_bits(0x3fcf9a5e7a5fa392),
    f64::from_bits(0x3fd00be05ac02ef5),
    f64::from_bits(0x3fd04a054d81a29e),
    f64::from_bits(0x3fd087a0835957c5),
    f64::from_bits(0x3fd0c4b4570994e1),
    f64::from_bits(0x3fd101431aa1fe1b),
    f64::from_bits(0x3fd13d4f08b98da2),
    f64::from_bits(0x3fd178da53edb85c),
    f64::from_bits(0x3fd1b3e71e9f9d22),
    f64::from_bits(0x3fd1ee777defdeb7),
    f64::from_bits(0x3fd2288d7b48e205),
    f64::from_bits(0x3fd2622b0f52e469),
    f64::from_bits(0x3fd29b522a4c62de),
    f64::from_bits(0x3fd2d404b0e30f4a),
    f64::from_bits(0x3fd30c4478f3fbaf),
    f64::from_bits(0x3fd34413509f78df),
];
const B: [f64; 3] = [
    f64::from_bits(0x3fdbcb7b15d35067),
    f64::from_bits(0xbfcbcbb1cd29cbaf),
    f64::from_bits(0x3fc2870e2624ce4e),
];
const C: [f64; 7] = [
    f64::from_bits(0x3fdbcb7b1526e50e),
    f64::from_bits(0xbfcbcb7b1526e53d),
    f64::from_bits(0x3fc287a7636f3fa2),
    f64::from_bits(0xbfbbcb7b146a14b3),
    f64::from_bits(0x3fb63c627d5219cb),
    f64::from_bits(0xbfb2880736c8762d),
    f64::from_bits(0x3fafc1ecf913961a),
];
/// Correctly rounded base-ten logarithm (nearest-even).
pub fn log10f(x: f32) -> f32 {
    let mut ux = x.to_bits();
    if ux >= 0x7f800000 || ux == 0 {
        return super::log_special(x);
    }
    // Exact powers of ten representable in binary32 (upstream's hashed table).
    const POW10: [u32; 11] = [
        0x3f800000, 0x41200000, 0x42c80000, 0x447a0000, 0x461c4000, 0x47c35000, 0x49742400,
        0x4b189680, 0x4cbebc20, 0x4e6e6b28, 0x501502f9,
    ];
    if let Some(n) = POW10.iter().position(|&b| b == ux) {
        return n as f32;
    }
    if ux < 1 << 23 {
        let n = ux.leading_zeros() - 8;
        ux = (ux << n).wrapping_sub(n << 23);
    }
    let e = (((ux as i32) >> 23) - 127) as f64;
    let m = ux & 0x7fffff;
    let j = ((m + (1 << 16)) >> 17) as usize;
    let z = f64::from_bits(((m as u64) << 29) | (0x3ff_u64 << 52)) * TR[j] - 1.0;
    let z2 = z * z;
    let mut r =
        ((e * f64::from_bits(0x3fd34413509f79ff) + TL[j]) + z * B[0]) + z2 * (B[1] + z * B[2]);
    let ub = r as f32;
    let lb = (r + f64::from_bits(0x3ddaf23f00000000)) as f32;
    if ub == lb {
        return ub;
    }
    let mut f =
        z * ((C[0] + z * C[1]) + z2 * ((C[2] + z * C[3]) + z2 * (C[4] + z * C[5] + z2 * C[6])));
    f += (-f64::from_bits(0x3c90cee0ed4ca7e9)) * e;
    f += TL[j] - TL[0];
    let el = e * f64::from_bits(0x3fd34413509f7a00);
    r = el + f;
    if r.to_bits() & ((1 << 28) - 1) == 0 {
        let dr = (el - r) + f;
        r += dr * 32.0;
    }
    r as f32
}

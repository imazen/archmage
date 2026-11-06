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
// Tables retained from zenmath at 52120bdf; CORE-MATH 660067df.
pub(super) const FAST_IX: [f64; 33] = [
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
pub(super) const FAST_LIX: [[f64; 2]; 33] = [
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
pub(super) const FAST_C: [f64; 8] = [
    f64::from_bits(0x3ff71547652b82fe),
    f64::from_bits(0xbfe71547652b82fe),
    f64::from_bits(0x3fdec709dc3a2d0b),
    f64::from_bits(0xbfd71547652bc4a9),
    f64::from_bits(0x3fd2776c441b72e0),
    f64::from_bits(0xbfcec709bdf453ec),
    f64::from_bits(0x3fca6406efd4b877),
    f64::from_bits(0xbfc717d824a520f7),
];
pub(super) const FAST_CE: [f64; 6] = [
    f64::from_bits(0x3fa62e42fefa398b),
    f64::from_bits(0x3f4ebfbdff84555a),
    f64::from_bits(0x3eec6b08d4ad86d3),
    f64::from_bits(0x3e83b2ad1b1716a2),
    f64::from_bits(0x3e15d7472718ce9d),
    f64::from_bits(0x3da4a1d7f457ac56),
];
pub(super) const FAST_TB: [f64; 16] = [
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

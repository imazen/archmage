/* Correctly-rounded arctangent of binary64 value.

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
use super::fma;
const A: [[f64; 2]; 129] = [
    [f64::from_bits(0x0), f64::from_bits(0x0)],
    [
        f64::from_bits(0x3f89224e047e368e),
        f64::from_bits(0x3c1a3ca6c727c59d),
    ],
    [
        f64::from_bits(0x3f992346247a91f0),
        f64::from_bits(0x3bf138b0ef96a186),
    ],
    [
        f64::from_bits(0x3fa2dbaae9a05db0),
        f64::from_bits(0x3c436e7f8a3f5e42),
    ],
    [
        f64::from_bits(0x3fa927278a3b1162),
        f64::from_bits(0xbbfac986efb92662),
    ],
    [
        f64::from_bits(0x3faf7495ea3f3783),
        f64::from_bits(0x3c406ec8011ee816),
    ],
    [
        f64::from_bits(0x3fb2e239ccff3831),
        f64::from_bits(0xbc5858437d431332),
    ],
    [
        f64::from_bits(0x3fb60b9f7597fdec),
        f64::from_bits(0xbc3cebd13eb7c513),
    ],
    [
        f64::from_bits(0x3fb936bb8c5b2da2),
        f64::from_bits(0xbc5840cac0d81db5),
    ],
    [
        f64::from_bits(0x3fbc63ce377fc802),
        f64::from_bits(0x3c5400b0fdaa109e),
    ],
    [
        f64::from_bits(0x3fbf93183a8db9e9),
        f64::from_bits(0x3c40e04e06c86e72),
    ],
    [
        f64::from_bits(0x3fc1626d85a91e70),
        f64::from_bits(0x3c4f7ad829163ca7),
    ],
    [
        f64::from_bits(0x3fc2fcac73a60640),
        f64::from_bits(0xbc52680735ce2cd8),
    ],
    [
        f64::from_bits(0x3fc4986a74cf4e57),
        f64::from_bits(0xbc690559690b42e4),
    ],
    [
        f64::from_bits(0x3fc635c990ce0d36),
        f64::from_bits(0x3c591d29110b41aa),
    ],
    [
        f64::from_bits(0x3fc7d4ec54fb5968),
        f64::from_bits(0xbc4ea90e27182780),
    ],
    [
        f64::from_bits(0x3fc975f5e0553158),
        f64::from_bits(0xbc2dc82ac14e3e1c),
    ],
    [
        f64::from_bits(0x3fcb1909efd8b762),
        f64::from_bits(0xbc573a10fd13daaf),
    ],
    [
        f64::from_bits(0x3fccbe4ceb4b4cf2),
        f64::from_bits(0xbc63a7ffbeabda0b),
    ],
    [
        f64::from_bits(0x3fce65e3f27c9f2a),
        f64::from_bits(0xbc6db6627a24d523),
    ],
    [
        f64::from_bits(0x3fd007fa758626ae),
        f64::from_bits(0xbc645f97dd3099f6),
    ],
    [
        f64::from_bits(0x3fd0de53475f3b3c),
        f64::from_bits(0xbc66293f68741816),
    ],
    [
        f64::from_bits(0x3fd1b6103d3597e9),
        f64::from_bits(0xbc6ab240d40633e9),
    ],
    [
        f64::from_bits(0x3fd28f459ecad74d),
        f64::from_bits(0xbc2de34d14e832e0),
    ],
    [
        f64::from_bits(0x3fd36a08355c63dc),
        f64::from_bits(0x3c6af540d9fb4926),
    ],
    [
        f64::from_bits(0x3fd4466d542bac92),
        f64::from_bits(0x3c6da60fdbc82ac4),
    ],
    [
        f64::from_bits(0x3fd5248ae1701b17),
        f64::from_bits(0xbc792a601170138a),
    ],
    [
        f64::from_bits(0x3fd604775fbb27df),
        f64::from_bits(0xbc67f1fca1d5d15b),
    ],
    [
        f64::from_bits(0x3fd6e649f7d78649),
        f64::from_bits(0xbc64e223ea716c7b),
    ],
    [
        f64::from_bits(0x3fd7ca1a832d0f84),
        f64::from_bits(0x3c7b24c824ac51fc),
    ],
    [
        f64::from_bits(0x3fd8b00196b3d022),
        f64::from_bits(0x3c64314cd132ba43),
    ],
    [
        f64::from_bits(0x3fd998188e816bf0),
        f64::from_bits(0xbc711f1e0817879a),
    ],
    [
        f64::from_bits(0x3fda827999fcef32),
        f64::from_bits(0xbc6c3dea4dbad538),
    ],
    [
        f64::from_bits(0x3fdb6f3fc8c61e5b),
        f64::from_bits(0x3c660d1b780ee3eb),
    ],
    [
        f64::from_bits(0x3fdc5e87185e67b6),
        f64::from_bits(0xbc4ab5edb7dfa545),
    ],
    [
        f64::from_bits(0x3fdd506c82a2c800),
        f64::from_bits(0xbc68e1437048b5bd),
    ],
    [
        f64::from_bits(0x3fde450e0d273e7a),
        f64::from_bits(0xbc706951c97b050f),
    ],
    [
        f64::from_bits(0x3fdf3c8ad985d9ee),
        f64::from_bits(0xbc414af9522ab518),
    ],
    [
        f64::from_bits(0x3fe01b819b5a7cf7),
        f64::from_bits(0xbc7aba0d7d97d1f2),
    ],
    [
        f64::from_bits(0x3fe09a4c59bd0d4d),
        f64::from_bits(0x3c4095bc4ebc2c42),
    ],
    [
        f64::from_bits(0x3fe11ab7190834ec),
        f64::from_bits(0x3c8798826fa27774),
    ],
    [
        f64::from_bits(0x3fe19cd3fe8e405d),
        f64::from_bits(0x3c8008f6258fc98f),
    ],
    [
        f64::from_bits(0x3fe220b5ef047825),
        f64::from_bits(0xbc5462af7ceb7de6),
    ],
    [
        f64::from_bits(0x3fe2a6709a74f289),
        f64::from_bits(0xbc71184dfd78b472),
    ],
    [
        f64::from_bits(0x3fe32e1889047ffd),
        f64::from_bits(0x3c79141876dc40c5),
    ],
    [
        f64::from_bits(0x3fe3b7c3289ed6f3),
        f64::from_bits(0x3c8481c20189726c),
    ],
    [
        f64::from_bits(0x3fe44386db9ce5db),
        f64::from_bits(0x3c82e851bd025441),
    ],
    [
        f64::from_bits(0x3fe4d17b087b265d),
        f64::from_bits(0x3c713ada9b8bc419),
    ],
    [
        f64::from_bits(0x3fe561b82ab7f990),
        f64::from_bits(0xbc805b4c3c4cbee8),
    ],
    [
        f64::from_bits(0x3fe5f457e4f4812e),
        f64::from_bits(0xbc85619249bd96f1),
    ],
    [
        f64::from_bits(0x3fe6897514751db6),
        f64::from_bits(0xbc6b0a0fbcafc671),
    ],
    [
        f64::from_bits(0x3fe7212be621be6d),
        f64::from_bits(0xbc819ff2dc66da45),
    ],
    [
        f64::from_bits(0x3fe7bb99ed2990cf),
        f64::from_bits(0x3c81320449592d92),
    ],
    [
        f64::from_bits(0x3fe858de3b716571),
        f64::from_bits(0xbc81fddcd2f3da8e),
    ],
    [
        f64::from_bits(0x3fe8f9197bf85eeb),
        f64::from_bits(0x3c6d44a42e35cc97),
    ],
    [
        f64::from_bits(0x3fe99c6e0f634394),
        f64::from_bits(0xbc7585a178b4a18d),
    ],
    [
        f64::from_bits(0x3fea43002ae42850),
        f64::from_bits(0x3c6f95a531b3a970),
    ],
    [
        f64::from_bits(0x3feaecf5f9ba35a6),
        f64::from_bits(0xbc396c2d43ca3392),
    ],
    [
        f64::from_bits(0x3feb9a77c18c1af2),
        f64::from_bits(0xbc6a5bed94b05def),
    ],
    [
        f64::from_bits(0x3fec4bb009e77983),
        f64::from_bits(0x3c454509d2bff511),
    ],
    [
        f64::from_bits(0x3fed00cbc7384d2e),
        f64::from_bits(0xbc6b4c867cef300c),
    ],
    [
        f64::from_bits(0x3fedb9fa89953fcf),
        f64::from_bits(0xbc1ddfac663d6bc6),
    ],
    [
        f64::from_bits(0x3fee776eafc91706),
        f64::from_bits(0xbc7a510683ff7cb6),
    ],
    [
        f64::from_bits(0x3fef395d9f0e3c92),
        f64::from_bits(0x3c44fdcd8e4e8710),
    ],
    [f64::from_bits(0x3ff0000000000000), f64::from_bits(0x0)],
    [
        f64::from_bits(0x3ff065c900aaf2d8),
        f64::from_bits(0xbc8deec7fc9042ad),
    ],
    [
        f64::from_bits(0x3ff0ce29d0883c99),
        f64::from_bits(0xbc8395ae45e0657d),
    ],
    [
        f64::from_bits(0x3ff139447e6a86ee),
        f64::from_bits(0x3c8332cf301a97f3),
    ],
    [
        f64::from_bits(0x3ff1a73d55278c4b),
        f64::from_bits(0xbc86cc8c4b78213b),
    ],
    [
        f64::from_bits(0x3ff2183b0c4573ff),
        f64::from_bits(0x3c870a90841da57a),
    ],
    [
        f64::from_bits(0x3ff28c66fdaf8f09),
        f64::from_bits(0xbc6ba39bad450ee0),
    ],
    [
        f64::from_bits(0x3ff303ed61109e20),
        f64::from_bits(0xbc88692946d9f93c),
    ],
    [
        f64::from_bits(0x3ff37efd8d87607e),
        f64::from_bits(0x3c63b711bf765b58),
    ],
    [
        f64::from_bits(0x3ff3fdca42847507),
        f64::from_bits(0x3c7c21387985b081),
    ],
    [
        f64::from_bits(0x3ff48089f8bf42cc),
        f64::from_bits(0xbc87ddb19d3d0efc),
    ],
    [
        f64::from_bits(0x3ff507773c537ead),
        f64::from_bits(0xbc7f5e354cf971f3),
    ],
    [
        f64::from_bits(0x3ff592d11142fa55),
        f64::from_bits(0xbc700f0ad675330d),
    ],
    [
        f64::from_bits(0x3ff622db63c8ecc2),
        f64::from_bits(0xbc82c93f50ab2c0e),
    ],
    [
        f64::from_bits(0x3ff6b7df86265200),
        f64::from_bits(0x3c7bec391adc37d5),
    ],
    [
        f64::from_bits(0x3ff7522cbdd428a8),
        f64::from_bits(0xbc69686ddc9ffcf5),
    ],
    [
        f64::from_bits(0x3ff7f218e25a7461),
        f64::from_bits(0xbc78d16529514246),
    ],
    [
        f64::from_bits(0x3ff89801106cc709),
        f64::from_bits(0xbc8092f51e9c2803),
    ],
    [
        f64::from_bits(0x3ff9444a7462122a),
        f64::from_bits(0xbc807c06755404c4),
    ],
    [
        f64::from_bits(0x3ff9f7632fa9e871),
        f64::from_bits(0x3c802e0d43abc92b),
    ],
    [
        f64::from_bits(0x3ffab1c35d8a74ea),
        f64::from_bits(0x3c5d0184e48af6f7),
    ],
    [
        f64::from_bits(0x3ffb73ee3c3ef16a),
        f64::from_bits(0x3c773be957380bc2),
    ],
    [
        f64::from_bits(0x3ffc3e738086bc0f),
        f64::from_bits(0xbc702b6e26c84462),
    ],
    [
        f64::from_bits(0x3ffd11f0dae40609),
        f64::from_bits(0x3c525c4f3ffa6e1f),
    ],
    [
        f64::from_bits(0x3ffdef13b73c1406),
        f64::from_bits(0xbc5e302db3c6823f),
    ],
    [
        f64::from_bits(0x3ffed69b4153a45d),
        f64::from_bits(0x3c73207830326c0e),
    ],
    [
        f64::from_bits(0x3fffc95abad6cf4a),
        f64::from_bits(0xbc66308cee7927bf),
    ],
    [
        f64::from_bits(0x4000641e192ceab3),
        f64::from_bits(0xbc70147ebf0df4c5),
    ],
    [
        f64::from_bits(0x4000ea21d716fbf7),
        f64::from_bits(0xbc7168533cc41d8b),
    ],
    [
        f64::from_bits(0x40017749711a6679),
        f64::from_bits(0xbc652a0b0333e9c5),
    ],
    [
        f64::from_bits(0x40020c36c6a7f38e),
        f64::from_bits(0x3c68659eece35395),
    ],
    [
        f64::from_bits(0x4002a99f50fd4f4f),
        f64::from_bits(0x3c820fcad18cb36f),
    ],
    [
        f64::from_bits(0x4003504f333f9de6),
        f64::from_bits(0xbc752afdbd5a8c74),
    ],
    [
        f64::from_bits(0x4004012ce2586a17),
        f64::from_bits(0xbc79747a792907d7),
    ],
    [
        f64::from_bits(0x4004bd3d87fe0650),
        f64::from_bits(0x3c790c59393b52c8),
    ],
    [
        f64::from_bits(0x400585aa4e1530fa),
        f64::from_bits(0x3c7af6934f13a3a8),
    ],
    [
        f64::from_bits(0x40065bc6cc825147),
        f64::from_bits(0xbc48534dcab5ad3e),
    ],
    [
        f64::from_bits(0x40074118e4b6a7c8),
        f64::from_bits(0xbc7555aa8bfca9a1),
    ],
    [
        f64::from_bits(0x400837626d70fdb8),
        f64::from_bits(0xbc556b3fee9ca72b),
    ],
    [
        f64::from_bits(0x400940ad30abc792),
        f64::from_bits(0x3c54b3fdd4fdc06c),
    ],
    [
        f64::from_bits(0x400a5f59e90600dd),
        f64::from_bits(0x3c6285d367c55ddc),
    ],
    [
        f64::from_bits(0x400b9633283b6d14),
        f64::from_bits(0xbc48712976f17a16),
    ],
    [
        f64::from_bits(0x400ce885653127e7),
        f64::from_bits(0xbc3abe8ab65d49fc),
    ],
    [
        f64::from_bits(0x400e5a3de972a377),
        f64::from_bits(0x3c5cd9be81ad764b),
    ],
    [
        f64::from_bits(0x400ff01305ecd8dc),
        f64::from_bits(0x3c4742c2922656fa),
    ],
    [
        f64::from_bits(0x4010d7dc7cff4c9e),
        f64::from_bits(0xbc77c842978bee09),
    ],
    [
        f64::from_bits(0x4011d0143e71565f),
        f64::from_bits(0x3c67bc7dea7c3c03),
    ],
    [
        f64::from_bits(0x4012e4ff1626b949),
        f64::from_bits(0x3c4aefbe25b404e9),
    ],
    [
        f64::from_bits(0x40141bfee2424771),
        f64::from_bits(0xbc34bcfaaa95cb2c),
    ],
    [
        f64::from_bits(0x40157be4eaa5e11b),
        f64::from_bits(0x3c50fe741e4ec679),
    ],
    [
        f64::from_bits(0x40170d751908c1b1),
        f64::from_bits(0x3c5fe74a5b0ec709),
    ],
    [
        f64::from_bits(0x4018dc25c117782b),
        f64::from_bits(0x3c50ca1c19f710ef),
    ],
    [
        f64::from_bits(0x401af73f4ca3310f),
        f64::from_bits(0x3c52867b40ba77d6),
    ],
    [
        f64::from_bits(0x401d7398d15e70db),
        f64::from_bits(0x3c60fd4e0d4b1547),
    ],
    [
        f64::from_bits(0x4020372fb36b87e2),
        f64::from_bits(0x3c5c16c9ecc1621d),
    ],
    [
        f64::from_bits(0x402208dbdae055ef),
        f64::from_bits(0x3c56b81a36e75e8c),
    ],
    [
        f64::from_bits(0x40244e6c595afdcc),
        f64::from_bits(0xbc57c22045771848),
    ],
    [
        f64::from_bits(0x4027398c57f3f1ad),
        f64::from_bits(0x3c5970503be105c0),
    ],
    [
        f64::from_bits(0x402b1d03c03d2f7f),
        f64::from_bits(0xbc3f299d010aead2),
    ],
    [
        f64::from_bits(0x403046e9fe60a77e),
        f64::from_bits(0x3c5d2b61deff33ec),
    ],
    [
        f64::from_bits(0x40345affed201b55),
        f64::from_bits(0x3bf0e84d9567203a),
    ],
    [
        f64::from_bits(0x403b267195b1ffae),
        f64::from_bits(0xbbfad44b44b92653),
    ],
    [
        f64::from_bits(0x40445e2455e4aaa7),
        f64::from_bits(0xbc3296d577b5e21d),
    ],
    [
        f64::from_bits(0x40545eed6854ce99),
        f64::from_bits(0x3c02db53886013ca),
    ],
    [f64::from_bits(0x0), f64::from_bits(0x0)],
];
const CH: [[f64; 2]; 3] = [
    [
        f64::from_bits(0xbfd5555555555555),
        f64::from_bits(0xbc75555555555555),
    ],
    [
        f64::from_bits(0x3fc999999999999a),
        f64::from_bits(0xbc6999999999bcb8),
    ],
    [
        f64::from_bits(0xbfc2492492492492),
        f64::from_bits(0xbc6249242093c016),
    ],
];
const CL: [f64; 4] = [
    f64::from_bits(0x3fbc71c71c71c71c),
    f64::from_bits(0xbfb745d1745d1265),
    f64::from_bits(0x3fb3b13b115bcbc4),
    f64::from_bits(0xbfb1107c41ad3253),
];
const DB: [[f64; 3]; 12] = [
    [
        f64::from_bits(0x3f80dc89a3b55010),
        f64::from_bits(0x3f80dc70ac228717),
        f64::from_bits(0x3c20000000000000),
    ],
    [
        f64::from_bits(0x3f7e3fb41d2d2260),
        f64::from_bits(0x3f7e3f9013a852f8),
        f64::from_bits(0x3c10000000000000),
    ],
    [
        f64::from_bits(0x3fe7ba49f739829f),
        f64::from_bits(0x3fe46ac372243536),
        f64::from_bits(0x3920000000000000),
    ],
    [
        f64::from_bits(0x3fca933fe176b375),
        f64::from_bits(0x3fca33f32ac5ceb5),
        f64::from_bits(0xb8f0000000000000),
    ],
    [
        f64::from_bits(0x3f7bb04a79820063),
        f64::from_bits(0x3f7bb02ed5c5e956),
        f64::from_bits(0xb8c0000000000000),
    ],
    [
        f64::from_bits(0x3f7cd30a9499618b),
        f64::from_bits(0x3f7cd2eb65f92a46),
        f64::from_bits(0xb8f0000000000000),
    ],
    [
        f64::from_bits(0x3f8f44aa37b8e66b),
        f64::from_bits(0x3f8f440b04187c87),
        f64::from_bits(0xb8f0000000000000),
    ],
    [
        f64::from_bits(0x3f7fd2ac95e57ef9),
        f64::from_bits(0x3f7fd2829febc03a),
        f64::from_bits(0xb8f0000000000000),
    ],
    [
        f64::from_bits(0x3f96419079bbf601),
        f64::from_bits(0x3f9640aade8f5427),
        f64::from_bits(0x38d0000000000000),
    ],
    [
        f64::from_bits(0x3fe7ba49f739829f),
        f64::from_bits(0x3fe46ac372243536),
        f64::from_bits(0x3910000000000000),
    ],
    [
        f64::from_bits(0x3fcd768804487b07),
        f64::from_bits(0x3fccf5676f373ec1),
        f64::from_bits(0xb910000000000000),
    ],
    [
        f64::from_bits(0x3f7bb04a79820063),
        f64::from_bits(0x3f7bb02ed5c5e956),
        f64::from_bits(0xb8c0000000000000),
    ],
];
const FAST: [f64; 4] = [
    f64::from_bits(0x3ff0000000000000),
    f64::from_bits(0xbfd555555555552b),
    f64::from_bits(0x3fc9999999069c20),
    f64::from_bits(0xbfc248d2c8444ac6),
];
const SMALL: [f64; 4] = [
    f64::from_bits(0xbfd5555555555555),
    f64::from_bits(0x3fc99999999998c1),
    f64::from_bits(0xbfc249249176aec0),
    f64::from_bits(0x3fbc711fd121ae80),
];
const INDEX: [[u64; 3]; 31] = [
    [419, 81, 0],
    [500, 81, 0],
    [582, 163, 0],
    [745, 163, 0],
    [908, 326, 0],
    [1234, 326, 0],
    [1559, 651, 0],
    [2210, 650, 1],
    [2860, 1299, 3],
    [4156, 1293, 4],
    [5444, 2569, 24],
    [7989, 2520, 32],
    [10476, 4917, 168],
    [15224, 4576, 200],
    [19601, 8341, 838],
    [27105, 6648, 731],
    [33036, 10210, 1998],
    [41266, 6292, 1117],
    [46469, 7926, 2048],
    [52375, 4038, 849],
    [55587, 4591, 1291],
    [58906, 2172, 479],
    [60612, 2390, 688],
    [62325, 1107, 247],
    [63192, 1207, 349],
    [64056, 556, 124],
    [64491, 605, 175],
    [64923, 278, 62],
    [65141, 303, 88],
    [65358, 139, 31],
    [65467, 151, 44],
];

fn two(x: f64, y: f64) -> (f64, f64) {
    let s = x + y;
    (s, y - (s - x))
}
fn add(xh: f64, xl: f64, ch: f64, cl: f64) -> (f64, f64) {
    let s = xh + ch;
    let d = s - xh;
    (s, ((ch - d) + (xh + (d - s))) + (xl + cl))
}
fn mul(xh: f64, xl: f64, ch: f64, cl: f64) -> (f64, f64) {
    let h = ch * xh;
    two(h, fma(ch, xh, -h) + (cl * xh + ch * xl))
}
fn signed(v: f64, x: f64) -> f64 {
    f64::from_bits((v.to_bits() & 0x7fffffffffffffff) | (x.to_bits() & 0x8000000000000000))
}
#[inline(never)]
fn refine(x: f64, a: f64) -> f64 {
    let i =
        (((a.abs() * f64::from_bits(0x40545f306dc9c883) + 256.5).to_bits() >> 44) & 255) as usize;
    let (h, hl) = if i == 128 {
        let h = -1.0 / x;
        (h, fma(h, x, 1.0) * h)
    } else {
        let ta = signed(A[i][0], x);
        let zta = x * ta;
        let ztal = fma(x, ta, -zta);
        let zmta = x - ta;
        let v = 1.0 + zta;
        let d = 1.0 - v;
        let ev = (d + zta) - ((d + v) - 1.0) + ztal;
        let r = 1.0 / v;
        let rl = (fma(r, -v, 1.0) - ev * r) * r;
        let h = r * zmta;
        (h, fma(r, zmta, -h) + rl * zmta)
    };
    let (h2, h2l) = mul(h, hl, h, hl);
    let h4 = h2 * h2;
    let (h3, h3l) = mul(h, hl, h2, h2l);
    let tail = h2 * ((CL[0] + h2 * CL[1]) + h4 * (CL[2] + h2 * CL[3]));
    let mut f = CH[2][0] + tail;
    let mut fl = (CH[2][0] - f) + tail + CH[2][1];
    for c in CH[..2].iter().rev() {
        (f, fl) = mul(h2, h2l, f, fl);
        let th = f + c[0];
        let tl = (c[0] - th) + f;
        f = th;
        fl += tl + c[1];
    }
    (f, fl) = mul(h3, h3l, f, fl);
    let (ah, al, at) = if i == 0 {
        (h, f, fl)
    } else {
        let df = if i < 128 {
            signed(1.0, x) * A[i][1]
        } else {
            0.0
        };
        let id = signed(i as f64, x);
        let ah = f64::from_bits(0x3f8921fb54442d00) * id;
        let al = f64::from_bits(0x3c88469898cc5180) * id;
        let at = -f64::from_bits(0x397fc8f8cbb5bf80) * id;
        let (al, at) = add(al, at, df, 0.0);
        let (al, at) = add(al, at, h, hl);
        let (al, at) = add(al, at, f, fl);
        (ah, al, at)
    };
    let (v0, v2) = two(ah, al);
    let (mut v1, v2) = two(v2, at);
    let t0 = v0.to_bits();
    let t1 = v1.to_bits();
    if t1.wrapping_add(1) & 0xfffffffffffff <= 2
        || ((t0 >> 52) & 0x7ff).wrapping_sub((t1 >> 52) & 0x7ff) > 103
    {
        if let Some(db) = DB.iter().find(|v| v[0] == x.abs()) {
            return signed(db[1], x) + signed(1.0, x) * db[2];
        }
        if t1 & 0xfffffffffffff == 0 {
            v1 = f64::from_bits(if (v2.to_bits() ^ t1) >> 63 != 0 {
                t1.wrapping_sub(1)
            } else {
                t1.wrapping_add(1)
            });
        }
    }
    v1 + v0
}
/// Arctangent, rounded to nearest with ties to even.
pub fn atan(x: f64) -> f64 {
    let at = x.to_bits() & 0x7fffffffffffffff;
    if at < 0x3f7b21c475e6362a {
        if at == 0 {
            return x;
        }
        if at < 0x3e40000000000000 {
            return fma(-f64::from_bits(0x3c90000000000000), x, x);
        }
        let x2 = x * x;
        let x3 = x * x2;
        let x4 = x2 * x2;
        let f = x3 * ((SMALL[0] + x2 * SMALL[1]) + x4 * (SMALL[2] + x2 * SMALL[3]));
        let ub = (f + f * f64::from_bits(0x3cd6000000000000)) + x;
        let lb = (f - f * f64::from_bits(0x3cc6000000000000)) + x;
        return if ub == lb { ub } else { refine(x, ub) };
    }
    let (h, ah, mut al) = if at > 0x4062ded8e34a9035 {
        let ah = signed(f64::from_bits(0x3ff921fb54442d18), x);
        let al = signed(f64::from_bits(0x3c91a62633145c07), x);
        if at >= 0x434d02967c31cdb5 {
            return if x.is_nan() { x + x } else { ah + al };
        }
        (-1.0 / x, ah, al)
    } else {
        let i = ((at >> 51) as i64 - 2030) as usize;
        let u = x.to_bits() & 0x7ffffffffffff;
        let ut = u >> 35;
        let ut2 = (ut * ut) >> 16;
        let i = (((INDEX[i][0] << 16) + ut * INDEX[i][1] - ut2 * INDEX[i][2]) >> 25) as usize;
        let sg = signed(1.0, x);
        let ta = sg * A[i][0];
        let id = sg * i as f64;
        (
            (x - ta) / (1.0 + x * ta),
            f64::from_bits(0x3f8921fb54442d00) * id,
            sg * A[i][1] + f64::from_bits(0x3c88469898cc5170) * id,
        )
    };
    let h2 = h * h;
    let h4 = h2 * h2;
    let f = (FAST[0] + h2 * FAST[1]) + h4 * (FAST[2] + h2 * FAST[3]);
    al = fma(h, f, al);
    let e = h * f64::from_bits(0x3ccf800000000000);
    let ub = (al + e) + ah;
    let lb = (al - e) + ah;
    if ub == lb { ub } else { refine(x, ub) }
}

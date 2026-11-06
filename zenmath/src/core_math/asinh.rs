/* Correctly-rounded inverse hyperbolic sine function for binary64.

Copyright (c) 2023-2025 Alexei Sibidanov <sibid@uvic.ca>.

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
const CH0: [[f64; 2]; 12] = [
    [
        f64::from_bits(0xbfc5555555555555),
        f64::from_bits(0xbc65555555555555),
    ],
    [
        f64::from_bits(0x3fb3333333333333),
        f64::from_bits(0x3c499999999949df),
    ],
    [
        f64::from_bits(0xbfa6db6db6db6db7),
        f64::from_bits(0x3c32492496091b0c),
    ],
    [
        f64::from_bits(0x3f9f1c71c71c71c7),
        f64::from_bits(0x3c1c71a35cfa0671),
    ],
    [
        f64::from_bits(0xbf96e8ba2e8ba2e9),
        f64::from_bits(0x3c317f937248cf81),
    ],
    [
        f64::from_bits(0x3f91c4ec4ec4ec4f),
        f64::from_bits(0xbc374e3c1dfd4c3d),
    ],
    [
        f64::from_bits(0xbf8c999999999977),
        f64::from_bits(0xbc238e7a467ecc55),
    ],
    [
        f64::from_bits(0x3f87a87878786c7e),
        f64::from_bits(0x3c2a83c7bace55eb),
    ],
    [
        f64::from_bits(0xbf83fde50d764083),
        f64::from_bits(0xbc2d024df7fa0542),
    ],
    [
        f64::from_bits(0x3f812ef3ceae4d12),
        f64::from_bits(0xbc2ba9c13deb261f),
    ],
    [
        f64::from_bits(0xbf7df3bd104aa267),
        f64::from_bits(0xbc1546da9bc5b32a),
    ],
    [
        f64::from_bits(0x3f7a685fc5de7a04),
        f64::from_bits(0x3c140d284a1d67f9),
    ],
];
const CL0: [f64; 5] = [
    f64::from_bits(0xbf77828d553ec800),
    f64::from_bits(0x3f751712f7bee368),
    f64::from_bits(0xbf72e6d98527bcc6),
    f64::from_bits(0x3f70095da47b392c),
    f64::from_bits(0xbf63b92d6368192c),
];
const R1: [f64; 33] = [
    f64::from_bits(0x3ff0000000000000),
    f64::from_bits(0x3fef507600000000),
    f64::from_bits(0x3feea4b000000000),
    f64::from_bits(0x3fedfc9800000000),
    f64::from_bits(0x3fed581800000000),
    f64::from_bits(0x3fecb72000000000),
    f64::from_bits(0x3fec199c00000000),
    f64::from_bits(0x3feb7f7600000000),
    f64::from_bits(0x3feae8a000000000),
    f64::from_bits(0x3fea550400000000),
    f64::from_bits(0x3fe9c49200000000),
    f64::from_bits(0x3fe9373800000000),
    f64::from_bits(0x3fe8ace600000000),
    f64::from_bits(0x3fe8258a00000000),
    f64::from_bits(0x3fe7a11400000000),
    f64::from_bits(0x3fe71f7600000000),
    f64::from_bits(0x3fe6a09e00000000),
    f64::from_bits(0x3fe6247e00000000),
    f64::from_bits(0x3fe5ab0800000000),
    f64::from_bits(0x3fe5342c00000000),
    f64::from_bits(0x3fe4bfda00000000),
    f64::from_bits(0x3fe44e0800000000),
    f64::from_bits(0x3fe3dea600000000),
    f64::from_bits(0x3fe371a800000000),
    f64::from_bits(0x3fe306fe00000000),
    f64::from_bits(0x3fe29e9e00000000),
    f64::from_bits(0x3fe2387a00000000),
    f64::from_bits(0x3fe1d48800000000),
    f64::from_bits(0x3fe172b800000000),
    f64::from_bits(0x3fe1130200000000),
    f64::from_bits(0x3fe0b55800000000),
    f64::from_bits(0x3fe059b000000000),
    f64::from_bits(0x3fe0000000000000),
];
const R2: [f64; 33] = [
    f64::from_bits(0x3ff0000000000000),
    f64::from_bits(0x3feffa7400000000),
    f64::from_bits(0x3feff4ea00000000),
    f64::from_bits(0x3fefef6200000000),
    f64::from_bits(0x3fefe9da00000000),
    f64::from_bits(0x3fefe45200000000),
    f64::from_bits(0x3fefdecc00000000),
    f64::from_bits(0x3fefd94600000000),
    f64::from_bits(0x3fefd3c200000000),
    f64::from_bits(0x3fefce3e00000000),
    f64::from_bits(0x3fefc8bc00000000),
    f64::from_bits(0x3fefc33a00000000),
    f64::from_bits(0x3fefbdba00000000),
    f64::from_bits(0x3fefb83a00000000),
    f64::from_bits(0x3fefb2bc00000000),
    f64::from_bits(0x3fefad3e00000000),
    f64::from_bits(0x3fefa7c200000000),
    f64::from_bits(0x3fefa24600000000),
    f64::from_bits(0x3fef9cca00000000),
    f64::from_bits(0x3fef975000000000),
    f64::from_bits(0x3fef91d800000000),
    f64::from_bits(0x3fef8c6000000000),
    f64::from_bits(0x3fef86e800000000),
    f64::from_bits(0x3fef817200000000),
    f64::from_bits(0x3fef7bfe00000000),
    f64::from_bits(0x3fef768a00000000),
    f64::from_bits(0x3fef711600000000),
    f64::from_bits(0x3fef6ba400000000),
    f64::from_bits(0x3fef663200000000),
    f64::from_bits(0x3fef60c200000000),
    f64::from_bits(0x3fef5b5200000000),
    f64::from_bits(0x3fef55e400000000),
    f64::from_bits(0x3fef507600000000),
];
const L1: [[f64; 2]; 33] = [
    [
        f64::from_bits(0x0000000000000000),
        f64::from_bits(0x0000000000000000),
    ],
    [
        f64::from_bits(0xbd1269e2038315b3),
        f64::from_bits(0x3f962e4eacd40000),
    ],
    [
        f64::from_bits(0xbd23f2558bddfc47),
        f64::from_bits(0x3fa62e3ce7218000),
    ],
    [
        f64::from_bits(0x3d207ea13c34efb5),
        f64::from_bits(0x3fb0a2ab6d3ec000),
    ],
    [
        f64::from_bits(0x3d38f3e77084d3ba),
        f64::from_bits(0x3fb62e4a86d8c000),
    ],
    [
        f64::from_bits(0xbd18d92a005f1a7e),
        f64::from_bits(0x3fbbb9db7062c000),
    ],
    [
        f64::from_bits(0x3d358239e799bfe5),
        f64::from_bits(0x3fc0a2b1a22cc000),
    ],
    [
        f64::from_bits(0xbd3a93fcf5f593b7),
        f64::from_bits(0x3fc3687f0a298000),
    ],
    [
        f64::from_bits(0xbd1db4cac32fd2b5),
        f64::from_bits(0x3fc62e4116b64000),
    ],
    [
        f64::from_bits(0xbd10e65a92ee0f3b),
        f64::from_bits(0x3fc8f409e4df6000),
    ],
    [
        f64::from_bits(0xbd38261383d475f1),
        f64::from_bits(0x3fcbb9d15001c000),
    ],
    [
        f64::from_bits(0xbd3359886207513b),
        f64::from_bits(0x3fce7f9a8c940000),
    ],
    [
        f64::from_bits(0x3d3811f87496ceb7),
        f64::from_bits(0x3fd0a2b052ddb000),
    ],
    [
        f64::from_bits(0x3d34991ec6cb435c),
        f64::from_bits(0x3fd205955ef73000),
    ],
    [
        f64::from_bits(0xbd34581abfeb8927),
        f64::from_bits(0x3fd3687bd9121000),
    ],
    [
        f64::from_bits(0x3d3cab48f6942703),
        f64::from_bits(0x3fd4cb5e8f2b5000),
    ],
    [
        f64::from_bits(0xbd0df2c452fde132),
        f64::from_bits(0x3fd62e4420e20000),
    ],
    [
        f64::from_bits(0x3d26109f4fdb74bd),
        f64::from_bits(0x3fd791292c46a000),
    ],
    [
        f64::from_bits(0xbd36b95fbdac7696),
        f64::from_bits(0x3fd8f40af84e7000),
    ],
    [
        f64::from_bits(0x3d17394fa880cbda),
        f64::from_bits(0x3fda56ed8f865000),
    ],
    [
        f64::from_bits(0xbd150b06a94eccab),
        f64::from_bits(0x3fdbb9d6505b4000),
    ],
    [
        f64::from_bits(0xbd3be2abf0b38989),
        f64::from_bits(0x3fdd1cb91e728000),
    ],
    [
        f64::from_bits(0xbd37d6bf1e34da04),
        f64::from_bits(0x3fde7f9d139e2000),
    ],
    [
        f64::from_bits(0xbd3423c1e14de6ed),
        f64::from_bits(0x3fdfe27db9b0e000),
    ],
    [
        f64::from_bits(0x3d3c46f1a0efbbc2),
        f64::from_bits(0x3fe0a2b25060a800),
    ],
    [
        f64::from_bits(0x3d2834fe4e3e6018),
        f64::from_bits(0x3fe154244482a000),
    ],
    [
        f64::from_bits(0x3d16a03d0f02b650),
        f64::from_bits(0x3fe2059731298800),
    ],
    [
        f64::from_bits(0x3d3d437056526f30),
        f64::from_bits(0x3fe2b707145de000),
    ],
    [
        f64::from_bits(0xbd2a0233728405c5),
        f64::from_bits(0x3fe3687b0e0b2800),
    ],
    [
        f64::from_bits(0xbd24dbdda10d2bf1),
        f64::from_bits(0x3fe419ec5d3f6800),
    ],
    [
        f64::from_bits(0x3d3f7d0a25d154f2),
        f64::from_bits(0x3fe4cb5f9fc02000),
    ],
    [
        f64::from_bits(0x3d315ede4d803b18),
        f64::from_bits(0x3fe57cd28421a800),
    ],
    [
        f64::from_bits(0x3d2ef35793c76730),
        f64::from_bits(0x3fe62e42fefa3800),
    ],
];
const L2: [[f64; 2]; 33] = [
    [
        f64::from_bits(0x0000000000000000),
        f64::from_bits(0x0000000000000000),
    ],
    [
        f64::from_bits(0x3d35abdac3638e99),
        f64::from_bits(0x3f4631ec81e00000),
    ],
    [
        f64::from_bits(0xbd216b8be9bbe239),
        f64::from_bits(0x3f562fd812700000),
    ],
    [
        f64::from_bits(0xbd3364c6315542eb),
        f64::from_bits(0x3f60a25205080000),
    ],
    [
        f64::from_bits(0x3d2734abe459c900),
        f64::from_bits(0x3f662dadc1d00000),
    ],
    [
        f64::from_bits(0x3d30cf8a761431bf),
        f64::from_bits(0x3f6bb9ff94d00000),
    ],
    [
        f64::from_bits(0x3d2da2718eb78708),
        f64::from_bits(0x3f70a2a2def80000),
    ],
    [
        f64::from_bits(0x3d334ada62c59b93),
        f64::from_bits(0x3f7368c0fae40000),
    ],
    [
        f64::from_bits(0x3d3d09ab376682d4),
        f64::from_bits(0x3f762e58e4f80000),
    ],
    [
        f64::from_bits(0xbd23cb7b94329211),
        f64::from_bits(0x3f78f46bd28c0000),
    ],
    [
        f64::from_bits(0xbd2eec5c297c41d0),
        f64::from_bits(0x3f7bb9f831200000),
    ],
    [
        f64::from_bits(0xbd36411b9395d150),
        f64::from_bits(0x3f7e7fff8f300000),
    ],
    [
        f64::from_bits(0xbd31c0e59a43053c),
        f64::from_bits(0x3f80a2c0006e0000),
    ],
    [
        f64::from_bits(0x3d16506596e077b6),
        f64::from_bits(0x3f8205bdb6f00000),
    ],
    [
        f64::from_bits(0x3d3e256bce6faa27),
        f64::from_bits(0x3f836877c86e0000),
    ],
    [
        f64::from_bits(0x3ccbd42467b0c8d1),
        f64::from_bits(0x3f84cb6f55780000),
    ],
    [
        f64::from_bits(0xbd3c4f92132ff0f0),
        f64::from_bits(0x3f862e230e8c0000),
    ],
    [
        f64::from_bits(0xbd380be08bfab390),
        f64::from_bits(0x3f87911440f60000),
    ],
    [
        f64::from_bits(0xbd3f0b1319ceb1f7),
        f64::from_bits(0x3f88f443020a0000),
    ],
    [
        f64::from_bits(0x3d2a65fcfb8de99b),
        f64::from_bits(0x3f8a572dbef40000),
    ],
    [
        f64::from_bits(0x3d14233885d3779c),
        f64::from_bits(0x3f8bb9d449a60000),
    ],
    [
        f64::from_bits(0x3d3f46a59e646edb),
        f64::from_bits(0x3f8d1cb8491c0000),
    ],
    [
        f64::from_bits(0xbd3c3d2f11c11446),
        f64::from_bits(0x3f8e7fd9d2aa0000),
    ],
    [
        f64::from_bits(0x3d27763f78a1e0cc),
        f64::from_bits(0x3f8fe2b6f9780000),
    ],
    [
        f64::from_bits(0x3d3b4c37fc60c043),
        f64::from_bits(0x3f90a2a7c7a50000),
    ],
    [
        f64::from_bits(0xbd15b8a822859be3),
        f64::from_bits(0x3f915412ca860000),
    ],
    [
        f64::from_bits(0xbd3f2d8c9fc06400),
        f64::from_bits(0x3f92059c90050000),
    ],
    [
        f64::from_bits(0xbd3e80e79c20378d),
        f64::from_bits(0x3f92b703f49b0000),
    ],
    [
        f64::from_bits(0x3d368256e4329bdb),
        f64::from_bits(0x3f93688a1a8d0000),
    ],
    [
        f64::from_bits(0x3d37e9741da248c3),
        f64::from_bits(0x3f9419edc7ba0000),
    ],
    [
        f64::from_bits(0x3d2e330dccce602b),
        f64::from_bits(0x3f94cb7034fa0000),
    ],
    [
        f64::from_bits(0x3ce2f32b5d18eefb),
        f64::from_bits(0x3f957cd011870000),
    ],
    [
        f64::from_bits(0xbd1269e2038315b3),
        f64::from_bits(0x3f962e4eacd40000),
    ],
];
const C1: [f64; 5] = [
    f64::from_bits(0xbfe0000000000000),
    f64::from_bits(0x3fd5555555555530),
    f64::from_bits(0xbfcfffffffffffa0),
    f64::from_bits(0x3fc99999e33a6366),
    f64::from_bits(0xbfc555559ef9525f),
];
const SM0: [f64; 1] = [f64::from_bits(0xbfc5555555555555)];
const SM1: [f64; 2] = [
    f64::from_bits(0xbfc5555555555555),
    f64::from_bits(0x3fb3333327c57c60),
];
const SM2: [f64; 4] = [
    f64::from_bits(0xbfc5555555555555),
    f64::from_bits(0x3fb333333332f2ff),
    f64::from_bits(0xbfa6db6d9a665159),
    f64::from_bits(0x3f9f186866d775f0),
];
const SM3: [f64; 7] = [
    f64::from_bits(0xbfc5555555555555),
    f64::from_bits(0x3fb3333333333310),
    f64::from_bits(0xbfa6db6db6da466c),
    f64::from_bits(0x3f9f1c71c2ea7be4),
    f64::from_bits(0xbf96e8b651b09d72),
    f64::from_bits(0x3f91c309fc0e69c2),
    f64::from_bits(0xbf8bab7833c1e000),
];
const DB: [[f64; 3]; 35] = [
    [
        f64::from_bits(0x3fd00f9476450863),
        f64::from_bits(0x3fcfcb35067f343c),
        f64::from_bits(0x3c60000000000000),
    ],
    [
        f64::from_bits(0x3fd1f0a79315b287),
        f64::from_bits(0x3fd1b68aae88feba),
        f64::from_bits(0x3c70000000000000),
    ],
    [
        f64::from_bits(0x3fd2b9618ff7acb7),
        f64::from_bits(0x3fd27781d9aa4e25),
        f64::from_bits(0xbc70000000000000),
    ],
    [
        f64::from_bits(0x3fd389ef683f3aa7),
        f64::from_bits(0x3fd33f52db6df1af),
        f64::from_bits(0x3c70000000000000),
    ],
    [
        f64::from_bits(0x3fd3b07e0c779dda),
        f64::from_bits(0x3fd364303e1ad8f6),
        f64::from_bits(0x3c70000000000000),
    ],
    [
        f64::from_bits(0x3fd48441df33b6d3),
        f64::from_bits(0x3fd42e385800f0a4),
        f64::from_bits(0x3c70000000000000),
    ],
    [
        f64::from_bits(0x3fd687bd068c1c1e),
        f64::from_bits(0x3fd616cc75d49226),
        f64::from_bits(0xbc70000000000000),
    ],
    [
        f64::from_bits(0x3fd8740c4453a056),
        f64::from_bits(0x3fd7e4f2ad132a1d),
        f64::from_bits(0x3c70000000000000),
    ],
    [
        f64::from_bits(0x3fd891acda11167e),
        f64::from_bits(0x3fd8009d924a3ffd),
        f64::from_bits(0x3c70000000000000),
    ],
    [
        f64::from_bits(0x3fdbafc3479fc9cc),
        f64::from_bits(0x3fdae3773250e7d2),
        f64::from_bits(0x3c70000000000000),
    ],
    [
        f64::from_bits(0x3fdc59869f17b483),
        f64::from_bits(0x3fdb7efa91915c95),
        f64::from_bits(0x3c70000000000000),
    ],
    [
        f64::from_bits(0x3fdc8be879787986),
        f64::from_bits(0x3fdbad0485e0fe0a),
        f64::from_bits(0xbc70000000000000),
    ],
    [
        f64::from_bits(0x3fde73b46abb01e1),
        f64::from_bits(0x3fdd68039861ab53),
        f64::from_bits(0x3c70000000000000),
    ],
    [
        f64::from_bits(0x3fded6236da268b0),
        f64::from_bits(0x3fddc0cb8f638126),
        f64::from_bits(0x3c70000000000000),
    ],
    [
        f64::from_bits(0x3fdf399ebafc1951),
        f64::from_bits(0x3fde1a4f519fab77),
        f64::from_bits(0xbc70000000000000),
    ],
    [
        f64::from_bits(0x3fdf70975ab0d471),
        f64::from_bits(0x3fde4bae8bcd6ea6),
        f64::from_bits(0x3c70000000000000),
    ],
    [
        f64::from_bits(0x3fdfbdd4a37760b7),
        f64::from_bits(0x3fde90f16eb88c09),
        f64::from_bits(0x3c70000000000000),
    ],
    [
        f64::from_bits(0x3fdfee72efb4bfdd),
        f64::from_bits(0x3fdebc791a88bed8),
        f64::from_bits(0x3c70000000000000),
    ],
    [
        f64::from_bits(0x3fe02339d6bdb741),
        f64::from_bits(0x3fdf0b2264e34555),
        f64::from_bits(0x3c70000000000000),
    ],
    [
        f64::from_bits(0x3fe09e7c831b1a23),
        f64::from_bits(0x3fdfe694c3c89138),
        f64::from_bits(0x3c70000000000000),
    ],
    [
        f64::from_bits(0x3fe16d32c862fc3b),
        f64::from_bits(0x3fe0a9c9334066db),
        f64::from_bits(0xbc80000000000000),
    ],
    [
        f64::from_bits(0x3fe857954132083d),
        f64::from_bits(0x3fe67425fe575c88),
        f64::from_bits(0xbc80000000000000),
    ],
    [
        f64::from_bits(0x3fe8a5c3b60f7e11),
        f64::from_bits(0x3fe6b23ad4415a17),
        f64::from_bits(0xbc80000000000000),
    ],
    [
        f64::from_bits(0x3fe9740eb419dd04),
        f64::from_bits(0x3fe754ab7535d47d),
        f64::from_bits(0x3c80000000000000),
    ],
    [
        f64::from_bits(0x3fea16d9cc06011a),
        f64::from_bits(0x3fe7d3755d851062),
        f64::from_bits(0xbc80000000000000),
    ],
    [
        f64::from_bits(0x3febb635be2213d1),
        f64::from_bits(0x3fe91167cae3cfa9),
        f64::from_bits(0x3c80000000000000),
    ],
    [
        f64::from_bits(0x3fed4b21ebf542f0),
        f64::from_bits(0x3fea3fc7e4dd47d1),
        f64::from_bits(0xbc80000000000000),
    ],
    [
        f64::from_bits(0x3ff7b8516ffd2406),
        f64::from_bits(0x3ff2f5d3b178914a),
        f64::from_bits(0x3c90000000000000),
    ],
    [
        f64::from_bits(0x3ff9295b9116e2e2),
        f64::from_bits(0x3ff3bffa88639760),
        f64::from_bits(0x3c90000000000000),
    ],
    [
        f64::from_bits(0x3fffedc65e327140),
        f64::from_bits(0x3ff710f91e844f9b),
        f64::from_bits(0x3c90000000000000),
    ],
    [
        f64::from_bits(0x40057e377b3f0b4b),
        f64::from_bits(0x3ffb6e2c73f41415),
        f64::from_bits(0x3c90000000000000),
    ],
    [
        f64::from_bits(0x4026056b06a21918),
        f64::from_bits(0x4008c0a26d055288),
        f64::from_bits(0x3ca0000000000000),
    ],
    [
        f64::from_bits(0x403843e1b5e5979c),
        f64::from_bits(0x400f0f978201eb84),
        f64::from_bits(0x3ca0000000000000),
    ],
    [
        f64::from_bits(0x409fee8f69c4cd25),
        f64::from_bits(0x4020a19aebb51e90),
        f64::from_bits(0xbcc0000000000000),
    ],
    [
        f64::from_bits(0x4170fbc6c02b1c90),
        f64::from_bits(0x40316369cd53bb69),
        f64::from_bits(0x3cd0000000000000),
    ],
];
const T1: [f64; 17] = [
    f64::from_bits(0x3ff0000000000000),
    f64::from_bits(0x3feea4afa0000000),
    f64::from_bits(0x3fed5818e0000000),
    f64::from_bits(0x3fec199be0000000),
    f64::from_bits(0x3feae89f98000000),
    f64::from_bits(0x3fe9c49180000000),
    f64::from_bits(0x3fe8ace540000000),
    f64::from_bits(0x3fe7a11470000000),
    f64::from_bits(0x3fe6a09e68000000),
    f64::from_bits(0x3fe5ab07e0000000),
    f64::from_bits(0x3fe4bfdad8000000),
    f64::from_bits(0x3fe3dea650000000),
    f64::from_bits(0x3fe306fe08000000),
    f64::from_bits(0x3fe2387a70000000),
    f64::from_bits(0x3fe172b840000000),
    f64::from_bits(0x3fe0b55870000000),
    f64::from_bits(0x3fe0000000000000),
];
const T2: [f64; 16] = [
    f64::from_bits(0x3ff0000000000000),
    f64::from_bits(0x3fefe9d968000000),
    f64::from_bits(0x3fefd3c228000000),
    f64::from_bits(0x3fefbdba38000000),
    f64::from_bits(0x3fefa7c180000000),
    f64::from_bits(0x3fef91d800000000),
    f64::from_bits(0x3fef7bfdb0000000),
    f64::from_bits(0x3fef663278000000),
    f64::from_bits(0x3fef507658000000),
    f64::from_bits(0x3fef3ac948000000),
    f64::from_bits(0x3fef252b38000000),
    f64::from_bits(0x3fef0f9c20000000),
    f64::from_bits(0x3feefa1bf0000000),
    f64::from_bits(0x3feee4aaa0000000),
    f64::from_bits(0x3feecf4830000000),
    f64::from_bits(0x3feeb9f488000000),
];
const T3: [f64; 16] = [
    f64::from_bits(0x3ff0000000000000),
    f64::from_bits(0x3feffe9d20000000),
    f64::from_bits(0x3feffd3a58000000),
    f64::from_bits(0x3feffbd798000000),
    f64::from_bits(0x3feffa74e8000000),
    f64::from_bits(0x3feff91248000000),
    f64::from_bits(0x3feff7afb8000000),
    f64::from_bits(0x3feff64d38000000),
    f64::from_bits(0x3feff4eac8000000),
    f64::from_bits(0x3feff38868000000),
    f64::from_bits(0x3feff22618000000),
    f64::from_bits(0x3feff0c3d0000000),
    f64::from_bits(0x3fefef61a0000000),
    f64::from_bits(0x3fefedff78000000),
    f64::from_bits(0x3fefec9d68000000),
    f64::from_bits(0x3fefeb3b60000000),
];
const T4: [f64; 16] = [
    f64::from_bits(0x3ff0000000000000),
    f64::from_bits(0x3fefffe9d0000000),
    f64::from_bits(0x3fefffd3a0000000),
    f64::from_bits(0x3fefffbd78000000),
    f64::from_bits(0x3fefffa748000000),
    f64::from_bits(0x3fefff9118000000),
    f64::from_bits(0x3fefff7ae8000000),
    f64::from_bits(0x3fefff64c0000000),
    f64::from_bits(0x3fefff4e90000000),
    f64::from_bits(0x3fefff3860000000),
    f64::from_bits(0x3fefff2238000000),
    f64::from_bits(0x3fefff0c08000000),
    f64::from_bits(0x3feffef5d8000000),
    f64::from_bits(0x3feffedfa8000000),
    f64::from_bits(0x3feffec980000000),
    f64::from_bits(0x3feffeb350000000),
];
const LL: [[[f64; 3]; 17]; 4] = [
    [
        [
            f64::from_bits(0x0000000000000000),
            f64::from_bits(0x0000000000000000),
            f64::from_bits(0x0000000000000000),
        ],
        [
            f64::from_bits(0x3f962e432b240000),
            f64::from_bits(0xbd5745af34bb54b8),
            f64::from_bits(0xb9e17e3ec05cde70),
        ],
        [
            f64::from_bits(0x3fa62e42e4a80000),
            f64::from_bits(0x3d3111a4eadf3120),
            f64::from_bits(0x3a2cff3027abb119),
        ],
        [
            f64::from_bits(0x3fb0a2b233f10000),
            f64::from_bits(0xbd588ac4ec78af80),
            f64::from_bits(0x3a24fa087ca75dfd),
        ],
        [
            f64::from_bits(0x3fb62e43056c0000),
            f64::from_bits(0x3d16bd65e8b0b700),
            f64::from_bits(0xba0b18e160362c24),
        ],
        [
            f64::from_bits(0x3fbbb9d3cbd60000),
            f64::from_bits(0x3d5de14aa55ec2b0),
            f64::from_bits(0xba1c6ac3f1862a6b),
        ],
        [
            f64::from_bits(0x3fc0a2b244da0000),
            f64::from_bits(0x3d594def487fea70),
            f64::from_bits(0xba1dead1a4581acf),
        ],
        [
            f64::from_bits(0x3fc3687aa9b78000),
            f64::from_bits(0x3d49cec9a50db220),
            f64::from_bits(0x3a234a70684f8e0e),
        ],
        [
            f64::from_bits(0x3fc62e42faba0000),
            f64::from_bits(0xbd3d69047a3aeb00),
            f64::from_bits(0xba04e061f79144e2),
        ],
        [
            f64::from_bits(0x3fc8f40b56d28000),
            f64::from_bits(0x3d5de7d755fd2e20),
            f64::from_bits(0x3a1bdc7ecf001489),
        ],
        [
            f64::from_bits(0x3fcbb9d3b61f0000),
            f64::from_bits(0x3d1c14f1445b1200),
            f64::from_bits(0x3a2a1d78cbdc5b58),
        ],
        [
            f64::from_bits(0x3fce7f9c11f08000),
            f64::from_bits(0xbd46e3e0000dae70),
            f64::from_bits(0x3a16a4559fadde98),
        ],
        [
            f64::from_bits(0x3fd0a2b242ec4000),
            f64::from_bits(0x3d5bb7cf852a5fe8),
            f64::from_bits(0x3a2a6aef11ee43bd),
        ],
        [
            f64::from_bits(0x3fd205966c764000),
            f64::from_bits(0x3d2ad3a5f2142940),
            f64::from_bits(0x3a25cc344fa10652),
        ],
        [
            f64::from_bits(0x3fd3687a98aac000),
            f64::from_bits(0x3d21623671842f00),
            f64::from_bits(0xba10b428fe1f9e43),
        ],
        [
            f64::from_bits(0x3fd4cb5ec93f4000),
            f64::from_bits(0x3d53d50980ea5130),
            f64::from_bits(0x3a267f0ea083b1c4),
        ],
        [
            f64::from_bits(0x3fd62e42fefa4000),
            f64::from_bits(0xbd38432a1b0e2640),
            f64::from_bits(0x3a2803f2f6af40f3),
        ],
    ],
    [
        [
            f64::from_bits(0x0000000000000000),
            f64::from_bits(0x0000000000000000),
            f64::from_bits(0x0000000000000000),
        ],
        [
            f64::from_bits(0x3f562e462b400000),
            f64::from_bits(0x3d5061d003b97318),
            f64::from_bits(0x3a2d7faee66a2e1e),
        ],
        [
            f64::from_bits(0x3f662e44c9200000),
            f64::from_bits(0x3d595a7bff5e2390),
            f64::from_bits(0xba0f7e788a871350),
        ],
        [
            f64::from_bits(0x3f70a2b1e3300000),
            f64::from_bits(0x3d42a3a1a65aa3a0),
            f64::from_bits(0xba254599c9605442),
        ],
        [
            f64::from_bits(0x3f762e4367c00000),
            f64::from_bits(0xbd24a995b6d9ddc0),
            f64::from_bits(0xb9b56bb79b254f33),
        ],
        [
            f64::from_bits(0x3f7bb9d449a00000),
            f64::from_bits(0x3d58a119c42e9bc0),
            f64::from_bits(0xba28ecf7d8d661f1),
        ],
        [
            f64::from_bits(0x3f80a2b1f1900000),
            f64::from_bits(0x3d58863771bd10a8),
            f64::from_bits(0x3a1e9731de7f0155),
        ],
        [
            f64::from_bits(0x3f83687ad1100000),
            f64::from_bits(0x3d5e026a347ca1c8),
            f64::from_bits(0x39efadc62522444d),
        ],
        [
            f64::from_bits(0x3f862e436f280000),
            f64::from_bits(0x3d525b84f71b70b8),
            f64::from_bits(0xb9ffcb3f98612d27),
        ],
        [
            f64::from_bits(0x3f88f40b7b380000),
            f64::from_bits(0xbd462a0a4fd47580),
            f64::from_bits(0x3a23cb3c35d9f6a1),
        ],
        [
            f64::from_bits(0x3f8bb9d3abb00000),
            f64::from_bits(0xbd50ec48f94d7860),
            f64::from_bits(0xba26b47d410e4cc7),
        ],
        [
            f64::from_bits(0x3f8e7f9bb2300000),
            f64::from_bits(0x3d4e4415cbc97a00),
            f64::from_bits(0xba23729fdb677231),
        ],
        [
            f64::from_bits(0x3f90a2b224780000),
            f64::from_bits(0xbd5cb73f4505b030),
            f64::from_bits(0xba21b3b3a3bc370a),
        ],
        [
            f64::from_bits(0x3f92059691e80000),
            f64::from_bits(0xbd4abcc3412f2640),
            f64::from_bits(0xba0fe6e998e48673),
        ],
        [
            f64::from_bits(0x3f93687a76800000),
            f64::from_bits(0xbd543901e5c97a90),
            f64::from_bits(0x39fb54cdd52a5d88),
        ],
        [
            f64::from_bits(0x3f94cb5eb5d80000),
            f64::from_bits(0xbd58f106f00f13b8),
            f64::from_bits(0xba28f793f5fce148),
        ],
        [
            f64::from_bits(0x3f962e432b240000),
            f64::from_bits(0xbd5745af34bb54b8),
            f64::from_bits(0xb9e17e3ec05cde70),
        ],
    ],
    [
        [
            f64::from_bits(0x0000000000000000),
            f64::from_bits(0x0000000000000000),
            f64::from_bits(0x0000000000000000),
        ],
        [
            f64::from_bits(0x3f162e7b00000000),
            f64::from_bits(0xbd3868625640a680),
            f64::from_bits(0xba234bf0db910f65),
        ],
        [
            f64::from_bits(0x3f262e35f6000000),
            f64::from_bits(0xbd42ee3d96b696a0),
            f64::from_bits(0x3a1a2948cd558655),
        ],
        [
            f64::from_bits(0x3f30a2b4b2000000),
            f64::from_bits(0x3d053edbcf116500),
            f64::from_bits(0xb9ecfc26ccf6d0e4),
        ],
        [
            f64::from_bits(0x3f362e4be1000000),
            f64::from_bits(0x3cb783e334614000),
            f64::from_bits(0xba204b96da30e63a),
        ],
        [
            f64::from_bits(0x3f3bb9e085000000),
            f64::from_bits(0xbd460785f20acb20),
            f64::from_bits(0xb9ff33369bf7dff1),
        ],
        [
            f64::from_bits(0x3f40a2b94d000000),
            f64::from_bits(0x3d5fd4b3a2733530),
            f64::from_bits(0xb9f685a35575eff1),
        ],
        [
            f64::from_bits(0x3f4368810f800000),
            f64::from_bits(0x3d07ded26dc81300),
            f64::from_bits(0xb9f4c4d1abca79bf),
        ],
        [
            f64::from_bits(0x3f462e4787800000),
            f64::from_bits(0x3d57d2bee9a1f630),
            f64::from_bits(0x3a2860233b7ad130),
        ],
        [
            f64::from_bits(0x3f48f40cb4800000),
            f64::from_bits(0xbd5af034eaf471c0),
            f64::from_bits(0x3a1ae748822d57b7),
        ],
        [
            f64::from_bits(0x3f4bb9d094000000),
            f64::from_bits(0xbd57a223013a20f0),
            f64::from_bits(0xba21e499087075b6),
        ],
        [
            f64::from_bits(0x3f4e7fa32c800000),
            f64::from_bits(0xbd4b2e67b1b59bd0),
            f64::from_bits(0xba254a41eda30fa6),
        ],
        [
            f64::from_bits(0x3f50a2b237000000),
            f64::from_bits(0xbd37ad97ff4ac7a0),
            f64::from_bits(0x3a2f932da91371dd),
        ],
        [
            f64::from_bits(0x3f52059a33800000),
            f64::from_bits(0xbd396422d90df400),
            f64::from_bits(0xba190800fbbf2ed3),
        ],
        [
            f64::from_bits(0x3f53687982400000),
            f64::from_bits(0x3d30f90540018120),
            f64::from_bits(0x3a29567e01e48f9a),
        ],
        [
            f64::from_bits(0x3f54cb602c000000),
            f64::from_bits(0xbd40d709a5ec0b50),
            f64::from_bits(0x3a1253dfd44635d2),
        ],
        [
            f64::from_bits(0x3f562e462b400000),
            f64::from_bits(0x3d5061d003b97318),
            f64::from_bits(0x3a2d7faee66a2e1e),
        ],
    ],
    [
        [
            f64::from_bits(0x0000000000000000),
            f64::from_bits(0x0000000000000000),
            f64::from_bits(0x0000000000000000),
        ],
        [
            f64::from_bits(0x3ed63007c0000000),
            f64::from_bits(0xbd4db0e38e5aaaa0),
            f64::from_bits(0x3a2259a7b94815b9),
        ],
        [
            f64::from_bits(0x3ee6300f60000000),
            f64::from_bits(0x3d32b1c755804380),
            f64::from_bits(0x3a278cabba01e3e4),
        ],
        [
            f64::from_bits(0x3ef0a21150000000),
            f64::from_bits(0xbd55ff2237307590),
            f64::from_bits(0x3a08074feacfe49d),
        ],
        [
            f64::from_bits(0x3ef62e1ec0000000),
            f64::from_bits(0xbd285d6f6487ce40),
            f64::from_bits(0x3a205485074b9276),
        ],
        [
            f64::from_bits(0x3efbba3010000000),
            f64::from_bits(0xbd4af5d58a7c9210),
            f64::from_bits(0xba230a8c0fd2ff5f),
        ],
        [
            f64::from_bits(0x3f00a32298000000),
            f64::from_bits(0x3d4590faa0883bd0),
            f64::from_bits(0x3a295e9bda999947),
        ],
        [
            f64::from_bits(0x3f03682f10000000),
            f64::from_bits(0x3d5f0224376efaf8),
            f64::from_bits(0xba25843c0db50d10),
        ],
        [
            f64::from_bits(0x3f062e3d80000000),
            f64::from_bits(0xbd4142c13daed4a0),
            f64::from_bits(0x3a2c68a61183ce87),
        ],
        [
            f64::from_bits(0x3f08f44dd8000000),
            f64::from_bits(0xbd4aa489f3999310),
            f64::from_bits(0x3a111c5c376854ea),
        ],
        [
            f64::from_bits(0x3f0bb96010000000),
            f64::from_bits(0x3d59904d8b6a3638),
            f64::from_bits(0x3a28c89554493c8f),
        ],
        [
            f64::from_bits(0x3f0e7f7440000000),
            f64::from_bits(0x3d55785ddbe7cba8),
            f64::from_bits(0x3a1e7ff3cde7d70c),
        ],
        [
            f64::from_bits(0x3f10a2c530000000),
            f64::from_bits(0xbd46d9e8780d0d50),
            f64::from_bits(0x3a1ad9c178106693),
        ],
        [
            f64::from_bits(0x3f1205d134000000),
            f64::from_bits(0xbd4214a2e893fcc0),
            f64::from_bits(0x3a2548a9500c9822),
        ],
        [
            f64::from_bits(0x3f13685e28000000),
            f64::from_bits(0x3d4e235886461030),
            f64::from_bits(0x3a12a97b26da2d88),
        ],
        [
            f64::from_bits(0x3f14cb6c18000000),
            f64::from_bits(0x3d52b7cfcea9e0d8),
            f64::from_bits(0xba25095048a6b824),
        ],
        [
            f64::from_bits(0x3f162e7b00000000),
            f64::from_bits(0xbd3868625640a680),
            f64::from_bits(0xba234bf0db910f65),
        ],
    ],
];
const CH1: [[f64; 2]; 3] = [
    [
        f64::from_bits(0x3fe0000000000000),
        f64::from_bits(0x39024b67ee516e3b),
    ],
    [
        f64::from_bits(0xbfd0000000000000),
        f64::from_bits(0xb91932ce43199a8d),
    ],
    [
        f64::from_bits(0x3fc5555555555555),
        f64::from_bits(0x3c655540c15cf91f),
    ],
];
const CL1: [f64; 3] = [
    f64::from_bits(0xbfc0000000000000),
    f64::from_bits(0x3fb9999999a0754f),
    f64::from_bits(0xbfb55555555c3157),
];
const B: [[i64; 2]; 32] = [
    [301, 27565],
    [7189, 24786],
    [13383, 22167],
    [18923, 19696],
    [23845, 17361],
    [28184, 15150],
    [31969, 13054],
    [35231, 11064],
    [37996, 9173],
    [40288, 7372],
    [42129, 5657],
    [43542, 4020],
    [44546, 2457],
    [45160, 962],
    [45399, -468],
    [45281, -1838],
    [44821, -3151],
    [44032, -4412],
    [42929, -5622],
    [41522, -6786],
    [39825, -7905],
    [37848, -8982],
    [35602, -10020],
    [33097, -11020],
    [30341, -11985],
    [27345, -12916],
    [24115, -13816],
    [20661, -14685],
    [16989, -15526],
    [13107, -16339],
    [9022, -17126],
    [4740, -17889],
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
    let l = fma(ch, xh, -h) + (cl * xh + ch * xl);
    let r = h + l;
    (r, (h - r) + l)
}
fn muld(xh: f64, xl: f64, ch: f64) -> (f64, f64) {
    let h = ch * xh;
    let l = fma(ch, xh, -h) + ch * xl;
    let r = h + l;
    (r, (h - r) + l)
}
fn poly(xh: f64, xl: f64, c: &[[f64; 2]], l: f64) -> (f64, f64) {
    let i = c.len() - 1;
    let mut ch = c[i][0] + l;
    let mut cl = ((c[i][0] - ch) + l) + c[i][1];
    for i in (0..i).rev() {
        (ch, cl) = mul(xh, xl, ch, cl);
        let th = ch + c[i][0];
        let tl = (c[i][0] - th) + ch;
        ch = th;
        cl += tl + c[i][1];
    }
    (ch, cl)
}
#[inline(never)]
fn zero(x: f64, x2h: f64, x2l: f64) -> f64 {
    let y2 = x2h * (CL0[0] + x2h * (CL0[1] + x2h * (CL0[2] + x2h * (CL0[3] + x2h * CL0[4]))));
    let (y1, y2) = poly(x2h, x2l, &CH0, y2);
    let (y1, y2) = mul(y1, y2, x2h, x2l);
    let (y1, y2) = muld(y1, y2, x);
    let (y0, y1) = two(x, y1);
    let (mut y1, y2) = two(y1, y2);
    let mut t = y1.to_bits();
    if t & 0xfffffffffffff == 0 {
        if (y2.to_bits() ^ t) >> 63 != 0 {
            t = t.wrapping_sub(1);
        } else {
            t = t.wrapping_add(1);
        }
        y1 = f64::from_bits(t);
    }
    y0 + y1
}
#[inline(never)]
fn refine(x: f64, zh: f64, zl: f64, a: f64) -> f64 {
    let e = (zh.to_bits() >> 52) as i64 - 1023 + i64::from(zl == 0.0);
    let t = f64::from_bits((zh.to_bits() & 0xfffffffffffff) | (1023 << 52));
    let ed = e as f64;
    let v = (a - ed + f64::from_bits(0x3ff0000800000000)).to_bits();
    let i = ((v - (1023 << 52)) >> 36) as usize;
    let (i1, i2, i3, i4) = ((i >> 12) & 31, (i >> 8) & 15, (i >> 4) & 15, i & 15);
    let el2 = -f64::from_bits(0x3a49ff0342542fc3) * ed;
    let el1 = f64::from_bits(0x3d1ef35793c76800) * ed;
    let el0 = f64::from_bits(0x3fd62e42fefa3800) * ed;
    let mut l = [0.0; 3];
    for j in 0..3 {
        l[j] = LL[0][i1][j] + LL[1][i2][j] + (LL[2][i3][j] + LL[3][i4][j]);
    }
    l[0] += el0;
    let t12 = T1[i1] * T2[i2];
    let t34 = T3[i3] * T4[i4];
    let th = t12 * t34;
    let tl = fma(t12, t34, -th);
    let dh = th * t;
    let dl = fma(th, t, -dh);
    let sh = tl * t;
    let sl = fma(tl, t, -sh);
    let (xh, mut xl) = two(dh - 1.0, dl);
    if zl != 0.0 {
        let t = f64::from_bits(zl.to_bits().wrapping_sub((e as u64) << 52));
        xl += th * t;
    }
    let (xh, xl) = add(xh, xl, sh, sl);
    let sl = xh * (CL1[0] + xh * (CL1[1] + xh * CL1[2]));
    let (sh, sl) = poly(xh, xl, &CH1, sl);
    let (sh, sl) = mul(xh, xl, sh, sl);
    let (sh, sl) = add(sh, sl, el1, el2);
    let (sh, sl) = add(sh, sl, l[1], l[2]);
    let (v0, v2) = two(l[0], sh);
    let (v1, v2) = two(v2, sl);
    let (v0, v1) = two(v0, v1);
    let (v1, v2) = two(v1, v2);
    let s = 2.0f64.copysign(x);
    let v0 = v0 * s;
    let mut v1 = v1 * s;
    let v2 = v2 * s;
    let mut t = v1.to_bits();
    if t & 0xfffffffffffff == 0 {
        if (v2.to_bits() ^ t) >> 63 != 0 {
            t = t.wrapping_sub(1);
        } else {
            t = t.wrapping_add(1);
        }
        v1 = f64::from_bits(t);
    }
    let er = t.wrapping_add(41) & 0xfffffffffffff;
    let de = ((v0.to_bits() >> 52) & 2047).wrapping_sub((t >> 52) & 2047);
    let res = v0 + v1;
    if de > 99 || er < 80 {
        let s = 1.0f64.copysign(x);
        for row in DB {
            if x.abs() == row[0] {
                return s * row[1] + s * row[2];
            }
        }
    }
    res
}
/// Correctly rounded inverse hyperbolic sine, nearest-even.
pub fn asinh(x: f64) -> f64 {
    let ax = x.abs();
    let u = ax.to_bits();
    if u < 0x3fbb000000000000 {
        if u < 0x3e57137449123ef7 {
            if u == 0 {
                return x;
            }
            return fma(-f64::from_bits(0x3c30000000000000), x, x);
        }
        let x2h = x * x;
        let x2l = fma(x, x, -x2h);
        let x3h = x2h * x;
        let sl = if u < 0x3f93000000000000 {
            if u < 0x3f30000000000000 {
                if u < 0x3e5a000000000000 {
                    x3h * SM0[0]
                } else {
                    x3h * (SM1[0] + x2h * SM1[1])
                }
            } else {
                x3h * (SM2[0] + x2h * (SM2[1] + x2h * (SM2[2] + x2h * SM2[3])))
            }
        } else {
            let c1 = SM3[1] + x2h * SM3[2];
            let c3 = SM3[3] + x2h * SM3[4];
            let c5 = SM3[5] + x2h * SM3[6];
            let x4 = x2h * x2h;
            x3h * (SM3[0] + x2h * (c1 + x4 * (c3 + x4 * c5)))
        };
        let eps = f64::from_bits(0x3ca7900000000000) * x3h;
        let lb = x + (sl - eps);
        let ub = x + (sl + eps);
        return if lb == ub { lb } else { zero(x, x2h, x2l) };
    }
    let (mut x2h, mut x2l) = (0.0, 0.0);
    let mut off = 1023;
    let (ah, al) = if u < 0x4190000000000000 {
        x2h = x * x;
        x2l = fma(x, x, -x2h);
        let (th, tl) = if u < 0x3ff0000000000000 {
            two(1.0, x2h)
        } else {
            two(x2h, 1.0)
        };
        let tl = tl + x2l;
        let ah = crate::reference::sqrt(th);
        let rs = 0.5 / th;
        let al = (tl - fma(ah, ah, -th)) * (rs * ah);
        let (ah, tl) = two(ah, ax);
        (ah, al + tl)
    } else if u < 0x4330000000000000 {
        (2.0 * ax, 0.5 / ax)
    } else {
        if !x.is_finite() {
            return x + x;
        }
        off = 1022;
        (ax, 0.0)
    };
    let bits = ah.to_bits();
    let ed = ((bits >> 52) as i64 - off) as f64;
    let m = bits & 0xfffffffffffff;
    let i = (m >> 47) as usize;
    let d = m & 0x7fffffffffff;
    let j = (m
        .wrapping_add((B[i][0] as u64) << 33)
        .wrapping_add((B[i][1] * (d >> 16) as i64) as u64)
        >> 42) as usize;
    let (i1, i2) = (j >> 5, j & 31);
    let t = f64::from_bits(m | (1023 << 52));
    let r = R1[i1] * R2[i2];
    let dx = fma(r, t, -1.0);
    let dx2 = dx * dx;
    let f = dx2 * ((C1[0] + dx * C1[1]) + dx2 * ((C1[2] + dx * C1[3]) + dx2 * C1[4]));
    let lh = f64::from_bits(0x3fe62e42fefa3800) * ed + (L1[i1][1] + L2[i2][1]);
    let mut ll = f64::from_bits(0x3d2ef35793c76730) * ed + L1[i1][0] + L2[i2][0] + al / ah + f;
    ll += dx;
    let s = 1.0f64.copysign(x);
    let lh = lh * s;
    let ll = ll * s;
    let eps = 1.63e-19;
    let lb = lh + (ll - eps);
    let ub = lh + (ll + eps);
    if lb == ub {
        return lb;
    }
    if ax < 0.25 {
        return zero(x, x2h, x2l);
    }
    refine(x, ah, al, f64::from_bits(0x3ff71547652b82fe) * lb.abs())
}

/* Correctly-rounded inverse hyperbolic tangent function for binary64.

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
SOFTWARE. */
use super::fma;
const CH0: [[f64; 2]; 13] = [
    [
        f64::from_bits(0x3fd5555555555555),
        f64::from_bits(0x3c75555555555555),
    ],
    [
        f64::from_bits(0x3fc999999999999a),
        f64::from_bits(0xbc6999999999611c),
    ],
    [
        f64::from_bits(0x3fc2492492492492),
        f64::from_bits(0x3c62492490f76b25),
    ],
    [
        f64::from_bits(0x3fbc71c71c71c71c),
        f64::from_bits(0x3c5c71cd5c38a112),
    ],
    [
        f64::from_bits(0x3fb745d1745d1746),
        f64::from_bits(0xbc47556c4165f4ca),
    ],
    [
        f64::from_bits(0x3fb3b13b13b13b14),
        f64::from_bits(0xbc4b893c3b36052e),
    ],
    [
        f64::from_bits(0x3fb1111111111105),
        f64::from_bits(0x3c44e1afd723ed1f),
    ],
    [
        f64::from_bits(0x3fae1e1e1e1e2678),
        f64::from_bits(0xbc4f86ea96fb1435),
    ],
    [
        f64::from_bits(0x3faaf286bc9f90cc),
        f64::from_bits(0x3c31e51a6e54fde9),
    ],
    [
        f64::from_bits(0x3fa8618618c779b6),
        f64::from_bits(0xbc2ab913de95c3bf),
    ],
    [
        f64::from_bits(0x3fa642c84aa383eb),
        f64::from_bits(0x3c4632e747641b12),
    ],
    [
        f64::from_bits(0x3fa47ae2d205013c),
        f64::from_bits(0xbc30c9617e7bcff2),
    ],
    [
        f64::from_bits(0x3fa2f664d60473f9),
        f64::from_bits(0x3c23adb3e2b7f35e),
    ],
];
const CL0: [f64; 5] = [
    f64::from_bits(0x3fa1a9a91fd692af),
    f64::from_bits(0x3fa06dfbb35e7f44),
    f64::from_bits(0x3fa037bed4d7588f),
    f64::from_bits(0x3f95aca6d6d720d6),
    f64::from_bits(0x3fa99ea5700d53a5),
];
const C0: [f64; 9] = [
    f64::from_bits(0x3fc999999999999a),
    f64::from_bits(0x3fc2492492492244),
    f64::from_bits(0x3fbc71c71c79715f),
    f64::from_bits(0x3fb745d16f777723),
    f64::from_bits(0x3fb3b13ca4174634),
    f64::from_bits(0x3fb110c9724989bd),
    f64::from_bits(0x3fae2d17608a5b2e),
    f64::from_bits(0x3faa0b56308cba0b),
    f64::from_bits(0x3fafb6341208ad2e),
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
        f64::from_bits(0xbe4532c1269e2038),
        f64::from_bits(0x3f862e5000000000),
    ],
    [
        f64::from_bits(0x3e4ce42d81b54e84),
        f64::from_bits(0x3f962e3c00000000),
    ],
    [
        f64::from_bits(0xbe525826f815ec3d),
        f64::from_bits(0x3fa0a2ac00000000),
    ],
    [
        f64::from_bits(0x3e50db1b1e7cee11),
        f64::from_bits(0x3fa62e4a00000000),
    ],
    [
        f64::from_bits(0xbe51f3a8c6c95003),
        f64::from_bits(0x3fabb9dc00000000),
    ],
    [
        f64::from_bits(0xbe5774cd4fb8c30d),
        f64::from_bits(0x3fb0a2b200000000),
    ],
    [
        f64::from_bits(0x3e2452e56c030a0a),
        f64::from_bits(0x3fb3687f00000000),
    ],
    [
        f64::from_bits(0x3e36b63c4966a79a),
        f64::from_bits(0x3fb62e4100000000),
    ],
    [
        f64::from_bits(0xbe3b20a21ccb525e),
        f64::from_bits(0x3fb8f40a00000000),
    ],
    [
        f64::from_bits(0x3e54006cfb3d8f85),
        f64::from_bits(0x3fbbb9d100000000),
    ],
    [
        f64::from_bits(0xbe5cdb026b310c41),
        f64::from_bits(0x3fbe7f9b00000000),
    ],
    [
        f64::from_bits(0xbe569124fdc0f16d),
        f64::from_bits(0x3fc0a2b080000000),
    ],
    [
        f64::from_bits(0xbe5084656cdc2727),
        f64::from_bits(0x3fc2059580000000),
    ],
    [
        f64::from_bits(0xbe5376fa8b0357fd),
        f64::from_bits(0x3fc3687c00000000),
    ],
    [
        f64::from_bits(0x3e3e56ae55a47b4a),
        f64::from_bits(0x3fc4cb5e80000000),
    ],
    [
        f64::from_bits(0x3e5070ff8834eeb4),
        f64::from_bits(0x3fc62e4400000000),
    ],
    [
        f64::from_bits(0x3e5623516109f4fe),
        f64::from_bits(0x3fc7912900000000),
    ],
    [
        f64::from_bits(0xbe2ec656b95fbdac),
        f64::from_bits(0x3fc8f40b00000000),
    ],
    [
        f64::from_bits(0x3e3f0ca2e729f510),
        f64::from_bits(0x3fca56ed80000000),
    ],
    [
        f64::from_bits(0xbe57d260a858354a),
        f64::from_bits(0x3fcbb9d680000000),
    ],
    [
        f64::from_bits(0x3e4e7279075503d3),
        f64::from_bits(0x3fcd1cb900000000),
    ],
    [
        f64::from_bits(0x3e439e1a0a503873),
        f64::from_bits(0x3fce7f9d00000000),
    ],
    [
        f64::from_bits(0x3e5cd86d7b87c3d6),
        f64::from_bits(0x3fcfe27d80000000),
    ],
    [
        f64::from_bits(0x3e5060ab88de341e),
        f64::from_bits(0x3fd0a2b240000000),
    ],
    [
        f64::from_bits(0x3e320a860d3f9390),
        f64::from_bits(0x3fd1542440000000),
    ],
    [
        f64::from_bits(0xbe4dacee95fc2f10),
        f64::from_bits(0x3fd2059740000000),
    ],
    [
        f64::from_bits(0x3e545de3a86e0aca),
        f64::from_bits(0x3fd2b70700000000),
    ],
    [
        f64::from_bits(0x3e4c164cbfb991af),
        f64::from_bits(0x3fd3687b00000000),
    ],
    [
        f64::from_bits(0x3e5d3f66b24225ef),
        f64::from_bits(0x3fd419ec40000000),
    ],
    [
        f64::from_bits(0x3e5fc023efa144ba),
        f64::from_bits(0x3fd4cb5f80000000),
    ],
    [
        f64::from_bits(0x3e3086a8af6f26c0),
        f64::from_bits(0x3fd57cd280000000),
    ],
    [
        f64::from_bits(0xbe105c610ca86c39),
        f64::from_bits(0x3fd62e4300000000),
    ],
];
const L2: [[f64; 2]; 33] = [
    [
        f64::from_bits(0x0000000000000000),
        f64::from_bits(0x0000000000000000),
    ],
    [
        f64::from_bits(0xbe337e152a129e4e),
        f64::from_bits(0x3f36320000000000),
    ],
    [
        f64::from_bits(0xbe53f6c916b8be9c),
        f64::from_bits(0x3f46300000000000),
    ],
    [
        f64::from_bits(0x3e520505936739d5),
        f64::from_bits(0x3f50a24000000000),
    ],
    [
        f64::from_bits(0xbe523e2e8cb541ba),
        f64::from_bits(0x3f562dc000000000),
    ],
    [
        f64::from_bits(0xbdfacb7983ac4f5e),
        f64::from_bits(0x3f5bba0000000000),
    ],
    [
        f64::from_bits(0x3e36f7c7689c63ae),
        f64::from_bits(0x3f60a2a000000000),
    ],
    [
        f64::from_bits(0x3e1f5ca695b4c58b),
        f64::from_bits(0x3f6368c000000000),
    ],
    [
        f64::from_bits(0xbe4c6c18bd953226),
        f64::from_bits(0x3f662e6000000000),
    ],
    [
        f64::from_bits(0x3e57a516c34846bd),
        f64::from_bits(0x3f68f46000000000),
    ],
    [
        f64::from_bits(0xbe4f3b83dd8b8530),
        f64::from_bits(0x3f6bba0000000000),
    ],
    [
        f64::from_bits(0xbe0c3459046e4e57),
        f64::from_bits(0x3f6e800000000000),
    ],
    [
        f64::from_bits(0x3d9b5c7e34cb79f6),
        f64::from_bits(0x3f70a2c000000000),
    ],
    [
        f64::from_bits(0xbe42487e9af9a692),
        f64::from_bits(0x3f7205c000000000),
    ],
    [
        f64::from_bits(0x3e5f21bbc4ad79ce),
        f64::from_bits(0x3f73687000000000),
    ],
    [
        f64::from_bits(0xbe2550ffc857b731),
        f64::from_bits(0x3f74cb7000000000),
    ],
    [
        f64::from_bits(0x3e487458ec1b7b34),
        f64::from_bits(0x3f762e2000000000),
    ],
    [
        f64::from_bits(0x3e5103d4fe83ee81),
        f64::from_bits(0x3f77911000000000),
    ],
    [
        f64::from_bits(0x3e4810483d3b398c),
        f64::from_bits(0x3f78f44000000000),
    ],
    [
        f64::from_bits(0xbe42085cb340608e),
        f64::from_bits(0x3f7a573000000000),
    ],
    [
        f64::from_bits(0x3e512698a119c42f),
        f64::from_bits(0x3f7bb9d000000000),
    ],
    [
        f64::from_bits(0xbe5edb8c172b4c33),
        f64::from_bits(0x3f7d1cc000000000),
    ],
    [
        f64::from_bits(0xbe58b55b87a5e238),
        f64::from_bits(0x3f7e7fe000000000),
    ],
    [
        f64::from_bits(0x3e5be5e17763f78a),
        f64::from_bits(0x3f7fe2b000000000),
    ],
    [
        f64::from_bits(0xbe1c2d496790073e),
        f64::from_bits(0x3f80a2a800000000),
    ],
    [
        f64::from_bits(0x3e56542f523abeec),
        f64::from_bits(0x3f81541000000000),
    ],
    [
        f64::from_bits(0xbe5b7fdbe5b193f8),
        f64::from_bits(0x3f8205a000000000),
    ],
    [
        f64::from_bits(0x3e5fa4d42fe30c7c),
        f64::from_bits(0x3f82b70000000000),
    ],
    [
        f64::from_bits(0x3e50d46ad04adc86),
        f64::from_bits(0x3f83688800000000),
    ],
    [
        f64::from_bits(0xbe51c22d02d17c4c),
        f64::from_bits(0x3f8419f000000000),
    ],
    [
        f64::from_bits(0x3e1a7d1e330dccce),
        f64::from_bits(0x3f84cb7000000000),
    ],
    [
        f64::from_bits(0x3e0187025e656ba3),
        f64::from_bits(0x3f857cd000000000),
    ],
    [
        f64::from_bits(0xbe4532c1269e2038),
        f64::from_bits(0x3f862e5000000000),
    ],
];
const C1: [f64; 5] = [
    f64::from_bits(0xbff0000000000000),
    f64::from_bits(0x3ff5555555555530),
    f64::from_bits(0xbfffffffffffffa0),
    f64::from_bits(0x40099999e33a6366),
    f64::from_bits(0xc01555559ef9525f),
];
const DB: [[f64; 3]; 2] = [
    [
        f64::from_bits(0x3fd2dbb7b1c91363),
        f64::from_bits(0x3fd36f33d51c264d),
        f64::from_bits(0x3c70000000000000),
    ],
    [
        f64::from_bits(0x3fdc493dc899e4a5),
        f64::from_bits(0x3fde611aa58ab608),
        f64::from_bits(0xbc70000000000000),
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
fn sub(x: f64, y: f64) -> (f64, f64) {
    let s = x - y;
    (s, (x - s) - y)
}
fn add(xh: f64, xl: f64, ch: f64, cl: f64) -> (f64, f64) {
    let s = xh + ch;
    let d = s - xh;
    (s, ((ch - d) + (xh + (d - s))) + (xl + cl))
}
fn mul(xh: f64, xl: f64, ch: f64, cl: f64) -> (f64, f64) {
    let ahlh = ch * xl;
    let alhh = cl * xh;
    let ahhh = ch * xh;
    let ahhl = fma(ch, xh, -ahhh);
    (ahhh, ahhl + (alhh + ahlh))
}
fn mul_acc(xh: f64, xl: f64, ch: f64, cl: f64) -> (f64, f64) {
    let (h, l) = mul(xh, xl, ch, cl);
    two(h, l)
}
fn mul_d(xh: f64, xl: f64, ch: f64) -> (f64, f64) {
    let h = xh * ch;
    (h, fma(ch, xh, -h) + xl * ch)
}
fn poly(xh: f64, xl: f64, c: &[[f64; 2]], l: f64) -> (f64, f64) {
    let i = c.len() - 1;
    let (mut ch, mut cl) = two(c[i][0], l);
    cl += c[i][1];
    for i in (0..i).rev() {
        (ch, cl) = mul_acc(xh, xl, ch, cl);
        let (th, tl) = two(c[i][0], ch);
        ch = th;
        cl += tl + c[i][1];
    }
    (ch, cl)
}
#[inline(never)]
fn zero(x: f64) -> f64 {
    let x2 = x * x;
    let x2l = fma(x, x, -x2);
    let y2 = x2 * (CL0[0] + x2 * (CL0[1] + x2 * (CL0[2] + x2 * (CL0[3] + x2 * CL0[4]))));
    let (y1, y2) = poly(x2, x2l, &CH0, y2);
    let (y1, y2) = mul_d(y1, y2, x);
    let (y1, y2) = mul_acc(x2, x2l, y1, y2);
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
    let e = (zh.to_bits() >> 52) as i64 - 1023;
    let t = f64::from_bits((zh.to_bits() & 0xfffffffffffff) | (1023 << 52));
    let ed = e as f64;
    let v = (a - ed + f64::from_bits(0x3ff0000800000000)).to_bits();
    let i = ((v - (1023 << 52)) >> 36) as usize;
    let (i1, i2, i3, i4) = (i >> 12, (i >> 8) & 15, (i >> 4) & 15, i & 15);
    let el2 = f64::from_bits(0x397f97b57a079a00) * ed;
    let el1 = -f64::from_bits(0x3cd0ca86c3898d00) * ed;
    let el0 = f64::from_bits(0x3fd62e42fefa3a00) * ed;
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
    let t = f64::from_bits(zl.to_bits().wrapping_sub((e as u64) << 52));
    xl += th * t;
    let (xh, xl) = add(xh, xl, sh, sl);
    let sl = xh * (CL1[0] + xh * (CL1[1] + xh * CL1[2]));
    let (sh, sl) = poly(xh, xl, &CH1, sl);
    let (sh, sl) = mul_acc(xh, xl, sh, sl);
    let (sh, sl) = add(sh, sl, el1, el2);
    let (sh, sl) = add(sh, sl, l[1], l[2]);
    let (v0, v2) = two(l[0], sh);
    let (mut v1, _) = two(v2, sl);
    let mut t = v1.to_bits();
    if t & 0xfffffffffffff == 0 {
        t = t.wrapping_add(1);
        v1 = f64::from_bits(t);
    }
    let er = t.wrapping_add(1) & 0xfffffffffffff;
    let de = ((v0.to_bits() >> 52) & 2047).wrapping_sub((t >> 52) & 2047);
    let s = 1.0f64.copysign(x);
    let res = v0 * s + v1 * s;
    if de > 104 || er < 3 {
        for row in DB {
            if x.abs() == row[0] {
                return s * row[1] + s * row[2];
            }
        }
    }
    res
}
/// Correctly rounded inverse hyperbolic tangent, nearest-even.
pub fn atanh(x: f64) -> f64 {
    let ax = x.abs();
    let aix = ax.to_bits();
    if aix >= 0x3ff0000000000000 {
        return if ax == 1.0 {
            f64::INFINITY.copysign(x)
        } else if x.is_nan() {
            x + x
        } else {
            f64::NAN
        };
    }
    if aix < 0x3fd0000000000000 {
        if aix < 0x3e4d12ed0af1a27f {
            return fma(x, f64::from_bits(0x3c80000000000000), x);
        }
        let x2 = x * x;
        let dx2 = fma(x, x, -x2);
        let x4 = x2 * x2;
        let x3 = x2 * x;
        let x8 = x4 * x4;
        let dx3 = fma(x2, x, -x3) + dx2 * x;
        let c = C0;
        let p = (c[0] + x2 * c[1])
            + x4 * (c[2] + x2 * c[3])
            + x8 * ((c[4] + x2 * c[5]) + x4 * (c[6] + x2 * c[7]) + x8 * c[8]);
        let t = fma(x2, p, f64::from_bits(0x3c75555555555555));
        let (ph, pl) = two(f64::from_bits(0x3fd5555555555555), t);
        let (ph, pl) = mul(ph, pl, x3, dx3);
        let (ph, tl) = two(x, ph);
        let pl = pl + tl;
        let eps =
            x * (x4 * f64::from_bits(0x3cad000000000000) + f64::from_bits(0x3980000000000000));
        let lb = ph + (pl - eps);
        let ub = ph + (pl + eps);
        return if lb == ub { lb } else { zero(x) };
    }
    let (ph, pl) = two(1.0, ax);
    let (qh, ql) = sub(1.0, ax);
    let iqh = 1.0 / qh;
    let th = ph * iqh;
    let tl = fma(ph, iqh, -th) + (pl + ph * (fma(-qh, iqh, 1.0) - ql * iqh)) * iqh;
    let bits = th.to_bits();
    let ed = ((bits >> 52) as i64 - 1023) as f64;
    let m = bits & 0xfffffffffffff;
    let i = (m >> 47) as usize;
    let d = m & 0x7fffffffffff;
    let j = (m
        .wrapping_add((B[i][0] as u64) << 33)
        .wrapping_add((B[i][1] * (d >> 16) as i64) as u64)
        >> 42) as usize;
    let (i1, i2) = (j >> 5, j & 31);
    let t = f64::from_bits(m | (1023 << 52));
    let r = (0.5 * R1[i1]) * R2[i2];
    let dx = fma(r, t, -0.5);
    let dx2 = dx * dx;
    let rx = r * t;
    let dxl = fma(r, t, -rx);
    let c = C1;
    let f = dx2 * ((c[0] + dx * c[1]) + dx2 * (c[2] + dx * c[3] + dx2 * c[4]));
    let lh = (L1[i1][1] + L2[i2][1]) + f64::from_bits(0x3fd62e42fefa3a00) * ed;
    let (lh, mut ll) = two(lh, rx - 0.5);
    ll += -f64::from_bits(0x3cd0ca86c3898d00) * ed + (L1[i1][0] + L2[i2][0]) + dxl + 0.5 * tl / th;
    ll += f;
    let s = 1.0f64.copysign(x);
    let lh = lh * s;
    let ll = ll * s;
    let eps = 38e-24 + dx2 * f64::from_bits(0x3ce0000000000000);
    let lb = lh + (ll - eps);
    let ub = lh + (ll + eps);
    if lb == ub {
        return lb;
    }
    let (th, tl) = two(th, tl);
    refine(
        x,
        th,
        tl,
        f64::from_bits(0x40071547652b82fe) * (lh + ll).abs(),
    )
}

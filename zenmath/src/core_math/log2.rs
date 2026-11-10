/* Correctly rounded base-2 logarithm of binary64 values.

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
const R1: [f64; 33] = [
    f64::from_bits(0x3ff7154800000000),
    f64::from_bits(0x3ff696af49200000),
    f64::from_bits(0x3ff61ab3fb680000),
    f64::from_bits(0x3ff5a18441680000),
    f64::from_bits(0x3ff52adadb480000),
    f64::from_bits(0x3ff4b6b7c9080000),
    f64::from_bits(0x3ff4451b0aa80000),
    f64::from_bits(0x3ff3d5ed8ae00000),
    f64::from_bits(0x3ff3691834680000),
    f64::from_bits(0x3ff2fe9b07400000),
    f64::from_bits(0x3ff2967603680000),
    f64::from_bits(0x3ff2307afe500000),
    f64::from_bits(0x3ff1cca9f7f80000),
    f64::from_bits(0x3ff16b02f0600000),
    f64::from_bits(0x3ff10b85e7880000),
    f64::from_bits(0x3ff0ae04b2e00000),
    f64::from_bits(0x3ff0527f52680000),
    f64::from_bits(0x3feff1eb8c400000),
    f64::from_bits(0x3fef42a1f1800000),
    f64::from_bits(0x3fee9721d4900000),
    f64::from_bits(0x3fedef6b35700000),
    f64::from_bits(0x3fed4b21bf000000),
    f64::from_bits(0x3fecaa739bd00000),
    f64::from_bits(0x3fec0d32a1500000),
    f64::from_bits(0x3feb735ecf800000),
    f64::from_bits(0x3feadcc9fbd00000),
    f64::from_bits(0x3fea497426400000),
    f64::from_bits(0x3fe9b92f24400000),
    f64::from_bits(0x3fe92c2920600000),
    f64::from_bits(0x3fe8a205c5800000),
    f64::from_bits(0x3fe81af33e300000),
    f64::from_bits(0x3fe796c35fe00000),
    f64::from_bits(0x3fe7154800000000),
];
const R2: [f64; 33] = [
    f64::from_bits(0x3ff0000000000000),
    f64::from_bits(0x3feffa7000000000),
    f64::from_bits(0x3feff4f000000000),
    f64::from_bits(0x3fefef6000000000),
    f64::from_bits(0x3fefe9e000000000),
    f64::from_bits(0x3fefe45000000000),
    f64::from_bits(0x3fefded000000000),
    f64::from_bits(0x3fefd94000000000),
    f64::from_bits(0x3fefd3c000000000),
    f64::from_bits(0x3fefce4000000000),
    f64::from_bits(0x3fefc8c000000000),
    f64::from_bits(0x3fefc34000000000),
    f64::from_bits(0x3fefbdc000000000),
    f64::from_bits(0x3fefb84000000000),
    f64::from_bits(0x3fefb2c000000000),
    f64::from_bits(0x3fefad4000000000),
    f64::from_bits(0x3fefa7c000000000),
    f64::from_bits(0x3fefa24000000000),
    f64::from_bits(0x3fef9cd000000000),
    f64::from_bits(0x3fef975000000000),
    f64::from_bits(0x3fef91e000000000),
    f64::from_bits(0x3fef8c6000000000),
    f64::from_bits(0x3fef86f000000000),
    f64::from_bits(0x3fef817000000000),
    f64::from_bits(0x3fef7c0000000000),
    f64::from_bits(0x3fef769000000000),
    f64::from_bits(0x3fef711000000000),
    f64::from_bits(0x3fef6ba000000000),
    f64::from_bits(0x3fef663000000000),
    f64::from_bits(0x3fef60c000000000),
    f64::from_bits(0x3fef5b5000000000),
    f64::from_bits(0x3fef55e000000000),
    f64::from_bits(0x3fef507000000000),
];
const L1: [[f64; 2]; 33] = [
    [f64::from_bits(0x0), f64::from_bits(0x0)],
    [
        f64::from_bits(0x3e51435edc775b51),
        f64::from_bits(0x3f9ffe3800000000),
    ],
    [
        f64::from_bits(0x3e5c8f1cbf9e4073),
        f64::from_bits(0x3fb000bc00000000),
    ],
    [
        f64::from_bits(0xbe57bf30fa53957b),
        f64::from_bits(0x3fb7ff9400000000),
    ],
    [
        f64::from_bits(0x3e4674d30b6276ed),
        f64::from_bits(0x3fbfffa600000000),
    ],
    [
        f64::from_bits(0x3e399fcf0d796ace),
        f64::from_bits(0x3fc4000580000000),
    ],
    [
        f64::from_bits(0xbe55ffd8b92706d2),
        f64::from_bits(0x3fc7ffe500000000),
    ],
    [
        f64::from_bits(0x3e5bef90bc5a116d),
        f64::from_bits(0x3fcbffc300000000),
    ],
    [
        f64::from_bits(0x3e5fca73b3d53f0d),
        f64::from_bits(0x3fcffffd00000000),
    ],
    [
        f64::from_bits(0xbe441024e560e04e),
        f64::from_bits(0x3fd2000d00000000),
    ],
    [
        f64::from_bits(0x3e5484024fad8461),
        f64::from_bits(0x3fd3ffcc00000000),
    ],
    [
        f64::from_bits(0x3e52d9ab90ba7694),
        f64::from_bits(0x3fd5ffe180000000),
    ],
    [
        f64::from_bits(0xbe52c998ea30ba7b),
        f64::from_bits(0x3fd80013c0000000),
    ],
    [
        f64::from_bits(0xbe475fc8682f918e),
        f64::from_bits(0x3fda0024c0000000),
    ],
    [
        f64::from_bits(0xbe5f02f268a85fb8),
        f64::from_bits(0x3fdbffd300000000),
    ],
    [
        f64::from_bits(0xbe450030ea7fae4b),
        f64::from_bits(0x3fddffd7c0000000),
    ],
    [
        f64::from_bits(0xbe3bffb8da5b849d),
        f64::from_bits(0x3fdffff980000000),
    ],
    [
        f64::from_bits(0x3e3f71993ff95475),
        f64::from_bits(0x3fe0fffd40000000),
    ],
    [
        f64::from_bits(0x3e288028e67f78fa),
        f64::from_bits(0x3fe20010c0000000),
    ],
    [
        f64::from_bits(0x3e5f915f5a0b4e89),
        f64::from_bits(0x3fe30018a0000000),
    ],
    [
        f64::from_bits(0x3e1c2fe288f968f8),
        f64::from_bits(0x3fe3fff480000000),
    ],
    [
        f64::from_bits(0x3e57375a75ae0837),
        f64::from_bits(0x3fe5001300000000),
    ],
    [
        f64::from_bits(0x3e291e48be920323),
        f64::from_bits(0x3fe6000ea0000000),
    ],
    [
        f64::from_bits(0x3e2ee7bc0d39a3db),
        f64::from_bits(0x3fe7001120000000),
    ],
    [
        f64::from_bits(0xbe2899e2ac5f778c),
        f64::from_bits(0x3fe7fffb40000000),
    ],
    [
        f64::from_bits(0x3e5f1f20176130a7),
        f64::from_bits(0x3fe8fffae0000000),
    ],
    [
        f64::from_bits(0x3e403fc59d34a4f3),
        f64::from_bits(0x3fe9fff1e0000000),
    ],
    [
        f64::from_bits(0xbdf68722010e4653),
        f64::from_bits(0x3feb0012a0000000),
    ],
    [
        f64::from_bits(0xbe558783d505a6ec),
        f64::from_bits(0x3febffeb80000000),
    ],
    [
        f64::from_bits(0x3e4b212ab9f8d51d),
        f64::from_bits(0x3fed0004e0000000),
    ],
    [
        f64::from_bits(0xbe5b3ecd767be776),
        f64::from_bits(0x3fedffeb20000000),
    ],
    [
        f64::from_bits(0x3e4b1ad41f07fc10),
        f64::from_bits(0x3feeffd520000000),
    ],
    [f64::from_bits(0x0), f64::from_bits(0x3ff0000000000000)],
];
const L2: [[f64; 2]; 33] = [
    [f64::from_bits(0x0), f64::from_bits(0x0)],
    [
        f64::from_bits(0xbe4e2b19f9c7b840),
        f64::from_bits(0x3f500e4000000000),
    ],
    [
        f64::from_bits(0xbe4b1d68137631fe),
        f64::from_bits(0x3f5ff10000000000),
    ],
    [
        f64::from_bits(0xbe4aa92227513fc3),
        f64::from_bits(0x3f68026000000000),
    ],
    [
        f64::from_bits(0xbe452918e3ab6f5e),
        f64::from_bits(0x3f6ff68000000000),
    ],
    [
        f64::from_bits(0x3e41a0b9b9010a9c),
        f64::from_bits(0x3f74019000000000),
    ],
    [
        f64::from_bits(0x3e509b87f57867ec),
        f64::from_bits(0x3f77fd0000000000),
    ],
    [
        f64::from_bits(0xbe33d6b70c673be6),
        f64::from_bits(0x3f7c04c000000000),
    ],
    [
        f64::from_bits(0x3e476d340a6780ab),
        f64::from_bits(0x3f8000c800000000),
    ],
    [
        f64::from_bits(0x3e5e4181e37d9e05),
        f64::from_bits(0x3f81ff8800000000),
    ],
    [
        f64::from_bits(0xbe48c181f042b901),
        f64::from_bits(0x3f83fea800000000),
    ],
    [
        f64::from_bits(0x3e49890ae7761d66),
        f64::from_bits(0x3f85fe1800000000),
    ],
    [
        f64::from_bits(0xbe555977ae613d5f),
        f64::from_bits(0x3f87fde800000000),
    ],
    [
        f64::from_bits(0x3e4c75f49acf5e56),
        f64::from_bits(0x3f89fe0800000000),
    ],
    [
        f64::from_bits(0xbe3e1fc84d0d42bc),
        f64::from_bits(0x3f8bfe8800000000),
    ],
    [
        f64::from_bits(0xbe55ab8a182ed279),
        f64::from_bits(0x3f8dff6000000000),
    ],
    [
        f64::from_bits(0xbe5b389a7d8a21d9),
        f64::from_bits(0x3f90004800000000),
    ],
    [
        f64::from_bits(0xbe57376f1a891fa5),
        f64::from_bits(0x3f91010c00000000),
    ],
    [
        f64::from_bits(0xbe30915581b87a8a),
        f64::from_bits(0x3f91ff1000000000),
    ],
    [
        f64::from_bits(0x3e4b7f2eaa894fca),
        f64::from_bits(0x3f93002c00000000),
    ],
    [
        f64::from_bits(0xbe5f8c41bdd38c23),
        f64::from_bits(0x3f93fe8c00000000),
    ],
    [
        f64::from_bits(0x3e304ffb1f3c8215),
        f64::from_bits(0x3f95000000000000),
    ],
    [
        f64::from_bits(0x3e5a398ccf2b137d),
        f64::from_bits(0x3f95feb400000000),
    ],
    [
        f64::from_bits(0x3e5373b5bce07f39),
        f64::from_bits(0x3f97008400000000),
    ],
    [
        f64::from_bits(0xbe4e42040fd9d454),
        f64::from_bits(0x3f97ff9400000000),
    ],
    [
        f64::from_bits(0x3e4ea00eb5770526),
        f64::from_bits(0x3f98fecc00000000),
    ],
    [
        f64::from_bits(0xbe53e2002bb54bb8),
        f64::from_bits(0x3f9a012400000000),
    ],
    [
        f64::from_bits(0x3e5797ec835db8d3),
        f64::from_bits(0x3f9b00b400000000),
    ],
    [
        f64::from_bits(0x3e3bdcd45fedd285),
        f64::from_bits(0x3f9c007400000000),
    ],
    [
        f64::from_bits(0xbe333b8f09d1a210),
        f64::from_bits(0x3f9d006000000000),
    ],
    [
        f64::from_bits(0xbe458937a8eea36f),
        f64::from_bits(0x3f9e007800000000),
    ],
    [
        f64::from_bits(0xbe43e76203ff54cd),
        f64::from_bits(0x3f9f00bc00000000),
    ],
    [
        f64::from_bits(0xbe16560f19fc3f41),
        f64::from_bits(0x3fa0009600000000),
    ],
];
const C: [f64; 4] = [
    f64::from_bits(0xbfd62e41d56c6400),
    f64::from_bits(0x3fc47fd2632d2d32),
    f64::from_bits(0xbfb5504497831ba7),
    f64::from_bits(0x3fa7a3314c5bef3c),
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
            f64::from_bits(0x0),
            f64::from_bits(0x0),
            f64::from_bits(0x0),
        ],
        [
            f64::from_bits(0x3fb000001fda0000),
            f64::from_bits(0x3d55ed58a7ff2c40),
            f64::from_bits(0x39f512acbb219717),
        ],
        [
            f64::from_bits(0x3fbfffffda070000),
            f64::from_bits(0xbd0b38ae1c2d5400),
            f64::from_bits(0xba210f11908c5c8d),
        ],
        [
            f64::from_bits(0x3fc7ffffefb50000),
            f64::from_bits(0x3d4785d91bb08320),
            f64::from_bits(0x3a2f5ded30ca48c0),
        ],
        [
            f64::from_bits(0x3fd0000004a60000),
            f64::from_bits(0x3d5380abf8fe7eb0),
            f64::from_bits(0xba2ebd3f567df886),
        ],
        [
            f64::from_bits(0x3fd4000009760000),
            f64::from_bits(0x3d4d321385bf0a10),
            f64::from_bits(0xba1c95a3d59b6c33),
        ],
        [
            f64::from_bits(0x3fd80000081b4000),
            f64::from_bits(0x3d5b6e8f25ff2610),
            f64::from_bits(0xba24072d39d44270),
        ],
        [
            f64::from_bits(0x3fdc00000f4f4000),
            f64::from_bits(0x3d4a5fed334580a0),
            f64::from_bits(0xb9f6f8fdabadd77a),
        ],
        [
            f64::from_bits(0x3fdffffff9de0000),
            f64::from_bits(0xbd4f85a54fb4a600),
            f64::from_bits(0xb9f50083897ee638),
        ],
        [
            f64::from_bits(0x3fe1fffffa35c000),
            f64::from_bits(0x3d5795db9d185140),
            f64::from_bits(0xba1594f07d4d693f),
        ],
        [
            f64::from_bits(0x3fe3fffff9cbc000),
            f64::from_bits(0x3d32fb09a99477a0),
            f64::from_bits(0x3a1b2c6248382152),
        ],
        [
            f64::from_bits(0x3fe5fffff6df2000),
            f64::from_bits(0xbd53ecf24077c3a0),
            f64::from_bits(0xba2382c6b4e2daa6),
        ],
        [
            f64::from_bits(0x3fe800000552e000),
            f64::from_bits(0x3d47044806f72380),
            f64::from_bits(0xba26da3eb126c999),
        ],
        [
            f64::from_bits(0x3fe9fffffc182000),
            f64::from_bits(0x3d45cf031d5d4cf0),
            f64::from_bits(0x3a2302c44da79f4b),
        ],
        [
            f64::from_bits(0x3febfffff6b62000),
            f64::from_bits(0x3d548d11d13372a8),
            f64::from_bits(0x3a2f045032543c0d),
        ],
        [
            f64::from_bits(0x3fedfffff7a42000),
            f64::from_bits(0xbd40cbf3e62fa860),
            f64::from_bits(0x3a21cfaf84227211),
        ],
        [
            f64::from_bits(0x3ff0000000000000),
            f64::from_bits(0x0),
            f64::from_bits(0x0),
        ],
    ],
    [
        [
            f64::from_bits(0x0),
            f64::from_bits(0x0),
            f64::from_bits(0x0),
        ],
        [
            f64::from_bits(0x3f7000024a000000),
            f64::from_bits(0xbd5799935d030650),
            f64::from_bits(0xba0cacad0b61c964),
        ],
        [
            f64::from_bits(0x3f8000014a880000),
            f64::from_bits(0xbd586dd926d35ee8),
            f64::from_bits(0xba2277ab54c00e64),
        ],
        [
            f64::from_bits(0x3f87ffff7b380000),
            f64::from_bits(0x3d15ffc7ff00e600),
            f64::from_bits(0x3a25870383c1225d),
        ],
        [
            f64::from_bits(0x3f9000004b940000),
            f64::from_bits(0xbd44aa9da7ea0120),
            f64::from_bits(0xba247decfe981022),
        ],
        [
            f64::from_bits(0x3f94000064380000),
            f64::from_bits(0xbd5015d79fd24c70),
            f64::from_bits(0x3a2ec26c231ddb4c),
        ],
        [
            f64::from_bits(0x3f97ffff8ff80000),
            f64::from_bits(0xbd5a62a2b755f048),
            f64::from_bits(0xba2eebe936e9a3f9),
        ],
        [
            f64::from_bits(0x3f9c000048180000),
            f64::from_bits(0x3d1d1a1bbce5f580),
            f64::from_bits(0xba24dd25e141ad16),
        ],
        [
            f64::from_bits(0x3fa0000050ee0000),
            f64::from_bits(0xbd5b32450688d5e0),
            f64::from_bits(0x3a2577a1c80a65b0),
        ],
        [
            f64::from_bits(0x3fa2000014760000),
            f64::from_bits(0xbd3bac8f6c103d40),
            f64::from_bits(0xba1157a2472c9532),
        ],
        [
            f64::from_bits(0x3fa3fffff2440000),
            f64::from_bits(0xbd4fbab0c9d05aa0),
            f64::from_bits(0xba13680ffabf1a5b),
        ],
        [
            f64::from_bits(0x3fa5ffffb1ce0000),
            f64::from_bits(0x3d4ef9ec2ba49140),
            f64::from_bits(0xba224850720b73ab),
        ],
        [
            f64::from_bits(0x3fa7ffffd9600000),
            f64::from_bits(0x3d514e3a91abeff0),
            f64::from_bits(0xba140e3565c114dd),
        ],
        [
            f64::from_bits(0x3faa0000321c0000),
            f64::from_bits(0x3d3e33c6a0732340),
            f64::from_bits(0xba27e8c937a981d8),
        ],
        [
            f64::from_bits(0x3fabffffc56a0000),
            f64::from_bits(0xbd52076781ff0460),
            f64::from_bits(0xba296e054ec8f31e),
        ],
        [
            f64::from_bits(0x3fadffffdba40000),
            f64::from_bits(0xbd4ea8b4476edde0),
            f64::from_bits(0xb9e07bfb9fde770f),
        ],
        [
            f64::from_bits(0x3fb000001fda0000),
            f64::from_bits(0x3d55ed58a7ff2c40),
            f64::from_bits(0x39f512acbb219717),
        ],
    ],
    [
        [
            f64::from_bits(0x0),
            f64::from_bits(0x0),
            f64::from_bits(0x0),
        ],
        [
            f64::from_bits(0x3f30002866000000),
            f64::from_bits(0xbd5329e31412b688),
            f64::from_bits(0x3a2279ddc3585863),
        ],
        [
            f64::from_bits(0x3f3fffed31000000),
            f64::from_bits(0x3d5546dcb40b0518),
            f64::from_bits(0xba26d048f41debec),
        ],
        [
            f64::from_bits(0x3f48000388000000),
            f64::from_bits(0x3d5fce6789678c88),
            f64::from_bits(0x3a24959dc2fea030),
        ],
        [
            f64::from_bits(0x3f50000668400000),
            f64::from_bits(0x3d5c5caec0aa6620),
            f64::from_bits(0xba289d58a2e0d287),
        ],
        [
            f64::from_bits(0x3f54000937000000),
            f64::from_bits(0xbd5df91bf5db20f8),
            f64::from_bits(0x3a2d4fc35005498a),
        ],
        [
            f64::from_bits(0x3f58000a2d800000),
            f64::from_bits(0x3d3e7464f400a700),
            f64::from_bits(0xba26365a147ce4a8),
        ],
        [
            f64::from_bits(0x3f5c00094a400000),
            f64::from_bits(0xbd5c55c526e65a68),
            f64::from_bits(0xba2a7ea2cf16d2ce),
        ],
        [
            f64::from_bits(0x3f60000345400000),
            f64::from_bits(0x3d48e815121136d0),
            f64::from_bits(0xba168f8f7a924fd6),
        ],
        [
            f64::from_bits(0x3f620000f6400000),
            f64::from_bits(0x3d57fb19a44da298),
            f64::from_bits(0xba2a05fc6ed4124d),
        ],
        [
            f64::from_bits(0x3f63fffdb7000000),
            f64::from_bits(0x3d5bfac3b1abe2e8),
            f64::from_bits(0xba09ab1f45a1efb1),
        ],
        [
            f64::from_bits(0x3f66000516a00000),
            f64::from_bits(0xbd402544d9faf830),
            f64::from_bits(0x3a2286c39adffa8c),
        ],
        [
            f64::from_bits(0x3f67fffff4200000),
            f64::from_bits(0xbd52878b893da790),
            f64::from_bits(0x3a2b55c5e09022b1),
        ],
        [
            f64::from_bits(0x3f6a00056f400000),
            f64::from_bits(0xbd59d7ccf4019608),
            f64::from_bits(0x3a2f4b05553fe340),
        ],
        [
            f64::from_bits(0x3f6bfffe65200000),
            f64::from_bits(0xbd5ee85f47c2b5a0),
            f64::from_bits(0x3a253fa913d4e9a6),
        ],
        [
            f64::from_bits(0x3f6e0001f7600000),
            f64::from_bits(0x3d42e594fb2af5e0),
            f64::from_bits(0xb9cafaec77ca8d86),
        ],
        [
            f64::from_bits(0x3f7000024a000000),
            f64::from_bits(0xbd5799935d030650),
            f64::from_bits(0xba0cacad0b61c964),
        ],
    ],
    [
        [
            f64::from_bits(0x0),
            f64::from_bits(0x0),
            f64::from_bits(0x0),
        ],
        [
            f64::from_bits(0x3ef0014690000000),
            f64::from_bits(0xbd4743d8ea40a4d0),
            f64::from_bits(0xb9e75086a21c12ab),
        ],
        [
            f64::from_bits(0x3f00014c18000000),
            f64::from_bits(0x3d48e7fa3cf59080),
            f64::from_bits(0x3a2a11bdcdb1ace3),
        ],
        [
            f64::from_bits(0x3f07ff17c8000000),
            f64::from_bits(0x3d5118dbbcce8470),
            f64::from_bits(0x3a009806ecd19760),
        ],
        [
            f64::from_bits(0x3f0fffcbb8000000),
            f64::from_bits(0xbd5eaa70b2051e38),
            f64::from_bits(0x39e0d287c5aa12b4),
        ],
        [
            f64::from_bits(0x3f14004294000000),
            f64::from_bits(0x3d5602b04d79b0c8),
            f64::from_bits(0x39f8d2c8ddf31648),
        ],
        [
            f64::from_bits(0x3f1800a218000000),
            f64::from_bits(0xbd50a23112178630),
            f64::from_bits(0xba2c4652435118f1),
        ],
        [
            f64::from_bits(0x3f1bff9304000000),
            f64::from_bits(0xbd4385dc475c16b0),
            f64::from_bits(0xb9fe7288f74bbbda),
        ],
        [
            f64::from_bits(0x3f1ffff810000000),
            f64::from_bits(0x3d4355edcff4e190),
            f64::from_bits(0x3a0f8daae6c1db11),
        ],
        [
            f64::from_bits(0x3f22002ff2000000),
            f64::from_bits(0x3d23fbb7cd366140),
            f64::from_bits(0x3a293c0e5b4dca92),
        ],
        [
            f64::from_bits(0x3f23ffac90000000),
            f64::from_bits(0xbd39bde416de25c0),
            f64::from_bits(0xba2f4df1d74776b3),
        ],
        [
            f64::from_bits(0x3f25ffe340000000),
            f64::from_bits(0xbd521735d0955dc0),
            f64::from_bits(0xba21b0e10765842f),
        ],
        [
            f64::from_bits(0x3f28001b52000000),
            f64::from_bits(0x3d40c0949e968810),
            f64::from_bits(0xba1d3771c0b220f3),
        ],
        [
            f64::from_bits(0x3f2a0054c8000000),
            f64::from_bits(0x3d426936197b1050),
            f64::from_bits(0x3a2e59caa334554a),
        ],
        [
            f64::from_bits(0x3f2bffd6f0000000),
            f64::from_bits(0x3d427097b0b27d40),
            f64::from_bits(0x3a278a37896a0d1f),
        ],
        [
            f64::from_bits(0x3f2e00132c000000),
            f64::from_bits(0x3d3df1e084120280),
            f64::from_bits(0x3a1c61f449a4b5d5),
        ],
        [
            f64::from_bits(0x3f30002866000000),
            f64::from_bits(0xbd5329e31412b688),
            f64::from_bits(0x3a2279ddc3585863),
        ],
    ],
];
const CY: [[f64; 2]; 3] = [
    [
        f64::from_bits(0x3ff71547652b82fe),
        f64::from_bits(0x3c7777d0ffda0d24),
    ],
    [
        f64::from_bits(0xbfe71547652b82fe),
        f64::from_bits(0xbc6777d0ffda0d24),
    ],
    [
        f64::from_bits(0x3fdec709dc3a03fd),
        f64::from_bits(0x3c7d27e96be541e5),
    ],
];
const CL: [f64; 3] = [
    f64::from_bits(0xbfd71547652b82fe),
    f64::from_bits(0x3fd2776c50f1ff14),
    f64::from_bits(0xbfcec709dc3eca5d),
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
    two(h, l)
}
fn normalized(x: f64) -> (u64, i32) {
    let mut b = x.to_bits();
    let ex = (b >> 52) as i32;
    let mut e = ex - 1023;
    if ex == 0 {
        let k = b.leading_zeros();
        e -= k as i32 - 12;
        b <<= k - 11;
    }
    (b & 0xfffffffffffff, e)
}
#[inline(never)]
fn refine(x: f64, a: f64) -> f64 {
    let (b, e) = normalized(x);
    let z = f64::from_bits(b | (0x3ff << 52));
    let ed = e as f64;
    let v = (a - ed + f64::from_bits(0x3ff0000800000000)).to_bits();
    let i = (v - (0x3ff << 52)) >> 36;
    let idx = [
        (i >> 12) as usize,
        ((i >> 8) & 15) as usize,
        ((i >> 4) & 15) as usize,
        (i & 15) as usize,
    ];
    let l = core::array::from_fn::<_, 3, _>(|k| {
        LL[0][idx[0]][k] + LL[1][idx[1]][k] + (LL[2][idx[2]][k] + LL[3][idx[3]][k])
    });
    let l0 = l[0] + ed;
    let t12 = T1[idx[0]] * T2[idx[1]];
    let t34 = T3[idx[2]] * T4[idx[3]];
    let th = t12 * t34;
    let tl = fma(t12, t34, -th);
    let dh = th * z;
    let dl = fma(th, z, -dh);
    let sh = tl * z;
    let sl = fma(tl, z, -sh);
    let (xh, xl) = two(dh - 1.0, dl);
    let (xh, xl) = add(xh, xl, sh, sl);
    let tail = xh * (CL[0] + xh * (CL[1] + xh * CL[2]));
    let mut sh = CY[2][0] + tail;
    let mut sl = ((CY[2][0] - sh) + tail) + CY[2][1];
    for c in CY[..2].iter().rev() {
        (sh, sl) = mul(xh, xl, sh, sl);
        let h = sh + c[0];
        let l = (c[0] - h) + sh;
        sh = h;
        sl += l + c[1];
    }
    (sh, sl) = mul(xh, xl, sh, sl);
    (sh, sl) = add(sh, sl, l[1], l[2]);
    let (v0, v2) = two(l0, sh);
    let (mut v1, v2) = two(v2, sl);
    if v1.to_bits() & 0xfffffffffffff == 0 {
        let d = if (v1.to_bits() ^ v2.to_bits()) >> 63 != 0 {
            -1i64
        } else {
            1
        };
        v1 = f64::from_bits(v1.to_bits().wrapping_add(d as u64));
    }
    v1 + v0
}
/// Base-two logarithm, rounded to nearest with ties to even.
pub fn log2(x: f64) -> f64 {
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    if x.is_nan() {
        return x + x;
    }
    if x < 0.0 {
        return f64::NAN;
    }
    if x == f64::INFINITY {
        return x;
    }
    let (b, e) = normalized(x);
    if b == 0 {
        return e as f64;
    }
    let i = (b >> 47) as usize;
    let d = (b & 0x7fffffffffff) as i64;
    let j = (b
        .wrapping_add((B[i][0] as u64) << 33)
        .wrapping_add((B[i][1] * (d >> 16)) as u64))
        >> 42;
    let i1 = (j >> 5) as usize;
    let i2 = (j & 31) as usize;
    let z = f64::from_bits(b | (0x3ff << 52));
    let r = R1[i1] * R2[i2];
    let o = r * z;
    let dxl = fma(r, z, -o);
    let dxh = o - f64::from_bits(0x3ff7154800000000);
    let dx = dxh + dxl;
    let dx2 = dx * dx;
    let f = dx2 * ((C[0] + dx * C[1]) + dx2 * (C[2] + dx * C[3]));
    let lt = (L1[i1][1] + L2[i2][1]) + e as f64;
    let lh = lt + dxh;
    let mut ll = (lt - lh) + dxh;
    ll += ((L1[i1][0] + L2[i2][0]) + dxl) + dxh * (-f64::from_bits(0x3e9ad47a2f472159));
    ll += f;
    let lb = lh + (ll - 2.64e-22);
    let ub = lh + (ll + 2.64e-22);
    if lb == ub { lb } else { refine(x, ub) }
}

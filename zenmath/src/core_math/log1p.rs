/* Correctly rounded log(1+x) for binary64 values.

Copyright (c) 2024-2026 Alexei Sibidanov <sibid@uvic.ca>.

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
const RT: [[f64; 16]; 4] = [
    [
        f64::from_bits(0x3ff6a09e68000000),
        f64::from_bits(0x3ff5ab07e0000000),
        f64::from_bits(0x3ff4bfdad8000000),
        f64::from_bits(0x3ff3dea650000000),
        f64::from_bits(0x3ff306fe08000000),
        f64::from_bits(0x3ff2387a70000000),
        f64::from_bits(0x3ff172b840000000),
        f64::from_bits(0x3ff0b55870000000),
        f64::from_bits(0x3ff0000000000000),
        f64::from_bits(0x3feea4afa0000000),
        f64::from_bits(0x3fed5818e0000000),
        f64::from_bits(0x3fec199be0000000),
        f64::from_bits(0x3feae89f98000000),
        f64::from_bits(0x3fe9c49180000000),
        f64::from_bits(0x3fe8ace540000000),
        f64::from_bits(0x3fe7a11470000000),
    ],
    [
        f64::from_bits(0x3ff059b0d0000000),
        f64::from_bits(0x3ff04e5f70000000),
        f64::from_bits(0x3ff04315e8000000),
        f64::from_bits(0x3ff037d430000000),
        f64::from_bits(0x3ff02c9a40000000),
        f64::from_bits(0x3ff0216818000000),
        f64::from_bits(0x3ff0163da8000000),
        f64::from_bits(0x3ff00b1af8000000),
        f64::from_bits(0x3ff0000000000000),
        f64::from_bits(0x3fefe9d968000000),
        f64::from_bits(0x3fefd3c228000000),
        f64::from_bits(0x3fefbdba38000000),
        f64::from_bits(0x3fefa7c180000000),
        f64::from_bits(0x3fef91d800000000),
        f64::from_bits(0x3fef7bfdb0000000),
        f64::from_bits(0x3fef663278000000),
    ],
    [
        f64::from_bits(0x3ff0058c88000000),
        f64::from_bits(0x3ff004dad8000000),
        f64::from_bits(0x3ff0042938000000),
        f64::from_bits(0x3ff0037798000000),
        f64::from_bits(0x3ff002c608000000),
        f64::from_bits(0x3ff0021478000000),
        f64::from_bits(0x3ff00162f0000000),
        f64::from_bits(0x3ff000b178000000),
        f64::from_bits(0x3ff0000000000000),
        f64::from_bits(0x3feffe9d20000000),
        f64::from_bits(0x3feffd3a58000000),
        f64::from_bits(0x3feffbd798000000),
        f64::from_bits(0x3feffa74e8000000),
        f64::from_bits(0x3feff91248000000),
        f64::from_bits(0x3feff7afb8000000),
        f64::from_bits(0x3feff64d38000000),
    ],
    [
        f64::from_bits(0x3ff00058b8000000),
        f64::from_bits(0x3ff0004da0000000),
        f64::from_bits(0x3ff0004288000000),
        f64::from_bits(0x3ff0003778000000),
        f64::from_bits(0x3ff0002c60000000),
        f64::from_bits(0x3ff0002148000000),
        f64::from_bits(0x3ff0001630000000),
        f64::from_bits(0x3ff0000b18000000),
        f64::from_bits(0x3ff0000000000000),
        f64::from_bits(0x3fefffe9d0000000),
        f64::from_bits(0x3fefffd3a0000000),
        f64::from_bits(0x3fefffbd78000000),
        f64::from_bits(0x3fefffa748000000),
        f64::from_bits(0x3fefff9118000000),
        f64::from_bits(0x3fefff7ae8000000),
        f64::from_bits(0x3fefff64c0000000),
    ],
];
const LN: [[[f64; 3]; 16]; 4] = [
    [
        [
            f64::from_bits(0x3a061fa45b636ea2),
            f64::from_bits(0xbd2497697cb31340),
            f64::from_bits(0xbfd62e43033a8000),
        ],
        [
            f64::from_bits(0xba242b671e5e6d5e),
            f64::from_bits(0x3d3fc50ee605bd80),
            f64::from_bits(0xbfd3687aa721c000),
        ],
        [
            f64::from_bits(0x3a00cc2ea968d329),
            f64::from_bits(0x3d4f48666c24eac0),
            f64::from_bits(0xbfd0a2b247d58000),
        ],
        [
            f64::from_bits(0xba2963904db0a34e),
            f64::from_bits(0xbd458495e50d36a0),
            f64::from_bits(0xbfcbb9d3d8080000),
        ],
        [
            f64::from_bits(0x3a0337e0d9f81652),
            f64::from_bits(0x3d2c89a0bede9780),
            f64::from_bits(0xbfc62e42f0378000),
        ],
        [
            f64::from_bits(0xba01bdf53871d50b),
            f64::from_bits(0x3d52d67e8a0c1d70),
            f64::from_bits(0xbfc0a2b24a100000),
        ],
        [
            f64::from_bits(0xb9d781dd6fc40530),
            f64::from_bits(0x3d507a22a9e81ed8),
            f64::from_bits(0xbfb62e43327c0000),
        ],
        [
            f64::from_bits(0xb9f84e4562b8f2f3),
            f64::from_bits(0xbd4868afe1489500),
            f64::from_bits(0xbfa62e435bae0000),
        ],
        [
            f64::from_bits(0x0),
            f64::from_bits(0x0),
            f64::from_bits(0x0),
        ],
        [
            f64::from_bits(0xb9f17e3ec05cde70),
            f64::from_bits(0x3d5174a196895690),
            f64::from_bits(0x3fa62e432b220000),
        ],
        [
            f64::from_bits(0xba08067ec2a27738),
            f64::from_bits(0x3d4111a4eadf3130),
            f64::from_bits(0x3fb62e42e4a80000),
        ],
        [
            f64::from_bits(0xba260bef06b14405),
            f64::from_bits(0x3d4dd4ec4e1d4210),
            f64::from_bits(0x3fc0a2b233f08000),
        ],
        [
            f64::from_bits(0xba1b18e160362c24),
            f64::from_bits(0x3d26bd65e8b0b700),
            f64::from_bits(0x3fc62e43056c0000),
        ],
        [
            f64::from_bits(0xba2c6ac3f1862a6b),
            f64::from_bits(0xbd30f5aad509ea80),
            f64::from_bits(0x3fcbb9d3cbd68000),
        ],
        [
            f64::from_bits(0xba2dead1a4581acf),
            f64::from_bits(0xbd4ac842de005640),
            f64::from_bits(0x3fd0a2b244da4000),
        ],
        [
            f64::from_bits(0xba296b1f2f60e3e4),
            f64::from_bits(0x3d59cec9a50db228),
            f64::from_bits(0x3fd3687aa9b78000),
        ],
    ],
    [
        [
            f64::from_bits(0x39fdbd93cdc08613),
            f64::from_bits(0x3d396f39c49fa8a0),
            f64::from_bits(0xbf962e423dd40000),
        ],
        [
            f64::from_bits(0xba13c008f058a015),
            f64::from_bits(0xbd4f26429af65b70),
            f64::from_bits(0xbf936879e5140000),
        ],
        [
            f64::from_bits(0x3a2f96a86136c1d1),
            f64::from_bits(0xbd5bfc5afb2bd378),
            f64::from_bits(0xbf90a2b2240c0000),
        ],
        [
            f64::from_bits(0x3a23917cea885be0),
            f64::from_bits(0xbd1421e0c2976380),
            f64::from_bits(0xbf8bb9d4b2880000),
        ],
        [
            f64::from_bits(0x3a2f0f1b5d989675),
            f64::from_bits(0x3d592a7506c90a18),
            f64::from_bits(0xbf862e43c1200000),
        ],
        [
            f64::from_bits(0xba2f7f8d6a190051),
            f64::from_bits(0xbd594925ae0e5240),
            f64::from_bits(0xbf80a2b41dd00000),
        ],
        [
            f64::from_bits(0x3a1ccb0919263869),
            f64::from_bits(0xbd50fb337ff2cf28),
            f64::from_bits(0xbf762e4106800000),
        ],
        [
            f64::from_bits(0x3a2c9a5efe4b7c6c),
            f64::from_bits(0xbd3966c27788d1c0),
            f64::from_bits(0xbf662e3e4cc00000),
        ],
        [
            f64::from_bits(0x0),
            f64::from_bits(0x0),
            f64::from_bits(0x0),
        ],
        [
            f64::from_bits(0xba040288ccae8f0f),
            f64::from_bits(0xbd5f3c5ff88d19c8),
            f64::from_bits(0x3f662e462b600000),
        ],
        [
            f64::from_bits(0xba1f7e788a871350),
            f64::from_bits(0xbd4a9610028771c0),
            f64::from_bits(0x3f762e44c9300000),
        ],
        [
            f64::from_bits(0x3a2574cc6d3f577d),
            f64::from_bits(0x3d52a3a1a65aa398),
            f64::from_bits(0x3f80a2b1e3300000),
        ],
        [
            f64::from_bits(0xb9c56bb79b254f33),
            f64::from_bits(0xbd34a995b6d9ddc0),
            f64::from_bits(0x3f862e4367c00000),
        ],
        [
            f64::from_bits(0x3a1c4c209ca6783d),
            f64::from_bits(0xbd4d7b98ef459110),
            f64::from_bits(0x3f8bb9d449a80000),
        ],
        [
            f64::from_bits(0x3a2e9731de7f0155),
            f64::from_bits(0xbd4de722390bbd60),
            f64::from_bits(0x3f90a2b1f1940000),
        ],
        [
            f64::from_bits(0x39ffadc62522444d),
            f64::from_bits(0xbd2fd95cb835e380),
            f64::from_bits(0x3f93687ad1140000),
        ],
    ],
    [
        [
            f64::from_bits(0xba01f72d2a6a460e),
            f64::from_bits(0x3d4a25045c37c330),
            f64::from_bits(0xbf562e4795000000),
        ],
        [
            f64::from_bits(0x3a22847e318fd3ff),
            f64::from_bits(0xbd5c633239e1cdc8),
            f64::from_bits(0xbf53686e5dc00000),
        ],
        [
            f64::from_bits(0x3a263f5750b9d826),
            f64::from_bits(0xbd5cb15590f1cd78),
            f64::from_bits(0xbf50a2b653800000),
        ],
        [
            f64::from_bits(0xba11382d6395c24c),
            f64::from_bits(0x3d47636a5400e2b0),
            f64::from_bits(0xbf4bb9bf13800000),
        ],
        [
            f64::from_bits(0x3a1be4491ec20322),
            f64::from_bits(0x3d5d0c5e2c9b6be8),
            f64::from_bits(0xbf462e53e5000000),
        ],
        [
            f64::from_bits(0xba2706f04cc2c9f6),
            f64::from_bits(0xbd5f3bc0ce9b9a08),
            f64::from_bits(0xbf40a2ab37000000),
        ],
        [
            f64::from_bits(0x3a2ab6f41df01d7f),
            f64::from_bits(0x3d3a7cbc9a97ba40),
            f64::from_bits(0xbf362e0a00000000),
        ],
        [
            f64::from_bits(0xb9b67e487663ca3e),
            f64::from_bits(0xbd4957976dc5f360),
            f64::from_bits(0xbf262e84fc000000),
        ],
        [
            f64::from_bits(0x0),
            f64::from_bits(0x0),
            f64::from_bits(0x0),
        ],
        [
            f64::from_bits(0x3a29681e48dde135),
            f64::from_bits(0xbd4868625640a690),
            f64::from_bits(0x3f262e7b00000000),
        ],
        [
            f64::from_bits(0x3a2a2948cd558655),
            f64::from_bits(0xbd52ee3d96b696a0),
            f64::from_bits(0x3f362e35f6000000),
        ],
        [
            f64::from_bits(0xb9fcfc26ccf6d0e4),
            f64::from_bits(0x3d153edbcf116500),
            f64::from_bits(0x3f40a2b4b2000000),
        ],
        [
            f64::from_bits(0x3a2f68d24b9e338d),
            f64::from_bits(0x3cc783e334613000),
            f64::from_bits(0x3f462e4be1000000),
        ],
        [
            f64::from_bits(0xba0f33369bf7dff1),
            f64::from_bits(0xbd560785f20acb20),
            f64::from_bits(0x3f4bb9e085000000),
        ],
        [
            f64::from_bits(0xba0685a35575eff1),
            f64::from_bits(0xbcf5a62ec6656800),
            f64::from_bits(0x3f50a2b94d400000),
        ],
        [
            f64::from_bits(0xba04c4d1abca79bf),
            f64::from_bits(0x3d17ded26dc81300),
            f64::from_bits(0x3f5368810f800000),
        ],
    ],
    [
        [
            f64::from_bits(0x3a140676dea39b19),
            f64::from_bits(0x3d525337681fa900),
            f64::from_bits(0xbf162dc284000000),
        ],
        [
            f64::from_bits(0xba1ec76c964ac8d6),
            f64::from_bits(0xbd5504032e2ed388),
            f64::from_bits(0xbf1367d0ec000000),
        ],
        [
            f64::from_bits(0x3a08ecaeb6d9523c),
            f64::from_bits(0x3d3c28fde83047e0),
            f64::from_bits(0xbf10a1dd6c000000),
        ],
        [
            f64::from_bits(0xba2e4069214576be),
            f64::from_bits(0x3d558f2757976ef8),
            f64::from_bits(0xbf0bbbcff0000000),
        ],
        [
            f64::from_bits(0x3a286146d1f9b91f),
            f64::from_bits(0xbd5d4717ca0a3230),
            f64::from_bits(0xbf062fe138000000),
        ],
        [
            f64::from_bits(0x3a21eb44b07102c5),
            f64::from_bits(0xbd486ffcda252780),
            f64::from_bits(0xbf00a3eeb0000000),
        ],
        [
            f64::from_bits(0xba295bbb5fd5a466),
            f64::from_bits(0x3d41ce399729e7b0),
            f64::from_bits(0xbef62ff0a0000000),
        ],
        [
            f64::from_bits(0xba279059023f8767),
            f64::from_bits(0xbd5dbf1c6a400408),
            f64::from_bits(0xbee62ff840000000),
        ],
        [
            f64::from_bits(0x0),
            f64::from_bits(0x0),
            f64::from_bits(0x0),
        ],
        [
            f64::from_bits(0xba2b4cb08d6fd48f),
            f64::from_bits(0xbd5db0e38e5aaa98),
            f64::from_bits(0x3ee63007c0000000),
        ],
        [
            f64::from_bits(0xba20e6a88bfc3838),
            f64::from_bits(0x3d42b1c755804390),
            f64::from_bits(0x3ef6300f60000000),
        ],
        [
            f64::from_bits(0x3a18074feacfe49d),
            f64::from_bits(0x3d5401bb919f14e0),
            f64::from_bits(0x3f00a21148000000),
        ],
        [
            f64::from_bits(0xba2f56f5f168db15),
            f64::from_bits(0xbd385d6f6487ce20),
            f64::from_bits(0x3f062e1ec0000000),
        ],
        [
            f64::from_bits(0x3a29eae7e05a0143),
            f64::from_bits(0xbd5af5d58a7c9218),
            f64::from_bits(0x3f0bba3010000000),
        ],
        [
            f64::from_bits(0xba1a859095999ae3),
            f64::from_bits(0x3d5590faa0883bd8),
            f64::from_bits(0x3f10a32298000000),
        ],
        [
            f64::from_bits(0x3a24f787e495e5e0),
            f64::from_bits(0xbd1fbb791220a180),
            f64::from_bits(0x3f13682f14000000),
        ],
    ],
];
const RF: [f64; 64] = [
    f64::from_bits(0x3ff6816818000000),
    f64::from_bits(0x3ff642c858000000),
    f64::from_bits(0x3ff6058160000000),
    f64::from_bits(0x3ff5c98828000000),
    f64::from_bits(0x3ff58ed230000000),
    f64::from_bits(0x3ff5555558000000),
    f64::from_bits(0x3ff51d07e8000000),
    f64::from_bits(0x3ff4e5e0a8000000),
    f64::from_bits(0x3ff4afd6a0000000),
    f64::from_bits(0x3ff47ae148000000),
    f64::from_bits(0x3ff446f868000000),
    f64::from_bits(0x3ff4141418000000),
    f64::from_bits(0x3ff3e22cc0000000),
    f64::from_bits(0x3ff3b13b10000000),
    f64::from_bits(0x3ff3813810000000),
    f64::from_bits(0x3ff3521cf8000000),
    f64::from_bits(0x3ff323e348000000),
    f64::from_bits(0x3ff2f684c0000000),
    f64::from_bits(0x3ff2c9fb50000000),
    f64::from_bits(0x3ff29e4128000000),
    f64::from_bits(0x3ff27350b8000000),
    f64::from_bits(0x3ff2492490000000),
    f64::from_bits(0x3ff21fb780000000),
    f64::from_bits(0x3ff1f70480000000),
    f64::from_bits(0x3ff1cf06b0000000),
    f64::from_bits(0x3ff1a7b960000000),
    f64::from_bits(0x3ff1811810000000),
    f64::from_bits(0x3ff15b1e60000000),
    f64::from_bits(0x3ff135c810000000),
    f64::from_bits(0x3ff1111110000000),
    f64::from_bits(0x3ff0ecf568000000),
    f64::from_bits(0x3ff0c97150000000),
    f64::from_bits(0x3ff0a68108000000),
    f64::from_bits(0x3ff0842108000000),
    f64::from_bits(0x3ff0624dd0000000),
    f64::from_bits(0x3ff0410410000000),
    f64::from_bits(0x3ff0204080000000),
    f64::from_bits(0x3ff0000000000000),
    f64::from_bits(0x3fef81f820000000),
    f64::from_bits(0x3fef07c1f0000000),
    f64::from_bits(0x3fee9131a8000000),
    f64::from_bits(0x3fee1e1e20000000),
    f64::from_bits(0x3fedae6078000000),
    f64::from_bits(0x3fed41d420000000),
    f64::from_bits(0x3fecd85688000000),
    f64::from_bits(0x3fec71c720000000),
    f64::from_bits(0x3fec0e0700000000),
    f64::from_bits(0x3febacf918000000),
    f64::from_bits(0x3feb4e81b8000000),
    f64::from_bits(0x3feaf286c0000000),
    f64::from_bits(0x3fea98ef60000000),
    f64::from_bits(0x3fea41a418000000),
    f64::from_bits(0x3fe9ec8e98000000),
    f64::from_bits(0x3fe9999998000000),
    f64::from_bits(0x3fe948b100000000),
    f64::from_bits(0x3fe8f9c190000000),
    f64::from_bits(0x3fe8acb910000000),
    f64::from_bits(0x3fe8618618000000),
    f64::from_bits(0x3fe8181818000000),
    f64::from_bits(0x3fe7d05f40000000),
    f64::from_bits(0x3fe78a4c80000000),
    f64::from_bits(0x3fe745d178000000),
    f64::from_bits(0x3fe702e060000000),
    f64::from_bits(0x3fe6c16c18000000),
];
const LF: [[f64; 2]; 64] = [
    [
        f64::from_bits(0xbd5f2f8281bade6a),
        f64::from_bits(0xbfd5d5bde3994000),
    ],
    [
        f64::from_bits(0x3d5c2843fdd367a4),
        f64::from_bits(0xbfd522ae0438c000),
    ],
    [
        f64::from_bits(0xbd306c10c34c14b0),
        f64::from_bits(0xbfd4718dc171c000),
    ],
    [
        f64::from_bits(0x3d4cfa4e853f5890),
        f64::from_bits(0xbfd3c2526cb34000),
    ],
    [
        f64::from_bits(0xbd5ce3ac179bd856),
        f64::from_bits(0xbfd314f1e0534000),
    ],
    [
        f64::from_bits(0xbd5b91f82deb8122),
        f64::from_bits(0xbfd269621934c000),
    ],
    [
        f64::from_bits(0x3d546bbb83d7163e),
        f64::from_bits(0xbfd1bf995a9a8000),
    ],
    [
        f64::from_bits(0x3d5b842e5a74bdb0),
        f64::from_bits(0xbfd1178e84a80000),
    ],
    [
        f64::from_bits(0xbd5862715e5bb534),
        f64::from_bits(0xbfd071385f4d4000),
    ],
    [
        f64::from_bits(0xbd59bcbcbea0cdf8),
        f64::from_bits(0xbfcf991c6eb38000),
    ],
    [
        f64::from_bits(0xbd401101cb605958),
        f64::from_bits(0xbfce530f10670000),
    ],
    [
        f64::from_bits(0x3d50c38c81ad8f06),
        f64::from_bits(0xbfcd10380b658000),
    ],
    [
        f64::from_bits(0x3d53aa40992a6d82),
        f64::from_bits(0xbfcbd0874c3c0000),
    ],
    [
        f64::from_bits(0x3d530f68780ae82e),
        f64::from_bits(0xbfca93ed248b0000),
    ],
    [
        f64::from_bits(0xbd07d116989d0980),
        f64::from_bits(0xbfc95a5ac5f70000),
    ],
    [
        f64::from_bits(0xbd51e0012ba619ca),
        f64::from_bits(0xbfc823c150518000),
    ],
    [
        f64::from_bits(0x3d454535d5671858),
        f64::from_bits(0xbfc6f0127cf58000),
    ],
    [
        f64::from_bits(0xbd5ed87db3498128),
        f64::from_bits(0xbfc5bf407b540000),
    ],
    [
        f64::from_bits(0xbd5aafde9c9fc39a),
        f64::from_bits(0xbfc4913d94338000),
    ],
    [
        f64::from_bits(0xbd4015868c234000),
        f64::from_bits(0xbfc365fca3158000),
    ],
    [
        f64::from_bits(0x3d5eff33f502c226),
        f64::from_bits(0xbfc23d7126ca0000),
    ],
    [
        f64::from_bits(0x3d4b8521e874d358),
        f64::from_bits(0xbfc1178e72280000),
    ],
    [
        f64::from_bits(0x3d454d75afe84568),
        f64::from_bits(0xbfbfe89129dc0000),
    ],
    [
        f64::from_bits(0xbd51a813f3fa7c1e),
        f64::from_bits(0xbfbda72783840000),
    ],
    [
        f64::from_bits(0xbd56c6676f40963e),
        f64::from_bits(0xbfbb6ac8afad0000),
    ],
    [
        f64::from_bits(0xbd52620b7957a7a6),
        f64::from_bits(0xbfb9335e4d590000),
    ],
    [
        f64::from_bits(0x3d4f8ffee5598f38),
        f64::from_bits(0xbfb700d2f4eb0000),
    ],
    [
        f64::from_bits(0xbd5fab0f5bf42ca2),
        f64::from_bits(0xbfb4d31165200000),
    ],
    [
        f64::from_bits(0xbd37a3e970b1c3a8),
        f64::from_bits(0xbfb2aa0492470000),
    ],
    [
        f64::from_bits(0xbd4d030435fecb50),
        f64::from_bits(0xbfb08598a59e0000),
    ],
    [
        f64::from_bits(0x3d435084a4fb8ab8),
        f64::from_bits(0xbfaccb7357de0000),
    ],
    [
        f64::from_bits(0x3d432f36d60b44c4),
        f64::from_bits(0xbfa894aa1ca00000),
    ],
    [
        f64::from_bits(0x3d2c1bcce5be8110),
        f64::from_bits(0xbfa466ae8a2e0000),
    ],
    [
        f64::from_bits(0x3d5777740b18714a),
        f64::from_bits(0xbfa0415d81e80000),
    ],
    [
        f64::from_bits(0xbd4955c057693d94),
        f64::from_bits(0xbf98492470c80000),
    ],
    [
        f64::from_bits(0x3d44f71addb8be00),
        f64::from_bits(0xbf90205648940000),
    ],
    [
        f64::from_bits(0xbd3bcda4e198afb0),
        f64::from_bits(0xbf801014f5880000),
    ],
    [f64::from_bits(0x3bcc000000000000), f64::from_bits(0x0)],
    [
        f64::from_bits(0xbd5fe0df75092c5e),
        f64::from_bits(0x3f8fc0a891000000),
    ],
    [
        f64::from_bits(0x3d298036ec7e0a10),
        f64::from_bits(0x3f9f829b1e780000),
    ],
    [
        f64::from_bits(0x3d5ba010f49e5ff0),
        f64::from_bits(0x3fa7745938320000),
    ],
    [
        f64::from_bits(0xbd53ab13c266d328),
        f64::from_bits(0x3faf0a30a0120000),
    ],
    [
        f64::from_bits(0xbd471798573e45d4),
        f64::from_bits(0x3fb341d78b1c0000),
    ],
    [
        f64::from_bits(0x3d5ad32f072669fc),
        f64::from_bits(0x3fb6f0d272e50000),
    ],
    [
        f64::from_bits(0xbd454e391e16ea38),
        f64::from_bits(0x3fba926d434b0000),
    ],
    [
        f64::from_bits(0xbd2a302bbaf05590),
        f64::from_bits(0x3fbe27074e2b0000),
    ],
    [
        f64::from_bits(0x3d3cb4cd66e31f30),
        f64::from_bits(0x3fc0d77e8cd08000),
    ],
    [
        f64::from_bits(0xbd35b7a5bc474128),
        f64::from_bits(0x3fc29552e9200000),
    ],
    [
        f64::from_bits(0xbd17062e8135f740),
        f64::from_bits(0x3fc44d2b5e4b8000),
    ],
    [
        f64::from_bits(0x3d43d4c88fe1f4b0),
        f64::from_bits(0x3fc5ff3060a78000),
    ],
    [
        f64::from_bits(0xbd537b70004a6946),
        f64::from_bits(0x3fc7ab8904110000),
    ],
    [
        f64::from_bits(0xbd54a5885167c1ec),
        f64::from_bits(0x3fc9525aa7f48000),
    ],
    [
        f64::from_bits(0x3d5ff9d5953004ac),
        f64::from_bits(0x3fcaf3c940008000),
    ],
    [
        f64::from_bits(0x3d4a21ec41d8219c),
        f64::from_bits(0x3fcc8ff7cf9a8000),
    ],
    [
        f64::from_bits(0xbd3a322bf2f02ae8),
        f64::from_bits(0x3fce27075e2b0000),
    ],
    [
        f64::from_bits(0x3d5f1548b8a33616),
        f64::from_bits(0x3fcfb9186b5e0000),
    ],
    [
        f64::from_bits(0x3d50e36401f7a006),
        f64::from_bits(0x3fd0a324e0f38000),
    ],
    [
        f64::from_bits(0xbd59f1fa55382a8a),
        f64::from_bits(0x3fd1675cacabc000),
    ],
    [
        f64::from_bits(0xbd3a69763deb0960),
        f64::from_bits(0x3fd22941fc0f8000),
    ],
    [
        f64::from_bits(0x3d5d30bc3ac91bda),
        f64::from_bits(0x3fd2e8e2bee10000),
    ],
    [
        f64::from_bits(0x3d37a79cf4d73b28),
        f64::from_bits(0x3fd3a64c59694000),
    ],
    [
        f64::from_bits(0x3d5ec345197b22de),
        f64::from_bits(0x3fd4618bb81c4000),
    ],
    [
        f64::from_bits(0xbd3f4810a30aeba8),
        f64::from_bits(0x3fd51aad7c2e0000),
    ],
    [
        f64::from_bits(0x3d4394d2371c1d1c),
        f64::from_bits(0x3fd5d1bdbbd80000),
    ],
];
const CZ: [[f64; 2]; 5] = [
    [
        f64::from_bits(0x3fd5555555555555),
        f64::from_bits(0x3c75555555555556),
    ],
    [
        f64::from_bits(0xbfd0000000000000),
        f64::from_bits(0x3a925558eff3c1ef),
    ],
    [
        f64::from_bits(0x3fc999999999999a),
        f64::from_bits(0xbc6999999a91d6ca),
    ],
    [
        f64::from_bits(0xbfc5555555555555),
        f64::from_bits(0xbc6588aab185593a),
    ],
    [
        f64::from_bits(0x3fc2492492492492),
        f64::from_bits(0x3c628c490abc528f),
    ],
];
const CZL: [f64; 4] = [
    f64::from_bits(0xbfbfffffffffc555),
    f64::from_bits(0x3fbc71c71c7185aa),
    f64::from_bits(0xbfb9999d44449a31),
    f64::from_bits(0x3fb745d51f1817c7),
];
const CY: [[f64; 2]; 5] = [
    [f64::from_bits(0x3ff0000000000000), f64::from_bits(0x0)],
    [f64::from_bits(0xbfe0000000000000), f64::from_bits(0x0)],
    [
        f64::from_bits(0x3fd5555555555555),
        f64::from_bits(0x3c75555555555555),
    ],
    [
        f64::from_bits(0xbfd0000000000000),
        f64::from_bits(0xb9480007ed9858b7),
    ],
    [
        f64::from_bits(0x3fc999999999999a),
        f64::from_bits(0xbc69999999999991),
    ],
];
const CL: [f64; 4] = [
    f64::from_bits(0xbfc5555555555555),
    f64::from_bits(0x3fc2492492492492),
    f64::from_bits(0xbfc0000000073334),
    f64::from_bits(0x3fbc71c71c802f68),
];
const C0: [f64; 2] = [
    f64::from_bits(0xbfe00000000001d1),
    f64::from_bits(0x3fd55555555558f7),
];
const C1: [f64; 4] = [
    f64::from_bits(0xbfdffffffffffffd),
    f64::from_bits(0x3fd5555555555551),
    f64::from_bits(0xbfd000000d5555e1),
    f64::from_bits(0x3fc99999b442f73f),
];
const C2: [f64; 12] = [
    f64::from_bits(0x3fd5555555555555),
    f64::from_bits(0xbfd0000000000000),
    f64::from_bits(0x3fc9999999999b41),
    f64::from_bits(0xbfc555555555583b),
    f64::from_bits(0x3fc24924923f39e0),
    f64::from_bits(0xbfbfffffffe42e43),
    f64::from_bits(0x3fbc71c75511d70b),
    f64::from_bits(0xbfb99999de10510f),
    f64::from_bits(0x3fb7457e81b175f6),
    f64::from_bits(0xbfb554fb43e54e0f),
    f64::from_bits(0x3fb3ed68744f3d18),
    f64::from_bits(0xbfb28558ad5a7ac4),
];
const C3: [f64; 6] = [
    f64::from_bits(0xbfe000000000003d),
    f64::from_bits(0x3fd5555555554cf5),
    f64::from_bits(0xbfcffffffeca2939),
    f64::from_bits(0x3fc99999a3661724),
    f64::from_bits(0xbfc555d345bfe6fd),
    f64::from_bits(0x3fc247b887a6e5ed),
];

fn two(x: f64, y: f64) -> (f64, f64) {
    let s = x + y;
    (s, y - (s - x))
}
fn twosum(x: f64, y: f64) -> (f64, f64) {
    let s = x + y;
    let d = s - x;
    (s, (y - d) + (x + (d - s)))
}
fn fastsum(xh: f64, xl: f64, yh: f64, yl: f64) -> (f64, f64) {
    let (h, l) = two(xh, yh);
    (h, (xl + yl) + l)
}
fn sum(xh: f64, xl: f64, yh: f64, yl: f64) -> (f64, f64) {
    let (h, l) = twosum(xh, yh);
    (h, (xl + yl) + l)
}
fn mul(xh: f64, xl: f64, ch: f64, cl: f64) -> (f64, f64) {
    let h = ch * xh;
    (h, (cl * xh + ch * xl) + fma(ch, xh, -h))
}
fn muld(x: f64, ch: f64, cl: f64) -> (f64, f64) {
    let h = ch * x;
    (h, cl * x + fma(ch, x, -h))
}
fn poly(xh: f64, xl: f64, c: &[[f64; 2]], tail: f64, single: bool) -> (f64, f64) {
    let last = c[c.len() - 1];
    let (mut h, l) = two(last[0], tail);
    let mut l = last[1] + l;
    for c in c[..c.len() - 1].iter().rev() {
        (h, l) = if single {
            muld(xh, h, l)
        } else {
            mul(xh, xl, h, l)
        };
        (h, l) = fastsum(c[0], c[1], h, l);
    }
    (h, l)
}
fn scale(x: f64, e: i64) -> f64 {
    f64::from_bits(x.to_bits().wrapping_sub((e as u64) << 52))
}
#[inline(never)]
fn refine(x: f64, a: f64) -> f64 {
    let bits = x.to_bits();
    let ax = bits << 1;
    let (ln22, ln21, ln20) = if ax < 0x7ea0000000000000 {
        if ax < 0x7940000000000000 {
            return if ax == 0 {
                x
            } else {
                fma(x.abs(), -f64::from_bits(0x3c90000000000000), x)
            };
        }
        let x2h = x * x;
        let x2l = fma(x, x, -x2h);
        let (x3h, x3l) = muld(x, x2h, x2l);
        let tail = x * ((CZL[0] + x * CZL[1]) + x2h * (CZL[2] + x * CZL[3]));
        let (sh, sl) = poly(x, 0.0, &CZ, tail, true);
        let (sh, sl) = mul(sh, sl, x3h, x3l);
        let (h, l) = fastsum(-0.5 * x2h, -0.5 * x2l, sh, sl);
        (x, h, l)
    } else {
        let j = crate::scalar::roundeven(a * f64::from_bits(0x40f71547652b82fe)) as i64;
        let i = j + 34952;
        let je = i >> 16;
        let idx = [
            ((i >> 12) & 15) as usize,
            ((i >> 8) & 15) as usize,
            ((i >> 4) & 15) as usize,
            (i & 15) as usize,
        ];
        let l = core::array::from_fn::<_, 3, _>(|k| {
            (LN[0][idx[0]][k] + LN[1][idx[1]][k]) + (LN[2][idx[2]][k] + LN[3][idx[3]][k])
        });
        let (t, dt) = if (bits as i64) < 0x4690000000000000 && bits < 0xbfe0000000000000 {
            twosum(1.0, x)
        } else {
            (
                if (bits as i64) >= 0x4690000000000000 {
                    x
                } else {
                    1.0 + x
                },
                0.0,
            )
        };
        let t = scale(t, je);
        let t12 = RT[0][idx[0]] * RT[1][idx[1]];
        let t34 = RT[2][idx[2]] * RT[3][idx[3]];
        let th = t12 * t34;
        let tl = fma(t12, t34, -th);
        let dh = th * t;
        let dl = fma(th, t, -dh);
        let sh = tl * t;
        let sl = fma(tl, t, -sh);
        let (xh, xl) = two(dh - 1.0, dl);
        let (mut xh, mut xl) = fastsum(xh, xl, sh, sl);
        if dt != 0.0 {
            let dt = scale(dt, je);
            let h = th * dt;
            let l = fma(th, dt, -h) + tl * dt;
            (xh, xl) = fastsum(xh, xl, h, l);
        }
        let tail = xh * ((CL[0] + xh * CL[1]) + (xh * xh) * (CL[2] + xh * CL[3]));
        let (sh, sl) = poly(xh, xl, &CY, tail, false);
        let (sh, sl) = mul(xh, xl, sh, sl);
        let (h, m, l) = if je != 0 {
            let e = je as f64;
            let h = f64::from_bits(0x3fe62e42fefa4000) * e + l[2];
            let m = -f64::from_bits(0x3d48432a1b0e2800) * e;
            let lo = f64::from_bits(0x3a8cc01f97b57a08) * e;
            let (m, lo) = fastsum(m, lo, l[1], l[0]);
            (h, m, lo)
        } else {
            (l[2], l[1], l[0])
        };
        let (m, l) = sum(m, l, sh, sl);
        (h, m, l)
    };
    let (h, m) = two(ln22, ln21);
    let (mut m, l) = two(m, ln20);
    if m.to_bits() & 0xfffffffffffff == 0 {
        let b = m.to_bits();
        m = f64::from_bits(if (l.to_bits() ^ b) >> 63 != 0 {
            b.wrapping_sub(1)
        } else {
            b.wrapping_add(1)
        });
    }
    h + m
}
/// Natural logarithm of one plus x, avoiding cancellation near zero.
pub fn log1p(x: f64) -> f64 {
    let bits = x.to_bits();
    let ax = bits << 1;
    let (ln1, ln0, eps) = if ax < 0x7f60000000000000 {
        if ax < 0x7940000000000000 {
            return if ax == 0 {
                x
            } else {
                fma(x.abs(), -f64::from_bits(0x3c90000000000000), x)
            };
        }
        let x2 = x * x;
        if ax < 0x7e60000000000000 {
            let lo = if ax < 0x7d43360000000000 {
                x2 * (C0[0] + x * C0[1])
            } else {
                x2 * ((C1[0] + x * C1[1]) + x2 * (C1[2] + x * C1[3]))
            };
            (x, lo, f64::from_bits(0x3bfa000000000000) * x)
        } else {
            let x3 = x2 * x;
            let x4 = x2 * x2;
            let hx = -0.5 * x;
            let h = fma(hx, x, x);
            let lo = fma(hx, x, x - h);
            let f = ((C2[0] + x * C2[1]) + x2 * (C2[2] + x * C2[3]))
                + x4 * (((C2[4] + x * C2[5]) + x2 * (C2[6] + x * C2[7]))
                    + x4 * ((C2[8] + x * C2[9]) + x2 * (C2[10] + x * C2[11])));
            (h, lo + x3 * f, x3 * f64::from_bits(0x3cbb600000000000))
        }
    } else {
        let (mut t, mut dt) = if (bits as i64) < 0x4340000000000000 && bits < 0xbff0000000000000 {
            two(1.0, x)
        } else if bits < 0x4690000000000000 {
            (x, 1.0)
        } else if bits < 0x7ff0000000000000 {
            (x, 0.0)
        } else {
            return if x.is_nan() {
                x + x
            } else if x == f64::INFINITY {
                x
            } else if x == -1.0 {
                f64::NEG_INFINITY
            } else {
                f64::NAN
            };
        };
        let j = (t.to_bits() as i64) - 0x3fe6a00000000000;
        let j1 = ((j >> 46) & 63) as usize;
        let je = j >> 52;
        let rs = if je < 1022 {
            scale(RF[j1], je)
        } else {
            let sc = [0.5, 0.25, 0.125][(je - 1022) as usize];
            t *= sc;
            dt *= sc;
            scale(RF[j1], 1021)
        };
        let dh = rs * t;
        let dl = fma(rs, t, -dh) + rs * dt;
        let (xh, mut xl) = two(dh - 1.0, dl);
        let x2 = xh * xh;
        xl += x2 * ((C3[0] + xh * C3[1]) + x2 * ((C3[2] + xh * C3[3]) + x2 * (C3[4] + xh * C3[5])));
        let e = je as f64;
        let l1 = f64::from_bits(0x3fe62e42fefa4000) * e;
        let l0 = -f64::from_bits(0x3d48432a1b0e2634) * e;
        let (h, l) = fastsum(LF[j1][1] + l1, LF[j1][0] + l0, xh, xl);
        (h, l, f64::from_bits(0x3bea000000000000))
    };
    let lb = ln1 + (ln0 - eps);
    let ub = ln1 + (ln0 + eps);
    if lb == ub { lb } else { refine(x, lb) }
}

/* Correctly-rounded atan2 function for two binary64 values.

Copyright (c) 2024-2025 Paul Zimmermann and Alexei Sibidanov

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
use super::{
    atan2_tint::{PI, PI2, Tint},
    fma,
};
const P: [Tint; 30] = [
    Tint {
        h: 0x82703f8b53112eca,
        m: 0x71ce6279d467aa45,
        l: 0x19c9b28466b6fc0,
        ex: -10,
        sgn: 0,
    },
    Tint {
        h: 0xb3cf74b427d53e03,
        m: 0xee590b839e751890,
        l: 0xf2570b12bcb2381a,
        ex: -8,
        sgn: 0,
    },
    Tint {
        h: 0xb08498b17b88c39a,
        m: 0x11e37821d784d8fa,
        l: 0x2795b14b84b9d78c,
        ex: -6,
        sgn: 0,
    },
    Tint {
        h: 0xfd4baa7bb52b0d83,
        m: 0xb25425beb0a04dc3,
        l: 0x92008d9fb34b68ff,
        ex: -5,
        sgn: 0,
    },
    Tint {
        h: 0x96541948b85fb386,
        m: 0xe0b815ace425a859,
        l: 0xbeca17c69bb4f7a0,
        ex: -3,
        sgn: 0,
    },
    Tint {
        h: 0x96aaa5d4baffc590,
        m: 0xf23db783b6e4ce55,
        l: 0x2c54e054d3b69f7e,
        ex: -2,
        sgn: 0,
    },
    Tint {
        h: 0x83cd41b5fa862b11,
        m: 0xad8f9d6e3eb9d833,
        l: 0x4ef470d801fa03ca,
        ex: -1,
        sgn: 0,
    },
    Tint {
        h: 0xcbbc7752197cc106,
        m: 0xbc66cfb3edefa6e3,
        l: 0x568b2f9e4feb8910,
        ex: -1,
        sgn: 0,
    },
    Tint {
        h: 0x8cf8f7c585767042,
        m: 0xcd92a5b9ac13d8c8,
        l: 0x43bd4a023dbe04d,
        ex: 0,
        sgn: 0,
    },
    Tint {
        h: 0xafdc36d6b36139bf,
        m: 0x11df63474e3859d4,
        l: 0x42b30b2c089a2b6c,
        ex: 0,
        sgn: 0,
    },
    Tint {
        h: 0xc6f8fb1463822519,
        m: 0x2f9f04cf55c10377,
        l: 0x1c44675c6d483606,
        ex: 0,
        sgn: 0,
    },
    Tint {
        h: 0xcce79f10f754e315,
        m: 0x91ce47edcbc8a4ee,
        l: 0xb2a6f721ac656b42,
        ex: 0,
        sgn: 0,
    },
    Tint {
        h: 0xc090384dbd77132e,
        m: 0x1dfb4d9a1d595de7,
        l: 0xe506ca899a394f70,
        ex: 0,
        sgn: 0,
    },
    Tint {
        h: 0xa556ac9eee25e867,
        m: 0xd84dafa25141d203,
        l: 0xf1c7c2a5156b3ec3,
        ex: 0,
        sgn: 0,
    },
    Tint {
        h: 0x81bd2677ddfcbc38,
        m: 0x1df33f27dbb60939,
        l: 0xcb4c2c73286a6c29,
        ex: 0,
        sgn: 0,
    },
    Tint {
        h: 0xb9ed12169d72e8d9,
        m: 0xc81d95dcef13c227,
        l: 0x974b95796b93c332,
        ex: -1,
        sgn: 0,
    },
    Tint {
        h: 0xf2df6da875b666fa,
        m: 0xc29dd3a69c274c65,
        l: 0x83b9260976c8f0c5,
        ex: -2,
        sgn: 0,
    },
    Tint {
        h: 0x902d34d4db910ed8,
        m: 0x6f3ecddfd6af0a9f,
        l: 0x8e950d7e70a8e9b2,
        ex: -2,
        sgn: 0,
    },
    Tint {
        h: 0x9aefa41961ce1207,
        m: 0xacd0317666753751,
        l: 0x3eaf42a53fcc6534,
        ex: -3,
        sgn: 0,
    },
    Tint {
        h: 0x95db9977c5e6e263,
        m: 0x85367a63fab8a720,
        l: 0xfc2b63bd115998d8,
        ex: -4,
        sgn: 0,
    },
    Tint {
        h: 0x817cb6d8236d5f4d,
        m: 0x70c7815f28909d9b,
        l: 0xc54e97155d12158,
        ex: -5,
        sgn: 0,
    },
    Tint {
        h: 0xc5eec4228c352098,
        m: 0x789e56be1dee9b67,
        l: 0xba25bda81292ccc1,
        ex: -7,
        sgn: 0,
    },
    Tint {
        h: 0x840c38ccfcb5dc64,
        m: 0xb55a44095bf7f312,
        l: 0x6668b8792d4ecd08,
        ex: -8,
        sgn: 0,
    },
    Tint {
        h: 0x97074af7b286b19e,
        m: 0xc20aac21f3130f1d,
        l: 0x369e9fdbdd48528a,
        ex: -10,
        sgn: 0,
    },
    Tint {
        h: 0x9054011c32d4b7a5,
        m: 0xc68d3c5d525f4af4,
        l: 0xeeb3c98615dee188,
        ex: -12,
        sgn: 0,
    },
    Tint {
        h: 0xdddb4a2eaab2bb15,
        m: 0x753761a8f0e9aa8d,
        l: 0x3f20620cd3e075bb,
        ex: -15,
        sgn: 0,
    },
    Tint {
        h: 0x80ef9a1b3e59abe5,
        m: 0x18e6afe0651fdf49,
        l: 0xd1c0329e6a702d9d,
        ex: -17,
        sgn: 0,
    },
    Tint {
        h: 0xca16dd49cd30854c,
        m: 0xbad8ca379401027c,
        l: 0x1fc7f47985486f16,
        ex: -21,
        sgn: 0,
    },
    Tint {
        h: 0xa0ba01b8742ba801,
        m: 0xe6dc14a9516d7f04,
        l: 0x1f5303d878905d2f,
        ex: -25,
        sgn: 0,
    },
    Tint {
        h: 0x9c0440896ce6515c,
        m: 0xcf3470fa7f10ac3,
        l: 0xa9dc80682c111a9d,
        ex: -59,
        sgn: 1,
    },
];
const Q: [Tint; 30] = [
    Tint {
        h: 0x82703f8b53112eca,
        m: 0x71ce6279d467aa45,
        l: 0x19c9b28466b6fc0,
        ex: -10,
        sgn: 0,
    },
    Tint {
        h: 0xb3cf74b427d53e03,
        m: 0xee590b839e751890,
        l: 0xf2570b12bcb238cf,
        ex: -8,
        sgn: 0,
    },
    Tint {
        h: 0xb33c44af0d43c9e8,
        m: 0xf4426f84614701db,
        l: 0x979e49dc5b858442,
        ex: -6,
        sgn: 0,
    },
    Tint {
        h: 0x8264d256f0bf4d61,
        m: 0xee21431cc0f29764,
        l: 0xce0cc1b593ac9952,
        ex: -4,
        sgn: 0,
    },
    Tint {
        h: 0x9d97c4947687b2ee,
        m: 0xef4087c49dde5229,
        l: 0x742bc6e02bfb6df8,
        ex: -3,
        sgn: 0,
    },
    Tint {
        h: 0xa0f888dc7aab949e,
        m: 0xf623f19ac413db23,
        l: 0x854744a164f70ecd,
        ex: -2,
        sgn: 0,
    },
    Tint {
        h: 0x8fd9c7172ab0828e,
        m: 0x1a9f72fadafc3dfb,
        l: 0xaf96a180ea297178,
        ex: -1,
        sgn: 0,
    },
    Tint {
        h: 0xe38168b78a51ee15,
        m: 0x604a47815c8091eb,
        l: 0x9faaa42312e8eb30,
        ex: -1,
        sgn: 0,
    },
    Tint {
        h: 0xa164ccdbb65ee133,
        m: 0xe623433c5e086402,
        l: 0xcb228afd335a245f,
        ex: 0,
        sgn: 0,
    },
    Tint {
        h: 0xced0c0cbe44fac2f,
        m: 0xd7c2adfc7c5dda9e,
        l: 0xa22b4a311cf5a5b1,
        ex: 0,
        sgn: 0,
    },
    Tint {
        h: 0xf0e6660438c90371,
        m: 0x3dca1a0f58f0d943,
        l: 0x8a92b9de68c4488c,
        ex: 0,
        sgn: 0,
    },
    Tint {
        h: 0x8000000000000000,
        m: 0x0,
        l: 0x0,
        ex: 1,
        sgn: 0,
    },
    Tint {
        h: 0xf8ea68f9ad849103,
        m: 0xb257e4bfc606967f,
        l: 0x818fa2816928c4aa,
        ex: 0,
        sgn: 0,
    },
    Tint {
        h: 0xddc64eeee3506234,
        m: 0x47a33ea40a2b22e4,
        l: 0xa6a2c129ace2470a,
        ex: 0,
        sgn: 0,
    },
    Tint {
        h: 0xb52ddfba2d37c8e2,
        m: 0x24551a9b19fa064a,
        l: 0x3d78c68b54bc3850,
        ex: 0,
        sgn: 0,
    },
    Tint {
        h: 0x87ac113d26b683a1,
        m: 0x4e97756c38785b0b,
        l: 0xf75f0a4506931ba8,
        ex: 0,
        sgn: 0,
    },
    Tint {
        h: 0xba065070b900f3d5,
        m: 0xe49d8f9f238f1eeb,
        l: 0x6805d48d67f1dcd9,
        ex: -1,
        sgn: 0,
    },
    Tint {
        h: 0xe902d2a4bfc22441,
        m: 0x945cfe6c83a5b828,
        l: 0x519ebb457f8ccca0,
        ex: -2,
        sgn: 0,
    },
    Tint {
        h: 0x84e405c915f7769b,
        m: 0xfaaf34c3c51dd39,
        l: 0xa97fa0ae99382c93,
        ex: -2,
        sgn: 0,
    },
    Tint {
        h: 0x896da46c7e6cebb3,
        m: 0xe0afc0b6db3dbbf5,
        l: 0x61ccff6e78efff4,
        ex: -3,
        sgn: 0,
    },
    Tint {
        h: 0x801a0c1a0dfefc92,
        m: 0xf8e0a97902e08d47,
        l: 0x81898e0b7d8ef0bc,
        ex: -4,
        sgn: 0,
    },
    Tint {
        h: 0xd59d088930df07db,
        m: 0xa8cb00021849fbaf,
        l: 0xf80cc91aa0bddf89,
        ex: -6,
        sgn: 0,
    },
    Tint {
        h: 0x9db56a6984b77231,
        m: 0xcf7c2074e0480959,
        l: 0x3316ecaa6aa4604e,
        ex: -7,
        sgn: 0,
    },
    Tint {
        h: 0xcb7450d9d62d2e68,
        m: 0x9787b4265977d004,
        l: 0xba615cee00a841d7,
        ex: -9,
        sgn: 0,
    },
    Tint {
        h: 0xe12a52e02b64b23e,
        m: 0x97c9139a8ced716c,
        l: 0x70a176ad2b5f16d2,
        ex: -11,
        sgn: 0,
    },
    Tint {
        h: 0xd057b6d0900ed30d,
        m: 0x5e04bb17eb879a5e,
        l: 0x23ed135b60bea070,
        ex: -13,
        sgn: 0,
    },
    Tint {
        h: 0x9b1f1ca22fef7ae5,
        m: 0x19a687f0a4073eee,
        l: 0x5a9ccbc8c14578b,
        ex: -15,
        sgn: 0,
    },
    Tint {
        h: 0xaeba710f053aac3b,
        m: 0x30ea5d6fb116f775,
        l: 0x8a9bcfef5698f8f1,
        ex: -18,
        sgn: 0,
    },
    Tint {
        h: 0x84b9ada076cd007d,
        m: 0x13e55006073f76b1,
        l: 0x4cf4259968228f3e,
        ex: -21,
        sgn: 0,
    },
    Tint {
        h: 0xcca4b9046786937d,
        m: 0x233c1e691c127c1f,
        l: 0xf493e0b7a3de459a,
        ex: -26,
        sgn: 0,
    },
];
const F2: [[f64; 2]; 65] = [
    [
        f64::from_bits(0x0000000000000000),
        f64::from_bits(0x0000000000000000),
    ],
    [
        f64::from_bits(0xbca95220c39d4dff),
        f64::from_bits(0x3f8fff555bbb7300),
    ],
    [
        f64::from_bits(0x3ca2542779d776de),
        f64::from_bits(0x3f9ffd55bba97600),
    ],
    [
        f64::from_bits(0xbca6061bbe3de53c),
        f64::from_bits(0x3fa7fb818430da40),
    ],
    [
        f64::from_bits(0xbca639269b0da47e),
        f64::from_bits(0x3faff55bb72cfe00),
    ],
    [
        f64::from_bits(0xbc84a7663af440f7),
        f64::from_bits(0x3fb3f59f0e7c55a0),
    ],
    [
        f64::from_bits(0x3cad1824d59f9e13),
        f64::from_bits(0x3fb7ee182602f100),
    ],
    [
        f64::from_bits(0x3c8bef71e5340b31),
        f64::from_bits(0x3fbbe39ebe6f07c0),
    ],
    [
        f64::from_bits(0x3cab8cb225e627d0),
        f64::from_bits(0x3fbfd5ba9aac2f60),
    ],
    [
        f64::from_bits(0x3cab92de9bac94c2),
        f64::from_bits(0x3fc1e1fafb043720),
    ],
    [
        f64::from_bits(0xbc9d3cb89e62dafd),
        f64::from_bits(0x3fc3d6eee8c66270),
    ],
    [
        f64::from_bits(0xbca882a55960087a),
        f64::from_bits(0x3fc5c9811e3ec270),
    ],
    [
        f64::from_bits(0x3c91347b0b4f881d),
        f64::from_bits(0x3fc7b97b4bce5b00),
    ],
    [
        f64::from_bits(0x3ca873d8079ed0d2),
        f64::from_bits(0x3fc9a6a8e96c8620),
    ],
    [
        f64::from_bits(0x3c9022f621a5c1cb),
        f64::from_bits(0x3fcb90d7529260a0),
    ],
    [
        f64::from_bits(0x3ca9c648d1534598),
        f64::from_bits(0x3fcd77d5df205730),
    ],
    [
        f64::from_bits(0xbc94ea9238610a08),
        f64::from_bits(0x3fcf5b75f92c80e0),
    ],
    [
        f64::from_bits(0x3ca2c5c8e721970d),
        f64::from_bits(0x3fd09dc597d86360),
    ],
    [
        f64::from_bits(0x3c630ca4748b1bf9),
        f64::from_bits(0x3fd18bf5a30bf178),
    ],
    [
        f64::from_bits(0xbca20ef9ba6dbf90),
        f64::from_bits(0x3fd278372057ef48),
    ],
    [
        f64::from_bits(0x3cae69c5abb498d2),
        f64::from_bits(0x3fd362773707ebc8),
    ],
    [
        f64::from_bits(0x3c9a8a86f0ea9311),
        f64::from_bits(0x3fd44aa436c2af08),
    ],
    [
        f64::from_bits(0x3c9db5336feef7f0),
        f64::from_bits(0x3fd530ad9951cd48),
    ],
    [
        f64::from_bits(0x3c99636a3aa3b840),
        f64::from_bits(0x3fd614840309cfe0),
    ],
    [
        f64::from_bits(0x3c81ce2a8c848b74),
        f64::from_bits(0x3fd6f61941e4def0),
    ],
    [
        f64::from_bits(0xbc84b1bbd1ea6db3),
        f64::from_bits(0x3fd7d5604b63b3f8),
    ],
    [
        f64::from_bits(0xbca4925e8b916e0b),
        f64::from_bits(0x3fd8b24d394a1b28),
    ],
    [
        f64::from_bits(0x3c79e6c988fd0a77),
        f64::from_bits(0x3fd98cd5454d6b18),
    ],
    [
        f64::from_bits(0xbcaa49bd836a1700),
        f64::from_bits(0x3fda64eec3cc2400),
    ],
    [
        f64::from_bits(0xbcaca3cf09c6b5f8),
        f64::from_bits(0x3fdb3a911da65c70),
    ],
    [
        f64::from_bits(0xbc7cc1ce70934c34),
        f64::from_bits(0x3fdc0db4c94ec9f0),
    ],
    [
        f64::from_bits(0x3c82e982ddf3872a),
        f64::from_bits(0x3fdcde53432c1350),
    ],
    [
        f64::from_bits(0xbc82ea406ee84d0f),
        f64::from_bits(0x3fddac670561bb50),
    ],
    [
        f64::from_bits(0xbcade35847c81979),
        f64::from_bits(0x3fde77eb7f175a38),
    ],
    [
        f64::from_bits(0xbc6a3992dc382a23),
        f64::from_bits(0x3fdf40dd0b541418),
    ],
    [
        f64::from_bits(0xbc8b32c949c9d593),
        f64::from_bits(0x3fe0039c73c1a40c),
    ],
    [
        f64::from_bits(0xbc7d5b495f6349e6),
        f64::from_bits(0x3fe0657e94db30d0),
    ],
    [
        f64::from_bits(0xbcaf34582f6255fe),
        f64::from_bits(0x3fe0c6145b5b43dc),
    ],
    [
        f64::from_bits(0x3c9ed42511e3f11d),
        f64::from_bits(0x3fe1255d9bfbd2a8),
    ],
    [
        f64::from_bits(0xbc91cef189ff9e7f),
        f64::from_bits(0x3fe1835a88be7c14),
    ],
    [
        f64::from_bits(0xbc5928df287a668f),
        f64::from_bits(0x3fe1e00babdefeb4),
    ],
    [
        f64::from_bits(0xbcae3bde360c7ddb),
        f64::from_bits(0x3fe23b71e2cc9e6c),
    ],
    [
        f64::from_bits(0x3c9bd86313ce4fde),
        f64::from_bits(0x3fe2958e59308e30),
    ],
    [
        f64::from_bits(0xbca8e8a85803cc1d),
        f64::from_bits(0x3fe2ee628406cbcc),
    ],
    [
        f64::from_bits(0xbc977ef7641c777f),
        f64::from_bits(0x3fe345f01cce37bc),
    ],
    [
        f64::from_bits(0x3cab73ef3389d02f),
        f64::from_bits(0x3fe39c391cd41718),
    ],
    [
        f64::from_bits(0x3c7ecf8b492644f0),
        f64::from_bits(0x3fe3f13fb89e96f4),
    ],
    [
        f64::from_bits(0x3cac1125fd3810c7),
        f64::from_bits(0x3fe445065b795b54),
    ],
    [
        f64::from_bits(0x3ca2483350fe548b),
        f64::from_bits(0x3fe4978fa3269ee0),
    ],
    [
        f64::from_bits(0x3c84a33dbeb3796c),
        f64::from_bits(0x3fe4e8de5bb6ec04),
    ],
    [
        f64::from_bits(0xbca46edd2af69483),
        f64::from_bits(0x3fe538f57b890620),
    ],
    [
        f64::from_bits(0xbca2bcb93b18b52a),
        f64::from_bits(0x3fe587d81f732fbc),
    ],
    [
        f64::from_bits(0x3c60028e4bc5e7ca),
        f64::from_bits(0x3fe5d58987169b18),
    ],
    [
        f64::from_bits(0x3caed487acaf1174),
        f64::from_bits(0x3fe6220d115d7b8c),
    ],
    [
        f64::from_bits(0xbca2dd4dfd7d1777),
        f64::from_bits(0x3fe66d663923e088),
    ],
    [
        f64::from_bits(0x3c92bfe3cf3b9d79),
        f64::from_bits(0x3fe6b798920b3d98),
    ],
    [
        f64::from_bits(0xbc78c34d25aadef6),
        f64::from_bits(0x3fe700a7c5784634),
    ],
    [
        f64::from_bits(0xbc9f426acf4d3bdb),
        f64::from_bits(0x3fe748978fba8e10),
    ],
    [
        f64::from_bits(0xbcaafe57dd9ff230),
        f64::from_bits(0x3fe78f6bbd5d3160),
    ],
    [
        f64::from_bits(0xbc954fbef0e862ab),
        f64::from_bits(0x3fe7d528289fa094),
    ],
    [
        f64::from_bits(0x3c990227758b11ba),
        f64::from_bits(0x3fe819d0b7158a4c),
    ],
    [
        f64::from_bits(0x3ca16b66e7fc8b8c),
        f64::from_bits(0x3fe85d69576cc2c4),
    ],
    [
        f64::from_bits(0xbc855b9a5e177a1b),
        f64::from_bits(0x3fe89ff5ff57f1f8),
    ],
    [
        f64::from_bits(0x3cac27cfaa9f7a14),
        f64::from_bits(0x3fe8e17aa99cc05c),
    ],
    [
        f64::from_bits(0x3c81a62633145c07),
        f64::from_bits(0x3fe921fb54442d18),
    ],
];
const O: [[f64; 2]; 8] = [
    [
        f64::from_bits(0x0000000000000000),
        f64::from_bits(0x0000000000000000),
    ],
    [
        f64::from_bits(0x3ff921fb54442d18),
        f64::from_bits(0x3c91a62633145c07),
    ],
    [
        f64::from_bits(0x0000000000000000),
        f64::from_bits(0x0000000000000000),
    ],
    [
        f64::from_bits(0xbff921fb54442d18),
        f64::from_bits(0xbc91a62633145c07),
    ],
    [
        f64::from_bits(0x400921fb54442d18),
        f64::from_bits(0x3ca1a62633145c07),
    ],
    [
        f64::from_bits(0x3ff921fb54442d18),
        f64::from_bits(0x3c91a62633145c07),
    ],
    [
        f64::from_bits(0xc00921fb54442d18),
        f64::from_bits(0xbca1a62633145c07),
    ],
    [
        f64::from_bits(0xbff921fb54442d18),
        f64::from_bits(0xbc91a62633145c07),
    ],
];

fn two(x: f64, y: f64) -> (f64, f64) {
    let s = x + y;
    (s, y - (s - x))
}
#[inline(never)]
fn accurate(y: f64, x: f64) -> f64 {
    let t = y / x;
    if t == 0.0 {
        return if x > 0.0 {
            t
        } else {
            core::f64::consts::PI.copysign(y)
        };
    }
    let corr = fma(t, x, -y);
    if corr == 0.0
        && x > 0.0
        && t.abs() <= f64::from_bits(0x3e4d12ed0af1a27f)
        && (y.abs() >= f64::from_bits(0x06c0000000000000)
            || fma(
                t * f64::from_bits(0x4680000000000000),
                x,
                -y * f64::from_bits(0x4680000000000000),
            ) == 0.0)
    {
        return fma(t, -f64::from_bits(0x3c90000000000000), t);
    }
    let inv = y.abs() > x.abs();
    let mut z = if inv {
        Tint::from_double(x).div(Tint::from_double(y))
    } else {
        Tint::from_double(y).div(Tint::from_double(x))
    };
    if !inv && x > 0.0 && z.ex <= -96 {
        z.l = z.l.wrapping_sub(2);
        z.m = z.m.wrapping_sub(u64::from(z.l < 2));
        z.h = z.h.wrapping_sub(u64::from(z.m < 1));
        return z.to_double(1);
    }
    let sz = z.sgn;
    z.sgn = 0;
    let (mut p, mut q) = (P[29], Q[29]);
    for i in (0..29).rev() {
        p = p.mul(z).add(P[i]);
        q = q.mul(z).add(Q[i]);
    }
    z = p.mul(z).div(q);
    z.sgn = sz;
    let mut err = 662;
    if inv {
        if z.sgn == 0 {
            z = PI2.add(z.neg());
        } else {
            z = PI2.add(z);
            z.sgn = 1;
        }
        err = 524;
    }
    if x < 0.0 {
        if z.sgn == 0 {
            z = PI.add(z.neg());
            z.sgn = 1;
        } else {
            z = PI.add(z);
        }
        err = 266;
    }
    z.to_double(err)
}
/// Correctly rounded atan2(y, x), nearest-even with gradual underflow.
pub fn atan2(y0: f64, x0: f64) -> f64 {
    if y0.is_nan() || x0.is_nan() {
        return y0 + x0;
    }
    if y0 == 0.0 {
        return if x0.is_sign_negative() {
            core::f64::consts::PI.copysign(y0)
        } else {
            y0
        };
    }
    if x0 == 0.0 {
        return core::f64::consts::FRAC_PI_2.copysign(y0);
    }
    if y0.is_infinite() {
        return (if x0.is_infinite() {
            if x0.is_sign_negative() {
                f64::from_bits(0x4002d97c7f3321d2)
            } else {
                core::f64::consts::FRAC_PI_4
            }
        } else {
            core::f64::consts::FRAC_PI_2
        })
        .copysign(y0);
    }
    if x0.is_infinite() {
        return (if x0.is_sign_negative() {
            core::f64::consts::PI
        } else {
            0.0
        })
        .copysign(y0);
    }
    let ax = x0.abs();
    let ay = y0.abs();
    let aix = ax.to_bits();
    let aiy = ay.to_bits();
    let gt = u64::from(aix < aiy);
    let sx = x0.to_bits() >> 63;
    let sy = y0.to_bits() >> 63;
    if (aix.wrapping_sub(aiy) ^ 0u64.wrapping_sub(gt)) >= (53u64 << 52) {
        return accurate(y0, x0);
    }
    let (mut x, mut y) = if gt != 0 { (ay, ax) } else { (ax, ay) };
    let s = if gt ^ sx ^ sy == 0 { 1.0 } else { -1.0 };
    let kw = ((sx << 2) | (sy << 1) | gt) as usize;
    let jt = (((y / x + (2.0 + 1.0 / 128.0)).to_bits() >> 45) & 127) as usize;
    let (mut fh, mut fl) = (F2[jt][1] * s + O[kw][0], F2[jt][0] * s + O[kw][1]);
    if x < f64::from_bits(0x0670000000000000) {
        x *= f64::from_bits(0x7970000000000000);
        y *= f64::from_bits(0x7970000000000000);
    }
    if x > f64::from_bits(0x7fd0000000000000) && jt != 0 {
        x *= 0.5;
        y *= 0.5;
    }
    let t0 = jt as f64 / 64.0;
    let zn = fma(-t0, x, y);
    let zd = fma(t0, y, x);
    let mut z = zn / zd;
    let mut z2 = z * z;
    z *= s;
    let dz = (z * z2)
        * (-f64::from_bits(0x3fd55555555554d2)
            + z2 * (f64::from_bits(0x3fc999999860e1ca)
                + z2 * (-f64::from_bits(0x3fc248ad469844a1))));
    let mut eps = z.abs() * f64::from_bits(0x3cc0510000000000) + f64::from_bits(0x3a50000000000000);
    let (rh, z) = two(fh, z);
    let rl = (fl + dz) + z;
    let mut lb = rh + (rl - eps);
    let mut ub = rh + (rl + eps);
    if lb != ub {
        let dh = y * t0;
        let mut dl = fma(y, t0, -dh);
        let (dh, e) = two(x, dh);
        let rdh = 1.0 / dh;
        dl += e;
        let nh = x * t0;
        let nl = fma(x, t0, -nh);
        let dt = y - nh;
        let y1 = dt + nh;
        let (nh, nl) = two(dt, if y1 == y { -nl } else { (y - y1) - nl });
        let mut zh = nh * rdh;
        z2 = zh * zh;
        let mut zl = rdh * (fma(dh, -zh, nh) + (nl - (nh * rdh) * dl));
        zl += zh
            * z2
            * ((-f64::from_bits(0x3fd5555555555555) + z2 * f64::from_bits(0x3fc999999999755e))
                + (z2 * z2)
                    * (-f64::from_bits(0x3fc24924883596f8)
                        + z2 * f64::from_bits(0x3fbc6f7d73531bc2)));
        zh *= s;
        zl *= s;
        eps = f64::from_bits(0x3cd4000000000000)
            * (zh.abs() * z2 + f64::from_bits(0x3cc0000000000000));
        let (sh, sl) = two(fh, zh);
        fh = sh;
        fl = (fl + zl) + sl;
        lb = fh + (fl - eps);
        ub = fh + (fl + eps);
        if lb != ub {
            return accurate(y0, x0);
        }
    }
    ub
}

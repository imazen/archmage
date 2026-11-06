/* Correctly-rounded tangent function for binary64 value.

Copyright (c) 2022-2025 Paul Zimmermann and Tom Hubrecht

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
const TINV: [u64; 256] = [
    0xff00ff00ff00ff02,
    0xfe03f80fe03f80ff,
    0xfd08e5500fd08e56,
    0xfc0fc0fc0fc0fc11,
    0xfb18856506ddaba7,
    0xfa232cf252138ac1,
    0xf92fb2211855a867,
    0xf83e0f83e0f83e11,
    0xf74e3fc22c700f76,
    0xf6603d980f6603db,
    0xf57403d5d00f5742,
    0xf4898d5f85bb3952,
    0xf3a0d52cba872338,
    0xf2b9d6480f2b9d66,
    0xf1d48bcee0d399fc,
    0xf0f0f0f0f0f0f0f2,
    0xf00f00f00f00f010,
    0xef2eb71fc4345239,
    0xee500ee500ee5010,
    0xed7303b5cc0ed731,
    0xec979118f3fc4da3,
    0xebbdb2a5c1619c8d,
    0xeae56403ab959010,
    0xea0ea0ea0ea0ea10,
    0xe939651fe2d8d35d,
    0xe865ac7b7603a198,
    0xe79372e225fe30da,
    0xe6c2b4481cd8568a,
    0xe5f36cb00e5f36cc,
    0xe525982af70c880f,
    0xe45932d7dc52100f,
    0xe38e38e38e38e38f,
    0xe2c4a6886a4c2e11,
    0xe1fc780e1fc780e3,
    0xe135a9c97500e137,
    0xe070381c0e070383,
    0xdfac1f74346c5760,
    0xdee95c4ca037ba58,
    0xde27eb2c41f3d9d2,
    0xdd67c8a60dd67c8b,
    0xdca8f158c7f91ab9,
    0xdbeb61eed19c5959,
    0xdb2f171df770291a,
    0xda740da740da740f,
    0xd9ba4256c0366e92,
    0xd901b2036406c80f,
    0xd84a598ec9151f44,
    0xd79435e50d79435f,
    0xd6df43fca482f00e,
    0xd62b80d62b80d62c,
    0xd578e97c3f5fe552,
    0xd4c77b03531dec0e,
    0xd4173289870ac52e,
    0xd3680d3680d3680e,
    0xd2ba083b445250ac,
    0xd20d20d20d20d20e,
    0xd161543e28e50275,
    0xd0b69fcbd2580d0c,
    0xd00d00d00d00d00e,
    0xcf6474a8819ec8ea,
    0xcebcf8bb5b4169cc,
    0xce168a7725080ce2,
    0xcd712752a886d243,
    0xccccccccccccccce,
    0xcc29786c7607f99f,
    0xcb8727c065c393e1,
    0xcae5d85f1bbd6c96,
    0xca4587e6b74f032a,
    0xc9a633fcd967300d,
    0xc907da4e871146ad,
    0xc86a78900c86a78a,
    0xc7ce0c7ce0c7ce0d,
    0xc73293d789b9f839,
    0xc6980c6980c6980d,
    0xc5fe740317f9d00d,
    0xc565c87b5f9d4d1c,
    0xc4ce07b00c4ce07c,
    0xc4372f855d824ca6,
    0xc3a13de60495c774,
    0xc30c30c30c30c30d,
    0xc2780613c0309e02,
    0xc1e4bbd595f6e948,
    0xc152500c152500c2,
    0xc0c0c0c0c0c0c0c1,
    0xc0300c0300c0300d,
    0xbfa02fe80bfa02ff,
    0xbf112a8ad278e8de,
    0xbe82fa0be82fa0bf,
    0xbdf59c91700bdf5a,
    0xbd69104707661aa3,
    0xbcdd535db1cc5b7c,
    0xbc52640bc52640bd,
    0xbbc8408cd63069a1,
    0xbb3ee721a54d880c,
    0xbab656100bab6562,
    0xba2e8ba2e8ba2e8c,
    0xb9a7862a0ff46588,
    0xb92143fa36f5e02f,
    0xb89bc36ce3e0453b,
    0xb81702e05c0b8171,
    0xb79300b79300b794,
    0xb70fbb5a19be3659,
    0xb68d31340e4307d9,
    0xb60b60b60b60b60c,
    0xb58a485518d1e7e4,
    0xb509e68a9b948220,
    0xb48a39d44685fe97,
    0xb40b40b40b40b40c,
    0xb38cf9b00b38cf9b,
    0xb30f63528917c80c,
    0xb2927c29da5519d0,
    0xb21642c8590b2165,
    0xb19ab5c45606f00c,
    0xb11fd3b80b11fd3c,
    0xb0a59b418d749d54,
    0xb02c0b02c0b02c0b,
    0xafb321a1496fdf0f,
    0xaf3addc680af3ade,
    0xaec33e1f671529a5,
    0xae4c415c9882b931,
    0xadd5e6323fd48a87,
    0xad602b580ad602b6,
    0xaceb0f891e6551bc,
    0xac7691840ac76919,
    0xac02b00ac02b00ac,
    0xab8f69e28359cd12,
    0xab1cbdd3e2970f60,
    0xaaaaaaaaaaaaaaab,
    0xaa392f35dc17f00b,
    0xa9c84a47a07f5638,
    0xa957fab5402a55ff,
    0xa8e83f5717c0a8e9,
    0xa87917088e262b70,
    0xa80a80a80a80a80b,
    0xa79c7b16ea64d422,
    0xa72f05397829cbc2,
    0xa6c21df6e1625c80,
    0xa655c4392d7b73a8,
    0xa5e9f6ed347f0721,
    0xa57eb50295fad40b,
    0xa513fd6bb00a5140,
    0xa4a9cf1d96833751,
    0xa44029100a440291,
    0xa3d70a3d70a3d70b,
    0xa36e71a2cb033129,
    0xa3065e3fae7cd0e0,
    0xa29ecf163bb6500a,
    0xa237c32b16cfd772,
    0xa1d139855f7268ee,
    0xa16b312ea8fc377d,
    0xa105a932f2ca891f,
    0xa0a0a0a0a0a0a0a1,
    0xa03c1688732b3032,
    0x9fd809fd809fd80a,
    0x9f747a152d7836d0,
    0x9f1165e7254813e2,
    0x9eaecc8d53ae2ddf,
    0x9e4cad23dd5f3a20,
    0x9deb06c9194aa416,
    0x9d89d89d89d89d8a,
    0x9d2921c3d6411308,
    0x9cc8e160c3fb19b9,
    0x9c69169b30446dfa,
    0x9c09c09c09c09c0a,
    0x9baade8e4a2f6e10,
    0x9b4c6f9ef03a3caa,
    0x9aee72fcf957c10f,
    0x9a90e7d95bc609a9,
    0x9a33cd67009a33ce,
    0x99d722dabde58f06,
    0x997ae76b50efd00a,
    0x991f1a515885fb37,
    0x98c3bac74f5db00a,
    0x9868c809868c8099,
    0x980e4156201301c8,
    0x97b425ed097b425f,
    0x975a750ff68a58af,
    0x97012e025c04b80a,
    0x96a850096a850097,
    0x964fda6c0964fda7,
    0x95f7cc72d1b887e9,
    0x95a02568095a0257,
    0x9548e4979e0829fd,
    0x94f2094f2094f209,
    0x949b92ddc02526e5,
    0x9445809445809446,
    0x93efd1c50e726b7c,
    0x939a85c40939a85c,
    0x93459be6b009345a,
    0x92f113840497889c,
    0x929cebf48bbd90e5,
    0x9249249249249249,
    0x91f5bcb8bb02d9cd,
    0x91a2b3c4d5e6f809,
    0x9150091500915009,
    0x90fdbc090fdbc091,
    0x90abcc0242af3009,
    0x905a38633e06c43b,
    0x9009009009009009,
    0x8fb823ee08fb823f,
    0x8f67a1e3fdc26179,
    0x8f1779d9fdc3a219,
    0x8ec7ab397255e41d,
    0x8e78356d1408e783,
    0x8e2917e0e702c6ce,
    0x8dda520237694809,
    0x8d8be33f95d71590,
    0x8d3dcb08d3dcb08d,
    0x8cf008cf008cf009,
    0x8ca29c046514e023,
    0x8c55841c815ed5ca,
    0x8c08c08c08c08c09,
    0x8bbc50c8deb420c0,
    0x8b70344a139bc75b,
    0x8b246a87e19008b2,
    0x8ad8f2fba9386823,
    0x8a8dcd1feeae465c,
    0x8a42f8705669db46,
    0x89f87469a23920e0,
    0x89ae4089ae4089ae,
    0x89645c4f6e055dec,
    0x891ac73ae9819b50,
    0x88d180cd3a4133d7,
    0x8888888888888889,
    0x883fddf00883fddf,
    0x87f78087f78087f8,
    0x87af6fd5992d0d40,
    0x8767ab5f34e47ef1,
    0x872032ac13008720,
    0x86d905447a34acc6,
    0x869222b1acf1ce96,
    0x864b8a7de6d1d608,
    0x86053c345a0b8473,
    0x85bf37612cee3c9b,
    0x85797b917765ab89,
    0x8534085340853408,
    0x84eedd357c1b0085,
    0x84a9f9c8084a9f9d,
    0x84655d9bab2f1008,
    0x8421084210842108,
    0x83dcf94dc7570ce1,
    0x839930523fbe3368,
    0x8355ace3c897db10,
    0x83126e978d4fdf3b,
    0x82cf750393ac3319,
    0x828cbfbeb9a020a3,
    0x824a4e60b3262bc5,
    0x8208208208208208,
    0x81c635bc123fdf8e,
    0x81848da8faf0d277,
    0x814327e3b94f462f,
    0x8102040810204081,
    0x80c121b28bd1ba98,
    0x8080808080808081,
    0x8040201008040201,
    0x8000000000000000,
];
const T: [u64; 20] = [
    0x28be60db9391054a,
    0x7f09d5f47d4d3770,
    0x36d8a5664f10e410,
    0x7f9458eaf7aef158,
    0x6dc91b8e909374b8,
    0x1924bba82746487,
    0x3f877ac72c4a69cf,
    0xba208d7d4baed121,
    0x3a671c09ad17df90,
    0x4e64758e60d4ce7d,
    0x272117e2ef7e4a0e,
    0xc7fe25fff7816603,
    0xfbcbc462d6829b47,
    0xdb4d9fb3c9f2c26d,
    0xd3d18fd9a797fa8b,
    0x5d49eeb1faf97c5e,
    0xcf41ce7de294a4ba,
    0x9afed7ec47e35742,
    0x1580cc11bf1edaea,
    0xfc33ef0826bd0d87,
];
const S: [D; 256] = [
    D {
        hi: 0x0,
        lo: 0x0,
        ex: 128,
        sgn: 0,
    },
    D {
        hi: 0xc90fc5f66525d257,
        lo: 0x480f7956b6470765,
        ex: -8,
        sgn: 0,
    },
    D {
        hi: 0xc90f87f3380388d5,
        lo: 0xcb3ff35bd4d81baa,
        ex: -7,
        sgn: 0,
    },
    D {
        hi: 0x96cb587284b81770,
        lo: 0xb767005691b9d9d1,
        ex: -6,
        sgn: 0,
    },
    D {
        hi: 0xc90e8fe6f63c2330,
        lo: 0xf1d7d06db39ea9fc,
        ex: -6,
        sgn: 0,
    },
    D {
        hi: 0xfb514b55ccbe541a,
        lo: 0xd784e031f9af76d6,
        ex: -6,
        sgn: 0,
    },
    D {
        hi: 0x96c9b5df1877e9b5,
        lo: 0xf91ee371d6467dca,
        ex: -5,
        sgn: 0,
    },
    D {
        hi: 0xafea690fd5912ef3,
        lo: 0xf56e3c87ae3c56df,
        ex: -5,
        sgn: 0,
    },
    D {
        hi: 0xc90aafbd1b33efc9,
        lo: 0xc539edcbfda0cf2c,
        ex: -5,
        sgn: 0,
    },
    D {
        hi: 0xe22a7a6729d8e453,
        lo: 0x850021e392744a4f,
        ex: -5,
        sgn: 0,
    },
    D {
        hi: 0xfb49b98e8e7807f6,
        lo: 0xb21ccebc9caac3,
        ex: -5,
        sgn: 0,
    },
    D {
        hi: 0x8a342eda160bf5ae,
        lo: 0xde5b1068d174be9c,
        ex: -4,
        sgn: 0,
    },
    D {
        hi: 0x96c32baca2ae68b4,
        lo: 0x37b2dd49d5fca3c0,
        ex: -4,
        sgn: 0,
    },
    D {
        hi: 0xa351cb7fc30bc889,
        lo: 0xb56007d16d4ad5a3,
        ex: -4,
        sgn: 0,
    },
    D {
        hi: 0xafe00694866a1b44,
        lo: 0xcd34d2751c2e1da7,
        ex: -4,
        sgn: 0,
    },
    D {
        hi: 0xbc6dd52c3a342eb5,
        lo: 0xf10bfca3d6464012,
        ex: -4,
        sgn: 0,
    },
    D {
        hi: 0xc8fb2f886ec09f37,
        lo: 0x6a17954b2b7c5171,
        ex: -4,
        sgn: 0,
    },
    D {
        hi: 0xd5880deafc18b534,
        lo: 0x73d1472472f4a390,
        ex: -4,
        sgn: 0,
    },
    D {
        hi: 0xe214689606bf1676,
        lo: 0x438b4a73aecd2541,
        ex: -4,
        sgn: 0,
    },
    D {
        hi: 0xeea037cc04764844,
        lo: 0xc4e92d01a2f42935,
        ex: -4,
        sgn: 0,
    },
    D {
        hi: 0xfb2b73cfc106ff68,
        lo: 0xf0a0e36a000c7350,
        ex: -4,
        sgn: 0,
    },
    D {
        hi: 0x83db0a7231831d8f,
        lo: 0x60e782313f6161af,
        ex: -3,
        sgn: 0,
    },
    D {
        hi: 0x8a2009a6b84d9402,
        lo: 0x77724a2b2a669bc4,
        ex: -3,
        sgn: 0,
    },
    D {
        hi: 0x9064b3a76a22640c,
        lo: 0x56e0a8b0d177b55d,
        ex: -3,
        sgn: 0,
    },
    D {
        hi: 0x96a9049670cfae65,
        lo: 0xf77574094d3c35c4,
        ex: -3,
        sgn: 0,
    },
    D {
        hi: 0x9cecf8962d14c822,
        lo: 0x50ffe4f5caa7f1fa,
        ex: -3,
        sgn: 0,
    },
    D {
        hi: 0xa3308bc93904ad69,
        lo: 0xdec1b7f2768bdafa,
        ex: -3,
        sgn: 0,
    },
    D {
        hi: 0xa973ba526a6850d9,
        lo: 0x76f8c63986598c79,
        ex: -3,
        sgn: 0,
    },
    D {
        hi: 0xafb68054d520c60b,
        lo: 0xfdd2fc0936594c2d,
        ex: -3,
        sgn: 0,
    },
    D {
        hi: 0xb5f8d9f3cd8945d6,
        lo: 0x924bef13600f9852,
        ex: -3,
        sgn: 0,
    },
    D {
        hi: 0xbc3ac352ead90abe,
        lo: 0xeb13e106732687f1,
        ex: -3,
        sgn: 0,
    },
    D {
        hi: 0xc27c389609850433,
        lo: 0xb228a03916371f6f,
        ex: -3,
        sgn: 0,
    },
    D {
        hi: 0xc8bd35e14da15f0e,
        lo: 0xc7396c894bbf7389,
        ex: -3,
        sgn: 0,
    },
    D {
        hi: 0xcefdb7592542e1e9,
        lo: 0x6b47b8c44e5b037e,
        ex: -3,
        sgn: 0,
    },
    D {
        hi: 0xd53db9224ae01bca,
        lo: 0x7337412cf70716cb,
        ex: -3,
        sgn: 0,
    },
    D {
        hi: 0xdb7d3761c7b263b6,
        lo: 0xbb286d23e11c8337,
        ex: -3,
        sgn: 0,
    },
    D {
        hi: 0xe1bc2e3cf616a7ac,
        lo: 0x31883b30137c6e62,
        ex: -3,
        sgn: 0,
    },
    D {
        hi: 0xe7fa99d983ee098f,
        lo: 0xeeb8f9c33340a2f2,
        ex: -3,
        sgn: 0,
    },
    D {
        hi: 0xee38765d74fe4897,
        lo: 0xed16b994af6c18ae,
        ex: -3,
        sgn: 0,
    },
    D {
        hi: 0xf475bfef2551f5b9,
        lo: 0x14e1a5488eaeab96,
        ex: -3,
        sgn: 0,
    },
    D {
        hi: 0xfab272b54b9871a2,
        lo: 0x704729ae56d78a37,
        ex: -3,
        sgn: 0,
    },
    D {
        hi: 0x8077456b7dc2d967,
        lo: 0x3eac8308f1113e5e,
        ex: -2,
        sgn: 0,
    },
    D {
        hi: 0x8395023dd418e919,
        lo: 0xdb1f70118c9c2198,
        ex: -2,
        sgn: 0,
    },
    D {
        hi: 0x86b26de5933c2e8e,
        lo: 0xc5a9decdfaad4db5,
        ex: -2,
        sgn: 0,
    },
    D {
        hi: 0x89cf8676d7abb55b,
        lo: 0x97965c9860c34e44,
        ex: -2,
        sgn: 0,
    },
    D {
        hi: 0x8cec4a05f12739e8,
        lo: 0xdcdca90cc73b116a,
        ex: -2,
        sgn: 0,
    },
    D {
        hi: 0x9008b6a763de75b7,
        lo: 0xa6e3df5975cca9da,
        ex: -2,
        sgn: 0,
    },
    D {
        hi: 0x9324ca6fe9a04b4e,
        lo: 0x899c4de737feec22,
        ex: -2,
        sgn: 0,
    },
    D {
        hi: 0x964083747309d113,
        lo: 0xa89a11e07c1fe,
        ex: -2,
        sgn: 0,
    },
    D {
        hi: 0x995bdfca28b53a54,
        lo: 0x49c4863de522b217,
        ex: -2,
        sgn: 0,
    },
    D {
        hi: 0x9c76dd866c689dcc,
        lo: 0xe7bc08111d0bfca4,
        ex: -2,
        sgn: 0,
    },
    D {
        hi: 0x9f917abeda4498df,
        lo: 0xf3ff913a4aadb85e,
        ex: -2,
        sgn: 0,
    },
    D {
        hi: 0xa2abb58949f2ced7,
        lo: 0xa5dbee6084ee1260,
        ex: -2,
        sgn: 0,
    },
    D {
        hi: 0xa5c58bfbcfd4436a,
        lo: 0x69fcb11e19f58619,
        ex: -2,
        sgn: 0,
    },
    D {
        hi: 0xa8defc2cbe2f8fcc,
        lo: 0xcd12a1f6ab6b095,
        ex: -2,
        sgn: 0,
    },
    D {
        hi: 0xabf80432a65ef190,
        lo: 0x8c95c4c91179176b,
        ex: -2,
        sgn: 0,
    },
    D {
        hi: 0xaf10a22459fe32a6,
        lo: 0x3feef3bb58b1f10d,
        ex: -2,
        sgn: 0,
    },
    D {
        hi: 0xb228d418ec1869ad,
        lo: 0x16031a34d4fc855d,
        ex: -2,
        sgn: 0,
    },
    D {
        hi: 0xb5409827b25591f0,
        lo: 0xcd73fb5d8d45d302,
        ex: -2,
        sgn: 0,
    },
    D {
        hi: 0xb857ec684627fa4c,
        lo: 0x187e26d290714d70,
        ex: -2,
        sgn: 0,
    },
    D {
        hi: 0xbb6ecef285f98a3a,
        lo: 0xbddd8a0365d6b1d3,
        ex: -2,
        sgn: 0,
    },
    D {
        hi: 0xbe853dde9658dc60,
        lo: 0xdfe1b074e22fc666,
        ex: -2,
        sgn: 0,
    },
    D {
        hi: 0xc19b3744e3262dcd,
        lo: 0xad5a41de48f6b26f,
        ex: -2,
        sgn: 0,
    },
    D {
        hi: 0xc4b0b93e20c0213f,
        lo: 0xdab4e426409b23a0,
        ex: -2,
        sgn: 0,
    },
    D {
        hi: 0xc7c5c1e34d3055b2,
        lo: 0x5cc8c00e4fccd850,
        ex: -2,
        sgn: 0,
    },
    D {
        hi: 0xcada4f4db157cf77,
        lo: 0xfa6171200ab2efc3,
        ex: -2,
        sgn: 0,
    },
    D {
        hi: 0xcdee5f96e21b332c,
        lo: 0x65a3132adfb7dfd5,
        ex: -2,
        sgn: 0,
    },
    D {
        hi: 0xd101f0d8c18ed1c1,
        lo: 0xaadb580a1eba209f,
        ex: -2,
        sgn: 0,
    },
    D {
        hi: 0xd415012d802284f0,
        lo: 0xdf4005ef6a64aa02,
        ex: -2,
        sgn: 0,
    },
    D {
        hi: 0xd7278eaf9dcd5b55,
        lo: 0x1779df36d1cc8912,
        ex: -2,
        sgn: 0,
    },
    D {
        hi: 0xda399779eb391377,
        lo: 0xcbabaeb97af8e8aa,
        ex: -2,
        sgn: 0,
    },
    D {
        hi: 0xdd4b19a78aed6515,
        lo: 0xece7f445cecf1e28,
        ex: -2,
        sgn: 0,
    },
    D {
        hi: 0xe05c1353f27b17e5,
        lo: 0xebc61ade6ca83cd,
        ex: -2,
        sgn: 0,
    },
    D {
        hi: 0xe36c829aeba6e720,
        lo: 0x26a0eecdb4f16266,
        ex: -2,
        sgn: 0,
    },
    D {
        hi: 0xe67c659895943123,
        lo: 0x82b0aecadf808123,
        ex: -2,
        sgn: 0,
    },
    D {
        hi: 0xe98bba6965ef725f,
        lo: 0xb91caf23416e7e80,
        ex: -2,
        sgn: 0,
    },
    D {
        hi: 0xec9a7f2a2a188aeb,
        lo: 0x7244ee20f591983b,
        ex: -2,
        sgn: 0,
    },
    D {
        hi: 0xefa8b1f8084ccdfc,
        lo: 0x1050cdf22f34182f,
        ex: -2,
        sgn: 0,
    },
    D {
        hi: 0xf2b650f080d0da8d,
        lo: 0x587f3fa044e2d27d,
        ex: -2,
        sgn: 0,
    },
    D {
        hi: 0xf5c35a316f1a3c80,
        lo: 0x643720de93ba81bd,
        ex: -2,
        sgn: 0,
    },
    D {
        hi: 0xf8cfcbd90af8d57a,
        lo: 0x4221dc4ba772598d,
        ex: -2,
        sgn: 0,
    },
    D {
        hi: 0xfbdba405e9c00cca,
        lo: 0xd24d3023da491920,
        ex: -2,
        sgn: 0,
    },
    D {
        hi: 0xfee6e0d6ff6fc5a4,
        lo: 0x8b74fe2508ab8fc2,
        ex: -2,
        sgn: 0,
    },
    D {
        hi: 0x80f8c035cfee8d76,
        lo: 0xfd958d68e8b49e6b,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0x827dc071bfed6ffa,
        lo: 0xfb4c92369f0cf008,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0x8402702f5b30f2a9,
        lo: 0xcb07b25a7b0372a7,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0x8586ce7ededc809d,
        lo: 0x9d3dc689006896f4,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0x870ada70ba4e6d49,
        lo: 0x9d52755ece3f70,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0x888e93158fb3bb04,
        lo: 0x984156f553344306,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0x8a11f77e349bc245,
        lo: 0xa66d1d936c38c329,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0x8b9506bbb28bb922,
        lo: 0x575f33366be0afef,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0x8d17bfdf47921ac8,
        lo: 0xcb590d74f64e77c9,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0x8e9a21fa66d9ee8d,
        lo: 0xf2be3ecae62789d4,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0x901c2c1eb93dee39,
        lo: 0x632b9cff5cfee724,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0x919ddd5e1ddb8b33,
        lo: 0x609c464b3dd676ec,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0x931f34caaaa5d23a,
        lo: 0x6a1ff8bfe6396e28,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0x94a03176acf82d45,
        lo: 0xae4ba773da6bf754,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0x9620d274aa290339,
        lo: 0xe06a955a5b8e301d,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0x97a116d7601c3515,
        lo: 0xfc8b7184b21f2d50,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0x9920fdb1c5d5783d,
        lo: 0x9dd1eedf18a2e4df,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0x9aa086170c0a8d86,
        lo: 0x9ffa0d23f3c26c62,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0x9c1faf1a9db554af,
        lo: 0xdab6b478577e7be5,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0x9d9e77d020a5bbe6,
        lo: 0xdb895384528d0d60,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0x9f1cdf4b76138b02,
        lo: 0x98dbd3555ebcdefe,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xa09ae4a0bb300a19,
        lo: 0x2f895f44a303cc0b,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xa21886e449b78316,
        lo: 0xd29d23a624acd00c,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xa395c52ab8829dfc,
        lo: 0x2be036401ba87cc2,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xa5129e88dc17976a,
        lo: 0x82d9495ead5be348,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xa68f1213c73b5124,
        lo: 0x17218792857f4c5a,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xa80b1ee0cb823c27,
        lo: 0x3269f4702b88324a,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xa986c40579e11c0a,
        lo: 0x8e3bdf8085321556,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xab020097a33da341,
        lo: 0xc1654b64a0081b46,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xac7cd3ad58fee7f0,
        lo: 0x811f953984eff83e,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xadf73c5ced9db0f3,
        lo: 0x9a5318ac6fe94e4d,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xaf7139bcf5349ac6,
        lo: 0x9fe5f4ea48965e2c,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xb0eacae4461013ed,
        lo: 0x63c66682bae74898,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xb263eee9f93e3088,
        lo: 0x695a5332090bb09b,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xb3dca4e56b1e54bb,
        lo: 0x992d96e5021e3c37,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xb554ebee3bf0b58e,
        lo: 0x971f4da709ad4378,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xb6ccc31c5065afee,
        lo: 0x35ebacd79f209137,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xb8442987d22cf576,
        lo: 0x9cc3ef36746de3b8,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xb9bb1e4930848ead,
        lo: 0xcdb0531c4e58484b,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xbb31a07920c7b256,
        lo: 0x55b92083658bb897,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xbca7af309efd7182,
        lo: 0xa4b0d21fc5036a5,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xbe1d4988ee67380c,
        lo: 0xd1f90f79f46c7e01,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xbf926e9b9a0f2127,
        lo: 0x91a1b5eb79658c67,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xc1071d8275561f9b,
        lo: 0x721853f8e528a934,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xc27b55579c81f96d,
        lo: 0xcdc2bd470675104d,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xc3ef1535754b168d,
        lo: 0x3122c2a59efddc37,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xc5625c36af6a222f,
        lo: 0xf4ff2895ab6ebe89,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xc6d5297645257e8d,
        lo: 0x14d24739de27e2e9,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xc8477c0f7bde8a98,
        lo: 0x4ce0246ad4fa74,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xc9b9531de49eb968,
        lo: 0x4319e5ad5b0dcb84,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xcb2aadbd5ca47af5,
        lo: 0xfaa3dfe675a65ee2,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xcc9b8b0a0deff5d4,
        lo: 0x2e663b3c7555a6c3,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xce0bea206fcf9192,
        lo: 0x3c540a9eec47af38,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xcf7bca1d476c516d,
        lo: 0xa81290bdbaad62e4,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xd0eb2a1da855fefd,
        lo: 0xb9302788604e88f1,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xd25a093ef50f2482,
        lo: 0x721fc87ba1d42456,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xd3c8669edf98d680,
        lo: 0x87967926fdcecec4,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xd536415b69fe4c54,
        lo: 0x1df22346611c6b4b,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xd6a39892e6e04764,
        lo: 0x3090d44db12c418c,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xd8106b63fa0048a0,
        lo: 0xa573f2aa90434ba5,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xd97cb8ed98cb93f5,
        lo: 0x2e349483e3fb2a6a,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xdae8804f0ae6015b,
        lo: 0x362cb974182e3030,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xdc53c0a7eab49b35,
        lo: 0x3ccca3982328ed8b,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xddbe791825e8099e,
        lo: 0x1a5bd9269d408d7e,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xdf28a8bffe06ca56,
        lo: 0xcce2634be2bf54df,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xe0924ec008f734fd,
        lo: 0x8aa895d5bf3e84ea,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xe1fb6a3931894b38,
        lo: 0xf7a1f9bd9ba13b6b,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xe363fa4cb8005482,
        lo: 0x7b32c72e31824e51,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xe4cbfe1c329c453a,
        lo: 0xd40e9e6b989f89e5,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xe63374c98e22f0b4,
        lo: 0x2872ce1bfc7ad1cd,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xe79a5d770e6905dc,
        lo: 0xf1b65cc5fd780262,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xe900b7474edad637,
        lo: 0x431626c10485bdda,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xea66815d4304e6c8,
        lo: 0xcc39cfcc29960b1,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xebcbbadc371c4aaa,
        lo: 0x1d90f780ae951140,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xed3062e7d086c6f0,
        lo: 0xc71debc372b6f9d4,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xee9478a40e62bf86,
        lo: 0x2a24164daec85ccb,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xeff7fb354a0eecb1,
        lo: 0x527233b40d3432bb,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xf15ae9c037b1d8f0,
        lo: 0x6c48e9e3420b0f1e,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xf2bd4369e6c126d3,
        lo: 0x7f232aee178c6323,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xf41f0757c2889e84,
        lo: 0x3c7f10db458c337c,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xf58034af92b102a7,
        lo: 0x93fa6107c4327527,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xf6e0ca977bc6ac45,
        lo: 0xe1079824233fef46,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xf840c835ffbfed66,
        lo: 0xa9a56012067c570c,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xf9a02cb1fe833a0d,
        lo: 0x8da894471de1a18,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xfafef732b66d1742,
        lo: 0x343fbf4a7d42af3,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xfc5d26dfc4d5cfda,
        lo: 0x27c07c911290b8d1,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xfdbabae12696eea4,
        lo: 0x2377c3799c052fa,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0xff17b25f38907dad,
        lo: 0xa9c6ba50490539f,
        ex: -1,
        sgn: 0,
    },
    D {
        hi: 0x803a06415c170525,
        lo: 0x6f53873e2f1477ff,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0x80e7e43a61f5b6cb,
        lo: 0x5ca183dc973abc22,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0x819572af6decac84,
        lo: 0x9fba97fdf0c4d24c,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0x8242b1357110d372,
        lo: 0x6fb2123fedfa6e22,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0x82ef9f618dc5b70e,
        lo: 0x91a965931f1a200a,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0x839c3cc917ff6cb4,
        lo: 0xbfd79717f2880abf,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0x8448890195846099,
        lo: 0x246efcff30cb064a,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0x84f483a0be2f0403,
        lo: 0x51917cac857fd5f5,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0x85a02c3c7c2f5ca5,
        lo: 0x327888fe4b62687b,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0x864b826aec4c74e5,
        lo: 0x85043222c9bdd18d,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0x86f685c25e25acf5,
        lo: 0x7e0b9b07548471a2,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0x87a135d95473ec89,
        lo: 0x4e091160e2430712,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0x884b9246854ab50b,
        lo: 0x4f14c8afe4560291,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0x88f59aa0da591421,
        lo: 0xb892ca8361d8c84c,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0x899f4e7f712a765e,
        lo: 0xc88302a31afce54a,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0x8a48ad799b6759f3,
        lo: 0x660558a02136130a,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0x8af1b726df15e13c,
        lo: 0x545f7d79ead8fa19,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0x8b9a6b1ef6da4502,
        lo: 0x21a6675f51580bc4,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0x8c42c8f9d2372644,
        lo: 0x101a5adbcb9ffb43,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0x8cead04f95cdbf66,
        lo: 0x4d49cbaf15aecd80,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0x8d9280b89b9df49b,
        lo: 0xde2d43c6b67a7cbe,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0x8e39d9cd73464364,
        lo: 0xbba4cfecbff54867,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0x8ee0db26e24390f8,
        lo: 0xaf0e2345f3bd24b4,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0x8f87845de430d777,
        lo: 0x9311a82459aa0f72,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0x902dd50bab06b1b7,
        lo: 0xb144016c7a30b39a,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0x90d3ccc99f5ac58b,
        lo: 0x9d1072e09b72292,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0x91796b31609f0c54,
        lo: 0x6714fe6925b78cc4,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0x921eafdcc560f9c5,
        lo: 0x33d0a284a8c954ad,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0x92c39a65db88809d,
        lo: 0x1f8481e704e4a767,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0x93682a66e896f544,
        lo: 0xb17821911e71c16e,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0x940c5f7a69e5ce1c,
        lo: 0x1489a97671a42,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0x94b0393b14e54156,
        lo: 0xd6c7af02d5c16fd9,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0x9553b743d75ac03f,
        lo: 0xac0106650f4ef023,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0x95f6d92fd79f4fba,
        lo: 0xd9f8e1a446e973b9,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0x96999e9a74ddbde3,
        lo: 0xa7a7556c3b33abc1,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0x973c071f4750b49c,
        lo: 0xc0a03934f0cce19b,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0x97de125a2080a8ed,
        lo: 0xd243aa0843a2c144,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0x987fbfe70b81a708,
        lo: 0x19cec845ac87a5c6,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0x99210f624d30facb,
        lo: 0xc4b992a37fb9b9bd,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0x99c200686472b4a8,
        lo: 0x1ab42d43235757b6,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0x9a6292960a6f0ab0,
        lo: 0x7e92c655656e6b85,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0x9b02c58832cf95c0,
        lo: 0x698b94f50326a043,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0x9ba298dc0bfc6a88,
        lo: 0x9a5614e8ffbeac6f,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0x9c420c2eff590e5f,
        lo: 0xc7fd954194e6d8aa,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0x9ce11f1eb18147b1,
        lo: 0x3e93627de8fd5779,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0x9d7fd1490285c9e3,
        lo: 0xe25e39549638ae68,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0x9e1e224c0e28bc94,
        lo: 0x2cad377d5c9c35d8,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0x9ebc11c62c1a1dfb,
        lo: 0xcc141e10c6460c8b,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0x9f599f55f0340061,
        lo: 0xa88d5f46834bbf8d,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0x9ff6ca9a2ab6a26d,
        lo: 0x22cc118a0c118aa0,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xa0939331e8846237,
        lo: 0x7cec6df5bea167cf,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xa12ff8bc735d8af6,
        lo: 0x71acea2819360c35,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xa1cbfad9521bfd1b,
        lo: 0x166c36e7bb3c402f,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xa267992848eeb0c0,
        lo: 0x3b5167ee359a234e,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xa302d34959951243,
        lo: 0x9443372e20d4377c,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xa39da8dcc39a38e5,
        lo: 0xca9a8a720d4c69c,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xa4381983048ff747,
        lo: 0xbf623cf5301a2dde,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xa4d224dcd849c5b0,
        lo: 0x23d251cc8d7975cc,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xa56bca8b391785db,
        lo: 0x189d39ffe11aaa2b,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xa6050a2f60002049,
        lo: 0x8c33ebf3aa8501fb,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xa69de36ac4fbfadc,
        lo: 0x9b3ad6e4022183d9,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xa73655df1f2f489e,
        lo: 0x149f6e75993468a3,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xa7ce612e65243291,
        lo: 0x6b2a39f856a69781,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xa86604facd04d969,
        lo: 0x3463a2c2e6e9cc55,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xa8fd40e6ccd52ffd,
        lo: 0x6cc14c4f53e2e82d,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xa99414951aacae5e,
        lo: 0xd147625fda929af8,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xaa2a7fa8acefdd63,
        lo: 0xb714ee81b53b4b9d,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xaac081c4ba89ba8a,
        lo: 0xe1b3dfc4dbda9bfd,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xab561a8cbb24f410,
        lo: 0xf17cee69b0d2ecde,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xabeb49a46764fd15,
        lo: 0x1becda8089c1a94c,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xac800eafb91ef9a9,
        lo: 0xf86ba0dde982fb59,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xad146952eb9282af,
        lo: 0x44bf16268608db96,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xada859327ba24151,
        lo: 0x9d30d4cfeb04f1fb,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xae3bddf3280c620d,
        lo: 0x3d53817865422565,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xaecef739f1a2df10,
        lo: 0xf74d099042e8f326,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xaf61a4ac1b83a1de,
        lo: 0xa89a9b8f726b95bf,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xaff3e5ef2b507c06,
        lo: 0x8c679e67fc462d51,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xb085baa8e966f6da,
        lo: 0xe4cad00d5c94bcd2,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xb117227f6117f9f9,
        lo: 0x8d8be132d576e614,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xb1a81d18e0df4889,
        lo: 0x24784f32c3e3e5bd,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xb238aa1bfa9ad507,
        lo: 0x8cc7d4bd05ffd5ae,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xb2c8c92f83c1eb87,
        lo: 0xac9f7ebbc469ef59,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xb35879fa959c323c,
        lo: 0x5d6635109164f740,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xb3e7bc248d78802e,
        lo: 0xa156468ef6c18c60,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xb4768f550ce389fd,
        lo: 0x4a85350f69018c55,
        ex: 0,
        sgn: 0,
    },
];
const C: [D; 256] = [
    D {
        hi: 0x8000000000000000,
        lo: 0x0,
        ex: 1,
        sgn: 0,
    },
    D {
        hi: 0xffffb10b10e80e95,
        lo: 0x3031437d7eccb9df,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xfffec42c7454926b,
        lo: 0x38e310779edfec68,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xfffd3964bc6275ba,
        lo: 0x69fff9ae0dedb047,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xfffb10b4dc96dabb,
        lo: 0xb47903f7a19f8ee2,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xfff84a1e29de8571,
        lo: 0x8cc193c5d508e13f,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xfff4e5a25a8d095b,
        lo: 0x43366df666fd54ff,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xfff0e343865bbb13,
        lo: 0x5428ed0647c9e5d1,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xffec4304266865d9,
        lo: 0x5657552366961732,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xffe704e71533c508,
        lo: 0x53aa9423bb0adc21,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xffe128ef8e9fc17a,
        lo: 0x7d209f32d42d864e,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xffdaaf212fed72db,
        lo: 0x4fd8f038449ec436,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xffd3977ff7bae4e9,
        lo: 0x664649b4d541b9c5,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xffcbe2104600a0a9,
        lo: 0x5595ca3f421ae09c,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xffc38ed6dc0ef98b,
        lo: 0x1c676208aa3be545,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xffba9dd8dc8b1e83,
        lo: 0xccfed60a91097c48,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xffb10f1bcb6bef1d,
        lo: 0x421e8edaaf59453e,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xffa6e2a58df6947d,
        lo: 0xd2c665c2da3e7844,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xff9c187c6abade6a,
        lo: 0x1e1862cca089938b,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xff90b0a7098f6443,
        lo: 0x2dabd3195a05710f,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xff84ab2c738d6a03,
        lo: 0x519c314973ccae6b,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xff780814130c893c,
        lo: 0x3ea4f30adda3016f,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xff6ac765b39e1e19,
        lo: 0x1b9d5851979f28fb,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xff5ce92982087867,
        lo: 0x50a7bb6a6ee3b0f1,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xff4e6d680c41d0a9,
        lo: 0xf668633f1ab858a,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xff3f542a416b0134,
        lo: 0xb085c1828f69296a,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xff2f9d7971ca0364,
        lo: 0x27e31939e2eec09c,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xff1f495f4ec430d7,
        lo: 0xf5971326a3540ea9,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xff0e57e5ead848d1,
        lo: 0x1f1901544271c3f8,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xfefcc917b99839a5,
        lo: 0xe0abd3a9b64df725,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xfeea9cff8fa2ae54,
        lo: 0xec34413e87ef2740,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xfed7d3a8a29c603b,
        lo: 0x2f88b949a72ff96c,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xfec46d1e89292cf0,
        lo: 0x41390efdc726e9ef,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xfeb0696d3ae4f04d,
        lo: 0xb7b6cc53c3abc817,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xfe9bc8a1105c22a5,
        lo: 0xd3af6ee4f2101c20,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xfe868ac6c3043b2e,
        lo: 0xb4f70c910505e10,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xfe70afeb6d33d6a2,
        lo: 0x2907cf2b3f6feac2,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xfe5a381c8a1aa224,
        lo: 0xd54faa364b7da8f6,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xfe432367f5b90a62,
        lo: 0x87b8875373a818a4,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xfe2b71dbecd7aefc,
        lo: 0x8598c2c429caf7,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xfe1323870cfe9a3d,
        lo: 0x90cd1d959db674ef,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xfdfa3878546c3d28,
        lo: 0x9bfe5c51e91cbdcd,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xfde0b0bf220c2fd4,
        lo: 0xe276d247626a23fd,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xfdc68c6b356db62f,
        lo: 0x499ddb331d19539d,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xfdabcb8caeba091b,
        lo: 0xfac7397cc07a6470,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xfd906e340eaa6401,
        lo: 0xd6e270740a186977,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xfd747472367dd6c5,
        lo: 0x61beb8cd2696fc78,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xfd57de5867eedc39,
        lo: 0x6c696582f346fd91,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xfd3aabf84528b50b,
        lo: 0xeae6bd951c1dabbe,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xfd1cdd63d0bc8735,
        lo: 0x863b87258f11ad7e,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xfcfe72ad6d9641f2,
        lo: 0xa06fab9f9d106709,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xfcdf6be7def1464c,
        lo: 0xa4e064308f4999f4,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xfcbfc926484cd43a,
        lo: 0xa3e22b4d38917e73,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xfc9f8a7c2d603c60,
        lo: 0x5d582cac7cb4391c,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xfc7eaffd720ed673,
        lo: 0x2880268f2e62955,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xfc5d39be5a5bbc4b,
        lo: 0x1c0d254b6c8da4bd,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xfc3b27d38a5d49ab,
        lo: 0x256778ffcb5c1769,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xfc187a52063060c2,
        lo: 0x9433b49289417ea2,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xfbf5314f31eb7375,
        lo: 0x25aafd7fdba12c5f,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xfbd14ce0d191516e,
        lo: 0x7190c94899dff1b8,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xfbaccd1d0903bb09,
        lo: 0xe63ae8632b84473c,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xfb87b21a5bf5b917,
        lo: 0x75df66f0ec3dd459,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xfb61fbefadddb985,
        lo: 0x61ce9d5ef5a81487,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xfb3baab441e770f7,
        lo: 0xb4b54683879c9c17,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xfb14be7fbae58156,
        lo: 0x2172a361fd2a722f,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xfaed376a1b42e559,
        lo: 0x2079880c450348ac,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xfac5158bc4f4211f,
        lo: 0x4a188aa367f90ab1,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xfa9c58fd796837d4,
        lo: 0x10655ecd5cc771d8,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xfa7301d859796671,
        lo: 0x1fe196a53fb5b237,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xfa491035e55da3a3,
        lo: 0xd24377c77a591e24,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xfa1e842ffc96e4e0,
        lo: 0x431c393c7f62da65,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xf9f35de0dde328ab,
        lo: 0xba5dbf4510eddc8f,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xf9c79d63272c4628,
        lo: 0x4504ae08d19b2980,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xf99b42d1d57781eb,
        lo: 0x78685d850f80ecdc,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xf96e4e4844d4e82a,
        lo: 0x80e8c17bf80e8f02,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xf940bfe2304e6c45,
        lo: 0xc0e2a1352ed7f292,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xf91297bbb1d6cdbe,
        lo: 0x68fc6e4d6a920bd2,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xf8e3d5f1423842a0,
        lo: 0x9701914c7f8fbcd7,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xf8b47a9fb902e76c,
        lo: 0xac9f07f54ff5bc14,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xf88485e44c7af48a,
        lo: 0xb36a9dfaadafc1e1,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xf853f7dc9186b952,
        lo: 0xc7adc6b4988891bb,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xf822d0a67b9c5cb5,
        lo: 0xa776175bd284fe05,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xf7f110605caf6390,
        lo: 0xa76f7efc19aed41c,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xf7beb728e51dfcb8,
        lo: 0x730785813f78aa1e,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xf78bc51f239e12c6,
        lo: 0x214cffcee9dd33ca,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xf7583a62852a23b2,
        lo: 0x4becad887680c197,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xf7241712d4edde49,
        lo: 0xf99107e50d631330,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xf6ef5b503c328589,
        lo: 0x50ca117eb18beed7,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xf6ba073b424b19e8,
        lo: 0x2c791f59cc1ffc23,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xf6841af4cc8048a4,
        lo: 0xce8c455197cdf8a7,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xf64d969e1dfc2119,
        lo: 0x119d358de0493956,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xf6167a58d7b59026,
        lo: 0x9dc7e5954c5a8f24,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xf5dec646f85ba1c6,
        lo: 0xc8c615e72768d6b5,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xf5a67a8adc4088ca,
        lo: 0xed0dd4bf62edd13f,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xf56d97473d446cda,
        lo: 0x275a2bbb2bab6c8a,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xf5341c9f32bffeb9,
        lo: 0x8da64484aaa0febc,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xf4fa0ab6316ed2ec,
        lo: 0x163c5c7f03b718c5,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xf4bf61b00b5982b7,
        lo: 0x890ac4aafa6a37bf,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xf48421b0efbf939b,
        lo: 0xf8f9d3b87d11fd52,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xf4484add6b01254b,
        lo: 0x667e06866c07c369,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xf40bdd5a6688662f,
        lo: 0x5019794a1f5896e5,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xf3ced94d28b2ce8a,
        lo: 0x18ef535a7ffa7a3d,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xf3913edb54ba2242,
        lo: 0x50f29b4b49f31c37,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xf3530e2aea9d3966,
        lo: 0xd981acdcf6bc3e4,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xf314476247088f74,
        lo: 0xa5486bdc455d56a2,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xf2d4eaa8233e997d,
        lo: 0x431be53f92ece9e6,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xf294f82394ffe320,
        lo: 0xebadcdbf915e8f6c,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xf2546ffc0e72f286,
        lo: 0xaf0eed81e8c51e55,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xf21352595e0bf350,
        lo: 0xe7112e89103cc0c7,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xf1d19f63ae7428a2,
        lo: 0x844e6a35ddc2b713,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xf18f574386712643,
        lo: 0x8f6bac72988088b0,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xf14c7a21c8cbd0f4,
        lo: 0x2730081c758fb42b,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xf1090827b43725fd,
        lo: 0x67127db35b287316,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xf0c5017ee336ca0f,
        lo: 0xc4e557b119ef3185,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xf08066514c055f7e,
        lo: 0x973ea9903ed5125f,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xf03b36c9407aa3e8,
        lo: 0x992d39ec5c561d28,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xeff573116df1555d,
        lo: 0x62aef7b55319d1d4,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xefaf1b54dd2cdf0f,
        lo: 0xf03a18a5e16ab641,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xef682fbef23ecda6,
        lo: 0x767c0e8ad33bc085,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xef20b07b6c6c0b37,
        lo: 0xe2398bf0eeb28cde,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xeed89db66611e307,
        lo: 0x86f8c20fb664b01b,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xee8ff79c548acd0f,
        lo: 0xa1d2c3d018a9279f,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xee46be5a0813016b,
        lo: 0x7872773830d368be,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xedfcf21cabacd3b1,
        lo: 0xfee6a1eebfa13b4a,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xedb29311c504d652,
        lo: 0x11815196b9fbf5df,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xed67a1673455c601,
        lo: 0x7289102076a125e5,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xed1c1d4b344c3d4f,
        lo: 0xddffe98c4f8aa031,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xecd006ec59ea306f,
        lo: 0xa8392eb238578ab0,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xec835e79946a3145,
        lo: 0x7e610231ac1d6181,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xec3624222d227bd1,
        lo: 0x278047ae3dd0889,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xebe85815c767cb00,
        lo: 0x1e99ccb9adc62ca6,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xeb99fa84606ff5ff,
        lo: 0xdae311e656e0661,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xeb4b0b9e4f345617,
        lo: 0x39e39c6c2ab3655d,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xeafb8b944453f52f,
        lo: 0x3383bbb5156bf1d7,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xeaab7a9749f584fe,
        lo: 0x24db98ad3a0647a1,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xea5ad8d8c3a91f05,
        lo: 0x4a0ca5ea449b1c83,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xea09a68a6e49cd62,
        lo: 0x15ad45b4a1b5e823,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xe9b7e3de5fdedc8b,
        lo: 0xcd24d4bd1056c826,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xe9659107077cf60f,
        lo: 0x89a92b199adfbafa,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xe912ae372d27045d,
        lo: 0xacb1c26a06e5ae02,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xe8bf3ba1f1aedfbb,
        lo: 0xf8972affb3d98e1f,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xe86b397ace95c46f,
        lo: 0x9fec1e78c4376186,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xe816a7f595ec9232,
        lo: 0xbfe8378abfb87b6f,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xe7c187467233d508,
        lo: 0xdbfb0fe56c6f80fe,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xe76bd7a1e63b9786,
        lo: 0x125129529d48a92f,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xe715993ccd02fe9c,
        lo: 0xe2ba81b9ce96e02e,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xe6becc4c5997af06,
        lo: 0x82fcedb4c6434d76,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xe667710616f4fc59,
        lo: 0xdd2a3e32c3859960,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xe60f879fe7e2e1e5,
        lo: 0x7613b68f6ab03130,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xe5b7105006d4c560,
        lo: 0x9b695cd67c93bd79,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xe55e0b4d05c80388,
        lo: 0x5a7c210a3a15e7ea,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xe50478cdce2246bc,
        lo: 0xe1f5a58c80292554,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xe4aa5909a08fa7b4,
        lo: 0x122785ae67f5515d,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xe44fac3814e09856,
        lo: 0x20d63b5b9e3cd6ac,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xe3f4729119e798d9,
        lo: 0x56992551ae074e99,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xe398ac4cf556b732,
        lo: 0xd1197dc12c63176,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xe33c59a4439cd8ec,
        lo: 0x36563e2ffad8351a,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xe2df7acff7c2cf83,
        lo: 0xd6fe4dd22e60a4a2,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xe28210095b483751,
        lo: 0xfd39138aa2d508ed,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xe224198a0e002123,
        lo: 0xe0521df01a1be6f5,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xe1c5978c05ed8691,
        lo: 0xf4e8a8372f8c5810,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xe1668a498f1f892c,
        lo: 0xe2f9d4600f4d0325,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xe106f1fd4b8d7c96,
        lo: 0x6ba8a9d9ba877899,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xe0a6cee232f2bb9c,
        lo: 0x6d6c98fe79817946,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xe046213392aa486c,
        lo: 0x55ff6038a5197367,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xdfe4e92d0d8a37f5,
        lo: 0x720588ff6547d884,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xdf83270a9bbee890,
        lo: 0xab01350f013d78dd,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xdf20db088aa60404,
        lo: 0x64a58b2f103485dd,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xdebe05637ca94cfb,
        lo: 0x4b19aa71fec3ae6d,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xde5aa65869193805,
        lo: 0x4248f15548f69ca,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xddf6be249c075037,
        lo: 0xd597b10a01676659,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xdd924d05b620678a,
        lo: 0x739c45b982193b5e,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xdd2d5339ac8692fd,
        lo: 0x49c6e0ea76cbcaac,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xdcc7d0fec8aaf2aa,
        lo: 0xb2069fd0b482b4e8,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xdc61c693a82745d5,
        lo: 0xaca8017e375b64e5,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xdbfb34373c974b0e,
        lo: 0xccb7fd40d543f4a1,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xdb941a28cb71ec87,
        lo: 0x2c19b63253da43fc,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xdb2c78a7ede238a9,
        lo: 0x5a98479cbef2ecbc,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xdac44ff490a02710,
        lo: 0x5b267c1bcff0ab62,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xda5ba04ef3c929f4,
        lo: 0xe257bde73d83dc1a,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xd9f269f7aab88c29,
        lo: 0x28e81dcb6dab91ac,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xd988ad2f9bdf9bbb,
        lo: 0xc4e4dc69fc2fff6f,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xd91e6a38009da15a,
        lo: 0x1bb35ad6d2e74b67,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xd8b3a1526517a48b,
        lo: 0x1ed1a8ff78f1b632,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xd84852c0a80ffcdb,
        lo: 0x24b9fe00663574a4,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xd7dc7ec4fabdb011,
        lo: 0xced12d2899b803db,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xd77025a1e0a39d8b,
        lo: 0xcb78e80e67ba1b8,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xd703479a2f6776cc,
        lo: 0x6cb3bfd65b38562b,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xd695e4f10ea88570,
        lo: 0x83f082b570611d7,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xd627fde9f7d63e7e,
        lo: 0x7afbefc05e9f7d99,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xd5b992c8b606a351,
        lo: 0x7190b755535d4f18,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xd54aa3d165cc7018,
        lo: 0x7d00ae97abaa4096,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xd4db3148750d1819,
        lo: 0xf630e8b6dac83e69,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xd46b3b72a2d68fc9,
        lo: 0xdc4663a3168698d2,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xd3fac294ff34e4d0,
        lo: 0xb77d4f6bd0ee8591,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xd389c6f4eb07a41c,
        lo: 0xa8faac741a6394dc,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xd31848d817d70e16,
        lo: 0xeeeaddb72f00e0dd,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xd2a6488487a91918,
        lo: 0x4300fd1c1ce507e5,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xd233c6408cd64236,
        lo: 0x981ba7e42537275f,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xd1c0c252c9de2c86,
        lo: 0xda7485a5aeffeb4c,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xd14d3d02313c0eed,
        lo: 0x744fea20e8abef92,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xd0d93696053af098,
        lo: 0x77a18eb13d2ecde5,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xd064af55d7c9b43e,
        lo: 0x6b8a685f6cb61c21,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xcfefa7898a4ef23c,
        lo: 0xdaf200dd81212d10,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xcf7a1f794d7ca1b1,
        lo: 0xdfcb60445c1bf973,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xcf04176da12390ac,
        lo: 0x4d27090f10c454e,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xce8d8faf5406ab8b,
        lo: 0xf5babff66def7892,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xce16888783ae13b3,
        lo: 0x93e391861a034684,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xcd9f023f9c3a059e,
        lo: 0x23af31db7179a4aa,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xcd26fd2158358e7d,
        lo: 0x649474e36b8db9d3,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xccae7976c0691177,
        lo: 0x83e907fbd7aaf0b0,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xcc35778a2bac9ca1,
        lo: 0xf839ce18e08bfb50,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xcbbbf7a63eba0dd5,
        lo: 0x70cbb7f3343451be,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xcb41fa15ebff0777,
        lo: 0x2293661be51140ab,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xcac77f24736eb553,
        lo: 0xd9944be1631846d8,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xca4c871d625361a9,
        lo: 0x5328edeb3e6784de,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xc9d1124c931fda7a,
        lo: 0x8335241be1693225,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xc95520fe2d40a74b,
        lo: 0x83b0e96e1249c2b0,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xc8d8b37ea4ed0f62,
        lo: 0xb562c00b34ee771,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xc85bca1abaf7f0a7,
        lo: 0x65862939b83382e0,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xc7de651f7ca06749,
        lo: 0x2b31bc86877fd2c,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xc76084da43624634,
        lo: 0xd5c149509e9059f1,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xc6e22998b4c6608e,
        lo: 0xcfe6c1b1a6b4e2a4,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xc66353a8c232a43c,
        lo: 0xe993503baf5afb41,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xc5e40358a8ba05a7,
        lo: 0x43da25d99267326b,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xc56438f6f0ec3cca,
        lo: 0xab4906075507e74,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xc4e3f4d26ea553b6,
        lo: 0xdd40950cf1ed92fa,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xc463373a40dd06a3,
        lo: 0x9dd768f30ca8e85c,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xc3e2007dd175f5a4,
        lo: 0xa87e78136665cdb2,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xc36050ecd50ca830,
        lo: 0x8ac9e1386e4cbabb,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xc2de28d74ac6628b,
        lo: 0x74c8f010d986a9e0,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xc25b888d7c1fcd38,
        lo: 0xb7041e9bc8c18b0d,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xc1d8705ffcbb6e90,
        lo: 0xbdf0715cb8b20bd7,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xc154e09faa2ff69a,
        lo: 0x17858573216e0a22,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xc0d0d99dabd65d44,
        lo: 0x2bda5328933c854a,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xc04c5bab7297d322,
        lo: 0x6dd06968e0ed1957,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xbfc7671ab8bb84c6,
        lo: 0xe4e62d86dd136e78,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xbf41fc3d81b430db,
        lo: 0xd46655d6b012455,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xbebc1b6619ed9116,
        lo: 0x2715ef03f8543355,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xbe35c4e716999630,
        lo: 0x29d7f7b67d43b177,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xbdaef913557d76f0,
        lo: 0xac85320f528d6d5d,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xbd27b83dfcbe9279,
        lo: 0x2ea36923d5d8e213,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xbca002ba7aaf25ea,
        lo: 0x4a48496734be336d,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xbc17d8dc859ad583,
        lo: 0x727c405ffc73af56,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xbb8f3af81b93095c,
        lo: 0xfce8d84068e825b6,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xbb062961823b1ddc,
        lo: 0x5120e35e1c1a250c,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xba7ca46d46946802,
        lo: 0x33201477347447d8,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xb9f2ac703cca0db3,
        lo: 0x39db32d014440024,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xb96841bf7ffcb21a,
        lo: 0x9de1e3b22b8bf4db,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xb8dd64b0720df647,
        lo: 0xa726f4f0828585c9,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xb8521598bb6bce26,
        lo: 0x1c041d1ea5fb3fdb,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xb7c654ce4adba9f2,
        lo: 0x2e7a35723f3ed035,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xb73a22a755457448,
        lo: 0x7f86f63bb23f496a,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xb6ad7f7a557e64f2,
        lo: 0xeb2d28ef943dc88c,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xb6206b9e0c13a892,
        lo: 0xea7c015f12b987f7,
        ex: 0,
        sgn: 0,
    },
    D {
        hi: 0xb592e7697f14dd4a,
        lo: 0x737dd2824b608d13,
        ex: 0,
        sgn: 0,
    },
];
const PS: [D; 6] = [
    D {
        hi: 0xc90fdaa22168c234,
        lo: 0xc4c6628b80dc1cd1,
        ex: 3,
        sgn: 0,
    },
    D {
        hi: 0xa55de7312df295f5,
        lo: 0x5dc72f712aa57db4,
        ex: 6,
        sgn: 1,
    },
    D {
        hi: 0xa335e33bad570e92,
        lo: 0x3f33be0021aa54d2,
        ex: 7,
        sgn: 0,
    },
    D {
        hi: 0x9969667315ec2d9d,
        lo: 0xe59d6ab8509a2025,
        ex: 7,
        sgn: 1,
    },
    D {
        hi: 0xa83c1a43bf1c6485,
        lo: 0x7d5f8f76fa7d74ed,
        ex: 6,
        sgn: 0,
    },
    D {
        hi: 0xf16ab2898eae62f9,
        lo: 0xa7f0339113b8b3c5,
        ex: 4,
        sgn: 1,
    },
];
const PC: [D; 6] = [
    D {
        hi: 0x8000000000000000,
        lo: 0x0,
        ex: 1,
        sgn: 0,
    },
    D {
        hi: 0x9de9e64df22ef2d2,
        lo: 0x56e26cd9808c1949,
        ex: 5,
        sgn: 1,
    },
    D {
        hi: 0x81e0f840dad61d9a,
        lo: 0x9980f00630cb655e,
        ex: 7,
        sgn: 0,
    },
    D {
        hi: 0xaae9e3f1e5ffcfe2,
        lo: 0xa508509534006249,
        ex: 7,
        sgn: 1,
    },
    D {
        hi: 0xf0fa83448dd1e094,
        lo: 0xe0603ce7044eeba,
        ex: 6,
        sgn: 0,
    },
    D {
        hi: 0xd368f6f4207cfe49,
        lo: 0xec63157807ebffa,
        ex: 5,
        sgn: 1,
    },
];
const PSFAST: [f64; 5] = [
    f64::from_bits(0x401921fb54442d18),
    f64::from_bits(0x3cb1a62645446203),
    f64::from_bits(0xc044abbce625be53),
    f64::from_bits(0x405466bc678d8d63),
    f64::from_bits(0xc05331554ca19669),
];
const PCFAST: [f64; 5] = [
    f64::from_bits(0x3ff0000000000000),
    f64::from_bits(0xbb2923015c000000),
    f64::from_bits(0xc033bd3cc9be45de),
    f64::from_bits(0x40503c1f080ad892),
    f64::from_bits(0xc0555a5c590f9e6a),
];
const SC: [[f64; 3]; 256] = [
    [
        f64::from_bits(0x0000000000000000),
        f64::from_bits(0x0000000000000000),
        f64::from_bits(0x3ff0000000000000),
    ],
    [
        f64::from_bits(0xbdcc0f6c00000000),
        f64::from_bits(0x3f6921f892b900fe),
        f64::from_bits(0x3feffff621623fa0),
    ],
    [
        f64::from_bits(0xbdc9c7935e000000),
        f64::from_bits(0x3f7921f0ea27ce01),
        f64::from_bits(0x3fefffd8858eca2e),
    ],
    [
        f64::from_bits(0xbddd14d1ac000000),
        f64::from_bits(0x3f82d96af779b0bb),
        f64::from_bits(0x3fefffa72c986392),
    ],
    [
        f64::from_bits(0xbdedba8f6a800000),
        f64::from_bits(0x3f8921d1ce2d0a1c),
        f64::from_bits(0x3fefff62169dddaa),
    ],
    [
        f64::from_bits(0x3dfa6b7cdf000000),
        f64::from_bits(0x3f8f6a29bdb73770),
        f64::from_bits(0x3fefff0943c02419),
    ],
    [
        f64::from_bits(0x3deb49618d000000),
        f64::from_bits(0x3f92d936d1506f3d),
        f64::from_bits(0x3feffe9cb44829c0),
    ],
    [
        f64::from_bits(0xbdc398d6fc000000),
        f64::from_bits(0x3f95fd4d1e21de6d),
        f64::from_bits(0x3feffe1c687174b1),
    ],
    [
        f64::from_bits(0xbe0e9e9a8c800000),
        f64::from_bits(0x3f99215597791e0a),
        f64::from_bits(0x3feffd886097afcf),
    ],
    [
        f64::from_bits(0xbdf34e844c000000),
        f64::from_bits(0x3f9c454f2e9480c7),
        f64::from_bits(0x3feffce09ce95933),
    ],
    [
        f64::from_bits(0xbdf989a8a4000000),
        f64::from_bits(0x3f9f693709b94f92),
        f64::from_bits(0x3feffc251dfbac0c),
    ],
    [
        f64::from_bits(0x3e104a9b99000000),
        f64::from_bits(0x3fa146860e69a571),
        f64::from_bits(0x3feffb55e40a5c43),
    ],
    [
        f64::from_bits(0xbdb56947c0000000),
        f64::from_bits(0x3fa2d865748774ad),
        f64::from_bits(0x3feffa72efff95d1),
    ],
    [
        f64::from_bits(0xbdcc348768000000),
        f64::from_bits(0x3fa46a396d34121a),
        f64::from_bits(0x3feff97c420a8451),
    ],
    [
        f64::from_bits(0x3df9e80552000000),
        f64::from_bits(0x3fa5fc00e6e4c65c),
        f64::from_bits(0x3feff871dacd8761),
    ],
    [
        f64::from_bits(0x3dd3f11d74000000),
        f64::from_bits(0x3fa78dbaa97099eb),
        f64::from_bits(0x3feff753bb18af95),
    ],
    [
        f64::from_bits(0x3dec039af4000000),
        f64::from_bits(0x3fa91f65fc0abc0a),
        f64::from_bits(0x3feff621e370ca7a),
    ],
    [
        f64::from_bits(0x3dc53e1f80000000),
        f64::from_bits(0x3faab101bf74ac2e),
        f64::from_bits(0x3feff4dc54b00181),
    ],
    [
        f64::from_bits(0x3e2114a649000000),
        f64::from_bits(0x3fac428d7de920e9),
        f64::from_bits(0x3feff3830f2e9043),
    ],
    [
        f64::from_bits(0x3e0adf0ef4000000),
        f64::from_bits(0x3fadd40723a3cdfb),
        f64::from_bits(0x3feff21614b9d9ad),
    ],
    [
        f64::from_bits(0xbe1d21f591800000),
        f64::from_bits(0x3faf656e1e9e59cd),
        f64::from_bits(0x3feff09565e83d77),
    ],
    [
        f64::from_bits(0xbe14f54d70800000),
        f64::from_bits(0x3fb07b612d6be078),
        f64::from_bits(0x3fefef0102c634e3),
    ],
    [
        f64::from_bits(0xbe11efec9a000000),
        f64::from_bits(0x3fb1440118ba7bd0),
        f64::from_bits(0x3fefed58ecf342da),
    ],
    [
        f64::from_bits(0x3e2cc17ba8800000),
        f64::from_bits(0x3fb20c96cf0a7eed),
        f64::from_bits(0x3fefeb9d24646fa6),
    ],
    [
        f64::from_bits(0x3de121dbe4000000),
        f64::from_bits(0x3fb2d5209628edf0),
        f64::from_bits(0x3fefe9cdacf99cff),
    ],
    [
        f64::from_bits(0xbdd9ecf610000000),
        f64::from_bits(0x3fb39d9f103bf7f7),
        f64::from_bits(0x3fefe7ea854e6b08),
    ],
    [
        f64::from_bits(0xbe004ede8e000000),
        f64::from_bits(0x3fb466116c629e5c),
        f64::from_bits(0x3fefe5f3af4ee201),
    ],
    [
        f64::from_bits(0xbe01821cec000000),
        f64::from_bits(0x3fb52e773c9920c7),
        f64::from_bits(0x3fefe3e92c0e4108),
    ],
    [
        f64::from_bits(0x3e0cdec726000000),
        f64::from_bits(0x3fb5f6d02131f0b2),
        f64::from_bits(0x3fefe1cafc7f1a24),
    ],
    [
        f64::from_bits(0xbe0edece4d000000),
        f64::from_bits(0x3fb6bf1b2653648c),
        f64::from_bits(0x3fefdf99233c230c),
    ],
    [
        f64::from_bits(0xbe02aa4d1c000000),
        f64::from_bits(0x3fb787585bc45f0f),
        f64::from_bits(0x3fefdd53a01d11d9),
    ],
    [
        f64::from_bits(0x3dfd461592000000),
        f64::from_bits(0x3fb84f871e32cf68),
        f64::from_bits(0x3fefdafa74f16482),
    ],
    [
        f64::from_bits(0x3e2f0cbd72800000),
        f64::from_bits(0x3fb917a71d3d2956),
        f64::from_bits(0x3fefd88da29f302e),
    ],
    [
        f64::from_bits(0xbe15832470000000),
        f64::from_bits(0x3fb9dfb6c9865b06),
        f64::from_bits(0x3fefd60d2e14a6b1),
    ],
    [
        f64::from_bits(0xbe12e81bf4000000),
        f64::from_bits(0x3fbaa7b706bfdbba),
        f64::from_bits(0x3fefd3791484ff50),
    ],
    [
        f64::from_bits(0xbe31394141800000),
        f64::from_bits(0x3fbb6fa680a05c27),
        f64::from_bits(0x3fefd0d15a4b8471),
    ],
    [
        f64::from_bits(0x3e171098ff000000),
        f64::from_bits(0x3fbc3785eba12b42),
        f64::from_bits(0x3fefce15fceddccf),
    ],
    [
        f64::from_bits(0xbdfc3519e8000000),
        f64::from_bits(0x3fbcff53302f0590),
        f64::from_bits(0x3fefcb4703b969e1),
    ],
    [
        f64::from_bits(0x3e42f522a5000000),
        f64::from_bits(0x3fbdc70fb84af16e),
        f64::from_bits(0x3fefc8646987fc1d),
    ],
    [
        f64::from_bits(0xbdeae9bed8000000),
        f64::from_bits(0x3fbe8eb7f8a589e2),
        f64::from_bits(0x3fefc56e3b91ca3a),
    ],
    [
        f64::from_bits(0x3e1f8868b2000000),
        f64::from_bits(0x3fbf564e87d2330f),
        f64::from_bits(0x3fefc264701f9a09),
    ],
    [
        f64::from_bits(0xbe2b07985f800000),
        f64::from_bits(0x3fc00ee8835051f4),
        f64::from_bits(0x3fefbf47105f7439),
    ],
    [
        f64::from_bits(0x3e1cbdaa94000000),
        f64::from_bits(0x3fc072a05e1d4d8e),
        f64::from_bits(0x3fefbc16172a9e36),
    ],
    [
        f64::from_bits(0x3e337c5b90800000),
        f64::from_bits(0x3fc0d64df9619f0d),
        f64::from_bits(0x3fefb8d18b635327),
    ],
    [
        f64::from_bits(0xbe3068b5fc800000),
        f64::from_bits(0x3fc139f09bc617f5),
        f64::from_bits(0x3fefb5797351da85),
    ],
    [
        f64::from_bits(0xbe28ea6681800000),
        f64::from_bits(0x3fc19d8919fa4ec8),
        f64::from_bits(0x3fefb20dc7da8aff),
    ],
    [
        f64::from_bits(0x3e36278ceb800000),
        f64::from_bits(0x3fc2011719d50b87),
        f64::from_bits(0x3fefae8e8bd4427f),
    ],
    [
        f64::from_bits(0xbe2096df84000000),
        f64::from_bits(0x3fc264993433763a),
        f64::from_bits(0x3fefaafbcbfca356),
    ],
    [
        f64::from_bits(0x3e29b2534f000000),
        f64::from_bits(0x3fc2c810967bbf70),
        f64::from_bits(0x3fefa7557d8d987e),
    ],
    [
        f64::from_bits(0x3dd215b4e0000000),
        f64::from_bits(0x3fc32b7bfa25c91b),
        f64::from_bits(0x3fefa39bac71954b),
    ],
    [
        f64::from_bits(0xbe194db891000000),
        f64::from_bits(0x3fc38edb9d29b39d),
        f64::from_bits(0x3fef9fce56700a6d),
    ],
    [
        f64::from_bits(0x3e27727f7b800000),
        f64::from_bits(0x3fc3f22f7c3cce3a),
        f64::from_bits(0x3fef9bed7b8c8d8c),
    ],
    [
        f64::from_bits(0xbe20cb3303800000),
        f64::from_bits(0x3fc45576971dd530),
        f64::from_bits(0x3fef97f925d53c83),
    ],
    [
        f64::from_bits(0xbe09071106000000),
        f64::from_bits(0x3fc4b8b175c71e22),
        f64::from_bits(0x3fef93f14feb8022),
    ],
    [
        f64::from_bits(0x3e262741e7800000),
        f64::from_bits(0x3fc51bdfa7ea30d5),
        f64::from_bits(0x3fef8fd5fe3efac8),
    ],
    [
        f64::from_bits(0x3e3f8e16d0c00000),
        f64::from_bits(0x3fc57f00e80e6e12),
        f64::from_bits(0x3fef8ba733a1ceb1),
    ],
    [
        f64::from_bits(0xbe076acbca000000),
        f64::from_bits(0x3fc5e2143b7bc1c2),
        f64::from_bits(0x3fef8764fad5e9bf),
    ],
    [
        f64::from_bits(0xbe10a0f73a000000),
        f64::from_bits(0x3fc6451a76411746),
        f64::from_bits(0x3fef830f4ad232d8),
    ],
    [
        f64::from_bits(0x3e3ca11d1bc00000),
        f64::from_bits(0x3fc6a8135d7bd143),
        f64::from_bits(0x3fef7ea625eb5af7),
    ],
    [
        f64::from_bits(0xbe202f2362800000),
        f64::from_bits(0x3fc70afd74071191),
        f64::from_bits(0x3fef7a299d3f182a),
    ],
    [
        f64::from_bits(0x3e2b34dcb8000000),
        f64::from_bits(0x3fc76dda08544b5c),
        f64::from_bits(0x3fef7599a1ac7ecd),
    ],
    [
        f64::from_bits(0x3df161ff40000000),
        f64::from_bits(0x3fc7d0a7bf2d4aba),
        f64::from_bits(0x3fef70f64322da74),
    ],
    [
        f64::from_bits(0xbe0c49b8b4000000),
        f64::from_bits(0x3fc83366ddb3de23),
        f64::from_bits(0x3fef6c3f7e7c2707),
    ],
    [
        f64::from_bits(0x3e221da851000000),
        f64::from_bits(0x3fc8961743b14290),
        f64::from_bits(0x3fef6775552a6ba2),
    ],
    [
        f64::from_bits(0x3e1ac63eda000000),
        f64::from_bits(0x3fc8f8b851098588),
        f64::from_bits(0x3fef6297cef0cdd6),
    ],
    [
        f64::from_bits(0x3e427ef489c00000),
        f64::from_bits(0x3fc95b4a5b9f2ceb),
        f64::from_bits(0x3fef5da6e7820551),
    ],
    [
        f64::from_bits(0x3e1ae89370000000),
        f64::from_bits(0x3fc9bdcc07900146),
        f64::from_bits(0x3fef58a2b0689c82),
    ],
    [
        f64::from_bits(0x3e2eb48c7e000000),
        f64::from_bits(0x3fca203e4a4f950e),
        f64::from_bits(0x3fef538b1d392049),
    ],
    [
        f64::from_bits(0xbe2bfd282f000000),
        f64::from_bits(0x3fca829ffaad0d79),
        f64::from_bits(0x3fef4e603d51f1aa),
    ],
    [
        f64::from_bits(0x3e27ccf638000000),
        f64::from_bits(0x3fcae4f1fa80e1b5),
        f64::from_bits(0x3fef492204c5ef9e),
    ],
    [
        f64::from_bits(0xbe32435c57800000),
        f64::from_bits(0x3fcb4732b72ebc86),
        f64::from_bits(0x3fef43d0890e1e72),
    ],
    [
        f64::from_bits(0x3e10293fec000000),
        f64::from_bits(0x3fcba9634155f866),
        f64::from_bits(0x3fef3e6bbb6c2ea4),
    ],
    [
        f64::from_bits(0xbe27bb1f92000000),
        f64::from_bits(0x3fcc0b82461f65e0),
        f64::from_bits(0x3fef38f3ae6f9afc),
    ],
    [
        f64::from_bits(0x3e227aaebc000000),
        f64::from_bits(0x3fcc6d906faacf65),
        f64::from_bits(0x3fef3368589e17a2),
    ],
    [
        f64::from_bits(0xbe42e2bcd5000000),
        f64::from_bits(0x3fcccf8c3f74a6c9),
        f64::from_bits(0x3fef2dc9cfb5fa74),
    ],
    [
        f64::from_bits(0xbe16f070ac000000),
        f64::from_bits(0x3fcd31773ba218a8),
        f64::from_bits(0x3fef2817fd4d045b),
    ],
    [
        f64::from_bits(0x3e2469adfc000000),
        f64::from_bits(0x3fcd935004779e57),
        f64::from_bits(0x3fef2252f59c122d),
    ],
    [
        f64::from_bits(0x3df4f51c18000000),
        f64::from_bits(0x3fcdf5164301377a),
        f64::from_bits(0x3fef1c7abdeaa3ef),
    ],
    [
        f64::from_bits(0x3e278e44da000000),
        f64::from_bits(0x3fce56ca4202807c),
        f64::from_bits(0x3fef168f51c5d5d5),
    ],
    [
        f64::from_bits(0x3df49bb5f8000000),
        f64::from_bits(0x3fceb86b4a1b7e9b),
        f64::from_bits(0x3fef1090bc4b6800),
    ],
    [
        f64::from_bits(0xbe367ba541000000),
        f64::from_bits(0x3fcf19f9369d5e93),
        f64::from_bits(0x3fef0a7effdc937f),
    ],
    [
        f64::from_bits(0x3e2c0cab95000000),
        f64::from_bits(0x3fcf7b74ab7219d2),
        f64::from_bits(0x3fef045a1219e594),
    ],
    [
        f64::from_bits(0xbe12b77e32000000),
        f64::from_bits(0x3fcfdcdc0ca3288d),
        f64::from_bits(0x3feefe220cf5c751),
    ],
    [
        f64::from_bits(0xbdee0d8cb0000000),
        f64::from_bits(0x3fd01f18054c8362),
        f64::from_bits(0x3feef7d6e54c347d),
    ],
    [
        f64::from_bits(0xbe2ecd5b9c000000),
        f64::from_bits(0x3fd04fb7f6d35d68),
        f64::from_bits(0x3feef178a6f9a987),
    ],
    [
        f64::from_bits(0x3e2eb24de5000000),
        f64::from_bits(0x3fd0804e1d369ff2),
        f64::from_bits(0x3feeeb074934fdf0),
    ],
    [
        f64::from_bits(0x3e14a897c4000000),
        f64::from_bits(0x3fd0b0d9d7b0d042),
        f64::from_bits(0x3feee482e14bcde0),
    ],
    [
        f64::from_bits(0x3e1336c376000000),
        f64::from_bits(0x3fd0e15b555e7bec),
        f64::from_bits(0x3feeddeb6908ca8c),
    ],
    [
        f64::from_bits(0xbe03952d90000000),
        f64::from_bits(0x3fd111d25efd48b8),
        f64::from_bits(0x3feed740e7eb8dd6),
    ],
    [
        f64::from_bits(0x3e0fc2a5d4000000),
        f64::from_bits(0x3fd1423ef5c7e1bd),
        f64::from_bits(0x3feed0835dc24e89),
    ],
    [
        f64::from_bits(0x3e2a88ed37000000),
        f64::from_bits(0x3fd172a0eb8361da),
        f64::from_bits(0x3feec9b2d0ec8288),
    ],
    [
        f64::from_bits(0xbe48ca4cb9400000),
        f64::from_bits(0x3fd1a2f7b10b6d70),
        f64::from_bits(0x3feec2cf55d6117c),
    ],
    [
        f64::from_bits(0x3e40144524000000),
        f64::from_bits(0x3fd1d3446fd0cd3f),
        f64::from_bits(0x3feebbd8c1d62f96),
    ],
    [
        f64::from_bits(0xbe3abf810c000000),
        f64::from_bits(0x3fd203855b85f89a),
        f64::from_bits(0x3feeb4cf57454132),
    ],
    [
        f64::from_bits(0x3e35d4c5d5800000),
        f64::from_bits(0x3fd233bbcca40561),
        f64::from_bits(0x3feeadb2e40746ca),
    ],
    [
        f64::from_bits(0xbe2a1b0c58000000),
        f64::from_bits(0x3fd263e685b1d714),
        f64::from_bits(0x3feea68396d87754),
    ],
    [
        f64::from_bits(0xbe277c8dac000000),
        f64::from_bits(0x3fd294061d2eb611),
        f64::from_bits(0x3fee9f41597393c8),
    ],
    [
        f64::from_bits(0x3e1915540e000000),
        f64::from_bits(0x3fd2c41a580014cf),
        f64::from_bits(0x3fee97ec348fb87f),
    ],
    [
        f64::from_bits(0xbe3abb6d9b000000),
        f64::from_bits(0x3fd2f422b2d0990c),
        f64::from_bits(0x3fee90843c55b996),
    ],
    [
        f64::from_bits(0xbe3b8ee5d5800000),
        f64::from_bits(0x3fd3241f8cea2836),
        f64::from_bits(0x3fee890962268c49),
    ],
    [
        f64::from_bits(0xbe31cd2982800000),
        f64::from_bits(0x3fd35410a8396266),
        f64::from_bits(0x3fee817baf85c094),
    ],
    [
        f64::from_bits(0xbdfe216af0000000),
        f64::from_bits(0x3fd383f5e08283e2),
        f64::from_bits(0x3fee79db2a188b0a),
    ],
    [
        f64::from_bits(0xbe024afc30000000),
        f64::from_bits(0x3fd3b3cef6993c0b),
        f64::from_bits(0x3fee7227dbf82004),
    ],
    [
        f64::from_bits(0xbe0aa1657c000000),
        f64::from_bits(0x3fd3e39be4767224),
        f64::from_bits(0x3fee6a61c62d5274),
    ],
    [
        f64::from_bits(0xbe1c5b65fa000000),
        f64::from_bits(0x3fd4135c898485bb),
        f64::from_bits(0x3fee6288ee07fea5),
    ],
    [
        f64::from_bits(0x3df23e8978000000),
        f64::from_bits(0x3fd44310de3c284b),
        f64::from_bits(0x3fee5a9d54bbd26c),
    ],
    [
        f64::from_bits(0xbe22b1d77a000000),
        f64::from_bits(0x3fd472b8976d498d),
        f64::from_bits(0x3fee529f06cb187d),
    ],
    [
        f64::from_bits(0xbe0daaa348000000),
        f64::from_bits(0x3fd4a253cb97efd1),
        f64::from_bits(0x3fee4a8e007231a2),
    ],
    [
        f64::from_bits(0xbe3322f570800000),
        f64::from_bits(0x3fd4d1e2260c3422),
        f64::from_bits(0x3fee426a500f6e33),
    ],
    [
        f64::from_bits(0x3e264758e8000000),
        f64::from_bits(0x3fd50163eca0b337),
        f64::from_bits(0x3fee3a33e996b722),
    ],
    [
        f64::from_bits(0x3e31248627800000),
        f64::from_bits(0x3fd530d89a17e007),
        f64::from_bits(0x3fee31eae3fb917b),
    ],
    [
        f64::from_bits(0xbe46c3416cc00000),
        f64::from_bits(0x3fd5603fcf8cd8a3),
        f64::from_bits(0x3fee298f502a579b),
    ],
    [
        f64::from_bits(0x3e2ab481ff000000),
        f64::from_bits(0x3fd58f9a896aa209),
        f64::from_bits(0x3fee2121016e14fc),
    ],
    [
        f64::from_bits(0xbe26eb838b000000),
        f64::from_bits(0x3fd5bee77aaf890b),
        f64::from_bits(0x3fee18a032eb4df5),
    ],
    [
        f64::from_bits(0xbdfd159b80000000),
        f64::from_bits(0x3fd5ee2734efeef5),
        f64::from_bits(0x3fee100ccaa6bd78),
    ],
    [
        f64::from_bits(0xbdda42e4a0000000),
        f64::from_bits(0x3fd61d595bedeabc),
        f64::from_bits(0x3fee0766d944915e),
    ],
    [
        f64::from_bits(0xbe143d0dc0000000),
        f64::from_bits(0x3fd64c7dd5cc0cd1),
        f64::from_bits(0x3fedfeae63903034),
    ],
    [
        f64::from_bits(0xbe48c7bdb7000000),
        f64::from_bits(0x3fd67b9453ca2122),
        f64::from_bits(0x3fedf5e378482eae),
    ],
    [
        f64::from_bits(0x3e11c0ead6000000),
        f64::from_bits(0x3fd6aa9d844c980a),
        f64::from_bits(0x3feded05f6a23a52),
    ],
    [
        f64::from_bits(0x3e07d52600000000),
        f64::from_bits(0x3fd6d99867e90d92),
        f64::from_bits(0x3fede4160e97b2e2),
    ],
    [
        f64::from_bits(0x3e3924e036800000),
        f64::from_bits(0x3fd7088555d3c816),
        f64::from_bits(0x3feddb13afb14e37),
    ],
    [
        f64::from_bits(0xbe174b7c3e000000),
        f64::from_bits(0x3fd73763c09fba09),
        f64::from_bits(0x3fedd1fef5335416),
    ],
    [
        f64::from_bits(0xbe17943ad0000000),
        f64::from_bits(0x3fd766340685c982),
        f64::from_bits(0x3fedc8d7ccf2567a),
    ],
    [
        f64::from_bits(0x3e279dd614000000),
        f64::from_bits(0x3fd794f5f7522b88),
        f64::from_bits(0x3fedbf9e402aa5c3),
    ],
    [
        f64::from_bits(0x3e17b64f32000000),
        f64::from_bits(0x3fd7c3a939c32d81),
        f64::from_bits(0x3fedb652607e0db1),
    ],
    [
        f64::from_bits(0xbe32bea5ce800000),
        f64::from_bits(0x3fd7f24db825141c),
        f64::from_bits(0x3fedacf43268b5b0),
    ],
    [
        f64::from_bits(0x3e1733c024000000),
        f64::from_bits(0x3fd820e3b8bf15a0),
        f64::from_bits(0x3feda383a7aed887),
    ],
    [
        f64::from_bits(0xbe4eac0fc9400000),
        f64::from_bits(0x3fd84f6a51d077b3),
        f64::from_bits(0x3fed9a00efd84537),
    ],
    [
        f64::from_bits(0x3e4aca3733800000),
        f64::from_bits(0x3fd87de2f4704f98),
        f64::from_bits(0x3fed906bbf17f4da),
    ],
    [
        f64::from_bits(0xbe1910c4f0000000),
        f64::from_bits(0x3fd8ac4b7dc0d986),
        f64::from_bits(0x3fed86c4862b5d6e),
    ],
    [
        f64::from_bits(0xbe033bb860000000),
        f64::from_bits(0x3fd8daa52b4dc041),
        f64::from_bits(0x3fed7d0b0374a559),
    ],
    [
        f64::from_bits(0xbe469e1507000000),
        f64::from_bits(0x3fd908ef408ad220),
        f64::from_bits(0x3fed733f5e71c3bc),
    ],
    [
        f64::from_bits(0x3e4cffacf0800000),
        f64::from_bits(0x3fd9372ab7784d36),
        f64::from_bits(0x3fed696161d786c9),
    ],
    [
        f64::from_bits(0xbe58629d9f000000),
        f64::from_bits(0x3fd965552b0849ab),
        f64::from_bits(0x3fed5f7190eeae23),
    ],
    [
        f64::from_bits(0x3e14150000000000),
        f64::from_bits(0x3fd99371687c64f3),
        f64::from_bits(0x3fed556f5155d9dd),
    ],
    [
        f64::from_bits(0xbe4bd37aad800000),
        f64::from_bits(0x3fd9c17cf40715cb),
        f64::from_bits(0x3fed4b5b2caf8386),
    ],
    [
        f64::from_bits(0x3e5d02cde7000000),
        f64::from_bits(0x3fd9ef79ea4d995d),
        f64::from_bits(0x3fed4134ac5eb246),
    ],
    [
        f64::from_bits(0xbe110547ac000000),
        f64::from_bits(0x3fda1d653d9adf5e),
        f64::from_bits(0x3fed36fc7d291602),
    ],
    [
        f64::from_bits(0xbe401a1a22800000),
        f64::from_bits(0x3fda4b40f9c0120b),
        f64::from_bits(0x3fed2cb22b45236b),
    ],
    [
        f64::from_bits(0x3e23ce2bac000000),
        f64::from_bits(0x3fda790ce2056b9a),
        f64::from_bits(0x3fed2255c3ae11a5),
    ],
    [
        f64::from_bits(0xbdfccb4a60000000),
        f64::from_bits(0x3fdaa6c828db4ea8),
        f64::from_bits(0x3fed17e774d4e3e2),
    ],
    [
        f64::from_bits(0x3e25db4b00000000),
        f64::from_bits(0x3fdad47321f29847),
        f64::from_bits(0x3fed0d672bc0b122),
    ],
    [
        f64::from_bits(0x3e232f6a6e000000),
        f64::from_bits(0x3fdb020d7a285e23),
        f64::from_bits(0x3fed02d4fb84d334),
    ],
    [
        f64::from_bits(0x3e5cf8e39bc00000),
        f64::from_bits(0x3fdb2f97c27f7494),
        f64::from_bits(0x3fecf830c2248c5e),
    ],
    [
        f64::from_bits(0x3e18927bb0000000),
        f64::from_bits(0x3fdb5d10129a750a),
        f64::from_bits(0x3feced7af22cb105),
    ],
    [
        f64::from_bits(0xbe33dec3c1000000),
        f64::from_bits(0x3fdb8a77f8d0bbc5),
        f64::from_bits(0x3fece2b32e50d6cd),
    ],
    [
        f64::from_bits(0xbe326ba536000000),
        f64::from_bits(0x3fdbb7cf08f0290d),
        f64::from_bits(0x3fecd7d98fcf3b1e),
    ],
    [
        f64::from_bits(0x3e223c568e000000),
        f64::from_bits(0x3fdbe51524e3aa53),
        f64::from_bits(0x3fecccee1da3d56e),
    ],
    [
        f64::from_bits(0xbe2f3b3af0000000),
        f64::from_bits(0x3fdc1249c1f5f2f6),
        f64::from_bits(0x3fecc1f0f95e1e24),
    ],
    [
        f64::from_bits(0xbe31286a47000000),
        f64::from_bits(0x3fdc3f6d2ef7054b),
        f64::from_bits(0x3fecb6e20ff37e81),
    ],
    [
        f64::from_bits(0x3e2641214e000000),
        f64::from_bits(0x3fdc6c7f594003d9),
        f64::from_bits(0x3fecabc165bf1b60),
    ],
    [
        f64::from_bits(0x3e40cda7c9000000),
        f64::from_bits(0x3fdc997ff2bffccb),
        f64::from_bits(0x3feca08f0dee434c),
    ],
    [
        f64::from_bits(0xbe35557ac9000000),
        f64::from_bits(0x3fdcc66e7b42e8f1),
        f64::from_bits(0x3fec954b28bca62e),
    ],
    [
        f64::from_bits(0x3e3555eb62000000),
        f64::from_bits(0x3fdcf34bccc567a1),
        f64::from_bits(0x3fec89f57f6e20f3),
    ],
    [
        f64::from_bits(0xbe34e0e361000000),
        f64::from_bits(0x3fdd2016cbb5e39a),
        f64::from_bits(0x3fec7e8e59999e1f),
    ],
    [
        f64::from_bits(0x3e2446da1e000000),
        f64::from_bits(0x3fdd4cd039d0ed05),
        f64::from_bits(0x3fec731585f970eb),
    ],
    [
        f64::from_bits(0x3e2103d328000000),
        f64::from_bits(0x3fdd797767638dec),
        f64::from_bits(0x3fec678b3174afe1),
    ],
    [
        f64::from_bits(0x3e35814d60000000),
        f64::from_bits(0x3fdda60c7ae9dc22),
        f64::from_bits(0x3fec5bef522be6fb),
    ],
    [
        f64::from_bits(0xbe25e2321e000000),
        f64::from_bits(0x3fddd28f054cbb3f),
        f64::from_bits(0x3fec5042052c8c42),
    ],
    [
        f64::from_bits(0xbe2a259ffe000000),
        f64::from_bits(0x3fddfeff54854631),
        f64::from_bits(0x3fec44833611bc7d),
    ],
    [
        f64::from_bits(0xbe04f28d80000000),
        f64::from_bits(0x3fde2b5d34665b35),
        f64::from_bits(0x3fec38b2f278ea7e),
    ],
    [
        f64::from_bits(0xbdbde57100000000),
        f64::from_bits(0x3fde57a86d137f20),
        f64::from_bits(0x3fec2cd1493d05c2),
    ],
    [
        f64::from_bits(0x3e2e0d8d14000000),
        f64::from_bits(0x3fde83e0ffb7bfb4),
        f64::from_bits(0x3fec20de3a08ea07),
    ],
    [
        f64::from_bits(0xbe312a858e000000),
        f64::from_bits(0x3fdeb0067e48baf4),
        f64::from_bits(0x3fec14d9e2bd511e),
    ],
    [
        f64::from_bits(0x3e49a17403000000),
        f64::from_bits(0x3fdedc19997a4431),
        f64::from_bits(0x3fec08c413089b2e),
    ],
    [
        f64::from_bits(0x3e268c8636000000),
        f64::from_bits(0x3fdf0819163d1bc0),
        f64::from_bits(0x3febfc9d21568f32),
    ],
    [
        f64::from_bits(0x3e24cc5eb8000000),
        f64::from_bits(0x3fdf3405a482e11d),
        f64::from_bits(0x3febf064dd580fc9),
    ],
    [
        f64::from_bits(0xbe4fce7cd8000000),
        f64::from_bits(0x3fdf5fde8f3f11d4),
        f64::from_bits(0x3febe41b798f6b97),
    ],
    [
        f64::from_bits(0xbe2af81690000000),
        f64::from_bits(0x3fdf8ba4c98a9816),
        f64::from_bits(0x3febd7c0b1a7f14b),
    ],
    [
        f64::from_bits(0x3de6e39e20000000),
        f64::from_bits(0x3fdfb7575d1ea750),
        f64::from_bits(0x3febcb54cac5dde5),
    ],
    [
        f64::from_bits(0x3e330f9256000000),
        f64::from_bits(0x3fdfe2f665dcd168),
        f64::from_bits(0x3febbed7bd1e17b0),
    ],
    [
        f64::from_bits(0x3e0626de20000000),
        f64::from_bits(0x3fe00740ca0d5fbb),
        f64::from_bits(0x3febb2499f9fe7a3),
    ],
    [
        f64::from_bits(0x3e15cc7030000000),
        f64::from_bits(0x3fe01cfc8afeea0e),
        f64::from_bits(0x3feba5aa650dd495),
    ],
    [
        f64::from_bits(0xbdf6191e60000000),
        f64::from_bits(0x3fe032ae54fe4057),
        f64::from_bits(0x3feb98fa2065a5e6),
    ],
    [
        f64::from_bits(0xbe06b14850000000),
        f64::from_bits(0x3fe0485624c328c8),
        f64::from_bits(0x3feb8c38d39737bc),
    ],
    [
        f64::from_bits(0xbe211fbc3a000000),
        f64::from_bits(0x3fe05df3e66a716d),
        f64::from_bits(0x3feb7f668a580fd0),
    ],
    [
        f64::from_bits(0xbe40eca7f0000000),
        f64::from_bits(0x3fe07387825589ec),
        f64::from_bits(0x3feb728352c44517),
    ],
    [
        f64::from_bits(0xbe68073bc9e00000),
        f64::from_bits(0x3fe089109ef1284d),
        f64::from_bits(0x3feb658f630112ed),
    ],
    [
        f64::from_bits(0xbe49dcf0ad000000),
        f64::from_bits(0x3fe09e9051603e29),
        f64::from_bits(0x3feb588a13ab750f),
    ],
    [
        f64::from_bits(0xbe206ea9f0000000),
        f64::from_bits(0x3fe0b405820e78e7),
        f64::from_bits(0x3feb4b740d3cc07b),
    ],
    [
        f64::from_bits(0xbe136a8d0c000000),
        f64::from_bits(0x3fe0c9704a1ea4e5),
        f64::from_bits(0x3feb3e4d40f5524d),
    ],
    [
        f64::from_bits(0x3e163d1f30000000),
        f64::from_bits(0x3fe0ded0bc01a533),
        f64::from_bits(0x3feb3115a3a628af),
    ],
    [
        f64::from_bits(0x3e5f3181f1400000),
        f64::from_bits(0x3fe0f4270e4787bf),
        f64::from_bits(0x3feb23cd1314c779),
    ],
    [
        f64::from_bits(0xbe2f269b78000000),
        f64::from_bits(0x3fe109723e75c5cf),
        f64::from_bits(0x3feb167430cfebdb),
    ],
    [
        f64::from_bits(0x3e41d84dc0800000),
        f64::from_bits(0x3fe11eb36bc9db52),
        f64::from_bits(0x3feb090a4915ee88),
    ],
    [
        f64::from_bits(0xbe408e6006800000),
        f64::from_bits(0x3fe133e9ba0061d8),
        f64::from_bits(0x3feafb8fe69a6527),
    ],
    [
        f64::from_bits(0x3e4cda72ab000000),
        f64::from_bits(0x3fe14915d557a7c9),
        f64::from_bits(0x3feaee049bc0aee0),
    ],
    [
        f64::from_bits(0xbe1f32f950000000),
        f64::from_bits(0x3fe15e36dfb6bb55),
        f64::from_bits(0x3feae068f6991699),
    ],
    [
        f64::from_bits(0x3e3138092d000000),
        f64::from_bits(0x3fe1734d6f34d7f0),
        f64::from_bits(0x3fead2bc96c1e1f5),
    ],
    [
        f64::from_bits(0x3e56b382dd400000),
        f64::from_bits(0x3fe188595ae376a5),
        f64::from_bits(0x3feac4ff962bdb6d),
    ],
    [
        f64::from_bits(0xbe3f12fafa000000),
        f64::from_bits(0x3fe19d59f592a587),
        f64::from_bits(0x3feab7326685eb57),
    ],
    [
        f64::from_bits(0xbe32909e5a000000),
        f64::from_bits(0x3fe1b2500aed7ac6),
        f64::from_bits(0x3feaa954823cf815),
    ],
    [
        f64::from_bits(0xbe6d66a897800000),
        f64::from_bits(0x3fe1c73aa0150cf9),
        f64::from_bits(0x3fea9b668fb0503f),
    ],
    [
        f64::from_bits(0x3e4311ea86000000),
        f64::from_bits(0x3fe1dc1b7db74db1),
        f64::from_bits(0x3fea8d675d9c6cc8),
    ],
    [
        f64::from_bits(0xbe041c02b8000000),
        f64::from_bits(0x3fe1f0f08a1a06a4),
        f64::from_bits(0x3fea7f5853bb4309),
    ],
    [
        f64::from_bits(0xbe5ca1f4ed000000),
        f64::from_bits(0x3fe205ba57211271),
        f64::from_bits(0x3fea71391146958f),
    ],
    [
        f64::from_bits(0xbe3910ce77000000),
        f64::from_bits(0x3fe21a7988f8326b),
        f64::from_bits(0x3fea63092626202f),
    ],
    [
        f64::from_bits(0x3e62bfadbee00000),
        f64::from_bits(0x3fe22f2dc71afab6),
        f64::from_bits(0x3fea54c8cd9fd0d9),
    ],
    [
        f64::from_bits(0xbe45f1c02a800000),
        f64::from_bits(0x3fe243d5df4afb93),
        f64::from_bits(0x3fea4678dbbe5e73),
    ],
    [
        f64::from_bits(0xbe1db12b90000000),
        f64::from_bits(0x3fe2587347f493a4),
        f64::from_bits(0x3fea38184db0df23),
    ],
    [
        f64::from_bits(0xbe17b29e00000000),
        f64::from_bits(0x3fe26d05490f2f61),
        f64::from_bits(0x3fea29a7a2f40b49),
    ],
    [
        f64::from_bits(0xbe2b3ddca4000000),
        f64::from_bits(0x3fe2818be6930629),
        f64::from_bits(0x3fea1b26d8f070d7),
    ],
    [
        f64::from_bits(0x3e2e112744000000),
        f64::from_bits(0x3fe2960730ff2bcd),
        f64::from_bits(0x3fea0c95e3df5e0e),
    ],
    [
        f64::from_bits(0xbe35269766000000),
        f64::from_bits(0x3fe2aa76dafcbbf4),
        f64::from_bits(0x3fe9fdf4fae1df6f),
    ],
    [
        f64::from_bits(0xbe309777e1000000),
        f64::from_bits(0x3fe2bedb1b6b4e15),
        f64::from_bits(0x3fe9ef43f6cbe162),
    ],
    [
        f64::from_bits(0x3e3ae2051f000000),
        f64::from_bits(0x3fe2d333e4617f25),
        f64::from_bits(0x3fe9e082e148680e),
    ],
    [
        f64::from_bits(0xbe436f6ced800000),
        f64::from_bits(0x3fe2e780cb47180e),
        f64::from_bits(0x3fe9d1b207f383c3),
    ],
    [
        f64::from_bits(0xbe323fdc6b000000),
        f64::from_bits(0x3fe2fbc23fba2f44),
        f64::from_bits(0x3fe9c2d1197130a7),
    ],
    [
        f64::from_bits(0x3debc540e0000000),
        f64::from_bits(0x3fe30ff7fd6d967d),
        f64::from_bits(0x3fe9b3e0478b961b),
    ],
    [
        f64::from_bits(0xbe3cfb4ed7000000),
        f64::from_bits(0x3fe32421da0bf0e9),
        f64::from_bits(0x3fe9a4dfb1c89326),
    ],
    [
        f64::from_bits(0x3e555802aec00000),
        f64::from_bits(0x3fe3384042a92b1d),
        f64::from_bits(0x3fe995cf06920d11),
    ],
    [
        f64::from_bits(0x3e360719e4000000),
        f64::from_bits(0x3fe34c52608e3a92),
        f64::from_bits(0x3fe986aee6d6837e),
    ],
    [
        f64::from_bits(0xbe1cbf2e48000000),
        f64::from_bits(0x3fe36058ac8863b6),
        f64::from_bits(0x3fe9777ef832c986),
    ],
    [
        f64::from_bits(0x3e49061c32000000),
        f64::from_bits(0x3fe374533ab707d0),
        f64::from_bits(0x3fe9683f2ad7e2ec),
    ],
    [
        f64::from_bits(0xbe4da84dfe000000),
        f64::from_bits(0x3fe3884160f9488f),
        f64::from_bits(0x3fe958f000fdd50a),
    ],
    [
        f64::from_bits(0x3e292e8a74000000),
        f64::from_bits(0x3fe39c23eba6b22a),
        f64::from_bits(0x3fe94990dd9cee51),
    ],
    [
        f64::from_bits(0xbe2bff5d9a000000),
        f64::from_bits(0x3fe3affa20756bdd),
        f64::from_bits(0x3fe93a225056084a),
    ],
    [
        f64::from_bits(0x3db4c46200000000),
        f64::from_bits(0x3fe3c3c4498e98eb),
        f64::from_bits(0x3fe92aa41fbb951c),
    ],
    [
        f64::from_bits(0xbe3e4613e9000000),
        f64::from_bits(0x3fe3d782261dff62),
        f64::from_bits(0x3fe91b167e92d706),
    ],
    [
        f64::from_bits(0x3e10eb2964000000),
        f64::from_bits(0x3fe3eb33ed579bbe),
        f64::from_bits(0x3fe90b794146043c),
    ],
    [
        f64::from_bits(0xbe260abec2000000),
        f64::from_bits(0x3fe3fed94c834d8a),
        f64::from_bits(0x3fe8fbcca9583479),
    ],
    [
        f64::from_bits(0x3e46954977000000),
        f64::from_bits(0x3fe4127281ddac03),
        f64::from_bits(0x3fe8ec1085083553),
    ],
    [
        f64::from_bits(0x3e2a16fec2000000),
        f64::from_bits(0x3fe425ff1f841235),
        f64::from_bits(0x3fe8dc452ca328d3),
    ],
    [
        f64::from_bits(0xbe427bcdd3000000),
        f64::from_bits(0x3fe4397f44aa44f2),
        f64::from_bits(0x3fe8cc6a8771e165),
    ],
    [
        f64::from_bits(0xbe360dded4000000),
        f64::from_bits(0x3fe44cf317a563db),
        f64::from_bits(0x3fe8bc8076122736),
    ],
    [
        f64::from_bits(0xbe59a8f405c00000),
        f64::from_bits(0x3fe4605a2b02d705),
        f64::from_bits(0x3fe8ac875232f3ef),
    ],
    [
        f64::from_bits(0x3e432777dc000000),
        f64::from_bits(0x3fe473b532bc5a67),
        f64::from_bits(0x3fe89c7e8713120c),
    ],
    [
        f64::from_bits(0xbe51418a7b000000),
        f64::from_bits(0x3fe4870306ca20e2),
        f64::from_bits(0x3fe88c670a0ea774),
    ],
    [
        f64::from_bits(0xbe3fed182e000000),
        f64::from_bits(0x3fe49a44886b5340),
        f64::from_bits(0x3fe87c401fdf05e5),
    ],
    [
        f64::from_bits(0x3e486144d8000000),
        f64::from_bits(0x3fe4ad796ea1410c),
        f64::from_bits(0x3fe86c0a04dbacc5),
    ],
    [
        f64::from_bits(0x3de1bc2e60000000),
        f64::from_bits(0x3fe4c0a14640d2af),
        f64::from_bits(0x3fe85bc51aa114c2),
    ],
    [
        f64::from_bits(0xbe3f53d2fe000000),
        f64::from_bits(0x3fe4d3bc5aaa8cd5),
        f64::from_bits(0x3fe84b7121b30a13),
    ],
    [
        f64::from_bits(0xbe12e100a0000000),
        f64::from_bits(0x3fe4e6cab91556be),
        f64::from_bits(0x3fe83b0e0e6b6ccc),
    ],
    [
        f64::from_bits(0xbe2fa58c62000000),
        f64::from_bits(0x3fe4f9cc1c69fdde),
        f64::from_bits(0x3fe82a9c1c1ab463),
    ],
    [
        f64::from_bits(0x3debb491e0000000),
        f64::from_bits(0x3fe50cc09fdcbd92),
        f64::from_bits(0x3fe81a1b3342f858),
    ],
    [
        f64::from_bits(0x3e3a115410000000),
        f64::from_bits(0x3fe51fa82c3aa029),
        f64::from_bits(0x3fe8098b67ea8509),
    ],
    [
        f64::from_bits(0x3e4ab0a5d3000000),
        f64::from_bits(0x3fe53282b20b96b6),
        f64::from_bits(0x3fe7f8ecc7919530),
    ],
    [
        f64::from_bits(0xbe3cba0438000000),
        f64::from_bits(0x3fe5454fe43a7d7c),
        f64::from_bits(0x3fe7e83f96af78a0),
    ],
    [
        f64::from_bits(0xbe20dd83a4000000),
        f64::from_bits(0x3fe5581033a81573),
        f64::from_bits(0x3fe7d783712e20ec),
    ],
    [
        f64::from_bits(0xbe3e9a8299000000),
        f64::from_bits(0x3fe56ac33fbb8253),
        f64::from_bits(0x3fe7c6b8acf90fa6),
    ],
    [
        f64::from_bits(0x3e2225c4aa000000),
        f64::from_bits(0x3fe57d6939d4b513),
        f64::from_bits(0x3fe7b5df1da18065),
    ],
    [
        f64::from_bits(0xbe482e66e0000000),
        f64::from_bits(0x3fe59001b9e64d79),
        f64::from_bits(0x3fe7a4f72157cfdf),
    ],
    [
        f64::from_bits(0x3e551a6a35400000),
        f64::from_bits(0x3fe5a28d5b36d597),
        f64::from_bits(0x3fe794002a7c9023),
    ],
    [
        f64::from_bits(0x3e513917f4000000),
        f64::from_bits(0x3fe5b50b4e10bec1),
        f64::from_bits(0x3fe782faf6dc7ba2),
    ],
    [
        f64::from_bits(0x3e149310cc000000),
        f64::from_bits(0x3fe5c77bc15ab4ef),
        f64::from_bits(0x3fe771e75c43942e),
    ],
    [
        f64::from_bits(0x3e124d493c000000),
        f64::from_bits(0x3fe5d9dee9de49db),
        f64::from_bits(0x3fe760c529bc17b0),
    ],
    [
        f64::from_bits(0xbe504638f7000000),
        f64::from_bits(0x3fe5ec347044e0f4),
        f64::from_bits(0x3fe74f94b0af9720),
    ],
    [
        f64::from_bits(0xbe23f41b28000000),
        f64::from_bits(0x3fe5fe7cb834600c),
        f64::from_bits(0x3fe73e55936a5160),
    ],
    [
        f64::from_bits(0xbe1a5f6f5c000000),
        f64::from_bits(0x3fe610b7515d1562),
        f64::from_bits(0x3fe72d083b8214eb),
    ],
    [
        f64::from_bits(0x3e319fb2e0000000),
        f64::from_bits(0x3fe622e459eafbc1),
        f64::from_bits(0x3fe71bac8c7b0592),
    ],
    [
        f64::from_bits(0xbe356d2c2b000000),
        f64::from_bits(0x3fe6350396fe4e62),
        f64::from_bits(0x3fe70a42bec51665),
    ],
    [
        f64::from_bits(0xbe33c156c2000000),
        f64::from_bits(0x3fe64715385bed93),
        f64::from_bits(0x3fe6f8caa4969708),
    ],
    [
        f64::from_bits(0xbe2f23e576000000),
        f64::from_bits(0x3fe659191d2fd57f),
        f64::from_bits(0x3fe6e7445d74f711),
    ],
    [
        f64::from_bits(0x3e11e4be38000000),
        f64::from_bits(0x3fe66b0f41d484c4),
        f64::from_bits(0x3fe6d5afecd4938d),
    ],
    [
        f64::from_bits(0xbe4397cc8d800000),
        f64::from_bits(0x3fe67cf76eac73df),
        f64::from_bits(0x3fe6c40d89625f63),
    ],
    [
        f64::from_bits(0xbe3202f686000000),
        f64::from_bits(0x3fe68ed1e0990551),
        f64::from_bits(0x3fe6b25cf728c350),
    ],
];

#[derive(Clone, Copy)]
struct D {
    hi: u64,
    lo: u64,
    ex: i64,
    sgn: u64,
}
const ZERO: D = D {
    hi: 0,
    lo: 0,
    ex: -1076,
    sgn: 0,
};
const ONE: D = D {
    hi: 1 << 63,
    lo: 0,
    ex: 1,
    sgn: 0,
};
const MAGIC: D = D { ex: -10, ..ONE };
impl D {
    fn bits(self) -> u128 {
        ((self.hi as u128) << 64) | self.lo as u128
    }
    fn with_bits(self, b: u128) -> Self {
        Self {
            hi: (b >> 64) as u64,
            lo: b as u64,
            ..self
        }
    }
    fn neg(mut self) -> Self {
        self.sgn ^= 1;
        self
    }
    fn normalize(self) -> Self {
        let b = self.bits();
        if b == 0 {
            return self;
        }
        let sh = b.leading_zeros();
        Self {
            ex: self.ex - i64::from(sh),
            ..self
        }
        .with_bits(b << sh)
    }
    fn add(self, b: Self) -> Self {
        if self.hi == 0 {
            return b;
        }
        if b.hi == 0 {
            return self;
        }
        let cmp = (self.ex, self.hi, self.lo).cmp(&(b.ex, b.hi, b.lo));
        if cmp.is_lt() {
            return b.add(self);
        }
        if cmp.is_eq() {
            return if self.sgn != b.sgn {
                ZERO
            } else {
                Self {
                    ex: self.ex + 1,
                    ..self
                }
            };
        }
        let a = self.bits();
        let k = self.ex - b.ex;
        let bb = if k < 128 { b.bits() >> k } else { 0 };
        let mut ex = self.ex;
        let c = if self.sgn != b.sgn {
            let mut c = a.wrapping_sub(bb);
            let sh = c.leading_zeros();
            if sh > 0 {
                c = if k == 1 {
                    (a << sh).wrapping_sub(b.bits() << (sh - 1))
                } else {
                    (a << sh).wrapping_sub(bb << sh)
                };
                ex -= i64::from(sh);
            }
            let sh = (c >> 64).leading_zeros() - 64;
            ex -= i64::from(sh);
            c << sh
        } else {
            let (c, carry) = a.overflowing_add(bb);
            if carry {
                ex += 1;
                (1 << 127) | (c >> 1)
            } else {
                c
            }
        };
        // The normalization above is exact when cancellation exposes discarded bits.

        Self { ex, ..self }.with_bits(c)
    }
    fn mul(self, b: Self) -> Self {
        let v = (self.hi as u128 * b.hi as u128)
            .wrapping_add((self.hi as u128 * b.lo as u128) >> 64)
            .wrapping_add((self.lo as u128 * b.hi as u128) >> 64);
        let sh = u32::from(v >> 127 == 0);
        Self {
            ex: self.ex + b.ex - i64::from(sh),
            sgn: self.sgn ^ b.sgn,
            ..self
        }
        .with_bits(v << sh)
    }
    fn inv(self) -> Self {
        let h = self.hi;
        let mut t = TINV[((h >> 55) & 255) as usize];
        for (sh, back) in [(55, 72), (47, 80), (31, 96)] {
            let e = (1u128 << 127).wrapping_sub(h as u128 * t as u128);
            let e = (t as u128).wrapping_mul(e >> sh);
            t = t.wrapping_add((e >> back) as u64);
        }
        let r = Self {
            hi: t,
            lo: 0,
            ex: 1 - self.ex,
            sgn: 0,
        };
        r.add(r.mul(ONE.add(self.mul(r.neg()))))
    }
    fn div(self, a: Self) -> Self {
        a.inv().mul(self)
    }
    fn from_double(x: f64) -> Self {
        let b = x.to_bits();
        let e = (b >> 52) & 2047;
        let m = (b & 0xfffffffffffff) + if e != 0 { 1 << 52 } else { 0 };
        let sh = m.leading_zeros();
        Self {
            hi: m << sh,
            lo: 0,
            ex: e as i64 - 1022 - if sh > 11 { i64::from(sh) - 12 } else { 0 },
            sgn: u64::from(x < 0.0),
        }
    }
    fn to_double(self) -> f64 {
        let sign = self.sgn << 63;
        if self.hi == 0 || self.ex < -1074 {
            return f64::from_bits(sign);
        }
        if self.ex > 1024 {
            return f64::from_bits(sign | 0x7ff0_0000_0000_0000);
        }
        // Round the retained 128-bit significand directly, including ties and
        // subnormal results; this is the nearest-even form of dint_tod.
        let shift = if self.ex >= -1021 {
            75
        } else {
            (-946 - self.ex) as u32
        };
        let bits = self.bits();
        let rounded = if shift == 128 {
            u64::from(bits > (1u128 << 127))
        } else {
            let hi = (bits >> shift) as u64;
            let low = bits & ((1u128 << shift) - 1);
            let mid = 1u128 << (shift - 1);
            hi + u64::from(low > mid || (low == mid && hi & 1 != 0))
        };
        let value = if self.ex >= -1021 {
            (((self.ex + 1022) as u64) << 52) + rounded - (1 << 52)
        } else {
            rounded
        };
        f64::from_bits(sign | value)
    }
    fn reduce(mut self) -> Self {
        let e = self.ex;
        if e <= 1 {
            let u = self.hi as u128 * T[1] as u128;
            let tiny = u as u64;
            let u = self.hi as u128 * T[0] as u128;
            let (lo, c) =
                (u as u64).overflowing_add(((self.hi as u128 * T[1] as u128) >> 64) as u64);
            self.lo = lo;
            self.hi = (u >> 64) as u64 + u64::from(c);
            self = self.normalize();
            let shift = e - self.ex;
            if shift != 0 {
                self.lo |= tiny >> (64 - shift);
            }
            return self;
        }
        let i = if e < 127 {
            0
        } else {
            ((e - 127 + 63) / 64) as usize
        };
        let mut c = [0u64; 5];
        let u = self.hi as u128 * T[i + 3] as u128;
        c[0] = u as u64;
        c[1] = (u >> 64) as u64;
        for j in 1..=3 {
            let u = self.hi as u128 * T[i + 3 - j] as u128;
            let (v, carry) = c[j].overflowing_add(u as u64);
            c[j] = v;
            c[j + 1] = (u >> 64) as u64 + u64::from(carry);
        }
        let f = e - 64 * i as i64;
        let tiny = if f < 64 {
            self.hi = (c[4] << f) | (c[3] >> (64 - f));
            self.lo = (c[3] << f) | (c[2] >> (64 - f));
            (c[2] << f) | (c[1] >> (64 - f))
        } else if f == 64 {
            self.hi = c[3];
            self.lo = c[2];
            c[1]
        } else {
            let g = f - 64;
            let u = ((self.hi as u128 * T[i + 4] as u128) >> 64) as u64;
            let (v, mut carry) = c[0].overflowing_add(u);
            c[0] = v;
            for v in &mut c[1..] {
                let (n, c2) = v.overflowing_add(u64::from(carry));
                *v = n;
                carry = c2;
            }
            self.hi = (c[3] << g) | (c[2] >> (64 - g));
            self.lo = (c[2] << g) | (c[1] >> (64 - g));
            (c[1] << g) | (c[0] >> (64 - g))
        };
        self.ex = 0;
        self = self.normalize();
        if self.ex < 0 {
            self.lo |= tiny >> (64 + self.ex);
        }
        self
    }
    fn reduce2(mut self) -> (Self, usize) {
        if self.ex <= -11 {
            return (self, 0);
        }
        let sh = 53 - self.ex;
        let i = (self.hi >> sh) as usize;
        self.hi &= (1 << sh) - 1;
        (self.normalize(), i)
    }
}
fn two(a: f64, b: f64) -> (f64, f64) {
    let h = a + b;
    (h, b - (h - a))
}
fn amul(a: f64, b: f64) -> (f64, f64) {
    let h = a * b;
    (h, fma(a, b, -h))
}
fn smul(a: f64, bh: f64, bl: f64) -> (f64, f64) {
    let (h, l) = amul(a, bh);
    (h, fma(a, bl, l))
}
fn dmul(ah: f64, al: f64, bh: f64, bl: f64) -> (f64, f64) {
    let (h, s) = amul(ah, bh);
    (h, fma(ah, bl, fma(al, bh, s)))
}
fn polyfast(p: &[f64; 5], uh: f64, ul: f64) -> (f64, f64) {
    let h = fma(fma(p[4], uh, p[3]), uh, p[2]);
    let (h, l) = smul(h, uh, ul);
    let (h, t) = two(p[0], h);
    (h, l + (p[1] + t))
}
fn set_dd(c1: u64, c0: u64) -> (f64, f64) {
    let b = ((c1 as u128) << 64) | c0 as u128;
    if b == 0 {
        return (0.0, 0.0);
    }
    let sh = b.leading_zeros();
    let b = b << sh;
    let exponent = 1022 - i64::from(sh);
    let h =
        f64::from_bits(((exponent as u64) << 52) | (((b >> 64) as u64 >> 11) & 0xfffffffffffff));
    let rest = b & ((1u128 << 75) - 1);
    if rest == 0 {
        return (h, 0.0);
    }
    let z = rest.leading_zeros();
    let normalized = rest << z;
    let l = f64::from_bits(
        (((exponent - i64::from(z)) as u64) << 52)
            | (((normalized >> 64) as u64 >> 11) & 0xfffffffffffff),
    );
    (h, l)
}
fn reduce_fast(x: f64) -> (f64, f64, usize) {
    let (mut h, l) = if x <= f64::from_bits(0x401921fb54442d17) {
        let (h, l) = amul(f64::from_bits(0x3fc45f306dc9c883), x);
        (h, fma(-f64::from_bits(0x3c66b01ec5417056), x, l))
    } else {
        let bits = x.to_bits();
        let mut e = ((bits >> 52) & 2047) as i64;
        let m = (1u64 << 52) | (bits & 0xfffffffffffff);
        let mut c = [0u64; 3];
        if e <= 1074 {
            let v = m as u128 * T[2] as u128;
            let u = m as u128 * T[1] as u128;
            let (c0, carry) = (u as u64).overflowing_add((v >> 64) as u64);
            c[0] = c0;
            c[1] = (u >> 64) as u64 + u64::from(carry);
            let u = m as u128 * T[0] as u128;
            let (c1, carry) = c[1].overflowing_add(u as u64);
            c[1] = c1;
            c[2] = (u >> 64) as u64 + u64::from(carry);
            e = 1075 - e;
        } else {
            let i = ((e - 1138 + 63) / 64) as usize;
            let v = m as u128 * T[i + 3] as u128;
            let u = m as u128 * T[i + 2] as u128;
            let (c0, carry) = (u as u64).overflowing_add((v >> 64) as u64);
            c[0] = c0;
            c[1] = (u >> 64) as u64 + u64::from(carry);
            let u = m as u128 * T[i + 1] as u128;
            let (c1, carry) = c[1].overflowing_add(u as u64);
            c[1] = c1;
            c[2] = (u >> 64) as u64 + u64::from(carry);
            let u = m as u128 * T[i] as u128;
            c[2] = c[2].wrapping_add(u as u64);
            e = 1139 + 64 * i as i64 - e;
        }
        if e == 64 {
            c[0] = c[1];
            c[1] = c[2];
        } else {
            c[0] = (c[1] << (64 - e)) | (c[0] >> e);
            c[1] = (c[2] << (64 - e)) | (c[1] >> e);
        }
        set_dd(c[1], c[0])
    };
    let i = (h * 2048.0) as usize;
    h = fma(i as f64, -f64::from_bits(0x3f40000000000000), h);
    (h, l, i)
}
fn divfast(bh: f64, bl: f64, ah: f64, al: f64) -> (f64, f64) {
    let y = 1.0 / ah;
    let h = bh * y;
    let eh = fma(ah, -h, bh);
    let el = fma(al, -h, bl);
    (h, y * (eh + el))
}
fn fast(x: f64) -> (f64, f64, f64) {
    let (mut h, mut l, mut i) = reduce_fast(x.abs());
    let mut neg = x < 0.0;
    let mut is_tan = true;
    i &= 1023;
    is_tan ^= i >> 9 != 0;
    neg ^= i >> 9 != 0;
    i &= 511;
    if i & 256 != 0 {
        is_tan = !is_tan;
        i = 511 - i;
        h = f64::from_bits(0x3f40000000000000) - h;
        l = -l;
    }
    if i == 0 && h < f64::from_bits(0x3da0000000000000) {
        return (h, l, 1.0);
    }
    h -= SC[i][0];
    (h, l) = two(h, l);
    let (uh, ul) = amul(h, h);
    let ul = fma(h + h, l, ul);
    let (sh, sl) = polyfast(&PSFAST, uh, ul);
    let (sh, sl) = dmul(sh, sl, h, l);
    let (ch, cl) = polyfast(&PCFAST, uh, ul);
    let (sh0, sl0) = smul(SC[i][2], sh, sl);
    let (ch0, cl0) = smul(SC[i][1], ch, cl);
    let (h1, l1) = two(ch0, sh0);
    let l1 = l1 + (sl0 + cl0);
    let (ch, cl) = smul(SC[i][2], ch, cl);
    let (sh, sl) = smul(SC[i][1], sh, sl);
    let (h2, l2) = two(ch, -sh);
    let l2 = l2 + (cl - sl);
    let (h, l) = if is_tan {
        divfast(h1, l1, h2, l2)
    } else {
        divfast(h2, l2, h1, l1)
    };
    let s = if neg { -1.0 } else { 1.0 };
    let h = h * s;
    let l = l * s;
    (h, l, h * f64::from_bits(0x3bd1a00000000000))
}
fn poly(p: &[D; 6], x2: D) -> D {
    let mut y = x2.mul(p[5]);
    for i in (0..5).rev() {
        y = y.add(p[i]);
        if i != 0 {
            y = y.mul(x2);
        }
    }
    y
}
#[inline(never)]
fn accurate(x: f64) -> f64 {
    let (xx, mut i) = D::from_double(x.abs()).reduce().reduce2();
    let mut xx = xx;
    let mut is_tan = true;
    let mut neg = x < 0.0;
    i &= 1023;
    if i & 512 != 0 {
        is_tan = false;
        neg = !neg;
        i &= 511;
    }
    if i & 256 != 0 {
        is_tan = !is_tan;
        xx = MAGIC.add(D { sgn: 1, ..xx });
        i = 511 - i;
    }
    let x2 = xx.mul(xx);
    let u = poly(&PC, x2);
    let v = poly(&PS, x2).mul(xx);
    let sin = S[i].mul(u).add(C[i].mul(v));
    let cos = C[i].mul(u).add(S[i].mul(v).neg());
    let mut u = if is_tan { sin.div(cos) } else { cos.div(sin) };
    let lo0 = u.lo.wrapping_sub(86);
    let hi0 = u.hi.wrapping_sub(u64::from(lo0 > u.lo));
    let lo1 = u.lo.wrapping_add(86);
    let hi1 = u.hi.wrapping_add(u64::from(lo1 < u.lo));
    if hi0 >> 10 != hi1 >> 10 {
        for (a, b, c) in [
            (
                f64::from_bits(0x3e9dffffffffff1f),
                f64::from_bits(0x3e9e000000000151),
                f64::from_bits(0x3b3fffffffffffff),
            ),
            (
                f64::from_bits(0x3eadfffffffffc7c),
                f64::from_bits(0x3eae000000000546),
                -f64::from_bits(0x36c658bcedb6e1d4),
            ),
        ] {
            if x.abs() == a {
                return if x > 0.0 { b + c } else { -b - c };
            }
        }
    }
    if neg {
        u = u.neg();
    }
    u.to_double()
}
/// Correctly rounded tangent, nearest-even with gradual underflow.
pub fn tan(x: f64) -> f64 {
    if !x.is_finite() {
        return if x.is_nan() { x + x } else { f64::NAN };
    }
    if x.to_bits() & 0x7fff_ffff_ffff_ffff <= 0x3e4d12ed0af1a27e {
        if x == 0.0 {
            return x;
        }
        return fma(x, f64::from_bits(0x3c90000000000000), x);
    }
    let (h, l, err) = fast(x);
    let left = h + (l - err);
    let right = h + (l + err);
    if left == right { left } else { accurate(x) }
}

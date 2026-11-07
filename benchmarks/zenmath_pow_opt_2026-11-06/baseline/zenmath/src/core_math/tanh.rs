/* Correctly rounded hyperbolic tangent function for binary64 values.

Copyright (c) 2023-2026 Alexei Sibidanov, Cyprien Peignier, Paul Zimmermann

Alexei Sibidanov designed the original algorithm, while Cyprien Peignier and
Paul Zimmermann extended the fma formula for |x0| <= 0x1.d12ed0af1a27fp-27,
and improved the minimax polynomial for x0 <= |x| < 0.25.

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
const CHE: [[f64; 2]; 3] = [
    [
        f64::from_bits(0x3ff0000000000000),
        f64::from_bits(0x3a16c16bd194535d),
    ],
    [
        f64::from_bits(0x3fe0000000000000),
        f64::from_bits(0xba28259d904fd34f),
    ],
    [
        f64::from_bits(0x3fc5555555555555),
        f64::from_bits(0x3c653e93e9f26e62),
    ],
];
const CH0: [[f64; 2]; 9] = [
    [
        f64::from_bits(0xbfd5555555555555),
        f64::from_bits(0xbc75555555554cc4),
    ],
    [
        f64::from_bits(0x3fc1111111111111),
        f64::from_bits(0x3c4111110f8c0178),
    ],
    [
        f64::from_bits(0xbfaba1ba1ba1ba1c),
        f64::from_bits(0x3c47917c1d676ff5),
    ],
    [
        f64::from_bits(0x3f9664f4882c10fa),
        f64::from_bits(0xbc09d5cb27c0af28),
    ],
    [
        f64::from_bits(0xbf8226e355e6c23c),
        f64::from_bits(0xbc2c9674586913f3),
    ],
    [
        f64::from_bits(0x3f6d6d3d0e157db3),
        f64::from_bits(0xbbe71376fa06ce94),
    ],
    [
        f64::from_bits(0xbf57da36452b5e46),
        f64::from_bits(0xbbeaba8d51bd9c00),
    ],
    [
        f64::from_bits(0x3f43558247faa32d),
        f64::from_bits(0xbbee0cfb423aedfd),
    ],
    [
        f64::from_bits(0xbf2f57d76ea30928),
        f64::from_bits(0xbbcc30601213cae9),
    ],
];
const CL0: [f64; 5] = [
    f64::from_bits(0x3f1967e0a63ca836),
    f64::from_bits(0xbf0497b99d2a77d1),
    f64::from_bits(0x3ef0ae346258cbde),
    f64::from_bits(0xbedaade68fb2f076),
    f64::from_bits(0x3ec22e609bf8671f),
];
const DB: [[f64; 3]; 12] = [
    [
        f64::from_bits(0x3fcac343b179fec4),
        f64::from_bits(0x3fca612499c53078),
        f64::from_bits(0x3c60000000000000),
    ],
    [
        f64::from_bits(0x3fd00764a988bf73),
        f64::from_bits(0x3fcf676484c0703b),
        f64::from_bits(0xb970000000000000),
    ],
    [
        f64::from_bits(0x3fd17d1e8a63711f),
        f64::from_bits(0x3fd110e96a6c2d96),
        f64::from_bits(0xb960000000000000),
    ],
    [
        f64::from_bits(0x3fd291c601a05276),
        f64::from_bits(0x3fd210b7d0c03743),
        f64::from_bits(0x3c70000000000000),
    ],
    [
        f64::from_bits(0x3fd36f33d51c264d),
        f64::from_bits(0x3fd2dbb7b1c91363),
        f64::from_bits(0xb950000000000000),
    ],
    [
        f64::from_bits(0x3fd43eaea23649c3),
        f64::from_bits(0x3fd39877ed028641),
        f64::from_bits(0xbc80000000000000),
    ],
    [
        f64::from_bits(0x3fdd88d7550b2826),
        f64::from_bits(0x3fdb9a3637366afd),
        f64::from_bits(0x3c70000000000000),
    ],
    [
        f64::from_bits(0x3fde611aa58ab608),
        f64::from_bits(0x3fdc493dc899e4a6),
        f64::from_bits(0xbc80000000000000),
    ],
    [
        f64::from_bits(0x3fe01efe7ac8c15d),
        f64::from_bits(0x3fddc3fe1b524821),
        f64::from_bits(0xb970000000000000),
    ],
    [
        f64::from_bits(0x3fe1005ec0bccabb),
        f64::from_bits(0x3fdf20b1c8557ded),
        f64::from_bits(0xbc80000000000000),
    ],
    [
        f64::from_bits(0x3fe33dfeb0fa4bfe),
        f64::from_bits(0x3fe1372f9ee76e99),
        f64::from_bits(0x3c80000000000000),
    ],
    [
        f64::from_bits(0x3fe49f24ac5cac35),
        f64::from_bits(0x3fe22c495ff06104),
        f64::from_bits(0xb970000000000000),
    ],
];
const T0: [[f64; 2]; 64] = [
    [
        f64::from_bits(0x0000000000000000),
        f64::from_bits(0x3ff0000000000000),
    ],
    [
        f64::from_bits(0xbc719083535b085e),
        f64::from_bits(0x3ff02c9a3e778061),
    ],
    [
        f64::from_bits(0x3c8d73e2a475b466),
        f64::from_bits(0x3ff059b0d3158574),
    ],
    [
        f64::from_bits(0x3c6186be4bb28500),
        f64::from_bits(0x3ff0874518759bc8),
    ],
    [
        f64::from_bits(0x3c98a62e4adc610a),
        f64::from_bits(0x3ff0b5586cf9890f),
    ],
    [
        f64::from_bits(0x3c403a1727c57b52),
        f64::from_bits(0x3ff0e3ec32d3d1a2),
    ],
    [
        f64::from_bits(0xbc96c51039449b3a),
        f64::from_bits(0x3ff11301d0125b51),
    ],
    [
        f64::from_bits(0xbc932fbf9af1369e),
        f64::from_bits(0x3ff1429aaea92de0),
    ],
    [
        f64::from_bits(0xbc819041b9d78a76),
        f64::from_bits(0x3ff172b83c7d517b),
    ],
    [
        f64::from_bits(0x3c8e5b4c7b4968e4),
        f64::from_bits(0x3ff1a35beb6fcb75),
    ],
    [
        f64::from_bits(0x3c9e016e00a2643c),
        f64::from_bits(0x3ff1d4873168b9aa),
    ],
    [
        f64::from_bits(0x3c8dc775814a8494),
        f64::from_bits(0x3ff2063b88628cd6),
    ],
    [
        f64::from_bits(0x3c99b07eb6c70572),
        f64::from_bits(0x3ff2387a6e756238),
    ],
    [
        f64::from_bits(0x3c82bd339940e9da),
        f64::from_bits(0x3ff26b4565e27cdd),
    ],
    [
        f64::from_bits(0x3c8612e8afad1256),
        f64::from_bits(0x3ff29e9df51fdee1),
    ],
    [
        f64::from_bits(0x3c90024754db41d4),
        f64::from_bits(0x3ff2d285a6e4030b),
    ],
    [
        f64::from_bits(0x3c86f46ad23182e4),
        f64::from_bits(0x3ff306fe0a31b715),
    ],
    [
        f64::from_bits(0x3c932721843659a6),
        f64::from_bits(0x3ff33c08b26416ff),
    ],
    [
        f64::from_bits(0xbc963aeabf42eae2),
        f64::from_bits(0x3ff371a7373aa9cb),
    ],
    [
        f64::from_bits(0xbc75e436d661f5e2),
        f64::from_bits(0x3ff3a7db34e59ff7),
    ],
    [
        f64::from_bits(0x3c8ada0911f09ebc),
        f64::from_bits(0x3ff3dea64c123422),
    ],
    [
        f64::from_bits(0xbc5ef3691c309278),
        f64::from_bits(0x3ff4160a21f72e2a),
    ],
    [
        f64::from_bits(0x3c489b7a04ef80d0),
        f64::from_bits(0x3ff44e086061892d),
    ],
    [
        f64::from_bits(0x3c73c1a3b69062f0),
        f64::from_bits(0x3ff486a2b5c13cd0),
    ],
    [
        f64::from_bits(0x3c7d4397afec42e2),
        f64::from_bits(0x3ff4bfdad5362a27),
    ],
    [
        f64::from_bits(0xbc94b309d25957e4),
        f64::from_bits(0x3ff4f9b2769d2ca7),
    ],
    [
        f64::from_bits(0xbc807abe1db13cac),
        f64::from_bits(0x3ff5342b569d4f82),
    ],
    [
        f64::from_bits(0x3c99bb2c011d93ac),
        f64::from_bits(0x3ff56f4736b527da),
    ],
    [
        f64::from_bits(0x3c96324c054647ac),
        f64::from_bits(0x3ff5ab07dd485429),
    ],
    [
        f64::from_bits(0x3c9ba6f93080e65e),
        f64::from_bits(0x3ff5e76f15ad2148),
    ],
    [
        f64::from_bits(0xbc9383c17e40b496),
        f64::from_bits(0x3ff6247eb03a5585),
    ],
    [
        f64::from_bits(0xbc9bb60987591c34),
        f64::from_bits(0x3ff6623882552225),
    ],
    [
        f64::from_bits(0xbc9bdd3413b26456),
        f64::from_bits(0x3ff6a09e667f3bcd),
    ],
    [
        f64::from_bits(0xbc6bbe3a683c88aa),
        f64::from_bits(0x3ff6dfb23c651a2f),
    ],
    [
        f64::from_bits(0xbc816e4786887a9a),
        f64::from_bits(0x3ff71f75e8ec5f74),
    ],
    [
        f64::from_bits(0xbc90245957316dd4),
        f64::from_bits(0x3ff75feb564267c9),
    ],
    [
        f64::from_bits(0xbc841577ee049930),
        f64::from_bits(0x3ff7a11473eb0187),
    ],
    [
        f64::from_bits(0x3c705d02ba15797e),
        f64::from_bits(0x3ff7e2f336cf4e62),
    ],
    [
        f64::from_bits(0xbc9d4c1dd41532d8),
        f64::from_bits(0x3ff82589994cce13),
    ],
    [
        f64::from_bits(0xbc9fc6f89bd4f6ba),
        f64::from_bits(0x3ff868d99b4492ed),
    ],
    [
        f64::from_bits(0x3c96e9f156864b26),
        f64::from_bits(0x3ff8ace5422aa0db),
    ],
    [
        f64::from_bits(0x3c85cc13a2e3976c),
        f64::from_bits(0x3ff8f1ae99157736),
    ],
    [
        f64::from_bits(0xbc675fc781b57ebc),
        f64::from_bits(0x3ff93737b0cdc5e5),
    ],
    [
        f64::from_bits(0xbc9d185b7c1b85d0),
        f64::from_bits(0x3ff97d829fde4e50),
    ],
    [
        f64::from_bits(0x3c7c7c46b071f2be),
        f64::from_bits(0x3ff9c49182a3f090),
    ],
    [
        f64::from_bits(0xbc9359495d1cd532),
        f64::from_bits(0x3ffa0c667b5de565),
    ],
    [
        f64::from_bits(0xbc9d2f6edb8d41e2),
        f64::from_bits(0x3ffa5503b23e255d),
    ],
    [
        f64::from_bits(0x3c90fac90ef7fd32),
        f64::from_bits(0x3ffa9e6b5579fdbf),
    ],
    [
        f64::from_bits(0x3c97a1cd345dcc82),
        f64::from_bits(0x3ffae89f995ad3ad),
    ],
    [
        f64::from_bits(0xbc62805e3084d708),
        f64::from_bits(0x3ffb33a2b84f15fb),
    ],
    [
        f64::from_bits(0xbc75584f7e54ac3a),
        f64::from_bits(0x3ffb7f76f2fb5e47),
    ],
    [
        f64::from_bits(0x3c823dd07a2d9e84),
        f64::from_bits(0x3ffbcc1e904bc1d2),
    ],
    [
        f64::from_bits(0x3c811065895048de),
        f64::from_bits(0x3ffc199bdd85529c),
    ],
    [
        f64::from_bits(0x3c92884dff483cac),
        f64::from_bits(0x3ffc67f12e57d14b),
    ],
    [
        f64::from_bits(0x3c7503cbd1e949dc),
        f64::from_bits(0x3ffcb720dcef9069),
    ],
    [
        f64::from_bits(0xbc9cbc3743797a9c),
        f64::from_bits(0x3ffd072d4a07897c),
    ],
    [
        f64::from_bits(0x3c82ed02d75b3706),
        f64::from_bits(0x3ffd5818dcfba487),
    ],
    [
        f64::from_bits(0x3c9c2300696db532),
        f64::from_bits(0x3ffda9e603db3285),
    ],
    [
        f64::from_bits(0xbc91a5cd4f184b5c),
        f64::from_bits(0x3ffdfc97337b9b5f),
    ],
    [
        f64::from_bits(0x3c839e8980a9cc90),
        f64::from_bits(0x3ffe502ee78b3ff6),
    ],
    [
        f64::from_bits(0xbc9e9c23179c2894),
        f64::from_bits(0x3ffea4afa2a490da),
    ],
    [
        f64::from_bits(0x3c9dc7f486a4b6b0),
        f64::from_bits(0x3ffefa1bee615a27),
    ],
    [
        f64::from_bits(0x3c99d3e12dd8a18a),
        f64::from_bits(0x3fff50765b6e4540),
    ],
    [
        f64::from_bits(0x3c874853f3a5931e),
        f64::from_bits(0x3fffa7c1819e90d8),
    ],
];
const T1: [[f64; 2]; 64] = [
    [
        f64::from_bits(0x0000000000000000),
        f64::from_bits(0x3ff0000000000000),
    ],
    [
        f64::from_bits(0x3c9ae8e38c59c72a),
        f64::from_bits(0x3ff000b175effdc7),
    ],
    [
        f64::from_bits(0xbc57b5d0d58ea8f4),
        f64::from_bits(0x3ff00162f3904052),
    ],
    [
        f64::from_bits(0x3c94115cb6b16a8e),
        f64::from_bits(0x3ff0021478e11ce6),
    ],
    [
        f64::from_bits(0xbc8d7c96f201bb2e),
        f64::from_bits(0x3ff002c605e2e8cf),
    ],
    [
        f64::from_bits(0x3c984711d4c35ea0),
        f64::from_bits(0x3ff003779a95f959),
    ],
    [
        f64::from_bits(0xbc80484245243778),
        f64::from_bits(0x3ff0042936faa3d8),
    ],
    [
        f64::from_bits(0xbc94b237da2025fa),
        f64::from_bits(0x3ff004dadb113da0),
    ],
    [
        f64::from_bits(0xbc75e00e62d6b30e),
        f64::from_bits(0x3ff0058c86da1c0a),
    ],
    [
        f64::from_bits(0x3c9a1d6cedbb9480),
        f64::from_bits(0x3ff0063e3a559473),
    ],
    [
        f64::from_bits(0xbc94acf197a00142),
        f64::from_bits(0x3ff006eff583fc3d),
    ],
    [
        f64::from_bits(0xbc6eaf2ea42391a6),
        f64::from_bits(0x3ff007a1b865a8ca),
    ],
    [
        f64::from_bits(0x3c7da93f90835f76),
        f64::from_bits(0x3ff0085382faef83),
    ],
    [
        f64::from_bits(0xbc86a79084ab093c),
        f64::from_bits(0x3ff00905554425d4),
    ],
    [
        f64::from_bits(0x3c986364f8fbe8f8),
        f64::from_bits(0x3ff009b72f41a12b),
    ],
    [
        f64::from_bits(0xbc882e8e14e3110e),
        f64::from_bits(0x3ff00a6910f3b6fd),
    ],
    [
        f64::from_bits(0xbc84f6b2a7609f72),
        f64::from_bits(0x3ff00b1afa5abcbf),
    ],
    [
        f64::from_bits(0xbc7e1a258ea8f71a),
        f64::from_bits(0x3ff00bcceb7707ec),
    ],
    [
        f64::from_bits(0x3c74362ca5bc26f2),
        f64::from_bits(0x3ff00c7ee448ee02),
    ],
    [
        f64::from_bits(0x3c9095a56c919d02),
        f64::from_bits(0x3ff00d30e4d0c483),
    ],
    [
        f64::from_bits(0xbc6406ac4e81a646),
        f64::from_bits(0x3ff00de2ed0ee0f5),
    ],
    [
        f64::from_bits(0x3c9b5a6902767e08),
        f64::from_bits(0x3ff00e94fd0398e0),
    ],
    [
        f64::from_bits(0xbc991b2060859320),
        f64::from_bits(0x3ff00f4714af41d3),
    ],
    [
        f64::from_bits(0x3c8427068ab22306),
        f64::from_bits(0x3ff00ff93412315c),
    ],
    [
        f64::from_bits(0x3c9c1d0660524e08),
        f64::from_bits(0x3ff010ab5b2cbd11),
    ],
    [
        f64::from_bits(0xbc9e7bdfb3204be8),
        f64::from_bits(0x3ff0115d89ff3a8b),
    ],
    [
        f64::from_bits(0x3c8843aa8b9cbbc6),
        f64::from_bits(0x3ff0120fc089ff63),
    ],
    [
        f64::from_bits(0xbc734104ee7edae8),
        f64::from_bits(0x3ff012c1fecd613b),
    ],
    [
        f64::from_bits(0xbc72b6aeb6176892),
        f64::from_bits(0x3ff0137444c9b5b5),
    ],
    [
        f64::from_bits(0x3c7a8cd33b8a1bb2),
        f64::from_bits(0x3ff01426927f5278),
    ],
    [
        f64::from_bits(0x3c72edc08e5da99a),
        f64::from_bits(0x3ff014d8e7ee8d2f),
    ],
    [
        f64::from_bits(0x3c857ba2dc7e0c72),
        f64::from_bits(0x3ff0158b4517bb88),
    ],
    [
        f64::from_bits(0x3c9b61299ab8cdb8),
        f64::from_bits(0x3ff0163da9fb3335),
    ],
    [
        f64::from_bits(0xbc990565902c5f44),
        f64::from_bits(0x3ff016f0169949ed),
    ],
    [
        f64::from_bits(0x3c870fc41c5c2d54),
        f64::from_bits(0x3ff017a28af25567),
    ],
    [
        f64::from_bits(0x3c94b9a6e145d76c),
        f64::from_bits(0x3ff018550706ab62),
    ],
    [
        f64::from_bits(0xbc7008eff5142bfa),
        f64::from_bits(0x3ff019078ad6a19f),
    ],
    [
        f64::from_bits(0xbc977669f033c7de),
        f64::from_bits(0x3ff019ba16628de2),
    ],
    [
        f64::from_bits(0xbc909bb78eeead0a),
        f64::from_bits(0x3ff01a6ca9aac5f3),
    ],
    [
        f64::from_bits(0x3c9371231477ece6),
        f64::from_bits(0x3ff01b1f44af9f9e),
    ],
    [
        f64::from_bits(0x3c75e7626621eb5a),
        f64::from_bits(0x3ff01bd1e77170b4),
    ],
    [
        f64::from_bits(0xbc9bc72b100828a4),
        f64::from_bits(0x3ff01c8491f08f08),
    ],
    [
        f64::from_bits(0xbc6ce39cbbab8bbe),
        f64::from_bits(0x3ff01d37442d5070),
    ],
    [
        f64::from_bits(0x3c816996709da2e2),
        f64::from_bits(0x3ff01de9fe280ac8),
    ],
    [
        f64::from_bits(0xbc8c11f5239bf536),
        f64::from_bits(0x3ff01e9cbfe113ef),
    ],
    [
        f64::from_bits(0x3c8e1d4eb5edc6b4),
        f64::from_bits(0x3ff01f4f8958c1c6),
    ],
    [
        f64::from_bits(0xbc9afb99946ee3f0),
        f64::from_bits(0x3ff020025a8f6a35),
    ],
    [
        f64::from_bits(0xbc98f06d8a148a32),
        f64::from_bits(0x3ff020b533856324),
    ],
    [
        f64::from_bits(0xbc82bf310fc54eb6),
        f64::from_bits(0x3ff02168143b0281),
    ],
    [
        f64::from_bits(0xbc9c95a035eb4176),
        f64::from_bits(0x3ff0221afcb09e3e),
    ],
    [
        f64::from_bits(0xbc9491793e46834c),
        f64::from_bits(0x3ff022cdece68c4f),
    ],
    [
        f64::from_bits(0xbc73e8d0d9c49090),
        f64::from_bits(0x3ff02380e4dd22ad),
    ],
    [
        f64::from_bits(0xbc9314aa16278aa4),
        f64::from_bits(0x3ff02433e494b755),
    ],
    [
        f64::from_bits(0x3c848daf888e9650),
        f64::from_bits(0x3ff024e6ec0da046),
    ],
    [
        f64::from_bits(0x3c856dc8046821f4),
        f64::from_bits(0x3ff02599fb483385),
    ],
    [
        f64::from_bits(0x3c945b42356b9d46),
        f64::from_bits(0x3ff0264d1244c719),
    ],
    [
        f64::from_bits(0xbc7082ef51b61d7e),
        f64::from_bits(0x3ff027003103b10e),
    ],
    [
        f64::from_bits(0x3c72106ed0920a34),
        f64::from_bits(0x3ff027b357854772),
    ],
    [
        f64::from_bits(0xbc9fd4cf26ea5d0e),
        f64::from_bits(0x3ff0286685c9e059),
    ],
    [
        f64::from_bits(0xbc909f8775e78084),
        f64::from_bits(0x3ff02919bbd1d1d8),
    ],
    [
        f64::from_bits(0x3c564cbba902ca28),
        f64::from_bits(0x3ff029ccf99d720a),
    ],
    [
        f64::from_bits(0x3c94383ef231d206),
        f64::from_bits(0x3ff02a803f2d170d),
    ],
    [
        f64::from_bits(0x3c94a47a505b3a46),
        f64::from_bits(0x3ff02b338c811703),
    ],
    [
        f64::from_bits(0x3c9e471202234680),
        f64::from_bits(0x3ff02be6e199c811),
    ],
];
const CH: [f64; 4] = [
    f64::from_bits(0x4000000000000000),
    f64::from_bits(0x4000000000000000),
    f64::from_bits(0x3ff55555557e54ff),
    f64::from_bits(0x3fe55555553a12f4),
];
const C: [f64; 8] = [
    f64::from_bits(0xbfd5555555555555),
    f64::from_bits(0x3fc1111111110f33),
    f64::from_bits(0xbfaba1ba1b9b8ea6),
    f64::from_bits(0x3f9664f4838e0a43),
    f64::from_bits(0xbf8226e17d1bc09b),
    f64::from_bits(0x3f6d6c64dfba2565),
    f64::from_bits(0xbf57bdd094d327af),
    f64::from_bits(0x3f41535ad0c31d0e),
];
fn two(x: f64, y: f64) -> (f64, f64) {
    let s = x + y;
    (s, y - (s - x))
}
fn sub(x: f64, y: f64) -> (f64, f64) {
    let s = x - y;
    (s, (x - s) - y)
}
fn mul(xh: f64, xl: f64, ch: f64, cl: f64) -> (f64, f64) {
    let h = xh * ch;
    let rest = fma(xh, ch, -h) + (xh * cl + xl * ch);
    two(h, rest)
}
fn muld(xh: f64, xl: f64, c: f64) -> (f64, f64) {
    let h = xh * c;
    two(h, fma(xh, c, -h) + xl * c)
}
fn poly(xh: f64, xl: f64, c: &[[f64; 2]], l: f64) -> (f64, f64) {
    let i = c.len() - 1;
    let (mut ch, t) = two(c[i][0], l);
    let mut cl = t + c[i][1];
    for i in (0..i).rev() {
        (ch, cl) = mul(xh, xl, ch, cl);
        let (th, tl) = two(c[i][0], ch);
        ch = th;
        cl = (cl + c[i][1]) + tl;
    }
    (ch, cl)
}
fn database(x: f64, f: f64) -> f64 {
    let s = 1.0f64.copysign(x);
    for row in DB {
        if row[0] == x.abs() {
            return s * row[1] + s * row[2];
        }
    }
    f
}
#[inline(never)]
fn zero(x: f64) -> f64 {
    let x2 = x * x;
    let x2l = fma(x, x, -x2);
    let y2 = x2 * (CL0[0] + x2 * (CL0[1] + x2 * (CL0[2] + x2 * (CL0[3] + x2 * CL0[4]))));
    let (y1, y2) = poly(x2, x2l, &CH0, y2);
    let (y1, y2) = muld(y1, y2, x);
    let (y1, y2) = mul(y1, y2, x2, x2l);
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
        if x.abs() == f64::from_bits(0x3fcac343b179fec4) {
            return database(x, y0 + y1);
        }
    }
    y0 + y1
}
#[inline(never)]
fn exp_accurate(x: f64, t: f64, th: f64, tl: f64) -> (f64, f64) {
    let dx = x - f64::from_bits(0x3f262e42ff000000) * t;
    let dxl = f64::from_bits(0x3d0718432a1b0e26) * t;
    let dxll =
        f64::from_bits(0x3999ff0342542fc3) * t + fma(f64::from_bits(0x3d0718432a1b0e26), t, -dxl);
    let dxh = dx + dxl;
    let dxl = ((dx - dxh) + dxl) + dxll;
    let fl = dxh
        * (f64::from_bits(0x3fa5555555555555)
            + dxh
                * (f64::from_bits(0x3f811111113e93e9) + dxh * f64::from_bits(0x3f56c16c169400a7)));
    let (fh, fl) = poly(dxh, dxl, &CHE, fl);
    let (fh, fl) = mul(dxh, dxl, fh, fl);
    let (fh, fl) = mul(th, tl, fh, fl);
    let zh = th + fh;
    let zl = (th - zh) + fh;
    let uh = zh + tl;
    let ul = ((zh - uh) + tl) + zl;
    let vh = uh + fl;
    let vl = ((uh - vh) + fl) + ul;
    (vh, vl)
}
/// Correctly rounded hyperbolic tangent, nearest-even.
pub fn tanh(x: f64) -> f64 {
    let ax = x.abs();
    let aix = ax.to_bits();
    if aix >= 0x40330fc1931f09ca {
        return if x.is_nan() {
            x + x
        } else {
            1.0f64.copysign(x)
        };
    }
    let v0 = fma(
        ax,
        -f64::from_bits(0x40c71547652b82fe),
        f64::from_bits(0x4188000004000000),
    );
    let jt = v0.to_bits();
    let t = f64::from_bits(jt & !((1u64 << 27) - 1)) - f64::from_bits(0x4188000000000000);
    let i1 = ((jt >> 27) & 63) as usize;
    let i0 = ((jt >> 33) & 63) as usize;
    let ie = ((jt << 13) as i64) >> 52;
    let sp = f64::from_bits(((1023 + ie) as u64) << 52);
    let t0h = T0[i0][1];
    let t1h = T1[i1][1];
    let mut th = t0h * t1h;
    let mut tl;
    if aix < 0x400d76c8b4395810 {
        if aix < 0x3fd0000000000000 {
            if aix <= 0x3e4d12ed0af1a27f {
                if aix == 0 {
                    return x;
                }
                return fma(x, -f64::from_bits(0x3c80000000000000), x);
            }
            let x2 = x * x;
            let x3 = x2 * x;
            let x4 = x2 * x2;
            let x8 = x4 * x4;
            let p1 = (C[4] + x2 * C[5]) + x4 * (C[6] + x2 * C[7]);
            let mut p0 = (C[0] + x2 * C[1]) + x4 * (C[2] + x2 * C[3]);
            p0 += x8 * p1;
            p0 *= x3;
            let (rh, rl) = two(x, p0);
            let e = x3 * f64::from_bits(0x3cbc000000000000);
            let lb = rh + (rl - e);
            let ub = rh + (rl + e);
            return if lb == ub { lb } else { zero(x) };
        }
        tl = t0h * T1[i1][0] + t1h * T0[i0][0] + fma(t0h, t1h, -th);
        th *= sp;
        tl *= sp;
        let dx = (-f64::from_bits(0x3f162e42ff000000) * t - ax)
            - (-f64::from_bits(0x3cf718432a1b0e26)) * t;
        let dx2 = dx * dx;
        let p = dx * ((CH[0] + dx * CH[1]) + dx2 * (CH[2] + dx * CH[3]));
        let (rh, rl) = two(th, tl + th * p);
        let (qh, qd) = two(1.0, rh);
        let ql = rl + qd;
        let rqh = 1.0 / qh;
        let rql = (ql * rqh + fma(rqh, qh, -1.0)) * (-rqh);
        let (ph, pl) = mul(rh, rl, rqh, rql);
        let e = rh * f64::from_bits(0x3c10d00000000000);
        let (rh, rl) = sub(0.5, ph);
        let s = 2.0f64.copysign(x);
        let rh = rh * s;
        let rl = (rl - pl) * s;
        let lb = rh + (rl - e);
        let ub = rh + (rl + e);
        if lb == ub {
            return lb;
        }
    } else {
        let dx = fma(-f64::from_bits(0x3f162e42fefa39ef), t, -ax);
        let dx2 = dx * dx;
        let p = dx * ((CH[0] + dx * CH[1]) + dx2 * (CH[2] + dx * CH[3]));
        let mut rh = th * sp;
        rh += (p + (2.0 * f64::from_bits(0x3c83000000000000)) * ax) * rh;
        let e = rh * f64::from_bits(0x3ce1000000000000);
        rh = (2.0 * rh) / (1.0 + rh);
        let one = 1.0f64.copysign(x);
        rh = rh.copysign(x);
        let lb = one - (rh + e);
        let ub = one - (rh - e);
        if lb == ub {
            return lb;
        }
        tl = t0h * T1[i1][0] + t1h * T0[i0][0] + fma(t0h, t1h, -th);
        th *= sp;
        tl *= sp;
    }
    let (rh, rl) = exp_accurate(-2.0 * ax, t, th, tl);
    let (qh, qd) = two(1.0, rh);
    let ql = rl + qd;
    let (qh, ql) = two(qh, ql);
    let rqh = 1.0 / qh;
    let rql = (ql * rqh + fma(rqh, qh, -1.0)) * (-rqh);
    let (ph, pl) = mul(rh, rl, rqh, rql);
    let (rh, rl) = sub(0.5, ph);
    let (rh, rl) = two(rh, rl - pl);
    let s = 2.0f64.copysign(x);
    let res = s * rh + s * rl;
    if rl.to_bits().wrapping_add(32) & 0xfffffffffffff < 65 {
        database(x, res)
    } else {
        res
    }
}

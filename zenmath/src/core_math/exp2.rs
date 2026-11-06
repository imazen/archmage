/* Correctly rounded exp2 function for binary64 values.

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
const DB: [f64; 43] = [
    f64::from_bits(0x3f5e4596526bf94d),
    f64::from_bits(0x3f5e76049073067f),
    f64::from_bits(0x3f6755aa6fa428cd),
    f64::from_bits(0x3f679015ce2843d7),
    f64::from_bits(0x3f7f99afefa30d65),
    f64::from_bits(0x3f98d040898b73f5),
    f64::from_bits(0x3fb673a7779d5293),
    f64::from_bits(0x3fb8859f5e252908),
    f64::from_bits(0x3fbfa18dfad6e466),
    f64::from_bits(0x3fc6c4175ea0c6e1),
    f64::from_bits(0x3fc926961243baba),
    f64::from_bits(0x3fe3e34fa6ab969e),
    f64::from_bits(0x3feb32a6c92d1185),
    f64::from_bits(0x3ff9f1a7d355cb4f),
    f64::from_bits(0xbf243c1cea9bd4d9),
    f64::from_bits(0xbf277970470a37ed),
    f64::from_bits(0xbf27d44c7c8229a6),
    f64::from_bits(0xbf395a914543eab7),
    f64::from_bits(0xbf399be01d01064a),
    f64::from_bits(0xbf413f898b1e4f28),
    f64::from_bits(0xbf468e7a49000b1c),
    f64::from_bits(0xbf486d2a6e5e8368),
    f64::from_bits(0xbf5120d3bdfb6ed8),
    f64::from_bits(0xbf53ec814d260d02),
    f64::from_bits(0xbf647b667916c4b2),
    f64::from_bits(0xbf6899e0474ba2d5),
    f64::from_bits(0xbf6ba84c6ebfb038),
    f64::from_bits(0xbf7111bc29ccdbb1),
    f64::from_bits(0xbf71fb57e1996e26),
    f64::from_bits(0xbf772e40977492c3),
    f64::from_bits(0xbf7ebf8cf367fcb8),
    f64::from_bits(0xbf807f812303f10a),
    f64::from_bits(0xbf9234ada2403885),
    f64::from_bits(0xbf935dd739305031),
    f64::from_bits(0xbfa526ce079b05a5),
    f64::from_bits(0xbfb3ea95a5c16e4a),
    f64::from_bits(0xbfc33564db4bb9ec),
    f64::from_bits(0xbfcd4854d9f87fca),
    f64::from_bits(0xbfcfe89353e31cbf),
    f64::from_bits(0xbfd83960b2a8d2c4),
    f64::from_bits(0xbfde242801b45d0d),
    f64::from_bits(0xbfecef4c143b5adf),
    f64::from_bits(0xbff60e582caa34b1),
];
const T0: [[f64; 2]; 64] = [
    [f64::from_bits(0x0), f64::from_bits(0x3ff0000000000000)],
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
    [f64::from_bits(0x0), f64::from_bits(0x3ff0000000000000)],
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
const CD: [[f64; 2]; 6] = [
    [
        f64::from_bits(0x3f262e42fefa39ef),
        f64::from_bits(0x3bbabc9e3b39873e),
    ],
    [
        f64::from_bits(0x3e4ebfbdff82c58f),
        f64::from_bits(0xbae5e43a53e44950),
    ],
    [
        f64::from_bits(0x3d6c6b08d704a0c0),
        f64::from_bits(0xba0d3a15710d3d83),
    ],
    [
        f64::from_bits(0x3c83b2ab6fba4e77),
        f64::from_bits(0x3914dd5d2a5e025a),
    ],
    [
        f64::from_bits(0x3b95d87fe7a66459),
        f64::from_bits(0xb83dc47e47beb9dd),
    ],
    [
        f64::from_bits(0x3aa430912f9fb79d),
        f64::from_bits(0xb744fcd51fcb7640),
    ],
];
const C: [f64; 4] = [
    f64::from_bits(0x3f262e42fefa39ef),
    f64::from_bits(0x3e4ebfbdff82c58f),
    f64::from_bits(0x3d6c6b08d73b3e01),
    f64::from_bits(0x3c83b2ab6fdda001),
];

fn two(x: f64, y: f64) -> (f64, f64) {
    let s = x + y;
    (s, y - (s - x))
}
fn sum(xh: f64, xl: f64, yh: f64, yl: f64) -> (f64, f64) {
    let (h, l) = two(xh, yh);
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
fn scale(x: f64, i: i64) -> f64 {
    f64::from_bits(x.to_bits().wrapping_add((i as u64) << 52))
}
fn denormal(x: f64) -> f64 {
    f64::from_bits(x.to_bits() & 0xfffffffffffff)
}
#[inline(never)]
fn database(x: f64, f: f64) -> f64 {
    if let Some(m) = DB.iter().position(|v| v.to_bits() == x.to_bits()) {
        let s = [0x3b216fbd5fd7665fu64, 0x34c797];
        let k = 8677191773140u64;
        let p = (s[m >> 5] >> ((m * 2) & 63)) & 3;
        let dy = f64::from_bits((0x3c90 | ((k >> m) << 15)) << 48);
        for i in -1i64..=1 {
            let b = f.to_bits().wrapping_add(i as u64);
            if b & 3 == p {
                return f64::from_bits(b) + dy;
            }
        }
    }
    f
}
#[inline(never)]
fn accurate(x: f64) -> f64 {
    let sx = 4096.0 * x;
    let fx = crate::reference::roundeven(sx);
    let z = sx - fx;
    let k = fx as i64;
    let i1 = (k & 63) as usize;
    let i0 = ((k >> 6) & 63) as usize;
    let ie = k >> 12;
    let (th, tl) = mul(T0[i0][1], T0[i0][0], T1[i1][1], T1[i1][0]);
    let (mut fh, mut fl) = (CD[5][0], CD[5][1]);
    for c in CD[..5].iter().rev() {
        (fh, fl) = muld(z, fh, fl);
        let h = fh + c[0];
        let l = (c[0] - h) + fh;
        fh = h;
        fl += l + c[1];
    }
    (fh, fl) = muld(z, fh, fl);
    if x.to_bits() <= 0xc08ff00000000000 {
        if -f64::from_bits(0x3c971547652b82fe) <= x && x <= f64::from_bits(0x3ca71547652b82fd) {
            return fma(x, 0.5, 1.0);
        }
        if k & 0xfff == 0 {
            let (h, e) = two(th, fh);
            fh = h;
            let (l, e) = two(e, fl);
            fl = l;
            if fl.to_bits() & 0xfffffffffffff == 0 {
                let d = if (fl.to_bits() ^ e.to_bits()) >> 63 != 0 {
                    -1i64
                } else {
                    1
                };
                fl = f64::from_bits(fl.to_bits().wrapping_add(d as u64));
            }
        } else {
            (fh, fl) = mul(fh, fl, th, tl);
            (fh, fl) = sum(th, tl, fh, fl);
        }
        (fh, fl) = two(fh, fl);
        if fl.to_bits().wrapping_add(2) & 0xfffffffffffff <= 2 {
            fh = database(x, fh);
        }
        scale(fh, ie)
    } else {
        (fh, fl) = mul(fh, fl, th, tl);
        (fh, fl) = sum(th, tl, fh, fl);
        let (h, e) = two(f64::from_bits(((1 - ie) as u64) << 52), fh);
        fl += e;
        denormal(h + fl)
    }
}
/// Base-two exponential, rounded to nearest with ties to even.
pub fn exp2(x: f64) -> f64 {
    let bits = x.to_bits();
    let ax = bits << 1;
    if ax == 0 {
        return 1.0;
    }
    if ax >= 0x8120000000000000 {
        if x.is_nan() {
            return x + x;
        }
        if x.is_infinite() {
            return if x.is_sign_negative() { 0.0 } else { x };
        }
        if x <= -1075.0 {
            return 0.0;
        }
        if x >= 1024.0 {
            return f64::INFINITY;
        }
    }
    if ax <= 0x792e2a8eca5705fc {
        return 1.0;
    }
    let m = bits << 12;
    let ex = (ax >> 53).wrapping_sub(0x3ff);
    let frac = (ex >> 63) | (m << (ex & 63));
    let sx = 4096.0 * x;
    let fx = crate::reference::roundeven(sx);
    let z = sx - fx;
    let z2 = z * z;
    let k = fx as i64;
    let i1 = (k & 63) as usize;
    let i0 = ((k >> 6) & 63) as usize;
    let ie = k >> 12;
    let (th, tl) = mul(T0[i0][1], T0[i0][0], T1[i1][1], T1[i1][0]);
    let mut fh = th;
    let mut fl = (th * z) * ((C[0] + z * C[1]) + z2 * (C[2] + z * C[3])) + tl;
    let eps = f64::from_bits(0x3c0fd00000000000);
    if bits <= 0xc08ff00000000000 {
        if frac != 0 {
            let ub = fh + (fl + eps);
            fh += fl - eps;
            if ub != fh {
                return accurate(x);
            }
        }
        scale(fh, ie)
    } else {
        let (h, e) = two(f64::from_bits(((1 - ie) as u64) << 52), fh);
        fh = h;
        fl += e;
        if frac != 0 {
            let ub = fh + (fl + eps);
            fh += fl - eps;
            if ub != fh {
                return accurate(x);
            }
        }
        denormal(fh)
    }
}

//! Cross-architecture value checks for f32x8 (W256) pixel block ops.
//!
//! The generated `cross_arch_parity` suite only covers W128 types, so the
//! f32x8 `to_u8` / `store_8_rgba_u8` paths (native on x86 AVX2 and the ARM
//! polyfill) would otherwise be exercised only on x86. These run on the native
//! token of each architecture (X64V3Token / NeonToken / Wasm128Token) so the
//! restored native pack+interleave is verified identical everywhere.

use archmage::SimdToken;
use magetypes::simd::generic::f32x8;

#[cfg(target_arch = "x86_64")]
type Tok = archmage::X64V3Token;
#[cfg(target_arch = "aarch64")]
type Tok = archmage::NeonToken;
#[cfg(target_arch = "wasm32")]
type Tok = archmage::Wasm128Token;

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
#[test]
fn f32x8_to_u8_clamp_and_round() {
    if let Some(t) = Tok::summon() {
        // Clamp (neg->0, >255->255) + round-half-to-even (128.5->128, 0.5->0, 2.5->2).
        let v = f32x8::<Tok>::from_array(t, [0.0, 127.6, 255.0, -5.0, 300.0, 128.5, 0.5, 2.5]);
        assert_eq!(v.to_u8(), [0, 128, 255, 0, 255, 128, 0, 2]);
    }
}

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
#[test]
fn f32x8_store_8_rgba_u8_interleave() {
    if let Some(t) = Tok::summon() {
        let r = f32x8::<Tok>::from_array(t, [0.0, 64.0, 128.0, 255.0, 1.0, 2.0, 3.0, 4.0]);
        let g = f32x8::<Tok>::from_array(t, [10.0, 74.0, 138.0, 245.0, 11.0, 12.0, 13.0, 14.0]);
        let b = f32x8::<Tok>::from_array(t, [20.0, 84.0, 148.0, 235.0, 21.0, 22.0, 23.0, 24.0]);
        let a = f32x8::<Tok>::from_array(t, [255.0, 300.0, -5.0, 128.5, 31.0, 32.0, 33.0, 34.0]);
        let want: [u8; 32] = [
            0, 10, 20, 255, 64, 74, 84, 255, 128, 138, 148, 0, 255, 245, 235, 128, 1, 11, 21, 31,
            2, 12, 22, 32, 3, 13, 23, 33, 4, 14, 24, 34,
        ];
        assert_eq!(f32x8::<Tok>::store_8_rgba_u8(r, g, b, a), want);
    }
}

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
#[test]
fn f32x8_transpose_8x8() {
    if let Some(t) = Tok::summon() {
        // rows[i][j] = i*8 + j ; after transpose rows[i][j] == original[j][i].
        let inp: [[f32; 8]; 8] =
            core::array::from_fn(|i| core::array::from_fn(|j| (i * 8 + j) as f32));
        let mut rows: [f32x8<Tok>; 8] =
            core::array::from_fn(|i| f32x8::<Tok>::from_array(t, inp[i]));
        f32x8::<Tok>::transpose_8x8(&mut rows);
        for i in 0..8 {
            let got = rows[i].to_array();
            for j in 0..8 {
                assert_eq!(got[j], inp[j][i], "transpose[{i}][{j}]");
            }
        }
    }
}

//! Auto-generated tests for SIMD types.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

#![cfg(target_arch = "x86_64")]
#![allow(clippy::needless_range_loop)]

use archmage::simd::*;
use archmage::{Avx2FmaToken, SimdToken};

#[test]
fn test_f32x8_basic() {
    if let Some(token) = Avx2FmaToken::try_new() {
        let a = f32x8::splat(token, 1.0);
        let b = f32x8::splat(token, 2.0);
        let c = a + b;
        let arr = c.to_array();
        for &v in &arr {
            assert_eq!(v, 1.0 + 2.0);
        }
    }
}

#[test]
fn test_f32x8_load_store() {
    if let Some(token) = Avx2FmaToken::try_new() {
        let data: [f32; f32x8::LANES] = [1.0; f32x8::LANES];
        let v = f32x8::load(token, &data);
        let mut out = [f32::default(); f32x8::LANES];
        v.store(&mut out);
        assert_eq!(data, out);
    }
}

#[test]
fn test_i32x8_basic() {
    if let Some(token) = Avx2FmaToken::try_new() {
        let a = i32x8::splat(token, 1);
        let b = i32x8::splat(token, 2);
        let c = a + b;
        let arr = c.to_array();
        for &v in &arr {
            assert_eq!(v, 1 + 2);
        }
    }
}

#[test]
fn test_i32x8_load_store() {
    if let Some(token) = Avx2FmaToken::try_new() {
        let data: [i32; i32x8::LANES] = [1; i32x8::LANES];
        let v = i32x8::load(token, &data);
        let mut out = [i32::default(); i32x8::LANES];
        v.store(&mut out);
        assert_eq!(data, out);
    }
}

#[test]
fn test_f32x8_transpose_8x8() {
    if let Some(token) = Avx2FmaToken::try_new() {
        // Create 8 row vectors: row[i] = [i*8, i*8+1, ..., i*8+7]
        let mut rows = [
            f32x8::from_array(token, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
            f32x8::from_array(token, [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]),
            f32x8::from_array(token, [16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0]),
            f32x8::from_array(token, [24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0]),
            f32x8::from_array(token, [32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0]),
            f32x8::from_array(token, [40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0]),
            f32x8::from_array(token, [48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0]),
            f32x8::from_array(token, [56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0]),
        ];

        f32x8::transpose_8x8(&mut rows);

        // After transpose: rows[i][j] should be original rows[j][i] = j*8 + i
        for i in 0..8 {
            let arr = rows[i].to_array();
            for j in 0..8 {
                let expected = (j * 8 + i) as f32;
                assert_eq!(arr[j], expected, "Mismatch at rows[{}][{}]", i, j);
            }
        }

        // Double transpose should restore original
        f32x8::transpose_8x8(&mut rows);
        for i in 0..8 {
            let arr = rows[i].to_array();
            for j in 0..8 {
                let expected = (i * 8 + j) as f32;
                assert_eq!(
                    arr[j], expected,
                    "Double transpose mismatch at rows[{}][{}]",
                    i, j
                );
            }
        }
    }
}

#[test]
fn test_f32x8_load_store_8x8() {
    if let Some(_token) = Avx2FmaToken::try_new() {
        let input: [f32; 64] = core::array::from_fn(|i| i as f32);
        let rows = f32x8::load_8x8(&input);

        // Verify load
        for i in 0..8 {
            let arr = rows[i].to_array();
            for j in 0..8 {
                assert_eq!(arr[j], (i * 8 + j) as f32);
            }
        }

        // Verify store roundtrip
        let mut output = [0.0f32; 64];
        f32x8::store_8x8(&rows, &mut output);
        assert_eq!(input, output);
    }
}

#[test]
fn test_f32x4_4ch_interleave() {
    if let Some(token) = archmage::Sse41Token::try_new() {
        // Create 4 channel vectors (SoA format)
        let r = f32x4::from_array(token, [1.0, 2.0, 3.0, 4.0]);
        let g = f32x4::from_array(token, [10.0, 20.0, 30.0, 40.0]);
        let b = f32x4::from_array(token, [100.0, 200.0, 300.0, 400.0]);
        let a = f32x4::from_array(token, [255.0, 255.0, 255.0, 255.0]);

        // Interleave to AoS: each output vector is one RGBA pixel
        let aos = f32x4::interleave_4ch([r, g, b, a]);
        assert_eq!(aos[0].to_array(), [1.0, 10.0, 100.0, 255.0]); // pixel 0
        assert_eq!(aos[1].to_array(), [2.0, 20.0, 200.0, 255.0]); // pixel 1
        assert_eq!(aos[2].to_array(), [3.0, 30.0, 300.0, 255.0]); // pixel 2
        assert_eq!(aos[3].to_array(), [4.0, 40.0, 400.0, 255.0]); // pixel 3

        // Deinterleave back to SoA
        let [r2, g2, b2, a2] = f32x4::deinterleave_4ch(aos);
        assert_eq!(r2.to_array(), r.to_array());
        assert_eq!(g2.to_array(), g.to_array());
        assert_eq!(b2.to_array(), b.to_array());
        assert_eq!(a2.to_array(), a.to_array());
    }
}

#[test]
fn test_f32x4_load_store_rgba_u8() {
    if let Some(_token) = archmage::Sse41Token::try_new() {
        // 4 RGBA pixels: red, green, blue, white
        let rgba: [u8; 16] = [
            255, 0, 0, 255, // red
            0, 255, 0, 255, // green
            0, 0, 255, 255, // blue
            255, 255, 255, 255, // white
        ];

        let (r, g, b, a) = f32x4::load_4_rgba_u8(&rgba);
        assert_eq!(r.to_array(), [255.0, 0.0, 0.0, 255.0]);
        assert_eq!(g.to_array(), [0.0, 255.0, 0.0, 255.0]);
        assert_eq!(b.to_array(), [0.0, 0.0, 255.0, 255.0]);
        assert_eq!(a.to_array(), [255.0, 255.0, 255.0, 255.0]);

        // Roundtrip
        let out = f32x4::store_4_rgba_u8(r, g, b, a);
        assert_eq!(out, rgba);
    }
}

#[test]
fn test_f32x8_load_store_rgba_u8() {
    if let Some(_token) = Avx2FmaToken::try_new() {
        // 8 RGBA pixels
        let rgba: [u8; 32] = [
            255, 0, 0, 255, // red
            0, 255, 0, 255, // green
            0, 0, 255, 255, // blue
            255, 255, 255, 255, // white
            128, 128, 128, 255, // gray
            0, 0, 0, 255, // black
            255, 128, 0, 255, // orange
            128, 0, 255, 255, // purple
        ];

        let (r, g, b, a) = f32x8::load_8_rgba_u8(&rgba);
        assert_eq!(
            r.to_array(),
            [255.0, 0.0, 0.0, 255.0, 128.0, 0.0, 255.0, 128.0]
        );
        assert_eq!(
            g.to_array(),
            [0.0, 255.0, 0.0, 255.0, 128.0, 0.0, 128.0, 0.0]
        );
        assert_eq!(
            b.to_array(),
            [0.0, 0.0, 255.0, 255.0, 128.0, 0.0, 0.0, 255.0]
        );
        assert_eq!(
            a.to_array(),
            [255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0]
        );

        // Roundtrip
        let out = f32x8::store_8_rgba_u8(r, g, b, a);
        assert_eq!(out, rgba);
    }
}

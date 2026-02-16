//! Tests for generic block operations on f32x4<T> and f32x8<T>.
//!
//! Covers: array/byte views, slice casting, u8 conversions, interleave/deinterleave,
//! RGBA load/store, matrix transpose, and cross-type bitcast references.

#![cfg(target_arch = "x86_64")]

use archmage::{ScalarToken, SimdToken, X64V3Token};
use magetypes::simd::backends::F32x8Backend;
use magetypes::simd::generic::{f32x4, f32x8};

fn get_v3() -> X64V3Token {
    X64V3Token::summon().expect("AVX2+FMA required for tests")
}

// ============================================================================
// f32x8 — Array/Byte Views
// ============================================================================

#[test]
fn f32x8_as_array() {
    let t = get_v3();
    let v = f32x8::<X64V3Token>::from_array(t, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let arr = v.as_array();
    assert_eq!(*arr, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn f32x8_as_array_mut() {
    let t = get_v3();
    let mut v = f32x8::<X64V3Token>::from_array(t, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    v.as_array_mut()[3] = 42.0;
    assert_eq!(v.to_array()[3], 42.0);
}

#[test]
fn f32x8_as_bytes_roundtrip() {
    let t = get_v3();
    let v = f32x8::<X64V3Token>::from_array(t, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let bytes = *v.as_bytes();
    let v2 = f32x8::<X64V3Token>::from_bytes_owned(t, bytes);
    assert_eq!(v.to_array(), v2.to_array());
}

#[test]
fn f32x8_from_bytes_ref() {
    let t = get_v3();
    let original = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let bytes: [u8; 32] = unsafe { core::mem::transmute(original) };
    let v = f32x8::<X64V3Token>::from_bytes(t, &bytes);
    assert_eq!(v.to_array(), original);
}

#[test]
fn f32x8_as_bytes_mut() {
    let t = get_v3();
    let mut v = f32x8::<X64V3Token>::splat(t, 0.0);
    let bytes = v.as_bytes_mut();
    // Write the bytes of 1.0_f32 into the first 4 bytes
    let one_bytes = 1.0_f32.to_ne_bytes();
    bytes[0..4].copy_from_slice(&one_bytes);
    assert_eq!(v.to_array()[0], 1.0);
}

// ============================================================================
// f32x8 — Slice Casting
// ============================================================================

#[test]
fn f32x8_cast_slice_aligned() {
    let t = get_v3();
    // Use from_array to ensure alignment
    let v = f32x8::<X64V3Token>::from_array(t, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let arr = v.as_array();
    // The array reference from as_array is properly aligned
    let cast = f32x8::<X64V3Token>::cast_slice(t, arr);
    assert!(cast.is_some());
    assert_eq!(cast.unwrap().len(), 1);
    assert_eq!(
        cast.unwrap()[0].to_array(),
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    );
}

#[test]
fn f32x8_cast_slice_wrong_length() {
    let t = get_v3();
    let data = [1.0_f32, 2.0, 3.0]; // not multiple of 8
    assert!(f32x8::<X64V3Token>::cast_slice(t, &data).is_none());
}

#[test]
fn f32x8_cast_slice_scalar_always_aligned() {
    let t = ScalarToken;
    let data = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    // ScalarToken uses [f32; 8] repr which has f32 alignment — any f32 slice qualifies
    let cast = f32x8::<ScalarToken>::cast_slice(t, &data);
    assert!(cast.is_some());
    assert_eq!(cast.unwrap()[0].to_array(), data);
}

// ============================================================================
// f32x8 — u8 Conversions
// ============================================================================

#[test]
fn f32x8_from_u8() {
    let bytes = [0u8, 128, 255, 1, 50, 100, 200, 42];
    let v = f32x8::<X64V3Token>::from_u8(&bytes);
    let arr = v.to_array();
    assert_eq!(arr, [0.0, 128.0, 255.0, 1.0, 50.0, 100.0, 200.0, 42.0]);
}

#[test]
fn f32x8_to_u8() {
    let t = get_v3();
    let v = f32x8::<X64V3Token>::from_array(t, [0.0, 127.6, 255.0, -5.0, 300.0, 0.4, 128.5, 1.0]);
    let bytes = v.to_u8();
    assert_eq!(bytes, [0, 128, 255, 0, 255, 0, 129, 1]); // clamped + rounded
}

#[test]
fn f32x8_u8_roundtrip() {
    let input = [10u8, 20, 30, 40, 50, 60, 70, 80];
    let v = f32x8::<X64V3Token>::from_u8(&input);
    let output = v.to_u8();
    assert_eq!(input, output);
}

// ============================================================================
// f32x8 — Interleave
// ============================================================================

#[test]
fn f32x8_interleave_lo() {
    let t = get_v3();
    let a = f32x8::<X64V3Token>::from_array(t, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    let b = f32x8::<X64V3Token>::from_array(t, [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]);
    let lo = a.interleave_lo(b);
    // Within 128-bit lanes: low pairs
    assert_eq!(lo.to_array(), [0.0, 10.0, 1.0, 11.0, 4.0, 14.0, 5.0, 15.0]);
}

#[test]
fn f32x8_interleave_hi() {
    let t = get_v3();
    let a = f32x8::<X64V3Token>::from_array(t, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    let b = f32x8::<X64V3Token>::from_array(t, [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]);
    let hi = a.interleave_hi(b);
    // Within 128-bit lanes: high pairs
    assert_eq!(hi.to_array(), [2.0, 12.0, 3.0, 13.0, 6.0, 16.0, 7.0, 17.0]);
}

#[test]
fn f32x8_interleave_both() {
    let t = get_v3();
    let a = f32x8::<X64V3Token>::from_array(t, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    let b = f32x8::<X64V3Token>::from_array(t, [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]);
    let (lo, hi) = a.interleave(b);
    assert_eq!(lo.to_array(), a.interleave_lo(b).to_array());
    assert_eq!(hi.to_array(), a.interleave_hi(b).to_array());
}

// ============================================================================
// f32x8 — 4-Channel Deinterleave/Interleave
// ============================================================================

#[test]
fn f32x8_deinterleave_4ch() {
    let t = get_v3();
    // 8 RGBA pixels in AoS format (4 vectors, 2 pixels each)
    let rgba = [
        f32x8::<X64V3Token>::from_array(t, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]), // R0,G0,B0,A0, R1,G1,B1,A1
        f32x8::<X64V3Token>::from_array(t, [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]), // R2,G2,B2,A2, R3,G3,B3,A3
        f32x8::<X64V3Token>::from_array(t, [17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0]),
        f32x8::<X64V3Token>::from_array(t, [25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0]),
    ];
    let [r, g, b, a] = f32x8::<X64V3Token>::deinterleave_4ch(rgba);
    assert_eq!(r.to_array(), [1.0, 5.0, 9.0, 13.0, 17.0, 21.0, 25.0, 29.0]); // all R
    assert_eq!(g.to_array(), [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0]); // all G
    assert_eq!(b.to_array(), [3.0, 7.0, 11.0, 15.0, 19.0, 23.0, 27.0, 31.0]); // all B
    assert_eq!(a.to_array(), [4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0]); // all A
}

#[test]
fn f32x8_interleave_4ch() {
    let t = get_v3();
    let channels = [
        f32x8::<X64V3Token>::from_array(t, [1.0, 5.0, 9.0, 13.0, 17.0, 21.0, 25.0, 29.0]), // R
        f32x8::<X64V3Token>::from_array(t, [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0]), // G
        f32x8::<X64V3Token>::from_array(t, [3.0, 7.0, 11.0, 15.0, 19.0, 23.0, 27.0, 31.0]), // B
        f32x8::<X64V3Token>::from_array(t, [4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0]), // A
    ];
    let result = f32x8::<X64V3Token>::interleave_4ch(channels);
    assert_eq!(
        result[0].to_array(),
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    );
    assert_eq!(
        result[1].to_array(),
        [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
    );
    assert_eq!(
        result[2].to_array(),
        [17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0]
    );
    assert_eq!(
        result[3].to_array(),
        [25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0]
    );
}

#[test]
fn f32x8_deinterleave_interleave_roundtrip() {
    let t = get_v3();
    let original = [
        f32x8::<X64V3Token>::from_array(t, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        f32x8::<X64V3Token>::from_array(t, [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]),
        f32x8::<X64V3Token>::from_array(t, [17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0]),
        f32x8::<X64V3Token>::from_array(t, [25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0]),
    ];
    let channels = f32x8::<X64V3Token>::deinterleave_4ch(original);
    let restored = f32x8::<X64V3Token>::interleave_4ch(channels);
    for i in 0..4 {
        assert_eq!(original[i].to_array(), restored[i].to_array());
    }
}

// ============================================================================
// f32x8 — RGBA Load/Store
// ============================================================================

#[test]
fn f32x8_load_8_rgba_u8() {
    // 8 RGBA pixels: R=10*i, G=10*i+1, B=10*i+2, A=10*i+3
    let mut rgba = [0u8; 32];
    for i in 0..8 {
        rgba[i * 4] = (i * 10) as u8;
        rgba[i * 4 + 1] = (i * 10 + 1) as u8;
        rgba[i * 4 + 2] = (i * 10 + 2) as u8;
        rgba[i * 4 + 3] = (i * 10 + 3) as u8;
    }
    let (r, g, b, a) = f32x8::<X64V3Token>::load_8_rgba_u8(&rgba);
    assert_eq!(
        r.to_array(),
        [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
    );
    assert_eq!(
        g.to_array(),
        [1.0, 11.0, 21.0, 31.0, 41.0, 51.0, 61.0, 71.0]
    );
    assert_eq!(
        b.to_array(),
        [2.0, 12.0, 22.0, 32.0, 42.0, 52.0, 62.0, 72.0]
    );
    assert_eq!(
        a.to_array(),
        [3.0, 13.0, 23.0, 33.0, 43.0, 53.0, 63.0, 73.0]
    );
}

#[test]
fn f32x8_store_8_rgba_u8() {
    let t = get_v3();
    let r = f32x8::<X64V3Token>::from_array(t, [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]);
    let g = f32x8::<X64V3Token>::from_array(t, [1.0, 11.0, 21.0, 31.0, 41.0, 51.0, 61.0, 71.0]);
    let b = f32x8::<X64V3Token>::from_array(t, [2.0, 12.0, 22.0, 32.0, 42.0, 52.0, 62.0, 72.0]);
    let a = f32x8::<X64V3Token>::from_array(t, [3.0, 13.0, 23.0, 33.0, 43.0, 53.0, 63.0, 73.0]);
    let out = f32x8::<X64V3Token>::store_8_rgba_u8(r, g, b, a);
    for i in 0..8 {
        assert_eq!(out[i * 4], (i * 10) as u8);
        assert_eq!(out[i * 4 + 1], (i * 10 + 1) as u8);
        assert_eq!(out[i * 4 + 2], (i * 10 + 2) as u8);
        assert_eq!(out[i * 4 + 3], (i * 10 + 3) as u8);
    }
}

#[test]
fn f32x8_rgba_load_store_roundtrip() {
    let mut rgba = [0u8; 32];
    for i in 0..32 {
        rgba[i] = i as u8;
    }
    let (r, g, b, a) = f32x8::<X64V3Token>::load_8_rgba_u8(&rgba);
    let out = f32x8::<X64V3Token>::store_8_rgba_u8(r, g, b, a);
    assert_eq!(rgba, out);
}

// ============================================================================
// f32x8 — Transpose
// ============================================================================

#[test]
fn f32x8_transpose_8x8() {
    let t = get_v3();
    let mut rows: [f32x8<X64V3Token>; 8] = core::array::from_fn(|i| {
        f32x8::<X64V3Token>::from_array(t, core::array::from_fn(|j| (i * 8 + j) as f32))
    });
    f32x8::<X64V3Token>::transpose_8x8(&mut rows);
    // After transpose: rows[i][j] should be (j*8 + i) as f32
    for i in 0..8 {
        let arr = rows[i].to_array();
        for j in 0..8 {
            assert_eq!(arr[j], (j * 8 + i) as f32, "rows[{i}][{j}]");
        }
    }
}

#[test]
fn f32x8_transpose_8x8_copy() {
    let t = get_v3();
    let rows: [f32x8<X64V3Token>; 8] = core::array::from_fn(|i| {
        f32x8::<X64V3Token>::from_array(t, core::array::from_fn(|j| (i * 8 + j) as f32))
    });
    let transposed = f32x8::<X64V3Token>::transpose_8x8_copy(rows);
    for i in 0..8 {
        let arr = transposed[i].to_array();
        for j in 0..8 {
            assert_eq!(arr[j], (j * 8 + i) as f32, "transposed[{i}][{j}]");
        }
    }
}

#[test]
fn f32x8_transpose_double_is_identity() {
    let t = get_v3();
    let original: [f32x8<X64V3Token>; 8] = core::array::from_fn(|i| {
        f32x8::<X64V3Token>::from_array(t, core::array::from_fn(|j| (i * 8 + j) as f32))
    });
    let transposed = f32x8::<X64V3Token>::transpose_8x8_copy(original);
    let restored = f32x8::<X64V3Token>::transpose_8x8_copy(transposed);
    for i in 0..8 {
        assert_eq!(original[i].to_array(), restored[i].to_array());
    }
}

#[test]
fn f32x8_load_store_8x8() {
    let block: [f32; 64] = core::array::from_fn(|i| i as f32);
    let rows = f32x8::<X64V3Token>::load_8x8(&block);
    for i in 0..8 {
        let arr = rows[i].to_array();
        for j in 0..8 {
            assert_eq!(arr[j], (i * 8 + j) as f32);
        }
    }
    let mut out = [0.0_f32; 64];
    f32x8::<X64V3Token>::store_8x8(&rows, &mut out);
    assert_eq!(block, out);
}

// ============================================================================
// f32x8 — Bitcast References
// ============================================================================

#[test]
fn f32x8_bitcast_ref_i32() {
    let t = get_v3();
    let v = f32x8::<X64V3Token>::from_array(t, [1.0, -1.0, 0.0, f32::INFINITY, 2.0, 3.0, 4.0, 5.0]);
    let i32_ref = v.bitcast_ref_i32();
    let arr = i32_ref.to_array();
    assert_eq!(arr[0], 1.0_f32.to_bits() as i32);
    assert_eq!(arr[1], (-1.0_f32).to_bits() as i32);
    assert_eq!(arr[2], 0.0_f32.to_bits() as i32);
    assert_eq!(arr[3], f32::INFINITY.to_bits() as i32);
}

#[test]
fn f32x8_bitcast_mut_i32() {
    let t = get_v3();
    let mut v = f32x8::<X64V3Token>::splat(t, 0.0);
    {
        let i32_ref = v.bitcast_mut_i32();
        let one_bits = 1.0_f32.to_bits() as i32;
        // Set first lane to bits of 1.0
        let mut arr = i32_ref.to_array();
        arr[0] = one_bits;
        *i32_ref = magetypes::simd::generic::i32x8::<X64V3Token>::from_array(t, arr);
    }
    assert_eq!(v.to_array()[0], 1.0);
}

// ============================================================================
// f32x4 — Array/Byte Views
// ============================================================================

#[test]
fn f32x4_as_array() {
    let t = get_v3();
    let v = f32x4::<X64V3Token>::from_array(t, [1.0, 2.0, 3.0, 4.0]);
    assert_eq!(*v.as_array(), [1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn f32x4_as_array_mut() {
    let t = get_v3();
    let mut v = f32x4::<X64V3Token>::from_array(t, [1.0, 2.0, 3.0, 4.0]);
    v.as_array_mut()[2] = 99.0;
    assert_eq!(v.to_array()[2], 99.0);
}

#[test]
fn f32x4_bytes_roundtrip() {
    let t = get_v3();
    let v = f32x4::<X64V3Token>::from_array(t, [1.0, 2.0, 3.0, 4.0]);
    let bytes = *v.as_bytes();
    let v2 = f32x4::<X64V3Token>::from_bytes_owned(t, bytes);
    assert_eq!(v.to_array(), v2.to_array());
}

// ============================================================================
// f32x4 — u8 Conversions
// ============================================================================

#[test]
fn f32x4_from_u8() {
    let bytes = [0u8, 128, 255, 42];
    let v = f32x4::<X64V3Token>::from_u8(&bytes);
    assert_eq!(v.to_array(), [0.0, 128.0, 255.0, 42.0]);
}

#[test]
fn f32x4_to_u8() {
    let t = get_v3();
    let v = f32x4::<X64V3Token>::from_array(t, [0.0, 127.6, 255.0, -5.0]);
    assert_eq!(v.to_u8(), [0, 128, 255, 0]); // clamped + rounded
}

#[test]
fn f32x4_u8_roundtrip() {
    let input = [10u8, 20, 30, 40];
    let v = f32x4::<X64V3Token>::from_u8(&input);
    assert_eq!(v.to_u8(), input);
}

// ============================================================================
// f32x4 — Interleave
// ============================================================================

#[test]
fn f32x4_interleave_lo() {
    let t = get_v3();
    let a = f32x4::<X64V3Token>::from_array(t, [0.0, 1.0, 2.0, 3.0]);
    let b = f32x4::<X64V3Token>::from_array(t, [10.0, 11.0, 12.0, 13.0]);
    assert_eq!(a.interleave_lo(b).to_array(), [0.0, 10.0, 1.0, 11.0]);
}

#[test]
fn f32x4_interleave_hi() {
    let t = get_v3();
    let a = f32x4::<X64V3Token>::from_array(t, [0.0, 1.0, 2.0, 3.0]);
    let b = f32x4::<X64V3Token>::from_array(t, [10.0, 11.0, 12.0, 13.0]);
    assert_eq!(a.interleave_hi(b).to_array(), [2.0, 12.0, 3.0, 13.0]);
}

// ============================================================================
// f32x4 — 4-Channel Deinterleave/Interleave (= transpose)
// ============================================================================

#[test]
fn f32x4_deinterleave_4ch() {
    let t = get_v3();
    let rgba = [
        f32x4::<X64V3Token>::from_array(t, [1.0, 2.0, 3.0, 4.0]), // pixel 0: R,G,B,A
        f32x4::<X64V3Token>::from_array(t, [5.0, 6.0, 7.0, 8.0]), // pixel 1
        f32x4::<X64V3Token>::from_array(t, [9.0, 10.0, 11.0, 12.0]),
        f32x4::<X64V3Token>::from_array(t, [13.0, 14.0, 15.0, 16.0]),
    ];
    let [r, g, b, a] = f32x4::<X64V3Token>::deinterleave_4ch(rgba);
    assert_eq!(r.to_array(), [1.0, 5.0, 9.0, 13.0]); // R channel
    assert_eq!(g.to_array(), [2.0, 6.0, 10.0, 14.0]); // G channel
    assert_eq!(b.to_array(), [3.0, 7.0, 11.0, 15.0]); // B channel
    assert_eq!(a.to_array(), [4.0, 8.0, 12.0, 16.0]); // A channel
}

#[test]
fn f32x4_interleave_4ch_roundtrip() {
    let t = get_v3();
    let original = [
        f32x4::<X64V3Token>::from_array(t, [1.0, 2.0, 3.0, 4.0]),
        f32x4::<X64V3Token>::from_array(t, [5.0, 6.0, 7.0, 8.0]),
        f32x4::<X64V3Token>::from_array(t, [9.0, 10.0, 11.0, 12.0]),
        f32x4::<X64V3Token>::from_array(t, [13.0, 14.0, 15.0, 16.0]),
    ];
    let channels = f32x4::<X64V3Token>::deinterleave_4ch(original);
    let restored = f32x4::<X64V3Token>::interleave_4ch(channels);
    for i in 0..4 {
        assert_eq!(original[i].to_array(), restored[i].to_array());
    }
}

// ============================================================================
// f32x4 — RGBA Load/Store
// ============================================================================

#[test]
fn f32x4_load_4_rgba_u8() {
    let rgba = [
        10u8, 20, 30, 40, // pixel 0
        50, 60, 70, 80, // pixel 1
        90, 100, 110, 120, // pixel 2
        130, 140, 150, 160, // pixel 3
    ];
    let (r, g, b, a) = f32x4::<X64V3Token>::load_4_rgba_u8(&rgba);
    assert_eq!(r.to_array(), [10.0, 50.0, 90.0, 130.0]);
    assert_eq!(g.to_array(), [20.0, 60.0, 100.0, 140.0]);
    assert_eq!(b.to_array(), [30.0, 70.0, 110.0, 150.0]);
    assert_eq!(a.to_array(), [40.0, 80.0, 120.0, 160.0]);
}

#[test]
fn f32x4_store_4_rgba_u8() {
    let t = get_v3();
    let r = f32x4::<X64V3Token>::from_array(t, [10.0, 50.0, 90.0, 130.0]);
    let g = f32x4::<X64V3Token>::from_array(t, [20.0, 60.0, 100.0, 140.0]);
    let b = f32x4::<X64V3Token>::from_array(t, [30.0, 70.0, 110.0, 150.0]);
    let a = f32x4::<X64V3Token>::from_array(t, [40.0, 80.0, 120.0, 160.0]);
    let out = f32x4::<X64V3Token>::store_4_rgba_u8(r, g, b, a);
    assert_eq!(
        out,
        [
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160
        ]
    );
}

#[test]
fn f32x4_rgba_roundtrip() {
    let rgba: [u8; 16] = core::array::from_fn(|i| (i * 15) as u8);
    let (r, g, b, a) = f32x4::<X64V3Token>::load_4_rgba_u8(&rgba);
    let out = f32x4::<X64V3Token>::store_4_rgba_u8(r, g, b, a);
    assert_eq!(rgba, out);
}

// ============================================================================
// f32x4 — Transpose
// ============================================================================

#[test]
fn f32x4_transpose_4x4() {
    let t = get_v3();
    let mut rows = [
        f32x4::<X64V3Token>::from_array(t, [0.0, 1.0, 2.0, 3.0]),
        f32x4::<X64V3Token>::from_array(t, [4.0, 5.0, 6.0, 7.0]),
        f32x4::<X64V3Token>::from_array(t, [8.0, 9.0, 10.0, 11.0]),
        f32x4::<X64V3Token>::from_array(t, [12.0, 13.0, 14.0, 15.0]),
    ];
    f32x4::<X64V3Token>::transpose_4x4(&mut rows);
    assert_eq!(rows[0].to_array(), [0.0, 4.0, 8.0, 12.0]);
    assert_eq!(rows[1].to_array(), [1.0, 5.0, 9.0, 13.0]);
    assert_eq!(rows[2].to_array(), [2.0, 6.0, 10.0, 14.0]);
    assert_eq!(rows[3].to_array(), [3.0, 7.0, 11.0, 15.0]);
}

#[test]
fn f32x4_transpose_double_is_identity() {
    let t = get_v3();
    let original = [
        f32x4::<X64V3Token>::from_array(t, [1.0, 2.0, 3.0, 4.0]),
        f32x4::<X64V3Token>::from_array(t, [5.0, 6.0, 7.0, 8.0]),
        f32x4::<X64V3Token>::from_array(t, [9.0, 10.0, 11.0, 12.0]),
        f32x4::<X64V3Token>::from_array(t, [13.0, 14.0, 15.0, 16.0]),
    ];
    let transposed = f32x4::<X64V3Token>::transpose_4x4_copy(original);
    let restored = f32x4::<X64V3Token>::transpose_4x4_copy(transposed);
    for i in 0..4 {
        assert_eq!(original[i].to_array(), restored[i].to_array());
    }
}

// ============================================================================
// f32x4 — Bitcast References
// ============================================================================

#[test]
fn f32x4_bitcast_ref_i32() {
    let t = get_v3();
    let v = f32x4::<X64V3Token>::from_array(t, [1.0, -1.0, 0.0, f32::INFINITY]);
    let i32_ref = v.bitcast_ref_i32();
    let arr = i32_ref.to_array();
    assert_eq!(arr[0], 1.0_f32.to_bits() as i32);
    assert_eq!(arr[1], (-1.0_f32).to_bits() as i32);
}

// ============================================================================
// Scalar backend — same operations work
// ============================================================================

#[test]
fn f32x8_scalar_as_array() {
    let t = ScalarToken;
    let v = f32x8::<ScalarToken>::from_array(t, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert_eq!(*v.as_array(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn f32x8_scalar_from_u8() {
    let bytes = [0u8, 128, 255, 1, 50, 100, 200, 42];
    let v = f32x8::<ScalarToken>::from_u8(&bytes);
    assert_eq!(
        v.to_array(),
        [0.0, 128.0, 255.0, 1.0, 50.0, 100.0, 200.0, 42.0]
    );
}

#[test]
fn f32x8_scalar_interleave_lo() {
    let t = ScalarToken;
    let a = f32x8::<ScalarToken>::from_array(t, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    let b = f32x8::<ScalarToken>::from_array(t, [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]);
    assert_eq!(
        a.interleave_lo(b).to_array(),
        [0.0, 10.0, 1.0, 11.0, 4.0, 14.0, 5.0, 15.0]
    );
}

#[test]
fn f32x8_scalar_transpose_8x8() {
    let t = ScalarToken;
    let mut rows: [f32x8<ScalarToken>; 8] = core::array::from_fn(|i| {
        f32x8::<ScalarToken>::from_array(t, core::array::from_fn(|j| (i * 8 + j) as f32))
    });
    f32x8::<ScalarToken>::transpose_8x8(&mut rows);
    for i in 0..8 {
        let arr = rows[i].to_array();
        for j in 0..8 {
            assert_eq!(arr[j], (j * 8 + i) as f32, "scalar rows[{i}][{j}]");
        }
    }
}

#[test]
fn f32x8_scalar_rgba_roundtrip() {
    let mut rgba = [0u8; 32];
    for i in 0..32 {
        rgba[i] = i as u8;
    }
    let (r, g, b, a) = f32x8::<ScalarToken>::load_8_rgba_u8(&rgba);
    let out = f32x8::<ScalarToken>::store_8_rgba_u8(r, g, b, a);
    assert_eq!(rgba, out);
}

#[test]
fn f32x4_scalar_transpose_4x4() {
    let t = ScalarToken;
    let mut rows = [
        f32x4::<ScalarToken>::from_array(t, [0.0, 1.0, 2.0, 3.0]),
        f32x4::<ScalarToken>::from_array(t, [4.0, 5.0, 6.0, 7.0]),
        f32x4::<ScalarToken>::from_array(t, [8.0, 9.0, 10.0, 11.0]),
        f32x4::<ScalarToken>::from_array(t, [12.0, 13.0, 14.0, 15.0]),
    ];
    f32x4::<ScalarToken>::transpose_4x4(&mut rows);
    assert_eq!(rows[0].to_array(), [0.0, 4.0, 8.0, 12.0]);
    assert_eq!(rows[1].to_array(), [1.0, 5.0, 9.0, 13.0]);
}

// ============================================================================
// Cross-backend consistency
// ============================================================================

#[test]
fn f32x8_v3_vs_scalar_interleave() {
    let t_v3 = get_v3();
    let t_sc = ScalarToken;
    let a_vals = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b_vals = [10.0_f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];

    let (lo_v3, hi_v3) =
        f32x8::<X64V3Token>::from_array(t_v3, a_vals)
            .interleave(f32x8::<X64V3Token>::from_array(t_v3, b_vals));
    let (lo_sc, hi_sc) =
        f32x8::<ScalarToken>::from_array(t_sc, a_vals)
            .interleave(f32x8::<ScalarToken>::from_array(t_sc, b_vals));

    assert_eq!(lo_v3.to_array(), lo_sc.to_array());
    assert_eq!(hi_v3.to_array(), hi_sc.to_array());
}

#[test]
fn f32x8_v3_vs_scalar_transpose() {
    let t_v3 = get_v3();
    let t_sc = ScalarToken;
    let data: [[f32; 8]; 8] =
        core::array::from_fn(|i| core::array::from_fn(|j| (i * 8 + j) as f32));

    let rows_v3 = f32x8::<X64V3Token>::transpose_8x8_copy(core::array::from_fn(|i| {
        f32x8::<X64V3Token>::from_array(t_v3, data[i])
    }));
    let rows_sc = f32x8::<ScalarToken>::transpose_8x8_copy(core::array::from_fn(|i| {
        f32x8::<ScalarToken>::from_array(t_sc, data[i])
    }));

    for i in 0..8 {
        assert_eq!(rows_v3[i].to_array(), rows_sc[i].to_array());
    }
}

#[test]
fn f32x8_v3_vs_scalar_rgba() {
    let rgba: [u8; 32] = core::array::from_fn(|i| (i * 7 + 13) as u8);

    let (r1, g1, b1, a1) = f32x8::<X64V3Token>::load_8_rgba_u8(&rgba);
    let (r2, g2, b2, a2) = f32x8::<ScalarToken>::load_8_rgba_u8(&rgba);

    assert_eq!(r1.to_array(), r2.to_array());
    assert_eq!(g1.to_array(), g2.to_array());
    assert_eq!(b1.to_array(), b2.to_array());
    assert_eq!(a1.to_array(), a2.to_array());
}

// ============================================================================
// Parity with old types
// ============================================================================

#[test]
fn f32x8_parity_interleave_lo() {
    use magetypes::simd::f32x8 as OldF32x8;
    let t = get_v3();
    let a_vals = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b_vals = [10.0_f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];

    let old = OldF32x8::from_array(t, a_vals).interleave_lo(OldF32x8::from_array(t, b_vals));
    let new = f32x8::<X64V3Token>::from_array(t, a_vals)
        .interleave_lo(f32x8::<X64V3Token>::from_array(t, b_vals));

    assert_eq!(old.to_array(), new.to_array());
}

#[test]
fn f32x8_parity_interleave_hi() {
    use magetypes::simd::f32x8 as OldF32x8;
    let t = get_v3();
    let a_vals = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b_vals = [10.0_f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];

    let old = OldF32x8::from_array(t, a_vals).interleave_hi(OldF32x8::from_array(t, b_vals));
    let new = f32x8::<X64V3Token>::from_array(t, a_vals)
        .interleave_hi(f32x8::<X64V3Token>::from_array(t, b_vals));

    assert_eq!(old.to_array(), new.to_array());
}

#[test]
fn f32x8_parity_deinterleave_4ch() {
    use magetypes::simd::f32x8 as OldF32x8;
    let t = get_v3();
    let vals: [[f32; 8]; 4] =
        core::array::from_fn(|i| core::array::from_fn(|j| (i * 8 + j + 1) as f32));

    let old =
        OldF32x8::deinterleave_4ch(core::array::from_fn(|i| OldF32x8::from_array(t, vals[i])));
    let new = f32x8::<X64V3Token>::deinterleave_4ch(core::array::from_fn(|i| {
        f32x8::<X64V3Token>::from_array(t, vals[i])
    }));

    for i in 0..4 {
        assert_eq!(old[i].to_array(), new[i].to_array(), "channel {i}");
    }
}

#[test]
fn f32x8_parity_interleave_4ch() {
    use magetypes::simd::f32x8 as OldF32x8;
    let t = get_v3();
    let vals: [[f32; 8]; 4] =
        core::array::from_fn(|i| core::array::from_fn(|j| (i * 8 + j + 1) as f32));

    let old = OldF32x8::interleave_4ch(core::array::from_fn(|i| OldF32x8::from_array(t, vals[i])));
    let new = f32x8::<X64V3Token>::interleave_4ch(core::array::from_fn(|i| {
        f32x8::<X64V3Token>::from_array(t, vals[i])
    }));

    for i in 0..4 {
        assert_eq!(old[i].to_array(), new[i].to_array(), "vector {i}");
    }
}

#[test]
fn f32x8_parity_transpose_8x8() {
    use magetypes::simd::f32x8 as OldF32x8;
    let t = get_v3();
    let data: [[f32; 8]; 8] =
        core::array::from_fn(|i| core::array::from_fn(|j| (i * 8 + j) as f32));

    let old =
        OldF32x8::transpose_8x8_copy(core::array::from_fn(|i| OldF32x8::from_array(t, data[i])));
    let new = f32x8::<X64V3Token>::transpose_8x8_copy(core::array::from_fn(|i| {
        f32x8::<X64V3Token>::from_array(t, data[i])
    }));

    for i in 0..8 {
        assert_eq!(old[i].to_array(), new[i].to_array(), "row {i}");
    }
}

#[test]
fn f32x8_parity_from_u8() {
    use magetypes::simd::f32x8 as OldF32x8;
    let bytes = [10u8, 20, 30, 40, 50, 60, 70, 80];
    let old = OldF32x8::from_u8(&bytes).to_array();
    let new = f32x8::<X64V3Token>::from_u8(&bytes).to_array();
    assert_eq!(old, new);
}

#[test]
fn f32x8_parity_to_u8() {
    use magetypes::simd::f32x8 as OldF32x8;
    let t = get_v3();
    // Avoid exact .5 values: old x86 uses round-to-nearest-even (128.5→128),
    // generic uses f32::round (ties-away-from-zero: 128.5→129).
    // Only exact .5 values differ; all other values are identical.
    let vals = [0.0_f32, 127.6, 255.0, -5.0, 300.0, 0.4, 128.7, 1.0];
    let old = OldF32x8::from_array(t, vals).to_u8();
    let new = f32x8::<X64V3Token>::from_array(t, vals).to_u8();
    assert_eq!(old, new);
}

#[test]
fn f32x4_parity_interleave_lo() {
    use magetypes::simd::f32x4 as OldF32x4;
    let t = get_v3();
    let a_vals = [1.0_f32, 2.0, 3.0, 4.0];
    let b_vals = [10.0_f32, 20.0, 30.0, 40.0];

    let old = OldF32x4::from_array(t, a_vals).interleave_lo(OldF32x4::from_array(t, b_vals));
    let new = f32x4::<X64V3Token>::from_array(t, a_vals)
        .interleave_lo(f32x4::<X64V3Token>::from_array(t, b_vals));

    assert_eq!(old.to_array(), new.to_array());
}

#[test]
fn f32x4_parity_transpose_4x4() {
    use magetypes::simd::f32x4 as OldF32x4;
    let t = get_v3();
    let data: [[f32; 4]; 4] =
        core::array::from_fn(|i| core::array::from_fn(|j| (i * 4 + j) as f32));

    let old =
        OldF32x4::transpose_4x4_copy(core::array::from_fn(|i| OldF32x4::from_array(t, data[i])));
    let new = f32x4::<X64V3Token>::transpose_4x4_copy(core::array::from_fn(|i| {
        f32x4::<X64V3Token>::from_array(t, data[i])
    }));

    for i in 0..4 {
        assert_eq!(old[i].to_array(), new[i].to_array(), "row {i}");
    }
}

#[test]
fn f32x8_parity_load_store_rgba() {
    use magetypes::simd::f32x8 as OldF32x8;
    let rgba: [u8; 32] = core::array::from_fn(|i| (i * 7 + 13) as u8);

    let (or, og, ob, oa) = OldF32x8::load_8_rgba_u8(&rgba);
    let (nr, ng, nb, na) = f32x8::<X64V3Token>::load_8_rgba_u8(&rgba);

    assert_eq!(or.to_array(), nr.to_array());
    assert_eq!(og.to_array(), ng.to_array());
    assert_eq!(ob.to_array(), nb.to_array());
    assert_eq!(oa.to_array(), na.to_array());
}

// ============================================================================
// Generic function test — image processing pipeline
// ============================================================================

/// Demonstrates a generic image processing function using block ops.
fn brighten_pixels<T: F32x8Backend>(pixels: &mut [u8; 32], amount: f32) {
    let (r, g, b, a) = f32x8::<T>::load_8_rgba_u8(pixels);
    // Use operator overloads — the scalar broadcast Add<f32> impl
    let r = r + amount;
    let g = g + amount;
    let b = b + amount;
    *pixels = f32x8::<T>::store_8_rgba_u8(r, g, b, a);
}

#[test]
fn generic_brighten_v3_and_scalar() {
    let mut pixels_v3 = [0u8; 32];
    let mut pixels_sc = [0u8; 32];
    for i in 0..32 {
        pixels_v3[i] = (i * 5) as u8;
        pixels_sc[i] = (i * 5) as u8;
    }
    brighten_pixels::<X64V3Token>(&mut pixels_v3, 10.0);
    brighten_pixels::<ScalarToken>(&mut pixels_sc, 10.0);
    assert_eq!(pixels_v3, pixels_sc);
}

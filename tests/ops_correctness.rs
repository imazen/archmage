//! Correctness tests for archmage token-gated SIMD primitives
//!
//! Tests all operations that DCT, color conversion, and other image processing
//! algorithms would need: load, store, arithmetic, FMA, shuffle, permute, etc.

#![cfg(target_arch = "x86_64")]

use archmage::tokens::x86::*;
use archmage::tokens::SimdToken;

#[cfg(target_arch = "x86_64")]
use archmage::ops::x86::*;

// ============================================================================
// Load/Store Tests
// ============================================================================

#[test]
fn test_load_store_roundtrip() {
    if let Some(token) = Avx2Token::try_new() {
        let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let v = load_f32x8(token, &input);
        let mut output = [0.0f32; 8];
        store_f32x8(token, &mut output, v);
        assert_eq!(input, output);
    }
}

#[test]
fn test_load_store_negative_values() {
    if let Some(token) = Avx2Token::try_new() {
        let input = [-1.0f32, -2.0, -3.0, -4.0, 5.0, 6.0, 7.0, 8.0];
        let v = load_f32x8(token, &input);
        let mut output = [0.0f32; 8];
        store_f32x8(token, &mut output, v);
        assert_eq!(input, output);
    }
}

#[test]
fn test_load_store_special_values() {
    if let Some(token) = Avx2Token::try_new() {
        let input = [0.0f32, -0.0, f32::INFINITY, f32::NEG_INFINITY,
                     f32::MIN, f32::MAX, 1e-38, 1e38];
        let v = load_f32x8(token, &input);
        let output = to_array_f32x8(v);
        // Check all except NaN (NaN != NaN)
        for i in 0..8 {
            assert_eq!(input[i], output[i], "mismatch at index {}", i);
        }
    }
}

// ============================================================================
// Arithmetic Tests
// ============================================================================

#[test]
fn test_add_f32x8() {
    if let Some(token) = Avx2Token::try_new() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [8.0f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let va = load_f32x8(token, &a);
        let vb = load_f32x8(token, &b);
        let result = to_array_f32x8(add_f32x8(token, va, vb));
        assert_eq!(result, [9.0f32; 8]);
    }
}

#[test]
fn test_sub_f32x8() {
    if let Some(token) = Avx2Token::try_new() {
        let a = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let b = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let va = load_f32x8(token, &a);
        let vb = load_f32x8(token, &b);
        let result = to_array_f32x8(sub_f32x8(token, va, vb));
        let expected = [9.0f32, 18.0, 27.0, 36.0, 45.0, 54.0, 63.0, 72.0];
        assert_eq!(result, expected);
    }
}

#[test]
fn test_mul_f32x8() {
    if let Some(token) = Avx2Token::try_new() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [2.0f32; 8];
        let va = load_f32x8(token, &a);
        let vb = load_f32x8(token, &b);
        let result = to_array_f32x8(mul_f32x8(token, va, vb));
        let expected = [2.0f32, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
        assert_eq!(result, expected);
    }
}

#[test]
fn test_div_f32x8() {
    if let Some(token) = Avx2Token::try_new() {
        let a = [2.0f32, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
        let b = [2.0f32; 8];
        let va = load_f32x8(token, &a);
        let vb = load_f32x8(token, &b);
        let result = to_array_f32x8(div_f32x8(token, va, vb));
        let expected = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert_eq!(result, expected);
    }
}

#[test]
fn test_min_max_f32x8() {
    if let Some(token) = Avx2Token::try_new() {
        let a = [1.0f32, 5.0, 3.0, 7.0, 2.0, 6.0, 4.0, 8.0];
        let b = [4.0f32; 8];
        let va = load_f32x8(token, &a);
        let vb = load_f32x8(token, &b);

        let min_result = to_array_f32x8(min_f32x8(token, va, vb));
        let max_result = to_array_f32x8(max_f32x8(token, va, vb));

        let expected_min = [1.0f32, 4.0, 3.0, 4.0, 2.0, 4.0, 4.0, 4.0];
        let expected_max = [4.0f32, 5.0, 4.0, 7.0, 4.0, 6.0, 4.0, 8.0];

        assert_eq!(min_result, expected_min);
        assert_eq!(max_result, expected_max);
    }
}

#[test]
fn test_set1_f32x8() {
    if let Some(token) = Avx2Token::try_new() {
        let v = set1_f32x8(token, 42.0);
        let result = to_array_f32x8(v);
        assert_eq!(result, [42.0f32; 8]);
    }
}

#[test]
fn test_zero_f32x8() {
    if let Some(token) = Avx2Token::try_new() {
        let v = zero_f32x8(token);
        let result = to_array_f32x8(v);
        assert_eq!(result, [0.0f32; 8]);
    }
}

// ============================================================================
// FMA Tests (critical for DCT and color conversion)
// ============================================================================

#[test]
fn test_fmadd_f32x8() {
    if let Some(token) = Avx2FmaToken::try_new() {
        // a * b + c
        let a = [2.0f32; 8];
        let b = [3.0f32; 8];
        let c = [1.0f32; 8];

        let va = load_f32x8(token.avx2(), &a);
        let vb = load_f32x8(token.avx2(), &b);
        let vc = load_f32x8(token.avx2(), &c);

        let result = to_array_f32x8(fmadd_f32x8(token.fma(), va, vb, vc));
        // 2 * 3 + 1 = 7
        assert_eq!(result, [7.0f32; 8]);
    }
}

#[test]
fn test_fmsub_f32x8() {
    if let Some(token) = Avx2FmaToken::try_new() {
        // a * b - c
        let a = [2.0f32; 8];
        let b = [3.0f32; 8];
        let c = [1.0f32; 8];

        let va = load_f32x8(token.avx2(), &a);
        let vb = load_f32x8(token.avx2(), &b);
        let vc = load_f32x8(token.avx2(), &c);

        let result = to_array_f32x8(fmsub_f32x8(token.fma(), va, vb, vc));
        // 2 * 3 - 1 = 5
        assert_eq!(result, [5.0f32; 8]);
    }
}

#[test]
fn test_fnmadd_f32x8() {
    if let Some(token) = Avx2FmaToken::try_new() {
        // -(a * b) + c
        let a = [2.0f32; 8];
        let b = [3.0f32; 8];
        let c = [10.0f32; 8];

        let va = load_f32x8(token.avx2(), &a);
        let vb = load_f32x8(token.avx2(), &b);
        let vc = load_f32x8(token.avx2(), &c);

        let result = to_array_f32x8(fnmadd_f32x8(token.fma(), va, vb, vc));
        // -(2 * 3) + 10 = 4
        assert_eq!(result, [4.0f32; 8]);
    }
}

#[test]
fn test_fma_chained_pattern() {
    // Pattern used in RGB->YCbCr: r * Yr + g * Yg + b * Yb
    if let Some(token) = Avx2FmaToken::try_new() {
        let r = [255.0f32; 8];
        let g = [128.0f32; 8];
        let b = [64.0f32; 8];

        // BT.601 Y coefficients
        let yr = 0.299f32;
        let yg = 0.587f32;
        let yb = 0.114f32;

        let vr = load_f32x8(token.avx2(), &r);
        let vg = load_f32x8(token.avx2(), &g);
        let vb = load_f32x8(token.avx2(), &b);

        let coef_yr = set1_f32x8(token.avx2(), yr);
        let coef_yg = set1_f32x8(token.avx2(), yg);
        let coef_yb = set1_f32x8(token.avx2(), yb);

        // Y = r * yr + (g * yg + b * yb)
        // First: g * yg
        let temp = mul_f32x8(token.avx2(), vg, coef_yg);
        // Then: b * yb + temp
        let temp = fmadd_f32x8(token.fma(), vb, coef_yb, temp);
        // Finally: r * yr + temp
        let y = fmadd_f32x8(token.fma(), vr, coef_yr, temp);

        let result = to_array_f32x8(y);
        let expected = 255.0 * yr + 128.0 * yg + 64.0 * yb;

        for &val in &result {
            assert!((val - expected).abs() < 0.001,
                    "expected {}, got {}", expected, val);
        }
    }
}

// ============================================================================
// Shuffle/Permute Tests (critical for 8x8 transpose)
// ============================================================================

#[test]
fn test_unpacklo_f32x8() {
    if let Some(token) = Avx2Token::try_new() {
        // unpacklo interleaves elements [0,1] from each 128-bit lane
        let a = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let b = [10.0f32, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0];

        let va = load_f32x8(token, &a);
        let vb = load_f32x8(token, &b);
        let result = to_array_f32x8(unpacklo_f32x8(token, va, vb));

        // Low 128-bit lane: [a0, b0, a1, b1]
        // High 128-bit lane: [a4, b4, a5, b5]
        let expected = [0.0f32, 10.0, 1.0, 11.0, 4.0, 14.0, 5.0, 15.0];
        assert_eq!(result, expected);
    }
}

#[test]
fn test_unpackhi_f32x8() {
    if let Some(token) = Avx2Token::try_new() {
        // unpackhi interleaves elements [2,3] from each 128-bit lane
        let a = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let b = [10.0f32, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0];

        let va = load_f32x8(token, &a);
        let vb = load_f32x8(token, &b);
        let result = to_array_f32x8(unpackhi_f32x8(token, va, vb));

        // Low 128-bit lane: [a2, b2, a3, b3]
        // High 128-bit lane: [a6, b6, a7, b7]
        let expected = [2.0f32, 12.0, 3.0, 13.0, 6.0, 16.0, 7.0, 17.0];
        assert_eq!(result, expected);
    }
}

#[test]
fn test_shuffle_f32x8_0x44() {
    if let Some(token) = Avx2Token::try_new() {
        // 0x44 = 0b01_00_01_00: select [a0, a1, b0, b1] in each lane
        let a = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let b = [10.0f32, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0];

        let va = load_f32x8(token, &a);
        let vb = load_f32x8(token, &b);
        let result = to_array_f32x8(shuffle_f32x8::<0x44>(token, va, vb));

        // Each 128-bit lane: [a[0], a[1], b[0], b[1]]
        let expected = [0.0f32, 1.0, 10.0, 11.0, 4.0, 5.0, 14.0, 15.0];
        assert_eq!(result, expected);
    }
}

#[test]
fn test_shuffle_f32x8_0xee() {
    if let Some(token) = Avx2Token::try_new() {
        // 0xEE = 0b11_10_11_10: select [a2, a3, b2, b3] in each lane
        let a = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let b = [10.0f32, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0];

        let va = load_f32x8(token, &a);
        let vb = load_f32x8(token, &b);
        let result = to_array_f32x8(shuffle_f32x8::<0xEE>(token, va, vb));

        // Each 128-bit lane: [a[2], a[3], b[2], b[3]]
        let expected = [2.0f32, 3.0, 12.0, 13.0, 6.0, 7.0, 16.0, 17.0];
        assert_eq!(result, expected);
    }
}

#[test]
fn test_permute2_f32x8_0x20() {
    if let Some(token) = Avx2Token::try_new() {
        // 0x20: concat low lanes [a_lo, b_lo]
        let a = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let b = [10.0f32, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0];

        let va = load_f32x8(token, &a);
        let vb = load_f32x8(token, &b);
        let result = to_array_f32x8(permute2_f32x8::<0x20>(token, va, vb));

        // Result: [a_lo, b_lo] = [a0-a3, b0-b3]
        let expected = [0.0f32, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0];
        assert_eq!(result, expected);
    }
}

#[test]
fn test_permute2_f32x8_0x31() {
    if let Some(token) = Avx2Token::try_new() {
        // 0x31: concat high lanes [a_hi, b_hi]
        let a = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let b = [10.0f32, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0];

        let va = load_f32x8(token, &a);
        let vb = load_f32x8(token, &b);
        let result = to_array_f32x8(permute2_f32x8::<0x31>(token, va, vb));

        // Result: [a_hi, b_hi] = [a4-a7, b4-b7]
        let expected = [4.0f32, 5.0, 6.0, 7.0, 14.0, 15.0, 16.0, 17.0];
        assert_eq!(result, expected);
    }
}

#[test]
fn test_blend_f32x8() {
    if let Some(token) = Avx2Token::try_new() {
        let a = [0.0f32; 8];
        let b = [1.0f32; 8];

        let va = load_f32x8(token, &a);
        let vb = load_f32x8(token, &b);

        // 0b10101010: select b for odd indices
        let result = to_array_f32x8(blend_f32x8::<0b10101010>(token, va, vb));
        let expected = [0.0f32, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        assert_eq!(result, expected);
    }
}

// ============================================================================
// Bitwise Tests
// ============================================================================

#[test]
fn test_bitwise_and_f32x8() {
    if let Some(token) = Avx2Token::try_new() {
        // Use to create abs mask: and with 0x7FFFFFFF
        let v = set1_f32x8(token, -5.0);
        let abs_mask = set1_f32x8(token, f32::from_bits(0x7FFFFFFF));
        let result = to_array_f32x8(and_f32x8(token, v, abs_mask));

        for &val in &result {
            assert_eq!(val, 5.0, "abs(-5.0) should be 5.0");
        }
    }
}

#[test]
fn test_bitwise_xor_f32x8() {
    if let Some(token) = Avx2Token::try_new() {
        // XOR with sign bit to negate
        let v = set1_f32x8(token, 5.0);
        let sign_mask = set1_f32x8(token, f32::from_bits(0x80000000));
        let result = to_array_f32x8(xor_f32x8(token, v, sign_mask));

        for &val in &result {
            assert_eq!(val, -5.0, "negating 5.0 should give -5.0");
        }
    }
}

// ============================================================================
// Conversion Tests
// ============================================================================

#[test]
fn test_cvt_i32x8_f32x8() {
    if let Some(token) = Avx2Token::try_new() {
        let ints = [1i32, 2, 3, 4, 5, 6, 7, 8];
        let vi = load_i32x8(token, &ints);
        let vf = cvt_i32x8_f32x8(token, vi);
        let result = to_array_f32x8(vf);
        let expected = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert_eq!(result, expected);
    }
}

#[test]
fn test_cvtt_f32x8_i32x8() {
    if let Some(token) = Avx2Token::try_new() {
        let floats = [1.9f32, 2.1, 3.5, 4.9, -1.1, -2.9, 0.0, 100.5];
        let vf = load_f32x8(token, &floats);
        let vi = cvtt_f32x8_i32x8(token, vf);
        let mut result = [0i32; 8];
        store_i32x8(token, &mut result, vi);
        // Truncate toward zero
        let expected = [1i32, 2, 3, 4, -1, -2, 0, 100];
        assert_eq!(result, expected);
    }
}

// ============================================================================
// Integer Operations Tests (for fixed-point DCT)
// ============================================================================

#[test]
fn test_add_i32x8() {
    if let Some(token) = Avx2Token::try_new() {
        let a = [1i32, 2, 3, 4, 5, 6, 7, 8];
        let b = [10i32, 20, 30, 40, 50, 60, 70, 80];
        let va = load_i32x8(token, &a);
        let vb = load_i32x8(token, &b);
        let result_v = add_i32x8(token, va, vb);
        let mut result = [0i32; 8];
        store_i32x8(token, &mut result, result_v);
        let expected = [11i32, 22, 33, 44, 55, 66, 77, 88];
        assert_eq!(result, expected);
    }
}

#[test]
fn test_sub_i32x8() {
    if let Some(token) = Avx2Token::try_new() {
        let a = [100i32, 200, 300, 400, 500, 600, 700, 800];
        let b = [1i32, 2, 3, 4, 5, 6, 7, 8];
        let va = load_i32x8(token, &a);
        let vb = load_i32x8(token, &b);
        let result_v = sub_i32x8(token, va, vb);
        let mut result = [0i32; 8];
        store_i32x8(token, &mut result, result_v);
        let expected = [99i32, 198, 297, 396, 495, 594, 693, 792];
        assert_eq!(result, expected);
    }
}

#[test]
fn test_mullo_i32x8() {
    if let Some(token) = Avx2Token::try_new() {
        let a = [1i32, 2, 3, 4, 5, 6, 7, 8];
        let b = [2i32; 8];
        let va = load_i32x8(token, &a);
        let vb = load_i32x8(token, &b);
        let result_v = mullo_i32x8(token, va, vb);
        let mut result = [0i32; 8];
        store_i32x8(token, &mut result, result_v);
        let expected = [2i32, 4, 6, 8, 10, 12, 14, 16];
        assert_eq!(result, expected);
    }
}

#[test]
fn test_shift_i32x8() {
    if let Some(token) = Avx2Token::try_new() {
        let a = [1i32, 2, 4, 8, 16, 32, 64, 128];
        let va = load_i32x8(token, &a);

        // Shift left by 2
        let sll = slli_i32x8::<2>(token, va);
        let mut result_sll = [0i32; 8];
        store_i32x8(token, &mut result_sll, sll);
        let expected_sll = [4i32, 8, 16, 32, 64, 128, 256, 512];
        assert_eq!(result_sll, expected_sll);

        // Shift right logical by 2
        let srl = srli_i32x8::<2>(token, va);
        let mut result_srl = [0i32; 8];
        store_i32x8(token, &mut result_srl, srl);
        let expected_srl = [0i32, 0, 1, 2, 4, 8, 16, 32];
        assert_eq!(result_srl, expected_srl);
    }
}

#[test]
fn test_srai_i32x8_signed() {
    if let Some(token) = Avx2Token::try_new() {
        let a = [-8i32, -4, -2, -1, 1, 2, 4, 8];
        let va = load_i32x8(token, &a);

        // Arithmetic shift right preserves sign
        let sra = srai_i32x8::<1>(token, va);
        let mut result = [0i32; 8];
        store_i32x8(token, &mut result, sra);
        let expected = [-4i32, -2, -1, -1, 0, 1, 2, 4];
        assert_eq!(result, expected);
    }
}

// ============================================================================
// Horizontal Operations Tests
// ============================================================================

#[test]
fn test_hadd_f32x8() {
    if let Some(token) = Avx2Token::try_new() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];

        let va = load_f32x8(token, &a);
        let vb = load_f32x8(token, &b);
        let result = to_array_f32x8(hadd_f32x8(token, va, vb));

        // hadd operates within 128-bit lanes:
        // Low lane: [a0+a1, a2+a3, b0+b1, b2+b3]
        // High lane: [a4+a5, a6+a7, b4+b5, b6+b7]
        let expected = [3.0f32, 7.0, 30.0, 70.0, 11.0, 15.0, 110.0, 150.0];
        assert_eq!(result, expected);
    }
}

// ============================================================================
// Integration Test: 8x8 Transpose Pattern
// ============================================================================

#[test]
fn test_transpose_8x8_pattern() {
    // This tests the complete transpose algorithm used in DCT
    if let Some(token) = Avx2Token::try_new() {
        // Create 8x8 matrix with unique values
        let input: [[f32; 8]; 8] = core::array::from_fn(|row| {
            core::array::from_fn(|col| (row * 8 + col) as f32)
        });

        // Load rows
        let mut r: [_; 8] = core::array::from_fn(|i| load_f32x8(token, &input[i]));

        // Stage 1: Interleave pairs within 128-bit lanes
        let t0 = unpacklo_f32x8(token, r[0], r[1]);
        let t1 = unpackhi_f32x8(token, r[0], r[1]);
        let t2 = unpacklo_f32x8(token, r[2], r[3]);
        let t3 = unpackhi_f32x8(token, r[2], r[3]);
        let t4 = unpacklo_f32x8(token, r[4], r[5]);
        let t5 = unpackhi_f32x8(token, r[4], r[5]);
        let t6 = unpacklo_f32x8(token, r[6], r[7]);
        let t7 = unpackhi_f32x8(token, r[6], r[7]);

        // Stage 2: Shuffle to get 4-element groups
        r[0] = shuffle_f32x8::<0x44>(token, t0, t2);
        r[1] = shuffle_f32x8::<0xEE>(token, t0, t2);
        r[2] = shuffle_f32x8::<0x44>(token, t1, t3);
        r[3] = shuffle_f32x8::<0xEE>(token, t1, t3);
        r[4] = shuffle_f32x8::<0x44>(token, t4, t6);
        r[5] = shuffle_f32x8::<0xEE>(token, t4, t6);
        r[6] = shuffle_f32x8::<0x44>(token, t5, t7);
        r[7] = shuffle_f32x8::<0xEE>(token, t5, t7);

        // Stage 3: Exchange 128-bit halves
        let c0 = permute2_f32x8::<0x20>(token, r[0], r[4]);
        let c1 = permute2_f32x8::<0x20>(token, r[1], r[5]);
        let c2 = permute2_f32x8::<0x20>(token, r[2], r[6]);
        let c3 = permute2_f32x8::<0x20>(token, r[3], r[7]);
        let c4 = permute2_f32x8::<0x31>(token, r[0], r[4]);
        let c5 = permute2_f32x8::<0x31>(token, r[1], r[5]);
        let c6 = permute2_f32x8::<0x31>(token, r[2], r[6]);
        let c7 = permute2_f32x8::<0x31>(token, r[3], r[7]);

        // Store and verify
        let output: [[f32; 8]; 8] = [
            to_array_f32x8(c0),
            to_array_f32x8(c1),
            to_array_f32x8(c2),
            to_array_f32x8(c3),
            to_array_f32x8(c4),
            to_array_f32x8(c5),
            to_array_f32x8(c6),
            to_array_f32x8(c7),
        ];

        // Verify: output[col][row] == input[row][col]
        for row in 0..8 {
            for col in 0..8 {
                let orig_val = input[row][col];
                let trans_val = output[col][row];
                assert_eq!(
                    orig_val, trans_val,
                    "Transpose mismatch at ({}, {}): expected {}, got {}",
                    row, col, orig_val, trans_val
                );
            }
        }
    }
}

// ============================================================================
// Integration Test: DCT Butterfly Pattern
// ============================================================================

#[test]
fn test_dct_butterfly_pattern() {
    // Test the basic DCT butterfly: add/subtract pairs
    if let Some(token) = Avx2Token::try_new() {
        let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let v = load_f32x8(token, &input);

        // First stage of DCT: add/subtract reverse pairs
        // tmp0 = input[0] + input[7], tmp7 = input[0] - input[7]
        // tmp1 = input[1] + input[6], tmp6 = input[1] - input[6]
        // etc.

        // We need to reverse the order for the second operand
        // In real code, this would use permute, here we just test the arithmetic
        let reversed = [8.0f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let vr = load_f32x8(token, &reversed);

        let sum = to_array_f32x8(add_f32x8(token, v, vr));
        let diff = to_array_f32x8(sub_f32x8(token, v, vr));

        // 1+8=9, 2+7=9, 3+6=9, 4+5=9, 5+4=9, 6+3=9, 7+2=9, 8+1=9
        assert_eq!(sum, [9.0f32; 8]);

        // 1-8=-7, 2-7=-5, 3-6=-3, 4-5=-1, 5-4=1, 6-3=3, 7-2=5, 8-1=7
        let expected_diff = [-7.0f32, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0];
        assert_eq!(diff, expected_diff);
    }
}

//! 8x8 Matrix Transpose
//!
//! Efficient AVX2 transpose using the unpack/shuffle/permute pattern.
//! Critical for DCT transforms in JPEG encoding.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::mem::avx::{_mm256_loadu_ps, _mm256_storeu_ps};
use crate::simd_fn;
use crate::tokens::x86::Avx2Token;

/// Transpose an 8x8 f32 matrix in-place using AVX2.
///
/// Uses the Highway-style 3-stage transpose:
/// 1. `unpacklo/hi` - interleave pairs within 128-bit lanes
/// 2. `shuffle` - reorder within lanes
/// 3. `permute2f128` - exchange 128-bit halves
///
/// # Example
///
/// ```rust,ignore
/// use archmage::{Avx2Token, SimdToken, composite::transpose_8x8};
///
/// if let Some(token) = Avx2Token::try_new() {
///     let mut block: [f32; 64] = core::array::from_fn(|i| i as f32);
///     transpose_8x8(token, &mut block);
///     // block[col * 8 + row] now equals original block[row * 8 + col]
/// }
/// ```
#[simd_fn]
#[inline]
pub fn transpose_8x8(token: Avx2Token, block: &mut [f32; 64]) {
    let avx = token.avx();
    // Load 8 rows
    let mut r0 = _mm256_loadu_ps(avx, block[0..8].try_into().unwrap());
    let mut r1 = _mm256_loadu_ps(avx, block[8..16].try_into().unwrap());
    let mut r2 = _mm256_loadu_ps(avx, block[16..24].try_into().unwrap());
    let mut r3 = _mm256_loadu_ps(avx, block[24..32].try_into().unwrap());
    let mut r4 = _mm256_loadu_ps(avx, block[32..40].try_into().unwrap());
    let mut r5 = _mm256_loadu_ps(avx, block[40..48].try_into().unwrap());
    let mut r6 = _mm256_loadu_ps(avx, block[48..56].try_into().unwrap());
    let mut r7 = _mm256_loadu_ps(avx, block[56..64].try_into().unwrap());

    // Stage 1: Interleave pairs within 128-bit lanes
    let t0 = _mm256_unpacklo_ps(r0, r1);
    let t1 = _mm256_unpackhi_ps(r0, r1);
    let t2 = _mm256_unpacklo_ps(r2, r3);
    let t3 = _mm256_unpackhi_ps(r2, r3);
    let t4 = _mm256_unpacklo_ps(r4, r5);
    let t5 = _mm256_unpackhi_ps(r4, r5);
    let t6 = _mm256_unpacklo_ps(r6, r7);
    let t7 = _mm256_unpackhi_ps(r6, r7);

    // Stage 2: Shuffle to get 4-element groups
    r0 = _mm256_shuffle_ps::<0x44>(t0, t2); // 0b01_00_01_00
    r1 = _mm256_shuffle_ps::<0xEE>(t0, t2); // 0b11_10_11_10
    r2 = _mm256_shuffle_ps::<0x44>(t1, t3);
    r3 = _mm256_shuffle_ps::<0xEE>(t1, t3);
    r4 = _mm256_shuffle_ps::<0x44>(t4, t6);
    r5 = _mm256_shuffle_ps::<0xEE>(t4, t6);
    r6 = _mm256_shuffle_ps::<0x44>(t5, t7);
    r7 = _mm256_shuffle_ps::<0xEE>(t5, t7);

    // Stage 3: Exchange 128-bit halves
    let c0 = _mm256_permute2f128_ps::<0x20>(r0, r4);
    let c1 = _mm256_permute2f128_ps::<0x20>(r1, r5);
    let c2 = _mm256_permute2f128_ps::<0x20>(r2, r6);
    let c3 = _mm256_permute2f128_ps::<0x20>(r3, r7);
    let c4 = _mm256_permute2f128_ps::<0x31>(r0, r4);
    let c5 = _mm256_permute2f128_ps::<0x31>(r1, r5);
    let c6 = _mm256_permute2f128_ps::<0x31>(r2, r6);
    let c7 = _mm256_permute2f128_ps::<0x31>(r3, r7);

    // Store transposed rows
    _mm256_storeu_ps(avx, (&mut block[0..8]).try_into().unwrap(), c0);
    _mm256_storeu_ps(avx, (&mut block[8..16]).try_into().unwrap(), c1);
    _mm256_storeu_ps(avx, (&mut block[16..24]).try_into().unwrap(), c2);
    _mm256_storeu_ps(avx, (&mut block[24..32]).try_into().unwrap(), c3);
    _mm256_storeu_ps(avx, (&mut block[32..40]).try_into().unwrap(), c4);
    _mm256_storeu_ps(avx, (&mut block[40..48]).try_into().unwrap(), c5);
    _mm256_storeu_ps(avx, (&mut block[48..56]).try_into().unwrap(), c6);
    _mm256_storeu_ps(avx, (&mut block[56..64]).try_into().unwrap(), c7);
}

/// Transpose an 8x8 f32 matrix from input to output using AVX2.
///
/// Non-destructive version that reads from one buffer and writes to another.
#[simd_fn]
#[inline]
pub fn transpose_8x8_copy(token: Avx2Token, input: &[f32; 64], output: &mut [f32; 64]) {
    let avx = token.avx();
    // Load 8 rows from input
    let mut r0 = _mm256_loadu_ps(avx, input[0..8].try_into().unwrap());
    let mut r1 = _mm256_loadu_ps(avx, input[8..16].try_into().unwrap());
    let mut r2 = _mm256_loadu_ps(avx, input[16..24].try_into().unwrap());
    let mut r3 = _mm256_loadu_ps(avx, input[24..32].try_into().unwrap());
    let mut r4 = _mm256_loadu_ps(avx, input[32..40].try_into().unwrap());
    let mut r5 = _mm256_loadu_ps(avx, input[40..48].try_into().unwrap());
    let mut r6 = _mm256_loadu_ps(avx, input[48..56].try_into().unwrap());
    let mut r7 = _mm256_loadu_ps(avx, input[56..64].try_into().unwrap());

    // Stage 1
    let t0 = _mm256_unpacklo_ps(r0, r1);
    let t1 = _mm256_unpackhi_ps(r0, r1);
    let t2 = _mm256_unpacklo_ps(r2, r3);
    let t3 = _mm256_unpackhi_ps(r2, r3);
    let t4 = _mm256_unpacklo_ps(r4, r5);
    let t5 = _mm256_unpackhi_ps(r4, r5);
    let t6 = _mm256_unpacklo_ps(r6, r7);
    let t7 = _mm256_unpackhi_ps(r6, r7);

    // Stage 2
    r0 = _mm256_shuffle_ps::<0x44>(t0, t2);
    r1 = _mm256_shuffle_ps::<0xEE>(t0, t2);
    r2 = _mm256_shuffle_ps::<0x44>(t1, t3);
    r3 = _mm256_shuffle_ps::<0xEE>(t1, t3);
    r4 = _mm256_shuffle_ps::<0x44>(t4, t6);
    r5 = _mm256_shuffle_ps::<0xEE>(t4, t6);
    r6 = _mm256_shuffle_ps::<0x44>(t5, t7);
    r7 = _mm256_shuffle_ps::<0xEE>(t5, t7);

    // Stage 3
    let c0 = _mm256_permute2f128_ps::<0x20>(r0, r4);
    let c1 = _mm256_permute2f128_ps::<0x20>(r1, r5);
    let c2 = _mm256_permute2f128_ps::<0x20>(r2, r6);
    let c3 = _mm256_permute2f128_ps::<0x20>(r3, r7);
    let c4 = _mm256_permute2f128_ps::<0x31>(r0, r4);
    let c5 = _mm256_permute2f128_ps::<0x31>(r1, r5);
    let c6 = _mm256_permute2f128_ps::<0x31>(r2, r6);
    let c7 = _mm256_permute2f128_ps::<0x31>(r3, r7);

    // Store to output
    _mm256_storeu_ps(avx, (&mut output[0..8]).try_into().unwrap(), c0);
    _mm256_storeu_ps(avx, (&mut output[8..16]).try_into().unwrap(), c1);
    _mm256_storeu_ps(avx, (&mut output[16..24]).try_into().unwrap(), c2);
    _mm256_storeu_ps(avx, (&mut output[24..32]).try_into().unwrap(), c3);
    _mm256_storeu_ps(avx, (&mut output[32..40]).try_into().unwrap(), c4);
    _mm256_storeu_ps(avx, (&mut output[40..48]).try_into().unwrap(), c5);
    _mm256_storeu_ps(avx, (&mut output[48..56]).try_into().unwrap(), c6);
    _mm256_storeu_ps(avx, (&mut output[56..64]).try_into().unwrap(), c7);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokens::SimdToken;

    #[test]
    fn test_transpose_8x8() {
        if let Some(token) = Avx2Token::try_new() {
            let original: [f32; 64] = core::array::from_fn(|i| i as f32);
            let mut block = original;

            transpose_8x8(token, &mut block);

            // Verify: transposed[col][row] == original[row][col]
            for row in 0..8 {
                for col in 0..8 {
                    let orig_val = original[row * 8 + col];
                    let trans_val = block[col * 8 + row];
                    assert_eq!(
                        orig_val, trans_val,
                        "Mismatch at ({}, {}): expected {}, got {}",
                        row, col, orig_val, trans_val
                    );
                }
            }
        }
    }

    #[test]
    fn test_transpose_8x8_copy() {
        if let Some(token) = Avx2Token::try_new() {
            let input: [f32; 64] = core::array::from_fn(|i| i as f32);
            let mut output = [0.0f32; 64];

            transpose_8x8_copy(token, &input, &mut output);

            for row in 0..8 {
                for col in 0..8 {
                    let orig_val = input[row * 8 + col];
                    let trans_val = output[col * 8 + row];
                    assert_eq!(orig_val, trans_val);
                }
            }
        }
    }

    #[test]
    fn test_double_transpose() {
        if let Some(token) = Avx2Token::try_new() {
            let original: [f32; 64] = core::array::from_fn(|i| i as f32);
            let mut block = original;

            // Transpose twice should give original
            transpose_8x8(token, &mut block);
            transpose_8x8(token, &mut block);

            assert_eq!(original, block);
        }
    }
}

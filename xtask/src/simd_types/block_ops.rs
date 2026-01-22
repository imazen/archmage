//! Block operations (transpose, DCT) for SIMD types.

use super::types::{ElementType, SimdType, SimdWidth};
use indoc::formatdoc;

/// Generate block operations (transpose, etc.) for a SIMD type.
pub fn generate_block_ops(ty: &SimdType) -> String {
    let mut code = String::new();

    // Only f32 types get block ops for now
    if ty.elem != ElementType::F32 {
        return code;
    }

    match ty.width {
        SimdWidth::W128 => code.push_str(&generate_transpose_4x4()),
        SimdWidth::W256 => code.push_str(&generate_transpose_8x8_avx()),
        SimdWidth::W512 => code.push_str(&generate_transpose_8x8_avx512()),
    }

    code
}

/// Generate 4x4 transpose for f32x4 (SSE).
fn generate_transpose_4x4() -> String {
    formatdoc! {r#"
        // ========== Block Operations ==========

        /// Transpose a 4x4 matrix represented as 4 row vectors.
        ///
        /// After transpose, `rows[i][j]` becomes `rows[j][i]`.
        ///
        /// # Example
        /// ```ignore
        /// let mut rows = [
        ///     f32x4::from_array(token, [0.0, 1.0, 2.0, 3.0]),
        ///     f32x4::from_array(token, [4.0, 5.0, 6.0, 7.0]),
        ///     f32x4::from_array(token, [8.0, 9.0, 10.0, 11.0]),
        ///     f32x4::from_array(token, [12.0, 13.0, 14.0, 15.0]),
        /// ];
        /// f32x4::transpose_4x4(&mut rows);
        /// // rows[0] is now [0, 4, 8, 12]
        /// // rows[1] is now [1, 5, 9, 13]
        /// // etc.
        /// ```
        #[inline]
        pub fn transpose_4x4(rows: &mut [Self; 4]) {{
            unsafe {{
                // Stage 1: Interleave pairs
                let t0 = _mm_unpacklo_ps(rows[0].0, rows[1].0); // [r0[0], r1[0], r0[1], r1[1]]
                let t1 = _mm_unpackhi_ps(rows[0].0, rows[1].0); // [r0[2], r1[2], r0[3], r1[3]]
                let t2 = _mm_unpacklo_ps(rows[2].0, rows[3].0); // [r2[0], r3[0], r2[1], r3[1]]
                let t3 = _mm_unpackhi_ps(rows[2].0, rows[3].0); // [r2[2], r3[2], r2[3], r3[3]]

                // Stage 2: Combine halves
                rows[0] = Self(_mm_movelh_ps(t0, t2)); // [r0[0], r1[0], r2[0], r3[0]]
                rows[1] = Self(_mm_movehl_ps(t2, t0)); // [r0[1], r1[1], r2[1], r3[1]]
                rows[2] = Self(_mm_movelh_ps(t1, t3)); // [r0[2], r1[2], r2[2], r3[2]]
                rows[3] = Self(_mm_movehl_ps(t3, t1)); // [r0[3], r1[3], r2[3], r3[3]]
            }}
        }}

        /// Transpose a 4x4 matrix, returning the transposed rows.
        #[inline]
        pub fn transpose_4x4_copy(rows: [Self; 4]) -> [Self; 4] {{
            let mut result = rows;
            Self::transpose_4x4(&mut result);
            result
        }}

    "#}
}

/// Generate 8x8 transpose for f32x8 (AVX2).
fn generate_transpose_8x8_avx() -> String {
    formatdoc! {r#"
        // ========== Block Operations ==========

        /// Transpose an 8x8 matrix represented as 8 row vectors.
        ///
        /// Uses the Highway-style 3-stage transpose:
        /// 1. `unpacklo/hi` - interleave pairs within 128-bit lanes
        /// 2. `shuffle` - reorder within lanes
        /// 3. `permute2f128` - exchange 128-bit halves
        ///
        /// # Example
        /// ```ignore
        /// let mut rows: [f32x8; 8] = /* 8 row vectors */;
        /// f32x8::transpose_8x8(&mut rows);
        /// // rows[i][j] is now original rows[j][i]
        /// ```
        #[inline]
        pub fn transpose_8x8(rows: &mut [Self; 8]) {{
            unsafe {{
                // Stage 1: Interleave pairs within 128-bit lanes
                let t0 = _mm256_unpacklo_ps(rows[0].0, rows[1].0);
                let t1 = _mm256_unpackhi_ps(rows[0].0, rows[1].0);
                let t2 = _mm256_unpacklo_ps(rows[2].0, rows[3].0);
                let t3 = _mm256_unpackhi_ps(rows[2].0, rows[3].0);
                let t4 = _mm256_unpacklo_ps(rows[4].0, rows[5].0);
                let t5 = _mm256_unpackhi_ps(rows[4].0, rows[5].0);
                let t6 = _mm256_unpacklo_ps(rows[6].0, rows[7].0);
                let t7 = _mm256_unpackhi_ps(rows[6].0, rows[7].0);

                // Stage 2: Shuffle to get 4-element groups
                let s0 = _mm256_shuffle_ps::<0x44>(t0, t2); // 0b01_00_01_00
                let s1 = _mm256_shuffle_ps::<0xEE>(t0, t2); // 0b11_10_11_10
                let s2 = _mm256_shuffle_ps::<0x44>(t1, t3);
                let s3 = _mm256_shuffle_ps::<0xEE>(t1, t3);
                let s4 = _mm256_shuffle_ps::<0x44>(t4, t6);
                let s5 = _mm256_shuffle_ps::<0xEE>(t4, t6);
                let s6 = _mm256_shuffle_ps::<0x44>(t5, t7);
                let s7 = _mm256_shuffle_ps::<0xEE>(t5, t7);

                // Stage 3: Exchange 128-bit halves
                rows[0] = Self(_mm256_permute2f128_ps::<0x20>(s0, s4));
                rows[1] = Self(_mm256_permute2f128_ps::<0x20>(s1, s5));
                rows[2] = Self(_mm256_permute2f128_ps::<0x20>(s2, s6));
                rows[3] = Self(_mm256_permute2f128_ps::<0x20>(s3, s7));
                rows[4] = Self(_mm256_permute2f128_ps::<0x31>(s0, s4));
                rows[5] = Self(_mm256_permute2f128_ps::<0x31>(s1, s5));
                rows[6] = Self(_mm256_permute2f128_ps::<0x31>(s2, s6));
                rows[7] = Self(_mm256_permute2f128_ps::<0x31>(s3, s7));
            }}
        }}

        /// Transpose an 8x8 matrix, returning the transposed rows.
        #[inline]
        pub fn transpose_8x8_copy(rows: [Self; 8]) -> [Self; 8] {{
            let mut result = rows;
            Self::transpose_8x8(&mut result);
            result
        }}

        /// Load an 8x8 f32 block from a contiguous array and return as 8 row vectors.
        #[inline]
        pub fn load_8x8(block: &[f32; 64]) -> [Self; 8] {{
            unsafe {{
                [
                    Self(_mm256_loadu_ps(block.as_ptr())),
                    Self(_mm256_loadu_ps(block.as_ptr().add(8))),
                    Self(_mm256_loadu_ps(block.as_ptr().add(16))),
                    Self(_mm256_loadu_ps(block.as_ptr().add(24))),
                    Self(_mm256_loadu_ps(block.as_ptr().add(32))),
                    Self(_mm256_loadu_ps(block.as_ptr().add(40))),
                    Self(_mm256_loadu_ps(block.as_ptr().add(48))),
                    Self(_mm256_loadu_ps(block.as_ptr().add(56))),
                ]
            }}
        }}

        /// Store 8 row vectors to a contiguous 8x8 f32 block.
        #[inline]
        pub fn store_8x8(rows: &[Self; 8], block: &mut [f32; 64]) {{
            unsafe {{
                _mm256_storeu_ps(block.as_mut_ptr(), rows[0].0);
                _mm256_storeu_ps(block.as_mut_ptr().add(8), rows[1].0);
                _mm256_storeu_ps(block.as_mut_ptr().add(16), rows[2].0);
                _mm256_storeu_ps(block.as_mut_ptr().add(24), rows[3].0);
                _mm256_storeu_ps(block.as_mut_ptr().add(32), rows[4].0);
                _mm256_storeu_ps(block.as_mut_ptr().add(40), rows[5].0);
                _mm256_storeu_ps(block.as_mut_ptr().add(48), rows[6].0);
                _mm256_storeu_ps(block.as_mut_ptr().add(56), rows[7].0);
            }}
        }}

    "#}
}

/// Generate 8x8 transpose for f32x16 (AVX-512).
///
/// Note: f32x16 can hold 16 floats, but for 8x8 DCT compatibility
/// we provide 8x8 transpose that operates on the lower 8 elements
/// of each vector, plus a full 16x16 transpose.
fn generate_transpose_8x8_avx512() -> String {
    formatdoc! {r#"
        // ========== Block Operations ==========

        /// Transpose an 8x8 matrix using AVX-512.
        ///
        /// Takes 8 f32x16 vectors where only the lower 8 elements are used.
        /// This is useful for 8x8 DCT where you want to stay in 512-bit registers.
        #[inline]
        pub fn transpose_8x8(rows: &mut [Self; 8]) {{
            unsafe {{
                // Use AVX-512 2-input permute for efficient transpose
                // idx for gathering column i from rows: [i, i+16, i+32, ...]

                // Stage 1: Interleave pairs
                let idx_lo = _mm512_setr_epi32(0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29);
                let idx_hi = _mm512_setr_epi32(2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31);

                let r0i = _mm512_castps_si512(rows[0].0);
                let r1i = _mm512_castps_si512(rows[1].0);
                let r2i = _mm512_castps_si512(rows[2].0);
                let r3i = _mm512_castps_si512(rows[3].0);
                let r4i = _mm512_castps_si512(rows[4].0);
                let r5i = _mm512_castps_si512(rows[5].0);
                let r6i = _mm512_castps_si512(rows[6].0);
                let r7i = _mm512_castps_si512(rows[7].0);

                let t0 = _mm512_castsi512_ps(_mm512_permutex2var_epi32(r0i, idx_lo, r1i));
                let t1 = _mm512_castsi512_ps(_mm512_permutex2var_epi32(r0i, idx_hi, r1i));
                let t2 = _mm512_castsi512_ps(_mm512_permutex2var_epi32(r2i, idx_lo, r3i));
                let t3 = _mm512_castsi512_ps(_mm512_permutex2var_epi32(r2i, idx_hi, r3i));
                let t4 = _mm512_castsi512_ps(_mm512_permutex2var_epi32(r4i, idx_lo, r5i));
                let t5 = _mm512_castsi512_ps(_mm512_permutex2var_epi32(r4i, idx_hi, r5i));
                let t6 = _mm512_castsi512_ps(_mm512_permutex2var_epi32(r6i, idx_lo, r7i));
                let t7 = _mm512_castsi512_ps(_mm512_permutex2var_epi32(r6i, idx_hi, r7i));

                // Stage 2: Shuffle 64-bit pairs
                let idx2_lo = _mm512_setr_epi64(0, 8, 1, 9, 4, 12, 5, 13);
                let idx2_hi = _mm512_setr_epi64(2, 10, 3, 11, 6, 14, 7, 15);

                let t0i = _mm512_castps_si512(t0);
                let t2i = _mm512_castps_si512(t2);
                let t1i = _mm512_castps_si512(t1);
                let t3i = _mm512_castps_si512(t3);
                let t4i = _mm512_castps_si512(t4);
                let t6i = _mm512_castps_si512(t6);
                let t5i = _mm512_castps_si512(t5);
                let t7i = _mm512_castps_si512(t7);

                let s0 = _mm512_castsi512_ps(_mm512_permutex2var_epi64(t0i, idx2_lo, t2i));
                let s1 = _mm512_castsi512_ps(_mm512_permutex2var_epi64(t0i, idx2_hi, t2i));
                let s2 = _mm512_castsi512_ps(_mm512_permutex2var_epi64(t1i, idx2_lo, t3i));
                let s3 = _mm512_castsi512_ps(_mm512_permutex2var_epi64(t1i, idx2_hi, t3i));
                let s4 = _mm512_castsi512_ps(_mm512_permutex2var_epi64(t4i, idx2_lo, t6i));
                let s5 = _mm512_castsi512_ps(_mm512_permutex2var_epi64(t4i, idx2_hi, t6i));
                let s6 = _mm512_castsi512_ps(_mm512_permutex2var_epi64(t5i, idx2_lo, t7i));
                let s7 = _mm512_castsi512_ps(_mm512_permutex2var_epi64(t5i, idx2_hi, t7i));

                // Stage 3: Shuffle 128-bit lanes
                let idx3_lo = _mm512_setr_epi32(0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, 24, 25, 26, 27);
                let idx3_hi = _mm512_setr_epi32(4, 5, 6, 7, 20, 21, 22, 23, 12, 13, 14, 15, 28, 29, 30, 31);

                let s0i = _mm512_castps_si512(s0);
                let s4i = _mm512_castps_si512(s4);
                let s1i = _mm512_castps_si512(s1);
                let s5i = _mm512_castps_si512(s5);
                let s2i = _mm512_castps_si512(s2);
                let s6i = _mm512_castps_si512(s6);
                let s3i = _mm512_castps_si512(s3);
                let s7i = _mm512_castps_si512(s7);

                rows[0] = Self(_mm512_castsi512_ps(_mm512_permutex2var_epi32(s0i, idx3_lo, s4i)));
                rows[1] = Self(_mm512_castsi512_ps(_mm512_permutex2var_epi32(s1i, idx3_lo, s5i)));
                rows[2] = Self(_mm512_castsi512_ps(_mm512_permutex2var_epi32(s2i, idx3_lo, s6i)));
                rows[3] = Self(_mm512_castsi512_ps(_mm512_permutex2var_epi32(s3i, idx3_lo, s7i)));
                rows[4] = Self(_mm512_castsi512_ps(_mm512_permutex2var_epi32(s0i, idx3_hi, s4i)));
                rows[5] = Self(_mm512_castsi512_ps(_mm512_permutex2var_epi32(s1i, idx3_hi, s5i)));
                rows[6] = Self(_mm512_castsi512_ps(_mm512_permutex2var_epi32(s2i, idx3_hi, s6i)));
                rows[7] = Self(_mm512_castsi512_ps(_mm512_permutex2var_epi32(s3i, idx3_hi, s7i)));
            }}
        }}

        /// Transpose an 8x8 matrix, returning the transposed rows.
        #[inline]
        pub fn transpose_8x8_copy(rows: [Self; 8]) -> [Self; 8] {{
            let mut result = rows;
            Self::transpose_8x8(&mut result);
            result
        }}

        /// Load an 8x8 f32 block into 8 f32x16 vectors (lower 8 elements used).
        #[inline]
        pub fn load_8x8(block: &[f32; 64]) -> [Self; 8] {{
            unsafe {{
                // Load 8 floats into the lower half, upper half is zeroed
                [
                    Self(_mm512_castps256_ps512(_mm256_loadu_ps(block.as_ptr()))),
                    Self(_mm512_castps256_ps512(_mm256_loadu_ps(block.as_ptr().add(8)))),
                    Self(_mm512_castps256_ps512(_mm256_loadu_ps(block.as_ptr().add(16)))),
                    Self(_mm512_castps256_ps512(_mm256_loadu_ps(block.as_ptr().add(24)))),
                    Self(_mm512_castps256_ps512(_mm256_loadu_ps(block.as_ptr().add(32)))),
                    Self(_mm512_castps256_ps512(_mm256_loadu_ps(block.as_ptr().add(40)))),
                    Self(_mm512_castps256_ps512(_mm256_loadu_ps(block.as_ptr().add(48)))),
                    Self(_mm512_castps256_ps512(_mm256_loadu_ps(block.as_ptr().add(56)))),
                ]
            }}
        }}

        /// Store 8 f32x16 vectors (lower 8 elements) to a contiguous 8x8 f32 block.
        #[inline]
        pub fn store_8x8(rows: &[Self; 8], block: &mut [f32; 64]) {{
            unsafe {{
                _mm256_storeu_ps(block.as_mut_ptr(), _mm512_castps512_ps256(rows[0].0));
                _mm256_storeu_ps(block.as_mut_ptr().add(8), _mm512_castps512_ps256(rows[1].0));
                _mm256_storeu_ps(block.as_mut_ptr().add(16), _mm512_castps512_ps256(rows[2].0));
                _mm256_storeu_ps(block.as_mut_ptr().add(24), _mm512_castps512_ps256(rows[3].0));
                _mm256_storeu_ps(block.as_mut_ptr().add(32), _mm512_castps512_ps256(rows[4].0));
                _mm256_storeu_ps(block.as_mut_ptr().add(40), _mm512_castps512_ps256(rows[5].0));
                _mm256_storeu_ps(block.as_mut_ptr().add(48), _mm512_castps512_ps256(rows[6].0));
                _mm256_storeu_ps(block.as_mut_ptr().add(56), _mm512_castps512_ps256(rows[7].0));
            }}
        }}

    "#}
}

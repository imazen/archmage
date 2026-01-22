//! Block operations (transpose, extend/widen, interleave) for SIMD types.

use super::types::{ElementType, SimdType, SimdWidth};
use indoc::formatdoc;

/// Generate block operations for a SIMD type.
pub fn generate_block_ops(ty: &SimdType) -> String {
    let mut code = String::new();

    // Extend/widen operations
    code.push_str(&generate_extend_ops(ty));

    // Interleave/deinterleave
    code.push_str(&generate_interleave_ops(ty));

    // Pack/narrow operations
    code.push_str(&generate_pack_ops(ty));

    // Transpose (f32 only)
    code.push_str(&generate_transpose_ops(ty));

    code
}

/// Generate extend/widen operations.
fn generate_extend_ops(ty: &SimdType) -> String {
    let mut code = String::new();

    match (ty.elem, ty.width) {
        // ========== u8 → i16 ==========
        (ElementType::U8, SimdWidth::W128) => {
            code.push_str(&formatdoc! {r#"
                // ========== Extend/Widen Operations ==========

                /// Zero-extend low 8 u8 values to i16x8.
                ///
                /// Takes the lower 8 bytes and zero-extends each to 16 bits.
                #[inline(always)]
                pub fn extend_lo_i16(self) -> i16x8 {{
                    i16x8(unsafe {{ _mm_cvtepu8_epi16(self.0) }})
                }}

                /// Zero-extend high 8 u8 values to i16x8.
                ///
                /// Takes the upper 8 bytes and zero-extends each to 16 bits.
                #[inline(always)]
                pub fn extend_hi_i16(self) -> i16x8 {{
                    i16x8(unsafe {{
                        // Shift right by 8 bytes to get high half into low position
                        let hi = _mm_srli_si128::<8>(self.0);
                        _mm_cvtepu8_epi16(hi)
                    }})
                }}

                /// Zero-extend all 16 u8 values to two i16x8 vectors.
                ///
                /// Returns (low 8 as i16x8, high 8 as i16x8).
                #[inline(always)]
                pub fn extend_i16(self) -> (i16x8, i16x8) {{
                    (self.extend_lo_i16(), self.extend_hi_i16())
                }}

                /// Zero-extend low 4 u8 values to i32x4.
                #[inline(always)]
                pub fn extend_lo_i32(self) -> i32x4 {{
                    i32x4(unsafe {{ _mm_cvtepu8_epi32(self.0) }})
                }}

                /// Zero-extend low 4 u8 values to f32x4.
                #[inline(always)]
                pub fn extend_lo_f32(self) -> f32x4 {{
                    f32x4(unsafe {{
                        let i32s = _mm_cvtepu8_epi32(self.0);
                        _mm_cvtepi32_ps(i32s)
                    }})
                }}

            "#});
        }
        (ElementType::U8, SimdWidth::W256) => {
            code.push_str(&formatdoc! {r#"
                // ========== Extend/Widen Operations ==========

                /// Zero-extend low 16 u8 values to i16x16.
                ///
                /// Takes the lower 16 bytes and zero-extends each to 16 bits.
                #[inline(always)]
                pub fn extend_lo_i16(self) -> i16x16 {{
                    i16x16(unsafe {{
                        let lo128 = _mm256_castsi256_si128(self.0);
                        _mm256_cvtepu8_epi16(lo128)
                    }})
                }}

                /// Zero-extend high 16 u8 values to i16x16.
                ///
                /// Takes the upper 16 bytes and zero-extends each to 16 bits.
                #[inline(always)]
                pub fn extend_hi_i16(self) -> i16x16 {{
                    i16x16(unsafe {{
                        let hi128 = _mm256_extracti128_si256::<1>(self.0);
                        _mm256_cvtepu8_epi16(hi128)
                    }})
                }}

                /// Zero-extend all 32 u8 values to two i16x16 vectors.
                ///
                /// Returns (low 16 as i16x16, high 16 as i16x16).
                #[inline(always)]
                pub fn extend_i16(self) -> (i16x16, i16x16) {{
                    (self.extend_lo_i16(), self.extend_hi_i16())
                }}

                /// Zero-extend low 8 u8 values to i32x8.
                #[inline(always)]
                pub fn extend_lo_i32(self) -> i32x8 {{
                    i32x8(unsafe {{
                        let lo128 = _mm256_castsi256_si128(self.0);
                        _mm256_cvtepu8_epi32(lo128)
                    }})
                }}

                /// Zero-extend low 8 u8 values to f32x8.
                ///
                /// Useful for image processing: load 8 pixel values and convert to float.
                #[inline(always)]
                pub fn extend_lo_f32(self) -> f32x8 {{
                    f32x8(unsafe {{
                        let lo128 = _mm256_castsi256_si128(self.0);
                        let i32s = _mm256_cvtepu8_epi32(lo128);
                        _mm256_cvtepi32_ps(i32s)
                    }})
                }}

                /// Zero-extend all 32 u8 values to four f32x8 vectors.
                ///
                /// Returns [bytes 0-7, bytes 8-15, bytes 16-23, bytes 24-31] as f32x8.
                /// Useful for processing 32 pixels as floats.
                #[inline(always)]
                pub fn extend_f32(self) -> [f32x8; 4] {{
                    unsafe {{
                        let lo128 = _mm256_castsi256_si128(self.0);
                        let hi128 = _mm256_extracti128_si256::<1>(self.0);

                        // bytes 0-7
                        let i0 = _mm256_cvtepu8_epi32(lo128);
                        let f0 = _mm256_cvtepi32_ps(i0);

                        // bytes 8-15: shift lo128 right by 8 bytes
                        let lo_hi = _mm_srli_si128::<8>(lo128);
                        let i1 = _mm256_cvtepu8_epi32(lo_hi);
                        let f1 = _mm256_cvtepi32_ps(i1);

                        // bytes 16-23
                        let i2 = _mm256_cvtepu8_epi32(hi128);
                        let f2 = _mm256_cvtepi32_ps(i2);

                        // bytes 24-31: shift hi128 right by 8 bytes
                        let hi_hi = _mm_srli_si128::<8>(hi128);
                        let i3 = _mm256_cvtepu8_epi32(hi_hi);
                        let f3 = _mm256_cvtepi32_ps(i3);

                        [f32x8(f0), f32x8(f1), f32x8(f2), f32x8(f3)]
                    }}
                }}

            "#});
        }

        // ========== i16 → i32 ==========
        (ElementType::I16, SimdWidth::W128) => {
            code.push_str(&formatdoc! {r#"
                // ========== Extend/Widen Operations ==========

                /// Sign-extend low 4 i16 values to i32x4.
                #[inline(always)]
                pub fn extend_lo_i32(self) -> i32x4 {{
                    i32x4(unsafe {{ _mm_cvtepi16_epi32(self.0) }})
                }}

                /// Sign-extend high 4 i16 values to i32x4.
                #[inline(always)]
                pub fn extend_hi_i32(self) -> i32x4 {{
                    i32x4(unsafe {{
                        let hi = _mm_srli_si128::<8>(self.0);
                        _mm_cvtepi16_epi32(hi)
                    }})
                }}

                /// Sign-extend all 8 i16 values to two i32x4 vectors.
                #[inline(always)]
                pub fn extend_i32(self) -> (i32x4, i32x4) {{
                    (self.extend_lo_i32(), self.extend_hi_i32())
                }}

                /// Sign-extend low 4 i16 values to f32x4.
                #[inline(always)]
                pub fn extend_lo_f32(self) -> f32x4 {{
                    f32x4(unsafe {{
                        let i32s = _mm_cvtepi16_epi32(self.0);
                        _mm_cvtepi32_ps(i32s)
                    }})
                }}

            "#});
        }
        (ElementType::I16, SimdWidth::W256) => {
            code.push_str(&formatdoc! {r#"
                // ========== Extend/Widen Operations ==========

                /// Sign-extend low 8 i16 values to i32x8.
                #[inline(always)]
                pub fn extend_lo_i32(self) -> i32x8 {{
                    i32x8(unsafe {{
                        let lo128 = _mm256_castsi256_si128(self.0);
                        _mm256_cvtepi16_epi32(lo128)
                    }})
                }}

                /// Sign-extend high 8 i16 values to i32x8.
                #[inline(always)]
                pub fn extend_hi_i32(self) -> i32x8 {{
                    i32x8(unsafe {{
                        let hi128 = _mm256_extracti128_si256::<1>(self.0);
                        _mm256_cvtepi16_epi32(hi128)
                    }})
                }}

                /// Sign-extend all 16 i16 values to two i32x8 vectors.
                #[inline(always)]
                pub fn extend_i32(self) -> (i32x8, i32x8) {{
                    (self.extend_lo_i32(), self.extend_hi_i32())
                }}

                /// Sign-extend low 8 i16 values to f32x8.
                #[inline(always)]
                pub fn extend_lo_f32(self) -> f32x8 {{
                    f32x8(unsafe {{
                        let lo128 = _mm256_castsi256_si128(self.0);
                        let i32s = _mm256_cvtepi16_epi32(lo128);
                        _mm256_cvtepi32_ps(i32s)
                    }})
                }}

                /// Sign-extend all 16 i16 values to two f32x8 vectors.
                #[inline(always)]
                pub fn extend_f32(self) -> (f32x8, f32x8) {{
                    unsafe {{
                        let lo128 = _mm256_castsi256_si128(self.0);
                        let hi128 = _mm256_extracti128_si256::<1>(self.0);

                        let i32_lo = _mm256_cvtepi16_epi32(lo128);
                        let i32_hi = _mm256_cvtepi16_epi32(hi128);

                        (f32x8(_mm256_cvtepi32_ps(i32_lo)), f32x8(_mm256_cvtepi32_ps(i32_hi)))
                    }}
                }}

            "#});
        }

        // ========== u16 → i32/f32 ==========
        (ElementType::U16, SimdWidth::W128) => {
            code.push_str(&formatdoc! {r#"
                // ========== Extend/Widen Operations ==========

                /// Zero-extend low 4 u16 values to i32x4.
                #[inline(always)]
                pub fn extend_lo_i32(self) -> i32x4 {{
                    i32x4(unsafe {{ _mm_cvtepu16_epi32(self.0) }})
                }}

                /// Zero-extend high 4 u16 values to i32x4.
                #[inline(always)]
                pub fn extend_hi_i32(self) -> i32x4 {{
                    i32x4(unsafe {{
                        let hi = _mm_srli_si128::<8>(self.0);
                        _mm_cvtepu16_epi32(hi)
                    }})
                }}

                /// Zero-extend all 8 u16 values to two i32x4 vectors.
                #[inline(always)]
                pub fn extend_i32(self) -> (i32x4, i32x4) {{
                    (self.extend_lo_i32(), self.extend_hi_i32())
                }}

                /// Zero-extend low 4 u16 values to f32x4.
                #[inline(always)]
                pub fn extend_lo_f32(self) -> f32x4 {{
                    f32x4(unsafe {{
                        let i32s = _mm_cvtepu16_epi32(self.0);
                        _mm_cvtepi32_ps(i32s)
                    }})
                }}

            "#});
        }
        (ElementType::U16, SimdWidth::W256) => {
            code.push_str(&formatdoc! {r#"
                // ========== Extend/Widen Operations ==========

                /// Zero-extend low 8 u16 values to i32x8.
                #[inline(always)]
                pub fn extend_lo_i32(self) -> i32x8 {{
                    i32x8(unsafe {{
                        let lo128 = _mm256_castsi256_si128(self.0);
                        _mm256_cvtepu16_epi32(lo128)
                    }})
                }}

                /// Zero-extend high 8 u16 values to i32x8.
                #[inline(always)]
                pub fn extend_hi_i32(self) -> i32x8 {{
                    i32x8(unsafe {{
                        let hi128 = _mm256_extracti128_si256::<1>(self.0);
                        _mm256_cvtepu16_epi32(hi128)
                    }})
                }}

                /// Zero-extend all 16 u16 values to two i32x8 vectors.
                #[inline(always)]
                pub fn extend_i32(self) -> (i32x8, i32x8) {{
                    (self.extend_lo_i32(), self.extend_hi_i32())
                }}

                /// Zero-extend low 8 u16 values to f32x8.
                #[inline(always)]
                pub fn extend_lo_f32(self) -> f32x8 {{
                    f32x8(unsafe {{
                        let lo128 = _mm256_castsi256_si128(self.0);
                        let i32s = _mm256_cvtepu16_epi32(lo128);
                        _mm256_cvtepi32_ps(i32s)
                    }})
                }}

                /// Zero-extend all 16 u16 values to two f32x8 vectors.
                #[inline(always)]
                pub fn extend_f32(self) -> (f32x8, f32x8) {{
                    unsafe {{
                        let lo128 = _mm256_castsi256_si128(self.0);
                        let hi128 = _mm256_extracti128_si256::<1>(self.0);

                        let i32_lo = _mm256_cvtepu16_epi32(lo128);
                        let i32_hi = _mm256_cvtepu16_epi32(hi128);

                        (f32x8(_mm256_cvtepi32_ps(i32_lo)), f32x8(_mm256_cvtepi32_ps(i32_hi)))
                    }}
                }}

            "#});
        }

        // ========== i32 → f32 ==========
        (ElementType::I32, SimdWidth::W128) => {
            code.push_str(&formatdoc! {r#"
                // ========== Extend/Widen Operations ==========

                /// Convert to f32x4.
                #[inline(always)]
                pub fn to_f32(self) -> f32x4 {{
                    f32x4(unsafe {{ _mm_cvtepi32_ps(self.0) }})
                }}

            "#});
        }
        (ElementType::I32, SimdWidth::W256) => {
            code.push_str(&formatdoc! {r#"
                // ========== Extend/Widen Operations ==========

                /// Convert to f32x8.
                #[inline(always)]
                pub fn to_f32(self) -> f32x8 {{
                    f32x8(unsafe {{ _mm256_cvtepi32_ps(self.0) }})
                }}

            "#});
        }

        // ========== f32: from_u8 convenience ==========
        (ElementType::F32, SimdWidth::W128) => {
            code.push_str(&formatdoc! {r#"
                // ========== Load and Convert ==========

                /// Load 4 u8 values and convert to f32x4.
                ///
                /// Useful for image processing: load pixel values directly to float.
                #[inline(always)]
                pub fn from_u8(bytes: &[u8; 4]) -> Self {{
                    unsafe {{
                        // Load 4 bytes into low part of XMM register
                        let b = _mm_cvtsi32_si128(i32::from_ne_bytes(*bytes));
                        let i32s = _mm_cvtepu8_epi32(b);
                        Self(_mm_cvtepi32_ps(i32s))
                    }}
                }}

                /// Convert to 4 u8 values with saturation.
                ///
                /// Values are clamped to [0, 255] and rounded.
                #[inline(always)]
                pub fn to_u8(self) -> [u8; 4] {{
                    unsafe {{
                        // Convert to i32, pack to i16, pack to u8
                        let i32s = _mm_cvtps_epi32(self.0);
                        let i16s = _mm_packs_epi32(i32s, i32s);
                        let u8s = _mm_packus_epi16(i16s, i16s);
                        let val = _mm_cvtsi128_si32(u8s) as u32;
                        val.to_ne_bytes()
                    }}
                }}

            "#});
        }
        (ElementType::F32, SimdWidth::W256) => {
            code.push_str(&formatdoc! {r#"
                // ========== Load and Convert ==========

                /// Load 8 u8 values and convert to f32x8.
                ///
                /// Useful for image processing: load pixel values directly to float.
                #[inline(always)]
                pub fn from_u8(bytes: &[u8; 8]) -> Self {{
                    unsafe {{
                        // Load 8 bytes into low part of XMM register
                        let b = _mm_loadl_epi64(bytes.as_ptr() as *const __m128i);
                        let i32s = _mm256_cvtepu8_epi32(b);
                        Self(_mm256_cvtepi32_ps(i32s))
                    }}
                }}

                /// Convert to 8 u8 values with saturation.
                ///
                /// Values are clamped to [0, 255] and rounded.
                #[inline(always)]
                pub fn to_u8(self) -> [u8; 8] {{
                    unsafe {{
                        // Convert to i32
                        let i32s = _mm256_cvtps_epi32(self.0);
                        // Pack i32 to i16 (within lanes, then combine)
                        let lo = _mm256_castsi256_si128(i32s);
                        let hi = _mm256_extracti128_si256::<1>(i32s);
                        let i16s = _mm_packs_epi32(lo, hi);
                        // Pack i16 to u8
                        let u8s = _mm_packus_epi16(i16s, i16s);
                        let mut result = [0u8; 8];
                        _mm_storel_epi64(result.as_mut_ptr() as *mut __m128i, u8s);
                        result
                    }}
                }}

            "#});
        }

        _ => {}
    }

    code
}

/// Generate interleave/deinterleave operations.
fn generate_interleave_ops(ty: &SimdType) -> String {
    let mut code = String::new();

    // Only for f32 types for now
    if ty.elem != ElementType::F32 {
        return code;
    }

    match ty.width {
        SimdWidth::W128 => {
            code.push_str(&formatdoc! {r#"
                // ========== Interleave Operations ==========

                /// Interleave low elements: [a0,a1,a2,a3] + [b0,b1,b2,b3] → [a0,b0,a1,b1]
                #[inline(always)]
                pub fn interleave_lo(self, other: Self) -> Self {{
                    Self(unsafe {{ _mm_unpacklo_ps(self.0, other.0) }})
                }}

                /// Interleave high elements: [a0,a1,a2,a3] + [b0,b1,b2,b3] → [a2,b2,a3,b3]
                #[inline(always)]
                pub fn interleave_hi(self, other: Self) -> Self {{
                    Self(unsafe {{ _mm_unpackhi_ps(self.0, other.0) }})
                }}

                /// Interleave two vectors: returns (interleave_lo, interleave_hi)
                #[inline(always)]
                pub fn interleave(self, other: Self) -> (Self, Self) {{
                    (self.interleave_lo(other), self.interleave_hi(other))
                }}

                // ========== 4-Channel Interleave/Deinterleave ==========

                /// Deinterleave 4 RGBA pixels from AoS to SoA format.
                ///
                /// Input: 4 vectors where each contains one pixel `[R, G, B, A]`.
                /// Output: 4 vectors where each contains one channel across all pixels.
                ///
                /// ```text
                /// Input:  rgba[0] = [R0, G0, B0, A0]  (pixel 0)
                ///         rgba[1] = [R1, G1, B1, A1]  (pixel 1)
                ///         rgba[2] = [R2, G2, B2, A2]  (pixel 2)
                ///         rgba[3] = [R3, G3, B3, A3]  (pixel 3)
                ///
                /// Output: [0] = [R0, R1, R2, R3]  (red channel)
                ///         [1] = [G0, G1, G2, G3]  (green channel)
                ///         [2] = [B0, B1, B2, B3]  (blue channel)
                ///         [3] = [A0, A1, A2, A3]  (alpha channel)
                /// ```
                #[inline]
                pub fn deinterleave_4ch(rgba: [Self; 4]) -> [Self; 4] {{
                    Self::transpose_4x4_copy(rgba)
                }}

                /// Interleave 4 channels from SoA to AoS format.
                ///
                /// Input: 4 vectors where each contains one channel across pixels.
                /// Output: 4 vectors where each contains one complete RGBA pixel.
                ///
                /// This is the inverse of `deinterleave_4ch`.
                #[inline]
                pub fn interleave_4ch(channels: [Self; 4]) -> [Self; 4] {{
                    Self::transpose_4x4_copy(channels)
                }}

                /// Load 4 RGBA u8 pixels and deinterleave to 4 f32x4 channel vectors.
                ///
                /// Input: 16 bytes = 4 RGBA pixels in interleaved format.
                /// Output: (R, G, B, A) where each is f32x4 with values in [0.0, 255.0].
                #[inline]
                pub fn load_4_rgba_u8(rgba: &[u8; 16]) -> (Self, Self, Self, Self) {{
                    unsafe {{
                        let v = _mm_loadu_si128(rgba.as_ptr() as *const __m128i);

                        // Shuffle masks to gather each channel
                        // R: bytes 0, 4, 8, 12 → positions 0, 1, 2, 3
                        let r_mask = _mm_setr_epi8(0, -1, -1, -1, 4, -1, -1, -1, 8, -1, -1, -1, 12, -1, -1, -1);
                        // G: bytes 1, 5, 9, 13
                        let g_mask = _mm_setr_epi8(1, -1, -1, -1, 5, -1, -1, -1, 9, -1, -1, -1, 13, -1, -1, -1);
                        // B: bytes 2, 6, 10, 14
                        let b_mask = _mm_setr_epi8(2, -1, -1, -1, 6, -1, -1, -1, 10, -1, -1, -1, 14, -1, -1, -1);
                        // A: bytes 3, 7, 11, 15
                        let a_mask = _mm_setr_epi8(3, -1, -1, -1, 7, -1, -1, -1, 11, -1, -1, -1, 15, -1, -1, -1);

                        // Shuffle and convert to f32
                        let r_i32 = _mm_shuffle_epi8(v, r_mask);
                        let g_i32 = _mm_shuffle_epi8(v, g_mask);
                        let b_i32 = _mm_shuffle_epi8(v, b_mask);
                        let a_i32 = _mm_shuffle_epi8(v, a_mask);

                        (
                            Self(_mm_cvtepi32_ps(r_i32)),
                            Self(_mm_cvtepi32_ps(g_i32)),
                            Self(_mm_cvtepi32_ps(b_i32)),
                            Self(_mm_cvtepi32_ps(a_i32)),
                        )
                    }}
                }}

                /// Interleave 4 f32x4 channels and store as 4 RGBA u8 pixels.
                ///
                /// Input: (R, G, B, A) channel vectors with values that will be clamped to [0, 255].
                /// Output: 16 bytes = 4 RGBA pixels in interleaved format.
                #[inline]
                pub fn store_4_rgba_u8(r: Self, g: Self, b: Self, a: Self) -> [u8; 16] {{
                    unsafe {{
                        // Convert to i32 with rounding
                        let ri = _mm_cvtps_epi32(r.0);
                        let gi = _mm_cvtps_epi32(g.0);
                        let bi = _mm_cvtps_epi32(b.0);
                        let ai = _mm_cvtps_epi32(a.0);

                        // Pack i32 to i16 (saturating)
                        let rg = _mm_packs_epi32(ri, gi); // [R0,R1,R2,R3,G0,G1,G2,G3]
                        let ba = _mm_packs_epi32(bi, ai); // [B0,B1,B2,B3,A0,A1,A2,A3]

                        // Pack i16 to u8 (saturating)
                        let rgba_packed = _mm_packus_epi16(rg, ba); // [R0-3,G0-3,B0-3,A0-3]

                        // Shuffle to interleaved RGBA format
                        let shuffle = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
                        let result = _mm_shuffle_epi8(rgba_packed, shuffle);

                        let mut out = [0u8; 16];
                        _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, result);
                        out
                    }}
                }}

            "#});
        }
        SimdWidth::W256 => {
            code.push_str(&formatdoc! {r#"
                // ========== Interleave Operations ==========

                /// Interleave low elements within 128-bit lanes.
                ///
                /// [a0,a1,a2,a3,a4,a5,a6,a7] + [b0,b1,b2,b3,b4,b5,b6,b7]
                /// → [a0,b0,a1,b1,a4,b4,a5,b5]
                #[inline(always)]
                pub fn interleave_lo(self, other: Self) -> Self {{
                    Self(unsafe {{ _mm256_unpacklo_ps(self.0, other.0) }})
                }}

                /// Interleave high elements within 128-bit lanes.
                ///
                /// [a0,a1,a2,a3,a4,a5,a6,a7] + [b0,b1,b2,b3,b4,b5,b6,b7]
                /// → [a2,b2,a3,b3,a6,b6,a7,b7]
                #[inline(always)]
                pub fn interleave_hi(self, other: Self) -> Self {{
                    Self(unsafe {{ _mm256_unpackhi_ps(self.0, other.0) }})
                }}

                /// Interleave two vectors: returns (interleave_lo, interleave_hi)
                #[inline(always)]
                pub fn interleave(self, other: Self) -> (Self, Self) {{
                    (self.interleave_lo(other), self.interleave_hi(other))
                }}

                // ========== 4-Channel Interleave/Deinterleave ==========

                /// Deinterleave 8 RGBA pixels from AoS to SoA format.
                ///
                /// Input: 4 f32x8 vectors, where pairs of adjacent elements form RGBA pixels.
                /// Each input vector contains 2 complete RGBA pixels:
                /// - `rgba[0]` = [R0, G0, B0, A0, R1, G1, B1, A1]
                /// - `rgba[1]` = [R2, G2, B2, A2, R3, G3, B3, A3]
                /// - `rgba[2]` = [R4, G4, B4, A4, R5, G5, B5, A5]
                /// - `rgba[3]` = [R6, G6, B6, A6, R7, G7, B7, A7]
                ///
                /// Output: 4 f32x8 vectors, one per channel:
                /// - `[0]` = [R0, R1, R2, R3, R4, R5, R6, R7]
                /// - `[1]` = [G0, G1, G2, G3, G4, G5, G6, G7]
                /// - `[2]` = [B0, B1, B2, B3, B4, B5, B6, B7]
                /// - `[3]` = [A0, A1, A2, A3, A4, A5, A6, A7]
                #[inline]
                pub fn deinterleave_4ch(rgba: [Self; 4]) -> [Self; 4] {{
                    unsafe {{
                        // Stage 1: Unpack pairs
                        // unpacklo: [a0,b0,a1,b1, a4,b4,a5,b5]
                        // unpackhi: [a2,b2,a3,b3, a6,b6,a7,b7]
                        let rg_lo = _mm256_unpacklo_ps(rgba[0].0, rgba[1].0); // [R0,R2,G0,G2, R1,R3,G1,G3]
                        let rg_hi = _mm256_unpackhi_ps(rgba[0].0, rgba[1].0); // [B0,B2,A0,A2, B1,B3,A1,A3]
                        let rg_lo2 = _mm256_unpacklo_ps(rgba[2].0, rgba[3].0); // [R4,R6,G4,G6, R5,R7,G5,G7]
                        let rg_hi2 = _mm256_unpackhi_ps(rgba[2].0, rgba[3].0); // [B4,B6,A4,A6, B5,B7,A5,A7]

                        // Stage 2: Shuffle to separate R,G and B,A
                        let r_g_01 = _mm256_unpacklo_ps(rg_lo, rg_lo2);   // [R0,R4,R2,R6, R1,R5,R3,R7]
                        let r_g_23 = _mm256_unpackhi_ps(rg_lo, rg_lo2);   // [G0,G4,G2,G6, G1,G5,G3,G7]
                        let b_a_01 = _mm256_unpacklo_ps(rg_hi, rg_hi2);   // [B0,B4,B2,B6, B1,B5,B3,B7]
                        let b_a_23 = _mm256_unpackhi_ps(rg_hi, rg_hi2);   // [A0,A4,A2,A6, A1,A5,A3,A7]

                        // Stage 3: Final permute to get contiguous channels
                        // Need to reorder: [0,4,2,6,1,5,3,7] → [0,1,2,3,4,5,6,7]
                        let perm = _mm256_setr_epi32(0, 4, 2, 6, 1, 5, 3, 7);
                        let r = _mm256_permutevar8x32_ps(r_g_01, perm);
                        let g = _mm256_permutevar8x32_ps(r_g_23, perm);
                        let b = _mm256_permutevar8x32_ps(b_a_01, perm);
                        let a = _mm256_permutevar8x32_ps(b_a_23, perm);

                        [Self(r), Self(g), Self(b), Self(a)]
                    }}
                }}

                /// Interleave 4 channels from SoA to AoS format.
                ///
                /// Input: 4 f32x8 vectors, one per channel (R, G, B, A).
                /// Output: 4 f32x8 vectors in interleaved AoS format.
                ///
                /// This is the inverse of `deinterleave_4ch`.
                #[inline]
                pub fn interleave_4ch(channels: [Self; 4]) -> [Self; 4] {{
                    unsafe {{
                        let r = channels[0].0;
                        let g = channels[1].0;
                        let b = channels[2].0;
                        let a = channels[3].0;

                        // Interleave R with G: [R0,G0,R1,G1, R4,G4,R5,G5]
                        let rg_lo = _mm256_unpacklo_ps(r, g);
                        // [R2,G2,R3,G3, R6,G6,R7,G7]
                        let rg_hi = _mm256_unpackhi_ps(r, g);

                        // Interleave B with A
                        let ba_lo = _mm256_unpacklo_ps(b, a);
                        let ba_hi = _mm256_unpackhi_ps(b, a);

                        // Combine RG with BA: [R0,G0,B0,A0, R4,G4,B4,A4]
                        let rgba_0 = _mm256_shuffle_ps::<0x44>(rg_lo, ba_lo);
                        let rgba_1 = _mm256_shuffle_ps::<0xEE>(rg_lo, ba_lo);
                        let rgba_2 = _mm256_shuffle_ps::<0x44>(rg_hi, ba_hi);
                        let rgba_3 = _mm256_shuffle_ps::<0xEE>(rg_hi, ba_hi);

                        // Permute to get final layout
                        let out0 = _mm256_permute2f128_ps::<0x20>(rgba_0, rgba_1);
                        let out1 = _mm256_permute2f128_ps::<0x20>(rgba_2, rgba_3);
                        let out2 = _mm256_permute2f128_ps::<0x31>(rgba_0, rgba_1);
                        let out3 = _mm256_permute2f128_ps::<0x31>(rgba_2, rgba_3);

                        [Self(out0), Self(out1), Self(out2), Self(out3)]
                    }}
                }}

                /// Load 8 RGBA u8 pixels and deinterleave to 4 f32x8 channel vectors.
                ///
                /// Input: 32 bytes = 8 RGBA pixels in interleaved format.
                /// Output: (R, G, B, A) where each is f32x8 with values in [0.0, 255.0].
                #[inline]
                pub fn load_8_rgba_u8(rgba: &[u8; 32]) -> (Self, Self, Self, Self) {{
                    unsafe {{
                        // Load 32 bytes
                        let v = _mm256_loadu_si256(rgba.as_ptr() as *const __m256i);

                        // Use vpshufb to gather channels within each 128-bit lane
                        // Lane 0: pixels 0-3, Lane 1: pixels 4-7
                        let r_mask = _mm256_setr_epi8(
                            0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                            0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
                        );
                        let g_mask = _mm256_setr_epi8(
                            1, 5, 9, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                            1, 5, 9, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
                        );
                        let b_mask = _mm256_setr_epi8(
                            2, 6, 10, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                            2, 6, 10, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
                        );
                        let a_mask = _mm256_setr_epi8(
                            3, 7, 11, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                            3, 7, 11, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
                        );

                        // Gather each channel's bytes into low 4 bytes of each lane
                        let r_bytes = _mm256_shuffle_epi8(v, r_mask);
                        let g_bytes = _mm256_shuffle_epi8(v, g_mask);
                        let b_bytes = _mm256_shuffle_epi8(v, b_mask);
                        let a_bytes = _mm256_shuffle_epi8(v, a_mask);

                        // Extract low 128-bit and high 128-bit lanes, combine low 4 bytes of each
                        // to get 8 consecutive bytes, then extend to f32x8
                        let r_lo = _mm256_castsi256_si128(r_bytes);
                        let r_hi = _mm256_extracti128_si256::<1>(r_bytes);
                        let r_combined = _mm_unpacklo_epi32(r_lo, r_hi); // [R0-3, R4-7, ...]
                        let r_f32 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(r_combined));

                        let g_lo = _mm256_castsi256_si128(g_bytes);
                        let g_hi = _mm256_extracti128_si256::<1>(g_bytes);
                        let g_combined = _mm_unpacklo_epi32(g_lo, g_hi);
                        let g_f32 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(g_combined));

                        let b_lo = _mm256_castsi256_si128(b_bytes);
                        let b_hi = _mm256_extracti128_si256::<1>(b_bytes);
                        let b_combined = _mm_unpacklo_epi32(b_lo, b_hi);
                        let b_f32 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b_combined));

                        let a_lo = _mm256_castsi256_si128(a_bytes);
                        let a_hi = _mm256_extracti128_si256::<1>(a_bytes);
                        let a_combined = _mm_unpacklo_epi32(a_lo, a_hi);
                        let a_f32 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(a_combined));

                        (Self(r_f32), Self(g_f32), Self(b_f32), Self(a_f32))
                    }}
                }}

                /// Interleave 4 f32x8 channels and store as 8 RGBA u8 pixels.
                ///
                /// Input: (R, G, B, A) channel vectors with values that will be clamped to [0, 255].
                /// Output: 32 bytes = 8 RGBA pixels in interleaved format.
                #[inline]
                pub fn store_8_rgba_u8(r: Self, g: Self, b: Self, a: Self) -> [u8; 32] {{
                    unsafe {{
                        // Convert f32 to i32
                        let ri = _mm256_cvtps_epi32(r.0);
                        let gi = _mm256_cvtps_epi32(g.0);
                        let bi = _mm256_cvtps_epi32(b.0);
                        let ai = _mm256_cvtps_epi32(a.0);

                        // Pack to i16 (need to handle AVX2's lane-wise packing)
                        // _mm256_packs_epi32 packs within lanes: [lo0-3, hi0-3] + [lo4-7, hi4-7]
                        // → [lo0-3 as i16, lo4-7 as i16, hi0-3 as i16, hi4-7 as i16]

                        // Pack R,G and B,A together
                        let rg = _mm256_packs_epi32(ri, gi); // [R0-3,G0-3, R4-7,G4-7] as i16
                        let ba = _mm256_packs_epi32(bi, ai); // [B0-3,A0-3, B4-7,A4-7] as i16

                        // Pack i16 to u8
                        let rgba = _mm256_packus_epi16(rg, ba); // [R0-3,G0-3,B0-3,A0-3, R4-7,G4-7,B4-7,A4-7]

                        // Shuffle within each lane to get RGBA order
                        let shuf = _mm256_setr_epi8(
                            0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
                            0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15
                        );
                        let shuffled = _mm256_shuffle_epi8(rgba, shuf);

                        let mut out = [0u8; 32];
                        _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, shuffled);
                        out
                    }}
                }}

            "#});
        }
        _ => {}
    }

    code
}

/// Generate pack/narrow operations.
fn generate_pack_ops(ty: &SimdType) -> String {
    let mut code = String::new();

    match (ty.elem, ty.width) {
        // i16 → u8 (saturating)
        (ElementType::I16, SimdWidth::W128) => {
            code.push_str(&formatdoc! {r#"
                // ========== Pack/Narrow Operations ==========

                /// Pack two i16x8 vectors to u8x16 with unsigned saturation.
                ///
                /// Values below 0 become 0, values above 255 become 255.
                /// `self` provides low 8 bytes, `other` provides high 8 bytes.
                #[inline(always)]
                pub fn pack_u8(self, other: Self) -> u8x16 {{
                    u8x16(unsafe {{ _mm_packus_epi16(self.0, other.0) }})
                }}

                /// Pack two i16x8 vectors to i8x16 with signed saturation.
                ///
                /// Values are clamped to [-128, 127].
                #[inline(always)]
                pub fn pack_i8(self, other: Self) -> i8x16 {{
                    i8x16(unsafe {{ _mm_packs_epi16(self.0, other.0) }})
                }}

            "#});
        }
        (ElementType::I16, SimdWidth::W256) => {
            code.push_str(&formatdoc! {r#"
                // ========== Pack/Narrow Operations ==========

                /// Pack two i16x16 vectors to u8x32 with unsigned saturation.
                ///
                /// Values below 0 become 0, values above 255 become 255.
                /// Note: AVX2 pack works within 128-bit lanes, so results are:
                /// [self_lo[0-7], other_lo[0-7], self_hi[0-7], other_hi[0-7]]
                #[inline(always)]
                pub fn pack_u8(self, other: Self) -> u8x32 {{
                    u8x32(unsafe {{ _mm256_packus_epi16(self.0, other.0) }})
                }}

                /// Pack two i16x16 vectors to i8x32 with signed saturation.
                ///
                /// Values are clamped to [-128, 127].
                #[inline(always)]
                pub fn pack_i8(self, other: Self) -> i8x32 {{
                    i8x32(unsafe {{ _mm256_packs_epi16(self.0, other.0) }})
                }}

            "#});
        }

        // i32 → i16 (saturating)
        (ElementType::I32, SimdWidth::W128) => {
            code.push_str(&formatdoc! {r#"
                // ========== Pack/Narrow Operations ==========

                /// Pack two i32x4 vectors to i16x8 with signed saturation.
                ///
                /// `self` provides low 4 values, `other` provides high 4 values.
                #[inline(always)]
                pub fn pack_i16(self, other: Self) -> i16x8 {{
                    i16x8(unsafe {{ _mm_packs_epi32(self.0, other.0) }})
                }}

                /// Pack two i32x4 vectors to u16x8 with unsigned saturation.
                ///
                /// Requires SSE4.1.
                #[inline(always)]
                pub fn pack_u16(self, other: Self) -> u16x8 {{
                    u16x8(unsafe {{ _mm_packus_epi32(self.0, other.0) }})
                }}

            "#});
        }
        (ElementType::I32, SimdWidth::W256) => {
            code.push_str(&formatdoc! {r#"
                // ========== Pack/Narrow Operations ==========

                /// Pack two i32x8 vectors to i16x16 with signed saturation.
                ///
                /// Note: AVX2 pack works within 128-bit lanes.
                #[inline(always)]
                pub fn pack_i16(self, other: Self) -> i16x16 {{
                    i16x16(unsafe {{ _mm256_packs_epi32(self.0, other.0) }})
                }}

                /// Pack two i32x8 vectors to u16x16 with unsigned saturation.
                #[inline(always)]
                pub fn pack_u16(self, other: Self) -> u16x16 {{
                    u16x16(unsafe {{ _mm256_packus_epi32(self.0, other.0) }})
                }}

            "#});
        }

        _ => {}
    }

    code
}

/// Generate transpose operations.
fn generate_transpose_ops(ty: &SimdType) -> String {
    // Only for f32 types
    if ty.elem != ElementType::F32 {
        return String::new();
    }

    match ty.width {
        SimdWidth::W128 => generate_transpose_4x4(),
        SimdWidth::W256 => generate_transpose_8x8_avx(),
        SimdWidth::W512 => generate_transpose_8x8_avx512(),
    }
}

/// Generate 4x4 transpose for f32x4 (SSE).
fn generate_transpose_4x4() -> String {
    formatdoc! {r#"
        // ========== Matrix Transpose ==========

        /// Transpose a 4x4 matrix represented as 4 row vectors.
        ///
        /// After transpose, `rows[i][j]` becomes `rows[j][i]`.
        #[inline]
        pub fn transpose_4x4(rows: &mut [Self; 4]) {{
            unsafe {{
                let t0 = _mm_unpacklo_ps(rows[0].0, rows[1].0);
                let t1 = _mm_unpackhi_ps(rows[0].0, rows[1].0);
                let t2 = _mm_unpacklo_ps(rows[2].0, rows[3].0);
                let t3 = _mm_unpackhi_ps(rows[2].0, rows[3].0);

                rows[0] = Self(_mm_movelh_ps(t0, t2));
                rows[1] = Self(_mm_movehl_ps(t2, t0));
                rows[2] = Self(_mm_movelh_ps(t1, t3));
                rows[3] = Self(_mm_movehl_ps(t3, t1));
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
        // ========== Matrix Transpose ==========

        /// Transpose an 8x8 matrix represented as 8 row vectors.
        ///
        /// Uses the Highway-style 3-stage transpose:
        /// 1. `unpacklo/hi` - interleave pairs within 128-bit lanes
        /// 2. `shuffle` - reorder within lanes
        /// 3. `permute2f128` - exchange 128-bit halves
        #[inline]
        pub fn transpose_8x8(rows: &mut [Self; 8]) {{
            unsafe {{
                let t0 = _mm256_unpacklo_ps(rows[0].0, rows[1].0);
                let t1 = _mm256_unpackhi_ps(rows[0].0, rows[1].0);
                let t2 = _mm256_unpacklo_ps(rows[2].0, rows[3].0);
                let t3 = _mm256_unpackhi_ps(rows[2].0, rows[3].0);
                let t4 = _mm256_unpacklo_ps(rows[4].0, rows[5].0);
                let t5 = _mm256_unpackhi_ps(rows[4].0, rows[5].0);
                let t6 = _mm256_unpacklo_ps(rows[6].0, rows[7].0);
                let t7 = _mm256_unpackhi_ps(rows[6].0, rows[7].0);

                let s0 = _mm256_shuffle_ps::<0x44>(t0, t2);
                let s1 = _mm256_shuffle_ps::<0xEE>(t0, t2);
                let s2 = _mm256_shuffle_ps::<0x44>(t1, t3);
                let s3 = _mm256_shuffle_ps::<0xEE>(t1, t3);
                let s4 = _mm256_shuffle_ps::<0x44>(t4, t6);
                let s5 = _mm256_shuffle_ps::<0xEE>(t4, t6);
                let s6 = _mm256_shuffle_ps::<0x44>(t5, t7);
                let s7 = _mm256_shuffle_ps::<0xEE>(t5, t7);

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

        /// Load an 8x8 f32 block from a contiguous array.
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
fn generate_transpose_8x8_avx512() -> String {
    formatdoc! {r#"
        // ========== Matrix Transpose ==========

        /// Transpose an 8x8 matrix using AVX-512.
        ///
        /// Takes 8 f32x16 vectors where only the lower 8 elements are used.
        #[inline]
        pub fn transpose_8x8(rows: &mut [Self; 8]) {{
            unsafe {{
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

//! Block operations (transpose, interleave) for WASM SIMD f32x4.
//!
//! Uses i32x4_shuffle for interleave since WASM SIMD lacks dedicated zip intrinsics.

use super::types::{ElementType, SimdType, SimdWidth};
use indoc::formatdoc;

/// Generate WASM block operations for f32x4.
pub fn generate_wasm_block_ops(ty: &SimdType) -> String {
    if ty.width != SimdWidth::W128 {
        return String::new();
    }

    match ty.elem {
        ElementType::F32 => generate_f32x4_block_ops(),
        _ => String::new(),
    }
}

fn generate_f32x4_block_ops() -> String {
    let mut code = formatdoc! {r#"
        // ========== Interleave Operations ==========

        /// Interleave low elements: [a0,a1,a2,a3] + [b0,b1,b2,b3] → [a0,b0,a1,b1]
        #[inline(always)]
        pub fn interleave_lo(self, other: Self) -> Self {{
            // i32x4_shuffle picks lanes: 0-3 from self, 4-7 from other
            Self(i32x4_shuffle::<0, 4, 1, 5>(self.0, other.0))
        }}

        /// Interleave high elements: [a0,a1,a2,a3] + [b0,b1,b2,b3] → [a2,b2,a3,b3]
        #[inline(always)]
        pub fn interleave_hi(self, other: Self) -> Self {{
            Self(i32x4_shuffle::<2, 6, 3, 7>(self.0, other.0))
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

        // ========== Matrix Transpose ==========

        /// Transpose a 4x4 matrix represented as 4 row vectors.
        ///
        /// After transpose, `rows[i][j]` becomes `rows[j][i]`.
        #[inline]
        pub fn transpose_4x4(rows: &mut [Self; 4]) {{
            // Step 1: interleave pairs
            // t0 = [a0,c0,a1,c1]
            let t0 = i32x4_shuffle::<0, 4, 1, 5>(rows[0].0, rows[2].0);
            // t1 = [a2,c2,a3,c3]
            let t1 = i32x4_shuffle::<2, 6, 3, 7>(rows[0].0, rows[2].0);
            // t2 = [b0,d0,b1,d1]
            let t2 = i32x4_shuffle::<0, 4, 1, 5>(rows[1].0, rows[3].0);
            // t3 = [b2,d2,b3,d3]
            let t3 = i32x4_shuffle::<2, 6, 3, 7>(rows[1].0, rows[3].0);

            // Step 2: interleave again to get final columns
            rows[0] = Self(i32x4_shuffle::<0, 4, 1, 5>(t0, t2)); // [a0,b0,c0,d0]
            rows[1] = Self(i32x4_shuffle::<2, 6, 3, 7>(t0, t2)); // [a1,b1,c1,d1]
            rows[2] = Self(i32x4_shuffle::<0, 4, 1, 5>(t1, t3)); // [a2,b2,c2,d2]
            rows[3] = Self(i32x4_shuffle::<2, 6, 3, 7>(t1, t3)); // [a3,b3,c3,d3]
        }}

        /// Transpose a 4x4 matrix, returning the transposed rows.
        #[inline]
        pub fn transpose_4x4_copy(rows: [Self; 4]) -> [Self; 4] {{
            let mut result = rows;
            Self::transpose_4x4(&mut result);
            result
        }}

    "#};

    // RGBA pixel operations
    code.push_str(&formatdoc! {r#"
        // ========== Load and Convert ==========

        /// Load 4 u8 values and convert to f32x4.
        ///
        /// Useful for image processing: load pixel values directly to float.
        #[inline(always)]
        pub fn from_u8(bytes: &[u8; 4]) -> Self {{
            // Load 4 bytes into lane 0 as u32, then extend through u8->u16->u32->f32
            let val = u32::from_ne_bytes(*bytes);
            let v = i32x4_replace_lane::<0>(i32x4_splat(0), val as i32);
            let v16 = u16x8_extend_low_u8x16(v);
            let v32 = u32x4_extend_low_u16x8(v16);
            Self(f32x4_convert_u32x4(v32))
        }}

        /// Convert to 4 u8 values with saturation.
        ///
        /// Values are clamped to [0, 255] and rounded.
        #[inline(always)]
        pub fn to_u8(self) -> [u8; 4] {{
            // Round to nearest, truncate with saturation, narrow through i16->u8
            let i32s = i32x4_trunc_sat_f32x4(f32x4_nearest(self.0));
            let i16s = i16x8_narrow_i32x4(i32s, i32s);
            let u8s = u8x16_narrow_i16x8(i16s, i16s);
            let val = u32x4_extract_lane::<0>(u8s);
            (val as u32).to_ne_bytes()
        }}

        /// Load 4 RGBA u8 pixels and deinterleave to 4 f32x4 channel vectors.
        ///
        /// Input: 16 bytes = 4 RGBA pixels in interleaved format.
        /// Output: (R, G, B, A) where each is f32x4 with values in [0.0, 255.0].
        #[inline]
        pub fn load_4_rgba_u8(rgba: &[u8; 16]) -> (Self, Self, Self, Self) {{
            unsafe {{
                // Load 16 bytes and interpret as 4 u32 pixels
                let v = v128_load(rgba.as_ptr() as *const v128);
                let mask = u32x4_splat(0xFF);

                // Extract channels via mask and shift
                let r = v128_and(v, mask);
                let g = v128_and(u32x4_shr(v, 8), mask);
                let b = v128_and(u32x4_shr(v, 16), mask);
                let a = u32x4_shr(v, 24);

                (
                    Self(f32x4_convert_u32x4(r)),
                    Self(f32x4_convert_u32x4(g)),
                    Self(f32x4_convert_u32x4(b)),
                    Self(f32x4_convert_u32x4(a)),
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
                // Round to nearest, truncate with saturation, clamp to [0, 255]
                let zero = i32x4_splat(0);
                let max_val = i32x4_splat(255);
                let ri = i32x4_min(i32x4_max(i32x4_trunc_sat_f32x4(f32x4_nearest(r.0)), zero), max_val);
                let gi = i32x4_min(i32x4_max(i32x4_trunc_sat_f32x4(f32x4_nearest(g.0)), zero), max_val);
                let bi = i32x4_min(i32x4_max(i32x4_trunc_sat_f32x4(f32x4_nearest(b.0)), zero), max_val);
                let ai = i32x4_min(i32x4_max(i32x4_trunc_sat_f32x4(f32x4_nearest(a.0)), zero), max_val);

                // Combine channels: R | (G << 8) | (B << 16) | (A << 24)
                let pixels = v128_or(
                    v128_or(ri, i32x4_shl(gi, 8)),
                    v128_or(i32x4_shl(bi, 16), i32x4_shl(ai, 24)),
                );

                let mut out = [0u8; 16];
                v128_store(out.as_mut_ptr() as *mut v128, pixels);
                out
            }}
        }}

    "#});

    code
}

//! Block operations (transpose, interleave) for ARM NEON f32x4.
//!
//! Uses NEON zip/unzip intrinsics for interleave and vtrn for transpose.

use super::types::{ElementType, SimdType, SimdWidth};
use indoc::formatdoc;

/// Generate ARM block operations for f32x4.
pub fn generate_arm_block_ops(ty: &SimdType) -> String {
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
            unsafe {{ Self(vzip1q_f32(self.0, other.0)) }}
        }}

        /// Interleave high elements: [a0,a1,a2,a3] + [b0,b1,b2,b3] → [a2,b2,a3,b3]
        #[inline(always)]
        pub fn interleave_hi(self, other: Self) -> Self {{
            unsafe {{ Self(vzip2q_f32(self.0, other.0)) }}
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
            unsafe {{
                // Step 1: zip pairs
                // t0 = [a0,c0,a1,c1], t1 = [a2,c2,a3,c3]
                let t0 = vzip1q_f32(rows[0].0, rows[2].0);
                let t1 = vzip2q_f32(rows[0].0, rows[2].0);
                // t2 = [b0,d0,b1,d1], t3 = [b2,d2,b3,d3]
                let t2 = vzip1q_f32(rows[1].0, rows[3].0);
                let t3 = vzip2q_f32(rows[1].0, rows[3].0);

                // Step 2: zip again to get final columns
                rows[0] = Self(vzip1q_f32(t0, t2)); // [a0,b0,c0,d0]
                rows[1] = Self(vzip2q_f32(t0, t2)); // [a1,b1,c1,d1]
                rows[2] = Self(vzip1q_f32(t1, t3)); // [a2,b2,c2,d2]
                rows[3] = Self(vzip2q_f32(t1, t3)); // [a3,b3,c3,d3]
            }}
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
            unsafe {{
                // Load 4 bytes as a u32 into lane 0 of a u8x16 vector
                let val = u32::from_ne_bytes(*bytes);
                let v = vsetq_lane_u32::<0>(val, vdupq_n_u32(0));
                let v8 = vreinterpretq_u8_u32(v);
                // Extend u8 -> u16 -> u32 -> f32
                let v16 = vmovl_u8(vget_low_u8(v8));
                let v32 = vmovl_u16(vget_low_u16(v16));
                Self(vcvtq_f32_u32(v32))
            }}
        }}

        /// Convert to 4 u8 values with saturation.
        ///
        /// Values are clamped to [0, 255] and rounded.
        #[inline(always)]
        pub fn to_u8(self) -> [u8; 4] {{
            unsafe {{
                // Round to nearest i32
                let i32s = vcvtnq_s32_f32(self.0);
                // Narrow i32 -> i16 (signed saturation)
                let i16s = vqmovn_s32(i32s);
                // Narrow i16 -> u8 (unsigned saturation, clamps to [0, 255])
                let u8s = vqmovun_s16(vcombine_s16(i16s, vdup_n_s16(0)));
                let val = vget_lane_u32::<0>(vreinterpret_u32_u8(u8s));
                val.to_ne_bytes()
            }}
        }}

        /// Load 4 RGBA u8 pixels and deinterleave to 4 f32x4 channel vectors.
        ///
        /// Input: 16 bytes = 4 RGBA pixels in interleaved format.
        /// Output: (R, G, B, A) where each is f32x4 with values in [0.0, 255.0].
        #[inline]
        pub fn load_4_rgba_u8(rgba: &[u8; 16]) -> (Self, Self, Self, Self) {{
            unsafe {{
                // Load 16 bytes and reinterpret as 4 u32 pixels
                let v = vld1q_u8(rgba.as_ptr());
                let v32 = vreinterpretq_u32_u8(v);
                let mask = vdupq_n_u32(0xFF);

                // Extract channels via mask and shift
                let r_u32 = vandq_u32(v32, mask);
                let g_u32 = vandq_u32(vshrq_n_u32::<8>(v32), mask);
                let b_u32 = vandq_u32(vshrq_n_u32::<16>(v32), mask);
                let a_u32 = vshrq_n_u32::<24>(v32);

                (
                    Self(vcvtq_f32_u32(r_u32)),
                    Self(vcvtq_f32_u32(g_u32)),
                    Self(vcvtq_f32_u32(b_u32)),
                    Self(vcvtq_f32_u32(a_u32)),
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
                // Round to nearest i32 and clamp to [0, 255]
                let zero = vdupq_n_s32(0);
                let max_val = vdupq_n_s32(255);
                let ri = vminq_s32(vmaxq_s32(vcvtnq_s32_f32(r.0), zero), max_val);
                let gi = vminq_s32(vmaxq_s32(vcvtnq_s32_f32(g.0), zero), max_val);
                let bi = vminq_s32(vmaxq_s32(vcvtnq_s32_f32(b.0), zero), max_val);
                let ai = vminq_s32(vmaxq_s32(vcvtnq_s32_f32(a.0), zero), max_val);

                // Combine channels: R | (G << 8) | (B << 16) | (A << 24)
                let ri = vreinterpretq_u32_s32(ri);
                let gi = vreinterpretq_u32_s32(gi);
                let bi = vreinterpretq_u32_s32(bi);
                let ai = vreinterpretq_u32_s32(ai);

                let pixels = vorrq_u32(
                    vorrq_u32(ri, vshlq_n_u32::<8>(gi)),
                    vorrq_u32(vshlq_n_u32::<16>(bi), vshlq_n_u32::<24>(ai)),
                );

                let mut out = [0u8; 16];
                vst1q_u8(out.as_mut_ptr(), vreinterpretq_u8_u32(pixels));
                out
            }}
        }}

    "#});

    code
}

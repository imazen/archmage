//! Block operations (transpose, interleave) for ARM NEON f32x4.
//!
//! Uses NEON zip/unzip intrinsics for interleave and vtrn for transpose.

use super::types::{ElementType, SimdType, SimdWidth};
use indoc::formatdoc;

/// Generate ARM block operations for f32x4.
pub fn generate_arm_block_ops(ty: &SimdType) -> String {
    // Only for f32x4
    if ty.elem != ElementType::F32 || ty.width != SimdWidth::W128 {
        return String::new();
    }

    formatdoc! {r#"
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

    "#}
}

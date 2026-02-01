//! Extend/widen and pack/narrow operations for ARM NEON types.
//!
//! NEON has native widen/narrow intrinsics that map cleanly to these ops.

use super::types::{ElementType, SimdType, SimdWidth};
use indoc::formatdoc;

/// Generate ARM extend and pack operations.
pub fn generate_arm_extend_ops(ty: &SimdType) -> String {
    if ty.width != SimdWidth::W128 {
        return String::new();
    }

    match ty.elem {
        ElementType::U8 => generate_u8x16_extend(),
        ElementType::I16 => generate_i16x8_extend_and_pack(),
        ElementType::U16 => generate_u16x8_extend(),
        ElementType::I32 => generate_i32x4_pack(),
        _ => String::new(),
    }
}

fn generate_u8x16_extend() -> String {
    formatdoc! {r#"
        // ========== Extend/Widen Operations ==========

        /// Zero-extend low 8 u8 values to i16x8.
        ///
        /// Takes the lower 8 bytes and zero-extends each to 16 bits.
        #[inline(always)]
        pub fn extend_lo_i16(self) -> i16x8 {{
            unsafe {{ i16x8(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(self.0)))) }}
        }}

        /// Zero-extend high 8 u8 values to i16x8.
        ///
        /// Takes the upper 8 bytes and zero-extends each to 16 bits.
        #[inline(always)]
        pub fn extend_hi_i16(self) -> i16x8 {{
            unsafe {{ i16x8(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(self.0)))) }}
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
            unsafe {{
                let u16s = vmovl_u8(vget_low_u8(self.0));
                i32x4(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(u16s))))
            }}
        }}

        /// Zero-extend low 4 u8 values to f32x4.
        #[inline(always)]
        pub fn extend_lo_f32(self) -> f32x4 {{
            unsafe {{
                let u16s = vmovl_u8(vget_low_u8(self.0));
                let u32s = vmovl_u16(vget_low_u16(u16s));
                f32x4(vcvtq_f32_u32(u32s))
            }}
        }}

    "#}
}

fn generate_i16x8_extend_and_pack() -> String {
    formatdoc! {r#"
        // ========== Extend/Widen Operations ==========

        /// Sign-extend low 4 i16 values to i32x4.
        #[inline(always)]
        pub fn extend_lo_i32(self) -> i32x4 {{
            unsafe {{ i32x4(vmovl_s16(vget_low_s16(self.0))) }}
        }}

        /// Sign-extend high 4 i16 values to i32x4.
        #[inline(always)]
        pub fn extend_hi_i32(self) -> i32x4 {{
            unsafe {{ i32x4(vmovl_s16(vget_high_s16(self.0))) }}
        }}

        /// Sign-extend all 8 i16 values to two i32x4 vectors.
        ///
        /// Returns (low 4 as i32x4, high 4 as i32x4).
        #[inline(always)]
        pub fn extend_i32(self) -> (i32x4, i32x4) {{
            (self.extend_lo_i32(), self.extend_hi_i32())
        }}

        /// Sign-extend low 4 i16 values to f32x4.
        #[inline(always)]
        pub fn extend_lo_f32(self) -> f32x4 {{
            unsafe {{
                let i32s = vmovl_s16(vget_low_s16(self.0));
                f32x4(vcvtq_f32_s32(i32s))
            }}
        }}

        // ========== Pack/Narrow Operations ==========

        /// Pack two i16x8 to u8x16 with unsigned saturation.
        ///
        /// `self` provides the low 8 bytes, `other` provides the high 8 bytes.
        /// Values are clamped to [0, 255].
        #[inline(always)]
        pub fn pack_u8(self, other: Self) -> u8x16 {{
            unsafe {{
                u8x16(vcombine_u8(
                    vqmovun_s16(self.0),
                    vqmovun_s16(other.0),
                ))
            }}
        }}

        /// Pack two i16x8 to i8x16 with signed saturation.
        ///
        /// `self` provides the low 8 bytes, `other` provides the high 8 bytes.
        /// Values are clamped to [-128, 127].
        #[inline(always)]
        pub fn pack_i8(self, other: Self) -> i8x16 {{
            unsafe {{
                i8x16(vcombine_s8(
                    vqmovn_s16(self.0),
                    vqmovn_s16(other.0),
                ))
            }}
        }}

    "#}
}

fn generate_u16x8_extend() -> String {
    formatdoc! {r#"
        // ========== Extend/Widen Operations ==========

        /// Zero-extend low 4 u16 values to i32x4.
        #[inline(always)]
        pub fn extend_lo_i32(self) -> i32x4 {{
            unsafe {{ i32x4(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(self.0)))) }}
        }}

        /// Zero-extend high 4 u16 values to i32x4.
        #[inline(always)]
        pub fn extend_hi_i32(self) -> i32x4 {{
            unsafe {{ i32x4(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(self.0)))) }}
        }}

        /// Zero-extend all 8 u16 values to two i32x4 vectors.
        ///
        /// Returns (low 4 as i32x4, high 4 as i32x4).
        #[inline(always)]
        pub fn extend_i32(self) -> (i32x4, i32x4) {{
            (self.extend_lo_i32(), self.extend_hi_i32())
        }}

        /// Zero-extend low 4 u16 values to f32x4.
        #[inline(always)]
        pub fn extend_lo_f32(self) -> f32x4 {{
            unsafe {{
                let u32s = vmovl_u16(vget_low_u16(self.0));
                f32x4(vcvtq_f32_u32(u32s))
            }}
        }}

    "#}
}

fn generate_i32x4_pack() -> String {
    formatdoc! {r#"
        // ========== Pack/Narrow Operations ==========

        /// Pack two i32x4 to i16x8 with signed saturation.
        ///
        /// `self` provides the low 4 values, `other` provides the high 4 values.
        /// Values are clamped to [-32768, 32767].
        #[inline(always)]
        pub fn pack_i16(self, other: Self) -> i16x8 {{
            unsafe {{
                i16x8(vcombine_s16(
                    vqmovn_s32(self.0),
                    vqmovn_s32(other.0),
                ))
            }}
        }}

        /// Pack two i32x4 to u16x8 with unsigned saturation.
        ///
        /// `self` provides the low 4 values, `other` provides the high 4 values.
        /// Values are clamped to [0, 65535].
        #[inline(always)]
        pub fn pack_u16(self, other: Self) -> u16x8 {{
            unsafe {{
                u16x8(vcombine_u16(
                    vqmovun_s32(self.0),
                    vqmovun_s32(other.0),
                ))
            }}
        }}

    "#}
}

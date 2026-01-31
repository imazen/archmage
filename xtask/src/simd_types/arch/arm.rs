//! ARM AArch64 architecture code generation (NEON).
//!
//! NEON provides 128-bit SIMD on all AArch64 processors.
//! Unlike x86 with multiple width tiers, NEON is exclusively 128-bit.
//! For 256-bit operations, use polyfills (2×128-bit).

#![allow(dead_code)]

use super::Arch;
use crate::simd_types::types::{ElementType, SimdWidth};

/// ARM AArch64 architecture (NEON)
pub struct Arm;

impl Arch for Arm {
    fn target_arch() -> &'static str {
        "aarch64"
    }

    fn intrinsic_type(elem: ElementType, width: SimdWidth) -> &'static str {
        // NEON only supports 128-bit natively
        assert!(
            width == SimdWidth::W128,
            "NEON only supports 128-bit vectors"
        );

        match elem {
            ElementType::F32 => "float32x4_t",
            ElementType::F64 => "float64x2_t",
            ElementType::I8 => "int8x16_t",
            ElementType::U8 => "uint8x16_t",
            ElementType::I16 => "int16x8_t",
            ElementType::U16 => "uint16x8_t",
            ElementType::I32 => "int32x4_t",
            ElementType::U32 => "uint32x4_t",
            ElementType::I64 => "int64x2_t",
            ElementType::U64 => "uint64x2_t",
        }
    }

    fn prefix(_width: SimdWidth) -> &'static str {
        // NEON intrinsics start with 'v'
        "v"
    }

    fn suffix(elem: ElementType) -> &'static str {
        // NEON suffix includes 'q' for 128-bit and type indicator
        match elem {
            ElementType::F32 => "q_f32",
            ElementType::F64 => "q_f64",
            ElementType::I8 => "q_s8",
            ElementType::U8 => "q_u8",
            ElementType::I16 => "q_s16",
            ElementType::U16 => "q_u16",
            ElementType::I32 => "q_s32",
            ElementType::U32 => "q_u32",
            ElementType::I64 => "q_s64",
            ElementType::U64 => "q_u64",
        }
    }

    fn minmax_suffix(elem: ElementType) -> &'static str {
        // Same as suffix for NEON (signed/unsigned already encoded)
        Self::suffix(elem)
    }

    fn required_token(_width: SimdWidth, _needs_int_ops: bool) -> &'static str {
        // NEON is baseline on AArch64
        "NeonToken"
    }

    fn required_feature(_width: SimdWidth) -> Option<&'static str> {
        // NEON is always available on AArch64, no feature flag needed
        None
    }

    fn supports_width(width: SimdWidth) -> bool {
        // NEON only supports 128-bit natively
        width == SimdWidth::W128
    }
}

// ============================================================================
// ARM-specific intrinsic helpers
// ============================================================================

impl Arm {
    /// Get the NEON intrinsic for a basic operation
    pub fn intrinsic(op: &str, elem: ElementType) -> String {
        let suffix = Self::suffix(elem);
        format!("v{op}{suffix}")
    }

    /// Get the load intrinsic (vld1q_*)
    pub fn load_intrinsic(elem: ElementType) -> String {
        let suffix = Self::suffix(elem);
        format!("vld1{suffix}")
    }

    /// Get the store intrinsic (vst1q_*)
    pub fn store_intrinsic(elem: ElementType) -> String {
        let suffix = Self::suffix(elem);
        format!("vst1{suffix}")
    }

    /// Get the splat/duplicate intrinsic (vdupq_n_*)
    pub fn splat_intrinsic(elem: ElementType) -> String {
        let type_suffix = Self::type_suffix(elem);
        format!("vdupq_n_{type_suffix}")
    }

    /// Get just the type suffix without 'q' (for some intrinsics)
    pub fn type_suffix(elem: ElementType) -> &'static str {
        match elem {
            ElementType::F32 => "f32",
            ElementType::F64 => "f64",
            ElementType::I8 => "s8",
            ElementType::U8 => "u8",
            ElementType::I16 => "s16",
            ElementType::U16 => "u16",
            ElementType::I32 => "s32",
            ElementType::U32 => "u32",
            ElementType::I64 => "s64",
            ElementType::U64 => "u64",
        }
    }

    /// Get the arithmetic intrinsic
    /// add, sub, mul (vadd, vsub, vmul)
    pub fn arith_intrinsic(op: &str, elem: ElementType) -> String {
        let suffix = Self::suffix(elem);
        format!("v{op}{suffix}")
    }

    /// Get the FMA intrinsic
    /// Note: NEON FMA is vfmaq_f32(a, b, c) = a + b*c
    /// We want self * a + b, so: vfmaq_f32(b, self, a)
    pub fn fma_intrinsic(elem: ElementType) -> String {
        let suffix = Self::suffix(elem);
        format!("vfma{suffix}")
    }

    /// Get the comparison intrinsic (vceq, vcgt, vclt, vcge, vcle)
    pub fn cmp_intrinsic(op: &str, elem: ElementType) -> String {
        let suffix = Self::suffix(elem);
        format!("vc{op}{suffix}")
    }

    /// Get the bitwise intrinsic (vand, vorr, veor)
    /// Note: For floats, need to cast to uint first
    pub fn bitwise_intrinsic(op: &str, elem: ElementType) -> String {
        // Bitwise ops use integer types
        let int_suffix = match elem {
            ElementType::F32 => "q_u32",
            ElementType::F64 => "q_u64",
            _ => Self::suffix(elem),
        };
        format!("v{op}{int_suffix}")
    }

    /// Get the cast intrinsic for float<->int bitwise ops
    pub fn reinterpret_to_uint(elem: ElementType) -> &'static str {
        match elem {
            ElementType::F32 => "vreinterpretq_u32_f32",
            ElementType::F64 => "vreinterpretq_u64_f64",
            _ => panic!("reinterpret only for floats"),
        }
    }

    /// Get the cast intrinsic for int->float after bitwise ops
    pub fn reinterpret_from_uint(elem: ElementType) -> &'static str {
        match elem {
            ElementType::F32 => "vreinterpretq_f32_u32",
            ElementType::F64 => "vreinterpretq_f64_u64",
            _ => panic!("reinterpret only for floats"),
        }
    }

    /// Get the min/max intrinsic
    pub fn minmax_intrinsic(op: &str, elem: ElementType) -> String {
        let suffix = Self::suffix(elem);
        format!("v{op}{suffix}")
    }

    /// Get the sqrt intrinsic (only for floats)
    pub fn sqrt_intrinsic(elem: ElementType) -> String {
        assert!(elem.is_float(), "sqrt only for floats");
        let suffix = Self::suffix(elem);
        format!("vsqrt{suffix}")
    }

    /// Get the abs intrinsic
    pub fn abs_intrinsic(elem: ElementType) -> String {
        let suffix = Self::suffix(elem);
        format!("vabs{suffix}")
    }

    /// Get the negate intrinsic
    pub fn neg_intrinsic(elem: ElementType) -> String {
        let suffix = Self::suffix(elem);
        format!("vneg{suffix}")
    }

    /// Get the floor intrinsic (vrndmq for "round toward minus infinity")
    pub fn floor_intrinsic(elem: ElementType) -> String {
        assert!(elem.is_float(), "floor only for floats");
        let suffix = Self::suffix(elem);
        format!("vrndm{suffix}")
    }

    /// Get the ceil intrinsic (vrndpq for "round toward plus infinity")
    pub fn ceil_intrinsic(elem: ElementType) -> String {
        assert!(elem.is_float(), "ceil only for floats");
        let suffix = Self::suffix(elem);
        format!("vrndp{suffix}")
    }

    /// Get the round intrinsic (vrndnq for "round to nearest")
    pub fn round_intrinsic(elem: ElementType) -> String {
        assert!(elem.is_float(), "round only for floats");
        let suffix = Self::suffix(elem);
        format!("vrndn{suffix}")
    }

    /// Get the truncate intrinsic (vrndq for "round toward zero")
    pub fn trunc_intrinsic(elem: ElementType) -> String {
        assert!(elem.is_float(), "trunc only for floats");
        let suffix = Self::suffix(elem);
        format!("vrnd{suffix}")
    }

    /// Get the pairwise add intrinsic (for horizontal operations)
    pub fn padd_intrinsic(elem: ElementType) -> String {
        let suffix = Self::suffix(elem);
        format!("vpadd{suffix}")
    }

    /// Get lane extraction intrinsic
    pub fn get_lane_intrinsic(elem: ElementType) -> String {
        let type_suffix = Self::type_suffix(elem);
        format!("vgetq_lane_{type_suffix}")
    }

    /// Get the blend/select intrinsic (vbslq - bit select)
    pub fn blend_intrinsic(elem: ElementType) -> String {
        // vbslq uses the same type for all element sizes
        let suffix = Self::suffix(elem);
        format!("vbsl{suffix}")
    }

    /// Get reciprocal estimate intrinsic
    pub fn recpe_intrinsic(elem: ElementType) -> String {
        assert!(elem.is_float(), "recpe only for floats");
        let suffix = Self::suffix(elem);
        format!("vrecpe{suffix}")
    }

    /// Get reciprocal square root estimate intrinsic
    pub fn rsqrte_intrinsic(elem: ElementType) -> String {
        assert!(elem.is_float(), "rsqrte only for floats");
        let suffix = Self::suffix(elem);
        format!("vrsqrte{suffix}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intrinsic_types() {
        assert_eq!(
            Arm::intrinsic_type(ElementType::F32, SimdWidth::W128),
            "float32x4_t"
        );
        assert_eq!(
            Arm::intrinsic_type(ElementType::I32, SimdWidth::W128),
            "int32x4_t"
        );
        assert_eq!(
            Arm::intrinsic_type(ElementType::U8, SimdWidth::W128),
            "uint8x16_t"
        );
    }

    #[test]
    fn test_intrinsic_names() {
        assert_eq!(Arm::arith_intrinsic("add", ElementType::F32), "vaddq_f32");
        assert_eq!(Arm::arith_intrinsic("mul", ElementType::I32), "vmulq_s32");
        assert_eq!(Arm::load_intrinsic(ElementType::F32), "vld1q_f32");
        assert_eq!(Arm::splat_intrinsic(ElementType::F32), "vdupq_n_f32");
        assert_eq!(Arm::sqrt_intrinsic(ElementType::F32), "vsqrtq_f32");
        assert_eq!(Arm::floor_intrinsic(ElementType::F32), "vrndmq_f32");
    }

    #[test]
    fn test_supports_width() {
        assert!(Arm::supports_width(SimdWidth::W128));
        assert!(!Arm::supports_width(SimdWidth::W256));
        assert!(!Arm::supports_width(SimdWidth::W512));
    }
}

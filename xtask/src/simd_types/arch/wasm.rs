//! WebAssembly SIMD128 architecture code generation.
//!
//! WASM SIMD provides 128-bit SIMD on all supporting browsers/runtimes.
//! Uses a single `v128` type with typed operations.

use super::Arch;
use crate::simd_types::types::{ElementType, SimdWidth};

/// WebAssembly SIMD128 architecture
pub struct Wasm;

impl Arch for Wasm {
    fn target_arch() -> &'static str {
        "wasm32"
    }

    fn intrinsic_type(_elem: ElementType, width: SimdWidth) -> &'static str {
        // WASM uses a single v128 type for all element types
        assert!(
            width == SimdWidth::W128,
            "WASM SIMD only supports 128-bit vectors"
        );
        "v128"
    }

    fn prefix(_width: SimdWidth) -> &'static str {
        // WASM intrinsics don't have a consistent prefix
        ""
    }

    fn suffix(elem: ElementType) -> &'static str {
        // WASM uses type prefix like f32x4_, i32x4_
        match elem {
            ElementType::F32 => "f32x4",
            ElementType::F64 => "f64x2",
            ElementType::I8 => "i8x16",
            ElementType::U8 => "u8x16",
            ElementType::I16 => "i16x8",
            ElementType::U16 => "u16x8",
            ElementType::I32 => "i32x4",
            ElementType::U32 => "u32x4",
            ElementType::I64 => "i64x2",
            ElementType::U64 => "u64x2",
        }
    }

    fn minmax_suffix(elem: ElementType) -> &'static str {
        Self::suffix(elem)
    }

    fn required_token(_width: SimdWidth, _needs_int_ops: bool) -> &'static str {
        "Simd128Token"
    }

    fn required_feature(_width: SimdWidth) -> Option<&'static str> {
        // simd128 is enabled at compile time via target_feature
        None
    }

    fn supports_width(width: SimdWidth) -> bool {
        width == SimdWidth::W128
    }
}

// ============================================================================
// WASM-specific intrinsic helpers
// ============================================================================

impl Wasm {
    /// Get the type prefix for WASM intrinsics (e.g., "f32x4", "i32x4")
    pub fn type_prefix(elem: ElementType) -> &'static str {
        Self::suffix(elem)
    }

    /// Get the arithmetic intrinsic (f32x4_add, i32x4_sub, etc.)
    pub fn arith_intrinsic(op: &str, elem: ElementType) -> String {
        let prefix = Self::type_prefix(elem);
        format!("{prefix}_{op}")
    }

    /// Get the load intrinsic (v128_load)
    pub fn load_intrinsic() -> &'static str {
        "v128_load"
    }

    /// Get the store intrinsic (v128_store)
    pub fn store_intrinsic() -> &'static str {
        "v128_store"
    }

    /// Get the splat intrinsic (f32x4_splat, i32x4_splat, etc.)
    pub fn splat_intrinsic(elem: ElementType) -> String {
        let prefix = Self::type_prefix(elem);
        format!("{prefix}_splat")
    }

    /// Get the extract lane intrinsic
    pub fn extract_lane_intrinsic(elem: ElementType) -> String {
        let prefix = Self::type_prefix(elem);
        format!("{prefix}_extract_lane")
    }

    /// Get the replace lane intrinsic
    pub fn replace_lane_intrinsic(elem: ElementType) -> String {
        let prefix = Self::type_prefix(elem);
        format!("{prefix}_replace_lane")
    }

    /// Get the min intrinsic
    pub fn min_intrinsic(elem: ElementType) -> String {
        let prefix = Self::type_prefix(elem);
        format!("{prefix}_min")
    }

    /// Get the max intrinsic
    pub fn max_intrinsic(elem: ElementType) -> String {
        let prefix = Self::type_prefix(elem);
        format!("{prefix}_max")
    }

    /// Get the abs intrinsic (floats and signed integers)
    pub fn abs_intrinsic(elem: ElementType) -> String {
        let prefix = Self::type_prefix(elem);
        format!("{prefix}_abs")
    }

    /// Get the neg intrinsic
    pub fn neg_intrinsic(elem: ElementType) -> String {
        let prefix = Self::type_prefix(elem);
        format!("{prefix}_neg")
    }

    /// Get the sqrt intrinsic (floats only)
    pub fn sqrt_intrinsic(elem: ElementType) -> String {
        assert!(elem.is_float(), "sqrt only for floats");
        let prefix = Self::type_prefix(elem);
        format!("{prefix}_sqrt")
    }

    /// Get the floor intrinsic (floats only)
    pub fn floor_intrinsic(elem: ElementType) -> String {
        assert!(elem.is_float(), "floor only for floats");
        let prefix = Self::type_prefix(elem);
        format!("{prefix}_floor")
    }

    /// Get the ceil intrinsic (floats only)
    pub fn ceil_intrinsic(elem: ElementType) -> String {
        assert!(elem.is_float(), "ceil only for floats");
        let prefix = Self::type_prefix(elem);
        format!("{prefix}_ceil")
    }

    /// Get the trunc intrinsic (floats only)
    pub fn trunc_intrinsic(elem: ElementType) -> String {
        assert!(elem.is_float(), "trunc only for floats");
        let prefix = Self::type_prefix(elem);
        format!("{prefix}_trunc")
    }

    /// Get the nearest intrinsic (round to nearest, floats only)
    pub fn nearest_intrinsic(elem: ElementType) -> String {
        assert!(elem.is_float(), "nearest only for floats");
        let prefix = Self::type_prefix(elem);
        format!("{prefix}_nearest")
    }

    /// Get the comparison intrinsic (eq, ne, lt, gt, le, ge)
    pub fn cmp_intrinsic(op: &str, elem: ElementType) -> String {
        let prefix = Self::type_prefix(elem);
        format!("{prefix}_{op}")
    }

    /// Get bitwise and intrinsic
    pub fn and_intrinsic() -> &'static str {
        "v128_and"
    }

    /// Get bitwise or intrinsic
    pub fn or_intrinsic() -> &'static str {
        "v128_or"
    }

    /// Get bitwise xor intrinsic
    pub fn xor_intrinsic() -> &'static str {
        "v128_xor"
    }

    /// Get bitwise not intrinsic
    pub fn not_intrinsic() -> &'static str {
        "v128_not"
    }

    /// Get bitselect (blend) intrinsic
    pub fn bitselect_intrinsic() -> &'static str {
        "v128_bitselect"
    }

    /// Get the pmin intrinsic (pseudo-min, NaN propagating)
    pub fn pmin_intrinsic(elem: ElementType) -> String {
        assert!(elem.is_float(), "pmin only for floats");
        let prefix = Self::type_prefix(elem);
        format!("{prefix}_pmin")
    }

    /// Get the pmax intrinsic (pseudo-max, NaN propagating)
    pub fn pmax_intrinsic(elem: ElementType) -> String {
        assert!(elem.is_float(), "pmax only for floats");
        let prefix = Self::type_prefix(elem);
        format!("{prefix}_pmax")
    }

    /// Get relaxed FMA intrinsic (requires relaxed-simd)
    pub fn relaxed_madd_intrinsic(elem: ElementType) -> String {
        assert!(elem.is_float(), "relaxed_madd only for floats");
        let prefix = Self::type_prefix(elem);
        format!("{prefix}_relaxed_madd")
    }

    /// Get shift left intrinsic
    pub fn shl_intrinsic(elem: ElementType) -> String {
        let prefix = Self::type_prefix(elem);
        format!("{prefix}_shl")
    }

    /// Get shift right intrinsic (signed for signed types, unsigned for unsigned)
    pub fn shr_intrinsic(elem: ElementType) -> String {
        let prefix = Self::type_prefix(elem);
        format!("{prefix}_shr")
    }

    /// Get all_true intrinsic (horizontal and)
    pub fn all_true_intrinsic(elem: ElementType) -> String {
        let prefix = Self::type_prefix(elem);
        format!("{prefix}_all_true")
    }

    /// Get any_true intrinsic
    pub fn any_true_intrinsic() -> &'static str {
        "v128_any_true"
    }

    /// Get bitmask intrinsic (extract high bit of each lane)
    pub fn bitmask_intrinsic(elem: ElementType) -> String {
        let prefix = Self::type_prefix(elem);
        format!("{prefix}_bitmask")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intrinsic_type() {
        assert_eq!(
            Wasm::intrinsic_type(ElementType::F32, SimdWidth::W128),
            "v128"
        );
        assert_eq!(
            Wasm::intrinsic_type(ElementType::I32, SimdWidth::W128),
            "v128"
        );
    }

    #[test]
    fn test_intrinsic_names() {
        assert_eq!(Wasm::arith_intrinsic("add", ElementType::F32), "f32x4_add");
        assert_eq!(Wasm::arith_intrinsic("sub", ElementType::I32), "i32x4_sub");
        assert_eq!(Wasm::splat_intrinsic(ElementType::F32), "f32x4_splat");
        assert_eq!(Wasm::sqrt_intrinsic(ElementType::F32), "f32x4_sqrt");
        assert_eq!(Wasm::min_intrinsic(ElementType::I16), "i16x8_min");
    }

    #[test]
    fn test_supports_width() {
        assert!(Wasm::supports_width(SimdWidth::W128));
        assert!(!Wasm::supports_width(SimdWidth::W256));
        assert!(!Wasm::supports_width(SimdWidth::W512));
    }
}

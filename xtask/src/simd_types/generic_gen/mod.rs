//! Generator for `magetypes/src/simd/generic/generated/` — the strategy-pattern wrapper types.
//!
//! Generates all 42 files: 30 `*_impl.rs`, 8 `block_ops_*.rs`,
//! 3 `transcendentals_*.rs`, and 1 `mod.rs`.

mod block_ops;
mod conversions;
mod transcendentals;
mod type_impl;

use std::collections::BTreeMap;

use super::types::{ElementType, SimdType, SimdWidth, all_simd_types};

// ============================================================================
// Type capability queries
// ============================================================================

/// Whether the element type supports hardware multiply.
/// Float + 16-bit + 32-bit ints have mul; 8-bit and 64-bit ints do not.
pub(crate) fn has_mul(elem: ElementType) -> bool {
    !matches!(
        elem,
        ElementType::I8 | ElementType::U8 | ElementType::I64 | ElementType::U64
    )
}

/// Whether the type supports Neg (float + signed int).
pub(crate) fn has_neg(elem: ElementType) -> bool {
    elem.is_float()
        || matches!(
            elem,
            ElementType::I8 | ElementType::I16 | ElementType::I32 | ElementType::I64
        )
}

/// Whether the type supports Div (float only).
pub(crate) fn has_div(elem: ElementType) -> bool {
    elem.is_float()
}

/// Whether the type has reduce_min/reduce_max (float only).
pub(crate) fn has_reduce_min_max(elem: ElementType) -> bool {
    elem.is_float()
}

/// Whether the type has rcp_approx/recip/rsqrt_approx/rsqrt (float only).
pub(crate) fn has_approx(elem: ElementType) -> bool {
    elem.is_float()
}

/// Whether the type has shift operations (int only).
pub(crate) fn has_shifts(elem: ElementType) -> bool {
    !elem.is_float()
}

/// Whether the type has shr_arithmetic (all signed ints including i64).
pub(crate) fn has_shr_arithmetic(elem: ElementType) -> bool {
    matches!(
        elem,
        ElementType::I8 | ElementType::I16 | ElementType::I32 | ElementType::I64
    )
}

/// Whether the type has all_true/any_true/bitmask (int only).
pub(crate) fn has_boolean(elem: ElementType) -> bool {
    !elem.is_float()
}

/// Whether the type has abs (float + signed int).
pub(crate) fn has_abs(elem: ElementType) -> bool {
    elem.is_signed()
}

/// Backend trait name: e.g., "F32x4Backend".
pub(crate) fn backend_trait(ty: &SimdType) -> String {
    let name = ty.name();
    let upper = uppercase_first(&name);
    format!("{upper}Backend")
}

/// Uppercase first char: "f32x4" -> "F32x4"
pub(crate) fn uppercase_first(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
    }
}

/// The bitmask return type: u32 for W128/W256, u64 for W512.
pub(crate) fn bitmask_type(width: SimdWidth) -> &'static str {
    match width {
        SimdWidth::W512 => "u64",
        _ => "u32",
    }
}

/// Doc string for comparison methods: "signed" for signed types, "unsigned" for unsigned ints.
pub(crate) fn signedness_doc(elem: ElementType) -> &'static str {
    match elem {
        ElementType::F32 | ElementType::F64 => "",
        ElementType::I8 | ElementType::I16 | ElementType::I32 | ElementType::I64 => "",
        ElementType::U8 | ElementType::U16 | ElementType::U32 | ElementType::U64 => ", unsigned",
    }
}

/// Doc string for boolean section: "high bit" vs "sign bit".
pub(crate) fn high_bit_doc(elem: ElementType) -> &'static str {
    if elem.is_signed() || elem.is_float() {
        "sign bit"
    } else {
        "high bit"
    }
}

/// ARM repr hint for doc comments (e.g., "float32x4_t", "int32x4_t", "uint8x16_t").
pub(crate) fn arm_repr_hint(ty: &SimdType) -> &'static str {
    match (ty.elem, ty.width) {
        (ElementType::F32, SimdWidth::W128) => "float32x4_t",
        (ElementType::F64, SimdWidth::W128) => "float64x2_t",
        (ElementType::I32, SimdWidth::W128) => "int32x4_t",
        (ElementType::U32, SimdWidth::W128) => "uint32x4_t",
        (ElementType::I8, SimdWidth::W128) => "int8x16_t",
        (ElementType::U8, SimdWidth::W128) => "uint8x16_t",
        (ElementType::I16, SimdWidth::W128) => "int16x8_t",
        (ElementType::U16, SimdWidth::W128) => "uint16x8_t",
        (ElementType::I64, SimdWidth::W128) => "int64x2_t",
        (ElementType::U64, SimdWidth::W128) => "uint64x2_t",
        _ => "[scalar array]",
    }
}

/// x86 repr hint for doc comments (e.g., "__m128", "__m256i").
pub(crate) fn x86_repr_hint(ty: &SimdType) -> &'static str {
    if ty.elem.is_float() {
        ty.width.x86_float_type(ty.elem)
    } else {
        ty.width.x86_int_type()
    }
}

/// The x86 raw type name for from_mXXX constructors (e.g., "__m128", "__m128i", "__m256d").
pub(crate) fn x86_raw_type(ty: &SimdType) -> &'static str {
    x86_repr_hint(ty)
}

/// The from_mXXX constructor name (e.g., "from_m128", "from_m128i", "from_m256d").
pub(crate) fn from_raw_fn_name(ty: &SimdType) -> String {
    let raw = x86_raw_type(ty);
    format!("from_{}", raw.trim_start_matches('_'))
}

/// Extract the element type prefix from a type name (e.g., "f32x4" -> "f32", "i8x16" -> "i8").
pub(crate) fn elem_prefix(type_name: &str) -> &str {
    type_name.split('x').next().unwrap_or(type_name)
}

/// Extract the lane count string from a type name (e.g., "f32x4" -> "4", "i8x16" -> "16").
pub(crate) fn lane_count(type_name: &str) -> &str {
    type_name.split('x').nth(1).unwrap_or("0")
}

/// x86 integer type hint for a SIMD type name (e.g., "i8x16" -> "__m128i", "i16x16" -> "__m256i").
pub(crate) fn x86_int_type_for_name(type_name: &str) -> &'static str {
    let elem = elem_prefix(type_name);
    let lanes: usize = lane_count(type_name).parse().unwrap_or(0);
    let elem_bytes = match elem {
        "i8" | "u8" => 1,
        "i16" | "u16" => 2,
        "i32" | "u32" | "f32" => 4,
        "i64" | "u64" | "f64" => 8,
        _ => 4,
    };
    match lanes * elem_bytes {
        16 => "__m128i",
        32 => "__m256i",
        64 => "__m512i",
        _ => "__m128i",
    }
}

// ============================================================================
// Conversion table
// ============================================================================

/// Cross-type conversion definition.
pub(crate) struct Conversion {
    /// Source type name (e.g., "f32x4").
    pub(crate) src: &'static str,
    /// The trait bound (e.g., "F32x4Convert", "I8x16Bitcast").
    pub(crate) trait_bound: &'static str,
    /// Whether the trait is in `crate::simd::backends::`.
    pub(crate) _in_backends: bool,
    /// Code generator function.
    pub(crate) gen_fn: fn(&str, &str) -> String,
}

/// All cross-type conversions. Each entry generates a separate `impl` block.
pub(crate) fn all_conversions() -> Vec<Conversion> {
    use conversions::*;
    vec![
        // f32x4 <-> i32x4
        Conversion {
            src: "f32x4",
            trait_bound: "F32x4Convert",
            _in_backends: true,
            gen_fn: gen_f32_i32_convert_on_float,
        },
        Conversion {
            src: "i32x4",
            trait_bound: "F32x4Convert",
            _in_backends: true,
            gen_fn: gen_f32_i32_convert_on_int,
        },
        // f32x8 <-> i32x8
        Conversion {
            src: "f32x8",
            trait_bound: "F32x8Convert",
            _in_backends: true,
            gen_fn: gen_f32_i32_convert_on_float,
        },
        Conversion {
            src: "i32x8",
            trait_bound: "F32x8Convert",
            _in_backends: true,
            gen_fn: gen_f32_i32_convert_on_int,
        },
        // f32x16 <-> i32x16
        Conversion {
            src: "f32x16",
            trait_bound: "F32x16Convert",
            _in_backends: true,
            gen_fn: gen_f32_i32_convert_on_float,
        },
        Conversion {
            src: "i32x16",
            trait_bound: "F32x16Convert",
            _in_backends: true,
            gen_fn: gen_f32_i32_convert_on_int,
        },
        // i8 <-> u8 bitcasts
        Conversion {
            src: "i8x16",
            trait_bound: "I8x16Bitcast",
            _in_backends: true,
            gen_fn: |src, _| gen_signed_unsigned_bitcast(src, "u8x16", "i8_to_u8", "u8_to_i8"),
        },
        Conversion {
            src: "u8x16",
            trait_bound: "I8x16Bitcast",
            _in_backends: true,
            gen_fn: |src, _| gen_unsigned_signed_bitcast(src, "i8x16", "u8_to_i8", "bitcast_i8x16"),
        },
        Conversion {
            src: "i8x32",
            trait_bound: "I8x32Bitcast",
            _in_backends: true,
            gen_fn: |src, _| gen_signed_unsigned_bitcast(src, "u8x32", "i8_to_u8", "u8_to_i8"),
        },
        Conversion {
            src: "u8x32",
            trait_bound: "I8x32Bitcast",
            _in_backends: true,
            gen_fn: |src, _| gen_unsigned_signed_bitcast(src, "i8x32", "u8_to_i8", "bitcast_i8x32"),
        },
        // i16 <-> u16 bitcasts
        Conversion {
            src: "i16x8",
            trait_bound: "I16x8Bitcast",
            _in_backends: true,
            gen_fn: |src, _| gen_signed_unsigned_bitcast(src, "u16x8", "i16_to_u16", "u16_to_i16"),
        },
        Conversion {
            src: "u16x8",
            trait_bound: "I16x8Bitcast",
            _in_backends: true,
            gen_fn: |src, _| {
                gen_unsigned_signed_bitcast(src, "i16x8", "u16_to_i16", "bitcast_i16x8")
            },
        },
        Conversion {
            src: "i16x16",
            trait_bound: "I16x16Bitcast",
            _in_backends: true,
            gen_fn: |src, _| gen_signed_unsigned_bitcast(src, "u16x16", "i16_to_u16", "u16_to_i16"),
        },
        Conversion {
            src: "u16x16",
            trait_bound: "I16x16Bitcast",
            _in_backends: true,
            gen_fn: |src, _| {
                gen_unsigned_signed_bitcast(src, "i16x16", "u16_to_i16", "bitcast_i16x16")
            },
        },
        // u32 -> i32 bitcasts
        Conversion {
            src: "u32x4",
            trait_bound: "U32x4Bitcast",
            _in_backends: true,
            gen_fn: |src, _| gen_u32_i32_bitcast(src, "i32x4", "u32_to_i32"),
        },
        Conversion {
            src: "u32x8",
            trait_bound: "U32x8Bitcast",
            _in_backends: true,
            gen_fn: |src, _| gen_u32_i32_bitcast(src, "i32x8", "u32_to_i32"),
        },
        // u64 -> i64 bitcasts
        Conversion {
            src: "u64x2",
            trait_bound: "U64x2Bitcast",
            _in_backends: true,
            gen_fn: |src, _| gen_u64_i64_bitcast(src, "i64x2", "u64_to_i64"),
        },
        Conversion {
            src: "u64x4",
            trait_bound: "U64x4Bitcast",
            _in_backends: true,
            gen_fn: |src, _| gen_u64_i64_bitcast(src, "i64x4", "u64_to_i64"),
        },
        // i64 -> f64 bitcasts
        Conversion {
            src: "i64x2",
            trait_bound: "I64x2Bitcast",
            _in_backends: true,
            gen_fn: |src, _| gen_i64_f64_bitcast(src, "f64x2", "i64_to_f64", true),
        },
        Conversion {
            src: "i64x4",
            trait_bound: "I64x4Bitcast",
            _in_backends: true,
            gen_fn: |src, _| gen_i64_f64_bitcast(src, "f64x4", "i64_to_f64", false),
        },
    ]
}

// ============================================================================
// Public entry point
// ============================================================================

/// Generate all files for `magetypes/src/simd/generic/generated/`.
///
/// Returns a map from relative path (e.g., `"generic/generated/f32x4_impl.rs"`) to file content.
pub fn generate_generic_files() -> BTreeMap<String, String> {
    let mut files = BTreeMap::new();
    let all_types = all_simd_types();

    // Generate 30 *_impl.rs files
    for ty in &all_types {
        let name = ty.name();
        let path = format!("generic/generated/{name}_impl.rs");
        let content = type_impl::gen_type_impl(ty);
        files.insert(path, content);
    }

    // Generate 8 block_ops files
    for (type_name, elem, width) in block_ops_types() {
        let ty = SimdType::new(elem, width);
        let path = format!("generic/generated/block_ops_{type_name}.rs");
        let content = block_ops::gen_block_ops(&ty);
        files.insert(path, content);
    }

    // Generate 3 transcendentals files
    files.insert(
        "generic/generated/transcendentals_f32x4.rs".to_string(),
        transcendentals::gen_transcendentals(
            "f32x4",
            "i32x4",
            4,
            "F32x4Backend",
            "F32x4Convert",
            "I32x4Backend",
        ),
    );
    files.insert(
        "generic/generated/transcendentals_f32x8.rs".to_string(),
        transcendentals::gen_transcendentals(
            "f32x8",
            "i32x8",
            8,
            "F32x8Backend",
            "F32x8Convert",
            "I32x8Backend",
        ),
    );
    files.insert(
        "generic/generated/transcendentals_f32x16.rs".to_string(),
        transcendentals::gen_transcendentals(
            "f32x16",
            "i32x16",
            16,
            "F32x16Backend",
            "F32x16Convert",
            "I32x16Backend",
        ),
    );

    // Generate mod.rs
    files.insert(
        "generic/generated/mod.rs".to_string(),
        transcendentals::gen_mod_rs(&all_types),
    );

    files
}

/// Types that get block_ops files.
fn block_ops_types() -> Vec<(&'static str, ElementType, SimdWidth)> {
    vec![
        ("f32x4", ElementType::F32, SimdWidth::W128),
        ("f32x8", ElementType::F32, SimdWidth::W256),
        ("f64x2", ElementType::F64, SimdWidth::W128),
        ("f64x4", ElementType::F64, SimdWidth::W256),
        ("i32x4", ElementType::I32, SimdWidth::W128),
        ("i32x8", ElementType::I32, SimdWidth::W256),
        ("i8x16", ElementType::I8, SimdWidth::W128),
        ("u32x4", ElementType::U32, SimdWidth::W128),
    ]
}

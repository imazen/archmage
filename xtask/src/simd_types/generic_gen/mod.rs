//! Generator for `magetypes/src/simd/generic/generated/` — the strategy-pattern wrapper types.
//!
//! Generates 30 `*_impl.rs` files, 9 `block_ops_*.rs` files,
//! 3 `transcendentals_*.rs`, and 1 `mod.rs`.

mod block_ops;
mod conversions;
mod token_aliases;
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

/// Whether the type carries the uniform-variable-shift family.
///
/// 16/32-bit integers. The audit (`docs/CROSS-ISA-INT-PRIMITIVES.md` §2a)
/// proves the uniform form universal at every one of these widths on every
/// tier: SSE2/AVX2/AVX-512BW `PSLL/PSRL/PSRA` with the count in an XMM,
/// NEON `vshlq_*` with a splatted (clamped) count, wasm's `u32` count plus
/// a keep mask, scalar guards. 64-bit is deliberately excluded: x86 has no
/// `sra_epi64` below AVX-512, so the arithmetic flavor would need a
/// sign-decompose polyfill on v3 — that goes in only with a measured
/// benchmark attached.
pub(crate) fn has_uniform_shifts(elem: ElementType) -> bool {
    matches!(
        elem,
        ElementType::I16 | ElementType::U16 | ElementType::I32 | ElementType::U32
    )
}

/// Whether the type carries saturating add/sub.
///
/// 8- and 16-bit integers only. Saturating add/sub has no 32/64-bit form on
/// x86 (`_mm*_adds_epi32` does not exist at any tier) or on wasm128, so
/// exposing it above 16 bits would be a portability lie.
/// See `docs/CROSS-ISA-INT-PRIMITIVES.md` §1.
pub(crate) fn has_saturating_arith(elem: ElementType) -> bool {
    matches!(
        elem,
        ElementType::I8 | ElementType::U8 | ElementType::I16 | ElementType::U16
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
    ty.trait_name()
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

// ============================================================================
// Conversion table
// ============================================================================

/// Cross-type conversion definition.
pub(crate) struct Conversion {
    /// Source type name (e.g., "f32x4").
    pub(crate) src: &'static str,
    /// The trait bound (e.g., "F32x4Convert", "I8x16Bitcast").
    pub(crate) trait_bound: &'static str,
    /// Code generator function.
    pub(crate) gen_fn: fn(&str, &str) -> String,
}

const fn conv(
    src: &'static str,
    trait_bound: &'static str,
    gen_fn: fn(&str, &str) -> String,
) -> Conversion {
    Conversion {
        src,
        trait_bound,
        gen_fn,
    }
}

/// All cross-type conversions. Each entry generates a separate `impl` block.
pub(crate) fn all_conversions() -> Vec<Conversion> {
    use conversions::*;
    vec![
        // f32 <-> i32
        conv("f32x4", "F32x4Convert", gen_f32_i32_convert_on_float),
        conv("i32x4", "F32x4Convert", gen_f32_i32_convert_on_int),
        conv("f32x8", "F32x8Convert", gen_f32_i32_convert_on_float),
        conv("i32x8", "F32x8Convert", gen_f32_i32_convert_on_int),
        conv("f32x16", "F32x16Convert", gen_f32_i32_convert_on_float),
        conv("i32x16", "F32x16Convert", gen_f32_i32_convert_on_int),
        // i8 <-> u8 bitcasts
        conv("i8x16", "I8x16Bitcast", |s, t| {
            gen_bitcast(s, t, "u8x16", "i8_to_u8", None, false)
        }),
        conv("u8x16", "I8x16Bitcast", |s, t| {
            gen_bitcast(s, t, "i8x16", "u8_to_i8", None, false)
        }),
        conv("i8x32", "I8x32Bitcast", |s, t| {
            gen_bitcast(s, t, "u8x32", "i8_to_u8", None, false)
        }),
        conv("u8x32", "I8x32Bitcast", |s, t| {
            gen_bitcast(s, t, "i8x32", "u8_to_i8", None, false)
        }),
        // i16 <-> u16 bitcasts
        conv("i16x8", "I16x8Bitcast", |s, t| {
            gen_bitcast(s, t, "u16x8", "i16_to_u16", None, false)
        }),
        conv("u16x8", "I16x8Bitcast", |s, t| {
            gen_bitcast(s, t, "i16x8", "u16_to_i16", None, false)
        }),
        conv("i16x16", "I16x16Bitcast", |s, t| {
            gen_bitcast(s, t, "u16x16", "i16_to_u16", None, false)
        }),
        conv("u16x16", "I16x16Bitcast", |s, t| {
            gen_bitcast(s, t, "i16x16", "u16_to_i16", None, false)
        }),
        // W512 i16/u16 bitcasts use the shared checked storage primitives.
        conv(
            "i16x32",
            "crate::simd::backends::I16x32Backend + crate::simd::backends::U16x32Backend",
            gen_w512_i16_bitcast,
        ),
        conv(
            "u16x32",
            "crate::simd::backends::I16x32Backend + crate::simd::backends::U16x32Backend",
            gen_w512_i16_bitcast,
        ),
        // u32 -> i32 bitcasts
        conv("u32x4", "U32x4Bitcast", |s, t| {
            gen_bitcast(s, t, "i32x4", "u32_to_i32", Some("bitcast_to_i32"), true)
        }),
        conv("u32x8", "U32x8Bitcast", |s, t| {
            gen_bitcast(s, t, "i32x8", "u32_to_i32", Some("bitcast_to_i32"), true)
        }),
        // u64 -> i64 bitcasts
        conv("u64x2", "U64x2Bitcast", |s, t| {
            gen_bitcast(s, t, "i64x2", "u64_to_i64", None, false)
        }),
        conv("u64x4", "U64x4Bitcast", |s, t| {
            gen_bitcast(s, t, "i64x4", "u64_to_i64", None, false)
        }),
        // i64 -> f64 bitcasts
        conv("i64x2", "I64x2Bitcast", |s, t| {
            gen_bitcast(s, t, "f64x2", "i64_to_f64", Some("bitcast_to_f64"), false)
        }),
        conv("i64x4", "I64x4Bitcast", |s, t| {
            gen_bitcast(s, t, "f64x4", "i64_to_f64", Some("bitcast_to_f64"), false)
        }),
    ]
}

// ============================================================================
// Public entry point
// ============================================================================

/// Generate all files for `magetypes/src/simd/generic/generated/`.
///
/// Returns a map from relative path (e.g., `"generic/generated/f32x4_impl.rs"`) to file content.
pub fn generate_generic_files(registry: &crate::registry::Registry) -> BTreeMap<String, String> {
    let mut files = BTreeMap::new();
    let all_types = all_simd_types();

    // Generate 30 *_impl.rs files
    for ty in &all_types {
        let name = ty.name();
        let path = format!("generic/generated/{name}_impl.rs");
        let content = type_impl::gen_type_impl(ty) + &type_impl::gen_raw_interop(ty, registry);
        files.insert(path, content);
    }

    // Generate block_ops files
    for (type_name, elem, width) in block_ops_types() {
        let ty = SimdType::new(elem, width);
        let path = format!("generic/generated/block_ops_{type_name}.rs");
        let content = block_ops::gen_block_ops(&ty);
        files.insert(path, content);
    }

    // Generate 3 transcendentals files (f32x4, f32x8, f32x16)
    for lanes in [4, 8, 16] {
        let f = format!("f32x{lanes}");
        let i = format!("i32x{lanes}");
        let fb = format!("F32x{lanes}Backend");
        let fc = format!("F32x{lanes}Convert");
        let ib = format!("I32x{lanes}Backend");
        files.insert(
            format!("generic/generated/transcendentals_{f}.rs"),
            transcendentals::gen_transcendentals(&f, &i, &fb, &fc, &ib),
        );
    }

    // Generate mod.rs
    files.insert(
        "generic/generated/mod.rs".to_string(),
        transcendentals::gen_mod_rs(&all_types),
    );

    for (path, source) in &mut files {
        if !path.ends_with("/mod.rs") {
            source.push_str(&token_aliases::generate(source));
        }
    }
    // These modules contain handwritten implementations. Only their deprecated forwarders
    // are generated, using the same signature-driven pass as the vector files.
    files.insert(
        "generic/generated/scalar_token_aliases.rs".into(),
        token_aliases::generate(include_str!("../../../../magetypes/src/simd/scalar.rs")),
    );
    files.insert(
        "generic/generated/cross_width_token_aliases.rs".into(),
        token_aliases::generate(include_str!(
            "../../../../magetypes/src/simd/generic/cross_width.rs"
        )),
    );

    files
}

/// Types that get block_ops files.
fn block_ops_types() -> Vec<(&'static str, ElementType, SimdWidth)> {
    vec![
        ("f32x4", ElementType::F32, SimdWidth::W128),
        ("f32x8", ElementType::F32, SimdWidth::W256),
        ("f32x16", ElementType::F32, SimdWidth::W512),
        ("f64x2", ElementType::F64, SimdWidth::W128),
        ("f64x4", ElementType::F64, SimdWidth::W256),
        ("i32x4", ElementType::I32, SimdWidth::W128),
        ("i32x8", ElementType::I32, SimdWidth::W256),
        ("i8x16", ElementType::I8, SimdWidth::W128),
        ("u32x4", ElementType::U32, SimdWidth::W128),
    ]
}

/// Documentation derives from the complete generated source and handwritten originals.
pub fn constructor_inventory(files: &BTreeMap<String, String>) -> BTreeMap<String, String> {
    token_aliases::inventory(files.values().cloned().chain([
        include_str!("../../../../magetypes/src/simd/scalar.rs").to_owned(),
        include_str!("../../../../magetypes/src/simd/generic/cross_width.rs").to_owned(),
    ]))
}

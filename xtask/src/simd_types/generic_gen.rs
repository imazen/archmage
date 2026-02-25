//! Generator for `magetypes/src/simd/generic/` — the strategy-pattern wrapper types.
//!
//! Generates all 41 files: 30 `*_impl.rs`, 8 `block_ops_*.rs`,
//! 2 `transcendentals_*.rs`, and 1 `mod.rs`.

use std::collections::BTreeMap;

use indoc::formatdoc;

use super::types::{ElementType, SimdType, SimdWidth, all_simd_types};

// ============================================================================
// Type capability queries
// ============================================================================

/// Whether the element type supports hardware multiply.
/// Float + 16-bit + 32-bit ints have mul; 8-bit and 64-bit ints do not.
fn has_mul(elem: ElementType) -> bool {
    !matches!(
        elem,
        ElementType::I8 | ElementType::U8 | ElementType::I64 | ElementType::U64
    )
}

/// Whether the type supports Neg (float + signed int).
fn has_neg(elem: ElementType) -> bool {
    elem.is_float()
        || matches!(
            elem,
            ElementType::I8 | ElementType::I16 | ElementType::I32 | ElementType::I64
        )
}

/// Whether the type supports Div (float only).
fn has_div(elem: ElementType) -> bool {
    elem.is_float()
}

/// Whether the type has reduce_min/reduce_max (float only).
fn has_reduce_min_max(elem: ElementType) -> bool {
    elem.is_float()
}

/// Whether the type has rcp_approx/recip/rsqrt_approx/rsqrt (float only).
fn has_approx(elem: ElementType) -> bool {
    elem.is_float()
}

/// Whether the type has shift operations (int only).
fn has_shifts(elem: ElementType) -> bool {
    !elem.is_float()
}

/// Whether the type has shr_arithmetic (all signed ints including i64).
fn has_shr_arithmetic(elem: ElementType) -> bool {
    matches!(
        elem,
        ElementType::I8 | ElementType::I16 | ElementType::I32 | ElementType::I64
    )
}

/// Whether the type has all_true/any_true/bitmask (int only).
fn has_boolean(elem: ElementType) -> bool {
    !elem.is_float()
}

/// Whether the type has abs (float + signed int).
fn has_abs(elem: ElementType) -> bool {
    elem.is_signed()
}

/// Backend trait name: e.g., "F32x4Backend".
fn backend_trait(ty: &SimdType) -> String {
    let name = ty.name();
    let upper = uppercase_first(&name);
    format!("{upper}Backend")
}

/// Uppercase first char: "f32x4" -> "F32x4"
fn uppercase_first(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
    }
}

/// The bitmask return type: u32 for W128/W256, u64 for W512.
fn bitmask_type(width: SimdWidth) -> &'static str {
    match width {
        SimdWidth::W512 => "u64",
        _ => "u32",
    }
}

/// Doc string for comparison methods: "signed" for signed types, "unsigned" for unsigned ints.
fn signedness_doc(elem: ElementType) -> &'static str {
    match elem {
        ElementType::F32 | ElementType::F64 => "",
        ElementType::I8 | ElementType::I16 | ElementType::I32 | ElementType::I64 => "",
        ElementType::U8 | ElementType::U16 | ElementType::U32 | ElementType::U64 => ", unsigned",
    }
}

/// Doc string for boolean section: "high bit" vs "sign bit".
fn high_bit_doc(elem: ElementType) -> &'static str {
    if elem.is_signed() || elem.is_float() {
        "sign bit"
    } else {
        "high bit"
    }
}

/// ARM repr hint for doc comments (e.g., "float32x4_t", "int32x4_t", "uint8x16_t").
fn arm_repr_hint(ty: &SimdType) -> &'static str {
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
fn x86_repr_hint(ty: &SimdType) -> &'static str {
    if ty.elem.is_float() {
        ty.width.x86_float_type(ty.elem)
    } else {
        ty.width.x86_int_type()
    }
}

/// The x86 raw type name for from_mXXX constructors (e.g., "__m128", "__m128i", "__m256d").
fn x86_raw_type(ty: &SimdType) -> &'static str {
    x86_repr_hint(ty)
}

/// The from_mXXX constructor name (e.g., "from_m128", "from_m128i", "from_m256d").
fn from_raw_fn_name(ty: &SimdType) -> String {
    let raw = x86_raw_type(ty);
    format!("from_{}", raw.trim_start_matches('_'))
}

/// Extract the element type prefix from a type name (e.g., "f32x4" -> "f32", "i8x16" -> "i8").
fn elem_prefix(type_name: &str) -> &str {
    type_name.split('x').next().unwrap_or(type_name)
}

/// Extract the lane count string from a type name (e.g., "f32x4" -> "4", "i8x16" -> "16").
fn lane_count(type_name: &str) -> &str {
    type_name.split('x').nth(1).unwrap_or("0")
}

/// x86 integer type hint for a SIMD type name (e.g., "i8x16" -> "__m128i", "i16x16" -> "__m256i").
fn x86_int_type_for_name(type_name: &str) -> &'static str {
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
struct Conversion {
    /// Source type name (e.g., "f32x4").
    src: &'static str,
    /// The trait bound (e.g., "F32x4Convert", "I8x16Bitcast").
    trait_bound: &'static str,
    /// Whether the trait is in `crate::simd::backends::`.
    _in_backends: bool,
    /// Code generator function.
    gen_fn: fn(&str, &str) -> String,
}

/// All cross-type conversions. Each entry generates a separate `impl` block.
fn all_conversions() -> Vec<Conversion> {
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

/// Generate all files for `magetypes/src/simd/generic/`.
///
/// Returns a map from relative path (e.g., `"generic/f32x4_impl.rs"`) to file content.
pub fn generate_generic_files() -> BTreeMap<String, String> {
    let mut files = BTreeMap::new();
    let all_types = all_simd_types();

    // Generate 30 *_impl.rs files
    for ty in &all_types {
        let name = ty.name();
        let path = format!("generic/{name}_impl.rs");
        let content = gen_type_impl(ty);
        files.insert(path, content);
    }

    // Generate 8 block_ops files
    for (type_name, elem, width) in block_ops_types() {
        let ty = SimdType::new(elem, width);
        let path = format!("generic/block_ops_{type_name}.rs");
        let content = gen_block_ops(&ty);
        files.insert(path, content);
    }

    // Generate 2 transcendentals files
    files.insert(
        "generic/transcendentals_f32x4.rs".to_string(),
        gen_transcendentals(
            "f32x4",
            "i32x4",
            4,
            "F32x4Backend",
            "F32x4Convert",
            "I32x4Backend",
        ),
    );
    files.insert(
        "generic/transcendentals_f32x8.rs".to_string(),
        gen_transcendentals(
            "f32x8",
            "i32x8",
            8,
            "F32x8Backend",
            "F32x8Convert",
            "I32x8Backend",
        ),
    );

    // Generate mod.rs
    files.insert("generic/mod.rs".to_string(), gen_mod_rs(&all_types));

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

// ============================================================================
// *_impl.rs generator
// ============================================================================

/// Generate a complete `*_impl.rs` file for one SIMD type.
fn gen_type_impl(ty: &SimdType) -> String {
    let mut code = String::new();
    code.push_str(&gen_header(ty));
    code.push_str(&gen_struct(ty));
    code.push_str(&gen_methods(ty));
    code.push_str(&gen_operators(ty));
    code.push_str(&gen_assign_operators(ty));
    code.push_str(&gen_scalar_broadcast(ty));
    code.push_str(&gen_index(ty));
    code.push_str(&gen_from_array(ty));
    code.push_str(&gen_debug(ty));
    code.push_str(&gen_cross_type(ty));
    code.push_str(&gen_platform(ty));
    code.push_str(&gen_popcnt(ty));
    code
}

// ============================================================================
// Header (module doc + imports)
// ============================================================================

/// Generate the module-level doc comment, clippy allow, and imports.
fn gen_header(ty: &SimdType) -> String {
    let name = ty.name();
    let elem = ty.elem.name();
    let lanes = ty.lanes();
    let backend = backend_trait(ty);

    // Token examples in module doc depend on width
    let token_examples = match ty.width {
        SimdWidth::W512 => format!("`X64V4Token`, `ScalarToken`"),
        _ => format!("`X64V3Token`, `NeonToken`, `ScalarToken`"),
    };

    // Example block for float types with >= 8 lanes
    let example_block = if ty.elem.is_float() && lanes >= 8 {
        let backend_module = match ty.width {
            SimdWidth::W512 => "x64v4",
            _ => "x64v3",
        };
        formatdoc! {"
            //!
            //! # Example
            //!
            //! ```ignore
            //! use magetypes::simd::backends::{{{backend}, {backend_module}}};
            //! use magetypes::simd::generic::{name};
            //!
            //! fn sum<T: {backend}>(token: T, data: &[{elem}]) -> {elem} {{
            //!     let mut acc = {name}::<T>::zero(token);
            //!     for chunk in data.chunks_exact({lanes}) {{
            //!         acc = acc + {name}::<T>::load(token, chunk.try_into().unwrap());
            //!     }}
            //!     acc.reduce_add()
            //! }}
            //! ```
        "}
    } else {
        String::new()
    };

    // Build ops import list
    let mut ops = vec![
        "Add",
        "AddAssign",
        "BitAnd",
        "BitAndAssign",
        "BitOr",
        "BitOrAssign",
        "BitXor",
        "BitXorAssign",
    ];
    if has_div(ty.elem) {
        ops.push("Div");
        ops.push("DivAssign");
    }
    ops.push("Index");
    ops.push("IndexMut");
    if has_mul(ty.elem) {
        ops.push("Mul");
        ops.push("MulAssign");
    }
    if has_neg(ty.elem) {
        ops.push("Neg");
    }
    ops.push("Sub");
    ops.push("SubAssign");

    // Sort for consistent output matching the handwritten files
    ops.sort();

    // Format ops import - need to match the specific line wrapping pattern
    // The handwritten files wrap at ~95 chars with specific line breaks
    let ops_str = format_ops_import(&ops);

    formatdoc! {"
        //! Generic `{name}<T>` — {lanes}-lane {elem} SIMD vector parameterized by backend.
        //!
        //! `T` is a token type (e.g., {token_examples})
        //! that determines the platform-native representation and intrinsics used.
        //! The struct delegates all operations to the [`{backend}`] trait.
        {example_block}
        #![allow(clippy::should_implement_trait)]

        use core::marker::PhantomData;
        {ops_str}
        use crate::simd::backends::{backend};

    "}
}

/// Format the ops import to match the handwritten pattern.
/// Groups ops on lines that fit within ~95 chars.
fn format_ops_import(ops: &[&str]) -> String {
    // Build comma-separated list
    let items: Vec<String> = ops.iter().map(|s| s.to_string()).collect();

    // Try single-line first
    let single_line = format!("use core::ops::{{{}}};", items.join(", "));
    if single_line.len() <= 100 {
        return single_line + "\n";
    }

    // Multi-line: wrap to fit within ~100 chars per line (matching rustfmt defaults)
    let mut lines: Vec<String> = Vec::new();
    let mut current_line = String::new();

    for (i, item) in items.iter().enumerate() {
        let separator = if i < items.len() - 1 { ", " } else { "," };
        let addition = format!("{item}{separator}");

        if current_line.is_empty() {
            current_line = addition;
        } else if format!("    {current_line}{addition}").len() <= 100 {
            current_line.push_str(&addition);
        } else {
            lines.push(current_line);
            current_line = addition;
        }
    }
    if !current_line.is_empty() {
        lines.push(current_line);
    }

    let mut result = "use core::ops::{\n".to_string();
    for line in &lines {
        result.push_str(&format!("    {line}\n"));
    }
    result.push_str("};\n");
    result
}

// ============================================================================
// Struct definition
// ============================================================================

/// Generate the struct definition with doc comment.
fn gen_struct(ty: &SimdType) -> String {
    let name = ty.name();
    let elem = ty.elem.name();
    let lanes = ty.lanes();
    let backend = backend_trait(ty);

    // Repr doc depends on width and element type
    let repr_doc = match ty.width {
        SimdWidth::W128 => {
            let x86_repr = x86_repr_hint(ty);
            let arm_repr = arm_repr_hint(ty);
            format!("`{x86_repr}` on x86, `{arm_repr}` on ARM")
        }
        SimdWidth::W256 => {
            let x86_repr = x86_repr_hint(ty);
            format!("`{x86_repr}` on AVX2, `[{elem}; {lanes}]` on scalar")
        }
        SimdWidth::W512 => {
            let x86_repr = x86_repr_hint(ty);
            format!("`{x86_repr}` on AVX-512, `[{elem}; {lanes}]` on scalar")
        }
    };

    // 64-bit integer note
    let note_section = if matches!(ty.elem, ElementType::I64) {
        formatdoc! {"
            ///
            /// # Note
            ///
            /// 64-bit integer SIMD has limited native support: no hardware multiply on
            /// AVX2/NEON/WASM, and arithmetic right shift requires AVX-512 on x86.
            /// Operations like `min`, `max`, and `abs` are polyfilled where needed.
        "}
    } else if matches!(ty.elem, ElementType::U64) {
        formatdoc! {"
            ///
            /// # Note
            ///
            /// 64-bit integer SIMD has limited native support: no hardware multiply on
            /// AVX2/NEON/WASM.
        "}
    } else {
        String::new()
    };

    // PhantomData ZST comment for float types with >= 8 lanes
    let phantom_comment = if ty.elem.is_float() && lanes >= 8 {
        format!("\n// PhantomData is ZST, so {name}<T> has the same size as T::Repr.\n")
    } else {
        String::new()
    };

    formatdoc! {"
        /// {lanes}-lane {elem} SIMD vector, generic over backend `T`.
        ///
        /// `T` is a token type that proves CPU support for the required SIMD features.
        /// The inner representation is `T::Repr` (e.g., {repr_doc}).
        ///
        /// Construction requires a token value to prove CPU support at runtime.
        /// After construction, operations don't need the token — it's baked into the type.
        {note_section}#[derive(Clone, Copy)]
        #[repr(transparent)]
        pub struct {name}<T: {backend}>(T::Repr, PhantomData<T>);
        {phantom_comment}
    "}
}

// ============================================================================
// Methods (main impl block)
// ============================================================================

/// Generate the main `impl` block with all methods.
fn gen_methods(ty: &SimdType) -> String {
    let name = ty.name();
    let elem = ty.elem.name();
    let lanes = ty.lanes();
    let backend = backend_trait(ty);
    let bitmask_ty = bitmask_type(ty.width);
    let elem_bits = ty.elem.size_bytes() * 8;
    let high_bit = high_bit_doc(ty.elem);

    let mut code = format!("impl<T: {backend}> {name}<T> {{\n");

    // LANES constant
    code.push_str(&formatdoc! {"
        \x20   /// Number of {elem} lanes.
            pub const LANES: usize = {lanes};

    "});

    // ====== Construction (token-gated) ======
    code.push_str("    // ====== Construction (token-gated) ======\n\n");

    code.push_str(&formatdoc! {"
        \x20   /// Broadcast scalar to all {lanes} lanes.
            #[inline(always)]
            pub fn splat(_: T, v: {elem}) -> Self {{
                Self(T::splat(v), PhantomData)
            }}

            /// All lanes zero.
            #[inline(always)]
            pub fn zero(_: T) -> Self {{
                Self(T::zero(), PhantomData)
            }}

            /// Load from a `[{elem}; {lanes}]` array.
            #[inline(always)]
            pub fn load(_: T, data: &[{elem}; {lanes}]) -> Self {{
                Self(T::load(data), PhantomData)
            }}

            /// Create from array (zero-cost where possible).
            #[inline(always)]
            pub fn from_array(_: T, arr: [{elem}; {lanes}]) -> Self {{
                Self(T::from_array(arr), PhantomData)
            }}

            /// Create from slice. Panics if `slice.len() < {lanes}`.
            #[inline(always)]
            pub fn from_slice(_: T, slice: &[{elem}]) -> Self {{
                let arr: [{elem}; {lanes}] = slice[..{lanes}].try_into().unwrap();
                Self(T::from_array(arr), PhantomData)
            }}

    "});

    // partition_slice
    code.push_str(&gen_partition_slice(ty));

    // partition_slice_mut
    code.push_str(&gen_partition_slice_mut(ty));

    // ====== Accessors ======
    code.push_str("    // ====== Accessors ======\n\n");

    code.push_str(&formatdoc! {"
        \x20   /// Store to array.
            #[inline(always)]
            pub fn store(self, out: &mut [{elem}; {lanes}]) {{
                T::store(self.0, out);
            }}

            /// Convert to array.
            #[inline(always)]
            pub fn to_array(self) -> [{elem}; {lanes}] {{
                T::to_array(self.0)
            }}

            /// Get the underlying platform representation.
            #[inline(always)]
            pub fn into_repr(self) -> T::Repr {{
                self.0
            }}

            /// Wrap a platform representation (token-gated).
            #[inline(always)]
            pub fn from_repr(_: T, repr: T::Repr) -> Self {{
                Self(repr, PhantomData)
            }}

            /// Wrap a repr without requiring a token value.
            /// Only usable within the `generic` module (for cross-type conversions).
            #[inline(always)]
            #[allow(dead_code)]
            pub(super) fn from_repr_unchecked(repr: T::Repr) -> Self {{
                Self(repr, PhantomData)
            }}

    "});

    // ====== Math ======
    code.push_str("    // ====== Math ======\n\n");

    if ty.elem.is_float() {
        // Float: min, max, clamp, sqrt, abs, floor, ceil, round, mul_add, mul_sub
        code.push_str(&formatdoc! {"
            \x20   /// Lane-wise minimum.
                #[inline(always)]
                pub fn min(self, other: Self) -> Self {{
                    Self(T::min(self.0, other.0), PhantomData)
                }}

                /// Lane-wise maximum.
                #[inline(always)]
                pub fn max(self, other: Self) -> Self {{
                    Self(T::max(self.0, other.0), PhantomData)
                }}

                /// Clamp between lo and hi.
                #[inline(always)]
                pub fn clamp(self, lo: Self, hi: Self) -> Self {{
                    Self(T::clamp(self.0, lo.0, hi.0), PhantomData)
                }}

                /// Square root.
                #[inline(always)]
                pub fn sqrt(self) -> Self {{
                    Self(T::sqrt(self.0), PhantomData)
                }}

                /// Absolute value.
                #[inline(always)]
                pub fn abs(self) -> Self {{
                    Self(T::abs(self.0), PhantomData)
                }}

                /// Round toward negative infinity.
                #[inline(always)]
                pub fn floor(self) -> Self {{
                    Self(T::floor(self.0), PhantomData)
                }}

                /// Round toward positive infinity.
                #[inline(always)]
                pub fn ceil(self) -> Self {{
                    Self(T::ceil(self.0), PhantomData)
                }}

                /// Round to nearest integer.
                #[inline(always)]
                pub fn round(self) -> Self {{
                    Self(T::round(self.0), PhantomData)
                }}

                /// Fused multiply-add: `self * a + b`.
                #[inline(always)]
                pub fn mul_add(self, a: Self, b: Self) -> Self {{
                    Self(T::mul_add(self.0, a.0, b.0), PhantomData)
                }}

                /// Fused multiply-sub: `self * a - b`.
                #[inline(always)]
                pub fn mul_sub(self, a: Self, b: Self) -> Self {{
                    Self(T::mul_sub(self.0, a.0, b.0), PhantomData)
                }}

        "});
    } else {
        // Integer types
        let unsigned_suffix = if !ty.elem.is_signed() {
            " (unsigned)"
        } else {
            ""
        };

        code.push_str(&formatdoc! {"
            \x20   /// Lane-wise minimum{unsigned_suffix}.
                #[inline(always)]
                pub fn min(self, other: Self) -> Self {{
                    Self(T::min(self.0, other.0), PhantomData)
                }}

                /// Lane-wise maximum{unsigned_suffix}.
                #[inline(always)]
                pub fn max(self, other: Self) -> Self {{
                    Self(T::max(self.0, other.0), PhantomData)
                }}

        "});

        // abs for signed int only (before clamp)
        if has_abs(ty.elem) && !ty.elem.is_float() {
            code.push_str(&formatdoc! {"
                \x20   /// Lane-wise absolute value.
                    #[inline(always)]
                    pub fn abs(self) -> Self {{
                        Self(T::abs(self.0), PhantomData)
                    }}

            "});
        }

        code.push_str(&formatdoc! {"
            \x20   /// Clamp between lo and hi.
                #[inline(always)]
                pub fn clamp(self, lo: Self, hi: Self) -> Self {{
                    Self(T::clamp(self.0, lo.0, hi.0), PhantomData)
                }}

        "});
    }

    // ====== Comparisons ======
    code.push_str("    // ====== Comparisons ======\n\n");

    let signedness = signedness_doc(ty.elem);

    code.push_str(&formatdoc! {"
        \x20   /// Lane-wise equality (returns mask).
            #[inline(always)]
            pub fn simd_eq(self, other: Self) -> Self {{
                Self(T::simd_eq(self.0, other.0), PhantomData)
            }}

            /// Lane-wise inequality (returns mask).
            #[inline(always)]
            pub fn simd_ne(self, other: Self) -> Self {{
                Self(T::simd_ne(self.0, other.0), PhantomData)
            }}

            /// Lane-wise less-than{signedness} (returns mask).
            #[inline(always)]
            pub fn simd_lt(self, other: Self) -> Self {{
                Self(T::simd_lt(self.0, other.0), PhantomData)
            }}

            /// Lane-wise less-than-or-equal{signedness} (returns mask).
            #[inline(always)]
            pub fn simd_le(self, other: Self) -> Self {{
                Self(T::simd_le(self.0, other.0), PhantomData)
            }}

            /// Lane-wise greater-than{signedness} (returns mask).
            #[inline(always)]
            pub fn simd_gt(self, other: Self) -> Self {{
                Self(T::simd_gt(self.0, other.0), PhantomData)
            }}

            /// Lane-wise greater-than-or-equal{signedness} (returns mask).
            #[inline(always)]
            pub fn simd_ge(self, other: Self) -> Self {{
                Self(T::simd_ge(self.0, other.0), PhantomData)
            }}

            /// Select lanes: where mask is all-1s pick `if_true`, else `if_false`.
            #[inline(always)]
            pub fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {{
                Self(T::blend(mask.0, if_true.0, if_false.0), PhantomData)
            }}

    "});

    // ====== Reductions ======
    code.push_str("    // ====== Reductions ======\n\n");

    let wrapping_suffix = if ty.elem.is_float() {
        ""
    } else {
        " (wrapping)"
    };
    code.push_str(&formatdoc! {"
        \x20   /// Sum all {lanes} lanes{wrapping_suffix}.
            #[inline(always)]
            pub fn reduce_add(self) -> {elem} {{
                T::reduce_add(self.0)
            }}

    "});

    if has_reduce_min_max(ty.elem) {
        code.push_str(&formatdoc! {"
            \x20   /// Minimum across all {lanes} lanes.
                #[inline(always)]
                pub fn reduce_min(self) -> {elem} {{
                    T::reduce_min(self.0)
                }}

                /// Maximum across all {lanes} lanes.
                #[inline(always)]
                pub fn reduce_max(self) -> {elem} {{
                    T::reduce_max(self.0)
                }}

        "});
    }

    // ====== Approximations ====== (float only)
    if has_approx(ty.elem) {
        code.push_str("    // ====== Approximations ======\n\n");
        code.push_str(&formatdoc! {"
            \x20   /// Fast reciprocal approximation (~12-bit precision).
                #[inline(always)]
                pub fn rcp_approx(self) -> Self {{
                    Self(T::rcp_approx(self.0), PhantomData)
                }}

                /// Precise reciprocal (Newton-Raphson refined).
                #[inline(always)]
                pub fn recip(self) -> Self {{
                    Self(T::recip(self.0), PhantomData)
                }}

                /// Fast reciprocal square root approximation (~12-bit precision).
                #[inline(always)]
                pub fn rsqrt_approx(self) -> Self {{
                    Self(T::rsqrt_approx(self.0), PhantomData)
                }}

                /// Precise reciprocal square root (Newton-Raphson refined).
                #[inline(always)]
                pub fn rsqrt(self) -> Self {{
                    Self(T::rsqrt(self.0), PhantomData)
                }}

        "});
    }

    // ====== Shifts ====== (int only)
    if has_shifts(ty.elem) {
        code.push_str("    // ====== Shifts ======\n\n");

        code.push_str(&formatdoc! {"
            \x20   /// Shift left by constant.
                #[inline(always)]
                pub fn shl_const<const N: i32>(self) -> Self {{
                    Self(T::shl_const::<N>(self.0), PhantomData)
                }}

        "});

        // shr_arithmetic (signed int only, excluding i64)
        if has_shr_arithmetic(ty.elem) {
            code.push_str(&formatdoc! {"
                \x20   /// Arithmetic shift right by constant (sign-extending).
                    #[inline(always)]
                    pub fn shr_arithmetic_const<const N: i32>(self) -> Self {{
                        Self(T::shr_arithmetic_const::<N>(self.0), PhantomData)
                    }}

            "});
        }

        code.push_str(&formatdoc! {"
            \x20   /// Logical shift right by constant (zero-filling).
                #[inline(always)]
                pub fn shr_logical_const<const N: i32>(self) -> Self {{
                    Self(T::shr_logical_const::<N>(self.0), PhantomData)
                }}

                /// Alias for [`shl_const`](Self::shl_const).
                #[inline(always)]
                pub fn shl<const N: i32>(self) -> Self {{
                    self.shl_const::<N>()
                }}

        "});

        if has_shr_arithmetic(ty.elem) {
            code.push_str(&formatdoc! {"
                \x20   /// Alias for [`shr_arithmetic_const`](Self::shr_arithmetic_const).
                    #[inline(always)]
                    pub fn shr_arithmetic<const N: i32>(self) -> Self {{
                        self.shr_arithmetic_const::<N>()
                    }}

            "});
        }

        code.push_str(&formatdoc! {"
            \x20   /// Alias for [`shr_logical_const`](Self::shr_logical_const).
                #[inline(always)]
                pub fn shr_logical<const N: i32>(self) -> Self {{
                    self.shr_logical_const::<N>()
                }}

        "});
    }

    // ====== Bitwise ======
    code.push_str("    // ====== Bitwise ======\n\n");
    code.push_str(&formatdoc! {"
        \x20   /// Bitwise NOT.
            #[inline(always)]
            pub fn not(self) -> Self {{
                Self(T::not(self.0), PhantomData)
            }}

    "});

    // ====== Boolean ====== (int only)
    if has_boolean(ty.elem) {
        code.push_str("    // ====== Boolean ======\n\n");

        code.push_str(&formatdoc! {"
            \x20   /// True if all lanes have their {high_bit} set (all-1s mask).
                #[inline(always)]
                pub fn all_true(self) -> bool {{
                    T::all_true(self.0)
                }}

                /// True if any lane has its {high_bit} set.
                #[inline(always)]
                pub fn any_true(self) -> bool {{
                    T::any_true(self.0)
                }}

                /// Extract the high bit of each {elem_bits}-bit lane as a bitmask.
                #[inline(always)]
                pub fn bitmask(self) -> {bitmask_ty} {{
                    T::bitmask(self.0)
                }}

        "});
    }

    code.push_str("}\n\n");
    code
}

/// Generate partition_slice method.
fn gen_partition_slice(ty: &SimdType) -> String {
    let elem = ty.elem.name();
    let lanes = ty.lanes();

    // lanes >= 16 makes the single-line too long (two-digit lane count)
    let needs_multiline = lanes >= 16;

    if needs_multiline {
        formatdoc! {"
            \x20   /// Split a slice into SIMD-width chunks and a scalar remainder.
                ///
                /// Returns `(&[[{elem}; {lanes}]], &[{elem}])` — the bulk portion reinterpreted
                /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
                #[inline(always)]
                pub fn partition_slice(_: T, data: &[{elem}]) -> (&[[{elem}; {lanes}]], &[{elem}]) {{
                    let bulk = data.len() / {lanes};
                    let (head, tail) = data.split_at(bulk * {lanes});
                    // SAFETY: head.len() is bulk * {lanes}, so it's exactly `bulk` chunks of [{elem}; {lanes}].
                    // The pointer cast is valid because [{elem}] and [[{elem}; {lanes}]] have the same alignment.
                    let chunks =
                        unsafe {{ core::slice::from_raw_parts(head.as_ptr().cast::<[{elem}; {lanes}]>(), bulk) }};
                    (chunks, tail)
                }}

        "}
    } else {
        formatdoc! {"
            \x20   /// Split a slice into SIMD-width chunks and a scalar remainder.
                ///
                /// Returns `(&[[{elem}; {lanes}]], &[{elem}])` — the bulk portion reinterpreted
                /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
                #[inline(always)]
                pub fn partition_slice(_: T, data: &[{elem}]) -> (&[[{elem}; {lanes}]], &[{elem}]) {{
                    let bulk = data.len() / {lanes};
                    let (head, tail) = data.split_at(bulk * {lanes});
                    // SAFETY: head.len() is bulk * {lanes}, so it's exactly `bulk` chunks of [{elem}; {lanes}].
                    // The pointer cast is valid because [{elem}] and [[{elem}; {lanes}]] have the same alignment.
                    let chunks = unsafe {{ core::slice::from_raw_parts(head.as_ptr().cast::<[{elem}; {lanes}]>(), bulk) }};
                    (chunks, tail)
                }}

        "}
    }
}

/// Generate partition_slice_mut method.
fn gen_partition_slice_mut(ty: &SimdType) -> String {
    let elem = ty.elem.name();
    let lanes = ty.lanes();

    // partition_slice_mut always uses multiline format (line too long otherwise)
    formatdoc! {"
        \x20   /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
            ///
            /// Returns `(&mut [[{elem}; {lanes}]], &mut [{elem}])` — the bulk portion reinterpreted
            /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
            #[inline(always)]
            pub fn partition_slice_mut(_: T, data: &mut [{elem}]) -> (&mut [[{elem}; {lanes}]], &mut [{elem}]) {{
                let bulk = data.len() / {lanes};
                let (head, tail) = data.split_at_mut(bulk * {lanes});
                // SAFETY: head.len() is bulk * {lanes}, so it's exactly `bulk` chunks of [{elem}; {lanes}].
                // The pointer cast is valid because [{elem}] and [[{elem}; {lanes}]] have the same alignment.
                let chunks =
                    unsafe {{ core::slice::from_raw_parts_mut(head.as_mut_ptr().cast::<[{elem}; {lanes}]>(), bulk) }};
                (chunks, tail)
            }}

    "}
}

// ============================================================================
// Operator implementations
// ============================================================================

/// Generate binary and unary operator impls (Add, Sub, Mul, Div, Neg, BitAnd, BitOr, BitXor).
fn gen_operators(ty: &SimdType) -> String {
    let name = ty.name();
    let backend = backend_trait(ty);

    let mut code = formatdoc! {"
        // ============================================================================
        // Operator implementations
        // ============================================================================

    "};

    // Always: Add, Sub
    code.push_str(&gen_binary_op(&name, &backend, "Add", "add"));
    code.push_str(&gen_binary_op(&name, &backend, "Sub", "sub"));

    // Conditional: Mul
    if has_mul(ty.elem) {
        code.push_str(&gen_binary_op(&name, &backend, "Mul", "mul"));
    }

    // Conditional: Div (float only)
    if has_div(ty.elem) {
        code.push_str(&gen_binary_op(&name, &backend, "Div", "div"));
    }

    // Conditional: Neg
    if has_neg(ty.elem) {
        code.push_str(&formatdoc! {"
            impl<T: {backend}> Neg for {name}<T> {{
                type Output = Self;
                #[inline(always)]
                fn neg(self) -> Self {{
                    Self(T::neg(self.0), PhantomData)
                }}
            }}

        "});
    }

    // Always: BitAnd, BitOr, BitXor
    code.push_str(&gen_binary_op(&name, &backend, "BitAnd", "bitand"));
    code.push_str(&gen_binary_op(&name, &backend, "BitOr", "bitor"));
    code.push_str(&gen_binary_op(&name, &backend, "BitXor", "bitxor"));

    code
}

/// Generate a single binary operator impl block.
fn gen_binary_op(name: &str, backend: &str, trait_name: &str, method: &str) -> String {
    formatdoc! {"
        impl<T: {backend}> {trait_name} for {name}<T> {{
            type Output = Self;
            #[inline(always)]
            fn {method}(self, rhs: Self) -> Self {{
                Self(T::{method}(self.0, rhs.0), PhantomData)
            }}
        }}

    "}
}

// ============================================================================
// Assign operators
// ============================================================================

/// Generate assign operator impls (AddAssign, SubAssign, etc.).
fn gen_assign_operators(ty: &SimdType) -> String {
    let name = ty.name();
    let backend = backend_trait(ty);

    let mut code = formatdoc! {"
        // ============================================================================
        // Assign operators
        // ============================================================================

    "};

    // Always: AddAssign, SubAssign
    code.push_str(&gen_assign_op(
        &name,
        &backend,
        "AddAssign",
        "add_assign",
        "+",
    ));
    code.push_str(&gen_assign_op(
        &name,
        &backend,
        "SubAssign",
        "sub_assign",
        "-",
    ));

    // Conditional: MulAssign
    if has_mul(ty.elem) {
        code.push_str(&gen_assign_op(
            &name,
            &backend,
            "MulAssign",
            "mul_assign",
            "*",
        ));
    }

    // Conditional: DivAssign (float only)
    if has_div(ty.elem) {
        code.push_str(&gen_assign_op(
            &name,
            &backend,
            "DivAssign",
            "div_assign",
            "/",
        ));
    }

    // Always: BitAndAssign, BitOrAssign, BitXorAssign
    code.push_str(&gen_assign_op(
        &name,
        &backend,
        "BitAndAssign",
        "bitand_assign",
        "&",
    ));
    code.push_str(&gen_assign_op(
        &name,
        &backend,
        "BitOrAssign",
        "bitor_assign",
        "|",
    ));
    code.push_str(&gen_assign_op(
        &name,
        &backend,
        "BitXorAssign",
        "bitxor_assign",
        "^",
    ));

    code
}

/// Generate a single assign operator impl block.
fn gen_assign_op(name: &str, backend: &str, trait_name: &str, method: &str, op: &str) -> String {
    formatdoc! {"
        impl<T: {backend}> {trait_name} for {name}<T> {{
            #[inline(always)]
            fn {method}(&mut self, rhs: Self) {{
                *self = *self {op} rhs;
            }}
        }}

    "}
}

// ============================================================================
// Scalar broadcast operators
// ============================================================================

/// Generate scalar broadcast operator impls (Add<elem>, Sub<elem>, etc.).
fn gen_scalar_broadcast(ty: &SimdType) -> String {
    let name = ty.name();
    let elem = ty.elem.name();
    let backend = backend_trait(ty);

    // Section header varies by type
    let header_comment = if ty.elem.is_float() {
        format!("Scalar broadcast operators (v + 2.0, v * 0.5, etc.)")
    } else if has_mul(ty.elem) {
        format!("Scalar broadcast operators (v + 2, v * 3, etc.)")
    } else {
        format!("Scalar broadcast operators (v + 2, etc.)")
    };

    let mut code = formatdoc! {"
        // ============================================================================
        // {header_comment}
        // ============================================================================

    "};

    // Always: Add<elem>, Sub<elem>
    code.push_str(&gen_scalar_op(&name, &backend, elem, "Add", "add"));
    code.push_str(&gen_scalar_op(&name, &backend, elem, "Sub", "sub"));

    // Conditional: Mul<elem>
    if has_mul(ty.elem) {
        code.push_str(&gen_scalar_op(&name, &backend, elem, "Mul", "mul"));
    }

    // Conditional: Div<elem> (float only)
    if has_div(ty.elem) {
        code.push_str(&gen_scalar_op(&name, &backend, elem, "Div", "div"));
    }

    code
}

/// Generate a single scalar broadcast operator impl block.
fn gen_scalar_op(name: &str, backend: &str, elem: &str, trait_name: &str, method: &str) -> String {
    formatdoc! {"
        impl<T: {backend}> {trait_name}<{elem}> for {name}<T> {{
            type Output = Self;
            #[inline(always)]
            fn {method}(self, rhs: {elem}) -> Self {{
                Self(T::{method}(self.0, T::splat(rhs)), PhantomData)
            }}
        }}

    "}
}

// ============================================================================
// Index
// ============================================================================

/// Generate Index and IndexMut impls.
fn gen_index(ty: &SimdType) -> String {
    let name = ty.name();
    let elem = ty.elem.name();
    let lanes = ty.lanes();
    let backend = backend_trait(ty);

    formatdoc! {"
        // ============================================================================
        // Index
        // ============================================================================

        impl<T: {backend}> Index<usize> for {name}<T> {{
            type Output = {elem};
            #[inline(always)]
            fn index(&self, i: usize) -> &{elem} {{
                assert!(i < {lanes}, \"{name} index out of bounds: {{i}}\");
                // SAFETY: {name}'s repr is layout-compatible with [{elem}; {lanes}], and i < {lanes}.
                unsafe {{ &*(core::ptr::from_ref(self).cast::<{elem}>()).add(i) }}
            }}
        }}

        impl<T: {backend}> IndexMut<usize> for {name}<T> {{
            #[inline(always)]
            fn index_mut(&mut self, i: usize) -> &mut {elem} {{
                assert!(i < {lanes}, \"{name} index out of bounds: {{i}}\");
                // SAFETY: {name}'s repr is layout-compatible with [{elem}; {lanes}], and i < {lanes}.
                unsafe {{ &mut *(core::ptr::from_mut(self).cast::<{elem}>()).add(i) }}
            }}
        }}

    "}
}

// ============================================================================
// From array
// ============================================================================

/// Generate `From<name<T>> for [elem; lanes]`.
fn gen_from_array(ty: &SimdType) -> String {
    let name = ty.name();
    let elem = ty.elem.name();
    let lanes = ty.lanes();
    let backend = backend_trait(ty);

    formatdoc! {"
        // ============================================================================
        // Conversions
        // ============================================================================

        impl<T: {backend}> From<{name}<T>> for [{elem}; {lanes}] {{
            #[inline(always)]
            fn from(v: {name}<T>) -> [{elem}; {lanes}] {{
                T::to_array(v.0)
            }}
        }}

    "}
}

// ============================================================================
// Debug
// ============================================================================

/// Generate Debug impl.
fn gen_debug(ty: &SimdType) -> String {
    let name = ty.name();
    let backend = backend_trait(ty);

    formatdoc! {"
        // ============================================================================
        // Debug
        // ============================================================================

        impl<T: {backend}> core::fmt::Debug for {name}<T> {{
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {{
                let arr = T::to_array(self.0);
                f.debug_tuple(\"{name}\").field(&arr).finish()
            }}
        }}

    "}
}

// ============================================================================
// Cross-type conversions

// ============================================================================
// Cross-type conversions
// ============================================================================

/// Generate cross-type conversion impl blocks.
fn gen_cross_type(ty: &SimdType) -> String {
    let name = ty.name();
    let conversions = all_conversions();

    let matching: Vec<&Conversion> = conversions.iter().filter(|c| c.src == name).collect();

    if matching.is_empty() {
        return String::new();
    }

    let mut code = String::new();
    for conv in &matching {
        code.push_str(&(conv.gen_fn)(&name, conv.trait_bound));
        code.push('\n');
    }
    code
}

// ============================================================================
// Platform-specific concrete impls
// ============================================================================

/// Generate platform-specific impl blocks (implementation_name, raw, from_mXXX).
fn gen_platform(ty: &SimdType) -> String {
    let name = ty.name();

    match ty.width {
        SimdWidth::W128 | SimdWidth::W256 => {
            // Non-W512: single impl block with raw/from methods
            let raw_type = x86_raw_type(ty);
            let from_fn = from_raw_fn_name(ty);

            formatdoc! {"
                // ============================================================================
                // Platform-specific concrete impls
                // ============================================================================

                #[cfg(target_arch = \"x86_64\")]
                impl {name}<archmage::X64V3Token> {{
                    /// Implementation identifier for this backend.
                    pub const fn implementation_name() -> &'static str {{
                        \"x86::v3::{name}\"
                    }}

                    /// Get the raw `{raw_type}` value.
                    #[inline(always)]
                    pub fn raw(self) -> core::arch::x86_64::{raw_type} {{
                        self.0
                    }}

                    /// Create from a raw `{raw_type}` (token-gated, zero-cost).
                    #[inline(always)]
                    pub fn {from_fn}(_: archmage::X64V3Token, v: core::arch::x86_64::{raw_type}) -> Self {{
                        Self(v, PhantomData)
                    }}
                }}
            "}
        }
        SimdWidth::W512 => {
            // W512: three impl blocks, no raw/from methods
            formatdoc! {"
                // ============================================================================
                // Platform-specific implementation info
                // ============================================================================

                #[cfg(target_arch = \"x86_64\")]
                impl {name}<archmage::X64V3Token> {{
                    /// Implementation identifier for this backend.
                    pub const fn implementation_name() -> &'static str {{
                        \"polyfill::v3_512::{name}\"
                    }}
                }}

                #[cfg(all(target_arch = \"x86_64\", feature = \"avx512\"))]
                impl {name}<archmage::X64V4Token> {{
                    /// Implementation identifier for this backend.
                    pub const fn implementation_name() -> &'static str {{
                        \"x86::v4::{name}\"
                    }}
                }}

                #[cfg(all(target_arch = \"x86_64\", feature = \"avx512\"))]
                impl {name}<archmage::X64V4xToken> {{
                    /// Implementation identifier for this backend.
                    pub const fn implementation_name() -> &'static str {{
                        \"x86::v4x::{name}\"
                    }}
                }}
            "}
        }
    }
}

// ============================================================================
// Popcnt extension
// ============================================================================

/// Generate popcnt impl block for W512 integer types only.
fn gen_popcnt(ty: &SimdType) -> String {
    // Only W512 integer types get popcnt
    if ty.width != SimdWidth::W512 || ty.elem.is_float() {
        return String::new();
    }

    let name = ty.name();

    formatdoc! {"

        // ============================================================================
        // Extension: popcnt (requires Modern token)
        // ============================================================================

        #[cfg(feature = \"avx512\")]
        impl<T: crate::simd::backends::{name}PopcntBackend> {name}<T> {{
            /// Count set bits in each lane (popcnt).
            ///
            /// Returns a vector where each lane contains the number of 1-bits
            /// in the corresponding lane of `self`.
            ///
            /// Requires AVX-512 Modern token (VPOPCNTDQ or BITALG extension).
            #[inline(always)]
            pub fn popcnt(self) -> Self {{
                Self(T::popcnt(self.0), core::marker::PhantomData)
            }}
        }}
    "}
}

// ============================================================================
// f32 <-> i32 conversions (full numeric + bitcast)
// ============================================================================

/// Generate f32->i32 conversions on the float type (f32x4 or f32x8).
///
/// Generates: bitcast_to_i32, from_i32_bitcast, to_i32, to_i32_round, from_i32,
/// plus backward-compatible aliases (bitcast_i32xN, to_i32xN, to_i32xN_round,
/// from_i32xN, bitcast_ref_i32xN, bitcast_mut_i32xN).
fn gen_f32_i32_convert_on_float(src: &str, trait_bound: &str) -> String {
    // Derive int type: "f32x4" -> "i32x4", "f32x8" -> "i32x8"
    let lanes = lane_count(src);
    let int_type = format!("i32x{lanes}");

    formatdoc! {r#"
        // ============================================================================
        // Cross-type conversions (available when T implements conversion traits)
        // ============================================================================

        impl<T: crate::simd::backends::{trait_bound}> {src}<T> {{
            /// Bitcast to {int_type} (reinterpret bits, no conversion).
            #[inline(always)]
            pub fn bitcast_to_i32(self) -> super::{int_type}<T> {{
                super::{int_type}::from_repr_unchecked(T::bitcast_f32_to_i32(self.0))
            }}

            /// Create from {int_type} via bitcast (reinterpret bits, no conversion).
            #[inline(always)]
            pub fn from_i32_bitcast(_: T, v: super::{int_type}<T>) -> Self {{
                Self(T::bitcast_i32_to_f32(v.into_repr()), PhantomData)
            }}

            /// Convert to {int_type} with truncation toward zero.
            #[inline(always)]
            pub fn to_i32(self) -> super::{int_type}<T> {{
                super::{int_type}::from_repr_unchecked(T::convert_f32_to_i32(self.0))
            }}

            /// Convert to {int_type} with rounding to nearest.
            #[inline(always)]
            pub fn to_i32_round(self) -> super::{int_type}<T> {{
                super::{int_type}::from_repr_unchecked(T::convert_f32_to_i32_round(self.0))
            }}

            /// Create from {int_type} via numeric conversion.
            #[inline(always)]
            pub fn from_i32(_: T, v: super::{int_type}<T>) -> Self {{
                Self(T::convert_i32_to_f32(v.into_repr()), PhantomData)
            }}

            // ====== Backward-compatible aliases (old generated API names) ======

            /// Alias for [`bitcast_to_i32`](Self::bitcast_to_i32).
            #[inline(always)]
            pub fn bitcast_{int_type}(self) -> super::{int_type}<T> {{
                self.bitcast_to_i32()
            }}

            /// Alias for [`to_i32`](Self::to_i32).
            #[inline(always)]
            pub fn to_{int_type}(self) -> super::{int_type}<T> {{
                self.to_i32()
            }}

            /// Alias for [`to_i32_round`](Self::to_i32_round).
            #[inline(always)]
            pub fn to_{int_type}_round(self) -> super::{int_type}<T> {{
                self.to_i32_round()
            }}

            /// Alias for [`from_i32`](Self::from_i32).
            #[inline(always)]
            pub fn from_{int_type}(token: T, v: super::{int_type}<T>) -> Self {{
                Self::from_i32(token, v)
            }}

            /// Alias for [`bitcast_ref_i32`](Self::bitcast_ref_i32) (from block_ops).
            #[inline(always)]
            pub fn bitcast_ref_{int_type}(&self) -> &super::{int_type}<T> {{
                self.bitcast_ref_i32()
            }}

            /// Alias for [`bitcast_mut_i32`](Self::bitcast_mut_i32) (from block_ops).
            #[inline(always)]
            pub fn bitcast_mut_{int_type}(&mut self) -> &mut super::{int_type}<T> {{
                self.bitcast_mut_i32()
            }}
        }}
    "#}
}

/// Generate i32->f32 conversions on the int type (i32x4 or i32x8).
///
/// Generates: bitcast_to_f32, to_f32, plus backward-compatible aliases
/// (bitcast_f32xN, to_f32xN).
fn gen_f32_i32_convert_on_int(src: &str, trait_bound: &str) -> String {
    // Derive float type: "i32x4" -> "f32x4", "i32x8" -> "f32x8"
    let lanes = lane_count(src);
    let float_type = format!("f32x{lanes}");

    formatdoc! {r#"
        // ============================================================================
        // Cross-type conversions (available when T implements conversion traits)
        // ============================================================================

        impl<T: crate::simd::backends::{trait_bound}> {src}<T> {{
            /// Bitcast to {float_type} (reinterpret bits, no conversion).
            #[inline(always)]
            pub fn bitcast_to_f32(self) -> super::{float_type}<T> {{
                super::{float_type}::from_repr_unchecked(T::bitcast_i32_to_f32(self.0))
            }}

            /// Convert to {float_type} (numeric conversion).
            #[inline(always)]
            pub fn to_f32(self) -> super::{float_type}<T> {{
                super::{float_type}::from_repr_unchecked(T::convert_i32_to_f32(self.0))
            }}

            // ====== Backward-compatible aliases (old generated API names) ======

            /// Alias for [`bitcast_to_f32`](Self::bitcast_to_f32).
            #[inline(always)]
            pub fn bitcast_{float_type}(self) -> super::{float_type}<T> {{
                self.bitcast_to_f32()
            }}

            /// Alias for [`to_f32`](Self::to_f32).
            #[inline(always)]
            pub fn to_{float_type}(self) -> super::{float_type}<T> {{
                self.to_f32()
            }}
        }}
    "#}
}

// ============================================================================
// Signed <-> Unsigned integer bitcasts (i8/u8, i16/u16)
// ============================================================================

/// Generate signed->unsigned bitcast on the signed type (e.g., i8x16->u8x16).
///
/// The trait_bound is derived from the source type: uppercase_first(src) + "Bitcast".
/// Methods: bitcast_{target}, bitcast_ref_{target}, bitcast_mut_{target}.
fn gen_signed_unsigned_bitcast(
    src: &str,
    target: &str,
    to_method: &str,
    _from_method: &str,
) -> String {
    let trait_bound = format!("{}Bitcast", uppercase_first(src));
    let src_elem = elem_prefix(src);
    let target_elem = elem_prefix(target);
    let src_lanes = lane_count(src);
    let x86_int = x86_int_type_for_name(src);

    formatdoc! {r#"
        // ============================================================================
        // Cross-type conversions ({src_elem} ↔ {target_elem} bitcast)
        // ============================================================================

        impl<T: crate::simd::backends::{trait_bound}> {src}<T> {{
            /// Bitcast to {target} (reinterpret bits, no conversion).
            #[inline(always)]
            pub fn bitcast_{target}(self) -> super::{target}<T> {{
                super::{target}::from_repr_unchecked(T::bitcast_{to_method}(self.0))
            }}

            /// Bitcast to {target} by reference (zero-cost).
            #[inline(always)]
            pub fn bitcast_ref_{target}(&self) -> &super::{target}<T> {{
                // SAFETY: {src} and {target} share the same repr ({x86_int} / [{src_elem};{src_lanes}] / etc.)
                unsafe {{ &*(core::ptr::from_ref(self).cast()) }}
            }}

            /// Bitcast to {target} by mutable reference (zero-cost).
            #[inline(always)]
            pub fn bitcast_mut_{target}(&mut self) -> &mut super::{target}<T> {{
                // SAFETY: {src} and {target} share the same repr
                unsafe {{ &mut *(core::ptr::from_mut(self).cast()) }}
            }}
        }}
    "#}
}

/// Generate unsigned->signed bitcast on the unsigned type (e.g., u8x16->i8x16).
///
/// The trait_bound is derived from the target (signed) type: uppercase_first(target) + "Bitcast".
/// Methods: bitcast_{alias_name}, bitcast_ref_{alias_name}, bitcast_mut_{alias_name}.
/// Here alias_name is typically the full target type name (e.g., "i8x16" -> method "bitcast_i8x16").
fn gen_unsigned_signed_bitcast(
    src: &str,
    target: &str,
    from_method: &str,
    _alias_name: &str,
) -> String {
    let trait_bound = format!("{}Bitcast", uppercase_first(target));
    let src_elem = elem_prefix(src);
    let target_elem = elem_prefix(target);
    let src_lanes = lane_count(src);
    let x86_int = x86_int_type_for_name(src);

    formatdoc! {r#"
        // ============================================================================
        // Cross-type conversions ({src_elem} ↔ {target_elem} bitcast)
        // ============================================================================

        impl<T: crate::simd::backends::{trait_bound}> {src}<T> {{
            /// Bitcast to {target} (reinterpret bits, no conversion).
            #[inline(always)]
            pub fn bitcast_{target}(self) -> super::{target}<T> {{
                super::{target}::from_repr_unchecked(T::bitcast_{from_method}(self.0))
            }}

            /// Bitcast to {target} by reference (zero-cost).
            #[inline(always)]
            pub fn bitcast_ref_{target}(&self) -> &super::{target}<T> {{
                // SAFETY: {src} and {target} share the same repr ({x86_int} / [{src_elem};{src_lanes}] / etc.)
                unsafe {{ &*(core::ptr::from_ref(self).cast()) }}
            }}

            /// Bitcast to {target} by mutable reference (zero-cost).
            #[inline(always)]
            pub fn bitcast_mut_{target}(&mut self) -> &mut super::{target}<T> {{
                // SAFETY: {src} and {target} share the same repr
                unsafe {{ &mut *(core::ptr::from_mut(self).cast()) }}
            }}
        }}
    "#}
}

// ============================================================================
// u32 -> i32 bitcasts (with backward-compatible alias)
// ============================================================================

/// Generate u32->i32 bitcast (e.g., u32x4->i32x4, u32x8->i32x8).
///
/// The trait_bound is derived from the source type: uppercase_first(src) + "Bitcast".
/// Methods: bitcast_to_i32, bitcast_ref_{target}, bitcast_mut_{target},
/// plus alias bitcast_{target}.
fn gen_u32_i32_bitcast(src: &str, target: &str, method: &str) -> String {
    let trait_bound = format!("{}Bitcast", uppercase_first(src));
    let src_elem = elem_prefix(src);
    let target_elem = elem_prefix(target);
    let src_lanes = lane_count(src);
    let x86_int = x86_int_type_for_name(src);

    formatdoc! {r#"
        // ============================================================================
        // Cross-type conversions ({target_elem} ↔ {src_elem} bitcast)
        // ============================================================================

        impl<T: crate::simd::backends::{trait_bound}> {src}<T> {{
            /// Bitcast to {target} (reinterpret bits, no conversion).
            #[inline(always)]
            pub fn bitcast_to_i32(self) -> super::{target}<T> {{
                super::{target}::from_repr_unchecked(T::bitcast_{method}(self.0))
            }}

            /// Bitcast to {target} by reference (zero-cost).
            #[inline(always)]
            pub fn bitcast_ref_{target}(&self) -> &super::{target}<T> {{
                // SAFETY: {src} and {target} share the same repr ({x86_int} / [{src_elem};{src_lanes}] / etc.)
                unsafe {{ &*(core::ptr::from_ref(self).cast()) }}
            }}

            /// Bitcast to {target} by mutable reference (zero-cost).
            #[inline(always)]
            pub fn bitcast_mut_{target}(&mut self) -> &mut super::{target}<T> {{
                // SAFETY: {src} and {target} share the same repr
                unsafe {{ &mut *(core::ptr::from_mut(self).cast()) }}
            }}

            // ====== Backward-compatible aliases ======

            /// Alias for [`bitcast_to_i32`](Self::bitcast_to_i32).
            #[inline(always)]
            pub fn bitcast_{target}(self) -> super::{target}<T> {{
                self.bitcast_to_i32()
            }}
        }}
    "#}
}

// ============================================================================
// u64 -> i64 bitcasts (no backward-compatible alias)
// ============================================================================

/// Generate u64->i64 bitcast (e.g., u64x2->i64x2, u64x4->i64x4).
///
/// The trait_bound is derived from the source type: uppercase_first(src) + "Bitcast".
/// Methods: bitcast_{target}, bitcast_ref_{target}, bitcast_mut_{target}.
/// No backward-compatible aliases (u64 bitcasts don't have `bitcast_to_i64`).
fn gen_u64_i64_bitcast(src: &str, target: &str, method: &str) -> String {
    let trait_bound = format!("{}Bitcast", uppercase_first(src));
    let src_elem = elem_prefix(src);
    let target_elem = elem_prefix(target);
    let src_lanes = lane_count(src);
    let x86_int = x86_int_type_for_name(src);

    formatdoc! {r#"
        // ============================================================================
        // Cross-type conversions ({src_elem} ↔ {target_elem} bitcast)
        // ============================================================================

        impl<T: crate::simd::backends::{trait_bound}> {src}<T> {{
            /// Bitcast to {target} (reinterpret bits, no conversion).
            #[inline(always)]
            pub fn bitcast_{target}(self) -> super::{target}<T> {{
                super::{target}::from_repr_unchecked(T::bitcast_{method}(self.0))
            }}

            /// Bitcast to {target} by reference (zero-cost).
            #[inline(always)]
            pub fn bitcast_ref_{target}(&self) -> &super::{target}<T> {{
                // SAFETY: {src} and {target} share the same repr ({x86_int} / [{src_elem};{src_lanes}] / etc.)
                unsafe {{ &*(core::ptr::from_ref(self).cast()) }}
            }}

            /// Bitcast to {target} by mutable reference (zero-cost).
            #[inline(always)]
            pub fn bitcast_mut_{target}(&mut self) -> &mut super::{target}<T> {{
                // SAFETY: {src} and {target} share the same repr
                unsafe {{ &mut *(core::ptr::from_mut(self).cast()) }}
            }}
        }}
    "#}
}

// ============================================================================
// i64 -> f64 bitcasts (with backward-compatible alias)
// ============================================================================

/// Generate i64->f64 bitcast (e.g., i64x2->f64x2, i64x4->f64x4).
///
/// The trait_bound is derived from the source type: uppercase_first(src) + "Bitcast".
/// Methods: bitcast_to_f64, bitcast_ref_{target}, bitcast_mut_{target},
/// plus backward-compatible alias `bitcast_{target}`.
///
/// The `_has_alias` parameter is retained for interface compatibility but ignored;
/// both i64x2 and i64x4 have the alias in the handwritten files.
fn gen_i64_f64_bitcast(src: &str, target: &str, method: &str, _has_alias: bool) -> String {
    let trait_bound = format!("{}Bitcast", uppercase_first(src));
    let src_elem = elem_prefix(src);
    let target_elem = elem_prefix(target);
    let src_lanes = lane_count(src);

    // The SAFETY comment for ref/mut uses a repr hint that varies by width
    let repr_hint = if src_lanes == "2" {
        format!("__m128i/__m128d / [{src_elem};2] / etc.")
    } else {
        format!("__m256i/__m256d / [{src_elem};{src_lanes}] / etc.")
    };

    formatdoc! {r#"
        // ============================================================================
        // Cross-type conversions ({src_elem} ↔ {target_elem} bitcast)
        // ============================================================================

        impl<T: crate::simd::backends::{trait_bound}> {src}<T> {{
            /// Bitcast to {target} (reinterpret bits, no conversion).
            #[inline(always)]
            pub fn bitcast_to_f64(self) -> super::{target}<T> {{
                super::{target}::from_repr_unchecked(T::bitcast_{method}(self.0))
            }}

            /// Bitcast to {target} by reference (zero-cost).
            #[inline(always)]
            pub fn bitcast_ref_{target}(&self) -> &super::{target}<T> {{
                // SAFETY: {src} and {target} share the same repr ({repr_hint})
                unsafe {{ &*(core::ptr::from_ref(self).cast()) }}
            }}

            /// Bitcast to {target} by mutable reference (zero-cost).
            #[inline(always)]
            pub fn bitcast_mut_{target}(&mut self) -> &mut super::{target}<T> {{
                // SAFETY: {src} and {target} share the same repr
                unsafe {{ &mut *(core::ptr::from_mut(self).cast()) }}
            }}

            // ====== Backward-compatible aliases ======

            /// Alias for [`bitcast_to_f64`](Self::bitcast_to_f64).
            #[inline(always)]
            pub fn bitcast_{target}(self) -> super::{target}<T> {{
                self.bitcast_to_f64()
            }}
        }}
    "#}
}

// ============================================================================
// Part B: Block ops generator
// ============================================================================

/// Generate the content for a `block_ops_{name}.rs` file.
///
/// All types get basic block ops (as_array, as_array_mut, as_bytes, as_bytes_mut,
/// from_bytes, from_bytes_owned, cast_slice, cast_slice_mut).
///
/// f32x4/f32x8 get additional image/interleave/transpose operations.
/// u32x4 gets cross-type bitcast to f32x4.
fn gen_block_ops(ty: &SimdType) -> String {
    let name = ty.name();
    let elem = ty.elem.name();
    let lanes = ty.lanes();
    let byte_size = ty.elem.size_bytes() * lanes;
    let trait_name = backend_trait(ty);

    let mut code = String::new();

    // File header
    match name.as_str() {
        "f32x4" | "f32x8" => {
            code.push_str(&formatdoc! {r#"
                //! Block, view, and image operations for `{name}<T>`.
                //!
                //! Array/byte views, slice casting, interleave/deinterleave,
                //! matrix transpose, RGBA pixel operations, and cross-type bitcast references.
            "#});
        }
        "u32x4" => {
            code.push_str(&formatdoc! {r#"
                //! Block and view operations for `{name}<T>`.
                //!
                //! Array/byte views, slice casting, and cross-type bitcast to f32x4.
            "#});
        }
        _ => {
            code.push_str(&formatdoc! {r#"
                //! Block and view operations for `{name}<T>`.
                //!
                //! Array/byte views and slice casting.
            "#});
        }
    }

    // Imports
    code.push_str(&formatdoc! {r#"

        use crate::simd::backends::{trait_name};
        use crate::simd::generic::{name};

    "#});

    // Main impl block: basic block ops (opens impl block, does NOT close it)
    code.push_str(&gen_basic_block_ops(
        &name,
        elem,
        lanes,
        byte_size,
        &trait_name,
    ));

    // Type-specific extras (f32x4/f32x8 extras close the impl block)
    match name.as_str() {
        "f32x4" => {
            code.push_str(&gen_f32x4_extras());
            code.push_str(&gen_f32_bitcast_ref_i32(&name, lanes));
        }
        "f32x8" => {
            code.push_str(&gen_f32x8_extras());
            code.push_str(&gen_f32_bitcast_ref_i32(&name, lanes));
        }
        "u32x4" => {
            // Close the basic impl block, then add separate cross-type impl block
            code.push_str("}\n");
            code.push_str(&gen_u32x4_bitcast_f32());
        }
        _ => {
            // Close the basic impl block
            code.push_str("}\n");
        }
    }

    code
}

/// Generate the basic block ops impl block (all types get these).
fn gen_basic_block_ops(
    name: &str,
    elem: &str,
    lanes: usize,
    byte_size: usize,
    trait_name: &str,
) -> String {
    formatdoc! {r#"
        impl<T: {trait_name}> {name}<T> {{
            // ====== Array/Byte Views ======

            /// Reference to underlying array (zero-copy).
            #[inline(always)]
            pub fn as_array(&self) -> &[{elem}; {lanes}] {{
                // SAFETY: {name}<T> is repr(transparent) over T::Repr, layout-compatible with [{elem}; {lanes}]
                unsafe {{ &*core::ptr::from_ref(self).cast::<[{elem}; {lanes}]>() }}
            }}

            /// Mutable reference to underlying array (zero-copy).
            #[inline(always)]
            pub fn as_array_mut(&mut self) -> &mut [{elem}; {lanes}] {{
                // SAFETY: same layout guarantee as as_array
                unsafe {{ &mut *core::ptr::from_mut(self).cast::<[{elem}; {lanes}]>() }}
            }}

            /// View as byte array.
            #[inline(always)]
            pub fn as_bytes(&self) -> &[u8; {byte_size}] {{
                // SAFETY: {name}<T> is exactly {byte_size} bytes for all backends
                unsafe {{ &*core::ptr::from_ref(self).cast::<[u8; {byte_size}]>() }}
            }}

            /// View as mutable byte array.
            #[inline(always)]
            pub fn as_bytes_mut(&mut self) -> &mut [u8; {byte_size}] {{
                // SAFETY: {name}<T> is exactly {byte_size} bytes for all backends
                unsafe {{ &mut *core::ptr::from_mut(self).cast::<[u8; {byte_size}]>() }}
            }}

            /// Create from byte array reference (token-gated).
            #[inline(always)]
            pub fn from_bytes(_: T, bytes: &[u8; {byte_size}]) -> Self {{
                // SAFETY: {name}<T> is exactly {byte_size} bytes; transmute_copy uses read_unaligned
                unsafe {{ core::mem::transmute_copy(bytes) }}
            }}

            /// Create from owned byte array (token-gated).
            #[inline(always)]
            pub fn from_bytes_owned(_: T, bytes: [u8; {byte_size}]) -> Self {{
                // SAFETY: {name}<T> is exactly {byte_size} bytes; transmute_copy uses read_unaligned
                unsafe {{ core::mem::transmute_copy(&bytes) }}
            }}

            // ====== Slice Casting ======

            /// Reinterpret a scalar slice as a SIMD vector slice (token-gated).
            ///
            /// Returns `None` if length is not a multiple of {lanes} or alignment is wrong.
            #[inline(always)]
            pub fn cast_slice(_: T, slice: &[{elem}]) -> Option<&[Self]> {{
                if !slice.len().is_multiple_of({lanes}) {{
                    return None;
                }}
                let ptr = slice.as_ptr();
                if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {{
                    return None;
                }}
                let len = slice.len() / {lanes};
                // SAFETY: alignment and length checked, layout is compatible
                Some(unsafe {{ core::slice::from_raw_parts(ptr.cast::<Self>(), len) }})
            }}

            /// Reinterpret a mutable scalar slice as a SIMD vector slice (token-gated).
            ///
            /// Returns `None` if length is not a multiple of {lanes} or alignment is wrong.
            #[inline(always)]
            pub fn cast_slice_mut(_: T, slice: &mut [{elem}]) -> Option<&mut [Self]> {{
                if !slice.len().is_multiple_of({lanes}) {{
                    return None;
                }}
                let ptr = slice.as_mut_ptr();
                if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {{
                    return None;
                }}
                let len = slice.len() / {lanes};
                // SAFETY: alignment and length checked, layout is compatible
                Some(unsafe {{ core::slice::from_raw_parts_mut(ptr.cast::<Self>(), len) }})
            }}
    "#}
}

/// Generate f32x4-specific extras (u8 conversions, interleave, deinterleave, rgba, transpose).
/// Returns the closing brace of the main impl block plus the extra methods.
fn gen_f32x4_extras() -> String {
    formatdoc! {r#"

            // ====== u8 Conversions ======

            /// Load 4 u8 values and convert to f32x4.
            ///
            /// Values are in `[0.0, 255.0]`. Useful for image processing.
            #[inline(always)]
            pub fn from_u8(bytes: &[u8; 4]) -> Self {{
                Self::from_repr_unchecked(T::from_array(core::array::from_fn(|i| bytes[i] as f32)))
            }}

            /// Convert to 4 u8 values with saturation.
            ///
            /// Values are rounded and clamped to `[0, 255]`.
            #[inline(always)]
            pub fn to_u8(self) -> [u8; 4] {{
                let arr = self.to_array();
                core::array::from_fn(|i| arr[i].round().clamp(0.0, 255.0) as u8)
            }}

            // ====== Interleave Operations ======

            /// Interleave low elements.
            ///
            /// ```text
            /// [a0,a1,a2,a3] + [b0,b1,b2,b3] → [a0,b0,a1,b1]
            /// ```
            #[inline(always)]
            pub fn interleave_lo(self, other: Self) -> Self {{
                let a = self.to_array();
                let b = other.to_array();
                Self::from_repr_unchecked(T::from_array([a[0], b[0], a[1], b[1]]))
            }}

            /// Interleave high elements.
            ///
            /// ```text
            /// [a0,a1,a2,a3] + [b0,b1,b2,b3] → [a2,b2,a3,b3]
            /// ```
            #[inline(always)]
            pub fn interleave_hi(self, other: Self) -> Self {{
                let a = self.to_array();
                let b = other.to_array();
                Self::from_repr_unchecked(T::from_array([a[2], b[2], a[3], b[3]]))
            }}

            /// Interleave two vectors: returns `(interleave_lo, interleave_hi)`.
            #[inline(always)]
            pub fn interleave(self, other: Self) -> (Self, Self) {{
                let a = self.to_array();
                let b = other.to_array();
                (
                    Self::from_repr_unchecked(T::from_array([a[0], b[0], a[1], b[1]])),
                    Self::from_repr_unchecked(T::from_array([a[2], b[2], a[3], b[3]])),
                )
            }}

            // ====== 4-Channel Interleave/Deinterleave ======

            /// Deinterleave 4 RGBA pixels from AoS to SoA format.
            ///
            /// Input: 4 vectors, each containing one pixel `[R, G, B, A]`.
            /// Output: 4 vectors, each containing one channel across all pixels.
            ///
            /// This is equivalent to `transpose_4x4_copy`.
            #[inline]
            pub fn deinterleave_4ch(rgba: [Self; 4]) -> [Self; 4] {{
                Self::transpose_4x4_copy(rgba)
            }}

            /// Interleave 4 channels from SoA to AoS format.
            ///
            /// Input: 4 vectors, each containing one channel across pixels.
            /// Output: 4 vectors, each containing one complete RGBA pixel.
            ///
            /// This is the inverse of `deinterleave_4ch` (also equivalent to transpose).
            #[inline]
            pub fn interleave_4ch(channels: [Self; 4]) -> [Self; 4] {{
                Self::transpose_4x4_copy(channels)
            }}

            // ====== RGBA Load/Store ======

            /// Load 4 RGBA u8 pixels and deinterleave to 4 f32x4 channel vectors.
            ///
            /// Input: 16 bytes = 4 RGBA pixels in interleaved format.
            /// Output: `(R, G, B, A)` where each is f32x4 with values in `[0.0, 255.0]`.
            #[inline]
            pub fn load_4_rgba_u8(rgba: &[u8; 16]) -> (Self, Self, Self, Self) {{
                let r: [f32; 4] = core::array::from_fn(|i| rgba[i * 4] as f32);
                let g: [f32; 4] = core::array::from_fn(|i| rgba[i * 4 + 1] as f32);
                let b: [f32; 4] = core::array::from_fn(|i| rgba[i * 4 + 2] as f32);
                let a: [f32; 4] = core::array::from_fn(|i| rgba[i * 4 + 3] as f32);
                (
                    Self::from_repr_unchecked(T::from_array(r)),
                    Self::from_repr_unchecked(T::from_array(g)),
                    Self::from_repr_unchecked(T::from_array(b)),
                    Self::from_repr_unchecked(T::from_array(a)),
                )
            }}

            /// Interleave 4 f32x4 channels and store as 4 RGBA u8 pixels.
            ///
            /// Values are rounded and clamped to `[0, 255]`.
            /// Output: 16 bytes = 4 RGBA pixels in interleaved format.
            #[inline]
            pub fn store_4_rgba_u8(r: Self, g: Self, b: Self, a: Self) -> [u8; 16] {{
                let rv = r.to_array();
                let gv = g.to_array();
                let bv = b.to_array();
                let av = a.to_array();
                let mut out = [0u8; 16];
                for i in 0..4 {{
                    out[i * 4] = rv[i].round().clamp(0.0, 255.0) as u8;
                    out[i * 4 + 1] = gv[i].round().clamp(0.0, 255.0) as u8;
                    out[i * 4 + 2] = bv[i].round().clamp(0.0, 255.0) as u8;
                    out[i * 4 + 3] = av[i].round().clamp(0.0, 255.0) as u8;
                }}
                out
            }}

            // ====== Matrix Transpose ======

            /// Transpose a 4x4 matrix represented as 4 row vectors (in-place).
            ///
            /// After transpose, `rows[i][j]` becomes `rows[j][i]`.
            #[inline]
            pub fn transpose_4x4(rows: &mut [Self; 4]) {{
                let a = rows[0].to_array();
                let b = rows[1].to_array();
                let c = rows[2].to_array();
                let d = rows[3].to_array();
                rows[0] = Self::from_repr_unchecked(T::from_array([a[0], b[0], c[0], d[0]]));
                rows[1] = Self::from_repr_unchecked(T::from_array([a[1], b[1], c[1], d[1]]));
                rows[2] = Self::from_repr_unchecked(T::from_array([a[2], b[2], c[2], d[2]]));
                rows[3] = Self::from_repr_unchecked(T::from_array([a[3], b[3], c[3], d[3]]));
            }}

            /// Transpose a 4x4 matrix, returning the transposed rows.
            #[inline]
            pub fn transpose_4x4_copy(rows: [Self; 4]) -> [Self; 4] {{
                let mut result = rows;
                Self::transpose_4x4(&mut result);
                result
            }}
        }}
    "#}
}

/// Generate f32x8-specific extras (u8 conversions, interleave, deinterleave, rgba, transpose,
/// load_8x8, store_8x8).
/// Returns the closing brace of the main impl block plus the extra methods.
fn gen_f32x8_extras() -> String {
    formatdoc! {r#"

            // ====== u8 Conversions ======

            /// Load 8 u8 values and convert to f32x8.
            ///
            /// Values are in `[0.0, 255.0]`. Useful for image processing.
            #[inline(always)]
            pub fn from_u8(bytes: &[u8; 8]) -> Self {{
                Self::from_repr_unchecked(T::from_array(core::array::from_fn(|i| bytes[i] as f32)))
            }}

            /// Convert to 8 u8 values with saturation.
            ///
            /// Values are rounded and clamped to `[0, 255]`.
            #[inline(always)]
            pub fn to_u8(self) -> [u8; 8] {{
                let arr = self.to_array();
                core::array::from_fn(|i| arr[i].round().clamp(0.0, 255.0) as u8)
            }}

            // ====== Interleave Operations ======

            /// Interleave low elements within 128-bit lanes.
            ///
            /// ```text
            /// [a0,a1,a2,a3,a4,a5,a6,a7] + [b0,b1,b2,b3,b4,b5,b6,b7]
            /// → [a0,b0,a1,b1,a4,b4,a5,b5]
            /// ```
            #[inline(always)]
            pub fn interleave_lo(self, other: Self) -> Self {{
                let a = self.to_array();
                let b = other.to_array();
                Self::from_repr_unchecked(T::from_array([
                    a[0], b[0], a[1], b[1], a[4], b[4], a[5], b[5],
                ]))
            }}

            /// Interleave high elements within 128-bit lanes.
            ///
            /// ```text
            /// [a0,a1,a2,a3,a4,a5,a6,a7] + [b0,b1,b2,b3,b4,b5,b6,b7]
            /// → [a2,b2,a3,b3,a6,b6,a7,b7]
            /// ```
            #[inline(always)]
            pub fn interleave_hi(self, other: Self) -> Self {{
                let a = self.to_array();
                let b = other.to_array();
                Self::from_repr_unchecked(T::from_array([
                    a[2], b[2], a[3], b[3], a[6], b[6], a[7], b[7],
                ]))
            }}

            /// Interleave two vectors: returns `(interleave_lo, interleave_hi)`.
            #[inline(always)]
            pub fn interleave(self, other: Self) -> (Self, Self) {{
                let a = self.to_array();
                let b = other.to_array();
                (
                    Self::from_repr_unchecked(T::from_array([
                        a[0], b[0], a[1], b[1], a[4], b[4], a[5], b[5],
                    ])),
                    Self::from_repr_unchecked(T::from_array([
                        a[2], b[2], a[3], b[3], a[6], b[6], a[7], b[7],
                    ])),
                )
            }}

            // ====== 4-Channel Interleave/Deinterleave ======

            /// Deinterleave 8 RGBA pixels from AoS to SoA format.
            ///
            /// Input: 4 f32x8 vectors, each containing 2 RGBA pixels:
            /// - `rgba[0]` = `[R0, G0, B0, A0, R1, G1, B1, A1]`
            /// - `rgba[1]` = `[R2, G2, B2, A2, R3, G3, B3, A3]`
            /// - `rgba[2]` = `[R4, G4, B4, A4, R5, G5, B5, A5]`
            /// - `rgba[3]` = `[R6, G6, B6, A6, R7, G7, B7, A7]`
            ///
            /// Output: `[R_all, G_all, B_all, A_all]` — one f32x8 per channel.
            #[inline]
            pub fn deinterleave_4ch(rgba: [Self; 4]) -> [Self; 4] {{
                let v: [[f32; 8]; 4] = core::array::from_fn(|i| rgba[i].to_array());
                // Each input vector has 2 RGBA pixels (4 elements each)
                // Pixel i: v[i/2][(i%2)*4 + channel]
                let r: [f32; 8] = core::array::from_fn(|i| v[i / 2][(i % 2) * 4]);
                let g: [f32; 8] = core::array::from_fn(|i| v[i / 2][(i % 2) * 4 + 1]);
                let b: [f32; 8] = core::array::from_fn(|i| v[i / 2][(i % 2) * 4 + 2]);
                let a: [f32; 8] = core::array::from_fn(|i| v[i / 2][(i % 2) * 4 + 3]);
                [
                    Self::from_repr_unchecked(T::from_array(r)),
                    Self::from_repr_unchecked(T::from_array(g)),
                    Self::from_repr_unchecked(T::from_array(b)),
                    Self::from_repr_unchecked(T::from_array(a)),
                ]
            }}

            /// Interleave 4 channels from SoA to AoS format.
            ///
            /// Input: `[R, G, B, A]` — one f32x8 per channel.
            /// Output: 4 f32x8 vectors in interleaved AoS format.
            ///
            /// This is the inverse of `deinterleave_4ch`.
            #[inline]
            pub fn interleave_4ch(channels: [Self; 4]) -> [Self; 4] {{
                let r = channels[0].to_array();
                let g = channels[1].to_array();
                let b = channels[2].to_array();
                let a = channels[3].to_array();
                core::array::from_fn(|i| {{
                    let base = i * 2;
                    Self::from_repr_unchecked(T::from_array([
                        r[base],
                        g[base],
                        b[base],
                        a[base],
                        r[base + 1],
                        g[base + 1],
                        b[base + 1],
                        a[base + 1],
                    ]))
                }})
            }}

            // ====== RGBA Load/Store ======

            /// Load 8 RGBA u8 pixels and deinterleave to 4 f32x8 channel vectors.
            ///
            /// Input: 32 bytes = 8 RGBA pixels in interleaved format.
            /// Output: `(R, G, B, A)` where each is f32x8 with values in `[0.0, 255.0]`.
            #[inline]
            pub fn load_8_rgba_u8(rgba: &[u8; 32]) -> (Self, Self, Self, Self) {{
                let r: [f32; 8] = core::array::from_fn(|i| rgba[i * 4] as f32);
                let g: [f32; 8] = core::array::from_fn(|i| rgba[i * 4 + 1] as f32);
                let b: [f32; 8] = core::array::from_fn(|i| rgba[i * 4 + 2] as f32);
                let a: [f32; 8] = core::array::from_fn(|i| rgba[i * 4 + 3] as f32);
                (
                    Self::from_repr_unchecked(T::from_array(r)),
                    Self::from_repr_unchecked(T::from_array(g)),
                    Self::from_repr_unchecked(T::from_array(b)),
                    Self::from_repr_unchecked(T::from_array(a)),
                )
            }}

            /// Interleave 4 f32x8 channels and store as 8 RGBA u8 pixels.
            ///
            /// Values are rounded and clamped to `[0, 255]`.
            /// Output: 32 bytes = 8 RGBA pixels in interleaved format.
            #[inline]
            pub fn store_8_rgba_u8(r: Self, g: Self, b: Self, a: Self) -> [u8; 32] {{
                let rv = r.to_array();
                let gv = g.to_array();
                let bv = b.to_array();
                let av = a.to_array();
                let mut out = [0u8; 32];
                for i in 0..8 {{
                    out[i * 4] = rv[i].round().clamp(0.0, 255.0) as u8;
                    out[i * 4 + 1] = gv[i].round().clamp(0.0, 255.0) as u8;
                    out[i * 4 + 2] = bv[i].round().clamp(0.0, 255.0) as u8;
                    out[i * 4 + 3] = av[i].round().clamp(0.0, 255.0) as u8;
                }}
                out
            }}

            // ====== Matrix Transpose ======

            /// Transpose an 8x8 matrix represented as 8 row vectors (in-place).
            ///
            /// After transpose, `rows[i][j]` becomes `rows[j][i]`.
            #[inline]
            pub fn transpose_8x8(rows: &mut [Self; 8]) {{
                let r: [[f32; 8]; 8] = core::array::from_fn(|i| rows[i].to_array());
                for i in 0..8 {{
                    rows[i] = Self::from_repr_unchecked(T::from_array(core::array::from_fn(|j| r[j][i])));
                }}
            }}

            /// Transpose an 8x8 matrix, returning the transposed rows.
            #[inline]
            pub fn transpose_8x8_copy(rows: [Self; 8]) -> [Self; 8] {{
                let mut result = rows;
                Self::transpose_8x8(&mut result);
                result
            }}

            /// Load an 8x8 f32 block from a contiguous array into 8 row vectors.
            #[inline]
            pub fn load_8x8(block: &[f32; 64]) -> [Self; 8] {{
                core::array::from_fn(|i| {{
                    let arr: [f32; 8] = block[i * 8..][..8].try_into().unwrap();
                    Self::from_repr_unchecked(T::from_array(arr))
                }})
            }}

            /// Store 8 row vectors to a contiguous 8x8 f32 block.
            #[inline]
            pub fn store_8x8(rows: &[Self; 8], block: &mut [f32; 64]) {{
                for i in 0..8 {{
                    let arr = rows[i].to_array();
                    block[i * 8..][..8].copy_from_slice(&arr);
                }}
            }}
        }}
    "#}
}

/// Generate the cross-type bitcast ref/mut impl block for f32x4/f32x8 -> i32x4/i32x8.
fn gen_f32_bitcast_ref_i32(name: &str, lanes: usize) -> String {
    let int_type = format!("i32x{lanes}");
    let convert_trait = format!("F32x{}Convert", lanes);

    formatdoc! {r#"

        // ============================================================================
        // Cross-type bitcast references (require {convert_trait})
        // ============================================================================

        impl<T: crate::simd::backends::{convert_trait}> {name}<T> {{
            /// Reinterpret bits as `&{int_type}<T>` (zero-cost pointer cast).
            #[inline(always)]
            pub fn bitcast_ref_i32(&self) -> &super::{int_type}<T> {{
                // SAFETY: {name}<T> and {int_type}<T> are both repr(transparent) with same size
                unsafe {{ &*core::ptr::from_ref(self).cast::<super::{int_type}<T>>() }}
            }}

            /// Reinterpret bits as `&mut {int_type}<T>` (zero-cost pointer cast).
            #[inline(always)]
            pub fn bitcast_mut_i32(&mut self) -> &mut super::{int_type}<T> {{
                // SAFETY: {name}<T> and {int_type}<T> are both repr(transparent) with same size
                unsafe {{ &mut *core::ptr::from_mut(self).cast::<super::{int_type}<T>>() }}
            }}
        }}
    "#}
}

/// Generate the cross-type bitcast impl block for u32x4 -> f32x4.
fn gen_u32x4_bitcast_f32() -> String {
    formatdoc! {r#"

        // ============================================================================
        // Cross-type bitcast to f32x4 (requires F32x4Backend)
        // ============================================================================

        impl<T: U32x4Backend + crate::simd::backends::F32x4Backend> u32x4<T> {{
            /// Bitcast to f32x4 (reinterpret bits, no conversion).
            #[inline(always)]
            pub fn bitcast_f32x4(self) -> super::f32x4<T> {{
                // SAFETY: u32x4<T> and f32x4<T> are both exactly 16 bytes
                unsafe {{ core::mem::transmute_copy(&self) }}
            }}

            /// Bitcast to f32x4 by reference (zero-cost pointer cast).
            #[inline(always)]
            pub fn bitcast_ref_f32x4(&self) -> &super::f32x4<T> {{
                // SAFETY: u32x4 and f32x4 are both repr(transparent) with same size (16 bytes)
                unsafe {{ &*core::ptr::from_ref(self).cast::<super::f32x4<T>>() }}
            }}

            /// Bitcast to f32x4 by mutable reference (zero-cost pointer cast).
            #[inline(always)]
            pub fn bitcast_mut_f32x4(&mut self) -> &mut super::f32x4<T> {{
                // SAFETY: u32x4 and f32x4 are both repr(transparent) with same size (16 bytes)
                unsafe {{ &mut *core::ptr::from_mut(self).cast::<super::f32x4<T>>() }}
            }}
        }}
    "#}
}

/// Generate the transcendentals file for a given float/int type pair.
///
/// Called as:
/// ```ignore
/// gen_transcendentals("f32x4", "i32x4", 4, "F32x4Backend", "F32x4Convert", "I32x4Backend")
/// gen_transcendentals("f32x8", "i32x8", 8, "F32x8Backend", "F32x8Convert", "I32x8Backend")
/// ```
fn gen_transcendentals(
    float_type: &str,
    int_type: &str,
    lanes: usize,
    float_backend: &str,
    convert_trait: &str,
    int_backend: &str,
) -> String {
    formatdoc! {r#"
        //! Transcendental math functions for `{float_type}<T>`.
        //!
        //! Generic implementations using IEEE 754 bit manipulation and polynomial
        //! approximation. Available when `T: {convert_trait}` (float↔int conversion).
        //!
        //! Two precision tiers:
        //! - **lowp** (~1% max error): Fast, suitable for perceptual/audio work
        //! - **midp** (~3 ULP): Accurate, suitable for most numerical work
        //!
        //! Variant suffixes:
        //! - `_unchecked`: No edge case handling (fastest, undefined for ≤0/NaN/Inf)
        //! - (normal): Basic edge case handling (0→-inf, negative→NaN for log)
        //! - `_precise`: Full handling including denormals

        use crate::simd::backends::{{{float_backend}, {convert_trait}, {int_backend}}};
        use crate::simd::generic::{{{float_type}, {int_type}}};

        /// Splat an i32 into {int_type} (disambiguates from f32 splat).
        #[inline(always)]
        fn splat_i32<T: {convert_trait}>(v: i32) -> {int_type}<T> {{
            {int_type}::from_repr_unchecked(<T as {int_backend}>::splat(v))
        }}

        /// Splat an f32 into {float_type}.
        #[inline(always)]
        fn splat_f32<T: {convert_trait}>(v: f32) -> {float_type}<T> {{
            {float_type}::from_repr_unchecked(<T as {float_backend}>::splat(v))
        }}

        impl<T: {convert_trait}> {float_type}<T> {{
            // ====== Low-Precision Transcendentals (~1% error) ======

            /// Low-precision base-2 logarithm (~1% max error).
            ///
            /// Uses rational polynomial approximation on the mantissa.
            /// Result is undefined for x <= 0.
            #[inline(always)]
            pub fn log2_lowp(self) -> Self {{
                const P0: f32 = -1.850_383_3e-6;
                const P1: f32 = 1.428_716_1;
                const P2: f32 = 0.742_458_7;
                const Q0: f32 = 0.990_328_14;
                const Q1: f32 = 1.009_671_8;
                const Q2: f32 = 0.174_093_43;

                let x_bits = self.bitcast_to_i32();
                let offset = splat_i32::<T>(0x3f2a_aaab_u32 as i32);
                let exp_bits = x_bits - offset;
                let exp_shifted = exp_bits.shr_arithmetic_const::<23>();
                let mantissa_bits = x_bits - exp_shifted.shl_const::<23>();
                let mantissa = mantissa_bits.bitcast_to_f32();
                let exp_val = exp_shifted.to_f32();

                let m = mantissa - splat_f32::<T>(1.0);

                let yp = splat_f32::<T>(P2).mul_add(m, splat_f32::<T>(P1));
                let yp = yp.mul_add(m, splat_f32::<T>(P0));

                let yq = splat_f32::<T>(Q2).mul_add(m, splat_f32::<T>(Q1));
                let yq = yq.mul_add(m, splat_f32::<T>(Q0));

                yp / yq + exp_val
            }}

            /// Low-precision base-2 logarithm, no edge case handling.
            #[inline(always)]
            pub fn log2_lowp_unchecked(self) -> Self {{
                self.log2_lowp()
            }}

            /// Low-precision base-2 exponential (~1% max error).
            #[inline(always)]
            pub fn exp2_lowp(self) -> Self {{
                const C0: f32 = 1.0;
                const C1: f32 = core::f32::consts::LN_2;
                const C2: f32 = 0.240_226_5;
                const C3: f32 = 0.055_504_11;

                let x = self.max(splat_f32::<T>(-126.0)).min(splat_f32::<T>(126.0));
                let xi = x.floor();
                let xf = x - xi;

                let poly = splat_f32::<T>(C3).mul_add(xf, splat_f32::<T>(C2));
                let poly = poly.mul_add(xf, splat_f32::<T>(C1));
                let poly = poly.mul_add(xf, splat_f32::<T>(C0));

                let xi_i32 = xi.to_i32_round();
                let scale_bits = (xi_i32 + splat_i32::<T>(127)).shl_const::<23>();
                poly * scale_bits.bitcast_to_f32()
            }}

            /// Low-precision base-2 exponential, no edge case handling.
            #[inline(always)]
            pub fn exp2_lowp_unchecked(self) -> Self {{
                self.exp2_lowp()
            }}

            /// Low-precision natural logarithm.
            #[inline(always)]
            pub fn ln_lowp(self) -> Self {{
                self.log2_lowp() * splat_f32::<T>(core::f32::consts::LN_2)
            }}

            /// Low-precision natural logarithm, no edge case handling.
            #[inline(always)]
            pub fn ln_lowp_unchecked(self) -> Self {{
                self.ln_lowp()
            }}

            /// Low-precision natural exponential.
            #[inline(always)]
            pub fn exp_lowp(self) -> Self {{
                (self * splat_f32::<T>(core::f32::consts::LOG2_E)).exp2_lowp()
            }}

            /// Low-precision natural exponential, no edge case handling.
            #[inline(always)]
            pub fn exp_lowp_unchecked(self) -> Self {{
                self.exp_lowp()
            }}

            /// Low-precision base-10 logarithm.
            #[inline(always)]
            pub fn log10_lowp(self) -> Self {{
                self.log2_lowp() * splat_f32::<T>(core::f32::consts::LN_2 / core::f32::consts::LN_10)
            }}

            /// Low-precision base-10 logarithm, no edge case handling.
            #[inline(always)]
            pub fn log10_lowp_unchecked(self) -> Self {{
                self.log10_lowp()
            }}

            /// Low-precision power function: `self^n`.
            #[inline(always)]
            pub fn pow_lowp(self, n: f32) -> Self {{
                (self.log2_lowp() * splat_f32::<T>(n)).exp2_lowp()
            }}

            /// Low-precision power function, no edge case handling.
            #[inline(always)]
            pub fn pow_lowp_unchecked(self, n: f32) -> Self {{
                self.pow_lowp(n)
            }}

            // ====== Mid-Precision Transcendentals (~3 ULP) ======

            /// Mid-precision base-2 logarithm (~3 ULP).
            ///
            /// Uses (a-1)/(a+1) transform with odd polynomial evaluation.
            /// Result is undefined for x <= 0.
            #[inline(always)]
            pub fn log2_midp_unchecked(self) -> Self {{
                const SQRT2_OVER_2: u32 = 0x3f35_04f3;
                const ONE_BITS: u32 = 0x3f80_0000;
                const MANTISSA_MASK: i32 = 0x007f_ffff_u32 as i32;

                const C0: f32 = 2.885_39;
                const C1: f32 = 0.961_800_76;
                const C2: f32 = 0.576_974_45;
                const C3: f32 = 0.434_411_97;

                let x_bits = self.bitcast_to_i32();

                let offset = splat_i32::<T>((ONE_BITS - SQRT2_OVER_2) as i32);
                let adjusted = x_bits + offset;

                let exp_raw = adjusted.shr_arithmetic_const::<23>();
                let n = (exp_raw - splat_i32::<T>(127)).to_f32();

                let mantissa_bits = adjusted & splat_i32::<T>(MANTISSA_MASK);
                let a = (mantissa_bits + splat_i32::<T>(SQRT2_OVER_2 as i32)).bitcast_to_f32();

                let one = splat_f32::<T>(1.0);
                let y = (a - one) / (a + one);
                let y2 = y * y;

                let poly = splat_f32::<T>(C3).mul_add(y2, splat_f32::<T>(C2));
                let poly = poly.mul_add(y2, splat_f32::<T>(C1));
                let poly = poly.mul_add(y2, splat_f32::<T>(C0));

                poly.mul_add(y, n)
            }}

            /// Mid-precision base-2 logarithm with edge case handling.
            ///
            /// Returns -inf for 0, NaN for negative values.
            #[inline(always)]
            pub fn log2_midp(self) -> Self {{
                let result = self.log2_midp_unchecked();
                let zero = splat_f32::<T>(0.0);
                let result = Self::blend(
                    self.simd_eq(zero),
                    splat_f32::<T>(f32::NEG_INFINITY),
                    result,
                );
                Self::blend(self.simd_lt(zero), splat_f32::<T>(f32::NAN), result)
            }}

            /// Mid-precision base-2 logarithm with denormal handling.
            #[inline(always)]
            pub fn log2_midp_precise(self) -> Self {{
                self.log2_midp()
            }}

            /// Mid-precision base-2 exponential (~3 ULP). Undefined for extreme inputs.
            #[inline(always)]
            pub fn exp2_midp_unchecked(self) -> Self {{
                const C0: f32 = 1.0;
                const C1: f32 = core::f32::consts::LN_2;
                const C2: f32 = 0.240_226_46;
                const C3: f32 = 0.055_504_545;
                const C4: f32 = 0.009_618_055;
                const C5: f32 = 0.001_333_37;
                const C6: f32 = 0.000_154_47;

                let xi = self.floor();
                let xf = self - xi;

                let poly = splat_f32::<T>(C6).mul_add(xf, splat_f32::<T>(C5));
                let poly = poly.mul_add(xf, splat_f32::<T>(C4));
                let poly = poly.mul_add(xf, splat_f32::<T>(C3));
                let poly = poly.mul_add(xf, splat_f32::<T>(C2));
                let poly = poly.mul_add(xf, splat_f32::<T>(C1));
                let poly = poly.mul_add(xf, splat_f32::<T>(C0));

                let xi_i32 = xi.to_i32_round();
                let scale_bits = (xi_i32 + splat_i32::<T>(127)).shl_const::<23>();
                poly * scale_bits.bitcast_to_f32()
            }}

            /// Mid-precision base-2 exponential with clamping.
            #[inline(always)]
            pub fn exp2_midp(self) -> Self {{
                self.max(splat_f32::<T>(-126.0))
                    .min(splat_f32::<T>(126.0))
                    .exp2_midp_unchecked()
            }}

            /// Mid-precision base-2 exponential with full edge case handling.
            #[inline(always)]
            pub fn exp2_midp_precise(self) -> Self {{
                self.exp2_midp()
            }}

            /// Mid-precision natural logarithm.
            #[inline(always)]
            pub fn ln_midp(self) -> Self {{
                self.log2_midp() * splat_f32::<T>(core::f32::consts::LN_2)
            }}

            /// Mid-precision natural logarithm, no edge case handling.
            #[inline(always)]
            pub fn ln_midp_unchecked(self) -> Self {{
                self.log2_midp_unchecked() * splat_f32::<T>(core::f32::consts::LN_2)
            }}

            /// Mid-precision natural logarithm with denormal handling.
            #[inline(always)]
            pub fn ln_midp_precise(self) -> Self {{
                self.ln_midp()
            }}

            /// Mid-precision natural exponential.
            #[inline(always)]
            pub fn exp_midp(self) -> Self {{
                (self * splat_f32::<T>(core::f32::consts::LOG2_E)).exp2_midp()
            }}

            /// Mid-precision natural exponential, no edge case handling.
            #[inline(always)]
            pub fn exp_midp_unchecked(self) -> Self {{
                (self * splat_f32::<T>(core::f32::consts::LOG2_E)).exp2_midp_unchecked()
            }}

            /// Mid-precision natural exponential with full edge case handling.
            #[inline(always)]
            pub fn exp_midp_precise(self) -> Self {{
                self.exp_midp()
            }}

            /// Mid-precision base-10 logarithm.
            #[inline(always)]
            pub fn log10_midp(self) -> Self {{
                self.log2_midp() * splat_f32::<T>(core::f32::consts::LN_2 / core::f32::consts::LN_10)
            }}

            /// Mid-precision base-10 logarithm, no edge case handling.
            #[inline(always)]
            pub fn log10_midp_unchecked(self) -> Self {{
                self.log2_midp_unchecked()
                    * splat_f32::<T>(core::f32::consts::LN_2 / core::f32::consts::LN_10)
            }}

            /// Mid-precision base-10 logarithm with denormal handling.
            #[inline(always)]
            pub fn log10_midp_precise(self) -> Self {{
                self.log10_midp()
            }}

            /// Mid-precision power function: `self^n`.
            #[inline(always)]
            pub fn pow_midp(self, n: f32) -> Self {{
                (self.log2_midp() * splat_f32::<T>(n)).exp2_midp()
            }}

            /// Mid-precision power function, no edge case handling.
            #[inline(always)]
            pub fn pow_midp_unchecked(self, n: f32) -> Self {{
                (self.log2_midp_unchecked() * splat_f32::<T>(n)).exp2_midp_unchecked()
            }}

            /// Mid-precision power function with full edge case handling.
            #[inline(always)]
            pub fn pow_midp_precise(self, n: f32) -> Self {{
                self.pow_midp(n)
            }}

            /// Mid-precision cube root.
            ///
            /// Uses Kahan's initial approximation via bit manipulation followed
            /// by 3 Newton-Raphson iterations.
            #[inline(always)]
            pub fn cbrt_midp(self) -> Self {{
                const MAGIC: u32 = 0x2a50_8c2d;
                const TWO_THIRDS: f32 = 0.666_666_6;

                let sign_mask = splat_f32::<T>(-0.0);
                let sign = self & sign_mask;
                let abs_x = self.abs();

                let abs_arr = abs_x.to_array();
                let approx_arr: [f32; {lanes}] =
                    core::array::from_fn(|i| f32::from_bits((abs_arr[i].to_bits() / 3) + MAGIC));
                let mut y = {float_type}::from_repr_unchecked(<T as {float_backend}>::from_array(approx_arr));

                let three = splat_f32::<T>(3.0);
                let two_thirds = splat_f32::<T>(TWO_THIRDS);
                for _ in 0..3 {{
                    let y2 = y * y;
                    let y3 = y2 * y;
                    y *= two_thirds + abs_x / (three * y3);
                }}

                y | sign
            }}

            /// Mid-precision cube root with denormal and zero handling.
            #[inline(always)]
            pub fn cbrt_midp_precise(self) -> Self {{
                let zero = splat_f32::<T>(0.0);
                let is_zero = self.simd_eq(zero);

                let abs_x = self.abs();
                let is_denorm = abs_x.simd_lt(splat_f32::<T>(1.175_494_4e-38));

                let scaled = self * splat_f32::<T>(16_777_216.0);
                let x_for_cbrt = Self::blend(is_denorm, scaled, self);

                let result = x_for_cbrt.cbrt_midp();

                let scaled_result = result * splat_f32::<T>(1.0 / 256.0);
                let result = Self::blend(is_denorm, scaled_result, result);

                Self::blend(is_zero, zero, result)
            }}
        }}
    "#}
}

/// Generate `generic/mod.rs` from the list of all SIMD types.
///
/// Produces the module declarations, transcendentals modules, and pub use
/// re-exports in the exact order matching the existing handwritten file.
fn gen_mod_rs(all_types: &[SimdType]) -> String {
    let mut code = String::new();

    // File header
    code.push_str(&formatdoc! {r#"
        //! Generic SIMD types parameterized by backend token.
        //!
        //! These types are the strategy-pattern wrappers: `f32x8<T>` where `T`
        //! determines the platform implementation. Write one generic function,
        //! get monomorphized per backend at dispatch time.
        //!
        //! # Example
        //!
        //! ```ignore
        //! use magetypes::simd::backends::F32x8Backend;
        //! use magetypes::simd::generic::f32x8;
        //!
        //! fn dot<T: F32x8Backend>(token: T, a: &[f32; 8], b: &[f32; 8]) -> f32 {{
        //!     let va = f32x8::<T>::load(token, a);
        //!     let vb = f32x8::<T>::load(token, b);
        //!     (va * vb).reduce_add()
        //! }}
        //! ```

        #![allow(non_camel_case_types)]

    "#});

    // Collect W128+W256 and W512 types
    let w128_w256: Vec<&SimdType> = all_types
        .iter()
        .filter(|t| t.width != SimdWidth::W512)
        .collect();
    let w512: Vec<&SimdType> = all_types
        .iter()
        .filter(|t| t.width == SimdWidth::W512)
        .collect();

    // Block ops modules (hardcoded list, alphabetical)
    let block_ops_types = [
        "f32x4", "f32x8", "f64x2", "f64x4", "i32x4", "i32x8", "i8x16", "u32x4",
    ];
    for name in &block_ops_types {
        code.push_str(&format!("mod block_ops_{name};\n"));
    }

    // W128+W256 type mod declarations + transcendentals, all sorted alphabetically.
    // The transcendentals modules sort between i8x32_impl and u16x16_impl naturally.
    let mut w128_w256_names: Vec<String> = w128_w256.iter().map(|t| t.name()).collect();
    w128_w256_names.sort();

    let mut all_mods: Vec<String> = w128_w256_names
        .iter()
        .map(|name| format!("{name}_impl"))
        .collect();
    all_mods.push("transcendentals_f32x4".to_string());
    all_mods.push("transcendentals_f32x8".to_string());
    all_mods.sort();

    for mod_name in &all_mods {
        code.push_str(&format!("mod {mod_name};\n"));
    }

    // W128+W256 pub use declarations (specific non-alphabetical order:
    // floats by width, then signed ints by element size, then unsigned ints by element size)
    let pub_use_order = [
        "f32x4", "f32x8", "f64x2", "f64x4", "i8x16", "i8x32", "i16x8", "i16x16", "i32x4", "i32x8",
        "i64x2", "i64x4", "u8x16", "u8x32", "u16x8", "u16x16", "u32x4", "u32x8", "u64x2", "u64x4",
    ];
    code.push('\n');
    for name in &pub_use_order {
        if w128_w256_names.contains(&name.to_string()) {
            code.push_str(&format!("pub use {name}_impl::{name};\n"));
        }
    }

    // 512-bit section
    code.push_str("\n// 512-bit generic wrapper types\n");

    // W512 mod declarations (alphabetical)
    let mut w512_names: Vec<String> = w512.iter().map(|t| t.name()).collect();
    w512_names.sort();
    for name in &w512_names {
        code.push_str(&format!("mod {name}_impl;\n"));
    }

    // W512 pub use declarations (specific order:
    // floats, then signed ints by element size, then unsigned ints by element size)
    let w512_pub_use_order = [
        "f32x16", "f64x8", "i8x64", "i16x32", "i32x16", "i64x8", "u8x64", "u16x32", "u32x16",
        "u64x8",
    ];
    code.push('\n');
    for name in &w512_pub_use_order {
        if w512_names.contains(&name.to_string()) {
            code.push_str(&format!("pub use {name}_impl::{name};\n"));
        }
    }

    code
}

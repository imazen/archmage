//! Width dispatch code generation.
//!
//! Generates `magetypes/src/width.rs` with the `WidthDispatch` trait and
//! a unified blanket implementation for every token supporting all 128/256-bit
//! backend traits.

use crate::registry::Registry;
use indoc::formatdoc;

/// Element type info for width dispatch generation.
struct ElemType {
    /// Short name like "f32", "i8", "u64"
    name: &'static str,
    /// Rust type for scalar values
    scalar: &'static str,
    /// Number of lanes at 128-bit width
    lanes_128: u32,
}

const ELEM_TYPES: &[ElemType] = &[
    ElemType {
        name: "f32",
        scalar: "f32",
        lanes_128: 4,
    },
    ElemType {
        name: "f64",
        scalar: "f64",
        lanes_128: 2,
    },
    ElemType {
        name: "i8",
        scalar: "i8",
        lanes_128: 16,
    },
    ElemType {
        name: "u8",
        scalar: "u8",
        lanes_128: 16,
    },
    ElemType {
        name: "i16",
        scalar: "i16",
        lanes_128: 8,
    },
    ElemType {
        name: "u16",
        scalar: "u16",
        lanes_128: 8,
    },
    ElemType {
        name: "i32",
        scalar: "i32",
        lanes_128: 4,
    },
    ElemType {
        name: "u32",
        scalar: "u32",
        lanes_128: 4,
    },
    ElemType {
        name: "i64",
        scalar: "i64",
        lanes_128: 2,
    },
    ElemType {
        name: "u64",
        scalar: "u64",
        lanes_128: 2,
    },
];

/// Width multipliers.
const WIDTHS: &[u32] = &[128, 256, 512];

/// Get lanes for a given element type at a given width.
fn lanes(elem: &ElemType, width: u32) -> u32 {
    elem.lanes_128 * (width / 128)
}

/// Type name like "f32x8".
fn type_name(elem: &ElemType, width: u32) -> String {
    format!("{}x{}", elem.name, lanes(elem, width))
}

/// Associated type name like "F32x8".
fn assoc_type_name(elem: &ElemType, width: u32) -> String {
    let l = lanes(elem, width);
    format!(
        "{}x{}",
        match elem.name {
            "f32" => "F32",
            "f64" => "F64",
            "i8" => "I8",
            "u8" => "U8",
            "i16" => "I16",
            "u16" => "U16",
            "i32" => "I32",
            "u32" => "U32",
            "i64" => "I64",
            "u64" => "U64",
            _ => unreachable!(),
        },
        l
    )
}

/// Generate the trait definition.
fn generate_trait_def() -> String {
    let mut code = String::with_capacity(4096);

    code.push_str(
        "/// Trait providing access to all SIMD sizes from a capability token.\n\
         ///\n\
         /// Every token implementing this trait can construct vectors of any size.\n\
         /// The associated types determine whether native or polyfilled implementations\n\
         /// are used based on the token's hardware capabilities.\n\
         pub trait WidthDispatch: SimdToken + Copy {\n",
    );

    // Associated types grouped by width
    for &width in WIDTHS {
        code.push_str(&format!("\n    // {width}-bit types\n"));
        for elem in ELEM_TYPES {
            let assoc = assoc_type_name(elem, width);
            code.push_str(&format!("    type {assoc};\n"));
        }
    }

    // Constructor methods grouped by type
    code.push('\n');
    for &width in WIDTHS {
        for elem in ELEM_TYPES {
            let tn = type_name(elem, width);
            let assoc = assoc_type_name(elem, width);
            let scalar = elem.scalar;
            let l = lanes(elem, width);

            code.push_str(&formatdoc! {r#"
                fn {tn}_splat(self, v: {scalar}) -> Self::{assoc};
                fn {tn}_zero(self) -> Self::{assoc};
                fn {tn}_load(self, data: &[{scalar}; {l}]) -> Self::{assoc};
            "#});
        }
    }

    code.push_str("}\n");
    code
}

/// Narrow (128-bit and 256-bit) backend traits required by the blanket
/// `WidthDispatch` implementation.
fn narrow_backend_bounds() -> String {
    let mut bounds = Vec::new();
    for &width in &[128u32, 256u32] {
        for elem in ELEM_TYPES {
            let assoc = assoc_type_name(elem, width);
            bounds.push(format!("crate::simd::backends::{assoc}Backend"));
        }
    }
    bounds.join("\n        + ")
}

/// Generate the single blanket `impl<T> WidthDispatch for T` block.
fn generate_blanket_impl_block() -> String {
    let mut code = String::with_capacity(8192);
    let bounds = narrow_backend_bounds();

    code.push_str(&formatdoc! {r#"
        use crate::simd::generic;

        impl<T> WidthDispatch for T
        where
            T: SimdToken
                + Copy
                + {bounds},
        {{
    "#});

    for &width in WIDTHS {
        code.push_str(&format!("    // {width}-bit types\n"));
        for elem in ELEM_TYPES {
            let assoc = assoc_type_name(elem, width);
            let concrete = if width == 512 {
                let w128 = type_name(elem, 128);
                format!("[generic::{w128}<T>; 4]")
            } else {
                let tn = type_name(elem, width);
                format!("generic::{tn}<T>")
            };
            code.push_str(&format!("    type {assoc} = {concrete};\n"));
        }
        code.push('\n');
    }

    for &width in WIDTHS {
        for elem in ELEM_TYPES {
            let tn = type_name(elem, width);
            let assoc = assoc_type_name(elem, width);
            let scalar = elem.scalar;
            let l = lanes(elem, width);

            let (splat_body, zero_body, load_body) = if width == 512 {
                let w128 = type_name(elem, 128);
                let w128_lanes = lanes(elem, 128);
                let mut parts = Vec::with_capacity(4);
                for i in 0..4 {
                    let start = i * w128_lanes;
                    let end = start + w128_lanes;
                    parts.push(format!(
                        "generic::{w128}::load_t(self, data[{start}..{end}].try_into().unwrap())"
                    ));
                }
                let joined = parts.join(",\n                ");
                (
                    formatdoc! {r#"
                        {{
                                    let part = generic::{w128}::splat_t(self, v);
                                    [part, part, part, part]
                                }}"#
                    },
                    formatdoc! {r#"
                        {{
                                    let part = generic::{w128}::zero_t(self);
                                    [part, part, part, part]
                                }}"#
                    },
                    formatdoc! {r#"
                        {{
                                    [
                                        {joined}
                                    ]
                                }}"#
                    },
                )
            } else {
                (
                    format!("generic::{tn}::splat_t(self, v)"),
                    format!("generic::{tn}::zero_t(self)"),
                    format!("generic::{tn}::load_t(self, data)"),
                )
            };

            code.push_str(&formatdoc! {r#"
                    #[inline(always)]
                    fn {tn}_splat(self, v: {scalar}) -> Self::{assoc} {{
                        {splat_body}
                    }}

                    #[inline(always)]
                    fn {tn}_zero(self) -> Self::{assoc} {{
                        {zero_body}
                    }}

                    #[inline(always)]
                    fn {tn}_load(self, data: &[{scalar}; {l}]) -> Self::{assoc} {{
                        {load_body}
                    }}
            "#});
        }
    }

    code.push_str("}\n");
    code
}

/// Generate the complete width.rs file.
pub fn generate_width_dispatch(_reg: &Registry) -> String {
    let mut code = String::with_capacity(16384);

    code.push_str(
        r#"//! Width dispatch trait for token-based SIMD type construction.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.
//!
//! The `WidthDispatch` trait provides access to ALL SIMD sizes from any token.
//! Native types are used where the hardware supports them; polyfills are used
//! for wider types on narrower hardware.

#![allow(missing_docs)]

use archmage::SimdToken;

"#,
    );

    code.push_str(&generate_trait_def());
    code.push('\n');
    code.push_str(&generate_blanket_impl_block());
    code
}

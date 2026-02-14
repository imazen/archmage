//! Width dispatch code generation.
//!
//! Generates `magetypes/src/width.rs` with the `WidthDispatch` trait and
//! implementations for X64V3Token, NeonToken, and Wasm128Token.

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

/// What kind of constructor strategy to use.
enum Strategy {
    /// Native width or narrower: pass self directly. `type_name::splat(self, v)`
    /// Works because wider tokens downcast to narrower tokens' required type.
    Native,
    /// Polyfill exists: `poly_mod::type_name::splat(self, v)`
    Polyfill { poly_mod: &'static str },
    /// No polyfill: array of w128. `[part, part, part, part]`
    Array { w128_type: String, chunks: u32 },
}

/// Token implementation config.
struct TokenConfig {
    token: &'static str,
    cfg: &'static str,
    mod_name: &'static str,
    native_width: u32,
    /// Polyfill module for w256 (when native is 128)
    poly_w256: Option<&'static str>,
    /// Polyfill module for w512
    poly_w512: Option<&'static str>,
}

const TOKEN_CONFIGS: &[TokenConfig] = &[
    TokenConfig {
        token: "X64V3Token",
        cfg: "target_arch = \"x86_64\"",
        mod_name: "x86_impl",
        native_width: 256,
        poly_w256: None, // native
        poly_w512: Some("crate::simd::polyfill::v3_512"),
    },
    TokenConfig {
        token: "NeonToken",
        cfg: "target_arch = \"aarch64\"",
        mod_name: "arm_impl",
        native_width: 128,
        poly_w256: Some("crate::simd::polyfill::neon"),
        poly_w512: None, // array fallback
    },
    TokenConfig {
        token: "Wasm128Token",
        cfg: "target_arch = \"wasm32\"",
        mod_name: "wasm_impl",
        native_width: 128,
        poly_w256: Some("crate::simd::polyfill::wasm128"),
        poly_w512: None, // array fallback
    },
];

fn strategy(tc: &TokenConfig, elem: &ElemType, width: u32) -> Strategy {
    match width {
        128 => {
            // Native or narrower than native: pass self directly.
            // e.g. X64V3Token (native 256) can be passed to w128 types
            // that accept X64V3Token — no forging needed.
            Strategy::Native
        }
        256 => {
            if tc.native_width >= 256 {
                Strategy::Native
            } else if let Some(poly) = tc.poly_w256 {
                Strategy::Polyfill { poly_mod: poly }
            } else {
                unreachable!("no w256 strategy for {}", tc.token);
            }
        }
        512 => {
            if let Some(poly) = tc.poly_w512 {
                Strategy::Polyfill { poly_mod: poly }
            } else {
                // Array of w128 parts
                let chunks = 512 / 128;
                Strategy::Array {
                    w128_type: type_name(elem, 128),
                    chunks,
                }
            }
        }
        _ => unreachable!(),
    }
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

/// Generate a constructor body based on strategy.
fn gen_splat(strat: &Strategy, _tc: &TokenConfig, tn: &str, scalar: &str) -> String {
    match strat {
        Strategy::Native => format!("{tn}::splat(self, v)"),
        Strategy::Polyfill { poly_mod } => format!("{poly_mod}::{tn}::splat(self, v)"),
        Strategy::Array { w128_type, .. } => {
            let _ = scalar;
            formatdoc! {r#"
                {{
                            let part = {w128_type}::splat(self, v);
                            [part, part, part, part]
                        }}"#
            }
        }
    }
}

fn gen_zero(strat: &Strategy, _tc: &TokenConfig, tn: &str) -> String {
    match strat {
        Strategy::Native => format!("{tn}::zero(self)"),
        Strategy::Polyfill { poly_mod } => format!("{poly_mod}::{tn}::zero(self)"),
        Strategy::Array { w128_type, .. } => {
            formatdoc! {r#"
                {{
                            let part = {w128_type}::zero(self);
                            [part, part, part, part]
                        }}"#
            }
        }
    }
}

fn gen_load(strat: &Strategy, _tc: &TokenConfig, tn: &str, elem: &ElemType, width: u32) -> String {
    match strat {
        Strategy::Native => format!("{tn}::load(self, data)"),
        Strategy::Polyfill { poly_mod } => format!("{poly_mod}::{tn}::load(self, data)"),
        Strategy::Array { w128_type, chunks } => {
            let w128_lanes = lanes(elem, 128);
            let mut parts = String::new();
            for i in 0..*chunks {
                let start = i * w128_lanes;
                let end = start + w128_lanes;
                if i > 0 {
                    parts.push_str(",\n                    ");
                }
                parts.push_str(&format!(
                    "{w128_type}::load(self, data[{start}..{end}].try_into().unwrap())"
                ));
            }
            let _ = width;
            formatdoc! {r#"
                {{
                            [
                                {parts}
                            ]
                        }}"#
            }
        }
    }
}

/// Generate the associated type for a given token config + element + width.
fn gen_assoc_type(tc: &TokenConfig, elem: &ElemType, width: u32) -> String {
    let strat = strategy(tc, elem, width);
    let tn = type_name(elem, width);
    match strat {
        Strategy::Native => tn,
        Strategy::Polyfill { poly_mod } => format!("{poly_mod}::{tn}"),
        Strategy::Array { w128_type, chunks } => {
            format!("[{w128_type}; {chunks}]")
        }
    }
}

/// Generate a full impl block for one token config.
fn generate_impl_block(tc: &TokenConfig) -> String {
    let mut code = String::with_capacity(8192);
    let token = tc.token;
    let cfg = tc.cfg;
    let mod_name = tc.mod_name;

    // Module wrapper with cfg + imports
    code.push_str(&format!(
        "#[cfg({cfg})]\nmod {mod_name} {{\n    use super::WidthDispatch;\n    use archmage::{token};\n"
    ));

    // Collect all native imports needed
    let mut native_imports: Vec<String> = Vec::new();

    for &width in WIDTHS {
        for elem in ELEM_TYPES {
            let strat = strategy(tc, elem, width);
            let tn = type_name(elem, width);
            match &strat {
                Strategy::Native => {
                    if !native_imports.contains(&tn) {
                        native_imports.push(tn);
                    }
                }
                Strategy::Polyfill { .. } => {
                    // Polyfill types use full path, no import needed
                }
                Strategy::Array { w128_type, .. } => {
                    if !native_imports.contains(w128_type) {
                        native_imports.push(w128_type.clone());
                    }
                }
            }
        }
    }

    // Write native type imports
    if !native_imports.is_empty() {
        native_imports.sort();
        let imports = native_imports.join(", ");
        code.push_str(&format!("\n    use crate::simd::{{{imports}}};\n"));
    }

    // Start impl block
    code.push_str(&format!("\n    impl WidthDispatch for {token} {{\n"));

    // Associated types
    for &width in WIDTHS {
        code.push_str(&format!("        // {width}-bit types\n"));
        for elem in ELEM_TYPES {
            let assoc = assoc_type_name(elem, width);
            let concrete = gen_assoc_type(tc, elem, width);
            code.push_str(&format!("        type {assoc} = {concrete};\n"));
        }
        code.push('\n');
    }

    // Constructor methods
    for &width in WIDTHS {
        for elem in ELEM_TYPES {
            let tn = type_name(elem, width);
            let assoc = assoc_type_name(elem, width);
            let scalar = elem.scalar;
            let l = lanes(elem, width);
            let strat = strategy(tc, elem, width);

            let splat_body = gen_splat(&strat, tc, &tn, scalar);
            let zero_body = gen_zero(&strat, tc, &tn);
            let load_body = gen_load(&strat, tc, &tn, elem, width);

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

    // Close impl + module
    code.push_str("    }\n}\n");
    code
}

/// Generate the complete width.rs file.
pub fn generate_width_dispatch(_reg: &Registry) -> String {
    let mut code = String::with_capacity(32768);

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

    // Trait definition
    code.push_str(&generate_trait_def());
    code.push('\n');

    // Impl blocks for each token
    for tc in TOKEN_CONFIGS {
        code.push_str(&generate_impl_block(tc));
        code.push('\n');
    }

    code
}

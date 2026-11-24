//! Width dispatch code generation.
//!
//! Generates `magetypes/src/width.rs` with the `WidthDispatch` trait and
//! implementations for X64V3Token, NeonToken, and Wasm128Token.

use indoc::formatdoc;

use super::types::{ElementType, SimdType, SimdWidth};

const ELEMS: &[ElementType] = &[
    ElementType::F32,
    ElementType::F64,
    ElementType::I8,
    ElementType::U8,
    ElementType::I16,
    ElementType::U16,
    ElementType::I32,
    ElementType::U32,
    ElementType::I64,
    ElementType::U64,
];

const WIDTHS: &[SimdWidth] = &[SimdWidth::W128, SimdWidth::W256, SimdWidth::W512];

/// Associated type name like "F32x8".
fn assoc_type_name(ty: SimdType) -> String {
    format!("{}x{}", ty.elem.upper_name(), ty.lanes())
}

/// What kind of constructor strategy to use.
enum Strategy {
    /// Native width or narrower: pass self directly (`type_name::splat_t(self, v)`).
    Native,
    /// Polyfill exists: `poly_mod::type_name::splat_t(self, v)`.
    Polyfill { poly_mod: &'static str },
    /// Array of 4×W128 parts (`[part, part, part, part]`).
    Array { w128_type: String },
}

/// Token implementation config.
struct TokenConfig {
    token: &'static str,
    cfg: &'static str,
    mod_name: &'static str,
    native_width: SimdWidth,
    /// Polyfill module for W256 (when native is W128)
    poly_w256: Option<&'static str>,
}

const TOKEN_CONFIGS: &[TokenConfig] = &[
    TokenConfig {
        token: "X64V3Token",
        cfg: "all(target_arch = \"x86_64\", target_feature = \"sse2\")",
        mod_name: "x86_impl",
        native_width: SimdWidth::W256,
        poly_w256: None,
    },
    TokenConfig {
        token: "NeonToken",
        cfg: "target_arch = \"aarch64\"",
        mod_name: "arm_impl",
        native_width: SimdWidth::W128,
        poly_w256: Some("crate::simd::neon"),
    },
    TokenConfig {
        token: "Wasm128Token",
        cfg: "target_arch = \"wasm32\"",
        mod_name: "wasm_impl",
        native_width: SimdWidth::W128,
        poly_w256: Some("crate::simd::wasm128"),
    },
];

fn strategy(tc: &TokenConfig, ty: SimdType) -> Strategy {
    match ty.width {
        SimdWidth::W128 => Strategy::Native,
        SimdWidth::W256 => {
            if tc.native_width.bits() >= 256 {
                Strategy::Native
            } else {
                Strategy::Polyfill {
                    poly_mod: tc
                        .poly_w256
                        .unwrap_or_else(|| unreachable!("no w256 strategy for {}", tc.token)),
                }
            }
        }
        SimdWidth::W512 => Strategy::Array {
            w128_type: SimdType::new(ty.elem, SimdWidth::W128).name(),
        },
    }
}

/// Generate the trait definition.
fn generate_trait_def() -> String {
    let mut code = String::with_capacity(4096);

    code.push_str(
        "/// **Superseded (queued for deprecation):** Use `#[attune(_*_t, define(...))]` or backend-generic bounds (such as [`F32x8Backend`](crate::simd::backends::F32x8Backend) on [`f32x8<T>`](crate::simd::generic::f32x8)) instead.\n\
         ///\n\
         /// Trait providing access to all SIMD sizes from a capability token.\n\
         ///\n\
         /// Every token implementing this trait can construct vectors of any size.\n\
         /// The associated types determine whether native or polyfilled implementations\n\
         /// are used based on the token's hardware capabilities.\n\
         pub trait WidthDispatch: SimdToken + Copy {\n",
    );

    for &width in WIDTHS {
        code.push_str(&format!("\n    // {}-bit types\n", width.bits()));
        for &elem in ELEMS {
            let assoc = assoc_type_name(SimdType::new(elem, width));
            code.push_str(&format!("    type {assoc};\n"));
        }
    }

    code.push('\n');
    for &width in WIDTHS {
        for &elem in ELEMS {
            let ty = SimdType::new(elem, width);
            let tn = ty.name();
            let assoc = assoc_type_name(ty);
            let scalar = elem.name();
            let l = ty.lanes();

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

fn gen_ctor(strat: &Strategy, tn: &str, ctor: &str, args: &str) -> String {
    match strat {
        Strategy::Native => format!("{tn}::{ctor}({args})"),
        Strategy::Polyfill { poly_mod } => format!("{poly_mod}::{tn}::{ctor}({args})"),
        Strategy::Array { w128_type } => formatdoc! {r#"
            {{
                        let part = {w128_type}::{ctor}({args});
                        [part, part, part, part]
                    }}"#
        },
    }
}

fn gen_load(strat: &Strategy, tn: &str, elem: ElementType) -> String {
    match strat {
        Strategy::Native => format!("{tn}::load_t(self, data)"),
        Strategy::Polyfill { poly_mod } => format!("{poly_mod}::{tn}::load_t(self, data)"),
        Strategy::Array { w128_type } => {
            let w128_lanes = SimdType::new(elem, SimdWidth::W128).lanes();
            let parts = (0..4)
                .map(|i| {
                    let start = i * w128_lanes;
                    let end = start + w128_lanes;
                    format!("{w128_type}::load_t(self, data[{start}..{end}].try_into().unwrap())")
                })
                .collect::<Vec<_>>()
                .join(",\n                    ");
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

fn generate_impl_block(tc: &TokenConfig) -> String {
    let mut code = String::with_capacity(8192);
    let token = tc.token;
    let cfg = tc.cfg;
    let mod_name = tc.mod_name;

    code.push_str(&format!(
        "#[cfg({cfg})]\nmod {mod_name} {{\n    use super::WidthDispatch;\n    use archmage::{token};\n"
    ));

    let mut native_imports: Vec<String> = Vec::new();
    for &width in WIDTHS {
        for &elem in ELEMS {
            let ty = SimdType::new(elem, width);
            let import = match strategy(tc, ty) {
                Strategy::Native => Some(ty.name()),
                Strategy::Polyfill { .. } => None,
                Strategy::Array { w128_type } => Some(w128_type),
            };
            if let Some(name) = import
                && !native_imports.contains(&name)
            {
                native_imports.push(name);
            }
        }
    }

    if !native_imports.is_empty() {
        native_imports.sort();
        code.push_str(&format!(
            "\n    use crate::simd::{{{}}};\n",
            native_imports.join(", ")
        ));
    }

    code.push_str(&format!("\n    impl WidthDispatch for {token} {{\n"));

    for &width in WIDTHS {
        code.push_str(&format!("        // {}-bit types\n", width.bits()));
        for &elem in ELEMS {
            let ty = SimdType::new(elem, width);
            let assoc = assoc_type_name(ty);
            let concrete = match strategy(tc, ty) {
                Strategy::Native => ty.name(),
                Strategy::Polyfill { poly_mod } => format!("{poly_mod}::{}", ty.name()),
                Strategy::Array { w128_type } => format!("[{w128_type}; 4]"),
            };
            code.push_str(&format!("        type {assoc} = {concrete};\n"));
        }
        code.push('\n');
    }

    for &width in WIDTHS {
        for &elem in ELEMS {
            let ty = SimdType::new(elem, width);
            let tn = ty.name();
            let assoc = assoc_type_name(ty);
            let scalar = elem.name();
            let l = ty.lanes();
            let strat = strategy(tc, ty);

            let splat_body = gen_ctor(&strat, &tn, "splat_t", "self, v");
            let zero_body = gen_ctor(&strat, &tn, "zero_t", "self");
            let load_body = gen_load(&strat, &tn, elem);

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

    code.push_str("    }\n}\n");
    code
}

/// Generate the complete `magetypes/src/width.rs` file.
pub fn generate_width_dispatch() -> String {
    let mut code = String::with_capacity(32768);

    code.push_str(
        r#"//! **Superseded (queued for deprecation):** Use `#[attune(_*_t, define(...))]` or backend-generic bounds (such as [`F32x8Backend`](crate::simd::backends::F32x8Backend) on [`f32x8<T>`](crate::simd::generic::f32x8)) instead.
//!
//! Width dispatch trait for token-based SIMD type construction.
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

    for tc in TOKEN_CONFIGS {
        code.push_str(&generate_impl_block(tc));
        code.push('\n');
    }

    code
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_strategies_cover_native_polyfilled_and_four_way_layouts() {
        let generated = generate_width_dispatch();
        syn::parse_file(&generated).expect("valid Rust width-dispatch implementations");

        assert!(generated.contains("i16x16::load_t(self, data)"));
        assert!(generated.contains("crate::simd::neon::i16x16::load_t(self, data)"));
        assert!(generated.contains("crate::simd::wasm128::i16x16::load_t(self, data)"));
        assert_eq!(generated.matches("type U8x64 = [u8x16; 4];").count(), 3);
        for range in ["0..16", "16..32", "32..48", "48..64"] {
            let load = format!("u8x16::load_t(self, data[{range}].try_into().unwrap())");
            assert_eq!(generated.matches(&load).count(), 3, "u8x64 chunk {range}");
        }
    }
}

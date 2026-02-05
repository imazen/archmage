//! SIMD type generation for wide-like ergonomic types.
//!
//! Generates token-gated SIMD types with:
//! - Safe construction (requires capability token)
//! - Operator overloads (+, -, *, /, &, |, ^, etc.)
//! - Math methods (min, max, abs, sqrt, fma, etc.)
//! - Integration with archmage::mem for load/store
//!
//! ## Module Structure
//!
//! - `types` - Core types (ElementType, SimdWidth, SimdType) and code generation helpers
//! - `arch` - Architecture-specific helpers (x86, arm)
//! - `structure` - Type definition and macro generation
//! - `ops` - Standard SIMD operations (comparison, blend, math, etc.)
//! - `transcendental` - Transcendental functions (log, exp, pow, cbrt)

pub mod arch;
mod block_ops;
pub mod block_ops_arm;
pub mod block_ops_wasm;
pub mod extend_ops_arm;
pub mod extend_ops_wasm;
mod ops;
pub mod ops_bitcast;
mod ops_comparison;
pub mod parity_tests;
mod structure;
mod structure_arm;
pub mod structure_polyfill;
mod structure_wasm;
mod transcendental;
pub mod transcendental_arm;
pub mod transcendental_wasm;
pub mod types;
pub mod width_dispatch;

pub use types::{SimdType, SimdWidth, all_simd_types};

use std::collections::BTreeMap;

/// File header with common imports and attributes
fn file_header() -> &'static str {
    r#"//! Token-gated SIMD types with natural operators.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

#![allow(dead_code)]
#![allow(non_camel_case_types)]
#![allow(missing_docs)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::approx_constant)]
#![allow(clippy::missing_transmute_annotations)]
#![allow(clippy::manual_is_multiple_of)]

"#
}

/// Generate the main mod.rs in generated/ with all the real code
fn generate_generated_mod_rs(types: &[SimdType]) -> String {
    let mut code = String::from(file_header());

    // Note: core::ops imports are in submodules where types are defined
    // Macros are #[macro_export] and expand in those submodules

    // Generate comparison traits
    code.push_str(&structure::generate_comparison_traits());

    // Generate macros
    code.push_str(&structure::generate_macros());

    // Module declarations and re-exports grouped by width
    code.push_str(
        "// ============================================================================\n",
    );
    code.push_str("// Type modules\n");
    code.push_str(
        "// ============================================================================\n\n",
    );

    // Group types by width for organized output
    let mut w128_types = Vec::new();
    let mut w256_types = Vec::new();
    let mut w512_types = Vec::new();

    for ty in types {
        let name = ty.name();
        match ty.width {
            SimdWidth::W128 => w128_types.push(name),
            SimdWidth::W256 => w256_types.push(name),
            SimdWidth::W512 => w512_types.push(name),
        }
    }

    // x86 module (SSE, AVX, AVX-512) - public for parity tests
    code.push_str("// x86-64 types (SSE, AVX, AVX-512)\n");
    code.push_str("#[cfg(target_arch = \"x86_64\")]\n");
    code.push_str("pub mod x86 {\n");
    code.push_str("    pub mod w128;\n");
    code.push_str("    pub mod w256;\n");
    code.push_str("    #[cfg(feature = \"avx512\")]\n");
    code.push_str("    pub mod w512;\n");
    code.push_str("}\n\n");

    // ARM module (NEON) - public for parity tests
    code.push_str("// AArch64 types (NEON)\n");
    code.push_str("#[cfg(target_arch = \"aarch64\")]\n");
    code.push_str("pub mod arm {\n");
    code.push_str("    pub mod w128;\n");
    code.push_str("}\n\n");

    // WASM module (SIMD128) - public for parity tests
    code.push_str("// WebAssembly types (SIMD128)\n");
    code.push_str("#[cfg(target_arch = \"wasm32\")]\n");
    code.push_str("pub mod wasm {\n");
    code.push_str("    pub mod w128;\n");
    code.push_str("}\n\n");

    // Re-exports for x86
    code.push_str("// Re-export all types\n");
    code.push_str("#[cfg(target_arch = \"x86_64\")]\n");
    code.push_str("pub use x86::w128::*;\n");
    code.push_str("#[cfg(target_arch = \"x86_64\")]\n");
    code.push_str("pub use x86::w256::*;\n");
    code.push_str("#[cfg(all(target_arch = \"x86_64\", feature = \"avx512\"))]\n");
    code.push_str("pub use x86::w512::*;\n\n");

    // Re-exports for ARM
    code.push_str("#[cfg(target_arch = \"aarch64\")]\n");
    code.push_str("pub use arm::w128::*;\n");
    code.push_str("#[cfg(target_arch = \"aarch64\")]\n");
    code.push_str("pub use polyfill::neon::*;\n\n");

    // Re-exports for WASM
    code.push_str("#[cfg(target_arch = \"wasm32\")]\n");
    code.push_str("pub use wasm::w128::*;\n");
    code.push_str("#[cfg(target_arch = \"wasm32\")]\n");
    code.push_str("pub use polyfill::wasm128::*;\n\n");

    // Polyfill module (auto-generated W256 from pairs of W128)
    code.push_str("// Polyfill module for emulating wider types on narrower hardware\n");
    code.push_str("pub mod polyfill;\n\n");

    // Generate width-aliased namespaces for multi-width dispatch
    code.push_str(&generate_width_namespaces(types));

    // Generate NEON namespace
    code.push_str(&generate_neon_namespace());

    // Generate WASM SIMD128 namespace
    code.push_str(&generate_simd128_namespace());

    code
}

/// Generate width-aliased namespace modules for multi-width dispatch
///
/// Creates modules like `v3`, `v4` that export ALL types available for that
/// token level: native types at natural width, plus narrower native types
/// and wider polyfilled types. The `xN` aliases point to the natural width.
fn generate_width_namespaces(types: &[SimdType]) -> String {
    let mut code = String::new();

    // Collect type names per width for selective re-exports
    let w128_types: Vec<String> = types
        .iter()
        .filter(|t| t.width == SimdWidth::W128)
        .map(|t| t.name())
        .collect();
    let w512_types: Vec<String> = types
        .iter()
        .filter(|t| t.width == SimdWidth::W512)
        .map(|t| t.name())
        .collect();

    code.push_str(
        "// ============================================================================\n",
    );
    code.push_str("// Per-token namespaces (all available widths per token level)\n");
    code.push_str("//\n");
    code.push_str("// Each module re-exports ALL types usable with that token:\n");
    code.push_str("//   - Native types at the token's natural width\n");
    code.push_str("//   - Narrower native types (same token or use .v3()/.v2() to downcast)\n");
    code.push_str("//   - Wider polyfilled types built from the natural width\n");
    code.push_str("//\n");
    code.push_str("// The `xN` aliases and `LANES_*` refer to the natural width.\n");
    code.push_str(
        "// ============================================================================\n\n",
    );

    // ── V3 namespace ──
    // Natural width: 256-bit (AVX2). Also includes:
    // - w128 native types (same X64V3Token)
    // - v3_512 polyfilled 512-bit types (same X64V3Token)
    code.push_str("#[cfg(target_arch = \"x86_64\")]\n");
    code.push_str("pub mod v3 {\n");
    code.push_str("    //! All SIMD types available with `X64V3Token`.\n");
    code.push_str("    //!\n");
    code.push_str("    //! Natural width: 256-bit (AVX2+FMA). `f32xN` = `f32x8`.\n");
    code.push_str("    //!\n");
    code.push_str("    //! Also includes 128-bit native types and 512-bit polyfills\n");
    code.push_str("    //! (emulated via 2×256-bit ops). All take `X64V3Token`.\n\n");

    // xN aliases (natural width = 256-bit)
    code.push_str("    pub use super::x86::w256::{\n");
    for ty in types.iter().filter(|t| t.width == SimdWidth::W256) {
        let name = ty.name();
        code.push_str(&format!("        {name} as {}xN,\n", ty.elem.name()));
    }
    code.push_str("    };\n\n");

    // Native 256-bit (natural width)
    code.push_str("    pub use super::x86::w256::*;\n\n");

    // Native 128-bit (narrower, same token)
    code.push_str("    // 128-bit native types (same X64V3Token)\n");
    code.push_str("    pub use super::x86::w128::{\n");
    code.push_str(&format!("        {},\n", w128_types.join(", ")));
    code.push_str("    };\n\n");

    // Polyfilled 512-bit (wider, same token)
    code.push_str("    // 512-bit polyfilled types (2×256-bit, same X64V3Token)\n");
    code.push_str("    pub use super::polyfill::v3_512::{\n");
    code.push_str(&format!("        {},\n", w512_types.join(", ")));
    code.push_str("    };\n\n");

    code.push_str("    /// Token type for this width level\n");
    code.push_str("    pub type Token = archmage::X64V3Token;\n\n");
    code.push_str("    pub const LANES_F32: usize = 8;\n");
    code.push_str("    pub const LANES_F64: usize = 4;\n");
    code.push_str("    pub const LANES_32: usize = 8;\n");
    code.push_str("    pub const LANES_16: usize = 16;\n");
    code.push_str("    pub const LANES_8: usize = 32;\n");
    code.push_str("}\n\n");

    // ── V4 namespace ──
    // Natural width: 512-bit (AVX-512). Also includes:
    // - w128 native types (take X64V3Token; use token.v3() to downcast)
    // - w256 native types (take X64V3Token; use token.v3() to downcast)
    code.push_str("#[cfg(all(target_arch = \"x86_64\", feature = \"avx512\"))]\n");
    code.push_str("pub mod v4 {\n");
    code.push_str("    //! All SIMD types available with `X64V4Token`.\n");
    code.push_str("    //!\n");
    code.push_str("    //! Natural width: 512-bit (AVX-512). `f32xN` = `f32x16`.\n");
    code.push_str("    //!\n");
    code.push_str("    //! Also includes 128-bit and 256-bit native types. These accept\n");
    code.push_str("    //! `X64V3Token` — use `token.v3()` to downcast your `X64V4Token`.\n\n");

    // xN aliases (natural width = 512-bit)
    code.push_str("    pub use super::x86::w512::{\n");
    for ty in types.iter().filter(|t| t.width == SimdWidth::W512) {
        let name = ty.name();
        code.push_str(&format!("        {name} as {}xN,\n", ty.elem.name()));
    }
    code.push_str("    };\n\n");

    // Native 512-bit (natural width)
    code.push_str("    pub use super::x86::w512::*;\n\n");

    // Native 128-bit (narrower, needs token.v3())
    code.push_str("    // 128-bit native types (use token.v3() to downcast)\n");
    code.push_str("    pub use super::x86::w128::{\n");
    code.push_str(&format!("        {},\n", w128_types.join(", ")));
    code.push_str("    };\n\n");

    // Native 256-bit (narrower, needs token.v3())
    code.push_str("    // 256-bit native types (use token.v3() to downcast)\n");
    code.push_str("    pub use super::x86::w256::{\n");
    let w256_types: Vec<String> = types
        .iter()
        .filter(|t| t.width == SimdWidth::W256)
        .map(|t| t.name())
        .collect();
    code.push_str(&format!("        {},\n", w256_types.join(", ")));
    code.push_str("    };\n\n");

    code.push_str("    /// Token type for this width level\n");
    code.push_str("    pub type Token = archmage::X64V4Token;\n\n");
    code.push_str("    pub const LANES_F32: usize = 16;\n");
    code.push_str("    pub const LANES_F64: usize = 8;\n");
    code.push_str("    pub const LANES_32: usize = 16;\n");
    code.push_str("    pub const LANES_16: usize = 32;\n");
    code.push_str("    pub const LANES_8: usize = 64;\n");
    code.push_str("}\n");

    code
}

/// Generate NEON width namespace for ARM
fn generate_neon_namespace() -> String {
    let mut code = String::new();

    // Polyfill type names (256-bit emulated via 2×128-bit NEON)
    let polyfill_types = "f32x8, f64x4, i8x32, u8x32, i16x16, u16x16, i32x8, u32x8, i64x4, u64x4";

    code.push_str("\n#[cfg(target_arch = \"aarch64\")]\n");
    code.push_str("pub mod neon {\n");
    code.push_str("    //! All SIMD types available with `NeonToken`.\n");
    code.push_str("    //!\n");
    code.push_str("    //! Natural width: 128-bit (NEON). `f32xN` = `f32x4`.\n");
    code.push_str("    //!\n");
    code.push_str("    //! Also includes 256-bit polyfills (emulated via 2×128-bit NEON ops).\n");
    code.push_str("    //! All take `NeonToken`.\n\n");

    // xN aliases (natural width = 128-bit)
    code.push_str("    pub use super::arm::w128::{\n");
    code.push_str("        f32x4 as f32xN, f64x2 as f64xN, i8x16 as i8xN, i16x8 as i16xN,\n");
    code.push_str("        i32x4 as i32xN, i64x2 as i64xN, u8x16 as u8xN, u16x8 as u16xN,\n");
    code.push_str("        u32x4 as u32xN, u64x2 as u64xN,\n");
    code.push_str("    };\n\n");

    // Native 128-bit (natural width)
    code.push_str("    pub use super::arm::w128::*;\n\n");

    // Polyfilled 256-bit (wider, same token)
    code.push_str("    // 256-bit polyfilled types (2×128-bit NEON, same NeonToken)\n");
    code.push_str(&format!("    pub use super::polyfill::neon::{{{polyfill_types}}};\n\n"));

    code.push_str("    /// Token type for this width level\n");
    code.push_str("    pub type Token = archmage::NeonToken;\n\n");
    code.push_str("    pub const LANES_F32: usize = 4;\n");
    code.push_str("    pub const LANES_F64: usize = 2;\n");
    code.push_str("    pub const LANES_32: usize = 4;\n");
    code.push_str("    pub const LANES_16: usize = 8;\n");
    code.push_str("    pub const LANES_8: usize = 16;\n");
    code.push_str("}\n");

    code
}

/// Generate WASM SIMD128 width namespace
fn generate_simd128_namespace() -> String {
    let mut code = String::new();

    // Polyfill type names (256-bit emulated via 2×128-bit WASM SIMD)
    let polyfill_types = "f32x8, f64x4, i8x32, u8x32, i16x16, u16x16, i32x8, u32x8, i64x4, u64x4";

    code.push_str("\n#[cfg(target_arch = \"wasm32\")]\n");
    code.push_str("pub mod wasm128 {\n");
    code.push_str("    //! All SIMD types available with `Wasm128Token`.\n");
    code.push_str("    //!\n");
    code.push_str("    //! Natural width: 128-bit (SIMD128). `f32xN` = `f32x4`.\n");
    code.push_str("    //!\n");
    code.push_str("    //! Also includes 256-bit polyfills (emulated via 2×128-bit WASM ops).\n");
    code.push_str("    //! All take `Wasm128Token`.\n\n");

    // xN aliases (natural width = 128-bit)
    code.push_str("    pub use super::wasm::w128::{\n");
    code.push_str("        f32x4 as f32xN, f64x2 as f64xN, i8x16 as i8xN, i16x8 as i16xN,\n");
    code.push_str("        i32x4 as i32xN, i64x2 as i64xN, u8x16 as u8xN, u16x8 as u16xN,\n");
    code.push_str("        u32x4 as u32xN, u64x2 as u64xN,\n");
    code.push_str("    };\n\n");

    // Native 128-bit (natural width)
    code.push_str("    pub use super::wasm::w128::*;\n\n");

    // Polyfilled 256-bit (wider, same token)
    code.push_str("    // 256-bit polyfilled types (2×128-bit WASM SIMD, same Wasm128Token)\n");
    code.push_str(&format!("    pub use super::polyfill::wasm128::{{{polyfill_types}}};\n\n"));

    code.push_str("    /// Token type for this width level\n");
    code.push_str("    pub type Token = archmage::Wasm128Token;\n\n");
    code.push_str("    pub const LANES_F32: usize = 4;\n");
    code.push_str("    pub const LANES_F64: usize = 2;\n");
    code.push_str("    pub const LANES_32: usize = 4;\n");
    code.push_str("    pub const LANES_16: usize = 8;\n");
    code.push_str("    pub const LANES_8: usize = 16;\n");
    code.push_str("}\n");

    code
}

/// Generate a file containing types of a specific width
fn generate_width_file(types: &[SimdType], width: SimdWidth) -> String {
    let mut code = String::new();

    // Header comment
    let width_name = match width {
        SimdWidth::W128 => "128-bit (SSE)",
        SimdWidth::W256 => "256-bit (AVX/AVX2)",
        SimdWidth::W512 => "512-bit (AVX-512)",
    };
    code.push_str(&format!("//! {} SIMD types.\n", width_name));
    code.push_str("//!\n");
    code.push_str("//! **Auto-generated** by `cargo xtask generate` - do not edit manually.\n\n");

    // Imports
    code.push_str("use core::arch::x86_64::*;\n");
    code.push_str("use core::ops::{\n");
    code.push_str(
        "    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign,\n",
    );
    code.push_str("    Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,\n");
    code.push_str("};\n\n");

    // Note: SimdEq, SimdNe, etc. traits are defined in parent but not currently used
    // Macros are exported at crate root via #[macro_export]

    // Generate each type of this width
    let width_types: Vec<_> = types.iter().filter(|t| t.width == width).collect();
    for ty in &width_types {
        code.push_str(&structure::generate_type(ty));
    }

    code
}

/// Generate SIMD types split into multiple files
/// Returns a map of relative path -> file content
///
/// All generated files go into `generated/` subfolders.
/// The root `magetypes/src/simd/mod.rs` is hand-written and just re-exports.
pub fn generate_simd_types_split() -> BTreeMap<String, String> {
    let mut files = BTreeMap::new();
    let types = all_simd_types();

    // Root mod.rs (magetypes/src/simd/mod.rs) is hand-written, not generated.
    // It just does: mod generated; pub use generated::*;

    // generated/mod.rs contains the real generated code
    files.insert(
        "generated/mod.rs".to_string(),
        generate_generated_mod_rs(&types),
    );

    // x86 directory mod.rs
    let x86_mod = r#"//! x86-64 SIMD types.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

pub mod w128;
pub mod w256;
#[cfg(feature = "avx512")]
pub mod w512;
"#;
    files.insert("generated/x86/mod.rs".to_string(), x86_mod.to_string());

    // Generate x86 width-specific files
    files.insert(
        "generated/x86/w128.rs".to_string(),
        generate_width_file(&types, SimdWidth::W128),
    );
    files.insert(
        "generated/x86/w256.rs".to_string(),
        generate_width_file(&types, SimdWidth::W256),
    );
    files.insert(
        "generated/x86/w512.rs".to_string(),
        generate_width_file(&types, SimdWidth::W512),
    );

    // ARM directory mod.rs
    let arm_mod = r#"//! ARM AArch64 SIMD types (NEON).
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

pub mod w128;
"#;
    files.insert("generated/arm/mod.rs".to_string(), arm_mod.to_string());

    // Generate ARM NEON types
    let neon_types = structure_arm::all_neon_types();
    files.insert(
        "generated/arm/w128.rs".to_string(),
        structure_arm::generate_arm_w128(&neon_types),
    );

    // WASM directory mod.rs
    let wasm_mod = r#"//! WebAssembly SIMD types (SIMD128).
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

pub mod w128;
"#;
    files.insert("generated/wasm/mod.rs".to_string(), wasm_mod.to_string());

    // Generate WASM SIMD128 types
    let wasm_types = structure_wasm::all_wasm_types();
    files.insert(
        "generated/wasm/w128.rs".to_string(),
        structure_wasm::generate_wasm_w128(&wasm_types),
    );

    // Generate polyfill types (W256 from pairs of W128)
    files.insert(
        "generated/polyfill.rs".to_string(),
        structure_polyfill::generate_polyfill(),
    );

    files
}

/// Generate all SIMD types (legacy single-file output)
#[allow(dead_code)]
pub fn generate_simd_types() -> String {
    let mut code = String::new();

    // Header
    code.push_str(file_header());
    code.push_str(
        r#"#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign,
    Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

"#,
    );

    // Generate comparison traits
    code.push_str(&structure::generate_comparison_traits());

    // Generate macros
    code.push_str(&structure::generate_macros());

    // Generate each type
    for ty in &all_simd_types() {
        code.push_str(&structure::generate_type(ty));
    }

    code
}

/// Generate tests for SIMD types
pub fn generate_simd_tests() -> String {
    let mut code = String::from(
        r#"//! Auto-generated tests for SIMD types.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

#![cfg(target_arch = "x86_64")]
#![allow(clippy::needless_range_loop)]

use magetypes::simd::*;
use archmage::{SimdToken, X64V3Token};

"#,
    );

    // Basic tests for 256-bit types (most common)
    let test_types = [
        ("f32x8", "X64V3Token", "f32", "1.0", "2.0"),
        ("i32x8", "X64V3Token", "i32", "1", "2"),
    ];

    for (ty_name, token, elem, val1, val2) in test_types {
        code.push_str(&format!(
            r#"
#[test]
fn test_{ty_name}_basic() {{
    if let Some(token) = {token}::summon() {{
        let a = {ty_name}::splat(token, {val1});
        let b = {ty_name}::splat(token, {val2});
        let c = a + b;
        let arr = c.to_array();
        for &v in &arr {{
            assert_eq!(v, {val1} + {val2});
        }}
    }}
}}

#[test]
fn test_{ty_name}_load_store() {{
    if let Some(token) = {token}::summon() {{
        let data: [{elem}; {ty_name}::LANES] = [{val1}; {ty_name}::LANES];
        let v = {ty_name}::load(token, &data);
        let mut out = [{elem}::default(); {ty_name}::LANES];
        v.store(&mut out);
        assert_eq!(data, out);
    }}
}}
"#
        ));
    }

    // Transpose test for f32x8
    code.push_str(
        r#"
#[test]
fn test_f32x8_transpose_8x8() {
    if let Some(token) = X64V3Token::summon() {
        // Create 8 row vectors: row[i] = [i*8, i*8+1, ..., i*8+7]
        let mut rows = [
            f32x8::from_array(token, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
            f32x8::from_array(token, [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]),
            f32x8::from_array(token, [16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0]),
            f32x8::from_array(token, [24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0]),
            f32x8::from_array(token, [32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0]),
            f32x8::from_array(token, [40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0]),
            f32x8::from_array(token, [48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0]),
            f32x8::from_array(token, [56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0]),
        ];

        f32x8::transpose_8x8(&mut rows);

        // After transpose: rows[i][j] should be original rows[j][i] = j*8 + i
        for i in 0..8 {
            let arr = rows[i].to_array();
            for j in 0..8 {
                let expected = (j * 8 + i) as f32;
                assert_eq!(arr[j], expected, "Mismatch at rows[{}][{}]", i, j);
            }
        }

        // Double transpose should restore original
        f32x8::transpose_8x8(&mut rows);
        for i in 0..8 {
            let arr = rows[i].to_array();
            for j in 0..8 {
                let expected = (i * 8 + j) as f32;
                assert_eq!(arr[j], expected, "Double transpose mismatch at rows[{}][{}]", i, j);
            }
        }
    }
}

#[test]
fn test_f32x8_load_store_8x8() {
    if let Some(_token) = X64V3Token::summon() {
        let input: [f32; 64] = core::array::from_fn(|i| i as f32);
        let rows = f32x8::load_8x8(&input);

        // Verify load
        for i in 0..8 {
            let arr = rows[i].to_array();
            for j in 0..8 {
                assert_eq!(arr[j], (i * 8 + j) as f32);
            }
        }

        // Verify store roundtrip
        let mut output = [0.0f32; 64];
        f32x8::store_8x8(&rows, &mut output);
        assert_eq!(input, output);
    }
}

#[test]
fn test_f32x4_4ch_interleave() {
    if let Some(token) = archmage::X64V3Token::summon() {
        // Create 4 channel vectors (SoA format)
        let r = f32x4::from_array(token, [1.0, 2.0, 3.0, 4.0]);
        let g = f32x4::from_array(token, [10.0, 20.0, 30.0, 40.0]);
        let b = f32x4::from_array(token, [100.0, 200.0, 300.0, 400.0]);
        let a = f32x4::from_array(token, [255.0, 255.0, 255.0, 255.0]);

        // Interleave to AoS: each output vector is one RGBA pixel
        let aos = f32x4::interleave_4ch([r, g, b, a]);
        assert_eq!(aos[0].to_array(), [1.0, 10.0, 100.0, 255.0]);   // pixel 0
        assert_eq!(aos[1].to_array(), [2.0, 20.0, 200.0, 255.0]);   // pixel 1
        assert_eq!(aos[2].to_array(), [3.0, 30.0, 300.0, 255.0]);   // pixel 2
        assert_eq!(aos[3].to_array(), [4.0, 40.0, 400.0, 255.0]);   // pixel 3

        // Deinterleave back to SoA
        let [r2, g2, b2, a2] = f32x4::deinterleave_4ch(aos);
        assert_eq!(r2.to_array(), r.to_array());
        assert_eq!(g2.to_array(), g.to_array());
        assert_eq!(b2.to_array(), b.to_array());
        assert_eq!(a2.to_array(), a.to_array());
    }
}

#[test]
fn test_f32x4_load_store_rgba_u8() {
    if let Some(_token) = archmage::X64V3Token::summon() {
        // 4 RGBA pixels: red, green, blue, white
        let rgba: [u8; 16] = [
            255, 0, 0, 255,     // red
            0, 255, 0, 255,     // green
            0, 0, 255, 255,     // blue
            255, 255, 255, 255, // white
        ];

        let (r, g, b, a) = f32x4::load_4_rgba_u8(&rgba);
        assert_eq!(r.to_array(), [255.0, 0.0, 0.0, 255.0]);
        assert_eq!(g.to_array(), [0.0, 255.0, 0.0, 255.0]);
        assert_eq!(b.to_array(), [0.0, 0.0, 255.0, 255.0]);
        assert_eq!(a.to_array(), [255.0, 255.0, 255.0, 255.0]);

        // Roundtrip
        let out = f32x4::store_4_rgba_u8(r, g, b, a);
        assert_eq!(out, rgba);
    }
}

#[test]
fn test_f32x8_load_store_rgba_u8() {
    if let Some(_token) = X64V3Token::summon() {
        // 8 RGBA pixels
        let rgba: [u8; 32] = [
            255, 0, 0, 255,     // red
            0, 255, 0, 255,     // green
            0, 0, 255, 255,     // blue
            255, 255, 255, 255, // white
            128, 128, 128, 255, // gray
            0, 0, 0, 255,       // black
            255, 128, 0, 255,   // orange
            128, 0, 255, 255,   // purple
        ];

        let (r, g, b, a) = f32x8::load_8_rgba_u8(&rgba);
        assert_eq!(r.to_array(), [255.0, 0.0, 0.0, 255.0, 128.0, 0.0, 255.0, 128.0]);
        assert_eq!(g.to_array(), [0.0, 255.0, 0.0, 255.0, 128.0, 0.0, 128.0, 0.0]);
        assert_eq!(b.to_array(), [0.0, 0.0, 255.0, 255.0, 128.0, 0.0, 0.0, 255.0]);
        assert_eq!(a.to_array(), [255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0]);

        // Roundtrip
        let out = f32x8::store_8_rgba_u8(r, g, b, a);
        assert_eq!(out, rgba);
    }
}
"#,
    );

    // Add AVX-512 tests
    code.push_str(
        r#"
// ============================================================================
// AVX-512 Tests (require avx512 feature)
// ============================================================================

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
mod avx512_tests {
    use magetypes::simd::*;
    use archmage::{SimdToken, Avx512Token};

    #[test]
    fn test_f32x16_basic() {
        if let Some(token) = Avx512Token::summon() {
            let a = f32x16::splat(token, 1.0);
            let b = f32x16::splat(token, 2.0);
            let c = a + b;
            let arr = c.to_array();
            for &v in &arr {
                assert_eq!(v, 3.0);
            }
        }
    }

    #[test]
    fn test_f32x16_load_store() {
        if let Some(token) = Avx512Token::summon() {
            let data: [f32; 16] = core::array::from_fn(|i| i as f32);
            let v = f32x16::load(token, &data);
            let mut out = [0.0f32; 16];
            v.store(&mut out);
            assert_eq!(data, out);
        }
    }

    #[test]
    fn test_i32x16_basic() {
        if let Some(token) = Avx512Token::summon() {
            let a = i32x16::splat(token, 10);
            let b = i32x16::splat(token, 20);
            let c = a + b;
            let arr = c.to_array();
            for &v in &arr {
                assert_eq!(v, 30);
            }
        }
    }

    #[test]
    fn test_i32x16_load_store() {
        if let Some(token) = Avx512Token::summon() {
            let data: [i32; 16] = core::array::from_fn(|i| i as i32);
            let v = i32x16::load(token, &data);
            let mut out = [0i32; 16];
            v.store(&mut out);
            assert_eq!(data, out);
        }
    }

    #[test]
    fn test_f64x8_basic() {
        if let Some(token) = Avx512Token::summon() {
            let a = f64x8::splat(token, 2.5);
            let b = f64x8::splat(token, 1.5);
            let sum = a + b;
            assert_eq!(sum.to_array(), [4.0; 8]);
        }
    }

    #[test]
    fn test_f32x16_math_ops() {
        if let Some(token) = Avx512Token::summon() {
            let v = f32x16::from_array(token, [
                1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0,
                81.0, 100.0, 121.0, 144.0, 169.0, 196.0, 225.0, 256.0
            ]);
            let sqrt_v = v.sqrt();
            let expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                          9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
            assert_eq!(sqrt_v.to_array(), expected);
        }
    }

    #[test]
    fn test_f32x16_fma() {
        if let Some(token) = Avx512Token::summon() {
            let a = f32x16::splat(token, 2.0);
            let b = f32x16::splat(token, 3.0);
            let c = f32x16::splat(token, 1.0);

            // a * b + c = 2 * 3 + 1 = 7
            let result = a.mul_add(b, c);
            assert_eq!(result.to_array(), [7.0; 16]);
        }
    }

    #[test]
    fn test_cast_slice_512() {
        if let Some(token) = Avx512Token::summon() {
            let data: [f32; 32] = core::array::from_fn(|i| i as f32);

            let vectors = f32x16::cast_slice(token, &data).unwrap();
            assert_eq!(vectors.len(), 2);
            assert_eq!(vectors[0].to_array()[0], 0.0);
            assert_eq!(vectors[1].to_array()[0], 16.0);
        }
    }
}
"#,
    );

    // Add ARM tests
    code.push_str(
        r#"
// ============================================================================
// AArch64 (NEON) Tests
// ============================================================================

#[cfg(target_arch = "aarch64")]
mod arm_tests {
    use magetypes::simd::*;
    use archmage::{SimdToken, NeonToken};

    #[test]
    fn test_f32x4_basic() {
        if let Some(token) = NeonToken::summon() {
            let a = f32x4::splat(token, 1.0);
            let b = f32x4::splat(token, 2.0);
            let c = a + b;
            let arr = c.to_array();
            for &v in &arr {
                assert_eq!(v, 3.0);
            }
        }
    }

    #[test]
    fn test_f32x4_load_store() {
        if let Some(token) = NeonToken::summon() {
            let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
            let v = f32x4::load(token, &data);
            let mut out = [0.0f32; 4];
            v.store(&mut out);
            assert_eq!(data, out);
        }
    }

    #[test]
    fn test_i32x4_basic() {
        if let Some(token) = NeonToken::summon() {
            let a = i32x4::splat(token, 10);
            let b = i32x4::splat(token, 20);
            let c = a + b;
            let arr = c.to_array();
            for &v in &arr {
                assert_eq!(v, 30);
            }
        }
    }

    #[test]
    fn test_i32x4_load_store() {
        if let Some(token) = NeonToken::summon() {
            let data: [i32; 4] = [1, 2, 3, 4];
            let v = i32x4::load(token, &data);
            let mut out = [0i32; 4];
            v.store(&mut out);
            assert_eq!(data, out);
        }
    }

    #[test]
    fn test_i64x2_min_max() {
        // Test the polyfilled 64-bit min/max
        if let Some(token) = NeonToken::summon() {
            let a = i64x2::from_array(token, [10, -5]);
            let b = i64x2::from_array(token, [5, -2]);

            let min_result = a.min(b);
            assert_eq!(min_result.to_array(), [5, -5]);

            let max_result = a.max(b);
            assert_eq!(max_result.to_array(), [10, -2]);
        }
    }

    #[test]
    fn test_u64x2_min_max() {
        // Test the polyfilled 64-bit unsigned min/max
        if let Some(token) = NeonToken::summon() {
            let a = u64x2::from_array(token, [100, 200]);
            let b = u64x2::from_array(token, [150, 50]);

            let min_result = a.min(b);
            assert_eq!(min_result.to_array(), [100, 50]);

            let max_result = a.max(b);
            assert_eq!(max_result.to_array(), [150, 200]);
        }
    }

    #[test]
    fn test_f64x2_operations() {
        if let Some(token) = NeonToken::summon() {
            let a = f64x2::splat(token, 2.5);
            let b = f64x2::splat(token, 1.5);

            let sum = a + b;
            assert_eq!(sum.to_array(), [4.0, 4.0]);

            let prod = a * b;
            assert_eq!(prod.to_array(), [3.75, 3.75]);
        }
    }

    #[test]
    fn test_f32x4_math_ops() {
        if let Some(token) = NeonToken::summon() {
            let v = f32x4::from_array(token, [4.0, 9.0, 16.0, 25.0]);

            let sqrt_v = v.sqrt();
            assert_eq!(sqrt_v.to_array(), [2.0, 3.0, 4.0, 5.0]);

            let abs_v = f32x4::from_array(token, [-1.0, 2.0, -3.0, 4.0]).abs();
            assert_eq!(abs_v.to_array(), [1.0, 2.0, 3.0, 4.0]);
        }
    }

    #[test]
    fn test_f32x4_fma() {
        if let Some(token) = NeonToken::summon() {
            let a = f32x4::splat(token, 2.0);
            let b = f32x4::splat(token, 3.0);
            let c = f32x4::splat(token, 1.0);

            // a * b + c = 2 * 3 + 1 = 7
            let result = a.mul_add(b, c);
            assert_eq!(result.to_array(), [7.0, 7.0, 7.0, 7.0]);
        }
    }

    #[test]
    fn test_cast_slice() {
        if let Some(token) = NeonToken::summon() {
            let data: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

            // Cast to f32x4 slice
            let vectors = f32x4::cast_slice(token, &data).unwrap();
            assert_eq!(vectors.len(), 2);
            assert_eq!(vectors[0].to_array(), [1.0, 2.0, 3.0, 4.0]);
            assert_eq!(vectors[1].to_array(), [5.0, 6.0, 7.0, 8.0]);
        }
    }

    #[test]
    fn test_from_bytes() {
        if let Some(token) = NeonToken::summon() {
            let bytes: [u8; 16] = [
                0x00, 0x00, 0x80, 0x3f, // 1.0f32
                0x00, 0x00, 0x00, 0x40, // 2.0f32
                0x00, 0x00, 0x40, 0x40, // 3.0f32
                0x00, 0x00, 0x80, 0x40, // 4.0f32
            ];

            let v = f32x4::from_bytes(token, &bytes);
            assert_eq!(v.to_array(), [1.0, 2.0, 3.0, 4.0]);
        }
    }
}
"#,
    );

    // Bitcast tests
    code.push_str(
        r#"
// ============================================================================
// Bitcast Tests
// ============================================================================

#[test]
fn test_f32x8_bitcast_i32x8_roundtrip() {
    if let Some(token) = X64V3Token::summon() {
        let f = f32x8::splat(token, 1.0f32);
        let i = f.bitcast_i32x8();
        // IEEE 754: 1.0f32 = 0x3F800000
        assert_eq!(i[0], 0x3F80_0000_i32);
        let f2 = i.bitcast_f32x8();
        assert_eq!(f2.to_array(), f.to_array());
    }
}

#[test]
fn test_f32x4_bitcast_i32x4_roundtrip() {
    if let Some(token) = X64V3Token::summon() {
        let f = f32x4::splat(token, -1.0f32);
        let i = f.bitcast_i32x4();
        // IEEE 754: -1.0f32 = 0xBF800000
        assert_eq!(i[0], -0x4080_0000_i32); // 0xBF800000 as i32
        let f2 = i.bitcast_f32x4();
        assert_eq!(f2.to_array(), f.to_array());
    }
}

#[test]
fn test_i32x8_bitcast_u32x8() {
    if let Some(token) = X64V3Token::summon() {
        let i = i32x8::splat(token, -1);
        let u = i.bitcast_u32x8();
        assert_eq!(u[0], u32::MAX);
        let i2 = u.bitcast_i32x8();
        assert_eq!(i2[0], -1);
    }
}

#[test]
fn test_f32x8_bitcast_ref() {
    if let Some(token) = X64V3Token::summon() {
        let f = f32x8::splat(token, 1.0f32);
        let i_ref: &i32x8 = f.bitcast_ref_i32x8();
        assert_eq!(i_ref[0], 0x3F80_0000_i32);
    }
}

#[test]
fn test_f32x8_bitcast_mut() {
    if let Some(token) = X64V3Token::summon() {
        let mut f = f32x8::splat(token, 1.0f32);
        let i_mut: &mut i32x8 = f.bitcast_mut_i32x8();
        // Modify via the bitcast reference
        i_mut[0] = 0x4000_0000; // 2.0f32 in IEEE 754
        assert_eq!(f[0], 2.0f32);
    }
}

#[test]
fn test_f64x4_bitcast_i64x4_roundtrip() {
    if let Some(token) = X64V3Token::summon() {
        let f = f64x4::splat(token, 1.0f64);
        let i = f.bitcast_i64x4();
        // IEEE 754: 1.0f64 = 0x3FF0000000000000
        assert_eq!(i[0], 0x3FF0_0000_0000_0000_i64);
        let f2 = i.bitcast_f64x4();
        assert_eq!(f2.to_array(), f.to_array());
    }
}

#[test]
fn test_i8x32_bitcast_u8x32() {
    if let Some(token) = X64V3Token::summon() {
        let i = i8x32::splat(token, -128);
        let u = i.bitcast_u8x32();
        assert_eq!(u[0], 128u8);
    }
}

#[test]
fn test_i16x16_bitcast_u16x16() {
    if let Some(token) = X64V3Token::summon() {
        let i = i16x16::splat(token, -1);
        let u = i.bitcast_u16x16();
        assert_eq!(u[0], u16::MAX);
    }
}
"#,
    );

    code
}

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
mod ops;
mod ops_comparison;
mod structure;
mod transcendental;
pub mod types;

pub use types::{all_simd_types, ElementType, SimdType, SimdWidth};

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

"#
}

/// Generate the main mod.rs with re-exports
fn generate_mod_rs(types: &[SimdType]) -> String {
    let mut code = String::from(file_header());

    // Note: core::ops imports are in submodules where types are defined
    // Macros are #[macro_export] and expand in those submodules

    // Generate comparison traits
    code.push_str(&structure::generate_comparison_traits());

    // Generate macros
    code.push_str(&structure::generate_macros());

    // Module declarations and re-exports grouped by width
    code.push_str("// ============================================================================\n");
    code.push_str("// Type modules\n");
    code.push_str("// ============================================================================\n\n");

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

    // 128-bit types (SSE)
    code.push_str("// 128-bit types (SSE/NEON)\n");
    code.push_str("#[cfg(target_arch = \"x86_64\")]\n");
    code.push_str("mod x86 {\n");
    code.push_str("    pub mod w128;\n");
    code.push_str("    pub mod w256;\n");
    code.push_str("    #[cfg(feature = \"avx512\")]\n");
    code.push_str("    pub mod w512;\n");
    code.push_str("}\n\n");

    // Re-exports
    code.push_str("// Re-export all types\n");
    code.push_str("#[cfg(target_arch = \"x86_64\")]\n");
    code.push_str("pub use x86::w128::*;\n");
    code.push_str("#[cfg(target_arch = \"x86_64\")]\n");
    code.push_str("pub use x86::w256::*;\n");
    code.push_str("#[cfg(all(target_arch = \"x86_64\", feature = \"avx512\"))]\n");
    code.push_str("pub use x86::w512::*;\n");

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
    code.push_str("    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign,\n");
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
pub fn generate_simd_types_split() -> BTreeMap<String, String> {
    let mut files = BTreeMap::new();
    let types = all_simd_types();

    // Main mod.rs
    files.insert("mod.rs".to_string(), generate_mod_rs(&types));

    // x86 directory mod.rs
    let x86_mod = r#"//! x86-64 SIMD types.

pub mod w128;
pub mod w256;
#[cfg(feature = "avx512")]
pub mod w512;
"#;
    files.insert("x86/mod.rs".to_string(), x86_mod.to_string());

    // Generate width-specific files
    files.insert(
        "x86/w128.rs".to_string(),
        generate_width_file(&types, SimdWidth::W128),
    );
    files.insert(
        "x86/w256.rs".to_string(),
        generate_width_file(&types, SimdWidth::W256),
    );
    files.insert(
        "x86/w512.rs".to_string(),
        generate_width_file(&types, SimdWidth::W512),
    );

    files
}

/// Generate all SIMD types (legacy single-file output)
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

use archmage::simd::*;
use archmage::{SimdToken, Avx2FmaToken};

"#,
    );

    // Basic tests for 256-bit types (most common)
    let test_types = [
        ("f32x8", "Avx2FmaToken", "f32", "1.0", "2.0"),
        ("i32x8", "Avx2FmaToken", "i32", "1", "2"),
    ];

    for (ty_name, token, elem, val1, val2) in test_types {
        code.push_str(&format!(
            r#"
#[test]
fn test_{ty_name}_basic() {{
    if let Some(token) = {token}::try_new() {{
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
    if let Some(token) = {token}::try_new() {{
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
    if let Some(token) = Avx2FmaToken::try_new() {
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
    if let Some(_token) = Avx2FmaToken::try_new() {
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
    if let Some(token) = archmage::Sse41Token::try_new() {
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
    if let Some(_token) = archmage::Sse41Token::try_new() {
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
    if let Some(_token) = Avx2FmaToken::try_new() {
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

    code
}

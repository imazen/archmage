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
//! - `structure` - Type definition and macro generation
//! - `ops` - Standard SIMD operations (comparison, blend, math, etc.)
//! - `transcendental` - Transcendental functions (log, exp, pow, cbrt)

mod ops;
mod structure;
mod transcendental;
mod types;

pub use types::{all_simd_types, ElementType, SimdType, SimdWidth};

/// Generate all SIMD types
pub fn generate_simd_types() -> String {
    let mut code = String::new();

    // Header
    code.push_str(
        r#"//! Token-gated SIMD types with natural operators.
//!
//! Provides `wide`-like ergonomics with token-gated construction.
//! There is NO way to construct these types without proving CPU support.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

#![allow(dead_code)]
#![allow(non_camel_case_types)]
#![allow(missing_docs)]

#[cfg(target_arch = "x86_64")]
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

    code
}

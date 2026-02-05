//! Auto-generated polyfill types.
//!
//! Generates W256 types from pairs of W128 types for all 10 element types,
//! across all platforms (SSE, NEON, WASM SIMD128).

use super::ops_bitcast;
use super::types::{ElementType, SimdType, SimdWidth};
use std::fmt::Write;

/// All 10 element types.
const ALL_ELEMENTS: &[ElementType] = &[
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

/// Check if a W128 type has `min`/`max` methods.
/// i64 and u64 don't have native min/max on x86/ARM/WASM W128.
fn has_min_max(elem: ElementType) -> bool {
    !matches!(elem, ElementType::I64 | ElementType::U64)
}

/// Check if a W128 type has `reduce_max`/`reduce_min` methods.
/// Only floats have these on x86 W128.
fn has_reduce_minmax(elem: ElementType) -> bool {
    elem.is_float()
}

/// Check if a W128 type has `Neg` impl.
/// Only floats have Neg on x86 W128.
fn has_neg(elem: ElementType) -> bool {
    elem.is_float()
}

/// Check if a W128 type has `abs` method.
/// Floats and signed integers (except i64) have abs.
fn has_abs(elem: ElementType) -> bool {
    matches!(
        elem,
        ElementType::F32 | ElementType::F64 | ElementType::I8 | ElementType::I16 | ElementType::I32
    )
}

/// Check if a type has `Mul` operator.
fn has_mul(elem: ElementType) -> bool {
    elem.is_float()
        || matches!(
            elem,
            ElementType::I16 | ElementType::U16 | ElementType::I32 | ElementType::U32
        )
}

/// Platform configuration for polyfill generation.
struct PlatformConfig {
    /// Module name (e.g., "sse", "neon", "wasm128")
    mod_name: &'static str,
    /// Tier name for implementation_name() (e.g., "v3", "neon", "simd128")
    tier_name: &'static str,
    /// Doc comment for the module
    doc: &'static str,
    /// cfg attribute (e.g., `target_arch = "x86_64"`)
    cfg: &'static str,
    /// Token type name
    token: &'static str,
    /// Import path prefix for W128 types
    w128_import: &'static str,
}

const PLATFORMS: &[PlatformConfig] = &[
    PlatformConfig {
        mod_name: "v3",
        tier_name: "v3",
        doc: "Polyfilled 256-bit types using x86-64-v3 (128-bit) operations.",
        cfg: "target_arch = \"x86_64\"",
        token: "X64V3Token",
        w128_import: "crate::simd::generated::x86::w128",
    },
    PlatformConfig {
        mod_name: "neon",
        tier_name: "neon",
        doc: "Polyfilled 256-bit types using NEON (128-bit) operations.",
        cfg: "target_arch = \"aarch64\"",
        token: "NeonToken",
        w128_import: "crate::simd::generated::arm::w128",
    },
    PlatformConfig {
        mod_name: "wasm128",
        tier_name: "wasm128",
        doc: "Polyfilled 256-bit types using WASM SIMD128 (128-bit) operations.",
        cfg: "target_arch = \"wasm32\"",
        token: "Wasm128Token",
        w128_import: "crate::simd::generated::wasm::w128",
    },
];

/// W512 platform configs: polyfill W512 from pairs of native W256
struct W512PlatformConfig {
    /// Module name
    mod_name: &'static str,
    /// Tier name for implementation_name() (e.g., "v3")
    tier_name: &'static str,
    /// Doc comment
    doc: &'static str,
    /// cfg attribute
    cfg: &'static str,
    /// Token type
    token: &'static str,
    /// Import path for W256 types
    w256_import: &'static str,
}

const W512_PLATFORMS: &[W512PlatformConfig] = &[W512PlatformConfig {
    mod_name: "v3",
    tier_name: "v3",
    doc: "Polyfilled 512-bit types using x86-64-v3 (256-bit) operations.",
    cfg: "target_arch = \"x86_64\"",
    token: "X64V3Token",
    w256_import: "crate::simd::generated::x86::w256",
}];

/// Generate the entire polyfill.rs file.
pub fn generate_polyfill() -> String {
    let mut code = String::new();

    code.push_str("//! Polyfill module for emulating wider SIMD types on narrower hardware.\n");
    code.push_str("//!\n");
    code.push_str(
        "//! This module provides types like `f32x8` that work on SSE/NEON/WASM hardware by\n",
    );
    code.push_str("//! internally using two W128 operations. This allows writing code\n");
    code.push_str("//! targeting AVX2 widths while still running (slower) on SSE-only systems.\n");
    code.push_str("//!\n");
    code.push_str("//! **Auto-generated** by `cargo xtask generate` - do not edit manually.\n\n");

    // W256 polyfills (from pairs of W128)
    for platform in PLATFORMS {
        code.push_str(&generate_platform_module(platform));
    }

    // W512 polyfills (from pairs of W256)
    for platform in W512_PLATFORMS {
        code.push_str(&generate_w512_platform_module(platform));
    }

    code
}

/// Generate a single platform module (e.g., `pub mod sse { ... }`)
fn generate_platform_module(platform: &PlatformConfig) -> String {
    let mut code = String::new();

    writeln!(code, "#[cfg({})]", platform.cfg).unwrap();
    writeln!(code, "pub mod {} {{", platform.mod_name).unwrap();
    writeln!(code, "    //! {}", platform.doc).unwrap();
    writeln!(code).unwrap();

    // Imports
    let w128_types: Vec<String> = ALL_ELEMENTS
        .iter()
        .map(|e| SimdType::new(*e, SimdWidth::W128).name())
        .collect();
    writeln!(
        code,
        "    use {}::{{{}}};",
        platform.w128_import,
        w128_types.join(", ")
    )
    .unwrap();
    writeln!(code, "    use archmage::{};", platform.token).unwrap();
    writeln!(code, "    use core::ops::{{Add, Sub, Mul, Div, Neg}};").unwrap();
    writeln!(code).unwrap();

    // Generate each W256 polyfill type
    for elem in ALL_ELEMENTS {
        code.push_str(&generate_polyfill_type(*elem, platform));
    }

    // Generate xN aliases and LANES constants
    writeln!(code).unwrap();
    writeln!(code, "    // Width-aliased type names").unwrap();
    for elem in ALL_ELEMENTS {
        let full = SimdType::new(*elem, SimdWidth::W256);
        let full_name = full.name();
        writeln!(code, "    pub type {}xN = {};", elem.name(), full_name).unwrap();
    }
    writeln!(code).unwrap();
    writeln!(code, "    /// Number of f32 lanes").unwrap();
    writeln!(
        code,
        "    pub const LANES_F32: usize = {};",
        SimdType::new(ElementType::F32, SimdWidth::W256).lanes()
    )
    .unwrap();
    writeln!(code, "    /// Number of f64 lanes").unwrap();
    writeln!(
        code,
        "    pub const LANES_F64: usize = {};",
        SimdType::new(ElementType::F64, SimdWidth::W256).lanes()
    )
    .unwrap();
    writeln!(code, "    /// Number of i32/u32 lanes").unwrap();
    writeln!(
        code,
        "    pub const LANES_32: usize = {};",
        SimdType::new(ElementType::I32, SimdWidth::W256).lanes()
    )
    .unwrap();
    writeln!(code, "    /// Number of i16/u16 lanes").unwrap();
    writeln!(
        code,
        "    pub const LANES_16: usize = {};",
        SimdType::new(ElementType::I16, SimdWidth::W256).lanes()
    )
    .unwrap();
    writeln!(code, "    /// Number of i8/u8 lanes").unwrap();
    writeln!(
        code,
        "    pub const LANES_8: usize = {};",
        SimdType::new(ElementType::I8, SimdWidth::W256).lanes()
    )
    .unwrap();
    writeln!(code).unwrap();
    writeln!(code, "    /// Token type for this polyfill level").unwrap();
    writeln!(code, "    pub type Token = archmage::{};", platform.token).unwrap();

    writeln!(code, "}}").unwrap();
    writeln!(code).unwrap();

    code
}

/// Generate a single polyfill type (e.g., `pub struct f32x8 { lo: f32x4, hi: f32x4 }`)
fn generate_polyfill_type(elem: ElementType, platform: &PlatformConfig) -> String {
    let mut code = String::new();

    let half = SimdType::new(elem, SimdWidth::W128);
    let full = SimdType::new(elem, SimdWidth::W256);
    let half_name = half.name();
    let full_name = full.name();
    let full_lanes = full.lanes();
    let half_lanes = half.lanes();
    let elem_name = elem.name();
    let token = platform.token;

    // Struct definition
    writeln!(
        code,
        "    /// Emulated {full_lanes}-wide {elem_name} vector using two {half_name} vectors."
    )
    .unwrap();
    writeln!(code, "    #[derive(Clone, Copy, Debug)]").unwrap();
    writeln!(code, "    pub struct {full_name} {{").unwrap();
    writeln!(code, "        pub(crate) lo: {half_name},").unwrap();
    writeln!(code, "        pub(crate) hi: {half_name},").unwrap();
    writeln!(code, "    }}\n").unwrap();

    // Impl block
    writeln!(code, "    impl {full_name} {{").unwrap();
    writeln!(code, "        pub const LANES: usize = {full_lanes};\n").unwrap();

    // Construction: load
    writeln!(code, "        /// Load from array (token-gated)").unwrap();
    writeln!(code, "        #[inline(always)]").unwrap();
    writeln!(
        code,
        "        pub fn load(token: {token}, data: &[{elem_name}; {full_lanes}]) -> Self {{"
    )
    .unwrap();
    writeln!(
        code,
        "            let lo_arr: &[{elem_name}; {half_lanes}] = data[0..{half_lanes}].try_into().unwrap();"
    )
    .unwrap();
    writeln!(
        code,
        "            let hi_arr: &[{elem_name}; {half_lanes}] = data[{half_lanes}..{full_lanes}].try_into().unwrap();"
    )
    .unwrap();
    writeln!(code, "            Self {{").unwrap();
    writeln!(
        code,
        "                lo: {half_name}::load(token, lo_arr),"
    )
    .unwrap();
    writeln!(
        code,
        "                hi: {half_name}::load(token, hi_arr),"
    )
    .unwrap();
    writeln!(code, "            }}").unwrap();
    writeln!(code, "        }}\n").unwrap();

    // Construction: splat
    writeln!(
        code,
        "        /// Broadcast scalar to all lanes (token-gated)"
    )
    .unwrap();
    writeln!(code, "        #[inline(always)]").unwrap();
    writeln!(
        code,
        "        pub fn splat(token: {token}, v: {elem_name}) -> Self {{"
    )
    .unwrap();
    writeln!(code, "            Self {{").unwrap();
    writeln!(code, "                lo: {half_name}::splat(token, v),").unwrap();
    writeln!(code, "                hi: {half_name}::splat(token, v),").unwrap();
    writeln!(code, "            }}").unwrap();
    writeln!(code, "        }}\n").unwrap();

    // Construction: zero
    writeln!(code, "        /// Zero vector (token-gated)").unwrap();
    writeln!(code, "        #[inline(always)]").unwrap();
    writeln!(code, "        pub fn zero(token: {token}) -> Self {{").unwrap();
    writeln!(code, "            Self {{").unwrap();
    writeln!(code, "                lo: {half_name}::zero(token),").unwrap();
    writeln!(code, "                hi: {half_name}::zero(token),").unwrap();
    writeln!(code, "            }}").unwrap();
    writeln!(code, "        }}\n").unwrap();

    // Construction: from_array
    writeln!(code, "        /// Create from array (token-gated)").unwrap();
    writeln!(code, "        #[inline(always)]").unwrap();
    writeln!(
        code,
        "        pub fn from_array(token: {token}, arr: [{elem_name}; {full_lanes}]) -> Self {{"
    )
    .unwrap();
    writeln!(code, "            Self::load(token, &arr)").unwrap();
    writeln!(code, "        }}\n").unwrap();

    // Store
    writeln!(code, "        /// Store to array").unwrap();
    writeln!(code, "        #[inline(always)]").unwrap();
    writeln!(
        code,
        "        pub fn store(self, out: &mut [{elem_name}; {full_lanes}]) {{"
    )
    .unwrap();
    writeln!(
        code,
        "            let (lo, hi) = out.split_at_mut({half_lanes});"
    )
    .unwrap();
    writeln!(
        code,
        "            let lo_arr: &mut [{elem_name}; {half_lanes}] = lo.try_into().unwrap();"
    )
    .unwrap();
    writeln!(
        code,
        "            let hi_arr: &mut [{elem_name}; {half_lanes}] = hi.try_into().unwrap();"
    )
    .unwrap();
    writeln!(code, "            self.lo.store(lo_arr);").unwrap();
    writeln!(code, "            self.hi.store(hi_arr);").unwrap();
    writeln!(code, "        }}\n").unwrap();

    // To array
    writeln!(code, "        /// Convert to array").unwrap();
    writeln!(code, "        #[inline(always)]").unwrap();
    writeln!(
        code,
        "        pub fn to_array(self) -> [{elem_name}; {full_lanes}] {{"
    )
    .unwrap();
    writeln!(
        code,
        "            let mut out = [{}; {}];",
        elem.zero_literal(),
        full_lanes
    )
    .unwrap();
    writeln!(code, "            self.store(&mut out);").unwrap();
    writeln!(code, "            out").unwrap();
    writeln!(code, "        }}\n").unwrap();

    // Min/Max/Clamp — only for types that have them on W128
    if has_min_max(elem) {
        generate_lo_hi_binary(&mut code, "min", "Element-wise minimum");
        generate_lo_hi_binary(&mut code, "max", "Element-wise maximum");

        writeln!(code, "        /// Clamp values between lo and hi").unwrap();
        writeln!(code, "        #[inline(always)]").unwrap();
        writeln!(
            code,
            "        pub fn clamp(self, lo: Self, hi: Self) -> Self {{"
        )
        .unwrap();
        writeln!(code, "            self.max(lo).min(hi)").unwrap();
        writeln!(code, "        }}\n").unwrap();
    }

    // Float-only operations
    if elem.is_float() {
        generate_lo_hi_unary(&mut code, "sqrt", "Square root");
        generate_lo_hi_unary(&mut code, "floor", "Floor");
        generate_lo_hi_unary(&mut code, "ceil", "Ceil");
        generate_lo_hi_unary(&mut code, "round", "Round to nearest");

        // mul_add
        writeln!(code, "        /// Fused multiply-add: self * a + b").unwrap();
        writeln!(code, "        #[inline(always)]").unwrap();
        writeln!(
            code,
            "        pub fn mul_add(self, a: Self, b: Self) -> Self {{"
        )
        .unwrap();
        writeln!(code, "            Self {{").unwrap();
        writeln!(code, "                lo: self.lo.mul_add(a.lo, b.lo),").unwrap();
        writeln!(code, "                hi: self.hi.mul_add(a.hi, b.hi),").unwrap();
        writeln!(code, "            }}").unwrap();
        writeln!(code, "        }}\n").unwrap();
    }

    // abs — for floats and signed integers (i8, i16, i32, not i64)
    if has_abs(elem) {
        generate_lo_hi_unary(&mut code, "abs", "Absolute value");
    }

    // Reduce operations
    // reduce_add — always available
    writeln!(code, "        /// Reduce: sum all lanes").unwrap();
    writeln!(code, "        #[inline(always)]").unwrap();
    writeln!(code, "        pub fn reduce_add(self) -> {elem_name} {{").unwrap();
    if elem.is_float() {
        writeln!(
            code,
            "            self.lo.reduce_add() + self.hi.reduce_add()"
        )
        .unwrap();
    } else {
        writeln!(
            code,
            "            self.lo.reduce_add().wrapping_add(self.hi.reduce_add())"
        )
        .unwrap();
    }
    writeln!(code, "        }}\n").unwrap();

    // reduce_max / reduce_min — only for floats (W128 types only have these for floats)
    if has_reduce_minmax(elem) {
        writeln!(code, "        /// Reduce: max of all lanes").unwrap();
        writeln!(code, "        #[inline(always)]").unwrap();
        writeln!(code, "        pub fn reduce_max(self) -> {elem_name} {{").unwrap();
        writeln!(
            code,
            "            self.lo.reduce_max().max(self.hi.reduce_max())"
        )
        .unwrap();
        writeln!(code, "        }}\n").unwrap();

        writeln!(code, "        /// Reduce: min of all lanes").unwrap();
        writeln!(code, "        #[inline(always)]").unwrap();
        writeln!(code, "        pub fn reduce_min(self) -> {elem_name} {{").unwrap();
        writeln!(
            code,
            "            self.lo.reduce_min().min(self.hi.reduce_min())"
        )
        .unwrap();
        writeln!(code, "        }}\n").unwrap();
    }

    // Bitcast operations
    code.push_str(
        &ops_bitcast::generate_polyfill_bitcasts(elem, SimdWidth::W256, SimdWidth::W128)
            .replace('\n', "\n    "),
    );

    // Implementation identification
    let tier_name = platform.tier_name;
    let impl_name = format!("polyfill::{tier_name}::{full_name}");
    writeln!(
        code,
        "        // ========== Implementation identification ==========\n"
    )
    .unwrap();
    writeln!(
        code,
        "        /// Returns a string identifying this type's implementation."
    )
    .unwrap();
    writeln!(code, "        ///").unwrap();
    writeln!(
        code,
        "        /// This is useful for verifying that the correct implementation is being used"
    )
    .unwrap();
    writeln!(
        code,
        "        /// at compile time (via `-Ctarget-cpu`) or at runtime (via `#[magetypes]` dispatch)."
    )
    .unwrap();
    writeln!(code, "        ///").unwrap();
    writeln!(code, "        /// Returns `\"{impl_name}\"`.",).unwrap();
    writeln!(code, "        #[inline(always)]").unwrap();
    writeln!(
        code,
        "        pub const fn implementation_name() -> &'static str {{"
    )
    .unwrap();
    writeln!(code, "            \"{impl_name}\"").unwrap();
    writeln!(code, "        }}\n").unwrap();

    writeln!(code, "    }}\n").unwrap();

    // Operator impls
    // Add and Sub — always present
    generate_op_impl(&mut code, &full_name, "Add", "add");
    generate_op_impl(&mut code, &full_name, "Sub", "sub");

    // Mul — float and 16/32-bit integers
    if has_mul(elem) {
        generate_op_impl(&mut code, &full_name, "Mul", "mul");
    }

    // Div — float only
    if elem.is_float() {
        generate_op_impl(&mut code, &full_name, "Div", "div");
    }

    // Neg — only floats on x86 W128
    if has_neg(elem) {
        writeln!(code, "    impl Neg for {full_name} {{").unwrap();
        writeln!(code, "        type Output = Self;").unwrap();
        writeln!(code, "        #[inline(always)]").unwrap();
        writeln!(code, "        fn neg(self) -> Self {{").unwrap();
        writeln!(code, "            Self {{").unwrap();
        writeln!(code, "                lo: -self.lo,").unwrap();
        writeln!(code, "                hi: -self.hi,").unwrap();
        writeln!(code, "            }}").unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();
    }

    // Assign ops
    generate_assign_op(&mut code, &full_name, "AddAssign", "add_assign", "+");
    generate_assign_op(&mut code, &full_name, "SubAssign", "sub_assign", "-");

    if has_mul(elem) {
        generate_assign_op(&mut code, &full_name, "MulAssign", "mul_assign", "*");
    }

    if elem.is_float() {
        generate_assign_op(&mut code, &full_name, "DivAssign", "div_assign", "/");
    }

    writeln!(code).unwrap();

    code
}

/// Generate a W512 platform module (e.g., `pub mod avx2 { ... }`)
fn generate_w512_platform_module(platform: &W512PlatformConfig) -> String {
    let mut code = String::new();

    writeln!(code, "#[cfg({})]", platform.cfg).unwrap();
    writeln!(code, "pub mod {} {{", platform.mod_name).unwrap();
    writeln!(code, "    //! {}", platform.doc).unwrap();
    writeln!(code).unwrap();

    // Imports
    let w256_types: Vec<String> = ALL_ELEMENTS
        .iter()
        .map(|e| SimdType::new(*e, SimdWidth::W256).name())
        .collect();
    writeln!(
        code,
        "    use {}::{{{}}};",
        platform.w256_import,
        w256_types.join(", ")
    )
    .unwrap();
    writeln!(code, "    use archmage::{};", platform.token).unwrap();
    writeln!(code, "    use core::ops::{{Add, Sub, Mul, Div, Neg}};").unwrap();
    writeln!(code).unwrap();

    // Generate each W512 polyfill type
    for elem in ALL_ELEMENTS {
        code.push_str(&generate_w512_polyfill_type(*elem, platform));
    }

    // Generate xN aliases and LANES constants
    writeln!(code).unwrap();
    writeln!(code, "    // Width-aliased type names").unwrap();
    for elem in ALL_ELEMENTS {
        let full = SimdType::new(*elem, SimdWidth::W512);
        let full_name = full.name();
        writeln!(code, "    pub type {}xN = {};", elem.name(), full_name).unwrap();
    }
    writeln!(code).unwrap();
    writeln!(code, "    /// Number of f32 lanes").unwrap();
    writeln!(
        code,
        "    pub const LANES_F32: usize = {};",
        SimdType::new(ElementType::F32, SimdWidth::W512).lanes()
    )
    .unwrap();
    writeln!(code, "    /// Number of f64 lanes").unwrap();
    writeln!(
        code,
        "    pub const LANES_F64: usize = {};",
        SimdType::new(ElementType::F64, SimdWidth::W512).lanes()
    )
    .unwrap();
    writeln!(code, "    /// Number of i32/u32 lanes").unwrap();
    writeln!(
        code,
        "    pub const LANES_32: usize = {};",
        SimdType::new(ElementType::I32, SimdWidth::W512).lanes()
    )
    .unwrap();
    writeln!(code, "    /// Number of i16/u16 lanes").unwrap();
    writeln!(
        code,
        "    pub const LANES_16: usize = {};",
        SimdType::new(ElementType::I16, SimdWidth::W512).lanes()
    )
    .unwrap();
    writeln!(code, "    /// Number of i8/u8 lanes").unwrap();
    writeln!(
        code,
        "    pub const LANES_8: usize = {};",
        SimdType::new(ElementType::I8, SimdWidth::W512).lanes()
    )
    .unwrap();
    writeln!(code).unwrap();
    writeln!(code, "    /// Token type for this polyfill level").unwrap();
    writeln!(code, "    pub type Token = archmage::{};", platform.token).unwrap();

    writeln!(code, "}}").unwrap();
    writeln!(code).unwrap();

    code
}

/// Generate a single W512 polyfill type from pairs of W256
fn generate_w512_polyfill_type(elem: ElementType, platform: &W512PlatformConfig) -> String {
    let mut code = String::new();

    let half = SimdType::new(elem, SimdWidth::W256);
    let full = SimdType::new(elem, SimdWidth::W512);
    let half_name = half.name();
    let full_name = full.name();
    let full_lanes = full.lanes();
    let half_lanes = half.lanes();
    let elem_name = elem.name();
    let token = platform.token;

    // Struct definition
    writeln!(
        code,
        "    /// Emulated {full_lanes}-wide {elem_name} vector using two {half_name} vectors."
    )
    .unwrap();
    writeln!(code, "    #[derive(Clone, Copy, Debug)]").unwrap();
    writeln!(code, "    pub struct {full_name} {{").unwrap();
    writeln!(code, "        pub(crate) lo: {half_name},").unwrap();
    writeln!(code, "        pub(crate) hi: {half_name},").unwrap();
    writeln!(code, "    }}\n").unwrap();

    writeln!(code, "    impl {full_name} {{").unwrap();
    writeln!(code, "        pub const LANES: usize = {full_lanes};\n").unwrap();

    // Construction: load
    writeln!(code, "        /// Load from array (token-gated)").unwrap();
    writeln!(code, "        #[inline(always)]").unwrap();
    writeln!(
        code,
        "        pub fn load(token: {token}, data: &[{elem_name}; {full_lanes}]) -> Self {{"
    )
    .unwrap();
    writeln!(
        code,
        "            let lo_arr: &[{elem_name}; {half_lanes}] = data[0..{half_lanes}].try_into().unwrap();"
    )
    .unwrap();
    writeln!(
        code,
        "            let hi_arr: &[{elem_name}; {half_lanes}] = data[{half_lanes}..{full_lanes}].try_into().unwrap();"
    )
    .unwrap();
    writeln!(code, "            Self {{").unwrap();
    writeln!(
        code,
        "                lo: {half_name}::load(token, lo_arr),"
    )
    .unwrap();
    writeln!(
        code,
        "                hi: {half_name}::load(token, hi_arr),"
    )
    .unwrap();
    writeln!(code, "            }}").unwrap();
    writeln!(code, "        }}\n").unwrap();

    // Construction: splat
    writeln!(
        code,
        "        /// Broadcast scalar to all lanes (token-gated)"
    )
    .unwrap();
    writeln!(code, "        #[inline(always)]").unwrap();
    writeln!(
        code,
        "        pub fn splat(token: {token}, v: {elem_name}) -> Self {{"
    )
    .unwrap();
    writeln!(code, "            Self {{").unwrap();
    writeln!(code, "                lo: {half_name}::splat(token, v),").unwrap();
    writeln!(code, "                hi: {half_name}::splat(token, v),").unwrap();
    writeln!(code, "            }}").unwrap();
    writeln!(code, "        }}\n").unwrap();

    // Construction: zero
    writeln!(code, "        /// Zero vector (token-gated)").unwrap();
    writeln!(code, "        #[inline(always)]").unwrap();
    writeln!(code, "        pub fn zero(token: {token}) -> Self {{").unwrap();
    writeln!(code, "            Self {{").unwrap();
    writeln!(code, "                lo: {half_name}::zero(token),").unwrap();
    writeln!(code, "                hi: {half_name}::zero(token),").unwrap();
    writeln!(code, "            }}").unwrap();
    writeln!(code, "        }}\n").unwrap();

    // Store
    writeln!(code, "        /// Store to array").unwrap();
    writeln!(code, "        #[inline(always)]").unwrap();
    writeln!(
        code,
        "        pub fn store(self, out: &mut [{elem_name}; {full_lanes}]) {{"
    )
    .unwrap();
    writeln!(
        code,
        "            let (lo, hi) = out.split_at_mut({half_lanes});"
    )
    .unwrap();
    writeln!(
        code,
        "            let lo_arr: &mut [{elem_name}; {half_lanes}] = lo.try_into().unwrap();"
    )
    .unwrap();
    writeln!(
        code,
        "            let hi_arr: &mut [{elem_name}; {half_lanes}] = hi.try_into().unwrap();"
    )
    .unwrap();
    writeln!(code, "            self.lo.store(lo_arr);").unwrap();
    writeln!(code, "            self.hi.store(hi_arr);").unwrap();
    writeln!(code, "        }}\n").unwrap();

    // To array
    writeln!(code, "        /// Convert to array").unwrap();
    writeln!(code, "        #[inline(always)]").unwrap();
    writeln!(
        code,
        "        pub fn to_array(self) -> [{elem_name}; {full_lanes}] {{"
    )
    .unwrap();
    writeln!(
        code,
        "            let mut out = [{}; {}];",
        elem.zero_literal(),
        full_lanes
    )
    .unwrap();
    writeln!(code, "            self.store(&mut out);").unwrap();
    writeln!(code, "            out").unwrap();
    writeln!(code, "        }}\n").unwrap();

    // Math ops (same availability as W256)
    if has_min_max(elem) {
        generate_lo_hi_binary(&mut code, "min", "Element-wise minimum");
        generate_lo_hi_binary(&mut code, "max", "Element-wise maximum");

        writeln!(code, "        /// Clamp values between lo and hi").unwrap();
        writeln!(code, "        #[inline(always)]").unwrap();
        writeln!(
            code,
            "        pub fn clamp(self, lo: Self, hi: Self) -> Self {{"
        )
        .unwrap();
        writeln!(code, "            self.max(lo).min(hi)").unwrap();
        writeln!(code, "        }}\n").unwrap();
    }

    if elem.is_float() {
        generate_lo_hi_unary(&mut code, "sqrt", "Square root");
        generate_lo_hi_unary(&mut code, "floor", "Floor");
        generate_lo_hi_unary(&mut code, "ceil", "Ceil");
        generate_lo_hi_unary(&mut code, "round", "Round to nearest");

        writeln!(code, "        /// Fused multiply-add: self * a + b").unwrap();
        writeln!(code, "        #[inline(always)]").unwrap();
        writeln!(
            code,
            "        pub fn mul_add(self, a: Self, b: Self) -> Self {{"
        )
        .unwrap();
        writeln!(code, "            Self {{").unwrap();
        writeln!(code, "                lo: self.lo.mul_add(a.lo, b.lo),").unwrap();
        writeln!(code, "                hi: self.hi.mul_add(a.hi, b.hi),").unwrap();
        writeln!(code, "            }}").unwrap();
        writeln!(code, "        }}\n").unwrap();
    }

    if has_abs(elem) {
        generate_lo_hi_unary(&mut code, "abs", "Absolute value");
    }

    // reduce_add
    writeln!(code, "        /// Reduce: sum all lanes").unwrap();
    writeln!(code, "        #[inline(always)]").unwrap();
    writeln!(code, "        pub fn reduce_add(self) -> {elem_name} {{").unwrap();
    if elem.is_float() {
        writeln!(
            code,
            "            self.lo.reduce_add() + self.hi.reduce_add()"
        )
        .unwrap();
    } else {
        writeln!(
            code,
            "            self.lo.reduce_add().wrapping_add(self.hi.reduce_add())"
        )
        .unwrap();
    }
    writeln!(code, "        }}\n").unwrap();

    if has_reduce_minmax(elem) {
        writeln!(code, "        /// Reduce: max of all lanes").unwrap();
        writeln!(code, "        #[inline(always)]").unwrap();
        writeln!(code, "        pub fn reduce_max(self) -> {elem_name} {{").unwrap();
        writeln!(
            code,
            "            self.lo.reduce_max().max(self.hi.reduce_max())"
        )
        .unwrap();
        writeln!(code, "        }}\n").unwrap();

        writeln!(code, "        /// Reduce: min of all lanes").unwrap();
        writeln!(code, "        #[inline(always)]").unwrap();
        writeln!(code, "        pub fn reduce_min(self) -> {elem_name} {{").unwrap();
        writeln!(
            code,
            "            self.lo.reduce_min().min(self.hi.reduce_min())"
        )
        .unwrap();
        writeln!(code, "        }}\n").unwrap();
    }

    // Bitcast operations
    code.push_str(
        &ops_bitcast::generate_polyfill_bitcasts(elem, SimdWidth::W512, SimdWidth::W256)
            .replace('\n', "\n    "),
    );

    // Implementation identification
    let tier_name = platform.tier_name;
    let impl_name = format!("polyfill::{tier_name}::{full_name}");
    writeln!(
        code,
        "        // ========== Implementation identification ==========\n"
    )
    .unwrap();
    writeln!(
        code,
        "        /// Returns a string identifying this type's implementation."
    )
    .unwrap();
    writeln!(code, "        ///").unwrap();
    writeln!(
        code,
        "        /// This is useful for verifying that the correct implementation is being used"
    )
    .unwrap();
    writeln!(
        code,
        "        /// at compile time (via `-Ctarget-cpu`) or at runtime (via `#[magetypes]` dispatch)."
    )
    .unwrap();
    writeln!(code, "        ///").unwrap();
    writeln!(code, "        /// Returns `\"{impl_name}\"`.",).unwrap();
    writeln!(code, "        #[inline(always)]").unwrap();
    writeln!(
        code,
        "        pub const fn implementation_name() -> &'static str {{"
    )
    .unwrap();
    writeln!(code, "            \"{impl_name}\"").unwrap();
    writeln!(code, "        }}\n").unwrap();

    writeln!(code, "    }}\n").unwrap();

    // Operator impls
    generate_op_impl(&mut code, &full_name, "Add", "add");
    generate_op_impl(&mut code, &full_name, "Sub", "sub");

    if has_mul(elem) {
        generate_op_impl(&mut code, &full_name, "Mul", "mul");
    }

    if elem.is_float() {
        generate_op_impl(&mut code, &full_name, "Div", "div");
    }

    if has_neg(elem) {
        writeln!(code, "    impl Neg for {full_name} {{").unwrap();
        writeln!(code, "        type Output = Self;").unwrap();
        writeln!(code, "        #[inline(always)]").unwrap();
        writeln!(code, "        fn neg(self) -> Self {{").unwrap();
        writeln!(code, "            Self {{").unwrap();
        writeln!(code, "                lo: -self.lo,").unwrap();
        writeln!(code, "                hi: -self.hi,").unwrap();
        writeln!(code, "            }}").unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();
    }

    generate_assign_op(&mut code, &full_name, "AddAssign", "add_assign", "+");
    generate_assign_op(&mut code, &full_name, "SubAssign", "sub_assign", "-");

    if has_mul(elem) {
        generate_assign_op(&mut code, &full_name, "MulAssign", "mul_assign", "*");
    }

    if elem.is_float() {
        generate_assign_op(&mut code, &full_name, "DivAssign", "div_assign", "/");
    }

    writeln!(code).unwrap();

    code
}

/// Generate a lo/hi delegating unary method.
fn generate_lo_hi_unary(code: &mut String, name: &str, doc: &str) {
    writeln!(code, "        /// {doc}").unwrap();
    writeln!(code, "        #[inline(always)]").unwrap();
    writeln!(code, "        pub fn {name}(self) -> Self {{").unwrap();
    writeln!(code, "            Self {{").unwrap();
    writeln!(code, "                lo: self.lo.{name}(),").unwrap();
    writeln!(code, "                hi: self.hi.{name}(),").unwrap();
    writeln!(code, "            }}").unwrap();
    writeln!(code, "        }}\n").unwrap();
}

/// Generate a lo/hi delegating binary method.
fn generate_lo_hi_binary(code: &mut String, name: &str, doc: &str) {
    writeln!(code, "        /// {doc}").unwrap();
    writeln!(code, "        #[inline(always)]").unwrap();
    writeln!(code, "        pub fn {name}(self, other: Self) -> Self {{").unwrap();
    writeln!(code, "            Self {{").unwrap();
    writeln!(code, "                lo: self.lo.{name}(other.lo),").unwrap();
    writeln!(code, "                hi: self.hi.{name}(other.hi),").unwrap();
    writeln!(code, "            }}").unwrap();
    writeln!(code, "        }}\n").unwrap();
}

/// Generate an operator impl (Add, Sub, Mul, Div).
fn generate_op_impl(code: &mut String, full_name: &str, trait_name: &str, fn_name: &str) {
    let op = match fn_name {
        "add" => "+",
        "sub" => "-",
        "mul" => "*",
        "div" => "/",
        _ => panic!("unknown op: {fn_name}"),
    };
    writeln!(code, "    impl {trait_name} for {full_name} {{").unwrap();
    writeln!(code, "        type Output = Self;").unwrap();
    writeln!(code, "        #[inline(always)]").unwrap();
    writeln!(code, "        fn {fn_name}(self, rhs: Self) -> Self {{").unwrap();
    writeln!(code, "            Self {{").unwrap();
    writeln!(code, "                lo: self.lo {op} rhs.lo,").unwrap();
    writeln!(code, "                hi: self.hi {op} rhs.hi,").unwrap();
    writeln!(code, "            }}").unwrap();
    writeln!(code, "        }}").unwrap();
    writeln!(code, "    }}\n").unwrap();
}

/// Generate an assign operator impl (AddAssign, SubAssign, etc.).
fn generate_assign_op(
    code: &mut String,
    full_name: &str,
    trait_name: &str,
    fn_name: &str,
    op: &str,
) {
    writeln!(code, "    impl core::ops::{trait_name} for {full_name} {{").unwrap();
    writeln!(code, "        #[inline(always)]").unwrap();
    writeln!(code, "        fn {fn_name}(&mut self, rhs: Self) {{").unwrap();
    writeln!(code, "            *self = *self {op} rhs;").unwrap();
    writeln!(code, "        }}").unwrap();
    writeln!(code, "    }}\n").unwrap();
}

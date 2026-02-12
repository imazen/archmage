//! Auto-generated polyfill types.
//!
//! Generates W256 types from pairs of W128 types for all 10 element types,
//! across all platforms (SSE, NEON, WASM SIMD128).

use super::ops_bitcast;
use super::types::{ElementType, SimdType, SimdWidth};
use indoc::formatdoc;

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
    mod_name: "v3_512",
    tier_name: "v3_512",
    doc: "Polyfilled 512-bit types using x86-64-v3 (256-bit) operations.",
    cfg: "target_arch = \"x86_64\"",
    token: "X64V3Token",
    w256_import: "crate::simd::generated::x86::w256",
}];

/// Generate the entire polyfill.rs file.
pub fn generate_polyfill() -> String {
    let mut code = String::new();

    code.push_str(&formatdoc! {"
        //! Polyfill module for emulating wider SIMD types on narrower hardware.
        //!
        //! This module provides types like `f32x8` that work on SSE/NEON/WASM hardware by
        //! internally using two W128 operations. This allows writing code
        //! targeting AVX2 widths while still running (slower) on SSE-only systems.
        //!
        //! **Auto-generated** by `cargo xtask generate` - do not edit manually.

    "});

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

    let cfg = platform.cfg;
    let mod_name = platform.mod_name;
    let doc = platform.doc;
    let w128_import = platform.w128_import;
    let token = platform.token;

    let w128_types: Vec<String> = ALL_ELEMENTS
        .iter()
        .map(|e| SimdType::new(*e, SimdWidth::W128).name())
        .collect();
    let w128_types_joined = w128_types.join(", ");

    code.push_str(&formatdoc! {"
        #[cfg({cfg})]
        pub mod {mod_name} {{
            //! {doc}

            use {w128_import}::{{{w128_types_joined}}};
            use archmage::{token};
            use core::ops::{{Add, Sub, Mul, Div, Neg}};

    "});

    // Generate each W256 polyfill type
    for elem in ALL_ELEMENTS {
        code.push_str(&generate_polyfill_type(*elem, platform));
    }

    // Generate xN aliases and LANES constants
    let mut aliases = String::from("\n    // Width-aliased type names\n");
    for elem in ALL_ELEMENTS {
        let full = SimdType::new(*elem, SimdWidth::W256);
        let full_name = full.name();
        let elem_name = elem.name();
        aliases.push_str(&format!("    pub type {elem_name}xN = {full_name};\n"));
    }

    let lanes_f32 = SimdType::new(ElementType::F32, SimdWidth::W256).lanes();
    let lanes_f64 = SimdType::new(ElementType::F64, SimdWidth::W256).lanes();
    let lanes_32 = SimdType::new(ElementType::I32, SimdWidth::W256).lanes();
    let lanes_16 = SimdType::new(ElementType::I16, SimdWidth::W256).lanes();
    let lanes_8 = SimdType::new(ElementType::I8, SimdWidth::W256).lanes();

    code.push_str(&aliases);
    code.push_str(&formatdoc! {"

        /// Number of f32 lanes
        pub const LANES_F32: usize = {lanes_f32};
        /// Number of f64 lanes
        pub const LANES_F64: usize = {lanes_f64};
        /// Number of i32/u32 lanes
        pub const LANES_32: usize = {lanes_32};
        /// Number of i16/u16 lanes
        pub const LANES_16: usize = {lanes_16};
        /// Number of i8/u8 lanes
        pub const LANES_8: usize = {lanes_8};

        /// Token type for this polyfill level
        pub type Token = archmage::{token};
    "});
    // Indent the above block
    code = code.replace("\n/// Number", "\n    /// Number");
    code = code.replace("\npub const", "\n    pub const");
    code = code.replace("\n/// Token", "\n    /// Token");
    code = code.replace("\npub type Token", "\n    pub type Token");

    code.push_str("}\n\n");

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
    let zero_lit = elem.zero_literal();

    // Struct definition + constructors + store/to_array
    code.push_str(&formatdoc! {"
            /// Emulated {full_lanes}-wide {elem_name} vector using two {half_name} vectors.
            #[derive(Clone, Copy, Debug)]
            pub struct {full_name} {{
                pub(crate) lo: {half_name},
                pub(crate) hi: {half_name},
            }}

            impl {full_name} {{
                pub const LANES: usize = {full_lanes};

                /// Load from array (token-gated)
                #[inline(always)]
                pub fn load(token: {token}, data: &[{elem_name}; {full_lanes}]) -> Self {{
                    let lo_arr: &[{elem_name}; {half_lanes}] = data[0..{half_lanes}].try_into().unwrap();
                    let hi_arr: &[{elem_name}; {half_lanes}] = data[{half_lanes}..{full_lanes}].try_into().unwrap();
                    Self {{
                        lo: {half_name}::load(token, lo_arr),
                        hi: {half_name}::load(token, hi_arr),
                    }}
                }}

                /// Broadcast scalar to all lanes (token-gated)
                #[inline(always)]
                pub fn splat(token: {token}, v: {elem_name}) -> Self {{
                    Self {{
                        lo: {half_name}::splat(token, v),
                        hi: {half_name}::splat(token, v),
                    }}
                }}

                /// Zero vector (token-gated)
                #[inline(always)]
                pub fn zero(token: {token}) -> Self {{
                    Self {{
                        lo: {half_name}::zero(token),
                        hi: {half_name}::zero(token),
                    }}
                }}

                /// Create from array (token-gated)
                #[inline(always)]
                pub fn from_array(token: {token}, arr: [{elem_name}; {full_lanes}]) -> Self {{
                    Self::load(token, &arr)
                }}

                /// Store to array
                #[inline(always)]
                pub fn store(self, out: &mut [{elem_name}; {full_lanes}]) {{
                    let (lo, hi) = out.split_at_mut({half_lanes});
                    let lo_arr: &mut [{elem_name}; {half_lanes}] = lo.try_into().unwrap();
                    let hi_arr: &mut [{elem_name}; {half_lanes}] = hi.try_into().unwrap();
                    self.lo.store(lo_arr);
                    self.hi.store(hi_arr);
                }}

                /// Convert to array
                #[inline(always)]
                pub fn to_array(self) -> [{elem_name}; {full_lanes}] {{
                    let mut out = [{zero_lit}; {full_lanes}];
                    self.store(&mut out);
                    out
                }}

    "});

    // Min/Max/Clamp
    if has_min_max(elem) {
        code.push_str(&gen_lo_hi_binary("min", "Element-wise minimum"));
        code.push_str(&gen_lo_hi_binary("max", "Element-wise maximum"));
        code.push_str(&formatdoc! {"
                    /// Clamp values between lo and hi
                    #[inline(always)]
                    pub fn clamp(self, lo: Self, hi: Self) -> Self {{
                        self.max(lo).min(hi)
                    }}

        "});
    }

    // Float-only operations
    if elem.is_float() {
        code.push_str(&gen_lo_hi_unary("sqrt", "Square root"));
        code.push_str(&gen_lo_hi_unary("floor", "Floor"));
        code.push_str(&gen_lo_hi_unary("ceil", "Ceil"));
        code.push_str(&gen_lo_hi_unary("round", "Round to nearest"));
        code.push_str(&formatdoc! {"
                    /// Fused multiply-add: self * a + b
                    #[inline(always)]
                    pub fn mul_add(self, a: Self, b: Self) -> Self {{
                        Self {{
                            lo: self.lo.mul_add(a.lo, b.lo),
                            hi: self.hi.mul_add(a.hi, b.hi),
                        }}
                    }}

        "});
    }

    // abs
    if has_abs(elem) {
        code.push_str(&gen_lo_hi_unary("abs", "Absolute value"));
    }

    // reduce_add
    let reduce_add_body = if elem.is_float() {
        "self.lo.reduce_add() + self.hi.reduce_add()"
    } else {
        "self.lo.reduce_add().wrapping_add(self.hi.reduce_add())"
    };
    code.push_str(&formatdoc! {"
                /// Reduce: sum all lanes
                #[inline(always)]
                pub fn reduce_add(self) -> {elem_name} {{
                    {reduce_add_body}
                }}

    "});

    // reduce_max / reduce_min
    if has_reduce_minmax(elem) {
        code.push_str(&formatdoc! {"
                    /// Reduce: max of all lanes
                    #[inline(always)]
                    pub fn reduce_max(self) -> {elem_name} {{
                        self.lo.reduce_max().max(self.hi.reduce_max())
                    }}

                    /// Reduce: min of all lanes
                    #[inline(always)]
                    pub fn reduce_min(self) -> {elem_name} {{
                        self.lo.reduce_min().min(self.hi.reduce_min())
                    }}

        "});
    }

    // Bitcast operations
    code.push_str(
        &ops_bitcast::generate_polyfill_bitcasts(elem, SimdWidth::W256, SimdWidth::W128)
            .replace('\n', "\n    "),
    );

    // Implementation identification
    let tier_name = platform.tier_name;
    let impl_name = format!("polyfill::{tier_name}::{full_name}");
    code.push_str(&formatdoc! {"
                // ========== Implementation identification ==========

                /// Returns a string identifying this type's implementation.
                ///
                /// This is useful for verifying that the correct implementation is being used
                /// at compile time (via `-Ctarget-cpu`) or at runtime (via `#[magetypes]` dispatch).
                ///
                /// Returns `\"{impl_name}\"`.
                #[inline(always)]
                pub const fn implementation_name() -> &'static str {{
                    \"{impl_name}\"
                }}

            }}

    "});

    // Operator impls
    code.push_str(&gen_op_impl(&full_name, "Add", "add"));
    code.push_str(&gen_op_impl(&full_name, "Sub", "sub"));

    if has_mul(elem) {
        code.push_str(&gen_op_impl(&full_name, "Mul", "mul"));
    }

    if elem.is_float() {
        code.push_str(&gen_op_impl(&full_name, "Div", "div"));
    }

    // Neg
    if has_neg(elem) {
        code.push_str(&formatdoc! {"
                impl Neg for {full_name} {{
                    type Output = Self;
                    #[inline(always)]
                    fn neg(self) -> Self {{
                        Self {{
                            lo: -self.lo,
                            hi: -self.hi,
                        }}
                    }}
                }}

        "});
    }

    // Assign ops
    code.push_str(&gen_assign_op(&full_name, "AddAssign", "add_assign", "+"));
    code.push_str(&gen_assign_op(&full_name, "SubAssign", "sub_assign", "-"));

    if has_mul(elem) {
        code.push_str(&gen_assign_op(&full_name, "MulAssign", "mul_assign", "*"));
    }

    if elem.is_float() {
        code.push_str(&gen_assign_op(&full_name, "DivAssign", "div_assign", "/"));
    }

    code.push('\n');

    code
}

/// Generate a W512 platform module (e.g., `pub mod avx2 { ... }`)
fn generate_w512_platform_module(platform: &W512PlatformConfig) -> String {
    let mut code = String::new();

    let cfg = platform.cfg;
    let mod_name = platform.mod_name;
    let doc = platform.doc;
    let w256_import = platform.w256_import;
    let token = platform.token;

    let w256_types: Vec<String> = ALL_ELEMENTS
        .iter()
        .map(|e| SimdType::new(*e, SimdWidth::W256).name())
        .collect();
    let w256_types_joined = w256_types.join(", ");

    code.push_str(&formatdoc! {"
        #[cfg({cfg})]
        pub mod {mod_name} {{
            //! {doc}

            use {w256_import}::{{{w256_types_joined}}};
            use archmage::{token};
            use core::ops::{{Add, Sub, Mul, Div, Neg}};

    "});

    // Generate each W512 polyfill type
    for elem in ALL_ELEMENTS {
        code.push_str(&generate_w512_polyfill_type(*elem, platform));
    }

    // Generate xN aliases and LANES constants
    let mut aliases = String::from("\n    // Width-aliased type names\n");
    for elem in ALL_ELEMENTS {
        let full = SimdType::new(*elem, SimdWidth::W512);
        let full_name = full.name();
        let elem_name = elem.name();
        aliases.push_str(&format!("    pub type {elem_name}xN = {full_name};\n"));
    }

    let lanes_f32 = SimdType::new(ElementType::F32, SimdWidth::W512).lanes();
    let lanes_f64 = SimdType::new(ElementType::F64, SimdWidth::W512).lanes();
    let lanes_32 = SimdType::new(ElementType::I32, SimdWidth::W512).lanes();
    let lanes_16 = SimdType::new(ElementType::I16, SimdWidth::W512).lanes();
    let lanes_8 = SimdType::new(ElementType::I8, SimdWidth::W512).lanes();

    code.push_str(&aliases);
    code.push_str(&formatdoc! {"

        /// Number of f32 lanes
        pub const LANES_F32: usize = {lanes_f32};
        /// Number of f64 lanes
        pub const LANES_F64: usize = {lanes_f64};
        /// Number of i32/u32 lanes
        pub const LANES_32: usize = {lanes_32};
        /// Number of i16/u16 lanes
        pub const LANES_16: usize = {lanes_16};
        /// Number of i8/u8 lanes
        pub const LANES_8: usize = {lanes_8};

        /// Token type for this polyfill level
        pub type Token = archmage::{token};
    "});
    // Indent the above block
    code = code.replace("\n/// Number", "\n    /// Number");
    code = code.replace("\npub const", "\n    pub const");
    code = code.replace("\n/// Token", "\n    /// Token");
    code = code.replace("\npub type Token", "\n    pub type Token");

    code.push_str("}\n\n");

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
    let zero_lit = elem.zero_literal();

    // Struct definition + constructors + store/to_array
    code.push_str(&formatdoc! {"
            /// Emulated {full_lanes}-wide {elem_name} vector using two {half_name} vectors.
            #[derive(Clone, Copy, Debug)]
            pub struct {full_name} {{
                pub(crate) lo: {half_name},
                pub(crate) hi: {half_name},
            }}

            impl {full_name} {{
                pub const LANES: usize = {full_lanes};

                /// Load from array (token-gated)
                #[inline(always)]
                pub fn load(token: {token}, data: &[{elem_name}; {full_lanes}]) -> Self {{
                    let lo_arr: &[{elem_name}; {half_lanes}] = data[0..{half_lanes}].try_into().unwrap();
                    let hi_arr: &[{elem_name}; {half_lanes}] = data[{half_lanes}..{full_lanes}].try_into().unwrap();
                    Self {{
                        lo: {half_name}::load(token, lo_arr),
                        hi: {half_name}::load(token, hi_arr),
                    }}
                }}

                /// Broadcast scalar to all lanes (token-gated)
                #[inline(always)]
                pub fn splat(token: {token}, v: {elem_name}) -> Self {{
                    Self {{
                        lo: {half_name}::splat(token, v),
                        hi: {half_name}::splat(token, v),
                    }}
                }}

                /// Zero vector (token-gated)
                #[inline(always)]
                pub fn zero(token: {token}) -> Self {{
                    Self {{
                        lo: {half_name}::zero(token),
                        hi: {half_name}::zero(token),
                    }}
                }}

                /// Store to array
                #[inline(always)]
                pub fn store(self, out: &mut [{elem_name}; {full_lanes}]) {{
                    let (lo, hi) = out.split_at_mut({half_lanes});
                    let lo_arr: &mut [{elem_name}; {half_lanes}] = lo.try_into().unwrap();
                    let hi_arr: &mut [{elem_name}; {half_lanes}] = hi.try_into().unwrap();
                    self.lo.store(lo_arr);
                    self.hi.store(hi_arr);
                }}

                /// Convert to array
                #[inline(always)]
                pub fn to_array(self) -> [{elem_name}; {full_lanes}] {{
                    let mut out = [{zero_lit}; {full_lanes}];
                    self.store(&mut out);
                    out
                }}

    "});

    // Math ops (same availability as W256)
    if has_min_max(elem) {
        code.push_str(&gen_lo_hi_binary("min", "Element-wise minimum"));
        code.push_str(&gen_lo_hi_binary("max", "Element-wise maximum"));
        code.push_str(&formatdoc! {"
                    /// Clamp values between lo and hi
                    #[inline(always)]
                    pub fn clamp(self, lo: Self, hi: Self) -> Self {{
                        self.max(lo).min(hi)
                    }}

        "});
    }

    if elem.is_float() {
        code.push_str(&gen_lo_hi_unary("sqrt", "Square root"));
        code.push_str(&gen_lo_hi_unary("floor", "Floor"));
        code.push_str(&gen_lo_hi_unary("ceil", "Ceil"));
        code.push_str(&gen_lo_hi_unary("round", "Round to nearest"));
        code.push_str(&formatdoc! {"
                    /// Fused multiply-add: self * a + b
                    #[inline(always)]
                    pub fn mul_add(self, a: Self, b: Self) -> Self {{
                        Self {{
                            lo: self.lo.mul_add(a.lo, b.lo),
                            hi: self.hi.mul_add(a.hi, b.hi),
                        }}
                    }}

        "});
    }

    if has_abs(elem) {
        code.push_str(&gen_lo_hi_unary("abs", "Absolute value"));
    }

    // reduce_add
    let reduce_add_body = if elem.is_float() {
        "self.lo.reduce_add() + self.hi.reduce_add()"
    } else {
        "self.lo.reduce_add().wrapping_add(self.hi.reduce_add())"
    };
    code.push_str(&formatdoc! {"
                /// Reduce: sum all lanes
                #[inline(always)]
                pub fn reduce_add(self) -> {elem_name} {{
                    {reduce_add_body}
                }}

    "});

    if has_reduce_minmax(elem) {
        code.push_str(&formatdoc! {"
                    /// Reduce: max of all lanes
                    #[inline(always)]
                    pub fn reduce_max(self) -> {elem_name} {{
                        self.lo.reduce_max().max(self.hi.reduce_max())
                    }}

                    /// Reduce: min of all lanes
                    #[inline(always)]
                    pub fn reduce_min(self) -> {elem_name} {{
                        self.lo.reduce_min().min(self.hi.reduce_min())
                    }}

        "});
    }

    // Bitcast operations
    code.push_str(
        &ops_bitcast::generate_polyfill_bitcasts(elem, SimdWidth::W512, SimdWidth::W256)
            .replace('\n', "\n    "),
    );

    // Implementation identification
    let tier_name = platform.tier_name;
    let impl_name = format!("polyfill::{tier_name}::{full_name}");
    code.push_str(&formatdoc! {"
                // ========== Implementation identification ==========

                /// Returns a string identifying this type's implementation.
                ///
                /// This is useful for verifying that the correct implementation is being used
                /// at compile time (via `-Ctarget-cpu`) or at runtime (via `#[magetypes]` dispatch).
                ///
                /// Returns `\"{impl_name}\"`.
                #[inline(always)]
                pub const fn implementation_name() -> &'static str {{
                    \"{impl_name}\"
                }}

            }}

    "});

    // Operator impls
    code.push_str(&gen_op_impl(&full_name, "Add", "add"));
    code.push_str(&gen_op_impl(&full_name, "Sub", "sub"));

    if has_mul(elem) {
        code.push_str(&gen_op_impl(&full_name, "Mul", "mul"));
    }

    if elem.is_float() {
        code.push_str(&gen_op_impl(&full_name, "Div", "div"));
    }

    if has_neg(elem) {
        code.push_str(&formatdoc! {"
                impl Neg for {full_name} {{
                    type Output = Self;
                    #[inline(always)]
                    fn neg(self) -> Self {{
                        Self {{
                            lo: -self.lo,
                            hi: -self.hi,
                        }}
                    }}
                }}

        "});
    }

    code.push_str(&gen_assign_op(&full_name, "AddAssign", "add_assign", "+"));
    code.push_str(&gen_assign_op(&full_name, "SubAssign", "sub_assign", "-"));

    if has_mul(elem) {
        code.push_str(&gen_assign_op(&full_name, "MulAssign", "mul_assign", "*"));
    }

    if elem.is_float() {
        code.push_str(&gen_assign_op(&full_name, "DivAssign", "div_assign", "/"));
    }

    code.push('\n');

    code
}

/// Generate a lo/hi delegating unary method.
fn gen_lo_hi_unary(name: &str, doc: &str) -> String {
    formatdoc! {"
                /// {doc}
                #[inline(always)]
                pub fn {name}(self) -> Self {{
                    Self {{
                        lo: self.lo.{name}(),
                        hi: self.hi.{name}(),
                    }}
                }}

    "}
}

/// Generate a lo/hi delegating binary method.
fn gen_lo_hi_binary(name: &str, doc: &str) -> String {
    formatdoc! {"
                /// {doc}
                #[inline(always)]
                pub fn {name}(self, other: Self) -> Self {{
                    Self {{
                        lo: self.lo.{name}(other.lo),
                        hi: self.hi.{name}(other.hi),
                    }}
                }}

    "}
}

/// Generate an operator impl (Add, Sub, Mul, Div).
fn gen_op_impl(full_name: &str, trait_name: &str, fn_name: &str) -> String {
    let op = match fn_name {
        "add" => "+",
        "sub" => "-",
        "mul" => "*",
        "div" => "/",
        _ => panic!("unknown op: {fn_name}"),
    };
    formatdoc! {"
            impl {trait_name} for {full_name} {{
                type Output = Self;
                #[inline(always)]
                fn {fn_name}(self, rhs: Self) -> Self {{
                    Self {{
                        lo: self.lo {op} rhs.lo,
                        hi: self.hi {op} rhs.hi,
                    }}
                }}
            }}

    "}
}

/// Generate an assign operator impl (AddAssign, SubAssign, etc.).
fn gen_assign_op(full_name: &str, trait_name: &str, fn_name: &str, op: &str) -> String {
    formatdoc! {"
            impl core::ops::{trait_name} for {full_name} {{
                #[inline(always)]
                fn {fn_name}(&mut self, rhs: Self) {{
                    *self = *self {op} rhs;
                }}
            }}

    "}
}

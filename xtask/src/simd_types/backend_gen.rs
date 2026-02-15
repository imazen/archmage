//! Backend trait + implementation generation for the generic SIMD strategy pattern.
//!
//! Generates:
//! - Backend trait definitions (e.g., `F32x8Backend`) in `backends/`
//! - Sealed trait in `backends/sealed.rs`
//! - Backend implementations for each token × type in `impls/`
//! - Module routing files (`backends/mod.rs`, `impls/mod.rs`)
//!
//! Currently supports float types (f32, f64) at all widths.
//! Integer types will be added in a future expansion.

use std::collections::BTreeMap;

use indoc::formatdoc;

// ============================================================================
// Data Model
// ============================================================================

/// A float vector type for backend generation.
#[derive(Clone, Debug)]
struct FloatVecType {
    /// Element type: "f32" or "f64"
    elem: &'static str,
    /// Number of lanes
    lanes: usize,
    /// Width in bits (128, 256, 512)
    width_bits: usize,
}

impl FloatVecType {
    /// Type name: "f32x8", "f64x4", etc.
    fn name(&self) -> String {
        format!("{}x{}", self.elem, self.lanes)
    }

    /// Trait name: "F32x8Backend", "F64x4Backend", etc.
    fn trait_name(&self) -> String {
        let upper_elem = match self.elem {
            "f32" => "F32",
            "f64" => "F64",
            _ => unreachable!(),
        };
        format!("{upper_elem}x{}Backend", self.lanes)
    }

    /// Array type for load/store: "[f32; 8]", "[f64; 4]", etc.
    fn array_type(&self) -> String {
        format!("[{}; {}]", self.elem, self.lanes)
    }

    /// x86 intrinsic prefix: "_mm", "_mm256", "_mm512"
    fn x86_prefix(&self) -> &'static str {
        match self.width_bits {
            128 => "_mm",
            256 => "_mm256",
            512 => "_mm512",
            _ => unreachable!(),
        }
    }

    /// x86 intrinsic suffix: "ps" for f32, "pd" for f64
    fn x86_suffix(&self) -> &'static str {
        match self.elem {
            "f32" => "ps",
            "f64" => "pd",
            _ => unreachable!(),
        }
    }

    /// x86 inner type: "__m128", "__m256", "__m512", "__m128d", etc.
    fn x86_inner_type(&self) -> &'static str {
        match (self.elem, self.width_bits) {
            ("f32", 128) => "__m128",
            ("f32", 256) => "__m256",
            ("f32", 512) => "__m512",
            ("f64", 128) => "__m128d",
            ("f64", 256) => "__m256d",
            ("f64", 512) => "__m512d",
            _ => unreachable!(),
        }
    }

    /// x86 token for this width
    #[allow(dead_code)]
    fn _x86_token(&self) -> &'static str {
        match self.width_bits {
            128 | 256 => "X64V3Token",
            512 => "X64V4Token",
            _ => unreachable!(),
        }
    }

    /// Whether this type is native on x86 V3 (AVX2+FMA)
    #[allow(dead_code)]
    fn _native_on_v3(&self) -> bool {
        self.width_bits <= 256
    }

    /// Whether this type is native on NEON (128-bit only)
    fn native_on_neon(&self) -> bool {
        self.width_bits == 128
    }

    /// Whether this type is native on WASM (128-bit only)
    fn native_on_wasm(&self) -> bool {
        self.width_bits == 128
    }

    /// NEON repr type
    fn neon_repr(&self) -> String {
        if self.native_on_neon() {
            match self.elem {
                "f32" => "float32x4_t".to_string(),
                "f64" => "float64x2_t".to_string(),
                _ => unreachable!(),
            }
        } else {
            let native = match self.elem {
                "f32" => "float32x4_t",
                "f64" => "float64x2_t",
                _ => unreachable!(),
            };
            let count = self.width_bits / 128;
            format!("[{native}; {count}]")
        }
    }

    /// WASM repr type
    fn wasm_repr(&self) -> String {
        if self.native_on_wasm() {
            "v128".to_string()
        } else {
            let count = self.width_bits / 128;
            format!("[v128; {count}]")
        }
    }

    /// Scalar repr type
    #[allow(dead_code)]
    fn _scalar_repr(&self) -> String {
        self.array_type()
    }

    /// NEON element intrinsic suffix: "f32" or "f64"
    fn neon_suffix(&self) -> &'static str {
        self.elem
    }

    /// WASM element prefix: "f32x4" or "f64x2"
    fn wasm_prefix(&self) -> &'static str {
        match self.elem {
            "f32" => "f32x4",
            "f64" => "f64x2",
            _ => unreachable!(),
        }
    }

    /// Number of 128-bit sub-vectors for polyfill
    fn sub_count(&self) -> usize {
        self.width_bits / 128
    }

    /// Whether f32 (has rcp/rsqrt approximation intrinsics on x86)
    #[allow(dead_code)]
    fn _has_native_approx(&self) -> bool {
        self.elem == "f32"
    }
}

/// All float vector types to generate backends for.
fn all_float_types() -> Vec<FloatVecType> {
    vec![
        FloatVecType { elem: "f32", lanes: 4, width_bits: 128 },
        FloatVecType { elem: "f32", lanes: 8, width_bits: 256 },
        FloatVecType { elem: "f64", lanes: 2, width_bits: 128 },
        FloatVecType { elem: "f64", lanes: 4, width_bits: 256 },
    ]
}

// ============================================================================
// Public Entry Point
// ============================================================================

/// Generate all backend trait definitions and implementations.
///
/// Returns a map of relative paths (under `magetypes/src/simd/`) to file contents.
pub fn generate_backend_files() -> BTreeMap<String, String> {
    let types = all_float_types();
    let mut files = BTreeMap::new();

    // 1. sealed.rs
    files.insert("backends/sealed.rs".to_string(), generate_sealed());

    // 2. Backend trait definitions
    for ty in &types {
        files.insert(
            format!("backends/{}.rs", ty.name()),
            generate_float_backend_trait(ty),
        );
    }

    // 3. backends/mod.rs
    files.insert("backends/mod.rs".to_string(), generate_backends_mod(&types));

    // 4. Implementation files
    files.insert("impls/x86_v3.rs".to_string(), generate_x86_impls(&types, "X64V3Token", 256));
    files.insert("impls/scalar.rs".to_string(), generate_scalar_impls(&types));
    files.insert("impls/arm_neon.rs".to_string(), generate_neon_impls(&types));
    files.insert("impls/wasm128.rs".to_string(), generate_wasm_impls(&types));

    // 5. impls/mod.rs
    files.insert("impls/mod.rs".to_string(), generate_impls_mod());

    files
}

// ============================================================================
// Sealed Trait
// ============================================================================

fn generate_sealed() -> String {
    formatdoc! {r#"
        //! Sealed trait to prevent external implementations of backend traits.
        //!
        //! **Auto-generated** by `cargo xtask generate` - do not edit manually.

        /// Sealed trait — only archmage token types can implement backend traits.
        pub trait Sealed {{}}

        impl Sealed for archmage::X64V1Token {{}}
        impl Sealed for archmage::X64V2Token {{}}
        impl Sealed for archmage::X64V3Token {{}}
        impl Sealed for archmage::X64V4Token {{}}
        impl Sealed for archmage::Avx512ModernToken {{}}
        impl Sealed for archmage::Avx512Fp16Token {{}}
        impl Sealed for archmage::NeonToken {{}}
        impl Sealed for archmage::NeonAesToken {{}}
        impl Sealed for archmage::NeonSha3Token {{}}
        impl Sealed for archmage::NeonCrcToken {{}}
        impl Sealed for archmage::Wasm128Token {{}}
        impl Sealed for archmage::ScalarToken {{}}
    "#}
}

// ============================================================================
// Backend Trait Definition Generation
// ============================================================================

fn generate_float_backend_trait(ty: &FloatVecType) -> String {
    let trait_name = ty.trait_name();
    let elem = ty.elem;
    let lanes = ty.lanes;
    let array = ty.array_type();

    formatdoc! {r#"
        //! Backend trait for `{name}<T>` — {lanes}-lane {elem} SIMD vector.
        //!
        //! Each token type implements this trait with its platform-native representation.
        //! The generic wrapper `{name}<T>` delegates all operations to these trait methods.
        //!
        //! **Auto-generated** by `cargo xtask generate` - do not edit manually.

        use super::sealed::Sealed;
        use archmage::SimdToken;

        /// Backend implementation for {lanes}-lane {elem} SIMD vectors.
        ///
        /// Trait methods are **associated functions** (no `self`/token parameter).
        /// The implementing type `Self` (a token type) determines which platform
        /// intrinsics are used. All methods are `#[inline(always)]` in implementations.
        ///
        /// # Sealed
        ///
        /// This trait is sealed — only archmage token types can implement it.
        /// The token proves CPU support was verified via `summon()`.
        pub trait {trait_name}: SimdToken + Sealed + Copy + 'static {{
            /// Platform-native SIMD representation.
            type Repr: Copy + Clone + Send + Sync;

            // ====== Construction ======

            /// Broadcast scalar to all {lanes} lanes.
            fn splat(v: {elem}) -> Self::Repr;

            /// All lanes zero.
            fn zero() -> Self::Repr;

            /// Load from an aligned array.
            fn load(data: &{array}) -> Self::Repr;

            /// Create from array (zero-cost transmute where possible).
            fn from_array(arr: {array}) -> Self::Repr;

            /// Store to array.
            fn store(repr: Self::Repr, out: &mut {array});

            /// Convert to array.
            fn to_array(repr: Self::Repr) -> {array};

            // ====== Arithmetic ======

            /// Lane-wise addition.
            fn add(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise subtraction.
            fn sub(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise multiplication.
            fn mul(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise division.
            fn div(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise negation.
            fn neg(a: Self::Repr) -> Self::Repr;

            // ====== Math ======

            /// Lane-wise minimum.
            fn min(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise maximum.
            fn max(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Square root.
            fn sqrt(a: Self::Repr) -> Self::Repr;

            /// Absolute value.
            fn abs(a: Self::Repr) -> Self::Repr;

            /// Round toward negative infinity.
            fn floor(a: Self::Repr) -> Self::Repr;

            /// Round toward positive infinity.
            fn ceil(a: Self::Repr) -> Self::Repr;

            /// Round to nearest integer.
            fn round(a: Self::Repr) -> Self::Repr;

            /// Fused multiply-add: a * b + c.
            fn mul_add(a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr;

            /// Fused multiply-sub: a * b - c.
            fn mul_sub(a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr;

            // ====== Comparisons ======
            // Return masks where each lane is all-1s (true) or all-0s (false).

            /// Lane-wise equality.
            fn simd_eq(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise inequality.
            fn simd_ne(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise less-than.
            fn simd_lt(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise less-than-or-equal.
            fn simd_le(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise greater-than.
            fn simd_gt(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise greater-than-or-equal.
            fn simd_ge(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Select lanes: where mask is all-1s pick `if_true`, else `if_false`.
            fn blend(mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr;

            // ====== Reductions ======

            /// Sum all {lanes} lanes.
            fn reduce_add(a: Self::Repr) -> {elem};

            /// Minimum across all {lanes} lanes.
            fn reduce_min(a: Self::Repr) -> {elem};

            /// Maximum across all {lanes} lanes.
            fn reduce_max(a: Self::Repr) -> {elem};

            // ====== Approximations ======

            /// Fast reciprocal approximation (~12-bit precision where available).
            ///
            /// On platforms without native approximation, falls back to full division.
            fn rcp_approx(a: Self::Repr) -> Self::Repr {{
                Self::div(Self::splat(1.0), a)
            }}

            /// Fast reciprocal square root approximation (~12-bit precision where available).
            ///
            /// On platforms without native approximation, falls back to 1/sqrt.
            fn rsqrt_approx(a: Self::Repr) -> Self::Repr {{
                Self::div(Self::splat(1.0), Self::sqrt(a))
            }}

            // ====== Bitwise ======

            /// Bitwise NOT.
            fn not(a: Self::Repr) -> Self::Repr;

            /// Bitwise AND.
            fn bitand(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Bitwise OR.
            fn bitor(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Bitwise XOR.
            fn bitxor(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            // ====== Default implementations ======

            /// Clamp values between lo and hi.
            #[inline(always)]
            fn clamp(a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {{
                Self::min(Self::max(a, lo), hi)
            }}

            /// Precise reciprocal (Newton-Raphson from rcp_approx).
            #[inline(always)]
            fn recip(a: Self::Repr) -> Self::Repr {{
                let approx = Self::rcp_approx(a);
                let two = Self::splat(2.0);
                // x' = x * (2 - a*x)
                Self::mul(approx, Self::sub(two, Self::mul(a, approx)))
            }}

            /// Precise reciprocal square root (Newton-Raphson from rsqrt_approx).
            #[inline(always)]
            fn rsqrt(a: Self::Repr) -> Self::Repr {{
                let approx = Self::rsqrt_approx(a);
                let half = Self::splat(0.5);
                let three = Self::splat(3.0);
                // y' = 0.5 * y * (3 - x * y * y)
                Self::mul(
                    Self::mul(half, approx),
                    Self::sub(three, Self::mul(a, Self::mul(approx, approx))),
                )
            }}
        }}
    "#,
        name = ty.name(),
    }
}

// ============================================================================
// backends/mod.rs Generation
// ============================================================================

fn generate_backends_mod(types: &[FloatVecType]) -> String {
    let mut code = formatdoc! {r#"
        //! Backend traits for generic SIMD types.
        //!
        //! **Auto-generated** by `cargo xtask generate` - do not edit manually.

        #![allow(non_camel_case_types)]

        pub(crate) mod sealed;

    "#};

    // Module declarations and re-exports
    for ty in types {
        let name = ty.name();
        let trait_name = ty.trait_name();
        code.push_str(&format!("mod {name};\npub use {name}::{trait_name};\n\n"));
    }

    // Type aliases for ergonomic use
    for (alias, full, doc) in [
        ("x64v1", "archmage::X64V1Token", "x86-64 baseline (SSE2)."),
        ("x64v2", "archmage::X64V2Token", "x86-64 v2 (SSE4.2 + POPCNT)."),
        ("x64v3", "archmage::X64V3Token", "x86-64 v3 (AVX2 + FMA)."),
        ("x64v4", "archmage::X64V4Token", "x86-64 v4 (AVX-512)."),
        (
            "avx512_modern",
            "archmage::Avx512ModernToken",
            "AVX-512 with modern extensions.",
        ),
        ("neon", "archmage::NeonToken", "AArch64 NEON."),
        ("wasm128", "archmage::Wasm128Token", "WASM SIMD128."),
        ("scalar", "archmage::ScalarToken", "Scalar fallback."),
    ] {
        code.push_str(&format!("/// {doc}\npub type {alias} = {full};\n"));
    }

    code
}

// ============================================================================
// impls/mod.rs Generation
// ============================================================================

fn generate_impls_mod() -> String {
    formatdoc! {r#"
        //! Backend trait implementations for each token type.
        //!
        //! Each file implements the backend traits (e.g., `F32x8Backend`) for one
        //! token, using that platform's native intrinsics.
        //!
        //! **Auto-generated** by `cargo xtask generate` - do not edit manually.

        #[cfg(target_arch = "x86_64")]
        mod x86_v3;

        #[cfg(target_arch = "aarch64")]
        mod arm_neon;

        #[cfg(target_arch = "wasm32")]
        mod wasm128;

        mod scalar;
    "#}
}

// ============================================================================
// x86 Implementation Generation
// ============================================================================

fn generate_x86_impls(types: &[FloatVecType], token: &str, max_width: usize) -> String {
    let mut code = formatdoc! {r#"
        //! Backend implementations for {token} (x86-64).
        //!
        //! **Auto-generated** by `cargo xtask generate` - do not edit manually.

        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;

        use crate::simd::backends::*;

    "#};

    for ty in types {
        if ty.width_bits > max_width {
            continue;
        }
        code.push_str("#[cfg(target_arch = \"x86_64\")]\n");
        code.push_str(&generate_x86_float_impl(ty, token));
        code.push('\n');
    }

    code
}

fn generate_x86_float_impl(ty: &FloatVecType, token: &str) -> String {
    let trait_name = ty.trait_name();
    let inner = ty.x86_inner_type();
    let p = ty.x86_prefix();
    let s = ty.x86_suffix();
    let bits = ty.width_bits;
    let array = ty.array_type();
    let elem = ty.elem;
    let lanes = ty.lanes;

    // set1 intrinsic: _mm256_set1_ps, _mm_set1_pd, etc.
    let set1 = format!("{p}_set1_{s}");
    // setzero: _mm256_setzero_ps, etc.
    let setzero = format!("{p}_setzero_{s}");
    // Cast si to float: _mm256_castsi256_ps
    let cast_int_to_float = format!("{p}_castsi{bits}_{s}");
    let cast_float_to_int = format!("{p}_cast{s}_si{bits}");
    // Integer set1 for abs mask
    let set1_int = if elem == "f32" { "epi32" } else if bits == 512 { "epi64" } else { "epi64x" };
    let abs_mask = if elem == "f32" { "0x7FFF_FFFFu32 as i32" } else { "0x7FFF_FFFF_FFFF_FFFFu64 as i64" };

    // Round intrinsics differ by width
    let (floor_intr, ceil_intr, round_intr) = if bits == 512 {
        (
            format!("{p}_roundscale_{s}::<0x01>"),
            format!("{p}_roundscale_{s}::<0x02>"),
            format!("{p}_roundscale_{s}::<0x00>"),
        )
    } else {
        (
            format!("{p}_floor_{s}"),
            format!("{p}_ceil_{s}"),
            format!("{p}_round_{s}::<{{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }}>"),
        )
    };

    // Reduction helpers
    let reduce_add_body = generate_x86_reduce_add(ty);
    let reduce_min_body = generate_x86_reduce_min(ty);
    let reduce_max_body = generate_x86_reduce_max(ty);

    // Comparison predicates
    let cmp = format!("{p}_cmp_{s}");

    // Approximation intrinsics (f32 only, or f64 on AVX-512)
    let (rcp_fn, rsqrt_fn) = if elem == "f32" {
        if bits == 512 { ("rcp14", "rsqrt14") } else { ("rcp", "rsqrt") }
    } else if bits == 512 {
        ("rcp14", "rsqrt14")
    } else {
        ("", "")
    };

    let approx_section = if !rcp_fn.is_empty() {
        formatdoc! {r#"
            #[inline(always)]
            fn rcp_approx(a: {inner}) -> {inner} {{
                unsafe {{ {p}_{rcp_fn}_{s}(a) }}
            }}

            #[inline(always)]
            fn rsqrt_approx(a: {inner}) -> {inner} {{
                unsafe {{ {p}_{rsqrt_fn}_{s}(a) }}
            }}
        "#}
    } else {
        String::new()
    };

    // Extract/cast for reduction helper types
    let extract_hi = match (elem, bits) {
        ("f32", 256) => format!("let hi = {p}_extractf128_ps::<1>(a);\n            let lo = {p}_castps256_ps128(a);"),
        ("f64", 256) => format!("let hi = {p}_extractf128_pd::<1>(a);\n            let lo = {p}_castpd256_pd128(a);"),
        ("f32", 128) | ("f64", 128) => String::new(), // No extraction needed
        _ => String::new(),
    };

    let _ = extract_hi; // Used in reduce_* bodies

    formatdoc! {r#"
        impl {trait_name} for archmage::{token} {{
            type Repr = {inner};

            // ====== Construction ======

            #[inline(always)]
            fn splat(v: {elem}) -> {inner} {{
                unsafe {{ {set1}(v) }}
            }}

            #[inline(always)]
            fn zero() -> {inner} {{
                unsafe {{ {setzero}() }}
            }}

            #[inline(always)]
            fn load(data: &{array}) -> {inner} {{
                unsafe {{ {p}_loadu_{s}(data.as_ptr()) }}
            }}

            #[inline(always)]
            fn from_array(arr: {array}) -> {inner} {{
                // SAFETY: {array} and {inner} have identical size and layout.
                unsafe {{ core::mem::transmute(arr) }}
            }}

            #[inline(always)]
            fn store(repr: {inner}, out: &mut {array}) {{
                unsafe {{ {p}_storeu_{s}(out.as_mut_ptr(), repr) }};
            }}

            #[inline(always)]
            fn to_array(repr: {inner}) -> {array} {{
                let mut out = [{zero_lit}; {lanes}];
                unsafe {{ {p}_storeu_{s}(out.as_mut_ptr(), repr) }};
                out
            }}

            // ====== Arithmetic ======

            #[inline(always)]
            fn add(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_add_{s}(a, b) }}
            }}

            #[inline(always)]
            fn sub(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_sub_{s}(a, b) }}
            }}

            #[inline(always)]
            fn mul(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_mul_{s}(a, b) }}
            }}

            #[inline(always)]
            fn div(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_div_{s}(a, b) }}
            }}

            #[inline(always)]
            fn neg(a: {inner}) -> {inner} {{
                unsafe {{ {p}_sub_{s}({setzero}(), a) }}
            }}

            // ====== Math ======

            #[inline(always)]
            fn min(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_min_{s}(a, b) }}
            }}

            #[inline(always)]
            fn max(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_max_{s}(a, b) }}
            }}

            #[inline(always)]
            fn sqrt(a: {inner}) -> {inner} {{
                unsafe {{ {p}_sqrt_{s}(a) }}
            }}

            #[inline(always)]
            fn abs(a: {inner}) -> {inner} {{
                unsafe {{
                    let mask = {cast_int_to_float}({p}_set1_{set1_int}({abs_mask}));
                    {p}_and_{s}(a, mask)
                }}
            }}

            #[inline(always)]
            fn floor(a: {inner}) -> {inner} {{
                unsafe {{ {floor_intr}(a) }}
            }}

            #[inline(always)]
            fn ceil(a: {inner}) -> {inner} {{
                unsafe {{ {ceil_intr}(a) }}
            }}

            #[inline(always)]
            fn round(a: {inner}) -> {inner} {{
                unsafe {{ {round_intr}(a) }}
            }}

            #[inline(always)]
            fn mul_add(a: {inner}, b: {inner}, c: {inner}) -> {inner} {{
                unsafe {{ {p}_fmadd_{s}(a, b, c) }}
            }}

            #[inline(always)]
            fn mul_sub(a: {inner}, b: {inner}, c: {inner}) -> {inner} {{
                unsafe {{ {p}_fmsub_{s}(a, b, c) }}
            }}

            // ====== Comparisons ======

            #[inline(always)]
            fn simd_eq(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {cmp}::<_CMP_EQ_OQ>(a, b) }}
            }}

            #[inline(always)]
            fn simd_ne(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {cmp}::<_CMP_NEQ_OQ>(a, b) }}
            }}

            #[inline(always)]
            fn simd_lt(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {cmp}::<_CMP_LT_OQ>(a, b) }}
            }}

            #[inline(always)]
            fn simd_le(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {cmp}::<_CMP_LE_OQ>(a, b) }}
            }}

            #[inline(always)]
            fn simd_gt(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {cmp}::<_CMP_GT_OQ>(a, b) }}
            }}

            #[inline(always)]
            fn simd_ge(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {cmp}::<_CMP_GE_OQ>(a, b) }}
            }}

            #[inline(always)]
            fn blend(mask: {inner}, if_true: {inner}, if_false: {inner}) -> {inner} {{
                unsafe {{ {p}_blendv_{s}(if_false, if_true, mask) }}
            }}

            // ====== Reductions ======

            #[inline(always)]
            fn reduce_add(a: {inner}) -> {elem} {{
        {reduce_add_body}
            }}

            #[inline(always)]
            fn reduce_min(a: {inner}) -> {elem} {{
        {reduce_min_body}
            }}

            #[inline(always)]
            fn reduce_max(a: {inner}) -> {elem} {{
        {reduce_max_body}
            }}

            // ====== Approximations ======

        {approx_section}
            // ====== Bitwise ======

            #[inline(always)]
            fn not(a: {inner}) -> {inner} {{
                unsafe {{
                    let ones = {p}_set1_{set1_int}(-1);
                    let as_int = {cast_float_to_int}(a);
                    {cast_int_to_float}({p}_xor_si{bits}(as_int, ones))
                }}
            }}

            #[inline(always)]
            fn bitand(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_and_{s}(a, b) }}
            }}

            #[inline(always)]
            fn bitor(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_or_{s}(a, b) }}
            }}

            #[inline(always)]
            fn bitxor(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_xor_{s}(a, b) }}
            }}
        }}
    "#,
        zero_lit = if elem == "f32" { "0.0f32" } else { "0.0f64" },
    }
}

fn generate_x86_reduce_add(ty: &FloatVecType) -> String {
    let p = ty.x86_prefix();
    let s = ty.x86_suffix();
    let cvt = if ty.elem == "f32" { "_mm_cvtss_f32" } else { "_mm_cvtsd_f64" };
    let hadd = if ty.elem == "f32" { "_mm_hadd_ps" } else { "_mm_hadd_pd" };

    match (ty.elem, ty.width_bits) {
        ("f32", 128) => formatdoc! {"
                unsafe {{
                    let h1 = {hadd}(a, a);
                    let h2 = {hadd}(h1, h1);
                    {cvt}(h2)
                }}"},
        ("f64", 128) => formatdoc! {"
                unsafe {{
                    let h = {hadd}(a, a);
                    {cvt}(h)
                }}"},
        ("f32", 256) => formatdoc! {"
                unsafe {{
                    let hi = {p}_extractf128_{s}::<1>(a);
                    let lo = {p}_cast{s}256_{s}128(a);
                    let sum = _mm_add_{s}(lo, hi);
                    let h1 = {hadd}(sum, sum);
                    let h2 = {hadd}(h1, h1);
                    {cvt}(h2)
                }}"},
        ("f64", 256) => formatdoc! {"
                unsafe {{
                    let hi = {p}_extractf128_{s}::<1>(a);
                    let lo = {p}_cast{s}256_{s}128(a);
                    let sum = _mm_add_{s}(lo, hi);
                    let h = {hadd}(sum, sum);
                    {cvt}(h)
                }}"},
        ("f32", 512) => "        unsafe { _mm512_reduce_add_ps(a) }".to_string(),
        ("f64", 512) => "        unsafe { _mm512_reduce_add_pd(a) }".to_string(),
        _ => unreachable!(),
    }
}

fn generate_x86_reduce_min(ty: &FloatVecType) -> String {
    let p = ty.x86_prefix();
    let s = ty.x86_suffix();
    let cvt = if ty.elem == "f32" { "_mm_cvtss_f32" } else { "_mm_cvtsd_f64" };

    match (ty.elem, ty.width_bits) {
        ("f32", 128) => formatdoc! {"
                unsafe {{
                    let shuf = _mm_shuffle_ps::<0b10_11_00_01>(a, a);
                    let m1 = _mm_min_ps(a, shuf);
                    let shuf2 = _mm_shuffle_ps::<0b00_00_10_10>(m1, m1);
                    let m2 = _mm_min_ps(m1, shuf2);
                    {cvt}(m2)
                }}"},
        ("f64", 128) => formatdoc! {"
                unsafe {{
                    let shuf = _mm_shuffle_pd::<0b01>(a, a);
                    let m = _mm_min_pd(a, shuf);
                    {cvt}(m)
                }}"},
        ("f32", 256) => formatdoc! {"
                unsafe {{
                    let hi = {p}_extractf128_{s}::<1>(a);
                    let lo = {p}_cast{s}256_{s}128(a);
                    let m = _mm_min_{s}(lo, hi);
                    let shuf = _mm_shuffle_ps::<0b10_11_00_01>(m, m);
                    let m1 = _mm_min_ps(m, shuf);
                    let shuf2 = _mm_shuffle_ps::<0b00_00_10_10>(m1, m1);
                    let m2 = _mm_min_ps(m1, shuf2);
                    {cvt}(m2)
                }}"},
        ("f64", 256) => formatdoc! {"
                unsafe {{
                    let hi = {p}_extractf128_{s}::<1>(a);
                    let lo = {p}_cast{s}256_{s}128(a);
                    let m = _mm_min_{s}(lo, hi);
                    let shuf = _mm_shuffle_pd::<0b01>(m, m);
                    let m2 = _mm_min_pd(m, shuf);
                    {cvt}(m2)
                }}"},
        ("f32", 512) => "        unsafe { _mm512_reduce_min_ps(a) }".to_string(),
        ("f64", 512) => "        unsafe { _mm512_reduce_min_pd(a) }".to_string(),
        _ => unreachable!(),
    }
}

fn generate_x86_reduce_max(ty: &FloatVecType) -> String {
    let p = ty.x86_prefix();
    let s = ty.x86_suffix();
    let cvt = if ty.elem == "f32" { "_mm_cvtss_f32" } else { "_mm_cvtsd_f64" };

    match (ty.elem, ty.width_bits) {
        ("f32", 128) => formatdoc! {"
                unsafe {{
                    let shuf = _mm_shuffle_ps::<0b10_11_00_01>(a, a);
                    let m1 = _mm_max_ps(a, shuf);
                    let shuf2 = _mm_shuffle_ps::<0b00_00_10_10>(m1, m1);
                    let m2 = _mm_max_ps(m1, shuf2);
                    {cvt}(m2)
                }}"},
        ("f64", 128) => formatdoc! {"
                unsafe {{
                    let shuf = _mm_shuffle_pd::<0b01>(a, a);
                    let m = _mm_max_pd(a, shuf);
                    {cvt}(m)
                }}"},
        ("f32", 256) => formatdoc! {"
                unsafe {{
                    let hi = {p}_extractf128_{s}::<1>(a);
                    let lo = {p}_cast{s}256_{s}128(a);
                    let m = _mm_max_{s}(lo, hi);
                    let shuf = _mm_shuffle_ps::<0b10_11_00_01>(m, m);
                    let m1 = _mm_max_ps(m, shuf);
                    let shuf2 = _mm_shuffle_ps::<0b00_00_10_10>(m1, m1);
                    let m2 = _mm_max_ps(m1, shuf2);
                    {cvt}(m2)
                }}"},
        ("f64", 256) => formatdoc! {"
                unsafe {{
                    let hi = {p}_extractf128_{s}::<1>(a);
                    let lo = {p}_cast{s}256_{s}128(a);
                    let m = _mm_max_{s}(lo, hi);
                    let shuf = _mm_shuffle_pd::<0b01>(m, m);
                    let m2 = _mm_max_pd(m, shuf);
                    {cvt}(m2)
                }}"},
        ("f32", 512) => "        unsafe { _mm512_reduce_max_ps(a) }".to_string(),
        ("f64", 512) => "        unsafe { _mm512_reduce_max_pd(a) }".to_string(),
        _ => unreachable!(),
    }
}

// ============================================================================
// Scalar Implementation Generation
// ============================================================================

fn generate_scalar_impls(types: &[FloatVecType]) -> String {
    let mut code = formatdoc! {r#"
        //! Backend implementations for ScalarToken (fallback).
        //!
        //! All operations are plain array math. Always available on all platforms.
        //!
        //! **Auto-generated** by `cargo xtask generate` - do not edit manually.

        use crate::simd::backends::*;

    "#};

    // Emit helper functions once per element type (not per vector type)
    let mut seen_elems = std::collections::BTreeSet::new();
    for ty in types {
        if seen_elems.insert(ty.elem) {
            code.push_str(&formatdoc! {r#"
                // Helpers to avoid trait method name shadowing inside the impl block.
                // Inside `impl XxxBackend`, names like `sqrt`, `floor`, etc. resolve to
                // the trait's associated functions instead of {elem}'s inherent methods.
                #[inline(always)]
                fn {elem}_sqrt(x: {elem}) -> {elem} {{
                    x.sqrt()
                }}
                #[inline(always)]
                fn {elem}_floor(x: {elem}) -> {elem} {{
                    x.floor()
                }}
                #[inline(always)]
                fn {elem}_ceil(x: {elem}) -> {elem} {{
                    x.ceil()
                }}
                #[inline(always)]
                fn {elem}_round(x: {elem}) -> {elem} {{
                    x.round()
                }}

            "#, elem = ty.elem});
        }
    }

    for ty in types {
        code.push_str(&generate_scalar_float_impl(ty));
        code.push('\n');
    }

    code
}

fn generate_scalar_float_impl(ty: &FloatVecType) -> String {
    let trait_name = ty.trait_name();
    let elem = ty.elem;
    let lanes = ty.lanes;
    let array = ty.array_type();
    let zero_lit = if elem == "f32" { "0.0f32" } else { "0.0f64" };

    // Generate lane-by-lane operations
    let binary_lanes = |op: &str| -> String {
        let items: Vec<String> = (0..lanes).map(|i| format!("a[{i}] {op} b[{i}]")).collect();
        format!("[{}]", items.join(", "))
    };

    let min_lanes_fn = || -> String {
        let items: Vec<String> = (0..lanes).map(|i| format!("a[{i}].min(b[{i}])")).collect();
        format!("[{}]", items.join(", "))
    };

    let max_lanes_fn = || -> String {
        let items: Vec<String> = (0..lanes).map(|i| format!("a[{i}].max(b[{i}])")).collect();
        format!("[{}]", items.join(", "))
    };

    let neg_lanes = || -> String {
        let items: Vec<String> = (0..lanes).map(|i| format!("-a[{i}]")).collect();
        format!("[{}]", items.join(", "))
    };

    let abs_mask = if elem == "f32" { "0x7FFF_FFFF" } else { "0x7FFF_FFFF_FFFF_FFFF" };
    let abs_lanes = || -> String {
        let items: Vec<String> = (0..lanes)
            .map(|i| format!("{elem}::from_bits(a[{i}].to_bits() & {abs_mask})"))
            .collect();
        format!("[{}]", items.join(", "))
    };

    let mul_add_lanes = || -> String {
        let items: Vec<String> = (0..lanes)
            .map(|i| format!("a[{i}] * b[{i}] + c[{i}]"))
            .collect();
        format!("[{}]", items.join(", "))
    };

    let mul_sub_lanes = || -> String {
        let items: Vec<String> = (0..lanes)
            .map(|i| format!("a[{i}] * b[{i}] - c[{i}]"))
            .collect();
        format!("[{}]", items.join(", "))
    };

    let bitwise_unary_lanes = |op: &str| -> String {
        let items: Vec<String> = (0..lanes)
            .map(|i| format!("{elem}::from_bits({op}a[{i}].to_bits())"))
            .collect();
        format!("[{}]", items.join(", "))
    };

    let bitwise_binary_lanes = |op: &str| -> String {
        let items: Vec<String> = (0..lanes)
            .map(|i| format!("{elem}::from_bits(a[{i}].to_bits() {op} b[{i}].to_bits())"))
            .collect();
        format!("[{}]", items.join(", "))
    };

    let reduce_add = || -> String {
        let items: Vec<String> = (0..lanes).map(|i| format!("a[{i}]")).collect();
        items.join(" + ")
    };

    // Helper functions (f32_sqrt etc.) are emitted once per element type
    // at the file level by generate_scalar_impls, not per-type.

    let true_mask = if elem == "f32" {
        "f32::from_bits(0xFFFF_FFFF)"
    } else {
        "f64::from_bits(0xFFFF_FFFF_FFFF_FFFF)"
    };

    let sign_shift = if elem == "f32" { "31" } else { "63" };

    formatdoc! {r#"
        impl {trait_name} for archmage::ScalarToken {{
            type Repr = {array};

            // ====== Construction ======

            #[inline(always)]
            fn splat(v: {elem}) -> {array} {{
                [v; {lanes}]
            }}

            #[inline(always)]
            fn zero() -> {array} {{
                [{zero_lit}; {lanes}]
            }}

            #[inline(always)]
            fn load(data: &{array}) -> {array} {{
                *data
            }}

            #[inline(always)]
            fn from_array(arr: {array}) -> {array} {{
                arr
            }}

            #[inline(always)]
            fn store(repr: {array}, out: &mut {array}) {{
                *out = repr;
            }}

            #[inline(always)]
            fn to_array(repr: {array}) -> {array} {{
                repr
            }}

            // ====== Arithmetic ======

            #[inline(always)]
            fn add(a: {array}, b: {array}) -> {array} {{
                {add_lanes}
            }}

            #[inline(always)]
            fn sub(a: {array}, b: {array}) -> {array} {{
                {sub_lanes}
            }}

            #[inline(always)]
            fn mul(a: {array}, b: {array}) -> {array} {{
                {mul_lanes}
            }}

            #[inline(always)]
            fn div(a: {array}, b: {array}) -> {array} {{
                {div_lanes}
            }}

            #[inline(always)]
            fn neg(a: {array}) -> {array} {{
                {neg}
            }}

            // ====== Math ======

            #[inline(always)]
            fn min(a: {array}, b: {array}) -> {array} {{
                {min_lanes}
            }}

            #[inline(always)]
            fn max(a: {array}, b: {array}) -> {array} {{
                {max_lanes}
            }}

            #[inline(always)]
            fn sqrt(a: {array}) -> {array} {{
                let mut r = [{zero_lit}; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = {elem}_sqrt(a[i]);
                }}
                r
            }}

            #[inline(always)]
            fn abs(a: {array}) -> {array} {{
                {abs}
            }}

            #[inline(always)]
            fn floor(a: {array}) -> {array} {{
                let mut r = [{zero_lit}; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = {elem}_floor(a[i]);
                }}
                r
            }}

            #[inline(always)]
            fn ceil(a: {array}) -> {array} {{
                let mut r = [{zero_lit}; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = {elem}_ceil(a[i]);
                }}
                r
            }}

            #[inline(always)]
            fn round(a: {array}) -> {array} {{
                let mut r = [{zero_lit}; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = {elem}_round(a[i]);
                }}
                r
            }}

            #[inline(always)]
            fn mul_add(a: {array}, b: {array}, c: {array}) -> {array} {{
                {mul_add}
            }}

            #[inline(always)]
            fn mul_sub(a: {array}, b: {array}, c: {array}) -> {array} {{
                {mul_sub}
            }}

            // ====== Comparisons ======

            #[inline(always)]
            fn simd_eq(a: {array}, b: {array}) -> {array} {{
                let mut r = [{zero_lit}; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if a[i] == b[i] {{ {true_mask} }} else {{ 0.0 }};
                }}
                r
            }}

            #[inline(always)]
            fn simd_ne(a: {array}, b: {array}) -> {array} {{
                let mut r = [{zero_lit}; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if a[i] != b[i] {{ {true_mask} }} else {{ 0.0 }};
                }}
                r
            }}

            #[inline(always)]
            fn simd_lt(a: {array}, b: {array}) -> {array} {{
                let mut r = [{zero_lit}; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if a[i] < b[i] {{ {true_mask} }} else {{ 0.0 }};
                }}
                r
            }}

            #[inline(always)]
            fn simd_le(a: {array}, b: {array}) -> {array} {{
                let mut r = [{zero_lit}; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if a[i] <= b[i] {{ {true_mask} }} else {{ 0.0 }};
                }}
                r
            }}

            #[inline(always)]
            fn simd_gt(a: {array}, b: {array}) -> {array} {{
                let mut r = [{zero_lit}; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if a[i] > b[i] {{ {true_mask} }} else {{ 0.0 }};
                }}
                r
            }}

            #[inline(always)]
            fn simd_ge(a: {array}, b: {array}) -> {array} {{
                let mut r = [{zero_lit}; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if a[i] >= b[i] {{ {true_mask} }} else {{ 0.0 }};
                }}
                r
            }}

            #[inline(always)]
            fn blend(mask: {array}, if_true: {array}, if_false: {array}) -> {array} {{
                let mut r = [{zero_lit}; {lanes}];
                for i in 0..{lanes} {{
                    // Check sign bit of mask (all-1s has sign bit set)
                    r[i] = if (mask[i].to_bits() >> {sign_shift}) != 0 {{
                        if_true[i]
                    }} else {{
                        if_false[i]
                    }};
                }}
                r
            }}

            // ====== Reductions ======

            #[inline(always)]
            fn reduce_add(a: {array}) -> {elem} {{
                {reduce_add}
            }}

            #[inline(always)]
            fn reduce_min(a: {array}) -> {elem} {{
                let mut m = a[0];
                for &v in &a[1..] {{
                    m = m.min(v);
                }}
                m
            }}

            #[inline(always)]
            fn reduce_max(a: {array}) -> {elem} {{
                let mut m = a[0];
                for &v in &a[1..] {{
                    m = m.max(v);
                }}
                m
            }}

            // ====== Approximations ======

            #[inline(always)]
            fn rcp_approx(a: {array}) -> {array} {{
                {rcp_lanes}
            }}

            #[inline(always)]
            fn rsqrt_approx(a: {array}) -> {array} {{
                let mut r = [{zero_lit}; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = 1.0 / {elem}_sqrt(a[i]);
                }}
                r
            }}

            // Override defaults: scalar doesn't need Newton-Raphson (already full precision)
            // Use FQS because ScalarToken implements multiple backend traits.
            #[inline(always)]
            fn recip(a: {array}) -> {array} {{
                <archmage::ScalarToken as {trait_name}>::rcp_approx(a)
            }}

            #[inline(always)]
            fn rsqrt(a: {array}) -> {array} {{
                <archmage::ScalarToken as {trait_name}>::rsqrt_approx(a)
            }}

            // ====== Bitwise ======

            #[inline(always)]
            fn not(a: {array}) -> {array} {{
                {not_lanes}
            }}

            #[inline(always)]
            fn bitand(a: {array}, b: {array}) -> {array} {{
                {and_lanes}
            }}

            #[inline(always)]
            fn bitor(a: {array}, b: {array}) -> {array} {{
                {or_lanes}
            }}

            #[inline(always)]
            fn bitxor(a: {array}, b: {array}) -> {array} {{
                {xor_lanes}
            }}
        }}
    "#,
        add_lanes = binary_lanes("+"),
        sub_lanes = binary_lanes("-"),
        mul_lanes = binary_lanes("*"),
        div_lanes = binary_lanes("/"),
        neg = neg_lanes(),
        min_lanes = min_lanes_fn(),
        max_lanes = max_lanes_fn(),
        // Actually, min/max need a different pattern...
        abs = abs_lanes(),
        mul_add = mul_add_lanes(),
        mul_sub = mul_sub_lanes(),
        reduce_add = reduce_add(),
        rcp_lanes = {
            let items: Vec<String> = (0..lanes).map(|i| format!("1.0 / a[{i}]")).collect();
            format!("[{}]", items.join(", "))
        },
        not_lanes = bitwise_unary_lanes("!"),
        and_lanes = bitwise_binary_lanes("&"),
        or_lanes = bitwise_binary_lanes("|"),
        xor_lanes = bitwise_binary_lanes("^"),
    }
}

// ============================================================================
// NEON Implementation Generation
// ============================================================================

fn generate_neon_impls(types: &[FloatVecType]) -> String {
    let mut code = formatdoc! {r#"
        //! Backend implementations for NeonToken (AArch64 NEON).
        //!
        //! **Auto-generated** by `cargo xtask generate` - do not edit manually.

        #[cfg(target_arch = "aarch64")]
        use core::arch::aarch64::*;

        use crate::simd::backends::*;

    "#};

    for ty in types {
        code.push_str("#[cfg(target_arch = \"aarch64\")]\n");
        code.push_str(&generate_neon_float_impl(ty));
        code.push('\n');
    }

    code
}

fn generate_neon_float_impl(ty: &FloatVecType) -> String {
    let trait_name = ty.trait_name();
    let repr = ty.neon_repr();
    let elem = ty.elem;
    let lanes = ty.lanes;
    let array = ty.array_type();
    let ns = ty.neon_suffix();
    let zero_lit = if elem == "f32" { "0.0f32" } else { "0.0f64" };
    let sub_count = ty.sub_count();

    // NEON native type names
    let _native_type = match elem {
        "f32" => "float32x4_t",
        "f64" => "float64x2_t",
        _ => unreachable!(),
    };
    let native_lanes = if elem == "f32" { 4 } else { 2 };

    // If this is a native 128-bit type, no polyfill needed
    if ty.native_on_neon() {
        return generate_neon_native_impl(ty);
    }

    // Polyfill: apply operation to each sub-vector
    let binary_op = |intrinsic: &str| -> String {
        let items: Vec<String> = (0..sub_count)
            .map(|i| format!("{intrinsic}(a[{i}], b[{i}])"))
            .collect();
        format!("unsafe {{ [{}] }}", items.join(", "))
    };

    let unary_op = |intrinsic: &str| -> String {
        let items: Vec<String> = (0..sub_count)
            .map(|i| format!("{intrinsic}(a[{i}])"))
            .collect();
        format!("unsafe {{ [{}] }}", items.join(", "))
    };

    let cmp_op = |intrinsic: &str| -> String {
        let items: Vec<String> = (0..sub_count)
            .map(|i| format!("vreinterpretq_{ns}_u{elem_bits}({intrinsic}(a[{i}], b[{i}]))",
                elem_bits = if elem == "f32" { 32 } else { 64 }))
            .collect();
        format!("unsafe {{ [{}] }}", items.join(", "))
    };

    let ne_op = || -> String {
        let items: Vec<String> = (0..sub_count)
            .map(|i| format!("vreinterpretq_{ns}_u{eb}(vmvnq_u{eb}(vceqq_{ns}(a[{i}], b[{i}])))",
                eb = if elem == "f32" { 32 } else { 64 },
                ns = ns))
            .collect();
        format!("unsafe {{ [{}] }}", items.join(", "))
    };

    let blend_op = || -> String {
        let eb = if elem == "f32" { 32 } else { 64 };
        let items: Vec<String> = (0..sub_count)
            .map(|i| format!("vbslq_{ns}(vreinterpretq_u{eb}_{ns}(mask[{i}]), if_true[{i}], if_false[{i}])"))
            .collect();
        format!("unsafe {{ [{}] }}", items.join(", "))
    };

    let bitwise_not_op = || -> String {
        let eb = if elem == "f32" { 32 } else { 64 };
        let items: Vec<String> = (0..sub_count)
            .map(|i| format!("vreinterpretq_{ns}_u{eb}(vmvnq_u{eb}(vreinterpretq_u{eb}_{ns}(a[{i}])))"))
            .collect();
        format!("unsafe {{ [{}] }}", items.join(", "))
    };

    let bitwise_binary_op = |neon_op: &str| -> String {
        let eb = if elem == "f32" { 32 } else { 64 };
        let items: Vec<String> = (0..sub_count)
            .map(|i| format!("vreinterpretq_{ns}_u{eb}({neon_op}(vreinterpretq_u{eb}_{ns}(a[{i}]), vreinterpretq_u{eb}_{ns}(b[{i}])))"))
            .collect();
        format!("unsafe {{ [{}] }}", items.join(", "))
    };

    // Reduction: combine halves then reduce
    let reduce_combine = |combine_intrinsic: &str, pairwise: &str| -> String {
        let mut body = String::from("unsafe {\n");
        body.push_str(&format!("            let m = {combine_intrinsic}(a[0], a[1]);\n"));
        // For wider types, combine more
        for i in 2..sub_count {
            body.push_str(&format!("            let m = {combine_intrinsic}(m, a[{i}]);\n"));
        }
        body.push_str(&format!("            let pair = {pairwise}(m, m);\n"));
        if native_lanes > 2 {
            body.push_str(&format!("            let pair = {pairwise}(pair, pair);\n"));
        }
        body.push_str(&format!("            vgetq_lane_{ns}::<0>(pair)\n"));
        body.push_str("        }");
        body
    };

    formatdoc! {r#"
        impl {trait_name} for archmage::NeonToken {{
            type Repr = {repr};

            // ====== Construction ======

            #[inline(always)]
            fn splat(v: {elem}) -> {repr} {{
                unsafe {{
                    let v4 = vdupq_n_{ns}(v);
                    [{v4_copies}]
                }}
            }}

            #[inline(always)]
            fn zero() -> {repr} {{
                unsafe {{
                    let z = vdupq_n_{ns}(0.0);
                    [{z_copies}]
                }}
            }}

            #[inline(always)]
            fn load(data: &{array}) -> {repr} {{
                unsafe {{
                    [{load_lanes}]
                }}
            }}

            #[inline(always)]
            fn from_array(arr: {array}) -> {repr} {{
                Self::load(&arr)
            }}

            #[inline(always)]
            fn store(repr: {repr}, out: &mut {array}) {{
                unsafe {{
                    {store_lanes}
                }}
            }}

            #[inline(always)]
            fn to_array(repr: {repr}) -> {array} {{
                let mut out = [{zero_lit}; {lanes}];
                Self::store(repr, &mut out);
                out
            }}

            // ====== Arithmetic ======

            #[inline(always)]
            fn add(a: {repr}, b: {repr}) -> {repr} {{
                {add_body}
            }}

            #[inline(always)]
            fn sub(a: {repr}, b: {repr}) -> {repr} {{
                {sub_body}
            }}

            #[inline(always)]
            fn mul(a: {repr}, b: {repr}) -> {repr} {{
                {mul_body}
            }}

            #[inline(always)]
            fn div(a: {repr}, b: {repr}) -> {repr} {{
                {div_body}
            }}

            #[inline(always)]
            fn neg(a: {repr}) -> {repr} {{
                {neg_body}
            }}

            // ====== Math ======

            #[inline(always)]
            fn min(a: {repr}, b: {repr}) -> {repr} {{
                {min_body}
            }}

            #[inline(always)]
            fn max(a: {repr}, b: {repr}) -> {repr} {{
                {max_body}
            }}

            #[inline(always)]
            fn sqrt(a: {repr}) -> {repr} {{
                {sqrt_body}
            }}

            #[inline(always)]
            fn abs(a: {repr}) -> {repr} {{
                {abs_body}
            }}

            #[inline(always)]
            fn floor(a: {repr}) -> {repr} {{
                {floor_body}
            }}

            #[inline(always)]
            fn ceil(a: {repr}) -> {repr} {{
                {ceil_body}
            }}

            #[inline(always)]
            fn round(a: {repr}) -> {repr} {{
                {round_body}
            }}

            #[inline(always)]
            fn mul_add(a: {repr}, b: {repr}, c: {repr}) -> {repr} {{
                // vfmaq = acc + x*y, so mul_add(a, b, c) = a*b + c => vfmaq(c, a, b)
                {mul_add_body}
            }}

            #[inline(always)]
            fn mul_sub(a: {repr}, b: {repr}, c: {repr}) -> {repr} {{
                // a*b - c => vfmaq(-c, a, b) = -c + a*b
                {mul_sub_body}
            }}

            // ====== Comparisons ======

            #[inline(always)]
            fn simd_eq(a: {repr}, b: {repr}) -> {repr} {{
                {eq_body}
            }}

            #[inline(always)]
            fn simd_ne(a: {repr}, b: {repr}) -> {repr} {{
                {ne_body}
            }}

            #[inline(always)]
            fn simd_lt(a: {repr}, b: {repr}) -> {repr} {{
                {lt_body}
            }}

            #[inline(always)]
            fn simd_le(a: {repr}, b: {repr}) -> {repr} {{
                {le_body}
            }}

            #[inline(always)]
            fn simd_gt(a: {repr}, b: {repr}) -> {repr} {{
                {gt_body}
            }}

            #[inline(always)]
            fn simd_ge(a: {repr}, b: {repr}) -> {repr} {{
                {ge_body}
            }}

            #[inline(always)]
            fn blend(mask: {repr}, if_true: {repr}, if_false: {repr}) -> {repr} {{
                {blend_body}
            }}

            // ====== Reductions ======

            #[inline(always)]
            fn reduce_add(a: {repr}) -> {elem} {{
                {reduce_add_body}
            }}

            #[inline(always)]
            fn reduce_min(a: {repr}) -> {elem} {{
                {reduce_min_body}
            }}

            #[inline(always)]
            fn reduce_max(a: {repr}) -> {elem} {{
                {reduce_max_body}
            }}

            // ====== Approximations ======

            #[inline(always)]
            fn rcp_approx(a: {repr}) -> {repr} {{
                {rcp_body}
            }}

            #[inline(always)]
            fn rsqrt_approx(a: {repr}) -> {repr} {{
                {rsqrt_body}
            }}

            // ====== Bitwise ======

            #[inline(always)]
            fn not(a: {repr}) -> {repr} {{
                {not_body}
            }}

            #[inline(always)]
            fn bitand(a: {repr}, b: {repr}) -> {repr} {{
                {and_body}
            }}

            #[inline(always)]
            fn bitor(a: {repr}, b: {repr}) -> {repr} {{
                {or_body}
            }}

            #[inline(always)]
            fn bitxor(a: {repr}, b: {repr}) -> {repr} {{
                {xor_body}
            }}
        }}
    "#,
        v4_copies = (0..sub_count).map(|_| "v4").collect::<Vec<_>>().join(", "),
        z_copies = (0..sub_count).map(|_| "z").collect::<Vec<_>>().join(", "),
        load_lanes = (0..sub_count)
            .map(|i| format!("vld1q_{ns}(data.as_ptr().add({}))", i * native_lanes))
            .collect::<Vec<_>>().join(", "),
        store_lanes = (0..sub_count)
            .map(|i| format!("vst1q_{ns}(out.as_mut_ptr().add({}), repr[{i}]);", i * native_lanes))
            .collect::<Vec<_>>().join("\n            "),
        add_body = binary_op(&format!("vaddq_{ns}")),
        sub_body = binary_op(&format!("vsubq_{ns}")),
        mul_body = binary_op(&format!("vmulq_{ns}")),
        div_body = binary_op(&format!("vdivq_{ns}")),
        neg_body = unary_op(&format!("vnegq_{ns}")),
        min_body = binary_op(&format!("vminq_{ns}")),
        max_body = binary_op(&format!("vmaxq_{ns}")),
        sqrt_body = unary_op(&format!("vsqrtq_{ns}")),
        abs_body = unary_op(&format!("vabsq_{ns}")),
        floor_body = unary_op(&format!("vrndmq_{ns}")),
        ceil_body = unary_op(&format!("vrndpq_{ns}")),
        round_body = unary_op(&format!("vrndnq_{ns}")),
        mul_add_body = {
            let items: Vec<String> = (0..sub_count)
                .map(|i| format!("vfmaq_{ns}(c[{i}], a[{i}], b[{i}])"))
                .collect();
            format!("unsafe {{ [{}] }}", items.join(", "))
        },
        mul_sub_body = {
            let items: Vec<String> = (0..sub_count)
                .map(|i| format!("vfmaq_{ns}(vnegq_{ns}(c[{i}]), a[{i}], b[{i}])"))
                .collect();
            format!("unsafe {{ [{}] }}", items.join(", "))
        },
        eq_body = cmp_op(&format!("vceqq_{ns}")),
        ne_body = ne_op(),
        lt_body = cmp_op(&format!("vcltq_{ns}")),
        le_body = cmp_op(&format!("vcleq_{ns}")),
        gt_body = cmp_op(&format!("vcgtq_{ns}")),
        ge_body = cmp_op(&format!("vcgeq_{ns}")),
        blend_body = blend_op(),
        reduce_add_body = reduce_combine(&format!("vaddq_{ns}"), &format!("vpaddq_{ns}")),
        reduce_min_body = reduce_combine(&format!("vminq_{ns}"), &format!("vpminq_{ns}")),
        reduce_max_body = reduce_combine(&format!("vmaxq_{ns}"), &format!("vpmaxq_{ns}")),
        rcp_body = unary_op(&format!("vrecpeq_{ns}")),
        rsqrt_body = unary_op(&format!("vrsqrteq_{ns}")),
        not_body = bitwise_not_op(),
        and_body = bitwise_binary_op("vandq_u32"),
        or_body = bitwise_binary_op("vorrq_u32"),
        xor_body = bitwise_binary_op("veorq_u32"),
    }
}

/// Generate NEON impl for a type that's native 128-bit (f32x4, f64x2).
fn generate_neon_native_impl(ty: &FloatVecType) -> String {
    let trait_name = ty.trait_name();
    let repr = ty.neon_repr();
    let elem = ty.elem;
    let lanes = ty.lanes;
    let array = ty.array_type();
    let ns = ty.neon_suffix();
    let zero_lit = if elem == "f32" { "0.0f32" } else { "0.0f64" };
    let eb = if elem == "f32" { 32 } else { 64 };
    let native_lanes = if elem == "f32" { 4 } else { 2 };

    // For native types, reduce pattern is different
    let reduce_pairwise = |pairwise: &str| -> String {
        let mut body = format!("unsafe {{\n            let pair = {pairwise}(a, a);\n");
        if native_lanes > 2 {
            body.push_str(&format!("            let pair = {pairwise}(pair, pair);\n"));
        }
        body.push_str(&format!("            vgetq_lane_{ns}::<0>(pair)\n"));
        body.push_str("        }");
        body
    };

    formatdoc! {r#"
        impl {trait_name} for archmage::NeonToken {{
            type Repr = {repr};

            #[inline(always)]
            fn splat(v: {elem}) -> {repr} {{
                unsafe {{ vdupq_n_{ns}(v) }}
            }}

            #[inline(always)]
            fn zero() -> {repr} {{
                unsafe {{ vdupq_n_{ns}(0.0) }}
            }}

            #[inline(always)]
            fn load(data: &{array}) -> {repr} {{
                unsafe {{ vld1q_{ns}(data.as_ptr()) }}
            }}

            #[inline(always)]
            fn from_array(arr: {array}) -> {repr} {{
                Self::load(&arr)
            }}

            #[inline(always)]
            fn store(repr: {repr}, out: &mut {array}) {{
                unsafe {{ vst1q_{ns}(out.as_mut_ptr(), repr) }};
            }}

            #[inline(always)]
            fn to_array(repr: {repr}) -> {array} {{
                let mut out = [{zero_lit}; {lanes}];
                Self::store(repr, &mut out);
                out
            }}

            #[inline(always)]
            fn add(a: {repr}, b: {repr}) -> {repr} {{ unsafe {{ vaddq_{ns}(a, b) }} }}
            #[inline(always)]
            fn sub(a: {repr}, b: {repr}) -> {repr} {{ unsafe {{ vsubq_{ns}(a, b) }} }}
            #[inline(always)]
            fn mul(a: {repr}, b: {repr}) -> {repr} {{ unsafe {{ vmulq_{ns}(a, b) }} }}
            #[inline(always)]
            fn div(a: {repr}, b: {repr}) -> {repr} {{ unsafe {{ vdivq_{ns}(a, b) }} }}
            #[inline(always)]
            fn neg(a: {repr}) -> {repr} {{ unsafe {{ vnegq_{ns}(a) }} }}
            #[inline(always)]
            fn min(a: {repr}, b: {repr}) -> {repr} {{ unsafe {{ vminq_{ns}(a, b) }} }}
            #[inline(always)]
            fn max(a: {repr}, b: {repr}) -> {repr} {{ unsafe {{ vmaxq_{ns}(a, b) }} }}
            #[inline(always)]
            fn sqrt(a: {repr}) -> {repr} {{ unsafe {{ vsqrtq_{ns}(a) }} }}
            #[inline(always)]
            fn abs(a: {repr}) -> {repr} {{ unsafe {{ vabsq_{ns}(a) }} }}
            #[inline(always)]
            fn floor(a: {repr}) -> {repr} {{ unsafe {{ vrndmq_{ns}(a) }} }}
            #[inline(always)]
            fn ceil(a: {repr}) -> {repr} {{ unsafe {{ vrndpq_{ns}(a) }} }}
            #[inline(always)]
            fn round(a: {repr}) -> {repr} {{ unsafe {{ vrndnq_{ns}(a) }} }}

            #[inline(always)]
            fn mul_add(a: {repr}, b: {repr}, c: {repr}) -> {repr} {{
                unsafe {{ vfmaq_{ns}(c, a, b) }}
            }}

            #[inline(always)]
            fn mul_sub(a: {repr}, b: {repr}, c: {repr}) -> {repr} {{
                unsafe {{ vfmaq_{ns}(vnegq_{ns}(c), a, b) }}
            }}

            #[inline(always)]
            fn simd_eq(a: {repr}, b: {repr}) -> {repr} {{
                unsafe {{ vreinterpretq_{ns}_u{eb}(vceqq_{ns}(a, b)) }}
            }}
            #[inline(always)]
            fn simd_ne(a: {repr}, b: {repr}) -> {repr} {{
                unsafe {{ vreinterpretq_{ns}_u{eb}(vmvnq_u{eb}(vceqq_{ns}(a, b))) }}
            }}
            #[inline(always)]
            fn simd_lt(a: {repr}, b: {repr}) -> {repr} {{
                unsafe {{ vreinterpretq_{ns}_u{eb}(vcltq_{ns}(a, b)) }}
            }}
            #[inline(always)]
            fn simd_le(a: {repr}, b: {repr}) -> {repr} {{
                unsafe {{ vreinterpretq_{ns}_u{eb}(vcleq_{ns}(a, b)) }}
            }}
            #[inline(always)]
            fn simd_gt(a: {repr}, b: {repr}) -> {repr} {{
                unsafe {{ vreinterpretq_{ns}_u{eb}(vcgtq_{ns}(a, b)) }}
            }}
            #[inline(always)]
            fn simd_ge(a: {repr}, b: {repr}) -> {repr} {{
                unsafe {{ vreinterpretq_{ns}_u{eb}(vcgeq_{ns}(a, b)) }}
            }}

            #[inline(always)]
            fn blend(mask: {repr}, if_true: {repr}, if_false: {repr}) -> {repr} {{
                unsafe {{ vbslq_{ns}(vreinterpretq_u{eb}_{ns}(mask), if_true, if_false) }}
            }}

            #[inline(always)]
            fn reduce_add(a: {repr}) -> {elem} {{
                {reduce_add}
            }}

            #[inline(always)]
            fn reduce_min(a: {repr}) -> {elem} {{
                {reduce_min}
            }}

            #[inline(always)]
            fn reduce_max(a: {repr}) -> {elem} {{
                {reduce_max}
            }}

            #[inline(always)]
            fn rcp_approx(a: {repr}) -> {repr} {{ unsafe {{ vrecpeq_{ns}(a) }} }}
            #[inline(always)]
            fn rsqrt_approx(a: {repr}) -> {repr} {{ unsafe {{ vrsqrteq_{ns}(a) }} }}

            #[inline(always)]
            fn not(a: {repr}) -> {repr} {{
                unsafe {{ vreinterpretq_{ns}_u{eb}(vmvnq_u{eb}(vreinterpretq_u{eb}_{ns}(a))) }}
            }}
            #[inline(always)]
            fn bitand(a: {repr}, b: {repr}) -> {repr} {{
                unsafe {{ vreinterpretq_{ns}_u{eb}(vandq_u{eb}(vreinterpretq_u{eb}_{ns}(a), vreinterpretq_u{eb}_{ns}(b))) }}
            }}
            #[inline(always)]
            fn bitor(a: {repr}, b: {repr}) -> {repr} {{
                unsafe {{ vreinterpretq_{ns}_u{eb}(vorrq_u{eb}(vreinterpretq_u{eb}_{ns}(a), vreinterpretq_u{eb}_{ns}(b))) }}
            }}
            #[inline(always)]
            fn bitxor(a: {repr}, b: {repr}) -> {repr} {{
                unsafe {{ vreinterpretq_{ns}_u{eb}(veorq_u{eb}(vreinterpretq_u{eb}_{ns}(a), vreinterpretq_u{eb}_{ns}(b))) }}
            }}
        }}
    "#,
        reduce_add = reduce_pairwise(&format!("vpaddq_{ns}")),
        reduce_min = reduce_pairwise(&format!("vpminq_{ns}")),
        reduce_max = reduce_pairwise(&format!("vpmaxq_{ns}")),
    }
}

// ============================================================================
// WASM Implementation Generation
// ============================================================================

fn generate_wasm_impls(types: &[FloatVecType]) -> String {
    let mut code = formatdoc! {r#"
        //! Backend implementations for Wasm128Token (WebAssembly SIMD).
        //!
        //! **Auto-generated** by `cargo xtask generate` - do not edit manually.

        #[cfg(target_arch = "wasm32")]
        use core::arch::wasm32::*;

        use crate::simd::backends::*;

    "#};

    for ty in types {
        code.push_str("#[cfg(target_arch = \"wasm32\")]\n");
        code.push_str(&generate_wasm_float_impl(ty));
        code.push('\n');
    }

    code
}

fn generate_wasm_float_impl(ty: &FloatVecType) -> String {
    let trait_name = ty.trait_name();
    let repr = ty.wasm_repr();
    let elem = ty.elem;
    let lanes = ty.lanes;
    let array = ty.array_type();
    let wp = ty.wasm_prefix(); // "f32x4" or "f64x2"
    let zero_lit = if elem == "f32" { "0.0f32" } else { "0.0f64" };
    let sub_count = ty.sub_count();
    let native_lanes = if elem == "f32" { 4 } else { 2 };

    if ty.native_on_wasm() {
        return generate_wasm_native_impl(ty);
    }

    // Polyfill: apply operation to each sub-vector
    let binary_op = |func: &str| -> String {
        let items: Vec<String> = (0..sub_count)
            .map(|i| format!("{func}(a[{i}], b[{i}])"))
            .collect();
        format!("[{}]", items.join(", "))
    };

    let unary_op = |func: &str| -> String {
        let items: Vec<String> = (0..sub_count)
            .map(|i| format!("{func}(a[{i}])"))
            .collect();
        format!("[{}]", items.join(", "))
    };

    // Reductions: extract all lanes and fold
    let reduce_add_body = || -> String {
        let mut items = Vec::new();
        for i in 0..sub_count {
            for j in 0..native_lanes {
                items.push(format!("{wp}_extract_lane::<{j}>(a[{i}])"));
            }
        }
        items.join("\n            + ")
    };

    let reduce_minmax = |combine: &str, fold_method: &str| -> String {
        let mut body = format!("let m = {combine}(a[0], a[1]);\n");
        for i in 2..sub_count {
            body.push_str(&format!("        let m = {combine}(m, a[{i}]);\n"));
        }
        let extracts: Vec<String> = (0..native_lanes)
            .map(|j| format!("let v{j} = {wp}_extract_lane::<{j}>(m);"))
            .collect();
        body.push_str(&format!("        {}\n", extracts.join("\n        ")));
        // Build fold
        let fold: String = if native_lanes == 4 {
            format!("v0.{fold_method}(v1).{fold_method}(v2.{fold_method}(v3))")
        } else {
            format!("v0.{fold_method}(v1)")
        };
        body.push_str(&format!("        {fold}"));
        body
    };

    formatdoc! {r#"
        impl {trait_name} for archmage::Wasm128Token {{
            type Repr = {repr};

            #[inline(always)]
            fn splat(v: {elem}) -> {repr} {{
                let v4 = {wp}_splat(v);
                [{v4_copies}]
            }}

            #[inline(always)]
            fn zero() -> {repr} {{
                let z = {wp}_splat(0.0);
                [{z_copies}]
            }}

            #[inline(always)]
            fn load(data: &{array}) -> {repr} {{
                unsafe {{
                    [{load_lanes}]
                }}
            }}

            #[inline(always)]
            fn from_array(arr: {array}) -> {repr} {{
                Self::load(&arr)
            }}

            #[inline(always)]
            fn store(repr: {repr}, out: &mut {array}) {{
                unsafe {{
                    {store_lanes}
                }}
            }}

            #[inline(always)]
            fn to_array(repr: {repr}) -> {array} {{
                let mut out = [{zero_lit}; {lanes}];
                Self::store(repr, &mut out);
                out
            }}

            #[inline(always)]
            fn add(a: {repr}, b: {repr}) -> {repr} {{ {add} }}
            #[inline(always)]
            fn sub(a: {repr}, b: {repr}) -> {repr} {{ {sub} }}
            #[inline(always)]
            fn mul(a: {repr}, b: {repr}) -> {repr} {{ {mul} }}
            #[inline(always)]
            fn div(a: {repr}, b: {repr}) -> {repr} {{ {div} }}
            #[inline(always)]
            fn neg(a: {repr}) -> {repr} {{ {neg} }}
            #[inline(always)]
            fn min(a: {repr}, b: {repr}) -> {repr} {{ {min} }}
            #[inline(always)]
            fn max(a: {repr}, b: {repr}) -> {repr} {{ {max} }}
            #[inline(always)]
            fn sqrt(a: {repr}) -> {repr} {{ {sqrt} }}
            #[inline(always)]
            fn abs(a: {repr}) -> {repr} {{ {abs} }}
            #[inline(always)]
            fn floor(a: {repr}) -> {repr} {{ {floor} }}
            #[inline(always)]
            fn ceil(a: {repr}) -> {repr} {{ {ceil} }}
            #[inline(always)]
            fn round(a: {repr}) -> {repr} {{ {round} }}

            #[inline(always)]
            fn mul_add(a: {repr}, b: {repr}, c: {repr}) -> {repr} {{
                // WASM has no native FMA
                [{mul_add_lanes}]
            }}

            #[inline(always)]
            fn mul_sub(a: {repr}, b: {repr}, c: {repr}) -> {repr} {{
                [{mul_sub_lanes}]
            }}

            #[inline(always)]
            fn simd_eq(a: {repr}, b: {repr}) -> {repr} {{ {eq} }}
            #[inline(always)]
            fn simd_ne(a: {repr}, b: {repr}) -> {repr} {{ {ne} }}
            #[inline(always)]
            fn simd_lt(a: {repr}, b: {repr}) -> {repr} {{ {lt} }}
            #[inline(always)]
            fn simd_le(a: {repr}, b: {repr}) -> {repr} {{ {le} }}
            #[inline(always)]
            fn simd_gt(a: {repr}, b: {repr}) -> {repr} {{ {gt} }}
            #[inline(always)]
            fn simd_ge(a: {repr}, b: {repr}) -> {repr} {{ {ge} }}

            #[inline(always)]
            fn blend(mask: {repr}, if_true: {repr}, if_false: {repr}) -> {repr} {{
                [{blend_lanes}]
            }}

            #[inline(always)]
            fn reduce_add(a: {repr}) -> {elem} {{
                {reduce_add}
            }}

            #[inline(always)]
            fn reduce_min(a: {repr}) -> {elem} {{
                {reduce_min}
            }}

            #[inline(always)]
            fn reduce_max(a: {repr}) -> {elem} {{
                {reduce_max}
            }}

            #[inline(always)]
            fn rcp_approx(a: {repr}) -> {repr} {{
                let one = {wp}_splat(1.0);
                [{rcp_lanes}]
            }}

            #[inline(always)]
            fn rsqrt_approx(a: {repr}) -> {repr} {{
                let one = {wp}_splat(1.0);
                [{rsqrt_lanes}]
            }}

            // Override defaults: WASM has no fast approximation, already full precision
            #[inline(always)]
            fn recip(a: {repr}) -> {repr} {{
                <archmage::Wasm128Token as {trait_name}>::rcp_approx(a)
            }}

            #[inline(always)]
            fn rsqrt(a: {repr}) -> {repr} {{
                <archmage::Wasm128Token as {trait_name}>::rsqrt_approx(a)
            }}

            #[inline(always)]
            fn not(a: {repr}) -> {repr} {{ {not} }}
            #[inline(always)]
            fn bitand(a: {repr}, b: {repr}) -> {repr} {{ {and} }}
            #[inline(always)]
            fn bitor(a: {repr}, b: {repr}) -> {repr} {{ {or} }}
            #[inline(always)]
            fn bitxor(a: {repr}, b: {repr}) -> {repr} {{ {xor} }}
        }}
    "#,
        v4_copies = (0..sub_count).map(|_| "v4").collect::<Vec<_>>().join(", "),
        z_copies = (0..sub_count).map(|_| "z").collect::<Vec<_>>().join(", "),
        load_lanes = (0..sub_count)
            .map(|i| format!("v128_load(data.as_ptr().add({}).cast())", i * native_lanes))
            .collect::<Vec<_>>().join(", "),
        store_lanes = (0..sub_count)
            .map(|i| format!("v128_store(out.as_mut_ptr().add({}).cast(), repr[{i}]);", i * native_lanes))
            .collect::<Vec<_>>().join("\n            "),
        add = binary_op(&format!("{wp}_add")),
        sub = binary_op(&format!("{wp}_sub")),
        mul = binary_op(&format!("{wp}_mul")),
        div = binary_op(&format!("{wp}_div")),
        neg = unary_op(&format!("{wp}_neg")),
        min = binary_op(&format!("{wp}_min")),
        max = binary_op(&format!("{wp}_max")),
        sqrt = unary_op(&format!("{wp}_sqrt")),
        abs = unary_op(&format!("{wp}_abs")),
        floor = unary_op(&format!("{wp}_floor")),
        ceil = unary_op(&format!("{wp}_ceil")),
        round = unary_op(&format!("{wp}_nearest")),
        mul_add_lanes = (0..sub_count)
            .map(|i| format!("{wp}_add({wp}_mul(a[{i}], b[{i}]), c[{i}])"))
            .collect::<Vec<_>>().join(", "),
        mul_sub_lanes = (0..sub_count)
            .map(|i| format!("{wp}_sub({wp}_mul(a[{i}], b[{i}]), c[{i}])"))
            .collect::<Vec<_>>().join(", "),
        eq = binary_op(&format!("{wp}_eq")),
        ne = binary_op(&format!("{wp}_ne")),
        lt = binary_op(&format!("{wp}_lt")),
        le = binary_op(&format!("{wp}_le")),
        gt = binary_op(&format!("{wp}_gt")),
        ge = binary_op(&format!("{wp}_ge")),
        blend_lanes = (0..sub_count)
            .map(|i| format!("v128_bitselect(if_true[{i}], if_false[{i}], mask[{i}])"))
            .collect::<Vec<_>>().join(", "),
        reduce_add = reduce_add_body(),
        reduce_min = reduce_minmax(&format!("{wp}_min"), "min"),
        reduce_max = reduce_minmax(&format!("{wp}_max"), "max"),
        rcp_lanes = (0..sub_count)
            .map(|i| format!("{wp}_div(one, a[{i}])"))
            .collect::<Vec<_>>().join(", "),
        rsqrt_lanes = (0..sub_count)
            .map(|i| format!("{wp}_div(one, {wp}_sqrt(a[{i}]))"))
            .collect::<Vec<_>>().join(", "),
        not = unary_op("v128_not"),
        and = binary_op("v128_and"),
        or = binary_op("v128_or"),
        xor = binary_op("v128_xor"),
    }
}

/// Generate WASM impl for native 128-bit types (f32x4, f64x2).
fn generate_wasm_native_impl(ty: &FloatVecType) -> String {
    let trait_name = ty.trait_name();
    let elem = ty.elem;
    let lanes = ty.lanes;
    let array = ty.array_type();
    let wp = ty.wasm_prefix();
    let zero_lit = if elem == "f32" { "0.0f32" } else { "0.0f64" };

    // Reduction: extract all lanes
    let reduce_add_body = || -> String {
        let items: Vec<String> = (0..lanes)
            .map(|j| format!("{wp}_extract_lane::<{j}>(a)"))
            .collect();
        items.join(" + ")
    };

    let reduce_minmax = |fold_method: &str| -> String {
        let extracts: Vec<String> = (0..lanes)
            .map(|j| format!("let v{j} = {wp}_extract_lane::<{j}>(a);"))
            .collect();
        let fold = if lanes == 4 {
            format!("v0.{fold_method}(v1).{fold_method}(v2.{fold_method}(v3))")
        } else {
            format!("v0.{fold_method}(v1)")
        };
        format!("{}\n        {fold}", extracts.join("\n        "))
    };

    formatdoc! {r#"
        impl {trait_name} for archmage::Wasm128Token {{
            type Repr = v128;

            #[inline(always)]
            fn splat(v: {elem}) -> v128 {{ {wp}_splat(v) }}
            #[inline(always)]
            fn zero() -> v128 {{ {wp}_splat(0.0) }}
            #[inline(always)]
            fn load(data: &{array}) -> v128 {{ unsafe {{ v128_load(data.as_ptr().cast()) }} }}
            #[inline(always)]
            fn from_array(arr: {array}) -> v128 {{ Self::load(&arr) }}
            #[inline(always)]
            fn store(repr: v128, out: &mut {array}) {{ unsafe {{ v128_store(out.as_mut_ptr().cast(), repr) }}; }}
            #[inline(always)]
            fn to_array(repr: v128) -> {array} {{
                let mut out = [{zero_lit}; {lanes}];
                Self::store(repr, &mut out);
                out
            }}

            #[inline(always)]
            fn add(a: v128, b: v128) -> v128 {{ {wp}_add(a, b) }}
            #[inline(always)]
            fn sub(a: v128, b: v128) -> v128 {{ {wp}_sub(a, b) }}
            #[inline(always)]
            fn mul(a: v128, b: v128) -> v128 {{ {wp}_mul(a, b) }}
            #[inline(always)]
            fn div(a: v128, b: v128) -> v128 {{ {wp}_div(a, b) }}
            #[inline(always)]
            fn neg(a: v128) -> v128 {{ {wp}_neg(a) }}
            #[inline(always)]
            fn min(a: v128, b: v128) -> v128 {{ {wp}_min(a, b) }}
            #[inline(always)]
            fn max(a: v128, b: v128) -> v128 {{ {wp}_max(a, b) }}
            #[inline(always)]
            fn sqrt(a: v128) -> v128 {{ {wp}_sqrt(a) }}
            #[inline(always)]
            fn abs(a: v128) -> v128 {{ {wp}_abs(a) }}
            #[inline(always)]
            fn floor(a: v128) -> v128 {{ {wp}_floor(a) }}
            #[inline(always)]
            fn ceil(a: v128) -> v128 {{ {wp}_ceil(a) }}
            #[inline(always)]
            fn round(a: v128) -> v128 {{ {wp}_nearest(a) }}
            #[inline(always)]
            fn mul_add(a: v128, b: v128, c: v128) -> v128 {{ {wp}_add({wp}_mul(a, b), c) }}
            #[inline(always)]
            fn mul_sub(a: v128, b: v128, c: v128) -> v128 {{ {wp}_sub({wp}_mul(a, b), c) }}
            #[inline(always)]
            fn simd_eq(a: v128, b: v128) -> v128 {{ {wp}_eq(a, b) }}
            #[inline(always)]
            fn simd_ne(a: v128, b: v128) -> v128 {{ {wp}_ne(a, b) }}
            #[inline(always)]
            fn simd_lt(a: v128, b: v128) -> v128 {{ {wp}_lt(a, b) }}
            #[inline(always)]
            fn simd_le(a: v128, b: v128) -> v128 {{ {wp}_le(a, b) }}
            #[inline(always)]
            fn simd_gt(a: v128, b: v128) -> v128 {{ {wp}_gt(a, b) }}
            #[inline(always)]
            fn simd_ge(a: v128, b: v128) -> v128 {{ {wp}_ge(a, b) }}
            #[inline(always)]
            fn blend(mask: v128, if_true: v128, if_false: v128) -> v128 {{
                v128_bitselect(if_true, if_false, mask)
            }}

            #[inline(always)]
            fn reduce_add(a: v128) -> {elem} {{ {reduce_add} }}
            #[inline(always)]
            fn reduce_min(a: v128) -> {elem} {{
                {reduce_min}
            }}
            #[inline(always)]
            fn reduce_max(a: v128) -> {elem} {{
                {reduce_max}
            }}

            #[inline(always)]
            fn rcp_approx(a: v128) -> v128 {{ {wp}_div({wp}_splat(1.0), a) }}
            #[inline(always)]
            fn rsqrt_approx(a: v128) -> v128 {{ {wp}_div({wp}_splat(1.0), {wp}_sqrt(a)) }}
            #[inline(always)]
            fn recip(a: v128) -> v128 {{ <archmage::Wasm128Token as {trait_name}>::rcp_approx(a) }}
            #[inline(always)]
            fn rsqrt(a: v128) -> v128 {{ <archmage::Wasm128Token as {trait_name}>::rsqrt_approx(a) }}

            #[inline(always)]
            fn not(a: v128) -> v128 {{ v128_not(a) }}
            #[inline(always)]
            fn bitand(a: v128, b: v128) -> v128 {{ v128_and(a, b) }}
            #[inline(always)]
            fn bitor(a: v128, b: v128) -> v128 {{ v128_or(a, b) }}
            #[inline(always)]
            fn bitxor(a: v128, b: v128) -> v128 {{ v128_xor(a, b) }}
        }}
    "#,
        reduce_add = reduce_add_body(),
        reduce_min = reduce_minmax("min"),
        reduce_max = reduce_minmax("max"),
    }
}

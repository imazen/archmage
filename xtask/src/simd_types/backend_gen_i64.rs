//! I64 backend codegen for i64x2 and i64x4 generic types.
//!
//! This will be merged into backend_gen.rs.

use indoc::formatdoc;

// ============================================================================
// Data Model
// ============================================================================

/// An i64 vector type for backend generation.
#[derive(Clone, Debug)]
pub(super) struct I64VecType {
    /// Number of lanes
    lanes: usize,
    /// Width in bits (128, 256)
    width_bits: usize,
}

impl I64VecType {
    /// Type name: "i64x2", "i64x4"
    pub(super) fn name(&self) -> String {
        format!("i64x{}", self.lanes)
    }

    /// Trait name: "I64x2Backend", "I64x4Backend"
    pub(super) fn trait_name(&self) -> String {
        format!("I64x{}Backend", self.lanes)
    }

    /// Array type: "[i64; 2]", "[i64; 4]"
    fn array_type(&self) -> String {
        format!("[i64; {}]", self.lanes)
    }

    /// x86 intrinsic prefix: "_mm", "_mm256"
    fn x86_prefix(&self) -> &'static str {
        match self.width_bits {
            128 => "_mm",
            256 => "_mm256",
            _ => unreachable!(),
        }
    }

    /// x86 inner type: "__m128i", "__m256i"
    fn x86_inner_type(&self) -> &'static str {
        match self.width_bits {
            128 => "__m128i",
            256 => "__m256i",
            _ => unreachable!(),
        }
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
            "int64x2_t".to_string()
        } else {
            let count = self.width_bits / 128;
            format!("[int64x2_t; {count}]")
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

    /// Number of 128-bit sub-vectors for polyfill
    fn sub_count(&self) -> usize {
        self.width_bits / 128
    }
}

/// All i64 vector types to generate backends for.
pub(super) fn all_i64_types() -> Vec<I64VecType> {
    vec![
        I64VecType {
            lanes: 2,
            width_bits: 128,
        },
        I64VecType {
            lanes: 4,
            width_bits: 256,
        },
    ]
}

// ============================================================================
// I64 Backend Trait Definition Generation
// ============================================================================

pub(super) fn generate_i64_backend_trait(ty: &I64VecType) -> String {
    let trait_name = ty.trait_name();
    let lanes = ty.lanes;
    let array = ty.array_type();

    formatdoc! {r#"
        //! Backend trait for `{name}<T>` — {lanes}-lane i64 SIMD vector.
        //!
        //! Each token type implements this trait with its platform-native representation.
        //! The generic wrapper `{name}<T>` delegates all operations to these trait methods.
        //!
        //! **Auto-generated** by `cargo xtask generate` - do not edit manually.

        use super::sealed::Sealed;
        use archmage::SimdToken;

        /// Backend implementation for {lanes}-lane i64 SIMD vectors.
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
            fn splat(v: i64) -> Self::Repr;

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

            // NOTE: No `mul` — no native i64 multiply on most platforms before AVX-512.

            /// Lane-wise negation.
            fn neg(a: Self::Repr) -> Self::Repr;

            // ====== Math ======

            /// Lane-wise minimum.
            fn min(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise maximum.
            fn max(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise absolute value.
            fn abs(a: Self::Repr) -> Self::Repr;

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
            fn reduce_add(a: Self::Repr) -> i64;

            // ====== Bitwise ======

            /// Bitwise NOT.
            fn not(a: Self::Repr) -> Self::Repr;

            /// Bitwise AND.
            fn bitand(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Bitwise OR.
            fn bitor(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Bitwise XOR.
            fn bitxor(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            // ====== Shifts ======

            /// Shift left by constant.
            fn shl_const<const N: i32>(a: Self::Repr) -> Self::Repr;

            /// Arithmetic shift right by constant (sign-extending).
            fn shr_arithmetic_const<const N: i32>(a: Self::Repr) -> Self::Repr;

            /// Logical shift right by constant (zero-filling).
            fn shr_logical_const<const N: i32>(a: Self::Repr) -> Self::Repr;

            // ====== Boolean ======

            /// True if all lanes have their sign bit set (all-1s mask).
            fn all_true(a: Self::Repr) -> bool;

            /// True if any lane has its sign bit set (any all-1s mask lane).
            fn any_true(a: Self::Repr) -> bool;

            /// Extract the high bit of each 64-bit lane as a bitmask.
            fn bitmask(a: Self::Repr) -> u32;

            // ====== Default implementations ======

            /// Clamp values between lo and hi.
            #[inline(always)]
            fn clamp(a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {{
                Self::min(Self::max(a, lo), hi)
            }}
        }}
    "#,
        name = ty.name(),
    }
}

// ============================================================================
// x86 I64 Implementation Generation
// ============================================================================

pub(super) fn generate_x86_i64_impls(
    types: &[I64VecType],
    token: &str,
    max_width: usize,
) -> String {
    let mut code = String::new();

    for ty in types {
        if ty.width_bits > max_width {
            continue;
        }
        code.push_str("\n#[cfg(target_arch = \"x86_64\")]\n");
        code.push_str(&generate_x86_i64_impl(ty, token));
        code.push('\n');
    }

    code
}

fn generate_x86_i64_impl(ty: &I64VecType, token: &str) -> String {
    let trait_name = ty.trait_name();
    let inner = ty.x86_inner_type();
    let p = ty.x86_prefix();
    let bits = ty.width_bits;
    let array = ty.array_type();
    let lanes = ty.lanes;

    let reduce_add_body = generate_x86_i64_reduce_add(ty);

    // Arithmetic right shift polyfill for i64 on SSE/AVX2:
    // No native _mm_srai_epi64 or _mm256_srai_epi64 on AVX2.
    // Use: logical shift + sign extension mask.
    //   let sign64 = shuffle high 32-bit of each 64 to both 32-bit positions,
    //                then arithmetic right shift 32-bit by 31 to get sign fill.
    //   let logical = srli_epi64(a, N)
    //   let mask = !srli_epi64(set1_epi64x(-1), N)  // upper N bits set
    //   result = logical | (sign64 & mask)
    let shr_arith_body = formatdoc! {"
                unsafe {{
                    // Broadcast sign of each 64-bit lane to all bits of that lane
                    let sign_ext = {p}_srai_epi32::<31>(a);
                    let sign64 = {p}_shuffle_epi32::<0xF5>(sign_ext);
                    let logical = {p}_srli_epi64::<N>(a);
                    // mask = NOT(srli(-1, N)) = upper N bits set, avoids {{64 - N}} const expr
                    let all_ones = {p}_set1_epi64x(-1);
                    let mask = {p}_andnot_si{bits}({p}_srli_epi64::<N>(all_ones), all_ones);
                    {p}_or_si{bits}(logical, {p}_and_si{bits}(sign64, mask))
                }}"};

    // all_true/any_true/bitmask for 64-bit lanes use _pd movemask (2 or 4 bits).
    let all_mask = match lanes {
        2 => "0x3",
        4 => "0xF",
        _ => unreachable!(),
    };

    formatdoc! {r#"
        impl {trait_name} for archmage::{token} {{
            type Repr = {inner};

            // ====== Construction ======

            #[inline(always)]
            fn splat(v: i64) -> {inner} {{
                unsafe {{ {p}_set1_epi64x(v) }}
            }}

            #[inline(always)]
            fn zero() -> {inner} {{
                unsafe {{ {p}_setzero_si{bits}() }}
            }}

            #[inline(always)]
            fn load(data: &{array}) -> {inner} {{
                unsafe {{ {p}_loadu_si{bits}(data.as_ptr().cast()) }}
            }}

            #[inline(always)]
            fn from_array(arr: {array}) -> {inner} {{
                // SAFETY: {array} and {inner} have identical size and layout.
                unsafe {{ core::mem::transmute(arr) }}
            }}

            #[inline(always)]
            fn store(repr: {inner}, out: &mut {array}) {{
                unsafe {{ {p}_storeu_si{bits}(out.as_mut_ptr().cast(), repr) }};
            }}

            #[inline(always)]
            fn to_array(repr: {inner}) -> {array} {{
                let mut out = [0i64; {lanes}];
                unsafe {{ {p}_storeu_si{bits}(out.as_mut_ptr().cast(), repr) }};
                out
            }}

            // ====== Arithmetic ======

            #[inline(always)]
            fn add(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_add_epi64(a, b) }}
            }}

            #[inline(always)]
            fn sub(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_sub_epi64(a, b) }}
            }}

            #[inline(always)]
            fn neg(a: {inner}) -> {inner} {{
                unsafe {{ {p}_sub_epi64({p}_setzero_si{bits}(), a) }}
            }}

            // ====== Math ======

            #[inline(always)]
            fn min(a: {inner}, b: {inner}) -> {inner} {{
                // Polyfill: compare+select (no native i64 min on AVX2)
                unsafe {{
                    let mask = {p}_cmpgt_epi64(a, b);
                    {p}_blendv_epi8(a, b, mask)
                }}
            }}

            #[inline(always)]
            fn max(a: {inner}, b: {inner}) -> {inner} {{
                // Polyfill: compare+select (no native i64 max on AVX2)
                unsafe {{
                    let mask = {p}_cmpgt_epi64(a, b);
                    {p}_blendv_epi8(b, a, mask)
                }}
            }}

            #[inline(always)]
            fn abs(a: {inner}) -> {inner} {{
                // Polyfill: (a ^ sign) - sign (two's complement trick)
                unsafe {{
                    let zero = {p}_setzero_si{bits}();
                    let sign = {p}_cmpgt_epi64(zero, a);
                    {p}_sub_epi64({p}_xor_si{bits}(a, sign), sign)
                }}
            }}

            // ====== Comparisons ======

            #[inline(always)]
            fn simd_eq(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_cmpeq_epi64(a, b) }}
            }}

            #[inline(always)]
            fn simd_ne(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{
                    let eq = {p}_cmpeq_epi64(a, b);
                    {p}_xor_si{bits}(eq, {p}_set1_epi64x(-1))
                }}
            }}

            #[inline(always)]
            fn simd_lt(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_cmpgt_epi64(b, a) }}
            }}

            #[inline(always)]
            fn simd_le(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{
                    let gt = {p}_cmpgt_epi64(a, b);
                    {p}_xor_si{bits}(gt, {p}_set1_epi64x(-1))
                }}
            }}

            #[inline(always)]
            fn simd_gt(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_cmpgt_epi64(a, b) }}
            }}

            #[inline(always)]
            fn simd_ge(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{
                    let lt = {p}_cmpgt_epi64(b, a);
                    {p}_xor_si{bits}(lt, {p}_set1_epi64x(-1))
                }}
            }}

            #[inline(always)]
            fn blend(mask: {inner}, if_true: {inner}, if_false: {inner}) -> {inner} {{
                unsafe {{ {p}_blendv_epi8(if_false, if_true, mask) }}
            }}

            // ====== Reductions ======

            #[inline(always)]
            fn reduce_add(a: {inner}) -> i64 {{
        {reduce_add_body}
            }}

            // ====== Bitwise ======

            #[inline(always)]
            fn not(a: {inner}) -> {inner} {{
                unsafe {{ {p}_xor_si{bits}(a, {p}_set1_epi64x(-1)) }}
            }}

            #[inline(always)]
            fn bitand(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_and_si{bits}(a, b) }}
            }}

            #[inline(always)]
            fn bitor(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_or_si{bits}(a, b) }}
            }}

            #[inline(always)]
            fn bitxor(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_xor_si{bits}(a, b) }}
            }}

            // ====== Shifts ======

            #[inline(always)]
            fn shl_const<const N: i32>(a: {inner}) -> {inner} {{
                unsafe {{ {p}_slli_epi64::<N>(a) }}
            }}

            #[inline(always)]
            fn shr_arithmetic_const<const N: i32>(a: {inner}) -> {inner} {{
                // Polyfill: no native _srai_epi64 on AVX2.
                // Use logical shift + sign extension.
        {shr_arith_body}
            }}

            #[inline(always)]
            fn shr_logical_const<const N: i32>(a: {inner}) -> {inner} {{
                unsafe {{ {p}_srli_epi64::<N>(a) }}
            }}

            // ====== Boolean ======

            #[inline(always)]
            fn all_true(a: {inner}) -> bool {{
                unsafe {{ {p}_movemask_pd({p}_castsi{bits}_pd(a)) == {all_mask} }}
            }}

            #[inline(always)]
            fn any_true(a: {inner}) -> bool {{
                unsafe {{ {p}_movemask_pd({p}_castsi{bits}_pd(a)) != 0 }}
            }}

            #[inline(always)]
            fn bitmask(a: {inner}) -> u32 {{
                unsafe {{ {p}_movemask_pd({p}_castsi{bits}_pd(a)) as u32 }}
            }}
        }}
    "#}
}

fn generate_x86_i64_reduce_add(ty: &I64VecType) -> String {
    match ty.width_bits {
        128 => formatdoc! {"
                unsafe {{
                    let hi = _mm_unpackhi_epi64(a, a);
                    let sum = _mm_add_epi64(a, hi);
                    // Extract low 64-bit lane
                    core::mem::transmute::<__m128i, [i64; 2]>(sum)[0]
                }}"},
        256 => formatdoc! {"
                unsafe {{
                    let lo = _mm256_castsi256_si128(a);
                    let hi = _mm256_extracti128_si256::<1>(a);
                    let sum = _mm_add_epi64(lo, hi);
                    let hi2 = _mm_unpackhi_epi64(sum, sum);
                    let sum2 = _mm_add_epi64(sum, hi2);
                    core::mem::transmute::<__m128i, [i64; 2]>(sum2)[0]
                }}"},
        _ => unreachable!(),
    }
}

// ============================================================================
// Scalar I64 Implementation Generation
// ============================================================================

pub(super) fn generate_scalar_i64_impls(types: &[I64VecType]) -> String {
    let mut code = String::new();

    for ty in types {
        code.push_str(&generate_scalar_i64_impl(ty));
        code.push('\n');
    }

    code
}

fn generate_scalar_i64_impl(ty: &I64VecType) -> String {
    let trait_name = ty.trait_name();
    let lanes = ty.lanes;
    let array = ty.array_type();

    // For infix operators like &, |, ^
    let binary_infix = |op: &str| -> String {
        let items: Vec<String> = (0..lanes).map(|i| format!("a[{i}] {op} b[{i}]")).collect();
        format!("[{}]", items.join(", "))
    };

    // For method-style ops like .wrapping_add(b)
    let binary_method = |method: &str| -> String {
        let items: Vec<String> = (0..lanes)
            .map(|i| format!("a[{i}].{method}(b[{i}])"))
            .collect();
        format!("[{}]", items.join(", "))
    };

    let min_lanes = || -> String {
        let items: Vec<String> = (0..lanes)
            .map(|i| format!("if a[{i}] < b[{i}] {{ a[{i}] }} else {{ b[{i}] }}"))
            .collect();
        format!("[{}]", items.join(", "))
    };

    let max_lanes = || -> String {
        let items: Vec<String> = (0..lanes)
            .map(|i| format!("if a[{i}] > b[{i}] {{ a[{i}] }} else {{ b[{i}] }}"))
            .collect();
        format!("[{}]", items.join(", "))
    };

    let neg_lanes = || -> String {
        let items: Vec<String> = (0..lanes)
            .map(|i| format!("a[{i}].wrapping_neg()"))
            .collect();
        format!("[{}]", items.join(", "))
    };

    let abs_lanes = || -> String {
        let items: Vec<String> = (0..lanes)
            .map(|i| format!("a[{i}].wrapping_abs()"))
            .collect();
        format!("[{}]", items.join(", "))
    };

    let reduce_add = || -> String {
        let items: Vec<String> = (0..lanes).map(|i| format!("a[{i}]")).collect();
        items.join(".wrapping_add(") + &")".repeat(lanes - 1)
    };

    let shl_lanes = || -> String {
        let items: Vec<String> = (0..lanes).map(|i| format!("a[{i}] << N")).collect();
        format!("[{}]", items.join(", "))
    };

    // Rust >> on i64 is arithmetic (sign-extending)
    let shr_arithmetic_lanes = || -> String {
        let items: Vec<String> = (0..lanes).map(|i| format!("a[{i}] >> N")).collect();
        format!("[{}]", items.join(", "))
    };

    let shr_logical_lanes = || -> String {
        let items: Vec<String> = (0..lanes)
            .map(|i| format!("((a[{i}] as u64) >> N) as i64"))
            .collect();
        format!("[{}]", items.join(", "))
    };

    let bitmask_expr = || -> String {
        let items: Vec<String> = (0..lanes)
            .map(|i| {
                if i == 0 {
                    "((a[0] as u64) >> 63) as u32".to_string()
                } else {
                    format!("((((a[{i}] as u64) >> 63) as u32) << {i})")
                }
            })
            .collect();
        items.join(" | ")
    };

    let all_true_expr = || -> String {
        let items: Vec<String> = (0..lanes).map(|i| format!("a[{i}] != 0")).collect();
        items.join(" && ")
    };

    let any_true_expr = || -> String {
        let items: Vec<String> = (0..lanes).map(|i| format!("a[{i}] != 0")).collect();
        items.join(" || ")
    };

    formatdoc! {r#"
        impl {trait_name} for archmage::ScalarToken {{
            type Repr = {array};

            // ====== Construction ======

            #[inline(always)]
            fn splat(v: i64) -> {array} {{
                [v; {lanes}]
            }}

            #[inline(always)]
            fn zero() -> {array} {{
                [0i64; {lanes}]
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
            fn abs(a: {array}) -> {array} {{
                {abs}
            }}

            // ====== Comparisons ======

            #[inline(always)]
            fn simd_eq(a: {array}, b: {array}) -> {array} {{
                let mut r = [0i64; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if a[i] == b[i] {{ -1 }} else {{ 0 }};
                }}
                r
            }}

            #[inline(always)]
            fn simd_ne(a: {array}, b: {array}) -> {array} {{
                let mut r = [0i64; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if a[i] != b[i] {{ -1 }} else {{ 0 }};
                }}
                r
            }}

            #[inline(always)]
            fn simd_lt(a: {array}, b: {array}) -> {array} {{
                let mut r = [0i64; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if a[i] < b[i] {{ -1 }} else {{ 0 }};
                }}
                r
            }}

            #[inline(always)]
            fn simd_le(a: {array}, b: {array}) -> {array} {{
                let mut r = [0i64; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if a[i] <= b[i] {{ -1 }} else {{ 0 }};
                }}
                r
            }}

            #[inline(always)]
            fn simd_gt(a: {array}, b: {array}) -> {array} {{
                let mut r = [0i64; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if a[i] > b[i] {{ -1 }} else {{ 0 }};
                }}
                r
            }}

            #[inline(always)]
            fn simd_ge(a: {array}, b: {array}) -> {array} {{
                let mut r = [0i64; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if a[i] >= b[i] {{ -1 }} else {{ 0 }};
                }}
                r
            }}

            #[inline(always)]
            fn blend(mask: {array}, if_true: {array}, if_false: {array}) -> {array} {{
                let mut r = [0i64; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if mask[i] != 0 {{ if_true[i] }} else {{ if_false[i] }};
                }}
                r
            }}

            // ====== Reductions ======

            #[inline(always)]
            fn reduce_add(a: {array}) -> i64 {{
                {reduce_add}
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

            // ====== Shifts ======

            #[inline(always)]
            fn shl_const<const N: i32>(a: {array}) -> {array} {{
                {shl}
            }}

            #[inline(always)]
            fn shr_arithmetic_const<const N: i32>(a: {array}) -> {array} {{
                {shr_arithmetic}
            }}

            #[inline(always)]
            fn shr_logical_const<const N: i32>(a: {array}) -> {array} {{
                {shr_logical}
            }}

            // ====== Boolean ======

            #[inline(always)]
            fn all_true(a: {array}) -> bool {{
                {all_true}
            }}

            #[inline(always)]
            fn any_true(a: {array}) -> bool {{
                {any_true}
            }}

            #[inline(always)]
            fn bitmask(a: {array}) -> u32 {{
                {bitmask}
            }}
        }}
    "#,
        add_lanes = binary_method("wrapping_add"),
        sub_lanes = binary_method("wrapping_sub"),
        neg = neg_lanes(),
        min_lanes = min_lanes(),
        max_lanes = max_lanes(),
        abs = abs_lanes(),
        reduce_add = reduce_add(),
        not_lanes = {
            let items: Vec<String> = (0..lanes).map(|i| format!("!a[{i}]")).collect();
            format!("[{}]", items.join(", "))
        },
        and_lanes = binary_infix("&"),
        or_lanes = binary_infix("|"),
        xor_lanes = binary_infix("^"),
        shl = shl_lanes(),
        shr_arithmetic = shr_arithmetic_lanes(),
        shr_logical = shr_logical_lanes(),
        all_true = all_true_expr(),
        any_true = any_true_expr(),
        bitmask = bitmask_expr(),
    }
}

// ============================================================================
// NEON I64 Implementation Generation
// ============================================================================

pub(super) fn generate_neon_i64_impls(types: &[I64VecType]) -> String {
    let mut code = String::new();

    for ty in types {
        code.push_str("\n#[cfg(target_arch = \"aarch64\")]\n");
        if ty.native_on_neon() {
            code.push_str(&generate_neon_native_i64_impl(ty));
        } else {
            code.push_str(&generate_neon_polyfill_i64_impl(ty));
        }
        code.push('\n');
    }

    code
}

fn generate_neon_native_i64_impl(ty: &I64VecType) -> String {
    let trait_name = ty.trait_name();
    let array = ty.array_type();
    let lanes = ty.lanes;

    formatdoc! {r#"
        impl {trait_name} for archmage::NeonToken {{
            type Repr = int64x2_t;

            #[inline(always)]
            fn splat(v: i64) -> int64x2_t {{
                unsafe {{ vdupq_n_s64(v) }}
            }}

            #[inline(always)]
            fn zero() -> int64x2_t {{
                unsafe {{ vdupq_n_s64(0i64) }}
            }}

            #[inline(always)]
            fn load(data: &{array}) -> int64x2_t {{
                unsafe {{ vld1q_s64(data.as_ptr()) }}
            }}

            #[inline(always)]
            fn from_array(arr: {array}) -> int64x2_t {{
                unsafe {{ vld1q_s64(arr.as_ptr()) }}
            }}

            #[inline(always)]
            fn store(repr: int64x2_t, out: &mut {array}) {{
                unsafe {{ vst1q_s64(out.as_mut_ptr(), repr) }};
            }}

            #[inline(always)]
            fn to_array(repr: int64x2_t) -> {array} {{
                let mut out = [0i64; {lanes}];
                unsafe {{ vst1q_s64(out.as_mut_ptr(), repr) }};
                out
            }}

            #[inline(always)]
            fn add(a: int64x2_t, b: int64x2_t) -> int64x2_t {{ unsafe {{ vaddq_s64(a, b) }} }}
            #[inline(always)]
            fn sub(a: int64x2_t, b: int64x2_t) -> int64x2_t {{ unsafe {{ vsubq_s64(a, b) }} }}
            #[inline(always)]
            fn neg(a: int64x2_t) -> int64x2_t {{ unsafe {{ vnegq_s64(a) }} }}
            #[inline(always)]
            fn min(a: int64x2_t, b: int64x2_t) -> int64x2_t {{
                // NEON lacks native i64 min; polyfill via compare+select
                unsafe {{
                    let mask = vcltq_s64(a, b);
                    vbslq_s64(mask, a, b)
                }}
            }}
            #[inline(always)]
            fn max(a: int64x2_t, b: int64x2_t) -> int64x2_t {{
                // NEON lacks native i64 max; polyfill via compare+select
                unsafe {{
                    let mask = vcgtq_s64(a, b);
                    vbslq_s64(mask, a, b)
                }}
            }}
            #[inline(always)]
            fn abs(a: int64x2_t) -> int64x2_t {{ unsafe {{ vabsq_s64(a) }} }}

            #[inline(always)]
            fn simd_eq(a: int64x2_t, b: int64x2_t) -> int64x2_t {{
                unsafe {{ vreinterpretq_s64_u64(vceqq_s64(a, b)) }}
            }}
            #[inline(always)]
            fn simd_ne(a: int64x2_t, b: int64x2_t) -> int64x2_t {{
                unsafe {{
                    let eq = vceqq_s64(a, b);
                    // NOT via XOR with all-ones
                    vreinterpretq_s64_u64(veorq_u64(eq, vdupq_n_u64(u64::MAX)))
                }}
            }}
            #[inline(always)]
            fn simd_lt(a: int64x2_t, b: int64x2_t) -> int64x2_t {{
                unsafe {{ vreinterpretq_s64_u64(vcltq_s64(a, b)) }}
            }}
            #[inline(always)]
            fn simd_le(a: int64x2_t, b: int64x2_t) -> int64x2_t {{
                unsafe {{ vreinterpretq_s64_u64(vcleq_s64(a, b)) }}
            }}
            #[inline(always)]
            fn simd_gt(a: int64x2_t, b: int64x2_t) -> int64x2_t {{
                unsafe {{ vreinterpretq_s64_u64(vcgtq_s64(a, b)) }}
            }}
            #[inline(always)]
            fn simd_ge(a: int64x2_t, b: int64x2_t) -> int64x2_t {{
                unsafe {{ vreinterpretq_s64_u64(vcgeq_s64(a, b)) }}
            }}

            #[inline(always)]
            fn blend(mask: int64x2_t, if_true: int64x2_t, if_false: int64x2_t) -> int64x2_t {{
                unsafe {{ vbslq_s64(vreinterpretq_u64_s64(mask), if_true, if_false) }}
            }}

            #[inline(always)]
            fn reduce_add(a: int64x2_t) -> i64 {{
                unsafe {{
                    let sum = vpaddq_s64(a, a);
                    vgetq_lane_s64::<0>(sum)
                }}
            }}

            #[inline(always)]
            fn not(a: int64x2_t) -> int64x2_t {{
                unsafe {{
                    let ones = vdupq_n_s64(-1i64);
                    veorq_s64(a, ones)
                }}
            }}
            #[inline(always)]
            fn bitand(a: int64x2_t, b: int64x2_t) -> int64x2_t {{
                unsafe {{ vandq_s64(a, b) }}
            }}
            #[inline(always)]
            fn bitor(a: int64x2_t, b: int64x2_t) -> int64x2_t {{
                unsafe {{ vorrq_s64(a, b) }}
            }}
            #[inline(always)]
            fn bitxor(a: int64x2_t, b: int64x2_t) -> int64x2_t {{
                unsafe {{ veorq_s64(a, b) }}
            }}

            #[inline(always)]
            fn shl_const<const N: i32>(a: int64x2_t) -> int64x2_t {{
                unsafe {{ vshlq_n_s64::<N>(a) }}
            }}

            #[inline(always)]
            fn shr_arithmetic_const<const N: i32>(a: int64x2_t) -> int64x2_t {{
                unsafe {{ vshrq_n_s64::<N>(a) }}
            }}

            #[inline(always)]
            fn shr_logical_const<const N: i32>(a: int64x2_t) -> int64x2_t {{
                unsafe {{ vreinterpretq_s64_u64(vshrq_n_u64::<N>(vreinterpretq_u64_s64(a))) }}
            }}

            #[inline(always)]
            fn all_true(a: int64x2_t) -> bool {{
                unsafe {{
                    let as_u64 = vreinterpretq_u64_s64(a);
                    vgetq_lane_u64::<0>(as_u64) != 0 && vgetq_lane_u64::<1>(as_u64) != 0
                }}
            }}

            #[inline(always)]
            fn any_true(a: int64x2_t) -> bool {{
                unsafe {{
                    let as_u64 = vreinterpretq_u64_s64(a);
                    (vgetq_lane_u64::<0>(as_u64) | vgetq_lane_u64::<1>(as_u64)) != 0
                }}
            }}

            #[inline(always)]
            fn bitmask(a: int64x2_t) -> u32 {{
                unsafe {{
                    let signs = vshrq_n_u64::<63>(vreinterpretq_u64_s64(a));
                    ((vgetq_lane_u64::<0>(signs) & 1) | ((vgetq_lane_u64::<1>(signs) & 1) << 1)) as u32
                }}
            }}
        }}
    "#}
}

fn generate_neon_polyfill_i64_impl(ty: &I64VecType) -> String {
    let trait_name = ty.trait_name();
    let repr = ty.neon_repr();
    let array = ty.array_type();
    let lanes = ty.lanes;
    let sub_count = ty.sub_count();

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
            .map(|i| format!("vreinterpretq_s64_u64({intrinsic}(a[{i}], b[{i}]))"))
            .collect();
        format!("unsafe {{ [{}] }}", items.join(", "))
    };

    let ne_op = || -> String {
        let items: Vec<String> = (0..sub_count)
            .map(|i| format!("vreinterpretq_s64_u64(veorq_u64(vceqq_s64(a[{i}], b[{i}]), vdupq_n_u64(u64::MAX)))"))
            .collect();
        format!("unsafe {{ [{}] }}", items.join(", "))
    };

    formatdoc! {r#"
        impl {trait_name} for archmage::NeonToken {{
            type Repr = {repr};

            #[inline(always)]
            fn splat(v: i64) -> {repr} {{
                unsafe {{
                    let v2 = vdupq_n_s64(v);
                    [{v2_copies}]
                }}
            }}

            #[inline(always)]
            fn zero() -> {repr} {{
                unsafe {{
                    let z = vdupq_n_s64(0i64);
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
                let mut out = [0i64; {lanes}];
                Self::store(repr, &mut out);
                out
            }}

            #[inline(always)]
            fn add(a: {repr}, b: {repr}) -> {repr} {{ {add} }}
            #[inline(always)]
            fn sub(a: {repr}, b: {repr}) -> {repr} {{ {sub} }}
            #[inline(always)]
            fn neg(a: {repr}) -> {repr} {{ {neg} }}
            #[inline(always)]
            fn min(a: {repr}, b: {repr}) -> {repr} {{
                // NEON lacks native i64 min; polyfill via compare+select per sub-vector
                {min}
            }}
            #[inline(always)]
            fn max(a: {repr}, b: {repr}) -> {repr} {{
                // NEON lacks native i64 max; polyfill via compare+select per sub-vector
                {max}
            }}
            #[inline(always)]
            fn abs(a: {repr}) -> {repr} {{ {abs} }}

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
                {blend}
            }}

            #[inline(always)]
            fn reduce_add(a: {repr}) -> i64 {{
                {reduce_add}
            }}

            #[inline(always)]
            fn not(a: {repr}) -> {repr} {{ {not} }}
            #[inline(always)]
            fn bitand(a: {repr}, b: {repr}) -> {repr} {{ {bitand} }}
            #[inline(always)]
            fn bitor(a: {repr}, b: {repr}) -> {repr} {{ {bitor} }}
            #[inline(always)]
            fn bitxor(a: {repr}, b: {repr}) -> {repr} {{ {bitxor} }}

            #[inline(always)]
            fn shl_const<const N: i32>(a: {repr}) -> {repr} {{
                {shl}
            }}

            #[inline(always)]
            fn shr_arithmetic_const<const N: i32>(a: {repr}) -> {repr} {{
                {shr_arith}
            }}

            #[inline(always)]
            fn shr_logical_const<const N: i32>(a: {repr}) -> {repr} {{
                {shr_logic}
            }}

            #[inline(always)]
            fn all_true(a: {repr}) -> bool {{
                {all_true}
            }}

            #[inline(always)]
            fn any_true(a: {repr}) -> bool {{
                {any_true}
            }}

            #[inline(always)]
            fn bitmask(a: {repr}) -> u32 {{
                {bitmask}
            }}
        }}
    "#,
        v2_copies = (0..sub_count).map(|_| "v2").collect::<Vec<_>>().join(", "),
        z_copies = (0..sub_count).map(|_| "z").collect::<Vec<_>>().join(", "),
        load_lanes = (0..sub_count)
            .map(|i| format!("vld1q_s64(data.as_ptr().add({}))", i * 2))
            .collect::<Vec<_>>().join(", "),
        store_lanes = (0..sub_count)
            .map(|i| format!("vst1q_s64(out.as_mut_ptr().add({}), repr[{i}]);", i * 2))
            .collect::<Vec<_>>().join("\n            "),
        add = binary_op("vaddq_s64"),
        sub = binary_op("vsubq_s64"),
        neg = unary_op("vnegq_s64"),
        min = {
            let items: Vec<String> = (0..sub_count)
                .map(|i| format!("vbslq_s64(vcltq_s64(a[{i}], b[{i}]), a[{i}], b[{i}])"))
                .collect();
            format!("unsafe {{ [{}] }}", items.join(", "))
        },
        max = {
            let items: Vec<String> = (0..sub_count)
                .map(|i| format!("vbslq_s64(vcgtq_s64(a[{i}], b[{i}]), a[{i}], b[{i}])"))
                .collect();
            format!("unsafe {{ [{}] }}", items.join(", "))
        },
        abs = unary_op("vabsq_s64"),
        eq = cmp_op("vceqq_s64"),
        ne = ne_op(),
        lt = cmp_op("vcltq_s64"),
        le = cmp_op("vcleq_s64"),
        gt = cmp_op("vcgtq_s64"),
        ge = cmp_op("vcgeq_s64"),
        blend = {
            let items: Vec<String> = (0..sub_count)
                .map(|i| format!("vbslq_s64(vreinterpretq_u64_s64(mask[{i}]), if_true[{i}], if_false[{i}])"))
                .collect();
            format!("unsafe {{ [{}] }}", items.join(", "))
        },
        reduce_add = {
            let mut body = "unsafe {\n".to_string();
            body.push_str("            let m = vaddq_s64(a[0], a[1]);\n");
            for i in 2..sub_count {
                body.push_str(&format!("            let m = vaddq_s64(m, a[{i}]);\n"));
            }
            body.push_str("            let sum = vpaddq_s64(m, m);\n");
            body.push_str("            vgetq_lane_s64::<0>(sum)\n");
            body.push_str("        }");
            body
        },
        not = {
            let items: Vec<String> = (0..sub_count)
                .map(|i| format!("veorq_s64(a[{i}], vdupq_n_s64(-1i64))"))
                .collect();
            format!("unsafe {{ [{}] }}", items.join(", "))
        },
        bitand = binary_op("vandq_s64"),
        bitor = binary_op("vorrq_s64"),
        bitxor = binary_op("veorq_s64"),
        shl = {
            let items: Vec<String> = (0..sub_count)
                .map(|i| format!("vshlq_n_s64::<N>(a[{i}])"))
                .collect();
            format!("unsafe {{ [{}] }}", items.join(", "))
        },
        shr_arith = {
            let items: Vec<String> = (0..sub_count)
                .map(|i| format!("vshrq_n_s64::<N>(a[{i}])"))
                .collect();
            format!("unsafe {{ [{}] }}", items.join(", "))
        },
        shr_logic = {
            let items: Vec<String> = (0..sub_count)
                .map(|i| format!("vreinterpretq_s64_u64(vshrq_n_u64::<N>(vreinterpretq_u64_s64(a[{i}])))"))
                .collect();
            format!("unsafe {{ [{}] }}", items.join(", "))
        },
        all_true = {
            let items: Vec<String> = (0..sub_count)
                .map(|i| {
                    format!("(vgetq_lane_u64::<0>(vreinterpretq_u64_s64(a[{i}])) != 0 && vgetq_lane_u64::<1>(vreinterpretq_u64_s64(a[{i}])) != 0)")
                })
                .collect();
            format!("unsafe {{ {} }}", items.join(" && "))
        },
        any_true = {
            let items: Vec<String> = (0..sub_count)
                .map(|i| {
                    format!("((vgetq_lane_u64::<0>(vreinterpretq_u64_s64(a[{i}])) | vgetq_lane_u64::<1>(vreinterpretq_u64_s64(a[{i}]))) != 0)")
                })
                .collect();
            format!("unsafe {{ {} }}", items.join(" || "))
        },
        bitmask = {
            let mut body = "unsafe {\n".to_string();
            body.push_str("            let mut bits = 0u32;\n");
            for i in 0..sub_count {
                let base = i * 2;
                body.push_str(&format!("            let s{i} = vshrq_n_u64::<63>(vreinterpretq_u64_s64(a[{i}]));\n"));
                body.push_str(&format!("            bits |= (vgetq_lane_u64::<0>(s{i}) as u32) << {base};\n"));
                body.push_str(&format!("            bits |= (vgetq_lane_u64::<1>(s{i}) as u32) << {};\n", base + 1));
            }
            body.push_str("            bits\n");
            body.push_str("        }");
            body
        },
    }
}

// ============================================================================
// WASM I64 Implementation Generation
// ============================================================================

pub(super) fn generate_wasm_i64_impls(types: &[I64VecType]) -> String {
    let mut code = String::new();

    for ty in types {
        code.push_str("\n#[cfg(target_arch = \"wasm32\")]\n");
        if ty.native_on_wasm() {
            code.push_str(&generate_wasm_native_i64_impl(ty));
        } else {
            code.push_str(&generate_wasm_polyfill_i64_impl(ty));
        }
        code.push('\n');
    }

    code
}

fn generate_wasm_native_i64_impl(ty: &I64VecType) -> String {
    let trait_name = ty.trait_name();
    let array = ty.array_type();
    let lanes = ty.lanes;

    let reduce_add_body = (0..lanes)
        .map(|j| format!("i64x2_extract_lane::<{j}>(a)"))
        .collect::<Vec<_>>()
        .join(" + ");

    formatdoc! {r#"
        impl {trait_name} for archmage::Wasm128Token {{
            type Repr = v128;

            #[inline(always)]
            fn splat(v: i64) -> v128 {{ i64x2_splat(v) }}
            #[inline(always)]
            fn zero() -> v128 {{ i64x2_splat(0i64) }}
            #[inline(always)]
            fn load(data: &{array}) -> v128 {{ unsafe {{ v128_load(data.as_ptr().cast()) }} }}
            #[inline(always)]
            fn from_array(arr: {array}) -> v128 {{ unsafe {{ v128_load(arr.as_ptr().cast()) }} }}
            #[inline(always)]
            fn store(repr: v128, out: &mut {array}) {{ unsafe {{ v128_store(out.as_mut_ptr().cast(), repr) }}; }}
            #[inline(always)]
            fn to_array(repr: v128) -> {array} {{
                let mut out = [0i64; {lanes}];
                unsafe {{ v128_store(out.as_mut_ptr().cast(), repr) }};
                out
            }}

            #[inline(always)]
            fn add(a: v128, b: v128) -> v128 {{ i64x2_add(a, b) }}
            #[inline(always)]
            fn sub(a: v128, b: v128) -> v128 {{ i64x2_sub(a, b) }}
            #[inline(always)]
            fn neg(a: v128) -> v128 {{ i64x2_neg(a) }}
            #[inline(always)]
            fn min(a: v128, b: v128) -> v128 {{
                // WASM SIMD lacks native i64 min; polyfill via compare+select
                let mask = i64x2_gt(a, b);
                v128_bitselect(b, a, mask)
            }}
            #[inline(always)]
            fn max(a: v128, b: v128) -> v128 {{
                // WASM SIMD lacks native i64 max; polyfill via compare+select
                let mask = i64x2_gt(a, b);
                v128_bitselect(a, b, mask)
            }}
            #[inline(always)]
            fn abs(a: v128) -> v128 {{
                // Polyfill: negate negative values
                let negated = i64x2_neg(a);
                let zero = i64x2_splat(0i64);
                let mask = i64x2_lt(a, zero);
                v128_bitselect(negated, a, mask)
            }}

            #[inline(always)]
            fn simd_eq(a: v128, b: v128) -> v128 {{ i64x2_eq(a, b) }}
            #[inline(always)]
            fn simd_ne(a: v128, b: v128) -> v128 {{ i64x2_ne(a, b) }}
            #[inline(always)]
            fn simd_lt(a: v128, b: v128) -> v128 {{ i64x2_lt(a, b) }}
            #[inline(always)]
            fn simd_le(a: v128, b: v128) -> v128 {{ i64x2_le(a, b) }}
            #[inline(always)]
            fn simd_gt(a: v128, b: v128) -> v128 {{ i64x2_gt(a, b) }}
            #[inline(always)]
            fn simd_ge(a: v128, b: v128) -> v128 {{ i64x2_ge(a, b) }}
            #[inline(always)]
            fn blend(mask: v128, if_true: v128, if_false: v128) -> v128 {{
                v128_bitselect(if_true, if_false, mask)
            }}

            #[inline(always)]
            fn reduce_add(a: v128) -> i64 {{ {reduce_add_body} }}

            #[inline(always)]
            fn not(a: v128) -> v128 {{ v128_not(a) }}
            #[inline(always)]
            fn bitand(a: v128, b: v128) -> v128 {{ v128_and(a, b) }}
            #[inline(always)]
            fn bitor(a: v128, b: v128) -> v128 {{ v128_or(a, b) }}
            #[inline(always)]
            fn bitxor(a: v128, b: v128) -> v128 {{ v128_xor(a, b) }}

            #[inline(always)]
            fn shl_const<const N: i32>(a: v128) -> v128 {{ i64x2_shl(a, N as u32) }}
            #[inline(always)]
            fn shr_arithmetic_const<const N: i32>(a: v128) -> v128 {{ i64x2_shr(a, N as u32) }}
            #[inline(always)]
            fn shr_logical_const<const N: i32>(a: v128) -> v128 {{ u64x2_shr(a, N as u32) }}

            #[inline(always)]
            fn all_true(a: v128) -> bool {{ i64x2_all_true(a) }}
            #[inline(always)]
            fn any_true(a: v128) -> bool {{ v128_any_true(a) }}
            #[inline(always)]
            fn bitmask(a: v128) -> u32 {{ i64x2_bitmask(a) as u32 }}
        }}
    "#}
}

fn generate_wasm_polyfill_i64_impl(ty: &I64VecType) -> String {
    let trait_name = ty.trait_name();
    let repr = ty.wasm_repr();
    let array = ty.array_type();
    let lanes = ty.lanes;
    let sub_count = ty.sub_count();

    let binary_op = |func: &str| -> String {
        let items: Vec<String> = (0..sub_count)
            .map(|i| format!("{func}(a[{i}], b[{i}])"))
            .collect();
        format!("[{}]", items.join(", "))
    };

    let unary_op = |func: &str| -> String {
        let items: Vec<String> = (0..sub_count).map(|i| format!("{func}(a[{i}])")).collect();
        format!("[{}]", items.join(", "))
    };

    let reduce_add_body = || -> String {
        let mut items = Vec::new();
        for i in 0..sub_count {
            for j in 0..2usize {
                items.push(format!("i64x2_extract_lane::<{j}>(a[{i}])"));
            }
        }
        items.join(" + ")
    };

    formatdoc! {r#"
        impl {trait_name} for archmage::Wasm128Token {{
            type Repr = {repr};

            #[inline(always)]
            fn splat(v: i64) -> {repr} {{
                let v2 = i64x2_splat(v);
                [{v2_copies}]
            }}

            #[inline(always)]
            fn zero() -> {repr} {{
                let z = i64x2_splat(0i64);
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
                <Self as {trait_name}>::load(&arr)
            }}

            #[inline(always)]
            fn store(repr: {repr}, out: &mut {array}) {{
                unsafe {{
                    {store_lanes}
                }}
            }}

            #[inline(always)]
            fn to_array(repr: {repr}) -> {array} {{
                let mut out = [0i64; {lanes}];
                <Self as {trait_name}>::store(repr, &mut out);
                out
            }}

            #[inline(always)]
            fn add(a: {repr}, b: {repr}) -> {repr} {{ {add} }}
            #[inline(always)]
            fn sub(a: {repr}, b: {repr}) -> {repr} {{ {sub} }}
            #[inline(always)]
            fn neg(a: {repr}) -> {repr} {{ {neg} }}
            #[inline(always)]
            fn min(a: {repr}, b: {repr}) -> {repr} {{
                // WASM SIMD lacks native i64 min; polyfill via compare+select per sub-vector
                [{min_lanes}]
            }}
            #[inline(always)]
            fn max(a: {repr}, b: {repr}) -> {repr} {{
                // WASM SIMD lacks native i64 max; polyfill via compare+select per sub-vector
                [{max_lanes}]
            }}
            #[inline(always)]
            fn abs(a: {repr}) -> {repr} {{
                // Polyfill: negate negative values per sub-vector
                [{abs_lanes}]
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
            fn reduce_add(a: {repr}) -> i64 {{ {reduce_add} }}

            #[inline(always)]
            fn not(a: {repr}) -> {repr} {{ {not} }}
            #[inline(always)]
            fn bitand(a: {repr}, b: {repr}) -> {repr} {{ {and} }}
            #[inline(always)]
            fn bitor(a: {repr}, b: {repr}) -> {repr} {{ {or} }}
            #[inline(always)]
            fn bitxor(a: {repr}, b: {repr}) -> {repr} {{ {xor} }}

            #[inline(always)]
            fn shl_const<const N: i32>(a: {repr}) -> {repr} {{
                [{shl_lanes}]
            }}

            #[inline(always)]
            fn shr_arithmetic_const<const N: i32>(a: {repr}) -> {repr} {{
                [{shr_arith_lanes}]
            }}

            #[inline(always)]
            fn shr_logical_const<const N: i32>(a: {repr}) -> {repr} {{
                [{shr_logic_lanes}]
            }}

            #[inline(always)]
            fn all_true(a: {repr}) -> bool {{
                {all_true}
            }}

            #[inline(always)]
            fn any_true(a: {repr}) -> bool {{
                {any_true}
            }}

            #[inline(always)]
            fn bitmask(a: {repr}) -> u32 {{
                {bitmask}
            }}
        }}
    "#,
        v2_copies = (0..sub_count).map(|_| "v2").collect::<Vec<_>>().join(", "),
        z_copies = (0..sub_count).map(|_| "z").collect::<Vec<_>>().join(", "),
        load_lanes = (0..sub_count)
            .map(|i| format!("v128_load(data.as_ptr().add({}).cast())", i * 2))
            .collect::<Vec<_>>().join(", "),
        store_lanes = (0..sub_count)
            .map(|i| format!("v128_store(out.as_mut_ptr().add({}).cast(), repr[{i}]);", i * 2))
            .collect::<Vec<_>>().join("\n            "),
        add = binary_op("i64x2_add"),
        sub = binary_op("i64x2_sub"),
        neg = unary_op("i64x2_neg"),
        min_lanes = (0..sub_count)
            .map(|i| {
                format!("{{ let mask = i64x2_gt(a[{i}], b[{i}]); v128_bitselect(b[{i}], a[{i}], mask) }}")
            })
            .collect::<Vec<_>>().join(", "),
        max_lanes = (0..sub_count)
            .map(|i| {
                format!("{{ let mask = i64x2_gt(a[{i}], b[{i}]); v128_bitselect(a[{i}], b[{i}], mask) }}")
            })
            .collect::<Vec<_>>().join(", "),
        abs_lanes = (0..sub_count)
            .map(|i| {
                format!("{{ let neg = i64x2_neg(a[{i}]); let zero = i64x2_splat(0i64); let mask = i64x2_lt(a[{i}], zero); v128_bitselect(neg, a[{i}], mask) }}")
            })
            .collect::<Vec<_>>().join(", "),
        eq = binary_op("i64x2_eq"),
        ne = binary_op("i64x2_ne"),
        lt = binary_op("i64x2_lt"),
        le = binary_op("i64x2_le"),
        gt = binary_op("i64x2_gt"),
        ge = binary_op("i64x2_ge"),
        blend_lanes = (0..sub_count)
            .map(|i| format!("v128_bitselect(if_true[{i}], if_false[{i}], mask[{i}])"))
            .collect::<Vec<_>>().join(", "),
        reduce_add = reduce_add_body(),
        not = unary_op("v128_not"),
        and = binary_op("v128_and"),
        or = binary_op("v128_or"),
        xor = binary_op("v128_xor"),
        shl_lanes = (0..sub_count)
            .map(|i| format!("i64x2_shl(a[{i}], N as u32)"))
            .collect::<Vec<_>>().join(", "),
        shr_arith_lanes = (0..sub_count)
            .map(|i| format!("i64x2_shr(a[{i}], N as u32)"))
            .collect::<Vec<_>>().join(", "),
        shr_logic_lanes = (0..sub_count)
            .map(|i| format!("u64x2_shr(a[{i}], N as u32)"))
            .collect::<Vec<_>>().join(", "),
        all_true = (0..sub_count)
            .map(|i| format!("i64x2_all_true(a[{i}])"))
            .collect::<Vec<_>>().join(" && "),
        any_true = (0..sub_count)
            .map(|i| format!("v128_any_true(a[{i}])"))
            .collect::<Vec<_>>().join(" || "),
        bitmask = {
            let items: Vec<String> = (0..sub_count)
                .map(|i| format!("((i64x2_bitmask(a[{i}]) as u32) << {})", i * 2))
                .collect();
            items.join(" | ")
        },
    }
}

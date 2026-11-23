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
        /// Trait methods take `self` (the token) as receiver — the token value
        /// is the proof of CPU support, and requiring it as the receiver means
        /// the methods cannot be invoked via UFCS without holding one. The
        /// implementing type `Self` (a token type) determines which platform
        /// intrinsics are used. All methods are `#[inline(always)]` in
        /// implementations.
        ///
        /// # Sealed
        ///
        /// This trait is sealed — only archmage token types can implement it.
        /// The token proves CPU support was verified via `summon()`.
        pub trait {trait_name}: SimdToken + Sealed + Copy + 'static {{
            /// Platform-native SIMD representation.
            #[allow(private_bounds)]
            type Repr: Copy + Clone + Send + Sync + crate::simd_storage::Pod;

            // ====== Construction ======

            /// Broadcast scalar to all {lanes} lanes.
            fn splat(self, v: i64) -> Self::Repr;

            /// All lanes zero.
            fn zero(self) -> Self::Repr;

            /// Load from an aligned array.
            fn load(self, data: &{array}) -> Self::Repr;

            /// Create from array (zero-cost transmute where possible).
            fn from_array(self, arr: {array}) -> Self::Repr;

            /// Store to array.
            fn store(self, repr: Self::Repr, out: &mut {array});

            /// Convert to array.
            fn to_array(self, repr: Self::Repr) -> {array};

            // ====== Arithmetic ======

            /// Lane-wise addition.
            fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise subtraction.
            fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            // NOTE: No `mul` — no native i64 multiply on most platforms before AVX-512.

            /// Lane-wise negation.
            fn neg(self, a: Self::Repr) -> Self::Repr;

            // ====== Math ======

            /// Lane-wise minimum.
            fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise maximum.
            fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise absolute value.
            fn abs(self, a: Self::Repr) -> Self::Repr;

            // ====== Comparisons ======
            // Return masks where each lane is all-1s (true) or all-0s (false).

            /// Lane-wise equality.
            fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise inequality.
            fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise less-than.
            fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise less-than-or-equal.
            fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise greater-than.
            fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise greater-than-or-equal.
            fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Select lanes: where mask is all-1s pick `if_true`, else `if_false`.
            fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr;

            // ====== Reductions ======

            /// Sum all {lanes} lanes.
            fn reduce_add(self, a: Self::Repr) -> i64;

            // ====== Bitwise ======

            /// Bitwise NOT.
            fn not(self, a: Self::Repr) -> Self::Repr;

            /// Bitwise AND.
            fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Bitwise OR.
            fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Bitwise XOR.
            fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            // ====== Shifts ======

            /// Shift left by constant.
            fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;

            /// Arithmetic shift right by constant (sign-extending).
            /// `N` must be in `0..=63`; the generic front-ends reject out-of-range `N` at compile time.
            fn shr_arithmetic_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;

            /// Logical shift right by constant (zero-filling).
            /// `N` must be in `0..=63`; the generic front-ends reject out-of-range `N` at compile time.
            fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;

            // ====== Boolean ======

            /// True if **every** lane has its sign bit set.
            ///
            /// This is a sign-bit test, not a "lane is nonzero" test, on
            /// every backend and at every width. A lane holding `1` is
            /// false; a lane holding `-1` (or `0x80` in its top bit) is
            /// true. Comparison results are all-ones or all-zeros per lane,
            /// so for masks — the intended input — the two readings agree
            /// and this is the cheap native reduction. They diverge only on
            /// hand-built vectors, which is where the backends used to
            /// disagree with each other.
            fn all_true(self, a: Self::Repr) -> bool;

            /// True if **any** lane has its sign bit set.
            ///
            /// Sign-bit test, not "lane is nonzero" — see `all_true`.
            fn any_true(self, a: Self::Repr) -> bool;

            /// Extract the high bit of each 64-bit lane as a bitmask.
            fn bitmask(self, a: Self::Repr) -> u32;

            // ====== Default implementations ======

            /// Clamp values between lo and hi.
            #[inline(always)]
            fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {{
                <Self as {trait_name}>::min(self, <Self as {trait_name}>::max(self, a, lo), hi)
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
        code.push_str("\n#[cfg(all(target_arch = \"x86_64\", target_feature = \"sse2\"))]\n");
        code.push_str(&generate_x86_i64_impl(ty, token));
        code.push('\n');
    }

    code
}

fn generate_x86_i64_impl(ty: &I64VecType, token: &str) -> String {
    let arcane = super::backend_syntax::arcane(token);
    let baseline = super::backend_syntax::sse2_or_arcane(token, ty.width_bits);
    let baseline_end = if ty.width_bits == 128 { "}" } else { "" };
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
                {{
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

            {arcane}
            fn splat(self, v: i64) -> {inner} {{
                {p}_set1_epi64x(v)
            }}

            {arcane}
            fn zero(self) -> {inner} {{
                {p}_setzero_si{bits}()
            }}

            #[inline(always)]
            fn load(self, data: &{array}) -> {inner} {{
                crate::simd_storage::copy(data)
            }}

            #[inline(always)]
            fn from_array(self, arr: {array}) -> {inner} {{
                crate::simd_storage::cast(arr)
            }}

            #[inline(always)]
            fn store(self, repr: {inner}, out: &mut {array}) {{
                crate::simd_storage::store(repr, out);
            }}

            #[inline(always)]
            fn to_array(self, repr: {inner}) -> {array} {{
                crate::simd_storage::cast(repr)
            }}

            // ====== Arithmetic ======

            {baseline}
            fn add(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_add_epi64(a, b)
            }}
            {baseline_end}

            {baseline}
            fn sub(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_sub_epi64(a, b)
            }}
            {baseline_end}

            {arcane}
            fn neg(self, a: {inner}) -> {inner} {{
                {p}_sub_epi64({p}_setzero_si{bits}(), a)
            }}

            // ====== Math ======

            {arcane}
            fn min(self, a: {inner}, b: {inner}) -> {inner} {{
                // Polyfill: compare+select (no native i64 min on AVX2)
                {{
                    let mask = {p}_cmpgt_epi64(a, b);
                    {p}_blendv_epi8(a, b, mask)
                }}
            }}

            {arcane}
            fn max(self, a: {inner}, b: {inner}) -> {inner} {{
                // Polyfill: compare+select (no native i64 max on AVX2)
                {{
                    let mask = {p}_cmpgt_epi64(a, b);
                    {p}_blendv_epi8(b, a, mask)
                }}
            }}

            {arcane}
            fn abs(self, a: {inner}) -> {inner} {{
                // Polyfill: (a ^ sign) - sign (two's complement trick)
                {{
                    let zero = {p}_setzero_si{bits}();
                    let sign = {p}_cmpgt_epi64(zero, a);
                    {p}_sub_epi64({p}_xor_si{bits}(a, sign), sign)
                }}
            }}

            // ====== Comparisons ======

            {arcane}
            fn simd_eq(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_cmpeq_epi64(a, b)
            }}

            {arcane}
            fn simd_ne(self, a: {inner}, b: {inner}) -> {inner} {{
                let eq = {p}_cmpeq_epi64(a, b);
                {p}_xor_si{bits}(eq, {p}_set1_epi64x(-1))
            }}

            {arcane}
            fn simd_lt(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_cmpgt_epi64(b, a)
            }}

            {arcane}
            fn simd_le(self, a: {inner}, b: {inner}) -> {inner} {{
                let gt = {p}_cmpgt_epi64(a, b);
                {p}_xor_si{bits}(gt, {p}_set1_epi64x(-1))
            }}

            {arcane}
            fn simd_gt(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_cmpgt_epi64(a, b)
            }}

            {arcane}
            fn simd_ge(self, a: {inner}, b: {inner}) -> {inner} {{
                let lt = {p}_cmpgt_epi64(b, a);
                {p}_xor_si{bits}(lt, {p}_set1_epi64x(-1))
            }}

            {arcane}
            fn blend(self, mask: {inner}, if_true: {inner}, if_false: {inner}) -> {inner} {{
                {p}_blendv_epi8(if_false, if_true, mask)
            }}

            // ====== Reductions ======

            {arcane}
            fn reduce_add(self, a: {inner}) -> i64 {{
        {reduce_add_body}
            }}

            // ====== Bitwise ======

            {baseline}
            fn not(self, a: {inner}) -> {inner} {{
                {p}_xor_si{bits}(a, {p}_set1_epi64x(-1))
            }}
            {baseline_end}

            {baseline}
            fn bitand(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_and_si{bits}(a, b)
            }}
            {baseline_end}

            {baseline}
            fn bitor(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_or_si{bits}(a, b)
            }}
            {baseline_end}

            {baseline}
            fn bitxor(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_xor_si{bits}(a, b)
            }}
            {baseline_end}

            // ====== Shifts ======

            {arcane}
            fn shl_const<const N: i32>(self, a: {inner}) -> {inner} {{
                {p}_slli_epi64::<N>(a)
            }}

            {arcane}
            fn shr_arithmetic_const<const N: i32>(self, a: {inner}) -> {inner} {{
                // Polyfill: no native _srai_epi64 on AVX2.
                // Use logical shift + sign extension.
        {shr_arith_body}
            }}

            {arcane}
            fn shr_logical_const<const N: i32>(self, a: {inner}) -> {inner} {{
                {p}_srli_epi64::<N>(a)
            }}

            // ====== Boolean ======

            {arcane}
            fn all_true(self, a: {inner}) -> bool {{
                {p}_movemask_pd({p}_castsi{bits}_pd(a)) == {all_mask}
            }}

            {arcane}
            fn any_true(self, a: {inner}) -> bool {{
                {p}_movemask_pd({p}_castsi{bits}_pd(a)) != 0
            }}

            {arcane}
            fn bitmask(self, a: {inner}) -> u32 {{
                {p}_movemask_pd({p}_castsi{bits}_pd(a)) as u32
            }}
        }}
    "#}
}

fn generate_x86_i64_reduce_add(ty: &I64VecType) -> String {
    match ty.width_bits {
        128 => formatdoc! {"
                {{
                    let hi = _mm_unpackhi_epi64(a, a);
                    let sum = _mm_add_epi64(a, hi);
                    // Extract low 64-bit lane
                    crate::simd_storage::cast::<__m128i, [i64; 2]>(sum)[0]
                }}"},
        256 => formatdoc! {"
                {{
                    let lo = _mm256_castsi256_si128(a);
                    let hi = _mm256_extracti128_si256::<1>(a);
                    let sum = _mm_add_epi64(lo, hi);
                    let hi2 = _mm_unpackhi_epi64(sum, sum);
                    let sum2 = _mm_add_epi64(sum, hi2);
                    crate::simd_storage::cast::<__m128i, [i64; 2]>(sum2)[0]
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

    let reduce_add = (0..lanes)
        .map(|i| format!("a[{i}]"))
        .collect::<Vec<_>>()
        .join(".wrapping_add(")
        + &")".repeat(lanes - 1);

    let bitmask = (0..lanes)
        .map(|i| {
            if i == 0 {
                "((a[0] as u64) >> 63) as u32".to_string()
            } else {
                format!("((((a[{i}] as u64) >> 63) as u32) << {i})")
            }
        })
        .collect::<Vec<_>>()
        .join(" | ");

    // Sign-bit contract: a lane is "true" when its top bit is set, not
    // merely when it is nonzero. Shifting keeps this correct for both the
    // signed lane types (arithmetic shift -> 0 or -1) and the unsigned ones.
    let all_true = (0..lanes)
        .map(|i| format!("(a[{i}] >> 63) != 0"))
        .collect::<Vec<_>>()
        .join(" && ");

    let any_true = (0..lanes)
        .map(|i| format!("(a[{i}] >> 63) != 0"))
        .collect::<Vec<_>>()
        .join(" || ");

    formatdoc! {r#"
        impl {trait_name} for archmage::ScalarToken {{
            type Repr = {array};

            // ====== Construction ======

            #[inline(always)]
            fn splat(self, v: i64) -> {array} {{
                [v; {lanes}]
            }}

            #[inline(always)]
            fn zero(self) -> {array} {{
                [0i64; {lanes}]
            }}

            #[inline(always)]
            fn load(self, data: &{array}) -> {array} {{
                *data
            }}

            #[inline(always)]
            fn from_array(self, arr: {array}) -> {array} {{
                arr
            }}

            #[inline(always)]
            fn store(self, repr: {array}, out: &mut {array}) {{
                *out = repr;
            }}

            #[inline(always)]
            fn to_array(self, repr: {array}) -> {array} {{
                repr
            }}

            // ====== Arithmetic ======

            #[inline(always)]
            fn add(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| a[i].wrapping_add(b[i]))
            }}

            #[inline(always)]
            fn sub(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| a[i].wrapping_sub(b[i]))
            }}

            #[inline(always)]
            fn neg(self, a: {array}) -> {array} {{
                a.map(i64::wrapping_neg)
            }}

            // ====== Math ======

            #[inline(always)]
            fn min(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| if a[i] < b[i] {{ a[i] }} else {{ b[i] }})
            }}

            #[inline(always)]
            fn max(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| if a[i] > b[i] {{ a[i] }} else {{ b[i] }})
            }}

            #[inline(always)]
            fn abs(self, a: {array}) -> {array} {{
                a.map(i64::wrapping_abs)
            }}

            // ====== Comparisons ======

            #[inline(always)]
            fn simd_eq(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| if a[i] == b[i] {{ -1 }} else {{ 0 }})
            }}

            #[inline(always)]
            fn simd_ne(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| if a[i] != b[i] {{ -1 }} else {{ 0 }})
            }}

            #[inline(always)]
            fn simd_lt(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| if a[i] < b[i] {{ -1 }} else {{ 0 }})
            }}

            #[inline(always)]
            fn simd_le(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| if a[i] <= b[i] {{ -1 }} else {{ 0 }})
            }}

            #[inline(always)]
            fn simd_gt(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| if a[i] > b[i] {{ -1 }} else {{ 0 }})
            }}

            #[inline(always)]
            fn simd_ge(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| if a[i] >= b[i] {{ -1 }} else {{ 0 }})
            }}

            #[inline(always)]
            fn blend(self, mask: {array}, if_true: {array}, if_false: {array}) -> {array} {{
                core::array::from_fn(|i| if mask[i] != 0 {{ if_true[i] }} else {{ if_false[i] }})
            }}

            // ====== Reductions ======

            #[inline(always)]
            fn reduce_add(self, a: {array}) -> i64 {{
                {reduce_add}
            }}

            // ====== Bitwise ======

            #[inline(always)]
            fn not(self, a: {array}) -> {array} {{
                a.map(|x| !x)
            }}

            #[inline(always)]
            fn bitand(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| a[i] & b[i])
            }}

            #[inline(always)]
            fn bitor(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| a[i] | b[i])
            }}

            #[inline(always)]
            fn bitxor(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| a[i] ^ b[i])
            }}

            // ====== Shifts ======

            #[inline(always)]
            fn shl_const<const N: i32>(self, a: {array}) -> {array} {{
                a.map(|x| x << N)
            }}

            #[inline(always)]
            fn shr_arithmetic_const<const N: i32>(self, a: {array}) -> {array} {{
                a.map(|x| x >> N)
            }}

            #[inline(always)]
            fn shr_logical_const<const N: i32>(self, a: {array}) -> {array} {{
                a.map(|x| ((x as u64) >> N) as i64)
            }}

            // ====== Boolean ======

            #[inline(always)]
            fn all_true(self, a: {array}) -> bool {{
                {all_true}
            }}

            #[inline(always)]
            fn any_true(self, a: {array}) -> bool {{
                {any_true}
            }}

            #[inline(always)]
            fn bitmask(self, a: {array}) -> u32 {{
                {bitmask}
            }}
        }}
    "#}
}

// ============================================================================
// NEON I64 Implementation Generation
// ============================================================================

pub(super) fn generate_neon_i64_impls(types: &[I64VecType]) -> String {
    let mut code = String::new();

    for ty in types {
        if ty.native_on_neon() {
            code.push_str("\n#[cfg(target_arch = \"aarch64\")]\n");
            code.push_str(&generate_neon_native_i64_impl(ty));
            code.push('\n');
        }
    }

    code
}

fn generate_neon_native_i64_impl(ty: &I64VecType) -> String {
    let arcane = super::backend_syntax::arcane("NeonToken");
    let trait_name = ty.trait_name();
    let array = ty.array_type();

    formatdoc! {r#"
        impl {trait_name} for archmage::NeonToken {{
            type Repr = int64x2_t;

            {arcane}
            fn splat(self, v: i64) -> int64x2_t {{
                vdupq_n_s64(v)
            }}

            {arcane}
            fn zero(self) -> int64x2_t {{
                vdupq_n_s64(0i64)
            }}

            #[inline(always)]
            fn load(self, data: &{array}) -> int64x2_t {{
                crate::simd_storage::copy(data)
            }}

            #[inline(always)]
            fn from_array(self, arr: {array}) -> int64x2_t {{
                crate::simd_storage::cast(arr)
            }}

            #[inline(always)]
            fn store(self, repr: int64x2_t, out: &mut {array}) {{
                crate::simd_storage::store(repr, out);
            }}

            #[inline(always)]
            fn to_array(self, repr: int64x2_t) -> {array} {{
                crate::simd_storage::cast(repr)
            }}

            {arcane}
            fn add(self, a: int64x2_t, b: int64x2_t) -> int64x2_t {{ vaddq_s64(a, b) }}
            {arcane}
            fn sub(self, a: int64x2_t, b: int64x2_t) -> int64x2_t {{ vsubq_s64(a, b) }}
            {arcane}
            fn neg(self, a: int64x2_t) -> int64x2_t {{ vnegq_s64(a) }}
            {arcane}
            fn min(self, a: int64x2_t, b: int64x2_t) -> int64x2_t {{
                // NEON lacks native i64 min; polyfill via compare+select
                {{
                    let mask = vcltq_s64(a, b);
                    vbslq_s64(mask, a, b)
                }}
            }}
            {arcane}
            fn max(self, a: int64x2_t, b: int64x2_t) -> int64x2_t {{
                // NEON lacks native i64 max; polyfill via compare+select
                {{
                    let mask = vcgtq_s64(a, b);
                    vbslq_s64(mask, a, b)
                }}
            }}
            {arcane}
            fn abs(self, a: int64x2_t) -> int64x2_t {{ vabsq_s64(a) }}

            {arcane}
            fn simd_eq(self, a: int64x2_t, b: int64x2_t) -> int64x2_t {{
                vreinterpretq_s64_u64(vceqq_s64(a, b))
            }}
            {arcane}
            fn simd_ne(self, a: int64x2_t, b: int64x2_t) -> int64x2_t {{
                let eq = vceqq_s64(a, b);
                // NOT via XOR with all-ones
                vreinterpretq_s64_u64(veorq_u64(eq, vdupq_n_u64(u64::MAX)))
            }}
            {arcane}
            fn simd_lt(self, a: int64x2_t, b: int64x2_t) -> int64x2_t {{
                vreinterpretq_s64_u64(vcltq_s64(a, b))
            }}
            {arcane}
            fn simd_le(self, a: int64x2_t, b: int64x2_t) -> int64x2_t {{
                vreinterpretq_s64_u64(vcleq_s64(a, b))
            }}
            {arcane}
            fn simd_gt(self, a: int64x2_t, b: int64x2_t) -> int64x2_t {{
                vreinterpretq_s64_u64(vcgtq_s64(a, b))
            }}
            {arcane}
            fn simd_ge(self, a: int64x2_t, b: int64x2_t) -> int64x2_t {{
                vreinterpretq_s64_u64(vcgeq_s64(a, b))
            }}

            {arcane}
            fn blend(self, mask: int64x2_t, if_true: int64x2_t, if_false: int64x2_t) -> int64x2_t {{
                vbslq_s64(vreinterpretq_u64_s64(mask), if_true, if_false)
            }}

            {arcane}
            fn reduce_add(self, a: int64x2_t) -> i64 {{
                let sum = vpaddq_s64(a, a);
                vgetq_lane_s64::<0>(sum)
            }}

            {arcane}
            fn not(self, a: int64x2_t) -> int64x2_t {{
                let ones = vdupq_n_s64(-1i64);
                veorq_s64(a, ones)
            }}
            {arcane}
            fn bitand(self, a: int64x2_t, b: int64x2_t) -> int64x2_t {{
                vandq_s64(a, b)
            }}
            {arcane}
            fn bitor(self, a: int64x2_t, b: int64x2_t) -> int64x2_t {{
                vorrq_s64(a, b)
            }}
            {arcane}
            fn bitxor(self, a: int64x2_t, b: int64x2_t) -> int64x2_t {{
                veorq_s64(a, b)
            }}

            {arcane}
            fn shl_const<const N: i32>(self, a: int64x2_t) -> int64x2_t {{
                vshlq_n_s64::<N>(a)
            }}

            {arcane}
            fn shr_arithmetic_const<const N: i32>(self, a: int64x2_t) -> int64x2_t {{
                const {{ assert!(N >= 0 && N <= 63) }};
                vshlq_s64(a, vdupq_n_s64((-N) as i64))
            }}

            {arcane}
            fn shr_logical_const<const N: i32>(self, a: int64x2_t) -> int64x2_t {{
                const {{ assert!(N >= 0 && N <= 63) }};
                vreinterpretq_s64_u64(vshlq_u64(vreinterpretq_u64_s64(a), vdupq_n_s64((-N) as i64)))
            }}

            {arcane}
            fn all_true(self, a: int64x2_t) -> bool {{
                // Sign-bit contract. NEON has no 64-bit horizontal reduction,
                // so combine the lanes and test one sign bit.
                let as_u64 = vreinterpretq_u64_s64(a);
                ((vgetq_lane_u64::<0>(as_u64) & vgetq_lane_u64::<1>(as_u64)) >> 63) != 0
            }}

            {arcane}
            fn any_true(self, a: int64x2_t) -> bool {{
                let as_u64 = vreinterpretq_u64_s64(a);
                ((vgetq_lane_u64::<0>(as_u64) | vgetq_lane_u64::<1>(as_u64)) >> 63) != 0
            }}

            {arcane}
            fn bitmask(self, a: int64x2_t) -> u32 {{
                let signs = vshrq_n_u64::<63>(vreinterpretq_u64_s64(a));
                ((vgetq_lane_u64::<0>(signs) & 1) | ((vgetq_lane_u64::<1>(signs) & 1) << 1)) as u32
            }}
        }}
    "#}
}

// ============================================================================
// WASM I64 Implementation Generation
// ============================================================================

pub(super) fn generate_wasm_i64_impls(types: &[I64VecType]) -> String {
    let mut code = String::new();

    for ty in types {
        if ty.native_on_wasm() {
            code.push_str("\n#[cfg(target_arch = \"wasm32\")]\n");
            code.push_str(&generate_wasm_native_i64_impl(ty));
            code.push('\n');
        }
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
            fn splat(self, v: i64) -> v128 {{ i64x2_splat(v) }}
            #[inline(always)]
            fn zero(self) -> v128 {{ i64x2_splat(0i64) }}
            #[inline(always)]
            fn load(self, data: &{array}) -> v128 {{ crate::simd_storage::copy(data) }}
            #[inline(always)]
            fn from_array(self, arr: {array}) -> v128 {{ crate::simd_storage::cast(arr) }}
            #[inline(always)]
            fn store(self, repr: v128, out: &mut {array}) {{ crate::simd_storage::store(repr, out); }}
            #[inline(always)]
            fn to_array(self, repr: v128) -> {array} {{
                let mut out = [0i64; {lanes}];
                crate::simd_storage::store(repr, &mut out);
                out
            }}

            #[inline(always)]
            fn add(self, a: v128, b: v128) -> v128 {{ i64x2_add(a, b) }}
            #[inline(always)]
            fn sub(self, a: v128, b: v128) -> v128 {{ i64x2_sub(a, b) }}
            #[inline(always)]
            fn neg(self, a: v128) -> v128 {{ i64x2_neg(a) }}
            #[inline(always)]
            fn min(self, a: v128, b: v128) -> v128 {{
                // WASM SIMD lacks native i64 min; polyfill via compare+select
                let mask = i64x2_gt(a, b);
                v128_bitselect(b, a, mask)
            }}
            #[inline(always)]
            fn max(self, a: v128, b: v128) -> v128 {{
                // WASM SIMD lacks native i64 max; polyfill via compare+select
                let mask = i64x2_gt(a, b);
                v128_bitselect(a, b, mask)
            }}
            #[inline(always)]
            fn abs(self, a: v128) -> v128 {{
                // Polyfill: negate negative values
                let negated = i64x2_neg(a);
                let zero = i64x2_splat(0i64);
                let mask = i64x2_lt(a, zero);
                v128_bitselect(negated, a, mask)
            }}

            #[inline(always)]
            fn simd_eq(self, a: v128, b: v128) -> v128 {{ i64x2_eq(a, b) }}
            #[inline(always)]
            fn simd_ne(self, a: v128, b: v128) -> v128 {{ i64x2_ne(a, b) }}
            #[inline(always)]
            fn simd_lt(self, a: v128, b: v128) -> v128 {{ i64x2_lt(a, b) }}
            #[inline(always)]
            fn simd_le(self, a: v128, b: v128) -> v128 {{ i64x2_le(a, b) }}
            #[inline(always)]
            fn simd_gt(self, a: v128, b: v128) -> v128 {{ i64x2_gt(a, b) }}
            #[inline(always)]
            fn simd_ge(self, a: v128, b: v128) -> v128 {{ i64x2_ge(a, b) }}
            #[inline(always)]
            fn blend(self, mask: v128, if_true: v128, if_false: v128) -> v128 {{
                v128_bitselect(if_true, if_false, mask)
            }}

            #[inline(always)]
            fn reduce_add(self, a: v128) -> i64 {{ {reduce_add_body} }}

            #[inline(always)]
            fn not(self, a: v128) -> v128 {{ v128_not(a) }}
            #[inline(always)]
            fn bitand(self, a: v128, b: v128) -> v128 {{ v128_and(a, b) }}
            #[inline(always)]
            fn bitor(self, a: v128, b: v128) -> v128 {{ v128_or(a, b) }}
            #[inline(always)]
            fn bitxor(self, a: v128, b: v128) -> v128 {{ v128_xor(a, b) }}

            #[inline(always)]
            fn shl_const<const N: i32>(self, a: v128) -> v128 {{ i64x2_shl(a, N as u32) }}
            #[inline(always)]
            fn shr_arithmetic_const<const N: i32>(self, a: v128) -> v128 {{ i64x2_shr(a, N as u32) }}
            #[inline(always)]
            fn shr_logical_const<const N: i32>(self, a: v128) -> v128 {{ u64x2_shr(a, N as u32) }}

            #[inline(always)]
            fn all_true(self, a: v128) -> bool {{ i64x2_bitmask(a) == 0x03 }}
            #[inline(always)]
            fn any_true(self, a: v128) -> bool {{ i64x2_bitmask(a) != 0 }}
            #[inline(always)]
            fn bitmask(self, a: v128) -> u32 {{ i64x2_bitmask(a) as u32 }}
        }}
    "#}
}

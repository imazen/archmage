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
// x86 I64 Implementation Generation
// ============================================================================

pub(super) fn generate_x86_i64_impls(types: &[I64VecType]) -> String {
    let mut code = String::new();

    for ty in types {
        code.push_str("\n#[cfg(all(target_arch = \"x86_64\", target_feature = \"sse2\"))]\n");
        code.push_str(&generate_x86_i64_impl(ty));
        code.push('\n');
    }

    code
}

fn generate_x86_i64_impl(ty: &I64VecType) -> String {
    let arcane = super::backend_syntax::arcane("X64V3Token");
    let baseline = super::backend_syntax::sse2_or_arcane("X64V3Token", ty.width_bits);
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
        impl {trait_name} for archmage::X64V3Token {{
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
